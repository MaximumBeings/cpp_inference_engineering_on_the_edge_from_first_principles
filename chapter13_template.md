# Chapter 13: The KV Cache Manager -- Paged Attention, Ring Buffers, and Smart Eviction

**What you will understand by the end of this chapter:**

- Why autoregressive decode is memory-bandwidth-bound rather than compute-bound, and the one equation — tokens/second is bounded by bandwidth divided by bytes read per token (model weights plus KV cache) — that makes this concrete enough to compute, not just assert.
- Why the KV cache's share of that per-token byte cost grows with context length until it dominates the model's own weights, and why that growth is the reason KV cache management, not raw compute throughput, is the dominant performance lever at long context.
- PagedAttention: dividing the KV cache into fixed-size blocks dispensed from a pool via a free list, so allocation and deallocation are both O(1), memory is never fragmented, and identical system prompts across many users can be shared by copying a handful of integers instead of the underlying data.
- The Ring Buffer's two-counter design — a wrapping physical write slot and a monotonically increasing logical RoPE position that are equal only during the initial fill — and why conflating them, a mistake that produces no crash and no warning, silently corrupts every attention score computed after the first wrap.
- H2O's use of attention probability, a signal the model already computes for free, as an importance score that lets eviction target the least-useful cached token instead of blindly discarding whichever token happens to be oldest — and how combining paging, wrapping, and smart eviction into one manager preserves every guarantee each technique proved on its own.

**What you need to know first:**

- Chapter 2's RoPE positional encoding — specifically that a Key vector's rotation angle is a function of the token's LOGICAL position in the sequence, a fact this chapter's Ring Buffer section depends on directly.
- Chapter 8's roofline model and its prefill/decode distinction: prefill processes a whole prompt at once and is compute-bound, while decode processes one token at a time and is memory-bound — this chapter is entirely about managing the resource that dominates the memory-bound side.
- This is a purely single-threaded, data-structure-focused chapter: no `-pthread`, no `std::mdspan`, no `-ffp-contract=off` for any file here, continuing Chapter 12's simplification — every file compiles with just `-std=c++23 -Wall -Wextra -O2`.

---

Every chapter through Part 2 asked how fast the CPU could compute a forward pass. This chapter asks a different question: once a model is generating token after token, what does it cost to remember everything it has already seen, and how do you keep that cost from destroying performance or correctness? Section 13.1 derives the memory wall itself — the equation that makes decode-time speed a function of bytes moved, not FLOPs computed — and shows exactly how much of that byte budget the KV cache claims as context grows. Section 13.2 solves the fragmentation half of the problem with PagedAttention: block-table indirection that allocates memory on demand and shares identical prefixes for free. Section 13.3 solves the unbounded-growth half with a Ring Buffer, and spends most of its attention on a subtle correctness property — RoPE position decoupling — that a naive implementation gets wrong in a way that looks fine until the model's output quietly stops making sense. Section 13.4 replaces FIFO eviction, which the Ring Buffer uses by default, with H2O's attention-probability-driven importance scoring, so eviction discards what the model has stopped needing rather than whatever merely arrived first. Section 13.5 closes the chapter by combining all three techniques into one manager and confirming that nothing about combining them weakens any guarantee proved in isolation.

## 13.1 The Memory Wall: Why the KV Cache Is the Real Bottleneck

### Intuition

An autoregressive model is stateful: generating the Nth token requires attending over all N-1 tokens that came before it. Storing their Key and Value vectors instead of recomputing them every step turns an O(N^2) total-FLOP generation into O(N) per step — a genuinely good trade, but one that swaps compute for memory traffic, and memory traffic has its own, much harder, speed limit.

### The Concept, In Detail

Every decode step must read the entire model weight matrix plus the current KV cache from memory before it can produce one token, so the theoretical ceiling on tokens per second is exactly (memory bandwidth) divided by (weight bytes plus KV cache bytes read per step). Neither more CPU cores, nor a higher clock, nor a wider FPU moves this ceiling at all, because none of them change how many bytes must cross the memory bus — this is the precise sense in which decode is memory-bound rather than compute-bound, the mirror image of Chapter 8's compute-bound prefill phase. The KV cache's own size is a direct product of the model's shape: for each of `n_layers` transformer layers, for each of `n_heads_kv` grouped-query-attention KV heads, for each cached token, two vectors (Key and Value) of `head_dim` elements each are stored at some number of bytes per element. For Llama 3 8B (32 layers, 8 KV heads, head_dim 128, FP16 storage) that works out to exactly 128 KB per token — a number small enough to ignore at a few hundred tokens of context and large enough, at 128K tokens, to require 16 GB on its own, dwarfing the roughly 4-4.5 GB the model's own quantized weights occupy. The consequence is that the KV cache's share of total bytes read per decode step is not fixed — it grows with context length until, at long enough context, it dominates the model's weights entirely, which is exactly why managing the KV cache's size, not further compute optimization, becomes the dominant performance lever once conversations or documents get long.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_kv_cache_bandwidth_estimator.cpp -o 01_kv_cache_bandwidth_estimator
./01_kv_cache_bandwidth_estimator
```

**Sample input:** real published model shapes (Llama 3 8B) checked against a hand calculation of KV cache size at 4096 tokens, a realistic Q4-quantized speed-limit computation, monotonic speed degradation as context grows from 512 to 8192 tokens, KV cache dominating over 70% of bandwidth at 128K context, and quantization's speedup shrinking as context (and therefore the KV cache's own share of bandwidth) grows.

@@OUT1@@

!!! warning "[COMMON TRAP] assuming more compute closes a memory-bandwidth gap"
    A CPU or GPU with dramatically more FLOPs per second generates decode tokens at the SAME speed as a slower one if the two have identical memory bandwidth, because decode's bottleneck is bytes moved per step, not arithmetic performed per step — the model's weights and KV cache must physically travel from RAM to the compute unit before a single multiply can happen, and no amount of extra ALU throughput makes that transfer faster. This is precisely why an H100 GPU (thousands of TFLOPS) and a desktop CPU (roughly one TFLOP) can show a far smaller decode-speed gap than their raw compute numbers would suggest, while showing close to the FULL gap in prefill, where compute rather than bandwidth is the limiting resource. Optimizing the wrong resource — adding compute throughput to a memory-bound workload — buys nothing measurable; the only levers that move decode speed are reducing bytes per token (quantization, a smaller KV cache) or increasing bandwidth itself.

## 13.2 PagedAttention: Block-Table Indirection

### Intuition

Two simple ways to allocate a KV cache both fail in production: reserving every connected user's worst-case context wastes nearly all of it on users who never approach that limit, and growing a buffer dynamically avoids that waste but pays for it with a stuttering reallocate-and-copy pause exactly when the buffer fills. PagedAttention borrows virtual-memory paging from operating systems to get neither failure mode.

### The Concept, In Detail

The KV cache is divided into fixed-size BLOCKS of tokens (this section uses 16), pre-allocated once into one big pool at startup, and dispensed to sequences on demand via a free list — a `BlockManager` that never allocates again after construction, so `alloc()` and `free()` are both O(1) integer-ID operations on that free list. Each sequence keeps a small BLOCK TABLE — an array mapping its own logical block index to a physical block ID somewhere in the pool — and physical blocks need not be, and generally are not, contiguous with each other. Because the block size is a power of two, translating a logical token index into (which block, which slot inside it) is exactly two single-cycle bit operations, a right shift and a mask, negligible next to the roughly 100ns DRAM access that follows either way. This indirection pays for itself doubly: it eliminates the waste of static pre-allocation, since a sequence only ever holds as many blocks as it has actually used, and it makes prefix sharing essentially free, since many users sharing an identical system prompt can have their block tables all reference the SAME physical blocks — "sharing" becomes copying a handful of integers, not copying any Key or Value data at all, and a user later adding their own unique tokens simply allocates NEW blocks for those without disturbing the shared ones.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_paged_attention_block_manager.cpp -o 02_paged_attention_block_manager
./02_paged_attention_block_manager
```

**Sample input:** bit-operation address translation checked against plain integer division and modulo across ten token indices; a ten-block pool's alloc/free lifecycle; 35 tokens written and read back across three block boundaries; paged attention output checked against a dense, unpaged reference on 20 random tokens; three users sharing one physical system-prompt block while adding unique tokens of their own; and a static-versus-paged memory utilization comparison across 20 simulated users.

@@OUT2@@

!!! warning "[COMMON TRAP] treating a shared block table entry as safe to write through directly"
    When three users' block tables all reference the same physical system-prompt block, that block must remain read-only from every user's perspective for as long as it is shared — writing a user's own new tokens must always go into a NEWLY allocated block, never into a shared one, even though the shared block's physical memory is sitting right there and technically writable. A block manager has no way to know a given physical block is currently referenced by more than one sequence's block table; it is the CALLER'S responsibility (`Sequence::store_kv` allocating a new block whenever the current one fills, rather than ever appending into an existing shared block past its System-prompt length) to preserve that invariant. Violating it silently corrupts every OTHER user sharing that block the moment one user's "unique" token overwrites shared data — a bug that produces plausible-looking garbage in a completely unrelated user's context, with no crash and no obvious cause.

## 13.3 Ring Buffer Contexts: RoPE Position Decoupling and Sink-Token Protection

### Intuition

PagedAttention solves fragmentation but not growth: total KV cache size still scales linearly with context length no matter how efficiently its blocks are packed. A Ring Buffer caps memory at a fixed size by overwriting the oldest entry once full — simple in concept, but hiding a correctness trap that produces no warning and no crash, only silently wrong attention.

### The Concept, In Detail

A Ring Buffer's physical WRITE SLOT wraps: slot equals total tokens processed, modulo capacity. But before a Key vector is ever cached, RoPE rotates it using that token's LOGICAL position in the overall sequence — a fact Chapter 2 established and this section now depends on directly. If an implementation reuses the wrapped physical slot number as the RoPE rotation angle, every token written after the FIRST wrap gets rotated by the wrong position: token 4096, physically landing in slot 0 because the buffer just wrapped, would be rotated as if it were token 0, and every later query's relative-distance computation against it becomes wrong by however far the sequence has wrapped. The fix keeps two separate counters that are equal only during the initial fill and deliberately diverge forever afterward: `write_slot()`, which wraps and answers only "where does the next token's data go," and `rope_position()`, which is simply the running total of tokens ever processed and answers only "what rotation does this token's Key vector need." A second, independent hazard survives even once RoPE positions are correct: trained transformers reliably dump a large share of softmax attention probability onto the first few tokens of a sequence regardless of their content, an "attention sink" behavior the model has learned to rely on as a safe destination for probability mass it has nowhere better to send. Evicting those sink tokens when the ring wraps measurably degrades coherence even with every rotation angle correct, so this section reserves a small PROTECTED region at the front of the buffer — write_slot's wrap logic only ever cycles through the ROLLING region past it, and the first few slots are never touched again once filled.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_ring_buffer_sink_tokens.cpp -o 03_ring_buffer_sink_tokens
./03_ring_buffer_sink_tokens
```

**Sample input:** 16 tokens written into a capacity-8, protected-2 ring, checking every write slot against a hand-derived expected sequence; the same tiny buffer checked to confirm write_slot equals rope_position only during the initial fill and diverges strictly afterward; three sink-token sentinel values confirmed intact after 60 total tokens through a capacity-12 buffer; memory confirmed capped at capacity after 100 tokens through a capacity-8 buffer; and ring attention checked against a dense reference before any wrap occurs.

@@OUT3@@

!!! warning "[COMMON TRAP] using the physical write slot as the RoPE rotation angle"
    It is a natural-looking shortcut to rotate a Key vector using the same slot index the ring buffer just computed for WHERE to store it — after all, that index is already sitting right there as a convenient integer. This works perfectly during the initial fill, when slot and logical position happen to be numerically identical, which is exactly what makes the bug so dangerous: it passes every test run only long enough to fill the buffer once, then silently corrupts every attention computation from the first wrap onward. There is no crash, no NaN, no obviously wrong-looking number — just relative-distance math that is quietly off by however many tokens the buffer has wrapped, producing degraded, sometimes bizarre generation that is easy to blame on the model itself rather than on a rotation angle that has been wrong since the first overwritten slot. The fix is never using `write_slot()` for anything except "where," and never using anything except the monotonically increasing total-tokens-processed counter for "what rotation."

## 13.4 Smart Eviction: The H2O Heavy-Hitter Policy

### Intuition

The Ring Buffer's default eviction is FIFO: whichever token is oldest goes, no matter what it contains. That is fine for genuinely disposable streaming data, but a real conversation's early tokens are not uniformly disposable — a user's stated name matters for the rest of the session; idle filler three tokens later does not — and FIFO cannot tell the two apart.

### The Concept, In Detail

H2O (Heavy Hitter Oracle) replaces "oldest goes" with "least useful goes," using a signal the model already computes as a byproduct of ordinary attention and normally discards afterward: the softmax probability each cached token receives at every decode step. Accumulating that probability into a per-token running total — one floating-point addition per cached token per step — is essentially free, since the attention computation already produces those probabilities; a token the model consistently attends to heavily accumulates high importance, and a token it consistently ignores accumulates almost none. When the cache is full and a new token needs a slot, eviction targets the single lowest-importance token currently eligible — with one deliberate exception: a small RECENT WINDOW of the most-just-added tokens is immune to eviction regardless of their importance score, because a token that has only existed for one or two decode steps has not yet had a fair chance to accumulate the importance its actual usefulness deserves. Evicting purely by recency (FIFO) discards a critical early fact and a piece of throwaway filler with equal indifference; H2O's importance signal is what lets eviction discriminate between the two using information the model has already paid to compute.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_h2o_eviction.cpp -o 04_h2o_eviction
./04_h2o_eviction
```

**Sample input:** a five-token cache with hand-assigned importance scores checked to confirm eviction targets the single lowest-importance eligible token; a four-token cache with tied importances checked to confirm the recent window correctly protects the three newest tokens; a 40-token simulated conversation with three designated heavy-hitter tokens, checked to confirm H2O retains all three while a FIFO baseline over the same data drops at least one; and ten simulated decode steps checked to confirm accumulated importance equals the exact sum of per-step attention probabilities.

@@OUT4@@

!!! warning "[COMMON TRAP] letting brand-new tokens compete on importance before they have any"
    A token that was just added to the cache has, by definition, accumulated at most one or two decode steps' worth of attention probability — nowhere near enough to reflect whether the model will actually rely on it going forward. Without a recent-window exemption, H2O's own importance-based eviction would systematically discriminate against every token the instant after it arrives, evicting genuinely important brand-new context (a name just stated, an instruction just given) simply because it has not yet had time to accumulate the score its true usefulness deserves — turning a smarter eviction policy into one that is, perversely, worse than FIFO for exactly the tokens that most need to survive their first few steps. The recent window's immunity is not a workaround for a limitation in the importance signal; it is a recognition that importance is a signal that needs TIME to become meaningful, and eviction must not judge a token before that time has passed.

## 13.5 A Complete KV Cache Manager

### Intuition

Each of this chapter's three techniques was verified in isolation. A production serving system needs all three working together — paged blocks for many concurrent users, a wrapping ring for unbounded conversations, and H2O eviction so wrapping discards the right tokens — and combining verified pieces is only useful if the combination does not quietly weaken any of the guarantees each piece proved on its own.

### The Concept, In Detail

This section reuses Section 13.2's `BlockManager`, Section 13.3's `RingBufferCtx`, and Section 13.4's `H2OPolicy` completely unchanged — no reimplementation, no simplification — and re-derives each one's core guarantee inside a combined scenario rather than merely asserting the combination works. A multi-user paged-serving scenario confirms three simulated users sharing one physical system-prompt block still see that same sharing (and the same O(1) pool free-count arithmetic) when they are running alongside the chapter's other two techniques rather than in isolation; a wrapping-ring scenario reproduces Section 13.3's own hand-derived write-slot sequence exactly to confirm nothing about combining it with paging or eviction logic elsewhere in the program perturbs its wrap arithmetic; and an H2O-versus-FIFO scenario confirms heavy hitters still survive under the combined manager while a plain-FIFO baseline over identical data still loses at least one. None of these checks are new claims this chapter has not already proven — they are the SAME claims, re-verified in the presence of the other two techniques, which is precisely what "does the combination preserve every guarantee" means operationally rather than as an assertion. A final memory-capacity analysis, scaled to Llama 3 8B's real per-block byte cost, quantifies the payoff of Section 13.2's paging concretely: reserving worst-case context for 50 users costs tens of gigabytes, while paging the same 50 users at their real average usage costs a small fraction of that.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_kv_cache_manager.cpp -o 05_kv_cache_manager
./05_kv_cache_manager
```

**Sample input:** three simulated users sharing a paged system-prompt block, with one disconnecting and returning its unique blocks to the pool; a 16-token run through a capacity-10, protected-2 ring buffer checked against Section 13.3's own wrap sequence; a 20-token H2O-versus-FIFO comparison with three designated heavy hitters; and a memory-capacity comparison between static worst-case reservation and paged allocation at real Llama 3 8B block sizes for 50 simulated users.

@@OUT5@@

!!! warning "[COMMON TRAP] assuming techniques verified separately compose correctly by default"
    Each of Sections 13.2 through 13.4 verified its own technique in complete isolation from the other two — PagedAttention's tests never wrapped a ring buffer, and the ring buffer's tests never ran an eviction policy. It is tempting to conclude that because each piece is independently correct, wiring them together in a real server must also be correct, with no further verification needed. This chapter's own Section 11.5-style discipline (Chapter 11's discovery that individually-correct parallel phases can still combine into a thread-count-dependent whole) applies here just as directly: combining independently-verified components is a NEW claim, not a free consequence of the old ones, and it earns verification of its own. This section's tests exist specifically because "I already proved each piece works" is not the same statement as "I proved the assembled system works" — the former is necessary, never sufficient, and this chapter's capstone treats it that way by re-checking each technique's own guarantee inside the combined scenario rather than skipping straight to a demo with no assertions at all.

## Chapter Summary

This chapter turned from the compute-bound optimization of Part 2 to decode's real bottleneck: the state an autoregressive model must remember between tokens, and the cost of keeping that state in memory. Section 13.1 derived the memory wall itself — tokens per second bounded by bandwidth divided by bytes read per step — and showed the KV cache's share of that byte cost growing from negligible to dominant as context length increases. Section 13.2 solved fragmentation with PagedAttention: fixed-size blocks dispensed from a pool via an O(1) free list, with block-table indirection making prefix sharing across users a matter of copying integers rather than data. Section 13.3 solved unbounded growth with a Ring Buffer, whose real lesson was a subtlety rather than the wrapping itself: physical write slot and logical RoPE position must be tracked as two genuinely separate counters, or every token written after the first wrap silently receives the wrong positional rotation, plus a protected region shielding attention-sink tokens the model has learned to rely on. Section 13.4 replaced FIFO's blind age-based eviction with H2O's attention-probability-driven importance score, a signal the model already computes for free, letting eviction discriminate between critical early context and disposable filler. Section 13.5 closed the chapter by combining all three techniques and re-verifying each one's own guarantee inside that combination, rather than treating separately-proven correctness as automatically transitive. Chapter 14 builds directly on this foundation, extending it to sliding-window attention at scale, a tiered cache that re-quantizes aging entries to coarser precision, and prefix caching across multi-turn conversations.

## Self-Check Questions

1. Why does adding more CPU cores or a faster clock fail to speed up decode, when the same changes would measurably speed up prefill?
2. Section 13.1's KV cache size formula uses `n_heads_kv` rather than the model's total number of query heads. Why does GQA (grouped-query attention) make that substitution correct, and what would go wrong if the formula used the query head count instead?
3. Why does PagedAttention's block size need to be a power of two specifically, rather than merely a small fixed number like 16?
4. In Section 13.2's zero-copy prefix sharing, what exactly gets copied when three users share an identical system prompt, and what would break if a user's later, unique tokens were appended directly into a shared block instead of a newly allocated one?
5. Section 13.3 tracks `write_slot()` and `rope_position()` as two separate counters that are equal only during a ring buffer's initial fill. Concretely, what goes wrong with attention scores if an implementation uses `write_slot()` for both purposes after the buffer has wrapped at least once?
6. Why does protecting a small number of "sink" tokens at the front of a ring buffer improve model coherence even when every RoPE position is already computed correctly?
7. Why is the attention probability H2O accumulates into each token's importance score described as a "free" signal, rather than one that costs additional computation to obtain?
8. In Section 13.4, why does H2O's eviction policy exempt a small recent window of tokens from competing on importance at all, rather than simply letting brand-new tokens start with an importance score of zero and compete normally?
9. Section 13.3's Test 3 writes 60 tokens through a 12-slot ring buffer with 3 protected slots and confirms the first 3 slots still hold their original sentinel values. Why does this specifically demonstrate that the protected region is never touched by the wrap logic, rather than merely that eviction happens to favor early tokens?
10. Section 13.5's capstone re-runs checks that Sections 13.2 through 13.4 already performed individually, rather than only adding new checks for behavior unique to the combination. Why is re-verifying each technique's own guarantee INSIDE the combined scenario a meaningfully different claim than having verified it in isolation?

## Where We Go Next

This chapter gave every future chapter a KV cache that pages efficiently, wraps without corrupting positional information, and evicts by actual usefulness rather than blind age. Chapter 14 extends this foundation three ways: sliding-window attention as a ring buffer applied at the scale a real long-context model needs, a tiered cache architecture that re-quantizes aging entries to progressively coarser precision as they leave the active working set, and prefix caching that lets a multi-turn conversation reuse everything computed in earlier turns — closing with a complete cache manager that combines every technique from both chapters behind one interface.

## Worked Solutions

**1.** Decode's speed ceiling is `bandwidth / bytes_read_per_token`, a ratio that depends only on how many bytes of model weights and KV cache must travel from RAM to the compute unit each step and how fast that memory channel can move them — it does not appear anywhere in terms of FLOPs or arithmetic throughput. Adding cores, raising clock speed, or widening the FPU all increase how much ARITHMETIC can be performed per second, but none of them increase how many BYTES per second can cross the memory bus, so a workload whose bottleneck is the bus (decode) sees no benefit, while prefill — which processes a whole prompt's worth of tokens in one compute-heavy pass and is genuinely limited by arithmetic throughput — benefits directly from exactly those same changes.

**2.** Grouped-query attention deliberately uses fewer Key/Value heads than Query heads, with multiple Query heads sharing one KV head's cached Key and Value vectors; the KV CACHE only ever stores one Key and one Value vector per KV head per token, never one per query head, so `n_heads_kv` is the number of distinct K/V vectors that actually exist in memory. Using the query head count instead would compute a KV cache size several times too large, over-reserving memory for K/V data that GQA's whole design never actually stores in the first place — Llama 3 8B's 8 KV heads versus a larger number of query heads is exactly the gap such a mistake would inflate by.

**3.** A power-of-two block size lets address translation — converting a logical token index into (which block, which slot within it) — compile to a bit shift (dividing by the block size) and a bit mask (taking the remainder), both single-cycle operations with no division instruction involved at all. A block size that is merely small but not a power of two, like 15 or 20, would require an actual integer division and modulo on every single cached-token access during attention — still fast in absolute terms, but needlessly slower than two bit operations for a piece of arithmetic that executes on every token, every head, every decode step.

**4.** Only the physical block IDs in each user's block table get copied — a handful of plain integers — while the actual Key and Value data for the shared system prompt exists exactly once in the pool, referenced by every sharing user's table. If a user's own later, unique tokens were appended directly into that SAME shared block rather than a freshly allocated one, the write would silently overwrite Key/Value data every OTHER user sharing that block is still relying on, corrupting their context with no crash, no error, and no indication anything went wrong beyond that other user's subsequent attention output becoming inexplicably wrong.

**5.** Before caching, a Key vector is rotated by RoPE using its token's LOGICAL position in the overall sequence, but `write_slot()` after the first wrap no longer equals that logical position — it is a physical index that has cycled back through values it has used before. If an implementation rotates using `write_slot()`, a token like T4096, which physically lands in slot 0 because the buffer just wrapped, gets rotated as though it were token 0 instead of token 4096; a later query computing its relative distance to that token then gets a wildly wrong distance (thousands of positions off) instead of the small, correct one, corrupting every attention score involving that token from that point forward.

**6.** Trained transformers reliably direct a large share of softmax attention probability toward the first few tokens of a sequence regardless of their actual content — a learned behavior where the model uses early tokens as a stable place to send probability mass it has nowhere more useful to put, since softmax must always sum to 1.0. This is independent of whether RoPE positions are computed correctly: even with perfect rotations, if those specific sink tokens are evicted when the ring wraps, the model loses the anchor its learned attention pattern depends on, and probability mass redistributes unpredictably across the remaining tokens, degrading coherence for a reason that has nothing to do with positional encoding at all.

**7.** The attention mechanism must compute a softmax probability for every cached token at every single decode step regardless of whether anything downstream ever records it — that computation happens purely to produce the weighted sum over Value vectors that IS the attention output. H2O's importance score is obtained by adding that already-computed probability into a running per-token total, a single floating-point addition that piggybacks on work the model was doing anyway; no additional forward pass, no extra matrix multiply, and no new probability computation is needed to obtain it.

**8.** A token that has existed in the cache for only one or two decode steps has, by definition, accumulated only one or two steps' worth of attention probability regardless of how genuinely important its content is — its LOW accumulated score at that point reflects its short lifetime, not its actual usefulness. Starting new tokens at zero and letting them compete immediately would systematically evict brand-new context (a name just stated, an instruction just given) purely because it has not yet had TIME to accumulate the score reflecting its true importance, which is worse than useless: it would make a "smart" eviction policy discriminate specifically against the tokens most likely to matter going forward. The recent window instead grants new tokens a fixed grace period, immune from competing on a score that is not yet meaningful, before importance-based competition applies to them at all.

**9.** If the protected region were merely favored by eviction rather than structurally exempt from the wrap logic, extended enough pressure (here, 57 additional tokens cycling through a 9-slot rolling region) would eventually still overwrite it, because "favored" implies a comparison that some other condition could still lose. The test's 60 tokens deliberately exceed the rolling region's own capacity many times over — proving that no amount of continued writing ever reaches slots 0 through 2 at all, because `write_slot()`'s wrap arithmetic is defined to only ever produce values in the range `[protected_size, capacity)` once wrapping begins; the protected slots are outside that arithmetic's output range entirely, not merely unlikely targets within it.

**10.** Verifying a technique in isolation proves it is correct under the specific conditions its own test constructed — Section 13.2's tests never had a ring buffer's wrap logic or an eviction policy running alongside PagedAttention, for instance. Re-running the SAME check (three users sharing one physical block, a hand-derived wrap sequence, heavy hitters surviving eviction) inside a scenario where all three techniques are actually operating together is a different claim: it confirms nothing about the OTHER two techniques' bookkeeping perturbs this one's guarantee, which is exactly the kind of interaction that unit tests run in isolation cannot see by construction. Chapter 11 demonstrated the general version of this lesson directly — individually thread-count-independent phases still produced a thread-count-DEPENDENT whole once combined — and this section's capstone treats "I verified each piece separately" as necessary but never sufficient for exactly that reason.
