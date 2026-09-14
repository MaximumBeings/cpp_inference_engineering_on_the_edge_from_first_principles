# Chapter 14: Advanced KV Cache Management -- Sliding Windows, Streaming Re-Quantization, and Prefix Caching

**What you will understand by the end of this chapter:**

- Sliding-window attention as the same two-counter ring buffer mechanism Chapter 13.3 introduced, but with the sink-token protected region removed entirely — the window boundary itself becomes the eviction policy, with no case-by-case judgment about which tokens matter.
- Streaming re-quantization: compressing aging cache entries to progressively coarser bit-widths instead of evicting them outright, and why a token's shrinking attention weight as it ages is what makes the resulting quantization error harmless to the output.
- Prefix caching for multi-turn conversations: reusing a previous turn's KV cache entries instead of recomputing an unchanged shared history, and why a KVBlock's fixed granularity means a block straddling the point where two turns diverge must be recomputed in full even though part of it was genuinely shared.
- Why combining independently-correct techniques can surface a NEW form of waste that none of them showed alone — specifically, that prefix caching can only reuse what the runtime cache still holds, and aggressive eviction from earlier sections can silently shrink prefix caching's own benefit.

**What you need to know first:**

- Chapter 13.3's Ring Buffer: the two-counter design (a wrapping physical write slot and a monotonically increasing logical position) that Section 14.1 reuses directly, minus its sink-token protection.
- Chapter 13.4's H2O eviction policy — attention-probability-driven importance scoring with a recent-window immunity — which Section 14.4's capstone reuses as one of several combined eviction mechanisms.
- This book's own Chapter 7 (TurboQuant), which Section 14.2 references by name for the production version of the multi-bit-width re-quantization it demonstrates with a simplified stand-in quantizer.
- This is a purely single-threaded, data-structure-focused chapter, continuing Chapters 12 and 13's simplification: no `-pthread`, no `std::mdspan`, no `-ffp-contract=off` for any file here — every file compiles with just `-std=c++23 -Wall -Wextra -O2`.

---

Chapter 13 gave every future chapter a KV cache that pages efficiently, wraps without corrupting positional information, and evicts by measured usefulness rather than blind age. This chapter extends that foundation in exactly the three directions Chapter 13 did not cover, plus a capstone that ties all four techniques together. Section 14.1 applies Chapter 13.3's ring-buffer mechanism at the scale a real long-context model actually needs: a hard window of thousands of tokens, with the window boundary itself standing in for any case-by-case eviction judgment. Section 14.2 offers a gentler alternative to hard eviction: instead of a token either being in the cache or gone, it is compressed to a coarser bit-width as it ages, trading precision for memory in a way that costs the output almost nothing because old tokens receive vanishingly small attention weight anyway. Section 14.3 tackles a problem earlier chapters never addressed at all — a multi-turn conversation recomputing its own unchanged history on every single turn — with an entirely original prefix-caching design, since this chapter's own source material describes the idea only in prose with an unverified speedup figure and no accompanying tested code. Section 14.4 closes the chapter by combining sliding windows, re-quantization, prefix caching, and Chapter 13.4's H2O eviction into one complete cache manager, and — following Chapter 13.5's precedent that combining verified pieces is a new claim requiring its own verification — surfaces a genuinely new interaction: prefix caching's benefit depends on what the runtime cache still retains, and the tokens most likely to have been evicted are exactly the ones deepest in a long shared prefix.

## 14.1 Sliding-Window Attention as a Ring Buffer at Scale

### Intuition

Chapter 13.3's Ring Buffer capped memory by wrapping a fixed-size buffer and protected a handful of sink tokens from that wrap. Sliding-window attention is the same wrapping mechanism with that protection removed: nothing is exempt, and the window's edge is the entire eviction policy.

### The Concept, In Detail

A sliding window keeps exactly the most recent `W` tokens and never attends to anything older, full stop — unlike H2O, which decides case by case whether a given token still matters, sliding window makes no such judgment at all. The implementation is Chapter 13.3's two-counter ring buffer verbatim, minus the protected region: `current_pos` counts every token ever inserted and never wraps, while the physical storage slot for a given position is `position % window_size`, so the valid attention range is always `[max(0, current_pos - window_size), current_pos)`. This is not merely a memory-saving approximation retrofitted onto models that were never designed for it — Mistral 7B was explicitly TRAINED with a 4096-token sliding window, meaning the model learned during training to compress anything it might need beyond that window into its hidden state rather than relying on attention to reach arbitrarily far back. For models trained with full, unrestricted attention instead, sliding window is only an approximation, but one that tends to work well in practice regardless, because attention weight empirically falls off sharply with distance: tokens thousands of positions back typically receive negligible attention whether or not a hard window forces them out.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_sliding_window_cache.cpp -o 01_sliding_window_cache
./01_sliding_window_cache
```

**Sample input:** 5 tokens inserted into an 8-slot window, checked byte-exact against every stored original; 12 tokens into the same 8-slot window, checked to confirm exactly the correct 8 remain visible and the oldest 4 are gone; memory measured before and after 1000 tokens to confirm it stays exactly constant; and a full attention computation over a 20-token stream's current window checked to produce a valid, non-zero-norm output.

@@OUT1@@

!!! warning "[COMMON TRAP] treating sliding window as free correctness for a full-attention-trained model"
    Mistral 7B's sliding window is not an approximation at all for that specific model — the model was trained to expect exactly this restriction and learned to compress older context into its hidden state accordingly, so applying the window changes nothing about the model's own expectations. Applying the identical mechanism to a model trained with full, unrestricted attention is a genuinely different situation: that model never learned to compress anything into its hidden state, because during training it could always attend arbitrarily far back. The fact that this approximation tends to work well in practice — because attention weight empirically decays with distance regardless of training regime — is an empirical property of typical attention distributions, not a guarantee, and a task that genuinely depends on precise recall of something far outside the window (a specific number stated thousands of tokens earlier, for instance) can fail silently: no crash, no warning, just an answer computed as though that information were never there at all.

## 14.2 Streaming Re-Quantization: Tiered Precision as Entries Age

### Intuition

Eviction, whether FIFO or H2O, is a binary decision: a token is either fully present in the cache or entirely gone. Streaming re-quantization offers a middle path — instead of discarding an aging token outright, compress it to a coarser bit-width, so the cache degrades gracefully rather than falling off a cliff.

### The Concept, In Detail

A freshly inserted token starts at high-fidelity 4-bit quantization; as it ages past configurable thresholds it is progressively re-quantized to 3-bit, then 2-bit, before eventual eviction — a tiered cache where recent tokens keep high-fidelity Key/Value vectors and old tokens hold increasingly coarse ones. The insight that makes this safe rather than merely convenient is that a token's actual influence on the CURRENT generation step is proportional to the attention weight it receives, and that weight shrinks the older a token gets (the same empirical decay Section 14.1 leaned on). A quantization error introduced into a token's Key or Value vector gets multiplied by that token's attention weight before it ever reaches the output, so a coarse quantization of a barely-attended-to old token produces an error that is itself barely-attended-to, while the SAME coarse quantization applied to a heavily-weighted recent token would visibly corrupt the output — which is exactly why the tiering is age-based rather than uniform. This section implements the mechanism with a simplified symmetric uniform quantizer at each bit-width rather than this book's own Chapter 7 (TurboQuant) machinery: a production system would re-quantize by dequantizing the current representation back to float and re-encoding with a coarser Lloyd-Max codebook from that chapter, sharing one rotation matrix across every bit-width and swapping only the codebook, but the cache-management policy — age thresholds, progressive compression, eventual eviction — is identical either way, so a uniform quantizer keeps this file's focus on that policy rather than re-deriving Chapter 7's codebook construction.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_streaming_requantization.cpp -o 02_streaming_requantization
./02_streaming_requantization
```

**Sample input:** a freshly inserted token checked to start at 4-bit; five tokens aged through 15, 50, and 170 simulated steps against thresholds of 10/40/160, checked to land at 3-bit, then 2-bit, then fully evicted at each stage; a 500-step simulation with periodic maintenance checked to use less total memory than a uniform 4-bit cache of the same live entries; and a single vector's reconstruction error checked to increase monotonically as it is re-quantized from 4-bit down through 3-bit to 2-bit.

@@OUT2@@

!!! warning "[COMMON TRAP] assuming a tiering scheme's claimed memory savings without running it"
    A tiering scheme's savings depend on exactly how many live entries land in each tier at the moment memory is measured, which in turn depends on the age-threshold parameters, how often maintenance actually runs, and how long the cache has been running when the measurement is taken — none of which can be derived from the tier table alone without actually simulating the policy. A back-of-envelope estimate assuming a specific, round distribution across tiers (for instance, assuming exactly evenly-spaced buckets of tokens at each bit-width) will not generally match what a real, running cache produces, because real insertion and maintenance timing rarely aligns with round numbers — this chapter's own honestly-computed savings figure, obtained by actually running the maintenance loop over 500 simulated steps, differs from a simpler static-bucket estimate for exactly this reason, and the book's standing rule is to report the number the code actually produced rather than the more convenient one a table suggests.

## 14.3 Prefix Caching for Multi-Turn Conversations

### Intuition

Every new turn in a multi-turn conversation shares a long prefix with the previous one: the system prompt, the conversation history, and the formatting tokens the chat template inserts. Recomputing the KV cache for that entire shared prefix on every single turn is pure waste — the point of a KV cache is that it is a checkpoint of the model's state, and resuming from a checkpoint should be free.

### The Concept, In Detail

This section's source material describes prefix caching only in prose, with a single narrative "speedup: 501x" figure and no accompanying tested code anywhere — this book's standing rule is to never carry forward a performance claim that was not itself computed and checked, so what follows is an original design built on Chapter 13.2's `BlockManager` and `Sequence` rather than a transcription of that figure. The mechanism compares the OLD turn's token-ID sequence against the NEW one element by element to find their longest common prefix (LCP); everything before the first mismatch is genuinely shared history, and everything from the mismatch onward is new or changed and must be recomputed. The subtlety is that a `KVBlock` is the smallest unit PagedAttention can reuse, so only WHOLE blocks that lie entirely within the matched prefix — `floor(lcp / BLOCK_SIZE)` of them — can actually be reused; a block that straddles the divergence point, partly inside the shared prefix and partly past it, must be recomputed in full even though part of its contents were technically shared, because there is no way to reuse "part of a block." This block-granularity waste is a genuine, permanent property of the design rather than an oversight to be optimized away: a conversation edit that diverges at token 25 inside a 16-token-block scheme reuses only the FIRST fully-contained block (tokens 0-15), discarding tokens 16-24 even though they matched, purely because they shared a block with tokens that did not.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_prefix_caching.cpp -o 03_prefix_caching
./03_prefix_caching
```

**Sample input:** a block-aligned 48-token history extended by 20 new tokens, checked for zero-waste full-prefix reuse and an honestly-computed speedup ratio; a 40-token history diverging at token 25 with a 45-token new turn, checked to demonstrate exactly 9 tokens of block-granularity waste; a byte-for-byte `memcmp` of a surviving block's raw storage before and after a turn transition; a check that the recomputed suffix matches the new token IDs while the reused prefix still matches the original ones; and a realistic 2048-token history extended by 48 tokens, checked for a token-count savings figure computed directly from the code's own counters.

@@OUT3@@

!!! warning "[COMMON TRAP] assuming a matched token means a reusable token"
    The longest common prefix between two token-ID sequences answers a purely textual question — how many leading tokens are identical — and says nothing on its own about whether those tokens' KV data can actually be reused, because reuse additionally requires that the match extends across an entire block boundary. A 9-token stretch that genuinely matches but sits inside a block that also contains 7 tokens past the divergence point is, in every sense that matters for reuse, indistinguishable from a stretch that never matched at all: it gets recomputed either way. Treating "tokens matched" and "tokens reused" as the same quantity overstates a prefix cache's real benefit by exactly the amount of block-granularity waste any given divergence point happens to produce, which is why this section reports both figures separately rather than only the more flattering one.

## 14.4 A Complete Cache Manager Combining Every Technique

### Intuition

Sections 14.1 through 14.3, plus Chapter 13.4's H2O policy, each solve a different piece of the same problem in isolation. Chapter 13.5 already established that combining independently-verified pieces is a new claim requiring its own verification, not a free consequence of the old ones — and combining these four specific techniques surfaces a genuinely new interaction none of them showed alone.

### The Concept, In Detail

This section's `CacheManager` runs sliding window trimming, age-based re-quantization, sink-token protection, and H2O-style importance eviction (with Chapter 13.4's own recent-window immunity preserved) together during a single turn's generation, then uses Section 14.3's prefix-caching mechanism to hand surviving entries off to the next turn. Running all four together re-confirms, inside the combined scenario, that each technique's own guarantee still holds: the sliding window's hard cutoff, H2O's recent-window immunity, re-quantization's progressive tiering, and sink-token survival all get checked again here rather than merely assumed to carry over from their own chapters. But the combination also does something none of the four techniques showed in isolation: Section 14.3 measured prefix-caching waste purely from block alignment, implicitly assuming that a token inside the matched prefix was still sitting in the cache waiting to be reused. Once a real eviction policy is running alongside prefix caching, that assumption can simply be false — a token can match the new turn's prefix perfectly and STILL require recomputation, because the sliding window or H2O policy already evicted it during the PREVIOUS turn, long before the new turn's prefix match was ever computed. Prefix caching can only reuse what the runtime cache still holds; matching token IDs is necessary but never sufficient, and the tokens most likely to have been evicted — the oldest ones — are exactly the tokens most likely to lie deep inside a long shared prefix, which is precisely where a naive accounting of prefix-caching savings would expect the largest payoff.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_complete_cache_manager.cpp -o 04_complete_cache_manager
./04_complete_cache_manager
```

**Sample input:** a 300-step combined simulation (200-token window, 50-entry budget, 4 sink tokens) checked for the sliding window's hard cutoff, H2O's recent-window immunity via an isolated probe, progressive re-quantization tiering, and sink-token survival with continuously valid attention output; a prefix handoff between two turns with no prior eviction, checked to show the theoretical and actual reuse counts matching exactly; and a prefix handoff after 100 steps of real eviction under a tight budget, checked to show a substantial, honestly-computed gap between tokens that matched the prefix and tokens the cache had actually retained.

@@OUT4@@

!!! warning "[COMMON TRAP] crediting prefix caching for tokens the cache no longer has"
    A prefix-caching implementation that only checks "does this token's ID match the previous turn's sequence at this position" and calls that a cache hit is answering the wrong question the moment any eviction policy is also running, because a matching token ID guarantees nothing about whether that token's KV data still physically exists anywhere in memory. This section's own combined scenario makes the gap concrete: a full 100-token history matches the new turn's prefix token for token, yet a tight window-and-budget policy had already evicted all but 20 of those 100 entries by the time the new turn begins, so 80 tokens that would appear "reusable" under a purely textual prefix check must be recomputed exactly as if they had never matched at all. The fix is that prefix-caching logic must always check actual cache membership, never token-ID equality alone, before counting anything as reused — and any reported prefix-caching speedup figure that was computed from matched-token counts rather than actually-reused-token counts should be treated as an upper bound, not a measurement.

## Chapter Summary

This chapter extended Chapter 13's KV cache management foundation in the three directions its own techniques had not yet covered. Section 14.1 applied Chapter 13.3's ring-buffer mechanism at real long-context scale, removing the sink-token protection so the window boundary itself becomes the entire eviction policy — exactly what Mistral 7B was trained to expect, and a reasonable approximation for models that were not. Section 14.2 replaced binary eviction with graceful degradation: aging tokens are compressed to coarser bit-widths rather than discarded outright, safely, because their shrinking attention weight makes the resulting error harmless to the output. Section 14.3 built an entirely original prefix-caching design, since this chapter's own source material offered only an unverified narrative claim, and showed that a KVBlock's fixed granularity means matched tokens are not always reusable tokens. Section 14.4 closed the chapter by combining all four techniques — sliding window, re-quantization, prefix caching, and Chapter 13.4's H2O eviction — into one manager, re-verifying each technique's own guarantee inside the combination and surfacing a genuinely new interaction: prefix caching can only reuse what the runtime cache still retains, and the tokens most likely to have been evicted are exactly the ones most likely to sit deep inside a long shared prefix. Chapters 13 and 14 together give every later chapter a KV cache that pages efficiently, bounds its own growth at real scale, degrades gracefully as entries age, evicts by measured usefulness, and avoids recomputing work a previous turn already did.

## Self-Check Questions

1. Section 14.1's sliding window removes Chapter 13.3's sink-token protection entirely. Why does that removal make sense for sliding window specifically, when Chapter 13.3 treated sink-token eviction as a real coherence problem worth solving?
2. Why is applying a fixed-size sliding window to a model trained with full, unrestricted attention described as an approximation, while applying the identical window to Mistral 7B is not?
3. Section 14.2's re-quantization tiering compresses OLD tokens to coarser bit-widths rather than NEW ones. Explain why this age-based direction, rather than the reverse, is what makes the resulting quantization error safe to ignore.
4. Section 14.2 keeps a simplified uniform quantizer rather than building this book's Chapter 7 TurboQuant machinery inline. What part of the cache-management POLICY would be identical either way, and what part would actually differ if Chapter 7's codebooks were used instead?
5. In Section 14.3, a 40-token conversation diverges at token 25 under a 16-token block size. Walk through why exactly 9 tokens (16 through 24) are recomputed despite having matched the new turn's token IDs.
6. Why does Section 14.3 insist that only WHOLE blocks entirely inside the longest common prefix can be reused, rather than reusing individual matched tokens directly regardless of block boundaries?
7. Section 14.3's source material claimed an unverified "501x" prefix-caching speedup with no accompanying code. What did this chapter do instead, and why does the book's standing rule require that choice?
8. Section 14.4 re-checks the sliding window's cutoff, H2O's recent-window immunity, and re-quantization's tiering all over again inside the combined manager, even though each was already verified in its own chapter section. Why isn't verifying each technique once, in isolation, sufficient?
9. In Section 14.4's Test 6, a 100-token history matches a new turn's prefix perfectly, yet only 20 tokens are actually reused. Explain concretely why "the tokens matched" and "the tokens are reusable" diverge so sharply in this specific scenario.
10. Suppose a prefix-caching implementation reports its speedup using only the longest-common-prefix length, without checking whether the matched tokens are still physically present in the cache. Under what conditions would that reported figure most overstate the real speedup, based on Section 14.4's own findings?

## Where We Go Next

Chapters 13 and 14 together gave every future chapter a KV cache that pages efficiently, bounds its own growth at real scale, degrades gracefully rather than falling off a cliff as entries age, evicts by measured usefulness rather than blind age or a hard window alone, and avoids recomputing a conversation's own unchanged history on every turn. With state management settled, Part 4 turns from the engineering of the inference engine itself to running it against a real, published model: Chapter 15 begins with HuggingFace's model format and produces a first generated token from Qwen2.5, the point at which everything built so far — quantization, kernels, threading, and now KV cache management — comes together into a working system.

## Worked Solutions

**1.** Chapter 13.3's Ring Buffer needed sink-token protection because it could not otherwise distinguish "this token is being evicted because the buffer is full" from "this token happens to be one the model has learned to treat as a stable attention anchor" — the wrap logic would otherwise blindly overwrite whichever slot came next, sink token or not. Sliding window's entire design point is that NO token gets that kind of individual protection: the window boundary itself is the complete eviction policy, with no case-by-case judgment at all, so adding sink protection back in would contradict the very simplicity that makes sliding window sliding window rather than a re-implementation of Chapter 13.3's ring buffer with extra steps.

**2.** Mistral 7B was explicitly trained with a 4096-token sliding window in place, meaning every gradient update during training already reflected the constraint that attention could never reach further back than that — the model learned to compress anything it might need beyond the window into its hidden state as a normal part of training, so applying the window at inference time changes nothing relative to what the model has always experienced. A model trained with full, unrestricted attention never faced that constraint during training and never learned to compress anything into its hidden state for this reason, so imposing a window at inference time asks it to operate under a restriction its own training never prepared it for — usually tolerable in practice because attention weight decays with distance anyway, but not guaranteed the way it is for a model actually trained under the window.

**3.** A token's actual contribution to the current generation step's output is its attention weight multiplied by its Key/Value vector's value; that attention weight empirically shrinks as a token ages, so a quantization error introduced into an OLD token's vector gets multiplied by a small weight before it ever reaches the output, making the error's effect on the result correspondingly small. Applying the same coarse quantization to a NEW, heavily-weighted token would multiply that same-sized error by a large weight instead, producing a visible corruption of the output — the direction of the tiering (old gets coarse, new stays precise) is exactly what keeps the growing quantization error harmless rather than merely hidden.

**4.** The cache-management POLICY — inserting fresh entries at high fidelity, checking age against configurable thresholds, re-quantizing to progressively coarser tiers, and eventually evicting — would be completely identical either way, because that policy only cares about WHEN to re-quantize, not HOW the re-quantization itself is implemented. What would actually differ is the re-quantize operation's internals: this section's simplified quantizer dequantizes to float and re-encodes with a fresh symmetric-uniform scale, while Chapter 7's TurboQuant would dequantize using its trained Lloyd-Max codebook and re-encode with a coarser codebook sharing the same rotation matrix, producing a different (generally better, since Lloyd-Max codebooks are optimized for the actual data distribution) reconstruction error at each bit-width for the identical policy timeline.

**5.** With a 16-token block size, tokens 0-15 form block 0 and tokens 16-31 form block 1; the divergence at token 25 falls inside block 1, meaning block 1 contains both genuinely-matched tokens (16 through 24) and genuinely-diverged tokens (25 through 31) at once. Because a `KVBlock` can only be reused or discarded as a whole unit — there is no mechanism to keep part of a block's stored tokens while discarding the rest — the entire block gets discarded and recomputed once ANY of its tokens are past the divergence point, which is why tokens 16 through 24, despite matching perfectly, get swept into the recomputation along with the tokens that actually changed.

**6.** Reusing an individual matched token independent of its block would require pulling that one token's Key/Value data out of a block that otherwise needs to be discarded — but a block's storage has no mechanism for partial retention; the block manager's entire interface operates on whole blocks (allocate one, free one, reference one via a block table entry), because that whole-block granularity is precisely what makes block-table indirection O(1) and allocation-free in Chapter 13.2. Building token-level partial reuse would mean reintroducing the fine-grained, per-token bookkeeping that PagedAttention's block design was built specifically to avoid, trading away the very property (cheap, block-granularity indirection) that makes the KV cache manageable at scale in exchange for a reuse optimization that only pays off on the relatively rare tokens sitting in a straddling block.

**7.** This chapter's own source material described prefix caching only in prose, with a bare "speedup: 501x" arithmetic claim and no test code, model, or measurement methodology anywhere behind it. This book's standing rule is to never carry forward a performance number that was not itself computed and checked by code in the chapter, so rather than reproduce that unverified figure, this section built its own original prefix-caching implementation from Chapter 13.2's `BlockManager` and measured its own, real speedup and savings figures directly from working code — numbers that are smaller and more specific than "501x," but that are honestly the code's own.

**8.** Verifying a technique once, in isolation, proves it behaves correctly under the specific conditions its own test constructed — Section 14.1's sliding-window tests never ran alongside an H2O eviction policy or a re-quantization schedule, for instance. Re-checking the same guarantee inside the fully combined manager is a different claim: it confirms that none of the OTHER three techniques' bookkeeping perturbs this one's behavior when all four are genuinely running together, which unit tests run in isolation cannot show by construction — exactly the same lesson Chapter 13.5 drew from Chapter 11's discovery that individually thread-count-independent phases still produced a thread-count-dependent whole once combined.

**9.** The 100-token history matches the new turn's token IDs at every single position, so the longest common prefix is the full 100 tokens — a purely textual fact about the two ID sequences. But the OLD cache was simultaneously running a tight window-and-budget eviction policy throughout those same 100 steps, and by the time the new turn begins, that policy has already discarded all but 20 of the original entries to stay within budget; those 20 survivors are the only entries that ANY handoff mechanism can actually copy over without recomputing them, regardless of how much of the token sequence textually matches. The 100-token match describes what theoretically could have been reused if nothing had ever been evicted; the 20-entry reuse describes what the runtime cache genuinely still had on hand.

**10.** That reported figure most overstates reality precisely when a lot of eviction has happened to old, deep-prefix tokens by the time the new turn begins — exactly Section 14.4's own combined scenario, where a full prefix match coincided with 80 of 100 matched tokens already being gone from the cache. A conversation with little or no eviction pressure (a generous budget, a wide window, or a short history that never triggered maintenance) would show the theoretical and actual reuse counts nearly matching, as Section 14.4's own first handoff test demonstrated; the overstatement grows specifically with how aggressively the OTHER techniques in the combined system have been evicting the very tokens a long shared prefix depends on.
