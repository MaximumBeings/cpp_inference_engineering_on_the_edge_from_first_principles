# Chapter 8: The Memory Wall and the Roofline Model -- Why Arithmetic Is Rarely the Bottleneck

**What you will understand by the end of this chapter:**

- Why "how many FLOPs does this kernel do" is the wrong first question to ask about performance, and "how many bytes does it move, relative to those FLOPs" is the right one — derived here as arithmetic intensity, not asserted as folklore.
- How to derive a machine's peak compute and peak memory bandwidth from stated architectural parameters (core count, clock speed, SIMD width; memory channels, transfer rate) rather than quoting an unverified marketing number, and how those two peaks combine into a ridge point that separates memory-bound kernels from compute-bound ones.
- Why three real transformer kernels — RMSNorm, softmax, and RoPE — are all memory-bound by a wide margin, measured by instrumenting the actual kernel code to count every FLOP and every byte as it runs, not by looking the answer up.
- Why prefill (a GEMM, many tokens at once) and decode (a GEMV, one token at a time) are the same weight matrix and the same arithmetic, sitting on opposite sides of the same ridge point — and why quantization's speedup at one sequence length can vanish entirely at another.
- Why continuous batching's throughput gain is the exact same GEMV-to-GEMM transition, viewed from the scheduler's side instead of the sequence's — and why that gain comes with a latency cost past the same crossover point.
- Why, in a full decode step, the part of the cost that visibly grows with context length (attention against the KV cache) is not the part that actually dominates the bytes moved at realistic context lengths — the weight matrices, read fresh on every single step regardless of context, are.

**What you need to know first:**

- Chapter 3's std::mdspan vocabulary, for viewing a genuinely two- or three-dimensional structure — a weight matrix, a KV cache — as more than a flat buffer with hand-computed offsets; Sections 8.3 and 8.5 use it directly.
- Chapter 3.4's GQA attention and KV cache, and Chapters 2/4's Q4_0 block format, both reused verbatim in this chapter's later sections rather than redefined.
- This book's standing policy against fabricated timing numbers applies with particular force here: this chapter never runs a wall-clock benchmark, because a shared, virtualized build environment gives no number that would be reproducible on a rerun or on a reader's own machine. Every quantity in this chapter is either a stated, labeled architectural parameter or an exact, deterministic count of operations and bytes — FLOPs and bytes are integers a real kernel actually produces, not estimates a clock has to guess at.

---

Every kernel in this book so far has been judged by what it computes and how accurately. This chapter asks a different question: given a kernel that is already correct, what determines how FAST it can possibly run on real hardware — not "how fast did it run just now," which a shared sandbox cannot answer reproducibly, but "what is the hard ceiling imposed by the machine's own arithmetic and memory limits." Section 8.1 builds that ceiling, the roofline model, from first principles: peak compute and peak memory bandwidth, derived from stated architectural parameters, combining into a ridge point that separates kernels whose speed is capped by arithmetic from kernels whose speed is capped by how fast bytes can be fetched from memory. Section 8.2 measures where three real transformer kernels actually land relative to that ridge, by instrumenting them to count their own FLOPs and bytes as they run. Section 8.3 applies the same framework to the single most consequential distinction in an inference engine's performance profile — prefill versus decode — and finds the exact sequence length at which one becomes the other. Section 8.4 asks the same question from a serving system's point of view: continuous batching turns many decode requests' single-token GEMVs into one batched GEMM, the identical transition Section 8.3 analyzed, and this section asks what that means for latency, not just throughput. Section 8.5 closes the chapter with a full transformer layer's roofline profile, combining every kernel this book has built — RMSNorm, QKVO projections, GQA attention against a real KV cache, and the FFN — into one decode step, and asks which part of that step's memory traffic actually dominates.

## 8.1 The Roofline Model, Derived From First Principles

### Intuition

A kernel's speed is capped by two entirely different resources: how many arithmetic operations the processor can perform per second, and how many bytes it can move between memory and the processor per second. Whichever of the two a given kernel demands more of, relative to what it does with each byte it touches, is the one that actually limits it — and that ratio, FLOPs per byte, is a property of the KERNEL, while the two peaks are properties of the MACHINE. Neither number is worth much on its own; the useful question is always how they compare for one specific kernel on one specific machine.

### The Concept, In Detail

Peak compute is derived, not quoted: a core that supports AVX2 and fused multiply-add can retire `lanes_per_register * fma_ports_per_cycle * 2` FLOPs per cycle (the `2` because one FMA does a multiply and an add), and multiplying by core count and sustained clock speed gives the machine's peak FP32 GFLOP/s. Peak bandwidth is derived the same way: `channels * transfer_rate * bytes_per_transfer` gives peak GB/s for a stated memory configuration. Dividing the two gives the ridge point, in FLOPs per byte — the arithmetic intensity at which a kernel's demand for compute and its demand for bandwidth are exactly balanced. A kernel below the ridge point is memory-bound: however fast its arithmetic could run in isolation, it spends more time waiting on bytes than the processor spends computing on them, so its achievable throughput is `arithmetic_intensity * peak_bandwidth` — linear in arithmetic intensity, capped by bandwidth. A kernel above the ridge point is compute-bound: bytes arrive faster than the processor can consume them, so achievable throughput flattens at `peak_compute` regardless of how much higher arithmetic intensity climbs. This produces the chapter's first COMMON TRAP directly: a faster processor — more peak compute — does nothing whatsoever for a kernel that is memory-bound, because compute was never the limiting resource to begin with.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_roofline_and_ridge_point.cpp -o 01_roofline_and_ridge_point
./01_roofline_and_ridge_point
```

**Sample input:** an 8-core, AVX2, dual-channel DDR4-3200 machine's peak compute and peak bandwidth, derived from those stated parameters and cross-checked against the same arithmetic done independently; the achievable-throughput formula evaluated on both sides of the resulting ridge point, checking linear growth below it and a hard ceiling above it; and a deliberate demonstration of quadrupling peak compute for a memory-bound kernel.

@@OUT1@@

!!! warning "[COMMON TRAP] assuming a faster CPU always helps"
    Peak compute and peak bandwidth are independent resources, and a kernel is only ever limited by whichever one it demands more of. For a memory-bound kernel — one whose arithmetic intensity sits below the ridge point — the achievable-throughput formula reduces to `arithmetic_intensity * peak_bandwidth`, which does not contain the peak-compute term at all. Quadrupling peak compute for such a kernel changes its achievable throughput by exactly zero, not approximately zero: bandwidth was the entire bottleneck, and no amount of additional arithmetic capacity relieves a constraint the kernel was never bound by. "Upgrade to a machine with more FLOPs" is sound advice for a compute-bound kernel and completely useless advice for a memory-bound one, and the roofline model is what tells the two situations apart before any hardware purchase is made.

## 8.2 Measuring Arithmetic Intensity in Real Kernels

### Intuition

Section 8.1's ridge point is only useful once a kernel's own arithmetic intensity is known — and that number should come from instrumenting the kernel's actual control flow, not from a hand-derived formula that could quietly drift out of sync with what the code really does. RMSNorm, softmax, and RoPE are three of the cheapest, most frequently executed kernels in a transformer's forward pass, and all three turn out to sit far on the memory-bound side of any realistic ridge point — for the same underlying reason in each case.

### The Concept, In Detail

A `Cost` counter threaded through each kernel increments its FLOP and byte totals as the kernel's real loops execute, so the resulting arithmetic-intensity number is a direct consequence of the code, not a separately asserted claim about it. RMSNorm reads a vector and a weight vector, does a handful of FLOPs per element (a square, a running sum, a scale, a multiply), and writes the result — a small, fixed amount of arithmetic per byte moved, regardless of dimension, so it lands memory-bound at any realistic size. Softmax makes three full passes over its input (find the max, exponentiate and sum, divide by the sum), each pass moving as many bytes as the vector has elements, for an arithmetic cost per element that a transcendental function's real cost (counted here at a stated, labeled convention of eight FLOPs per `exp`, not treated as free) still does not come close to matching. RoPE rotates pairs of dimensions using precomputed cosine and sine tables — cheap per-pair arithmetic (six FLOPs) against the pair's own four bytes, plus the often-forgotten cost of reading the tables themselves. All three land far below any realistic ridge point, which is exactly why they are typically fused into neighboring kernels in a real engine rather than optimized in isolation — optimizing a memory-bound kernel's arithmetic buys almost nothing, per Section 8.1's own lesson.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_kernel_arithmetic_intensity.cpp -o 02_kernel_arithmetic_intensity
./02_kernel_arithmetic_intensity
```

**Sample input:** RMSNorm at dimension 4096, softmax at sequence length 2048, and RoPE at head dimension 128, each instrumented and classified against Section 8.1's derived ridge point; and a deliberate demonstration of forgetting to count RoPE's cosine/sine table reads as real memory traffic.

@@OUT2@@

!!! warning "[COMMON TRAP] forgetting a real memory access when counting bytes"
    The cosine and sine tables RoPE reads are easy to leave out of a byte count — they are "just a lookup," not the vector actually being transformed — but a lookup table read from memory is exactly as real a byte of traffic as the vector itself, especially when, as here, it is not reused across many calls but re-read for each one. Omitting it does not produce an obviously broken number; it produces an arithmetic-intensity figure that is quietly too high, because the same FLOP count is now being divided by too few bytes. At head dimension 128 this inflates the measured intensity by fifty percent — not enough to flip RoPE's classification here, since it is memory-bound by a wide margin either way, but for any kernel whose true arithmetic intensity sits close to the ridge point, an overstatement of this size is exactly enough to misclassify it as compute-bound when it is not.

## 8.3 Prefill vs. Decode: The Same Weights, Different Arithmetic

### Intuition

Prefill and decode use the identical weight matrix for the identical linear projection, and yet one is routinely described as compute-bound and the other as memory-bound. The difference is not the weights or the arithmetic — it is how many columns of input share that one weight read before it is done with. A GEMM (many token columns at once) and a GEMV (one column) are the same operation at two different points on the very same ridge point Section 8.1 derived, and the sequence length at which one becomes the other is a number this section finds by search, not by assumption.

### The Concept, In Detail

A linear layer's weight matrix, read once, can be reused across every column of a batch before the next matrix's weights are needed — so a batch of `seq` columns pays the weight-read cost exactly once while paying the FLOP cost `seq` times, and arithmetic intensity climbs linearly with `seq` until it crosses the ridge point. At `seq=1` (decode), that reuse buys nothing: the entire weight matrix is read to produce a single output column, which is why decode is memory-bound almost everywhere in practice. Quantizing the weight matrix to Q4_0 raises arithmetic intensity at any given `seq` (fewer bytes for the same FLOPs), which shifts the crossover to a SMALLER sequence length — but once a sequence is already long enough to be deep in the compute-bound regime for both formats, both are capped at the same peak-compute ceiling, and quantization's byte-count advantage stops mattering entirely. The weight matrix and the batch of activations are genuinely two-dimensional objects, viewed here through std::mdspan rather than through hand-computed row/column offsets, with a small real GEMM kernel run at reduced, illustrative dimensions specifically so its FLOP and byte counts can be checked against the closed-form formula this section then applies, unmaterialized, at the FFN projection's real (14336 x 4096) size.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_prefill_vs_decode.cpp -o 03_prefill_vs_decode
./03_prefill_vs_decode
```

**Sample input:** a real, std::mdspan-viewed FP32 and Q4_0 GEMM kernel at small illustrative dimensions, cross-checked against the closed-form cost formula it verifies; decode-time (seq=1) FP32-vs-Q4 byte and arithmetic-intensity comparison at the FFN projection's real 14336x4096 size; a search for the exact sequence length at which each format crosses from memory-bound to compute-bound; and a deliberate demonstration of quantization's throughput advantage at a sequence length well past both crossovers.

@@OUT3@@

!!! warning "[COMMON TRAP] assuming quantization's speedup holds at any sequence length"
    "Quantization gives roughly a 4x speedup" is a claim about the DECODE regime, where reading four times fewer weight bytes for the same arithmetic directly translates into a proportionally faster memory-bound kernel. Once the sequence length grows past both formats' compute-bound crossover, that logic no longer applies: both FP32 and Q4 are capped at the same peak-compute ceiling, and the achievable-throughput formula does not depend on arithmetic intensity at all once a kernel is compute-bound. Quantizing the weights of an already deeply compute-bound GEMM changes memory traffic that was never the bottleneck to begin with — exactly Section 8.1's trap, in a new form. The speedup is real and large at decode time; it is measured here to be exactly zero at a sufficiently long, already compute-bound sequence length, and the honest claim is "quantization speeds up decode," not "quantization speeds up inference" unconditionally.

## 8.4 Continuous Batching: the Same Crossover, a Different Question

### Intuition

A production inference server rarely runs one request's GEMV at a time. Continuous batching stacks many DIFFERENT requests' single decode tokens into one batched matrix multiply, sharing the same weight read across all of them — which is mathematically the identical GEMV-to-GEMM transition Section 8.3 analyzed for one request's own prompt tokens, just triggered by the scheduler instead of by prompt length. Reusing that section's own formula unchanged, with `seq` reinterpreted as batch size, this section asks the question prefill's analysis did not need to: what happens to each individual REQUEST's latency, not just the batch's aggregate throughput, on both sides of the same crossover.

### The Concept, In Detail

Below the batch-size crossover, a batch is memory-bound, and its total processing time is dominated by the one shared weight read — so doubling, or even multiplying by eight, the number of requests sharing that batch barely changes how long the batch takes to process, because the dominant cost did not change. Every request in a synchronously-served batch waits for the WHOLE batch to finish before its own next token is ready, so that near-flat batch time is also, directly, each request's own per-token latency — batching in this regime is close to free. Past the crossover, the batch is compute-bound, and batch time grows roughly linearly with batch size, since the shared-weight-read discount no longer has anywhere left to apply: FLOPs, not bytes, are now the limit, and FLOPs scale with every additional request. Aggregate throughput (tokens served per unit time) keeps growing on both sides of the crossover, but far more slowly past it than before — while each request's own latency, tied directly to the now-linearly-growing batch time, gets steadily worse the more requests are piled into an already compute-bound batch. This is the same shape as Section 8.1's and 8.3's traps, applied to a scheduling decision rather than a hardware or format choice.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_continuous_batching.cpp -o 04_continuous_batching
./04_continuous_batching
```

**Sample input:** batch processing time at batch size 1 versus batch size 8, well below the crossover; a search for the batch-size crossover B* of the same FFN projection Section 8.3 used; and a deliberate comparison of aggregate throughput and per-request latency at a batch size well past B*, against a batch size still below it.

@@OUT4@@

!!! warning "[COMMON TRAP] assuming more batching always helps throughput proportionally"
    Below the batch-size crossover, adding requests to a batch is close to free — batch time barely grows, so throughput grows almost proportionally with batch size, and it is tempting to extrapolate that relationship indefinitely. Past the crossover, the batch is compute-bound, and batch time grows roughly in step with batch size instead of staying flat — so throughput growth falls sharply behind batch-size growth, while every request's own per-token latency, tied to that now-growing batch time, gets measurably worse. A scheduler that keeps adding requests to an already compute-bound batch is not buying the throughput gain it bought earlier; it is trading request latency for a shrinking throughput return, the same trade a memory-bound-to-compute-bound crossover always forces once bytes stop being the bottleneck.

## 8.5 A Full Layer's Roofline Profile

### Intuition

A single decode step touches far more than one kernel: two RMSNorms, four attention projections, attention itself against a growing KV cache, and three FFN projections. It is tempting to assume that because attention's cost visibly GROWS with context length, it must be the part that dominates a long-context decode step's memory traffic. Summing every kernel's own cost formula from this chapter shows that, at realistic context lengths, it is not — the weight matrices, read completely fresh on every single step regardless of how long the conversation has been, are far larger.

### The Concept, In Detail

Every projection's cost reuses Section 8.3's verified linear-layer formula directly. Attention's cost is new to this section but built to match, term for term, a real GQA attention kernel: each KV head's cached rows are read once and reused by every query head in its group — the same "read once, reuse across the group" shape Section 8.3 used for weight matrices — while the score dot products, softmax, and weighted sum are real per-element arithmetic paid by every query head, verified here against Chapter 3.4's own std::mdspan-viewed KVCache and gqa_attention, reused verbatim with a cost counter threaded through it. Summing all of these for one decode step at a realistic context length confirms the whole step lands memory-bound, consistent with Section 8.1's own choice of a representative decode arithmetic intensity. Separating the step's bytes into the part that is FIXED (the QKVO and FFN weight matrices, read in full on every step regardless of context length) and the part that GROWS with context (the KV cache read) shows the fixed part dominating by more than an order of magnitude at a two-thousand-token context, and finds the context length at which the two would actually cross to be far beyond most deployed context windows.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 05_full_layer_roofline_profile.cpp -o 05_full_layer_roofline_profile
./05_full_layer_roofline_profile
```

**Sample input:** the closed-form attention cost formula checked against a real, std::mdspan-viewed GQA kernel reused from Chapter 3.4, at small dimensions; a full decode step at DIM=4096, 32 query heads sharing 8 KV heads (GQA group 4), D_FF=14336, and context length 2048, classified against the roofline; and a deliberate comparison of the step's fixed weight-read bytes against its context-dependent KV-cache-read bytes at that same context length.

@@OUT5@@

!!! warning "[COMMON TRAP] assuming attention dominates a long-context decode step"
    Attention's KV-cache read is the part of a decode step's cost that visibly grows with context length, which makes it the natural first place to look when a long-context deployment is slow — but "the part that grows" and "the part that dominates" are not the same claim. At a two-thousand-token context, the fixed QKVO-and-FFN weight-read cost, paid again in full on every single decode step no matter how long the conversation has been, outweighs the KV-cache read by more than fifty times, because the weight matrices are simply that much larger than one step's worth of cached keys and values. The KV-cache read only catches up to the fixed weight cost at a context length measured in the hundreds of thousands of tokens — far beyond most deployed context windows. Profiling effort spent shrinking attention's footprint at ordinary context lengths addresses the part of the bottleneck that is easiest to see, not the part that is actually largest.

## Chapter Summary

This chapter built the roofline model from first principles — peak compute and peak bandwidth derived from stated architectural parameters, not quoted from a data sheet, combining into a ridge point that separates memory-bound kernels from compute-bound ones — and then applied it, section by section, to increasingly complete pieces of a real inference engine. Section 8.1 showed that a faster processor does nothing for a memory-bound kernel, because the two peaks are independent resources and a kernel is only ever limited by whichever one it demands more of. Section 8.2 measured three real transformer kernels' own arithmetic intensity by instrumenting their actual control flow, finding RMSNorm, softmax, and RoPE all memory-bound by a wide margin, and showed how easily an incomplete byte count inflates that measurement. Section 8.3 showed that prefill and decode are the identical weight matrix and arithmetic sitting on opposite sides of the same ridge point, found the exact sequence length where one becomes the other, and showed quantization's speedup vanishing once a sequence is deep enough into the compute-bound regime. Section 8.4 showed that continuous batching is the identical crossover viewed from a scheduler's perspective, and that the throughput gains it provides below the crossover become a latency cost above it. Section 8.5 closed the chapter by summing every kernel this book has built into one full decode step's roofline profile, and showed that the part of the step's cost that visibly grows with context — attention against the KV cache — is not the part that dominates its memory traffic at realistic context lengths; the weight matrices, read fresh on every step regardless of context, are. Part 1 asked what a model's numbers should be represented as; Part 2, which this chapter opens, asks how fast an engine can actually move those numbers through a CPU, and the roofline model built here is the framework every remaining optimization decision in Part 2 will be checked against.

## Self-Check Questions

1. Why does quadrupling a machine's peak compute change a memory-bound kernel's achievable throughput by exactly zero, not merely by a small amount?
2. Explain, in terms of the achievable-throughput formula, why a kernel's classification as memory-bound or compute-bound depends on both the kernel's own arithmetic intensity and the machine's ridge point, not on the kernel alone.
3. Section 8.2 counts a transcendental function like `exp` as costing a stated eight FLOPs, not one. Why does treating it as free (one FLOP, or zero) distort a kernel's measured arithmetic intensity, and in which direction?
4. What specifically was forgotten in Section 8.2's COMMON TRAP, and why does omitting it inflate arithmetic intensity rather than deflate it?
5. Why are prefill and decode described as "the same weight matrix, different arithmetic" rather than as two fundamentally different kernels?
6. Explain why quantizing a weight matrix shifts a GEMM's memory-bound-to-compute-bound crossover to a SMALLER sequence length, and why that same quantization stops helping at all once the sequence is comfortably past the crossover on both sides.
7. In Section 8.4, why is a batch's total processing time also each individual request's own per-token latency, and what does that fact depend on about how the batch is served?
8. Explain why continuous batching's throughput benefit shrinks, rather than continuing proportionally, once a batch's size passes the compute-bound crossover.
9. In Section 8.5, why is the KV-cache read described as "the part that visibly grows with context" while the QKVO/FFN weight reads are described as "fixed," and why does that distinction matter for where a long-context deployment's actual bottleneck is likely to be?
10. Section 8.3's real GEMM kernel runs at small, illustrative dimensions rather than at the FFN projection's real 14336x4096 size. What is that small kernel actually used to verify, and why does the chapter trust the closed-form formula at the full size as a result?

## Where We Go Next

This chapter established the framework — the roofline model — that every remaining optimization in this book will be checked against: is a given kernel memory-bound or compute-bound, and does a proposed change actually address the resource that is limiting it. Chapter 9 puts that framework to work on the compute side of the ridge, covering SIMD vectorization — AVX2 on x86 and NEON on Arm — the mechanism by which a compute-bound kernel's peak FLOP rate is actually achieved in practice, and, just as importantly, the mechanism that determines how large a MEMORY-bound kernel's peak bandwidth figure really is once real cache and prefetch behavior are accounted for.

## Worked Solutions

**1.** The achievable-throughput formula for a memory-bound kernel is `arithmetic_intensity * peak_bandwidth` — an expression that does not contain the peak-compute term at all, because a memory-bound kernel's arithmetic intensity, by definition, sits below the ridge point where compute would first become the limit. Quadrupling peak compute changes a quantity that this formula never depended on in the first place, so the change is not merely small; it is exactly zero, because bandwidth was the entire bottleneck and no additional arithmetic capacity relieves a constraint that was never binding.

**2.** Whether a kernel is memory-bound or compute-bound is decided by comparing the kernel's own arithmetic intensity against the RATIO of the machine's two peaks (the ridge point), not against either peak in isolation. The identical kernel, with the identical arithmetic intensity, can be memory-bound on one machine and compute-bound on another if the two machines have different ridge points — so "is this kernel memory-bound" is never a question about the kernel alone; it is a question about the kernel running on a SPECIFIC machine.

**3.** Treating `exp` as one FLOP (or as free) understates the true arithmetic cost of every element softmax touches, which understates the numerator of the arithmetic-intensity ratio (FLOPs) while leaving the denominator (bytes) unchanged. This makes the kernel look MORE memory-bound than it actually is — its true arithmetic intensity is higher than the undercounted figure suggests, because the real hardware cost of a vectorized `exp` is measured, in practice, to be several times a single FLOP, not one.

**4.** Section 8.2's trap forgot to count the bytes read from the cosine and sine lookup tables RoPE depends on, counting only the rotated vector's own read and write. Omitting a real memory access removes bytes from the denominator of the arithmetic-intensity ratio while leaving the FLOP numerator unchanged, which makes the computed ratio LARGER than the true one — an inflation, not a deflation, and specifically dangerous for any kernel whose true intensity sits close to the ridge point, since an inflated number is exactly what could push a genuinely memory-bound kernel across the line into an apparent, incorrect compute-bound classification.

**5.** Prefill and decode invoke the identical weight matrix through the identical linear-algebra operation — a matrix multiplied by a batch of columns — differing only in how many columns that batch contains (many prompt tokens for prefill, one generated token for decode). Because the weight matrix is read once and reused across every column in the batch, the SAME kernel's arithmetic intensity rises with batch size; prefill and decode are two points on that single curve, not two different kernels with two different formulas.

**6.** Quantizing the weight matrix reduces the bytes the weight-read term of the cost model contributes without changing the FLOP count at all, which raises arithmetic intensity at every sequence length — including smaller ones — so the sequence length at which that raised intensity first reaches the ridge point is smaller than it was for the unquantized format. Once a sequence length is comfortably past BOTH formats' crossover, however, both are compute-bound, and the achievable-throughput formula in that regime is simply `peak_compute`, a constant that does not depend on arithmetic intensity (and therefore not on the format's byte count) at all — so quantization's byte-count advantage, which only mattered while bytes were the bottleneck, stops producing any speedup whatsoever.

**7.** A batch of decode requests served synchronously means no request's next token is available until the entire batch's GEMM has finished computing every request's output column — there is no mechanism by which one request's token could be ready earlier than another's within the same batch. Because of this, the roofline-predicted time to process the whole batch IS the time every individual request must wait for its own next token, so "batch processing time" and "per-request latency" are the same measured quantity, not two quantities that happen to correlate.

**8.** Below the compute-bound crossover, batch processing time grows far more slowly than batch size (the shared weight read dominates and barely changes), so throughput — requests served per unit time — grows almost as fast as batch size does. Past the crossover, the shared-weight-read discount has nowhere further to apply, since FLOPs, not bytes, are now the limiting resource, and FLOPs scale directly with the number of requests in the batch — so batch time now grows roughly in proportion to batch size, and throughput's growth falls correspondingly far behind the batch size's own growth, rather than continuing at the earlier, near-proportional rate.

**9.** The KV cache accumulates one more cached position with every token generated, so the number of bytes attention must read from it grows with the length of the conversation so far; the QKVO and FFN weight matrices are a fixed size set once at model-load time and are read in their entirety on every decode step regardless of how many tokens have been generated. This distinction matters because it is tempting to assume that whatever grows with context length must eventually dominate the cost — and it eventually does, but only at a context length this section shows to be far longer than most real conversations reach, meaning the FIXED weight-read cost is the actual dominant term across the range of context lengths most deployments will ever see.

**10.** The small kernel is used to verify that the closed-form cost FORMULA — the arithmetic used to predict FLOPs and bytes without materializing a buffer — exactly matches what a real, running kernel measures at an identical shape, both in total FLOPs and in total bytes. Because the formula is pure arithmetic over shape parameters (n_out, n_in, seq), a match at one shape is a match at every shape the same formula is evaluated at — so once the small-scale check passes, the chapter can trust the same formula applied to the FFN projection's real, much larger dimensions without needing to allocate and run a kernel over a buffer of that full size just to re-confirm arithmetic already shown to be correct.
