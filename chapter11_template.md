# Chapter 11: Parallelizing the Forward Pass -- Row-Parallel, Head-Parallel, and Reproducible Across Thread Counts

**What you will understand by the end of this chapter:**

- The two structurally different ways to split a matrix-vector product across threads — row-parallel (partition the OUTPUT, no combination step, bit-for-bit reproducible against a serial reference) and column-parallel (partition the INPUT, requires a combination step whose order determines the result) — and a third, well-defined-but-not-reproducible way to combine column-parallel's partial sums that this book will not lock into its own checked output.
- How to generalize Chapter 10.3's single-purpose GEMV thread pool into a persistent pool that runs ANY row-partitioned task, so one pool, created once, carries an entire transformer layer's several different matmul phases — and the real, structural constraint on when independent matmuls may be BATCHED into a single parallel phase together.
- Why parallelizing GQA attention by query head is embarrassingly parallel for correctness, but why thread boundaries that ignore KV-group boundaries cause a shared KV head's cache rows to be read redundantly by more than one thread — and how aligning boundaries to whole KV groups fixes it.
- Why "reproducible across repeated runs at a FIXED thread count" — this book's entire discipline since Chapter 10 — is a strictly weaker claim than "reproducible across a CHANGE in thread count," and the fixed-block reduction technique that actually delivers the stronger guarantee.
- How this chapter's four earlier techniques combine into one complete, real decode step whose output is bit-for-bit identical regardless of how many worker threads compute it — and how a single non-thread-count-independent reduction, left anywhere in that pipeline, is enough to break the guarantee for the whole thing.

**What you need to know first:**

- Chapter 10's `std::barrier`-based persistent thread pool and its row-parallel, bit-for-bit verification discipline, extended in this chapter rather than replaced.
- Chapter 3.1's RMSNorm, 3.2's SwiGLU FFN, and 3.4's GQA attention with its `std::mdspan`-backed KVCache — reused verbatim throughout this chapter as the real kernels being parallelized, never re-derived.
- This chapter continues Chapter 10's `-pthread` requirement, and every file touching the mdspan-backed KVCache also needs the vendored mdspan flags Chapter 8 established.
- A hazard this chapter's own authoring surfaced, not merely one described in the abstract: identical source, compiled once for x86_64 and once for aarch64, silently disagreed about serial-vs-parallel bit-exactness, purely because of where the compiler's default FMA-contraction heuristic chose to fuse a multiply and an add. Every file in this chapter compiles with `-ffp-contract=off` because of this, and Section 11.2 documents the discovery — what broke, why, and how it was found — in full.

---

Chapter 10 built the primitives — atomics, mutexes, a barrier-synchronized thread pool, an awareness of false sharing, a work-stealing scheduler — and verified a single GEMV split across them. This chapter spends those primitives on the actual forward pass this book has built one real kernel at a time since Part 0. Section 11.1 names the two fundamental ways to split a matrix-vector product and shows exactly what each one costs and guarantees. Section 11.2 generalizes Chapter 10.3's fixed-shape thread pool into one that can run an entire layer's several different matmul phases, and documents a genuine, hard-won lesson about floating-point reproducibility across CPU architectures along the way. Section 11.3 parallelizes GQA attention itself, across query heads, and finds a real memory-locality hazard in how naively that parallelization is usually done. Section 11.4 confronts a harder version of this book's reproducibility standard: not merely "the same answer on a rerun," but "the same answer no matter how many threads happen to be available." Section 11.5 closes the chapter, and Part 2, by combining every one of these techniques into one complete, real, multi-layer decode step — verified bit-for-bit identical across a change in thread count, and verified just as rigorously to break the moment even one piece of it does not honor that standard.

## 11.1 Row-Parallel vs. Column-Parallel Matmul

### Intuition

A matrix-vector product can be split across threads two structurally different ways: by OUTPUT row, or by INPUT column. Chapter 10.3 already used the first without naming it as a choice. This section names both, and shows precisely what each one guarantees and what each one costs — because the difference is not a matter of taste, it changes whether the result is bit-for-bit reproducible at all.

### The Concept, In Detail

Row-parallel assigns each thread a disjoint range of OUTPUT rows; every thread computes its rows' dot products completely independently, using its own full copy of the input vector, and writes to elements no other thread ever touches. Because no combination step exists at all, and because each row's own accumulation runs in exactly the same left-to-right order the serial reference uses, the result is bit-for-bit identical to serial — exactly Chapter 10.3's finding, now given its proper name. Column-parallel instead assigns each thread a disjoint range of INPUT columns; every thread computes a FULL n_out-length partial output using only its slice of the input, and after joining, those n_threads partial vectors must be combined element-by-element into the final output. That combination's order matters, because floating-point addition is not associative: combining the partial vectors in a FIXED sequential order (thread 0's, then thread 1's, and so on), after every thread has written into its own exclusively-owned buffer with no shared mutable state during the parallel phase itself, is well-defined and reproducible against itself across repeated calls at a fixed thread count — though it lands only within a tight tolerance of the serial reference, not bit-identical to it, since the summation order genuinely differs from the serial reference's own. A third combination strategy exists and is worth naming precisely because it is tempting: skipping the combine step entirely by having every thread call `std::atomic<float>::fetch_add` directly on the shared output during the parallel phase. This is well-defined by the C++ standard — no lost updates, no data race, no undefined behavior of any kind — but it is NOT reproducible run to run, because the arrival order of concurrent `fetch_add` calls is not fixed by anything the standard promises, and floating-point addition's non-associativity means that unfixed order can change the exact final bit pattern. This book includes that function as real, compilable source and never executes it from this file's own checked `main()`, for the identical honesty reason Chapter 10.2 never executed its racy counter. Column-parallel also pays a real, measured, recurring cost that row-parallel never does: a combination buffer of `n_threads * n_out` floats, allocated and summed again on every single call, in every layer, on every decode step.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 01_row_vs_column_parallel_matmul.cpp -o 01_row_vs_column_parallel_matmul
./01_row_vs_column_parallel_matmul
```

**Sample input:** a 64-row, 256-column matrix split across 4 threads two ways — row-parallel, checked bit-for-bit against a serial reference with no combination step at all, and column-parallel with a fixed-order combine, checked within a `1e-3` tolerance against the same serial reference and additionally checked bit-for-bit reproducible against a second call to itself; and a direct measurement of column-parallel's real combination-buffer memory cost against row-parallel's zero extra bytes.

@@OUT1@@

!!! warning "[COMMON TRAP] combining column-parallel's partial sums with a shared atomic instead of a fixed order"
    `std::atomic<float>::fetch_add` on a shared output element, called by every thread directly during the parallel phase, looks like the simplest possible way to combine column-parallel's partial results — no separate combine step, no per-thread buffer. It is also completely well-defined: the C++ standard guarantees no lost updates and no data race. What it does NOT guarantee is which thread's `fetch_add` lands first, and because floating-point addition is not associative, a different arrival order can produce a different final bit pattern on a different run of the identical program on the identical machine. This is a genuinely different hazard from Chapter 10.2's data race — there is no undefined behavior here at all — and it is exactly why this section's checked, locked output uses the fixed-order combine instead: reproducibility, not just well-definedness, is the standard this book holds every locked number to.

## 11.2 A Persistent Pool Across a Full Layer's Phases

### Intuition

Chapter 10.3's `TensorThreadPool` ran exactly one task shape, forever: a GEMV, every round. A real transformer layer needs that same persistent pool to run several DIFFERENT task shapes back to back within a single token — the Q/K/V projections, the output projection, the FFN gate/up projections, and the FFN down projection — with RMSNorm, attention, and SwiGLU's elementwise gate running sequentially in between. This section generalizes the pool from "always run a GEMV" to "run any row-partitioned task the caller hands it."

### The Concept, In Detail

The generalized pool's task descriptor becomes an arbitrary `std::function<void(size_t,size_t)>` — "compute this row range, however the caller defines that" — so the SAME pool instance, created once, runs a Q/K/V projection phase this round and an FFN down-projection phase the next, with no new threads and no new synchronization primitives created in between. This section verifies that generalization two ways: one full layer's worth of phases, run through the pool once, matches a serial reference bit-for-bit (every phase is still row-parallel with disjoint writes, so Chapter 10.3's bit-exactness carries over unchanged); and the SAME still-alive pool, run across six consecutive layers with six different weight sets, still matches a serial reference applied the same six times — proving the pool genuinely carries a multi-phase, multi-layer forward pass, not merely one repeated task shape. A second, independent point this section makes concrete: independent matmuls can only be BATCHED into a single parallel phase — one pair of barrier arrivals instead of several — when they share the same output row count. Q, K, and V all project into the same DIM-row space, so one phase computes all three; the FFN gate and up projections both produce D_FF rows, so they batch together too — but D_FF is not DIM, so gate/up cannot be batched into the same phase as the output or down projections without silently leaving rows uncomputed. A third hazard surfaced only while cross-checking this file on aarch64 during authoring: the very first version of this file matched serial-vs-parallel bit-for-bit on x86_64 but NOT on aarch64, even though both platforms were internally deterministic across repeated runs on each platform separately. The cause was GCC's default `-ffp-contract=fast`, which permits fusing `a*b+c` into one rounding step (a hardware FMA) wherever the compiler judges it profitable — a judgment made per call site, after inlining, so the identical accumulation, textually written once, can get fused on one architecture where scalar FMA is always available (aarch64) and stay unfused on another where it is not, absent an explicit `-mfma` flag (this book's baseline x86_64 build). Passing `-ffp-contract=off` restores the literal, separate multiply-then-add the source asks for on every architecture, and every file in this chapter compiles with it as a result.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_persistent_pool_full_layer.cpp -o 02_persistent_pool_full_layer
./02_persistent_pool_full_layer
```

**Sample input:** a real, small GQA transformer layer (DIM=64, D_FF=176, 8 query heads sharing 2 KV heads), its RMSNorm, QKV projections, GQA attention, output projection, and SwiGLU FFN all reused verbatim from Chapters 3.1/3.2/3.4; one full layer run through a persistent `ParallelPool` and checked bit-for-bit against a serial reference; the SAME still-alive pool reused across six consecutive layers, checked bit-for-bit the same way; and a deliberate demonstration of a batched phase told the wrong shared row count, leaving a checkable, exact number of FFN rows uncomputed.

@@OUT2@@

!!! warning "[COMMON TRAP] batching phases whose output row counts do not match"
    Batching the Q, K, and V projections into one parallel phase works because all three matmuls happen to produce output vectors of the same length, so a single row range is valid for every one of them at once. It is tempting to extend that same batching instinct to any two matmuls that are both "parallel phases" — for instance, telling the FFN gate/up phase it has DIM rows when it actually has D_FF, as if copy-pasting the output-projection phase's row count into the wrong place. The pool has no way to know the caller made this mistake: it faithfully partitions and runs exactly the row count it was told, and the rows past that count — here, `D_FF - DIM` of them, in both the gate and up projections — are simply never assigned to any thread and never computed, left at whatever the output buffer contained beforehand. The fix is not a smarter pool; it is remembering that batching is only valid when every matmul folded into one phase genuinely shares the same output row count.

## 11.3 Head-Parallel Attention

### Intuition

Section 11.2 kept attention itself sequential. Attention is, in fact, the most embarrassingly parallel phase in the whole layer: GQA computes an entirely independent score vector, softmax, and weighted sum for every query head, writing to a disjoint slice of the output — no combination step, no shared mutable state between heads. The real design question is not whether attention can be parallelized by head, but how heads should be assigned to threads, given that GQA's whole premise is several query heads SHARING one KV head's cached rows.

### The Concept, In Detail

Partitioning query heads evenly by raw INDEX, ignoring which KV head each one belongs to, routinely splits a single KV head's group across two different threads — with 8 query heads sharing 2 KV heads in a group of 4, even an evenly-divided 4-thread partition (2 heads per thread) puts heads 0-1 and heads 2-3 in different threads despite BOTH pairs belonging to KV head 0's group, so KV head 0's cache rows get read by two different threads instead of one. This section counts that redundancy directly — for a given partition, which KV head index each thread's assigned heads resolve to, and how many DISTINCT threads end up touching each one — rather than timing anything, the same structural-arithmetic verification Chapter 10.3's Test 1 used for its own row partition, before ever running a single thread. The fix assigns whole KV GROUPS to threads instead of raw query-head ranges: ceiling-divide the KV HEAD count across threads, then expand each thread's KV-head range back into the query heads that belong to it. Every KV head's group is now owned by exactly one thread, because a KV head only ever belongs to one range in this scheme — verified both structurally (zero KV heads touched by more than one thread) and functionally (the real threaded computation using this partition still matches the serial reference bit-for-bit, so the fix costs nothing in correctness). Every head's own full computation — scores, softmax, weighted sum — is written as a single free function, `attend_one_head`, called identically from the serial loop and from every parallel partition tried in this section, applying Section 11.2's own lesson about not giving the compiler two textually different call sites to treat differently.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_head_parallel_attention.cpp -o 03_head_parallel_attention
./03_head_parallel_attention
```

**Sample input:** 8 query heads sharing 2 KV heads (group size 4) over a 10-position cache; head-parallel attention with a naive 4-thread, per-index partition checked bit-for-bit against a serial reference; that same naive partition's KV-head redundancy counted directly (2 of 2 KV heads touched by more than one thread); and a group-aligned 2-thread partition checked both to eliminate the redundancy entirely (0 KV heads touched by more than one thread) and to still match the serial reference bit-for-bit.

@@OUT3@@

!!! warning "[COMMON TRAP] partitioning query heads by index instead of by KV group"
    An even, per-index partition of query heads across threads is completely CORRECT — every test in this section confirms it matches the serial reference exactly, no matter how the heads are divided up, because each head's computation is fully independent of every other head's. What an even index partition does not protect is memory locality: with more query heads than KV heads (GQA's entire premise), a thread boundary drawn by raw head index has no reason to land on a KV-group boundary, and when it does not, the KV head whose group straddles that boundary gets its cache rows read once per thread that touches any part of its group, rather than once total. The fix — assigning whole KV groups to threads instead of raw head ranges — is not a correctness fix (both partitions are equally correct); it is a structural fix to how many times the same cached bytes get fetched, verified here as a direct count of KV-head touches rather than left as an unverified assumption about "obviously" parallel work.

## 11.4 Deterministic Floating-Point Reduction, Independent of Thread Count

### Intuition

Combining per-thread-ID slots in fixed thread order — Section 11.1's own fixed-order combine, and the technique this book has relied on since Chapter 10 for "reproducible" — is reproducible run to run FOR A GIVEN thread count. This section asks the harder question that guarantee was never tested against: is it reproducible when the THREAD COUNT ITSELF changes? A real engine does not always run with the same thread count, and floating-point addition is not associative, so a different thread count means different chunk boundaries, which means a different summation order, which can mean a different final bit pattern — even though every run is still numerically correct.

### The Concept, In Detail

A per-thread-indexed reduction chunks the input by `ceil(n / n_threads)`, so changing `n_threads` moves every chunk boundary, changing which elements get summed together first in each thread's own partial sum — and this section confirms empirically that this really does change the final bit pattern: across six tested thread counts (1, 2, 3, 4, 5, 8) on a 997-element dot product, five distinct bit patterns appear, even though every single one of them is reproducible on its own if that same thread count is used again. The fix generalizes "write to your own slot, combine in fixed order" from per-THREAD slots to per-BLOCK slots, where block boundaries are a FIXED CONSTANT (16 elements per block here) chosen independently of how many threads happen to be running. A block's own internal sum is always computed the same way — left-to-right over that block's fixed element range — no matter which thread is assigned to compute it, and the final combination always walks the same fixed sequence of block indices, 0 through the last block, regardless of thread count. However many threads did the work, and however the blocks were divided up among them, the exact same set of intermediate sums gets combined in the exact same order every time — verified here as a BIT-IDENTICAL result across all six tested thread counts, where the thread-indexed version produced five different ones.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread 04_deterministic_reduction.cpp -o 04_deterministic_reduction
./04_deterministic_reduction
```

**Sample input:** a 997-element dot product (deliberately not a clean multiple of any tested thread count or of the fixed 16-element block size), reduced with fixed-size blocks across thread counts 1, 2, 3, 4, 5, and 8 and checked bit-identical across all six; the same dot product reduced with per-thread-indexed slots across the identical six thread counts, checked reproducible for a single fixed thread count but shown to produce multiple distinct bit patterns across the six.

@@OUT4@@

!!! warning "[COMMON TRAP] mistaking \"reproducible\" for \"reproducible under any configuration\""
    A test that creates a pool with one fixed thread count, runs a reduction against it twice, and confirms the two results match — exactly what Chapter 11.1 checked, and exactly what this section's own Test 3 checks first — genuinely does verify something real: the reduction is not internally flaky. It does NOT verify that the SAME reduction would produce the SAME result if the deployment happened to run with a different thread count next time, and this section shows concretely that the per-thread-indexed technique does not have that stronger property, even though it passes every single-thread-count reproducibility check perfectly. Only a reduction whose intermediate boundaries are fixed independently of thread count — this section's block scheme — can honestly claim reproducibility across a configuration change, and the two claims should never be described with the same word without checking which one was actually tested.

## 11.5 A Fully Multi-Threaded Decode Step

### Intuition

Every technique this chapter built, combined into one real, complete decode step — and checked against a stricter standard than any single section needed on its own: the exact same result, bit-for-bit, no matter how many worker threads compute it.

### The Concept, In Detail

Section 11.2's generalized pool and row-parallel matmul phases, and Section 11.3's group-aligned head-parallel attention, are already thread-count-independent by construction — both assign disjoint OUTPUT units (rows, or heads) to threads and never reorder any single row's or head's own internal accumulation, so however many threads share the work, the result cannot depend on the count. RMSNorm's own reduction is the one piece of a full layer that is a genuine cross-element reduction, and this section uses Section 11.4's fixed-block technique specifically so it does not undo the guarantee every other phase already provides for free. Combined — a persistent pool carrying a parallel RMSNorm reduction, a batched QKV projection phase, group-aligned head-parallel attention, an output projection phase, a batched FFN gate/up phase, and an FFN down-projection phase, run across four full transformer layers — the whole decode step matches a serial reference bit-for-bit with 4 worker threads, and matches it again, still bit-for-bit, when the exact same pipeline runs with 3 worker threads through a freshly created pool instead. This is the chapter's real payoff: not merely correct, and not merely reproducible on a rerun, but reproducible ACROSS a configuration change, because every phase that combines partial results was built with that specific property in mind from the start. The final test makes the fragility of that property concrete: swapping ONLY the RMSNorm reduction for Section 11.4's thread-indexed alternative — leaving every matmul phase and every attention head exactly as row/head-parallel as before — is enough to make the WHOLE decode step's final output depend on thread count again, because the broken reduction's result feeds into the residual stream that every later phase in every later layer builds on. One non-thread-count-independent piece, anywhere in an otherwise perfectly parallel pipeline, is enough to break the guarantee for the entire pipeline.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 05_full_decode_step_multithreaded.cpp -o 05_full_decode_step_multithreaded
./05_full_decode_step_multithreaded
```

**Sample input:** a real 4-layer GQA decode step (DIM=64, D_FF=176, 8 query heads sharing 2 KV heads), combining a fixed-block-reduced parallel RMSNorm, batched QKV projections, group-aligned head-parallel attention, an output projection, batched FFN gate/up projections, and an FFN down projection through one persistent pool; checked bit-for-bit against a serial reference at 4 threads, checked bit-for-bit again at 3 threads through a freshly created pool, and checked to DIFFER across those same two thread counts once RMSNorm's reduction alone is swapped for the non-thread-count-independent alternative.

@@OUT5@@

!!! warning "[COMMON TRAP] assuming perfectly parallel phases guarantee a perfectly parallel pipeline"
    Every matmul phase in this file is row-parallel, and attention is head-parallel, and both are thread-count-independent by construction — true before Test 3 and still true during it. It is tempting to conclude that a pipeline built entirely out of such phases must itself be thread-count-independent as a whole. Test 3 shows that conclusion does not follow: a single reduction — one piece, appearing twice per layer, only mixed AMONG otherwise-perfect parallel phases — feeds its result into the residual stream that every subsequent projection, every subsequent attention call, and every subsequent layer builds on, so that one non-thread-count-independent result propagates and contaminates everything computed afterward. A pipeline's overall reproducibility guarantee is only as strong as its WEAKEST reduction, not the average of its phases, and verifying the strong phases individually is no substitute for verifying the assembled whole.

## Chapter Summary

This chapter took every real kernel this book has built since Part 0 and put Chapter 10's threading primitives to work parallelizing them, closing Part 2 with a complete, verified, multi-threaded forward pass. Section 11.1 named the two fundamental ways to split a matrix-vector product — row-parallel, bit-for-bit reproducible with no combination step, and column-parallel, which requires one, and is reproducible only when that combination happens in a fixed order rather than through a well-defined-but-order-dependent shared atomic. Section 11.2 generalized Chapter 10.3's single-shape thread pool into one that carries an entire layer's several different matmul phases, and along the way discovered and fixed a genuine cross-architecture hazard: the compiler's default FMA-contraction heuristic can silently break bit-exactness between a serial and a parallel code path on one architecture while leaving it intact on another. Section 11.3 parallelized GQA attention across query heads and found that naive, per-index thread boundaries — while perfectly correct — redundantly re-read a shared KV head's cache rows across threads, fixed by aligning boundaries to whole KV groups instead. Section 11.4 confronted a version of this book's reproducibility standard no earlier chapter needed: not just reproducible on a rerun at a fixed configuration, but reproducible ACROSS a change in that configuration, delivered by a fixed-block reduction rather than a per-thread-indexed one. Section 11.5 closed the chapter by combining all four techniques into one real, multi-layer decode step, verified bit-for-bit identical at two different thread counts, and then deliberately broke that guarantee by swapping out a single reduction, demonstrating that a pipeline's overall reproducibility is only as strong as its weakest link, never the average of its parts. Chapter 10 showed how to coordinate real threads correctly; this chapter showed how to make an entire real forward pass not just fast and correct, but reproducible in the specific, stronger sense a production inference engine actually needs.

## Self-Check Questions

1. Why is row-parallel matmul bit-for-bit reproducible against a serial reference with no tolerance needed, while column-parallel matmul (even with a correct, fixed-order combine) is only reproducible within a tolerance against that same serial reference?
2. Section 11.1 includes `gemv_column_parallel_racy_order` as real, compilable source but never calls it from `main()`. What specific guarantee does the C++ standard make about this function that Chapter 10.2's racy counter did NOT have, and why does the function still not get executed here?
3. In Section 11.2, why can the Q, K, and V projections be batched into a single parallel phase, but the FFN gate/up projections cannot be batched into that SAME phase alongside the output projection?
4. Section 11.2 discovered that identical source code produced bit-for-bit matching serial-vs-parallel results on x86_64 but not on aarch64. What compiler behavior caused this, and why did it appear on one architecture but not the other?
5. In Section 11.3, why does an EVEN, per-index partition of query heads across threads remain fully CORRECT even when it splits a KV group across two threads — what specifically does that partition cost, if not correctness?
6. Explain the fix Section 11.3 uses to eliminate redundant KV-head reads, and why it does not change the attention computation's actual result.
7. Section 11.4 distinguishes "reproducible for a fixed thread count" from "reproducible across a change in thread count." Why does the per-thread-indexed reduction satisfy the first but not the second?
8. Why does Section 11.4's fixed-block reduction's block boundary NOT depend on the number of threads running, and why is that specific independence what makes it reproducible across thread counts?
9. In Section 11.5, every matmul phase and every attention head is thread-count-independent on its own, yet Test 3 shows the WHOLE decode step becomes thread-count-dependent when only the RMSNorm reduction changes. Why does breaking one piece contaminate phases that are individually still correct?
10. Section 11.5's capstone reuses real kernels from Chapters 3.1, 3.2, and 3.4 rather than a cost-model abstraction like Chapter 8.5's. Why does that choice matter for what "bit-for-bit identical" is actually claiming in this chapter, compared to what Chapter 8.5 verified?

## Where We Go Next

This chapter closes Part 2 with a complete, real, multi-threaded transformer layer whose correctness and reproducibility are both verified exactly, at more than one thread count, all the way through a full decode step. Part 3 turns to the concerns of a real serving system built around that engine: how tokens enter and leave it, how a KV cache is actually managed as a growing, evictable resource across many concurrent requests, and the state a production inference server has to track that a single decode step, however well-parallelized, does not.

## Worked Solutions

**1.** Row-parallel assigns each thread a disjoint set of OUTPUT rows and never combines anything across threads — each row's own dot product runs in exactly the same left-to-right accumulation order the serial reference uses, so the two computations are literally the same sequence of floating-point operations, just executed by different threads for different rows. Column-parallel instead has every thread compute a full partial output vector from a different slice of INPUT columns, and the final result requires SUMMING those partial vectors together — an operation whose order differs from the serial reference's own single-pass accumulation order, and since floating-point addition is not associative, a different summation order can produce a different (though numerically very close) final value.

**2.** The C++ standard guarantees that `std::atomic<float>::fetch_add` is well-defined — no data race, no lost updates, no undefined behavior of any kind — unlike Chapter 10.2's `racy_increment_worker`, which was a genuine data race and therefore undefined behavior. What the standard does NOT guarantee is the ORDER in which concurrent `fetch_add` calls from different threads are applied, and because floating-point addition is not associative, that unfixed order means the exact final bit pattern is not guaranteed to reproduce from one run to the next, even though every individual run is perfectly well-defined and numerically correct. This section excludes it from checked output for a reproducibility reason, not a safety reason — a different, narrower version of Chapter 10.2's honesty standard.

**3.** Batching requires every matmul folded into one phase to share the same output row count, because the pool's single row-range partition is applied identically to every matmul inside that phase's lambda. Q, K, and V all project into DIM rows (or a subset of DIM for K and V's smaller KV-head dimension, which the lambda handles by clamping its own range), so one DIM-sized partition validly indexes all three. The FFN gate and up projections produce D_FF rows — a different size from DIM — so a phase batching gate/up with the (DIM-sized) output projection would either run out of bounds or, if told DIM as its row count, silently never compute the D_FF - DIM rows past that count, exactly the bug Section 11.2's Test 3 demonstrates deliberately.

**4.** GCC's default `-ffp-contract=fast` allows fusing a multiply immediately followed by an add (`a*b + c`) into a single hardware fused-multiply-add instruction wherever it judges doing so profitable — a decision made per call site, after inlining, not guaranteed to be applied consistently to two textually different pieces of source that compute the same arithmetic. aarch64's baseline instruction set always includes scalar FMA hardware, so the compiler can and does fuse readily there; this book's x86_64 baseline build has no `-mfma` flag, so no scalar hardware FMA exists to fuse into at all, and every accumulation stays as separate multiply-then-add regardless of call site. The result was two code paths (a standalone function and an inlined lambda body) that were textually different enough to receive different fusion treatment on aarch64 specifically, producing a genuine, silent bit-level divergence between serial and parallel results that did not appear on x86_64.

**5.** Every query head's attention computation — its scores, its softmax, its weighted sum over V — depends only on that head's own query vector and its assigned KV head's cached rows, never on any other head's computation or any shared mutable state between heads. Whichever thread computes a given head, and in whatever order threads happen to finish, each head's own result is identical to what the serial reference would compute for that head, so the overall output is bit-for-bit correct regardless of the partition. What an index-based partition costs is not correctness but memory locality: it can cause the SAME KV head's cache rows to be read by more than one thread instead of exactly one, a real but purely structural inefficiency in how many times shared bytes get fetched, not a wrong answer anywhere.

**6.** The fix ceiling-divides the KV HEAD count (not the query-head count) across threads, then expands each thread's resulting KV-head range back into the full set of query heads that belong to those KV heads' groups. Because every query head within one KV group is now guaranteed to be assigned to the SAME thread as every other query head in that group, no KV head's cache rows are ever needed by more than one thread. The computation each thread performs for each of its assigned heads is identical to before — the same `attend_one_head` function, called with the same arguments it would have received under any other partition — so the result is unchanged; only which heads get grouped onto which thread changes.

**7.** The per-thread-indexed reduction's chunk boundaries are computed as `ceil(n / n_threads)`, a value that depends directly on `n_threads`. For a FIXED value of `n_threads`, those boundaries — and therefore the summation order within and across chunks — never change between runs, so the result reproduces exactly every time, satisfying the first, weaker claim. But changing `n_threads` changes the chunk size and therefore moves every boundary, which changes which elements get summed together first in each chunk and in what order the chunk totals themselves get combined — a different summation order for floating-point values that are not guaranteed to sum identically regardless of order, so the second, stronger claim does not hold.

**8.** The fixed-block scheme's block boundaries are computed purely from a compile-time constant block size and the array's own length — block `b` always covers elements `[b*BLOCK_SIZE, min((b+1)*BLOCK_SIZE, n))`, a fact that involves `n_threads` nowhere in its definition. Thread count only affects which BLOCKS get assigned to which thread (via the same ceiling-division partition used everywhere else in this chapter), never where the blocks themselves begin or end, and each block's own internal sum is computed the same fixed way regardless of which thread happens to compute it. Since the final combination also walks every block index in the same fixed order (0 through the last block) regardless of how many threads existed, the entire computation — which elements get summed with which, and in what order the results get combined — is completely independent of `n_threads`, which is exactly the property a per-thread-indexed reduction lacks.

**9.** Every matmul phase and every attention head computes its own OWN piece of the layer's state independently and writes it to a disjoint location, so those computations genuinely do not depend on thread count on their own. But RMSNorm's reduction result becomes an INPUT to every phase that follows it — the normalized activations feed the QKV projections, which feed attention, which feeds the output projection, and the residual sum built from all of that feeds the next layer's own RMSNorm in turn. A different bit pattern out of one reduction is not an isolated error confined to that reduction; it is a different NUMBER flowing into every downstream computation that reads it, so phases that are individually perfectly thread-count-independent still produce different final results once their shared input differs.

**10.** Chapter 8.5 verified that a closed-form COST FORMULA (FLOPs and bytes) matched what a real kernel actually counted as it ran — a claim about resource accounting, not about the kernel's own numerical OUTPUT matching anything else bit-for-bit. Section 11.5's capstone instead runs the actual RMSNorm, matmul, attention, and SwiGLU arithmetic and checks that the resulting activation VALUES are bit-for-bit identical between a serial reference and a parallel, multi-threaded, multi-thread-count computation. "Bit-for-bit identical" here is a claim about the actual numbers a real decode step would produce — the same claim Chapter 10.3 first made for one GEMV — extended across an entire real, multi-layer forward pass, which is a substantially stronger and more directly useful guarantee for a real inference engine than a cost formula matching a cost counter.
