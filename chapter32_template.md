# Chapter 32: Flash Attention and CUDA Kernels: Taking the Engine to the GPU

<!-- edge:opener:start -->
<figure class="chapter-figure" markdown="1">

[![Chapter 32: Tiled attention keeps the score matrix out of global memory](../images/chapter-openers/32.svg?v=f13d2b11c378)](../images/chapter-openers/32.svg?v=f13d2b11c378)

<figcaption>Chapter roadmap. Open the image for a full-size view.</figcaption>
</figure>
<!-- edge:opener:end -->

**What you will understand by the end of this chapter:**

- Why standard attention's own real O(N^2) score-matrix memory becomes the actual bottleneck at real sequence lengths, quantified in exact bytes rather than described in the abstract, and why the online-softmax recurrence -- a real, provable algebraic reformulation of softmax, not an approximation -- fixes it.
- How to implement online softmax without a quadratic score matrix and check numerical agreement within the stated floating-point tolerances.
- How to implement that identical real algorithm on top of `std::mdspan`, checked directly against a naive full-matrix reference across genuinely different real shapes and tile sizes, and to "benchmark" it the way this book's own honesty discipline requires: with real, deterministic operation counts and peak-memory byte counts, never with wall-clock timing.
- How to implement the algorithm as a CUDA kernel, compare its output with a CPU reference on an A100, and distinguish device execution from a no-device validation run.
- How a real kernel-validation suite is actually structured: a real CPU golden reference, a real comparison harness proven correct independently of any specific hardware, and a real device-detection path that honestly reports what it could and could not check on the machine actually running it.

**What you need to know first:**

- Section 30.2's own real numerically stable softmax -- the shift-invariant max-subtraction identity -- is the exact real building block this chapter's own online-softmax recurrence generalizes from a single row, computed all at once, to a real streaming computation over blocks that are never all in memory simultaneously.
- Chapter 2.1's own `std::mdspan` technique (a non-owning view over flat memory, indexed through the `idx2()`/`idx3()` helpers Chapter 13's own appendix established for real compatibility with this book's own GCC 11.4.0 aarch64 hardware) is applied here, unchanged, to this book's own final numerical kernel.
- This book's own running discipline of never claiming a wall-clock timing result as part of a locked, deterministic self-test contract (stated plainly in Appendix D) is what shapes this chapter's own definition of "benchmark": real FLOP counts and real peak-memory byte counts, both exactly reproducible on every real run, stand in for timing throughout this chapter.

---

This chapter extends the book's C++ inference foundations to tiled attention and CUDA. The November 20, 2026 validation ran the attention kernel on a physical NVIDIA A100-SXM4-40GB with CUDA 12.8 and GCC 13.3. All 71 checks passed, including 60 device-to-CPU comparisons across different shapes and partial tiles. Compute Sanitizer memory, race and synchronization checks also passed. A separate no-device run verified the explicit skip behavior. See [Validation status](../validation-status.md) for the evidence and the distinction between tested correctness, target-specific execution and performance measurements.

<!-- chapter-figure -->
<figure class="chapter-figure" markdown="1">

![Tiled attention keeps the score matrix out of global memory. Each query retains running softmax statistics while K/V tiles pass through shared memory.](../images/32-overview.svg?v=f75001fc09d7){: loading="lazy" width="760" }

  <figcaption>Validate partial tiles and query blocks against the CPU reference on a physical GPU.</figcaption>
</figure>
<!-- /chapter-figure -->

## 32.1 The O(N^2) Memory Wall and the Online-Softmax Fix

<!-- edge:concept-32.1:start -->
<figure class="chapter-figure" markdown="1">

[![Online softmax rescales old and new contributions](../images/technical/ch32/32-01.svg?v=e486d74bf453)](../images/technical/ch32/32-01.svg?v=e486d74bf453)

<figcaption>Tile sums and weighted values use the same new maximum. Normalize o by l after the final tile.</figcaption>
</figure>
<!-- edge:concept-32.1:end -->


### Intuition

All implementations in this chapter compute unscaled, noncausal single-head attention, `softmax(Q K^T) V`, with supplied finite Q/K/V arrays. They omit the usual `1/sqrt(d)` scale, causal/padding masks, batching and model integration. CPU/GPU agreement tests this stated contract, not equivalence to a production decoder. See the [scaled dot-product attention definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

Standard attention computes a full N x N score matrix -- one entry per query-key pair -- before it can take a single softmax over any row of it. That matrix's own real memory cost grows with the SQUARE of sequence length, and at the real sequence lengths modern serving systems actually handle, it becomes the genuine bottleneck long before raw compute does.

### The Concept, In Detail

Test 1 quantifies this in real, exact bytes rather than describing it abstractly: a real 8192-token sequence's own full score matrix, at 4 bytes per element, costs exactly 268,435,456 bytes -- 256 MiB -- for a SINGLE batch-and-head pair, and doubling the sequence length to 16384 costs exactly 4x that, a real 1 GiB. Test 2 confirms the real fix's own defining property before building it: a streaming approach's score-buffer storage -- one block's worth of scores -- is a fixed constant that does not depend on N at all.

Tests 3 through 5 check the online-softmax recurrence on finite supplied inputs. Test 3 uses a hand-verifiable degenerate case; Tests 4 and 5 compare streaming and full-row outputs within 1e-9 for several block sizes, including 1. The recurrence is algebraically equivalent in real arithmetic, but floating-point evaluation order changes rounding. Passing these cases is neither bitwise equality nor a proof for all inputs.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_memory_wall_and_online_softmax.cpp -o 01_memory_wall_and_online_softmax
./01_memory_wall_and_online_softmax
```

**Sample input:** the real O(N^2) score-matrix memory cost checked against an exact hand computation at a real, production-scale sequence length, and its own real quadratic growth confirmed directly; streaming attention's score-buffer storage checked to be a fixed constant independent of N; naive attention checked against a real, hand-verifiable degenerate case; and streaming attention, at multiple real block sizes, checked to agree with naive attention to within 1e-9 on a genuinely non-degenerate example.

@@OUT1@@

!!! warning "[COMMON TRAP] treating online-softmax as an approximation that trades accuracy for memory"
    It is easy to assume any technique that avoids materializing the full real score matrix must be giving something up numerically to get there. Test 4 exists specifically to rule this out: streaming attention with a block size of 1 -- processing one real key at a time, the maximally memory-frugal case -- produces a result matching naive full-row attention to within 1e-9, not "close enough for practical purposes." The online-softmax recurrence is a real algebraic identity: the running max, sum, and output correction terms are derived directly from the same shift-invariant softmax identity Chapter 30.2 already proved, rearranged to update incrementally rather than requiring the whole row up front. There is no accuracy given up for the memory saved -- which is exactly why Flash Attention became the real, universal default rather than a memory-constrained fallback used only when the full matrix does not fit.

## 32.2 A std::mdspan-Based Flash Attention Implementation and Benchmark

<!-- edge:concept-32.2:start -->
<figure class="chapter-figure" markdown="1">

[![Tiling changes intermediate storage, not the target result](../images/technical/ch32/32-02.svg?v=60f617affa6a)](../images/technical/ch32/32-02.svg?v=60f617affa6a)

<figcaption>The tiled path avoids storing the full score matrix but remains subject to finite-precision differences.</figcaption>
</figure>
<!-- edge:concept-32.2:end -->


### Intuition

Section 32.1 proved the online-softmax recurrence correct on raw vectors. This section implements the identical real algorithm as a proper tiled Flash Attention pass over `std::mdspan`-viewed Q, K, and V matrices, and "benchmarks" it the only way this book's own discipline allows: with real, deterministic counts, never wall-clock time.

### The Concept, In Detail

`naive_attention_full` genuinely materializes the real, full Nq x Nk score matrix as one real mdspan-viewed allocation -- the actual behavior Section 32.1 quantified the cost of, not a stand-in for it. `tiled_flash_attention` applies Section 32.1's own recurrence per real Q-block over real K/V-blocks, reusing a SINGLE block-sized score buffer for every block rather than ever allocating the full matrix. Test 1 confirms the naive reference is itself correct via mdspan on the identical degenerate case Section 32.1 used. Test 2 confirms the tiled implementation matches the naive reference to within 1e-9 across 5 genuinely different real shapes and tile sizes.

Test 3 is this section's own central, real finding, and it is counted directly rather than assumed: both implementations perform EXACTLY 2 x Nq x Nk x d real multiply-accumulate operations, an empirically counted fact from each implementation's own innermost loop -- confirming Flash Attention's real benefit is reduced memory TRAFFIC, not reduced compute. Test 4 completes the honest benchmark: naive attention's own real peak score-buffer size, read directly from an actual allocation's own byte count, grows from 128 to 512 to 2048 bytes as the sequence grows, while tiled attention's score-buffer size stays fixed at exactly `block_rows * block_cols * sizeof(double)` regardless.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_mdspan_flash_attention_implementation_and_benchmark.cpp -o 02_mdspan_flash_attention_implementation_and_benchmark
./02_mdspan_flash_attention_implementation_and_benchmark
```

**Sample input:** naive mdspan-based attention checked against a real, hand-verifiable degenerate case; tiled Flash Attention checked against the naive reference to within 1e-9 across 5 genuinely different real shapes and tile sizes; both implementations' own real multiply-accumulate operation counts checked to match each other and an exact closed-form count across those identical shapes; and each implementation's own real peak score-buffer byte count, read directly from an actual allocation, checked against the expected growth (naive) or fixed constant (tiled) as sequence length grows.

@@OUT2@@

!!! warning "[COMMON TRAP] assuming Flash Attention's own real speedup comes from doing less arithmetic"
    Test 3's own real, counted result is easy to misread in the opposite direction from the Section 32.1 trap: it is tempting to assume Flash Attention must ALSO be doing fewer real floating-point operations, since it is famous for being faster in practice. Test 3 proves directly that it is not: both implementations perform the identical `2 * Nq * Nk * d` real multiply-accumulate operations, counted at each one's own innermost loop, every single time. Flash Attention's own genuine real speedup on actual hardware comes from a different resource entirely -- it never writes the full O(N^2) score matrix out to slow real memory and reads it back in for the softmax and the second matmul, which Chapter 30's own roofline framing already established is the ACTUAL bottleneck for a memory-bound operation. Same real compute, drastically less real memory traffic -- that distinction is the entire point of this section's own choice to benchmark bytes and operation counts rather than assume the FLOP count itself must have changed.

## 32.3 An Educational CUDA Attention Kernel and Validation Suite

<!-- edge:concept-32.3:start -->
<figure class="chapter-figure" markdown="1">

[![Validate kernels on real device work](../images/technical/ch32/32-03.svg?v=4da01d52f616)](../images/technical/ch32/32-03.svg?v=4da01d52f616)

<figcaption>A skipped device run is not a pass. Partial tiles and block tails need explicit coverage.</figcaption>
</figure>
<!-- edge:concept-32.3:end -->


### From a CPU reference to a device result

The kernel assigns one query row to each thread. Threads cooperatively load a tile of keys and values into shared memory, synchronize, and update the online-softmax state for their query. The last query block and final key tile can be partially filled.

Validation compares device output with a CPU attention reference for 60 combinations of query count, key count, and head dimension. The shapes include 129 queries, key counts around the 32-key tile boundary, and head dimensions up to the kernel's stated limit of 64. The comparison rejects non-finite output. CUDA allocation, copy, launch, and synchronization failures cause validation to fail, and the input checks reject unsupported dimensions before a launch.

### Build and run

Choose the architecture for the device that will run the binary. The A100 validation used CUDA 12.8 with `sm_80`. Jetson Orin uses `sm_87`.

```bash
nvcc -std=c++20 -arch=sm_80 03_cuda_kernel_and_validation_suite.cu -o cuda_validation
./cuda_validation
```

@@CODE3@@

The captured transcript below is from the real A100 run. A machine with no visible device reports skipped device comparisons while still testing the reference, comparison function, and argument validation. That transcript will differ from a device run.

@@OUT3@@

!!! warning "A passing host reference does not establish device correctness"
    Check the kernel on a real GPU. The suite must reject failed CUDA operations, NaN output, and numerical mismatches. Host-only simulation cannot expose a shared-memory race or an invalid device access.

## Chapter Summary

This chapter demonstrates unscaled, noncausal attention and validates the CUDA implementation against its CPU reference on an A100. Score-tile storage is fixed for fixed tile dimensions; total Q/K/V, outputs and running state are not constant in sequence length. Comparisons are tolerance-based, not bitwise equality or proof for arbitrary inputs. Architecture compilation is not physical Jetson execution.

## Self-Check Questions

1. Section 32.1's Test 1 shows the real score-matrix memory quadruples when sequence length doubles. Explain, from the formula itself, why this growth is quadratic rather than linear.
2. Section 32.1's Test 4 uses a block size of exactly 1 as the section's own central proof. Explain why this specific choice is a stronger test of the online recurrence's own correctness than a larger, more "realistic" block size would be.
3. Section 32.1's online-softmax recurrence relies on `exp(-infinity)` evaluating to exactly `0.0` for its own initial-block correctness. Explain what would go wrong with the very first block's own computed result if this were not true.
4. Section 32.2's Test 3 shows naive and tiled attention perform the identical number of real multiply-accumulate operations. Given that result, explain in your own words what Flash Attention's own real speedup on actual hardware actually comes from instead.
5. Section 32.2's `naive_attention_full` genuinely allocates the full real Nq x Nk score matrix, rather than only computing one row at a time. Explain why this specific choice matters for Test 4's own real peak-memory comparison to be a fair, honest one.
6. Section 32.3 explains why a clean `nvcc` compile does not prove the kernel's output is correct. Name one specific real category of bug that compilation could never catch, and explain why it could not.
7. Section 32.3 checks the CUDA Runtime API error code before using the device count. Why must this check remain even though discovery succeeded during the A100 validation?
8. Section 32.3's `validate_kernel_on_device` function contains a real code path that would allocate device memory and launch the real kernel, and that path was exercised on an A100. Explain what specifically would need to be true of the machine running this exact file for that path to execute instead.
9. Section 32.3's Test 3 validates the comparison harness using only CPU-computed vectors, with no GPU involved at all. Explain why this test is still a meaningful, real check of the kernel-validation suite's own correctness, despite never touching a GPU.
10. What algebraic property of online softmax is shared by the implementations, and what can tolerance-based finite-input tests establish about their floating-point behavior?

## Where We Go Next

This chapter closes the main text of this book. Every real technique built across 32 chapters -- from `std::mdspan` tensors and affine quantization through SIMD, threading, the KV cache, real deployed models, twelve real edge deployments, the mathematics underlying every kernel, a real continuous-batching scheduler, and finally a real GPU kernel -- was built from scratch and verified directly rather than merely asserted, on real, portable C++23 that compiles and runs identically across an x86 development machine and real aarch64 hardware. What remains is the book's own set of appendices: cross-compilation setup for real edge targets, a practice quiz spanning every Part, a consolidated decision-tree reference, a profiling and benchmarking guide for real edge hardware, and a Rosetta Stone for readers arriving fluent in the Python inference ecosystem.

## Worked Solutions

**1.** The real score-matrix formula is `N * N * dtype_bytes` -- N appears TWICE, once for the number of queries and once for the number of keys, since every query is scored against every key. Doubling N therefore multiplies the result by `2 * 2 = 4`, not `2`: quadratic growth is a direct, mechanical consequence of N appearing as a product with itself in the formula, not an incidental property of this specific example.

**2.** A block size of 1 forces the recurrence to update its own running max, sum, and output after processing a SINGLE key at a time, which is the maximum possible number of real update steps (and real correction-term applications) for a given sequence length. If the online recurrence's own algebra had any subtle error -- in the correction term, in the order of operations, in the handling of the running max -- the more update steps that occur, the more real opportunities that error has to compound or reveal itself. A single large block, in the extreme case one block covering the entire sequence, would apply the correction step zero or one times and could mask a real bug that only manifests when the running state is actually updated repeatedly.

**3.** The very first block's own real correction term is computed as `exp(m_old - m_new)` where `m_old` is initialized to `-infinity`. If this did not evaluate to exactly `0.0`, the first block's own running sum and output would be corrupted by multiplying the (empty, zero-valued) initial state by whatever `exp(-infinity - m_new)` actually returned instead -- if it returned NaN, for instance, every subsequent real update would also become NaN, since any arithmetic involving a NaN produces NaN, corrupting the entire computation from the very first block onward.

**4.** Since both implementations perform the identical number of real multiply-accumulate operations, Flash Attention's own real speedup cannot come from doing less arithmetic. It comes instead from real memory traffic: naive attention writes the entire real O(N^2) score matrix out to memory and reads it back in for the softmax and the second matmul, while tiled Flash Attention never writes more than one real block's worth of scores to memory at any point, keeping the running state in fast on-chip storage (registers or shared memory) instead. Chapter 30's own roofline framing already established that a memory-bound operation's real bottleneck is bandwidth, not FLOPs -- Flash Attention's real benefit is reducing that memory traffic, not the arithmetic.

**5.** The baseline deliberately materializes the full score matrix. Other valid baselines compute rows without storing that full matrix, so the demonstrated allocation comparison must not be generalized to every non-Flash implementation. It measures score-buffer allocations, not complete process or GPU memory.

**6.** A real race condition in the shared-memory tile load -- for instance, if a thread began reading from `k_tile` or `v_tile` before every thread in the block had finished writing its own portion of that same tile -- would compile without any error at all, since the CUDA compiler has no way to know, purely from the kernel's own source code, whether a `__syncthreads()` call is missing or misplaced relative to how the tile is actually used. This category of bug only manifests as an actual incorrect numerical result (or, worse, one that is only wrong nondeterministically depending on real thread-scheduling timing) when the kernel is genuinely executed on real hardware -- compilation checks syntax and generates valid instructions, it does not simulate the real, concurrent execution of thousands of real threads.

**7.** Device discovery succeeded in the A100 run, but an API failure on another host can leave the output count unusable. Check `err == cudaSuccess` before interpreting the count. A missing device must produce an explicit skip, while allocation, transfer, launch or synchronization errors during a device test must fail validation.

**8.** Compile for the target GPU and execute with a working CUDA driver and a visible device. For A100 use `-arch=sm_80`. The runtime discovery call must succeed and report at least one device. Validation then allocates buffers, launches attention, synchronizes, and compares the copied result with the CPU reference. The documented A100 run exercised this path for all 60 shapes.

**9.** The comparison function is tested independently with known matches, tolerable rounding differences, mismatches and non-finite values. Those host checks validate the decision rule. The A100 run then exercised that rule on actual kernel output for 60 shapes. Host harness checks and device execution are complementary tests, not substitutes for one another.

**10.** All three sections depend on the SAME real property: that the online-softmax recurrence -- computing a running max, then rescaling the running sum and output by `exp(old_max - new_max)` before incorporating each new block -- produces an algebraically equivalent result in real arithmetic; Section 32.1 checks floating-point agreement within tolerance on finite test inputs. Section 32.2's tiled CPU implementation depends on this to claim its own output matches the naive reference; Section 32.3's CUDA kernel implements this identical recurrence in device code and depends on the same property to claim the device-to-reference comparisons performed on the A100 validate the tested implementations and shapes. The tests do not prove arbitrary-input correctness or bitwise equality.
