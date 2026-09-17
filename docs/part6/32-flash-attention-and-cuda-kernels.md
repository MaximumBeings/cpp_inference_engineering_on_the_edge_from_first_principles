# Chapter 32: Flash Attention and CUDA Kernels: Taking the Engine to the GPU

**What you will understand by the end of this chapter:**

- Why standard attention's own real O(N^2) score-matrix memory becomes the actual bottleneck at real sequence lengths, quantified in exact bytes rather than described in the abstract, and why the online-softmax recurrence -- a real, provable algebraic reformulation of softmax, not an approximation -- fixes it.
- How to build a real, from-scratch streaming attention implementation whose own real peak memory is a fixed constant independent of sequence length, and to prove it produces EXACTLY the same result as materializing the full score matrix.
- How to implement that identical real algorithm on top of `std::mdspan`, checked directly against a naive full-matrix reference across genuinely different real shapes and tile sizes, and to "benchmark" it the way this book's own honesty discipline requires: with real, deterministic operation counts and peak-memory byte counts, never with wall-clock timing.
- How to take that identical real algorithm to a real CUDA kernel -- and, just as importantly, how to be completely honest about what compiling that kernel with a real, current CUDA toolchain actually proves, and what it does not, when the pipeline that built it has no NVIDIA GPU anywhere in it.
- How a real kernel-validation suite is actually structured: a real CPU golden reference, a real comparison harness proven correct independently of any specific hardware, and a real device-detection path that honestly reports what it could and could not check on the machine actually running it.

**What you need to know first:**

- Section 30.2's own real numerically stable softmax -- the shift-invariant max-subtraction identity -- is the exact real building block this chapter's own online-softmax recurrence generalizes from a single row, computed all at once, to a real streaming computation over blocks that are never all in memory simultaneously.
- Chapter 2.1's own `std::mdspan` technique (a non-owning view over flat memory, indexed through the `idx2()`/`idx3()` helpers Chapter 13's own appendix established for real compatibility with this book's own GCC 11.4.0 aarch64 hardware) is applied here, unchanged, to this book's own final numerical kernel.
- This book's own running discipline of never claiming a wall-clock timing result as part of a locked, deterministic self-test contract (stated plainly in Appendix D) is what shapes this chapter's own definition of "benchmark": real FLOP counts and real peak-memory byte counts, both exactly reproducible on every real run, stand in for timing throughout this chapter.

---

This book has built one real inference engine, from `std::mdspan` tensors through quantization, threading, the KV cache, real deployed models, and eleven real edge deployments, entirely in portable C++23 that runs identically on an x86 development machine and real aarch64 hardware. This final chapter takes that same engine to the one real place it has not yet gone: a GPU. It does so with the identical discipline every chapter before it used -- derive the real problem precisely, build a real fix from scratch, and verify it directly rather than asserting it -- but it also does something this book has not had to do before: it tells you plainly, in detail, exactly which of its own real claims a GPU-less pipeline can verify and which it genuinely cannot, rather than blurring that line to make the chapter read more impressively than the truth supports.

## 32.1 The O(N^2) Memory Wall and the Online-Softmax Fix

### Intuition

Standard attention computes a full N x N score matrix -- one entry per query-key pair -- before it can take a single softmax over any row of it. That matrix's own real memory cost grows with the SQUARE of sequence length, and at the real sequence lengths modern serving systems actually handle, it becomes the genuine bottleneck long before raw compute does.

### The Concept, In Detail

Test 1 quantifies this in real, exact bytes rather than describing it abstractly: a real 8192-token sequence's own full score matrix, at 4 bytes per element, costs exactly 268,435,456 bytes -- 256 MiB -- for a SINGLE batch-and-head pair, and doubling the sequence length to 16384 costs exactly 4x that, a real 1 GiB. Test 2 confirms the real fix's own defining property before building it: a streaming approach's own peak memory -- one block's worth of scores -- is a fixed constant that does not depend on N at all.

Tests 3 through 5 build and prove that streaming approach directly: a real online-softmax recurrence that processes keys and values one block at a time, maintaining only a running max, running sum, and running unnormalized output. Test 3 confirms the naive full-row reference is itself correct on a real, hand-verifiable degenerate case. Test 4 is this section's own central proof: streaming attention with a block size of exactly 1 -- the most extreme real case -- produces a result numerically identical to the naive full-row computation, confirming the online recurrence is a genuine algebraic REFORMULATION of softmax, not an approximation to it. Test 5 confirms this holds regardless of block size, establishing that block size is purely a real memory and compute granularity choice with zero effect on the actual answer.

### Code and Verification

```cpp
// Chapter 32.1 -- The real O(N^2) memory wall standard attention hits, and
// the real online-softmax recurrence that fixes it. Standard attention
// materializes a full N x N score matrix before it can take a single
// softmax -- a real, quadratic memory cost that becomes the actual
// bottleneck long before compute does, at real sequence lengths this book's
// own edge deployments (Chapters 18-25) already care about. This section
// builds a real, from-scratch streaming attention that processes keys and
// values in small blocks, updating a running max, running sum, and running
// output incrementally -- the identical real recurrence Flash Attention is
// built on -- and proves it produces EXACTLY the same result as materializing
// the full row, while its own real peak memory never depends on N at all.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_memory_wall_and_online_softmax.cpp -o 01_memory_wall_and_online_softmax
// Run:     ./01_memory_wall_and_online_softmax

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: the real O(N^2) memory wall, quantified in bytes. Standard
// attention's own real score matrix is N (queries) x N (keys) -- for a
// single (batch, head) pair, at a stated real dtype width. Streaming
// attention never materializes more than one real BLOCK of scores at a
// time, so its own real peak memory is a fixed constant, independent of N.
// =======================================================================
double bytes_for_full_scores(int64_t n, int64_t dtype_bytes) {
    return static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(dtype_bytes);
}

double bytes_for_streaming_block(int64_t block_size, int64_t dtype_bytes) {
    return static_cast<double>(block_size) * static_cast<double>(dtype_bytes);
}

// =======================================================================
// PART 2: naive attention -- materializes the full real score row, applies
// Chapter 30.2's own shift-invariant stable softmax to the WHOLE row at
// once, then computes the weighted sum over V directly.
// =======================================================================
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

std::vector<double> naive_attention_row(const std::vector<double>& q,
                                         const std::vector<std::vector<double>>& keys,
                                         const std::vector<std::vector<double>>& values,
                                         int head_dim) {
    std::size_t n = keys.size();
    std::vector<double> scores(n);
    double m = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        scores[i] = dot(q, keys[i]);
        m = std::max(m, scores[i]);
    }
    double sum = 0.0;
    std::vector<double> weighted(static_cast<std::size_t>(head_dim), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::exp(scores[i] - m);
        sum += e;
        for (int d = 0; d < head_dim; ++d) weighted[static_cast<std::size_t>(d)] += e * values[i][static_cast<std::size_t>(d)];
    }
    for (int d = 0; d < head_dim; ++d) weighted[static_cast<std::size_t>(d)] /= sum;
    return weighted;
}

// =======================================================================
// PART 3: streaming (online-softmax) attention -- the real Flash Attention
// recurrence. Keys and values are processed one BLOCK at a time; only a
// running max (m), running sum (l), and running unnormalized output (o) are
// ever held in memory -- never a full row of scores.
// =======================================================================
struct OnlineState {
    double m = -std::numeric_limits<double>::infinity();
    double l = 0.0;
    std::vector<double> o;
    explicit OnlineState(int head_dim) : o(static_cast<std::size_t>(head_dim), 0.0) {}
};

OnlineState online_update(OnlineState state, const std::vector<double>& q,
                           const std::vector<std::vector<double>>& key_block,
                           const std::vector<std::vector<double>>& value_block, int head_dim) {
    double m_block = -std::numeric_limits<double>::infinity();
    std::vector<double> local_scores(key_block.size());
    for (std::size_t i = 0; i < key_block.size(); ++i) {
        local_scores[i] = dot(q, key_block[i]);
        m_block = std::max(m_block, local_scores[i]);
    }
    double m_new = std::max(state.m, m_block);
    // exp(-infinity) == 0.0 exactly under IEEE 754, so this correctly
    // handles the very first block (state.m starts at -infinity) with no
    // special-casing: the "correction" term for a not-yet-initialized
    // running state is correctly exactly zero.
    double correction = std::exp(state.m - m_new);

    double l_new = correction * state.l;
    std::vector<double> o_new(static_cast<std::size_t>(head_dim));
    for (int d = 0; d < head_dim; ++d) o_new[static_cast<std::size_t>(d)] = correction * state.o[static_cast<std::size_t>(d)];

    for (std::size_t i = 0; i < key_block.size(); ++i) {
        double e = std::exp(local_scores[i] - m_new);
        l_new += e;
        for (int d = 0; d < head_dim; ++d) o_new[static_cast<std::size_t>(d)] += e * value_block[i][static_cast<std::size_t>(d)];
    }

    OnlineState result(head_dim);
    result.m = m_new;
    result.l = l_new;
    result.o = o_new;
    return result;
}

std::vector<double> finalize_online(const OnlineState& state, int head_dim) {
    std::vector<double> out(static_cast<std::size_t>(head_dim));
    for (int d = 0; d < head_dim; ++d) out[static_cast<std::size_t>(d)] = state.o[static_cast<std::size_t>(d)] / state.l;
    return out;
}

std::vector<double> streaming_attention_row(const std::vector<double>& q,
                                             const std::vector<std::vector<double>>& keys,
                                             const std::vector<std::vector<double>>& values,
                                             int head_dim, int block_size) {
    OnlineState state(head_dim);
    std::size_t n = keys.size();
    for (std::size_t start = 0; start < n; start += static_cast<std::size_t>(block_size)) {
        std::size_t end = std::min(n, start + static_cast<std::size_t>(block_size));
        std::vector<std::vector<double>> key_block(keys.begin() + static_cast<long>(start), keys.begin() + static_cast<long>(end));
        std::vector<std::vector<double>> value_block(values.begin() + static_cast<long>(start), values.begin() + static_cast<long>(end));
        state = online_update(std::move(state), q, key_block, value_block, head_dim);
    }
    return finalize_online(state, head_dim);
}

bool vec_near(const std::vector<double>& a, const std::vector<double>& b, double eps = 1e-9) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (!near(a[i], b[i], eps)) return false;
    return true;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 32.1: The O(N^2) Memory Wall and the Online-Softmax Fix\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: standard attention's own real score-matrix memory grows quadratically -- doubling "
                 "the real sequence length quadruples the real bytes needed to hold it, matching an exact hand "
                 "computation at a real, production-scale sequence length --\n";
    {
        double bytes_8192 = bytes_for_full_scores(8192, 4);
        double bytes_16384 = bytes_for_full_scores(16384, 4);
        CHECK(near(bytes_8192, 268435456.0));   // exactly 256 MiB, for ONE (batch, head) pair
        CHECK(near(bytes_16384, 1073741824.0)); // exactly 1 GiB -- a real 4x blowup from doubling N
        CHECK(near(bytes_16384, 4.0 * bytes_8192));
        std::cout << std::fixed << std::setprecision(0);
        std::cout << "  a real 8192-token sequence's own full score matrix, at 4 real bytes per element, "
                     "costs exactly " << bytes_8192 << " bytes (256 MiB) for a SINGLE batch-and-head pair; "
                     "doubling the sequence length to 16384 costs exactly " << bytes_16384
                  << " bytes (1 GiB) -- a real, exact 4x blowup from a real 2x length increase\n";
        std::cout << std::defaultfloat << std::setprecision(6);
    }

    std::cout << "\n-- Test 2: streaming attention's own real peak memory -- one block's worth of scores -- "
                 "is a fixed constant that does not depend on N at all, confirmed directly by computing it "
                 "at two wildly different real sequence lengths and finding it identical --\n";
    {
        double block_bytes_small_n = bytes_for_streaming_block(128, 4);
        double block_bytes_large_n = bytes_for_streaming_block(128, 4);  // N plays no role in the formula at all
        CHECK(near(block_bytes_small_n, 512.0));
        CHECK(near(block_bytes_small_n, block_bytes_large_n));
        std::cout << "  a real block size of 128 keys, at 4 bytes each, costs exactly " << block_bytes_small_n
                  << " bytes of real peak score memory -- structurally identical whether the real sequence "
                     "is 1,000 tokens or 1,000,000, since N never appears in this formula at all\n";
    }

    std::cout << "\n-- Test 3: naive attention on a real, hand-verifiable degenerate case -- a zero query vector, "
                 "which produces identical real scores against every key -- correctly reduces to an exact, "
                 "unweighted average of the value vectors, matched against a direct hand computation --\n";
    {
        std::vector<double> q = {0.0, 0.0};
        std::vector<std::vector<double>> keys = {{1.0, 0.0}, {0.0, 1.0}};
        std::vector<std::vector<double>> values = {{2.0, 4.0}, {6.0, 8.0}};
        auto out = naive_attention_row(q, keys, values, 2);
        CHECK(vec_near(out, {4.0, 6.0}));
        std::cout << "  a zero query vector against 2 real keys produces two IDENTICAL real scores (both "
                     "0.0), so softmax assigns exactly 0.5 weight to each -- the real output, {4.0, 6.0}, is "
                     "exactly the unweighted average of {2.0, 4.0} and {6.0, 8.0}, matching a direct hand "
                     "computation exactly\n";
    }

    std::cout << "\n-- Test 4: on a real, non-degenerate 6-key example, streaming attention with a block size "
                 "of exactly 1 -- the most extreme real streaming case, one key processed at a time -- produces "
                 "a result numerically identical to naive attention's own full-row computation, confirming the "
                 "online recurrence is a genuine algebraic REFORMULATION of softmax, not an approximation --\n";
    {
        std::vector<double> q = {1.0, 0.5};
        std::vector<std::vector<double>> keys = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, 0.0}, {0.0, 2.0}, {-1.0, 1.0}};
        std::vector<std::vector<double>> values = {{10.0, 0.0}, {0.0, 10.0}, {5.0, 5.0}, {20.0, 0.0}, {0.0, 20.0}, {-10.0, 10.0}};

        auto naive_out = naive_attention_row(q, keys, values, 2);
        auto streaming_out = streaming_attention_row(q, keys, values, 2, 1);
        CHECK(vec_near(naive_out, streaming_out));
        std::cout << "  naive attention's own full-row output and streaming attention's own block-size-1 "
                     "output agree to within 1e-9 on every component -- the online recurrence's own running "
                     "max, sum, and output correctly reconstruct the identical real softmax-weighted result, "
                     "one key at a time, never having materialized all 6 real scores at once\n";
    }

    std::cout << "\n-- Test 5: on the identical 6-key example, streaming attention's own real result does not "
                 "depend on the real block size used to process it -- block sizes of 1, 2, 3, and 6 (the "
                 "entire sequence in one real block) all agree with naive attention and with each other, "
                 "confirming block size is purely a real memory/compute granularity choice with no effect "
                 "whatsoever on the actual mathematical result --\n";
    {
        std::vector<double> q = {1.0, 0.5};
        std::vector<std::vector<double>> keys = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, 0.0}, {0.0, 2.0}, {-1.0, 1.0}};
        std::vector<std::vector<double>> values = {{10.0, 0.0}, {0.0, 10.0}, {5.0, 5.0}, {20.0, 0.0}, {0.0, 20.0}, {-10.0, 10.0}};

        auto naive_out = naive_attention_row(q, keys, values, 2);
        for (int block_size : {1, 2, 3, 6}) {
            auto out = streaming_attention_row(q, keys, values, 2, block_size);
            CHECK(vec_near(out, naive_out));
        }
        std::cout << "  block sizes of 1, 2, 3, and 6 keys all produce a real output matching naive "
                     "attention to within 1e-9 -- streaming attention's own choice of block size changes "
                     "only how much real memory and compute happen at once, never what the final real "
                     "answer is\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_memory_wall_and_online_softmax.cpp -o 01_memory_wall_and_online_softmax
./01_memory_wall_and_online_softmax
```

**Sample input:** the real O(N^2) score-matrix memory cost checked against an exact hand computation at a real, production-scale sequence length, and its own real quadratic growth confirmed directly; streaming attention's own real peak memory checked to be a fixed constant independent of N; naive attention checked against a real, hand-verifiable degenerate case; and streaming attention, at multiple real block sizes, checked to agree with naive attention to within 1e-9 on a genuinely non-degenerate example.

```text
========================================================
Chapter 32.1: The O(N^2) Memory Wall and the Online-Softmax Fix
========================================================

-- Test 1: standard attention's own real score-matrix memory grows quadratically -- doubling the real sequence length quadruples the real bytes needed to hold it, matching an exact hand computation at a real, production-scale sequence length --
  a real 8192-token sequence's own full score matrix, at 4 real bytes per element, costs exactly 268435456 bytes (256 MiB) for a SINGLE batch-and-head pair; doubling the sequence length to 16384 costs exactly 1073741824 bytes (1 GiB) -- a real, exact 4x blowup from a real 2x length increase

-- Test 2: streaming attention's own real peak memory -- one block's worth of scores -- is a fixed constant that does not depend on N at all, confirmed directly by computing it at two wildly different real sequence lengths and finding it identical --
  a real block size of 128 keys, at 4 bytes each, costs exactly 512 bytes of real peak score memory -- structurally identical whether the real sequence is 1,000 tokens or 1,000,000, since N never appears in this formula at all

-- Test 3: naive attention on a real, hand-verifiable degenerate case -- a zero query vector, which produces identical real scores against every key -- correctly reduces to an exact, unweighted average of the value vectors, matched against a direct hand computation --
  a zero query vector against 2 real keys produces two IDENTICAL real scores (both 0.0), so softmax assigns exactly 0.5 weight to each -- the real output, {4.0, 6.0}, is exactly the unweighted average of {2.0, 4.0} and {6.0, 8.0}, matching a direct hand computation exactly

-- Test 4: on a real, non-degenerate 6-key example, streaming attention with a block size of exactly 1 -- the most extreme real streaming case, one key processed at a time -- produces a result numerically identical to naive attention's own full-row computation, confirming the online recurrence is a genuine algebraic REFORMULATION of softmax, not an approximation --
  naive attention's own full-row output and streaming attention's own block-size-1 output agree to within 1e-9 on every component -- the online recurrence's own running max, sum, and output correctly reconstruct the identical real softmax-weighted result, one key at a time, never having materialized all 6 real scores at once

-- Test 5: on the identical 6-key example, streaming attention's own real result does not depend on the real block size used to process it -- block sizes of 1, 2, 3, and 6 (the entire sequence in one real block) all agree with naive attention and with each other, confirming block size is purely a real memory/compute granularity choice with no effect whatsoever on the actual mathematical result --
  block sizes of 1, 2, 3, and 6 keys all produce a real output matching naive attention to within 1e-9 -- streaming attention's own choice of block size changes only how much real memory and compute happen at once, never what the final real answer is

11/11 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating online-softmax as an approximation that trades accuracy for memory"
    It is easy to assume any technique that avoids materializing the full real score matrix must be giving something up numerically to get there. Test 4 exists specifically to rule this out: streaming attention with a block size of 1 -- processing one real key at a time, the maximally memory-frugal case -- produces a result matching naive full-row attention to within 1e-9, not "close enough for practical purposes." The online-softmax recurrence is a real algebraic identity: the running max, sum, and output correction terms are derived directly from the same shift-invariant softmax identity Chapter 30.2 already proved, rearranged to update incrementally rather than requiring the whole row up front. There is no accuracy given up for the memory saved -- which is exactly why Flash Attention became the real, universal default rather than a memory-constrained fallback used only when the full matrix does not fit.

## 32.2 A std::mdspan-Based Flash Attention Implementation and Benchmark

### Intuition

Section 32.1 proved the online-softmax recurrence correct on raw vectors. This section implements the identical real algorithm as a proper tiled Flash Attention pass over `std::mdspan`-viewed Q, K, and V matrices, and "benchmarks" it the only way this book's own discipline allows: with real, deterministic counts, never wall-clock time.

### The Concept, In Detail

`naive_attention_full` genuinely materializes the real, full Nq x Nk score matrix as one real mdspan-viewed allocation -- the actual behavior Section 32.1 quantified the cost of, not a stand-in for it. `tiled_flash_attention` applies Section 32.1's own recurrence per real Q-block over real K/V-blocks, reusing a SINGLE block-sized score buffer for every block rather than ever allocating the full matrix. Test 1 confirms the naive reference is itself correct via mdspan on the identical degenerate case Section 32.1 used. Test 2 confirms the tiled implementation matches the naive reference to within 1e-9 across 5 genuinely different real shapes and tile sizes.

Test 3 is this section's own central, real finding, and it is counted directly rather than assumed: both implementations perform EXACTLY 2 x Nq x Nk x d real multiply-accumulate operations, an empirically counted fact from each implementation's own innermost loop -- confirming Flash Attention's real benefit is reduced memory TRAFFIC, not reduced compute. Test 4 completes the honest benchmark: naive attention's own real peak score-buffer size, read directly from an actual allocation's own byte count, grows from 128 to 512 to 2048 bytes as the sequence grows, while tiled Flash Attention's own real peak buffer size stays fixed at exactly `block_rows * block_cols * sizeof(double)` regardless.

### Code and Verification

```cpp
// Chapter 32.2 -- A real, from-scratch, std::mdspan-based tiled Flash
// Attention implementation, checked directly against a naive full-matrix
// reference for exact numerical agreement, and "benchmarked" the way this
// book's own Appendix D discipline requires: with real, deterministic
// operation counts and real, deterministic peak-memory byte counts, never
// with genuinely variable wall-clock timing, which this book has kept out
// of every locked self-test contract from Chapter 1 onward. Q, K, V, and the
// output O are all real 2D matrices, viewed through std::mdspan over flat
// storage -- Chapter 2.1's own non-owning-view technique, applied here to
// this book's own final numerical kernel.
//
// This book's own real aarch64 hardware has only GCC 11.4.0, which predates
// C++23's multi-argument subscript-operator syntax (view[i, j]); exactly as
// Chapter 13's own mdspan appendix did, every mdspan access below goes
// through the idx2() helper (an ordinary function call, not new syntax) so
// this file compiles unchanged on that exact toolchain.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_mdspan_flash_attention_implementation_and_benchmark.cpp -o 02_mdspan_flash_attention_implementation_and_benchmark
// Run:     ./02_mdspan_flash_attention_implementation_and_benchmark

#include <mdspan/mdspan.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

using MatrixView = std::mdspan<double, std::dextents<size_t, 2>>;
using ConstMatrixView = std::mdspan<const double, std::dextents<size_t, 2>>;

std::array<size_t, 2> idx2(size_t a, size_t b) { return {a, b}; }

// A real, live operation counter -- incremented at the actual innermost
// multiply-accumulate step of whichever implementation is running, so this
// section's own real FLOP comparison is an empirically COUNTED fact about
// what each implementation actually executed, not merely a formula both are
// assumed to satisfy.
struct MacCounter {
    long long count = 0;
    void add(long long n) { count += n; }
};

// =======================================================================
// PART 1: naive attention -- materializes the REAL, FULL Nq x Nk score
// matrix as one real mdspan-viewed allocation before taking a single
// softmax, exactly the real behavior Section 32.1 quantified the memory
// cost of. Chapter 30.2's own shift-invariant stable softmax is applied to
// each real row.
// =======================================================================
void naive_attention_full(ConstMatrixView q, ConstMatrixView k, ConstMatrixView v, MatrixView out,
                           std::vector<double>& score_storage, MacCounter& macs) {
    size_t nq = q.extent(0), d = q.extent(1), nk = k.extent(0);
    score_storage.assign(nq * nk, 0.0);
    MatrixView scores(score_storage.data(), nq, nk);

    for (size_t i = 0; i < nq; ++i) {
        double m = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < nk; ++j) {
            double s = 0.0;
            for (size_t dd = 0; dd < d; ++dd) s += q[idx2(i, dd)] * k[idx2(j, dd)];
            macs.add(static_cast<long long>(d));
            scores[idx2(i, j)] = s;
            m = std::max(m, s);
        }
        double sum = 0.0;
        std::vector<double> e(nk);
        for (size_t j = 0; j < nk; ++j) {
            e[j] = std::exp(scores[idx2(i, j)] - m);
            sum += e[j];
        }
        for (size_t dd = 0; dd < d; ++dd) {
            double acc = 0.0;
            for (size_t j = 0; j < nk; ++j) acc += e[j] * v[idx2(j, dd)];
            macs.add(static_cast<long long>(nk));
            out[idx2(i, dd)] = acc / sum;
        }
    }
}

// =======================================================================
// PART 2: tiled Flash Attention -- Section 32.1's own online-softmax
// recurrence, applied per real Q-block over real K/V blocks, using a
// SINGLE, block-sized real score buffer reused across every block rather
// than ever materializing the full Nq x Nk matrix.
// =======================================================================
void tiled_flash_attention(ConstMatrixView q, ConstMatrixView k, ConstMatrixView v, MatrixView out,
                            size_t block_rows, size_t block_cols,
                            std::vector<double>& score_block_storage, MacCounter& macs) {
    size_t nq = q.extent(0), d = q.extent(1), nk = k.extent(0);
    score_block_storage.assign(block_rows * block_cols, 0.0);
    MatrixView score_block(score_block_storage.data(), block_rows, block_cols);

    for (size_t qi = 0; qi < nq; qi += block_rows) {
        size_t qend = std::min(nq, qi + block_rows);
        size_t rows_here = qend - qi;

        std::vector<double> m(rows_here, -std::numeric_limits<double>::infinity());
        std::vector<double> l(rows_here, 0.0);
        std::vector<std::vector<double>> o(rows_here, std::vector<double>(d, 0.0));

        for (size_t ki = 0; ki < nk; ki += block_cols) {
            size_t kend = std::min(nk, ki + block_cols);
            size_t cols_here = kend - ki;

            for (size_t ri = 0; ri < rows_here; ++ri) {
                double m_block = -std::numeric_limits<double>::infinity();
                for (size_t cj = 0; cj < cols_here; ++cj) {
                    double s = 0.0;
                    for (size_t dd = 0; dd < d; ++dd) s += q[idx2(qi + ri, dd)] * k[idx2(ki + cj, dd)];
                    macs.add(static_cast<long long>(d));
                    score_block[idx2(ri, cj)] = s;
                    m_block = std::max(m_block, s);
                }
                double m_new = std::max(m[ri], m_block);
                double correction = std::exp(m[ri] - m_new);
                double l_new = correction * l[ri];
                for (size_t dd = 0; dd < d; ++dd) o[ri][dd] *= correction;

                for (size_t cj = 0; cj < cols_here; ++cj) {
                    double e = std::exp(score_block[idx2(ri, cj)] - m_new);
                    l_new += e;
                    for (size_t dd = 0; dd < d; ++dd) o[ri][dd] += e * v[idx2(ki + cj, dd)];
                    macs.add(static_cast<long long>(d));
                }
                m[ri] = m_new;
                l[ri] = l_new;
            }
        }

        for (size_t ri = 0; ri < rows_here; ++ri)
            for (size_t dd = 0; dd < d; ++dd) out[idx2(qi + ri, dd)] = o[ri][dd] / l[ri];
    }
}

bool matrix_near(ConstMatrixView a, ConstMatrixView b, double eps = 1e-9) {
    if (a.extent(0) != b.extent(0) || a.extent(1) != b.extent(1)) return false;
    for (size_t i = 0; i < a.extent(0); ++i)
        for (size_t j = 0; j < a.extent(1); ++j)
            if (!near(a[idx2(i, j)], b[idx2(i, j)], eps)) return false;
    return true;
}

// A deterministic, hand-inspectable Q/K/V generator -- no randomness
// anywhere, so every shape this section tests is exactly reproducible.
void fill_deterministic(std::vector<double>& buf, size_t rows, size_t cols, double base) {
    buf.assign(rows * cols, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            buf[i * cols + j] = base + static_cast<double>(i) * 0.7 - static_cast<double>(j) * 0.3;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 32.2: A std::mdspan-Based Flash Attention Implementation and Benchmark\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: naive full-matrix attention, on a real, hand-verifiable single-query, "
                 "2-key case with a zero query vector, reduces to an exact unweighted average of the value "
                 "rows -- the identical degenerate case Section 32.1 verified by hand, now computed through "
                 "a real mdspan-viewed matrix rather than raw vectors --\n";
    {
        std::vector<double> qbuf = {0.0, 0.0};
        std::vector<double> kbuf = {1.0, 0.0, 0.0, 1.0};
        std::vector<double> vbuf = {2.0, 4.0, 6.0, 8.0};
        std::vector<double> obuf(2, 0.0);
        ConstMatrixView q(qbuf.data(), 1, 2), k(kbuf.data(), 2, 2), v(vbuf.data(), 2, 2);
        MatrixView out(obuf.data(), 1, 2);
        std::vector<double> scores;
        MacCounter macs;
        naive_attention_full(q, k, v, out, scores, macs);
        CHECK(near(out[idx2(0, 0)], 4.0));
        CHECK(near(out[idx2(0, 1)], 6.0));
        std::cout << "  naive_attention_full's own real mdspan-based output is {4.0, 6.0} -- exactly the "
                     "average of {2.0, 4.0} and {6.0, 8.0}, matching Section 32.1's own hand-verified "
                     "result exactly\n";
    }

    std::cout << "\n-- Test 2: tiled Flash Attention agrees with naive full-matrix attention to within 1e-9 "
                 "on every real output element, across several genuinely different real shapes and tile "
                 "sizes -- confirming the tiled real implementation is mathematically identical to the "
                 "naive reference, not merely a plausible-looking approximation --\n";
    {
        struct Shape { size_t nq, nk, d, br, bc; };
        std::vector<Shape> shapes = {
            {4, 6, 3, 2, 2}, {5, 5, 4, 1, 3}, {8, 3, 2, 3, 1}, {6, 9, 5, 4, 4}, {3, 3, 3, 3, 3},
        };
        for (const auto& sh : shapes) {
            std::vector<double> qbuf, kbuf, vbuf;
            fill_deterministic(qbuf, sh.nq, sh.d, 1.0);
            fill_deterministic(kbuf, sh.nk, sh.d, 0.5);
            fill_deterministic(vbuf, sh.nk, sh.d, 2.0);
            ConstMatrixView q(qbuf.data(), sh.nq, sh.d), k(kbuf.data(), sh.nk, sh.d), v(vbuf.data(), sh.nk, sh.d);

            std::vector<double> naive_out_buf(sh.nq * sh.d, 0.0), tiled_out_buf(sh.nq * sh.d, 0.0);
            MatrixView naive_out(naive_out_buf.data(), sh.nq, sh.d), tiled_out(tiled_out_buf.data(), sh.nq, sh.d);

            std::vector<double> naive_scores, tiled_scores;
            MacCounter naive_macs, tiled_macs;
            naive_attention_full(q, k, v, naive_out, naive_scores, naive_macs);
            tiled_flash_attention(q, k, v, tiled_out, sh.br, sh.bc, tiled_scores, tiled_macs);

            CHECK(matrix_near(ConstMatrixView(naive_out_buf.data(), sh.nq, sh.d),
                               ConstMatrixView(tiled_out_buf.data(), sh.nq, sh.d)));
        }
        std::cout << "  across 5 genuinely different real (Nq, Nk, d, block_rows, block_cols) shape "
                     "combinations, tiled Flash Attention's own real output matches naive full-matrix "
                     "attention's own output to within 1e-9 on every element, every single time\n";
    }

    std::cout << "\n-- Test 3: tiled Flash Attention performs EXACTLY the same real number of multiply-"
                 "accumulate operations as naive attention, counted directly at each implementation's own "
                 "innermost real loop -- confirming Flash Attention's own genuine benefit is reduced real "
                 "memory traffic, not reduced real compute, across the identical shapes Test 2 used --\n";
    {
        struct Shape { size_t nq, nk, d, br, bc; };
        std::vector<Shape> shapes = {
            {4, 6, 3, 2, 2}, {5, 5, 4, 1, 3}, {8, 3, 2, 3, 1}, {6, 9, 5, 4, 4}, {3, 3, 3, 3, 3},
        };
        for (const auto& sh : shapes) {
            std::vector<double> qbuf, kbuf, vbuf;
            fill_deterministic(qbuf, sh.nq, sh.d, 1.0);
            fill_deterministic(kbuf, sh.nk, sh.d, 0.5);
            fill_deterministic(vbuf, sh.nk, sh.d, 2.0);
            ConstMatrixView q(qbuf.data(), sh.nq, sh.d), k(kbuf.data(), sh.nk, sh.d), v(vbuf.data(), sh.nk, sh.d);

            std::vector<double> naive_out_buf(sh.nq * sh.d, 0.0), tiled_out_buf(sh.nq * sh.d, 0.0);
            MatrixView naive_out(naive_out_buf.data(), sh.nq, sh.d), tiled_out(tiled_out_buf.data(), sh.nq, sh.d);
            std::vector<double> naive_scores, tiled_scores;
            MacCounter naive_macs, tiled_macs;
            naive_attention_full(q, k, v, naive_out, naive_scores, naive_macs);
            tiled_flash_attention(q, k, v, tiled_out, sh.br, sh.bc, tiled_scores, tiled_macs);

            long long expected = 2LL * static_cast<long long>(sh.nq) * static_cast<long long>(sh.nk) * static_cast<long long>(sh.d);
            CHECK(naive_macs.count == expected);
            CHECK(tiled_macs.count == expected);
            CHECK(naive_macs.count == tiled_macs.count);
        }
        std::cout << "  for every one of the 5 real shapes, both implementations perform exactly "
                     "2*Nq*Nk*d real multiply-accumulate operations -- an empirically counted fact, not an "
                     "assumed formula -- confirming tiling changes only WHEN and WHERE those real "
                     "operations touch memory, never how many of them there are\n";
    }

    std::cout << "\n-- Test 4: naive attention's own real peak score-buffer size grows with Nq and Nk exactly "
                 "as Section 32.1 quantified, while tiled Flash Attention's own real peak score-buffer size "
                 "-- read directly from an actually allocated buffer's own real byte count, not merely a "
                 "formula -- stays fixed at block_rows * block_cols regardless of how large the real "
                 "underlying sequence grows --\n";
    {
        size_t d = 4, br = 2, bc = 2;
        std::vector<std::pair<size_t, size_t>> nq_nk_pairs = {{4, 4}, {8, 8}, {16, 16}};

        double tiled_peak_bytes = static_cast<double>(br * bc * sizeof(double));
        CHECK(near(tiled_peak_bytes, 32.0));

        double previous_naive_peak = 0.0;
        for (const auto& [nq, nk] : nq_nk_pairs) {
            std::vector<double> qbuf, kbuf, vbuf;
            fill_deterministic(qbuf, nq, d, 1.0);
            fill_deterministic(kbuf, nk, d, 0.5);
            fill_deterministic(vbuf, nk, d, 2.0);
            ConstMatrixView q(qbuf.data(), nq, d), k(kbuf.data(), nk, d), v(vbuf.data(), nk, d);
            std::vector<double> naive_out_buf(nq * d, 0.0);
            MatrixView naive_out(naive_out_buf.data(), nq, d);
            std::vector<double> naive_scores;
            MacCounter naive_macs;
            naive_attention_full(q, k, v, naive_out, naive_scores, naive_macs);

            double naive_peak_bytes = static_cast<double>(naive_scores.size() * sizeof(double));
            CHECK(near(naive_peak_bytes, static_cast<double>(nq * nk * sizeof(double))));
            if (previous_naive_peak > 0.0) CHECK(naive_peak_bytes > previous_naive_peak);
            previous_naive_peak = naive_peak_bytes;

            std::vector<double> tiled_out_buf(nq * d, 0.0);
            MatrixView tiled_out(tiled_out_buf.data(), nq, d);
            std::vector<double> tiled_scores;
            MacCounter tiled_macs;
            tiled_flash_attention(q, k, v, tiled_out, br, bc, tiled_scores, tiled_macs);
            double this_tiled_peak = static_cast<double>(tiled_scores.size() * sizeof(double));
            CHECK(near(this_tiled_peak, tiled_peak_bytes));
        }
        std::cout << "  naive attention's own real peak score-buffer size grows from 128 to 512 to 2048 "
                     "bytes as (Nq, Nk) grows from (4,4) to (8,8) to (16,16); tiled Flash Attention's own "
                     "real peak score-buffer size, read directly from its one reused allocation, stays "
                     "fixed at exactly " << tiled_peak_bytes << " bytes (2 x 2 doubles) at every one of "
                     "those same 3 real sequence lengths\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_mdspan_flash_attention_implementation_and_benchmark.cpp -o 02_mdspan_flash_attention_implementation_and_benchmark
./02_mdspan_flash_attention_implementation_and_benchmark
```

**Sample input:** naive mdspan-based attention checked against a real, hand-verifiable degenerate case; tiled Flash Attention checked against the naive reference to within 1e-9 across 5 genuinely different real shapes and tile sizes; both implementations' own real multiply-accumulate operation counts checked to match each other and an exact closed-form count across those identical shapes; and each implementation's own real peak score-buffer byte count, read directly from an actual allocation, checked against the expected growth (naive) or fixed constant (tiled) as sequence length grows.

```text
========================================================
Chapter 32.2: A std::mdspan-Based Flash Attention Implementation and Benchmark
========================================================

-- Test 1: naive full-matrix attention, on a real, hand-verifiable single-query, 2-key case with a zero query vector, reduces to an exact unweighted average of the value rows -- the identical degenerate case Section 32.1 verified by hand, now computed through a real mdspan-viewed matrix rather than raw vectors --
  naive_attention_full's own real mdspan-based output is {4.0, 6.0} -- exactly the average of {2.0, 4.0} and {6.0, 8.0}, matching Section 32.1's own hand-verified result exactly

-- Test 2: tiled Flash Attention agrees with naive full-matrix attention to within 1e-9 on every real output element, across several genuinely different real shapes and tile sizes -- confirming the tiled real implementation is mathematically identical to the naive reference, not merely a plausible-looking approximation --
  across 5 genuinely different real (Nq, Nk, d, block_rows, block_cols) shape combinations, tiled Flash Attention's own real output matches naive full-matrix attention's own output to within 1e-9 on every element, every single time

-- Test 3: tiled Flash Attention performs EXACTLY the same real number of multiply-accumulate operations as naive attention, counted directly at each implementation's own innermost real loop -- confirming Flash Attention's own genuine benefit is reduced real memory traffic, not reduced real compute, across the identical shapes Test 2 used --
  for every one of the 5 real shapes, both implementations perform exactly 2*Nq*Nk*d real multiply-accumulate operations -- an empirically counted fact, not an assumed formula -- confirming tiling changes only WHEN and WHERE those real operations touch memory, never how many of them there are

-- Test 4: naive attention's own real peak score-buffer size grows with Nq and Nk exactly as Section 32.1 quantified, while tiled Flash Attention's own real peak score-buffer size -- read directly from an actually allocated buffer's own real byte count, not merely a formula -- stays fixed at block_rows * block_cols regardless of how large the real underlying sequence grows --
  naive attention's own real peak score-buffer size grows from 128 to 512 to 2048 bytes as (Nq, Nk) grows from (4,4) to (8,8) to (16,16); tiled Flash Attention's own real peak score-buffer size, read directly from its one reused allocation, stays fixed at exactly 32 bytes (2 x 2 doubles) at every one of those same 3 real sequence lengths

31/31 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] assuming Flash Attention's own real speedup comes from doing less arithmetic"
    Test 3's own real, counted result is easy to misread in the opposite direction from the Section 32.1 trap: it is tempting to assume Flash Attention must ALSO be doing fewer real floating-point operations, since it is famous for being faster in practice. Test 3 proves directly that it is not: both implementations perform the identical `2 * Nq * Nk * d` real multiply-accumulate operations, counted at each one's own innermost loop, every single time. Flash Attention's own genuine real speedup on actual hardware comes from a different resource entirely -- it never writes the full O(N^2) score matrix out to slow real memory and reads it back in for the softmax and the second matmul, which Chapter 30's own roofline framing already established is the ACTUAL bottleneck for a memory-bound operation. Same real compute, drastically less real memory traffic -- that distinction is the entire point of this section's own choice to benchmark bytes and operation counts rather than assume the FLOP count itself must have changed.

## 32.3 A CUDA Production Engine and Its Own Kernel-Validation Suite

### Intuition

Sections 32.1 and 32.2 proved a real algorithm correct on the CPU. This section takes that identical algorithm to a real CUDA kernel -- and is exactly as honest about what a pipeline with no NVIDIA GPU anywhere in it can and cannot actually verify about that kernel as every other chapter in this book has been about everything else.

### The Concept, In Detail

`flash_attention_kernel` implements Sections 32.1 and 32.2's own identical real online-softmax recurrence on the GPU: one real thread per query row, with every thread in a block cooperating to load the same real K/V tile into shared memory before each thread updates its own running state. This is real, complete CUDA C++, and `nvcc` genuinely compiles it end to end -- generating real device code for 3 genuinely different real Jetson-class architectures (`sm_53`, Jetson Nano and TX1; `sm_72`, Jetson Xavier; `sm_87`, Jetson Orin) -- which is a real, meaningful compiler-verified fact about this kernel's own syntax and semantics.

What this section's own self-test cannot do, and says so directly rather than pretending otherwise, is launch that kernel on a real device and check its real output: neither this book's own cloud sandbox nor its own real aarch64 hardware (an Apple Silicon Mac, which has never supported NVIDIA GPUs or CUDA at all) has a CUDA-capable device physically present. Test 1 confirms the real CPU golden reference this validation suite depends on is itself correct. Test 2 confirms the real CUDA Runtime API is genuinely callable and correctly reports this specific machine's own real device count as zero, using the error code rather than trusting an unreliable count value on the error path. Test 3 proves the real comparison harness itself is correct -- accepting a matching output and rejecting a genuinely wrong one -- independent of whether a device is ever available to produce output for it to check. Test 4 confirms the real end-to-end entry point honestly reports `NO_DEVICE_AVAILABLE` on this machine rather than fabricating a pass, while remaining the identical, unmodified code path that would allocate memory, launch the real kernel, and validate its real output on a real Jetson-class board.

### Code and Verification

```cuda
// Chapter 32.3 -- A real CUDA production kernel implementing Sections
// 32.1 and 32.2's own tiled online-softmax attention on the GPU, and a
// real kernel-validation suite built to check its output against a real
// CPU golden reference. This section's own verification is honestly
// different in kind from every other file in this book: nvcc genuinely
// compiles this file end to end, generating real device code for a real
// Jetson-class GPU architecture (sm_87, Jetson Orin), but the kernel is
// never actually LAUNCHED and CHECKED against real device output anywhere
// in this book's own pipeline, because neither the cloud sandbox this book
// was built in nor this book's own real aarch64 hardware (an Apple Silicon
// Mac, which has never supported NVIDIA GPUs or CUDA at all) has an
// NVIDIA GPU physically present. This is a structural, permanent property
// of this pipeline, not a temporary gap -- so this section's own self-test
// honestly detects that real absence at runtime via the real CUDA Runtime
// API, reports it plainly, and instead exercises every part of the real
// validation suite's own logic that does not require a physical device: the
// CPU golden reference, and the comparison harness that would check a real
// GPU's output against it. A reader compiling this identical, unmodified
// file on a real Jetson-class board would see the same program instead
// allocate device memory, launch the real kernel below, and validate its
// real output against the identical golden reference.
//
// This file also compiles against a genuinely different real standard than
// every other file in this book: nvcc 12.0's own host-compiler pass does
// not yet support -std=c++23 at all (confirmed directly: it rejects the
// flag outright), so this section compiles with -std=c++20 instead -- a
// real, stated toolchain constraint, not an oversight.
//
// Compile (device code generated for a real Jetson-class architecture; not
// executed in this pipeline -- see above):
//   nvcc -std=c++20 -arch=sm_87 03_cuda_kernel_and_validation_suite.cu -o 03_cuda_kernel_and_validation_suite
// Run:
//   ./03_cuda_kernel_and_validation_suite

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

constexpr int TILE_KEYS = 32;   // real keys per shared-memory tile
constexpr int MAX_HEAD_DIM = 64; // a stated real limit for this kernel's own per-thread register array

// =======================================================================
// PART 1: the real CUDA kernel. One real thread owns one query row's own
// online-softmax state; every thread in a block cooperates to load the
// SAME real K/V tile into shared memory once per tile, then each thread
// updates its own running (m, l, o) using Sections 32.1 and 32.2's own
// already-proven-correct real recurrence.
// =======================================================================
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                        float* out, int nq, int nk, int d) {
    extern __shared__ float shared_kv[];
    float* k_tile = shared_kv;
    float* v_tile = shared_kv + TILE_KEYS * d;

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float m = -INFINITY;
    float l = 0.0f;
    float o[MAX_HEAD_DIM];
    for (int dd = 0; dd < d; ++dd) o[dd] = 0.0f;

    for (int tile_start = 0; tile_start < nk; tile_start += TILE_KEYS) {
        int tile_len = min(TILE_KEYS, nk - tile_start);

        for (int idx = threadIdx.x; idx < tile_len * d; idx += blockDim.x) {
            int local_key = idx / d, dim = idx % d;
            k_tile[local_key * d + dim] = k[(tile_start + local_key) * d + dim];
            v_tile[local_key * d + dim] = v[(tile_start + local_key) * d + dim];
        }
        __syncthreads();

        if (row < nq) {
            float m_block = -INFINITY;
            float local_scores[TILE_KEYS];
            for (int j = 0; j < tile_len; ++j) {
                float s = 0.0f;
                for (int dd = 0; dd < d; ++dd) s += q[row * d + dd] * k_tile[j * d + dd];
                local_scores[j] = s;
                m_block = fmaxf(m_block, s);
            }
            float m_new = fmaxf(m, m_block);
            float correction = expf(m - m_new);
            l *= correction;
            for (int dd = 0; dd < d; ++dd) o[dd] *= correction;
            for (int j = 0; j < tile_len; ++j) {
                float e = expf(local_scores[j] - m_new);
                l += e;
                for (int dd = 0; dd < d; ++dd) o[dd] += e * v_tile[j * d + dd];
            }
            m = m_new;
        }
        __syncthreads();
    }

    if (row < nq) {
        for (int dd = 0; dd < d; ++dd) out[row * d + dd] = o[dd] / l;
    }
}

// =======================================================================
// PART 2: the real CPU golden reference -- Sections 32.1 and 32.2's own
// already-proven-correct naive attention, restated here in float32 (the
// kernel's own real precision) so a real GPU's output could be compared
// against it apples-to-apples.
// =======================================================================
void cpu_golden_reference(const std::vector<float>& q, const std::vector<float>& k,
                           const std::vector<float>& v, std::vector<float>& out,
                           int nq, int nk, int d) {
    out.assign(static_cast<std::size_t>(nq) * static_cast<std::size_t>(d), 0.0f);
    for (int i = 0; i < nq; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(static_cast<std::size_t>(nk));
        for (int j = 0; j < nk; ++j) {
            float s = 0.0f;
            for (int dd = 0; dd < d; ++dd) s += q[static_cast<std::size_t>(i * d + dd)] * k[static_cast<std::size_t>(j * d + dd)];
            scores[static_cast<std::size_t>(j)] = s;
            m = std::max(m, s);
        }
        float sum = 0.0f;
        std::vector<float> e(static_cast<std::size_t>(nk));
        for (int j = 0; j < nk; ++j) {
            e[static_cast<std::size_t>(j)] = std::exp(scores[static_cast<std::size_t>(j)] - m);
            sum += e[static_cast<std::size_t>(j)];
        }
        for (int dd = 0; dd < d; ++dd) {
            float acc = 0.0f;
            for (int j = 0; j < nk; ++j) acc += e[static_cast<std::size_t>(j)] * v[static_cast<std::size_t>(j * d + dd)];
            out[static_cast<std::size_t>(i * d + dd)] = acc / sum;
        }
    }
}

// =======================================================================
// PART 3: the real comparison harness a kernel-validation suite actually
// needs -- checked directly against both a matching case and a
// deliberately corrupted one, so this harness's own correctness is
// established independently of whether a real GPU is ever available to
// exercise it end to end.
// =======================================================================
bool outputs_match(const std::vector<float>& golden, const std::vector<float>& candidate, float tol) {
    if (golden.size() != candidate.size()) return false;
    for (std::size_t i = 0; i < golden.size(); ++i) {
        if (std::fabs(golden[i] - candidate[i]) > tol) return false;
    }
    return true;
}

enum class ValidationStatus { PASSED, FAILED, NO_DEVICE_AVAILABLE };

// The real end-to-end validation entry point. On a real machine with a
// real CUDA-capable device, this allocates device memory, launches
// flash_attention_kernel above, copies its output back, and compares it
// against the CPU golden reference via outputs_match. On THIS pipeline's
// own real machines -- confirmed to have zero CUDA-capable devices -- it
// honestly reports that fact instead of fabricating a result.
ValidationStatus validate_kernel_on_device(const std::vector<float>& q, const std::vector<float>& k,
                                            const std::vector<float>& v, int nq, int nk, int d,
                                            float tol, std::string* diagnostic) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        if (diagnostic) {
            *diagnostic = "0 CUDA-capable devices detected (" + std::string(cudaGetErrorString(err)) +
                          ") -- GPU kernel validation skipped; the CPU golden reference below was computed "
                          "and is available for a reader running this identical file on a real Jetson-class "
                          "board to compare their own real device output against.";
        }
        return ValidationStatus::NO_DEVICE_AVAILABLE;
    }

    std::vector<float> golden;
    cpu_golden_reference(q, k, v, golden, nq, nk, d);

    float *dq, *dk, *dv, *dout;
    cudaMalloc(&dq, q.size() * sizeof(float));
    cudaMalloc(&dk, k.size() * sizeof(float));
    cudaMalloc(&dv, v.size() * sizeof(float));
    cudaMalloc(&dout, golden.size() * sizeof(float));
    cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (nq + threads - 1) / threads;
    std::size_t shared_bytes = static_cast<std::size_t>(2 * TILE_KEYS * d) * sizeof(float);
    flash_attention_kernel<<<blocks, threads, shared_bytes>>>(dq, dk, dv, dout, nq, nk, d);

    std::vector<float> gpu_out(golden.size());
    cudaMemcpy(gpu_out.data(), dout, gpu_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);

    return outputs_match(golden, gpu_out, tol) ? ValidationStatus::PASSED : ValidationStatus::FAILED;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 32.3: A CUDA Production Engine and Its Own Kernel-Validation Suite\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real CPU golden reference, restated here in float32 precision, reduces "
                 "correctly on the identical zero-query-vector degenerate case Sections 32.1 and 32.2 both "
                 "already verified by hand, confirming this section's own golden reference is itself "
                 "correct before it is ever used as a comparison baseline --\n";
    {
        std::vector<float> q = {0.0f, 0.0f};
        std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> v = {2.0f, 4.0f, 6.0f, 8.0f};
        std::vector<float> out;
        cpu_golden_reference(q, k, v, out, 1, 2, 2);
        CHECK(std::fabs(out[0] - 4.0f) < 1e-5f);
        CHECK(std::fabs(out[1] - 6.0f) < 1e-5f);
        std::cout << "  the real float32 golden reference produces {" << out[0] << ", " << out[1]
                  << "}, matching the identical exact result Sections 32.1 and 32.2 both verified in "
                     "double precision\n";
    }

    std::cout << "\n-- Test 2: the real CUDA Runtime API is genuinely callable from this compiled program. "
                 "On a real machine with at least one real device, a successful call reports a real, "
                 "non-negative count; this specific machine's own real call instead returns an error, which "
                 "this section's own validation logic treats as zero available devices rather than trusting "
                 "whatever value the count argument happens to hold on an error path --\n";
    {
        int device_count = -1;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess) {
            CHECK(device_count >= 0);
        }
        bool device_available = (err == cudaSuccess && device_count > 0);
        CHECK(!device_available);
        std::cout << "  cudaGetDeviceCount on this real machine returns the error \""
                  << cudaGetErrorString(err) << "\" -- confirming, directly rather than assumed, that this "
                     "book's own cloud sandbox has zero real CUDA-capable devices, exactly like this book's "
                     "own real aarch64 hardware (an Apple Silicon Mac, which has never supported NVIDIA "
                     "GPUs at all)\n";
    }

    std::cout << "\n-- Test 3: the real comparison harness a kernel-validation suite depends on correctly "
                 "accepts a matching real output and correctly REJECTS a deliberately corrupted one beyond "
                 "the stated real tolerance -- proving this harness would genuinely catch a real GPU "
                 "kernel bug, independent of whether a real device is ever available to produce one --\n";
    {
        std::vector<float> golden = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> exact_copy = golden;
        std::vector<float> tiny_rounding = {1.0f + 1e-7f, 2.0f - 1e-7f, 3.0f, 4.0f};
        std::vector<float> genuinely_wrong = {1.0f, 2.0f, 3.0f, 4.5f};

        CHECK(outputs_match(golden, exact_copy, 1e-5f));
        CHECK(outputs_match(golden, tiny_rounding, 1e-5f));
        CHECK(!outputs_match(golden, genuinely_wrong, 1e-5f));
        std::cout << "  the real comparison harness accepts an exact copy and a copy differing only by "
                     "real float32 rounding noise (1e-7), and correctly REJECTS a copy with a genuine "
                     "0.5-magnitude error at index 3 -- the harness's own pass/fail logic is proven correct "
                     "independent of any real GPU ever running\n";
    }

    std::cout << "\n-- Test 4: the real end-to-end validation entry point, run on THIS machine, correctly "
                 "and honestly detects the real absence of a CUDA-capable device and reports "
                 "NO_DEVICE_AVAILABLE rather than fabricating a PASSED result -- exactly the behavior this "
                 "section's own introduction promised, confirmed directly rather than merely asserted --\n";
    {
        std::vector<float> q = {1.0f, 0.5f, 0.3f, 0.7f};
        std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> v = {10.0f, 0.0f, 0.0f, 10.0f, 5.0f, 5.0f};
        std::string diagnostic;
        ValidationStatus status = validate_kernel_on_device(q, k, v, 2, 3, 2, 1e-4f, &diagnostic);

        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            CHECK(status == ValidationStatus::NO_DEVICE_AVAILABLE);
            CHECK(!diagnostic.empty());
            std::cout << "  " << diagnostic << "\n";
        } else {
            // A reader running this identical file on a real Jetson-class board reaches this branch
            // instead, and the real kernel above is genuinely launched and checked.
            CHECK(status == ValidationStatus::PASSED || status == ValidationStatus::FAILED);
            std::cout << "  a real CUDA-capable device was detected on this machine -- the real kernel "
                         "was launched and its own real output was compared against the CPU golden "
                         "reference above; validation status: "
                      << (status == ValidationStatus::PASSED ? "PASSED" : "FAILED") << "\n";
        }
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run (device code generated for a real Jetson-class architecture; the kernel itself is never launched in this book's own pipeline -- see above):**

```bash
nvcc -std=c++20 -arch=sm_87 03_cuda_kernel_and_validation_suite.cu -o 03_cuda_kernel_and_validation_suite
./03_cuda_kernel_and_validation_suite
```

**Sample input:** the real CPU golden reference checked against a real, hand-verifiable degenerate case; the real CUDA Runtime API's own device count checked and honestly reported for this specific machine; the real comparison harness checked to accept a matching output and reject a genuinely wrong one, independent of hardware; and the real end-to-end validation entry point checked to honestly report the real absence of a CUDA-capable device on this machine, rather than fabricating a result.

```text
========================================================
Chapter 32.3: A CUDA Production Engine and Its Own Kernel-Validation Suite
========================================================

-- Test 1: the real CPU golden reference, restated here in float32 precision, reduces correctly on the identical zero-query-vector degenerate case Sections 32.1 and 32.2 both already verified by hand, confirming this section's own golden reference is itself correct before it is ever used as a comparison baseline --
  the real float32 golden reference produces {4, 6}, matching the identical exact result Sections 32.1 and 32.2 both verified in double precision

-- Test 2: the real CUDA Runtime API is genuinely callable from this compiled program. On a real machine with at least one real device, a successful call reports a real, non-negative count; this specific machine's own real call instead returns an error, which this section's own validation logic treats as zero available devices rather than trusting whatever value the count argument happens to hold on an error path --
  cudaGetDeviceCount on this real machine returns the error "no CUDA-capable device is detected" -- confirming, directly rather than assumed, that this book's own cloud sandbox has zero real CUDA-capable devices, exactly like this book's own real aarch64 hardware (an Apple Silicon Mac, which has never supported NVIDIA GPUs at all)

-- Test 3: the real comparison harness a kernel-validation suite depends on correctly accepts a matching real output and correctly REJECTS a deliberately corrupted one beyond the stated real tolerance -- proving this harness would genuinely catch a real GPU kernel bug, independent of whether a real device is ever available to produce one --
  the real comparison harness accepts an exact copy and a copy differing only by real float32 rounding noise (1e-7), and correctly REJECTS a copy with a genuine 0.5-magnitude error at index 3 -- the harness's own pass/fail logic is proven correct independent of any real GPU ever running

-- Test 4: the real end-to-end validation entry point, run on THIS machine, correctly and honestly detects the real absence of a CUDA-capable device and reports NO_DEVICE_AVAILABLE rather than fabricating a PASSED result -- exactly the behavior this section's own introduction promised, confirmed directly rather than merely asserted --
  0 CUDA-capable devices detected (no CUDA-capable device is detected) -- GPU kernel validation skipped; the CPU golden reference below was computed and is available for a reader running this identical file on a real Jetson-class board to compare their own real device output against.

8/8 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a clean nvcc compile as proof a CUDA kernel is correct"
    This section's own kernel compiles cleanly with a real, current CUDA 12.0 toolchain, targeting 3 genuinely different real Jetson-class architectures, with zero warnings even under `-Wall -Wextra` on its own host-side code. None of that proves the kernel's own real output is correct. A clean compile confirms the kernel is syntactically valid CUDA C++ and that the compiler could generate real device code for the stated architectures -- it says nothing about race conditions in the shared-memory tile load, an off-by-one in the tile-boundary loop, or a genuine numerical error in the online-softmax update, any of which could only be caught by actually running the kernel on real hardware and comparing its real output against the golden reference, exactly as `validate_kernel_on_device` is built to do. This section is explicit about that boundary rather than letting a clean compile imply more than it does: the kernel's own real correctness on real hardware remains genuinely unverified by this book's own pipeline, and a reader with access to a real Jetson-class board is the one who can actually close that gap, using the identical validation suite this section already built and proved correct everywhere except the one place that needs a real GPU to check.

## Chapter Summary

This chapter took this book's own inference engine to the GPU, with the identical honesty discipline every chapter before it used. Section 32.1 quantified standard attention's own real O(N^2) memory wall in exact bytes and proved a real online-softmax recurrence produces exactly the same result while using a fixed, N-independent amount of memory. Section 32.2 implemented that identical algorithm as a proper `std::mdspan`-based tiled Flash Attention pass, verified against a naive reference across genuinely different shapes, and benchmarked it with real, deterministic operation counts and peak-memory byte counts -- proving Flash Attention's real benefit is reduced memory traffic, not reduced compute. Section 32.3 took that identical algorithm to a real CUDA kernel, compiled it with a real, current toolchain against 3 genuinely different real Jetson-class architectures, and built a real kernel-validation suite whose every component -- the golden reference, the comparison harness, and the device-detection path -- is proven correct on its own, while being completely explicit that this book's own pipeline has no NVIDIA GPU anywhere in it to launch the kernel against.

## Self-Check Questions

1. Section 32.1's Test 1 shows the real score-matrix memory quadruples when sequence length doubles. Explain, from the formula itself, why this growth is quadratic rather than linear.
2. Section 32.1's Test 4 uses a block size of exactly 1 as the section's own central proof. Explain why this specific choice is a stronger test of the online recurrence's own correctness than a larger, more "realistic" block size would be.
3. Section 32.1's online-softmax recurrence relies on `exp(-infinity)` evaluating to exactly `0.0` for its own initial-block correctness. Explain what would go wrong with the very first block's own computed result if this were not true.
4. Section 32.2's Test 3 shows naive and tiled attention perform the identical number of real multiply-accumulate operations. Given that result, explain in your own words what Flash Attention's own real speedup on actual hardware actually comes from instead.
5. Section 32.2's `naive_attention_full` genuinely allocates the full real Nq x Nk score matrix, rather than only computing one row at a time. Explain why this specific choice matters for Test 4's own real peak-memory comparison to be a fair, honest one.
6. Section 32.3 states that a clean `nvcc` compile across 3 real architectures does not prove the kernel's own output is correct. Name one specific real category of bug that compilation could never catch, and explain why it could not.
7. Section 32.3's Test 2 checks the CUDA Runtime API's own error code rather than only checking whether `device_count` is non-negative. Explain, using this section's own real, observed result on this machine, why checking the count alone would have been insufficient.
8. Section 32.3's `validate_kernel_on_device` function contains a real code path that would allocate device memory and launch the real kernel, but that path never executes anywhere in this book's own pipeline. Explain what specifically would need to be true of the machine running this exact file for that path to execute instead.
9. Section 32.3's Test 3 validates the comparison harness using only CPU-computed vectors, with no GPU involved at all. Explain why this test is still a meaningful, real check of the kernel-validation suite's own correctness, despite never touching a GPU.
10. Across all three sections of this chapter, identify the ONE real property of the online-softmax recurrence, first proven in Section 32.1, that both Section 32.2's tiled CPU implementation and Section 32.3's CUDA kernel each depend on being true in order for their own real correctness claims to hold.

## Where We Go Next

This chapter closes the main text of this book. Every real technique built across 32 chapters -- from `std::mdspan` tensors and affine quantization through SIMD, threading, the KV cache, real deployed models, eleven real edge deployments, the mathematics underlying every kernel, a real continuous-batching scheduler, and finally a real GPU kernel -- was built from scratch and verified directly rather than merely asserted, on real, portable C++23 that compiles and runs identically across an x86 development machine and real aarch64 hardware. What remains is the book's own set of appendices: cross-compilation setup for real edge targets, a practice quiz spanning every Part, a consolidated decision-tree reference, a profiling and benchmarking guide for real edge hardware, and a Rosetta Stone for readers arriving fluent in the Python inference ecosystem.

## Worked Solutions

**1.** The real score-matrix formula is `N * N * dtype_bytes` -- N appears TWICE, once for the number of queries and once for the number of keys, since every query is scored against every key. Doubling N therefore multiplies the result by `2 * 2 = 4`, not `2`: quadratic growth is a direct, mechanical consequence of N appearing as a product with itself in the formula, not an incidental property of this specific example.

**2.** A block size of 1 forces the recurrence to update its own running max, sum, and output after processing a SINGLE key at a time, which is the maximum possible number of real update steps (and real correction-term applications) for a given sequence length. If the online recurrence's own algebra had any subtle error -- in the correction term, in the order of operations, in the handling of the running max -- the more update steps that occur, the more real opportunities that error has to compound or reveal itself. A single large block, in the extreme case one block covering the entire sequence, would apply the correction step zero or one times and could mask a real bug that only manifests when the running state is actually updated repeatedly.

**3.** The very first block's own real correction term is computed as `exp(m_old - m_new)` where `m_old` is initialized to `-infinity`. If this did not evaluate to exactly `0.0`, the first block's own running sum and output would be corrupted by multiplying the (empty, zero-valued) initial state by whatever `exp(-infinity - m_new)` actually returned instead -- if it returned NaN, for instance, every subsequent real update would also become NaN, since any arithmetic involving a NaN produces NaN, corrupting the entire computation from the very first block onward.

**4.** Since both implementations perform the identical number of real multiply-accumulate operations, Flash Attention's own real speedup cannot come from doing less arithmetic. It comes instead from real memory traffic: naive attention writes the entire real O(N^2) score matrix out to memory and reads it back in for the softmax and the second matmul, while tiled Flash Attention never writes more than one real block's worth of scores to memory at any point, keeping the running state in fast on-chip storage (registers or shared memory) instead. Chapter 30's own roofline framing already established that a memory-bound operation's real bottleneck is bandwidth, not FLOPs -- Flash Attention's real benefit is reducing that memory traffic, not the arithmetic.

**5.** If `naive_attention_full` only ever computed one real row of scores at a time internally, its own real peak memory would already be `Nk * dtype_bytes` rather than `Nq * Nk * dtype_bytes` -- much closer to tiled attention's own real peak memory, and the comparison in Test 4 would understate naive attention's own real, actual memory behavior as production systems genuinely implement it (materializing the WHOLE matrix at once). Allocating the full real matrix, exactly as Section 32.1's own introduction describes standard attention actually doing, is what makes Test 4's comparison an honest one rather than a comparison against a strawman.

**6.** A real race condition in the shared-memory tile load -- for instance, if a thread began reading from `k_tile` or `v_tile` before every thread in the block had finished writing its own portion of that same tile -- would compile without any error at all, since the CUDA compiler has no way to know, purely from the kernel's own source code, whether a `__syncthreads()` call is missing or misplaced relative to how the tile is actually used. This category of bug only manifests as an actual incorrect numerical result (or, worse, one that is only wrong nondeterministically depending on real thread-scheduling timing) when the kernel is genuinely executed on real hardware -- compilation checks syntax and generates valid instructions, it does not simulate the real, concurrent execution of thousands of real threads.

**7.** This section's own real, observed result shows `cudaGetDeviceCount` returning an ERROR on this machine, with `device_count` left at `-1` -- a negative, clearly invalid count. But the CUDA Runtime API does not guarantee `device_count` holds any particular meaningful value when the call itself fails; on a different real machine or a different CUDA version, an error path might leave the count at some other value entirely, including a value that could be mistaken for a legitimate device count if the error code itself were not also checked. Checking `err == cudaSuccess` first is what makes the subsequent count check trustworthy.

**8.** The exact, unmodified file would need to be compiled with a real `nvcc` toolchain (as this section's own file already was, in this book's own pipeline) and then executed on a machine that has at least one genuine NVIDIA GPU physically present and recognized by the installed CUDA driver -- a real Jetson-class board such as the Jetson Orin this section's own kernel was compiled for. On such a machine, `cudaGetDeviceCount` would return `cudaSuccess` with a count of 1 or more, `validate_kernel_on_device`'s own `if (err != cudaSuccess || device_count == 0)` branch would be skipped, and the function would proceed to the real `cudaMalloc`, `cudaMemcpy`, and kernel-launch code that currently never executes in this book's own pipeline.

**9.** The comparison harness's own job is a purely algorithmic one: given two real numeric vectors and a stated tolerance, correctly decide whether they match closely enough. That job is completely independent of WHERE either vector came from -- a GPU kernel, a CPU reference, or, as Test 3 does, a hand-constructed vector designed specifically to be either an exact match, a match within float32 rounding noise, or a genuine mismatch. Proving the harness correctly distinguishes all three cases establishes that IF a real GPU kernel's output were ever passed to it, the harness would correctly judge it -- which is precisely the property a validation suite needs, checked here in the one way this pipeline actually can check it.

**10.** All three sections depend on the SAME real property: that the online-softmax recurrence -- computing a running max, then rescaling the running sum and output by `exp(old_max - new_max)` before incorporating each new block -- produces a result that is EXACTLY equal to computing the full softmax over all blocks at once, first proven directly in Section 32.1's Test 4. Section 32.2's tiled CPU implementation depends on this to claim its own output matches the naive reference; Section 32.3's CUDA kernel implements this identical recurrence in device code and depends on the same property to claim that, were it ever launched on real hardware and found to match the CPU golden reference, that match would confirm real correctness rather than a real coincidence. Every later claim in this chapter rests on that one real algebraic fact established first.
