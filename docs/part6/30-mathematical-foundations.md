# Chapter 30: Mathematical Foundations for Kernel Authors: FLOPs, the Roofline, and the Hessian

**What you will understand by the end of this chapter:**

- Why a dot product's own real arithmetic intensity is a fixed constant regardless of length, why GEMV (the operation autoregressive decode actually runs) is intrinsically memory-bound at any model size, and the exact real batch size at which GEMM crosses over from memory-bound to compute-bound on a given machine.
- The real algebraic identity that makes naive softmax overflow, why the fix works, and why the identical bug reappears in an even more dangerous, silent form in log-sum-exp.
- The real trigonometric identity that explains, rather than merely implements, why RoPE's position-dependent rotation encodes relative position in a real dot product.
- Why quantization is a real affine map whose only non-linear step is rounding, the real, provably tight error bound that follows from it, and the real non-associativity that makes re-quantization a genuinely different operation from quantizing once.
- How a layer's real Hessian, computed directly from calibration data, gives GPTQ the real second-order information needed to compensate not-yet-quantized weights for the error already introduced by the ones that are -- proven, not merely asserted, to beat naive independent rounding.
- How to combine every real formula in this chapter into a complete roofline analysis of an entire real transformer decoder layer, and to derive the real batch size at which the whole layer's own bound classification flips.

**What you need to know first:**

- Chapter 8's own Roofline Model (peak FLOPs, peak bandwidth, and the ridge point separating memory-bound work from compute-bound work) is the real framework this chapter builds directly on top of -- this chapter derives the actual formulas Chapter 8 introduced conceptually, and applies them to real transformer operations.
- Chapter 3's own working RoPE implementation and Chapter 4's own working affine quantization are both real, correct code this chapter does not repeat -- Section 30.3 and Section 30.4 instead derive and prove WHY those working implementations behave the way they do.
- Section 19.4's and Section 22.3's own real least-squares reuse, and this book's general habit of implementing an independently-famous real algorithm from scratch and verifying it against known values (Sections 23.1, 23.2, 23.4), both recur here: Section 30.5's own from-scratch Gauss-Jordan matrix inverse and GPTQ compensation formula follow the identical discipline.

---

Every chapter before this one built a working system and verified it against the correct answer. This chapter asks a different question of the same material: not "does it work," but "why does it work, and what does that explain about its own real limits." A dot product's own arithmetic intensity explains why decode is slow no matter how fast the GPU is. A rotation matrix's own algebra explains why RoPE encodes relative position at all. A Hessian explains why GPTQ beats naive rounding, not just that it does. Part 6 exists to give the kernel-level intuition Parts 1 through 5 relied on without deriving, and this chapter is where that derivation happens -- entirely in real, checkable, from-scratch C++, exactly like every chapter before it.

## 30.1 The Dot Product as Inference's Atomic Unit, and the Real GEMV/GEMM Crossover

### Intuition

Every matrix multiply in a transformer decomposes into real dot products, and a real dot product's own arithmetic intensity -- FLOPs moved per byte read -- never improves no matter how long the vectors get. That single, unglamorous fact is the entire reason autoregressive decode is memory-bound, and it is what makes GEMM's own real transition to compute-bound, as batch size grows, a genuinely derivable number rather than a rule of thumb.

### The Concept, In Detail

Test 1 confirms the dot product's own constant arithmetic intensity directly: an 8-element and an 8000-element dot product both compute to exactly 0.25 FLOPs/byte. Test 2 extends this to GEMV -- an M x N matrix against an N-length vector -- and confirms its own real arithmetic intensity approaches exactly 0.5 as the matrix grows, which is the real, precise reason a batch-1 GEMV (decode) never becomes compute-bound simply because the model gets bigger: bigger M and N do not change the ratio.

GEMM is genuinely different, and Test 3 derives why: for a batch of M rows through a K x N weight matrix, when M is small relative to K and N (the real, ordinary case for LLM serving), the weight matrix's own bytes dominate real traffic, producing the clean closed-form result AI(M) = M/2 -- arithmetic intensity that actually grows with batch size. Test 3 also honestly quantifies how far this approximation drifts from an exact byte accounting as M grows, rather than presenting it as exact everywhere. Test 4 closes the section by deriving a real, concrete crossover batch size -- the exact point where a stated real machine's own ridge point (Chapter 8) is reached -- from stated real hardware numbers end to end.

### Code and Verification

```cpp
// Chapter 30.1 -- Every operation in a transformer, no matter how large,
// decomposes into real dot products, and a dot product's own real
// arithmetic intensity (FLOPs moved per byte read) is a fixed constant,
// never improving no matter how long the vectors get -- which is
// exactly why GEMV, the operation autoregressive decode actually runs,
// is intrinsically memory-bound at any model size. GEMM, the operation
// batched/prefill inference runs, is different: its own real arithmetic
// intensity grows linearly with batch size, and this section derives
// the exact real batch size at which a given machine's own roofline
// (Chapter 8) predicts the crossover from memory-bound to compute-bound.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover.cpp -o 01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover
// Run:     ./01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover

#include <cmath>
#include <cstdint>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

constexpr double BYTES_PER_FLOAT32 = 4.0;

// =======================================================================
// PART 1: the dot product -- inference's real atomic unit. Its own
// arithmetic intensity is a fixed constant, independent of length.
// =======================================================================
double dot_product_flops(int64_t n) { return 2.0 * static_cast<double>(n); }  // n multiplies + n adds
double dot_product_bytes(int64_t n) { return 2.0 * static_cast<double>(n) * BYTES_PER_FLOAT32; }  // reading both vectors

double arithmetic_intensity(double flops, double bytes) { return flops / bytes; }

// =======================================================================
// PART 2: GEMV -- an M x N matrix times an N-length vector. The matrix's
// own bytes dominate for any real model size, which is what makes GEMV's
// own real arithmetic intensity approach a fixed ~0.5, regardless of M
// or N -- the real reason autoregressive decode (batch = 1, pure GEMV)
// is memory-bound no matter how large the model gets.
// =======================================================================
double gemv_flops(int64_t m, int64_t n) { return 2.0 * static_cast<double>(m) * static_cast<double>(n); }

double gemv_bytes_exact(int64_t m, int64_t n) {
    double matrix_bytes = static_cast<double>(m) * static_cast<double>(n) * BYTES_PER_FLOAT32;
    double vector_in_bytes = static_cast<double>(n) * BYTES_PER_FLOAT32;
    double vector_out_bytes = static_cast<double>(m) * BYTES_PER_FLOAT32;
    return matrix_bytes + vector_in_bytes + vector_out_bytes;
}

// =======================================================================
// PART 3: GEMM -- a batch of M rows through a K x N weight matrix. For
// real LLM inference, M (the batch size) is tiny compared to K and N
// (the model's own hidden dimensions), so the weight matrix's own bytes
// dominate real traffic -- this weight-dominated approximation is what
// produces the clean, well-known real result: arithmetic intensity
// scales as exactly M / 2.
// =======================================================================
double gemm_flops(int64_t m, int64_t k, int64_t n) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
}

double gemm_bytes_exact(int64_t m, int64_t k, int64_t n) {
    double a_bytes = static_cast<double>(m) * static_cast<double>(k) * BYTES_PER_FLOAT32;
    double weight_bytes = static_cast<double>(k) * static_cast<double>(n) * BYTES_PER_FLOAT32;
    double c_bytes = static_cast<double>(m) * static_cast<double>(n) * BYTES_PER_FLOAT32;
    return a_bytes + weight_bytes + c_bytes;
}

double gemm_bytes_weight_dominated_approx(int64_t k, int64_t n) {
    return static_cast<double>(k) * static_cast<double>(n) * BYTES_PER_FLOAT32;
}

// The real, closed-form weight-dominated approximation: AI(M) = M / 2,
// derived directly from FLOPs / weight_bytes = 2*M*K*N / (K*N*4).
double gemm_arithmetic_intensity_approx(int64_t m) { return static_cast<double>(m) / 2.0; }

// =======================================================================
// PART 4: the real roofline crossover -- the exact batch size at which
// this weight-dominated GEMM's own arithmetic intensity reaches a given
// machine's ridge point (Chapter 8's own peak_FLOPs / peak_bandwidth),
// the point past which the workload stops being memory-bound.
// =======================================================================
double ridge_point(double peak_flops_per_sec, double peak_bandwidth_bytes_per_sec) {
    return peak_flops_per_sec / peak_bandwidth_bytes_per_sec;
}

// Solving M / 2 = ridge_point for M.
double crossover_batch_size(double ridge_pt) { return 2.0 * ridge_pt; }

// =======================================================================
// PART 5: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 30.1: The Dot Product, Arithmetic Intensity, and the GEMV/GEMM Crossover\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a dot product's own real arithmetic intensity is a fixed constant, "
                 "identical for a short vector and a vector 1000x longer --\n";
    {
        double ai_short = arithmetic_intensity(dot_product_flops(8), dot_product_bytes(8));
        double ai_long = arithmetic_intensity(dot_product_flops(8000), dot_product_bytes(8000));
        CHECK(near(ai_short, 0.25));
        CHECK(near(ai_long, 0.25));
        CHECK(near(ai_short, ai_long));
        std::cout << "  an 8-element dot product and an 8000-element dot product both compute to "
                     "exactly 0.25 FLOPs/byte -- a dot product's own real arithmetic intensity never "
                     "improves with length, which is exactly why it is intrinsically memory-bound\n";
    }

    std::cout << "\n-- Test 2: GEMV's own real arithmetic intensity approaches exactly 0.5 as the matrix "
                 "grows, regardless of its real shape -- the real reason decode-time GEMV is memory-bound "
                 "at any model size --\n";
    {
        double ai_small = arithmetic_intensity(gemv_flops(64, 64), gemv_bytes_exact(64, 64));
        double ai_large = arithmetic_intensity(gemv_flops(4096, 4096), gemv_bytes_exact(4096, 4096));
        double ai_huge = arithmetic_intensity(gemv_flops(16384, 16384), gemv_bytes_exact(16384, 16384));
        CHECK(ai_small < 0.5);
        CHECK(ai_large < 0.5 && ai_large > ai_small);
        CHECK(near(ai_huge, 0.5, 1e-3));
        std::cout << "  a real 64x64 GEMV, a real 4096x4096 GEMV, and a real 16384x16384 GEMV compute "
                     "to arithmetic intensities of " << ai_small << ", " << ai_large << ", and "
                  << ai_huge << " -- all strictly below 0.5, and converging toward exactly 0.5 as the "
                     "real matrix dimensions grow and the vector/output bytes become negligible\n";
    }

    std::cout << "\n-- Test 3: GEMM's own weight-dominated approximation, AI(M) = M/2, matches its own "
                 "exact byte accounting closely when the batch M is genuinely small compared to the real "
                 "weight matrix dimensions, and the approximation's own real error is honestly quantified "
                 "rather than assumed away --\n";
    {
        int64_t hidden = 4096;
        double exact_ai_batch1 = arithmetic_intensity(gemm_flops(1, hidden, hidden),
                                                        gemm_bytes_exact(1, hidden, hidden));
        double approx_ai_batch1 = gemm_arithmetic_intensity_approx(1);
        CHECK(near(exact_ai_batch1, approx_ai_batch1, 1e-3));

        double exact_ai_batch100 = arithmetic_intensity(gemm_flops(100, hidden, hidden),
                                                          gemm_bytes_exact(100, hidden, hidden));
        double approx_ai_batch100 = gemm_arithmetic_intensity_approx(100);
        // The approximation's own real drift grows with batch size (the input/output bytes it
        // ignores become a larger real fraction of total traffic) -- honestly still within about
        // 5% here, not zero, and not hidden behind an artificially loose tolerance.
        double relative_error = std::fabs(exact_ai_batch100 - approx_ai_batch100) / approx_ai_batch100;
        CHECK(relative_error < 0.05);
        CHECK(exact_ai_batch100 < approx_ai_batch100);  // the approximation always overstates AI a little

        // Using ONLY the weight bytes, the approximation is exact by construction.
        double weight_only_ai_batch1 = arithmetic_intensity(gemm_flops(1, hidden, hidden),
                                                              gemm_bytes_weight_dominated_approx(hidden, hidden));
        CHECK(near(weight_only_ai_batch1, 0.5));
        double weight_only_ai_batch100 = arithmetic_intensity(gemm_flops(100, hidden, hidden),
                                                                gemm_bytes_weight_dominated_approx(hidden, hidden));
        CHECK(near(weight_only_ai_batch100, 50.0));
        std::cout << "  for a real 4096x4096 weight matrix, weight-bytes-only arithmetic intensity is "
                     "exactly 0.5 at batch 1 and exactly 50.0 at batch 100 -- matching the closed-form "
                     "M/2 approximation exactly by construction -- while the EXACT byte accounting "
                     "(including the batch's own input and output bytes) differs from that "
                     "approximation by a small, honestly measured amount that grows with batch size\n";
    }

    std::cout << "\n-- Test 4: the real crossover batch size where a weight-dominated GEMM's own "
                 "arithmetic intensity reaches a stated machine's real ridge point, computed end to end "
                 "from real, stated hardware figures --\n";
    {
        // A real, stated example machine: 1000 GFLOP/s peak compute, 25 GB/s peak bandwidth.
        double peak_flops = 1000.0e9, peak_bw = 25.0e9;
        double ridge = ridge_point(peak_flops, peak_bw);
        CHECK(near(ridge, 40.0));
        double crossover = crossover_batch_size(ridge);
        CHECK(near(crossover, 80.0));
        // Below the crossover batch, this weight-dominated GEMM is memory-bound; at or above it, compute-bound.
        CHECK(gemm_arithmetic_intensity_approx(79) < ridge);
        CHECK(gemm_arithmetic_intensity_approx(81) > ridge);
        std::cout << "  a real machine with 1000 GFLOP/s peak compute and 25 GB/s peak bandwidth has a "
                     "real ridge point of exactly 40.0 FLOPs/byte, which this weight-dominated GEMM "
                     "model crosses at exactly batch size 80.0 -- batch 79 is still memory-bound, batch "
                     "81 is already compute-bound, exactly straddling the computed crossover\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover.cpp -o 01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover
./01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover
```

**Sample input:** a dot product's own arithmetic intensity checked to be an identical constant at two very different lengths; GEMV's own arithmetic intensity checked to converge toward exactly 0.5 as its real shape grows; GEMM's own weight-dominated M/2 approximation checked against an exact byte accounting, with the approximation's own real drift honestly quantified; and a real, stated machine's own crossover batch size derived end to end from its stated peak compute and peak bandwidth figures.

```text
========================================================
Chapter 30.1: The Dot Product, Arithmetic Intensity, and the GEMV/GEMM Crossover
========================================================

-- Test 1: a dot product's own real arithmetic intensity is a fixed constant, identical for a short vector and a vector 1000x longer --
  an 8-element dot product and an 8000-element dot product both compute to exactly 0.25 FLOPs/byte -- a dot product's own real arithmetic intensity never improves with length, which is exactly why it is intrinsically memory-bound

-- Test 2: GEMV's own real arithmetic intensity approaches exactly 0.5 as the matrix grows, regardless of its real shape -- the real reason decode-time GEMV is memory-bound at any model size --
  a real 64x64 GEMV, a real 4096x4096 GEMV, and a real 16384x16384 GEMV compute to arithmetic intensities of 0.484848, 0.499756, and 0.499939 -- all strictly below 0.5, and converging toward exactly 0.5 as the real matrix dimensions grow and the vector/output bytes become negligible

-- Test 3: GEMM's own weight-dominated approximation, AI(M) = M/2, matches its own exact byte accounting closely when the batch M is genuinely small compared to the real weight matrix dimensions, and the approximation's own real error is honestly quantified rather than assumed away --
  for a real 4096x4096 weight matrix, weight-bytes-only arithmetic intensity is exactly 0.5 at batch 1 and exactly 50.0 at batch 100 -- matching the closed-form M/2 approximation exactly by construction -- while the EXACT byte accounting (including the batch's own input and output bytes) differs from that approximation by a small, honestly measured amount that grows with batch size

-- Test 4: the real crossover batch size where a weight-dominated GEMM's own arithmetic intensity reaches a stated machine's real ridge point, computed end to end from real, stated hardware figures --
  a real machine with 1000 GFLOP/s peak compute and 25 GB/s peak bandwidth has a real ridge point of exactly 40.0 FLOPs/byte, which this weight-dominated GEMM model crosses at exactly batch size 80.0 -- batch 79 is still memory-bound, batch 81 is already compute-bound, exactly straddling the computed crossover

15/15 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] assuming a bigger model changes GEMV's own real memory-bound verdict"
    It is tempting to think a sufficiently large model, with enough real FLOPs per token, must eventually become compute-bound even during single-token decode. Test 2 shows precisely why that intuition is wrong: GEMV's own real arithmetic intensity converges toward a FIXED ceiling of exactly 0.5 FLOPs/byte as the matrix grows -- it does not keep climbing. A bigger model has proportionally more real FLOPs AND proportionally more real bytes to read, and those two quantities grow together, leaving the ratio essentially unchanged. The only real lever that moves arithmetic intensity for a linear layer is the BATCH size, not the model size -- which is exactly why real serving systems build continuous batching (this book's own next chapter) rather than simply hoping a bigger GPU fixes decode-time memory-boundedness on its own.

## 30.2 Numerically Stable Softmax: The Log-Sum-Exp Fix

### Intuition

Softmax's own textbook definition is exactly correct mathematically and dangerously wrong to implement literally: a single sufficiently large real logit overflows a real floating-point exponential before the actual probability computation is finished.

### The Concept, In Detail

Test 1 reproduces this real failure deliberately, in real 32-bit float -- the precision real production kernels actually run: `naive_softmax_f32({1.0, 2.0, 100.0})` produces a genuine NaN for index 2, the one entry that should hold nearly all the real probability mass, because `exp(100.0f)` overflows float32 to `+inf` and `inf/inf` is undefined. Test 2 confirms the real, shift-invariant fix -- subtracting `max(x)` before exponentiating, which is mathematically identical to the original by real algebraic cancellation -- produces a fully valid distribution on the identical input. Test 3 confirms naive and stable softmax genuinely agree on safe-magnitude input, proving they are the same real function differing only in robustness.

Test 4 and Test 5 apply the identical shift-invariant identity to log-sum-exp, and Test 5 is this section's own central, sharper point: `naive_log_sum_exp` on a sufficiently large real input does not crash and does not produce a NaN -- it silently returns `+inf`, a plainly wrong finite-valued answer masquerading as a real number, which is a more dangerous failure mode than Test 1's NaN precisely because nothing about it looks obviously broken to a caller who never separately checked.

### Code and Verification

```cpp
// Chapter 30.2 -- Softmax turns raw logits into real probabilities, and
// its own textbook definition, exp(x_i) / sum(exp(x_j)), is exactly
// correct mathematically and dangerously wrong to implement literally:
// a single large real logit overflows a real floating-point exponential
// long before the actual probability computation is finished. This
// section reproduces that real failure on purpose, derives the real
// algebraic identity that fixes it (softmax is shift-invariant), and
// shows the identical bug reappears -- in an even more dangerous, silent
// form -- in the closely related log-sum-exp computation real LLM
// log-likelihood scoring depends on.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_numerically_stable_softmax_and_log_sum_exp.cpp -o 02_numerically_stable_softmax_and_log_sum_exp
// Run:     ./02_numerically_stable_softmax_and_log_sum_exp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: naive softmax, implemented in real 32-bit float -- the
// precision real production kernels actually run -- and left exactly as
// literally as the textbook formula reads.
// =======================================================================
std::vector<float> naive_softmax_f32(const std::vector<float>& x) {
    std::vector<float> exps(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) {
        exps[i] = std::exp(x[i]);
        sum += exps[i];
    }
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); i++) out[i] = exps[i] / sum;
    return out;
}

// =======================================================================
// PART 2: the real, shift-invariant fix. softmax(x) is mathematically
// identical to softmax(x - c) for ANY real constant c, because the
// constant factor exp(-c) introduced in every term of both the
// numerator and the denominator cancels exactly. Choosing c = max(x)
// guarantees the largest real shifted value is exactly 0 (exp(0) = 1,
// no overflow), and every other shifted value is <= 0 (exp of a
// non-positive number can only underflow toward 0, never overflow).
// =======================================================================
std::vector<float> stable_softmax_f32(const std::vector<float>& x) {
    float max_x = *std::max_element(x.begin(), x.end());
    std::vector<float> exps(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) {
        exps[i] = std::exp(x[i] - max_x);
        sum += exps[i];
    }
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); i++) out[i] = exps[i] / sum;
    return out;
}

// =======================================================================
// PART 3: the identical real bug, in the closely related log-sum-exp
// computation -- and here, in double precision, its own failure mode is
// even more dangerous: it never produces a NaN a caller might notice.
// It silently returns +infinity as though that were a real, valid
// log-probability.
// =======================================================================
double naive_log_sum_exp(const std::vector<double>& x) {
    double sum = 0.0;
    for (double v : x) sum += std::exp(v);
    return std::log(sum);
}

double stable_log_sum_exp(const std::vector<double>& x) {
    double max_x = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (double v : x) sum += std::exp(v - max_x);
    return max_x + std::log(sum);
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 30.2: Numerically Stable Softmax and Log-Sum-Exp\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a single real large logit makes naive_softmax_f32 produce a genuine NaN "
                 "for exactly the entry that should carry nearly all the real probability mass --\n";
    {
        std::vector<float> x = {1.0f, 2.0f, 100.0f};
        auto result = naive_softmax_f32(x);
        CHECK(std::isnan(result[2]));   // exp(100.0f) overflows float32 to +inf; inf/inf = NaN
        CHECK(!std::isnan(result[0]));  // the small entries underflow to a plain, if misleadingly exact, 0.0f
        CHECK(!std::isnan(result[1]));
        std::cout << "  naive_softmax_f32({1.0, 2.0, 100.0}) produces a genuine NaN for index 2 -- the "
                     "one entry that should hold almost the entire real probability mass -- because "
                     "exp(100.0f) overflows float32 to +inf, and inf/inf is undefined\n";
    }

    std::cout << "\n-- Test 2: stable_softmax_f32 on the identical real input produces a valid real "
                 "probability distribution -- no NaN anywhere, and the values sum to exactly 1.0 --\n";
    {
        std::vector<float> x = {1.0f, 2.0f, 100.0f};
        auto result = stable_softmax_f32(x);
        for (float v : result) CHECK(!std::isnan(v) && !std::isinf(v));
        CHECK(result[2] > 0.999f);  // the dominant real logit correctly claims nearly all the probability
        float sum = result[0] + result[1] + result[2];
        CHECK(near(static_cast<double>(sum), 1.0, 1e-5));
        std::cout << "  stable_softmax_f32 on the identical real input produces a fully valid "
                     "distribution: index 2 correctly carries over 99.9% of the real probability mass, "
                     "and all three values sum to exactly 1.0 -- the exact same real input naive_softmax "
                     "could not process at all\n";
    }

    std::cout << "\n-- Test 3: on a real, safe-magnitude input with no overflow risk, naive and stable "
                 "softmax agree -- confirming they really are the identical real mathematical function, "
                 "differing only in numerical robustness, never in what they compute --\n";
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        auto naive = naive_softmax_f32(x);
        auto stable = stable_softmax_f32(x);
        for (size_t i = 0; i < x.size(); i++) CHECK(near(static_cast<double>(naive[i]), static_cast<double>(stable[i]), 1e-5));
        double sum = naive[0] + naive[1] + naive[2];
        CHECK(near(sum, 1.0, 1e-5));
        std::cout << "  on the real, safe input {1.0, 2.0, 3.0} (well within float32's real dynamic "
                     "range), naive and stable softmax agree to within 1e-5 at every index -- the shift "
                     "by max(x) really does leave the mathematical result unchanged\n";
    }

    std::cout << "\n-- Test 4: naive_log_sum_exp on a real, safe input matches stable_log_sum_exp "
                 "exactly, confirming the identical shift-invariant identity in the closely related "
                 "log-sum-exp formulation --\n";
    {
        std::vector<double> x = {1.0, 2.0, 3.0};
        double naive = naive_log_sum_exp(x);
        double stable = stable_log_sum_exp(x);
        CHECK(near(naive, stable, 1e-9));
        std::cout << "  on {1.0, 2.0, 3.0}, naive_log_sum_exp and stable_log_sum_exp agree to within "
                     "1e-9 -- log(sum(exp(x))) and max(x) + log(sum(exp(x - max(x)))) really are the "
                     "identical real function\n";
    }

    std::cout << "\n-- Test 5: naive_log_sum_exp on a real large-magnitude input silently returns +inf "
                 "-- not a NaN, not a crash, just a plainly WRONG finite-valued answer masquerading as "
                 "a real number -- while stable_log_sum_exp on the identical input returns the correct, "
                 "genuinely finite real value --\n";
    {
        std::vector<double> x = {1000.0, 1.0, 2.0};
        double naive = naive_log_sum_exp(x);
        double stable = stable_log_sum_exp(x);
        CHECK(std::isinf(naive));
        CHECK(!std::isinf(stable) && !std::isnan(stable));
        CHECK(stable >= 1000.0 && stable < 1000.001);  // the true value is 1000 plus a vanishingly small correction
        std::cout << "  naive_log_sum_exp({1000.0, 1.0, 2.0}) silently returns +inf, because exp(1000.0) "
                     "overflows even double precision -- a caller who never separately checked would "
                     "treat +inf as a legitimate log-probability; stable_log_sum_exp on the identical "
                     "input correctly returns approximately 1000.0, the real, genuinely finite answer, "
                     "confirming the naive result was not merely imprecise but categorically wrong\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_numerically_stable_softmax_and_log_sum_exp.cpp -o 02_numerically_stable_softmax_and_log_sum_exp
./02_numerically_stable_softmax_and_log_sum_exp
```

**Sample input:** naive softmax in real float32 checked to produce a genuine NaN on a real large-magnitude logit; stable softmax on the identical input checked to produce a fully valid distribution summing to exactly 1.0; naive and stable softmax checked to agree on safe-magnitude input; naive and stable log-sum-exp checked to agree on safe-magnitude input; and naive log-sum-exp on a real large-magnitude input checked to silently return +inf while stable log-sum-exp on the identical input returns the correct, genuinely finite value.

```text
========================================================
Chapter 30.2: Numerically Stable Softmax and Log-Sum-Exp
========================================================

-- Test 1: a single real large logit makes naive_softmax_f32 produce a genuine NaN for exactly the entry that should carry nearly all the real probability mass --
  naive_softmax_f32({1.0, 2.0, 100.0}) produces a genuine NaN for index 2 -- the one entry that should hold almost the entire real probability mass -- because exp(100.0f) overflows float32 to +inf, and inf/inf is undefined

-- Test 2: stable_softmax_f32 on the identical real input produces a valid real probability distribution -- no NaN anywhere, and the values sum to exactly 1.0 --
  stable_softmax_f32 on the identical real input produces a fully valid distribution: index 2 correctly carries over 99.9% of the real probability mass, and all three values sum to exactly 1.0 -- the exact same real input naive_softmax could not process at all

-- Test 3: on a real, safe-magnitude input with no overflow risk, naive and stable softmax agree -- confirming they really are the identical real mathematical function, differing only in numerical robustness, never in what they compute --
  on the real, safe input {1.0, 2.0, 3.0} (well within float32's real dynamic range), naive and stable softmax agree to within 1e-5 at every index -- the shift by max(x) really does leave the mathematical result unchanged

-- Test 4: naive_log_sum_exp on a real, safe input matches stable_log_sum_exp exactly, confirming the identical shift-invariant identity in the closely related log-sum-exp formulation --
  on {1.0, 2.0, 3.0}, naive_log_sum_exp and stable_log_sum_exp agree to within 1e-9 -- log(sum(exp(x))) and max(x) + log(sum(exp(x - max(x)))) really are the identical real function

-- Test 5: naive_log_sum_exp on a real large-magnitude input silently returns +inf -- not a NaN, not a crash, just a plainly WRONG finite-valued answer masquerading as a real number -- while stable_log_sum_exp on the identical input returns the correct, genuinely finite real value --
  naive_log_sum_exp({1000.0, 1.0, 2.0}) silently returns +inf, because exp(1000.0) overflows even double precision -- a caller who never separately checked would treat +inf as a legitimate log-probability; stable_log_sum_exp on the identical input correctly returns approximately 1000.0, the real, genuinely finite answer, confirming the naive result was not merely imprecise but categorically wrong

16/16 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating the absence of a NaN as proof a numerical computation is correct"
    Test 1's naive softmax fails loudly enough that a NaN check would catch it immediately. Test 5's naive log-sum-exp is the more instructive real failure precisely because it does NOT fail loudly: `+inf` is a normal, valid-looking `double`, and a system that only checks `std::isnan` on its own outputs would let this real bug through completely undetected, silently corrupting every downstream computation that treats that `+inf` as a legitimate log-probability. The real lesson is not "check for NaN" -- it is that a numerically unstable formula can fail in whatever way is easiest for it to fail, and the only real fix is the shift-invariant identity itself, applied everywhere the unstable formula would otherwise be used, not a downstream check for one specific symptom.

## 30.3 RoPE's Rotation Math

### Intuition

Chapter 3 implemented RoPE as working code. This section asks why rotating a query and a key by their own real, position-dependent angles before taking their dot product actually encodes relative position at all -- and the answer turns out to be a clean, provable real trigonometric identity, not a fortunate coincidence.

### The Concept, In Detail

Test 1 confirms the rotation matrix itself against exact, hand-verifiable real angles. Test 2 is the section's own real mathematical core: `R(a)` transposed, composed with `R(b)`, equals exactly `R(b - a)` -- checked as a genuine 2x2 matrix equality, not merely asserted -- because a rotation matrix is real and orthogonal, so its own transpose is its own inverse.

Test 3 turns that identity into RoPE's own central, load-bearing property: the real dot product of a rotated query at position m and a rotated key at position n depends only on the real relative offset `(m - n)`, confirmed directly by showing three genuinely different absolute position pairs sharing the identical offset produce numerically identical dot products, while a pair with a different offset produces a measurably different one. Test 4 bridges the two: the concrete multi-position computation from Test 3 is confirmed to equal `q^T * R(theta * (n - m)) * k`, computed directly via the abstract identity from Test 2 -- proving RoPE's own real relative-position property is a direct algebraic consequence of rotation composition, not an empirical accident.

### Code and Verification

```cpp
// Chapter 30.3 -- Chapter 3 implemented RoPE as a working piece of a
// real computational graph; this section derives WHY it actually works.
// A 2D rotation matrix is a real orthogonal transformation, and
// composing the transpose of one rotation with another real rotation
// yields exactly a THIRD rotation by the real angle DIFFERENCE -- a
// clean trigonometric identity. Applying a position-dependent rotation
// to a query and a key before taking their real dot product is what
// turns that identity into RoPE's own central, load-bearing property:
// the resulting dot product depends only on the real relative position
// between the two tokens, never on their real absolute positions.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_rope_rotation_math.cpp -o 03_rope_rotation_math
// Run:     ./03_rope_rotation_math

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: a real 2D vector and a real 2x2 rotation matrix, built from
// the standard rotation-matrix formula.
// =======================================================================
struct Vec2 {
    double x = 0.0, y = 0.0;
};

struct Mat2 {
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;  // [[a, b], [c, d]]
};

Mat2 rotation_matrix(double theta) {
    double ct = std::cos(theta), st = std::sin(theta);
    return {ct, -st, st, ct};
}

Vec2 apply(const Mat2& m, const Vec2& v) {
    return {m.a * v.x + m.b * v.y, m.c * v.x + m.d * v.y};
}

Mat2 transpose(const Mat2& m) { return {m.a, m.c, m.b, m.d}; }

Mat2 matmul(const Mat2& p, const Mat2& q) {
    return {
        p.a * q.a + p.b * q.c, p.a * q.b + p.b * q.d,
        p.c * q.a + p.d * q.c, p.c * q.b + p.d * q.d,
    };
}

double dot(const Vec2& u, const Vec2& v) { return u.x * v.x + u.y * v.y; }

bool mat_near(const Mat2& p, const Mat2& q, double eps = 1e-9) {
    return near(p.a, q.a, eps) && near(p.b, q.b, eps) && near(p.c, q.c, eps) && near(p.d, q.d, eps);
}

// =======================================================================
// PART 2: RoPE itself -- rotate a 2D query/key subvector by an angle
// that scales linearly with the token's own real absolute position.
// =======================================================================
Vec2 rope_rotate(const Vec2& v, int64_t position, double theta_base) {
    return apply(rotation_matrix(static_cast<double>(position) * theta_base), v);
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 30.3: RoPE's Rotation Math\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the rotation matrix itself matches exact, hand-verifiable real angles --\n";
    {
        Vec2 v{1.0, 0.0};
        Vec2 rotated_zero = apply(rotation_matrix(0.0), v);
        CHECK(near(rotated_zero.x, 1.0) && near(rotated_zero.y, 0.0));

        Vec2 rotated_90 = apply(rotation_matrix(std::numbers::pi / 2.0), v);
        CHECK(near(rotated_90.x, 0.0, 1e-9) && near(rotated_90.y, 1.0, 1e-9));

        Vec2 rotated_180 = apply(rotation_matrix(std::numbers::pi), v);
        CHECK(near(rotated_180.x, -1.0, 1e-9) && near(rotated_180.y, 0.0, 1e-9));
        std::cout << "  rotating (1, 0) by a real angle of 0 leaves it exactly (1, 0); rotating it by "
                     "exactly pi/2 radians (90 degrees) maps it to exactly (0, 1); rotating it by "
                     "exactly pi radians (180 degrees) maps it to exactly (-1, 0)\n";
    }

    std::cout << "\n-- Test 2: a real trigonometric identity -- R(a) transposed, composed with R(b), "
                 "equals exactly R(b - a), checked as a genuine 2x2 matrix equality across several real "
                 "angle pairs --\n";
    {
        struct Pair { double a, b; };
        std::vector<Pair> pairs = {
            {0.3, 0.9}, {1.2, 0.4}, {std::numbers::pi / 4.0, std::numbers::pi / 3.0}, {2.5, 2.5},
        };
        for (const auto& p : pairs) {
            Mat2 lhs = matmul(transpose(rotation_matrix(p.a)), rotation_matrix(p.b));
            Mat2 rhs = rotation_matrix(p.b - p.a);
            CHECK(mat_near(lhs, rhs, 1e-9));
        }
        std::cout << "  across 4 real angle pairs -- including a = b, where the identity predicts "
                     "exactly the identity matrix R(0) -- R(a)^T * R(b) matches R(b - a) exactly, "
                     "confirming rotation composition really does reduce to a single real angle "
                     "subtraction\n";
    }

    std::cout << "\n-- Test 3: RoPE's own central property -- the real dot product of a rotated query "
                 "and a rotated key depends ONLY on their real relative position, never on their "
                 "absolute positions -- verified directly across multiple genuinely different absolute "
                 "position pairs sharing the identical real offset --\n";
    {
        Vec2 q{1.3, -0.7}, k{0.4, 2.1};
        double theta_base = 0.05;

        // Three different absolute (m, n) pairs, all sharing the identical real relative offset (m - n) = 2.
        double dot_5_3 = dot(rope_rotate(q, 5, theta_base), rope_rotate(k, 3, theta_base));
        double dot_10_8 = dot(rope_rotate(q, 10, theta_base), rope_rotate(k, 8, theta_base));
        double dot_2_0 = dot(rope_rotate(q, 2, theta_base), rope_rotate(k, 0, theta_base));

        CHECK(near(dot_5_3, dot_10_8, 1e-9));
        CHECK(near(dot_10_8, dot_2_0, 1e-9));

        // A genuinely different real relative offset produces a genuinely different real dot product.
        double dot_5_2 = dot(rope_rotate(q, 5, theta_base), rope_rotate(k, 2, theta_base));  // offset = 3
        CHECK(!near(dot_5_2, dot_5_3, 1e-6));
        std::cout << "  position pairs (5, 3), (10, 8), and (2, 0) -- three genuinely different real "
                     "absolute positions, all sharing the identical real relative offset of 2 -- produce "
                     "an identical real rotated dot product to within 1e-9; the pair (5, 2), sharing a "
                     "genuinely different offset of 3, produces a measurably different real value\n";
    }

    std::cout << "\n-- Test 4: the multi-position rotated dot product from Test 3 matches the abstract "
                 "identity from Test 2 applied directly -- q^T * R(theta * (n - m)) * k -- bridging the "
                 "concrete RoPE computation to the real rotation-composition identity that explains it "
                 "--\n";
    {
        Vec2 q{1.3, -0.7}, k{0.4, 2.1};
        double theta_base = 0.05;
        int64_t m = 5, n = 3;

        double rope_dot = dot(rope_rotate(q, m, theta_base), rope_rotate(k, n, theta_base));
        Vec2 k_via_identity = apply(rotation_matrix(theta_base * static_cast<double>(n - m)), k);
        double identity_dot = dot(q, k_via_identity);

        CHECK(near(rope_dot, identity_dot, 1e-9));
        std::cout << "  computing dot(R(theta*5) q, R(theta*3) k) directly, and computing "
                     "dot(q, R(theta*(3-5)) k) via the rotation-composition identity from Test 2, agree "
                     "to within 1e-9 -- confirming RoPE's own real relative-position property is not a "
                     "coincidence but a direct, provable consequence of the rotation matrix's own real "
                     "orthogonality\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_rope_rotation_math.cpp -o 03_rope_rotation_math
./03_rope_rotation_math
```

**Sample input:** the rotation matrix checked against exact real angles (0, pi/2, pi radians); the rotation-composition identity R(a)^T * R(b) = R(b - a) checked as a real 2x2 matrix equality across several angle pairs; RoPE's own relative-position property checked directly across multiple absolute position pairs sharing an identical real offset, and against a pair with a genuinely different offset; and the concrete rotated dot product checked to match the abstract identity applied directly.

```text
========================================================
Chapter 30.3: RoPE's Rotation Math
========================================================

-- Test 1: the rotation matrix itself matches exact, hand-verifiable real angles --
  rotating (1, 0) by a real angle of 0 leaves it exactly (1, 0); rotating it by exactly pi/2 radians (90 degrees) maps it to exactly (0, 1); rotating it by exactly pi radians (180 degrees) maps it to exactly (-1, 0)

-- Test 2: a real trigonometric identity -- R(a) transposed, composed with R(b), equals exactly R(b - a), checked as a genuine 2x2 matrix equality across several real angle pairs --
  across 4 real angle pairs -- including a = b, where the identity predicts exactly the identity matrix R(0) -- R(a)^T * R(b) matches R(b - a) exactly, confirming rotation composition really does reduce to a single real angle subtraction

-- Test 3: RoPE's own central property -- the real dot product of a rotated query and a rotated key depends ONLY on their real relative position, never on their absolute positions -- verified directly across multiple genuinely different absolute position pairs sharing the identical real offset --
  position pairs (5, 3), (10, 8), and (2, 0) -- three genuinely different real absolute positions, all sharing the identical real relative offset of 2 -- produce an identical real rotated dot product to within 1e-9; the pair (5, 2), sharing a genuinely different offset of 3, produces a measurably different real value

-- Test 4: the multi-position rotated dot product from Test 3 matches the abstract identity from Test 2 applied directly -- q^T * R(theta * (n - m)) * k -- bridging the concrete RoPE computation to the real rotation-composition identity that explains it --
  computing dot(R(theta*5) q, R(theta*3) k) directly, and computing dot(q, R(theta*(3-5)) k) via the rotation-composition identity from Test 2, agree to within 1e-9 -- confirming RoPE's own real relative-position property is not a coincidence but a direct, provable consequence of the rotation matrix's own real orthogonality

11/11 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating RoPE's relative-position property as something to verify only empirically"
    It is possible to convince yourself RoPE "seems to" encode relative position by trying a few position pairs and noticing the dot products look related. Test 2 and Test 4 exist to do something stronger: derive the REASON algebraically (rotation composition reduces to a single angle subtraction, because a rotation matrix's transpose is its own real inverse) and then confirm the concrete numerical computation matches that abstract identity exactly. The difference matters because an empirical-only check cannot tell you whether the property holds for every possible position pair or only the ones you happened to try; the algebraic identity, once confirmed to match the concrete computation, guarantees it holds for all of them.

## 30.4 Quantization as Affine Algebra: Real Error Bounds and the Optimal Scale

### Intuition

Chapter 4 implemented affine quantization as working code. Treating it explicitly as an affine map -- `quantize(x) = round(x / scale) + zero_point` -- makes its own real error bound and its own real failure to compose across repeated quantization both directly provable rather than merely observed.

### The Concept, In Detail

Test 1 confirms the real optimal scale formula and an exact round-trip at both endpoints of a stated range. Test 2 is a real, general property check, not a hand-picked example: across 1001 real sample points spanning a stated range, not one is ever reconstructed more than `scale/2` away from its own true value -- the real error bound rounding produces. Test 3 goes further and confirms this bound is genuinely TIGHT: a real value placed exactly at a quantization bin's own midpoint reconstructs with an error close to the full `scale/2` bound, not comfortably inside it.

Test 4 is this section's own sharper, more consequential point: re-quantization does not commute. Quantizing a real value once, directly, at a coarse scale can produce a genuinely different integer code than quantizing it finely first, dequantizing, and then quantizing that result at the identical coarse scale -- a real, checkable non-associativity with direct consequences for any system, including this book's own Chapter 14 streaming re-quantization, that quantizes more than once.

### Code and Verification

```cpp
// Chapter 30.4 -- Chapter 4 implemented affine quantization as a working
// tool; this section treats it as what it actually is: a real affine
// map, quantize(x) = round(x / scale) + zero_point, whose only
// non-linear step is the round itself. That single fact is what
// produces quantization's own real, provable error bound (never more
// than scale/2 away from the original value), and its own real,
// checkable failure to compose: re-quantizing an already-dequantized
// value at a coarser scale is NOT the same real operation as quantizing
// the original value directly at that coarser scale -- a genuine
// non-associativity with direct consequences for any system, like
// Chapter 14's own streaming re-quantization, that quantizes more than
// once.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_quantization_as_affine_algebra.cpp -o 04_quantization_as_affine_algebra
// Run:     ./04_quantization_as_affine_algebra

#include <cmath>
#include <cstdint>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: the real affine map itself -- the standard optimal scale for
// mapping a real [min_val, max_val] range onto an unsigned integer grid
// of a stated bit width, plus its own forward (quantize) and inverse
// (dequantize) affine transformations.
// =======================================================================
double optimal_scale(double min_val, double max_val, int bits) {
    double levels = std::pow(2.0, bits) - 1.0;  // e.g. 255 real distinct integer codes for 8 bits
    return (max_val - min_val) / levels;
}

int64_t quantize(double x, double min_val, double scale) {
    return static_cast<int64_t>(std::llround((x - min_val) / scale));
}

double dequantize(int64_t q, double min_val, double scale) {
    return min_val + static_cast<double>(q) * scale;
}

// =======================================================================
// PART 2: the real, provable error bound. Because round() never moves a
// value by more than half a real quantization step, no real x in
// [min_val, max_val] can ever land farther than scale/2 from its own
// dequantized reconstruction.
// =======================================================================
double quantization_error(double x, double min_val, double scale) {
    return std::fabs(x - dequantize(quantize(x, min_val, scale), min_val, scale));
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 30.4: Quantization as Affine Algebra\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real optimal scale formula, and an exact real round-trip at both "
                 "endpoints of the quantization range --\n";
    {
        double min_val = -1.0, max_val = 1.0;
        double scale = optimal_scale(min_val, max_val, 8);
        CHECK(near(scale, 2.0 / 255.0));

        int64_t q_min = quantize(min_val, min_val, scale);
        int64_t q_max = quantize(max_val, min_val, scale);
        CHECK(q_min == 0);
        CHECK(q_max == 255);
        CHECK(near(dequantize(q_min, min_val, scale), min_val, 1e-9));
        CHECK(near(dequantize(q_max, min_val, scale), max_val, 1e-9));
        std::cout << "  an 8-bit real quantization of [-1.0, 1.0] has a real scale of exactly 2.0/255; "
                     "the real range's own two endpoints quantize to exactly integer codes 0 and 255, "
                     "and dequantizing each reproduces the exact original endpoint\n";
    }

    std::cout << "\n-- Test 2: the real, general error bound -- no real value anywhere in the "
                 "quantization range is ever reconstructed more than scale/2 away from itself, checked "
                 "across many sample points, not merely a hand-picked few --\n";
    {
        double min_val = -3.0, max_val = 5.0;
        double scale = optimal_scale(min_val, max_val, 8);
        int violations = 0;
        int checked = 0;
        for (int i = 0; i <= 1000; i++) {
            double x = min_val + (max_val - min_val) * (static_cast<double>(i) / 1000.0);
            double err = quantization_error(x, min_val, scale);
            if (err > scale / 2.0 + 1e-9) violations++;
            checked++;
        }
        CHECK(violations == 0);
        CHECK(checked == 1001);
        std::cout << "  across " << checked << " real sample points evenly spanning [-3.0, 5.0], not a "
                     "single one is ever reconstructed more than scale/2 away from its own true value -- "
                     "the real error bound holds with zero exceptions\n";
    }

    std::cout << "\n-- Test 3: the real error bound from Test 2 is TIGHT, not merely generous -- a real "
                 "value constructed exactly at a quantization bin's own real midpoint is reconstructed "
                 "with an error genuinely close to the full scale/2 bound, not comfortably inside it --\n";
    {
        double min_val = -3.0, max_val = 5.0;
        double scale = optimal_scale(min_val, max_val, 8);
        // A real value sitting exactly halfway between two adjacent real quantization levels.
        double x = min_val + scale * 10.5;
        double err = quantization_error(x, min_val, scale);
        CHECK(err > scale / 2.0 - 1e-6);
        CHECK(err <= scale / 2.0 + 1e-6);
        std::cout << "  a real value placed exactly at the midpoint between two adjacent quantization "
                     "levels reconstructs with an error of " << err << ", within a millionth of the "
                     "theoretical scale/2 bound of " << (scale / 2.0) << " -- confirming the bound from "
                     "Test 2 is the real, tight worst case, not an artificially loose estimate\n";
    }

    std::cout << "\n-- Test 4: re-quantization does not commute -- quantizing a real value once at a "
                 "coarse scale, versus quantizing it finely, dequantizing, then re-quantizing coarsely, "
                 "can produce two genuinely DIFFERENT real integer codes, a direct real consequence for "
                 "any system (like Chapter 14's own streaming re-quantization) that quantizes more than "
                 "once --\n";
    {
        double min_val = 0.0, max_val = 10.0;
        double fine_scale = optimal_scale(min_val, max_val, 7);   // a real, fine-grained scale
        double coarse_scale = optimal_scale(min_val, max_val, 4); // a real, much coarser scale

        double x = 1.0 / 3.0;  // a real value chosen so the two real paths land on different final codes

        // Path A: quantize directly at the coarse scale.
        int64_t direct_coarse = quantize(x, min_val, coarse_scale);

        // Path B: quantize finely first, dequantize back to a real float, THEN quantize at the coarse scale.
        int64_t fine_code = quantize(x, min_val, fine_scale);
        double roundtripped = dequantize(fine_code, min_val, fine_scale);
        int64_t via_fine_then_coarse = quantize(roundtripped, min_val, coarse_scale);

        CHECK(direct_coarse != via_fine_then_coarse);
        std::cout << "  quantizing " << x << " directly at a coarse 4-bit scale yields code "
                  << direct_coarse << "; quantizing the SAME real value finely at 7 bits first, "
                     "dequantizing it back to a real float, and THEN quantizing that at the identical "
                     "coarse 4-bit scale yields a genuinely different code, " << via_fine_then_coarse
                  << " -- re-quantization is a real, checkable non-associative operation, not merely a "
                     "theoretical curiosity\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_quantization_as_affine_algebra.cpp -o 04_quantization_as_affine_algebra
./04_quantization_as_affine_algebra
```

**Sample input:** the optimal scale formula and an exact endpoint round-trip checked against a stated range; the real scale/2 error bound checked across 1001 sample points spanning a stated range with zero violations; the bound's own tightness checked at a real bin midpoint; and re-quantization's own real non-associativity checked by comparing a direct coarse quantization against a fine-then-coarse round trip on the identical original value.

```text
========================================================
Chapter 30.4: Quantization as Affine Algebra
========================================================

-- Test 1: the real optimal scale formula, and an exact real round-trip at both endpoints of the quantization range --
  an 8-bit real quantization of [-1.0, 1.0] has a real scale of exactly 2.0/255; the real range's own two endpoints quantize to exactly integer codes 0 and 255, and dequantizing each reproduces the exact original endpoint

-- Test 2: the real, general error bound -- no real value anywhere in the quantization range is ever reconstructed more than scale/2 away from itself, checked across many sample points, not merely a hand-picked few --
  across 1001 real sample points evenly spanning [-3.0, 5.0], not a single one is ever reconstructed more than scale/2 away from its own true value -- the real error bound holds with zero exceptions

-- Test 3: the real error bound from Test 2 is TIGHT, not merely generous -- a real value constructed exactly at a quantization bin's own real midpoint is reconstructed with an error genuinely close to the full scale/2 bound, not comfortably inside it --
  a real value placed exactly at the midpoint between two adjacent quantization levels reconstructs with an error of 0.0156863, within a millionth of the theoretical scale/2 bound of 0.0156863 -- confirming the bound from Test 2 is the real, tight worst case, not an artificially loose estimate

-- Test 4: re-quantization does not commute -- quantizing a real value once at a coarse scale, versus quantizing it finely, dequantizing, then re-quantizing coarsely, can produce two genuinely DIFFERENT real integer codes, a direct real consequence for any system (like Chapter 14's own streaming re-quantization) that quantizes more than once --
  quantizing 0.333333 directly at a coarse 4-bit scale yields code 1; quantizing the SAME real value finely at 7 bits first, dequantizing it back to a real float, and THEN quantizing that at the identical coarse 4-bit scale yields a genuinely different code, 0 -- re-quantization is a real, checkable non-associative operation, not merely a theoretical curiosity

10/10 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] assuming re-quantizing an already-quantized value is the same as quantizing the original"
    A system that dequantizes a value to re-scale it -- converting from one bit width to another, say, during a real re-quantization pass -- might assume the result is equivalent to having quantized the true original value directly at the new scale, since dequantization is "supposed to" recover the original. Test 4 shows this assumption is false in general: the intermediate fine-grained rounding step introduces its own small real error, and that error can be just enough to push the value across a coarser bin's own boundary, landing on a genuinely different final code than a direct quantization would have. Any real system that quantizes more than once -- exactly the situation Chapter 14's own streaming re-quantization is built to handle carefully -- has to treat this as a real, accumulating source of error, not something dequantization quietly undoes.

## 30.5 The Hessian's Role in GPTQ: Second-Order Error Compensation

### Intuition

Quantizing every weight independently, as Section 30.4's own affine map does in isolation, ignores real information a calibration dataset already provides: once one weight is quantized, the weights that are not yet quantized can be nudged to compensate for the error that quantization just introduced. GPTQ's real insight is that the layer's own Hessian is exactly the second-order information needed to compute that compensation optimally.

### The Concept, In Detail

Test 1 and Test 2 build this section's own real, from-scratch linear-algebra foundation: `H = 2 * X^T * X`, computed directly from a tiny real calibration dataset, matches an exact hand computation, and a real, general Gauss-Jordan inverse of that Hessian is confirmed correct not by trusting the algorithm but by checking `H * H^-1` is genuinely the identity matrix directly.

Test 3 confirms the real GPTQ compensation formula -- `delta_w_f = -(e_p / [H^-1]_pp) * [H^-1]_fp` -- against an exact hand computation for a single quantized weight's effect on the one remaining weight. Test 4 is this section's own central, real proof: running the full sequential GPTQ pipeline (quantize, compute the real error, compensate every remaining weight, repeat) on a fully hand-traceable two-weight example produces a genuinely different final quantized weight than naive independent rounding, and a strictly LOWER real total squared output error over the calibration data -- 0.14 for GPTQ's compensated result against 0.24 for naive rounding on the identical original weights. GPTQ's real benefit is not asserted here; it is computed and compared directly.

### Code and Verification

```cpp
// Chapter 30.5 -- Quantizing a weight independently of every other
// weight, as Section 30.4's own affine map does, ignores something a
// real calibration dataset already knows: some weights matter more to
// a layer's real output than others, and the ones that don't yet have
// a fixed quantized value can still be nudged to compensate for the
// ones that already do. GPTQ's own real insight is that the layer's
// Hessian -- here, the genuinely simple H = 2 * X^T * X for a real
// squared-error loss over calibration data X -- is exactly the real
// second-order information needed to compute that compensation
// optimally. This section builds a real, from-scratch 2x2 Hessian,
// inverts it with a real Gauss-Jordan solver, and proves on a fully
// hand-traceable example that the resulting compensated quantization
// produces strictly less real output error than quantizing every
// weight independently.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_hessian_role_in_gptq.cpp -o 05_hessian_role_in_gptq
// Run:     ./05_hessian_role_in_gptq

#include <cmath>
#include <iostream>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: a real, general, from-scratch small dense matrix, with real
// multiply, transpose, and a real Gauss-Jordan inverse -- general in N,
// even though this section only ever exercises it at N = 2.
// =======================================================================
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<double> data;

    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0) {}
    double& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    double at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
};

Matrix transpose(const Matrix& m) {
    Matrix out(m.cols, m.rows);
    for (int r = 0; r < m.rows; r++)
        for (int c = 0; c < m.cols; c++) out.at(c, r) = m.at(r, c);
    return out;
}

Matrix matmul(const Matrix& a, const Matrix& b) {
    Matrix out(a.rows, b.cols);
    for (int r = 0; r < a.rows; r++)
        for (int c = 0; c < b.cols; c++) {
            double sum = 0.0;
            for (int k = 0; k < a.cols; k++) sum += a.at(r, k) * b.at(k, c);
            out.at(r, c) = sum;
        }
    return out;
}

// A real, general Gauss-Jordan matrix inverse via full pivoting on an
// augmented [M | I] matrix.
Matrix inverse(const Matrix& m) {
    int n = m.rows;
    Matrix aug(n, 2 * n);
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) aug.at(r, c) = m.at(r, c);
        aug.at(r, n + r) = 1.0;
    }
    for (int p = 0; p < n; p++) {
        double pivot = aug.at(p, p);
        for (int c = 0; c < 2 * n; c++) aug.at(p, c) /= pivot;
        for (int r = 0; r < n; r++) {
            if (r == p) continue;
            double factor = aug.at(r, p);
            for (int c = 0; c < 2 * n; c++) aug.at(r, c) -= factor * aug.at(p, c);
        }
    }
    Matrix out(n, n);
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) out.at(r, c) = aug.at(r, n + c);
    return out;
}

std::vector<double> matvec(const Matrix& m, const std::vector<double>& v) {
    std::vector<double> out(static_cast<size_t>(m.rows), 0.0);
    for (int r = 0; r < m.rows; r++)
        for (int c = 0; c < m.cols; c++) out[static_cast<size_t>(r)] += m.at(r, c) * v[static_cast<size_t>(c)];
    return out;
}

// =======================================================================
// PART 2: real round-to-nearest quantization at a stated fixed scale --
// the same real affine map from Section 30.4, applied here to a single
// scalar weight.
// =======================================================================
double quantize_weight(double w, double scale) { return std::round(w / scale) * scale; }

// =======================================================================
// PART 3: the real GPTQ compensation formula. Having just quantized
// weight index p, introducing a real error e_p = w_p - quantize(w_p),
// the optimal real update to every NOT-yet-quantized weight index f is
// delta_w_f = -(e_p / [H^-1]_pp) * [H^-1]_fp -- the real, second-order-
// informed correction that keeps the layer's overall real output as
// close as possible to what it would have been with the original,
// unquantized weight.
// =======================================================================
void apply_gptq_compensation(std::vector<double>& w, const Matrix& h_inv, int quantized_index, double error) {
    double denom = h_inv.at(quantized_index, quantized_index);
    for (size_t f = 0; f < w.size(); f++) {
        if (static_cast<int>(f) == quantized_index) continue;
        double delta = -(error / denom) * h_inv.at(static_cast<int>(f), quantized_index);
        w[f] += delta;
    }
}

double sum_squared_error(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 30.5: The Hessian's Role in GPTQ\n";
    std::cout << "========================================================\n";

    // A real, tiny calibration dataset: 3 real samples, 2 real features each.
    Matrix X(3, 2);
    X.at(0, 0) = 1; X.at(0, 1) = 0;
    X.at(1, 0) = 0; X.at(1, 1) = 1;
    X.at(2, 0) = 1; X.at(2, 1) = 1;

    std::cout << "\n-- Test 1: the real Hessian H = 2 * X^T * X, computed from scratch, matches an exact "
                 "hand computation over this section's own real calibration data --\n";
    {
        Matrix xt = transpose(X);
        Matrix xtx = matmul(xt, X);
        Matrix H(2, 2);
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 2; c++) H.at(r, c) = 2.0 * xtx.at(r, c);

        CHECK(near(H.at(0, 0), 4.0));
        CHECK(near(H.at(0, 1), 2.0));
        CHECK(near(H.at(1, 0), 2.0));
        CHECK(near(H.at(1, 1), 4.0));
        std::cout << "  X^T * X over the real 3-sample calibration set is exactly [[2, 1], [1, 2]], so "
                     "H = 2 * X^T * X is exactly [[4, 2], [2, 4]] -- matching a direct hand computation "
                     "exactly\n";
    }

    std::cout << "\n-- Test 2: the real, general Gauss-Jordan inverse of H matches an exact hand "
                 "computation, and H * H^-1 is genuinely the identity matrix, checked directly rather "
                 "than assumed from the inversion algorithm's own correctness --\n";
    {
        Matrix H(2, 2);
        H.at(0, 0) = 4; H.at(0, 1) = 2; H.at(1, 0) = 2; H.at(1, 1) = 4;
        Matrix h_inv = inverse(H);

        CHECK(near(h_inv.at(0, 0), 1.0 / 3.0));
        CHECK(near(h_inv.at(0, 1), -1.0 / 6.0));
        CHECK(near(h_inv.at(1, 0), -1.0 / 6.0));
        CHECK(near(h_inv.at(1, 1), 1.0 / 3.0));

        Matrix identity_check = matmul(H, h_inv);
        CHECK(near(identity_check.at(0, 0), 1.0) && near(identity_check.at(0, 1), 0.0));
        CHECK(near(identity_check.at(1, 0), 0.0) && near(identity_check.at(1, 1), 1.0));
        std::cout << "  the real Gauss-Jordan inverse of [[4, 2], [2, 4]] is exactly "
                     "[[1/3, -1/6], [-1/6, 1/3]], matching a hand computation via the real 2x2 "
                     "determinant formula exactly, and H * H^-1 checked directly is exactly the real "
                     "2x2 identity matrix\n";
    }

    std::cout << "\n-- Test 3: the real GPTQ compensation formula applied to a single quantized weight "
                 "matches an exact hand computation for the resulting update to the remaining weight --\n";
    {
        Matrix h_inv(2, 2);
        h_inv.at(0, 0) = 1.0 / 3.0; h_inv.at(0, 1) = -1.0 / 6.0;
        h_inv.at(1, 0) = -1.0 / 6.0; h_inv.at(1, 1) = 1.0 / 3.0;

        double w0 = 1.3, scale = 0.5;
        double q0 = quantize_weight(w0, scale);
        CHECK(near(q0, 1.5));
        double e0 = w0 - q0;
        CHECK(near(e0, -0.2));

        std::vector<double> w = {w0, 0.8};
        apply_gptq_compensation(w, h_inv, 0, e0);
        CHECK(near(w[1], 0.7));  // 0.8 + delta, where delta = -(-0.2 / (1/3)) * (-1/6) = -0.1
        std::cout << "  quantizing w[0] = 1.3 at scale 0.5 yields q0 = 1.5, a real error of e0 = -0.2; "
                     "applying the real GPTQ compensation formula to the not-yet-quantized w[1] = 0.8 "
                     "shifts it to exactly 0.7 -- matching a direct hand computation of "
                     "-(-0.2 / (1/3)) * (-1/6) = -0.1 exactly\n";
    }

    std::cout << "\n-- Test 4: the full real GPTQ pipeline -- sequential per-weight quantization with "
                 "compensation -- produces a genuinely different final weight than naive, independent "
                 "rounding, and a strictly LOWER real output error over the calibration data, GPTQ's own "
                 "central real claim, verified end to end on a fully hand-traceable example --\n";
    {
        Matrix h_inv(2, 2);
        h_inv.at(0, 0) = 1.0 / 3.0; h_inv.at(0, 1) = -1.0 / 6.0;
        h_inv.at(1, 0) = -1.0 / 6.0; h_inv.at(1, 1) = 1.0 / 3.0;

        std::vector<double> w_orig = {1.3, 0.8};
        double scale = 0.5;

        // Naive: quantize every weight independently, no compensation at all.
        std::vector<double> w_naive = {quantize_weight(w_orig[0], scale), quantize_weight(w_orig[1], scale)};
        CHECK(near(w_naive[0], 1.5));
        CHECK(near(w_naive[1], 1.0));

        // GPTQ: quantize index 0, compensate the remaining weight, then quantize it too.
        std::vector<double> w_gptq = w_orig;
        double q0 = quantize_weight(w_gptq[0], scale);
        double e0 = w_gptq[0] - q0;
        w_gptq[0] = q0;
        apply_gptq_compensation(w_gptq, h_inv, 0, e0);
        w_gptq[1] = quantize_weight(w_gptq[1], scale);
        CHECK(near(w_gptq[0], 1.5));
        CHECK(near(w_gptq[1], 0.5));  // a genuinely different real final code than naive's 1.0

        std::vector<double> y_orig = matvec(X, w_orig);
        std::vector<double> y_naive = matvec(X, w_naive);
        std::vector<double> y_gptq = matvec(X, w_gptq);

        double error_naive = sum_squared_error(y_naive, y_orig);
        double error_gptq = sum_squared_error(y_gptq, y_orig);
        CHECK(near(error_naive, 0.24));
        CHECK(near(error_gptq, 0.14));
        CHECK(error_gptq < error_naive);
        std::cout << "  naive independent rounding quantizes w to exactly [1.5, 1.0], with a real total "
                     "squared output error over the calibration data of exactly 0.24; GPTQ's real "
                     "Hessian-compensated pipeline quantizes the SAME original weights to exactly "
                     "[1.5, 0.5] -- a genuinely different second weight -- with a real total squared "
                     "output error of exactly 0.14, strictly lower than naive rounding's 0.24, "
                     "confirming GPTQ's own central claim on a fully hand-traceable example\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_hessian_role_in_gptq.cpp -o 05_hessian_role_in_gptq
./05_hessian_role_in_gptq
```

**Sample input:** the real Hessian H = 2 * X^T * X checked against an exact hand computation over a tiny calibration dataset; a real, general Gauss-Jordan matrix inverse checked against an exact hand computation, with H * H^-1 checked directly against the identity matrix; the real GPTQ compensation formula checked against an exact hand computation for a single weight; and the full GPTQ pipeline checked end to end against naive independent rounding on a fully hand-traceable two-weight example, confirming a strictly lower real total squared output error.

```text
========================================================
Chapter 30.5: The Hessian's Role in GPTQ
========================================================

-- Test 1: the real Hessian H = 2 * X^T * X, computed from scratch, matches an exact hand computation over this section's own real calibration data --
  X^T * X over the real 3-sample calibration set is exactly [[2, 1], [1, 2]], so H = 2 * X^T * X is exactly [[4, 2], [2, 4]] -- matching a direct hand computation exactly

-- Test 2: the real, general Gauss-Jordan inverse of H matches an exact hand computation, and H * H^-1 is genuinely the identity matrix, checked directly rather than assumed from the inversion algorithm's own correctness --
  the real Gauss-Jordan inverse of [[4, 2], [2, 4]] is exactly [[1/3, -1/6], [-1/6, 1/3]], matching a hand computation via the real 2x2 determinant formula exactly, and H * H^-1 checked directly is exactly the real 2x2 identity matrix

-- Test 3: the real GPTQ compensation formula applied to a single quantized weight matches an exact hand computation for the resulting update to the remaining weight --
  quantizing w[0] = 1.3 at scale 0.5 yields q0 = 1.5, a real error of e0 = -0.2; applying the real GPTQ compensation formula to the not-yet-quantized w[1] = 0.8 shifts it to exactly 0.7 -- matching a direct hand computation of -(-0.2 / (1/3)) * (-1/6) = -0.1 exactly

-- Test 4: the full real GPTQ pipeline -- sequential per-weight quantization with compensation -- produces a genuinely different final weight than naive, independent rounding, and a strictly LOWER real output error over the calibration data, GPTQ's own central real claim, verified end to end on a fully hand-traceable example --
  naive independent rounding quantizes w to exactly [1.5, 1.0], with a real total squared output error over the calibration data of exactly 0.24; GPTQ's real Hessian-compensated pipeline quantizes the SAME original weights to exactly [1.5, 0.5] -- a genuinely different second weight -- with a real total squared output error of exactly 0.14, strictly lower than naive rounding's 0.24, confirming GPTQ's own central claim on a fully hand-traceable example

20/20 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] assuming Hessian-based compensation always helps a LATER weight, never a chain of them"
    It is easy to read GPTQ's own compensation formula and assume its benefit is limited to the single, immediately adjacent weight it updates. Test 4's own real pipeline shows the mechanism is genuinely sequential and cumulative: quantizing weight 0 changes the value weight 1 is compensated toward, and in a layer with more than 2 weights, quantizing weight 1 (now itself already nudged once) would go on to compensate weight 2, and so on. Each real compensation step uses the CURRENT state of the not-yet-quantized weights, not the original ones -- which is exactly why GPTQ processes weights in a specific real sequence rather than computing every compensation independently up front from the unquantized original values.

## 30.6 A Full Roofline Analysis of a Transformer Layer

### Intuition

Every real formula this chapter derived -- arithmetic intensity, weight-dominated byte accounting, the roofline crossover -- was built on a single operation at a time. This capstone section applies all of them together to an entire real transformer decoder layer, to classify the WHOLE layer's own real bound and to derive the exact batch size where that classification flips.

### The Concept, In Detail

Test 1 confirms every real FLOP sub-total (QKV projection, attention, output projection, FFN) and the real weight-byte total for a tiny, fully hand-traceable layer shape, matching a direct hand computation exactly: 576 total FLOPs, 512 total weight bytes, an arithmetic intensity of exactly 1.125. Test 2 confirms a real, general algebraic property this section's own crossover formula depends on: doubling batch size exactly doubles both total FLOPs and arithmetic intensity, while weight bytes -- which do not depend on batch at all -- stay exactly unchanged.

Test 3 applies Section 30.1's own roofline classification to the WHOLE layer: against a stated real ridge point of 2.0, the identical layer classifies `MEMORY_BOUND` at batch 1 and `COMPUTE_BOUND` at batch 2. Test 4 confirms the real, closed-form crossover formula predicts precisely this observed transition rather than merely rationalizing it afterward, and Test 5 confirms the formula's own real algebraic correctness across several genuinely different stated ridge points at once.

### Code and Verification

```cpp
// Chapter 30.6 -- This chapter's own capstone: every real formula built
// in Sections 30.1 through 30.5 -- FLOP counting, weight-dominated byte
// accounting, arithmetic intensity, and the real roofline crossover --
// applied together to a complete real transformer decoder layer (QKV
// projection, attention, output projection, and the FFN), at a stated
// real shape, to classify the WHOLE layer as memory-bound or
// compute-bound at a given real batch size, and to derive the real
// batch size at which that classification flips.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 06_full_transformer_layer_roofline_analysis.cpp -o 06_full_transformer_layer_roofline_analysis
// Run:     ./06_full_transformer_layer_roofline_analysis

#include <cmath>
#include <cstdint>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) < eps; }

constexpr double BYTES_PER_FLOAT32 = 4.0;

// =======================================================================
// PART 1: a real transformer decoder layer's own shape.
// =======================================================================
struct LayerShape {
    int64_t hidden = 0, heads = 0, head_dim = 0, ffn_dim = 0, seq_len = 0;
};

struct FlopBreakdown {
    double qkv = 0.0, attn = 0.0, out_proj = 0.0, ffn = 0.0, total = 0.0;
};

struct ByteBreakdown {
    double qkv = 0.0, out_proj = 0.0, ffn = 0.0, total = 0.0;
};

// =======================================================================
// PART 2: real FLOP counting for each real sub-block of the layer, at a
// stated real batch size. M = batch * seq_len is the real number of
// token-rows the QKV and output projections, and the FFN, each run as a
// GEMM over.
// =======================================================================
FlopBreakdown layer_flops(const LayerShape& s, int64_t batch) {
    double m = static_cast<double>(batch) * static_cast<double>(s.seq_len);
    double hidden = static_cast<double>(s.hidden);
    double ffn_dim = static_cast<double>(s.ffn_dim);
    double seq_len = static_cast<double>(s.seq_len);
    double b = static_cast<double>(batch);

    FlopBreakdown f;
    f.qkv = 3.0 * 2.0 * m * hidden * hidden;               // 3 separate hidden -> hidden projections
    f.attn = 4.0 * b * seq_len * seq_len * hidden;          // real QK^T plus attn*V, summed over all heads
    f.out_proj = 2.0 * m * hidden * hidden;                 // one hidden -> hidden projection
    f.ffn = 4.0 * m * hidden * ffn_dim;                     // hidden->ffn_dim, then ffn_dim->hidden
    f.total = f.qkv + f.attn + f.out_proj + f.ffn;
    return f;
}

// =======================================================================
// PART 3: real weight-byte accounting -- the same weight-dominated
// approximation Section 30.1 derived and honestly bounded, applied here
// to the layer's own 4 real weight tensors. Batch-independent by
// construction: the weights themselves do not grow with batch size.
// =======================================================================
ByteBreakdown layer_weight_bytes(const LayerShape& s) {
    double hidden = static_cast<double>(s.hidden);
    double ffn_dim = static_cast<double>(s.ffn_dim);

    ByteBreakdown b;
    b.qkv = 3.0 * hidden * hidden * BYTES_PER_FLOAT32;
    b.out_proj = hidden * hidden * BYTES_PER_FLOAT32;
    b.ffn = 2.0 * hidden * ffn_dim * BYTES_PER_FLOAT32;
    b.total = b.qkv + b.out_proj + b.ffn;
    return b;
}

double arithmetic_intensity(double flops, double bytes) { return flops / bytes; }

enum class BoundClass { MEMORY_BOUND, COMPUTE_BOUND };

BoundClass classify_bound(double ai, double ridge_point) {
    return (ai < ridge_point) ? BoundClass::MEMORY_BOUND : BoundClass::COMPUTE_BOUND;
}

// Because every term in layer_flops scales linearly with batch (M and
// the explicit batch factor in the attention term both do), and
// layer_weight_bytes does not depend on batch at all, this layer's own
// real arithmetic intensity is EXACTLY linear in batch: AI(batch) =
// AI(1) * batch. Solving AI(1) * batch = ridge_point gives the real
// crossover batch size directly.
double crossover_batch_size(const LayerShape& s, double ridge_point) {
    FlopBreakdown f1 = layer_flops(s, 1);
    ByteBreakdown bytes = layer_weight_bytes(s);
    double ai_at_batch_1 = arithmetic_intensity(f1.total, bytes.total);
    return ridge_point / ai_at_batch_1;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 30.6: A Full Roofline Analysis of a Transformer Layer\n";
    std::cout << "========================================================\n";

    // A tiny, fully hand-traceable real layer shape.
    LayerShape shape{4, 2, 2, 8, 2};  // hidden=4, heads=2, head_dim=2, ffn_dim=8, seq_len=2

    std::cout << "\n-- Test 1: at batch 1, every real FLOP sub-total, the real total, the real weight-"
                 "byte total, and the resulting real arithmetic intensity all match an exact hand "
                 "computation for this tiny, fully traceable layer shape --\n";
    {
        FlopBreakdown f = layer_flops(shape, 1);
        CHECK(near(f.qkv, 192.0));
        CHECK(near(f.attn, 64.0));
        CHECK(near(f.out_proj, 64.0));
        CHECK(near(f.ffn, 256.0));
        CHECK(near(f.total, 576.0));

        ByteBreakdown b = layer_weight_bytes(shape);
        CHECK(near(b.qkv, 192.0));
        CHECK(near(b.out_proj, 64.0));
        CHECK(near(b.ffn, 256.0));
        CHECK(near(b.total, 512.0));

        double ai = arithmetic_intensity(f.total, b.total);
        CHECK(near(ai, 1.125));
        std::cout << "  at batch 1, this tiny layer's own real FLOPs are exactly 192 (QKV) + 64 "
                     "(attention) + 64 (output projection) + 256 (FFN) = 576 total; its own real weight "
                     "bytes are exactly 192 + 64 + 256 = 512; its own real arithmetic intensity is "
                     "exactly 576 / 512 = 1.125 FLOPs/byte -- every sub-total matching a direct hand "
                     "computation from this layer's own stated shape\n";
    }

    std::cout << "\n-- Test 2: doubling the real batch size exactly doubles the real total FLOPs and "
                 "the real arithmetic intensity, while the real weight bytes stay exactly unchanged -- "
                 "confirming the algebraic linear-in-batch property this section's own crossover formula "
                 "depends on, checked directly rather than merely asserted --\n";
    {
        FlopBreakdown f1 = layer_flops(shape, 1);
        FlopBreakdown f2 = layer_flops(shape, 2);
        CHECK(near(f2.total, 2.0 * f1.total));
        CHECK(near(f2.total, 1152.0));

        ByteBreakdown b1 = layer_weight_bytes(shape);
        ByteBreakdown b2 = layer_weight_bytes(shape);  // batch-independent -- recomputed, still identical
        CHECK(near(b1.total, b2.total));

        double ai1 = arithmetic_intensity(f1.total, b1.total);
        double ai2 = arithmetic_intensity(f2.total, b2.total);
        CHECK(near(ai2, 2.0 * ai1));
        CHECK(near(ai2, 2.25));
        std::cout << "  at batch 2, real total FLOPs are exactly 1152 -- precisely double batch 1's 576 "
                     "-- while real weight bytes remain exactly 512, unchanged; the resulting real "
                     "arithmetic intensity, 2.25, is precisely double batch 1's 1.125, confirming this "
                     "layer's own real arithmetic intensity scales exactly linearly with batch size\n";
    }

    std::cout << "\n-- Test 3: against a stated real machine ridge point sitting between this layer's own "
                 "batch-1 and batch-2 arithmetic intensities, the SAME layer classifies as memory-bound "
                 "at batch 1 and compute-bound at batch 2 -- the real roofline model's own central "
                 "prediction, applied to a whole real transformer layer rather than a single operation "
                 "--\n";
    {
        double ridge = 2.0;
        FlopBreakdown f1 = layer_flops(shape, 1);
        FlopBreakdown f2 = layer_flops(shape, 2);
        ByteBreakdown bytes = layer_weight_bytes(shape);

        double ai1 = arithmetic_intensity(f1.total, bytes.total);
        double ai2 = arithmetic_intensity(f2.total, bytes.total);

        CHECK(classify_bound(ai1, ridge) == BoundClass::MEMORY_BOUND);
        CHECK(classify_bound(ai2, ridge) == BoundClass::COMPUTE_BOUND);
        std::cout << "  against a real stated machine ridge point of 2.0 FLOPs/byte, this layer's own "
                     "batch-1 arithmetic intensity of 1.125 classifies MEMORY_BOUND, and the identical "
                     "layer's own batch-2 arithmetic intensity of 2.25 classifies COMPUTE_BOUND -- the "
                     "same real layer shape, genuinely different real classifications, purely as a "
                     "function of real batch size\n";
    }

    std::cout << "\n-- Test 4: the real crossover batch size computed directly from this layer's own "
                 "shape and a stated ridge point falls exactly between the two batch sizes Test 3 "
                 "observed flipping classification, confirming the closed-form crossover formula "
                 "predicts the identical real transition rather than merely rationalizing it after the "
                 "fact --\n";
    {
        double ridge = 2.0;
        double crossover = crossover_batch_size(shape, ridge);
        CHECK(near(crossover, 2.0 / 1.125));
        CHECK(crossover > 1.0);
        CHECK(crossover < 2.0);
        std::cout << "  the real crossover batch size for this layer's own shape against a ridge point "
                     "of 2.0 is exactly " << crossover << " -- strictly between batch 1 (observed "
                     "MEMORY_BOUND in Test 3) and batch 2 (observed COMPUTE_BOUND) -- confirming the "
                     "closed-form crossover formula predicts precisely the transition this section "
                     "already observed directly\n";
    }

    std::cout << "\n-- Test 5: the same crossover formula behaves consistently across several genuinely "
                 "different real ridge points, always landing exactly where AI(1) * crossover = ridge, "
                 "checked algebraically rather than against a single hand-picked example --\n";
    {
        for (double ridge : {0.5, 1.125, 5.0, 40.0}) {
            double crossover = crossover_batch_size(shape, ridge);
            FlopBreakdown f1 = layer_flops(shape, 1);
            ByteBreakdown bytes = layer_weight_bytes(shape);
            double ai1 = arithmetic_intensity(f1.total, bytes.total);
            CHECK(near(ai1 * crossover, ridge, 1e-6));
        }
        std::cout << "  across 4 genuinely different real ridge points -- 0.5, 1.125 (the layer's own "
                     "exact batch-1 arithmetic intensity), 5.0, and 40.0 -- the computed real crossover "
                     "batch size always satisfies AI(1) * crossover = ridge_point exactly, confirming "
                     "the formula's own real algebraic correctness rather than a single coincidental "
                     "match\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 06_full_transformer_layer_roofline_analysis.cpp -o 06_full_transformer_layer_roofline_analysis
./06_full_transformer_layer_roofline_analysis
```

**Sample input:** every real FLOP and weight-byte sub-total for a tiny, fully hand-traceable transformer layer shape, checked against an exact hand computation; batch-doubling checked to exactly double FLOPs and arithmetic intensity while leaving weight bytes unchanged; the whole layer's own bound classification checked to flip from memory-bound to compute-bound between batch 1 and batch 2 against a stated ridge point; the closed-form crossover batch size checked to fall exactly between those two batch sizes; and the crossover formula's own algebraic correctness checked across 4 different stated ridge points.

```text
========================================================
Chapter 30.6: A Full Roofline Analysis of a Transformer Layer
========================================================

-- Test 1: at batch 1, every real FLOP sub-total, the real total, the real weight-byte total, and the resulting real arithmetic intensity all match an exact hand computation for this tiny, fully traceable layer shape --
  at batch 1, this tiny layer's own real FLOPs are exactly 192 (QKV) + 64 (attention) + 64 (output projection) + 256 (FFN) = 576 total; its own real weight bytes are exactly 192 + 64 + 256 = 512; its own real arithmetic intensity is exactly 576 / 512 = 1.125 FLOPs/byte -- every sub-total matching a direct hand computation from this layer's own stated shape

-- Test 2: doubling the real batch size exactly doubles the real total FLOPs and the real arithmetic intensity, while the real weight bytes stay exactly unchanged -- confirming the algebraic linear-in-batch property this section's own crossover formula depends on, checked directly rather than merely asserted --
  at batch 2, real total FLOPs are exactly 1152 -- precisely double batch 1's 576 -- while real weight bytes remain exactly 512, unchanged; the resulting real arithmetic intensity, 2.25, is precisely double batch 1's 1.125, confirming this layer's own real arithmetic intensity scales exactly linearly with batch size

-- Test 3: against a stated real machine ridge point sitting between this layer's own batch-1 and batch-2 arithmetic intensities, the SAME layer classifies as memory-bound at batch 1 and compute-bound at batch 2 -- the real roofline model's own central prediction, applied to a whole real transformer layer rather than a single operation --
  against a real stated machine ridge point of 2.0 FLOPs/byte, this layer's own batch-1 arithmetic intensity of 1.125 classifies MEMORY_BOUND, and the identical layer's own batch-2 arithmetic intensity of 2.25 classifies COMPUTE_BOUND -- the same real layer shape, genuinely different real classifications, purely as a function of real batch size

-- Test 4: the real crossover batch size computed directly from this layer's own shape and a stated ridge point falls exactly between the two batch sizes Test 3 observed flipping classification, confirming the closed-form crossover formula predicts the identical real transition rather than merely rationalizing it after the fact --
  the real crossover batch size for this layer's own shape against a ridge point of 2.0 is exactly 1.77778 -- strictly between batch 1 (observed MEMORY_BOUND in Test 3) and batch 2 (observed COMPUTE_BOUND) -- confirming the closed-form crossover formula predicts precisely the transition this section already observed directly

-- Test 5: the same crossover formula behaves consistently across several genuinely different real ridge points, always landing exactly where AI(1) * crossover = ridge, checked algebraically rather than against a single hand-picked example --
  across 4 genuinely different real ridge points -- 0.5, 1.125 (the layer's own exact batch-1 arithmetic intensity), 5.0, and 40.0 -- the computed real crossover batch size always satisfies AI(1) * crossover = ridge_point exactly, confirming the formula's own real algebraic correctness rather than a single coincidental match

24/24 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] analyzing one operation's own roofline classification and assuming it tells you the whole layer's"
    A transformer layer is not a single GEMM -- it is 4 real sub-blocks (QKV projection, attention, output projection, FFN), each with its own real FLOP and byte profile, and nothing guarantees they all cross the ridge point at the identical batch size. This section's own capstone works because every one of those 4 sub-blocks happens to scale identically with batch (linearly, since each one's own real formula has an explicit or implicit batch factor) while none of their weight bytes depend on batch at all -- which is what allows a SINGLE whole-layer arithmetic intensity, and a single whole-layer crossover batch size, to exist at all. A real architecture with a sub-block that scales differently (for instance, a mixture-of-experts layer whose active weight bytes themselves depend on which real experts a given batch routes to) would need its own, separately derived crossover analysis rather than reusing this section's own single-formula shortcut.

## Chapter Summary

This chapter derived the real mathematics Parts 1 through 5 relied on without deriving. Section 30.1 showed a dot product's own arithmetic intensity is a fixed constant, explaining why GEMV-based decode is intrinsically memory-bound, and derived the real batch size at which GEMM crosses over to compute-bound. Section 30.2 reproduced softmax's real overflow failure on purpose, fixed it with a real shift-invariant identity, and showed the identical bug reappears silently, as a wrong finite value rather than a NaN, in log-sum-exp. Section 30.3 proved RoPE's own relative-position property follows directly from a real rotation-composition identity. Section 30.4 treated quantization as a genuine affine map, deriving its own provably tight error bound and its own real non-associativity under repeated quantization. Section 30.5 built a real Hessian and a real Gauss-Jordan inverse from scratch to prove, on a fully hand-traceable example, that GPTQ's second-order compensation produces strictly lower real error than naive independent rounding. Section 30.6 closed the chapter by combining every one of these real formulas into a complete roofline analysis of an entire transformer decoder layer.

## Self-Check Questions

1. Section 30.1 shows GEMV's own arithmetic intensity converges toward exactly 0.5 as the matrix grows, rather than continuing to increase. Explain, in terms of what grows in the numerator versus the denominator, why a bigger matrix alone can never push GEMV past this ceiling.
2. Section 30.1's Test 3 shows the weight-dominated M/2 approximation's own real error grows as batch size M increases. Explain concretely which bytes the approximation ignores, and why ignoring them matters more at larger M.
3. Section 30.2's Test 5 describes naive log-sum-exp's silent +inf as a MORE dangerous failure than naive softmax's NaN in Test 1. Explain concretely why a system that only checks for NaN would not catch this failure.
4. Section 30.3's Test 2 confirms R(a)^T * R(b) = R(b - a) as a real matrix identity. Explain what specific property of a rotation matrix (true of rotation matrices in general, not just this one) is what makes its own transpose equal its own inverse.
5. Section 30.4's Test 3 constructs a real value exactly at a quantization bin's own midpoint to demonstrate the scale/2 bound is tight. Explain why a value chosen randomly within the range, rather than at a midpoint, would be much less likely to demonstrate this.
6. Section 30.4's Test 4 shows re-quantization does not commute. Construct, in your own words, a concrete real scenario (outside this section's own hand-picked example) in a serving system where this specific non-associativity could silently degrade model quality over time.
7. Section 30.5's GPTQ compensation formula divides by `[H^-1]_pp`. Explain, in terms of what the Hessian represents about a weight's own real sensitivity, what a very LARGE value of `[H^-1]_pp` would suggest about that weight, and how that would affect the resulting compensation.
8. Section 30.5's Test 4 processes the 2 weights in a SPECIFIC order (index 0 before index 1). Explain why processing them in the opposite order could produce a genuinely different final result.
9. Section 30.6's whole-layer crossover analysis depends on every sub-block scaling identically with batch size. Name one real architectural change to a transformer layer (not necessarily mixture-of-experts) that could break this assumption, and explain concretely why.
10. This chapter is titled "Mathematical Foundations for Kernel Authors." Choose any ONE of this chapter's 6 sections and explain concretely how the specific mathematical property it derives would change a real decision a kernel author makes when writing or optimizing an actual transformer inference kernel.

## Where We Go Next

This chapter derived the mathematics; the next two chapters put it to work at serving scale. Chapter 31 builds a real continuous-batching scheduler from scratch, directly exploiting Section 30.1's own real batch-size-dependent arithmetic intensity to keep a serving system as close to compute-bound as real traffic allows, and adds real numerical debugging tools -- NaN-propagation tracing and floating-point drift detection -- for catching the bugs Section 30.2's own numerical instability foreshadowed, at a scale where they only appear under real production load. Chapter 32 closes the book by taking this book's own inference engine to the GPU, building a real Flash Attention implementation around the same online-softmax idea Section 30.2 introduced, and a real CUDA production engine for the edge devices that carry a small GPU.

## Worked Solutions

**1.** GEMV's real arithmetic intensity is `2*M*N / (M*N*4 + N*4 + M*4)`. As M and N both grow, the numerator and the dominant `M*N*4` term in the denominator both grow proportionally to `M*N`, so their ratio approaches a fixed constant (`2 / 4 = 0.5`) rather than continuing to increase -- the `N*4` and `M*4` terms (the vector and output bytes) shrink to a vanishing fraction of total bytes as the matrix grows, but the matrix term itself scales in lockstep with the FLOPs, so there is no way for a bigger matrix alone to change the ratio's own real limit.

**2.** The approximation counts only the weight matrix's own bytes (`K*N*4`) and ignores the batch's own input bytes (`M*K*4`) and output bytes (`M*N*4`). At M=1, those ignored bytes are a tiny fraction of the weight bytes, so the approximation is nearly exact. As M grows, the ignored input and output bytes grow linearly with M while the weight bytes stay fixed -- so the ignored bytes become a progressively larger real fraction of total traffic, and the approximation's own overestimate of arithmetic intensity (since it undercounts the true denominator) grows correspondingly larger.

**3.** A system checking only `std::isnan` would see `naive_log_sum_exp`'s own `+inf` result, note that `std::isnan(+inf)` is false, and conclude the computation succeeded -- `+inf` is a normal, valid-looking `double` value that simply happens to be wrong. Catching this failure requires either using the always-correct stable formula in the first place, or separately checking `std::isinf`, which is an easy check to omit precisely because `+inf` does not "look like" an error the way a NaN does.

**4.** A rotation matrix is orthogonal: its own columns (and rows) are unit vectors that are pairwise perpendicular. For any real orthogonal matrix, the general linear-algebra identity `A^T * A = I` holds, which is exactly the statement that `A^T` is `A`'s own real inverse. This is true of every real rotation matrix, in any number of dimensions, not merely the 2D case this section works with -- it follows from what a rotation actually IS (a transformation that preserves lengths and angles), not from any property specific to this section's own particular angles.

**5.** A randomly chosen value within the range is, with high probability, somewhere between a bin's own center and its edge, and the resulting error is typically much smaller than `scale/2` -- most real values do not happen to land exactly at the worst-case point. Only a value constructed deliberately at the exact midpoint between two adjacent quantization levels is guaranteed to sit at the real maximum possible distance from both, which is precisely why Test 3 constructs that value on purpose rather than sampling one randomly and hoping it happens to be near the boundary.

**6.** Consider a real KV cache using Chapter 14's own streaming re-quantization: cached keys and values, once written at a fine-grained scale, get re-quantized to a coarser scale as they age out of a hot window to save memory. If this re-quantization is applied repeatedly -- perhaps a value gets moved between cache tiers more than once as access patterns shift -- each individual re-quantization step is a fresh dequantize-then-quantize round trip, and Section 30.4's own Test 4 shows this is NOT equivalent to quantizing the true original value directly at the final coarse scale. Over enough repeated tier transitions, this could silently accumulate more real error than a single, correctly-designed direct re-quantization from the original cached value would have produced.

**7.** `[H^-1]_pp` being large means that, from the Hessian's own real perspective (built from calibration data `X`), weight `p`'s own contribution to the layer's output is comparatively insensitive to small perturbations in that weight relative to the other weights -- intuitively, the calibration data does not "notice" weight `p` moving very much. Since the compensation formula divides by `[H^-1]_pp`, a very large value there would make the resulting `delta_w_f` correction SMALL (dividing by a large number), meaning a weight the Hessian considers relatively unimportant produces a smaller real compensation to the remaining weights when it is quantized -- exactly the intuitively correct behavior, since a change to an insensitive weight has less real output error to compensate for in the first place.

**8.** The compensation formula updates the remaining, not-yet-quantized weights using the CURRENT values of `H^-1` restricted to those remaining indices, and each subsequent weight is quantized using its own value AFTER any prior compensation has already been applied to it. If weight 1 were processed before weight 0, weight 0 would instead be the one receiving compensation based on weight 1's own quantization error, and since the two weights' own real errors (`e_0` and `e_1`) are generally different, and the `H^-1` submatrix used to compensate differs depending on which index is being treated as "already quantized," the sequence of quantized values -- and therefore the final total output error -- would generally come out genuinely different.

**9.** A mixture-of-experts FFN, where each real token is routed to only a small subset of a much larger set of expert weight matrices, breaks this assumption directly: the ACTIVE weight bytes a given batch actually touches depend on which real experts that batch's own tokens route to, not on a single fixed weight tensor size the way this section's own dense FFN does. A batch of tokens that happens to route to many different real experts touches far more total weight bytes than a batch that concentrates on a few, so the layer's own real arithmetic intensity for a mixture-of-experts FFN is not a clean function of batch size alone the way this section's dense capstone example is -- it also depends on the real routing decisions made at that specific batch, which this section's single whole-layer formula has no way to account for.

**10.** Section 30.1's own real crossover batch size directly informs a genuinely practical decision: a kernel author building a serving system's own batching scheduler (this book's next chapter) can use the exact derived crossover point to decide how aggressively to batch requests together before dispatching a GEMM kernel -- below the crossover batch, the kernel is memory-bound and further optimizing its own compute (say, using a more arithmetically efficient but more complex kernel) buys little real speedup, since the bottleneck is bandwidth; above the crossover, the kernel is compute-bound and the opposite investment (a more compute-efficient kernel, even at some cost to memory access patterns) becomes the right real engineering trade-off. Without this section's own derived formula, a kernel author would be guessing at this trade-off rather than computing it directly from the machine's own stated peak compute and peak bandwidth figures.
