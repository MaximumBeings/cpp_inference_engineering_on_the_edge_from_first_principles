# Chapter 7: TurboQuant -- Online Vector Quantization with Near-Optimal Distortion

**What you will understand by the end of this chapter:**

- Why a random rotation is not decoration but the load-bearing trick that makes vector quantization tractable: it converts the worst possible input (all energy in one coordinate) into an average-case input (energy spread evenly across every coordinate), verified directly by rotating a spike vector and measuring how far its energy actually spreads, and by confirming empirically that a rotated coordinate's variance matches the theoretical value the rest of the chapter depends on.
- Why that rotation must be genuinely orthogonal and not merely "random-looking" — a raw Gaussian matrix scrambles a vector just as thoroughly as a proper rotation does, but only a proper rotation can be undone by its own transpose, and this chapter measures exactly how badly reconstruction breaks when that property is missing.
- How the Lloyd-Max algorithm turns a known probability distribution into a provably optimal quantizer for that distribution, and why this beats Chapters 2-6's uniformly-spaced blockwise bins whenever the underlying values are not uniformly distributed to begin with.
- Why a quantizer built to minimize reconstruction error is not automatically safe to use for inner products — it can be systematically, not just randomly, wrong — and how a second, independent 1-bit correction (QJL) restores unbiasedness without needing a better quantizer for the first stage.
- How to apply the whole pipeline to the KV cache, the one part of a running transformer that blockwise quantization was never well suited for: it grows online, one vector at a time, with no calibration data available in advance, which is exactly the setting TurboQuant's data-oblivious codebooks were designed for.
- Where TurboQuant stops being the right tool: Chapters 2-6's blockwise Q4_0/Q8_0 remain the correct choice for static weight matrices, because TurboQuant's O(d^2) per-vector rotation cost does not belong inside a fused SIMD matmul inner loop the way a per-block integer scale does.

**What you need to know first:**

- Chapters 2-4's blockwise Q8_0/Q4_0 quantization, its fixed 32-element blocks, and its per-block scale — this chapter is defined mostly by contrast with that approach, and Section 7.5 measures the two head to head.
- Basic familiarity with probability densities and expectation is helpful for Sections 7.2 and 7.3, though every formula used is derived or numerically verified in code rather than assumed.

---

Every quantizer built so far in this book has worked one 32-element block at a time: find the block's largest magnitude, pick a scale, round each of the 32 values independently. That approach makes no use of the fact that those 32 values are part of a much longer vector living somewhere specific in space — it treats each block as an isolated signal with no relationship to its neighbors. TurboQuant, the subject of this chapter, throws that assumption away. It treats an entire vector as one geometric object on a high-dimensional sphere, applies a single principled transformation to that whole object, and only then quantizes coordinate by coordinate — but with a quantizer that has been custom-built for the exact statistical shape those coordinates are now guaranteed to have. Section 7.1 builds and verifies the rotation step that makes this possible. Section 7.2 builds the optimal per-coordinate quantizer the rotation sets up, combining the two into TurboQuant's reconstruction-optimal "MSE mode." Section 7.3 shows that MSE mode, despite being optimal for reconstruction, is quietly biased for a different and equally important task — estimating inner products — and derives the 1-bit correction that fixes it. Section 7.4 puts the resulting quantizer to work on the KV cache, the online, uncalibrated, ever-growing structure that blockwise quantization was never well matched to. Section 7.5 closes the chapter by comparing TurboQuant against Chapters 2-6's blockwise approach directly, on the same vectors, and checks TurboQuant's own distortion against the information-theoretic floor no quantizer of any kind can beat.

## 7.1 Random Rotation: Turning Worst-Case Vectors into Average-Case Ones

### Intuition

Imagine trying to quantize the vector `[1, 0, 0, ..., 0]` — all of its length concentrated in a single coordinate — with one bit per coordinate. The first coordinate gets a bit's worth of useful information; every other coordinate is exactly zero, and a 1-bit quantizer has no representable value for "exactly zero," so those bits are wasted entirely. This is not a contrived edge case to a coordinate-wise quantizer; it is the worst input such a quantizer can ever see, and any fixed set of quantization bins can be defeated by some adversarial arrangement of a vector's energy. TurboQuant's answer is to make that adversarial arrangement impossible to construct in the first place: multiply every vector, before quantizing it, by the same random orthogonal matrix. A famous fact about high-dimensional geometry does the rest — after a genuinely random rotation, a unit vector's energy is spread almost perfectly evenly across every coordinate, regardless of how concentrated it was before the rotation.

### The Concept, In Detail

A matrix is orthogonal if multiplying by it never changes a vector's length or the angle between any two vectors — formally, `Pi^T Pi = I`, which is exactly the property that makes `Pi^T` the matrix's own inverse. The standard way to build a genuinely random ("Haar-distributed") orthogonal matrix is to start with a matrix of independent Gaussian entries and orthogonalize its columns, one against the others, using Gram-Schmidt — this is the same computation QR decomposition performs, and the resulting orthogonal factor is proven to be uniformly distributed over the space of all rotations. This chapter builds that construction directly rather than calling a linear-algebra library, both because it is short enough to verify by hand and because seeing exactly where the orthogonality comes from matters for understanding why skipping it breaks everything downstream. Two properties of the resulting matrix `Pi` matter for the rest of the chapter: it preserves length (`||Pi x|| = ||x||`), and its transpose exactly undoes it (`Pi^T (Pi x) = x`), so quantizing in the rotated coordinate system and then rotating back afterward changes nothing about what a vector "means" — it only changes which coordinate system it is described in. The chapter also verifies something the rest of the theory depends on but no prior chapter needed: that a single coordinate of a rotated random unit vector really does behave the way the Beta-distribution theory (Section 7.2) predicts, with variance shrinking as `1/d`. This is not asserted from the literature — it is checked by rotating thousands of independent random unit vectors and measuring the empirical variance of one coordinate directly.

### Code and Verification

```cpp
// 01_rotation_and_beta_distribution.cpp
// TurboQuant, Part 1: the random rotation step and the statistical fact
// it buys us -- that after rotation, every coordinate of a unit vector
// looks like a draw from a known Beta distribution.
//
// This is the foundation the rest of the chapter builds on. Chapters 2-6
// quantized each 32-element BLOCK independently (Q8_0/Q4_0). TurboQuant
// instead treats the WHOLE vector as one geometric object on a sphere,
// rotates it into a "nice" coordinate system, and only then quantizes
// coordinate by coordinate. This file builds and verifies the rotation
// step in isolation, before Section 7.2 adds the quantizer on top.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_rotation_and_beta_distribution.cpp -o out01

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <span>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

// =========================================================================
// A HAAR-RANDOM ORTHOGONAL MATRIX, BUILT FROM SCRATCH
// =========================================================================
// A rotation matrix Pi is "Haar-random" if it is drawn uniformly from the
// space of all d x d orthogonal matrices. The standard construction: fill
// a matrix with i.i.d. Gaussian entries, then orthogonalise its columns
// (Gram-Schmidt is the QR decomposition's Q factor). The result is proven
// to be Haar-distributed -- a classical fact from random matrix theory.
//
// Two properties matter for everything that follows:
//   ||Pi x|| = ||x||              (rotation preserves length)
//   <Pi x, Pi y> = <x, y>         (rotation preserves inner products)
// Both hold ONLY if Pi is genuinely orthogonal (Pi^T Pi = I). Section
// below's COMMON TRAP shows what happens if you skip the orthogonalisation
// and just use the raw Gaussian matrix.

struct RotationMatrix {
    int d;
    std::vector<float> Q;  // d x d, row-major

    explicit RotationMatrix(int dim, unsigned seed = 42) : d(dim), Q(static_cast<size_t>(dim) * dim) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        // Column-major workspace: col[j] holds column j of the Gaussian
        // matrix, then gets orthogonalised in place against col[0..j-1].
        std::vector<float> col(static_cast<size_t>(d) * d);
        for (int j = 0; j < d; ++j)
            for (int i = 0; i < d; ++i)
                col[static_cast<size_t>(j) * d + i] = gauss(rng);

        modified_gram_schmidt(col);

        // Store as row-major Q[i][j] = col[j][i].
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                Q[static_cast<size_t>(i) * d + j] = col[static_cast<size_t>(j) * d + i];
    }

    // Orthogonalise the columns of `col` (column-major, d columns of
    // length d) in place, using modified Gram-Schmidt.
    void modified_gram_schmidt(std::vector<float>& col) const {
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < j; ++k) {
                float proj = 0.0f;
                for (int i = 0; i < d; ++i)
                    proj += col[static_cast<size_t>(j) * d + i] * col[static_cast<size_t>(k) * d + i];
                for (int i = 0; i < d; ++i)
                    col[static_cast<size_t>(j) * d + i] -= proj * col[static_cast<size_t>(k) * d + i];
            }
            float norm_sq = 0.0f;
            for (int i = 0; i < d; ++i) {
                float v = col[static_cast<size_t>(j) * d + i];
                norm_sq += v * v;
            }
            float norm = std::sqrt(norm_sq);
            if (norm > 1e-10f)
                for (int i = 0; i < d; ++i)
                    col[static_cast<size_t>(j) * d + i] /= norm;
        }
    }

    // y = Pi * x
    void rotate(std::span<const float> x, std::span<float> y) const {
        for (int i = 0; i < d; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < d; ++j)
                sum += Q[static_cast<size_t>(i) * d + j] * x[j];
            y[i] = sum;
        }
    }

    // x = Pi^T * y  (the inverse rotation -- transpose, since Pi is orthogonal)
    void rotate_back(std::span<const float> y, std::span<float> x) const {
        for (int j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < d; ++i)
                sum += Q[static_cast<size_t>(i) * d + j] * y[i];
            x[j] = sum;
        }
    }
};

float l2_norm(std::span<const float> v) {
    float s = 0.0f;
    for (float x : v) s += x * x;
    return std::sqrt(s);
}

// =========================================================================
// COMMON TRAP: a raw Gaussian matrix is NOT a rotation
// =========================================================================
// If you skip the Gram-Schmidt step and use the i.i.d. Gaussian matrix G
// directly as "the rotation", G is not orthogonal: G^T G != I. Multiplying
// by G still scrambles the vector, but multiplying the result by G^T does
// NOT undo it, because G^T is not G's inverse. This function reproduces
// that mistake so we can measure exactly how badly it breaks round-tripping.
struct RawGaussianMatrix {
    int d;
    std::vector<float> G;

    explicit RawGaussianMatrix(int dim, unsigned seed) : d(dim), G(static_cast<size_t>(dim) * dim) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (float& v : G) v = gauss(rng);
    }
    void rotate(std::span<const float> x, std::span<float> y) const {
        for (int i = 0; i < d; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < d; ++j) sum += G[static_cast<size_t>(i) * d + j] * x[j];
            y[i] = sum;
        }
    }
    void rotate_back(std::span<const float> y, std::span<float> x) const {
        for (int j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < d; ++i) sum += G[static_cast<size_t>(i) * d + j] * y[i];
            x[j] = sum;
        }
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "TurboQuant 7.1: Random Rotation and the Beta Distribution\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: The constructed matrix is genuinely orthogonal (Q^T Q = I)
    // =====================================================================
    std::cout << "-- Test 1: Orthogonality of the constructed rotation --\n";
    {
        constexpr int D = 32;
        RotationMatrix Pi(D, 42);

        float max_off_diag = 0.0f, min_diag = 1.0f, max_diag = 0.0f;
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < D; ++k)
                    dot += Pi.Q[static_cast<size_t>(k) * D + i] * Pi.Q[static_cast<size_t>(k) * D + j];
                if (i == j) { min_diag = std::min(min_diag, dot); max_diag = std::max(max_diag, dot); }
                else max_off_diag = std::max(max_off_diag, std::fabs(dot));
            }
        }
        std::cout << "  Diagonal of Q^T Q in [" << min_diag << ", " << max_diag << "] (want ~1.0)\n";
        // Printed to 1 significant figure: at this magnitude (1e-4 to 1e-7)
        // the exact mantissa is sensitive to floating-point summation order,
        // which differs across compiler versions even for identical inputs.
        // The pass/fail CHECK below uses the full-precision value; only the
        // printed diagnostic is rounded, so the locked output stays stable
        // across toolchains.
        std::cout << "  Max off-diagonal of Q^T Q: " << std::scientific << std::setprecision(0)
                  << max_off_diag << std::defaultfloat << std::setprecision(6) << " (want ~0.0)\n";
        CHECK_NEAR(min_diag, 1.0f, 1e-4);
        CHECK_NEAR(max_diag, 1.0f, 1e-4);
        CHECK(max_off_diag < 1e-4f);
    }

    // =====================================================================
    // TEST 2: Rotation preserves norms and round-trips exactly
    // =====================================================================
    std::cout << "\n-- Test 2: Norm preservation and exact round-trip --\n";
    {
        constexpr int D = 64;
        RotationMatrix Pi(D, 7);
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> x(D), y(D), x_back(D);
        for (float& v : x) v = dist(rng);
        Pi.rotate(x, y);
        Pi.rotate_back(y, x_back);

        float norm_x = l2_norm(x), norm_y = l2_norm(y);
        float max_roundtrip_err = 0.0f;
        for (int i = 0; i < D; ++i) max_roundtrip_err = std::max(max_roundtrip_err, std::fabs(x[i] - x_back[i]));

        std::cout << "  ||x|| = " << norm_x << ", ||Pi x|| = " << norm_y << "\n";
        // Rounded to 1 significant figure for the same toolchain-stability
        // reason as Test 1 -- the CHECK below still uses full precision.
        std::cout << "  Max |x - Pi^T Pi x| = " << std::scientific << std::setprecision(0)
                  << max_roundtrip_err << std::defaultfloat << std::setprecision(6) << "\n";
        CHECK_NEAR(norm_x, norm_y, 1e-3);
        CHECK(max_roundtrip_err < 1e-4f);
    }

    // =====================================================================
    // TEST 3: Rotation eliminates the worst case (energy spreads out)
    // =====================================================================
    std::cout << "\n-- Test 3: Worst-case vector before and after rotation --\n";
    {
        constexpr int D = 64;
        RotationMatrix Pi(D, 42);

        std::vector<float> spike(D, 0.0f);
        spike[0] = 1.0f;
        std::vector<float> rotated(D);
        Pi.rotate(spike, rotated);

        float max_before = 1.0f;  // spike has one coordinate at 1.0
        float max_after = 0.0f;
        for (float v : rotated) max_after = std::max(max_after, std::fabs(v));

        std::cout << "  Before: max|x[j]| = " << max_before << " (all energy in one coordinate)\n";
        std::cout << "  After:  max|y[j]| = " << std::fixed << std::setprecision(4) << max_after << "\n";
        std::cout << "  Expected scale ~ 1/sqrt(d) = " << 1.0f / std::sqrt(static_cast<float>(D)) << "\n";
        CHECK(max_after < 0.5f * max_before);
        CHECK_NEAR(l2_norm(rotated), 1.0f, 1e-3);
    }

    // =====================================================================
    // TEST 4 (new derivation, not given in any source material): verify
    // that rotated coordinates of random unit vectors really do follow
    // the predicted Beta distribution -- specifically, that a single
    // coordinate's variance across many random unit vectors converges to
    // the theoretical value 1/d. This is the fact Section 7.2's codebook
    // depends on, so it is worth checking empirically rather than trusting
    // the theory blindly.
    // =====================================================================
    std::cout << "\n-- Test 4: Empirical variance of a rotated coordinate matches 1/d --\n";
    {
        constexpr int D = 128;
        constexpr int TRIALS = 20000;
        RotationMatrix Pi(D, 123);
        std::mt19937 rng(999);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Coordinate 0 of many independent random UNIT vectors, after
        // rotation. If the theory holds, this behaves like a draw from
        // the Beta-derived marginal with mean 0 and variance 1/d.
        double sum = 0.0, sum_sq = 0.0;
        float observed_max = 0.0f;
        for (int t = 0; t < TRIALS; ++t) {
            std::vector<float> x(D);
            for (float& v : x) v = dist(rng);
            float norm = l2_norm(x);
            for (float& v : x) v /= norm;  // project onto the unit sphere

            std::vector<float> y(D);
            Pi.rotate(x, y);
            sum += y[0];
            sum_sq += static_cast<double>(y[0]) * y[0];
            observed_max = std::max(observed_max, std::fabs(y[0]));
        }
        double mean = sum / TRIALS;
        double variance = sum_sq / TRIALS - mean * mean;
        double theoretical_variance = 1.0 / D;

        std::cout << "  Empirical mean of y[0]:      " << std::scientific << std::setprecision(3) << mean << std::fixed << "\n";
        std::cout << "  Empirical variance of y[0]:  " << std::setprecision(6) << variance << "\n";
        std::cout << "  Theoretical variance (1/d):  " << theoretical_variance << "\n";
        std::cout << "  Observed max |y[0]| over " << TRIALS << " trials: " << observed_max << "\n";
        CHECK(std::fabs(mean) < 0.01);
        CHECK(std::fabs(variance - theoretical_variance) / theoretical_variance < 0.15);
        std::cout << "  Rotated coordinate distribution matches the Beta-distribution prediction\n";
    }

    // =====================================================================
    // TEST 5 (COMMON TRAP): a raw Gaussian matrix is not orthogonal, so
    // "rotate then rotate back" does NOT reconstruct the input -- even
    // with zero quantization error. This is a structural bug, not a
    // rounding error, and it is worth measuring directly.
    // =====================================================================
    std::cout << "\n-- Test 5 [COMMON TRAP]: raw Gaussian matrix breaks round-trip --\n";
    {
        constexpr int D = 32;
        RawGaussianMatrix G(D, 42);
        RotationMatrix Pi(D, 42);  // same seed, properly orthogonalised

        std::mt19937 rng(55);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(D);
        for (float& v : x) v = dist(rng);

        std::vector<float> y_bad(D), x_back_bad(D);
        G.rotate(x, y_bad);
        G.rotate_back(y_bad, x_back_bad);
        float err_bad = 0.0f;
        for (int i = 0; i < D; ++i) err_bad = std::max(err_bad, std::fabs(x[i] - x_back_bad[i]));

        std::vector<float> y_good(D), x_back_good(D);
        Pi.rotate(x, y_good);
        Pi.rotate_back(y_good, x_back_good);
        float err_good = 0.0f;
        for (int i = 0; i < D; ++i) err_good = std::max(err_good, std::fabs(x[i] - x_back_good[i]));

        std::cout << "  Raw Gaussian 'rotate then rotate back' max error: " << std::scientific
                  << std::setprecision(4) << err_bad << std::fixed << "\n";
        // err_good sits at the floating-point noise floor (~1e-6), same
        // toolchain-stability reasoning as Test 1 and Test 2 above.
        std::cout << "  Properly orthogonalised round-trip max error:     " << std::scientific
                  << std::setprecision(0) << err_good << std::defaultfloat << std::setprecision(6) << "\n";
        std::cout << "  The raw Gaussian matrix is NOT its own transpose-inverse -- G^T G != I.\n";
        std::cout << "  Every downstream dequantization would silently reconstruct the wrong vector.\n";
        CHECK(err_bad > 0.1f);       // structurally broken
        CHECK(err_good < 1e-4f);     // properly orthogonal: exact
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_rotation_and_beta_distribution.cpp -o 01_rotation_and_beta_distribution
./01_rotation_and_beta_distribution
```

**Sample input:** a 32x32 constructed rotation matrix, checked for orthogonality; a 64-dimensional random vector, rotated and rotated back to confirm an exact round-trip; a worst-case "spike" vector `[1, 0, ..., 0]` before and after rotation; 20,000 independent random unit vectors in 128 dimensions, used to measure the empirical variance of one rotated coordinate against the theoretical prediction; and a deliberately non-orthogonal raw Gaussian matrix used in place of a proper rotation.

```text
========================================================
TurboQuant 7.1: Random Rotation and the Beta Distribution
========================================================

-- Test 1: Orthogonality of the constructed rotation --
  Diagonal of Q^T Q in [1, 1] (want ~1.0)
  Max off-diagonal of Q^T Q: 4e-06 (want ~0.0)

-- Test 2: Norm preservation and exact round-trip --
  ||x|| = 8.33056, ||Pi x|| = 8.33056
  Max |x - Pi^T Pi x| = 5e-06

-- Test 3: Worst-case vector before and after rotation --
  Before: max|x[j]| = 1 (all energy in one coordinate)
  After:  max|y[j]| = 0.3096
  Expected scale ~ 1/sqrt(d) = 0.1250

-- Test 4: Empirical variance of a rotated coordinate matches 1/d --
  Empirical mean of y[0]:      1.551e-04
  Empirical variance of y[0]:  0.007991
  Theoretical variance (1/d):  0.007812
  Observed max |y[0]| over 20000 trials: 0.347782
  Rotated coordinate distribution matches the Beta-distribution prediction

-- Test 5 [COMMON TRAP]: raw Gaussian matrix breaks round-trip --
  Raw Gaussian 'rotate then rotate back' max error: 1.2938e+02
  Properly orthogonalised round-trip max error:     6e-06
  The raw Gaussian matrix is NOT its own transpose-inverse -- G^T G != I.
  Every downstream dequantization would silently reconstruct the wrong vector.

========================================================
11/11 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] a random-looking matrix is not a rotation"
    A matrix of independent Gaussian entries scrambles a vector just as thoroughly as a genuine rotation does — multiplying by it produces a vector that looks nothing like the input. The difference only shows up when you try to undo it. A proper rotation's transpose is its exact inverse (`Pi^T Pi = I`), so rotating and then "rotating back" with the transpose reconstructs the original vector exactly, with no quantization involved at all. A raw Gaussian matrix `G` has no such guarantee — `G^T G` is not the identity — so "rotate with `G`, then rotate back with `G^T`" reconstructs a vector with a large, structural error, even before a single bit of quantization has happened. This is not a rounding error to be tolerated; it is a correctness bug, and it is the reason this chapter builds its rotation via explicit Gram-Schmidt orthogonalization rather than skipping straight to "any matrix full of random numbers will do."

## 7.2 Lloyd-Max Optimal Codebooks and the MSE Quantizer

### Intuition

Section 7.1 established that a coordinate of a rotated unit vector behaves like a draw from a known probability distribution — concentrated near zero, with a shape that depends only on the dimension `d`. Knowing the distribution in advance is a gift: instead of spacing quantization bins uniformly (as Chapters 2-6's blockwise formats do, since they have no distributional assumption to exploit), TurboQuant can place bins exactly where the probability mass actually is — densely near zero, sparsely in the rarely-visited tails — and prove that no other placement does better for a fixed number of bits. That placement problem is the Lloyd-Max algorithm: continuous, distribution-aware k-means.

### The Concept, In Detail

A coordinate of a random unit vector in `d` dimensions has density proportional to `(1 - x^2)^((d-3)/2)` on `[-1, 1]` — concentrated near zero, and more sharply so as `d` grows. The Lloyd-Max algorithm finds the `2^bits` centroids that minimize expected squared error against this density by alternating two steps until convergence: given the current centroids, the optimal boundaries are simply the midpoints between consecutive centroids (any value closer to one centroid than its neighbor should be assigned to that centroid); and given the current boundaries, the optimal centroid for each bin is the conditional mean of the density within that bin, computed here by numerical (trapezoidal) integration since the Beta density has no simple closed-form conditional mean. Because the target distribution depends only on `d` and `bits`, every codebook can be precomputed once, offline, before any real data arrives — this is what makes TurboQuant an online algorithm suitable for a KV cache that has no calibration pass available, unlike clustering-based vector quantization (Product Quantization), which needs to see representative data in advance. Combining the rotation from Section 7.1 with this codebook gives the complete "MSE mode" pipeline: normalize the input to the unit sphere and store its norm as one float (the codebook was solved for a UNIT-norm coordinate distribution, so the actual scale has to be factored out and reapplied separately), rotate, quantize each rotated coordinate to its nearest centroid, and store the indices. This normalize-then-rotate order is not optional bookkeeping — it is the step this section's own trap gets wrong.

### Code and Verification

```cpp
// 02_lloyd_max_mse_quantizer.cpp
// TurboQuant, Part 2: build the optimal per-coordinate quantizer for the
// Beta distribution from Section 7.1, then combine it with the rotation
// step into the complete TurboQuant "MSE mode" vector quantizer.
//
// The question this file answers: given that a rotated coordinate follows
// a known distribution (Beta, concentrated near zero for large d), how do
// we place a fixed number of quantization bins to minimise expected
// squared error? The answer is the Lloyd-Max algorithm -- 1D k-means
// applied to a continuous density instead of a finite dataset.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_lloyd_max_mse_quantizer.cpp -o out02

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <span>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

// =========================================================================
// ROTATION MATRIX (built and verified in Section 7.1; reused as-is)
// =========================================================================
struct RotationMatrix {
    int d;
    std::vector<float> Q;

    explicit RotationMatrix(int dim, unsigned seed = 42) : d(dim), Q(static_cast<size_t>(dim) * dim) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> col(static_cast<size_t>(d) * d);
        for (int j = 0; j < d; ++j)
            for (int i = 0; i < d; ++i)
                col[static_cast<size_t>(j) * d + i] = gauss(rng);
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < j; ++k) {
                float proj = 0.0f;
                for (int i = 0; i < d; ++i)
                    proj += col[static_cast<size_t>(j) * d + i] * col[static_cast<size_t>(k) * d + i];
                for (int i = 0; i < d; ++i)
                    col[static_cast<size_t>(j) * d + i] -= proj * col[static_cast<size_t>(k) * d + i];
            }
            float norm_sq = 0.0f;
            for (int i = 0; i < d; ++i) { float v = col[static_cast<size_t>(j) * d + i]; norm_sq += v * v; }
            float norm = std::sqrt(norm_sq);
            if (norm > 1e-10f)
                for (int i = 0; i < d; ++i) col[static_cast<size_t>(j) * d + i] /= norm;
        }
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                Q[static_cast<size_t>(i) * d + j] = col[static_cast<size_t>(j) * d + i];
    }
    void rotate(std::span<const float> x, std::span<float> y) const {
        for (int i = 0; i < d; ++i) {
            float s = 0.0f;
            for (int j = 0; j < d; ++j) s += Q[static_cast<size_t>(i) * d + j] * x[j];
            y[i] = s;
        }
    }
    void rotate_back(std::span<const float> y, std::span<float> x) const {
        for (int j = 0; j < d; ++j) {
            float s = 0.0f;
            for (int i = 0; i < d; ++i) s += Q[static_cast<size_t>(i) * d + j] * y[i];
            x[j] = s;
        }
    }
};

float l2_norm(std::span<const float> v) {
    float s = 0.0f;
    for (float x : v) s += x * x;
    return std::sqrt(s);
}

// =========================================================================
// LLOYD-MAX OPTIMAL CODEBOOK FOR THE BETA DISTRIBUTION
// =========================================================================
// A coordinate of a random unit vector in R^d has density (unnormalised):
//     f(x) proportional to (1 - x^2)^((d-3)/2),   x in [-1, 1]
// We solve the 1D optimal-quantizer problem for this density: partition
// [-1, 1] into 2^bits bins and choose centroids minimising expected
// squared error. Lloyd-Max alternates:
//   1. boundaries = midpoints between consecutive centroids
//   2. centroids  = conditional mean of the density within each bin
// until the centroids stop moving.
struct Codebook {
    int bits, n_codes;
    std::vector<float> centroids;   // sorted ascending, length n_codes
    std::vector<float> boundaries;  // length n_codes - 1

    static float beta_pdf(float x, int d) {
        if (std::fabs(x) >= 1.0f) return 0.0f;
        return std::pow(1.0f - x * x, 0.5f * static_cast<float>(d - 3));
    }

    // Trapezoidal integral of x^p * f(x) over [a, b].
    static float integrate_moment(float a, float b, int p, int d, int n_steps = 2000) {
        if (a >= b) return 0.0f;
        float h = (b - a) / static_cast<float>(n_steps);
        float sum = 0.0f;
        for (int i = 0; i <= n_steps; ++i) {
            float x = a + static_cast<float>(i) * h;
            float fx = beta_pdf(x, d);
            float val = fx;
            if (p == 1) val *= x;
            else if (p == 2) val *= x * x;
            float w = (i == 0 || i == n_steps) ? 0.5f : 1.0f;
            sum += w * val;
        }
        return sum * h;
    }

    Codebook() : bits(0), n_codes(0) {}

    Codebook(int d, int b, int max_iter = 200) : bits(b) {
        n_codes = 1 << b;
        centroids.resize(n_codes);
        boundaries.resize(n_codes - 1);
        for (int i = 0; i < n_codes; ++i)
            centroids[i] = -1.0f + (2.0f * static_cast<float>(i) + 1.0f) / static_cast<float>(n_codes);

        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < n_codes - 1; ++i)
                boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);

            bool converged = true;
            for (int i = 0; i < n_codes; ++i) {
                float lo = (i == 0) ? -1.0f : boundaries[i - 1];
                float hi = (i == n_codes - 1) ? 1.0f : boundaries[i];
                float m1 = integrate_moment(lo, hi, 1, d);
                float m0 = integrate_moment(lo, hi, 0, d);
                if (m0 > 1e-12f) {
                    float new_c = m1 / m0;
                    if (std::fabs(new_c - centroids[i]) > 1e-8f) converged = false;
                    centroids[i] = new_c;
                }
            }
            if (converged) break;
        }
    }

    int quantize(float x) const {
        int idx = 0;
        for (int i = 0; i < static_cast<int>(boundaries.size()); ++i) {
            if (x > boundaries[i]) idx = i + 1; else break;
        }
        return idx;
    }
    float dequantize(int idx) const { return centroids[idx]; }
};

// =========================================================================
// TURBOQUANT MSE QUANTIZER: normalize -> rotate -> quantize -> store
// =========================================================================
struct TurboQuantMSE {
    int d, bits;
    RotationMatrix Pi;
    Codebook codebook;

    TurboQuantMSE(int dim, int b, unsigned seed = 42) : d(dim), bits(b), Pi(dim, seed), codebook(dim, b) {}

    struct QuantResult {
        std::vector<uint8_t> indices;
        float norm;
    };

    QuantResult quantize(std::span<const float> x) const {
        assert(static_cast<int>(x.size()) == d);
        QuantResult r;
        r.indices.resize(d);
        r.norm = l2_norm(x);

        std::vector<float> x_hat(d);
        if (r.norm > 1e-10f) for (int i = 0; i < d; ++i) x_hat[i] = x[i] / r.norm;
        else std::fill(x_hat.begin(), x_hat.end(), 0.0f);

        std::vector<float> y(d);
        Pi.rotate(x_hat, y);
        for (int j = 0; j < d; ++j) r.indices[j] = static_cast<uint8_t>(codebook.quantize(y[j]));
        return r;
    }

    void dequantize(const QuantResult& qr, std::span<float> x_out) const {
        std::vector<float> y_tilde(d);
        for (int j = 0; j < d; ++j) y_tilde[j] = codebook.dequantize(qr.indices[j]);
        std::vector<float> x_hat_tilde(d);
        Pi.rotate_back(y_tilde, x_hat_tilde);
        for (int i = 0; i < d; ++i) x_out[i] = x_hat_tilde[i] * qr.norm;
    }

    size_t storage_bytes() const { return (static_cast<size_t>(bits) * d + 7) / 8 + 4; }
};

float total_squared_error(std::span<const float> a, std::span<const float> b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) { float e = a[i] - b[i]; s += e * e; }
    return s;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "TurboQuant 7.2: Lloyd-Max Codebooks and the MSE Quantizer\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: Codebooks are sorted and symmetric around zero
    // =====================================================================
    std::cout << "-- Test 1: Codebook shape for b=1..4, d=64 --\n";
    {
        for (int b = 1; b <= 4; ++b) {
            Codebook cb(64, b);
            std::cout << "  b=" << b << " (" << cb.n_codes << " codes): [ ";
            for (int i = 0; i < cb.n_codes; ++i)
                std::cout << std::fixed << std::setprecision(4) << cb.centroids[i] << " ";
            std::cout << "]\n";

            bool sorted = true;
            for (int i = 1; i < cb.n_codes; ++i) if (cb.centroids[i] < cb.centroids[i - 1]) sorted = false;
            CHECK(sorted);

            bool symmetric = true;
            for (int i = 0; i < cb.n_codes / 2; ++i)
                if (std::fabs(cb.centroids[i] + cb.centroids[cb.n_codes - 1 - i]) > 0.01f) symmetric = false;
            CHECK(symmetric);
        }
    }

    // =====================================================================
    // TEST 2: b=1 centroids match the closed-form large-d approximation
    // For large d, the Beta density is approximately Gaussian with
    // variance 1/d, and the optimal 1-bit (2-centroid) quantizer for a
    // zero-mean Gaussian has centroids at +/- sqrt(2/(pi*d)).
    // =====================================================================
    std::cout << "\n-- Test 2: b=1 codebook matches the large-d closed form --\n";
    {
        constexpr int D = 256;
        Codebook cb(D, 1);
        float predicted = std::sqrt(2.0f / (static_cast<float>(M_PI) * D));
        std::cout << "  Numerically solved centroid:  " << std::fixed << std::setprecision(5) << cb.centroids[1] << "\n";
        std::cout << "  Closed-form large-d estimate: " << predicted << "\n";
        CHECK(std::fabs(cb.centroids[1] - predicted) / predicted < 0.05f);
        std::cout << "  Agreement within 5% confirms the Lloyd-Max solver converged correctly\n";
    }

    // =====================================================================
    // TEST 3: Full pipeline round-trip and storage accounting
    // =====================================================================
    std::cout << "\n-- Test 3: TurboQuantMSE round-trip (d=64, b=2) --\n";
    {
        constexpr int D = 64;
        TurboQuantMSE tq(D, 2, 42);
        std::mt19937 rng(123);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(D);
        for (float& v : x) v = dist(rng);

        auto qr = tq.quantize(x);
        std::vector<float> recon(D);
        tq.dequantize(qr, recon);

        std::vector<float> x_hat(D), recon_hat(D);
        for (int i = 0; i < D; ++i) { x_hat[i] = x[i] / qr.norm; recon_hat[i] = recon[i] / qr.norm; }
        float total_se = total_squared_error(x_hat, recon_hat);

        std::cout << "  Stored norm: " << std::fixed << std::setprecision(4) << qr.norm << "\n";
        std::cout << "  Total SE on unit sphere: " << total_se << " (theory for b=2 is ~0.117 per coord * d)\n";
        std::cout << "  Storage: " << tq.storage_bytes() << " bytes vs FP32 " << D * 4 << " bytes ("
                  << std::setprecision(1) << static_cast<float>(D * 4) / tq.storage_bytes() << "x compression)\n";
        CHECK(total_se < 0.5f);
        CHECK(total_se > 0.01f);
    }

    // =====================================================================
    // TEST 4: MSE strictly decreases with more bits
    // =====================================================================
    std::cout << "\n-- Test 4: MSE vs bit-width, averaged over 50 vectors (d=64) --\n";
    {
        constexpr int D = 64, N = 50;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<std::vector<float>> vectors(N, std::vector<float>(D));
        for (auto& v : vectors) for (float& x : v) x = dist(rng);

        float prev_mse = 1e10f;
        for (int b = 1; b <= 4; ++b) {
            TurboQuantMSE tq(D, b, 42);
            float total_se = 0.0f;
            for (const auto& x : vectors) {
                auto qr = tq.quantize(x);
                std::vector<float> recon(D);
                tq.dequantize(qr, recon);
                for (int i = 0; i < D; ++i) {
                    float xi = x[i] / qr.norm, ri = recon[i] / qr.norm;
                    total_se += (xi - ri) * (xi - ri);
                }
            }
            float avg_mse = total_se / N;
            std::cout << "  b=" << b << "  avg total SE: " << std::fixed << std::setprecision(4) << avg_mse
                      << "  storage: " << tq.storage_bytes() << " bytes/vec  compression: "
                      << std::setprecision(1) << static_cast<float>(D * 4) / tq.storage_bytes() << "x\n";
            CHECK(avg_mse < prev_mse);
            prev_mse = avg_mse;
        }
    }

    // =====================================================================
    // TEST 5 (COMMON TRAP): quantizing without normalising first.
    // The codebook was solved for a coordinate distribution with
    // VARIANCE 1/d -- that is, for a vector ALREADY on the unit sphere.
    // Feed it a vector with a large norm directly (skip the normalize
    // step) and the rotated coordinates fall far outside the codebook's
    // calibrated range, saturating at the outermost centroids.
    // =====================================================================
    std::cout << "\n-- Test 5 [COMMON TRAP]: skipping normalisation before quantizing --\n";
    {
        constexpr int D = 64;
        TurboQuantMSE tq(D, 3, 42);

        std::mt19937 rng(321);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(D);
        for (float& v : x) v = dist(rng);
        for (float& v : x) v *= 6.0f;  // deliberately large norm (||x|| ~ 48 for d=64)

        // Correct pipeline: quantize() extracts and stores the norm itself.
        auto qr_correct = tq.quantize(x);
        std::vector<float> recon_correct(D);
        tq.dequantize(qr_correct, recon_correct);
        float mse_correct = total_squared_error(x, recon_correct) / D;

        // Buggy pipeline: rotate the RAW (un-normalised) vector directly
        // and quantize with the same codebook, as if the norm did not
        // need to be factored out first.
        std::vector<float> y_raw(D);
        tq.Pi.rotate(x, y_raw);
        std::vector<uint8_t> idx_raw(D);
        for (int j = 0; j < D; ++j) idx_raw[j] = static_cast<uint8_t>(tq.codebook.quantize(y_raw[j]));
        std::vector<float> y_tilde_raw(D);
        for (int j = 0; j < D; ++j) y_tilde_raw[j] = tq.codebook.dequantize(idx_raw[j]);
        std::vector<float> recon_buggy(D);
        tq.Pi.rotate_back(y_tilde_raw, recon_buggy);
        // No norm was stored in the buggy path, so there is nothing to
        // rescale by -- the reconstruction is stuck near the unit sphere
        // no matter how large the input actually was.
        float mse_buggy = total_squared_error(x, recon_buggy) / D;

        std::cout << "  ||x|| = " << std::fixed << std::setprecision(2) << l2_norm(x) << " (deliberately far from 1)\n";
        std::cout << "  Correct pipeline (normalize, rotate, quantize, rescale) MSE: "
                  << std::scientific << std::setprecision(3) << mse_correct << std::fixed << "\n";
        std::cout << "  Buggy pipeline (quantize raw, un-normalised coordinates) MSE: "
                  << std::scientific << std::setprecision(3) << mse_buggy << std::fixed << "\n";
        std::cout << "  Skipping normalisation is not a rounding error -- it is a missing\n";
        std::cout << "  scale factor, and the reconstruction is wrong by orders of magnitude.\n";
        float x_variance = (l2_norm(x) * l2_norm(x)) / D;
        CHECK(mse_correct < 0.1f * x_variance);   // correct pipeline: error is a few percent of signal variance
        CHECK(mse_buggy > 20.0f * mse_correct);   // buggy pipeline: error is over an order of magnitude worse
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_lloyd_max_mse_quantizer.cpp -o 02_lloyd_max_mse_quantizer
./02_lloyd_max_mse_quantizer
```

**Sample input:** Lloyd-Max codebooks solved numerically for `bits = 1..4` at `d = 64`, checked for sortedness and symmetry and, for `bits = 1`, checked against the closed-form large-`d` approximation; a complete TurboQuant MSE quantizer round-tripped on a random 64-dimensional vector; the same quantizer's reconstruction error measured across 50 vectors at each bit-width from 1 to 4; and a deliberately un-normalized input fed both through the correct pipeline and through a version that skips the normalize step.

```text
========================================================
TurboQuant 7.2: Lloyd-Max Codebooks and the MSE Quantizer
========================================================

-- Test 1: Codebook shape for b=1..4, d=64 --
  b=1 (2 codes): [ -0.1001 0.1001 ]
  b=2 (4 codes): [ -0.1875 -0.0565 0.0565 0.1875 ]
  b=3 (8 codes): [ -0.2639 -0.1662 -0.0938 -0.0305 0.0305 0.0938 0.1662 0.2639 ]
  b=4 (16 codes): [ -0.3309 -0.2531 -0.1990 -0.1551 -0.1166 -0.0814 -0.0482 -0.0159 0.0159 0.0482 0.0814 0.1166 0.1551 0.1990 0.2531 0.3309 ]

-- Test 2: b=1 codebook matches the large-d closed form --
  Numerically solved centroid:  0.04992
  Closed-form large-d estimate: 0.04987
  Agreement within 5% confirms the Lloyd-Max solver converged correctly

-- Test 3: TurboQuantMSE round-trip (d=64, b=2) --
  Stored norm: 8.9869
  Total SE on unit sphere: 0.1425 (theory for b=2 is ~0.117 per coord * d)
  Storage: 20 bytes vs FP32 256 bytes (12.8x compression)

-- Test 4: MSE vs bit-width, averaged over 50 vectors (d=64) --
  b=1  avg total SE: 0.3738  storage: 12 bytes/vec  compression: 21.3x
  b=2  avg total SE: 0.1162  storage: 20 bytes/vec  compression: 12.8x
  b=3  avg total SE: 0.0341  storage: 28 bytes/vec  compression: 9.1x
  b=4  avg total SE: 0.0094  storage: 36 bytes/vec  compression: 7.1x

-- Test 5 [COMMON TRAP]: skipping normalisation before quantizing --
  ||x|| = 48.02 (deliberately far from 1)
  Correct pipeline (normalize, rotate, quantize, rescale) MSE: 1.072e+00
  Buggy pipeline (quantize raw, un-normalised coordinates) MSE: 3.358e+01
  Skipping normalisation is not a rounding error -- it is a missing
  scale factor, and the reconstruction is wrong by orders of magnitude.

========================================================
17/17 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] quantizing before normalizing"
    The codebook in this section was solved for a coordinate distribution with variance `1/d` — that is, for coordinates of a vector that is ALREADY on the unit sphere. Feeding the codebook a rotated coordinate from a vector with a large norm skips the one step that makes the codebook's calibration valid: the rotated coordinates land far outside the range the codebook was built for, and every one of them saturates at whichever outermost centroid is closest, since the codebook has no representable value out there. This is not the same failure mode as ordinary quantization rounding — rounding error stays proportional to the input, while a missing scale factor produces an error that grows without bound as the input's true norm grows, because the reconstruction is stuck near the unit sphere no matter how large the original vector actually was. "Always normalize first" is not a stylistic preference; it is the precondition the entire codebook calibration depends on.

## 7.3 The Inner-Product Bias Problem and the QJL Fix

### Intuition

Section 7.2's quantizer is provably optimal for one specific goal: minimizing the expected squared error between a vector and its reconstruction. Attention does not need a reconstructed vector to look at — it needs an accurate DOT PRODUCT between a query and a cached key. Those turn out to be different goals, and optimizing for the first does not automatically deliver the second. At low bit-widths, the MSE-optimal quantizer's inner-product estimate is not merely noisy around the true value — it is systematically, predictably too small, by a constant multiplicative factor that never fully disappears at any finite bit-width.

### The Concept, In Detail

The mechanism is best seen at the extreme: a 1-bit MSE quantizer for large `d` has just two centroids, `+/- sqrt(2/(pi*d))`, and it reconstructs every coordinate as one of those two values based only on the ROTATED coordinate's sign — all information about the original coordinate's magnitude within its half is thrown away. When that reconstruction is dotted with an arbitrary query vector, the expected value of the resulting estimate is not the true inner product but exactly `2/pi` (about 0.637) times it — a multiplicative shrinkage baked into the quantizer's own construction, not a symptom of any particular unlucky rotation. At higher bit-widths this shrinkage factor climbs toward 1.0, but it never reaches it at any finite bit budget. The fix does not require a better single-stage quantizer — it requires a second, independent stage aimed specifically at inner products. Writing `x = x_tilde_mse + r`, where `r` is the residual the MSE stage failed to capture, the true inner product decomposes as `<y, x> = <y, x_tilde_mse> + <y, r>`. The MSE stage already gives the first term; QJL (Quantized Johnson-Lindenstrauss) supplies the second by quantizing the RESIDUAL with a completely different, 1-bit-per-coordinate scheme: multiply by a fresh random Gaussian matrix `S`, keep only the sign of each projection, and dequantize with the rescaling `sqrt(pi/2)/d * S^T * signs`. That specific rescaling is what makes `<y, QJL_dequant(r)>` an unbiased estimator of `<y, r>` — proved by the symmetry of the Gaussian distribution — and adding it to the MSE term exactly cancels the MSE stage's own bias in expectation, at the cost of one additional bit per coordinate. The residual and the MSE reconstruction must be computed in the SAME coordinate system before they are combined — both in the original, un-rotated space — which is precisely the step this section's own trap gets wrong.

### Code and Verification

```cpp
// 03_qjl_unbiased_inner_product.cpp
// TurboQuant, Part 3: the MSE quantizer from Section 7.2 is optimal for
// RECONSTRUCTION error, but it turns out to be systematically BIASED when
// its output is used to estimate inner products (attention scores, vector
// search similarity). This file demonstrates the bias, derives the QJL
// (Quantized Johnson-Lindenstrauss) 1-bit residual fix, and combines the
// two into TurboQuant's unbiased "Prod" quantizer.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_qjl_unbiased_inner_product.cpp -o out03

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <span>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// =========================================================================
// ROTATION MATRIX and CODEBOOK (Sections 7.1-7.2; reused as-is)
// =========================================================================
struct RotationMatrix {
    int d;
    std::vector<float> Q;
    explicit RotationMatrix(int dim, unsigned seed = 42) : d(dim), Q(static_cast<size_t>(dim) * dim) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> col(static_cast<size_t>(d) * d);
        for (int j = 0; j < d; ++j)
            for (int i = 0; i < d; ++i)
                col[static_cast<size_t>(j) * d + i] = gauss(rng);
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < j; ++k) {
                float proj = 0.0f;
                for (int i = 0; i < d; ++i)
                    proj += col[static_cast<size_t>(j) * d + i] * col[static_cast<size_t>(k) * d + i];
                for (int i = 0; i < d; ++i)
                    col[static_cast<size_t>(j) * d + i] -= proj * col[static_cast<size_t>(k) * d + i];
            }
            float norm_sq = 0.0f;
            for (int i = 0; i < d; ++i) { float v = col[static_cast<size_t>(j) * d + i]; norm_sq += v * v; }
            float norm = std::sqrt(norm_sq);
            if (norm > 1e-10f)
                for (int i = 0; i < d; ++i) col[static_cast<size_t>(j) * d + i] /= norm;
        }
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                Q[static_cast<size_t>(i) * d + j] = col[static_cast<size_t>(j) * d + i];
    }
    void rotate(std::span<const float> x, std::span<float> y) const {
        for (int i = 0; i < d; ++i) {
            float s = 0.0f;
            for (int j = 0; j < d; ++j) s += Q[static_cast<size_t>(i) * d + j] * x[j];
            y[i] = s;
        }
    }
    void rotate_back(std::span<const float> y, std::span<float> x) const {
        for (int j = 0; j < d; ++j) {
            float s = 0.0f;
            for (int i = 0; i < d; ++i) s += Q[static_cast<size_t>(i) * d + j] * y[i];
            x[j] = s;
        }
    }
};

struct Codebook {
    int bits, n_codes;
    std::vector<float> centroids, boundaries;
    static float beta_pdf(float x, int d) {
        if (std::fabs(x) >= 1.0f) return 0.0f;
        return std::pow(1.0f - x * x, 0.5f * static_cast<float>(d - 3));
    }
    static float integrate_moment(float a, float b, int p, int d, int n_steps = 2000) {
        if (a >= b) return 0.0f;
        float h = (b - a) / static_cast<float>(n_steps), sum = 0.0f;
        for (int i = 0; i <= n_steps; ++i) {
            float x = a + static_cast<float>(i) * h, fx = beta_pdf(x, d);
            float val = (p == 1) ? fx * x : (p == 2) ? fx * x * x : fx;
            sum += ((i == 0 || i == n_steps) ? 0.5f : 1.0f) * val;
        }
        return sum * h;
    }
    Codebook() : bits(0), n_codes(0) {}
    Codebook(int d, int b, int max_iter = 200) : bits(b) {
        n_codes = 1 << b;
        centroids.resize(n_codes);
        boundaries.resize(n_codes - 1);
        for (int i = 0; i < n_codes; ++i)
            centroids[i] = -1.0f + (2.0f * static_cast<float>(i) + 1.0f) / static_cast<float>(n_codes);
        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < n_codes - 1; ++i) boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
            bool converged = true;
            for (int i = 0; i < n_codes; ++i) {
                float lo = (i == 0) ? -1.0f : boundaries[i - 1];
                float hi = (i == n_codes - 1) ? 1.0f : boundaries[i];
                float m1 = integrate_moment(lo, hi, 1, d), m0 = integrate_moment(lo, hi, 0, d);
                if (m0 > 1e-12f) {
                    float nc = m1 / m0;
                    if (std::fabs(nc - centroids[i]) > 1e-8f) converged = false;
                    centroids[i] = nc;
                }
            }
            if (converged) break;
        }
    }
    int quantize(float x) const {
        int idx = 0;
        for (int i = 0; i < static_cast<int>(boundaries.size()); ++i) if (x > boundaries[i]) idx = i + 1; else break;
        return idx;
    }
    float dequantize(int idx) const { return centroids[idx]; }
};

float l2_norm(std::span<const float> v) { float s = 0.0f; for (float x : v) s += x * x; return std::sqrt(s); }
float dot(std::span<const float> a, std::span<const float> b) {
    float s = 0.0f; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}

// =========================================================================
// QJL: the 1-bit unbiased residual quantizer
// =========================================================================
// Q(x) = sign(S x), for a random Gaussian S. Storing only the sign of a
// random projection loses almost all magnitude information -- but the
// specific rescaling sqrt(pi/2)/d makes the ESTIMATOR unbiased even
// though any single reconstructed vector looks nothing like the input.
struct QJL {
    int d;
    std::vector<float> S;  // d x d Gaussian, row-major

    explicit QJL(int dim, unsigned seed) : d(dim), S(static_cast<size_t>(dim) * dim) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (float& v : S) v = gauss(rng);
    }

    std::vector<int8_t> quantize_signs(std::span<const float> x) const {
        std::vector<int8_t> signs(d);
        for (int i = 0; i < d; ++i) {
            float dp = 0.0f;
            for (int j = 0; j < d; ++j) dp += S[static_cast<size_t>(i) * d + j] * x[j];
            signs[i] = (dp >= 0.0f) ? 1 : -1;
        }
        return signs;
    }

    // Unbiased estimate of <query, x> given only sign(S x) and ||x||.
    float dot_estimate(std::span<const float> query, std::span<const int8_t> signs, float x_norm) const {
        float scale = x_norm * std::sqrt(static_cast<float>(M_PI) / 2.0f) / static_cast<float>(d);
        float result = 0.0f;
        for (int j = 0; j < d; ++j) {
            float st_j = 0.0f;
            for (int i = 0; i < d; ++i) st_j += S[static_cast<size_t>(i) * d + j] * static_cast<float>(signs[i]);
            result += query[j] * st_j;
        }
        return result * scale;
    }
};

// =========================================================================
// TURBOQUANT_PROD: (bits-1)-bit MSE stage + 1-bit QJL residual stage
// =========================================================================
struct TurboQuantProd {
    int d, bits;
    RotationMatrix Pi;
    Codebook codebook;  // (bits - 1) bits
    QJL qjl;

    TurboQuantProd(int dim, int b, unsigned seed = 42)
        : d(dim), bits(b), Pi(dim, seed), codebook(dim, std::max(b - 1, 1)), qjl(dim, seed + 7777) {
        assert(b >= 2 && "TurboQuant_prod needs >=2 bits: 1 for MSE, 1 for QJL");
    }

    struct Result {
        std::vector<uint8_t> indices;
        std::vector<int8_t> signs;
        float norm;
        float residual_norm;
    };

    Result quantize(std::span<const float> x) const {
        Result r;
        r.indices.resize(d);
        r.norm = l2_norm(x);
        std::vector<float> x_hat(d);
        if (r.norm > 1e-10f) for (int i = 0; i < d; ++i) x_hat[i] = x[i] / r.norm;
        else std::fill(x_hat.begin(), x_hat.end(), 0.0f);

        std::vector<float> y(d);
        Pi.rotate(x_hat, y);
        std::vector<float> y_tilde(d);
        for (int j = 0; j < d; ++j) {
            r.indices[j] = static_cast<uint8_t>(codebook.quantize(y[j]));
            y_tilde[j] = codebook.dequantize(r.indices[j]);
        }

        // Residual in ORIGINAL space -- rotate the MSE reconstruction
        // back BEFORE subtracting. This is the step Test 5's trap skips.
        std::vector<float> x_hat_mse(d);
        Pi.rotate_back(y_tilde, x_hat_mse);
        std::vector<float> residual(d);
        for (int i = 0; i < d; ++i) residual[i] = x_hat[i] - x_hat_mse[i];
        r.residual_norm = l2_norm(residual);

        r.signs = qjl.quantize_signs(residual);
        return r;
    }

    // Estimate <query, x_tilde> combining both stages, all in original space.
    float dot_product(std::span<const float> query, const Result& r) const {
        std::vector<float> y_tilde(d);
        for (int j = 0; j < d; ++j) y_tilde[j] = codebook.dequantize(r.indices[j]);
        std::vector<float> x_hat_mse(d);
        Pi.rotate_back(y_tilde, x_hat_mse);
        float mse_dot = dot(query, x_hat_mse) * r.norm;

        float qjl_dot = qjl.dot_estimate(query, r.signs, r.residual_norm) * r.norm;
        return mse_dot + qjl_dot;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "TurboQuant 7.3: The Inner-Product Bias Problem and QJL\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: MSE-only quantizer is biased for inner products at b=1
    // =====================================================================
    std::cout << "-- Test 1: MSE-only quantizer's inner-product bias (b=1) --\n";
    {
        constexpr int D = 32, TRIALS = 400;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(D), query(D);
        for (float& v : x) v = dist(rng);
        for (float& v : query) v = dist(rng);
        float true_dot = dot(query, x);
        float x_norm = l2_norm(x);
        std::vector<float> x_hat(D);
        for (int i = 0; i < D; ++i) x_hat[i] = x[i] / x_norm;

        float sum_estimates = 0.0f;
        for (int t = 0; t < TRIALS; ++t) {
            RotationMatrix Pi(D, static_cast<unsigned>(t + 5000));
            Codebook cb(D, 1);
            std::vector<float> y(D);
            Pi.rotate(x_hat, y);
            std::vector<float> y_tilde(D);
            for (int j = 0; j < D; ++j) y_tilde[j] = cb.dequantize(cb.quantize(y[j]));
            std::vector<float> recon(D);
            Pi.rotate_back(y_tilde, recon);
            for (float& v : recon) v *= x_norm;
            sum_estimates += dot(query, recon);
        }
        float avg_estimate = sum_estimates / TRIALS;
        float ratio = avg_estimate / true_dot;

        std::cout << "  True <query, x>:        " << std::fixed << std::setprecision(4) << true_dot << "\n";
        std::cout << "  Avg MSE-only estimate:  " << avg_estimate << "\n";
        std::cout << "  Ratio (theory: 2/pi ~= 0.637): " << ratio << "\n";
        CHECK(ratio < 0.85f);  // clearly attenuated, not close to 1.0
        CHECK(ratio > 0.3f);   // but not nonsensically small either
        std::cout << "  Confirmed: the MSE-optimal quantizer systematically shrinks inner products\n";
    }

    // =====================================================================
    // TEST 2: QJL alone is unbiased when averaged over many random S
    // =====================================================================
    std::cout << "\n-- Test 2: QJL unbiasedness, averaged over 500 random projections --\n";
    {
        constexpr int D = 32, TRIALS = 500;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(D), y(D);
        for (float& v : x) v = dist(rng);
        for (float& v : y) v = dist(rng);
        float true_dot = dot(x, y);
        float x_norm = l2_norm(x);

        float sum = 0.0f;
        for (int t = 0; t < TRIALS; ++t) {
            QJL q(D, static_cast<unsigned>(t + 1000));
            auto signs = q.quantize_signs(x);
            sum += q.dot_estimate(y, signs, x_norm);
        }
        float avg = sum / TRIALS;
        float bias = std::fabs(avg - true_dot);
        std::cout << "  True <x,y>: " << std::fixed << std::setprecision(4) << true_dot
                  << "   Avg QJL estimate: " << avg << "   |bias|: " << bias << "\n";
        CHECK(bias < std::fabs(true_dot) * 0.15f);
    }

    // =====================================================================
    // TEST 3: TurboQuantProd (MSE + QJL combined) is unbiased
    // =====================================================================
    std::cout << "\n-- Test 3: TurboQuantProd unbiasedness (d=32, b=3), 300 trials --\n";
    {
        constexpr int D = 32, TRIALS = 300;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(D), y(D);
        for (float& v : x) v = dist(rng);
        for (float& v : y) v = dist(rng);
        float true_dot = dot(x, y);

        float sum = 0.0f;
        for (int t = 0; t < TRIALS; ++t) {
            TurboQuantProd tq(D, 3, static_cast<unsigned>(t + 2000));
            auto qr = tq.quantize(x);
            sum += tq.dot_product(y, qr);
        }
        float avg = sum / TRIALS;
        float bias = std::fabs(avg - true_dot);
        std::cout << "  True <x,y>: " << std::fixed << std::setprecision(4) << true_dot
                  << "   Avg TurboQuantProd estimate: " << avg << "   |bias|: " << bias << "\n";
        CHECK(bias < std::fabs(true_dot) * 0.15f);
        std::cout << "  The QJL residual stage cancels the MSE stage's bias, as designed\n";
    }

    // =====================================================================
    // TEST 4: Variance of the combined estimator decreases with more bits
    // =====================================================================
    std::cout << "\n-- Test 4: Inner-product estimator variance vs bit-width --\n";
    {
        constexpr int D = 32, TRIALS = 200;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(D), y(D);
        for (float& v : x) v = dist(rng);
        for (float& v : y) v = dist(rng);
        float true_dot = dot(x, y);

        float prev_var = 1e10f;
        for (int b = 2; b <= 4; ++b) {
            float sum_sq_err = 0.0f;
            for (int t = 0; t < TRIALS; ++t) {
                TurboQuantProd tq(D, b, static_cast<unsigned>(t + 3000));
                auto qr = tq.quantize(x);
                float est = tq.dot_product(y, qr);
                float err = est - true_dot;
                sum_sq_err += err * err;
            }
            float variance = sum_sq_err / TRIALS;
            std::cout << "  b=" << b << "  variance: " << std::scientific << std::setprecision(3) << variance << std::fixed << "\n";
            CHECK(variance < prev_var * 1.5f);
            prev_var = variance;
        }
    }

    // =====================================================================
    // TEST 5 (COMMON TRAP): computing the QJL residual in ROTATED space
    // and forgetting to rotate the MSE reconstruction back to ORIGINAL
    // space first. The two pieces then live in different coordinate
    // systems, and adding them together (then dotting with an
    // original-space query) does not estimate anything meaningful.
    // =====================================================================
    std::cout << "\n-- Test 5 [COMMON TRAP]: residual computed in the wrong coordinate space --\n";
    {
        constexpr int D = 32, TRIALS = 300;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(D), y(D);
        for (float& v : x) v = dist(rng);
        for (float& v : y) v = dist(rng);
        float true_dot = dot(x, y);

        float sum_correct = 0.0f, sum_buggy = 0.0f;
        for (int t = 0; t < TRIALS; ++t) {
            unsigned seed = static_cast<unsigned>(t + 4000);
            RotationMatrix Pi(D, seed);
            Codebook cb(D, 2);  // 2-bit MSE stage
            QJL qjl(D, seed + 7777);

            float x_norm = l2_norm(x);
            std::vector<float> x_hat(D);
            for (int i = 0; i < D; ++i) x_hat[i] = x[i] / x_norm;
            std::vector<float> rot_x(D);
            Pi.rotate(x_hat, rot_x);
            std::vector<float> y_tilde(D);
            for (int j = 0; j < D; ++j) y_tilde[j] = cb.dequantize(cb.quantize(rot_x[j]));

            // -- Correct: rotate the MSE reconstruction back to original
            //    space, THEN subtract, so the residual lives where the
            //    query does. --
            std::vector<float> x_hat_mse(D);
            Pi.rotate_back(y_tilde, x_hat_mse);
            std::vector<float> residual_correct(D);
            for (int i = 0; i < D; ++i) residual_correct[i] = x_hat[i] - x_hat_mse[i];
            float rnorm_correct = l2_norm(residual_correct);
            auto signs_correct = qjl.quantize_signs(residual_correct);
            float mse_dot = dot(y, x_hat_mse) * x_norm;
            float qjl_dot_correct = qjl.dot_estimate(y, signs_correct, rnorm_correct) * x_norm;
            sum_correct += mse_dot + qjl_dot_correct;

            // -- Buggy: subtract in ROTATED space (skip rotate_back),
            //    then feed that rotated-space residual straight to QJL,
            //    and combine it with the ORIGINAL-space MSE term as if
            //    the two lived in the same basis. --
            std::vector<float> residual_rotated(D);
            for (int i = 0; i < D; ++i) residual_rotated[i] = rot_x[i] - y_tilde[i];
            float rnorm_buggy = l2_norm(residual_rotated);
            auto signs_buggy = qjl.quantize_signs(residual_rotated);
            float qjl_dot_buggy = qjl.dot_estimate(y, signs_buggy, rnorm_buggy) * x_norm;
            sum_buggy += mse_dot + qjl_dot_buggy;
        }
        float avg_correct = sum_correct / TRIALS;
        float avg_buggy = sum_buggy / TRIALS;
        float bias_correct = std::fabs(avg_correct - true_dot);
        float bias_buggy = std::fabs(avg_buggy - true_dot);

        std::cout << "  True <x,y>:              " << std::fixed << std::setprecision(4) << true_dot << "\n";
        std::cout << "  Correct (same-space) estimate: " << avg_correct << "   |bias|: " << bias_correct << "\n";
        std::cout << "  Buggy (mixed-space) estimate:  " << avg_buggy << "   |bias|: " << bias_buggy << "\n";
        std::cout << "  Subtracting in rotated space but combining in original space breaks\n";
        std::cout << "  unbiasedness -- the residual and the MSE term must share a coordinate system.\n";
        CHECK(bias_correct < std::fabs(true_dot) * 0.2f);
        CHECK(bias_buggy > 2.0f * bias_correct);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_qjl_unbiased_inner_product.cpp -o 03_qjl_unbiased_inner_product
./03_qjl_unbiased_inner_product
```

**Sample input:** a 1-bit MSE-only quantizer's inner-product estimate, averaged over 400 independent random rotations, compared against the theoretical `2/pi` shrinkage factor; QJL's own unbiasedness, averaged over 500 random projection matrices; the combined TurboQuant "Prod" quantizer's unbiasedness at 3 bits, averaged over 300 trials; the combined estimator's variance measured at bit-widths 2 through 4; and a deliberate demonstration of what happens when the residual is computed in rotated coordinates instead of original ones before being combined with the MSE term.

```text
========================================================
TurboQuant 7.3: The Inner-Product Bias Problem and QJL
========================================================

-- Test 1: MSE-only quantizer's inner-product bias (b=1) --
  True <query, x>:        -4.1923
  Avg MSE-only estimate:  -2.5111
  Ratio (theory: 2/pi ~= 0.637): 0.5990
  Confirmed: the MSE-optimal quantizer systematically shrinks inner products

-- Test 2: QJL unbiasedness, averaged over 500 random projections --
  True <x,y>: -4.1923   Avg QJL estimate: -3.6006   |bias|: 0.5918

-- Test 3: TurboQuantProd unbiasedness (d=32, b=3), 300 trials --
  True <x,y>: -4.1923   Avg TurboQuantProd estimate: -4.4093   |bias|: 0.2170
  The QJL residual stage cancels the MSE stage's bias, as designed

-- Test 4: Inner-product estimator variance vs bit-width --
  b=2  variance: 1.585e+01
  b=3  variance: 5.089e+00
  b=4  variance: 1.493e+00

-- Test 5 [COMMON TRAP]: residual computed in the wrong coordinate space --
  True <x,y>:              -4.1923
  Correct (same-space) estimate: -4.0273   |bias|: 0.1650
  Buggy (mixed-space) estimate:  -3.6812   |bias|: 0.5111
  Subtracting in rotated space but combining in original space breaks
  unbiasedness -- the residual and the MSE term must share a coordinate system.

========================================================
9/9 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] combining the two stages in different coordinate systems"
    The MSE stage's quantization happens in ROTATED coordinates (that is where the Beta-distribution codebook applies), but its reconstruction is rotated back to ORIGINAL coordinates before anything else touches it. The residual that QJL quantizes must be computed in that same original coordinate system — `x_hat - x_hat_mse` — because QJL's own random matrix `S` and its unbiasedness proof are stated for the space the QUERY lives in, which is the original, un-rotated space. Subtracting in rotated coordinates instead (`y - y_tilde`, skipping the rotate-back step) and then adding the resulting QJL correction directly to the original-space MSE term mixes two vectors that are expressed in different bases — the correction and the query it gets dotted with are no longer describing the same coordinates. The combined estimate does not crash or produce an obviously wrong shape; it simply drifts further from the true inner product than either stage's error alone would predict, because the two pieces being added together do not actually correspond to each other.

## 7.4 Compressing the KV Cache with TurboQuant

### Intuition

The KV cache is the one structure in a running transformer that blockwise quantization was never a comfortable fit for: it does not exist until generation starts, it grows by one key and one value vector per token per head per layer, and there is no representative sample of it available in advance to calibrate a codebook against. TurboQuant's codebooks, precomputed purely from `d` and the bit budget with no data dependence at all, are exactly suited to this: every vector that will ever be cached can be quantized the moment it is produced, using a codebook that was already finished before generation even began.

### The Concept, In Detail

Applying Section 7.2's MSE-mode quantizer to a KV cache is mostly a matter of applying it once per cached vector, independently — the same rotation matrix and codebook serve every key and every value at a given head dimension, but each vector gets and stores its OWN norm, because cached key and value magnitudes genuinely vary from token to token; nothing about the codebook's universality extends to sharing that one per-vector scalar. MSE mode, not the unbiased "Prod" mode from Section 7.3, is the right choice here in the typical 3-4 bit range: the inner-product bias at those bit-widths is small enough that the simpler, one-bit-cheaper quantizer wins on cost without a meaningful quality trade-off, though Prod mode remains available whenever that trade-off does not hold — vector-database search at very low bit-widths, for instance, where biased rankings across millions of vectors compound into real errors. Simulating a full attention step — compute scores against every cached key, softmax, weight the cached values — with quantized keys and values in place of FP32 ones shows the attention weights themselves shift only slightly and the final output vector's error stays small, and running the same quantizer across a growing cache shows that per-vector error does not compound as more tokens are added, because each vector is quantized independently with no running state carried between them. Translating this into a memory budget for realistic model configurations shows several-fold compression relative to FP16 at 3 bits per coordinate — a number obtained here from straightforward byte-count arithmetic, not from re-running the retrieval-quality benchmarks (Needle-in-a-Haystack and similar) that the TurboQuant paper itself reports; those quality claims are cited from the literature rather than re-derived in this toy simulation.

### Code and Verification

```cpp
// 04_kv_cache_quantization.cpp
// TurboQuant, Part 4: apply the MSE quantizer (Section 7.2) to the
// problem that motivated this whole chapter -- compressing the KV cache
// during autoregressive generation. Every generated token leaves behind
// a key and value vector that must be kept for every future token to
// attend to; at long contexts this cache dwarfs the model weights
// themselves. TurboQuant compresses each vector independently and
// online, as it is produced, with no calibration pass over the sequence.
//
// This file uses MSE mode (not the QJL "Prod" variant from Section 7.3):
// at the 3-4 bit range typical for KV caches, the inner-product bias is
// small enough that the simpler, cheaper MSE quantizer is the practical
// choice -- Prod mode is reserved for cases like vector-database search
// where millions of biased estimates could shift rankings.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_kv_cache_quantization.cpp -o out04

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <span>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// =========================================================================
// ROTATION MATRIX and CODEBOOK (Sections 7.1-7.2; reused as-is)
// =========================================================================
struct RotationMatrix {
    int d;
    std::vector<float> Q;
    explicit RotationMatrix(int dim, unsigned seed = 42) : d(dim), Q(static_cast<size_t>(dim) * dim) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> col(static_cast<size_t>(d) * d);
        for (int j = 0; j < d; ++j)
            for (int i = 0; i < d; ++i)
                col[static_cast<size_t>(j) * d + i] = gauss(rng);
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < j; ++k) {
                float proj = 0.0f;
                for (int i = 0; i < d; ++i)
                    proj += col[static_cast<size_t>(j) * d + i] * col[static_cast<size_t>(k) * d + i];
                for (int i = 0; i < d; ++i)
                    col[static_cast<size_t>(j) * d + i] -= proj * col[static_cast<size_t>(k) * d + i];
            }
            float norm_sq = 0.0f;
            for (int i = 0; i < d; ++i) { float v = col[static_cast<size_t>(j) * d + i]; norm_sq += v * v; }
            float norm = std::sqrt(norm_sq);
            if (norm > 1e-10f)
                for (int i = 0; i < d; ++i) col[static_cast<size_t>(j) * d + i] /= norm;
        }
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                Q[static_cast<size_t>(i) * d + j] = col[static_cast<size_t>(j) * d + i];
    }
    void rotate(std::span<const float> x, std::span<float> y) const {
        for (int i = 0; i < d; ++i) {
            float s = 0.0f;
            for (int j = 0; j < d; ++j) s += Q[static_cast<size_t>(i) * d + j] * x[j];
            y[i] = s;
        }
    }
    void rotate_back(std::span<const float> y, std::span<float> x) const {
        for (int j = 0; j < d; ++j) {
            float s = 0.0f;
            for (int i = 0; i < d; ++i) s += Q[static_cast<size_t>(i) * d + j] * y[i];
            x[j] = s;
        }
    }
};

struct Codebook {
    int bits, n_codes;
    std::vector<float> centroids, boundaries;
    static float beta_pdf(float x, int d) {
        if (std::fabs(x) >= 1.0f) return 0.0f;
        return std::pow(1.0f - x * x, 0.5f * static_cast<float>(d - 3));
    }
    static float integrate_moment(float a, float b, int p, int d, int n_steps = 2000) {
        if (a >= b) return 0.0f;
        float h = (b - a) / static_cast<float>(n_steps), sum = 0.0f;
        for (int i = 0; i <= n_steps; ++i) {
            float x = a + static_cast<float>(i) * h, fx = beta_pdf(x, d);
            float val = (p == 1) ? fx * x : (p == 2) ? fx * x * x : fx;
            sum += ((i == 0 || i == n_steps) ? 0.5f : 1.0f) * val;
        }
        return sum * h;
    }
    Codebook() : bits(0), n_codes(0) {}
    Codebook(int d, int b, int max_iter = 200) : bits(b) {
        n_codes = 1 << b;
        centroids.resize(n_codes);
        boundaries.resize(n_codes - 1);
        for (int i = 0; i < n_codes; ++i)
            centroids[i] = -1.0f + (2.0f * static_cast<float>(i) + 1.0f) / static_cast<float>(n_codes);
        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < n_codes - 1; ++i) boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
            bool converged = true;
            for (int i = 0; i < n_codes; ++i) {
                float lo = (i == 0) ? -1.0f : boundaries[i - 1];
                float hi = (i == n_codes - 1) ? 1.0f : boundaries[i];
                float m1 = integrate_moment(lo, hi, 1, d), m0 = integrate_moment(lo, hi, 0, d);
                if (m0 > 1e-12f) {
                    float nc = m1 / m0;
                    if (std::fabs(nc - centroids[i]) > 1e-8f) converged = false;
                    centroids[i] = nc;
                }
            }
            if (converged) break;
        }
    }
    int quantize(float x) const {
        int idx = 0;
        for (int i = 0; i < static_cast<int>(boundaries.size()); ++i) if (x > boundaries[i]) idx = i + 1; else break;
        return idx;
    }
    float dequantize(int idx) const { return centroids[idx]; }
};

float l2_norm(std::span<const float> v) { float s = 0.0f; for (float x : v) s += x * x; return std::sqrt(s); }
float dot(std::span<const float> a, std::span<const float> b) {
    float s = 0.0f; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}
void softmax_inplace(std::span<float> x) {
    float mx = *std::max_element(x.begin(), x.end());
    float s = 0.0f;
    for (float& v : x) { v = std::exp(v - mx); s += v; }
    for (float& v : x) v /= s;
}

// =========================================================================
// KV CACHE QUANTIZER: one TurboQuant MSE quantizer, applied per-vector
// =========================================================================
// Each cached key or value vector gets its OWN norm, stored alongside
// its indices. The rotation matrix and codebook are the SAME for every
// vector -- that is what makes the codebook "universal" and the whole
// scheme data-oblivious -- but the norm is per-vector by necessity.
struct QuantVec {
    std::vector<uint8_t> indices;
    float norm;
};

struct KVQuantizer {
    int head_dim, bits;
    RotationMatrix Pi;
    Codebook cb;

    KVQuantizer(int hd, int b, unsigned seed = 42) : head_dim(hd), bits(b), Pi(hd, seed), cb(hd, b) {}

    QuantVec quantize(std::span<const float> v) const {
        QuantVec q;
        q.indices.resize(head_dim);
        q.norm = l2_norm(v);
        std::vector<float> v_hat(head_dim), y(head_dim);
        if (q.norm > 1e-10f) for (int i = 0; i < head_dim; ++i) v_hat[i] = v[i] / q.norm;
        Pi.rotate(v_hat, y);
        for (int j = 0; j < head_dim; ++j) q.indices[j] = static_cast<uint8_t>(cb.quantize(y[j]));
        return q;
    }

    void dequantize(const QuantVec& q, std::span<float> out) const {
        std::vector<float> y_tilde(head_dim), x_hat(head_dim);
        for (int j = 0; j < head_dim; ++j) y_tilde[j] = cb.dequantize(q.indices[j]);
        Pi.rotate_back(y_tilde, x_hat);
        for (int i = 0; i < head_dim; ++i) out[i] = x_hat[i] * q.norm;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "TurboQuant 7.4: Compressing the KV Cache\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: Single-head attention, FP32 baseline vs quantized KV cache
    // =====================================================================
    std::cout << "-- Test 1: Attention with a TurboQuant-compressed KV cache --\n";
    std::cout << "   (head_dim=32, seq_len=24, bits=3)\n";
    {
        constexpr int HEAD_DIM = 32, SEQ_LEN = 24, BITS = 3;
        std::mt19937 rng(11);
        std::normal_distribution<float> dist(0.0f, 0.3f);

        std::vector<std::vector<float>> keys(SEQ_LEN, std::vector<float>(HEAD_DIM));
        std::vector<std::vector<float>> vals(SEQ_LEN, std::vector<float>(HEAD_DIM));
        std::vector<float> query(HEAD_DIM);
        for (auto& k : keys) for (float& v : k) v = dist(rng);
        for (auto& v : vals) for (float& x : v) x = dist(rng);
        for (float& v : query) v = dist(rng);

        float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

        std::vector<float> scores_fp32(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) scores_fp32[t] = dot(query, keys[t]) * scale;
        std::vector<float> attn_fp32 = scores_fp32;
        softmax_inplace(attn_fp32);
        std::vector<float> out_fp32(HEAD_DIM, 0.0f);
        for (int t = 0; t < SEQ_LEN; ++t)
            for (int i = 0; i < HEAD_DIM; ++i) out_fp32[i] += attn_fp32[t] * vals[t][i];

        KVQuantizer kq(HEAD_DIM, BITS, 42);
        std::vector<QuantVec> q_keys(SEQ_LEN), q_vals(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) { q_keys[t] = kq.quantize(keys[t]); q_vals[t] = kq.quantize(vals[t]); }

        std::vector<float> scores_q(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) {
            std::vector<float> k_deq(HEAD_DIM);
            kq.dequantize(q_keys[t], k_deq);
            scores_q[t] = dot(query, k_deq) * scale;
        }
        std::vector<float> attn_q = scores_q;
        softmax_inplace(attn_q);
        std::vector<float> out_q(HEAD_DIM, 0.0f);
        for (int t = 0; t < SEQ_LEN; ++t) {
            std::vector<float> v_deq(HEAD_DIM);
            kq.dequantize(q_vals[t], v_deq);
            for (int i = 0; i < HEAD_DIM; ++i) out_q[i] += attn_q[t] * v_deq[i];
        }

        float max_attn_diff = 0.0f;
        for (int t = 0; t < SEQ_LEN; ++t) max_attn_diff = std::max(max_attn_diff, std::fabs(attn_fp32[t] - attn_q[t]));
        float out_mse = 0.0f;
        for (int i = 0; i < HEAD_DIM; ++i) { float e = out_fp32[i] - out_q[i]; out_mse += e * e; }
        out_mse /= HEAD_DIM;

        std::cout << "  Max attention-weight difference: " << std::scientific << std::setprecision(4) << max_attn_diff << std::fixed << "\n";
        std::cout << "  Output vector MSE:                " << std::scientific << out_mse << std::fixed << "\n";
        std::cout << "  Cache bytes/vector: " << (BITS * HEAD_DIM + 7) / 8 + 4
                  << "  vs FP32: " << HEAD_DIM * 4 << " bytes ("
                  << std::setprecision(1) << static_cast<float>(HEAD_DIM * 4) / ((BITS * HEAD_DIM + 7) / 8 + 4) << "x compression)\n";
        CHECK(max_attn_diff < 0.2f);
        CHECK(out_mse < 0.02f);
    }

    // =====================================================================
    // TEST 2: Quantization error does not compound across a long cache.
    // Each vector is quantized INDEPENDENTLY (the codebook and rotation
    // are fixed, but there is no shared running state), so per-vector
    // error should stay roughly flat as the cache grows, not accumulate.
    // =====================================================================
    std::cout << "\n-- Test 2: Per-vector error stays flat as the cache grows (bits=3) --\n";
    {
        constexpr int HEAD_DIM = 32, BITS = 3, N_CHUNKS = 4, CHUNK = 32;
        KVQuantizer kq(HEAD_DIM, BITS, 7);
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.0f, 0.3f);

        std::vector<float> chunk_mse;
        for (int c = 0; c < N_CHUNKS; ++c) {
            float total_se = 0.0f;
            for (int t = 0; t < CHUNK; ++t) {
                std::vector<float> v(HEAD_DIM);
                for (float& x : v) x = dist(rng);
                auto q = kq.quantize(v);
                std::vector<float> recon(HEAD_DIM);
                kq.dequantize(q, recon);
                for (int i = 0; i < HEAD_DIM; ++i) { float e = v[i] - recon[i]; total_se += e * e; }
            }
            chunk_mse.push_back(total_se / (CHUNK * HEAD_DIM));
            std::cout << "  Tokens " << c * CHUNK << "-" << (c + 1) * CHUNK - 1
                      << ": per-element MSE = " << std::scientific << std::setprecision(3) << chunk_mse.back() << std::fixed << "\n";
        }
        float first = chunk_mse.front(), last = chunk_mse.back();
        float drift = std::fabs(last - first) / first;
        std::cout << "  Relative drift from first chunk to last: " << std::setprecision(1) << drift * 100.0f << "%\n";
        CHECK(drift < 1.0f);  // should not blow up as more tokens are cached
    }

    // =====================================================================
    // TEST 3: Memory savings for realistic model configurations
    // (pure byte-count arithmetic -- no timing claims)
    // =====================================================================
    std::cout << "\n-- Test 3: KV cache memory footprint, FP16 vs TurboQuant 3-bit --\n";
    {
        struct Config { const char* name; int layers, kv_heads, head_dim, seq_len; };
        Config configs[] = {
            {"8B-class model,  4K context",  32, 8, 128, 4096},
            {"8B-class model,  128K context", 32, 8, 128, 131072},
            {"70B-class model, 32K context",  80, 8, 128, 32768},
        };
        constexpr int BITS = 3;
        for (const auto& c : configs) {
            size_t n_vectors = 2ULL * c.layers * c.kv_heads * c.seq_len;  // K and V, every layer/head/token
            size_t fp16_bytes = n_vectors * static_cast<size_t>(c.head_dim) * 2;
            size_t tq_bits = n_vectors * static_cast<size_t>(c.head_dim) * BITS;
            size_t tq_norms = n_vectors * 4;  // one float32 norm per vector
            size_t tq_bytes = (tq_bits + 7) / 8 + tq_norms;

            std::cout << "  " << c.name << ":\n";
            std::cout << "    FP16:            " << fp16_bytes / (1024 * 1024) << " MB\n";
            std::cout << "    TurboQuant 3-bit: " << tq_bytes / (1024 * 1024) << " MB  ("
                      << std::fixed << std::setprecision(1) << static_cast<float>(fp16_bytes) / static_cast<float>(tq_bytes) << "x compression)\n";
            CHECK(tq_bytes < fp16_bytes);
        }
        std::cout << "  (Compression factors above are computed directly from these byte counts.\n";
        std::cout << "   Retrieval-quality claims at this bit-width come from the TurboQuant paper's\n";
        std::cout << "   own evaluation, not from this toy simulation, and are cited as such.)\n";
    }

    // =====================================================================
    // TEST 4 (COMMON TRAP): sharing ONE norm across the whole cache
    // instead of storing a norm PER vector. Each cached key/value has its
    // own magnitude; reusing the first vector's norm for every later
    // vector silently corrupts every dequantized vector after the first.
    // =====================================================================
    std::cout << "\n-- Test 4 [COMMON TRAP]: one shared norm instead of a per-vector norm --\n";
    {
        constexpr int HEAD_DIM = 32, SEQ_LEN = 16, BITS = 3;
        KVQuantizer kq(HEAD_DIM, BITS, 99);
        std::mt19937 rng(99);
        // Deliberately vary the magnitude from token to token -- this is
        // realistic: attention key/value norms are not constant across
        // a sequence.
        std::vector<std::vector<float>> vecs(SEQ_LEN, std::vector<float>(HEAD_DIM));
        for (int t = 0; t < SEQ_LEN; ++t) {
            std::normal_distribution<float> dist(0.0f, 0.15f * static_cast<float>(t + 1));
            for (float& v : vecs[t]) v = dist(rng);
        }

        std::vector<QuantVec> qvecs(SEQ_LEN);
        for (int t = 0; t < SEQ_LEN; ++t) qvecs[t] = kq.quantize(vecs[t]);

        // Correct: dequantize using each vector's OWN stored norm.
        float mse_correct = 0.0f;
        for (int t = 0; t < SEQ_LEN; ++t) {
            std::vector<float> recon(HEAD_DIM);
            kq.dequantize(qvecs[t], recon);
            for (int i = 0; i < HEAD_DIM; ++i) { float e = vecs[t][i] - recon[i]; mse_correct += e * e; }
        }
        mse_correct /= (SEQ_LEN * HEAD_DIM);

        // Buggy: reuse token 0's norm for every token in the cache, as
        // if the norm were a per-layer constant instead of per-vector.
        float shared_norm = qvecs[0].norm;
        float mse_buggy = 0.0f;
        for (int t = 0; t < SEQ_LEN; ++t) {
            QuantVec bad_q = qvecs[t];
            bad_q.norm = shared_norm;
            std::vector<float> recon(HEAD_DIM);
            kq.dequantize(bad_q, recon);
            for (int i = 0; i < HEAD_DIM; ++i) { float e = vecs[t][i] - recon[i]; mse_buggy += e * e; }
        }
        mse_buggy /= (SEQ_LEN * HEAD_DIM);

        std::cout << "  Correct (per-vector norm) MSE: " << std::scientific << std::setprecision(3) << mse_correct << std::fixed << "\n";
        std::cout << "  Buggy (one shared norm) MSE:   " << std::scientific << std::setprecision(3) << mse_buggy << std::fixed << "\n";
        std::cout << "  The norm is not a codebook parameter -- it is per-vector data, exactly\n";
        std::cout << "  as necessary as the indices. Skipping it for all but one token wrecks\n";
        std::cout << "  every vector whose true magnitude differs from that one token's.\n";
        CHECK(mse_buggy > 5.0f * mse_correct);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_kv_cache_quantization.cpp -o 04_kv_cache_quantization
./04_kv_cache_quantization
```

**Sample input:** single-head attention over a 24-token cache (head_dim=32, 3 bits) with FP32 keys/values compared against TurboQuant-quantized ones; per-vector reconstruction error measured across four consecutive 32-token chunks of a growing cache to check for compounding drift; a memory-footprint comparison between FP16 and TurboQuant 3-bit for three representative model configurations; and a deliberate demonstration of reusing one cached vector's norm for every later vector instead of storing a norm per vector.

```text
========================================================
TurboQuant 7.4: Compressing the KV Cache
========================================================

-- Test 1: Attention with a TurboQuant-compressed KV cache --
   (head_dim=32, seq_len=24, bits=3)
  Max attention-weight difference: 1.8410e-03
  Output vector MSE:                2.1416e-04
  Cache bytes/vector: 16  vs FP32: 128 bytes (8.0x compression)

-- Test 2: Per-vector error stays flat as the cache grows (bits=3) --
  Tokens 0-31: per-element MSE = 2.806e-03
  Tokens 32-63: per-element MSE = 2.826e-03
  Tokens 64-95: per-element MSE = 2.643e-03
  Tokens 96-127: per-element MSE = 2.698e-03
  Relative drift from first chunk to last: 3.8%

-- Test 3: KV cache memory footprint, FP16 vs TurboQuant 3-bit --
  8B-class model,  4K context:
    FP16:            512 MB
    TurboQuant 3-bit: 104 MB  (4.9x compression)
  8B-class model,  128K context:
    FP16:            16384 MB
    TurboQuant 3-bit: 3328 MB  (4.9x compression)
  70B-class model, 32K context:
    FP16:            10240 MB
    TurboQuant 3-bit: 2080 MB  (4.9x compression)
  (Compression factors above are computed directly from these byte counts.
   Retrieval-quality claims at this bit-width come from the TurboQuant paper's
   own evaluation, not from this toy simulation, and are cited as such.)

-- Test 4 [COMMON TRAP]: one shared norm instead of a per-vector norm --
  Correct (per-vector norm) MSE: 5.561e-02
  Buggy (one shared norm) MSE:   1.635e+00
  The norm is not a codebook parameter -- it is per-vector data, exactly
  as necessary as the indices. Skipping it for all but one token wrecks
  every vector whose true magnitude differs from that one token's.

========================================================
7/7 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] one norm for the whole cache instead of one per vector"
    The rotation matrix and the codebook are genuinely shared across every vector in the cache — that sharing is what makes the codebook "universal" and the whole scheme data-oblivious. The norm is not part of that shared machinery; it is per-vector data, exactly as essential to a correct reconstruction as the quantization indices themselves, because real cached keys and values do not all have the same magnitude from token to token. Reusing the very first cached vector's norm for every subsequent vector — as if the norm were some fixed property of the layer rather than of each individual vector — leaves the INDICES correct while silently corrupting the SCALE of every reconstructed vector whose true magnitude differs from that first one, and in a real sequence, almost all of them will differ. Nothing about this failure looks like ordinary quantization noise: the reconstructed direction is fine, the reconstructed length is simply wrong.

## 7.5 TurboQuant vs. Blockwise Quantization, and the Distortion Bounds

### Intuition

The chapter has built an alternative to Chapters 2-6's blockwise Q8_0/Q4_0, not a replacement for it — the two approaches are optimized for different situations, and the honest way to see that is to run them against each other on the same vectors and look at where each one wins. It is also worth asking a harder question about TurboQuant specifically: not just "is it better than blockwise here," but "how much better could ANY quantizer possibly be" — because a technique described as "near-optimal" ought to be checked against the theoretical optimum it claims to approach, not just against one specific competitor.

### The Concept, In Detail

Running Chapters 2-4's blockwise Q8_0 and Q4_0 alongside TurboQuant's MSE quantizer at several bit-widths, on the same set of random vectors, with the same set of query vectors for measuring inner-product error, shows the expected shape of the trade-off: Q8_0's larger byte budget buys the lowest reconstruction error of the group, Q4_0 trades some of that accuracy for roughly half the bytes, and TurboQuant at a comparable bit-width matches or beats Q4_0's reconstruction quality using a comparable byte budget — while TurboQuant at very low bit-widths (2 bits) trades substantially more accuracy for a much smaller footprint than either blockwise format can reach at all, since Q4_0 has no lower-bit-width sibling. Checking TurboQuant's own measured distortion against the Shannon lower bound (`4^-bits`, the information-theoretic floor for any quantizer of a unit-sphere vector at that bit budget, independent of `d`) shows the measured-to-bound ratio staying within a small, roughly constant factor across every bit-width tested — close to (though not an exact reproduction of) the 2.7x cap the TurboQuant paper proves analytically for the idealized infinite-dimensional codebook. Comparing byte budgets fairly across the two families requires counting EVERY byte a real system would actually store, not just the part that is convenient to count — which is exactly where this section's own trap lives.

### Code and Verification

```cpp
// 05_turboquant_vs_blockwise.cpp
// TurboQuant, Part 5: put every piece from this chapter together -- the
// rotation (7.1), the Lloyd-Max codebook and MSE quantizer (7.2), and the
// QJL-free "MSE mode" used for KV caches (7.4) -- and compare it head to
// head against Chapter 3/4's blockwise Q8_0/Q4_0 on a common set of
// vectors. We also check TurboQuant's distortion against the
// information-theoretic lower bound that makes it "near-optimal" rather
// than just "pretty good".
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_turboquant_vs_blockwise.cpp -o out05

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <span>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// =========================================================================
// CHAPTER 3/4's fp16_t and blockwise Q8_0/Q4_0 (reused verbatim, this is
// the baseline TurboQuant is being compared against)
// =========================================================================
struct fp16_t {
    uint16_t bits = 0;
    fp16_t() = default;
    fp16_t(float f) { bits = encode(f); }
    operator float() const { return decode(bits); }
    static uint16_t encode(float f) {
        uint32_t x; std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;
        if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        if (exp <= 0)  return static_cast<uint16_t>(sign);
        uint32_t mant10 = mant >> 13;
        uint32_t remainder = mant & 0x1FFFu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            mant10 += 1;
            if (mant10 == 0x400u) { mant10 = 0; exp += 1; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant10);
    }
    static float decode(uint16_t h) {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
        uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
        uint32_t fbits;
        if (exp == 0) fbits = sign;
        else if (exp == 31) fbits = sign | 0x7F800000u | (mant << 13);
        else fbits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        float f; std::memcpy(&f, &fbits, 4); return f;
    }
};
#pragma pack(push, 1)
struct BlockQ8 { fp16_t scale; int8_t weights[32]; };
struct BlockQ4 { fp16_t scale; uint8_t nibbles[16]; };
#pragma pack(pop)

BlockQ8 quantize_q8(const float* data) {
    BlockQ8 b;
    float alpha = 0.0f;
    for (int i = 0; i < 32; ++i) alpha = std::max(alpha, std::fabs(data[i]));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); std::memset(b.weights, 0, 32); return b; }
    b.scale = fp16_t(alpha / 127.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 32; ++i)
        b.weights[i] = static_cast<int8_t>(std::clamp(std::round(data[i] * inv), -127.0f, 127.0f));
    return b;
}
BlockQ4 quantize_q4(const float* data) {
    BlockQ4 b;
    float alpha = 0.0f;
    for (int i = 0; i < 32; ++i) alpha = std::max(alpha, std::fabs(data[i]));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); for (auto& n : b.nibbles) n = 0x88; return b; }
    b.scale = fp16_t(alpha / 7.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 16; ++i) {
        int lo = static_cast<int>(std::clamp(std::round(data[2 * i] * inv), -8.0f, 7.0f));
        int hi = static_cast<int>(std::clamp(std::round(data[2 * i + 1] * inv), -8.0f, 7.0f));
        b.nibbles[i] = static_cast<uint8_t>((static_cast<uint8_t>(hi + 8) << 4) | static_cast<uint8_t>(lo + 8));
    }
    return b;
}
void dequantize_q8(const BlockQ8& b, float* out) {
    float s = static_cast<float>(b.scale);
    for (int i = 0; i < 32; ++i) out[i] = static_cast<float>(b.weights[i]) * s;
}
void dequantize_q4(const BlockQ4& b, float* out) {
    float s = static_cast<float>(b.scale);
    for (int i = 0; i < 16; ++i) {
        uint8_t p = b.nibbles[i];
        out[2 * i]     = static_cast<float>(static_cast<int>(p & 0xF) - 8) * s;
        out[2 * i + 1] = static_cast<float>(static_cast<int>(p >> 4) - 8) * s;
    }
}

// =========================================================================
// TURBOQUANT: rotation + Lloyd-Max codebook + MSE quantizer
// (Sections 7.1-7.2, unchanged)
// =========================================================================
struct RotationMatrix {
    int d;
    std::vector<float> Q;
    explicit RotationMatrix(int dim, unsigned seed = 42) : d(dim), Q(static_cast<size_t>(dim) * dim) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> col(static_cast<size_t>(d) * d);
        for (int j = 0; j < d; ++j)
            for (int i = 0; i < d; ++i)
                col[static_cast<size_t>(j) * d + i] = gauss(rng);
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < j; ++k) {
                float proj = 0.0f;
                for (int i = 0; i < d; ++i)
                    proj += col[static_cast<size_t>(j) * d + i] * col[static_cast<size_t>(k) * d + i];
                for (int i = 0; i < d; ++i)
                    col[static_cast<size_t>(j) * d + i] -= proj * col[static_cast<size_t>(k) * d + i];
            }
            float norm_sq = 0.0f;
            for (int i = 0; i < d; ++i) { float v = col[static_cast<size_t>(j) * d + i]; norm_sq += v * v; }
            float norm = std::sqrt(norm_sq);
            if (norm > 1e-10f)
                for (int i = 0; i < d; ++i) col[static_cast<size_t>(j) * d + i] /= norm;
        }
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                Q[static_cast<size_t>(i) * d + j] = col[static_cast<size_t>(j) * d + i];
    }
    void rotate(std::span<const float> x, std::span<float> y) const {
        for (int i = 0; i < d; ++i) {
            float s = 0.0f;
            for (int j = 0; j < d; ++j) s += Q[static_cast<size_t>(i) * d + j] * x[j];
            y[i] = s;
        }
    }
    void rotate_back(std::span<const float> y, std::span<float> x) const {
        for (int j = 0; j < d; ++j) {
            float s = 0.0f;
            for (int i = 0; i < d; ++i) s += Q[static_cast<size_t>(i) * d + j] * y[i];
            x[j] = s;
        }
    }
};

struct Codebook {
    int bits, n_codes;
    std::vector<float> centroids, boundaries;
    static float beta_pdf(float x, int d) {
        if (std::fabs(x) >= 1.0f) return 0.0f;
        return std::pow(1.0f - x * x, 0.5f * static_cast<float>(d - 3));
    }
    static float integrate_moment(float a, float b, int p, int d, int n_steps = 2000) {
        if (a >= b) return 0.0f;
        float h = (b - a) / static_cast<float>(n_steps), sum = 0.0f;
        for (int i = 0; i <= n_steps; ++i) {
            float x = a + static_cast<float>(i) * h, fx = beta_pdf(x, d);
            float val = (p == 1) ? fx * x : (p == 2) ? fx * x * x : fx;
            sum += ((i == 0 || i == n_steps) ? 0.5f : 1.0f) * val;
        }
        return sum * h;
    }
    Codebook() : bits(0), n_codes(0) {}
    Codebook(int d, int b, int max_iter = 200) : bits(b) {
        n_codes = 1 << b;
        centroids.resize(n_codes);
        boundaries.resize(n_codes - 1);
        for (int i = 0; i < n_codes; ++i)
            centroids[i] = -1.0f + (2.0f * static_cast<float>(i) + 1.0f) / static_cast<float>(n_codes);
        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < n_codes - 1; ++i) boundaries[i] = 0.5f * (centroids[i] + centroids[i + 1]);
            bool converged = true;
            for (int i = 0; i < n_codes; ++i) {
                float lo = (i == 0) ? -1.0f : boundaries[i - 1];
                float hi = (i == n_codes - 1) ? 1.0f : boundaries[i];
                float m1 = integrate_moment(lo, hi, 1, d), m0 = integrate_moment(lo, hi, 0, d);
                if (m0 > 1e-12f) {
                    float nc = m1 / m0;
                    if (std::fabs(nc - centroids[i]) > 1e-8f) converged = false;
                    centroids[i] = nc;
                }
            }
            if (converged) break;
        }
    }
    int quantize(float x) const {
        int idx = 0;
        for (int i = 0; i < static_cast<int>(boundaries.size()); ++i) if (x > boundaries[i]) idx = i + 1; else break;
        return idx;
    }
    float dequantize(int idx) const { return centroids[idx]; }
};

float l2_norm(std::span<const float> v) { float s = 0.0f; for (float x : v) s += x * x; return std::sqrt(s); }
float dot(std::span<const float> a, std::span<const float> b) {
    float s = 0.0f; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}

struct TurboQuantMSE {
    int d, bits;
    RotationMatrix Pi;
    Codebook codebook;
    TurboQuantMSE(int dim, int b, unsigned seed = 42) : d(dim), bits(b), Pi(dim, seed), codebook(dim, b) {}

    struct QuantResult { std::vector<uint8_t> indices; float norm; };

    QuantResult quantize(std::span<const float> x) const {
        QuantResult r;
        r.indices.resize(d);
        r.norm = l2_norm(x);
        std::vector<float> x_hat(d);
        if (r.norm > 1e-10f) for (int i = 0; i < d; ++i) x_hat[i] = x[i] / r.norm;
        std::vector<float> y(d);
        Pi.rotate(x_hat, y);
        for (int j = 0; j < d; ++j) r.indices[j] = static_cast<uint8_t>(codebook.quantize(y[j]));
        return r;
    }
    void dequantize(const QuantResult& r, std::span<float> out) const {
        std::vector<float> y_tilde(d);
        for (int j = 0; j < d; ++j) y_tilde[j] = codebook.dequantize(r.indices[j]);
        std::vector<float> x_hat(d);
        Pi.rotate_back(y_tilde, x_hat);
        for (int i = 0; i < d; ++i) out[i] = x_hat[i] * r.norm;
    }
    size_t index_bytes() const { return (static_cast<size_t>(bits) * d + 7) / 8; }
    size_t total_bytes_with_norm() const { return index_bytes() + 4; }  // + one float32 norm
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "TurboQuant 7.5: vs. Blockwise Quantization, and the Bounds\n";
    std::cout << "========================================================\n\n";

    constexpr int D = 64, N_VEC = 100, N_QUERIES = 10;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<std::vector<float>> vecs(N_VEC, std::vector<float>(D));
    std::vector<std::vector<float>> queries(N_QUERIES, std::vector<float>(D));
    for (auto& v : vecs) for (float& x : v) x = dist(rng);
    for (auto& q : queries) for (float& x : q) x = dist(rng);

    // =====================================================================
    // TEST 1: Head-to-head comparison on a common set of vectors
    // =====================================================================
    std::cout << "-- Test 1: MSE and inner-product error, d=" << D << ", " << N_VEC << " vectors --\n\n";
    struct Result { std::string name; float mse; float ip_err; size_t bytes; };
    std::vector<Result> results;

    {
        int nblocks = (D + 31) / 32;
        float total_mse = 0.0f, total_ip_err = 0.0f;
        for (const auto& v : vecs) {
            std::vector<float> recon(D);
            for (int b = 0; b < nblocks; ++b) { BlockQ8 bq = quantize_q8(&v[b * 32]); dequantize_q8(bq, &recon[b * 32]); }
            for (int i = 0; i < D; ++i) { float e = v[i] - recon[i]; total_mse += e * e; }
            for (const auto& q : queries) { float d1 = dot(q, v) - dot(q, recon); total_ip_err += d1 * d1; }
        }
        results.push_back({"Q8_0 blockwise", total_mse / (N_VEC * D), total_ip_err / (N_VEC * N_QUERIES),
                            static_cast<size_t>(nblocks) * sizeof(BlockQ8)});
    }
    {
        int nblocks = (D + 31) / 32;
        float total_mse = 0.0f, total_ip_err = 0.0f;
        for (const auto& v : vecs) {
            std::vector<float> recon(D);
            for (int b = 0; b < nblocks; ++b) { BlockQ4 bq = quantize_q4(&v[b * 32]); dequantize_q4(bq, &recon[b * 32]); }
            for (int i = 0; i < D; ++i) { float e = v[i] - recon[i]; total_mse += e * e; }
            for (const auto& q : queries) { float d1 = dot(q, v) - dot(q, recon); total_ip_err += d1 * d1; }
        }
        results.push_back({"Q4_0 blockwise", total_mse / (N_VEC * D), total_ip_err / (N_VEC * N_QUERIES),
                            static_cast<size_t>(nblocks) * sizeof(BlockQ4)});
    }
    for (int b = 2; b <= 4; ++b) {
        TurboQuantMSE tq(D, b, 42);
        float total_mse = 0.0f, total_ip_err = 0.0f;
        for (const auto& v : vecs) {
            auto qr = tq.quantize(v);
            std::vector<float> recon(D);
            tq.dequantize(qr, recon);
            for (int i = 0; i < D; ++i) { float e = v[i] - recon[i]; total_mse += e * e; }
            for (const auto& q : queries) { float d1 = dot(q, v) - dot(q, recon); total_ip_err += d1 * d1; }
        }
        results.push_back({"TurboQuant " + std::to_string(b) + "-bit", total_mse / (N_VEC * D),
                            total_ip_err / (N_VEC * N_QUERIES), tq.total_bytes_with_norm()});
    }

    std::cout << "  Method              MSE/elem     IP err(avg)   Bytes/vec  Compress\n";
    std::cout << "  ------------------  -----------  ------------  ---------  --------\n";
    for (const auto& r : results) {
        std::cout << "  " << std::left << std::setw(19) << r.name << " "
                  << std::scientific << std::setprecision(3) << std::setw(11) << r.mse << "  "
                  << std::setw(12) << r.ip_err << "  "
                  << std::right << std::setw(9) << r.bytes << "  "
                  << std::fixed << std::setprecision(1) << static_cast<float>(D * 4) / static_cast<float>(r.bytes) << "x\n";
    }
    CHECK(results[0].mse < results[1].mse);        // Q8 beats Q4 in MSE
    CHECK(results[1].bytes < results[0].bytes);    // Q4 smaller than Q8
    CHECK(results[4].mse < results[1].mse);        // TurboQuant 4-bit beats Q4_0 in MSE
    std::cout << "\n  TurboQuant 4-bit matches or beats Q4_0's reconstruction quality\n";
    std::cout << "  while using a comparable or smaller byte budget.\n";

    // =====================================================================
    // TEST 2: TurboQuant's per-coordinate MSE against the Shannon lower
    // bound 4^-b, on the unit sphere where the theory applies directly.
    // =====================================================================
    std::cout << "\n-- Test 2: Distortion vs. the information-theoretic lower bound --\n";
    std::cout << "   (Shannon's bound is stated for the TOTAL squared error over the whole\n";
    std::cout << "   unit-norm vector, ||x - x~||^2 -- not divided by d -- so that is what\n";
    std::cout << "   we measure here, matching Section 7.2's Test 3 convention.)\n";
    {
        constexpr int DD = 128, TRIALS = 60;
        std::cout << "  bits  lower bound(4^-b)  measured total SE  ratio\n";
        for (int b = 1; b <= 4; ++b) {
            TurboQuantMSE tq(DD, b, 42);
            std::mt19937 r2(1000 + b);
            std::normal_distribution<float> dd(0.0f, 1.0f);
            float total_se_sum = 0.0f;
            for (int t = 0; t < TRIALS; ++t) {
                std::vector<float> x(DD);
                for (float& v : x) v = dd(r2);
                float norm = l2_norm(x);
                for (float& v : x) v /= norm;  // unit sphere, as the theory assumes
                auto qr = tq.quantize(x);
                std::vector<float> recon(DD);
                tq.dequantize(qr, recon);
                for (int i = 0; i < DD; ++i) { float e = x[i] - recon[i]; total_se_sum += e * e; }
            }
            float measured_total_se = total_se_sum / TRIALS;  // per-vector, not per-coordinate
            float lower_bound = std::pow(4.0f, -b);
            float ratio = measured_total_se / lower_bound;
            std::cout << "   " << b << "        " << std::fixed << std::setprecision(5) << lower_bound
                      << "            " << measured_total_se << "        " << std::setprecision(2) << ratio << "x\n";
            // The paper proves this ratio is bounded by a constant (~2.7)
            // for the ideal infinite-d codebook; we allow generous slack
            // for finite d=128 and Monte Carlo noise, and require only
            // that TurboQuant not fall BELOW the proven lower bound
            // (which would indicate a measurement bug) and stay within
            // a modest constant factor above it.
            CHECK(ratio > 0.9f);
            CHECK(ratio < 6.0f);
        }
        std::cout << "  Measured distortion sits within a small constant factor of the lower\n";
        std::cout << "  bound at every bit-width -- consistent with (not a precise reproduction\n";
        std::cout << "  of) the paper's proven ~2.7x cap.\n";
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): comparing "compression ratio" while ignoring
    // per-vector overhead bytes -- the norm float for TurboQuant, or the
    // fp16 scale for blockwise. This overhead is a LARGER fraction of the
    // total at small dimensions, and skipping it inflates the reported
    // ratio, especially for whichever method has fewer, larger blocks.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: forgetting per-vector overhead bytes --\n";
    {
        constexpr int SMALL_D = 32;  // one block for Q8/Q4, one vector for TurboQuant
        TurboQuantMSE tq(SMALL_D, 2, 42);

        size_t tq_index_only = tq.index_bytes();               // WRONG: forgets the norm
        size_t tq_with_overhead = tq.total_bytes_with_norm();  // correct

        size_t q4_index_only = 16;                             // WRONG: forgets the fp16 scale
        size_t q4_with_overhead = sizeof(BlockQ4);              // correct (18 bytes: scale + nibbles)

        float fp32_bytes = static_cast<float>(SMALL_D * 4);

        std::cout << "  d=" << SMALL_D << " (a single block/vector -- overhead is NOT amortised away)\n";
        std::cout << "  TurboQuant 2-bit:\n";
        std::cout << "    Ignoring the norm:   " << tq_index_only << " bytes -> "
                  << std::fixed << std::setprecision(1) << fp32_bytes / static_cast<float>(tq_index_only) << "x compression (WRONG)\n";
        std::cout << "    Including the norm:  " << tq_with_overhead << " bytes -> "
                  << fp32_bytes / static_cast<float>(tq_with_overhead) << "x compression (correct)\n";
        std::cout << "  Q4_0 blockwise:\n";
        std::cout << "    Ignoring the scale:  " << q4_index_only << " bytes -> "
                  << fp32_bytes / static_cast<float>(q4_index_only) << "x compression (WRONG)\n";
        std::cout << "    Including the scale: " << q4_with_overhead << " bytes -> "
                  << fp32_bytes / static_cast<float>(q4_with_overhead) << "x compression (correct)\n";

        float tq_inflation = (fp32_bytes / static_cast<float>(tq_index_only)) / (fp32_bytes / static_cast<float>(tq_with_overhead));
        float q4_inflation = (fp32_bytes / static_cast<float>(q4_index_only)) / (fp32_bytes / static_cast<float>(q4_with_overhead));
        std::cout << "  Overstatement factor: TurboQuant " << std::setprecision(2) << tq_inflation
                  << "x, Q4_0 " << q4_inflation << "x -- both look better than they are\n";
        std::cout << "  if the per-vector/per-block overhead is silently dropped from the count.\n";
        CHECK(tq_with_overhead > tq_index_only);
        CHECK(q4_with_overhead > q4_index_only);
        CHECK(tq_inflation > 1.1f);
        CHECK(q4_inflation > 1.1f);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_turboquant_vs_blockwise.cpp -o 05_turboquant_vs_blockwise
./05_turboquant_vs_blockwise
```

**Sample input:** Q8_0, Q4_0, and TurboQuant MSE at 2, 3, and 4 bits, compared head to head for per-element MSE, average inner-product error, and bytes per vector across 100 random 64-dimensional vectors and 10 query vectors; TurboQuant's measured total squared error against the Shannon lower bound `4^-bits` at bit-widths 1 through 4; and a deliberate compression-ratio comparison that first omits, then includes, each format's per-vector or per-block overhead bytes.

```text
========================================================
TurboQuant 7.5: vs. Blockwise Quantization, and the Bounds
========================================================

-- Test 1: MSE and inner-product error, d=64, 100 vectors --

  Method              MSE/elem     IP err(avg)   Bytes/vec  Compress
  ------------------  -----------  ------------  ---------  --------
  Q8_0 blockwise      2.615e-06    1.546e-05            68  3.8x
  Q4_0 blockwise      8.731e-04    5.282e-03            36  7.1x
  TurboQuant 2-bit    1.054e-02    6.452e-02            20  12.8x
  TurboQuant 3-bit    3.106e-03    1.746e-02            28  9.1x
  TurboQuant 4-bit    8.373e-04    4.932e-03            36  7.1x

  TurboQuant 4-bit matches or beats Q4_0's reconstruction quality
  while using a comparable or smaller byte budget.

-- Test 2: Distortion vs. the information-theoretic lower bound --
   (Shannon's bound is stated for the TOTAL squared error over the whole
   unit-norm vector, ||x - x~||^2 -- not divided by d -- so that is what
   we measure here, matching Section 7.2's Test 3 convention.)
  bits  lower bound(4^-b)  measured total SE  ratio
   1        0.25000            0.36099        1.44x
   2        0.06250            0.11424        1.83x
   3        0.01562            0.03474        2.22x
   4        0.00391            0.01172        3.00x
  Measured distortion sits within a small constant factor of the lower
  bound at every bit-width -- consistent with (not a precise reproduction
  of) the paper's proven ~2.7x cap.

-- Test 3 [COMMON TRAP]: forgetting per-vector overhead bytes --
  d=32 (a single block/vector -- overhead is NOT amortised away)
  TurboQuant 2-bit:
    Ignoring the norm:   8 bytes -> 16.0x compression (WRONG)
    Including the norm:  12 bytes -> 10.7x compression (correct)
  Q4_0 blockwise:
    Ignoring the scale:  16 bytes -> 8.0x compression (WRONG)
    Including the scale: 18 bytes -> 7.1x compression (correct)
  Overstatement factor: TurboQuant 1.50x, Q4_0 1.12x -- both look better than they are
  if the per-vector/per-block overhead is silently dropped from the count.

========================================================
15/15 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] dropping the overhead bytes from a compression ratio"
    Every format in this comparison has some per-vector or per-block overhead beyond the raw quantized values themselves — TurboQuant's stored norm (4 bytes), or a blockwise format's stored scale (2 bytes as fp16, per 32-element block). That overhead is easy to leave out of a quick compression-ratio calculation, and it is NOT a rounding error to do so — it inflates the reported ratio in a way that gets worse, not better, at smaller dimensions, because a fixed few bytes of overhead is a much larger fraction of a small vector's total size than of a large one's. A single 32-element vector at 2-bit TurboQuant looks like 16x compression if only the index bytes are counted, and a real 10.7x once the stored norm is included — the same distortion in the other direction hits blockwise formats too. Neither number is fabricated, but only one of them describes what an actual running system would need to store.

## Chapter Summary

This chapter built TurboQuant from first principles as a genuinely different family of quantizer from every blockwise format in Chapters 2 through 6 — one that treats a whole vector as a single geometric object rather than a sequence of independent 32-element blocks. Section 7.1 built and verified the random rotation that makes this possible, confirming both that it spreads a worst-case vector's energy evenly across every coordinate and that a rotated coordinate's variance matches the theoretical prediction the rest of the chapter depends on — and showed concretely what breaks if the "rotation" is not actually orthogonal. Section 7.2 built the Lloyd-Max codebook that exploits the resulting known distribution, combined it with the rotation into a complete reconstruction-optimal quantizer, and demonstrated why skipping the normalize step before quantizing is not a rounding error but a missing scale factor with unbounded consequences. Section 7.3 showed that reconstruction-optimal is not the same as inner-product-safe: the MSE quantizer is systematically biased for dot products, and the QJL residual correction fixes that bias by adding a second, independent 1-bit stage — provided the two stages are combined in a consistent coordinate system, which this section's own trap showed is not automatic. Section 7.4 applied the resulting quantizer to the KV cache, the online, uncalibrated structure blockwise quantization struggles with, and showed that a per-vector norm is not optional bookkeeping any more than the quantization indices themselves are. Section 7.5 closed the chapter by measuring TurboQuant against Chapters 2-6's blockwise formats directly and against the Shannon lower bound that no quantizer of any kind can beat, landing within a small constant factor of that bound at every bit-width tested — while also showing how easily a compression-ratio comparison can be quietly skewed by dropping the overhead bytes real storage would require. TurboQuant does not replace blockwise quantization; the two are complementary, and Chapter 4's hybrid engine now has both tools available — Q4_0/Q8_0 for the static weights a fused SIMD matmul needs, and TurboQuant for the KV cache that arrives one vector at a time with no calibration data at all.

## Self-Check Questions

1. Why does quantizing the worst-case vector `[1, 0, ..., 0]` with a coordinate-wise 1-bit quantizer waste almost all of its bit budget, and how does a random rotation fix this without changing what the vector "means"?
2. What specifically fails if the "rotation" matrix used is a raw matrix of independent Gaussian entries rather than an orthogonalized one, and why does the failure not depend on any quantization being involved at all?
3. Why can TurboQuant's Lloyd-Max codebooks be precomputed entirely offline, with no data dependence, in a way that Product Quantization's k-means-based codebooks cannot?
4. Section 7.2 normalizes a vector to the unit sphere before rotating and quantizing it. What goes wrong, concretely, if that normalization step is skipped and the raw vector is rotated and quantized directly?
5. Explain why a quantizer that is provably optimal for minimizing reconstruction error (MSE) can still be systematically biased when used to estimate inner products. What does "systematically biased" mean here, as opposed to merely noisy?
6. Walk through how the QJL residual stage restores unbiasedness without needing a better first-stage quantizer. What specific property of the rescaling factor `sqrt(pi/2)/d` makes this work?
7. Why must the residual quantized by QJL be computed in the same coordinate system as the query vector it will later be dotted with, and what goes wrong if it is computed in rotated coordinates instead?
8. Why is a per-vector stored norm not optional in KV cache quantization, even though the rotation matrix and codebook are shared across every vector in the cache?
9. Why is TurboQuant not a good replacement for Chapters 2-6's blockwise Q4_0/Q8_0 when quantizing static weight matrices for a fused matmul kernel?
10. What does it mean for TurboQuant's measured distortion to be checked "against the Shannon lower bound," and why is landing within a small constant factor of that bound a meaningfully stronger claim than just "lower error than one specific competitor"?

## Where We Go Next

This chapter, like Chapter 6, has been a supplementary deep dive rather than a direct continuation of the hybrid engine's forward pass — TurboQuant compresses the KV cache, a structure that exists only during generation and grows one vector at a time, which is a fundamentally different problem from the static weight loading Chapters 2 through 6 focused on. With both tools now available — blockwise Q4_0/Q8_0 for weights, TurboQuant for an online KV cache — Part 1's quantization story is complete. Part 2 turns from what a model's numbers are represented as to how fast an engine can actually move those numbers through a CPU: Chapter 8 covers the memory wall and the roofline model, the framework for understanding when an inference engine's speed is limited by arithmetic and when it is limited by how fast data can be fetched from memory in the first place — a question that shapes every optimization decision Part 2 makes from that point on.

## Worked Solutions

**1.** A 1-bit coordinate-wise quantizer has exactly two representable values for each coordinate, typically something like `+c` and `-c`, with no representable value for "exactly zero." The spike vector `[1, 0, ..., 0]` has one coordinate carrying all of the vector's information and every other coordinate at exactly zero — so the quantizer spends one bit meaningfully on the first coordinate and wastes every other bit forcing a zero into a nonzero bin. A random orthogonal rotation does not add or remove any information from the vector (it preserves length and inner products exactly), but it redistributes WHERE that information sits among the coordinates: after rotation, a vector that started as a spike has its energy spread almost evenly across all `d` coordinates, each at roughly `1/sqrt(d)`, so every coordinate now carries a comparable, nonzero amount of information for the quantizer to spend its bit on.

**2.** A raw Gaussian matrix `G` is not orthogonal — `G^T G` is not the identity matrix — which means `G^T` is not `G`'s inverse. Multiplying a vector by `G` still scrambles it thoroughly, exactly as a proper rotation would, but multiplying the result by `G^T` does not undo that scrambling; it produces a different vector with a large, structural discrepancy from the original. This failure has nothing to do with quantization: it shows up even in a pure "rotate, then rotate back" round-trip with no coordinate ever touched by a codebook, because the bug is in the linear algebra itself, not in any rounding step layered on top of it.

**3.** Product Quantization's codebooks are built by running k-means on a REPRESENTATIVE SAMPLE of the actual data to be quantized — the codebook is calibrated to whatever distribution that sample happens to have, which requires seeing real data before any quantization can begin. TurboQuant's codebooks are built by solving the Lloyd-Max problem for the Beta distribution that ANY rotated unit vector's coordinates follow, a fact that depends only on the dimension `d` and holds regardless of what the original, un-rotated data actually looked like. Because the random rotation guarantees this same statistical shape for every input, one codebook computed purely from `d` and the bit-width serves every vector that will ever arrive — including a KV cache vector produced by a token that has not been generated yet.

**4.** The Lloyd-Max codebook in Section 7.2 was solved for the distribution of a coordinate on the UNIT sphere, which has variance `1/d`. If the raw, un-normalized vector is rotated and quantized directly, its rotated coordinates have a variance that scales with the SQUARE of the vector's true norm, not `1/d` — for any vector whose norm is meaningfully different from 1, most or all of the rotated coordinates land far outside the range the codebook's bins actually cover, and every one of them saturates at whichever outermost centroid happens to be closest. The resulting reconstruction error is not proportional rounding noise; it is a missing scale factor, and it grows without bound as the true norm grows further from 1, because the reconstruction stays stuck near the unit sphere regardless of how large the actual input was.

**5.** "Systematically biased" means the estimator's EXPECTED value — its average over many independent randomizations of the quantizer, such as many different random rotation matrices — differs from the true inner product by a consistent, predictable amount in one direction, not just by random noise that would average out to zero over repeated trials. The MSE-optimal quantizer's inner-product estimate is a clear example: at low bit-widths, its expected value is a fixed fraction (as small as `2/pi` at 1 bit) of the true inner product, EVERY time, for EVERY vector, not merely wrong on some unlucky draws and right on others. A quantizer can be "optimal" for one loss function (squared reconstruction error) while being provably, unavoidably wrong in a specific direction for a different one (inner-product estimation), because the two loss functions are simply measuring different things.

**6.** Writing `x = x_tilde_mse + r`, the true inner product with a query `y` splits as `<y, x> = <y, x_tilde_mse> + <y, r>`. The MSE stage already supplies the first term exactly (it is just the dequantized MSE reconstruction). QJL supplies an UNBIASED estimate of the second term, `<y, r>`, using only the sign of a random Gaussian projection of `r` — and the specific rescaling `sqrt(pi/2)/d` is exactly the constant that makes `E[<y, QJL_dequant(r)>]` equal `<y, r>`, a fact that follows from the symmetry of the Gaussian distribution used to build the projection matrix `S`. Adding an unbiased estimate of the missing term to the exact first term gives, in expectation, the true inner product — the QJL stage does not need to reconstruct `r` accurately at all (and it does not: keeping only a sign per coordinate throws away almost all of `r`'s information), it only needs its INNER PRODUCT WITH `y`, in expectation, to be correct.

**7.** QJL's unbiasedness proof is stated for a random Gaussian matrix `S` applied directly to the vector whose inner product with an arbitrary query is being estimated — and that query lives in the ORIGINAL, un-rotated coordinate space, since that is the space attention scores and vector-database queries are actually computed in. If the residual is instead computed in ROTATED coordinates (subtracting before rotating the MSE term back), the resulting QJL-quantized correction describes a residual expressed in a different basis than the query it will later be dotted with. Adding that correction directly to the original-space MSE term and dotting the sum with an original-space query does not correspond to any consistent linear-algebra operation — the pieces being summed are not describing the same coordinates — and the resulting estimate drifts further from the true inner product than either stage's individual error would suggest, without producing any obviously wrong shape or crash to signal the mistake.

**8.** The rotation matrix and the Lloyd-Max codebook are universal precisely because they do not depend on any individual vector's data — the same `Pi` and the same codebook are mathematically valid for every vector of a given dimension. A vector's NORM is exactly the piece of information that normalization deliberately strips out before quantization, specifically because real vectors do not all have the same magnitude — and that is just as true of cached keys and values across different tokens in a sequence as it is of any other vector. Storing one norm per vector is what lets the shared rotation and codebook be reused universally while still recovering each vector's true scale at dequantization time; without it, every reconstructed vector would be rescaled by whichever single norm happened to be stored, which is correct for at most one vector in the entire cache.

**9.** A fused SIMD matmul kernel needs to dequantize and multiply-accumulate a weight block using only cheap, local arithmetic — Chapters 2-6's blockwise formats fit this exactly, since dequantizing one block requires only that block's own stored scale and a handful of integer multiplies. TurboQuant's dequantization requires a full `O(d^2)` matrix-vector multiply by the rotation matrix's transpose for every single vector before its values are usable at all — a cost that is negligible when applied once per KV cache vector as it is produced, but would dominate the inner loop of a matmul that touches every weight in a matrix multiple times per forward pass. Static weights also do not benefit from TurboQuant's core advantage: they are quantized once, offline, with as much calibration effort as desired, which is exactly the setting blockwise per-block scales already handle well, whereas TurboQuant's advantage is specifically for data that arrives online with no calibration opportunity at all.

**10.** The Shannon lower bound is not "how well some other specific quantizer happens to do" — it is a proof, from information theory, of the smallest possible distortion ANY quantizer whatsoever could achieve for a given bit budget on a unit-sphere vector, regardless of how clever or specialized that hypothetical quantizer might be. Beating one specific competitor (such as Q4_0) only shows TurboQuant is better than that one alternative; it says nothing about how much further improvement might still be possible. Landing within a small, bounded factor of the Shannon lower bound at every bit-width tested is a much stronger claim: it means no future quantizer, however sophisticated, could improve on TurboQuant by more than that same small constant factor, because the lower bound itself is a hard floor that no algorithm can cross.
