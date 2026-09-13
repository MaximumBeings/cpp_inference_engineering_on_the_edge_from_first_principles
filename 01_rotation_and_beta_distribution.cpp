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
