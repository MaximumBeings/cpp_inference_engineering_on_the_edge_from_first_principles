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
