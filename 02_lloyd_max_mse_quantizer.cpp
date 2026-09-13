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
