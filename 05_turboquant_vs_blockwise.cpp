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
