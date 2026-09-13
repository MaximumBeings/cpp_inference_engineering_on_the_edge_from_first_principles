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
