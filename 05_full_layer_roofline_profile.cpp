// 05_full_layer_roofline_profile.cpp
// Chapter 8, Part 5 (capstone): profile one full transformer layer's
// decode step -- both RMSNorms, the four attention projections (Q, K, V,
// O), attention itself against a GQA KV cache, and the three FFN
// projections (gate, up, down) -- by summing the SAME cost formulas this
// chapter has already built and, in Section 8.3's case, already verified
// against a real kernel: linear_fp32_cost for every projection, and a
// new attention-cost formula verified here the same way, against a real,
// std::mdspan-viewed GQA attention kernel reused verbatim from Chapter
// 3.4's KVCache.
//
// The question this file answers: of everything a decode step touches,
// what actually dominates the bytes moved -- the weights, or the KV
// cache that grows with every generated token?
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 05_full_layer_roofline_profile.cpp -o out05

#include <mdspan/mdspan.hpp>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

// =========================================================================
// COST COUNTER (Sections 8.2/8.3's convention, reused)
// =========================================================================
struct Cost {
    long long flops = 0;
    long long bytes = 0;
    void flop(long long n = 1) { flops += n; }
    void access(long long n_bytes) { bytes += n_bytes; }
    double arithmetic_intensity() const { return static_cast<double>(flops) / static_cast<double>(bytes); }
};
Cost operator+(Cost a, const Cost& b) { a.flops += b.flops; a.bytes += b.bytes; return a; }
Cost& operator+=(Cost& a, const Cost& b) { a.flops += b.flops; a.bytes += b.bytes; return a; }

// Section 8.3's verified linear-layer formula, unchanged.
Cost linear_fp32_cost(long long n_out, long long n_in, long long seq) {
    Cost cost;
    cost.access(n_out * n_in * 4);
    cost.access(n_in * seq * 4);
    cost.access(n_out * seq * 4);
    cost.flop(2 * n_out * n_in * seq);
    return cost;
}
// Section 8.2's verified RMSNorm formula, generalized from a fixed DIM
// to a parameter (the derivation is identical; only the symbol changes).
constexpr long long EXP_FLOP_COST = 8;
Cost rmsnorm_cost(long long dim) {
    Cost c; c.flop(4 * dim + 4); c.access(3 * dim * 4); return c;
}

// =========================================================================
// CLOSED-FORM ATTENTION COST: one new query token, against `cache_len`
// already-cached positions, for n_heads_q query heads sharing n_heads_kv
// KV heads (GQA). Each KV head's cache rows are read ONCE (bytes) and
// reused by every query head in its group -- the same "read once, reuse
// across the group" shape Section 8.3 used for weight matrices -- while
// the score dot products, softmax, and weighted sum are real per-element
// arithmetic paid by every query head. This formula is written to match,
// term for term, what the real kernel below actually counts as it runs
// (Test 0 checks the two agree exactly), rather than being independently
// asserted.
// =========================================================================
Cost attention_cost_formula(long long n_heads_q, long long n_heads_kv, long long head_dim, long long cache_len) {
    Cost c;
    c.access(n_heads_kv * 2 * cache_len * head_dim * 4);      // K and V, one read each, per KV head
    c.flop(n_heads_q * cache_len * head_dim * 2);             // Q.K^T score dot products, all query heads
    c.flop(n_heads_q * cache_len * head_dim * 2);             // weighted sum over V, all query heads
    c.flop(n_heads_q * cache_len * (2 + EXP_FLOP_COST));      // softmax: subtract+exp, then scale, per element
    c.access(n_heads_q * 8 * cache_len);                      // softmax's own read-then-write of the score buffer
    return c;
}

// =========================================================================
// REAL, SMALL-SCALE VALIDATION: Chapter 3.4's KVCache and gqa_attention,
// reused verbatim (the exact struct and function, mdspan/submdspan and
// all), with a Cost counter threaded through so the closed-form formula
// above can be checked against bytes and FLOPs a real kernel actually
// produced -- not merely asserted.
// =========================================================================
void softmax_inplace_counted(std::span<float> scores, Cost& cost) {
    int len = static_cast<int>(scores.size());
    cost.access(4LL * len);
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; cost.flop(1 + EXP_FLOP_COST); }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) { s *= inv_sum; cost.flop(1); }
    cost.access(4LL * len);
}

struct KVCache {
    std::vector<float> K, V;
    int n_heads_kv, max_seq_len, head_dim;
    using View = std::mdspan<float, std::dextents<size_t, 3>>;

    KVCache(int nh, int seq, int hd) : n_heads_kv(nh), max_seq_len(seq), head_dim(hd) {
        K.assign(static_cast<size_t>(nh) * seq * hd, 0.0f);
        V.assign(static_cast<size_t>(nh) * seq * hd, 0.0f);
    }
    View k_view() { return View(K.data(), n_heads_kv, max_seq_len, head_dim); }
    View v_view() { return View(V.data(), n_heads_kv, max_seq_len, head_dim); }
    auto k_at(int h, int t) { return std::submdspan(k_view(), h, t, std::full_extent); }
    auto v_at(int h, int t) { return std::submdspan(v_view(), h, t, std::full_extent); }
    void store(int h, int t, std::span<const float> k, std::span<const float> v) {
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[i]; vslice[i] = v[i]; }
    }
};

// Chapter 3.4's gqa_attention, reused verbatim, with one addition: a
// Cost& thread that counts each K/V slot exactly once per KV head (read
// once, reused by every query head in its group) and every FLOP as the
// real dot products and weighted sums execute.
void gqa_attention_counted(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                            int cache_len, int n_heads_q, int group_size, Cost& cost) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(cache_len);
    int last_kv_h_counted = -1;

    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, head_dim);

        // This KV head's cache rows are read once per kv_h -- reused by
        // every query head in its group -- exactly the shape
        // attention_cost_formula charges.
        bool first_in_group = (kv_h != last_kv_h_counted);
        if (first_in_group) { cost.access(static_cast<long long>(cache_len) * head_dim * 4); last_kv_h_counted = kv_h; }

        for (int t = 0; t < cache_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            float d = 0.0f;
            for (int i = 0; i < head_dim; ++i) { d += q[i] * k[i]; cost.flop(2); }
            scores[t] = d * scale;
        }
        softmax_inplace_counted(std::span<float>(scores.data(), cache_len), cost);

        float* out = output.data() + h * head_dim;
        std::fill(out, out + head_dim, 0.0f);
        if (first_in_group) cost.access(static_cast<long long>(cache_len) * head_dim * 4);  // V, same reuse
        for (int t = 0; t < cache_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            float w = scores[t];
            for (int i = 0; i < head_dim; ++i) { out[i] += w * v[i]; cost.flop(2); }
        }
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 8.5: A Full Layer's Roofline Profile\n";
    std::cout << "========================================================\n\n";

    constexpr double PEAK_COMPUTE_GFLOPS = 896.0, PEAK_BANDWIDTH_GBPS = 51.2;
    constexpr double RIDGE_POINT = PEAK_COMPUTE_GFLOPS / PEAK_BANDWIDTH_GBPS;

    // =====================================================================
    // TEST 0: the closed-form attention formula matches a real,
    // mdspan-viewed GQA kernel's measured Cost, at small dimensions.
    // =====================================================================
    std::cout << "-- Test 0: closed-form attention cost vs. a real GQA kernel, same shape --\n";
    {
        constexpr int HEAD_DIM = 8, N_Q = 4, N_KV = 2, GROUP = 2, CACHE_LEN = 6;
        KVCache cache(N_KV, CACHE_LEN, HEAD_DIM);
        std::vector<float> k(HEAD_DIM), v(HEAD_DIM);
        for (int h = 0; h < N_KV; ++h) {
            for (int t = 0; t < CACHE_LEN; ++t) {
                for (int i = 0; i < HEAD_DIM; ++i) { k[i] = 0.01f * ((h * 37 + t * 11 + i) % 13 - 6); v[i] = 0.02f * ((h * 19 + t * 7 + i) % 9 - 4); }
                cache.store(h, t, k, v);
            }
        }
        std::vector<float> q(N_Q * HEAD_DIM), output(N_Q * HEAD_DIM);
        for (size_t i = 0; i < q.size(); ++i) q[i] = 0.05f * (static_cast<float>(i % 7) - 3.0f);

        Cost real_cost;
        gqa_attention_counted(q, cache, output, CACHE_LEN, N_Q, GROUP, real_cost);
        Cost formula_cost = attention_cost_formula(N_Q, N_KV, HEAD_DIM, CACHE_LEN);

        std::cout << "  Real GQA kernel: FLOPs=" << real_cost.flops << " Bytes=" << real_cost.bytes << "\n";
        std::cout << "  Closed formula:  FLOPs=" << formula_cost.flops << " Bytes=" << formula_cost.bytes << "\n";
        CHECK(real_cost.flops == formula_cost.flops);
        CHECK(real_cost.bytes == formula_cost.bytes);
    }

    // =====================================================================
    // Real illustrative model dimensions (Section 8.3's FFN convention,
    // extended to a full layer): DIM=4096, 32 query heads sharing 8 KV
    // heads (GQA group=4, the Llama-3-8B-class ratio), D_FF=14336.
    // =====================================================================
    constexpr long long DIM = 4096, N_HEADS_Q = 32, N_HEADS_KV = 8, HEAD_DIM = DIM / N_HEADS_Q, D_FF = 14336;
    constexpr long long KV_DIM = N_HEADS_KV * HEAD_DIM;

    auto decode_step_cost = [&](long long cache_len_before) {
        Cost c;
        c += rmsnorm_cost(DIM); c += rmsnorm_cost(DIM);                              // attn_norm, ffn_norm
        c += linear_fp32_cost(DIM, DIM, 1);      // W_q
        c += linear_fp32_cost(KV_DIM, DIM, 1);   // W_k
        c += linear_fp32_cost(KV_DIM, DIM, 1);   // W_v
        c += linear_fp32_cost(DIM, DIM, 1);      // W_o
        c += attention_cost_formula(N_HEADS_Q, N_HEADS_KV, HEAD_DIM, cache_len_before + 1);  // +1: this step's own new token
        c += linear_fp32_cost(D_FF, DIM, 1);     // W_gate
        c += linear_fp32_cost(D_FF, DIM, 1);     // W_up
        c += linear_fp32_cost(DIM, D_FF, 1);     // W_down
        return c;
    };
    // Bytes belonging only to the four QKVO projections and three FFN
    // projections -- the part of a decode step that does NOT depend on
    // how long the conversation so far has been.
    auto fixed_weight_bytes = [&]() {
        Cost c;
        c += linear_fp32_cost(DIM, DIM, 1);
        c += linear_fp32_cost(KV_DIM, DIM, 1);
        c += linear_fp32_cost(KV_DIM, DIM, 1);
        c += linear_fp32_cost(DIM, DIM, 1);
        c += linear_fp32_cost(D_FF, DIM, 1);
        c += linear_fp32_cost(D_FF, DIM, 1);
        c += linear_fp32_cost(DIM, D_FF, 1);
        return c.bytes;
    };

    // =====================================================================
    // TEST 1: at a realistic context length, classify the whole decode
    // step against the roofline, and check it lands memory-bound (as
    // Section 8.1's Test 3 assumed when it picked AI=0.5 as "a
    // representative decode-time GEMV").
    // =====================================================================
    std::cout << "\n-- Test 1: one decode step at cache_len=2048, classified against the roofline --\n";
    {
        constexpr long long L = 2048;
        Cost step = decode_step_cost(L);
        double ai = step.arithmetic_intensity();
        std::cout << "  FLOPs=" << step.flops << "  Bytes=" << step.bytes
                  << "  AI=" << std::fixed << std::setprecision(4) << ai << " FLOPs/byte\n";
        std::cout << "  Ridge point=" << std::setprecision(2) << RIDGE_POINT << " -> "
                  << (ai < RIDGE_POINT ? "MEMORY-BOUND" : "COMPUTE-BOUND") << "\n";
        CHECK(ai < RIDGE_POINT);
    }

    // =====================================================================
    // TEST 2 (COMMON TRAP): the KV cache read is the part of a decode
    // step that grows with context length, so it is tempting to assume
    // it dominates a long-context decode step's cost. At realistic
    // context lengths it does not -- the QKVO and FFN weight matrices,
    // read completely fresh on EVERY step regardless of context length,
    // are far larger.
    // =====================================================================
    std::cout << "\n-- Test 2 [COMMON TRAP]: attention does not dominate decode bytes at realistic context lengths --\n";
    {
        constexpr long long L = 2048;
        long long weight_bytes = fixed_weight_bytes();
        // Bytes attributable to K+V cache reads alone: one read of each
        // cached K and V slot, per KV head (shared across its group of
        // query heads, exactly as attention_cost_formula charges it).
        long long kv_cache_bytes = N_HEADS_KV * 2 * L * HEAD_DIM * 4;

        std::cout << "  Fixed QKVO+FFN weight bytes (every step, any context length): "
                  << weight_bytes / (1024 * 1024) << " MB\n";
        std::cout << "  KV-cache read bytes at context length " << L << ": "
                  << kv_cache_bytes / 1024 << " KB\n";
        std::cout << "  Weight bytes exceed KV-cache bytes by " << std::setprecision(1)
                  << (static_cast<double>(weight_bytes) / static_cast<double>(kv_cache_bytes)) << "x at this context length.\n";

        long long crossover_L = weight_bytes / (N_HEADS_KV * 2 * HEAD_DIM * 4);
        std::cout << "  The KV cache would need to reach " << crossover_L
                  << " tokens of context before its read bytes alone caught up to the fixed\n";
        std::cout << "  per-step weight-read cost -- far beyond a " << L << "-token context, and beyond\n";
        std::cout << "  most deployed context windows. Attention's cost is the part that VISIBLY\n";
        std::cout << "  grows with context, which is exactly why it gets blamed first -- but the\n";
        std::cout << "  fixed weight-read cost, paid again every single step regardless of context,\n";
        std::cout << "  is what a decode step's memory traffic is actually dominated by until the\n";
        std::cout << "  context grows far longer than most conversations ever do.\n";

        CHECK(weight_bytes > kv_cache_bytes * 10);       // weight reads dominate by an order of magnitude here
        CHECK(crossover_L > 50000);                       // and only stop dominating at a very long context
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
