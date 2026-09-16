// Chapter 15.3 -- Section 15.2 established five concrete differences
// between Llama's architecture (this book's own Chapter 3 and Chapter
// 12) and the real downloaded Qwen2.5 checkpoint. This section adapts
// the actual forward-pass and tokenizer CODE for those five
// differences, rather than merely describing them: Chapter 3.1's
// RMSNorm, Chapter 3.2's SwiGLU FFN, Chapter 3.3's RoPE, and Chapter
// 3.4's GQA attention are all reused verbatim (none of them needed to
// change), while a QKV bias step is added between projection and RoPE,
// the attention call is driven by a real 7:1 group size instead of an
// arbitrary example ratio, and Chapter 12.3's Llama-3-only chat
// template is generalized into one function that produces either
// convention from a single boolean -- exactly the
// tokenizer.ggml.add_bos_token flag Section 15.1's reader already
// knows how to read from a real file.
//
// Every check in this section runs against a small SYNTHETIC model at
// real-shaped proportions (7 query heads sharing 1 KV head, echoing the
// real file's own 14:2 ratio in miniature) rather than the actual
// 644 MB checkpoint -- this section is about proving the CODE is
// correct, which small deterministic weights can do exactly as well as
// real ones and a great deal faster. Loading the real weights and
// running this exact machinery against them is Section 15.4's job.
//
// This section's block combines enough sequential floating-point
// reductions (QKV projections, GQA attention, the SwiGLU FFN, two
// residual additions, all repeated across 3 sequential positions) that
// Chapter 11's own cross-architecture floating-point fix applies here
// too: GCC's default -ffp-contract=fast lets x86 and aarch64 fuse
// multiply-add differently, producing a last-digit difference in Test
// 5's printed norm between this book's x86_64 cross-check and its
// aarch64 targets. -ffp-contract=off is the same fix Chapter 11 already
// established, applied here for the same reason.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_qwen2_transformer_block.cpp -o 03_qwen2_transformer_block

#include <mdspan/mdspan.hpp>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// ---------------------------------------------------------------------
// Chapter 3.1's RMSNorm, reused verbatim -- Qwen2 also uses RMSNorm for
// both attn_norm and ffn_norm, so nothing here needed to change.
// ---------------------------------------------------------------------
void rms_norm(std::span<float> out, std::span<const float> x,
              std::span<const float> weights, float epsilon = 1e-6f) {
    const size_t d = x.size();
    // Accumulated in double: Section 15.4 discovered that a naive float32
    // running sum here, repeated across 24 real layers and thousands of
    // elements per reduction, drifts enough to flip the argmax of a real
    // generation -- see that section's account of the bug. The fix is
    // ordinary numerical-analysis practice (accumulate reductions in
    // higher precision than the input/output type), not an architecture
    // change, so it belongs in this shared function, used identically at
    // every scale from this section's tiny synthetic block up to the
    // real 896-wide reductions.
    double sum_sq = 0.0;
    for (size_t i = 0; i < d; ++i) sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    float rms_inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(d)) + epsilon);
    for (size_t i = 0; i < d; ++i) out[i] = (x[i] * rms_inv) * weights[i];
}

// ---------------------------------------------------------------------
// Chapter 3.2's matmul and SwiGLU FFN, reused verbatim except for the
// same double-precision accumulation fix as rms_norm above (same
// reduction-precision reasoning, same real-file discovery in Section
// 15.4).
// ---------------------------------------------------------------------
void matmul(std::span<float> out, std::span<const float> x,
            std::span<const float> W, size_t in_dim, size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < in_dim; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(W[j * in_dim + i]);
        out[j] = static_cast<float>(sum);
    }
}
inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
void swiglu_ffn(std::span<float> out, std::span<const float> x,
                 std::span<const float> W_gate, std::span<const float> W_up,
                 std::span<const float> W_down, size_t dim, size_t d_ff) {
    std::vector<float> gate_proj(d_ff), up_proj(d_ff), hidden(d_ff);
    matmul(gate_proj, x, W_gate, dim, d_ff);
    matmul(up_proj, x, W_up, dim, d_ff);
    for (size_t i = 0; i < d_ff; ++i) hidden[i] = silu(gate_proj[i]) * up_proj[i];
    matmul(out, hidden, W_down, d_ff, dim);
}

// ---------------------------------------------------------------------
// ADAPTATION 1 (new in this section): a bias-add step. Real Qwen2 adds
// a learned per-output-dimension bias after the Q/K/V projection
// matmul; Llama's attention (Chapter 3.4's gqa_attention, built against
// Chapter 12's source material) has no such tensors at all, so Chapter
// 3 never needed this function. It is a two-line addition, not a
// redesign -- exactly the kind of "small, local, well-understood
// change" a real architecture diff often turns out to require.
// ---------------------------------------------------------------------
void linear_with_bias(std::span<float> out, std::span<const float> x,
                       std::span<const float> W, std::span<const float> bias,
                       size_t in_dim, size_t out_dim) {
    matmul(out, x, W, in_dim, out_dim);
    for (size_t j = 0; j < out_dim; ++j) out[j] += bias[j];
}

// ---------------------------------------------------------------------
// Chapter 3.3's RoPE, reused verbatim -- its base parameter already
// existed, so ADAPTATION 2 (rope_theta=1,000,000 instead of Chapter
// 3.3's own 10,000 default) is a call-site argument, not a code change.
// ---------------------------------------------------------------------
struct RoPETables {
    std::vector<float> cos_vals, sin_vals;
    int half_dim;
    RoPETables(int seq_len, int head_dim, float base) : half_dim(head_dim / 2) {
        cos_vals.resize(seq_len * half_dim);
        sin_vals.resize(seq_len * half_dim);
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * k) / head_dim);
                float angle = static_cast<float>(pos) * theta;
                cos_vals[pos * half_dim + k] = std::cos(angle);
                sin_vals[pos * half_dim + k] = std::sin(angle);
            }
        }
    }
    float cos_at(int pos, int k) const { return cos_vals[pos * half_dim + k]; }
    float sin_at(int pos, int k) const { return sin_vals[pos * half_dim + k]; }
};
void apply_rope(std::span<float> vec, int pos, const RoPETables& tables) {
    // Split-half (NeoX/HF-style) pairing: (vec[k], vec[k+half_dim]),
    // not the interleaved (vec[2k], vec[2k+1]) pairing this function
    // used through Chapter 11. Section 15.4's real-file debugging (see
    // that section) confirmed against an independently built llama.cpp
    // that real HF-derived architectures -- Qwen2 included -- rotate
    // this way; Chapter 3's own synthetic self-tests never had a way to
    // tell the two conventions apart, since they only ever checked
    // internal consistency against this same file's own (then
    // interleaved) convention. The algebraic property Test 4 below
    // checks -- relative-position invariance -- holds under either
    // pairing, which is exactly why the self-tests alone could not have
    // caught this.
    const int half_dim = tables.half_dim;
    for (int k = 0; k < half_dim; ++k) {
        float x1 = vec[k], x2 = vec[k + half_dim];
        float c = tables.cos_at(pos, k), s = tables.sin_at(pos, k);
        vec[k] = x1 * c - x2 * s;
        vec[k + half_dim] = x1 * s + x2 * c;
    }
}
float dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

// ---------------------------------------------------------------------
// Chapter 3.4's KVCache and gqa_attention, reused verbatim -- its
// group_size parameter already existed, so ADAPTATION 3 (a real 7:1
// ratio instead of Chapter 3.4's own 1:1 and 2:1 worked examples) is
// also a call-site argument, not a code change.
// ---------------------------------------------------------------------
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
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
void gqa_attention(std::span<const float> q_heads, KVCache& cache,
                    std::span<float> output, int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len);
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, head_dim);
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            double d = 0.0;   // double accumulation -- same fix as matmul/rms_norm above
            for (int i = 0; i < head_dim; ++i) d += static_cast<double>(q[i]) * static_cast<double>(k[i]);
            scores[t] = static_cast<float>(d) * scale;
        }
        softmax_inplace(std::span<float>(scores.data(), seq_len));
        std::vector<double> acc(static_cast<size_t>(head_dim), 0.0);
        for (int t = 0; t < seq_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            double w = scores[t];
            for (int i = 0; i < head_dim; ++i) acc[static_cast<size_t>(i)] += w * static_cast<double>(v[i]);
        }
        float* out = output.data() + h * head_dim;
        for (int i = 0; i < head_dim; ++i) out[i] = static_cast<float>(acc[static_cast<size_t>(i)]);
    }
}

// ---------------------------------------------------------------------
// The complete adapted block: RMSNorm -> QKV projection WITH bias ->
// RoPE(Q,K) -> GQA attention -> output projection (no bias -- real
// Qwen2 has none on attn_output, confirmed in Section 15.1's real-file
// tensor listing) -> residual -> RMSNorm -> SwiGLU FFN -> residual.
// This is Chapter 11's "assemble one full transformer layer" pattern,
// re-run here with the three adaptations wired in instead of assumed
// away.
// ---------------------------------------------------------------------
struct QwenBlockWeights {
    std::vector<float> attn_norm, Wq, bq, Wk, bk, Wv, bv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};
struct QwenShape {
    int dim, n_heads, n_heads_kv, head_dim, d_ff;
    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_heads_kv * head_dim; }
    int group_size() const { return n_heads / n_heads_kv; }
};

void qwen2_block_forward(std::span<float> x, const QwenShape& shape, const QwenBlockWeights& w,
                          KVCache& cache, int pos, const RoPETables& rope) {
    std::vector<float> normed(shape.dim), q(shape.q_dim()), k(shape.kv_dim()), v(shape.kv_dim());
    std::vector<float> attn_out(shape.q_dim()), proj_out(shape.dim);

    rms_norm(normed, x, w.attn_norm);
    linear_with_bias(q, normed, w.Wq, w.bq, shape.dim, shape.q_dim());
    linear_with_bias(k, normed, w.Wk, w.bk, shape.dim, shape.kv_dim());
    linear_with_bias(v, normed, w.Wv, w.bv, shape.dim, shape.kv_dim());

    for (int h = 0; h < shape.n_heads; ++h)
        apply_rope(std::span<float>(q.data() + h * shape.head_dim, shape.head_dim), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        apply_rope(std::span<float>(k.data() + h * shape.head_dim, shape.head_dim), pos, rope);

    for (int h = 0; h < shape.n_heads_kv; ++h)
        cache.store(h, pos, std::span<const float>(k.data() + h * shape.head_dim, shape.head_dim),
                             std::span<const float>(v.data() + h * shape.head_dim, shape.head_dim));

    gqa_attention(q, cache, attn_out, pos + 1, shape.n_heads, shape.group_size());
    matmul(proj_out, attn_out, w.Wo, shape.q_dim(), shape.dim);   // no bias -- Section 15.1 confirmed none exists
    for (int i = 0; i < shape.dim; ++i) x[i] += proj_out[i];       // residual 1

    std::vector<float> normed2(shape.dim), ffn_out(shape.dim);
    rms_norm(normed2, x, w.ffn_norm);
    swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, shape.dim, shape.d_ff);
    for (int i = 0; i < shape.dim; ++i) x[i] += ffn_out[i];        // residual 2
}

// ---------------------------------------------------------------------
// ADAPTATION 4 (tokenizer side): Chapter 12.3 hard-coded the Llama-3
// convention (BOS unconditionally at position 0). Section 15.2 showed
// ChatML uses no BOS-equivalent at all. Rather than maintain two
// separate template functions, this generalizes both into one function
// driven by a single boolean -- exactly Section 15.1's
// tokenizer.ggml.add_bos_token, read directly from whichever real file
// is loaded, rather than assumed from the model family's name.
// ---------------------------------------------------------------------
struct ChatEncoderConfig {
    bool prepend_bos;
    int32_t bos_id;         // meaningless when prepend_bos is false
    int32_t turn_start_id;  // Llama: start_header_id : Qwen2: im_start
    int32_t turn_end_id;    // Llama: eot_id          : Qwen2: im_end
};
std::vector<int32_t> encode_conversation(const ChatEncoderConfig& cfg, int n_turns) {
    std::vector<int32_t> ids;
    if (cfg.prepend_bos) ids.push_back(cfg.bos_id);
    for (int i = 0; i < n_turns; ++i) {
        ids.push_back(cfg.turn_start_id);
        ids.push_back(cfg.turn_end_id);
    }
    return ids;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 15.3: Adapting the Forward Pass and Tokenizer for Qwen2\n";
    std::cout << "========================================================\n";

    // =====================================================================
    // TEST 1: linear_with_bias, hand-traced against a tiny known matrix.
    // =====================================================================
    std::cout << "\n-- Test 1: linear_with_bias (hand-traced) --\n";
    {
        std::vector<float> x = {1.0f, 2.0f};
        std::vector<float> W = {1.0f, 0.0f,   0.0f, 1.0f,   1.0f, 1.0f};   // 3x2, row-major
        std::vector<float> bias = {0.5f, -0.5f, 2.0f};
        std::vector<float> out(3);
        linear_with_bias(out, x, W, bias, 2, 3);
        CHECK_NEAR(out[0], 1.5f, 1e-6f);    // (1*1 + 2*0) + 0.5
        CHECK_NEAR(out[1], 1.5f, 1e-6f);    // (1*0 + 2*1) - 0.5
        CHECK_NEAR(out[2], 5.0f, 1e-6f);    // (1*1 + 2*1) + 2.0
        std::cout << "  linear_with_bias([1,2]) = [" << out[0] << ", " << out[1] << ", " << out[2] << "]\n";
    }

    // =====================================================================
    // TEST 2: the bias term genuinely changes the projection -- with a
    // nonzero bias, Wx+b != Wx, so QKV bias is not a no-op Qwen2 merely
    // carries around unused.
    // =====================================================================
    std::cout << "\n-- Test 2: bias is not a no-op --\n";
    {
        std::vector<float> x = {0.3f, -0.7f, 1.1f};
        std::vector<float> W = {1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f};   // 2x3
        std::vector<float> zero_bias = {0.0f, 0.0f};
        std::vector<float> real_bias = {0.25f, -0.1f};
        std::vector<float> out_zero(2), out_real(2);
        linear_with_bias(out_zero, x, W, zero_bias, 3, 2);
        linear_with_bias(out_real, x, W, real_bias, 3, 2);
        CHECK(out_zero[0] != out_real[0]);
        CHECK(out_zero[1] != out_real[1]);
        std::cout << "  Wx (zero bias) = [" << out_zero[0] << ", " << out_zero[1] << "]\n";
        std::cout << "  Wx+b (real bias) = [" << out_real[0] << ", " << out_real[1] << "]\n";
    }

    // =====================================================================
    // TEST 3: the real 7:1 GQA ratio -- 7 query heads all sharing ONE KV
    // head. Every query head reads the identical K, V; only the query
    // itself can make their outputs differ, exactly Chapter 3.4's own
    // sharing property, now checked at the real ratio instead of 2:1.
    // =====================================================================
    std::cout << "\n-- Test 3: real 7:1 GQA ratio (7 query heads, 1 KV head) --\n";
    {
        constexpr int HEAD_DIM = 4, N_HEADS_KV = 1, N_HEADS_Q = 7, GROUP = 7, SEQ = 2;
        KVCache cache(N_HEADS_KV, SEQ + 1, HEAD_DIM);
        cache.store(0, 0, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
        cache.store(0, 1, std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f}, std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f});

        std::vector<float> q(N_HEADS_Q * HEAD_DIM, 0.0f);
        for (int h = 0; h < N_HEADS_Q; ++h) q[h * HEAD_DIM + (h % 2)] = 1.0f;  // alternate alignment with K0/K1

        std::vector<float> output(N_HEADS_Q * HEAD_DIM, 0.0f);
        gqa_attention(q, cache, output, SEQ, N_HEADS_Q, GROUP);

        // Heads aligned with K0 (even h) all produce the SAME output as
        // each other; heads aligned with K1 (odd h) all produce a
        // DIFFERENT, mutually-identical output -- the group-sharing
        // structure holding at 7:1 exactly as it did at 2:1 in Chapter 3.4.
        for (int h = 2; h < N_HEADS_Q; h += 2)
            for (int i = 0; i < HEAD_DIM; ++i)
                CHECK_NEAR(output[h * HEAD_DIM + i], output[0 * HEAD_DIM + i], 1e-5f);
        for (int h = 3; h < N_HEADS_Q; h += 2)
            for (int i = 0; i < HEAD_DIM; ++i)
                CHECK_NEAR(output[h * HEAD_DIM + i], output[1 * HEAD_DIM + i], 1e-5f);
        CHECK(std::fabs(output[0] - output[HEAD_DIM]) > 0.01f);   // the two groups genuinely differ
        std::cout << "  head 0 (K0-aligned) output[0]=" << output[0]
                   << ", head 1 (K1-aligned) output[0]=" << output[HEAD_DIM] << "\n";
        std::cout << "  all 4 K0-aligned heads (0,2,4,6) agree, all 3 K1-aligned heads (1,3,5) agree: yes\n";
    }

    // =====================================================================
    // TEST 4: RoPE's relative-position invariance still holds at the
    // real base=1,000,000 -- the algebraic property Chapter 3.3 proved
    // does not depend on which base is used.
    // =====================================================================
    std::cout << "\n-- Test 4: RoPE relative-position invariance at base=1000000 --\n";
    {
        RoPETables tables(16, 4, 1000000.0f);
        std::vector<float> Q = {0.6f, 0.4f, -0.2f, 0.8f};
        std::vector<float> K = {0.3f, 0.7f, 0.5f, 0.1f};
        std::vector<float> Qa = Q, Ka = K;
        apply_rope(Qa, 5, tables); apply_rope(Ka, 3, tables);
        std::vector<float> Qb = Q, Kb = K;
        apply_rope(Qb, 2, tables); apply_rope(Kb, 0, tables);
        CHECK_NEAR(dot(Qa, Ka), dot(Qb, Kb), 1e-4f);
        std::cout << "  dot(Q@5,K@3) == dot(Q@2,K@0) at base=1000000: "
                   << (std::fabs(dot(Qa, Ka) - dot(Qb, Kb)) < 1e-4f ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 5: the complete adapted block, at real-shaped proportions
    // (7:1 GQA), run for 3 sequential positions -- deterministic (same
    // weights and inputs reproduce the same output bit-for-bit), and the
    // residual connections keep the output finite and nonzero.
    // =====================================================================
    std::cout << "\n-- Test 5: complete Qwen2 block, 7:1 GQA, 3 positions --\n";
    std::vector<float> block_trace;
    {
        QwenShape shape{.dim = 28, .n_heads = 7, .n_heads_kv = 1, .head_dim = 4, .d_ff = 16};
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.05f);
        auto rand_vec = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };

        QwenBlockWeights w;
        w.attn_norm.assign(shape.dim, 1.0f);
        w.Wq = rand_vec(static_cast<size_t>(shape.q_dim()) * shape.dim);
        w.bq = rand_vec(shape.q_dim());
        w.Wk = rand_vec(static_cast<size_t>(shape.kv_dim()) * shape.dim);
        w.bk = rand_vec(shape.kv_dim());
        w.Wv = rand_vec(static_cast<size_t>(shape.kv_dim()) * shape.dim);
        w.bv = rand_vec(shape.kv_dim());
        w.Wo = rand_vec(static_cast<size_t>(shape.dim) * shape.q_dim());
        w.ffn_norm.assign(shape.dim, 1.0f);
        w.Wgate = rand_vec(static_cast<size_t>(shape.d_ff) * shape.dim);
        w.Wup = rand_vec(static_cast<size_t>(shape.d_ff) * shape.dim);
        w.Wdown = rand_vec(static_cast<size_t>(shape.dim) * shape.d_ff);

        RoPETables rope(8, shape.head_dim, 1000000.0f);
        KVCache cache(shape.n_heads_kv, 8, shape.head_dim);

        std::vector<float> x = rand_vec(shape.dim);
        std::vector<float> x_run1 = x;
        for (int pos = 0; pos < 3; ++pos) qwen2_block_forward(x_run1, shape, w, cache, pos, rope);

        bool all_finite = true;
        for (float v : x_run1) if (!std::isfinite(v)) all_finite = false;
        CHECK(all_finite);
        float norm_sq = 0.0f;
        for (float v : x_run1) norm_sq += v * v;
        CHECK(norm_sq > 0.0f);
        block_trace = x_run1;
        std::cout << "  after 3 positions: finite=" << all_finite << ", ||x||^2=" << norm_sq << "\n";

        // Determinism: re-running from the same starting state with a
        // FRESH cache and the same weights must reproduce bit-identical
        // output -- nothing in this block reads any source of randomness
        // at inference time (the randomness above only built the weights).
        KVCache cache2(shape.n_heads_kv, 8, shape.head_dim);
        std::vector<float> x_run2 = x;
        for (int pos = 0; pos < 3; ++pos) qwen2_block_forward(x_run2, shape, w, cache2, pos, rope);
        bool identical = (x_run1 == x_run2);
        CHECK(identical);
        std::cout << "  re-run with fresh cache, same weights and input: bit-identical: " << (identical ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 6: one ChatEncoder function produces BOTH Chapter 12.3's
    // Llama-3 convention and Section 15.2's ChatML convention, selected
    // entirely by the add_bos_token boolean Section 15.1's reader
    // already knows how to read from a real file -- no per-family
    // branching anywhere in encode_conversation itself.
    // =====================================================================
    std::cout << "\n-- Test 6: one tokenizer function, two model families --\n";
    {
        ChatEncoderConfig llama_cfg{.prepend_bos = true, .bos_id = 128000, .turn_start_id = 128006, .turn_end_id = 128009};
        ChatEncoderConfig qwen_cfg{.prepend_bos = false, .bos_id = -1, .turn_start_id = 100000, .turn_end_id = 100001};

        auto llama_ids = encode_conversation(llama_cfg, 3);
        auto qwen_ids = encode_conversation(qwen_cfg, 3);

        CHECK(llama_ids.front() == 128000);
        CHECK(llama_ids.size() == 1 + 3 * 2);
        CHECK(qwen_ids.front() != 128000);
        CHECK(qwen_ids.size() == 3 * 2);
        bool qwen_has_no_bos = true;
        for (int32_t id : qwen_ids) if (id == 128000) qwen_has_no_bos = false;
        CHECK(qwen_has_no_bos);
        std::cout << "  add_bos_token=true  (Llama-3 convention): " << llama_ids.size() << " ids, starts with BOS\n";
        std::cout << "  add_bos_token=false (Qwen2 ChatML convention): " << qwen_ids.size() << " ids, no BOS anywhere\n";
        std::cout << "  same encode_conversation() function, driven by one boolean read from the GGUF file\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
