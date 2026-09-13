# Chapter 6: The Hybrid Quantized Engine -- Mixing Precisions for Different Jobs

**What you will understand by the end of this chapter:**

- Why giving every weight matrix the same quantization format is a mistake that has nothing to do with some matrices being harder to quantize than others — the raw rounding error a block format introduces depends only on the format's bit width, verified directly by quantizing an attention-shaped and an FFN-shaped matrix from the same distribution and finding statistically identical error.
- Why the real argument for hybrid quantization is about DOWNSTREAM CONSEQUENCES, not raw error: a quantization error fed through a softmax's discrete argmax decision can flip which token a model attends to, measurably more often at Q4_0 than at Q8_0, while the same-sized error fed through an FFN's smooth SiLU gate produces only a continuous, proportional output error with no equivalent decision to flip.
- How to encode a per-tensor quantization policy directly in the format GGUF already provides — a tensor descriptor's own type field, dispatched on by a reader that never guesses a format from a tensor's name — and why guessing from the name is not a hypothetical risk: Chapter 5.3's own naming convention has a tensor ("ffn_norm") whose real role contradicts what its name would suggest.
- How to compute, rather than assume, whether a hybrid policy is actually worth it: this chapter's own first attempt at a policy (keep norm AND embedding weights at F32) turned out to cost 86% more than uniform Q4_0, almost entirely because of the embedding table's size at a modern vocabulary — a mistake the chapter corrects in the same section it is discovered in.
- How to wire a complete hybrid engine end to end: a real multi-layer GGUF file, memory-mapped with Chapter 5.3's zero-copy technique, feeding Chapter 3's RMSNorm, RoPE, and grouped-query attention through Chapter 4's fused Q8_0/Q4_0 dot products, generating tokens autoregressively — with a plain FP32 KV cache, because TurboQuant is Chapter 7's subject, not this one's.

**What you need to know first:**

- Chapter 3's RMSNorm, SwiGLU, RoPE, and grouped-query attention with a KV cache, and Chapter 4's blockwise Q4_0/Q8_0 quantization with fused dot products — this chapter assembles those pieces rather than re-deriving them.
- Chapter 5's GGUF binary format, its per-tensor type tag, and its `mmap`-based zero-copy weight access — this chapter's hybrid policy is stored using exactly that mechanism, nothing new.

---

Chapter 5 built one quantization pipeline and applied it the same way to every tensor in a GGUF file. A real inference engine does not do this: it routes different weight classes through different formats, on purpose, because different classes fail differently when quantized and cost different amounts of memory to protect. This chapter builds that routing from first principles rather than asserting it. Section 6.1 verifies, by actually measuring it, that the case for hybrid quantization is about consequences rather than raw quantizability. Section 6.2 turns that case into a policy encoded the way GGUF already supports — a per-tensor type tag, dispatched on correctly. Section 6.3 prices the policy in real bytes for a large model, and corrects a mistake in its own first draft when the numbers do not support it. Sections 6.4 and 6.5 wire the corrected policy into a genuine forward pass and then a complete multi-layer, multi-token engine, reading real weights from a memory-mapped GGUF file — the culmination of Chapters 3 through 5, with the KV cache still at plain FP32 until Chapter 7 gives it a reason to change.

## 6.1 Why Hybrid: Same Format, Different Consequences

### Intuition

It would be convenient if some weight matrices were simply more "quantizable" than others — if an attention matrix's values were, in some measurable sense, harder to round to 4 bits than an FFN matrix's values. That is not the story this section finds. Rounding a block of 32 weights to Q4_0 or Q8_0 introduces an error that depends on the block's own values and the format's bit width — nothing about the matrix's role in a transformer enters into that calculation. What actually differs by role is what happens to that error AFTER it is introduced: an attention weight feeds a dot product that gets normalized by softmax and then very often consumed through a discrete decision (which cached position wins the most attention weight), and a small perturbation can flip that decision outright whenever two positions are nearly tied. An FFN weight feeds a smooth, non-competitive SiLU gate with no equivalent discrete decision to flip. Same format, same raw error, different consequences.

### The Concept, In Detail

Two experiments make this precise. The first quantizes an attention-shaped (DIM x DIM) and an FFN-shaped (D_FF x DIM) matrix drawn from the same random distribution at both Q4_0 and Q8_0, and measures the relative error each format introduces. The two shapes' errors match each other within measurement noise for a given format — the shape carries no information the format doesn't already determine. The second experiment isolates a single Q4-quantized attention weight matrix (the key projection), feeds it a genuine query-key dot product across several cached positions, and measures two different things: the raw relative error in the resulting scores, and — across many independent trials — how often the SOFTMAX ARGMAX (which position the query ends up attending to most) changes compared to the FP32 result. A first version of this section measured the wrong second quantity: the total-variation distance between the full FP32 and Q4 softmax distributions, expecting it to be larger than the raw score error (an "amplification" story). It was not — softmax's outputs are bounded in [0, 1] and sum to 1, which caps how far the BULK distribution can move regardless of the input error, so the total-variation number came out smaller than the raw error, not larger. The discrete argmax decision, measured across many trials instead of one distribution, is where the real fragility shows up: at Q4_0 it flips measurably more often than at Q8_0. Feeding the identical Q4 format into an FFN's gate matrix instead produces a raw output error of the same order of magnitude as the raw attention score error — unsurprising, since the arithmetic is identical — but there is no discrete decision downstream for that error to disrupt; it stays a continuous, bounded output error. The final piece of the argument is not about error at all, but about population size: with this chapter's own D_FF = 4 x DIM convention, a layer's FFN matrices (gate, up, down) outnumber its attention matrices (Q, K, V, O) three to one, which is why aggressive Q4_0 quantization there buys the largest absolute memory savings while attention's smaller footprint makes the extra precision of Q8_0 comparatively cheap to afford.

### Code and Verification

```cpp
// Chapter 6.1 -- Chapters 4 and 5 built one quantization pipeline and
// applied it uniformly: every weight matrix went through the same Q4_0
// or Q8_0 block format. A real inference engine does not do this. It
// gives different weight CLASSES different formats -- and the reason
// is not that some matrices are more "quantizable" than others in the
// sense of producing bigger rounding errors. This section verifies
// that claim directly: the raw dequantization error a block format
// introduces is a property of the format (how many bits it spends per
// weight), not of which matrix the block came from. What actually
// differs by weight class is what happens to that error AFTER it is
// introduced -- and how many elements of that class exist in the first
// place.
//
// Two separate experiments make this concrete. First, the same-format,
// same-distribution comparison: quantizing an "attention-shaped" matrix
// and an "FFN-shaped" matrix with the same random distribution at Q4_0
// produces statistically indistinguishable relative error -- the shape
// does not matter, only the bit width does (Test 1). Second, a
// downstream-consumption comparison: attention's softmax gets consumed
// through a DISCRETE decision -- which position wins the most weight --
// and whenever two positions are nearly tied, a small score perturbation
// can flip that decision outright; measured across many trials, Q4's
// larger raw score error flips it measurably more often than Q8's does
// (Test 2). An FFN's SiLU-gated output has no equivalent discrete
// decision to flip -- its output error is continuous and stays at the
// same order of magnitude as the raw quantization error that produced
// it, no matter how the scores happen to be spaced (Test 3). Different
// FAILURE MODES for the same format, not a difference in how
// "quantizable" each matrix is -- and, worth stating plainly, NOT that
// softmax makes the whole attention distribution move further on
// average than the FFN's output does. A first attempt at this section
// measured exactly that (total variation distance between the FP32 and
// Q4 softmax outputs) and found it was actually SMALLER than the raw
// score error, not larger -- softmax's outputs are bounded in [0, 1]
// and sum to 1, which caps how far the bulk distribution can move
// regardless of the input error's size. The discrete argmax decision,
// not the bulk distribution, is where attention's real fragility shows
// up, and that only becomes visible by running many trials and counting
// how often the decision itself changes.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>
#include <span>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// -- Chapter 4.2/4.3's fp16_t and block layouts (repeated per this
// book's one-file-per-section convention). --
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
struct BlockQ8 { fp16_t scale; int8_t weights[32]; };   // 34 bytes -- Chapter 4.2
struct BlockQ4 { fp16_t scale; uint8_t nibbles[16]; };  // 18 bytes -- Chapter 4.3
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
        int lo = static_cast<int>(std::clamp(std::round(data[2*i] * inv), -8.0f, 7.0f));
        int hi = static_cast<int>(std::clamp(std::round(data[2*i+1] * inv), -8.0f, 7.0f));
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
        out[2*i]   = static_cast<float>(static_cast<int>(p & 0xF) - 8) * s;
        out[2*i+1] = static_cast<float>(static_cast<int>(p >> 4) - 8) * s;
    }
}

// -- Chapter 4.4/5.3's fused dot products, against spans of blocks. --
float dot_q8(std::span<const BlockQ8> blocks, const float* x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 32; ++i) bs += static_cast<float>(blocks[b].weights[i]) * x[b * 32 + i];
        total += bs * s;
    }
    return total;
}
float dot_q4(std::span<const BlockQ4> blocks, const float* x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 16; ++i) {
            uint8_t p = blocks[b].nibbles[i];
            bs += static_cast<float>(static_cast<int>(p & 0xF) - 8) * x[b * 32 + 2 * i];
            bs += static_cast<float>(static_cast<int>(p >> 4) - 8) * x[b * 32 + 2 * i + 1];
        }
        total += bs * s;
    }
    return total;
}

// A full row-major weight matrix, quantized block by block: rows *
// (cols/32) blocks total, laid out row 0's blocks first, then row 1's.
std::vector<BlockQ4> quantize_matrix_q4(const std::vector<float>& W, int rows, int cols) {
    int blocks_per_row = cols / 32;
    std::vector<BlockQ4> out(static_cast<size_t>(rows) * blocks_per_row);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < blocks_per_row; ++b)
            out[r * blocks_per_row + b] = quantize_q4(&W[(static_cast<size_t>(r) * cols) + b * 32]);
    return out;
}
std::vector<BlockQ8> quantize_matrix_q8(const std::vector<float>& W, int rows, int cols) {
    int blocks_per_row = cols / 32;
    std::vector<BlockQ8> out(static_cast<size_t>(rows) * blocks_per_row);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < blocks_per_row; ++b)
            out[r * blocks_per_row + b] = quantize_q8(&W[(static_cast<size_t>(r) * cols) + b * 32]);
    return out;
}

// Row-major matmul: out[j] = sum_i x[i] * W[j*in_dim + i] -- Chapter 3.2's matmul.
void matmul(std::span<float> out, std::span<const float> x, std::span<const float> W, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; ++i) sum += x[i] * W[j * in_dim + i];
        out[j] = sum;
    }
}
// Same matmul, but reading each output row from quantized blocks instead
// of an FP32 matrix -- the "hybrid" half of this comparison.
void matmul_q4(std::span<float> out, const float* x, const std::vector<BlockQ4>& W, int in_dim, int out_dim) {
    int bpr = in_dim / 32;
    for (int j = 0; j < out_dim; ++j)
        out[j] = dot_q4(std::span<const BlockQ4>(&W[j * bpr], bpr), x);
}
void matmul_q8(std::span<float> out, const float* x, const std::vector<BlockQ8>& W, int in_dim, int out_dim) {
    int bpr = in_dim / 32;
    for (int j = 0; j < out_dim; ++j)
        out[j] = dot_q8(std::span<const BlockQ8>(&W[j * bpr], bpr), x);
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void softmax_inplace(std::span<float> x) {
    float mx = *std::max_element(x.begin(), x.end()), s = 0.0f;
    for (float& v : x) { v = std::exp(v - mx); s += v; }
    for (float& v : x) v /= s;
}

float relative_l2_error(std::span<const float> a, std::span<const float> b) {
    float num = 0.0f, den = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) { float d = a[i] - b[i]; num += d * d; den += a[i] * a[i]; }
    return (den > 0.0f) ? std::sqrt(num / den) : 0.0f;
}
// Total variation distance between two probability distributions: half
// the L1 distance, bounded in [0, 1] -- the standard way to measure how
// much probability mass moved between two softmax outputs.
float total_variation(std::span<const float> p, std::span<const float> q) {
    float s = 0.0f;
    for (size_t i = 0; i < p.size(); ++i) s += std::fabs(p[i] - q[i]);
    return 0.5f * s;
}

int main() {
    std::cout << "================================================\n";
    std::cout << "Weight Class Sensitivity: Same Format, Different Consequences\n";
    std::cout << "================================================\n\n";

    constexpr int DIM = 64, D_FF = 256, SEQ_LEN = 8;

    // =====================================================================
    // TEST 1: The SAME format, applied to an "attention-shaped" (DIM x
    // DIM) and an "FFN-shaped" (D_FF x DIM) matrix drawn from the same
    // distribution, produces statistically indistinguishable relative
    // error. The block format does not know or care what role the
    // matrix plays -- only how many bits it spends per weight.
    // =====================================================================
    std::cout << "-- Test 1: Quantization error depends on format, not matrix shape --\n";
    float err_attn_q4 = 0.0f, err_ffn_q4 = 0.0f, err_attn_q8 = 0.0f, err_ffn_q8 = 0.0f;
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);

        std::vector<float> attn_w(DIM * DIM), ffn_w(D_FF * DIM);
        for (float& v : attn_w) v = dist(rng);
        for (float& v : ffn_w) v = dist(rng);

        auto measure_error = [](const std::vector<float>& W, int rows, int cols, bool use_q8) -> float {
            int bpr = cols / 32;
            float num = 0.0f, den = 0.0f;
            float buf[32];
            for (int r = 0; r < rows; ++r) {
                for (int b = 0; b < bpr; ++b) {
                    const float* orig = &W[static_cast<size_t>(r) * cols + b * 32];
                    if (use_q8) { BlockQ8 blk = quantize_q8(orig); dequantize_q8(blk, buf); }
                    else        { BlockQ4 blk = quantize_q4(orig); dequantize_q4(blk, buf); }
                    for (int i = 0; i < 32; ++i) { float d = orig[i] - buf[i]; num += d * d; den += orig[i] * orig[i]; }
                }
            }
            return std::sqrt(num / den);
        };

        err_attn_q4 = measure_error(attn_w, DIM, DIM, false);
        err_ffn_q4  = measure_error(ffn_w, D_FF, DIM, false);
        err_attn_q8 = measure_error(attn_w, DIM, DIM, true);
        err_ffn_q8  = measure_error(ffn_w, D_FF, DIM, true);

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  Q4_0 relative L2 error -- attention-shaped: " << err_attn_q4
                   << ", FFN-shaped: " << err_ffn_q4 << "\n";
        std::cout << "  Q8_0 relative L2 error -- attention-shaped: " << err_attn_q8
                   << ", FFN-shaped: " << err_ffn_q8 << "\n";
        std::cout << "  same-format errors match across shapes (within 10% relative): "
                   << ((std::fabs(err_attn_q4 - err_ffn_q4) / err_attn_q4 < 0.10f) ? "yes" : "no") << "\n";

        CHECK(std::fabs(err_attn_q4 - err_ffn_q4) / err_attn_q4 < 0.10f);
        CHECK(std::fabs(err_attn_q8 - err_ffn_q8) / err_attn_q8 < 0.10f);
        CHECK(err_attn_q8 < err_attn_q4);   // Q8 is unambiguously more precise than Q4
    }

    // =====================================================================
    // TEST 2: Attention's real fragility is a DISCRETE decision, not a
    // bulk distributional shift. Across many independent trials (fresh
    // query and key-source activations each time, same fixed W_k), this
    // measures how often quantizing W_k changes WHICH cached position
    // wins the most softmax weight (argmax) compared to the FP32 result
    // -- the event that actually matters when attention is picking which
    // earlier token to look at. Whenever two positions' scores are
    // nearly tied, Q4's larger raw error is more likely to flip which
    // one wins than Q8's smaller error is.
    // =====================================================================
    std::cout << "\n-- Test 2: Attention's discrete failure mode -- argmax flips --\n";
    float raw_score_relerr_q4 = 0.0f, raw_score_relerr_q8 = 0.0f;
    int flips_q4 = 0, flips_q8 = 0;
    constexpr int N_TRIALS = 500;
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> wdist(0.0f, 0.02f);
        std::normal_distribution<float> xdist(0.0f, 1.0f);

        std::vector<float> W_k(DIM * DIM);
        for (float& v : W_k) v = wdist(rng);
        auto Wk_q4 = quantize_matrix_q4(W_k, DIM, DIM);
        auto Wk_q8 = quantize_matrix_q8(W_k, DIM, DIM);

        float scale = 1.0f / std::sqrt(static_cast<float>(DIM));
        double sum_relerr_q4 = 0.0, sum_relerr_q8 = 0.0;

        for (int trial = 0; trial < N_TRIALS; ++trial) {
            std::vector<float> q(DIM);
            for (float& v : q) v = xdist(rng);

            std::vector<float> scores_fp32(SEQ_LEN), scores_q4(SEQ_LEN), scores_q8(SEQ_LEN);
            for (int t = 0; t < SEQ_LEN; ++t) {
                std::vector<float> x_t(DIM);
                for (float& v : x_t) v = xdist(rng);

                std::vector<float> K_fp32(DIM), K_q4(DIM), K_q8(DIM);
                matmul(K_fp32, x_t, W_k, DIM, DIM);
                matmul_q4(K_q4, x_t.data(), Wk_q4, DIM, DIM);
                matmul_q8(K_q8, x_t.data(), Wk_q8, DIM, DIM);

                scores_fp32[t] = 0.0f; scores_q4[t] = 0.0f; scores_q8[t] = 0.0f;
                for (int i = 0; i < DIM; ++i) {
                    scores_fp32[t] += q[i] * K_fp32[i];
                    scores_q4[t]   += q[i] * K_q4[i];
                    scores_q8[t]   += q[i] * K_q8[i];
                }
                scores_fp32[t] *= scale; scores_q4[t] *= scale; scores_q8[t] *= scale;
            }

            sum_relerr_q4 += relative_l2_error(scores_fp32, scores_q4);
            sum_relerr_q8 += relative_l2_error(scores_fp32, scores_q8);

            auto argmax = [](const std::vector<float>& v) {
                return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
            };
            int am_fp32 = argmax(scores_fp32), am_q4 = argmax(scores_q4), am_q8 = argmax(scores_q8);
            if (am_q4 != am_fp32) ++flips_q4;
            if (am_q8 != am_fp32) ++flips_q8;
        }

        raw_score_relerr_q4 = static_cast<float>(sum_relerr_q4 / N_TRIALS);
        raw_score_relerr_q8 = static_cast<float>(sum_relerr_q8 / N_TRIALS);

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  mean raw score relative error over " << N_TRIALS << " trials -- Q4 keys: "
                   << raw_score_relerr_q4 << ", Q8 keys: " << raw_score_relerr_q8 << "\n";
        std::cout << "  argmax (\"which position wins attention\") flip rate -- Q4 keys: "
                   << flips_q4 << "/" << N_TRIALS << " (" << std::setprecision(1)
                   << (100.0 * flips_q4 / N_TRIALS) << "%), Q8 keys: " << flips_q8 << "/" << N_TRIALS
                   << " (" << (100.0 * flips_q8 / N_TRIALS) << "%)\n";

        CHECK(raw_score_relerr_q8 < raw_score_relerr_q4);
        CHECK(flips_q8 <= flips_q4);
        CHECK(flips_q4 > 0);   // Q4's error is large enough to flip the decision at least sometimes
    }

    // =====================================================================
    // TEST 3: The SAME Q4 format, applied to an FFN gate matrix instead
    // of an attention key matrix, produces a raw output error of the
    // SAME ORDER as attention's mean raw score error (both come from
    // identical Q4 arithmetic against similarly-scaled random weights)
    // -- but the FFN has no discrete decision for that error to flip.
    // SiLU is a smooth, non-competitive elementwise function; its output
    // error stays continuous and close to the input error's own scale,
    // with nothing analogous to Test 2's argmax to disrupt.
    // =====================================================================
    std::cout << "\n-- Test 3: No discrete decision, no flip -- just a continuous error --\n";
    float raw_output_relerr_q4 = 0.0f;
    {
        std::mt19937 rng(99);
        std::normal_distribution<float> wdist(0.0f, 0.02f);
        std::normal_distribution<float> xdist(0.0f, 1.0f);

        std::vector<float> W_gate(D_FF * DIM), W_up(D_FF * DIM), W_down(DIM * D_FF);
        for (float& v : W_gate) v = wdist(rng);
        for (float& v : W_up) v = wdist(rng);
        for (float& v : W_down) v = wdist(rng);
        auto Wgate_q4 = quantize_matrix_q4(W_gate, D_FF, DIM);

        std::vector<float> x(DIM);
        for (float& v : x) v = xdist(rng);

        // FP32 path: every matrix at full precision.
        std::vector<float> gate_fp32(D_FF), up_fp32(D_FF), hidden_fp32(D_FF), out_fp32(DIM);
        matmul(gate_fp32, x, W_gate, DIM, D_FF);
        matmul(up_fp32, x, W_up, DIM, D_FF);
        for (int i = 0; i < D_FF; ++i) hidden_fp32[i] = silu(gate_fp32[i]) * up_fp32[i];
        matmul(out_fp32, hidden_fp32, W_down, D_FF, DIM);

        // Hybrid path: ONLY the gate matrix is Q4 -- isolating the same
        // single quantized matrix that Test 2 isolated for attention.
        std::vector<float> gate_q4(D_FF), up_q4(D_FF), hidden_q4(D_FF), out_q4(DIM);
        matmul_q4(gate_q4, x.data(), Wgate_q4, DIM, D_FF);
        matmul(up_q4, x, W_up, DIM, D_FF);
        for (int i = 0; i < D_FF; ++i) hidden_q4[i] = silu(gate_q4[i]) * up_q4[i];
        matmul(out_q4, hidden_q4, W_down, D_FF, DIM);

        raw_output_relerr_q4 = relative_l2_error(out_fp32, out_q4);

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  FFN output relative error (Q4 gate weights): " << raw_output_relerr_q4 << "\n";
        std::cout << "  attention mean raw score relative error (Q4 key weights, Test 2): " << raw_score_relerr_q4 << "\n";
        std::cout << "  same order of magnitude (within 5x either way): "
                   << ((raw_output_relerr_q4 / raw_score_relerr_q4 > 0.2f &&
                        raw_output_relerr_q4 / raw_score_relerr_q4 < 5.0f) ? "yes" : "no") << "\n";
        std::cout << "  FFN has nothing analogous to Test 2's argmax to flip -- the error above is\n";
        std::cout << "  the entire downstream consequence, whereas attention's same-sized error also\n";
        std::cout << "  produced a " << std::setprecision(1) << (100.0 * flips_q4 / N_TRIALS)
                   << "% chance of silently changing WHICH position the model attends to.\n";

        CHECK(raw_output_relerr_q4 / raw_score_relerr_q4 > 0.2f && raw_output_relerr_q4 / raw_score_relerr_q4 < 5.0f);
    }

    // =====================================================================
    // TEST 4: The element-count argument. Using this chapter's own
    // D_FF = 4*DIM convention, one layer's FFN matrices (gate, up, down)
    // outnumber its attention matrices (Q, K, V, O) three to one -- so
    // aggressive (Q4) quantization there buys the largest absolute
    // memory savings, while attention's smaller footprint means paying
    // for Q8's extra precision costs comparatively little in bytes.
    // =====================================================================
    std::cout << "\n-- Test 4: Element counts per layer -- where the parameters actually live --\n";
    {
        long long attn_elements = 4LL * DIM * DIM;          // Q, K, V, O
        long long ffn_elements = 3LL * DIM * D_FF;           // gate, up, down
        double ratio = static_cast<double>(ffn_elements) / static_cast<double>(attn_elements);

        std::cout << "  attention elements/layer (Q+K+V+O, " << DIM << "x" << DIM << " each): " << attn_elements << "\n";
        std::cout << "  FFN elements/layer (gate+up+down, " << D_FF << "x" << DIM << " each, D_FF=4*DIM): " << ffn_elements << "\n";
        std::cout << "  FFN:attention element ratio: " << ratio << "x\n";
        std::cout << "  (production SwiGLU ratios vary with the FFN multiplier chosen, typically\n";
        std::cout << "   landing somewhere between roughly 2.5x and 4x depending on architecture --\n";
        std::cout << "   this chapter uses a clean 4x for its own worked examples throughout.)\n";

        CHECK(ratio == 3.0);
    }

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_weight_class_quantization_error.cpp -o 01_weight_class_quantization_error
./01_weight_class_quantization_error
```

**Sample input:** an attention-shaped (64x64) and an FFN-shaped (256x64) weight matrix from the same Gaussian distribution, quantized at Q4_0 and Q8_0; a 500-trial Monte Carlo experiment measuring how often Q4_0 vs Q8_0 key-weight quantization flips which cached position a query attends to most; the same Q4_0 format applied to an FFN gate matrix, measured for continuous output error instead; and the FFN:attention element-count ratio implied by this chapter's D_FF = 4 x DIM convention.

```text
================================================
Weight Class Sensitivity: Same Format, Different Consequences
================================================

-- Test 1: Quantization error depends on format, not matrix shape --
  Q4_0 relative L2 error -- attention-shaped: 0.09972, FFN-shaped: 0.09643
  Q8_0 relative L2 error -- attention-shaped: 0.00544, FFN-shaped: 0.00532
  same-format errors match across shapes (within 10% relative): yes

-- Test 2: Attention's discrete failure mode -- argmax flips --
  mean raw score relative error over 500 trials -- Q4 keys: 0.10072, Q8 keys: 0.00555
  argmax ("which position wins attention") flip rate -- Q4 keys: 36/500 (7.2%), Q8 keys: 2/500 (0.4%)

-- Test 3: No discrete decision, no flip -- just a continuous error --
  FFN output relative error (Q4 gate weights): 0.10485
  attention mean raw score relative error (Q4 key weights, Test 2): 0.10072
  same order of magnitude (within 5x either way): yes
  FFN has nothing analogous to Test 2's argmax to flip -- the error above is
  the entire downstream consequence, whereas attention's same-sized error also
  produced a 7.2% chance of silently changing WHICH position the model attends to.

-- Test 4: Element counts per layer -- where the parameters actually live --
  attention elements/layer (Q+K+V+O, 64x64 each): 16384
  FFN elements/layer (gate+up+down, 256x64 each, D_FF=4*DIM): 49152
  FFN:attention element ratio: 3.0x
  (production SwiGLU ratios vary with the FFN multiplier chosen, typically
   landing somewhere between roughly 2.5x and 4x depending on architecture --
   this chapter uses a clean 4x for its own worked examples throughout.)

================================================
8/8 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] assuming a bigger measured shift means amplification"
    This section's own first draft measured the total-variation distance between the FP32 and Q4-quantized softmax distributions, expecting a small raw score error to come out amplified into a larger distributional shift. It came out SMALLER instead: softmax's outputs live in a bounded probability simplex, which caps how far the bulk distribution can move no matter how large the input perturbation is. The real fragility was invisible in that number because total variation measures the WRONG thing — a small average shift across many possible outcomes, not whether the single outcome that gets acted on (the argmax) actually changed. Measuring the argmax flip rate across many independent trials instead of one distribution's total variation is what actually surfaces the effect this section is about.

## 6.2 A Per-Tensor Quantization Policy, Encoded in the GGUF Type Tag

### Intuition

Section 6.1 established why different weight classes deserve different formats. Nothing new needs to be invented to store that decision: Chapter 5's tensor descriptor already carries a per-tensor type tag, because that is what real GGUF files do — a single file with a norm vector in F32 sitting next to an FFN matrix in Q4_0 is not a special case, it is the normal case. The policy itself is just a small function from a weight's role to a format. The part that has to be gotten right is the READER'S half of the bargain: it must dispatch on the descriptor's own recorded type, tensor by tensor, and never guess a format from the tensor's name.

### The Concept, In Detail

The policy maps four roles to three formats: norm weights to F32 (there are only ever a few tens of thousands of them per model, so F32 costs essentially nothing), attention weights to Q8_0 (Section 6.1's discrete decision to protect), FFN weights to Q4_0 (Section 6.1's bulk of the parameters), and — this is stated here and justified in Section 6.3 — the embedding table also to Q8_0, not F32, because a modern vocabulary makes it far too numerous for F32 to be free. Writing a file under this policy means calling the policy function once per tensor at write time and recording whatever format it returns in that tensor's own descriptor; nothing about the file format changes. Reading it back correctly means writing a dispatcher that switches on `TensorInfo::type` and calls the matching dequantization or dot-product routine — never a dispatcher that pattern-matches the tensor's NAME. The risk in doing the latter is not hypothetical: Chapter 5.3's own naming convention gives every layer an "ffn_norm" tensor, the normalization weight applied before the FFN block, which the policy above correctly assigns to F32 despite the substring "ffn" appearing in its name. A dispatcher that trusts the name would guess Q4_0 for it and reinterpret its true F32 bytes as fp16-scaled 4-bit nibbles — not a rounding error, a structurally wrong reinterpretation of bytes that happen to mean something completely different.

### Code and Verification

```cpp
// Chapter 6.2 -- Section 6.1 established WHY different weight classes
// deserve different formats. This section builds the mechanism: a
// per-tensor quantization POLICY (a small function mapping a weight's
// role -- norm, embedding, attention, FFN -- to a GGUF quantization
// type), and a GGUF file that actually uses it, writing some tensors as
// F32, some as Q8_0, and some as Q4_0 in the SAME file. Nothing new has
// to be invented to store this: Chapter 5's tensor descriptor already
// carries a per-tensor type tag (GGMLType), because that is exactly
// what GGUF real files do -- a single file with blk.0.attn_norm.weight
// in F32 sitting next to blk.0.ffn_gate.weight in Q4_0 is not a special
// case, it is the normal case.
//
// The embedding table gets Q8_0 here, not F32 -- Section 6.3 will show
// why: a large vocabulary makes the embedding table numerous enough
// that F32 stops being "free" the way it is for the genuinely tiny
// per-layer norm vectors. Only NORM stays at F32 in this chapter's
// policy; EMBEDDING joins ATTENTION at Q8_0.
//
// The reader's job is to dispatch on that recorded type, tensor by
// tensor, never to assume a format from the tensor's name. This section
// verifies why that distinction matters with a real GGUF naming
// collision: Chapter 5.3 established that every transformer layer has
// an "ffn_norm" tensor -- the normalization weight applied before the
// FFN block, which is almost always stored at full F32 precision
// despite "ffn" appearing in its name. A dispatcher that guesses the
// format from a substring match on the name would misclassify it as
// Q4_0 and try to reinterpret its raw F32 bytes as fp16-scaled 4-bit
// nibbles -- structurally nonsense output, not just numerically
// imprecise output. A dispatcher that reads the descriptor's own
// recorded type gets it right regardless of what the name says.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <span>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include <unistd.h>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// -- Chapter 4.2/4.3's fp16_t and block layouts. --
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
        int lo = static_cast<int>(std::clamp(std::round(data[2*i] * inv), -8.0f, 7.0f));
        int hi = static_cast<int>(std::clamp(std::round(data[2*i+1] * inv), -8.0f, 7.0f));
        b.nibbles[i] = static_cast<uint8_t>((static_cast<uint8_t>(hi + 8) << 4) | static_cast<uint8_t>(lo + 8));
    }
    return b;
}
float dot_q8(std::span<const BlockQ8> blocks, const float* x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 32; ++i) bs += static_cast<float>(blocks[b].weights[i]) * x[b * 32 + i];
        total += bs * s;
    }
    return total;
}
float dot_q4(std::span<const BlockQ4> blocks, const float* x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 16; ++i) {
            uint8_t p = blocks[b].nibbles[i];
            bs += static_cast<float>(static_cast<int>(p & 0xF) - 8) * x[b * 32 + 2 * i];
            bs += static_cast<float>(static_cast<int>(p >> 4) - 8) * x[b * 32 + 2 * i + 1];
        }
        total += bs * s;
    }
    return total;
}
float dot_fp32_row(const float* w, const float* x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += w[i] * x[i];
    return s;
}

// -- Chapter 5.2's GGUF constants, writer, and reader (repeated per this
// book's one-file-per-section convention). --
static constexpr uint32_t GGUF_MAGIC = 0x46475547;
static constexpr uint32_t GGUF_VERSION = 3;
enum GGUFType : uint32_t { T_STRING = 8, T_UINT32 = 4 };
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q4_0 = 2, GGML_Q8_0 = 7 };

static const char* ggml_type_name(uint32_t t) {
    switch (t) { case 0: return "F32"; case 2: return "Q4_0"; case 7: return "Q8_0"; default: return "???"; }
}

class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* data, size_t size) { out.write(reinterpret_cast<const char*>(data), size); pos += size; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_string(const std::string& s) { uint64_t len = s.size(); write_raw(&len, 8); write_raw(s.data(), s.size()); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(T_STRING); write_string(v); }
    void write_kv_u32(const std::string& k, uint32_t v) { write_string(k); write_u32(T_UINT32); write_u32(v); }
    void write_tensor_info(const std::string& name, uint32_t n_dims, const uint64_t* dims, GGMLType type, uint64_t offset) {
        write_string(name);
        write_u32(n_dims);
        for (uint32_t i = 0; i < n_dims; ++i) write_u64(dims[i]);
        write_u32(static_cast<uint32_t>(type));
        write_u64(offset);
    }
    void align(size_t alignment) {
        size_t rem = pos % alignment;
        if (rem != 0) { std::vector<char> zeros(alignment - rem, 0); write_raw(zeros.data(), zeros.size()); }
    }
    void write_bytes(const void* data, size_t size) { write_raw(data, size); }
    size_t tell() const { return pos; }
    bool good() const { return out.good(); }
};

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset;
    uint64_t n_elements;
    uint64_t data_size;

    uint64_t compute_data_size() const {
        switch (type) {
            case GGML_F32:  return n_elements * 4;
            case GGML_Q8_0: return (n_elements / 32) * sizeof(BlockQ8);
            case GGML_Q4_0: return (n_elements / 32) * sizeof(BlockQ4);
            default: return 0;
        }
    }
};

using MetaValue = std::variant<std::string, uint32_t>;

class GGUFReader {
    std::ifstream in;
    void read_raw(void* data, size_t size) { in.read(reinterpret_cast<char*>(data), size); }
public:
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    std::unordered_map<std::string, MetaValue> metadata;
    std::vector<TensorInfo> tensors;
    std::vector<uint8_t> data_section;   // everything from data_section_offset to EOF, loaded once
    size_t data_section_offset = 0;

    bool open(const std::string& path) {
        in.open(path, std::ios::binary);
        if (!in.is_open()) return false;

        read_raw(&magic, 4);
        if (magic != GGUF_MAGIC) return false;
        read_raw(&version, 4);
        read_raw(&n_tensors, 8);
        read_raw(&n_kv, 8);

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string();
            uint32_t type; read_raw(&type, 4);
            switch (type) {
                case T_STRING: metadata[key] = read_string(); break;
                case T_UINT32: { uint32_t v; read_raw(&v, 4); metadata[key] = v; break; }
                default: return false;
            }
        }

        tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            auto& t = tensors[i];
            t.name = read_string();
            read_raw(&t.n_dims, 4);
            t.dims.resize(t.n_dims);
            for (uint32_t d = 0; d < t.n_dims; ++d) read_raw(&t.dims[d], 8);
            read_raw(&t.type, 4);
            read_raw(&t.offset, 8);
            t.n_elements = 1;
            for (auto d : t.dims) t.n_elements *= d;
            t.data_size = t.compute_data_size();
        }

        size_t current_pos = static_cast<size_t>(in.tellg());
        size_t rem = current_pos % 32;
        data_section_offset = (rem == 0) ? current_pos : current_pos + (32 - rem);

        in.seekg(0, std::ios::end);
        size_t file_size = static_cast<size_t>(in.tellg());
        data_section.resize(file_size - data_section_offset);
        in.seekg(static_cast<std::streamoff>(data_section_offset));
        read_raw(data_section.data(), data_section.size());
        return true;
    }

    std::string read_string() {
        uint64_t len; read_raw(&len, 8);
        std::string s(len, '\0'); read_raw(s.data(), len); return s;
    }
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
    const uint8_t* tensor_data(const TensorInfo& t) const { return data_section.data() + t.offset; }
};

// -- The quantization policy: role -> format --
enum class WeightRole { NORM, EMBEDDING, ATTENTION, FFN };

GGMLType pick_format(WeightRole role) {
    switch (role) {
        case WeightRole::NORM:      return GGML_F32;    // tens of thousands of elements -- F32 is free
        case WeightRole::EMBEDDING: return GGML_Q8_0;    // Section 6.3: large vocabularies are NOT free at F32
        case WeightRole::ATTENTION: return GGML_Q8_0;    // Section 6.1: a discrete decision to protect
        case WeightRole::FFN:       return GGML_Q4_0;    // Section 6.1: the bulk of the parameters
    }
    return GGML_F32;
}
const char* role_name(WeightRole role) {
    switch (role) {
        case WeightRole::NORM: return "NORM";
        case WeightRole::EMBEDDING: return "EMBEDDING";
        case WeightRole::ATTENTION: return "ATTENTION";
        case WeightRole::FFN: return "FFN";
    }
    return "?";
}

// -- The dispatcher: reads TensorInfo::type, never the tensor's name. --
// Computes tensor . x for one output row (row_offset selects which row
// of a multi-row tensor's blocks/floats to use), regardless of format.
float dot_generic_row(const GGUFReader& reader, const TensorInfo& t, int row, int row_width, const float* x) {
    const uint8_t* base = reader.tensor_data(t);
    switch (t.type) {
        case GGML_F32: {
            const float* row_ptr = reinterpret_cast<const float*>(base) + static_cast<size_t>(row) * row_width;
            return dot_fp32_row(row_ptr, x, row_width);
        }
        case GGML_Q8_0: {
            int bpr = row_width / 32;
            auto* blocks = reinterpret_cast<const BlockQ8*>(base) + static_cast<size_t>(row) * bpr;
            return dot_q8(std::span<const BlockQ8>(blocks, bpr), x);
        }
        case GGML_Q4_0: {
            int bpr = row_width / 32;
            auto* blocks = reinterpret_cast<const BlockQ4*>(base) + static_cast<size_t>(row) * bpr;
            return dot_q4(std::span<const BlockQ4>(blocks, bpr), x);
        }
        default: return 0.0f;
    }
}

int main() {
    std::cout << "================================================\n";
    std::cout << "A Per-Tensor Quantization Policy, Encoded in the GGUF Type Tag\n";
    std::cout << "================================================\n\n";

    constexpr int DIM = 64, D_FF = 256;
    const std::string path = "/tmp/ch6_hybrid_policy.gguf";

    // =====================================================================
    // TEST 1: The policy itself -- a pure function from role to format.
    // =====================================================================
    std::cout << "-- Test 1: The quantization policy --\n";
    {
        WeightRole roles[] = {WeightRole::NORM, WeightRole::EMBEDDING, WeightRole::ATTENTION, WeightRole::FFN};
        for (auto r : roles)
            std::cout << "  " << std::left << std::setw(10) << role_name(r) << std::right
                       << " -> " << ggml_type_name(pick_format(r)) << "\n";
        CHECK(pick_format(WeightRole::NORM) == GGML_F32);
        CHECK(pick_format(WeightRole::EMBEDDING) == GGML_Q8_0);
        CHECK(pick_format(WeightRole::ATTENTION) == GGML_Q8_0);
        CHECK(pick_format(WeightRole::FFN) == GGML_Q4_0);
    }

    // =====================================================================
    // TEST 2: Write one hybrid GGUF file -- six tensors, three roles'
    // worth of formats, chosen by the policy above at write time. Real
    // GGUF files from llama.cpp do exactly this: every tensor carries
    // its own type tag, independent of every other tensor's.
    // =====================================================================
    std::cout << "\n-- Test 2: Writing a hybrid file (F32 + Q8_0 + Q4_0 in one file) --\n";
    std::vector<float> attn_norm_data(DIM), ffn_norm_data(DIM);
    std::vector<float> wq_data(DIM * DIM), wk_data(DIM * DIM);
    std::vector<float> wgate_data(D_FF * DIM), wup_data(D_FF * DIM);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> ndist(0.0f, 0.02f);
        std::normal_distribution<float> normw_dist(1.0f, 0.01f);   // norm weights cluster near 1.0
        for (float& v : attn_norm_data) v = normw_dist(rng);
        for (float& v : ffn_norm_data) v = normw_dist(rng);
        for (float& v : wq_data) v = ndist(rng);
        for (float& v : wk_data) v = ndist(rng);
        for (float& v : wgate_data) v = ndist(rng);
        for (float& v : wup_data) v = ndist(rng);

        auto wq_q8 = std::vector<BlockQ8>((DIM * DIM) / 32);
        for (size_t b = 0; b < wq_q8.size(); ++b) wq_q8[b] = quantize_q8(&wq_data[b * 32]);
        auto wk_q8 = std::vector<BlockQ8>((DIM * DIM) / 32);
        for (size_t b = 0; b < wk_q8.size(); ++b) wk_q8[b] = quantize_q8(&wk_data[b * 32]);
        auto wgate_q4 = std::vector<BlockQ4>((D_FF * DIM) / 32);
        for (size_t b = 0; b < wgate_q4.size(); ++b) wgate_q4[b] = quantize_q4(&wgate_data[b * 32]);
        auto wup_q4 = std::vector<BlockQ4>((D_FF * DIM) / 32);
        for (size_t b = 0; b < wup_q4.size(); ++b) wup_q4[b] = quantize_q4(&wup_data[b * 32]);

        size_t off = 0;
        size_t off_attn_norm = off; off += attn_norm_data.size() * 4;
        size_t off_ffn_norm  = off; off += ffn_norm_data.size() * 4;
        size_t off_wq        = off; off += wq_q8.size() * sizeof(BlockQ8);
        size_t off_wk        = off; off += wk_q8.size() * sizeof(BlockQ8);
        size_t off_wgate     = off; off += wgate_q4.size() * sizeof(BlockQ4);
        size_t off_wup       = off; off += wup_q4.size() * sizeof(BlockQ4);

        GGUFWriter writer(path);
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(6);   // n_tensors
        writer.write_u64(1);   // n_kv
        writer.write_kv_string("general.architecture", "llama");

        uint64_t d1[] = {static_cast<uint64_t>(DIM)};
        writer.write_tensor_info("blk.0.attn_norm.weight", 1, d1, pick_format(WeightRole::NORM), off_attn_norm);
        writer.write_tensor_info("blk.0.ffn_norm.weight", 1, d1, pick_format(WeightRole::NORM), off_ffn_norm);
        uint64_t d2[] = {static_cast<uint64_t>(DIM), static_cast<uint64_t>(DIM)};
        writer.write_tensor_info("blk.0.attn_q.weight", 2, d2, pick_format(WeightRole::ATTENTION), off_wq);
        writer.write_tensor_info("blk.0.attn_k.weight", 2, d2, pick_format(WeightRole::ATTENTION), off_wk);
        uint64_t d3[] = {static_cast<uint64_t>(DIM), static_cast<uint64_t>(D_FF)};
        writer.write_tensor_info("blk.0.ffn_gate.weight", 2, d3, pick_format(WeightRole::FFN), off_wgate);
        writer.write_tensor_info("blk.0.ffn_up.weight", 2, d3, pick_format(WeightRole::FFN), off_wup);

        writer.align(32);
        writer.write_bytes(attn_norm_data.data(), attn_norm_data.size() * 4);
        writer.write_bytes(ffn_norm_data.data(), ffn_norm_data.size() * 4);
        writer.write_bytes(wq_q8.data(), wq_q8.size() * sizeof(BlockQ8));
        writer.write_bytes(wk_q8.data(), wk_q8.size() * sizeof(BlockQ8));
        writer.write_bytes(wgate_q4.data(), wgate_q4.size() * sizeof(BlockQ4));
        writer.write_bytes(wup_q4.data(), wup_q4.size() * sizeof(BlockQ4));

        std::cout << "  wrote 6 tensors: 2x NORM (F32), 2x ATTENTION (Q8_0), 2x FFN (Q4_0)\n";
        CHECK(writer.good());
    }

    // =====================================================================
    // TEST 3: Read the file back and dispatch on each tensor's OWN
    // recorded type -- never on its name -- against a genuine FP32
    // reference for every tensor, of every format, in the file.
    // =====================================================================
    std::cout << "\n-- Test 3: Reading back and dispatching by recorded type --\n";
    GGUFReader reader;
    {
        bool ok = reader.open(path);
        CHECK(ok);
        CHECK(reader.tensors.size() == 6);

        for (const auto& t : reader.tensors)
            std::cout << "  " << std::left << std::setw(24) << t.name << std::right
                       << ggml_type_name(t.type) << "  n_elements=" << t.n_elements << "\n";

        std::mt19937 rng(7);
        std::normal_distribution<float> xdist(0.0f, 1.0f);
        std::vector<float> x(DIM);
        for (float& v : x) v = xdist(rng);

        auto* attn_norm_t = reader.find_tensor("blk.0.attn_norm.weight");
        auto* wq_t = reader.find_tensor("blk.0.attn_q.weight");
        auto* wgate_t = reader.find_tensor("blk.0.ffn_gate.weight");
        CHECK(attn_norm_t != nullptr && wq_t != nullptr && wgate_t != nullptr);
        CHECK(attn_norm_t->type == GGML_F32);
        CHECK(wq_t->type == GGML_Q8_0);
        CHECK(wgate_t->type == GGML_Q4_0);

        // Row 0 of each multi-row tensor, dispatched generically by type.
        float attn_norm_val = dot_generic_row(reader, *attn_norm_t, 0, DIM, x.data());   // whole vector as "row 0"
        float attn_norm_ref = dot_fp32_row(attn_norm_data.data(), x.data(), DIM);
        float wq_row0 = dot_generic_row(reader, *wq_t, 0, DIM, x.data());
        float wq_row0_ref = dot_fp32_row(&wq_data[0], x.data(), DIM);
        float wgate_row0 = dot_generic_row(reader, *wgate_t, 0, DIM, x.data());
        float wgate_row0_ref = dot_fp32_row(&wgate_data[0], x.data(), DIM);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  NORM (F32) dispatch:      " << attn_norm_val << " vs FP32 ref " << attn_norm_ref
                   << " (exact: " << (attn_norm_val == attn_norm_ref ? "yes" : "no") << ")\n";
        std::cout << "  ATTENTION (Q8_0) dispatch: " << wq_row0 << " vs FP32 ref " << wq_row0_ref
                   << " (error " << std::fabs(wq_row0 - wq_row0_ref) << ")\n";
        std::cout << "  FFN (Q4_0) dispatch:      " << wgate_row0 << " vs FP32 ref " << wgate_row0_ref
                   << " (error " << std::fabs(wgate_row0 - wgate_row0_ref) << ")\n";

        CHECK(attn_norm_val == attn_norm_ref);   // F32 dispatch is bit-exact, not merely close
        CHECK(std::fabs(wq_row0 - wq_row0_ref) < 0.05f);
        CHECK(std::fabs(wgate_row0 - wgate_row0_ref) < 0.05f);
    }

    // =====================================================================
    // TEST 4: [COMMON TRAP] a name-based dispatcher misclassifies
    // "ffn_norm.weight" -- a real GGUF tensor (Chapter 5.3) that is
    // norm-role (F32) despite "ffn" appearing in its name. Reinterpreting
    // its true F32 bytes as a BlockQ4 (fp16 scale + 4-bit nibbles)
    // produces structurally wrong values, not just imprecise ones --
    // proof that dispatch must read the descriptor's own type field.
    // =====================================================================
    std::cout << "\n-- Test 4: [COMMON TRAP] dispatch-by-name misreads ffn_norm.weight --\n";
    {
        auto* ffn_norm_t = reader.find_tensor("blk.0.ffn_norm.weight");
        CHECK(ffn_norm_t != nullptr);
        CHECK(ffn_norm_t->type == GGML_F32);   // correctly recorded as F32 by the writer's policy

        // A naive dispatcher that guesses format from a substring match
        // on the name: sees "ffn" and assumes Q4_0, exactly the mistake
        // this section warns against.
        bool name_says_q4 = ffn_norm_t->name.find("ffn") != std::string::npos;
        std::cout << "  tensor name: \"" << ffn_norm_t->name << "\"\n";
        std::cout << "  recorded type (correct):    " << ggml_type_name(ffn_norm_t->type) << "\n";
        std::cout << "  name-substring guess:       " << (name_says_q4 ? "Q4_0 (WRONG)" : "F32") << "\n";

        // Reinterpret the SAME true-F32 bytes as if they were Q4_0
        // blocks -- what the naive dispatcher would actually do.
        const uint8_t* base = reader.tensor_data(*ffn_norm_t);
        int bpr = DIM / 32;
        auto* misread_blocks = reinterpret_cast<const BlockQ4*>(base);
        std::vector<float> misread(DIM);
        for (int b = 0; b < bpr; ++b) {
            float s = static_cast<float>(misread_blocks[b].scale);
            for (int i = 0; i < 16; ++i) {
                uint8_t p = misread_blocks[b].nibbles[i];
                misread[b * 32 + 2 * i]     = static_cast<float>(static_cast<int>(p & 0xF) - 8) * s;
                misread[b * 32 + 2 * i + 1] = static_cast<float>(static_cast<int>(p >> 4) - 8) * s;
            }
        }

        float true_relerr = 0.0f;
        {
            float num = 0.0f, den = 0.0f;
            for (int i = 0; i < DIM; ++i) {
                float d = ffn_norm_data[i] - misread[i];
                num += d * d; den += ffn_norm_data[i] * ffn_norm_data[i];
            }
            true_relerr = std::sqrt(num / den);
        }
        std::cout << "  true value[0..3]:    " << ffn_norm_data[0] << ", " << ffn_norm_data[1]
                   << ", " << ffn_norm_data[2] << ", " << ffn_norm_data[3] << "\n";
        std::cout << "  misread as Q4_0[0..3]: " << misread[0] << ", " << misread[1]
                   << ", " << misread[2] << ", " << misread[3] << "\n";
        std::cout << "  relative L2 error from misreading the format: " << true_relerr
                   << " (not a rounding error -- structurally wrong bytes)\n";

        CHECK(name_says_q4);           // the naive heuristic really would get this one wrong
        CHECK(true_relerr > 1.0f);     // misreading the format, not merely quantizing it, is catastrophic
    }

    unlink(path.c_str());

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_gguf_type_dispatch.cpp -o 02_gguf_type_dispatch
./02_gguf_type_dispatch
```

**Sample input:** a small hybrid GGUF file with six tensors — two norm vectors (F32), two attention weight matrices (Q8_0), and two FFN weight matrices (Q4_0) — written under the policy above, read back, and dispatched on by recorded type; then a deliberate demonstration of what a name-based dispatcher would do to the file's "ffn_norm.weight" tensor.

```text
================================================
A Per-Tensor Quantization Policy, Encoded in the GGUF Type Tag
================================================

-- Test 1: The quantization policy --
  NORM       -> F32
  EMBEDDING  -> Q8_0
  ATTENTION  -> Q8_0
  FFN        -> Q4_0

-- Test 2: Writing a hybrid file (F32 + Q8_0 + Q4_0 in one file) --
  wrote 6 tensors: 2x NORM (F32), 2x ATTENTION (Q8_0), 2x FFN (Q4_0)

-- Test 3: Reading back and dispatching by recorded type --
  blk.0.attn_norm.weight  F32  n_elements=64
  blk.0.ffn_norm.weight   F32  n_elements=64
  blk.0.attn_q.weight     Q8_0  n_elements=4096
  blk.0.attn_k.weight     Q8_0  n_elements=4096
  blk.0.ffn_gate.weight   Q4_0  n_elements=16384
  blk.0.ffn_up.weight     Q4_0  n_elements=16384
  NORM (F32) dispatch:      -9.221472 vs FP32 ref -9.221472 (exact: yes)
  ATTENTION (Q8_0) dispatch: -0.008333 vs FP32 ref -0.010321 (error 0.001988)
  FFN (Q4_0) dispatch:      0.003833 vs FP32 ref 0.027152 (error 0.023319)

-- Test 4: [COMMON TRAP] dispatch-by-name misreads ffn_norm.weight --
  tensor name: "blk.0.ffn_norm.weight"
  recorded type (correct):    F32
  name-substring guess:       Q4_0 (WRONG)
  true value[0..3]:    0.992979, 0.982336, 0.985402, 1.001003
  misread as Q4_0[0..3]: 1.472168, -0.245361, 1.717529, -1.226807
  relative L2 error from misreading the format: 7.220647 (not a rounding error -- structurally wrong bytes)

================================================
18/18 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] guessing a tensor's format from its name"
    A dispatcher that checks whether a tensor's name contains the substring "ffn" and assumes Q4_0 on a match gets `blk.0.ffn_norm.weight` wrong — that tensor is genuinely F32 despite its name, because it is a norm vector that happens to sit before the FFN block, not an FFN weight matrix itself. Reinterpreting its true F32 bytes as Q4_0 blocks (an fp16 scale followed by packed 4-bit nibbles) does not fail loudly: it produces values that are numerically plausible-looking floats, just completely wrong ones, with a relative error over 7 — not a quantization rounding error, a wholesale misreading of what the bytes mean. The descriptor's own `type` field is the only source of truth; a tensor's name is a human-readable label, never a format specification.

## 6.3 The Memory Budget: Uniform vs. Hybrid Quantization

### Intuition

A hybrid policy is only worth adopting if it is cheap. This section prices Section 6.2's policy in real bytes for a large-scale model, using Chapter 5.3's own tensor-count formula and Section 6.1's own per-layer element-count formulas — and it prices the WRONG policy first, on purpose, because that is what actually happened while writing this chapter: the first version of Section 6.2's policy kept both norm weights and the embedding table at F32, on the reasoning that both are "small and sensitive." That reasoning holds for norm weights, which really are small — tens of thousands of elements regardless of model size. It does not hold for the embedding table, whose size scales with VOCABULARY, not model depth, and a modern large-vocabulary embedding table is tens to hundreds of millions of elements, not tens of thousands.

### The Concept, In Detail

Plugging illustrative large-model dimensions into Chapter 5.3's "3 + L x 9" tensor-count formula and Section 6.1's element-count formulas gives a total parameter count in the single-digit billions, split across four roles: embedding, norm, attention, and FFN. Computing total bytes under the NAIVE policy (F32 for both norm and embedding) against uniform Q4_0 shows an 86% overhead — nearly doubling the size a fully Q4_0 model would have been, for a policy that was supposed to be a minor refinement. Breaking that overhead down by role shows almost all of it (essentially the entire difference) comes from the embedding table, not the norm weights, which really do stay negligible under any format. The corrected policy — Section 6.2's actual policy — moves the embedding table to Q8_0 alongside attention, leaving only the genuinely tiny norm weights at F32, and the overhead against uniform Q4_0 drops from 86% to under 30%, while still saving roughly a third of the bytes uniform Q8_0 would have cost. The lesson is not "protect small, sensitive tensors" — it is "protect tensors that are small in POPULATION, and check that the population is actually small before assuming it," because a role can look conceptually similar to another (both "not weight matrices," both "touched by every forward pass") while having a wildly different element count.

### Code and Verification

```cpp
// Chapter 6.3 -- Section 6.2 settled on a policy: norm weights at F32,
// embedding and attention at Q8_0, FFN at Q4_0. This section is where
// that policy actually came from -- not a guess, but the result of
// computing the byte cost of the more obvious-looking policy (norm AND
// embedding both at F32) and finding it did not hold up.
//
// The first version of this section gave embedding weights the same
// treatment as norm weights: both "small, sensitive, keep them exact."
// That reasoning is right for norm weights, which number in the tens of
// thousands of elements per model and cost nothing extra at F32 no
// matter what. It is wrong for the embedding table, whose element count
// scales with VOCABULARY SIZE, not with model depth -- and a modern
// large-vocabulary embedding table is tens to hundreds of millions of
// elements, not tens of thousands. Running the numbers below with
// embeddings at F32 produced a hybrid file costing 86% more than
// uniform Q4_0 -- most of a Q4_0-sized model's memory budget again, just
// for keeping one table exact. Section 6.2's actual policy (embedding
// at Q8_0, only norm at F32) is the corrected version, verified here by
// computing both and showing which one the numbers actually support.

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

int main() {
    std::cout << "================================================\n";
    std::cout << "The Memory Budget: Uniform vs. Hybrid Quantization\n";
    std::cout << "================================================\n\n";

    // Round, illustrative dimensions for a large-scale model -- not a
    // claim about any specific named model's exact published spec, just
    // numbers big enough to make the byte counts meaningful. D_FF here
    // follows this chapter's own 4x-DIM convention from Section 6.1.
    constexpr long long DIM = 4096, N_LAYERS = 32, D_FF = 16384, VOCAB = 128000;

    // =====================================================================
    // TEST 1: Tensor count, cross-checked against Chapter 5.3's own
    // "3 + L*9" formula (3 global tensors, 9 per layer) for continuity.
    // =====================================================================
    std::cout << "-- Test 1: Tensor count (Chapter 5.3's formula, re-verified) --\n";
    {
        constexpr long long GLOBAL_TENSORS = 3;       // token_embd, output_norm, output
        constexpr long long PER_LAYER_TENSORS = 9;    // attn_norm, q, k, v, attn_output, ffn_norm, gate, up, down
        long long total_tensors = GLOBAL_TENSORS + N_LAYERS * PER_LAYER_TENSORS;
        std::cout << "  " << GLOBAL_TENSORS << " global + " << N_LAYERS << " layers x "
                   << PER_LAYER_TENSORS << " per-layer = " << total_tensors << " tensors\n";
        CHECK(total_tensors == 3 + N_LAYERS * 9);
        CHECK(total_tensors == 291);   // same 32-layer model Chapter 5.3 counted
    }

    // =====================================================================
    // TEST 2: Element counts by role, computed from the same per-layer
    // formulas Section 6.1 verified (4*DIM^2 attention, 3*DIM*D_FF FFN),
    // plus the two embedding-space tensors (token_embd, output) and the
    // norm weights (2 per layer, plus one final output_norm).
    // =====================================================================
    std::cout << "\n-- Test 2: Element counts by role --\n";
    long long embedding_elements = 0, norm_elements = 0, attn_elements = 0, ffn_elements = 0, total_elements = 0;
    {
        embedding_elements = 2 * VOCAB * DIM;                    // token_embd.weight + output.weight
        norm_elements = N_LAYERS * 2 * DIM + DIM;                // attn_norm + ffn_norm per layer, + output_norm
        attn_elements = N_LAYERS * 4 * DIM * DIM;                // Q, K, V, O per layer
        ffn_elements = N_LAYERS * 3 * DIM * D_FF;                // gate, up, down per layer
        total_elements = embedding_elements + norm_elements + attn_elements + ffn_elements;

        double total_b = static_cast<double>(total_elements) / 1e9;
        std::cout << "  dimensions: DIM=" << DIM << " N_LAYERS=" << N_LAYERS
                   << " D_FF=" << D_FF << " VOCAB=" << VOCAB << "\n";
        std::cout << "  EMBEDDING elements (token_embd + output): " << embedding_elements << "\n";
        std::cout << "  NORM elements (attn_norm + ffn_norm + output_norm): " << norm_elements << "\n";
        std::cout << "  ATTENTION elements (Q+K+V+O, all layers): " << attn_elements << "\n";
        std::cout << "  FFN elements (gate+up+down, all layers): " << ffn_elements << "\n";
        std::cout << "  total: " << total_elements << " (" << std::fixed << std::setprecision(2)
                   << total_b << " billion parameters)\n";

        CHECK(total_elements == embedding_elements + norm_elements + attn_elements + ffn_elements);
        CHECK(norm_elements < embedding_elements / 100);   // norm weights are numerically negligible
    }

    // =====================================================================
    // TEST 3: Total bytes under uniform Q4_0, uniform Q8_0, and TWO
    // hybrid variants -- the naive one (F32 for both norm and embedding)
    // and the corrected one from Section 6.2 (F32 for norm only, Q8_0
    // for embedding). Computing both, rather than just asserting the
    // corrected one is better, is the entire point of this section.
    // =====================================================================
    std::cout << "\n-- Test 3: Total weight bytes -- naive vs. corrected hybrid policy --\n";
    double uniform_q4_gb = 0.0, uniform_q8_gb = 0.0, hybrid_naive_gb = 0.0, hybrid_gb = 0.0;
    {
        constexpr double Q4_BPB = 18.0, Q8_BPB = 34.0, BW = 32.0;   // bytes per block, weights per block

        double uniform_q4_bytes = static_cast<double>(total_elements) / BW * Q4_BPB;
        double uniform_q8_bytes = static_cast<double>(total_elements) / BW * Q8_BPB;

        // Naive: norm AND embedding both at F32 -- the "small, sensitive,
        // keep it exact" reasoning applied uniformly, without checking
        // whether "small" actually holds for the embedding table.
        double hybrid_naive_bytes =
            static_cast<double>(norm_elements + embedding_elements) * 4.0 +
            static_cast<double>(attn_elements) / BW * Q8_BPB +
            static_cast<double>(ffn_elements) / BW * Q4_BPB;

        // Corrected (Section 6.2's actual policy): only norm at F32;
        // embedding joins attention at Q8_0.
        double hybrid_bytes =
            static_cast<double>(norm_elements) * 4.0 +
            static_cast<double>(embedding_elements + attn_elements) / BW * Q8_BPB +
            static_cast<double>(ffn_elements) / BW * Q4_BPB;

        uniform_q4_gb = uniform_q4_bytes / 1e9;
        uniform_q8_gb = uniform_q8_bytes / 1e9;
        hybrid_naive_gb = hybrid_naive_bytes / 1e9;
        hybrid_gb = hybrid_bytes / 1e9;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  uniform Q4_0 everywhere:            " << uniform_q4_gb << " GB\n";
        std::cout << "  uniform Q8_0 everywhere:             " << uniform_q8_gb << " GB\n";
        std::cout << "  hybrid, naive (F32 norm+embedding):  " << hybrid_naive_gb << " GB\n";
        std::cout << "  hybrid, corrected (F32 norm only):   " << hybrid_gb << " GB\n";

        double naive_overhead_pct = 100.0 * (hybrid_naive_gb - uniform_q4_gb) / uniform_q4_gb;
        double corrected_overhead_pct = 100.0 * (hybrid_gb - uniform_q4_gb) / uniform_q4_gb;
        double corrected_savings_pct = 100.0 * (uniform_q8_gb - hybrid_gb) / uniform_q8_gb;
        std::cout << std::setprecision(1);
        std::cout << "  naive hybrid costs "     << naive_overhead_pct     << "% more than uniform Q4_0\n";
        std::cout << "  corrected hybrid costs " << corrected_overhead_pct << "% more than uniform Q4_0\n";
        std::cout << "  corrected hybrid saves "  << corrected_savings_pct  << "% versus uniform Q8_0\n";

        CHECK(hybrid_gb < hybrid_naive_gb);       // the correction actually helps
        CHECK(hybrid_gb > uniform_q4_gb);
        CHECK(hybrid_gb < uniform_q8_gb);
        CHECK(naive_overhead_pct > 50.0);         // the naive policy's cost is not a rounding error
        CHECK(corrected_overhead_pct < 40.0);      // the corrected policy stays much closer to uniform Q4_0
    }

    // =====================================================================
    // TEST 4: Where the naive policy's extra bytes actually came from --
    // broken down by role, in absolute GB. The embedding table, not the
    // norm weights, turns out to be almost the entire difference: this
    // is the number that should have been checked before choosing F32
    // for embeddings in the first place.
    // =====================================================================
    std::cout << "\n-- Test 4: The naive policy's overhead, broken down by role --\n";
    {
        constexpr double Q4_BPB = 18.0, Q8_BPB = 34.0, BW = 32.0;

        double norm_as_f32_gb = static_cast<double>(norm_elements) * 4.0 / 1e9;
        double norm_as_q4_gb = static_cast<double>(norm_elements) / BW * Q4_BPB / 1e9;
        double embed_as_f32_gb = static_cast<double>(embedding_elements) * 4.0 / 1e9;
        double embed_as_q8_gb = static_cast<double>(embedding_elements) / BW * Q8_BPB / 1e9;

        double norm_delta = norm_as_f32_gb - norm_as_q4_gb;
        double embed_delta = embed_as_f32_gb - embed_as_q8_gb;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  norm weights:     F32 costs " << norm_as_f32_gb << " GB vs. Q4_0's " << norm_as_q4_gb
                   << " GB -- delta " << norm_delta << " GB (genuinely negligible)\n";
        std::cout << "  embedding table:  F32 costs " << embed_as_f32_gb << " GB vs. Q8_0's " << embed_as_q8_gb
                   << " GB -- delta " << embed_delta << " GB (the actual cost of the naive policy)\n";
        std::cout << "  embedding's share of the naive policy's total overhead vs. corrected: "
                   << std::setprecision(1) << (100.0 * embed_delta / (hybrid_naive_gb - hybrid_gb)) << "%\n";

        CHECK(embed_delta > norm_delta * 100.0);   // the embedding table dwarfs norm weights as a cost driver
        CHECK(embed_delta / (hybrid_naive_gb - hybrid_gb) > 0.9);   // embeddings explain nearly all of it
    }

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_hybrid_memory_budget.cpp -o 03_hybrid_memory_budget
./03_hybrid_memory_budget
```

**Sample input:** illustrative large-model dimensions (4096-dimensional, 32 layers, a 4x feed-forward multiplier, a 128,000-token vocabulary) run through Chapter 5.3's tensor-count formula and this chapter's element-count formulas, computing total bytes under uniform Q4_0, uniform Q8_0, the naive hybrid policy (F32 norm and embedding), and the corrected policy (F32 norm only, Q8_0 embedding).

```text
================================================
The Memory Budget: Uniform vs. Hybrid Quantization
================================================

-- Test 1: Tensor count (Chapter 5.3's formula, re-verified) --
  3 global + 32 layers x 9 per-layer = 291 tensors

-- Test 2: Element counts by role --
  dimensions: DIM=4096 N_LAYERS=32 D_FF=16384 VOCAB=128000
  EMBEDDING elements (token_embd + output): 1048576000
  NORM elements (attn_norm + ffn_norm + output_norm): 266240
  ATTENTION elements (Q+K+V+O, all layers): 2147483648
  FFN elements (gate+up+down, all layers): 6442450944
  total: 9638776832 (9.64 billion parameters)

-- Test 3: Total weight bytes -- naive vs. corrected hybrid policy --
  uniform Q4_0 everywhere:            5.422 GB
  uniform Q8_0 everywhere:             10.241 GB
  hybrid, naive (F32 norm+embedding):  10.101 GB
  hybrid, corrected (F32 norm only):   7.021 GB
  naive hybrid costs 86.3% more than uniform Q4_0
  corrected hybrid costs 29.5% more than uniform Q4_0
  corrected hybrid saves 31.4% versus uniform Q8_0

-- Test 4: The naive policy's overhead, broken down by role --
  norm weights:     F32 costs 0.001 GB vs. Q4_0's 0.000 GB -- delta 0.001 GB (genuinely negligible)
  embedding table:  F32 costs 4.194 GB vs. Q8_0's 1.114 GB -- delta 3.080 GB (the actual cost of the naive policy)
  embedding's share of the naive policy's total overhead vs. corrected: 100.0%

================================================
11/11 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] treating 'small and sensitive' as one category"
    Norm weights and the embedding table both sound like they belong in the same "small, sensitive, keep it exact" bucket — neither is a big square weight matrix, and both feed something that touches every token. But "small" is doing all the work in that sentence, and it is true for one of them and false for the other: norm weights total a few tens of thousands of elements per model regardless of scale, while an embedding table's element count is `vocabulary_size x hidden_dimension`, and a 128,000-token vocabulary at a 4096-dimensional hidden size alone is over half a billion elements. Grouping them by a shared adjective instead of checking their actual element counts is exactly how this chapter's own first-draft policy ended up costing 86% more than uniform Q4_0 — a mistake that a five-line calculation, run before committing to the policy rather than after, would have caught immediately.

## 6.4 A Hybrid Forward Pass, Loaded via mmap From a Real GGUF File

### Intuition

Every section so far has worked with weights generated in-process. This section writes an actual GGUF file to disk under Section 6.2's corrected policy, memory-maps it with Chapter 5.3's zero-copy technique, and runs one complete transformer layer's forward pass against the mapped weights — RMSNorm, RoPE, grouped-query attention with a KV cache, and a SwiGLU FFN, exactly as Chapter 3 built each piece, with the projections now reading fused dot products directly off mmap'd Q8_0 and Q4_0 blocks instead of in-process float arrays. The KV cache stays exactly Chapter 3.4's KVCache, unchanged and still FP32 — TurboQuant is Chapter 7's subject, and using a technique this book has not built yet would be getting ahead of the material.

### The Concept, In Detail

The one genuine wrinkle since Chapter 5.3 is the embedding table's format. Chapter 5.3's `MappedGGUF::get_f32` accessor does not check what format a tensor actually is — it assumes whatever tensor you hand it IS F32 and reinterprets its bytes as floats unconditionally. That was safe in Chapter 5.3 because the embedding table really was F32 there. Section 6.2's policy stores it as Q8_0 instead, so calling `get_f32` on it here — copying Chapter 5.3's lookup code unchanged, which is exactly the kind of mistake copying old code without checking its assumptions produces — does not crash or error. It silently reinterprets Q8_0 blocks (an fp16 scale packed next to 32 signed bytes) as 8 raw floats per block and returns numbers that are wrong by many orders of magnitude, with no indication anything went wrong. The fix is the same lesson Section 6.2 already taught: check the tensor descriptor's own recorded type before choosing an accessor, then call `get_q8` and dequantize the requested row properly. With that fixed, the forward pass itself is a straight assembly of established pieces: an RMSNorm using the mmap'd F32 norm weights, Q/K/V/O projections as fused Q8_0 dot products (with W_k and W_v genuinely narrower than W_q and W_o, the real GQA weight-count savings from Chapter 3.4), RoPE applied per head, insertion into Chapter 3.4's KVCache, attention, a residual connection, and a SwiGLU FFN built from fused Q4_0 dot products — all reading directly from mapped pages, with zero heap bytes allocated for any weight tensor.

### Code and Verification

```cpp
// Chapter 6.4 -- everything so far in this chapter has worked with
// weights generated in-process. This section writes a real hybrid GGUF
// file to disk -- ten tensors, three formats, following Section 6.2's
// policy exactly (F32 norms, Q8_0 embedding and attention, Q4_0 FFN) --
// memory-maps it with Chapter 5.3's zero-copy technique, and runs one
// full transformer layer's forward pass against the mapped weights:
// RMSNorm, RoPE, grouped-query attention with a genuine (F32, not yet
// TurboQuant) KV cache, and a SwiGLU FFN, exactly as Chapter 3 built
// each piece. TurboQuant is deliberately NOT here -- Chapter 7 covers
// it, and bolting it on early would mean using a technique this book
// has not built yet. This section's KV cache is Chapter 3.4's KVCache,
// unchanged, growing one position at a time as tokens are processed.
//
// The one new wrinkle since Chapter 5.3 is the embedding table's
// format. Chapter 5.3's MappedGGUF::get_f32 assumes whatever tensor you
// hand it IS F32 -- it does not check the descriptor's own type field,
// it just reinterprets the tensor's bytes as floats. That was safe in
// Chapter 5.3 because token_embd.weight really was F32 there. Section
// 6.2's policy stores it as Q8_0 instead (large vocabularies are not
// free at F32 -- Section 6.3), so calling get_f32 on it here silently
// reinterprets Q8_0 blocks (an fp16 scale packed next to 32 int8
// weights) as 8 raw floats -- no crash, no error, just wrong numbers.
// This section verifies that mistake directly before doing the lookup
// correctly, which is the entire lesson of Section 6.2 restated as
// code: check the descriptor's type field before choosing an accessor,
// never assume a format from what a previous chapter happened to use.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <span>
#include <algorithm>
#include <random>
#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <mdspan/mdspan.hpp>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// -- Chapter 4.2/4.3's fp16_t and block layouts --
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
        int lo = static_cast<int>(std::clamp(std::round(data[2*i] * inv), -8.0f, 7.0f));
        int hi = static_cast<int>(std::clamp(std::round(data[2*i+1] * inv), -8.0f, 7.0f));
        b.nibbles[i] = static_cast<uint8_t>((static_cast<uint8_t>(hi + 8) << 4) | static_cast<uint8_t>(lo + 8));
    }
    return b;
}
float dot_q8(std::span<const BlockQ8> blocks, const float* x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 32; ++i) bs += static_cast<float>(blocks[b].weights[i]) * x[b * 32 + i];
        total += bs * s;
    }
    return total;
}
float dot_q4(std::span<const BlockQ4> blocks, const float* x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 16; ++i) {
            uint8_t p = blocks[b].nibbles[i];
            bs += static_cast<float>(static_cast<int>(p & 0xF) - 8) * x[b * 32 + 2 * i];
            bs += static_cast<float>(static_cast<int>(p >> 4) - 8) * x[b * 32 + 2 * i + 1];
        }
        total += bs * s;
    }
    return total;
}
void matmul_q8(std::span<float> out, const float* x, std::span<const BlockQ8> W, int in_dim, int out_dim) {
    int bpr = in_dim / 32;
    for (int j = 0; j < out_dim; ++j) out[j] = dot_q8(W.subspan(static_cast<size_t>(j) * bpr, bpr), x);
}
void matmul_q4(std::span<float> out, const float* x, std::span<const BlockQ4> W, int in_dim, int out_dim) {
    int bpr = in_dim / 32;
    for (int j = 0; j < out_dim; ++j) out[j] = dot_q4(W.subspan(static_cast<size_t>(j) * bpr, bpr), x);
}
// Dequantize row `row` of a Q8_0-packed matrix into `out` (row_width
// floats) -- a lookup, not a dot product, for embedding table access.
void dequant_q8_row(std::span<const BlockQ8> blocks, int row, int row_width, float* out) {
    int bpr = row_width / 32;
    auto row_blocks = blocks.subspan(static_cast<size_t>(row) * bpr, bpr);
    for (int b = 0; b < bpr; ++b) {
        float s = static_cast<float>(row_blocks[b].scale);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(row_blocks[b].weights[i]) * s;
    }
}

// -- Chapter 3.1's RMSNorm, 3.2's SwiGLU, 3.3's RoPE, 3.4's GQA + KVCache --
void rms_norm(std::span<float> out, std::span<const float> x, std::span<const float> w, float eps = 1e-6f) {
    float ss = 0.0f;
    for (float v : x) ss += v * v;
    float inv = 1.0f / std::sqrt(ss / static_cast<float>(x.size()) + eps);
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * inv * w[i];
}
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

struct RoPETables {
    std::vector<float> cos_vals, sin_vals;
    int half_dim;
    RoPETables(int seq_len, int head_dim, float base = 10000.0f) : half_dim(head_dim / 2) {
        cos_vals.resize(seq_len * half_dim); sin_vals.resize(seq_len * half_dim);
        for (int pos = 0; pos < seq_len; ++pos)
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * k) / head_dim);
                float angle = static_cast<float>(pos) * theta;
                cos_vals[pos * half_dim + k] = std::cos(angle);
                sin_vals[pos * half_dim + k] = std::sin(angle);
            }
    }
    float cos_at(int pos, int k) const { return cos_vals[pos * half_dim + k]; }
    float sin_at(int pos, int k) const { return sin_vals[pos * half_dim + k]; }
};
void apply_rope(std::span<float> vec, int pos, const RoPETables& t) {
    for (int k = 0; k < t.half_dim; ++k) {
        float x1 = vec[2*k], x2 = vec[2*k+1];
        float c = t.cos_at(pos, k), s = t.sin_at(pos, k);
        vec[2*k] = x1*c - x2*s; vec[2*k+1] = x1*s + x2*c;
    }
}

void softmax_inplace(std::span<float> x) {
    float mx = *std::max_element(x.begin(), x.end()), s = 0.0f;
    for (float& v : x) { v = std::exp(v - mx); s += v; }
    for (float& v : x) v /= s;
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
        auto ks = k_at(h, t); auto vs = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { ks[i] = k[i]; vs[i] = v[i]; }
    }
};
void gqa_attention(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                    int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len);
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        const float* q = q_heads.data() + h * head_dim;
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            float d = 0.0f; for (int i = 0; i < head_dim; ++i) d += q[i] * k[i];
            scores[t] = d * scale;
        }
        softmax_inplace(std::span<float>(scores.data(), seq_len));
        float* out = output.data() + h * head_dim;
        std::fill(out, out + head_dim, 0.0f);
        for (int t = 0; t < seq_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            float w = scores[t];
            for (int i = 0; i < head_dim; ++i) out[i] += w * v[i];
        }
    }
}

// -- GGUF writing (Chapter 5.2's writer) and mmap reading (5.3's MappedGGUF) --
static constexpr uint32_t GGUF_MAGIC = 0x46475547;
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q4_0 = 2, GGML_Q8_0 = 7 };
static const char* ggml_type_name(uint32_t t) {
    switch (t) { case 0: return "F32"; case 2: return "Q4_0"; case 7: return "Q8_0"; default: return "???"; }
}

class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* d, size_t n) { out.write(reinterpret_cast<const char*>(d), n); pos += n; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_string(const std::string& s) { uint64_t n = s.size(); write_raw(&n, 8); write_raw(s.data(), n); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(8); write_string(v); }
    void write_tensor_info(const std::string& name, std::vector<uint64_t> dims, GGMLType type, uint64_t offset) {
        write_string(name); write_u32(static_cast<uint32_t>(dims.size()));
        for (auto d : dims) write_u64(d);
        write_u32(static_cast<uint32_t>(type)); write_u64(offset);
    }
    void align(size_t a) { size_t r = pos % a; if (r) { std::vector<char> z(a - r, 0); write_raw(z.data(), z.size()); } }
    void write_bytes(const void* d, size_t n) { write_raw(d, n); }
    size_t tell() const { return pos; }
    bool good() const { return out.good(); }
};

struct TensorDesc { std::string name; uint32_t type; uint64_t offset; uint64_t n_elements; };
struct MappedGGUF {
    int fd = -1;
    void* mapped = MAP_FAILED;
    size_t file_size = 0, data_offset = 0;
    std::vector<TensorDesc> tensors;
    std::unordered_map<std::string, size_t> tensor_index;

    bool open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st);
        file_size = static_cast<size_t>(st.st_size);
        mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) return false;

        const uint8_t* p = static_cast<const uint8_t*>(mapped);
        size_t pos = 0;
        auto r32 = [&]() -> uint32_t { uint32_t v; std::memcpy(&v, p + pos, 4); pos += 4; return v; };
        auto r64 = [&]() -> uint64_t { uint64_t v; std::memcpy(&v, p + pos, 8); pos += 8; return v; };
        auto rstr = [&]() -> std::string { uint64_t l = r64(); std::string s(reinterpret_cast<const char*>(p + pos), l); pos += l; return s; };

        if (r32() != GGUF_MAGIC) return false;
        r32();   // version
        uint64_t nt = r64(), nkv = r64();
        for (uint64_t i = 0; i < nkv; ++i) { rstr(); uint32_t t = r32(); if (t == 8) rstr(); else return false; }

        tensors.resize(nt);
        for (uint64_t i = 0; i < nt; ++i) {
            auto& t = tensors[i];
            t.name = rstr();
            uint32_t nd = r32();
            t.n_elements = 1;
            for (uint32_t d = 0; d < nd; ++d) t.n_elements *= r64();
            t.type = r32(); t.offset = r64();
            tensor_index[t.name] = i;
        }
        size_t rem = pos % 32;
        data_offset = (rem == 0) ? pos : pos + (32 - rem);
        return true;
    }
    const TensorDesc* find(const std::string& name) const {
        auto it = tensor_index.find(name);
        return (it == tensor_index.end()) ? nullptr : &tensors[it->second];
    }
    // -- Chapter 5.3's accessors: each ASSUMES its own format. Neither
    // checks the descriptor's recorded type -- that check is the
    // CALLER's job, which is exactly what Test 3 below demonstrates. --
    std::span<const float> get_f32(const std::string& name) const {
        auto* t = find(name); if (!t) return {};
        const void* p = static_cast<const uint8_t*>(mapped) + data_offset + t->offset;
        return std::span<const float>(reinterpret_cast<const float*>(p), t->n_elements);
    }
    std::span<const BlockQ8> get_q8(const std::string& name) const {
        auto* t = find(name); if (!t) return {};
        const void* p = static_cast<const uint8_t*>(mapped) + data_offset + t->offset;
        return std::span<const BlockQ8>(reinterpret_cast<const BlockQ8*>(p), t->n_elements / 32);
    }
    std::span<const BlockQ4> get_q4(const std::string& name) const {
        auto* t = find(name); if (!t) return {};
        const void* p = static_cast<const uint8_t*>(mapped) + data_offset + t->offset;
        return std::span<const BlockQ4>(reinterpret_cast<const BlockQ4*>(p), t->n_elements / 32);
    }
    ~MappedGGUF() { if (mapped != MAP_FAILED) munmap(mapped, file_size); if (fd >= 0) ::close(fd); }
};

int main() {
    std::cout << "================================================\n";
    std::cout << "A Hybrid Forward Pass, Loaded via mmap From a Real GGUF File\n";
    std::cout << "================================================\n\n";

    constexpr int DIM = 64, D_FF = 256, VOCAB = 256;
    constexpr int N_HEADS_Q = 4, N_HEADS_KV = 2, GROUP = 2, HEAD_DIM = DIM / N_HEADS_Q;   // HEAD_DIM=16
    constexpr int KV_DIM = N_HEADS_KV * HEAD_DIM;    // 32 -- narrower than DIM, the GQA savings
    constexpr int SEQ_LEN = 8;
    const std::string path = "/tmp/ch6_hybrid_layer.gguf";

    // =====================================================================
    // TEST 1: Write the hybrid GGUF file -- ten tensors, three formats,
    // exactly Section 6.2's policy (F32 norms, Q8_0 embedding+attention,
    // Q4_0 FFN). W_k and W_v are narrower than W_q/W_o (KV_DIM rows
    // instead of DIM), the real GQA weight-count savings from Chapter 3.4.
    // =====================================================================
    std::cout << "-- Test 1: Writing the hybrid layer --\n";
    std::vector<float> emb_data(VOCAB * DIM), attn_norm_data(DIM), ffn_norm_data(DIM);
    std::vector<float> wq_data(DIM * DIM), wk_data(KV_DIM * DIM), wv_data(KV_DIM * DIM), wo_data(DIM * DIM);
    std::vector<float> wgate_data(D_FF * DIM), wup_data(D_FF * DIM), wdown_data(DIM * D_FF);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> wdist(0.0f, 0.02f);
        std::normal_distribution<float> normw_dist(1.0f, 0.01f);
        for (float& v : emb_data) v = wdist(rng);
        for (float& v : attn_norm_data) v = normw_dist(rng);
        for (float& v : ffn_norm_data) v = normw_dist(rng);
        for (float& v : wq_data) v = wdist(rng);
        for (float& v : wk_data) v = wdist(rng);
        for (float& v : wv_data) v = wdist(rng);
        for (float& v : wo_data) v = wdist(rng);
        for (float& v : wgate_data) v = wdist(rng);
        for (float& v : wup_data) v = wdist(rng);
        for (float& v : wdown_data) v = wdist(rng);

        auto to_q8 = [](const std::vector<float>& src) {
            std::vector<BlockQ8> out(src.size() / 32);
            for (size_t b = 0; b < out.size(); ++b) out[b] = quantize_q8(&src[b * 32]);
            return out;
        };
        auto to_q4 = [](const std::vector<float>& src) {
            std::vector<BlockQ4> out(src.size() / 32);
            for (size_t b = 0; b < out.size(); ++b) out[b] = quantize_q4(&src[b * 32]);
            return out;
        };
        auto emb_q8 = to_q8(emb_data);
        auto wq_q8 = to_q8(wq_data), wk_q8 = to_q8(wk_data), wv_q8 = to_q8(wv_data), wo_q8 = to_q8(wo_data);
        auto wgate_q4 = to_q4(wgate_data), wup_q4 = to_q4(wup_data), wdown_q4 = to_q4(wdown_data);

        size_t off = 0;
        auto place = [&](size_t bytes) { size_t o = off; off += bytes; return o; };
        size_t off_emb = place(emb_q8.size() * sizeof(BlockQ8));
        size_t off_an = place(attn_norm_data.size() * 4);
        size_t off_fn = place(ffn_norm_data.size() * 4);
        size_t off_wq = place(wq_q8.size() * sizeof(BlockQ8));
        size_t off_wk = place(wk_q8.size() * sizeof(BlockQ8));
        size_t off_wv = place(wv_q8.size() * sizeof(BlockQ8));
        size_t off_wo = place(wo_q8.size() * sizeof(BlockQ8));
        size_t off_wg = place(wgate_q4.size() * sizeof(BlockQ4));
        size_t off_wu = place(wup_q4.size() * sizeof(BlockQ4));
        size_t off_wd = place(wdown_q4.size() * sizeof(BlockQ4));

        GGUFWriter w(path);
        w.write_u32(GGUF_MAGIC); w.write_u32(3);
        w.write_u64(10);   // n_tensors
        w.write_u64(1);    // n_kv
        w.write_kv_string("general.architecture", "llama");

        w.write_tensor_info("token_embd.weight", {DIM, VOCAB}, GGML_Q8_0, off_emb);
        w.write_tensor_info("blk.0.attn_norm.weight", {DIM}, GGML_F32, off_an);
        w.write_tensor_info("blk.0.ffn_norm.weight", {DIM}, GGML_F32, off_fn);
        w.write_tensor_info("blk.0.attn_q.weight", {DIM, DIM}, GGML_Q8_0, off_wq);
        w.write_tensor_info("blk.0.attn_k.weight", {DIM, KV_DIM}, GGML_Q8_0, off_wk);
        w.write_tensor_info("blk.0.attn_v.weight", {DIM, KV_DIM}, GGML_Q8_0, off_wv);
        w.write_tensor_info("blk.0.attn_output.weight", {DIM, DIM}, GGML_Q8_0, off_wo);
        w.write_tensor_info("blk.0.ffn_gate.weight", {DIM, D_FF}, GGML_Q4_0, off_wg);
        w.write_tensor_info("blk.0.ffn_up.weight", {DIM, D_FF}, GGML_Q4_0, off_wu);
        w.write_tensor_info("blk.0.ffn_down.weight", {D_FF, DIM}, GGML_Q4_0, off_wd);

        w.align(32);
        w.write_bytes(emb_q8.data(), emb_q8.size() * sizeof(BlockQ8));
        w.write_bytes(attn_norm_data.data(), attn_norm_data.size() * 4);
        w.write_bytes(ffn_norm_data.data(), ffn_norm_data.size() * 4);
        w.write_bytes(wq_q8.data(), wq_q8.size() * sizeof(BlockQ8));
        w.write_bytes(wk_q8.data(), wk_q8.size() * sizeof(BlockQ8));
        w.write_bytes(wv_q8.data(), wv_q8.size() * sizeof(BlockQ8));
        w.write_bytes(wo_q8.data(), wo_q8.size() * sizeof(BlockQ8));
        w.write_bytes(wgate_q4.data(), wgate_q4.size() * sizeof(BlockQ4));
        w.write_bytes(wup_q4.data(), wup_q4.size() * sizeof(BlockQ4));
        w.write_bytes(wdown_q4.data(), wdown_q4.size() * sizeof(BlockQ4));

        std::cout << "  wrote 10 tensors (1 embedding + 2 norm + 4 attention + 3 FFN), "
                   << w.tell() << " bytes\n";
        std::cout << "  W_k/W_v rows: " << KV_DIM << " (GQA-narrowed), W_q/W_o rows: " << DIM << "\n";
        CHECK(w.good());
    }

    // =====================================================================
    // TEST 2: mmap the file and verify every tensor's recorded type
    // matches Section 6.2's policy -- zero read() calls, exactly Chapter
    // 5.3's technique.
    // =====================================================================
    std::cout << "\n-- Test 2: mmap and verify the policy round-tripped --\n";
    MappedGGUF gguf;
    {
        bool ok = gguf.open(path.c_str());
        CHECK(ok);
        CHECK(gguf.tensors.size() == 10);
        auto* emb_t = gguf.find("token_embd.weight");
        auto* an_t = gguf.find("blk.0.attn_norm.weight");
        auto* wq_t = gguf.find("blk.0.attn_q.weight");
        auto* wg_t = gguf.find("blk.0.ffn_gate.weight");
        CHECK(emb_t && emb_t->type == GGML_Q8_0);
        CHECK(an_t && an_t->type == GGML_F32);
        CHECK(wq_t && wq_t->type == GGML_Q8_0);
        CHECK(wg_t && wg_t->type == GGML_Q4_0);
        std::cout << "  file size: " << gguf.file_size << " bytes, mmap'd, tensors: " << gguf.tensors.size() << "\n";
        std::cout << "  token_embd.weight recorded as: " << ggml_type_name(emb_t->type) << "\n";
    }

    // =====================================================================
    // TEST 3: [COMMON TRAP] Chapter 5.3's get_f32, called unchanged on
    // the now-Q8_0 embedding table, does not error -- it reinterprets
    // Q8_0 bytes (fp16 scale + 32 int8 weights) as raw floats and
    // returns structurally wrong values silently. Checking the
    // descriptor's own type field first, and calling get_q8 + a proper
    // dequantized row lookup, is the only fix.
    // =====================================================================
    std::cout << "\n-- Test 3: [COMMON TRAP] get_f32 on a Q8_0 tensor is silently wrong --\n";
    {
        int token = 65;   // 'A'
        std::vector<float> wrong_lookup(DIM), correct_lookup(DIM);

        // The Chapter 5.3 way -- correct THERE, wrong HERE.
        auto emb_as_f32 = gguf.get_f32("token_embd.weight");   // reinterprets Q8_0 bytes as float
        bool got_any_f32_bytes = !emb_as_f32.empty();
        if (got_any_f32_bytes && static_cast<size_t>(token) * DIM + DIM <= emb_as_f32.size())
            for (int i = 0; i < DIM; ++i) wrong_lookup[i] = emb_as_f32[token * DIM + i];

        // The correct way -- check the type, then dequantize the row.
        auto* emb_t = gguf.find("token_embd.weight");
        CHECK(emb_t->type == GGML_Q8_0);
        auto emb_blocks = gguf.get_q8("token_embd.weight");
        dequant_q8_row(emb_blocks, token, DIM, correct_lookup.data());

        float true_relerr = 0.0f;
        {
            float num = 0.0f, den = 0.0f;
            for (int i = 0; i < DIM; ++i) {
                float d = emb_data[token * DIM + i] - correct_lookup[i];
                num += d * d; den += emb_data[token * DIM + i] * emb_data[token * DIM + i];
            }
            true_relerr = std::sqrt(num / den);
        }
        std::cout << "  true embedding[token=65][0..3]:      " << emb_data[token*DIM] << ", "
                   << emb_data[token*DIM+1] << ", " << emb_data[token*DIM+2] << ", " << emb_data[token*DIM+3] << "\n";
        std::cout << "  get_f32 misread [0..3]:              " << wrong_lookup[0] << ", " << wrong_lookup[1]
                   << ", " << wrong_lookup[2] << ", " << wrong_lookup[3] << " (garbage -- wrong format)\n";
        std::cout << "  type-checked get_q8 + dequant [0..3]: " << correct_lookup[0] << ", " << correct_lookup[1]
                   << ", " << correct_lookup[2] << ", " << correct_lookup[3] << "\n";
        std::cout << "  correct lookup's relative L2 error (quantization only): " << true_relerr << "\n";

        CHECK(true_relerr < 0.05f);   // the CORRECT path is just ordinary Q8_0 rounding error
        // The misread path used the wrong element STRIDE (float span
        // sized by n_elements, i.e. VOCAB*DIM assuming 4 bytes/element,
        // but the tensor is really n_elements/32 blocks of 34 bytes) --
        // it does not even land on the right byte range for this token,
        // let alone interpret it correctly.
        CHECK(wrong_lookup != correct_lookup);
    }

    // =====================================================================
    // TEST 4: The full single-layer forward pass -- embedding lookup
    // (Q8_0, dequantized correctly per Test 3), RMSNorm, Q/K/V/O
    // projections (Q8_0 fused dot products against mmap'd blocks), RoPE,
    // GQA attention with a plain FP32 KV cache (Chapter 3.4, unchanged --
    // TurboQuant is Chapter 7), residual, FFN (Q4_0 fused dot products +
    // SwiGLU), second residual. Eight tokens, processed one at a time.
    // =====================================================================
    std::cout << "\n-- Test 4: Full single-layer forward pass, 8 tokens --\n";
    {
        auto emb_blocks = gguf.get_q8("token_embd.weight");
        auto attn_norm_w = gguf.get_f32("blk.0.attn_norm.weight");
        auto ffn_norm_w = gguf.get_f32("blk.0.ffn_norm.weight");
        auto wq_blocks = gguf.get_q8("blk.0.attn_q.weight");
        auto wk_blocks = gguf.get_q8("blk.0.attn_k.weight");
        auto wv_blocks = gguf.get_q8("blk.0.attn_v.weight");
        auto wo_blocks = gguf.get_q8("blk.0.attn_output.weight");
        auto wgate_blocks = gguf.get_q4("blk.0.ffn_gate.weight");
        auto wup_blocks = gguf.get_q4("blk.0.ffn_up.weight");
        auto wdown_blocks = gguf.get_q4("blk.0.ffn_down.weight");

        RoPETables rope(SEQ_LEN, HEAD_DIM);
        KVCache cache(N_HEADS_KV, SEQ_LEN, HEAD_DIM);

        std::string prompt = "Hello!!!";   // exactly SEQ_LEN=8 bytes
        CHECK(prompt.size() == static_cast<size_t>(SEQ_LEN));

        for (int pos = 0; pos < SEQ_LEN; ++pos) {
            int token = static_cast<unsigned char>(prompt[pos]);
            std::vector<float> x(DIM);
            dequant_q8_row(emb_blocks, token, DIM, x.data());

            std::vector<float> xn(DIM);
            rms_norm(xn, x, attn_norm_w);

            std::vector<float> q(DIM), k(KV_DIM), v(KV_DIM);
            matmul_q8(q, xn.data(), wq_blocks, DIM, DIM);
            matmul_q8(k, xn.data(), wk_blocks, DIM, KV_DIM);
            matmul_q8(v, xn.data(), wv_blocks, DIM, KV_DIM);

            for (int h = 0; h < N_HEADS_Q; ++h) apply_rope(std::span<float>(&q[h*HEAD_DIM], HEAD_DIM), pos, rope);
            for (int h = 0; h < N_HEADS_KV; ++h) apply_rope(std::span<float>(&k[h*HEAD_DIM], HEAD_DIM), pos, rope);

            for (int h = 0; h < N_HEADS_KV; ++h)
                cache.store(h, pos, std::span<const float>(&k[h*HEAD_DIM], HEAD_DIM),
                            std::span<const float>(&v[h*HEAD_DIM], HEAD_DIM));

            std::vector<float> attn_out(DIM);
            gqa_attention(q, cache, attn_out, pos + 1, N_HEADS_Q, GROUP);

            std::vector<float> proj(DIM);
            matmul_q8(proj, attn_out.data(), wo_blocks, DIM, DIM);
            for (int i = 0; i < DIM; ++i) x[i] += proj[i];

            std::vector<float> xn2(DIM);
            rms_norm(xn2, x, ffn_norm_w);
            std::vector<float> gate(D_FF), up(D_FF), hidden(D_FF), down(DIM);
            matmul_q4(gate, xn2.data(), wgate_blocks, DIM, D_FF);
            matmul_q4(up, xn2.data(), wup_blocks, DIM, D_FF);
            for (int i = 0; i < D_FF; ++i) hidden[i] = silu(gate[i]) * up[i];
            matmul_q4(down, hidden.data(), wdown_blocks, D_FF, DIM);
            for (int i = 0; i < DIM; ++i) x[i] += down[i];

            float xnorm = 0.0f; for (float v2 : x) xnorm += v2 * v2;
            xnorm = std::sqrt(xnorm);
            if (pos < 3 || pos == SEQ_LEN - 1)
                std::cout << "  token " << pos << " ('" << prompt[pos] << "'): ||output|| = "
                           << std::fixed << std::setprecision(4) << xnorm
                           << " (Q8_0 attn + Q4_0 FFN, mmap'd, FP32 KV cache)\n";
            if (pos == 3) std::cout << "  ...\n";

            CHECK(std::isfinite(xnorm));
            CHECK(xnorm > 0.0f);
        }

        std::cout << "  heap bytes allocated for weight data: 0 (every weight span aliases the mmap'd file)\n";
    }

    unlink(path.c_str());

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_hybrid_forward_pass.cpp -o 04_hybrid_forward_pass
./04_hybrid_forward_pass
```

**Sample input:** a ten-tensor, single-layer GGUF file (an embedding table, two norm vectors, four Q8_0 attention matrices with GQA-narrowed K/V rows, and three Q4_0 FFN matrices) written to `/tmp`, memory-mapped, and run through one full transformer layer's forward pass across an 8-token prompt ("Hello!!!"), printing the output vector's norm at selected positions.

```text
================================================
A Hybrid Forward Pass, Loaded via mmap From a Real GGUF File
================================================

-- Test 1: Writing the hybrid layer --
  wrote 10 tensors (1 embedding + 2 norm + 4 attention + 3 FFN), 59296 bytes
  W_k/W_v rows: 32 (GQA-narrowed), W_q/W_o rows: 64

-- Test 2: mmap and verify the policy round-tripped --
  file size: 59296 bytes, mmap'd, tensors: 10
  token_embd.weight recorded as: Q8_0

-- Test 3: [COMMON TRAP] get_f32 on a Q8_0 tensor is silently wrong --
  true embedding[token=65][0..3]:      -0.000460914, -0.0451334, -0.00238367, 0.00947825
  get_f32 misread [0..3]:              6.58707e-18, -2.2497, -6.86367e+26, -7.88984e+33 (garbage -- wrong format)
  type-checked get_q8 + dequant [0..3]: -0.000413656, -0.0450885, -0.00248194, 0.00951409
  correct lookup's relative L2 error (quantization only): 0.0060154

-- Test 4: Full single-layer forward pass, 8 tokens --
  token 0 ('H'): ||output|| = 0.2766 (Q8_0 attn + Q4_0 FFN, mmap'd, FP32 KV cache)
  token 1 ('e'): ||output|| = 0.2107 (Q8_0 attn + Q4_0 FFN, mmap'd, FP32 KV cache)
  token 2 ('l'): ||output|| = 0.2017 (Q8_0 attn + Q4_0 FFN, mmap'd, FP32 KV cache)
  ...
  token 7 ('!'): ||output|| = 0.1848 (Q8_0 attn + Q4_0 FFN, mmap'd, FP32 KV cache)
  heap bytes allocated for weight data: 0 (every weight span aliases the mmap'd file)

================================================
27/27 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] an accessor that assumes its own format instead of checking"
    `MappedGGUF::get_f32`, `get_q8`, and `get_q4` each reinterpret a tensor's raw bytes according to whichever format their NAME promises — none of them looks at the tensor descriptor's own recorded `type` field to confirm that promise is true. That is fine as long as every caller already knows the right format to ask for, which was true in Chapter 5.3 (the embedding table really was F32 there) and stopped being true the moment Section 6.2 changed the policy. Calling `get_f32("token_embd.weight")` after that change compiles, links, runs, and returns a `std::span<const float>` that looks completely normal — it just contains values reinterpreted from the wrong byte layout, off by up to 33 orders of magnitude in this section's own measurement, with nothing in the type system or the accessor itself to catch it. The fix is never to harden the accessors (they cannot know your intent); it is to check `TensorInfo::type` at the call site before deciding which accessor to call, exactly as Section 6.2's dispatcher already does.

## 6.5 The Complete Hybrid Engine: Multi-Layer, Multi-Token, End to End

### Intuition

This is the payoff the previous four sections built toward: a complete inference engine, wired end to end, reading a real multi-layer GGUF file via `mmap`, running Section 6.2's hybrid policy through every layer, and generating tokens autoregressively. Nothing here is new machinery — Section 6.4's single layer is simply run once per layer, each with its own independent KV cache, and the final layer's output feeds one more genuine architectural step that this section's own experiment shows is not optional: a final RMSNorm before the projection back to vocabulary logits.

### The Concept, In Detail

Extending Section 6.4 to multiple layers means two things have to be gotten right that a single layer does not test. First, every layer needs its OWN `KVCache` instance — sharing one cache across layers would silently let layer 1's keys and values overwrite layer 0's in the same storage, since nothing about `KVCache::store` prevents writing the same (head, position) slot from two unrelated layers. Second, after the last layer's output, a production transformer applies one more RMSNorm (using a final `output_norm.weight`, itself F32 under this chapter's policy, since it is exactly as small as every other norm vector) before the LM head projects the result into logits over the vocabulary. Skipping that step is a real bug, not a stylistic simplification, and this section verifies exactly how real: computing logits from the identical hidden state with and without the final norm shows the bulk next-token distribution barely moving (a total-variation distance under 1%) while the ARGMAX — the token that would actually get sampled at low temperature — changes to a completely different token. That is Section 6.1's own finding, encountered again in a different part of the same engine: a small, easily-overlooked change to a bulk distribution can still flip the one discrete decision that matters. With both of those handled, generating text is autoregressive sampling exactly as it sounds: a two-token prompt seeds the engine, each position's final-layer output produces logits, temperature-scaled softmax sampling (from a fixed-seed generator, so the run is reproducible) picks the next token, and that token becomes the next position's input. The weights are random, so the output is not real language — but every code path a trained model would exercise runs for real, and the chapter closes with a genuinely computed memory summary: this specific file's actual byte count against the same tensors stored uniformly at F32, a compression ratio computed from real numbers rather than asserted from a table.

### Code and Verification

```cpp
// Chapter 6.5 -- this is the payoff Sections 6.1 through 6.4 built
// toward: a complete inference engine, wired end to end, reading a real
// multi-layer GGUF file via mmap, running Section 6.2's hybrid policy
// through every layer, and generating tokens autoregressively. Nothing
// here is new machinery -- Section 6.4's single layer is simply run
// N_LAYERS times, each with its OWN KV cache (Chapter 3.4's KVCache,
// one instance per layer; sharing a single cache across layers would
// silently overwrite one layer's keys and values with another's), and
// the final layer's output feeds a genuine architectural step this
// section's own COMMON TRAP verifies is not optional: a final RMSNorm
// before the LM head projects back to vocabulary logits. The KV cache
// stays plain FP32 throughout -- Chapter 7 is where TurboQuant replaces
// it, not before.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <span>
#include <algorithm>
#include <random>
#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <mdspan/mdspan.hpp>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// -- Chapter 4.2/4.3's fp16_t and block layouts --
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
        int lo = static_cast<int>(std::clamp(std::round(data[2*i] * inv), -8.0f, 7.0f));
        int hi = static_cast<int>(std::clamp(std::round(data[2*i+1] * inv), -8.0f, 7.0f));
        b.nibbles[i] = static_cast<uint8_t>((static_cast<uint8_t>(hi + 8) << 4) | static_cast<uint8_t>(lo + 8));
    }
    return b;
}
float dot_q8(std::span<const BlockQ8> blocks, const float* x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 32; ++i) bs += static_cast<float>(blocks[b].weights[i]) * x[b * 32 + i];
        total += bs * s;
    }
    return total;
}
float dot_q4(std::span<const BlockQ4> blocks, const float* x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 16; ++i) {
            uint8_t p = blocks[b].nibbles[i];
            bs += static_cast<float>(static_cast<int>(p & 0xF) - 8) * x[b * 32 + 2 * i];
            bs += static_cast<float>(static_cast<int>(p >> 4) - 8) * x[b * 32 + 2 * i + 1];
        }
        total += bs * s;
    }
    return total;
}
void matmul_q8(std::span<float> out, const float* x, std::span<const BlockQ8> W, int in_dim, int out_dim) {
    int bpr = in_dim / 32;
    for (int j = 0; j < out_dim; ++j) out[j] = dot_q8(W.subspan(static_cast<size_t>(j) * bpr, bpr), x);
}
void matmul_q4(std::span<float> out, const float* x, std::span<const BlockQ4> W, int in_dim, int out_dim) {
    int bpr = in_dim / 32;
    for (int j = 0; j < out_dim; ++j) out[j] = dot_q4(W.subspan(static_cast<size_t>(j) * bpr, bpr), x);
}
void dequant_q8_row(std::span<const BlockQ8> blocks, int row, int row_width, float* out) {
    int bpr = row_width / 32;
    auto row_blocks = blocks.subspan(static_cast<size_t>(row) * bpr, bpr);
    for (int b = 0; b < bpr; ++b) {
        float s = static_cast<float>(row_blocks[b].scale);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(row_blocks[b].weights[i]) * s;
    }
}

// -- Chapter 3.1/3.2/3.3/3.4's math kernels --
void rms_norm(std::span<float> out, std::span<const float> x, std::span<const float> w, float eps = 1e-6f) {
    float ss = 0.0f;
    for (float v : x) ss += v * v;
    float inv = 1.0f / std::sqrt(ss / static_cast<float>(x.size()) + eps);
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * inv * w[i];
}
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

struct RoPETables {
    std::vector<float> cos_vals, sin_vals;
    int half_dim;
    RoPETables(int seq_len, int head_dim, float base = 10000.0f) : half_dim(head_dim / 2) {
        cos_vals.resize(seq_len * half_dim); sin_vals.resize(seq_len * half_dim);
        for (int pos = 0; pos < seq_len; ++pos)
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * k) / head_dim);
                float angle = static_cast<float>(pos) * theta;
                cos_vals[pos * half_dim + k] = std::cos(angle);
                sin_vals[pos * half_dim + k] = std::sin(angle);
            }
    }
    float cos_at(int pos, int k) const { return cos_vals[pos * half_dim + k]; }
    float sin_at(int pos, int k) const { return sin_vals[pos * half_dim + k]; }
};
void apply_rope(std::span<float> vec, int pos, const RoPETables& t) {
    for (int k = 0; k < t.half_dim; ++k) {
        float x1 = vec[2*k], x2 = vec[2*k+1];
        float c = t.cos_at(pos, k), s = t.sin_at(pos, k);
        vec[2*k] = x1*c - x2*s; vec[2*k+1] = x1*s + x2*c;
    }
}
void softmax_inplace(std::span<float> x) {
    float mx = *std::max_element(x.begin(), x.end()), s = 0.0f;
    for (float& v : x) { v = std::exp(v - mx); s += v; }
    for (float& v : x) v /= s;
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
        auto ks = k_at(h, t); auto vs = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { ks[i] = k[i]; vs[i] = v[i]; }
    }
};
void gqa_attention(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                    int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len);
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        const float* q = q_heads.data() + h * head_dim;
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            float d = 0.0f; for (int i = 0; i < head_dim; ++i) d += q[i] * k[i];
            scores[t] = d * scale;
        }
        softmax_inplace(std::span<float>(scores.data(), seq_len));
        float* out = output.data() + h * head_dim;
        std::fill(out, out + head_dim, 0.0f);
        for (int t = 0; t < seq_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            float w = scores[t];
            for (int i = 0; i < head_dim; ++i) out[i] += w * v[i];
        }
    }
}

// -- GGUF writing and mmap reading (Chapter 5.2/5.3) --
static constexpr uint32_t GGUF_MAGIC = 0x46475547;
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q4_0 = 2, GGML_Q8_0 = 7 };
static const char* ggml_type_name(uint32_t t) {
    switch (t) { case 0: return "F32"; case 2: return "Q4_0"; case 7: return "Q8_0"; default: return "???"; }
}
class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* d, size_t n) { out.write(reinterpret_cast<const char*>(d), n); pos += n; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_string(const std::string& s) { uint64_t n = s.size(); write_raw(&n, 8); write_raw(s.data(), n); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(8); write_string(v); }
    void write_tensor_info(const std::string& name, std::vector<uint64_t> dims, GGMLType type, uint64_t offset) {
        write_string(name); write_u32(static_cast<uint32_t>(dims.size()));
        for (auto d : dims) write_u64(d);
        write_u32(static_cast<uint32_t>(type)); write_u64(offset);
    }
    void align(size_t a) { size_t r = pos % a; if (r) { std::vector<char> z(a - r, 0); write_raw(z.data(), z.size()); } }
    void write_bytes(const void* d, size_t n) { write_raw(d, n); }
    size_t tell() const { return pos; }
    bool good() const { return out.good(); }
};
struct TensorDesc { std::string name; uint32_t type; uint64_t offset; uint64_t n_elements; };
struct MappedGGUF {
    int fd = -1;
    void* mapped = MAP_FAILED;
    size_t file_size = 0, data_offset = 0;
    std::vector<TensorDesc> tensors;
    std::unordered_map<std::string, size_t> tensor_index;

    bool open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st);
        file_size = static_cast<size_t>(st.st_size);
        mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) return false;
        const uint8_t* p = static_cast<const uint8_t*>(mapped);
        size_t pos = 0;
        auto r32 = [&]() -> uint32_t { uint32_t v; std::memcpy(&v, p + pos, 4); pos += 4; return v; };
        auto r64 = [&]() -> uint64_t { uint64_t v; std::memcpy(&v, p + pos, 8); pos += 8; return v; };
        auto rstr = [&]() -> std::string { uint64_t l = r64(); std::string s(reinterpret_cast<const char*>(p + pos), l); pos += l; return s; };
        if (r32() != GGUF_MAGIC) return false;
        r32();
        uint64_t nt = r64(), nkv = r64();
        for (uint64_t i = 0; i < nkv; ++i) { rstr(); uint32_t t = r32(); if (t == 8) rstr(); else return false; }
        tensors.resize(nt);
        for (uint64_t i = 0; i < nt; ++i) {
            auto& t = tensors[i];
            t.name = rstr();
            uint32_t nd = r32();
            t.n_elements = 1;
            for (uint32_t d = 0; d < nd; ++d) t.n_elements *= r64();
            t.type = r32(); t.offset = r64();
            tensor_index[t.name] = i;
        }
        size_t rem = pos % 32;
        data_offset = (rem == 0) ? pos : pos + (32 - rem);
        return true;
    }
    const TensorDesc* find(const std::string& name) const {
        auto it = tensor_index.find(name);
        return (it == tensor_index.end()) ? nullptr : &tensors[it->second];
    }
    std::span<const float> get_f32(const std::string& name) const {
        auto* t = find(name); if (!t) return {};
        const void* p = static_cast<const uint8_t*>(mapped) + data_offset + t->offset;
        return std::span<const float>(reinterpret_cast<const float*>(p), t->n_elements);
    }
    std::span<const BlockQ8> get_q8(const std::string& name) const {
        auto* t = find(name); if (!t) return {};
        const void* p = static_cast<const uint8_t*>(mapped) + data_offset + t->offset;
        return std::span<const BlockQ8>(reinterpret_cast<const BlockQ8*>(p), t->n_elements / 32);
    }
    std::span<const BlockQ4> get_q4(const std::string& name) const {
        auto* t = find(name); if (!t) return {};
        const void* p = static_cast<const uint8_t*>(mapped) + data_offset + t->offset;
        return std::span<const BlockQ4>(reinterpret_cast<const BlockQ4*>(p), t->n_elements / 32);
    }
    ~MappedGGUF() { if (mapped != MAP_FAILED) munmap(mapped, file_size); if (fd >= 0) ::close(fd); }
};

std::string tname(int layer, const char* suffix) { return "blk." + std::to_string(layer) + "." + suffix; }

int main() {
    std::cout << "================================================\n";
    std::cout << "The Complete Hybrid Engine: Multi-Layer, Multi-Token, End to End\n";
    std::cout << "================================================\n\n";

    constexpr int DIM = 64, D_FF = 256, VOCAB = 256, N_LAYERS = 3;
    constexpr int N_HEADS_Q = 4, N_HEADS_KV = 2, GROUP = 2, HEAD_DIM = DIM / N_HEADS_Q;
    constexpr int KV_DIM = N_HEADS_KV * HEAD_DIM;
    constexpr int SEQ_LEN = 8;
    const std::string path = "/tmp/ch6_full_engine.gguf";

    // =====================================================================
    // TEST 1: Write a complete N_LAYERS-layer GGUF file -- 3 global
    // tensors (embedding, final norm, LM head) plus 9 per layer, all
    // formatted by Section 6.2's policy. 3 + 3*9 = 30 tensors.
    // =====================================================================
    std::cout << "-- Test 1: Writing a " << N_LAYERS << "-layer hybrid model --\n";
    std::mt19937 wgen(42);
    std::normal_distribution<float> wdist(0.0f, 0.02f);
    std::normal_distribution<float> normw_dist(1.0f, 0.01f);

    auto rand_vec = [&](size_t n) { std::vector<float> v(n); for (float& x : v) x = wdist(wgen); return v; };
    auto rand_norm_vec = [&](size_t n) { std::vector<float> v(n); for (float& x : v) x = normw_dist(wgen); return v; };
    auto to_q8 = [](const std::vector<float>& src) {
        std::vector<BlockQ8> out(src.size() / 32);
        for (size_t b = 0; b < out.size(); ++b) out[b] = quantize_q8(&src[b * 32]);
        return out;
    };
    auto to_q4 = [](const std::vector<float>& src) {
        std::vector<BlockQ4> out(src.size() / 32);
        for (size_t b = 0; b < out.size(); ++b) out[b] = quantize_q4(&src[b * 32]);
        return out;
    };

    std::vector<float> emb_data = rand_vec(static_cast<size_t>(VOCAB) * DIM);
    std::vector<float> out_norm_data = rand_norm_vec(DIM);
    std::vector<float> output_data = rand_vec(static_cast<size_t>(VOCAB) * DIM);
    std::vector<std::vector<float>> attn_norm_data(N_LAYERS), ffn_norm_data(N_LAYERS);
    std::vector<std::vector<float>> wq_data(N_LAYERS), wk_data(N_LAYERS), wv_data(N_LAYERS), wo_data(N_LAYERS);
    std::vector<std::vector<float>> wgate_data(N_LAYERS), wup_data(N_LAYERS), wdown_data(N_LAYERS);
    for (int L = 0; L < N_LAYERS; ++L) {
        attn_norm_data[L] = rand_norm_vec(DIM);
        ffn_norm_data[L] = rand_norm_vec(DIM);
        wq_data[L] = rand_vec(static_cast<size_t>(DIM) * DIM);
        wk_data[L] = rand_vec(static_cast<size_t>(KV_DIM) * DIM);
        wv_data[L] = rand_vec(static_cast<size_t>(KV_DIM) * DIM);
        wo_data[L] = rand_vec(static_cast<size_t>(DIM) * DIM);
        wgate_data[L] = rand_vec(static_cast<size_t>(D_FF) * DIM);
        wup_data[L] = rand_vec(static_cast<size_t>(D_FF) * DIM);
        wdown_data[L] = rand_vec(static_cast<size_t>(DIM) * D_FF);
    }

    size_t total_file_bytes = 0;
    {
        auto emb_q8 = to_q8(emb_data);
        auto output_q8 = to_q8(output_data);
        std::vector<std::vector<BlockQ8>> wq_q8(N_LAYERS), wk_q8(N_LAYERS), wv_q8(N_LAYERS), wo_q8(N_LAYERS);
        std::vector<std::vector<BlockQ4>> wgate_q4(N_LAYERS), wup_q4(N_LAYERS), wdown_q4(N_LAYERS);
        for (int L = 0; L < N_LAYERS; ++L) {
            wq_q8[L] = to_q8(wq_data[L]); wk_q8[L] = to_q8(wk_data[L]);
            wv_q8[L] = to_q8(wv_data[L]); wo_q8[L] = to_q8(wo_data[L]);
            wgate_q4[L] = to_q4(wgate_data[L]); wup_q4[L] = to_q4(wup_data[L]); wdown_q4[L] = to_q4(wdown_data[L]);
        }

        size_t off = 0;
        auto place = [&](size_t bytes) { size_t o = off; off += bytes; return o; };
        size_t off_emb = place(emb_q8.size() * sizeof(BlockQ8));
        size_t off_outnorm = place(out_norm_data.size() * 4);
        size_t off_output = place(output_q8.size() * sizeof(BlockQ8));
        std::vector<size_t> off_an(N_LAYERS), off_fn(N_LAYERS), off_wq(N_LAYERS), off_wk(N_LAYERS),
            off_wv(N_LAYERS), off_wo(N_LAYERS), off_wg(N_LAYERS), off_wu(N_LAYERS), off_wd(N_LAYERS);
        for (int L = 0; L < N_LAYERS; ++L) {
            off_an[L] = place(attn_norm_data[L].size() * 4);
            off_fn[L] = place(ffn_norm_data[L].size() * 4);
            off_wq[L] = place(wq_q8[L].size() * sizeof(BlockQ8));
            off_wk[L] = place(wk_q8[L].size() * sizeof(BlockQ8));
            off_wv[L] = place(wv_q8[L].size() * sizeof(BlockQ8));
            off_wo[L] = place(wo_q8[L].size() * sizeof(BlockQ8));
            off_wg[L] = place(wgate_q4[L].size() * sizeof(BlockQ4));
            off_wu[L] = place(wup_q4[L].size() * sizeof(BlockQ4));
            off_wd[L] = place(wdown_q4[L].size() * sizeof(BlockQ4));
        }

        GGUFWriter w(path);
        w.write_u32(GGUF_MAGIC); w.write_u32(3);
        w.write_u64(static_cast<uint64_t>(3 + N_LAYERS * 9));
        w.write_u64(1);
        w.write_kv_string("general.architecture", "llama");

        w.write_tensor_info("token_embd.weight", {DIM, VOCAB}, GGML_Q8_0, off_emb);
        w.write_tensor_info("output_norm.weight", {DIM}, GGML_F32, off_outnorm);
        w.write_tensor_info("output.weight", {DIM, VOCAB}, GGML_Q8_0, off_output);
        for (int L = 0; L < N_LAYERS; ++L) {
            w.write_tensor_info(tname(L, "attn_norm.weight"), {DIM}, GGML_F32, off_an[L]);
            w.write_tensor_info(tname(L, "ffn_norm.weight"), {DIM}, GGML_F32, off_fn[L]);
            w.write_tensor_info(tname(L, "attn_q.weight"), {DIM, DIM}, GGML_Q8_0, off_wq[L]);
            w.write_tensor_info(tname(L, "attn_k.weight"), {DIM, KV_DIM}, GGML_Q8_0, off_wk[L]);
            w.write_tensor_info(tname(L, "attn_v.weight"), {DIM, KV_DIM}, GGML_Q8_0, off_wv[L]);
            w.write_tensor_info(tname(L, "attn_output.weight"), {DIM, DIM}, GGML_Q8_0, off_wo[L]);
            w.write_tensor_info(tname(L, "ffn_gate.weight"), {DIM, D_FF}, GGML_Q4_0, off_wg[L]);
            w.write_tensor_info(tname(L, "ffn_up.weight"), {DIM, D_FF}, GGML_Q4_0, off_wu[L]);
            w.write_tensor_info(tname(L, "ffn_down.weight"), {D_FF, DIM}, GGML_Q4_0, off_wd[L]);
        }

        w.align(32);
        w.write_bytes(emb_q8.data(), emb_q8.size() * sizeof(BlockQ8));
        w.write_bytes(out_norm_data.data(), out_norm_data.size() * 4);
        w.write_bytes(output_q8.data(), output_q8.size() * sizeof(BlockQ8));
        for (int L = 0; L < N_LAYERS; ++L) {
            w.write_bytes(attn_norm_data[L].data(), attn_norm_data[L].size() * 4);
            w.write_bytes(ffn_norm_data[L].data(), ffn_norm_data[L].size() * 4);
            w.write_bytes(wq_q8[L].data(), wq_q8[L].size() * sizeof(BlockQ8));
            w.write_bytes(wk_q8[L].data(), wk_q8[L].size() * sizeof(BlockQ8));
            w.write_bytes(wv_q8[L].data(), wv_q8[L].size() * sizeof(BlockQ8));
            w.write_bytes(wo_q8[L].data(), wo_q8[L].size() * sizeof(BlockQ8));
            w.write_bytes(wgate_q4[L].data(), wgate_q4[L].size() * sizeof(BlockQ4));
            w.write_bytes(wup_q4[L].data(), wup_q4[L].size() * sizeof(BlockQ4));
            w.write_bytes(wdown_q4[L].data(), wdown_q4[L].size() * sizeof(BlockQ4));
        }
        total_file_bytes = w.tell();
        std::cout << "  wrote " << (3 + N_LAYERS * 9) << " tensors across " << N_LAYERS
                   << " layers, " << total_file_bytes << " bytes\n";
        CHECK(w.good());
    }

    // =====================================================================
    // TEST 2: mmap the complete file and spot-check tensor count/types.
    // =====================================================================
    std::cout << "\n-- Test 2: mmap the complete model --\n";
    MappedGGUF gguf;
    {
        bool ok = gguf.open(path.c_str());
        CHECK(ok);
        CHECK(gguf.tensors.size() == static_cast<size_t>(3 + N_LAYERS * 9));
        CHECK(gguf.find("token_embd.weight")->type == GGML_Q8_0);
        CHECK(gguf.find("output_norm.weight")->type == GGML_F32);
        CHECK(gguf.find("output.weight")->type == GGML_Q8_0);
        CHECK(gguf.find(tname(1, "ffn_gate.weight"))->type == GGML_Q4_0);
        CHECK(gguf.find(tname(2, "attn_q.weight"))->type == GGML_Q8_0);
        std::cout << "  file size: " << gguf.file_size << " bytes, " << gguf.tensors.size() << " tensors, mmap'd\n";
        std::cout << "  token_embd.weight: " << ggml_type_name(gguf.find("token_embd.weight")->type)
                   << ", output.weight: " << ggml_type_name(gguf.find("output.weight")->type)
                   << ", blk.1.ffn_gate.weight: " << ggml_type_name(gguf.find(tname(1, "ffn_gate.weight"))->type) << "\n";
    }

    // =====================================================================
    // TEST 3: [COMMON TRAP] skipping the final RMSNorm before the LM
    // head is a real architectural bug, not a stylistic choice -- every
    // production transformer applies one. Computing logits with and
    // without it, from the same hidden state, and comparing which token
    // argmax picks, echoes Section 6.1's finding directly: the bulk
    // softmax distribution barely moves, but the discrete sampling
    // decision can flip anyway.
    // =====================================================================
    std::cout << "\n-- Test 3: [COMMON TRAP] the final norm before the LM head is not optional --\n";
    {
        auto out_norm_w = gguf.get_f32("output_norm.weight");
        auto output_blocks = gguf.get_q8("output.weight");
        CHECK(out_norm_w.size() == static_cast<size_t>(DIM));

        std::mt19937 rng(3);
        std::normal_distribution<float> hdist(0.0f, 1.0f);
        std::vector<float> hidden(DIM);
        for (float& v : hidden) v = hdist(rng);

        std::vector<float> logits_correct(VOCAB), logits_skipped(VOCAB);
        std::vector<float> hidden_normed(DIM);
        rms_norm(hidden_normed, hidden, out_norm_w);
        matmul_q8(logits_correct, hidden_normed.data(), output_blocks, DIM, VOCAB);
        matmul_q8(logits_skipped, hidden.data(), output_blocks, DIM, VOCAB);   // BUG: skipped the norm

        std::vector<float> p_correct = logits_correct, p_skipped = logits_skipped;
        softmax_inplace(p_correct); softmax_inplace(p_skipped);
        float tv = 0.0f;
        for (int i = 0; i < VOCAB; ++i) tv += std::fabs(p_correct[i] - p_skipped[i]);
        tv *= 0.5f;

        int argmax_correct = static_cast<int>(std::max_element(p_correct.begin(), p_correct.end()) - p_correct.begin());
        int argmax_skipped = static_cast<int>(std::max_element(p_skipped.begin(), p_skipped.end()) - p_skipped.begin());

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  next-token distribution shift from skipping the final norm (TV): " << tv << "\n";
        std::cout << "  argmax with norm: " << argmax_correct << ", argmax without: " << argmax_skipped
                   << " (" << (argmax_correct == argmax_skipped ? "same" : "DIFFERENT") << ")\n";
        std::cout << "  same lesson as Section 6.1: the bulk distribution barely moves (TV=" << tv
                   << "), but the DISCRETE decision -- which token gets sampled -- changed anyway\n";

        CHECK(argmax_correct != argmax_skipped);   // the discrete decision is what actually broke, per Section 6.1
    }

    // =====================================================================
    // TEST 4: The complete engine -- N_LAYERS layers, each with its OWN
    // KV cache, processing a 2-token prompt and generating 6 more tokens
    // autoregressively. Every projection is a fused dot product against
    // mmap'd Q8_0/Q4_0 blocks; the KV cache is plain FP32 throughout
    // (Chapter 7 replaces this with TurboQuant, not this chapter).
    // =====================================================================
    std::cout << "\n-- Test 4: Full engine -- " << N_LAYERS << " layers, prompt + autoregressive generation --\n";
    {
        auto emb_blocks = gguf.get_q8("token_embd.weight");
        auto out_norm_w = gguf.get_f32("output_norm.weight");
        auto output_blocks = gguf.get_q8("output.weight");

        std::vector<std::span<const float>> attn_norm_w(N_LAYERS), ffn_norm_w(N_LAYERS);
        std::vector<std::span<const BlockQ8>> wq_blocks(N_LAYERS), wk_blocks(N_LAYERS), wv_blocks(N_LAYERS), wo_blocks(N_LAYERS);
        std::vector<std::span<const BlockQ4>> wgate_blocks(N_LAYERS), wup_blocks(N_LAYERS), wdown_blocks(N_LAYERS);
        std::vector<KVCache> caches;
        for (int L = 0; L < N_LAYERS; ++L) {
            attn_norm_w[L] = gguf.get_f32(tname(L, "attn_norm.weight"));
            ffn_norm_w[L] = gguf.get_f32(tname(L, "ffn_norm.weight"));
            wq_blocks[L] = gguf.get_q8(tname(L, "attn_q.weight"));
            wk_blocks[L] = gguf.get_q8(tname(L, "attn_k.weight"));
            wv_blocks[L] = gguf.get_q8(tname(L, "attn_v.weight"));
            wo_blocks[L] = gguf.get_q8(tname(L, "attn_output.weight"));
            wgate_blocks[L] = gguf.get_q4(tname(L, "ffn_gate.weight"));
            wup_blocks[L] = gguf.get_q4(tname(L, "ffn_up.weight"));
            wdown_blocks[L] = gguf.get_q4(tname(L, "ffn_down.weight"));
            caches.emplace_back(N_HEADS_KV, SEQ_LEN, HEAD_DIM);
        }

        RoPETables rope(SEQ_LEN, HEAD_DIM);
        std::string prompt = "Hi";
        std::vector<int> tokens;
        for (char c : prompt) tokens.push_back(static_cast<unsigned char>(c));

        std::mt19937 gen_rng(12345);
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        constexpr float TEMPERATURE = 0.9f;

        std::cout << "  prompt: \"" << prompt << "\", generating " << (SEQ_LEN - static_cast<int>(prompt.size()))
                   << " tokens...\n  output: " << prompt << std::flush;

        for (int pos = 0; pos < SEQ_LEN; ++pos) {
            int token = tokens[pos];
            std::vector<float> x(DIM);
            dequant_q8_row(emb_blocks, token, DIM, x.data());

            for (int L = 0; L < N_LAYERS; ++L) {
                std::vector<float> xn(DIM);
                rms_norm(xn, x, attn_norm_w[L]);

                std::vector<float> q(DIM), k(KV_DIM), v(KV_DIM);
                matmul_q8(q, xn.data(), wq_blocks[L], DIM, DIM);
                matmul_q8(k, xn.data(), wk_blocks[L], DIM, KV_DIM);
                matmul_q8(v, xn.data(), wv_blocks[L], DIM, KV_DIM);

                for (int h = 0; h < N_HEADS_Q; ++h) apply_rope(std::span<float>(&q[h*HEAD_DIM], HEAD_DIM), pos, rope);
                for (int h = 0; h < N_HEADS_KV; ++h) apply_rope(std::span<float>(&k[h*HEAD_DIM], HEAD_DIM), pos, rope);

                for (int h = 0; h < N_HEADS_KV; ++h)
                    caches[L].store(h, pos, std::span<const float>(&k[h*HEAD_DIM], HEAD_DIM),
                                     std::span<const float>(&v[h*HEAD_DIM], HEAD_DIM));

                std::vector<float> attn_out(DIM);
                gqa_attention(q, caches[L], attn_out, pos + 1, N_HEADS_Q, GROUP);

                std::vector<float> proj(DIM);
                matmul_q8(proj, attn_out.data(), wo_blocks[L], DIM, DIM);
                for (int i = 0; i < DIM; ++i) x[i] += proj[i];

                std::vector<float> xn2(DIM);
                rms_norm(xn2, x, ffn_norm_w[L]);
                std::vector<float> gate(D_FF), up(D_FF), hidden(D_FF), down(DIM);
                matmul_q4(gate, xn2.data(), wgate_blocks[L], DIM, D_FF);
                matmul_q4(up, xn2.data(), wup_blocks[L], DIM, D_FF);
                for (int i = 0; i < D_FF; ++i) hidden[i] = silu(gate[i]) * up[i];
                matmul_q4(down, hidden.data(), wdown_blocks[L], D_FF, DIM);
                for (int i = 0; i < DIM; ++i) x[i] += down[i];
            }

            if (pos == static_cast<int>(tokens.size()) - 1) {
                std::vector<float> x_normed(DIM);
                rms_norm(x_normed, x, out_norm_w);
                std::vector<float> logits(VOCAB);
                matmul_q8(logits, x_normed.data(), output_blocks, DIM, VOCAB);
                for (float& l : logits) l /= TEMPERATURE;
                softmax_inplace(logits);

                float r = ud(gen_rng), cs = 0.0f;
                int next = VOCAB - 1;
                for (int vtok = 0; vtok < VOCAB; ++vtok) { cs += logits[vtok]; if (r < cs) { next = vtok; break; } }

                if (pos + 1 < SEQ_LEN) {
                    tokens.push_back(next);
                    if (next >= 32 && next < 127) std::cout << static_cast<char>(next);
                    else std::cout << "[" << next << "]";
                    std::cout << std::flush;
                }
            }
        }
        std::cout << "\n";

        CHECK(static_cast<int>(tokens.size()) == SEQ_LEN);
        std::cout << "  generated " << (SEQ_LEN - static_cast<int>(prompt.size()))
                   << " tokens (random weights -> not real language, but every code path exercised)\n";

        // =================================================================
        // Memory summary -- this ACTUAL file's real byte count against a
        // genuinely computed all-F32 equivalent for the SAME tensors.
        // =================================================================
        long long total_elements = 0;
        for (const auto& t : gguf.tensors) total_elements += static_cast<long long>(t.n_elements);
        size_t all_f32_bytes = static_cast<size_t>(total_elements) * 4;
        double ratio = static_cast<double>(all_f32_bytes) / static_cast<double>(total_file_bytes);

        std::cout << "\n  --- Memory summary ---\n";
        std::cout << "  this model, hybrid GGUF (mmap'd, on disk): " << total_file_bytes << " bytes\n";
        std::cout << "  the same tensors, all at F32:               " << all_f32_bytes << " bytes\n";
        std::cout << std::fixed << std::setprecision(2) << "  compression ratio: " << ratio << "x\n";
        std::cout << "  heap bytes allocated for weight data across all " << N_LAYERS << " layers: 0\n";

        CHECK(ratio > 3.0);
        CHECK(total_file_bytes == gguf.file_size);
    }

    unlink(path.c_str());

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_hybrid_engine_multilayer.cpp -o 05_hybrid_engine_multilayer
./05_hybrid_engine_multilayer
```

**Sample input:** a complete 3-layer, 30-tensor GGUF file (an embedding table, a final norm and LM head, and nine per-layer tensors each) written under Section 6.2's policy, memory-mapped, and run through a full autoregressive generation loop starting from the 2-token prompt "Hi" and generating 6 further tokens, each layer using its own KV cache.

```text
================================================
The Complete Hybrid Engine: Multi-Layer, Multi-Token, End to End
================================================

-- Test 1: Writing a 3-layer hybrid model --
  wrote 30 tensors across 3 layers, 160544 bytes

-- Test 2: mmap the complete model --
  file size: 160544 bytes, 30 tensors, mmap'd
  token_embd.weight: Q8_0, output.weight: Q8_0, blk.1.ffn_gate.weight: Q4_0

-- Test 3: [COMMON TRAP] the final norm before the LM head is not optional --
  next-token distribution shift from skipping the final norm (TV): 0.00628
  argmax with norm: 237, argmax without: 89 (DIFFERENT)
  same lesson as Section 6.1: the bulk distribution barely moves (TV=0.00628), but the DISCRETE decision -- which token gets sampled -- changed anyway

-- Test 4: Full engine -- 3 layers, prompt + autoregressive generation --
  prompt: "Hi", generating 6 tokens...
  output: Hi[238][228]Q!0[10]
  generated 6 tokens (random weights -> not real language, but every code path exercised)

  --- Memory summary ---
  this model, hybrid GGUF (mmap'd, on disk): 160544 bytes
  the same tensors, all at F32:               870144 bytes
  compression ratio: 5.42x
  heap bytes allocated for weight data across all 3 layers: 0

================================================
13/13 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] one shared KV cache across every layer"
    `KVCache::store(head, position, k, v)` has no way to know which LAYER is calling it — it only knows about heads and positions. Passing the SAME `KVCache` instance to every layer's attention call, instead of giving each layer its own instance, means layer 1's key and value vectors at (head 0, position 3) silently overwrite whatever layer 0 stored at that exact same slot, and every later position's attention computation for BOTH layers reads back whichever layer wrote last. Nothing crashes; nothing produces an obviously wrong shape or a NaN. The output is simply wrong in a way that only shows up as degraded quality on real weights — and would be invisible on this section's own random weights, since "wrong attention over random keys" looks exactly as random as "correct attention over random keys." Allocating one `KVCache` per layer, up front, before the generation loop starts, is the only fix, and it costs nothing extra: each cache is already sized for exactly the sequence length and head count its own layer needs.

## Chapter Summary

This chapter took Chapter 5's uniform quantization pipeline and made it heterogeneous, one section's finding correcting the next. Section 6.1 established that a block format's raw rounding error is a property of the format's bit width, not of which weight matrix it came from — and, in the process of verifying that, discovered that the "obvious" mechanism (softmax amplifying a small error into a larger distributional shift) is not what actually happens; the real fragility shows up only when the discrete argmax decision, not the bulk distribution, is measured across many trials. Section 6.2 turned that finding into a policy encoded directly in GGUF's existing per-tensor type tag, and demonstrated concretely why a reader must dispatch on that recorded type rather than guessing from a tensor's name, using a real naming collision Chapter 5.3 had already introduced. Section 6.3 priced that policy in real bytes for a large-scale model and caught its own first draft's mistake — treating the embedding table like the genuinely tiny norm weights it superficially resembles — by computing the actual cost rather than assuming it, cutting the policy's overhead from 86% to under 30% in the same section that discovered the problem. Sections 6.4 and 6.5 wired the corrected policy into a real memory-mapped GGUF file, a genuine forward pass, and finally a complete multi-layer, multi-token engine generating tokens autoregressively — exposing one more instance of Section 6.1's lesson (skipping the final norm barely moves the bulk distribution but still flips the sampled token) and one new failure mode of its own (a shared KV cache across layers, which corrupts silently rather than crashing). The KV cache at every stage stayed plain FP32, exactly Chapter 3.4's KVCache, because TurboQuant — the technique that will finally compress it — is Chapter 7's subject.

## Self-Check Questions

1. Section 6.1 quantizes an attention-shaped and an FFN-shaped matrix from the same distribution at the same format and finds their relative errors match closely. What does this imply about the claim "attention weights are harder to quantize than FFN weights"?
2. Why did measuring total-variation distance between FP32 and Q4-quantized softmax distributions fail to show the "amplification" effect Section 6.1 was looking for, and what measurement found it instead?
3. In Section 6.2's policy, why does the FFN gate matrix get Q4_0 while the attention query matrix gets Q8_0, given that Section 6.1 showed their raw quantization error is the same order of magnitude?
4. Explain concretely what happens, byte by byte, when a dispatcher that guesses format from a tensor's name encounters `blk.0.ffn_norm.weight`.
5. Section 6.3's naive hybrid policy put both norm weights and the embedding table at F32. Why did this cost so much more than expected, and why didn't the same reasoning apply to norm weights?
6. What formula determines an embedding table's element count, and why does that make it fundamentally different from a norm vector's element count as a function of model scale?
7. In Section 6.4, why does calling `MappedGGUF::get_f32` on the embedding table produce wrong values rather than a crash or an obvious error?
8. Why must W_k and W_v have fewer rows than W_q and W_o under grouped-query attention, and where does that show up in Section 6.4's GGUF file?
9. What specifically goes wrong if two transformer layers share a single `KVCache` instance, and why would this bug be invisible when testing against random (untrained) weights?
10. Why does skipping the final RMSNorm before the LM head in Section 6.5 barely change the total-variation distance of the resulting softmax distribution, yet still change which token gets sampled?

## Where We Go Next

Every KV cache in this chapter stored its keys and values as plain FP32 vectors, exactly Chapter 3.4's `KVCache`, unchanged. That was a deliberate choice, not an oversight: this chapter's hybrid engine mixes precisions across WEIGHT classes, but the KV cache is not a weight at all — it is created fresh for every prompt, grows with every generated token, and for a long enough context can outweigh the model's own weights in memory. Chapter 7 covers TurboQuant, a vector quantization technique built specifically for exactly that job: online (no calibration pass), accurate enough for attention's inner products, and aggressive enough to compress a KV cache far past what a blockwise scalar format like Q4_0 or Q8_0 could manage. Everything this chapter built — the mmap'd hybrid weights, the per-layer forward pass, the multi-layer engine — carries forward unchanged; only the KV cache's storage format is about to change.

## Worked Solutions

**1.** It implies the claim is false as stated: the raw rounding error a block format introduces depends only on the format's bit width and the distribution of values inside a block, not on which matrix the block came from. An attention-shaped and an FFN-shaped matrix drawn from the same distribution and quantized at the same format produce statistically indistinguishable relative error, because the arithmetic (find a block's max magnitude, pick a scale, round each weight) has no notion of what role the matrix plays. Whatever justifies treating attention and FFN weights differently has to come from something other than raw quantizability — which is exactly what the rest of Section 6.1 goes looking for.

**2.** Total variation distance measures how much probability mass moved across the WHOLE distribution on average, and softmax's outputs are bounded in [0, 1] and sum to 1 — that boundedness caps how far the bulk distribution can move regardless of how large the input perturbation was, so a 13%-relative-error input can still produce a distributional shift smaller than 1%. The effect Section 6.1 was looking for only shows up in the ARGMAX — which single position ends up with the most weight — measured across many independent trials rather than read off one distribution. Near-tied scores are where a small perturbation flips which position wins, and that is a discrete, trial-dependent event that an average-case distance metric smooths away.

**3.** Raw quantization error being the same order of magnitude does not mean the two matrices deserve the same format, because Section 6.1 showed the CONSEQUENCE of that error differs: the query/key path feeds a discrete argmax decision that a given error size flips at a measurable rate, while the FFN's SiLU-gated path has no such decision to flip and simply produces a proportional continuous output error. Giving attention weights Q8_0 buys a real reduction in how often that discrete decision changes; giving FFN weights the same protection would buy comparatively little, since there's no analogous discrete failure mode there to protect against, while costing far more in bytes since FFN weights vastly outnumber attention weights.

**4.** The name-based dispatcher sees the substring "ffn" in `blk.0.ffn_norm.weight` and assumes the tensor is Q4_0, so it reads the tensor's raw bytes as a sequence of `BlockQ4` structs: the first 2 bytes as an fp16 scale, the next 16 bytes as 32 packed 4-bit nibbles, and so on. But the tensor is genuinely F32 — its true bytes are 16 raw 4-byte floats (for a 64-element vector). Reinterpreting those float bytes as an fp16 scale and nibbles produces a "scale" that is really the first two bytes of an unrelated float, and "nibbles" that are really fragments of three other floats, packed and unpacked according to a completely different bit layout than what actually produced those bytes. The result is not a rounding error on the true values; it is numbers with no mathematical relationship to the true values at all.

**5.** Norm weights and the embedding table both sound like a "small, sensitive" category, but only norm weights are actually small in element count regardless of model scale (a few tens of thousands of elements total). The embedding table's element count is `vocabulary_size x hidden_dimension`, which scales with the tokenizer's vocabulary, not with the model's depth or width in the way norm weights do — at a 128,000-token vocabulary and a 4096-dimensional hidden size, the embedding table alone is over half a billion elements, more than a thousand times a typical norm-weight population. Storing that many elements at F32 (4 bytes each) instead of Q8_0 (about 1.0625 bytes each after amortizing the fp16 scale) costs several gigabytes at model scale, while doing the same for norm weights costs essentially nothing either way.

**6.** An embedding table's element count is `vocabulary_size x hidden_dimension`: one row per token the tokenizer can produce, each row as long as the model's hidden dimension. A norm vector's element count is just the hidden dimension, full stop — it does not multiply by anything that grows with the tokenizer or the training corpus. As vocabularies have grown (from tens of thousands of tokens to well over a hundred thousand in many current tokenizers), the embedding table's size has grown proportionally, while a norm vector's size has stayed exactly as small as the hidden dimension it lives in, which is why the two populations diverge so sharply at scale even though both "sound like" small, universally-touched tensors.

**7.** `get_f32` never inspects the tensor descriptor's own recorded `type` field — it simply reinterprets whatever bytes sit at that tensor's offset as an array of 4-byte floats, trusting the CALLER to have picked the right accessor. When the embedding table is genuinely F32 (as in Chapter 5.3), that trust is warranted and the accessor works correctly. Once the tensor is actually stored as Q8_0 blocks (an fp16 scale followed by 32 signed bytes, packed together), calling `get_f32` on it reads those exact same bytes but interprets each contiguous 4 bytes as an IEEE-754 float instead of as fragments of a scale-and-weights structure — a well-defined, crash-free operation that simply produces numbers with no relationship to the true embedding values, since nothing in the mmap'd memory itself carries a runtime type tag that `get_f32` could check.

**8.** Grouped-query attention has multiple query heads share a smaller number of key/value heads — a "group" of query heads all read the same K and V projection. That means the key and value projection matrices only need to produce `n_heads_kv x head_dim` output dimensions rather than `n_heads_q x head_dim` (which equals the full model dimension), so W_k and W_v have fewer OUTPUT rows than W_q and W_o, which each need the full model dimension of outputs. In Section 6.4's GGUF file, this shows up directly in the tensor descriptors: `blk.0.attn_q.weight` and `blk.0.attn_output.weight` are declared with `DIM` rows, while `blk.0.attn_k.weight` and `blk.0.attn_v.weight` are declared with the narrower `KV_DIM` (`n_heads_kv x head_dim`) rows — fewer elements, fewer Q8_0 blocks, less file space, exactly the GQA weight-count savings Chapter 3.4 first introduced.

**9.** `KVCache::store(head, position, k, v)` writes into a flat buffer indexed only by head and position, with no notion of which transformer layer is calling it. If every layer is handed the SAME `KVCache` instance, then layer 1 writing to (head 0, position 3) overwrites whatever layer 0 previously wrote to that identical slot, and every later attention computation — for both layers — reads back whichever layer's data was written most recently rather than its own. This bug produces no crash, no shape mismatch, and no NaN; it silently substitutes one layer's keys and values for another's. On trained weights this would corrupt attention and degrade output quality in a way that might be hard to diagnose; on random, untrained weights (as used throughout this chapter's own examples) the corrupted attention output is statistically indistinguishable from correct attention over equally-random keys and values, so the bug would produce no visibly different symptom at all — it would need to be caught by code review or a targeted test, not by "the output looks wrong."

**10.** Total variation distance measures the average shift across the WHOLE probability distribution over the vocabulary, and skipping the final norm changes the logits by a bounded, roughly uniform-feeling rescaling rather than reshuffling their relative order dramatically — so the bulk distribution barely moves in an average sense, exactly the same boundedness argument from Question 2. But the ARGMAX only cares about which single logit is largest, and a rescaling that leaves most of the distribution's shape intact can still change the relative ordering of the top one or two candidates if they were close together before the change — which is a discrete, threshold-crossing event that an average-case distance metric is not designed to detect. This is the same lesson as Section 6.1's argmax-flip experiment, encountered again in a completely different part of the same engine: bulk distributional stability and discrete decision stability are different properties, and a change can preserve one while breaking the other.
