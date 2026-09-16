// Chapter 20.5 -- Every earlier section in this chapter built a real
// structural guarantee: Section 20.1 made human sign-off unavoidable,
// Section 20.4 made the same true across a whole worklist queue. None of
// that answers a different, harder question a real deployment will be
// asked: WHY did the model suggest this study was urgent at all? This
// section builds one real, honest answer -- occlusion-based saliency,
// reusing Chapter 18.2's own vision encoder frozen exactly as written --
// and then, just as carefully, builds the proof of what that answer
// cannot tell you.
//
// The technique itself is genuinely simple and genuinely real: mask one
// patch out of the image, rerun the SAME frozen encoder, and measure how
// far the resulting whole-image representation moved. A patch whose
// removal barely moves that representation was not doing much work in
// the model's own computation; a patch whose removal moves it a long
// way was. That is a real, computable fact about this specific frozen
// network's own sensitivity to its own input -- not a guess, not a
// post-hoc story invented to sound plausible.
//
// What it is NOT, and what this section states as plainly as every
// other limitation this book has ever stated: a displacement score
// tells you THAT the representation moved and roughly how far, never
// WHY, never in what SEMANTIC direction, and never whether the model's
// own suggested priority was moved for a clinically sound reason or a
// spurious one. Test 4 below builds a direct, computable proof of
// exactly that gap: two differently-located anomalies can move the
// representation by comparable amounts while moving it in almost
// entirely different directions in the model's own hidden space -- proof
// that the scalar score alone cannot distinguish "the same kind of
// shift happened twice" from "two completely different things happened
// to look similarly important." A saliency map is a real, useful WHERE.
// It is never a WHY, and this section builds no code that pretends
// otherwise.
//
// A note on scope, in this chapter's own recurring voice: this is NOT
// legal or regulatory advice, and nothing here claims that occlusion
// saliency (or any other explainability technique) satisfies any
// specific jurisdiction's transparency or explainability requirement for
// software assisting a clinical decision -- that determination needs
// real regulatory and legal review this book cannot substitute for.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off 05_occlusion_saliency_and_explainability_limits.cpp -o 05_occlusion_saliency_and_explainability_limits
// Run:     ./05_occlusion_saliency_and_explainability_limits

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: raw-buffer preprocessing. Repeated verbatim from Chapter
// 18.2's own Section 18.2 -- this file's own encoder must be BYTE-FOR-
// BYTE the same frozen network Section 18.2 already verified, or a
// saliency score computed against it would not actually mean anything.
// =======================================================================
struct RawImage {
    uint32_t width = 0, height = 0, channels = 0;
    std::vector<uint8_t> pixels;   // row-major, HxWxC, matching FrameResult::pixels exactly
};

RawImage resize_nearest(const RawImage& src, uint32_t dst_w, uint32_t dst_h) {
    RawImage dst;
    dst.width = dst_w; dst.height = dst_h; dst.channels = src.channels;
    dst.pixels.resize(static_cast<size_t>(dst_w) * dst_h * src.channels);
    for (uint32_t y = 0; y < dst_h; ++y) {
        uint32_t sy = std::min(src.height - 1, (y * src.height) / dst_h);
        for (uint32_t x = 0; x < dst_w; ++x) {
            uint32_t sx = std::min(src.width - 1, (x * src.width) / dst_w);
            const uint8_t* sp = &src.pixels[(static_cast<size_t>(sy) * src.width + sx) * src.channels];
            uint8_t* dp = &dst.pixels[(static_cast<size_t>(y) * dst_w + x) * src.channels];
            std::memcpy(dp, sp, src.channels);
        }
    }
    return dst;
}

// CLIP/SigLIP-style per-channel normalization constants -- stated, not
// derived: a real deployment would use whichever mean/std the specific
// checkpoint's own preprocessor_config.json declares.
struct NormStats { float mean[3] = {0.481f, 0.458f, 0.408f}; float std[3] = {0.269f, 0.261f, 0.276f}; };

std::vector<float> extract_patch(const RawImage& img, uint32_t patch_row, uint32_t patch_col,
                                  uint32_t patch_size, const NormStats& norm) {
    std::vector<float> out(static_cast<size_t>(patch_size) * patch_size * img.channels);
    size_t idx = 0;
    for (uint32_t py = 0; py < patch_size; ++py) {
        uint32_t y = patch_row * patch_size + py;
        for (uint32_t px = 0; px < patch_size; ++px) {
            uint32_t x = patch_col * patch_size + px;
            const uint8_t* sp = &img.pixels[(static_cast<size_t>(y) * img.width + x) * img.channels];
            for (uint32_t c = 0; c < img.channels; ++c) {
                float v = static_cast<float>(sp[c]) / 255.0f;
                out[idx++] = (v - norm.mean[c % 3]) / norm.std[c % 3];
            }
        }
    }
    return out;
}

std::vector<std::vector<float>> patchify(const RawImage& img, uint32_t patch_size, const NormStats& norm,
                                          uint32_t& grid_h, uint32_t& grid_w) {
    grid_h = img.height / patch_size;
    grid_w = img.width / patch_size;
    std::vector<std::vector<float>> patches(static_cast<size_t>(grid_h) * grid_w);
    for (uint32_t r = 0; r < grid_h; ++r)
        for (uint32_t c = 0; c < grid_w; ++c)
            patches[static_cast<size_t>(r) * grid_w + c] = extract_patch(img, r, c, patch_size, norm);
    return patches;
}

// =======================================================================
// PART 2: this book's own RMSNorm/matmul/SwiGLU, repeated verbatim.
// =======================================================================
void rms_norm(std::span<float> out, std::span<const float> x, std::span<const float> weights, float epsilon = 1e-6f) {
    const size_t d = x.size();
    double sum_sq = 0.0;
    for (size_t i = 0; i < d; ++i) sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    float rms_inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(d)) + epsilon);
    for (size_t i = 0; i < d; ++i) out[i] = (x[i] * rms_inv) * weights[i];
}
void matmul(std::span<float> out, std::span<const float> x, std::span<const float> W, size_t in_dim, size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < in_dim; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(W[j * in_dim + i]);
        out[j] = static_cast<float>(sum);
    }
}
inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
void swiglu_ffn(std::span<float> out, std::span<const float> x, std::span<const float> W_gate,
                 std::span<const float> W_up, std::span<const float> W_down, size_t dim, size_t d_ff) {
    std::vector<float> gate_proj(d_ff), up_proj(d_ff), hidden(d_ff);
    matmul(gate_proj, x, W_gate, dim, d_ff);
    matmul(up_proj, x, W_up, dim, d_ff);
    for (size_t i = 0; i < d_ff; ++i) hidden[i] = silu(gate_proj[i]) * up_proj[i];
    matmul(out, hidden, W_down, d_ff, dim);
}
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
}

// =======================================================================
// PART 3: two-dimensional rotary position encoding, repeated verbatim.
// =======================================================================
struct RoPE2DTables {
    std::vector<float> cos_row, sin_row, cos_col, sin_col;
    int quarter_dim;
    RoPE2DTables(int max_coord, int head_dim, float base) : quarter_dim(head_dim / 4) {
        cos_row.resize(static_cast<size_t>(max_coord) * quarter_dim);
        sin_row.resize(static_cast<size_t>(max_coord) * quarter_dim);
        cos_col.resize(static_cast<size_t>(max_coord) * quarter_dim);
        sin_col.resize(static_cast<size_t>(max_coord) * quarter_dim);
        for (int coord = 0; coord < max_coord; ++coord) {
            for (int k = 0; k < quarter_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim / 2));
                float angle = static_cast<float>(coord) * theta;
                size_t idx = static_cast<size_t>(coord) * quarter_dim + k;
                cos_row[idx] = std::cos(angle); sin_row[idx] = std::sin(angle);
                cos_col[idx] = std::cos(angle); sin_col[idx] = std::sin(angle);
            }
        }
    }
};
void rotate_half_inplace(std::span<float> vec, std::span<const float> cos_tab, std::span<const float> sin_tab,
                          int coord, int quarter_dim) {
    for (int k = 0; k < quarter_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + quarter_dim)];
        float c = cos_tab[static_cast<size_t>(coord) * quarter_dim + k];
        float s = sin_tab[static_cast<size_t>(coord) * quarter_dim + k];
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + quarter_dim)] = x1 * s + x2 * c;
    }
}
void apply_rope2d(std::span<float> head_vec, int row, int col, const RoPE2DTables& t) {
    const int half = static_cast<int>(head_vec.size()) / 2;
    rotate_half_inplace(head_vec.subspan(0, static_cast<size_t>(half)), t.cos_row, t.sin_row, row, t.quarter_dim);
    rotate_half_inplace(head_vec.subspan(static_cast<size_t>(half), static_cast<size_t>(half)), t.cos_col, t.sin_col, col, t.quarter_dim);
}

// =======================================================================
// PART 4: the vision transformer block itself, repeated verbatim.
// =======================================================================
struct ViTShape { int dim, n_heads, head_dim, d_ff; int qkv_dim() const { return n_heads * head_dim; } };
struct ViTBlockWeights {
    std::vector<float> attn_norm, Wq, Wk, Wv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};

void vit_full_attention(const std::vector<std::vector<float>>& q, const std::vector<std::vector<float>>& k,
                         const std::vector<std::vector<float>>& v, std::vector<std::vector<float>>& out,
                         int n_heads, int head_dim) {
    const int n = static_cast<int>(q.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < n; ++i) {
            std::vector<float> scores(static_cast<size_t>(n));
            std::span<const float> qi(q[static_cast<size_t>(i)].data() + h * head_dim, static_cast<size_t>(head_dim));
            for (int j = 0; j < n; ++j) {
                std::span<const float> kj(k[static_cast<size_t>(j)].data() + h * head_dim, static_cast<size_t>(head_dim));
                double d = 0.0;
                for (int c = 0; c < head_dim; ++c) d += static_cast<double>(qi[static_cast<size_t>(c)]) * kj[static_cast<size_t>(c)];
                scores[static_cast<size_t>(j)] = static_cast<float>(d) * scale;
            }
            softmax_inplace(scores);
            float* o = out[static_cast<size_t>(i)].data() + h * head_dim;
            for (int c = 0; c < head_dim; ++c) o[c] = 0.0f;
            for (int j = 0; j < n; ++j) {
                std::span<const float> vj(v[static_cast<size_t>(j)].data() + h * head_dim, static_cast<size_t>(head_dim));
                float w = scores[static_cast<size_t>(j)];
                for (int c = 0; c < head_dim; ++c) o[c] += w * vj[static_cast<size_t>(c)];
            }
        }
    }
}

void vit_block_forward(std::vector<std::vector<float>>& x, const ViTShape& shape, const ViTBlockWeights& w,
                        const std::vector<int>& rows, const std::vector<int>& cols, const RoPE2DTables& rope) {
    const int n = static_cast<int>(x.size());
    std::vector<std::vector<float>> q(static_cast<size_t>(n)), k(static_cast<size_t>(n)), v(static_cast<size_t>(n)), attn_out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        std::vector<float> normed(static_cast<size_t>(shape.dim));
        rms_norm(normed, x[static_cast<size_t>(i)], w.attn_norm);
        q[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        k[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        v[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        attn_out[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        matmul(q[static_cast<size_t>(i)], normed, w.Wq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        matmul(k[static_cast<size_t>(i)], normed, w.Wk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        matmul(v[static_cast<size_t>(i)], normed, w.Wv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        for (int h = 0; h < shape.n_heads; ++h) {
            apply_rope2d(std::span<float>(q[static_cast<size_t>(i)].data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                         rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)], rope);
            apply_rope2d(std::span<float>(k[static_cast<size_t>(i)].data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                         rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)], rope);
        }
    }
    vit_full_attention(q, k, v, attn_out, shape.n_heads, shape.head_dim);
    for (int i = 0; i < n; ++i) {
        std::vector<float> proj(static_cast<size_t>(shape.dim));
        matmul(proj, attn_out[static_cast<size_t>(i)], w.Wo, static_cast<size_t>(shape.qkv_dim()), static_cast<size_t>(shape.dim));
        for (int d = 0; d < shape.dim; ++d) x[static_cast<size_t>(i)][static_cast<size_t>(d)] += proj[static_cast<size_t>(d)];
        std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
        rms_norm(normed2, x[static_cast<size_t>(i)], w.ffn_norm);
        swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
        for (int d = 0; d < shape.dim; ++d) x[static_cast<size_t>(i)][static_cast<size_t>(d)] += ffn_out[static_cast<size_t>(d)];
    }
}

// =======================================================================
// PART 5: the 2x2 spatial patch merger, repeated verbatim.
// =======================================================================
struct MergerWeights { std::vector<float> W1, b1, W2, b2; int hidden_dim; };

std::array<int, 4> block_patch_indices(int br, int bc, int grid_w) {
    const int r0 = 2 * br, r1 = 2 * br + 1, c0 = 2 * bc, c1 = 2 * bc + 1;
    return {r0 * grid_w + c0, r0 * grid_w + c1, r1 * grid_w + c0, r1 * grid_w + c1};
}

std::vector<std::vector<float>> merge_all_2x2(const std::vector<std::vector<float>>& patches, int grid_h, int grid_w,
                                               int vit_dim, const MergerWeights& mw, int llm_dim) {
    const int out_h = grid_h / 2, out_w = grid_w / 2;
    std::vector<std::vector<float>> merged(static_cast<size_t>(out_h) * out_w);
    for (int br = 0; br < out_h; ++br) {
        for (int bc = 0; bc < out_w; ++bc) {
            std::vector<float> concat(static_cast<size_t>(4 * vit_dim));
            auto idx = block_patch_indices(br, bc, grid_w);
            for (int slot = 0; slot < 4; ++slot) {
                const auto& p = patches[static_cast<size_t>(idx[static_cast<size_t>(slot)])];
                std::copy(p.begin(), p.end(), concat.begin() + slot * vit_dim);
            }
            std::vector<float> h1(static_cast<size_t>(mw.hidden_dim));
            matmul(h1, concat, mw.W1, static_cast<size_t>(4 * vit_dim), static_cast<size_t>(mw.hidden_dim));
            for (int i = 0; i < mw.hidden_dim; ++i) h1[static_cast<size_t>(i)] = silu(h1[static_cast<size_t>(i)] + mw.b1[static_cast<size_t>(i)]);
            std::vector<float> out(static_cast<size_t>(llm_dim));
            matmul(out, h1, mw.W2, static_cast<size_t>(mw.hidden_dim), static_cast<size_t>(llm_dim));
            for (int i = 0; i < llm_dim; ++i) out[static_cast<size_t>(i)] += mw.b2[static_cast<size_t>(i)];
            merged[static_cast<size_t>(br) * out_w + bc] = std::move(out);
        }
    }
    return merged;
}

// =======================================================================
// PART 6: occlusion-based saliency, built on top of the frozen PARTS
// 1-5 above with no changes to any of them.
// =======================================================================

// A patch is "occluded" by replacing it with the per-channel MEAN color
// -- which, after PART 1's own normalization, is EXACTLY the zero vector
// (v == mean => (v - mean) / std == 0). Occluding with black instead
// would introduce a large, artificial dark-patch edge of its own, which
// is itself a strong signal the model could react to; occluding with the
// dataset's own mean is the smallest, most neutral edit this section can
// make to remove a patch's content without inserting a new, unrelated
// one in its place.
std::vector<float> mean_occluded_patch(size_t patch_vec_len) {
    return std::vector<float>(patch_vec_len, 0.0f);
}

struct EncoderWeights {
    std::vector<float> W_embed;
    std::vector<ViTBlockWeights> layers;
    MergerWeights merger;
    int vit_dim = 0, n_heads = 0, head_dim = 0, d_ff = 0, llm_dim = 0;
};

// Mean-pools the merged visual tokens down to ONE fixed-length vector: a
// single whole-image representation whose length never changes no
// matter which (if any) patch was occluded, so any two runs' outputs can
// always be compared with plain L2 distance.
std::vector<float> mean_pool(const std::vector<std::vector<float>>& merged) {
    const size_t dim = merged.empty() ? 0 : merged[0].size();
    std::vector<float> out(dim, 0.0f);
    for (const auto& tok : merged) for (size_t i = 0; i < dim; ++i) out[i] += tok[i];
    if (!merged.empty()) for (float& v : out) v /= static_cast<float>(merged.size());
    return out;
}

double l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq);
}

// Runs the frozen PART 1-5 pipeline end to end on an already-patchified
// image, with `occluded_patch` (if set) replaced by the neutral
// mean-color vector before embedding, and returns the whole-image
// mean-pooled representation.
std::vector<float> run_encoder_pooled(const std::vector<std::vector<float>>& raw_patches, uint32_t grid_h,
                                       uint32_t grid_w, const EncoderWeights& ew, const RoPE2DTables& rope,
                                       std::optional<int> occluded_patch) {
    const int n_patches = static_cast<int>(raw_patches.size());
    const size_t patch_vec_len = raw_patches[0].size();
    std::vector<std::vector<float>> x(static_cast<size_t>(n_patches));
    std::vector<int> rows(static_cast<size_t>(n_patches)), cols(static_cast<size_t>(n_patches));
    for (int i = 0; i < n_patches; ++i) {
        std::vector<float> occluded_buf;
        const std::vector<float>* src = &raw_patches[static_cast<size_t>(i)];
        if (occluded_patch && *occluded_patch == i) {
            occluded_buf = mean_occluded_patch(patch_vec_len);
            src = &occluded_buf;
        }
        x[static_cast<size_t>(i)].resize(static_cast<size_t>(ew.vit_dim));
        matmul(x[static_cast<size_t>(i)], *src, ew.W_embed, patch_vec_len, static_cast<size_t>(ew.vit_dim));
        rows[static_cast<size_t>(i)] = i / static_cast<int>(grid_w);
        cols[static_cast<size_t>(i)] = i % static_cast<int>(grid_w);
    }
    ViTShape shape{ew.vit_dim, ew.n_heads, ew.head_dim, ew.d_ff};
    for (const auto& l : ew.layers) vit_block_forward(x, shape, l, rows, cols, rope);
    auto merged = merge_all_2x2(x, static_cast<int>(grid_h), static_cast<int>(grid_w), ew.vit_dim, ew.merger, ew.llm_dim);
    return mean_pool(merged);
}

// The saliency map itself: one non-negative displacement score per
// patch. score[i] == l2_distance(baseline, output-with-patch-i-
// occluded) -- large where removing that patch moved the whole-image
// representation a long way, near zero where it barely moved it at all.
std::vector<double> occlusion_saliency_map(const std::vector<std::vector<float>>& raw_patches, uint32_t grid_h,
                                            uint32_t grid_w, const EncoderWeights& ew, const RoPE2DTables& rope) {
    auto baseline = run_encoder_pooled(raw_patches, grid_h, grid_w, ew, rope, std::nullopt);
    std::vector<double> scores(raw_patches.size());
    for (size_t i = 0; i < raw_patches.size(); ++i) {
        auto occluded_output = run_encoder_pooled(raw_patches, grid_h, grid_w, ew, rope, static_cast<int>(i));
        scores[i] = l2_distance(baseline, occluded_output);
    }
    return scores;
}

// The raw per-patch DISPLACEMENT VECTOR, kept separate from the scalar
// score above specifically so this section's own Test 4 can compare the
// DIRECTION two different occlusions move the representation in, not
// merely how far.
std::vector<float> occlusion_displacement_vector(const std::vector<float>& baseline,
                                                  const std::vector<float>& occluded_output) {
    std::vector<float> d(baseline.size());
    for (size_t i = 0; i < baseline.size(); ++i) d[i] = occluded_output[i] - baseline[i];
    return d;
}

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// =======================================================================
// PART 7: self-tests.
// =======================================================================
std::vector<float> rand_vec(std::mt19937& rng, size_t n) {
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

EncoderWeights build_weights(int vit_dim, int n_heads, int head_dim, int d_ff, int n_layers,
                              int merger_hidden, int llm_dim, size_t patch_vec_len) {
    EncoderWeights ew;
    ew.vit_dim = vit_dim; ew.n_heads = n_heads; ew.head_dim = head_dim; ew.d_ff = d_ff; ew.llm_dim = llm_dim;
    std::mt19937 w_rng(42);
    ew.W_embed = rand_vec(w_rng, static_cast<size_t>(vit_dim) * patch_vec_len);
    ew.layers.resize(static_cast<size_t>(n_layers));
    for (auto& l : ew.layers) {
        l.attn_norm.assign(static_cast<size_t>(vit_dim), 1.0f);
        l.Wq = rand_vec(w_rng, static_cast<size_t>(n_heads * head_dim) * vit_dim);
        l.Wk = rand_vec(w_rng, static_cast<size_t>(n_heads * head_dim) * vit_dim);
        l.Wv = rand_vec(w_rng, static_cast<size_t>(n_heads * head_dim) * vit_dim);
        l.Wo = rand_vec(w_rng, static_cast<size_t>(vit_dim) * (n_heads * head_dim));
        l.ffn_norm.assign(static_cast<size_t>(vit_dim), 1.0f);
        l.Wgate = rand_vec(w_rng, static_cast<size_t>(d_ff) * vit_dim);
        l.Wup = rand_vec(w_rng, static_cast<size_t>(d_ff) * vit_dim);
        l.Wdown = rand_vec(w_rng, static_cast<size_t>(vit_dim) * d_ff);
    }
    std::mt19937 m_rng(99);
    ew.merger.hidden_dim = merger_hidden;
    ew.merger.W1 = rand_vec(m_rng, static_cast<size_t>(merger_hidden) * (4 * vit_dim));
    ew.merger.b1 = rand_vec(m_rng, static_cast<size_t>(merger_hidden));
    ew.merger.W2 = rand_vec(m_rng, static_cast<size_t>(llm_dim) * merger_hidden);
    ew.merger.b2 = rand_vec(m_rng, static_cast<size_t>(llm_dim));
    return ew;
}

// Fills every patch of a 4x4-grid, 56x56 synthetic image with `bg_value`,
// then overwrites the patches at `anomaly_indices` with their paired
// values from `anomaly_values` -- letting the tests below build images
// with a controlled number of deliberately anomalous regions.
RawImage make_grid_image(uint32_t patch, uint32_t grid, uint8_t bg_value,
                          const std::vector<int>& anomaly_indices, const std::vector<uint8_t>& anomaly_values) {
    RawImage img;
    img.width = patch * grid; img.height = patch * grid; img.channels = 3;
    img.pixels.assign(static_cast<size_t>(img.width) * img.height * img.channels, bg_value);
    for (size_t a = 0; a < anomaly_indices.size(); ++a) {
        int idx = anomaly_indices[a];
        uint32_t pr = static_cast<uint32_t>(idx) / grid, pc = static_cast<uint32_t>(idx) % grid;
        for (uint32_t py = 0; py < patch; ++py) {
            uint32_t y = pr * patch + py;
            for (uint32_t px = 0; px < patch; ++px) {
                uint32_t x = pc * patch + px;
                uint8_t* p = &img.pixels[(static_cast<size_t>(y) * img.width + x) * img.channels];
                p[0] = p[1] = p[2] = anomaly_values[a];
            }
        }
    }
    return img;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.5: Occlusion-Based Saliency and the Honest Limits of Explainability\n";
    std::cout << "========================================================\n";

    constexpr uint32_t PATCH = 14, GRID = 4;
    constexpr int VIT_DIM = 24, N_HEADS = 2, HEAD_DIM = 8, D_FF = 32, N_LAYERS = 2;
    constexpr int LLM_DIM = 32, MERGER_HIDDEN = 40;

    NormStats norm;

    std::cout << "\n-- Test 1: the frozen encoder's whole-image representation is deterministic --\n";
    {
        RawImage img = make_grid_image(PATCH, GRID, 128, {}, {});
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(img, PATCH, norm, grid_h, grid_w);
        auto ew = build_weights(VIT_DIM, N_HEADS, HEAD_DIM, D_FF, N_LAYERS, MERGER_HIDDEN, LLM_DIM, patches[0].size());
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);
        auto out1 = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, std::nullopt);
        auto out2 = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, std::nullopt);
        CHECK(out1 == out2);
        CHECK(out1.size() == static_cast<size_t>(LLM_DIM));
        std::cout << "  two runs of the unmodified " << grid_h << "x" << grid_w
                   << "-patch image through the frozen encoder produce byte-identical "
                   << out1.size() << "-dim pooled representations\n";
    }

    std::cout << "\n-- Test 2: the saliency map has one non-negative score per patch, and is not "
                 "trivially all-zero --\n";
    {
        RawImage img = make_grid_image(PATCH, GRID, 100, {5, 9}, {30, 220});
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(img, PATCH, norm, grid_h, grid_w);
        auto ew = build_weights(VIT_DIM, N_HEADS, HEAD_DIM, D_FF, N_LAYERS, MERGER_HIDDEN, LLM_DIM, patches[0].size());
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);
        auto scores = occlusion_saliency_map(patches, grid_h, grid_w, ew, rope);
        CHECK(scores.size() == 16);
        bool all_non_negative = true, any_nontrivial = false;
        for (double s : scores) {
            if (s < 0.0) all_non_negative = false;
            if (s > 1e-6) any_nontrivial = true;
        }
        CHECK(all_non_negative);
        CHECK(any_nontrivial);
        std::cout << "  16 patches produce 16 non-negative displacement scores, and at least "
                     "one is clearly nonzero -- occlusion genuinely moves the representation\n";
    }

    std::cout << "\n-- Test 3: occluding a deliberately anomalous patch moves the representation "
                 "further than occluding an ordinary background patch --\n";
    {
        constexpr int ANOMALY_IDX = 6;   // row 1, col 2 of a 4x4 grid
        RawImage img = make_grid_image(PATCH, GRID, 128, {ANOMALY_IDX}, {12});
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(img, PATCH, norm, grid_h, grid_w);
        auto ew = build_weights(VIT_DIM, N_HEADS, HEAD_DIM, D_FF, N_LAYERS, MERGER_HIDDEN, LLM_DIM, patches[0].size());
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);
        auto scores = occlusion_saliency_map(patches, grid_h, grid_w, ew, rope);
        double max_background = 0.0;
        for (int i = 0; i < 16; ++i) if (i != ANOMALY_IDX) max_background = std::max(max_background, scores[static_cast<size_t>(i)]);
        CHECK(scores[static_cast<size_t>(ANOMALY_IDX)] > max_background);
        std::cout << "  the anomalous patch's own score (" << scores[static_cast<size_t>(ANOMALY_IDX)]
                   << ") exceeds the highest score among all 15 ordinary background patches ("
                   << max_background << ") -- occlusion saliency correctly localizes WHERE the "
                      "output is sensitive to input content\n";
    }

    std::cout << "\n-- Test 4: THE HONEST LIMIT -- two different anomalies can move the "
                 "representation by comparable magnitudes while moving it in substantially "
                 "different directions, so the scalar score alone cannot tell you WHY --\n";
    {
        constexpr int ANOMALY_A = 3, ANOMALY_B = 12;
        RawImage img = make_grid_image(PATCH, GRID, 128, {ANOMALY_A, ANOMALY_B}, {12, 230});
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(img, PATCH, norm, grid_h, grid_w);
        auto ew = build_weights(VIT_DIM, N_HEADS, HEAD_DIM, D_FF, N_LAYERS, MERGER_HIDDEN, LLM_DIM, patches[0].size());
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);

        auto baseline = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, std::nullopt);
        auto out_a = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, ANOMALY_A);
        auto out_b = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, ANOMALY_B);
        double score_a = l2_distance(baseline, out_a);
        double score_b = l2_distance(baseline, out_b);
        auto disp_a = occlusion_displacement_vector(baseline, out_a);
        auto disp_b = occlusion_displacement_vector(baseline, out_b);
        double cos_sim = cosine_similarity(disp_a, disp_b);

        // "Comparable magnitude" is checked directly, not assumed: the
        // larger score is required to be within a factor of 5 of the
        // smaller one, so this test would fail loudly if the two
        // anomalies had turned out wildly mismatched in importance
        // rather than the deliberately similar disruption this section
        // built them to cause.
        double ratio = std::max(score_a, score_b) / std::min(score_a, score_b);
        CHECK(ratio < 5.0);
        // The actual claim this test exists to prove: direction is NOT
        // preserved just because magnitude is comparable. A cosine
        // similarity anywhere near 1.0 would mean both anomalies moved
        // the representation the same way, which would undermine the
        // entire point -- so this section checks it is well below that.
        CHECK(cos_sim < 0.9);
        std::cout << "  occluding patch " << ANOMALY_A << " moves the representation by "
                   << score_a << "; occluding patch " << ANOMALY_B << " moves it by " << score_b
                   << " (ratio " << ratio << ", comparable magnitude); but the two displacement "
                      "vectors have cosine similarity " << cos_sim << " -- pointing in "
                      "substantially different directions in the model's own hidden space. "
                      "Magnitude alone cannot tell these two, very different, causes apart\n";
    }

    std::cout << "\n-- Test 5: occlusion really does replace a patch with the dataset's own "
                 "per-channel mean color, not black --\n";
    {
        // mean_occluded_patch itself is exactly zero by construction --
        // checked directly here, not just asserted in a comment.
        auto occ = mean_occluded_patch(PATCH * PATCH * 3);
        bool all_exactly_zero = true;
        for (float v : occ) if (v != 0.0f) all_exactly_zero = false;
        CHECK(all_exactly_zero);

        // And a real patch filled with the ROUNDED mean pixel value in
        // every channel normalizes to (approximately) that same zero
        // vector -- proving "occlude with the mean" and "occlude with
        // mean_occluded_patch" are the same real edit, not two
        // unrelated ideas that happen to share a name.
        uint8_t mean_r = static_cast<uint8_t>(std::lround(norm.mean[0] * 255.0f));
        uint8_t mean_g = static_cast<uint8_t>(std::lround(norm.mean[1] * 255.0f));
        uint8_t mean_b = static_cast<uint8_t>(std::lround(norm.mean[2] * 255.0f));
        RawImage img; img.width = PATCH; img.height = PATCH; img.channels = 3;
        img.pixels.resize(static_cast<size_t>(PATCH) * PATCH * 3);
        for (size_t i = 0; i < img.pixels.size(); i += 3) {
            img.pixels[i] = mean_r; img.pixels[i + 1] = mean_g; img.pixels[i + 2] = mean_b;
        }
        auto mean_patch = extract_patch(img, 0, 0, PATCH, norm);
        bool all_near_zero = true;
        for (float v : mean_patch) if (std::fabs(v) > 0.01f) all_near_zero = false;
        CHECK(all_near_zero);
        std::cout << "  mean_occluded_patch() is exactly the zero vector, and a real patch "
                     "filled with the rounded per-channel mean pixel value normalizes to "
                     "within 0.01 of that same zero vector -- confirming occlusion removes a "
                     "patch's content by substituting the dataset's own neutral color, never "
                     "an artificial black edge\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
