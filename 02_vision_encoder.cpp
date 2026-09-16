// Chapter 18.2 -- Section 18.1's FrameReassembler already hands this
// section exactly the input a real deployment has: a raw MONO8/RGB8
// pixel buffer, straight off the sensor. That is the one genuine
// advantage a machine-vision pipeline has over a general-purpose photo
// pipeline: there is no JPEG or PNG to decode at all, because a GigE
// Vision camera never encodes one in the first place. This section
// starts from exactly that raw buffer and builds everything Qwen2.5-VL's
// own vision encoder needs from it: patchification, a linear patch
// embedding, the SAME two-dimensional rotary position encoding the real
// model uses to tell a patch's row and column apart, a small stack of
// bidirectional (non-causal) transformer blocks reusing this book's own
// already-verified RMSNorm/matmul/SwiGLU building blocks, and the
// 2x2 spatial patch merger that turns four neighboring patches into one
// visual token sized to match the text decoder's own embedding space.
//
// A note on scope, in this book's own recurring voice: this section's
// transformer blocks use FULL bidirectional attention over every patch,
// not the real Qwen2.5-VL's own window-attention scheme (full attention
// in only a few of its many layers, windowed attention restricted to
// nearby patches everywhere else, to keep a high-resolution image
// affordable). Teaching the real windowing scheme correctly would need
// its own section's worth of index arithmetic without changing a single
// idea this section is actually here to teach -- patch embedding, 2D
// RoPE, and the merger -- so this section states the simplification
// plainly rather than quietly shipping windowed attention as if it were
// the genuine article.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_vision_encoder.cpp -o 02_vision_encoder
// Run:     ./02_vision_encoder

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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
// PART 1: raw-buffer preprocessing. Section 18.1's FrameResult::pixels
// is already exactly this input -- an HxWxC row-major byte buffer, no
// codec involved anywhere in this pipeline.
// =======================================================================
struct RawImage {
    uint32_t width = 0, height = 0, channels = 0;
    std::vector<uint8_t> pixels;   // row-major, HxWxC, matching FrameResult::pixels exactly
};

// Nearest-neighbor resize to a target size that is an exact multiple of
// the ViT's own patch size. Real deployments use a higher-quality
// filter; nearest-neighbor is this section's own stated simplification,
// chosen because it is exactly reproducible in integer arithmetic across
// every architecture this book locks against, with no floating-point
// rounding difference to chase.
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

// One patch's flattened, normalized pixel data: patch_size * patch_size
// * channels floats, in row-major (row, then col, then channel) order.
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

// Patchifies the WHOLE image into a row-major grid of flattened patches:
// patches[row * grid_w + col] is that (row, col) patch's own flattened,
// normalized pixel vector.
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
// PART 2: this book's own RMSNorm/matmul/SwiGLU, repeated verbatim from
// Section 15.3 -- the vision encoder's own transformer blocks are built
// from the identical primitives the text decoder already uses.
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
// PART 3: two-dimensional rotary position encoding. A text token has one
// position; a patch has TWO (its row and its column), and Qwen2-VL's own
// real fix is splitting each attention head's dimension in half, rotating
// the FIRST half by the patch's row and the SECOND half by its column --
// the same 1D rotate-half mechanism this book has used since Chapter 9,
// applied twice, to two different coordinates, over two disjoint slices
// of the same vector.
// =======================================================================
struct RoPE2DTables {
    std::vector<float> cos_row, sin_row, cos_col, sin_col;
    int quarter_dim;   // half_dim (per axis) is head_dim/2; each axis's own rotate-half pairs cover head_dim/4
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
// Rotates a length-`2*half`-element slice using the standard rotate-half
// pairing (index k paired with index k+half), exactly Section 9's own
// `apply_rope` -- factored out so both the row-half and the col-half of
// a patch's query/key vector can call the identical primitive.
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
// Applies 2D RoPE to one attention head's full head_dim vector: the
// FIRST half rotated by `row` (using its own rotate-half pairing over
// that half's two quarters), the SECOND half rotated by `col`.
void apply_rope2d(std::span<float> head_vec, int row, int col, const RoPE2DTables& t) {
    const int half = static_cast<int>(head_vec.size()) / 2;
    rotate_half_inplace(head_vec.subspan(0, static_cast<size_t>(half)), t.cos_row, t.sin_row, row, t.quarter_dim);
    rotate_half_inplace(head_vec.subspan(static_cast<size_t>(half), static_cast<size_t>(half)), t.cos_col, t.sin_col, col, t.quarter_dim);
}

// =======================================================================
// PART 4: the vision transformer block itself -- RMSNorm, QKV
// projection, 2D RoPE, FULL (non-causal, no KV cache) bidirectional
// attention over every patch, output projection, a residual, a second
// RMSNorm, the SwiGLU FFN, and a second residual.
// =======================================================================
struct ViTShape { int dim, n_heads, head_dim, d_ff; int qkv_dim() const { return n_heads * head_dim; } };
struct ViTBlockWeights {
    std::vector<float> attn_norm, Wq, Wk, Wv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};

// One full self-attention pass over ALL patches -- unlike the text
// decoder's causal, one-position-at-a-time attention, every patch here
// attends to every other patch in a single call, because an image has
// no "future" to mask and no cache to build incrementally.
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
// PART 5: the 2x2 spatial patch merger. Four SPATIALLY ADJACENT patches
// -- (2r,2c), (2r,2c+1), (2r+1,2c), (2r+1,2c+1) -- are concatenated and
// projected down to the LLM's own embedding dimension, cutting the
// visual token count by 4x. Grouping by spatial adjacency, not by
// row-major list order, matters: for any grid wider than two patches,
// four row-major-consecutive patches are the START OF ONE ROW, not a 2x2
// neighborhood, and merging them would blend unrelated regions of the
// image into one token.
// =======================================================================
struct MergerWeights { std::vector<float> W1, b1, W2, b2; int hidden_dim; };

// Returns the flat patch-list indices of the 4 patches spatially
// adjacent to merged block (br, bc), in a fixed, testable order:
// top-left, top-right, bottom-left, bottom-right. Factored out on its
// own specifically so this section's own self-tests can check the
// SPATIAL grouping directly, independent of the matmul math around it.
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
// PART 6: self-tests, against a small synthetic image and small
// synthetic weights -- fast, deterministic, and needing no real
// checkpoint to prove this section's own machinery is correct.
// =======================================================================
std::vector<float> rand_vec(std::mt19937& rng, size_t n) {
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.2: Image Preprocessing and the Qwen2.5-VL Vision Encoder\n";
    std::cout << "========================================================\n";

    constexpr int PATCH = 14, CH = 3;
    constexpr int VIT_DIM = 24, N_HEADS = 2, HEAD_DIM = 8, D_FF = 32, N_LAYERS = 2;
    constexpr int LLM_DIM = 32, MERGER_HIDDEN = 40;

    std::cout << "\n-- Test 1: resize and patchify produce the expected grid and patch shape --\n";
    {
        RawImage src; src.width = 100; src.height = 80; src.channels = CH;
        std::mt19937 rng(1);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        src.pixels.resize(static_cast<size_t>(src.width) * src.height * CH);
        for (auto& b : src.pixels) b = static_cast<uint8_t>(byte_dist(rng));

        RawImage resized = resize_nearest(src, 56, 56);   // 4x4 grid of 14x14 patches
        CHECK(resized.width == 56 && resized.height == 56);

        NormStats norm;
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(resized, PATCH, norm, grid_h, grid_w);
        CHECK(grid_h == 4 && grid_w == 4);
        CHECK(patches.size() == 16);
        CHECK(patches[0].size() == static_cast<size_t>(PATCH) * PATCH * CH);
        bool all_finite = true;
        for (const auto& p : patches) for (float v : p) if (!std::isfinite(v)) all_finite = false;
        CHECK(all_finite);
        std::cout << "  resized to 56x56, patchified into " << grid_h << "x" << grid_w
                   << " grid, each patch " << patches[0].size() << " floats, all finite: "
                   << (all_finite ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Test 2: 2D RoPE rotates by row in the first half-dim and by column in the second --\n";
    {
        RoPE2DTables rope(8, HEAD_DIM, 10000.0f);
        std::vector<float> base(HEAD_DIM);
        std::mt19937 rng(2);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : base) v = dist(rng);

        // Position (0,0): every rotation angle is coord * theta = 0, so
        // the identity rotation should leave the vector byte-for-byte
        // unchanged, exactly like Section 9's own 1D RoPE at position 0.
        std::vector<float> at_origin = base;
        apply_rope2d(at_origin, 0, 0, rope);
        CHECK(at_origin == base);

        // Identical content at two DIFFERENT (row, col) positions must
        // produce DIFFERENT vectors -- this is the entire reason a
        // position encoding exists at all.
        std::vector<float> at_a = base, at_b = base;
        apply_rope2d(at_a, 1, 3, rope);
        apply_rope2d(at_b, 5, 2, rope);
        CHECK(at_a != at_b);

        // Changing ONLY the row (column held fixed) must still change
        // the result, and changing ONLY the column must too -- proving
        // both halves are actually wired to their own coordinate rather
        // than one half silently controlling both axes' output.
        std::vector<float> at_c = base, at_d = base;
        apply_rope2d(at_c, 1, 3, rope);
        apply_rope2d(at_d, 7, 3, rope);   // same col, different row
        CHECK(at_c != at_d);
        std::vector<float> at_e = base, at_f = base;
        apply_rope2d(at_e, 1, 3, rope);
        apply_rope2d(at_f, 1, 6, rope);   // same row, different col
        CHECK(at_e != at_f);

        std::cout << "  position (0,0) is the identity rotation: yes; identical content at different "
                     "(row,col) positions produces different vectors: yes; row-only and col-only "
                     "changes each independently change the output: yes\n";
    }

    std::cout << "\n-- Test 3: the 2x2 merger groups spatially adjacent patches, not row-major-consecutive ones --\n";
    {
        // A 4x4 grid: block (0,0) must gather patches (0,0),(0,1),(1,0),
        // (1,1) -- flat indices 0,1,4,5 -- NOT the row-major-consecutive
        // 0,1,2,3, which for a grid wider than 2 patches would silently
        // merge the first FOUR PATCHES OF ONE ROW into a single token
        // instead of a genuine 2x2 spatial neighborhood.
        auto idx00 = block_patch_indices(0, 0, /*grid_w=*/4);
        CHECK((idx00 == std::array<int, 4>{0, 1, 4, 5}));
        auto idx11 = block_patch_indices(1, 1, /*grid_w=*/4);
        CHECK((idx11 == std::array<int, 4>{10, 11, 14, 15}));
        auto idx01 = block_patch_indices(0, 1, /*grid_w=*/4);
        CHECK((idx01 == std::array<int, 4>{2, 3, 6, 7}));
        std::cout << "  block(0,0) on a 4x4 grid gathers patches {0,1,4,5} (a real 2x2 neighborhood), "
                     "not {0,1,2,3} (one row): correct\n";
    }

    std::cout << "\n-- Test 4: full encoder forward pass (embed -> " << N_LAYERS << " ViT blocks -> merger) --\n";
    {
        RawImage src; src.width = 56; src.height = 56; src.channels = CH;
        std::mt19937 img_rng(3);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        src.pixels.resize(static_cast<size_t>(src.width) * src.height * CH);
        for (auto& b : src.pixels) b = static_cast<uint8_t>(byte_dist(img_rng));

        NormStats norm;
        uint32_t grid_h = 0, grid_w = 0;
        auto raw_patches = patchify(src, PATCH, norm, grid_h, grid_w);
        const int n_patches = static_cast<int>(raw_patches.size());
        const size_t patch_vec_len = raw_patches[0].size();

        std::mt19937 w_rng(42);
        std::vector<float> W_embed = rand_vec(w_rng, static_cast<size_t>(VIT_DIM) * patch_vec_len);

        std::vector<std::vector<float>> x(static_cast<size_t>(n_patches));
        std::vector<int> rows(static_cast<size_t>(n_patches)), cols(static_cast<size_t>(n_patches));
        for (int i = 0; i < n_patches; ++i) {
            x[static_cast<size_t>(i)].resize(VIT_DIM);
            matmul(x[static_cast<size_t>(i)], raw_patches[static_cast<size_t>(i)], W_embed, patch_vec_len, VIT_DIM);
            rows[static_cast<size_t>(i)] = i / static_cast<int>(grid_w);
            cols[static_cast<size_t>(i)] = i % static_cast<int>(grid_w);
        }

        std::vector<ViTBlockWeights> layers(N_LAYERS);
        for (auto& l : layers) {
            l.attn_norm.assign(VIT_DIM, 1.0f);
            l.Wq = rand_vec(w_rng, static_cast<size_t>(N_HEADS * HEAD_DIM) * VIT_DIM);
            l.Wk = rand_vec(w_rng, static_cast<size_t>(N_HEADS * HEAD_DIM) * VIT_DIM);
            l.Wv = rand_vec(w_rng, static_cast<size_t>(N_HEADS * HEAD_DIM) * VIT_DIM);
            l.Wo = rand_vec(w_rng, static_cast<size_t>(VIT_DIM) * (N_HEADS * HEAD_DIM));
            l.ffn_norm.assign(VIT_DIM, 1.0f);
            l.Wgate = rand_vec(w_rng, static_cast<size_t>(D_FF) * VIT_DIM);
            l.Wup = rand_vec(w_rng, static_cast<size_t>(D_FF) * VIT_DIM);
            l.Wdown = rand_vec(w_rng, static_cast<size_t>(VIT_DIM) * D_FF);
        }
        ViTShape shape{VIT_DIM, N_HEADS, HEAD_DIM, D_FF};
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);

        auto run_encoder = [&]() {
            auto xx = x;
            for (const auto& l : layers) vit_block_forward(xx, shape, l, rows, cols, rope);
            MergerWeights mw;
            mw.hidden_dim = MERGER_HIDDEN;
            std::mt19937 m_rng(99);
            mw.W1 = rand_vec(m_rng, static_cast<size_t>(MERGER_HIDDEN) * (4 * VIT_DIM));
            mw.b1 = rand_vec(m_rng, static_cast<size_t>(MERGER_HIDDEN));
            mw.W2 = rand_vec(m_rng, static_cast<size_t>(LLM_DIM) * MERGER_HIDDEN);
            mw.b2 = rand_vec(m_rng, static_cast<size_t>(LLM_DIM));
            return merge_all_2x2(xx, static_cast<int>(grid_h), static_cast<int>(grid_w), VIT_DIM, mw, LLM_DIM);
        };

        auto merged1 = run_encoder();
        auto merged2 = run_encoder();

        CHECK(merged1.size() == (grid_h / 2) * (grid_w / 2));
        CHECK(merged1[0].size() == static_cast<size_t>(LLM_DIM));
        bool all_finite = true;
        for (const auto& tok : merged1) for (float v : tok) if (!std::isfinite(v)) all_finite = false;
        CHECK(all_finite);
        bool deterministic = (merged1.size() == merged2.size());
        for (size_t i = 0; deterministic && i < merged1.size(); ++i) deterministic = (merged1[i] == merged2[i]);
        CHECK(deterministic);

        std::cout << "  " << n_patches << " patches (" << grid_h << "x" << grid_w << ") -> " << N_LAYERS
                   << " ViT blocks -> " << merged1.size() << " merged visual tokens, each " << LLM_DIM
                   << "-dim; all finite: " << (all_finite ? "yes" : "no")
                   << "; deterministic across two runs: " << (deterministic ? "yes" : "no") << "\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
