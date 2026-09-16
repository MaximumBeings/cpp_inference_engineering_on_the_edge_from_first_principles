// Chapter 18.3 -- Section 18.2 turned a raw camera frame into a small
// handful of visual tokens, each one already projected into the text
// decoder's own embedding space. This section does the one thing that
// makes "vision-LANGUAGE model" more than two separate models bolted
// together: splicing those visual tokens directly into a text token
// sequence, in place of a reserved placeholder ID, and running the
// IDENTICAL, unchanged Qwen2 decoder this book has verified since
// Chapter 15 over the combined sequence -- no separate "vision decoder,"
// no special-cased attention path for image positions. A visual token
// and a text token differ only in how their own embedding vector was
// produced; once both are sitting in the same sequence of vectors, one
// per position, `qwen2_block_forward` cannot tell the difference and
// was never asked to.
//
// The one real seam this requires is in exactly one place: today,
// `decode_step` always computes a position's own starting vector by
// looking up `token_id` in `token_embd.weight`. A visual token has no
// such row to look up -- its vector was already computed by Section
// 18.2's own merger. `decode_step_with_embedding` factors the ORIGINAL
// `decode_step` into "get a starting embedding, then run every layer,"
// and `decode_step` becomes the thin, unchanged-behavior special case
// of it that looks the embedding up by ID -- exactly Chapter 16.4's own
// pattern of extending a locked function with an optional capability
// rather than rewriting it, applied to embeddings instead of profiling.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_multimodal_fusion.cpp -o 03_multimodal_fusion
// Run:     ./03_multimodal_fusion

#include <mdspan/mdspan.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: Section 15.1's GGUF reader/writer (Section 15.4's own copy).
// =======================================================================
enum GGUFValueType : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12,
};
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q8_0 = 8 };
size_t scalar_byte_size(uint32_t t) {
    switch (t) {
        case V_UINT8: case V_INT8: case V_BOOL: return 1;
        case V_UINT16: case V_INT16: return 2;
        case V_UINT32: case V_INT32: case V_FLOAT32: return 4;
        case V_UINT64: case V_INT64: case V_FLOAT64: return 8;
        default: return 0;
    }
}
class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* data, size_t size) { out.write(reinterpret_cast<const char*>(data), size); pos += size; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_magic() { write_raw("GGUF", 4); }
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_f32(float v) { write_raw(&v, 4); }
    void write_string(const std::string& s) { uint64_t len = s.size(); write_raw(&len, 8); write_raw(s.data(), s.size()); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(V_STRING); write_string(v); }
    void write_kv_u32(const std::string& k, uint32_t v) { write_string(k); write_u32(V_UINT32); write_u32(v); }
    void write_kv_f32(const std::string& k, float v) { write_string(k); write_u32(V_FLOAT32); write_f32(v); }
    void write_tensor_info(const std::string& name, const std::vector<uint64_t>& dims, GGMLType type, uint64_t offset) {
        write_string(name);
        write_u32(static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) write_u64(d);
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
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t n_elements = 0;
};
using MetaValue = std::variant<std::string, uint32_t, float, bool, std::vector<std::string>>;
class GGUFReader {
    std::ifstream in;
    void read_raw(void* data, size_t size) { in.read(reinterpret_cast<char*>(data), size); }
    void skip_value(uint32_t type) {
        if (type == V_STRING) { read_string(); return; }
        if (type == V_ARRAY) {
            uint32_t elem_type; read_raw(&elem_type, 4);
            uint64_t count; read_raw(&count, 8);
            for (uint64_t i = 0; i < count; ++i) skip_value(elem_type);
            return;
        }
        in.seekg(static_cast<std::streamoff>(scalar_byte_size(type)), std::ios::cur);
    }
public:
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    std::unordered_map<std::string, MetaValue> metadata;
    std::vector<TensorInfo> tensors;
    uint64_t data_section_offset = 0;
    bool open(const std::string& path) {
        in.open(path, std::ios::binary);
        if (!in.is_open()) return false;
        char magic_bytes[4];
        read_raw(magic_bytes, 4);
        if (std::memcmp(magic_bytes, "GGUF", 4) != 0) return false;
        std::memcpy(&magic, magic_bytes, 4);
        read_raw(&version, 4);
        read_raw(&n_tensors, 8);
        read_raw(&n_kv, 8);
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string();
            uint32_t type; read_raw(&type, 4);
            switch (type) {
                case V_STRING: metadata[key] = read_string(); break;
                case V_UINT32: { uint32_t v; read_raw(&v, 4); metadata[key] = v;
                                 if (key == "general.alignment") alignment = v;
                                 break; }
                case V_FLOAT32: { float v; read_raw(&v, 4); metadata[key] = v; break; }
                case V_BOOL: { uint8_t v; read_raw(&v, 1); metadata[key] = (v != 0); break; }
                case V_ARRAY: {
                    uint32_t elem_type; read_raw(&elem_type, 4);
                    uint64_t count; read_raw(&count, 8);
                    if (elem_type == V_STRING) {
                        std::vector<std::string> arr(count);
                        for (uint64_t j = 0; j < count; ++j) arr[j] = read_string();
                        metadata[key] = std::move(arr);
                    } else {
                        for (uint64_t j = 0; j < count; ++j) skip_value(elem_type);
                    }
                    break;
                }
                default: skip_value(type); break;
            }
        }
        tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            auto& t = tensors[i];
            t.name = read_string();
            uint32_t n_dims; read_raw(&n_dims, 4);
            t.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) read_raw(&t.dims[d], 8);
            read_raw(&t.type, 4);
            read_raw(&t.offset, 8);
            t.n_elements = 1;
            for (auto d : t.dims) t.n_elements *= d;
        }
        uint64_t header_end = static_cast<uint64_t>(in.tellg());
        uint64_t rem = header_end % alignment;
        data_section_offset = (rem == 0) ? header_end : header_end + (alignment - rem);
        return true;
    }
    std::string read_string() {
        uint64_t len; read_raw(&len, 8);
        std::string s(len, '\0'); read_raw(s.data(), len); return s;
    }
    uint32_t get_u32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0 : std::get<uint32_t>(it->second);
    }
    float get_f32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0.0f : std::get<float>(it->second);
    }
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

// =======================================================================
// PART 2: Chapter 4.2's fp16_t/BlockQ8 (Section 15.4's own copy).
// =======================================================================
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
#pragma pack(pop)
static_assert(sizeof(BlockQ8) == 34);
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

// =======================================================================
// PART 3: memory-mapped file + dequantization (Section 15.4's own copy).
// =======================================================================
struct MappedFile {
    void* data = nullptr;
    size_t size = 0;
    int fd = -1;
    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0) { ::close(fd); fd = -1; return false; }
        size = static_cast<size_t>(st.st_size);
        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; ::close(fd); fd = -1; return false; }
        return true;
    }
    const uint8_t* at(uint64_t byte_offset) const { return static_cast<const uint8_t*>(data) + byte_offset; }
    ~MappedFile() { if (data) ::munmap(data, size); if (fd >= 0) ::close(fd); }
};
std::vector<float> dequantize_tensor(const MappedFile& mf, const GGUFReader& r, const TensorInfo& t) {
    std::vector<float> out(t.n_elements);
    const uint8_t* p = mf.at(r.data_section_offset + t.offset);
    if (t.type == GGML_F32) {
        std::memcpy(out.data(), p, t.n_elements * sizeof(float));
    } else if (t.type == GGML_Q8_0) {
        uint64_t n_blocks = t.n_elements / 32;
        for (uint64_t b = 0; b < n_blocks; ++b) {
            BlockQ8 blk;
            std::memcpy(&blk, p + b * sizeof(BlockQ8), sizeof(BlockQ8));
            float s = static_cast<float>(blk.scale);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(blk.weights[i]) * s;
        }
    }
    return out;
}
void dequantize_row_q8(const MappedFile& mf, uint64_t abs_row_offset, size_t n_elements, std::span<float> out) {
    const uint8_t* p = mf.at(abs_row_offset);
    size_t n_blocks = n_elements / 32;
    for (size_t b = 0; b < n_blocks; ++b) {
        BlockQ8 blk;
        std::memcpy(&blk, p + b * sizeof(BlockQ8), sizeof(BlockQ8));
        float s = static_cast<float>(blk.scale);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(blk.weights[i]) * s;
    }
}

// =======================================================================
// PART 4: Section 15.3's adapted transformer block (Section 15.4's own
// copy).
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
void linear_with_bias(std::span<float> out, std::span<const float> x, std::span<const float> W,
                       std::span<const float> bias, size_t in_dim, size_t out_dim) {
    matmul(out, x, W, in_dim, out_dim);
    for (size_t j = 0; j < out_dim; ++j) out[j] += bias[j];
}
struct RoPETables {
    std::vector<float> cos_vals, sin_vals;
    int half_dim;
    RoPETables(int seq_len, int head_dim, float base) : half_dim(head_dim / 2) {
        cos_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        sin_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * theta;
                cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::cos(angle);
                sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::sin(angle);
            }
        }
    }
    float cos_at(int pos, int k) const { return cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
    float sin_at(int pos, int k) const { return sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
};
void apply_rope(std::span<float> vec, int pos, const RoPETables& tables) {
    const int half_dim = tables.half_dim;
    for (int k = 0; k < half_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + half_dim)];
        float c = tables.cos_at(pos, k), s = tables.sin_at(pos, k);
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + half_dim)] = x1 * s + x2 * c;
    }
}
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
        K.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
        V.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
    }
    View k_view() { return View(K.data(), n_heads_kv, max_seq_len, head_dim); }
    View v_view() { return View(V.data(), n_heads_kv, max_seq_len, head_dim); }
    auto k_at(int h, int t) { return std::submdspan(k_view(), h, t, std::full_extent); }
    auto v_at(int h, int t) { return std::submdspan(v_view(), h, t, std::full_extent); }
    void store(int h, int t, std::span<const float> k, std::span<const float> v) {
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[static_cast<size_t>(i)]; vslice[i] = v[static_cast<size_t>(i)]; }
    }
};
void gqa_attention(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                    int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(seq_len));
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, static_cast<size_t>(head_dim));
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            double d = 0.0;
            for (int i = 0; i < head_dim; ++i) d += static_cast<double>(q[i]) * static_cast<double>(k[i]);
            scores[static_cast<size_t>(t)] = static_cast<float>(d) * scale;
        }
        softmax_inplace(std::span<float>(scores.data(), static_cast<size_t>(seq_len)));
        std::vector<double> acc(static_cast<size_t>(head_dim), 0.0);
        for (int t = 0; t < seq_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            double w = scores[static_cast<size_t>(t)];
            for (int i = 0; i < head_dim; ++i) acc[static_cast<size_t>(i)] += w * static_cast<double>(v[i]);
        }
        float* out = output.data() + h * head_dim;
        for (int i = 0; i < head_dim; ++i) out[i] = static_cast<float>(acc[static_cast<size_t>(i)]);
    }
}
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
    std::vector<float> normed(static_cast<size_t>(shape.dim)), q(static_cast<size_t>(shape.q_dim())),
        k(static_cast<size_t>(shape.kv_dim())), v(static_cast<size_t>(shape.kv_dim()));
    std::vector<float> attn_out(static_cast<size_t>(shape.q_dim())), proj_out(static_cast<size_t>(shape.dim));
    rms_norm(normed, x, w.attn_norm);
    linear_with_bias(q, normed, w.Wq, w.bq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.q_dim()));
    linear_with_bias(k, normed, w.Wk, w.bk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    linear_with_bias(v, normed, w.Wv, w.bv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    for (int h = 0; h < shape.n_heads; ++h)
        apply_rope(std::span<float>(q.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        apply_rope(std::span<float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        cache.store(h, pos, std::span<const float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                             std::span<const float>(v.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)));
    gqa_attention(q, cache, attn_out, pos + 1, shape.n_heads, shape.group_size());
    matmul(proj_out, attn_out, w.Wo, static_cast<size_t>(shape.q_dim()), static_cast<size_t>(shape.dim));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += proj_out[static_cast<size_t>(i)];
    std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
    rms_norm(normed2, x, w.ffn_norm);
    swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += ffn_out[static_cast<size_t>(i)];
}
struct QwenModel {
    MappedFile mf;
    GGUFReader r;
    QwenShape shape{};
    RoPETables* rope = nullptr;
    bool load(const std::string& path) {
        if (!mf.open(path)) return false;
        if (!r.open(path)) return false;
        shape.dim = static_cast<int>(r.get_u32("qwen2.embedding_length"));
        shape.n_heads = static_cast<int>(r.get_u32("qwen2.attention.head_count"));
        shape.n_heads_kv = static_cast<int>(r.get_u32("qwen2.attention.head_count_kv"));
        shape.head_dim = shape.dim / shape.n_heads;
        shape.d_ff = static_cast<int>(r.get_u32("qwen2.feed_forward_length"));
        float base = r.get_f32("qwen2.rope.freq_base");
        rope = new RoPETables(4096, shape.head_dim, base);
        return true;
    }
    ~QwenModel() { delete rope; }
    int n_layers() const { return static_cast<int>(r.get_u32("qwen2.block_count")); }
    std::vector<float> tensor(const std::string& name) const {
        const auto* t = r.find_tensor(name);
        return dequantize_tensor(mf, r, *t);
    }
    QwenBlockWeights layer(int idx) const {
        std::string p = "blk." + std::to_string(idx) + ".";
        QwenBlockWeights w;
        w.attn_norm = tensor(p + "attn_norm.weight");
        w.Wq = tensor(p + "attn_q.weight");   w.bq = tensor(p + "attn_q.bias");
        w.Wk = tensor(p + "attn_k.weight");   w.bk = tensor(p + "attn_k.bias");
        w.Wv = tensor(p + "attn_v.weight");   w.bv = tensor(p + "attn_v.bias");
        w.Wo = tensor(p + "attn_output.weight");
        w.ffn_norm = tensor(p + "ffn_norm.weight");
        w.Wgate = tensor(p + "ffn_gate.weight");
        w.Wup = tensor(p + "ffn_up.weight");
        w.Wdown = tensor(p + "ffn_down.weight");
        return w;
    }
    std::vector<float> embedding(int token_id) const {
        const auto* t = r.find_tensor("token_embd.weight");
        uint64_t row_offset = r.data_section_offset + t->offset
                             + (static_cast<uint64_t>(token_id) * static_cast<uint64_t>(shape.dim) / 32) * sizeof(BlockQ8);
        std::vector<float> out(static_cast<size_t>(shape.dim));
        dequantize_row_q8(mf, row_offset, static_cast<size_t>(shape.dim), out);
        return out;
    }
};

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
// PART 6 (new): the fusion itself. `decode_step_with_embedding` is
// Section 16.3's own `decode_step`, factored so the "how do I get this
// position's starting vector" question is answered by the CALLER --
// `decode_step` (below) answers it the original way (look up a real
// token id); `build_fused_embeddings` answers it a second way (splice in
// an already-computed visual token) for exactly the positions a caller
// marks as image positions.
// =======================================================================
struct DecodeStepResult { std::vector<float> hidden; int nan_at_layer = -1; };
DecodeStepResult decode_step_with_embedding(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                                              std::vector<KVCache>& caches, std::vector<float> x, int pos) {
    DecodeStepResult res;
    for (int layer = 0; layer < model.n_layers(); ++layer) {
        qwen2_block_forward(x, model.shape, layers[static_cast<size_t>(layer)], caches[static_cast<size_t>(layer)], pos, *model.rope);
        bool has_nan = false;
        for (float v : x) if (std::isnan(v)) { has_nan = true; break; }
        if (has_nan) { res.nan_at_layer = layer; break; }
    }
    res.hidden = std::move(x);
    return res;
}
DecodeStepResult decode_step(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                               std::vector<KVCache>& caches, int token_id, int pos) {
    return decode_step_with_embedding(model, layers, caches, model.embedding(token_id), pos);
}

// A sentinel token id reserved (by this section's own synthetic
// vocabulary, exactly the way real Qwen2-VL reserves `<|image_pad|>`'s
// own real id) to mark "a visual token belongs here" inside an otherwise
// ordinary text token-id sequence.
constexpr int IMAGE_PLACEHOLDER_ID = -1;

struct FusionResult {
    bool ok = false;
    std::string error;
    std::vector<std::vector<float>> embeddings;   // one per position, text-looked-up or visual-injected
};

// Walks `token_ids` in order, replacing every `IMAGE_PLACEHOLDER_ID`
// with the NEXT unused visual token from `visual_tokens`, in sequence --
// and refuses, rather than silently truncating or zero-padding, if the
// placeholder count and the visual token count do not match exactly.
// A mismatch here is exactly the kind of silent-misalignment bug this
// book has refused to let slide since Chapter 15's own KVCache-sharing
// bug: every later position's own attention would still run and produce
// SOME finite number, with nothing about the output signaling that the
// image tokens landed in the wrong places, or that some were reused, or
// dropped.
FusionResult build_fused_embeddings(const QwenModel& model, const std::vector<int>& token_ids,
                                     const std::vector<std::vector<float>>& visual_tokens) {
    FusionResult res;
    size_t used = 0;
    res.embeddings.reserve(token_ids.size());
    for (int tid : token_ids) {
        if (tid == IMAGE_PLACEHOLDER_ID) {
            if (used >= visual_tokens.size()) {
                res.error = "more image placeholders than visual tokens";
                return res;
            }
            res.embeddings.push_back(visual_tokens[used++]);
        } else {
            res.embeddings.push_back(model.embedding(tid));
        }
    }
    if (used != visual_tokens.size()) {
        res.error = "fewer image placeholders than visual tokens (" + std::to_string(used) +
                    " used, " + std::to_string(visual_tokens.size()) + " provided)";
        return res;
    }
    res.ok = true;
    return res;
}

struct MultimodalPrefillResult { std::vector<float> hidden; bool exceeded_capacity = false; bool fusion_failed = false;
                                  std::string fusion_error; int nan_at_layer = -1; };
MultimodalPrefillResult prefill_multimodal(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                                            std::vector<KVCache>& caches, const std::vector<int>& token_ids,
                                            const std::vector<std::vector<float>>& visual_tokens,
                                            int start_pos, int kv_capacity) {
    MultimodalPrefillResult res;
    auto fused = build_fused_embeddings(model, token_ids, visual_tokens);
    if (!fused.ok) { res.fusion_failed = true; res.fusion_error = fused.error; return res; }
    for (size_t i = 0; i < fused.embeddings.size(); ++i) {
        int pos = start_pos + static_cast<int>(i);
        if (pos >= kv_capacity) { res.exceeded_capacity = true; break; }
        auto step = decode_step_with_embedding(model, layers, caches, fused.embeddings[i], pos);
        res.hidden = std::move(step.hidden);
        if (step.nan_at_layer >= 0) { res.nan_at_layer = step.nan_at_layer; break; }
    }
    return res;
}

// Runs Section 18.2's own encoder end to end -- patchify, embed, N ViT
// blocks, merge -- and returns the resulting visual tokens, projected to
// the LLM's own embedding dimension so `build_fused_embeddings` can drop
// them straight into a text sequence with no further adaptation.
struct VisionEncoderWeights {
    std::vector<float> W_patch_embed;
    std::vector<ViTBlockWeights> layers;
    MergerWeights merger;
};
std::vector<std::vector<float>> run_vision_encoder(const RawImage& image, int patch_size, const NormStats& norm,
                                                    const ViTShape& shape, const VisionEncoderWeights& w,
                                                    int llm_dim, uint32_t& out_grid_h, uint32_t& out_grid_w) {
    uint32_t grid_h = 0, grid_w = 0;
    auto raw_patches = patchify(image, static_cast<uint32_t>(patch_size), norm, grid_h, grid_w);
    const int n_patches = static_cast<int>(raw_patches.size());
    const size_t patch_vec_len = raw_patches[0].size();

    std::vector<std::vector<float>> x(static_cast<size_t>(n_patches));
    std::vector<int> rows(static_cast<size_t>(n_patches)), cols(static_cast<size_t>(n_patches));
    for (int i = 0; i < n_patches; ++i) {
        x[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.dim));
        matmul(x[static_cast<size_t>(i)], raw_patches[static_cast<size_t>(i)], w.W_patch_embed, patch_vec_len, static_cast<size_t>(shape.dim));
        rows[static_cast<size_t>(i)] = i / static_cast<int>(grid_w);
        cols[static_cast<size_t>(i)] = i % static_cast<int>(grid_w);
    }
    RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, shape.head_dim, 10000.0f);
    for (const auto& l : w.layers) vit_block_forward(x, shape, l, rows, cols, rope);

    out_grid_h = grid_h; out_grid_w = grid_w;
    return merge_all_2x2(x, static_cast<int>(grid_h), static_cast<int>(grid_w), shape.dim, w.merger, llm_dim);
}


// =======================================================================
// PART 7: self-tests, against a small synthetic Qwen2-shaped decoder
// (Section 15.4's own synthetic-model pattern, with `rope.freq_base`
// written explicitly this time -- Section 17.2's own lesson about that
// field's zero default applies here exactly as it did there) and a
// small synthetic image, so every check is deterministic and needs
// neither a real checkpoint nor a real camera.
// =======================================================================
std::vector<float> rand_vec(std::mt19937& rng, size_t n) {
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 18.3: Wiring the Vision Encoder into the Qwen2 Decoder\n";
    std::cout << "========================================================\n";

    // ---- synthetic Qwen2-shaped decoder, exactly Section 17.2's own shape ----
    constexpr int S_DIM = 32, S_HEADS = 4, S_HEADS_KV = 2, S_HEAD_DIM = 8, S_FF = 64, S_LAYERS = 2, S_VOCAB = 64;
    const std::string synth_path = "/tmp/ch18_3_synthetic_model.gguf";
    auto write_synthetic_model = [&](unsigned seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        struct Pending { std::string name; std::vector<uint64_t> dims; GGMLType type; std::vector<uint8_t> bytes; };
        std::vector<Pending> pending;
        auto add_f32 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& data) {
            std::vector<uint8_t> bytes(data.size() * 4);
            std::memcpy(bytes.data(), data.data(), bytes.size());
            pending.push_back({name, dims, GGML_F32, bytes});
        };
        auto add_q8 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& data) {
            size_t n_blocks = data.size() / 32;
            std::vector<uint8_t> bytes(n_blocks * sizeof(BlockQ8));
            for (size_t b = 0; b < n_blocks; ++b) {
                BlockQ8 blk = quantize_q8(data.data() + b * 32);
                std::memcpy(bytes.data() + b * sizeof(BlockQ8), &blk, sizeof(BlockQ8));
            }
            pending.push_back({name, dims, GGML_Q8_0, bytes});
        };
        auto rand_v = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
        add_q8("token_embd.weight", {S_DIM, S_VOCAB}, rand_v(static_cast<size_t>(S_DIM) * S_VOCAB));
        add_f32("output_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
        for (int layer = 0; layer < S_LAYERS; ++layer) {
            std::string p = "blk." + std::to_string(layer) + ".";
            add_f32(p + "attn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "attn_q.weight", {S_DIM, S_HEADS * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS * S_HEAD_DIM));
            add_f32(p + "attn_q.bias", {static_cast<uint64_t>(S_HEADS * S_HEAD_DIM)}, rand_v(S_HEADS * S_HEAD_DIM));
            add_q8(p + "attn_k.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_k.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_v(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_v.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_v.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_v(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_output.weight", {S_HEADS * S_HEAD_DIM, S_DIM}, rand_v(static_cast<size_t>(S_HEADS) * S_HEAD_DIM * S_DIM));
            add_f32(p + "ffn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "ffn_gate.weight", {S_DIM, S_FF}, rand_v(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_up.weight", {S_DIM, S_FF}, rand_v(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_down.weight", {S_FF, S_DIM}, rand_v(static_cast<size_t>(S_FF) * S_DIM));
        }
        GGUFWriter w(synth_path);
        w.write_magic(); w.write_u32(3);
        w.write_u64(pending.size()); w.write_u64(7);   // 7 kv pairs written below -- Section 16.3's own lesson
        w.write_kv_string("general.architecture", "qwen2");
        w.write_kv_u32("qwen2.block_count", S_LAYERS);
        w.write_kv_u32("qwen2.embedding_length", S_DIM);
        w.write_kv_u32("qwen2.attention.head_count", S_HEADS);
        w.write_kv_u32("qwen2.attention.head_count_kv", S_HEADS_KV);
        w.write_kv_u32("qwen2.feed_forward_length", S_FF);
        w.write_kv_f32("qwen2.rope.freq_base", 10000.0f);
        uint64_t off = 0;
        for (auto& p : pending) { w.write_tensor_info(p.name, p.dims, p.type, off); off += p.bytes.size(); }
        w.align(32);
        for (auto& p : pending) w.write_bytes(p.bytes.data(), p.bytes.size());
        return w.good();
    };
    CHECK(write_synthetic_model(7));
    QwenModel model;
    CHECK(model.load(synth_path));
    std::vector<QwenBlockWeights> layers;
    for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));

    // ---- synthetic ViT-shaped vision encoder ----
    constexpr int PATCH = 14, CH = 3, VIT_DIM = 16, V_HEADS = 2, V_HEAD_DIM = 8, V_FF = 24, V_LAYERS = 1;
    NormStats norm;
    auto make_vision_weights = [&](unsigned seed) {
        std::mt19937 rng(seed);
        VisionEncoderWeights vw;
        vw.W_patch_embed = rand_vec(rng, static_cast<size_t>(VIT_DIM) * (PATCH * PATCH * CH));
        vw.layers.resize(V_LAYERS);
        for (auto& l : vw.layers) {
            l.attn_norm.assign(VIT_DIM, 1.0f);
            l.Wq = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wk = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wv = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wo = rand_vec(rng, static_cast<size_t>(VIT_DIM) * (V_HEADS * V_HEAD_DIM));
            l.ffn_norm.assign(VIT_DIM, 1.0f);
            l.Wgate = rand_vec(rng, static_cast<size_t>(V_FF) * VIT_DIM);
            l.Wup = rand_vec(rng, static_cast<size_t>(V_FF) * VIT_DIM);
            l.Wdown = rand_vec(rng, static_cast<size_t>(VIT_DIM) * V_FF);
        }
        vw.merger.hidden_dim = 20;
        vw.merger.W1 = rand_vec(rng, static_cast<size_t>(vw.merger.hidden_dim) * (4 * VIT_DIM));
        vw.merger.b1 = rand_vec(rng, static_cast<size_t>(vw.merger.hidden_dim));
        vw.merger.W2 = rand_vec(rng, static_cast<size_t>(S_DIM) * vw.merger.hidden_dim);
        vw.merger.b2 = rand_vec(rng, static_cast<size_t>(S_DIM));
        return vw;
    };
    VisionEncoderWeights vision_weights = make_vision_weights(123);
    ViTShape vit_shape{VIT_DIM, V_HEADS, V_HEAD_DIM, V_FF};

    auto make_image = [&](unsigned seed) {
        RawImage img; img.width = 56; img.height = 56; img.channels = CH;   // 4x4 patch grid -> 4 merged tokens
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        img.pixels.resize(static_cast<size_t>(img.width) * img.height * CH);
        for (auto& b : img.pixels) b = static_cast<uint8_t>(byte_dist(rng));
        return img;
    };

    std::cout << "\n-- Test 1: fusion refuses a placeholder/visual-token count mismatch --\n";
    {
        std::vector<std::vector<float>> four_tokens(4, std::vector<float>(S_DIM, 0.1f));
        std::vector<int> ok_ids = {5, 6, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, 7};
        auto ok_res = build_fused_embeddings(model, ok_ids, four_tokens);
        CHECK(ok_res.ok);
        CHECK(ok_res.embeddings.size() == ok_ids.size());

        std::vector<int> too_few_placeholders = {5, 6, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, 7};
        auto short_res = build_fused_embeddings(model, too_few_placeholders, four_tokens);
        CHECK(!short_res.ok);
        CHECK(!short_res.error.empty());

        std::vector<int> too_many_placeholders = {5, 6, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID,
                                                   IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, 7};
        auto long_res = build_fused_embeddings(model, too_many_placeholders, four_tokens);
        CHECK(!long_res.ok);
        CHECK(!long_res.error.empty());

        std::cout << "  matching count (4 placeholders, 4 visual tokens): ok; fewer placeholders than "
                     "visual tokens: rejected (\"" << short_res.error << "\"); more placeholders than "
                     "visual tokens: rejected (\"" << long_res.error << "\")\n";
    }

    std::cout << "\n-- Test 2: end-to-end multimodal prefill (image + text) is finite and deterministic --\n";
    {
        constexpr int KV_CAP = 32;
        auto image = make_image(50);
        auto run_once = [&]() {
            uint32_t gh = 0, gw = 0;
            auto visual_tokens = run_vision_encoder(image, PATCH, norm, vit_shape, vision_weights, S_DIM, gh, gw);
            std::vector<int> token_ids = {5, 6};
            for (size_t i = 0; i < visual_tokens.size(); ++i) token_ids.push_back(IMAGE_PLACEHOLDER_ID);
            token_ids.insert(token_ids.end(), {7, 8, 9});
            std::vector<KVCache> caches;
            for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, KV_CAP, model.shape.head_dim);
            return prefill_multimodal(model, layers, caches, token_ids, visual_tokens, 0, KV_CAP);
        };
        auto r1 = run_once();
        auto r2 = run_once();
        CHECK(!r1.fusion_failed && !r1.exceeded_capacity && r1.nan_at_layer < 0);
        bool finite = true;
        for (float v : r1.hidden) if (!std::isfinite(v)) finite = false;
        CHECK(finite);
        CHECK(r1.hidden == r2.hidden);
        std::cout << "  9-position fused sequence (2 text + 4 image + 3 text) prefilled successfully, "
                     "final hidden state finite: " << (finite ? "yes" : "no") << ", deterministic across "
                     "two independent runs: " << (r1.hidden == r2.hidden ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Test 3: changing the image changes downstream text computation --\n";
    {
        constexpr int KV_CAP = 32;
        auto run_with_image = [&](const RawImage& image) {
            uint32_t gh = 0, gw = 0;
            auto visual_tokens = run_vision_encoder(image, PATCH, norm, vit_shape, vision_weights, S_DIM, gh, gw);
            std::vector<int> token_ids = {5, 6};
            for (size_t i = 0; i < visual_tokens.size(); ++i) token_ids.push_back(IMAGE_PLACEHOLDER_ID);
            token_ids.insert(token_ids.end(), {7, 8, 9});
            std::vector<KVCache> caches;
            for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, KV_CAP, model.shape.head_dim);
            return prefill_multimodal(model, layers, caches, token_ids, visual_tokens, 0, KV_CAP);
        };
        auto image_a = make_image(50);
        auto image_b = make_image(999);   // genuinely different pixel content
        auto res_a = run_with_image(image_a);
        auto res_b = run_with_image(image_b);
        CHECK(res_a.hidden != res_b.hidden);
        std::cout << "  the SAME text tokens with two DIFFERENT images produce different final hidden "
                     "states: " << (res_a.hidden != res_b.hidden ? "yes" : "no")
                   << " -- the image is genuinely being attended to, not silently ignored\n";
    }

    std::cout << "\n-- Test 4: KV-capacity exhaustion is still caught in a fused sequence --\n";
    {
        constexpr int KV_CAP = 5;   // smaller than the 9-position fused sequence below
        auto image = make_image(50);
        uint32_t gh = 0, gw = 0;
        auto visual_tokens = run_vision_encoder(image, PATCH, norm, vit_shape, vision_weights, S_DIM, gh, gw);
        std::vector<int> token_ids = {5, 6};
        for (size_t i = 0; i < visual_tokens.size(); ++i) token_ids.push_back(IMAGE_PLACEHOLDER_ID);
        token_ids.insert(token_ids.end(), {7, 8, 9});
        std::vector<KVCache> caches;
        for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, KV_CAP, model.shape.head_dim);
        auto res = prefill_multimodal(model, layers, caches, token_ids, visual_tokens, 0, KV_CAP);
        CHECK(res.exceeded_capacity);
        CHECK(!res.fusion_failed);
        std::cout << "  a " << token_ids.size() << "-position fused sequence against a " << KV_CAP
                   << "-position cache correctly reports exceeded_capacity=" << (res.exceeded_capacity ? "true" : "false") << "\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
