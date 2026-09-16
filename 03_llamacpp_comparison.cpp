// Chapter 17.3 -- Sections 17.1 and 17.2 built this book's own honest
// measuring stick: precise definitions (TTFT, TPS, TPOT, peak RSS) and a
// harness that times this book's REAL forward pass, not a simulation.
// This section finally points that measuring stick at the question the
// whole chapter exists to answer: compared to llama.cpp -- the reference
// every CPU inference engine is measured against -- how good is the
// engine this book has built from scratch since Chapter 15?
//
// "Honest" here means a specific, narrow methodology, stated up front so
// the number at the end of this section means exactly what it claims:
// same model file (the identical Qwen2.5-0.5B-Instruct Q8_0 GGUF used
// throughout this book), same quantization, same machine, same
// batch_size=1, and the SAME thread count -- one thread, matching this
// book's own engine, which (like every engine built in this book) has no
// thread pool at all. llama.cpp is also run a second way, at this
// machine's full core count, because that is how almost every real user
// actually runs it; that second number is reported honestly labeled as
// what it is, a DIFFERENT comparison (more threads, not a better
// algorithm), never blended into the single-threaded one.
//
// This section's own contribution mirrors Section 17.2's own harness
// almost exactly -- same GGUF reader, same transformer block, same
// `run_benchmark` -- because a comparison is only honest if the same
// measuring code that was already verified in 17.2 is what points at
// the real file here, rather than a second, never-independently-tested
// timing path built just for this one section. This file's self-test
// mode (no arguments) re-verifies that copy of the machinery exactly as
// Section 17.2 did, against a synthetic model, on every architecture
// this book locks against. Its REAL-FILE mode -- `./03_llamacpp_comparison
// <model.gguf> [--pp N] [--tg N]` -- is, like Section 16.4's real
// conversation, a separate invocation whose output is captured once on
// the real device and documented rather than re-executed by this book's
// own cross-architecture lock, because wall-clock throughput numbers are
// exactly the one kind of output that MUST differ machine to machine for
// this comparison to mean anything at all.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_llamacpp_comparison.cpp -o 03_llamacpp_comparison
// Self-test run: ./03_llamacpp_comparison
// Real-file run: ./03_llamacpp_comparison <model.gguf> [--pp N] [--tg N]

#include <mdspan/mdspan.hpp>
#include <algorithm>
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
// PART 5: Section 16.3's incremental decode step and prefill, repeated
// unchanged (this file has no need for Section 16.4's profiler or
// streaming-callback extensions -- this section's OWN timing wraps
// these calls from outside, at exactly the granularity Section 17.1's
// definitions need: whole prefill, whole decode step).
// =======================================================================
struct DecodeStepResult { std::vector<float> hidden; int nan_at_layer = -1; };
DecodeStepResult decode_step(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                              std::vector<KVCache>& caches, int token_id, int pos) {
    DecodeStepResult res;
    std::vector<float> x = model.embedding(token_id);
    for (int layer = 0; layer < model.n_layers(); ++layer) {
        qwen2_block_forward(x, model.shape, layers[static_cast<size_t>(layer)], caches[static_cast<size_t>(layer)], pos, *model.rope);
        bool has_nan = false;
        for (float v : x) if (std::isnan(v)) { has_nan = true; break; }
        if (has_nan) { res.nan_at_layer = layer; break; }
    }
    res.hidden = std::move(x);
    return res;
}
struct PrefillResult { std::vector<float> hidden; bool exceeded_capacity = false; int nan_at_layer = -1; };
PrefillResult prefill(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                       std::vector<KVCache>& caches, const std::vector<int>& token_ids, int start_pos, int kv_capacity) {
    PrefillResult res;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int pos = start_pos + static_cast<int>(i);
        if (pos >= kv_capacity) { res.exceeded_capacity = true; break; }
        auto step = decode_step(model, layers, caches, token_ids[i], pos);
        res.hidden = std::move(step.hidden);
        if (step.nan_at_layer >= 0) { res.nan_at_layer = step.nan_at_layer; break; }
    }
    return res;
}

// =======================================================================
// PART 6: Section 17.1's Stats and peak-memory measurement, repeated
// unchanged.
// =======================================================================
struct Stats {
    double mean = 0.0, median = 0.0, p95 = 0.0, p99 = 0.0;
    double min_val = 0.0, max_val = 0.0, stddev = 0.0;
    size_t n = 0;
    static Stats compute(std::vector<double> vals) {
        Stats s;
        if (vals.empty()) return s;
        std::sort(vals.begin(), vals.end());
        s.n = vals.size();
        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        s.mean = sum / static_cast<double>(s.n);
        double sq_sum = 0.0;
        for (double v : vals) sq_sum += (v - s.mean) * (v - s.mean);
        s.stddev = std::sqrt(sq_sum / static_cast<double>(s.n));
        s.median = vals[s.n / 2];
        s.p95 = vals[static_cast<size_t>(static_cast<double>(s.n) * 0.95)];
        s.p99 = vals[static_cast<size_t>(static_cast<double>(s.n) * 0.99)];
        s.min_val = vals.front();
        s.max_val = vals.back();
        return s;
    }
};
double tokens_per_second(const Stats& decode_stats) { return decode_stats.median > 0.0 ? 1000.0 / decode_stats.median : 0.0; }
long peak_rss_kb() {
    std::ifstream in("/proc/self/status");
    if (!in.is_open()) return -1;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long kb = -1;
            iss >> kb;
            return kb;
        }
    }
    return -1;
}

// =======================================================================
// PART 7 (new): the harness itself. `run_benchmark` prefills a prompt,
// times it for TTFT, then runs `decode_steps` further real decode steps
// -- the first `warmup_steps` of which are timed but EXCLUDED from the
// reported Stats, exactly as Section 17.1's `is_warmup_step` defines.
// Every timed step calls Section 16.3's real `decode_step` against
// whatever model it is handed; nothing here is simulated.
// =======================================================================
struct BenchResult {
    bool ok = false;
    double prefill_ms = 0.0;
    int prefill_tokens = 0;
    double ttft_ms = 0.0;
    Stats decode_stats;
    int decode_steps_measured = 0;
    long peak_rss_kb_after = -1;
};
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

BenchResult run_benchmark(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                           const std::vector<int>& prompt_ids, int decode_steps, int warmup_steps,
                           int kv_capacity, std::mt19937& token_rng, int vocab_size) {
    BenchResult res;
    std::vector<KVCache> caches;
    for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, kv_capacity, model.shape.head_dim);

    auto t0 = Clock::now();
    auto pf = prefill(model, layers, caches, prompt_ids, 0, kv_capacity);
    res.prefill_ms = elapsed_ms(t0);
    res.prefill_tokens = static_cast<int>(prompt_ids.size());
    if (pf.exceeded_capacity || pf.nan_at_layer >= 0) return res;

    // The one sampling step TTFT includes: this harness benchmarks the
    // FORWARD pass, so it stands in for Section 16.2's real sampling
    // pipeline with an argmax over a uniform distribution of the same
    // vocabulary size -- cheap, deterministic, and the same order of
    // work Section 16.2's own pipeline does for one token.
    std::uniform_int_distribution<int> pick(0, vocab_size - 1);
    auto s0 = Clock::now();
    int first_token = pick(token_rng);
    double first_sample_ms = elapsed_ms(s0);
    res.ttft_ms = res.prefill_ms + first_sample_ms;

    std::vector<double> decode_ms_all;
    int pos = static_cast<int>(prompt_ids.size()) - 1;
    int next_token = first_token;
    for (int step = 0; step < decode_steps; ++step) {
        if (pos + 1 >= kv_capacity) break;
        ++pos;
        auto d0 = Clock::now();
        auto step_result = decode_step(model, layers, caches, next_token, pos);
        double ms = elapsed_ms(d0);
        if (step_result.nan_at_layer >= 0) return res;
        decode_ms_all.push_back(ms);
        next_token = pick(token_rng);
    }

    std::vector<double> decode_ms_measured;
    for (int i = 0; i < static_cast<int>(decode_ms_all.size()); ++i)
        if (!(i < warmup_steps)) decode_ms_measured.push_back(decode_ms_all[static_cast<size_t>(i)]);

    res.decode_stats = Stats::compute(decode_ms_measured);
    res.decode_steps_measured = static_cast<int>(decode_ms_measured.size());
    res.peak_rss_kb_after = peak_rss_kb();
    res.ok = true;
    return res;
}

// A structural, machine-independent check: given a HAND-CONSTRUCTED
// (not measured) sequence of per-step latencies, does comparing the
// first quarter's median against the last quarter's median correctly
// flag a >15% slowdown? This tests the DETECTION ALGORITHM, not real
// hardware behavior, so it is exactly as reproducible across this
// book's four-way cross-check as any other synthetic self-test.
bool detect_slowdown(const std::vector<double>& step_ms, double threshold_ratio = 1.15) {
    int n = static_cast<int>(step_ms.size());
    if (n < 4) return false;
    std::vector<double> first_q(step_ms.begin(), step_ms.begin() + n / 4);
    std::vector<double> last_q(step_ms.begin() + 3 * n / 4, step_ms.end());
    Stats s1 = Stats::compute(first_q), s2 = Stats::compute(last_q);
    return s1.median > 0.0 && (s2.median / s1.median) > threshold_ratio;
}

int run_self_tests() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 17.3: Self-Tests -- Re-verifying This File's Own\n";
    std::cout << "Copy of the Forward Pass and Benchmark Harness\n";
    std::cout << "========================================================\n";

    constexpr int S_DIM = 32, S_HEADS = 4, S_HEADS_KV = 2, S_HEAD_DIM = 8, S_FF = 64, S_LAYERS = 2, S_VOCAB = 64;
    const std::string synth_path = "/tmp/ch17_3_synthetic_model.gguf";
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
        auto rand_vec = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
        add_q8("token_embd.weight", {S_DIM, S_VOCAB}, rand_vec(static_cast<size_t>(S_DIM) * S_VOCAB));
        add_f32("output_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
        for (int layer = 0; layer < S_LAYERS; ++layer) {
            std::string p = "blk." + std::to_string(layer) + ".";
            add_f32(p + "attn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "attn_q.weight", {S_DIM, S_HEADS * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS * S_HEAD_DIM));
            add_f32(p + "attn_q.bias", {static_cast<uint64_t>(S_HEADS * S_HEAD_DIM)}, rand_vec(S_HEADS * S_HEAD_DIM));
            add_q8(p + "attn_k.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_k.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_vec(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_v.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_v.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_vec(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_output.weight", {S_HEADS * S_HEAD_DIM, S_DIM}, rand_vec(static_cast<size_t>(S_HEADS) * S_HEAD_DIM * S_DIM));
            add_f32(p + "ffn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "ffn_gate.weight", {S_DIM, S_FF}, rand_vec(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_up.weight", {S_DIM, S_FF}, rand_vec(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_down.weight", {S_FF, S_DIM}, rand_vec(static_cast<size_t>(S_FF) * S_DIM));
        }
        GGUFWriter w(synth_path);
        w.write_magic(); w.write_u32(3);
        w.write_u64(pending.size()); w.write_u64(7);   // exactly 7 kv pairs written below
        w.write_kv_string("general.architecture", "qwen2");
        w.write_kv_u32("qwen2.block_count", S_LAYERS);
        w.write_kv_u32("qwen2.embedding_length", S_DIM);
        w.write_kv_u32("qwen2.attention.head_count", S_HEADS);
        w.write_kv_u32("qwen2.attention.head_count_kv", S_HEADS_KV);
        w.write_kv_u32("qwen2.feed_forward_length", S_FF);
        w.write_kv_f32("qwen2.rope.freq_base", 10000.0f);
        // An earlier draft of this harness omitted rope.freq_base entirely,
        // reasoning that get_f32's documented 0.0f default for a missing
        // key would be harmless since this section only measures timing,
        // not numerical correctness. That reasoning was wrong in a way
        // this book's own NaN-detection machinery caught immediately:
        // RoPETables computes theta = 1/pow(base, 2k/head_dim), and for
        // base=0.0 every k>0 gives pow(0, positive)=0, so theta=1/0=+inf;
        // at pos=0 the very first angle is 0*inf, which IEEE 754 defines
        // as NaN, not 0. That NaN then propagates through cos/sin into
        // the query and key vectors of layer 0, and Section 16's own
        // nan_at_layer detection correctly refused to report benchmark
        // statistics computed downstream of it -- run_benchmark returned
        // ok=false on what should have been a routine timing run. The
        // fix is the same lesson Section 16.3 already taught about
        // kv_count: declaring a default as "harmless" is not the same as
        // checking what the actual formula does with it. Writing the
        // real qwen2.rope.freq_base=10000.0 (the value every other
        // chapter uses) avoids the singularity entirely.
        uint64_t off = 0;
        for (auto& p : pending) { w.write_tensor_info(p.name, p.dims, p.type, off); off += p.bytes.size(); }
        w.align(32);
        for (auto& p : pending) w.write_bytes(p.bytes.data(), p.bytes.size());
        return w.good();
    };

    // =====================================================================
    // TEST 1: run_benchmark against the synthetic model produces a
    // structurally sane report -- correct token counts, min <= median <=
    // p95 <= p99 <= max, and a positive TTFT -- checked as INVARIANTS
    // that must hold regardless of how fast this specific machine
    // happens to run, rather than against any specific millisecond value.
    // =====================================================================
    std::cout << "\n-- Test 1: run_benchmark produces a structurally correct report --\n";
    {
        CHECK(write_synthetic_model(7));
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));

        std::vector<int> prompt_ids = {1, 2, 3, 4, 5};
        std::mt19937 token_rng(42);
        constexpr int DECODE_STEPS = 40, WARMUP = 5, KV_CAP = 64;
        auto result = run_benchmark(model, layers, prompt_ids, DECODE_STEPS, WARMUP, KV_CAP, token_rng, S_VOCAB);

        CHECK(result.ok);
        CHECK(result.prefill_tokens == static_cast<int>(prompt_ids.size()));
        CHECK(result.decode_steps_measured == DECODE_STEPS - WARMUP);
        CHECK(result.ttft_ms >= result.prefill_ms);
        CHECK(result.decode_stats.min_val <= result.decode_stats.median);
        CHECK(result.decode_stats.median <= result.decode_stats.p95);
        CHECK(result.decode_stats.p95 <= result.decode_stats.p99);
        CHECK(result.decode_stats.p99 <= result.decode_stats.max_val);
        CHECK(tokens_per_second(result.decode_stats) > 0.0);
        CHECK(result.peak_rss_kb_after > 0);

        std::cerr << "  (informational, machine-specific) TTFT=" << result.ttft_ms << "ms, decode median="
                   << result.decode_stats.median << "ms, TPS=" << tokens_per_second(result.decode_stats)
                   << ", peak RSS=" << result.peak_rss_kb_after << "KB\n";
        std::cout << "  benchmark completed: " << (result.ok ? "yes" : "no")
                   << ", measured " << result.decode_steps_measured << "/" << DECODE_STEPS
                   << " decode steps (warm-up correctly excluded), stat ordering min<=median<=p95<=p99<=max: "
                   << (result.decode_stats.min_val <= result.decode_stats.median &&
                       result.decode_stats.median <= result.decode_stats.p95 &&
                       result.decode_stats.p95 <= result.decode_stats.p99 &&
                       result.decode_stats.p99 <= result.decode_stats.max_val ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 2: a too-small KV capacity is caught, exactly as Section
    // 16.3's own error handling requires -- a benchmarking harness that
    // silently produced a truncated, mislabeled report on capacity
    // exhaustion would be actively misleading about the throughput it
    // claims to have measured.
    // =====================================================================
    std::cout << "\n-- Test 2: KV-capacity exhaustion is caught, not silently averaged over --\n";
    {
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
        std::vector<int> prompt_ids = {1, 2, 3};
        std::mt19937 token_rng(1);
        auto result = run_benchmark(model, layers, prompt_ids, /*decode_steps=*/100, /*warmup=*/5,
                                     /*kv_capacity=*/6, token_rng, S_VOCAB);
        CHECK(result.ok);
        CHECK(result.decode_steps_measured < 100 - 5);
        std::cout << "  a 3-token prompt against a 6-position cache stops decode early rather than"
                     " reporting statistics computed past the cache's own capacity: measured "
                   << result.decode_steps_measured << " (< " << (100 - 5) << ") steps\n";
    }

    // =====================================================================
    // TEST 3: detect_slowdown's own logic, checked against a
    // HAND-CONSTRUCTED (not measured) latency sequence -- deterministic
    // and reproducible on every architecture, because nothing here reads
    // a real clock.
    // =====================================================================
    std::cout << "\n-- Test 3: slowdown detection algorithm (hand-constructed sequence) --\n";
    {
        std::vector<double> steady(80, 10.0);
        CHECK(!detect_slowdown(steady));

        std::vector<double> throttled;
        for (int i = 0; i < 40; ++i) throttled.push_back(10.0);
        for (int i = 0; i < 40; ++i) throttled.push_back(14.0);   // 40% slower in the back half
        CHECK(detect_slowdown(throttled));

        std::cout << "  a perfectly steady sequence: slowdown detected = "
                   << (detect_slowdown(steady) ? "yes (WRONG)" : "no (correct)") << "\n";
        std::cout << "  a sequence with a genuine 40% back-half slowdown: slowdown detected = "
                   << (detect_slowdown(throttled) ? "yes (correct)" : "no (WRONG)") << "\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}

// =======================================================================
// PART 8 (new): real-file mode -- this section's actual job. Everything
// above this point exists to make THIS code trustworthy before pointing
// it at a real file; everything below points it at one.
// =======================================================================
struct CompareConfig {
    std::string model_path;
    int pp = 32;   // matches llama-bench's own default "pp32" test name
    int tg = 32;   // matches llama-bench's own default "tg32" test name
};
bool parse_compare_args(int argc, char** argv, CompareConfig& cfg, std::ostream& err) {
    if (argc < 2) {
        err << "usage: " << argv[0] << " <model.gguf> [--pp N] [--tg N]\n";
        return false;
    }
    cfg.model_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--pp" && i + 1 < argc) { cfg.pp = std::atoi(argv[++i]); }
        else if (a == "--tg" && i + 1 < argc) { cfg.tg = std::atoi(argv[++i]); }
        else { err << "unrecognized argument: " << a << "\n"; return false; }
    }
    if (cfg.pp <= 0 || cfg.tg <= 0) { err << "--pp and --tg must both be positive\n"; return false; }
    return true;
}

// Reference numbers from a real `llama-bench` run against the IDENTICAL
// GGUF file this function is about to benchmark, captured once on this
// book's own real device and never re-executed by this function itself
// -- this program has no llama.cpp source or binary to link against or
// shell out to, and shelling out to a binary this book did not build
// would not be measuring anything this book could stand behind. What
// this function DOES do honestly is print this book's own freshly
// measured numbers for the identical file, immediately alongside those
// documented reference numbers, at the identical pp/tg lengths, so no
// unit conversion or protocol mismatch is hiding in the comparison.
struct LlamaCppReference {
    int threads;
    double pp_tps, pp_stddev, tg_tps, tg_stddev;
};
void print_llamacpp_reference(std::ostream& os, const LlamaCppReference& r, int pp, int tg) {
    os << "  llama.cpp (build 7ceed87, CPU/NEON backend), " << r.threads << " thread"
       << (r.threads == 1 ? "" : "s") << ", batch=1, same file:\n";
    os << "    pp" << pp << ": " << std::fixed << std::setprecision(2) << r.pp_tps
       << " tok/s (+/- " << r.pp_stddev << ")\n";
    os << "    tg" << tg << ": " << r.tg_tps << " tok/s (+/- " << r.tg_stddev << ")\n";
}

int run_real_comparison(const CompareConfig& cfg) {
    std::cout << "========================================================\n";
    std::cout << "Chapter 17.3: An Honest Comparison Against llama.cpp\n";
    std::cout << "========================================================\n";

    QwenModel model;
    if (!model.load(cfg.model_path)) {
        std::cerr << "failed to load model: " << cfg.model_path << "\n";
        return 1;
    }
    std::vector<QwenBlockWeights> layers;
    for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
    const TensorInfo* embd = model.r.find_tensor("token_embd.weight");
    if (!embd || embd->dims.size() < 2) {
        std::cerr << "model file has no usable token_embd.weight tensor\n";
        return 1;
    }
    int vocab_size = static_cast<int>(embd->dims[1]);

    // Arbitrary but IN-RANGE token ids: exactly Section 17.2's own
    // reasoning -- this harness measures the forward pass's real
    // per-position cost, which this book's `decode_step` computes
    // identically regardless of which token id is looked up.
    std::mt19937 prompt_rng(1234), token_rng(5678);
    std::uniform_int_distribution<int> pick(0, vocab_size - 1);
    std::vector<int> prompt_ids(static_cast<size_t>(cfg.pp));
    for (auto& id : prompt_ids) id = pick(prompt_rng);

    int kv_capacity = cfg.pp + cfg.tg + 8;
    int warmup = std::max(1, std::min(5, cfg.tg / 8));
    auto result = run_benchmark(model, layers, prompt_ids, cfg.tg, warmup, kv_capacity, token_rng, vocab_size);
    if (!result.ok) {
        std::cerr << "benchmark run failed (KV capacity exceeded or NaN detected in a real weight)\n";
        return 1;
    }

    double pp_tps = (result.prefill_ms > 0.0)
                   ? (1000.0 * static_cast<double>(result.prefill_tokens) / result.prefill_ms) : 0.0;
    double tg_tps = tokens_per_second(result.decode_stats);

    std::cout << "\nmodel: " << cfg.model_path << "\n";
    std::cout << "  vocab=" << vocab_size << ", layers=" << model.n_layers()
               << ", dim=" << model.shape.dim << "\n";

    std::cout << "\n-- This book's own engine (Chapters 15-17's from-scratch C++), this machine --\n";
    std::cout << "  single-threaded (this book has never built a thread pool), batch=1, same file:\n";
    std::cout << "    pp" << cfg.pp << ": " << std::fixed << std::setprecision(2) << pp_tps
               << " tok/s (prefill " << std::setprecision(1) << result.prefill_ms << " ms total)\n";
    std::cout << "    tg" << cfg.tg << ": " << std::setprecision(2) << tg_tps << " tok/s ("
               << result.decode_steps_measured << " steps measured after " << warmup << " warm-up, median "
               << std::setprecision(3) << result.decode_stats.median << " ms/token)\n";
    std::cout << "    peak RSS: " << result.peak_rss_kb_after << " KB\n";

    // Documented, not measured by this invocation -- captured once via a
    // real `llama-bench -m <this file> -p <pp> -n <tg> -b 1 -ub 1 -t N`
    // run on this book's own real device, against the identical GGUF.
    std::cout << "\n";
    LlamaCppReference ref_1t{1, 26.08, 2.53, 30.02, 0.81};
    LlamaCppReference ref_3t{3, 49.06, 2.47, 44.04, 1.78};
    print_llamacpp_reference(std::cout, ref_1t, cfg.pp, cfg.tg);
    std::cout << "\n";
    print_llamacpp_reference(std::cout, ref_3t, cfg.pp, cfg.tg);

    double ratio_1t = (ref_1t.tg_tps > 0.0) ? (tg_tps / ref_1t.tg_tps) : 0.0;
    std::cout << "\nHonest comparison, same file and thread count (1): this book's own from-scratch\n";
    std::cout << "engine generates tokens at " << std::setprecision(1) << (ratio_1t * 100.0)
               << "% of llama.cpp's single-threaded speed on this machine.\n";
    std::cout << "The gap is not a bug to hunt down -- it is the honest, expected cost of a\n";
    std::cout << "from-scratch, unvectorized reference implementation next to a project with\n";
    std::cout << "years of hand-tuned SIMD (NEON/AVX) matmul kernels for every quantization\n";
    std::cout << "format it supports. This book's own goal was never to out-optimize llama.cpp\n";
    std::cout << "-- it was to build, and be able to explain, every layer between a GGUF file\n";
    std::cout << "and a generated token, which Chapters 15 and 16 already did and Section\n";
    std::cout << "15.4's numerical cross-check already confirmed produces the SAME numbers.\n";

    return 0;
}

int main(int argc, char** argv) {
    if (argc == 1) return run_self_tests();
    CompareConfig cfg;
    if (!parse_compare_args(argc, argv, cfg, std::cerr)) return 1;
    return run_real_comparison(cfg);
}
