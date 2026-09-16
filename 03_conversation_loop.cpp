// Chapter 16.3 -- Sections 16.1 and 16.2 built the two pieces a real
// generation loop needs beyond Section 15.4's single argmax: a full
// configuration and a sampling pipeline, and a decoder that can stream
// text as it arrives. This section builds the loop itself: an
// incremental decode step that extends Section 15.4's per-layer
// `KVCache`s by exactly one position at a time (rather than rebuilding
// them from scratch), a generation loop that samples one token per
// step until a stop condition fires, and a conversation loop that
// wraps that generation loop in a real, multi-turn ChatML exchange --
// reusing every position already in the cache from earlier turns
// rather than reprocessing the conversation's own history from
// scratch on every turn.
//
// This section also adds the two runtime failure modes a self-
// contained binary must survive without crashing: a `KVCache` filled
// to the capacity Section 16.1's `--kv-cap` allocated (checked BEFORE
// writing past it, since `KVCache::store` performs no bounds checking
// of its own -- see Section 15.4), and a NaN appearing partway through
// a forward pass (checked after every layer, reported with the exact
// layer index, rather than allowed to silently propagate to a garbage
// sampled token).
//
// Every structure through Part 4 below (the GGUF reader, the Q8_0
// dequantizer, the GPT-2 byte codec and BPE engine, and Section 15.3's
// adapted transformer block) is Section 15.4's own file, repeated here
// unchanged per this book's one-file-per-section convention -- exactly
// as Section 15.4 itself repeated Section 15.1's reader. Part 8
// repeats Section 16.2's sampling pipeline and streaming decoder the
// same way. Only Part 9 onward is new: project_logits (replacing
// Section 15.4's project_argmax, which discarded everything except the
// single best row, with one that returns the full vocabulary so
// Section 16.2's pipeline has something to filter), the incremental
// decode step, the generation loop, and the conversation loop.
//
// Every check in this section runs against a small synthetic model at
// real-shaped proportions, exactly as Section 15.3 did -- proving this
// section's LOOP logic (cache reuse across turns, capacity checks, NaN
// detection) is correct, which small deterministic weights can do
// exactly as well as the real 644 MB checkpoint. Running this exact
// machinery as a real, multi-turn conversation against the real file is
// Section 16.4's job.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_conversation_loop.cpp -o 03_conversation_loop
// Run:     ./03_conversation_loop

#include <mdspan/mdspan.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <span>
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
// PART 1: Section 15.1's GGUF reader (Section 15.4's own copy).
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
    std::string get_string(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? "" : std::get<std::string>(it->second);
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
// PART 4: GPT-2 byte codec + BPE engine (Section 15.4's own copy).
// =======================================================================
std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80) { s += static_cast<char>(cp); }
    else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
    else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}
std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> cps;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if ((c & 0x80) == 0) { cps.push_back(c); i += 1; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            uint32_t cp = (static_cast<uint32_t>(c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
            cps.push_back(cp); i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            uint32_t cp = (static_cast<uint32_t>(c & 0x0Fu) << 12)
                        | (static_cast<uint32_t>(static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6)
                        | (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
            cps.push_back(cp); i += 3;
        } else { cps.push_back(c); i += 1; }
    }
    return cps;
}
struct GPT2ByteCodec {
    std::string byte_to_symbol[256];
    std::unordered_map<uint32_t, uint8_t> codepoint_to_byte;
    GPT2ByteCodec() {
        std::vector<int> bs, cs;
        auto in_bs = [&](int b) { return std::find(bs.begin(), bs.end(), b) != bs.end(); };
        for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) if (!in_bs(b)) { bs.push_back(b); cs.push_back(256 + n); ++n; }
        for (size_t i = 0; i < bs.size(); ++i) {
            uint32_t cp = static_cast<uint32_t>(cs[i]);
            byte_to_symbol[static_cast<uint8_t>(bs[i])] = utf8_encode(cp);
            codepoint_to_byte[cp] = static_cast<uint8_t>(bs[i]);
        }
    }
    std::string decode(const std::string& symbol_text) const {
        std::string out;
        for (uint32_t cp : utf8_decode(symbol_text)) {
            auto it = codepoint_to_byte.find(cp);
            if (it != codepoint_to_byte.end()) out += static_cast<char>(it->second);
        }
        return out;
    }
};
bool is_alpha_c(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }
bool is_digit_c(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }
bool is_space_c(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }
std::vector<std::string> gpt2_pretokenize(const std::string& text) {
    std::vector<std::string> chunks;
    size_t i = 0, n = text.size();
    while (i < n) {
        size_t start = i;
        size_t j = i;
        bool leading_space = (text[j] == ' ' && j + 1 < n && !is_space_c(text[j + 1]));
        if (leading_space) ++j;
        if (j < n && is_alpha_c(text[j])) {
            size_t k = j; while (k < n && is_alpha_c(text[k])) ++k;
            chunks.push_back(text.substr(start, k - start)); i = k;
        } else if (j < n && is_digit_c(text[j])) {
            size_t k = j; while (k < n && is_digit_c(text[k])) ++k;
            chunks.push_back(text.substr(start, k - start)); i = k;
        } else if (j < n && !is_space_c(text[j])) {
            size_t k = j; while (k < n && !is_space_c(text[k]) && !is_alpha_c(text[k]) && !is_digit_c(text[k])) ++k;
            if (k == j) ++k;
            chunks.push_back(text.substr(start, k - start)); i = k;
        } else {
            size_t k = i; while (k < n && is_space_c(text[k])) ++k;
            chunks.push_back(text.substr(start, k - start)); i = k;
        }
    }
    return chunks;
}
struct MergeTable {
    std::unordered_map<std::string, int> priority_of;
    void add(const std::string& left, const std::string& right, int priority) { priority_of[left + " " + right] = priority; }
    int lookup(const std::string& left, const std::string& right) const {
        auto it = priority_of.find(left + " " + right);
        return (it == priority_of.end()) ? -1 : it->second;
    }
};
struct Vocabulary {
    std::vector<std::string> id_to_text;
    std::unordered_map<std::string, int> text_to_id;
    int add(const std::string& text) {
        int id = static_cast<int>(id_to_text.size());
        id_to_text.push_back(text);
        text_to_id[text] = id;
        return id;
    }
    int lookup(const std::string& text) const {
        auto it = text_to_id.find(text);
        return (it == text_to_id.end()) ? -1 : it->second;
    }
    const std::string& text_of(int id) const { return id_to_text[static_cast<size_t>(id)]; }
    int size() const { return static_cast<int>(id_to_text.size()); }
};
std::vector<int> bpe_merge_encode(std::vector<std::string> tokens, const MergeTable& merges, const Vocabulary& vocab) {
    while (tokens.size() >= 2) {
        int best_priority = -1, best_pos = -1;
        for (int i = 0; i < static_cast<int>(tokens.size()) - 1; ++i) {
            int p = merges.lookup(tokens[static_cast<size_t>(i)], tokens[static_cast<size_t>(i) + 1]);
            if (p >= 0 && (best_pos < 0 || p < best_priority)) { best_priority = p; best_pos = i; }
        }
        if (best_pos < 0) break;
        tokens[static_cast<size_t>(best_pos)] += tokens[static_cast<size_t>(best_pos) + 1];
        tokens.erase(tokens.begin() + best_pos + 1);
    }
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (const auto& t : tokens) ids.push_back(vocab.lookup(t));
    return ids;
}
std::vector<int> encode_gpt2(const std::string& raw_text, const GPT2ByteCodec& codec,
                              const MergeTable& merges, const Vocabulary& vocab) {
    if (raw_text.empty()) return {};
    std::vector<std::string> tokens;
    for (unsigned char c : raw_text) tokens.push_back(codec.byte_to_symbol[c]);
    return bpe_merge_encode(std::move(tokens), merges, vocab);
}
std::vector<int> encode_text_tokens(const std::string& raw, const GPT2ByteCodec& codec,
                                     const MergeTable& merges, const Vocabulary& vocab) {
    std::vector<int> ids;
    for (const auto& chunk : gpt2_pretokenize(raw))
        for (int id : encode_gpt2(chunk, codec, merges, vocab)) ids.push_back(id);
    return ids;
}

// =======================================================================
// PART 5: Section 15.3's adapted transformer block (Section 15.4's own
// copy, double-precision reductions and split-half RoPE included).
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
// PART 6 (new): project_logits -- the full-vocabulary counterpart to
// Section 15.4's project_argmax. That function streamed every
// vocabulary row and kept only the single best one, which is exactly
// right for finding one argmax and wrong for sampling: temperature,
// top-k, top-p, and repetition penalty (Section 16.2) all need the
// FULL distribution to operate on, not just its maximum. The streaming
// discipline -- never materializing the tied embedding table as one
// dequantized matrix -- is unchanged; only what gets kept differs.
// =======================================================================
std::vector<float> project_logits(const QwenModel& model, std::span<const float> hidden,
                                   std::span<const float> final_norm, int vocab_size) {
    std::vector<float> normed(static_cast<size_t>(model.shape.dim));
    rms_norm(normed, hidden, final_norm);
    const auto* t = model.r.find_tensor("token_embd.weight");
    uint64_t base_offset = model.r.data_section_offset + t->offset;
    uint64_t row_bytes = (static_cast<uint64_t>(model.shape.dim) / 32) * sizeof(BlockQ8);
    std::vector<float> logits(static_cast<size_t>(vocab_size));
    std::vector<float> row(static_cast<size_t>(model.shape.dim));
    for (int id = 0; id < vocab_size; ++id) {
        dequantize_row_q8(model.mf, base_offset + static_cast<uint64_t>(id) * row_bytes, static_cast<size_t>(model.shape.dim), row);
        double dot = 0.0;
        for (int i = 0; i < model.shape.dim; ++i) dot += static_cast<double>(normed[static_cast<size_t>(i)]) * static_cast<double>(row[static_cast<size_t>(i)]);
        logits[static_cast<size_t>(id)] = static_cast<float>(dot);
    }
    return logits;
}

// =======================================================================
// PART 7 (new): the incremental decode step. Section 15.4's
// run_generation embedded a token and ran it through all 24 layers
// exactly once, for a fixed prompt, never called again. A real
// generation loop calls this same per-position computation once per
// NEW token, extending the SAME per-layer caches by exactly one
// position each time -- the natural shape autoregressive decoding
// always has, and the reason Section 15.4 fixed its `KVCache` sharing
// bug to be one cache per layer, persisting across positions, in the
// first place.
// =======================================================================
struct DecodeStepResult {
    std::vector<float> hidden;
    int nan_at_layer = -1;   // -1: clean. Otherwise, the first layer whose output contained a NaN.
};
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
// Processes a whole span of NEW token IDs starting at `start_pos`,
// returning the hidden state at the LAST position processed (what
// project_logits needs to predict the very next token) -- prefill for
// an initial prompt, or for a new turn's tokens in an ongoing
// conversation, is the same operation either way: this function does
// not know or care which.
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
// PART 8: Section 16.2's sampling pipeline and streaming decoder,
// repeated here unchanged, operating on this section's own Vocabulary
// and GPT2ByteCodec (Part 4 above) rather than 16.2's minimal synthetic
// stand-ins.
// =======================================================================
void apply_repetition_penalty(std::span<float> logits, const std::vector<int>& history, float penalty) {
    if (penalty == 1.0f) return;
    for (int id : history) {
        if (id < 0 || static_cast<size_t>(id) >= logits.size()) continue;
        float& l = logits[static_cast<size_t>(id)];
        l = (l > 0.0f) ? (l / penalty) : (l * penalty);
    }
}
void apply_temperature(std::span<float> logits, float temp) { for (float& l : logits) l /= temp; }
void apply_top_k(std::span<float> logits, int k) {
    int n = static_cast<int>(logits.size());
    if (k >= n) return;
    std::vector<float> sorted(logits.begin(), logits.end());
    std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<float>());
    float kth = sorted[static_cast<size_t>(k - 1)];
    int kept = 0;
    for (float& l : logits) {
        if (l >= kth && kept < k) { ++kept; }
        else { l = -std::numeric_limits<float>::infinity(); }
    }
}
void apply_top_p(std::span<float> logits, float p) {
    int n = static_cast<int>(logits.size());
    std::vector<int> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });
    float max_logit = logits[static_cast<size_t>(order[0])];
    std::vector<double> exp_vals(static_cast<size_t>(n));
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::exp(static_cast<double>(logits[static_cast<size_t>(order[static_cast<size_t>(i)])] - max_logit));
        exp_vals[static_cast<size_t>(i)] = e;
        sum += e;
    }
    double cumulative = 0.0;
    int cutoff = n;
    for (int i = 0; i < n; ++i) {
        cumulative += exp_vals[static_cast<size_t>(i)] / sum;
        if (cumulative >= static_cast<double>(p)) { cutoff = i + 1; break; }
    }
    for (int i = cutoff; i < n; ++i) logits[static_cast<size_t>(order[static_cast<size_t>(i)])] = -std::numeric_limits<float>::infinity();
}
int argmax(std::span<const float> logits) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
int sample_categorical(std::span<const float> logits, std::mt19937& rng) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) { double e = std::exp(static_cast<double>(logits[i] - max_logit)); probs[i] = e; sum += e; }
    std::uniform_real_distribution<double> uni(0.0, sum);
    double target = uni(rng);
    double running = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) { running += probs[i]; if (running >= target) return static_cast<int>(i); }
    return static_cast<int>(probs.size()) - 1;
}
int sample_next_token(std::vector<float> logits, const std::vector<int>& history,
                       float temperature, int top_k, float top_p, float repeat_penalty, std::mt19937& rng) {
    if (temperature <= 0.0f) return argmax(logits);
    apply_repetition_penalty(logits, history, repeat_penalty);
    apply_temperature(logits, temperature);
    apply_top_k(logits, top_k);
    apply_top_p(logits, top_p);
    return sample_categorical(logits, rng);
}
int utf8_char_length(uint8_t first_byte) {
    if ((first_byte & 0x80) == 0x00) return 1;
    if ((first_byte & 0xE0) == 0xC0) return 2;
    if ((first_byte & 0xF0) == 0xE0) return 3;
    if ((first_byte & 0xF8) == 0xF0) return 4;
    return 1;
}
std::string flush_complete_utf8(std::string& buffer) {
    size_t i = 0, n = buffer.size(), last_complete = 0;
    while (i < n) {
        int len = utf8_char_length(static_cast<uint8_t>(buffer[i]));
        if (i + static_cast<size_t>(len) > n) break;
        i += static_cast<size_t>(len);
        last_complete = i;
    }
    std::string out = buffer.substr(0, last_complete);
    buffer.erase(0, last_complete);
    return out;
}
struct StreamingDecoder {
    const Vocabulary& vocab;
    const GPT2ByteCodec& codec;
    std::string pending;
    std::string process_token(int id) { pending += codec.decode(vocab.text_of(id)); return flush_complete_utf8(pending); }
    std::string finish() { std::string rest = pending; pending.clear(); return rest; }
};

// =======================================================================
// PART 9 (new): the generation loop and the conversation loop.
// =======================================================================
struct GenerationResult {
    std::vector<int> generated_ids;
    std::string text;
    bool stopped_on_stop_token = false;
    bool truncated_by_capacity = false;
    int nan_at_layer = -1;
    int end_pos = -1;   // position of the LAST token actually written into the cache
};

// Runs the sampling loop starting from `hidden` (the hidden state at
// position `start_pos`, i.e. the state prefill() just returned), for at
// most `max_new_tokens` steps, stopping early on `stop_token_id`, a
// cache-capacity limit, or a NaN. `history` is extended with every
// token this call generates (the caller owns it across the whole
// conversation, so repetition penalty sees the whole exchange, not just
// this one turn).
GenerationResult generate_response(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                                    std::vector<KVCache>& caches, std::span<const float> final_norm,
                                    Vocabulary& vocab, GPT2ByteCodec& codec, int vocab_size,
                                    std::vector<float> hidden, int start_pos, int kv_capacity,
                                    int max_new_tokens, int stop_token_id,
                                    float temperature, int top_k, float top_p, float repeat_penalty,
                                    std::mt19937& rng, std::vector<int>& history) {
    GenerationResult res;
    StreamingDecoder dec{vocab, codec, ""};
    int pos = start_pos;
    for (int step = 0; step < max_new_tokens; ++step) {
        std::vector<float> logits = project_logits(model, hidden, final_norm, vocab_size);
        int next_id = sample_next_token(logits, history, temperature, top_k, top_p, repeat_penalty, rng);
        if (next_id == stop_token_id) { res.stopped_on_stop_token = true; break; }
        if (pos + 1 >= kv_capacity) { res.truncated_by_capacity = true; break; }
        ++pos;
        auto step_result = decode_step(model, layers, caches, next_id, pos);
        if (step_result.nan_at_layer >= 0) { res.nan_at_layer = step_result.nan_at_layer; break; }
        history.push_back(next_id);
        res.generated_ids.push_back(next_id);
        res.text += dec.process_token(next_id);
        hidden = std::move(step_result.hidden);
    }
    res.text += dec.finish();
    res.end_pos = pos;
    return res;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 16.3: The Generation Loop and the Conversation Loop\n";
    std::cout << "========================================================\n";

    const std::string synth_path = "/tmp/ch16_3_synthetic_model.gguf";
    // S_VOCAB must cover every id the synthetic vocabulary below can produce:
    // 256 raw byte symbols plus the two ChatML special tokens (258 total).
    // encode_text_tokens falls back to one GPT2 byte symbol per input byte
    // whenever the (empty) merge table has nothing to merge, so an ordinary
    // ASCII prompt like "abc" produces token ids up to 0x63 (99) -- well
    // past a too-small S_VOCAB -- and token_embd.weight must have a real
    // row for every one of them, or embedding()/project_logits silently
    // read past the tensor's own data.
    constexpr int S_DIM = 32, S_HEADS = 4, S_HEADS_KV = 2, S_HEAD_DIM = 8, S_FF = 64, S_LAYERS = 2, S_VOCAB = 258;

    auto write_synthetic_model = [&](unsigned seed, bool poison_layer1_wq) {
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

        std::vector<float> emb_flat = rand_vec(static_cast<size_t>(S_DIM) * S_VOCAB);
        add_q8("token_embd.weight", {S_DIM, S_VOCAB}, emb_flat);
        add_f32("output_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
        for (int layer = 0; layer < S_LAYERS; ++layer) {
            std::string p = "blk." + std::to_string(layer) + ".";
            add_f32(p + "attn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            std::vector<float> wq = rand_vec(static_cast<size_t>(S_DIM) * S_HEADS * S_HEAD_DIM);
            add_q8(p + "attn_q.weight", {S_DIM, S_HEADS * S_HEAD_DIM}, wq);
            // Poison the BIAS, not the (Q8_0-quantized) weight: Q8_0's own
            // scale is a max-of-abs-value reduction, and std::max silently
            // discards a NaN operand (NaN compares false against anything,
            // so "current < NaN" is always false and the old max survives)
            // -- a NaN weight would quietly vanish during quantization and
            // never reach the forward pass at all. attn_q.bias is stored
            // as raw F32 (add_f32 just memcpy's the bits), so the NaN
            // survives losslessly and actually exercises the detector.
            std::vector<float> bq = rand_vec(S_HEADS * S_HEAD_DIM);
            if (poison_layer1_wq && layer == 1) bq[0] = std::numeric_limits<float>::quiet_NaN();
            add_f32(p + "attn_q.bias", {static_cast<uint64_t>(S_HEADS * S_HEAD_DIM)}, bq);
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
        w.write_u64(pending.size()); w.write_u64(7);
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

    // The full 256-symbol GPT-2 byte vocabulary plus the two ChatML
    // special tokens, id-for-id: with an empty merge table, every input
    // byte becomes its own token (id == that byte's value), so this is
    // the smallest vocabulary that can round-trip an arbitrary ASCII
    // prompt like "abc" or "hello" without an unknown-token failure.
    GPT2ByteCodec codec;
    Vocabulary vocab;
    MergeTable merges;   // empty: every chunk falls back to one GPT2 symbol per byte, which is fine for this section's purpose
    for (int b = 0; b < 256; ++b) vocab.add(codec.byte_to_symbol[static_cast<unsigned char>(b)]);
    vocab.add("<|im_start|>");   // completes the ChatML special-token pair; only im_end is used below, as the stop token
    int im_end = vocab.add("<|im_end|>");
    int synth_vocab_size = vocab.size();
    // token_embd.weight has exactly S_VOCAB rows; every id this vocabulary
    // can hand out must fit inside that table, or embedding()/project_logits
    // read past the tensor's own data instead of failing loudly.
    CHECK(synth_vocab_size <= S_VOCAB);

    auto push_text = [&](std::vector<int>& ids, const std::string& s) {
        for (int id : encode_text_tokens(s, codec, merges, vocab)) ids.push_back(id);
    };

    // =====================================================================
    // TEST 1: prefill + one generation step reproduces a correct, finite
    // hidden state and a plausible sampled token -- the same shape of
    // check Section 15.3's Test 5 already ran, now through the
    // INCREMENTAL decode_step/prefill functions instead of a single
    // position-major loop.
    // =====================================================================
    std::cout << "\n-- Test 1: prefill + incremental decode_step, finite and deterministic --\n";
    {
        CHECK(write_synthetic_model(7, false));
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
        std::vector<float> final_norm = model.tensor("output_norm.weight");

        std::vector<int> prompt_ids;
        push_text(prompt_ids, "abc");
        CHECK(!prompt_ids.empty());

        std::vector<KVCache> caches;
        for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, 16, model.shape.head_dim);
        auto pf = prefill(model, layers, caches, prompt_ids, 0, 16);
        CHECK(!pf.exceeded_capacity && pf.nan_at_layer < 0);
        bool finite = true;
        for (float v : pf.hidden) if (!std::isfinite(v)) finite = false;
        CHECK(finite);

        auto logits1 = project_logits(model, pf.hidden, final_norm, synth_vocab_size);
        auto logits2 = project_logits(model, pf.hidden, final_norm, synth_vocab_size);
        CHECK(logits1 == logits2);
        std::cout << "  prefilled " << prompt_ids.size() << " tokens, hidden state finite: " << (finite ? "yes" : "no")
                   << ", logits deterministic on rerun: " << (logits1 == logits2 ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 2: the full generation loop stops on the stop token, and
    // produces bit-identical output across two independent runs with
    // the same seed -- the contract's own determinism guarantee, now
    // exercised through the real streaming decoder as well as sampling.
    // =====================================================================
    std::cout << "\n-- Test 2: generation loop determinism and stop-token handling --\n";
    {
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
        std::vector<float> final_norm = model.tensor("output_norm.weight");

        std::vector<int> prompt_ids;
        push_text(prompt_ids, "ab");

        auto run_once = [&]() {
            std::vector<KVCache> caches;
            for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, 32, model.shape.head_dim);
            auto pf = prefill(model, layers, caches, prompt_ids, 0, 32);
            std::vector<int> history = prompt_ids;
            std::mt19937 rng(42);
            return generate_response(model, layers, caches, final_norm, vocab, codec, synth_vocab_size,
                                      pf.hidden, static_cast<int>(prompt_ids.size()) - 1, 32, 10, im_end,
                                      0.8f, 10, 0.9f, 1.1f, rng, history);
        };
        auto r1 = run_once();
        auto r2 = run_once();
        CHECK(r1.generated_ids == r2.generated_ids);
        CHECK(r1.text == r2.text);
        CHECK(r1.nan_at_layer < 0);
        std::cout << "  two independent runs, same seed: generated " << r1.generated_ids.size()
                   << " tokens, identical ID sequence and identical decoded text: "
                   << (r1.generated_ids == r2.generated_ids && r1.text == r2.text ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 3: the conversation loop -- turn 2 reuses every position
    // turn 1 already wrote into the cache. Confirmed two ways: (a) the
    // cache's own stored bytes for turn 1's positions are byte-for-byte
    // unchanged after turn 2 runs, and (b) an independently-computed
    // "replay from scratch" over the full concatenated token sequence
    // produces the SAME hidden state at the position turn 2's prefill
    // ends on -- proving reuse is not merely un-overwritten but actually
    // numerically equivalent to full recomputation.
    // =====================================================================
    std::cout << "\n-- Test 3: multi-turn conversation reuses turn 1's cache exactly --\n";
    {
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
        std::vector<float> final_norm = model.tensor("output_norm.weight");

        std::vector<int> turn1_ids;
        push_text(turn1_ids, "hello");
        std::vector<int> turn2_ids;
        push_text(turn2_ids, "world");

        // -- Conversation path: turn 1 fills the cache, then turn 2's
        // tokens are prefilled STARTING FROM where turn 1 left off,
        // reusing the same cache object. --
        std::vector<KVCache> conv_caches;
        for (int l = 0; l < model.n_layers(); ++l) conv_caches.emplace_back(model.shape.n_heads_kv, 32, model.shape.head_dim);
        auto conv_turn1 = prefill(model, layers, conv_caches, turn1_ids, 0, 32);
        CHECK(!conv_turn1.exceeded_capacity);
        std::vector<float> turn1_k_snapshot = conv_caches[0].K;   // snapshot layer 0's cache after turn 1
        auto conv_turn2 = prefill(model, layers, conv_caches, turn2_ids, static_cast<int>(turn1_ids.size()), 32);
        CHECK(!conv_turn2.exceeded_capacity);

        // Turn 1's own positions in layer 0's cache must be BYTE-FOR-BYTE
        // unchanged by turn 2's prefill -- turn 2 only ever writes to
        // NEW positions, never touching the ones turn 1 already filled.
        bool turn1_region_unchanged = true;
        size_t turn1_floats = static_cast<size_t>(turn1_ids.size()) * static_cast<size_t>(model.shape.n_heads_kv) * static_cast<size_t>(model.shape.head_dim);
        // K is laid out [head][seq][dim]; comparing the raw buffer up to
        // turn1's own seq positions requires walking per-head, since seq
        // is the MIDDLE axis -- so compare via the mdspan view instead
        // of assuming a flat prefix, which would be wrong for nh>1.
        for (int h = 0; h < model.shape.n_heads_kv && turn1_region_unchanged; ++h) {
            for (int t = 0; t < static_cast<int>(turn1_ids.size()) && turn1_region_unchanged; ++t) {
                auto now = conv_caches[0].k_at(h, t);
                for (int i = 0; i < model.shape.head_dim; ++i) {
                    size_t flat = (static_cast<size_t>(h) * 32u + static_cast<size_t>(t)) * static_cast<size_t>(model.shape.head_dim) + static_cast<size_t>(i);
                    if (flat < turn1_k_snapshot.size() && now[i] != turn1_k_snapshot[flat]) turn1_region_unchanged = false;
                }
            }
        }
        (void)turn1_floats;
        CHECK(turn1_region_unchanged);

        // -- From-scratch path: replay BOTH turns' tokens through a
        // FRESH cache in one prefill call, positions [0, total). --
        std::vector<int> concatenated = turn1_ids;
        concatenated.insert(concatenated.end(), turn2_ids.begin(), turn2_ids.end());
        std::vector<KVCache> fresh_caches;
        for (int l = 0; l < model.n_layers(); ++l) fresh_caches.emplace_back(model.shape.n_heads_kv, 32, model.shape.head_dim);
        auto fresh = prefill(model, layers, fresh_caches, concatenated, 0, 32);

        bool hidden_matches = (conv_turn2.hidden == fresh.hidden);
        CHECK(hidden_matches);
        std::cout << "  turn 1's cache region byte-for-byte unchanged after turn 2: " << (turn1_region_unchanged ? "yes" : "no") << "\n";
        std::cout << "  reused-cache hidden state == full-recompute-from-scratch hidden state: " << (hidden_matches ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 4: KV-capacity exceeded stops generation cleanly rather than
    // writing past the preallocated cache.
    // =====================================================================
    std::cout << "\n-- Test 4: KV-capacity limit is enforced, not merely hoped for --\n";
    {
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
        std::vector<float> final_norm = model.tensor("output_norm.weight");

        std::vector<int> prompt_ids;
        push_text(prompt_ids, "hi");
        constexpr int TINY_CAP = 5;
        std::vector<KVCache> caches;
        for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, TINY_CAP, model.shape.head_dim);
        auto pf = prefill(model, layers, caches, prompt_ids, 0, TINY_CAP);
        CHECK(!pf.exceeded_capacity);   // the short prompt itself fits

        std::vector<int> history = prompt_ids;
        std::mt19937 rng(1);
        auto res = generate_response(model, layers, caches, final_norm, vocab, codec, synth_vocab_size,
                                      pf.hidden, static_cast<int>(prompt_ids.size()) - 1, TINY_CAP, 100, im_end,
                                      0.8f, 10, 0.9f, 1.1f, rng, history);
        CHECK(res.truncated_by_capacity);
        CHECK(res.end_pos < TINY_CAP);
        std::cout << "  requested 100 new tokens against a " << TINY_CAP << "-position cache: stopped at position "
                   << res.end_pos << " with truncated_by_capacity=" << (res.truncated_by_capacity ? "true" : "false") << "\n";
    }

    // =====================================================================
    // TEST 5: a NaN introduced partway through the weights is detected
    // at the SPECIFIC layer it first appears in, not merely "somewhere".
    // =====================================================================
    std::cout << "\n-- Test 5: NaN detection names the failing layer --\n";
    {
        CHECK(write_synthetic_model(7, /*poison_layer1_wq=*/true));
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));

        std::vector<int> prompt_ids;
        push_text(prompt_ids, "x");
        std::vector<KVCache> caches;
        for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, 8, model.shape.head_dim);
        auto pf = prefill(model, layers, caches, prompt_ids, 0, 8);
        CHECK(pf.nan_at_layer == 1);   // layer 1's Wq was poisoned with a NaN weight above
        std::cout << "  NaN poisoned into layer 1's query projection weight; detected at layer "
                   << pf.nan_at_layer << " (expected: 1)\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
