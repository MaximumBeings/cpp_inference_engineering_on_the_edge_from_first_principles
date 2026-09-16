// Chapter 15.4 (capstone) -- every piece this chapter needs already
// exists: Section 15.1's reader (extended for BOOL, real ggml type IDs,
// and array metadata), Chapter 4.2's fp16_t/BlockQ8 Q8_0 dequantizer,
// Chapter 12.1's BPE merge engine, and Section 15.3's bias-adapted,
// 7:1-GQA-capable transformer block. This section wires all of them
// together, adds the one genuinely new piece a REAL byte-level BPE
// vocabulary requires -- GPT-2's reversible byte-to-unicode mapping,
// which Chapter 12's own synthetic vocabulary never needed because it
// used raw bytes as vocabulary strings directly -- and runs the result
// against the actual downloaded Qwen2.5-0.5B-Instruct checkpoint to
// produce one real, computed, decoded token.
//
// This is the same honest exception Sections 15.1 and 15.2 already
// established, now for the largest computation in the book so far: the
// self-tests below (no argument) build a tiny two-layer synthetic model
// with this file's own GGUFWriter and verify every piece of loading,
// dequantization, and the forward-projection-argmax pipeline against
// it, reproducing identically on this book's usual four-way
// cross-check. Real-file mode (a path as argv[1]) loads the actual
// ~644 MB, 24-layer, 151,936-token-vocabulary checkpoint, encodes a
// real ChatML prompt, runs a real 24-layer forward pass with real
// dequantized Q8_0 weights, and decodes a real generated token --
// executed via a shell on the reader's own machine, since that is the
// only place the real file exists.
//
// A stated simplification: gpt2_pretokenize below reproduces the
// STRUCTURE of the real GPT-2/tiktoken pre-tokenization regex (optional
// leading space attached to a following letter/digit/symbol run,
// trailing whitespace as its own chunk) for plain ASCII text without
// contractions, which is sufficient for the prompt this section chooses
// but is not a full Unicode-property-aware regex engine.
//
// Like Section 15.3, this file compiles with -ffp-contract=off --
// Chapter 11's cross-architecture floating-point fix, needed again
// here because a 2-layer synthetic forward pass (Test 4) is enough
// sequential reduction for GCC's default -ffp-contract=fast to fuse
// multiply-adds differently on x86_64 than on aarch64, producing a
// last-digit difference in the printed logit between this book's
// native and cross-compiled targets.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 04_first_real_token.cpp -o 04_first_real_token
// Run (self-tests only, works anywhere):    ./04_first_real_token
// Run (adds real generation):               ./04_first_real_token /path/to/model.gguf

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
static int r_tests = 0, r_passed = 0;
#define RCHECK(...) do { \
    r_tests++; \
    if (__VA_ARGS__) { r_passed++; } \
    else { std::cerr << "REAL-FILE FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: Section 15.1's GGUF reader, extended with array-metadata
// support (Chapter 12.5's contribution) for the real tokens/merges
// arrays this section actually needs to load.
// =======================================================================
enum GGUFValueType : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12,
};
enum GGMLType : uint32_t {
    GGML_F32 = 0, GGML_Q8_0 = 8,
};
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
    void write_kv_bool(const std::string& k, bool v) { write_string(k); write_u32(V_BOOL); write_raw(&v, 1); }
    void write_kv_array_string(const std::string& k, const std::vector<std::string>& values) {
        write_string(k); write_u32(V_ARRAY); write_u32(V_STRING); write_u64(values.size());
        for (const auto& v : values) write_string(v);
    }
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
    bool get_bool(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? false : std::get<bool>(it->second);
    }
    const std::vector<std::string>& get_string_array(const std::string& key) const {
        return std::get<std::vector<std::string>>(metadata.at(key));
    }
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

// =======================================================================
// PART 2: Chapter 4.2's fp16_t and BlockQ8, reused verbatim -- this is
// the machinery that turns the real file's Q8_0 bytes back into floats.
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
// PART 3: a memory-mapped file -- genuinely zero-copy tensor access,
// the loading strategy Chapter 5's own description promised. Dims are
// stored by GGUF with dims[0] as the CONTIGUOUS (innermost) axis and
// dims[1] as the outer axis; confirmed directly against the real file's
// own tensors (ffn_down.weight[4864,896] projects FROM 4864 TO 896, so
// dims[0]=in_dim is the contiguous axis) -- exactly the layout Chapter
// 3.2's matmul(out, x, W, in_dim, out_dim) already expects, so weight
// bytes read directly off disk need no transposition at all.
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
    ~MappedFile() {
        if (data) ::munmap(data, size);
        if (fd >= 0) ::close(fd);
    }
};

// Dequantizes a whole tensor (F32: a direct copy; Q8_0: block by block)
// into a fresh vector<float> in file order -- ready to hand straight to
// Chapter 3's matmul/rms_norm/etc. with no further rearrangement.
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

// Streams ONE row (contiguous n_elements-long slice) of a large Q8_0
// tensor without materializing the whole tensor -- used for both a
// single embedding lookup and, row by row, the full vocabulary
// projection at the end, so the 151,936-row tied embedding table never
// needs to exist as one dequantized 544 MB array in memory.
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
// PART 4: GPT-2's reversible byte<->unicode mapping -- new in this
// section. A real byte-level BPE vocabulary never stores raw bytes as
// vocabulary strings (many raw bytes, including plain space and
// newline, are not valid standalone UTF-8 text); instead every one of
// the 256 possible input bytes is mapped to one visible, always-valid
// UTF-8 symbol BEFORE the merge algorithm ever sees it, and the real
// vocabulary's own strings (things like the two-byte UTF-8 sequence for
// "Gspace", conventionally rendered "Ġ") are written in terms of
// those symbols. This is the standard, publicly documented GPT-2
// tokenizer construction (the same one OpenAI's own reference
// tokenizer and every GPT-2-vocabulary-compatible tokenizer use) --
// Chapter 12's synthetic vocabulary never needed it because that
// vocabulary was this book's own invention, free to use raw bytes
// directly as strings.
// =======================================================================
std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
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
        for (int b = 0; b < 256; ++b) {
            if (!in_bs(b)) { bs.push_back(b); cs.push_back(256 + n); ++n; }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            uint32_t cp = static_cast<uint32_t>(cs[i]);
            byte_to_symbol[static_cast<uint8_t>(bs[i])] = utf8_encode(cp);
            codepoint_to_byte[cp] = static_cast<uint8_t>(bs[i]);
        }
    }
    std::string encode(const std::string& raw_text) const {
        std::string out;
        for (unsigned char c : raw_text) out += byte_to_symbol[c];
        return out;
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

// GPT-2/tiktoken-style pre-tokenization, structurally: an optional
// single leading space attaches to a following letter run, digit run,
// or symbol run; a whitespace run with nothing non-space after it is
// its own chunk. See the file header for the stated scope of this
// approximation.
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

// =======================================================================
// PART 5: Chapter 12.1's MergeTable/Vocabulary/BPEEncoder, reused
// unchanged -- the merge ALGORITHM never needed to know whether its
// initial symbols are raw bytes or GPT-2-mapped byte symbols; only the
// SEEDING (encode_gpt2 below) differs from Chapter 12's own encode().
// =======================================================================
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
    const std::string& text_of(int id) const { return id_to_text[id]; }
    int size() const { return static_cast<int>(id_to_text.size()); }
};
std::vector<int> bpe_merge_encode(std::vector<std::string> tokens, const MergeTable& merges, const Vocabulary& vocab) {
    while (tokens.size() >= 2) {
        int best_priority = -1, best_pos = -1;
        for (int i = 0; i < static_cast<int>(tokens.size()) - 1; ++i) {
            int p = merges.lookup(tokens[i], tokens[i + 1]);
            if (p >= 0 && (best_pos < 0 || p < best_priority)) { best_priority = p; best_pos = i; }
        }
        if (best_pos < 0) break;
        tokens[best_pos] += tokens[best_pos + 1];
        tokens.erase(tokens.begin() + best_pos + 1);
    }
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (const auto& t : tokens) ids.push_back(vocab.lookup(t));
    return ids;
}
// The new seeding this section adds: one GPT-2-mapped symbol per raw
// input byte, instead of Chapter 12's one-raw-byte-as-one-char-string.
std::vector<int> encode_gpt2(const std::string& raw_text, const GPT2ByteCodec& codec,
                              const MergeTable& merges, const Vocabulary& vocab) {
    if (raw_text.empty()) return {};
    std::vector<std::string> tokens;
    for (unsigned char c : raw_text) tokens.push_back(codec.byte_to_symbol[c]);
    return bpe_merge_encode(std::move(tokens), merges, vocab);
}

// =======================================================================
// PART 6: Section 15.3's adapted block. rms_norm, matmul and
// gqa_attention below accumulate in double rather than float -- the fix
// this section's own real-generation debugging led to (see the account
// further down): a naive float32 running sum, repeated across 24 real
// layers and thousands of elements per reduction, drifts enough by the
// last layer to flip the argmax of a real generation. Everything else
// here is reused verbatim from Section 15.3.
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
    // Split-half (NeoX/HF-style) pairing -- (vec[k], vec[k+half_dim]) --
    // not the interleaved pairing this function used before this
    // section's real-generation debugging. Confirmed against an
    // independently built llama.cpp running the same real GGUF file: see
    // the account below of how this was found.
    const int half_dim = tables.half_dim;
    for (int k = 0; k < half_dim; ++k) {
        float x1 = vec[k], x2 = vec[k + half_dim];
        float c = tables.cos_at(pos, k), s = tables.sin_at(pos, k);
        vec[k] = x1 * c - x2 * s;
        vec[k + half_dim] = x1 * s + x2 * c;
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
void gqa_attention(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                    int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len);
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, head_dim);
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            double d = 0.0;
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
    matmul(proj_out, attn_out, w.Wo, shape.q_dim(), shape.dim);
    for (int i = 0; i < shape.dim; ++i) x[i] += proj_out[i];
    std::vector<float> normed2(shape.dim), ffn_out(shape.dim);
    rms_norm(normed2, x, w.ffn_norm);
    swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, shape.dim, shape.d_ff);
    for (int i = 0; i < shape.dim; ++i) x[i] += ffn_out[i];
}

// =======================================================================
// PART 7: the model wrapper -- loads one layer's weights on demand
// (peak memory: one layer at a time, never all 24 simultaneously) and
// streams the final vocabulary projection row by row.
// =======================================================================
struct QwenModel {
    MappedFile mf;
    GGUFReader r;
    QwenShape shape{};
    RoPETables* rope = nullptr;
    bool add_bos = false;

    bool load(const std::string& path) {
        if (!mf.open(path)) return false;
        if (!r.open(path)) return false;
        shape.dim = static_cast<int>(r.get_u32("qwen2.embedding_length"));
        shape.n_heads = static_cast<int>(r.get_u32("qwen2.attention.head_count"));
        shape.n_heads_kv = static_cast<int>(r.get_u32("qwen2.attention.head_count_kv"));
        shape.head_dim = shape.dim / shape.n_heads;
        shape.d_ff = static_cast<int>(r.get_u32("qwen2.feed_forward_length"));
        add_bos = r.get_bool("tokenizer.ggml.add_bos_token");
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
                             + (static_cast<uint64_t>(token_id) * shape.dim / 32) * sizeof(BlockQ8);
        std::vector<float> out(shape.dim);
        dequantize_row_q8(mf, row_offset, static_cast<size_t>(shape.dim), out);
        return out;
    }
    // Streams every vocabulary row against `hidden`, returning the
    // argmax id and its logit without ever materializing the full
    // vocab x dim dequantized matrix.
    std::pair<int, float> project_argmax(std::span<const float> hidden, int vocab_size) const {
        const auto* t = r.find_tensor("token_embd.weight");   // tied -- Section 15.2 proved this byte for byte
        uint64_t base_offset = r.data_section_offset + t->offset;
        uint64_t row_bytes = (static_cast<uint64_t>(shape.dim) / 32) * sizeof(BlockQ8);
        std::vector<float> row(shape.dim);
        int best_id = -1;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int id = 0; id < vocab_size; ++id) {
            dequantize_row_q8(mf, base_offset + static_cast<uint64_t>(id) * row_bytes, static_cast<size_t>(shape.dim), row);
            float dot = 0.0f;
            for (int i = 0; i < shape.dim; ++i) dot += hidden[i] * row[i];
            if (dot > best_logit) { best_logit = dot; best_id = id; }
        }
        return {best_id, best_logit};
    }
};

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "Chapter 15.4: A First Real Token From Qwen2.5-0.5B-Instruct\n";
    std::cout << "========================================================\n";
    GPT2ByteCodec codec;

    // =====================================================================
    // TEST 1: GPT-2 byte<->unicode round trip for all 256 bytes, plus the
    // two spot-checked values every gpt2-vocabulary reader relies on.
    // =====================================================================
    std::cout << "\n-- Test 1: GPT-2 byte<->unicode codec --\n";
    {
        bool all_round_trip = true;
        for (int b = 0; b < 256; ++b) {
            std::string sym = codec.byte_to_symbol[b];
            std::string back = codec.decode(sym);
            if (back.size() != 1 || static_cast<unsigned char>(back[0]) != b) all_round_trip = false;
        }
        CHECK(all_round_trip);
        CHECK(codec.byte_to_symbol[static_cast<unsigned char>('A')] == "A");           // printable ASCII: unchanged
        CHECK(codec.byte_to_symbol[static_cast<unsigned char>(' ')].size() == 2);      // space: mapped to a 2-byte symbol
        std::cout << "  all 256 bytes round-trip through encode+decode: " << (all_round_trip ? "yes" : "no") << "\n";
        std::cout << "  byte 'A' (printable) maps to itself; byte ' ' maps to a "
                   << codec.byte_to_symbol[static_cast<unsigned char>(' ')].size() << "-byte UTF-8 symbol\n";
    }

    // =====================================================================
    // TEST 2: pre-tokenization chunk boundaries on plain ASCII text.
    // =====================================================================
    std::cout << "\n-- Test 2: gpt2_pretokenize chunk boundaries --\n";
    {
        auto chunks = gpt2_pretokenize("The capital of France is");
        std::vector<std::string> expected = {"The", " capital", " of", " France", " is"};
        CHECK(chunks == expected);
        std::cout << "  \"The capital of France is\" -> " << chunks.size() << " chunks: ";
        for (auto& c : chunks) std::cout << "[" << c << "]";
        std::cout << "\n";

        auto chunks2 = gpt2_pretokenize("What is the capital of France?");
        std::vector<std::string> expected2 = {"What", " is", " the", " capital", " of", " France", "?"};
        CHECK(chunks2 == expected2);
        std::cout << "  \"What is the capital of France?\" -> " << chunks2.size() << " chunks (trailing \"?\" has no leading space)\n";
    }

    const std::string synth_path = "/tmp/ch15_4_synthetic_model.gguf";
    constexpr int S_DIM = 32, S_HEADS = 4, S_HEADS_KV = 2, S_HEAD_DIM = 8, S_FF = 64, S_LAYERS = 2, S_VOCAB = 64;

    // =====================================================================
    // TEST 3: build a tiny, real-shaped (2-layer) synthetic model with
    // this file's own GGUFWriter, and confirm embedding-row dequantization
    // matches the floats that were quantized into it, within Q8_0's own
    // known error bound.
    // =====================================================================
    std::cout << "\n-- Test 3: synthetic model, embedding dequantization --\n";
    std::vector<std::vector<float>> synth_embeddings(S_VOCAB, std::vector<float>(S_DIM));
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        for (auto& row : synth_embeddings) for (float& v : row) v = dist(rng);

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

        std::vector<float> emb_flat;
        for (auto& row : synth_embeddings) emb_flat.insert(emb_flat.end(), row.begin(), row.end());
        add_q8("token_embd.weight", {S_DIM, S_VOCAB}, emb_flat);
        add_f32("output_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));

        auto rand_vec = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
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
        w.write_u64(pending.size());
        w.write_u64(7);
        w.write_kv_string("general.architecture", "qwen2");
        w.write_kv_u32("qwen2.block_count", S_LAYERS);
        w.write_kv_u32("qwen2.embedding_length", S_DIM);
        w.write_kv_u32("qwen2.attention.head_count", S_HEADS);
        w.write_kv_u32("qwen2.attention.head_count_kv", S_HEADS_KV);
        w.write_kv_u32("qwen2.feed_forward_length", S_FF);
        w.write_kv_f32("qwen2.rope.freq_base", 10000.0f);   // omitted before -- caused RoPE theta=1/0=inf
        uint64_t off = 0;
        for (auto& p : pending) { w.write_tensor_info(p.name, p.dims, p.type, off); off += p.bytes.size(); }
        w.align(32);
        for (auto& p : pending) w.write_bytes(p.bytes.data(), p.bytes.size());
        CHECK(w.good());
    }
    {
        QwenModel model;
        CHECK(model.load(synth_path));
        CHECK(model.shape.dim == S_DIM && model.shape.n_heads == S_HEADS && model.shape.n_heads_kv == S_HEADS_KV);
        CHECK(model.n_layers() == S_LAYERS);
        auto row5 = model.embedding(5);
        float max_err = 0.0f;
        for (int i = 0; i < S_DIM; ++i) max_err = std::max(max_err, std::fabs(row5[i] - synth_embeddings[5][i]));
        CHECK(max_err < 0.02f);   // Q8_0's own known error bound (Chapter 4.2), not exact equality
        std::cout << "  loaded shape: dim=" << model.shape.dim << " heads=" << model.shape.n_heads
                   << " heads_kv=" << model.shape.n_heads_kv << " layers=" << model.n_layers() << "\n";
        std::cout << "  embedding row 5 max dequantization error: " << max_err << " (Q8_0 tolerance)\n";
    }

    // =====================================================================
    // TEST 4: the complete pipeline on the synthetic model -- embedding
    // lookup -> both layers -> final norm -> tied-projection argmax --
    // deterministic across two fresh runs.
    // =====================================================================
    std::cout << "\n-- Test 4: full synthetic pipeline, embedding to argmax --\n";
    {
        QwenModel model;
        CHECK(model.load(synth_path));
        auto run_once = [&]() {
            std::vector<float> x = model.embedding(3);
            KVCache cache(model.shape.n_heads_kv, 8, model.shape.head_dim);
            for (int layer = 0; layer < model.n_layers(); ++layer) {
                auto w = model.layer(layer);
                qwen2_block_forward(x, model.shape, w, cache, 0, *model.rope);
            }
            auto final_norm = model.tensor("output_norm.weight");
            std::vector<float> normed(model.shape.dim);
            rms_norm(normed, x, final_norm);
            return model.project_argmax(normed, S_VOCAB);
        };
        auto [id1, logit1] = run_once();
        auto [id2, logit2] = run_once();
        CHECK(id1 >= 0 && id1 < S_VOCAB);
        CHECK(id1 == id2);
        CHECK(logit1 == logit2);
        CHECK(std::isfinite(logit1));
        std::cout << "  argmax token id: " << id1 << ", logit=" << logit1 << "\n";
        std::cout << "  re-run bit-identical: " << (id1 == id2 && logit1 == logit2 ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Self-tests: " << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS --\n" : "FAILURES --\n");

    // =====================================================================
    // REAL GENERATION -- the same honest exception as Sections 15.1 and
    // 15.2: executed via a shell on the reader's own machine against the
    // actual downloaded file, not reproducible in the four-way
    // cross-check environment.
    // =====================================================================
    if (argc >= 2) {
        std::cout << "\n=== REAL GENERATION (run via a shell on the reader's own machine"
                     " against the actual downloaded file; not reproduced in the four-way"
                     " cross-check environment) ===\n";
        QwenModel model;
        bool loaded = model.load(argv[1]);
        RCHECK(loaded);
        if (!loaded) { std::cerr << "could not load " << argv[1] << "\n"; return 1; }
        RCHECK(model.shape.dim == 896 && model.shape.n_heads == 14 && model.shape.n_heads_kv == 2);
        RCHECK(model.n_layers() == 24);
        int vocab_size = static_cast<int>(model.r.get_string_array("tokenizer.ggml.tokens").size());
        RCHECK(vocab_size == 151936);
        std::cout << "model loaded: dim=" << model.shape.dim << " heads=" << model.shape.n_heads
                   << " heads_kv=" << model.shape.n_heads_kv << " layers=" << model.n_layers()
                   << " vocab=" << vocab_size << "\n";

        std::cout << "\nbuilding vocabulary and merge table from the real file...\n";
        Vocabulary vocab;
        for (const auto& t : model.r.get_string_array("tokenizer.ggml.tokens")) vocab.add(t);
        MergeTable merges;
        const auto& merge_strings = model.r.get_string_array("tokenizer.ggml.merges");
        for (size_t i = 0; i < merge_strings.size(); ++i) {
            size_t sp = merge_strings[i].find(' ');
            merges.add(merge_strings[i].substr(0, sp), merge_strings[i].substr(sp + 1), static_cast<int>(i));
        }
        RCHECK(vocab.size() == 151936);
        std::cout << "vocabulary: " << vocab.size() << " tokens, " << merges.priority_of.size() << " merge rules\n";

        int im_start = vocab.lookup("<|im_start|>");
        int im_end = vocab.lookup("<|im_end|>");
        RCHECK(im_start >= 0 && im_end >= 0);
        std::cout << "special tokens: <|im_start|>=" << im_start << " <|im_end|>=" << im_end << "\n";

        auto encode_text = [&](const std::string& raw) { return encode_gpt2(raw, codec, merges, vocab); };
        auto encode_chunks = [&](const std::string& raw) {
            std::vector<int> ids;
            for (const auto& chunk : gpt2_pretokenize(raw))
                for (int id : encode_text(chunk)) ids.push_back(id);
            return ids;
        };

        std::string content = "What is the capital of France?";
        // The real file's own tokenizer.chat_template (read directly from
        // the GGUF, not assumed) always injects a system turn -- the
        // caller's own message if one is given, otherwise this exact
        // fallback string -- before the first user turn. A first attempt
        // at this section omitted the system turn entirely, on the
        // assumption that a single user question needed nothing else; the
        // resulting generation was fluent but wrong (a real, documented
        // bug -- see this section's own account of finding it), and a
        // from-source llama.cpp build applying the file's real template
        // confirmed the system turn is exactly what was missing. A 0.5B
        // model, tuned against a fixed conversational shape, apparently
        // cannot be relied on to answer sensibly outside that shape.
        const std::string default_system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        std::vector<int> prompt_ids;
        if (model.add_bos) prompt_ids.push_back(vocab.lookup("<|endoftext|>"));   // not exercised for this file (add_bos_token=false)
        prompt_ids.push_back(im_start);
        for (int id : encode_text("system")) prompt_ids.push_back(id);
        for (int id : encode_text("\n")) prompt_ids.push_back(id);
        for (int id : encode_chunks(default_system)) prompt_ids.push_back(id);
        prompt_ids.push_back(im_end);
        for (int id : encode_text("\n")) prompt_ids.push_back(id);
        prompt_ids.push_back(im_start);
        for (int id : encode_text("user")) prompt_ids.push_back(id);
        for (int id : encode_text("\n")) prompt_ids.push_back(id);
        for (int id : encode_chunks(content)) prompt_ids.push_back(id);
        prompt_ids.push_back(im_end);
        for (int id : encode_text("\n")) prompt_ids.push_back(id);
        prompt_ids.push_back(im_start);
        for (int id : encode_text("assistant")) prompt_ids.push_back(id);
        for (int id : encode_text("\n")) prompt_ids.push_back(id);

        RCHECK(prompt_ids.size() > 0);
        bool all_valid = true;
        for (int id : prompt_ids) if (id < 0) all_valid = false;
        RCHECK(all_valid);
        std::cout << "\nChatML prompt (add_bos_token=" << (model.add_bos ? "true" : "false")
                   << "): " << prompt_ids.size() << " tokens\n";
        std::cout << "  raw content: \"" << content << "\"\n";

        std::cout << "\ndequantizing all " << model.n_layers()
                   << " real layers once (reused across every prompt position, rather than"
                     " re-dequantizing the same weights on every position)...\n";
        std::vector<QwenBlockWeights> all_layers;
        for (int layer = 0; layer < model.n_layers(); ++layer) all_layers.push_back(model.layer(layer));
        auto final_norm = model.tensor("output_norm.weight");
        std::cout << "dequantization done.\n";

        std::cout << "\nrunning the real 24-layer forward pass over " << prompt_ids.size() << " prompt positions...\n";
        // One KVCache PER LAYER, not one shared across all 24 -- a real,
        // and more serious, bug lived here through an earlier draft of
        // this section: a single cache object holds only one layer's
        // worth of key/value, so processing position-major (all 24 layers
        // for position 0, then all 24 for position 1, ...) through ONE
        // shared cache means each layer's slot for an earlier position
        // gets overwritten by the NEXT layer's key/value before a later
        // position ever attends back to it -- every attention lookup for
        // an earlier position silently read whichever layer wrote there
        // last, not the current layer's own key/value. A short 5-token
        // continuation prompt didn't expose it clearly enough to notice;
        // this section's real 36-token ChatML prompt did, once compared
        // token-for-token and layer-for-layer against an independently
        // built llama.cpp on the same file (see this section's own
        // account of finding it). The fix is exactly what real streaming
        // inference already requires: a separate cache per layer,
        // persisting across positions within that layer.
        auto run_generation = [&]() -> std::pair<int, float> {
            std::vector<KVCache> caches;
            for (int layer = 0; layer < model.n_layers(); ++layer)
                caches.emplace_back(model.shape.n_heads_kv, static_cast<int>(prompt_ids.size()) + 1, model.shape.head_dim);
            std::vector<float> x;
            for (size_t pos = 0; pos < prompt_ids.size(); ++pos) {
                x = model.embedding(prompt_ids[pos]);
                for (int layer = 0; layer < model.n_layers(); ++layer)
                    qwen2_block_forward(x, model.shape, all_layers[layer], caches[static_cast<size_t>(layer)], static_cast<int>(pos), *model.rope);
            }
            std::vector<float> normed(model.shape.dim);
            rms_norm(normed, x, final_norm);
            return model.project_argmax(normed, vocab_size);
        };

        auto [best_id, best_logit] = run_generation();
        RCHECK(best_id >= 0 && best_id < vocab_size);
        RCHECK(std::isfinite(best_logit));
        std::string decoded = codec.decode(vocab.text_of(best_id));
        RCHECK(!vocab.text_of(best_id).empty());
        std::cout << "\nfirst generated token: id=" << best_id << " logit=" << best_logit
                   << " vocab_text=\"" << vocab.text_of(best_id) << "\" decoded=\"" << decoded << "\"\n";

        std::cout << "\nre-running the full generation once more to confirm determinism"
                     " (this repeats the entire 24-layer pass)...\n";
        auto [best_id2, best_logit2] = run_generation();
        RCHECK(best_id2 == best_id);
        RCHECK(best_logit2 == best_logit);
        std::cout << "re-run: id=" << best_id2 << " logit=" << best_logit2
                   << " -- bit-identical to first run: " << (best_id2 == best_id && best_logit2 == best_logit ? "yes" : "no") << "\n";

        std::cout << "\n-- Real-generation checks: " << r_passed << "/" << r_tests << " checks passed ";
        std::cout << (r_passed == r_tests ? "ALL PASS --\n" : "FAILURES --\n");
    }

    std::cout << "\n========================================================\n";
    bool all_ok = (g_passed == g_tests) && (argc < 2 || r_passed == r_tests);
    std::cout << (all_ok ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    std::cout << "========================================================\n";
    return all_ok ? 0 : 1;
}
