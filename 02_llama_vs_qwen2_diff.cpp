// Chapter 15.2 -- Section 15.1 built a reader that can survive an actual
// downloaded GGUF file. This section puts that reader to work answering
// a specific question: which of this book's Part 0 assumptions, all
// built and verified against Llama-style architecture choices, does a
// real Qwen2.5 checkpoint actually violate? Five differences, each
// checked against the real file rather than asserted from a spec sheet:
// QKV bias tensors Llama has none of, a grouped-query-attention ratio
// this book's own TOC states incorrectly, byte-identical (tied) input
// and output embeddings, a RoPE base frequency two orders of magnitude
// above this book's own Chapter 3.3 default, and a ChatML turn format
// that -- unlike every Llama-3-style conversation this book has built
// so far -- never prepends a beginning-of-text token at all.
//
// Like Section 15.1, this file runs in two modes. Self-tests (no
// argument) exercise every piece of comparison logic against small
// synthetic fixtures and reproduce identically on every one of this
// book's usual cross-check targets. Real-file mode (a path as argv[1])
// re-opens the actual downloaded model and checks these five claims
// against its real bytes -- executed via a shell on the reader's own
// machine, for the same reason Section 15.1 could not run that part in
// the book's usual build environment: the real ~644 MB file lives only
// there.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_llama_vs_qwen2_diff.cpp -o 02_llama_vs_qwen2_diff
// Run (self-tests only, works anywhere):    ./02_llama_vs_qwen2_diff
// Run (adds real-file verification):        ./02_llama_vs_qwen2_diff /path/to/model.gguf

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

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

// ---------------------------------------------------------------------
// Section 15.1's reader, repeated per this book's one-file-per-section
// convention (the magic-number fix, BOOL support, real ggml type IDs,
// and generic value-skipping all carry over unchanged).
// ---------------------------------------------------------------------
static constexpr uint32_t GGUF_VERSION = 3;
enum GGUFValueType : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12,
};
enum GGMLType : uint32_t {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3,
    GGML_Q5_0 = 6, GGML_Q5_1 = 7, GGML_Q8_0 = 8, GGML_Q8_1 = 9,
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

using MetaValue = std::variant<std::string, uint32_t, float, bool>;

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
    std::string path;

    bool open(const std::string& p) {
        path = p;
        in.open(p, std::ios::binary);
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
                    for (uint64_t j = 0; j < count; ++j) skip_value(elem_type);   // this section needs no arrays
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
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

// ---------------------------------------------------------------------
// Difference 1: QKV bias. A plain, reusable predicate -- does this file
// have bias tensors for block 0's Q/K/V projections at all?
// ---------------------------------------------------------------------
struct QKVBiasReport { bool has_q, has_k, has_v; };
QKVBiasReport check_qkv_bias(const GGUFReader& r) {
    return { r.find_tensor("blk.0.attn_q.bias") != nullptr,
             r.find_tensor("blk.0.attn_k.bias") != nullptr,
             r.find_tensor("blk.0.attn_v.bias") != nullptr };
}

// ---------------------------------------------------------------------
// Difference 3: tied embeddings, byte for byte -- Section 15.1's
// gguf_tie_check.cpp reconnaissance tool, formalized as a reusable
// function with its own synthetic self-test.
// ---------------------------------------------------------------------
uint64_t q8_0_bytes(uint64_t n_elements) { return (n_elements / 32) * 34; }
uint64_t f32_bytes(uint64_t n_elements) { return n_elements * 4; }

bool tensors_byte_identical(const std::string& path, const GGUFReader& r,
                              const TensorInfo& a, const TensorInfo& b, uint64_t byte_len) {
    std::ifstream data(path, std::ios::binary);
    constexpr size_t CHUNK = 1 << 20;
    std::vector<char> buf_a(CHUNK), buf_b(CHUNK);
    uint64_t pos_a = r.data_section_offset + a.offset;
    uint64_t pos_b = r.data_section_offset + b.offset;
    uint64_t remaining = byte_len, checked = 0;
    while (remaining > 0) {
        size_t n = static_cast<size_t>(std::min<uint64_t>(CHUNK, remaining));
        data.seekg(static_cast<std::streamoff>(pos_a + checked));
        data.read(buf_a.data(), static_cast<std::streamsize>(n));
        data.seekg(static_cast<std::streamoff>(pos_b + checked));
        data.read(buf_b.data(), static_cast<std::streamsize>(n));
        if (std::memcmp(buf_a.data(), buf_b.data(), n) != 0) return false;
        checked += n;
        remaining -= n;
    }
    return true;
}

// ---------------------------------------------------------------------
// Difference 4: RoPE base frequency -- Chapter 3.3's RoPETables,
// reused verbatim (its base parameter already existed there; nothing
// about it needed to change to make this comparison).
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
    float angle_at(int pos, int k) const {
        return std::atan2(sin_vals[pos * half_dim + k], cos_vals[pos * half_dim + k]);
    }
};

// ---------------------------------------------------------------------
// Difference 5: ChatML vs Chapter 12.3's Llama-3-style template. Both
// use a trivial identity "encoder" here (role/content -> themselves as
// single pseudo-tokens) since this section's point is purely structural
// -- whether a BOS-equivalent is prepended and how a turn is delimited
// -- not real BPE encoding, which Section 15.3 handles with the real
// 151,936-token vocabulary.
// ---------------------------------------------------------------------
namespace llama_special {
constexpr int32_t BEGIN_OF_TEXT = 128000;
constexpr int32_t EOT_ID = 128009;
}
namespace qwen_special {
constexpr int32_t IM_START = 100000;   // placeholder IDs -- Section 15.3 maps these to the real vocab
constexpr int32_t IM_END = 100001;
}

std::vector<int32_t> llama_template(const std::vector<std::pair<std::string, std::string>>& turns) {
    std::vector<int32_t> ids;
    ids.push_back(llama_special::BEGIN_OF_TEXT);   // Chapter 12.3: BOS unconditionally, once, at position 0
    for (size_t i = 0; i < turns.size(); ++i) ids.push_back(llama_special::EOT_ID);
    return ids;
}
std::vector<int32_t> qwen_chatml_template(const std::vector<std::pair<std::string, std::string>>& turns) {
    std::vector<int32_t> ids;
    // No BOS-equivalent prepended at all -- this is Difference 5 itself,
    // not an omission: the real file's tokenizer.ggml.add_bos_token is
    // false, confirmed directly from its bytes in real-file mode below.
    for (size_t i = 0; i < turns.size(); ++i) {
        ids.push_back(qwen_special::IM_START);
        ids.push_back(qwen_special::IM_END);
    }
    return ids;
}

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "Chapter 15.2: Five Ways a Real Qwen2 Checkpoint Differs From Llama\n";
    std::cout << "========================================================\n";

    // =====================================================================
    // TEST 1: QKV bias detection against two synthetic models -- one with
    // all three bias tensors, one (a stand-in for Llama's own convention)
    // with none.
    // =====================================================================
    std::cout << "\n-- Test 1: QKV bias detection (synthetic) --\n";
    {
        GGUFWriter w("/tmp/ch15_2_with_bias.gguf");
        w.write_magic(); w.write_u32(GGUF_VERSION);
        w.write_u64(3); w.write_u64(0);
        w.write_tensor_info("blk.0.attn_q.bias", {8}, GGML_F32, 0);
        w.write_tensor_info("blk.0.attn_k.bias", {4}, GGML_F32, 32);
        w.write_tensor_info("blk.0.attn_v.bias", {4}, GGML_F32, 48);
        w.align(32);
        std::vector<float> zeros(16, 0.0f);
        w.write_bytes(zeros.data(), zeros.size() * 4);
        CHECK(w.good());
        }
        {
        GGUFWriter w2("/tmp/ch15_2_no_bias.gguf");
        w2.write_magic(); w2.write_u32(GGUF_VERSION);
        w2.write_u64(1); w2.write_u64(0);
        w2.write_tensor_info("blk.0.attn_q.weight", {8, 8}, GGML_F32, 0);
        w2.align(32);
        std::vector<float> zeros2(64, 0.0f);
        w2.write_bytes(zeros2.data(), zeros2.size() * 4);
        CHECK(w2.good());
        }
        {
        GGUFReader with_bias, no_bias;
        CHECK(with_bias.open("/tmp/ch15_2_with_bias.gguf"));
        CHECK(no_bias.open("/tmp/ch15_2_no_bias.gguf"));
        auto r1 = check_qkv_bias(with_bias);
        auto r2 = check_qkv_bias(no_bias);
        CHECK(r1.has_q && r1.has_k && r1.has_v);
        CHECK(!r2.has_q && !r2.has_k && !r2.has_v);
        std::cout << "  model with bias tensors: q=" << r1.has_q << " k=" << r1.has_k << " v=" << r1.has_v << "\n";
        std::cout << "  model with no bias tensors (Llama's convention): q=" << r2.has_q
                   << " k=" << r2.has_k << " v=" << r2.has_v << "\n";
    }

    // =====================================================================
    // TEST 2: tied-embedding detection against a genuinely-tied pair and
    // a genuinely-different pair, same shape, same byte length.
    // =====================================================================
    std::cout << "\n-- Test 2: tied-embedding byte comparison (synthetic) --\n";
    {
        constexpr uint32_t VOCAB = 64, DIM = 32;
        std::vector<float> shared(VOCAB * DIM);
        for (size_t i = 0; i < shared.size(); ++i) shared[i] = static_cast<float>(i) * 0.001f;
        std::vector<float> different = shared;
        different[different.size() / 2] += 1.0f;   // one differing element is enough to break tying

        const std::string tied_path = "/tmp/ch15_2_tied.gguf";
        {
            GGUFWriter w(tied_path);
            w.write_magic(); w.write_u32(GGUF_VERSION);
            w.write_u64(2); w.write_u64(0);
            uint64_t bytes = static_cast<uint64_t>(shared.size()) * 4;
            w.write_tensor_info("output.weight", {DIM, VOCAB}, GGML_F32, 0);
            w.write_tensor_info("token_embd.weight", {DIM, VOCAB}, GGML_F32, bytes);
            w.align(32);
            w.write_bytes(shared.data(), bytes);
            w.write_bytes(shared.data(), bytes);   // byte-identical copy: genuinely tied
            CHECK(w.good());
        }
        const std::string untied_path = "/tmp/ch15_2_untied.gguf";
        {
            GGUFWriter w(untied_path);
            w.write_magic(); w.write_u32(GGUF_VERSION);
            w.write_u64(2); w.write_u64(0);
            uint64_t bytes = static_cast<uint64_t>(shared.size()) * 4;
            w.write_tensor_info("output.weight", {DIM, VOCAB}, GGML_F32, 0);
            w.write_tensor_info("token_embd.weight", {DIM, VOCAB}, GGML_F32, bytes);
            w.align(32);
            w.write_bytes(shared.data(), bytes);
            w.write_bytes(different.data(), bytes);
            CHECK(w.good());
        }

        GGUFReader tied, untied;
        CHECK(tied.open(tied_path));
        CHECK(untied.open(untied_path));
        auto* t_out = tied.find_tensor("output.weight");
        auto* t_emb = tied.find_tensor("token_embd.weight");
        auto* u_out = untied.find_tensor("output.weight");
        auto* u_emb = untied.find_tensor("token_embd.weight");
        uint64_t byte_len = f32_bytes(t_out->n_elements);
        bool is_tied = tensors_byte_identical(tied_path, tied, *t_out, *t_emb, byte_len);
        bool is_untied = tensors_byte_identical(untied_path, untied, *u_out, *u_emb, byte_len);
        CHECK(is_tied);
        CHECK(!is_untied);
        std::cout << "  identical-bytes pair detected as tied: " << (is_tied ? "yes" : "no") << "\n";
        std::cout << "  one-element-different pair detected as tied: " << (is_untied ? "yes (WRONG)" : "no (correct)") << "\n";
    }

    // =====================================================================
    // TEST 3: RoPE base frequency -- a higher base produces a SMALLER
    // rotation angle at every (position, dimension-pair) beyond k=0, a
    // general mathematical property of theta_k = base^(-2k/head_dim)
    // checkable without needing the real model's specific numbers yet.
    // =====================================================================
    std::cout << "\n-- Test 3: RoPE base frequency changes rotation speed (synthetic) --\n";
    {
        constexpr int HEAD_DIM = 8, SEQ = 32;
        RoPETables low_base(SEQ, HEAD_DIM, 10000.0f);     // this book's own Chapter 3.3 default
        RoPETables high_base(SEQ, HEAD_DIM, 1000000.0f);  // the real Qwen2.5 file's actual value

        // k=0 is unaffected (theta_0 = base^0 = 1 regardless of base) --
        // the difference only shows up for k>0, and grows with k.
        CHECK(std::fabs(low_base.angle_at(10, 0) - high_base.angle_at(10, 0)) < 1e-4f);
        float low_angle_k3 = std::fabs(low_base.angle_at(10, 3));
        float high_angle_k3 = std::fabs(high_base.angle_at(10, 3));
        CHECK(high_angle_k3 < low_angle_k3);
        std::cout << "  at k=0: both bases rotate identically (angle depends only on position there)\n";
        std::cout << "  at k=3, pos=10: base=10000 angle=" << low_angle_k3
                   << ", base=1000000 angle=" << high_angle_k3 << " (higher base rotates slower)\n";
        CHECK(low_angle_k3 > 0.0f);
    }

    // =====================================================================
    // TEST 4: ChatML structurally never prepends a BOS-equivalent, while
    // this book's own Chapter 12.3 Llama-3-style template always does.
    // =====================================================================
    std::cout << "\n-- Test 4: chat template structure (synthetic) --\n";
    {
        std::vector<std::pair<std::string, std::string>> turns = {{"system", "x"}, {"user", "y"}};
        auto llama_ids = llama_template(turns);
        auto qwen_ids = qwen_chatml_template(turns);
        CHECK(llama_ids.front() == llama_special::BEGIN_OF_TEXT);
        CHECK(qwen_ids.front() != llama_special::BEGIN_OF_TEXT);
        bool qwen_has_bos_anywhere = false;
        for (int32_t id : qwen_ids) if (id == llama_special::BEGIN_OF_TEXT) qwen_has_bos_anywhere = true;
        CHECK(!qwen_has_bos_anywhere);
        std::cout << "  Llama-3-style template (Chapter 12.3): starts with BEGIN_OF_TEXT, always\n";
        std::cout << "  ChatML template (this section): no BOS-equivalent anywhere in " << qwen_ids.size() << " ids\n";
    }

    std::cout << "\n-- Self-tests: " << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS --\n" : "FAILURES --\n");

    // =====================================================================
    // REAL FILE VERIFICATION -- same honest exception as Section 15.1:
    // executed via a shell on the reader's own machine against the real
    // downloaded file, not reproducible in the four-way cross-check
    // environment because that file was never transferred there.
    // =====================================================================
    if (argc >= 2) {
        std::cout << "\n=== REAL FILE VERIFICATION (run via a shell on the reader's own machine"
                     " against the actual downloaded file; not reproduced in the four-way"
                     " cross-check environment) ===\n";
        GGUFReader r;
        bool opened = r.open(argv[1]);
        RCHECK(opened);
        if (!opened) { std::cerr << "could not open " << argv[1] << "\n"; return 1; }

        std::cout << "\n[Difference 1: QKV bias]\n";
        auto bias = check_qkv_bias(r);
        RCHECK(bias.has_q && bias.has_k && bias.has_v);
        auto* qb = r.find_tensor("blk.0.attn_q.bias");
        auto* kb = r.find_tensor("blk.0.attn_k.bias");
        auto* vb = r.find_tensor("blk.0.attn_v.bias");
        std::cout << "  blk.0.attn_q.bias present, shape=[" << qb->dims[0] << "]\n";
        std::cout << "  blk.0.attn_k.bias present, shape=[" << kb->dims[0] << "]\n";
        std::cout << "  blk.0.attn_v.bias present, shape=[" << vb->dims[0] << "]\n";
        std::cout << "  Llama's attention convention (this book's Chapter 3.4 gqa_attention) has no such tensors at all\n";

        std::cout << "\n[Difference 2: grouped-query attention ratio]\n";
        uint32_t n_heads = r.get_u32("qwen2.attention.head_count");
        uint32_t n_heads_kv = r.get_u32("qwen2.attention.head_count_kv");
        RCHECK(n_heads == 14);
        RCHECK(n_heads_kv == 2);
        RCHECK(n_heads / n_heads_kv == 7);
        std::cout << "  head_count=" << n_heads << " head_count_kv=" << n_heads_kv
                   << " -> real ratio " << n_heads << ":" << n_heads_kv << " = " << (n_heads / n_heads_kv) << ":1\n";
        std::cout << "  this book's own TOC (Chapter 15's own description) states \"6:1\" -- the real file"
                     " says 7:1, and this section reports what the bytes actually contain rather than"
                     " carrying the stated figure forward unverified\n";

        std::cout << "\n[Difference 3: tied embeddings]\n";
        auto* out_w = r.find_tensor("output.weight");
        auto* emb_w = r.find_tensor("token_embd.weight");
        RCHECK(out_w != nullptr && emb_w != nullptr);
        uint64_t byte_len = q8_0_bytes(out_w->n_elements);
        RCHECK(byte_len == q8_0_bytes(emb_w->n_elements));
        bool tied = tensors_byte_identical(argv[1], r, *out_w, *emb_w, byte_len);
        RCHECK(tied);
        std::cout << "  output.weight and token_embd.weight: " << byte_len << " bytes each, compared byte for byte\n";
        std::cout << "  result: " << (tied ? "BYTE-IDENTICAL -- genuinely tied" : "DIFFER -- not tied") << "\n";

        std::cout << "\n[Difference 4: RoPE base frequency]\n";
        float real_base = r.get_f32("qwen2.rope.freq_base");
        RCHECK(real_base == 1000000.0f);
        constexpr int HEAD_DIM = 64;   // derived in Section 15.1: embedding_length / head_count = 896/14
        RoPETables book_default(64, HEAD_DIM, 10000.0f);
        RoPETables real_rope(64, HEAD_DIM, real_base);
        float angle_book = std::fabs(book_default.angle_at(50, 20));
        float angle_real = std::fabs(real_rope.angle_at(50, 20));
        RCHECK(angle_real < angle_book);
        std::cout << "  real qwen2.rope.freq_base=" << real_base
                   << " vs this book's Chapter 3.3 default base=10000\n";
        std::cout << "  at position=50, dimension-pair k=20: this book's default base gives angle="
                   << angle_book << " rad, the real base gives angle=" << angle_real
                   << " rad (" << (angle_book / (angle_real > 0.0f ? angle_real : 1.0f))
                   << "x slower at this dimension pair)\n";

        std::cout << "\n[Difference 5: ChatML, no BOS prepend]\n";
        bool add_bos = r.get_bool("tokenizer.ggml.add_bos_token");
        std::string chat_template = r.get_string("tokenizer.chat_template");
        RCHECK(add_bos == false);
        RCHECK(!chat_template.empty());
        bool has_im_start = chat_template.find("<|im_start|>") != std::string::npos;
        bool has_im_end = chat_template.find("<|im_end|>") != std::string::npos;
        RCHECK(has_im_start && has_im_end);
        std::cout << "  tokenizer.ggml.add_bos_token=" << (add_bos ? "true" : "false")
                   << " -- unlike Chapter 12.3's Llama-3-style template, no token is ever prepended\n";
        std::cout << "  tokenizer.chat_template: " << chat_template.size()
                   << " bytes, contains \"<|im_start|>\": " << (has_im_start ? "yes" : "no")
                   << ", contains \"<|im_end|>\": " << (has_im_end ? "yes" : "no") << "\n";

        std::cout << "\n-- Real-file checks: " << r_passed << "/" << r_tests << " checks passed ";
        std::cout << (r_passed == r_tests ? "ALL PASS --\n" : "FAILURES --\n");
    }

    std::cout << "\n========================================================\n";
    bool all_ok = (g_passed == g_tests) && (argc < 2 || r_passed == r_tests);
    std::cout << (all_ok ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    std::cout << "========================================================\n";
    return all_ok ? 0 : 1;
}
