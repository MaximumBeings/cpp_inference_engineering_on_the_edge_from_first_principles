// Chapter 15.1 -- every earlier GGUF reader in this book (Chapter 5.2's
// writer/reader, Chapter 12.5's array-metadata extension) was built and
// verified exclusively against files this book's OWN writer produced --
// files whose exact shape was already known before the reader was asked
// to parse them. A real, published GGUF file makes no such promise: its
// metadata can use a value type this book's readers never needed
// (BOOL(7), the real Qwen2.5 checkpoint's tokenizer.ggml.add_bos_token
// field), and its tensor descriptors use the REAL ggml type-ID numbering
// rather than the simplified placeholder values Chapter 5.2 chose for
// its own synthetic worked example. This section builds a reader that
// closes both gaps, then points it at an actual downloaded model file.
//
// This file cannot be verified the same way every earlier chapter file
// was. The real ~644 MB Qwen2.5-0.5B-Instruct GGUF file this section
// inspects lives only on the reader's own machine -- it was never
// transferred into (and is not committed alongside) this book's own
// build environment, both because organizational network policy blocks
// downloading it there directly and because committing a 644 MB binary
// to source control defeats the purpose of a text-based book. So this
// file runs in two genuinely different modes: with no argument, it
// builds a small synthetic fixture with this section's own GGUFWriter
// (mirroring the real file's own proportions: non-trivial GQA ratio,
// multiple metadata value types, an unknown value type the reader must
// skip rather than choke on) and verifies the reader against it -- this
// half is what compiles and reruns identically on every one of this
// book's usual cross-check targets. With a real file path as argv[1],
// it ALSO opens that real file and prints and checks the genuine
// architecture facts read directly from its bytes -- this half's locked
// output was captured by running this exact binary via a shell on the
// reader's own machine against the real, downloaded file, and is
// reproduced here as an honest exception to this book's usual
// "verified identically on four targets" rule, clearly marked below.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_real_gguf_inspection.cpp -o 01_real_gguf_inspection
// Run (self-tests only, works anywhere):    ./01_real_gguf_inspection
// Run (adds real-file inspection):          ./01_real_gguf_inspection /path/to/model.gguf

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
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

// Chapters 5.2 and 12.5 both defined a magic-number CONSTANT,
// GGUF_MAGIC = 0x46475547, and compared the file's first four bytes
// against it as a little-endian uint32_t. That constant is wrong: its
// four little-endian bytes decode to the ASCII sequence 'G','U','G','F',
// not the real spec's 'G','G','U','F'. Neither chapter's own round-trip
// test could ever catch this, because each one only ever read back a
// file its OWN writer had just produced with the identical (wrong)
// constant -- writer and reader agreed with each other while both
// disagreed with the real GGUF spec, and nothing in either chapter's
// test suite could tell the difference. Pointing a reader at a REAL
// file (this section's whole point) surfaces it immediately: the real
// file's first four bytes, read the same way, do not equal 0x46475547.
// The fix used here is the same one this chapter's own reconnaissance
// tools (gguf_dump.cpp, gguf_tie_check.cpp) already used and this
// section's writer/reader now match: compare the four raw bytes
// directly against the literal string "GGUF" with memcmp, which
// sidesteps the little-endian encoding question entirely rather than
// getting it wrong a second time.
static constexpr uint32_t GGUF_VERSION = 3;

// ---------------------------------------------------------------------
// KV metadata VALUE types -- the tag that precedes every metadata
// value's bytes. This is its own independent numbering, unrelated to
// the tensor ELEMENT types below even though both happen to start at 0.
// Chapter 12.5 only ever needed UINT32(4), INT32(5), FLOAT32(6),
// STRING(8), and ARRAY(9); the real file adds exactly one this book has
// not parsed before -- BOOL(7), used by tokenizer.ggml.add_bos_token.
// ---------------------------------------------------------------------
enum GGUFValueType : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12,
};

// ---------------------------------------------------------------------
// Tensor ELEMENT types -- ggml's own type IDs, a SEPARATE numbering from
// GGUFValueType above. This is the real spec's numbering, not Chapter
// 5.2's: that section assigned its synthetic worked example's Q4_0 and
// Q8_0 the placeholder values 2 and 7 (chosen only to keep two example
// tensors distinguishable from F32=0), and never claimed those values
// matched any real file, because Chapter 5.2 never opened one. The real
// downloaded Qwen2.5 file's own tensor descriptors settle the question:
// Q8_0 is 8, not 7 -- confirmed directly below, not assumed from a spec
// sheet, because every quantized tensor in the real file reads back with
// type=8 and every norm/bias tensor reads back with type=0 (F32).
// ---------------------------------------------------------------------
enum GGMLType : uint32_t {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3,
    GGML_Q5_0 = 6, GGML_Q5_1 = 7, GGML_Q8_0 = 8, GGML_Q8_1 = 9,
    // 10 and above are the K-quant family (Q2_K .. Q8_K, IQ*) -- this
    // book's own quantizer only ever produces F32 or Q8_0 tensors, so
    // compute_data_size() below reports 0 (unknown) for anything else
    // rather than guessing a byte layout it cannot verify.
};

const char* ggml_type_name(uint32_t t) {
    switch (t) {
        case GGML_F32: return "F32"; case GGML_F16: return "F16";
        case GGML_Q4_0: return "Q4_0"; case GGML_Q4_1: return "Q4_1";
        case GGML_Q5_0: return "Q5_0"; case GGML_Q5_1: return "Q5_1";
        case GGML_Q8_0: return "Q8_0"; case GGML_Q8_1: return "Q8_1";
        default: return "K-QUANT-OR-UNKNOWN";
    }
}
const char* value_type_name(uint32_t t) {
    switch (t) {
        case V_UINT8: return "UINT8"; case V_INT8: return "INT8";
        case V_UINT16: return "UINT16"; case V_INT16: return "INT16";
        case V_UINT32: return "UINT32"; case V_INT32: return "INT32";
        case V_FLOAT32: return "FLOAT32"; case V_BOOL: return "BOOL";
        case V_STRING: return "STRING"; case V_ARRAY: return "ARRAY";
        case V_UINT64: return "UINT64"; case V_INT64: return "INT64";
        case V_FLOAT64: return "FLOAT64";
        default: return "UNKNOWN";
    }
}
size_t scalar_byte_size(uint32_t t) {
    switch (t) {
        case V_UINT8: case V_INT8: case V_BOOL: return 1;
        case V_UINT16: case V_INT16: return 2;
        case V_UINT32: case V_INT32: case V_FLOAT32: return 4;
        case V_UINT64: case V_INT64: case V_FLOAT64: return 8;
        default: return 0;
    }
}

// -- Writer: Chapter 5.2's write_u32/write_u64/write_string/tensor-info
//    writer, plus Chapter 12.5's array writers, plus a new write_kv_bool. --
class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* data, size_t size) { out.write(reinterpret_cast<const char*>(data), size); pos += size; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_magic() { write_raw("GGUF", 4); }   // the four literal bytes, not a numeric constant
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_i32(int32_t v) { write_raw(&v, 4); }
    void write_f32(float v) { write_raw(&v, 4); }
    void write_string(const std::string& s) { uint64_t len = s.size(); write_raw(&len, 8); write_raw(s.data(), s.size()); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(V_STRING); write_string(v); }
    void write_kv_u32(const std::string& k, uint32_t v) { write_string(k); write_u32(V_UINT32); write_u32(v); }
    void write_kv_f32(const std::string& k, float v) { write_string(k); write_u32(V_FLOAT32); write_f32(v); }
    void write_kv_bool(const std::string& k, bool v) { write_string(k); write_u32(V_BOOL); write_raw(&v, 1); }
    // A value type this reader deliberately does NOT understand, written
    // on purpose so Test 4 below can prove the reader skips it cleanly
    // instead of aborting the whole parse the way Chapter 12.5's did.
    void write_kv_unknown_int16(const std::string& k, int16_t v) { write_string(k); write_u32(V_INT16); write_raw(&v, 2); }

    void write_kv_array_string(const std::string& k, const std::vector<std::string>& values) {
        write_string(k); write_u32(V_ARRAY); write_u32(V_STRING); write_u64(values.size());
        for (const auto& v : values) write_string(v);
    }
    void write_kv_array_i32(const std::string& k, const std::vector<int32_t>& values) {
        write_string(k); write_u32(V_ARRAY); write_u32(V_INT32); write_u64(values.size());
        for (int32_t v : values) write_i32(v);
    }
    void write_kv_array_f32(const std::string& k, const std::vector<float>& values) {
        write_string(k); write_u32(V_ARRAY); write_u32(V_FLOAT32); write_u64(values.size());
        for (float v : values) write_f32(v);
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
    uint64_t data_size = 0;

    uint64_t compute_data_size() const {
        switch (type) {
            case GGML_F32: return n_elements * 4;
            case GGML_Q8_0: return (n_elements / 32) * 34;   // fp16 scale + 32 int8 codes, Chapter 4.2
            default: return 0;   // this book's own quantizer never produces anything else
        }
    }
};

// -- Reader: a union of Chapter 5.2's tensor-descriptor parsing and
//    Chapter 12.5's array-metadata parsing, plus BOOL support and a
//    generic skip path for any value type neither of those two sections
//    anticipated -- the specific gap a real, externally-produced file
//    can expose that a file this book wrote for itself never will. --
using MetaValue = std::variant<std::string, uint32_t, float, bool,
                                std::vector<std::string>, std::vector<int32_t>, std::vector<float>>;

class GGUFReader {
    std::ifstream in;
    void read_raw(void* data, size_t size) { in.read(reinterpret_cast<char*>(data), size); }

    // Consumes (without storing) one value of the given type -- the
    // generic fallback for a value type this reader does not have a
    // MetaValue case for. Recurses into ARRAY so an unsupported element
    // type inside an array is skipped correctly too.
    void skip_value(uint32_t type) {
        if (type == V_STRING) { read_string(); return; }
        if (type == V_ARRAY) {
            uint32_t elem_type; read_raw(&elem_type, 4);
            uint64_t count; read_raw(&count, 8);
            for (uint64_t i = 0; i < count; ++i) skip_value(elem_type);
            return;
        }
        size_t sz = scalar_byte_size(type);
        in.seekg(static_cast<std::streamoff>(sz), std::ios::cur);
    }

public:
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    std::unordered_map<std::string, MetaValue> metadata;
    std::vector<TensorInfo> tensors;
    uint64_t data_section_offset = 0;
    int unknown_keys_skipped = 0;

    bool open(const std::string& path) {
        in.open(path, std::ios::binary);
        if (!in.is_open()) return false;

        char magic_bytes[4];
        read_raw(magic_bytes, 4);
        if (std::memcmp(magic_bytes, "GGUF", 4) != 0) return false;
        std::memcpy(&magic, magic_bytes, 4);   // for display only, not compared numerically
        read_raw(&version, 4);
        read_raw(&n_tensors, 8);
        read_raw(&n_kv, 8);

        uint64_t alignment = 32;   // GGUF's documented default
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
                    } else if (elem_type == V_INT32) {
                        std::vector<int32_t> arr(count);
                        for (uint64_t j = 0; j < count; ++j) read_raw(&arr[j], 4);
                        metadata[key] = std::move(arr);
                    } else if (elem_type == V_FLOAT32) {
                        std::vector<float> arr(count);
                        for (uint64_t j = 0; j < count; ++j) read_raw(&arr[j], 4);
                        metadata[key] = std::move(arr);
                    } else {
                        // An array element type this reader does not
                        // model as a MetaValue: skip every element and
                        // keep going, rather than Chapter 12.5's
                        // "return false" -- one metadata key this reader
                        // cannot represent must not cost the reader
                        // every OTHER key that comes after it.
                        for (uint64_t j = 0; j < count; ++j) skip_value(elem_type);
                        ++unknown_keys_skipped;
                    }
                    break;
                }
                default:
                    skip_value(type);
                    ++unknown_keys_skipped;
                    break;
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
            t.data_size = t.compute_data_size();
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
    bool has(const std::string& key) const { return metadata.count(key) != 0; }
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
    const std::vector<int32_t>& get_i32_array(const std::string& key) const {
        return std::get<std::vector<int32_t>>(metadata.at(key));
    }
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "Chapter 15.1: A Reader That Survives Contact With a Real File\n";
    std::cout << "========================================================\n";

    const std::string path = "/tmp/ch15_synthetic_model.gguf";

    // =====================================================================
    // TEST 1: write a synthetic fixture mirroring the real file's own
    // proportions in miniature -- a non-1 GQA ratio (4 query heads : 1 KV
    // head, echoing the real file's own 14:2), QKV bias tensors present
    // (echoing a real Qwen2 architecture detail Section 15.2 covers), a
    // BOOL metadata value, and one deliberately-unsupported value type.
    // =====================================================================
    std::cout << "\n-- Test 1: writing a synthetic fixture --\n";
    constexpr uint32_t DIM = 16, N_HEADS = 4, N_HEADS_KV = 1;
    {
        GGUFWriter w(path);
        w.write_magic();
        w.write_u32(GGUF_VERSION);
        w.write_u64(2);    // n_tensors
        w.write_u64(9);    // n_kv
        w.write_kv_string("general.architecture", "qwen2");
        w.write_kv_u32("qwen2.embedding_length", DIM);
        w.write_kv_u32("qwen2.attention.head_count", N_HEADS);
        w.write_kv_u32("qwen2.attention.head_count_kv", N_HEADS_KV);
        w.write_kv_f32("qwen2.rope.freq_base", 1000000.0f);
        w.write_kv_bool("tokenizer.ggml.add_bos_token", false);
        w.write_kv_unknown_int16("some.future.field", 42);   // must not break the parse
        w.write_kv_array_string("tokenizer.ggml.tokens", {"a", "b", "c"});
        w.write_kv_array_i32("tokenizer.ggml.token_type", {1, 1, 3});

        std::vector<float> norm_data(DIM, 0.0f);
        for (uint32_t i = 0; i < DIM; ++i) norm_data[i] = static_cast<float>(i) * 0.01f;
        std::vector<uint8_t> q8_data((DIM * DIM / 32) * 34, 0);   // zeroed Q8_0 blocks -- shape only matters here

        uint64_t off_norm = 0, off_q = norm_data.size() * 4;
        w.write_tensor_info("attn_norm.weight", {DIM}, GGML_F32, off_norm);
        w.write_tensor_info("attn_q.weight", {DIM, DIM}, GGML_Q8_0, off_q);
        w.align(32);
        w.write_bytes(norm_data.data(), norm_data.size() * 4);
        w.write_bytes(q8_data.data(), q8_data.size());
        std::cout << "  wrote " << w.tell() << " bytes, GQA ratio " << N_HEADS << ":" << N_HEADS_KV << "\n";
        CHECK(w.good());
    }

    std::cout << "\n-- Test 2: reopening and parsing the header --\n";
    GGUFReader reader;
    {
        CHECK(reader.open(path));
        CHECK(reader.magic == 0x46554747u);  // correct LE encoding of the ASCII bytes 'G','G','U','F'
        CHECK(reader.version == GGUF_VERSION);
        CHECK(reader.n_tensors == 2);
        CHECK(reader.n_kv == 9);
        std::cout << "  magic ok, n_tensors=" << reader.n_tensors << " n_kv=" << reader.n_kv << "\n";
    }

    std::cout << "\n-- Test 3: scalar and BOOL metadata --\n";
    {
        CHECK(reader.get_string("general.architecture") == "qwen2");
        CHECK(reader.get_u32("qwen2.embedding_length") == DIM);
        CHECK(reader.get_u32("qwen2.attention.head_count") == N_HEADS);
        CHECK(reader.get_u32("qwen2.attention.head_count_kv") == N_HEADS_KV);
        CHECK(reader.get_f32("qwen2.rope.freq_base") == 1000000.0f);
        CHECK(reader.get_bool("tokenizer.ggml.add_bos_token") == false);
        std::cout << "  add_bos_token (BOOL) read back as: "
                  << (reader.get_bool("tokenizer.ggml.add_bos_token") ? "true" : "false") << "\n";
        std::cout << "  GQA ratio from file: " << reader.get_u32("qwen2.attention.head_count") << ":"
                  << reader.get_u32("qwen2.attention.head_count_kv") << "\n";
    }

    std::cout << "\n-- Test 4: an unsupported value type is skipped, not fatal --\n";
    {
        // "some.future.field" (INT16) has no MetaValue case; the reader
        // must skip exactly its 2 bytes and keep parsing everything that
        // follows it correctly -- proven by the two array keys AFTER it
        // in file order still reading back exactly right.
        CHECK(reader.unknown_keys_skipped == 1);
        CHECK(!reader.has("some.future.field"));
        const auto& toks = reader.get_string_array("tokenizer.ggml.tokens");
        const auto& types = reader.get_i32_array("tokenizer.ggml.token_type");
        CHECK(toks.size() == 3 && toks[0] == "a" && toks[2] == "c");
        CHECK(types.size() == 3 && types[2] == 3);
        std::cout << "  unknown_keys_skipped=" << reader.unknown_keys_skipped
                  << ", subsequent keys (tokens, token_type) parsed correctly anyway\n";
    }

    std::cout << "\n-- Test 5: tensor descriptors, real ggml type IDs --\n";
    {
        auto* norm = reader.find_tensor("attn_norm.weight");
        auto* q = reader.find_tensor("attn_q.weight");
        CHECK(norm != nullptr && q != nullptr);
        CHECK(norm->type == GGML_F32);
        CHECK(q->type == GGML_Q8_0);
        CHECK(std::string(ggml_type_name(norm->type)) == "F32");
        CHECK(std::string(ggml_type_name(q->type)) == "Q8_0");
        CHECK(norm->n_elements == DIM);
        CHECK(q->n_elements == static_cast<uint64_t>(DIM) * DIM);
        CHECK(q->data_size == (q->n_elements / 32) * 34);
        CHECK(reader.data_section_offset % 32 == 0);
        std::cout << "  attn_norm.weight: type=" << norm->type << " (" << ggml_type_name(norm->type) << ")\n";
        std::cout << "  attn_q.weight:    type=" << q->type << " (" << ggml_type_name(q->type)
                  << ") data_size=" << q->data_size << " bytes\n";
        std::cout << "  data section offset: " << reader.data_section_offset << " (32-byte aligned: yes)\n";
    }

    std::cout << "\n-- Self-tests: " << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS --\n" : "FAILURES --\n");

    // =====================================================================
    // REAL FILE INSPECTION -- executed via a shell on the reader's own
    // aarch64 machine against the actual downloaded
    // qwen2.5-0.5b-instruct-q8_0.gguf, not reproducible in this book's
    // usual four-way cross-check environment because the ~644 MB file
    // itself was never transferred there. Everything printed past this
    // marker is genuine data read directly from that real file's bytes.
    // =====================================================================
    if (argc >= 2) {
        std::cout << "\n=== REAL FILE INSPECTION (run via a shell on the reader's own machine"
                     " against the actual downloaded file; not reproduced in the four-way"
                     " cross-check environment) ===\n";
        GGUFReader real;
        std::string path2 = argv[1];
        bool opened = real.open(path2);
        RCHECK(opened);
        if (!opened) { std::cerr << "could not open " << path2 << "\n"; return 1; }

        RCHECK(real.magic == 0x46554747u);
        std::cout << "file: " << path2 << "\n";
        std::cout << "magic=GGUF version=" << real.version
                   << " n_tensors=" << real.n_tensors << " n_kv=" << real.n_kv << "\n";

        std::string arch = real.get_string("general.architecture");
        std::string name = real.get_string("general.name");
        RCHECK(arch == "qwen2");
        std::cout << "general.architecture=\"" << arch << "\" general.name=\"" << name << "\"\n";

        uint32_t n_layers = real.get_u32("qwen2.block_count");
        uint32_t ctx_len = real.get_u32("qwen2.context_length");
        uint32_t emb = real.get_u32("qwen2.embedding_length");
        uint32_t ffn = real.get_u32("qwen2.feed_forward_length");
        uint32_t n_heads = real.get_u32("qwen2.attention.head_count");
        uint32_t n_heads_kv = real.get_u32("qwen2.attention.head_count_kv");
        float rope_base = real.get_f32("qwen2.rope.freq_base");
        float rms_eps = real.get_f32("qwen2.attention.layer_norm_rms_epsilon");
        uint32_t file_type = real.get_u32("general.file_type");

        RCHECK(n_layers == 24);
        RCHECK(emb == 896);
        RCHECK(ffn == 4864);
        RCHECK(n_heads == 14);
        RCHECK(n_heads_kv == 2);
        RCHECK(file_type == 7);   // llama.cpp's llm_ftype enum: MOSTLY_Q8_0 = 7 (a DIFFERENT
                                  // numbering from ggml_type's own GGML_TYPE_Q8_0 = 8 below --
                                  // see the COMMON TRAP after this listing)

        std::cout << "qwen2.block_count=" << n_layers << " context_length=" << ctx_len << "\n";
        std::cout << "qwen2.embedding_length=" << emb << " feed_forward_length=" << ffn << "\n";
        std::cout << "attention.head_count=" << n_heads << " head_count_kv=" << n_heads_kv
                   << " -> GQA ratio " << n_heads << ":" << n_heads_kv << "\n";
        uint32_t head_dim = emb / n_heads;
        std::cout << "derived head_dim = embedding_length / head_count = " << emb << "/" << n_heads
                   << " = " << head_dim << "\n";
        std::cout << "rope.freq_base=" << std::fixed << std::setprecision(1) << rope_base << "\n";
        std::cout << "attention.layer_norm_rms_epsilon=" << std::scientific << rms_eps << std::fixed << "\n";
        std::cout << "general.file_type=" << file_type << " (llama.cpp llm_ftype: MOSTLY_Q8_0)\n";

        std::string tok_model = real.get_string("tokenizer.ggml.model");
        std::string tok_pre = real.get_string("tokenizer.ggml.pre");
        RCHECK(tok_model == "gpt2");
        const auto& vocab = real.get_string_array("tokenizer.ggml.tokens");
        const auto& merges = real.get_string_array("tokenizer.ggml.merges");
        RCHECK(vocab.size() == 151936);
        RCHECK(merges.size() == 151387);
        uint32_t bos_id = real.get_u32("tokenizer.ggml.bos_token_id");
        uint32_t eos_id = real.get_u32("tokenizer.ggml.eos_token_id");
        bool add_bos = real.get_bool("tokenizer.ggml.add_bos_token");
        RCHECK(eos_id == 151645);
        RCHECK(add_bos == false);

        std::cout << "tokenizer.ggml.model=\"" << tok_model << "\" pre=\"" << tok_pre << "\"\n";
        std::cout << "vocab size=" << vocab.size() << " merges=" << merges.size() << "\n";
        std::cout << "bos_token_id=" << bos_id << " eos_token_id=" << eos_id
                   << " add_bos_token=" << (add_bos ? "true" : "false") << "\n";

        int q8_tensors = 0, f32_tensors = 0, other_tensors = 0;
        for (const auto& t : real.tensors) {
            if (t.type == GGML_Q8_0) ++q8_tensors;
            else if (t.type == GGML_F32) ++f32_tensors;
            else ++other_tensors;
        }
        RCHECK(real.n_tensors == 291);
        RCHECK(static_cast<uint64_t>(q8_tensors + f32_tensors + other_tensors) == real.n_tensors);
        RCHECK(other_tensors == 0);
        std::cout << "tensors: " << real.n_tensors << " total (" << q8_tensors << " Q8_0 [type="
                   << GGML_Q8_0 << "], " << f32_tensors << " F32 [type=" << GGML_F32
                   << "], " << other_tensors << " other)\n";
        std::cout << "derivation check: 24 blocks x 12 tensors/block + 3 top-level = "
                   << (24 * 12 + 3) << " (matches n_tensors: " << (24 * 12 + 3 == static_cast<int>(real.n_tensors) ? "yes" : "no") << ")\n";
        std::cout << "unknown_keys_skipped=" << real.unknown_keys_skipped
                   << " (0 means this real file used only value types this reader already models)\n";

        auto* q_bias = real.find_tensor("blk.0.attn_q.bias");
        auto* k_bias = real.find_tensor("blk.0.attn_k.bias");
        auto* v_bias = real.find_tensor("blk.0.attn_v.bias");
        RCHECK(q_bias != nullptr && k_bias != nullptr && v_bias != nullptr);
        std::cout << "blk.0 attention bias tensors present: q="
                   << (q_bias ? "yes" : "no") << " k=" << (k_bias ? "yes" : "no")
                   << " v=" << (v_bias ? "yes" : "no") << " (Llama has none -- Section 15.2)\n";

        std::cout << "\n-- Real-file checks: " << r_passed << "/" << r_tests << " checks passed ";
        std::cout << (r_passed == r_tests ? "ALL PASS --\n" : "FAILURES --\n");
    }

    std::cout << "\n========================================================\n";
    bool all_ok = (g_passed == g_tests) && (argc < 2 || r_passed == r_tests);
    std::cout << (all_ok ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    std::cout << "========================================================\n";
    return all_ok ? 0 : 1;
}
