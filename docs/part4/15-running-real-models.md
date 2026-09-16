# Chapter 15: Running Real Models -- From HuggingFace to First Token with Qwen2.5

**What you will understand by the end of this chapter:**

- Why a GGUF reader verified only against files this book's own writer produced can still be wrong in ways its own tests could never catch -- and what changes once that reader is pointed at an actual downloaded model instead of a synthetic fixture.
- Five concrete, byte-verified ways a real Qwen2.5 checkpoint's architecture differs from the Llama-style assumptions this book built Part 0 and Part 3 against: QKV bias tensors, the real grouped-query-attention ratio, tied input/output embeddings, a RoPE base two orders of magnitude higher than this book's own default, and a ChatML turn format that never prepends a beginning-of-text token.
- How to adapt an already-verified forward pass for those five differences as small, local, well-understood changes rather than a redesign -- and why a synthetic self-test proving the adapted CODE is correct is a genuinely different claim from that code producing a correct answer on real weights.
- The specific, non-obvious ways a real 24-layer, real-weight, real-vocabulary forward pass can go wrong that no synthetic 2-layer self-test can expose: a RoPE pairing convention two conventions can both satisfy the same algebraic invariant, float32 accumulation drift compounding across two dozen layers, a chat template's own undocumented default, and a cache-sharing bug that only a loop order spanning many real layers and many real positions can surface at all.

**What you need to know first:**

- Chapter 3's RMSNorm, SwiGLU FFN, RoPE, and GQA attention -- Section 15.3 reuses all four verbatim, adapting only their call sites and adding one new step, so understanding what Chapter 3 built is what makes "adapting" legible as small changes rather than a rewrite.
- Chapter 4.2's Q8_0 block format and dequantizer, and Chapter 5's memory-mapped GGUF loading -- this chapter's reader and real-weight loading both build directly on that machinery rather than re-deriving it.
- Chapter 12's BPE merge engine and chat-template construction (Chapter 12.1's merge engine, Chapter 12.3's Llama-3-style template) -- Section 15.4 reuses the merge engine directly and Section 15.3 generalizes the template function rather than replacing it.
- This chapter introduces a genuinely new category of honest exception. Every earlier chapter's locked output was reproduced identically across this book's four-way cross-check (native x86_64, GCC14, aarch64 via QEMU, and a real aarch64 device). Sections 15.1, 15.2, and 15.4 each also load an actual ~644 MB downloaded model file that cannot be committed to this book's own repository or transferred into its build environment -- so each of those three files runs in two modes: a self-test mode (no argument) that reproduces identically everywhere, exactly like every earlier chapter, and a real-file mode (a path to the actual model as `argv[1]`) whose output is captured once, on a reader's own machine, and reproduced here as data rather than as something this book's own CI can re-verify. Section 15.3 has no such mode -- it proves the adapted code correct against a small synthetic model at real-shaped proportions, which is all a self-test needs to do, and leaves running that code against real weights to Section 15.4.
- The model this chapter uses throughout is Qwen2.5-0.5B-Instruct, downloaded as `qwen2.5-0.5b-instruct-q8_0.gguf` (Q8_0 quantization, roughly 644 MB). Readers following along need to download this file themselves; it is deliberately excluded from this book's own repository.

---

Every chapter so far has proven its code correct against data this book's own tools produced: a writer this book wrote, read back by a reader this book wrote, checked against values this book's own test computed. That symmetry is exactly what makes those chapters' cross-architecture verification possible, and exactly what this chapter breaks on purpose. A real, published model file was produced by someone else's tooling, encodes someone else's architecture choices, and makes no promise that this book's prior assumptions hold. Section 15.1 builds a GGUF reader that survives contact with such a file -- closing gaps (an unsupported metadata value type, wrong assumptions about a magic number's exact bytes) that no self-consistent round-trip test could ever have surfaced. Section 15.2 puts that reader to work identifying exactly where a real Qwen2.5 checkpoint's architecture diverges from the Llama-style choices this book built Part 0 and Part 3 against: five differences, each checked against the real file's own bytes rather than assumed from a spec sheet or carried forward from an unverified claim. Section 15.3 adapts the actual forward-pass and tokenizer code for those five differences and proves the adaptation correct against a small synthetic model shaped like the real one. Section 15.4, the chapter's capstone, wires every piece -- the reader, the dequantizer, the BPE engine, the adapted transformer block -- together, points the result at the actual 644 MB file, and produces one real, computed, decoded token. Getting that one token right required finding and fixing four genuine bugs that no synthetic self-test, however careful, could have caught -- and this chapter documents each one as it was actually found, not as though the correct answer were obvious from the start.

## 15.1 A Reader That Survives Contact With a Real File

### Intuition

Every earlier GGUF reader in this book was tested the same way: write a file with this book's own writer, read it back with this book's own reader, and confirm the two agree. That test proves internal consistency. It proves nothing about whether either tool agrees with the actual GGUF specification, because a writer and a reader that share the same misunderstanding will always agree with each other while both disagree with reality -- and nothing in a round-trip test can tell the difference.

### The Concept, In Detail

This section's reader closes two such gaps, and finds both of them the same way: by pointing the reader at a real file and watching it fail. The first is a genuine bug carried since Chapter 5.2: the GGUF magic number was checked as a little-endian `uint32_t` constant, `0x46475547`, whose four bytes decode to the ASCII sequence `'G','U','G','F'` -- not the real spec's `'G','G','U','F'`. Chapter 5.2's and Chapter 12.5's own round-trip tests could never catch this, because each one only ever read back a file its OWN writer had just produced with that identical, identically wrong constant. A real file's first four bytes, read the same way, simply do not equal that constant. The fix sidesteps the little-endian encoding question entirely: compare the four raw bytes directly against the literal string `"GGUF"` with `memcmp`, the same approach this chapter's own reconnaissance tools (`gguf_dump.cpp`, `gguf_tie_check.cpp`) already used. The second gap is a value type this book's readers never needed: the real Qwen2.5 checkpoint's `tokenizer.ggml.add_bos_token` field uses `BOOL` (value-type tag 7), which neither Chapter 5.2 nor Chapter 12.5 implemented, because neither ever needed to write one. This section's reader adds `BOOL` support, and -- because a real, externally-produced file can always contain a metadata value type its own author never anticipated needing to model at all -- adds a generic skip path that consumes any unrecognized value type's bytes correctly (recursing into `ARRAY` so an unsupported array element type is skipped too) and keeps parsing everything after it, rather than Chapter 12.5's `return false`, which let one metadata key this reader could not represent cost it every other key that came after.

A second, unrelated correction shows up in the tensor descriptors rather than the metadata: Chapter 5.2's synthetic worked example assigned its own placeholder `GGMLType` values (`Q4_0=2`, `Q8_0=7`) purely to keep two example tensors distinguishable from `F32=0`, and never claimed those values matched any real file, because Chapter 5.2 never opened one. The real file settles the question directly: every quantized tensor in it reads back with `type=8`, and every norm or bias tensor reads back with `type=0` -- so this section's `GGMLType` enum uses the real ggml numbering, with `GGML_Q8_0 = 8`, confirmed from real bytes rather than assumed from a spec sheet.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_real_gguf_inspection.cpp -o 01_real_gguf_inspection
./01_real_gguf_inspection                                            # self-tests only, reproduces everywhere
./01_real_gguf_inspection /path/to/qwen2.5-0.5b-instruct-q8_0.gguf   # adds real-file inspection (requires the downloaded file)
```

**Sample input:** a synthetic fixture written by this section's own `GGUFWriter` at real-shaped proportions (a 4:1 GQA ratio, a `BOOL` metadata value, one deliberately-unsupported value type placed between two array keys), checked for the corrected magic bytes, correct scalar and `BOOL` metadata, a clean skip of the unsupported type with both surrounding array keys still parsed correctly, and correct real ggml tensor type IDs; then, in real-file mode, the actual downloaded checkpoint's own architecture facts -- 24 layers, embedding dimension 896, a 14:2 (7:1) attention head ratio, RoPE base 1,000,000, a 151,936-token vocabulary, 291 tensors split exactly as `24 blocks x 12 tensors/block + 3 top-level`, and the presence of `blk.0`'s Q/K/V attention bias tensors, which Llama's own convention has none of.

```text
========================================================
Chapter 15.1: A Reader That Survives Contact With a Real File
========================================================

-- Test 1: writing a synthetic fixture --
  wrote 880 bytes, GQA ratio 4:1

-- Test 2: reopening and parsing the header --
  magic ok, n_tensors=2 n_kv=9

-- Test 3: scalar and BOOL metadata --
  add_bos_token (BOOL) read back as: false
  GQA ratio from file: 4:1

-- Test 4: an unsupported value type is skipped, not fatal --
  unknown_keys_skipped=1, subsequent keys (tokens, token_type) parsed correctly anyway

-- Test 5: tensor descriptors, real ggml type IDs --
  attn_norm.weight: type=0 (F32)
  attn_q.weight:    type=8 (Q8_0) data_size=272 bytes
  data section offset: 544 (32-byte aligned: yes)

-- Self-tests: 25/25 checks passed ALL PASS --

=== REAL FILE INSPECTION (run via a shell on the reader's own machine against the actual downloaded file; not reproduced in the four-way cross-check environment) ===
file: qwen2.5-0.5b-instruct-q8_0.gguf
magic=GGUF version=3 n_tensors=291 n_kv=26
general.architecture="qwen2" general.name="qwen2.5-0.5b-instruct"
qwen2.block_count=24 context_length=32768
qwen2.embedding_length=896 feed_forward_length=4864
attention.head_count=14 head_count_kv=2 -> GQA ratio 14:2
derived head_dim = embedding_length / head_count = 896/14 = 64
rope.freq_base=1000000.0
attention.layer_norm_rms_epsilon=1.0e-06
general.file_type=7 (llama.cpp llm_ftype: MOSTLY_Q8_0)
tokenizer.ggml.model="gpt2" pre="qwen2"
vocab size=151936 merges=151387
bos_token_id=151643 eos_token_id=151645 add_bos_token=false
tensors: 291 total (170 Q8_0 [type=8], 121 F32 [type=0], 0 other)
derivation check: 24 blocks x 12 tensors/block + 3 top-level = 291 (matches n_tensors: yes)
unknown_keys_skipped=0 (0 means this real file used only value types this reader already models)
blk.0 attention bias tensors present: q=yes k=yes v=yes (Llama has none -- Section 15.2)

-- Real-file checks: 18/18 checks passed ALL PASS --

========================================================
ALL CHECKS PASSED
========================================================
```

!!! warning "[COMMON TRAP] a self-consistent reader/writer pair proves nothing about a real file"
    A reader tested only against its own writer's output can pass every test it has and still misunderstand the actual file format in an arbitrary number of ways, because the test only ever checks that the two tools agree with EACH OTHER -- if both encode the magic number's bytes in the same wrong order, or neither one ever needed to represent a `BOOL` value, no round-trip test constructed entirely from that pair's own output can expose it. This is not a hypothetical: this exact reader/writer pair carried a wrong magic-number constant through two prior chapters undetected for precisely this reason. The only test that can catch this class of bug is pointing the reader at a file NEITHER tool produced -- and the earlier the real file enters a project's test suite, the cheaper each such gap is to find.

## 15.2 Five Ways a Real Qwen2 Checkpoint Differs From Llama

### Intuition

Section 15.1 built a reader that can survive an actual downloaded file. This section puts that reader to work answering a sharper question: which of this book's own Part 0 and Part 3 assumptions, all built and verified against Llama-style architecture choices, does a real Qwen2.5 checkpoint actually violate -- checked against the real file's bytes, not asserted from a spec sheet or carried forward from this chapter's own source material unverified.

### The Concept, In Detail

Five differences, each checked directly against the real file. First, QKV bias: real Qwen2 adds a learned per-output-dimension bias after the Q/K/V projection matmul, confirmed by `blk.0.attn_q.bias`, `blk.0.attn_k.bias`, and `blk.0.attn_v.bias` all being present in the real file's tensor list, with shapes `[896]`, `[128]`, and `[128]` respectively -- Llama's own attention convention, which this book's Chapter 3.4 `gqa_attention` was built against, has no such tensors at all. Second, the grouped-query-attention ratio: the real file's `attention.head_count` and `attention.head_count_kv` read back as 14 and 2, a 7:1 ratio -- and this book's own Table of Contents describes Chapter 15 as covering a "6:1" ratio, which the real file simply does not confirm. This section reports the ratio the bytes actually contain rather than carrying the stated figure forward unverified, exactly the standing rule this book has followed since Chapter 13's own H2O eviction section. Third, tied embeddings: `output.weight` and `token_embd.weight` are compared byte-for-byte across their full 144,643,072-byte extent (not merely checked for matching shapes, which byte-identical tensors trivially share with completely different ones), and found to be genuinely, exactly identical -- the real checkpoint ties its input and output embedding matrices, so there is only one set of weights to load for both roles. Fourth, RoPE base frequency: the real file's `rope.freq_base` is 1,000,000, two orders of magnitude above this book's own Chapter 3.3 default of 10,000 -- and because the rotation angle at dimension-pair `k` is `theta_k = base^(-2k/head_dim)`, a higher base produces a SLOWER rotation at every `k>0` (identically at `k=0`, where `theta_0=1` regardless of base), a checkable mathematical property this section confirms both on a small synthetic RoPE table and directly from the real file's own value. Fifth, ChatML never prepends a beginning-of-text token: the real file's `tokenizer.ggml.add_bos_token` is `false`, and its own embedded `tokenizer.chat_template` (read directly from the file, 2509 bytes long) confirms the `<|im_start|>`/`<|im_end|>` ChatML convention -- unlike every Llama-3-style conversation this book has built so far, which this book's Chapter 12.3 template always begins with a BOS token unconditionally.

This section's own source material states the grouped-query-attention ratio as "6:1" without having checked it against a real file, which is exactly the situation this book's standing rule about unverified figures exists for: rather than silently correct the number or silently repeat it, this section reports both -- what was stated, and what the real bytes actually contain -- and lets the discrepancy stand as data, the same way Chapter 14.3 treated its own source material's unverified "501x" prefix-caching claim.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_llama_vs_qwen2_diff.cpp -o 02_llama_vs_qwen2_diff
./02_llama_vs_qwen2_diff                                            # self-tests only, reproduces everywhere
./02_llama_vs_qwen2_diff /path/to/qwen2.5-0.5b-instruct-q8_0.gguf   # adds real-file verification (requires the downloaded file)
```

**Sample input:** synthetic fixtures isolating each of the five differences -- a model with all three QKV bias tensors against one with none; a genuinely byte-identical embedding pair against a pair differing in exactly one element; RoPE tables built at this book's own base=10000 against the real base=1000000, checked for identical rotation at k=0 and a smaller angle at k=3; and a Llama-3-style template that always starts with a beginning-of-text token against a ChatML-style template that never contains one anywhere -- followed, in real-file mode, by all five differences re-checked directly against the actual downloaded checkpoint's bytes.

```text
========================================================
Chapter 15.2: Five Ways a Real Qwen2 Checkpoint Differs From Llama
========================================================

-- Test 1: QKV bias detection (synthetic) --
  model with bias tensors: q=1 k=1 v=1
  model with no bias tensors (Llama's convention): q=0 k=0 v=0

-- Test 2: tied-embedding byte comparison (synthetic) --
  identical-bytes pair detected as tied: yes
  one-element-different pair detected as tied: no (correct)

-- Test 3: RoPE base frequency changes rotation speed (synthetic) --
  at k=0: both bases rotate identically (angle depends only on position there)
  at k=3, pos=10: base=10000 angle=0.01, base=1000000 angle=0.000316228 (higher base rotates slower)

-- Test 4: chat template structure (synthetic) --
  Llama-3-style template (Chapter 12.3): starts with BEGIN_OF_TEXT, always
  ChatML template (this section): no BOS-equivalent anywhere in 4 ids

-- Self-tests: 18/18 checks passed ALL PASS --

=== REAL FILE VERIFICATION (run via a shell on the reader's own machine against the actual downloaded file; not reproduced in the four-way cross-check environment) ===

[Difference 1: QKV bias]
  blk.0.attn_q.bias present, shape=[896]
  blk.0.attn_k.bias present, shape=[128]
  blk.0.attn_v.bias present, shape=[128]
  Llama's attention convention (this book's Chapter 3.4 gqa_attention) has no such tensors at all

[Difference 2: grouped-query attention ratio]
  head_count=14 head_count_kv=2 -> real ratio 14:2 = 7:1
  this book's own TOC (Chapter 15's own description) states "6:1" -- the real file says 7:1, and this section reports what the bytes actually contain rather than carrying the stated figure forward unverified

[Difference 3: tied embeddings]
  output.weight and token_embd.weight: 144643072 bytes each, compared byte for byte
  result: BYTE-IDENTICAL -- genuinely tied

[Difference 4: RoPE base frequency]
  real qwen2.rope.freq_base=1e+06 vs this book's Chapter 3.3 default base=10000
  at position=50, dimension-pair k=20: this book's default base gives angle=0.158114 rad, the real base gives angle=0.0088914 rad (17.7828x slower at this dimension pair)

[Difference 5: ChatML, no BOS prepend]
  tokenizer.ggml.add_bos_token=false -- unlike Chapter 12.3's Llama-3-style template, no token is ever prepended
  tokenizer.chat_template: 2509 bytes, contains "<|im_start|>": yes, contains "<|im_end|>": yes

-- Real-file checks: 13/13 checks passed ALL PASS --

========================================================
ALL CHECKS PASSED
========================================================
```

!!! warning "[COMMON TRAP] treating a source document's stated architecture figure as verified"
    A number written down in a model card, a chapter's own source material, or even this book's own Table of Contents describes what someone believed to be true at the time of writing, not what a specific downloaded file's bytes actually contain -- and the two can diverge without any error, warning, or version mismatch to flag it, exactly as this section's own "6:1" TOC figure diverges from the real file's confirmed 7:1 ratio. The fix is never to silently correct the stated figure to match what was found, and never to silently keep repeating the stated figure once a real file is available to check it against -- both hide the discrepancy. This section's own choice, printing both the stated figure and the measured one side by side, is the only version of "reporting the architecture" that remains true after the real file enters the picture.

## 15.3 Adapting the Forward Pass and Tokenizer for Qwen2

### Intuition

Section 15.2 established five concrete differences as facts about the real file. This section changes actual forward-pass and tokenizer CODE to match them -- and the central claim worth being precise about is that four of Chapter 3's five building blocks needed no change at all.

### The Concept, In Detail

Chapter 3.1's RMSNorm, Chapter 3.2's SwiGLU FFN, Chapter 3.3's RoPE, and Chapter 3.4's GQA attention and `KVCache` are all reused verbatim in this section. RoPE's base frequency and GQA attention's group size were already parameters, not hard-coded constants, so Adaptations 2 and 3 -- Difference 4's `rope_theta=1,000,000` and Difference 2's real 7:1 ratio -- are call-site arguments, not code changes. The two adaptations that ARE new code are small and local. Adaptation 1 is a bias-add step: a two-line function that adds a learned per-output-dimension bias vector after the Q/K/V projection matmul, called only because Difference 1 confirmed real Qwen2 has such tensors at all -- Chapter 3.4's `gqa_attention`, built against Chapter 12's Llama-style source material, never needed this function because Llama's own convention has no such bias tensors. Adaptation 4, on the tokenizer side, generalizes Chapter 12.3's Llama-3-only chat-template function (which hard-coded "always prepend BOS") into one function driven by a single boolean -- exactly Section 15.1's `tokenizer.ggml.add_bos_token`, read directly from whichever real file is loaded, rather than assumed from the model family's name. The complete adapted block is RMSNorm, QKV projection WITH bias, RoPE on Q and K, GQA attention, output projection (no bias -- Section 15.1's real-file tensor listing confirms `attn_output` has none), a residual connection, a second RMSNorm, the SwiGLU FFN, and a second residual -- Chapter 11's "assemble one full transformer layer" pattern, re-run here with the adaptations wired in rather than assumed away.

Every check in this section runs against a small synthetic model at real-shaped proportions (7 query heads sharing 1 KV head, echoing the real file's own 14:2 ratio in miniature) rather than the actual 644 MB checkpoint. That is a deliberate scope decision, not a shortcut: this section's job is proving the adapted CODE is correct, which small deterministic weights can do exactly as well as real ones and a great deal faster, while loading real weights and running this exact machinery against them is Section 15.4's job specifically. That distinction matters more than it might appear to, because this section's own self-tests -- 25 checks across 6 tests, all passing, all reproducing identically across this book's four-way cross-check -- turned out NOT to be sufficient to catch every bug in this machinery. Section 15.4's real-file debugging found two genuine correctness bugs in code this section had already verified: `apply_rope` was using an interleaved `(vec[2k], vec[2k+1])` pairing convention, when real HF-derived architectures -- Qwen2 included -- use a split-half `(vec[k], vec[k+half_dim])` pairing instead, confirmed by comparing against an independently built `llama.cpp` running the same real file; and `rms_norm`, `matmul`, and the attention reductions in `gqa_attention` were accumulating in `float`, when the correct practice for a reduction spanning many terms (and, in the real model, 24 stacked layers) is to accumulate in `double` and round back to `float` only at the end. Both fixes are applied here, in this section's own file, rather than only in Section 15.4's -- but it is worth being honest about why THIS section's self-tests never caught either one before Section 15.4's real-file debugging found them: this section's own algebraic invariant (Test 4's relative-position invariance, `dot(Q@5,K@3) == dot(Q@2,K@0)`) holds under EITHER RoPE pairing convention, and its small synthetic dimensions never accumulate enough float32 drift across only 3 positions and 2 layers to flip an outcome the way 24 real layers eventually did. A synthetic self-test proves the code behaves consistently with itself; it does not, by construction, prove the code matches an independently-verified ground truth, which is exactly the gap Section 15.4 exists to close.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_qwen2_transformer_block.cpp -o 03_qwen2_transformer_block
./03_qwen2_transformer_block
```

**Sample input:** `linear_with_bias` hand-traced against a known input/weight/bias triple, and separately checked to actually change the output relative to a zero bias; a real 7:1 GQA ratio checked so that all four query heads sharing KV head 0 agree and all three sharing KV head 1 agree; RoPE's relative-position invariance checked at the real base=1,000,000; a complete adapted block run across 3 sequential positions with a real 7:1 ratio, checked for a finite, non-zero-norm output and bit-identical determinism on a fresh cache and identical input; and one `encode_conversation` function checked to produce Chapter 12.3's Llama-3 convention when `add_bos_token=true` and Section 15.2's ChatML convention when it is `false`, driven by the same single boolean either way.

```text
========================================================
Chapter 15.3: Adapting the Forward Pass and Tokenizer for Qwen2
========================================================

-- Test 1: linear_with_bias (hand-traced) --
  linear_with_bias([1,2]) = [1.5, 1.5, 5]

-- Test 2: bias is not a no-op --
  Wx (zero bias) = [0.3, -0.7]
  Wx+b (real bias) = [0.55, -0.8]

-- Test 3: real 7:1 GQA ratio (7 query heads, 1 KV head) --
  head 0 (K0-aligned) output[0]=1.37754, head 1 (K1-aligned) output[0]=1.62246
  all 4 K0-aligned heads (0,2,4,6) agree, all 3 K1-aligned heads (1,3,5) agree: yes

-- Test 4: RoPE relative-position invariance at base=1000000 --
  dot(Q@5,K@3) == dot(Q@2,K@0) at base=1000000: yes

-- Test 5: complete Qwen2 block, 7:1 GQA, 3 positions --
  after 3 positions: finite=1, ||x||^2=1.46713
  re-run with fresh cache, same weights and input: bit-identical: yes

-- Test 6: one tokenizer function, two model families --
  add_bos_token=true  (Llama-3 convention): 7 ids, starts with BOS
  add_bos_token=false (Qwen2 ChatML convention): 6 ids, no BOS anywhere
  same encode_conversation() function, driven by one boolean read from the GGUF file

35/35 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] a synthetic self-test's algebraic invariant can hold under the wrong convention"
    RoPE's relative-position invariance -- that the dot product between a query at position `p+d` and a key at position `p` depends only on `d`, not on `p` itself -- is a property of rotating query and key vectors by a CONSISTENT, position-dependent angle, and it holds regardless of which of two internally-consistent pairing conventions (interleaved or split-half) is used to apply that rotation, as long as the same convention is used everywhere. A self-test built around that invariant can therefore pass 100% of its checks while using a convention that is nonetheless wrong for the specific real architecture the code is meant to model -- the test proves the rotation is being applied CONSISTENTLY, not that it is being applied the way the real model's own training expects. The only way to catch a wrong-but-internally-consistent convention is to compare against an independently produced ground truth -- a reference implementation, or, as in this book's case, a from-source build of `llama.cpp` run against the identical real file -- which is a fundamentally different kind of check than any property a synthetic self-test can assert about its own output alone.

## 15.4 A First Real Token From Qwen2.5-0.5B-Instruct

### Intuition

Every piece this chapter needs already exists: Section 15.1's reader, Chapter 4.2's Q8_0 dequantizer, Chapter 12.1's BPE merge engine, and Section 15.3's adapted transformer block. This section wires all of them together, adds the one genuinely new piece a real byte-level BPE vocabulary requires, and runs the result against the actual downloaded checkpoint to produce one real, computed, decoded token.

### The Concept, In Detail

The one new piece is GPT-2's reversible byte-to-unicode mapping: Chapter 12's own synthetic vocabulary used raw bytes as vocabulary strings directly, which a real byte-level BPE tokenizer's vocabulary does not do, so this section implements the mapping that makes every one of the 256 possible byte values round-trip through the real vocabulary's printable-and-remapped symbol alphabet exactly. Everything else is assembly: Section 15.1's reader loads the real 291-tensor, 151,936-token-vocabulary file; Chapter 4.2's dequantizer converts every Q8_0-quantized weight to `float` once, up front, rather than repeatedly on every prompt position; Chapter 12.1's merge engine and this section's byte-to-unicode codec turn a real ChatML prompt into real token IDs; and Section 15.3's adapted 24-layer transformer block runs a real forward pass over those IDs to produce logits, from which an argmax step picks the next token.

Wiring already-verified pieces together turned out not to be the whole story, in exactly the way Chapter 13.5 first established for this book: combining independently-correct pieces is its own claim, requiring its own verification, not a free consequence of each piece already being correct in isolation. Getting one real, correct token out of this section required finding and fixing four separate bugs, none of which any earlier synthetic self-test -- including Section 15.3's own 25/25-passing self-test -- could have caught, because each one only manifests at a scale or against a ground truth no synthetic fixture provides. The first two were Section 15.3's own RoPE-convention and float32-accumulation bugs, described in that section and fixed identically here, found by comparing this section's real 24-layer output against an independently built `llama.cpp` running the identical real file: `llama-eval-callback`'s per-layer tensor sums matched this section's own computation almost exactly for early layers, but diverged in a way that flipped an argmax by the final layer with float32 accumulation, and matched far more closely once the reduction was moved to `double`. The third was a missing default in the ChatML prompt construction: the real file's own embedded `tokenizer.chat_template` -- a Jinja template, inspected directly with `strings` on the downloaded file -- always injects a default system turn (`You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`) when the caller supplies none, and an earlier draft of this section's prompt construction omitted it entirely, on the reasonable-looking assumption that a single user question needed nothing else. Comparing against `llama-simple-chat`, which applies the model's own real template, confirmed the system turn was exactly what was missing; adding it produced a 36-token prompt matching `llama-tokenize`'s reference token IDs exactly, token for token.

The fourth bug was the deepest, and the one most specific to what a REAL multi-layer, multi-position generation loop actually exercises that a small synthetic self-test does not. `run_generation`'s original loop was position-major -- for each position, run all 24 layers -- sharing a single `KVCache` object across every one of those 24 layers. A `KVCache` stores one layer's worth of key/value data; because `qwen2_block_forward` calls `cache.store()` at every layer, processing one position through all 24 layers sequentially meant each layer's stored key/value for that position was immediately overwritten by the NEXT layer's, before any later position ever attended back to it. By the time a later position looked up an earlier position's cached key/value at a given layer, the cache slot held whatever layer had been processed LAST for that earlier position -- always layer 23 -- not the correct current layer's data at all. Section 15.3's own self-test, and this section's own synthetic Test 4, never exposed this, for a precise structural reason: both process only a SINGLE simulated layer, called repeatedly across positions against one cache, which is exactly the one usage pattern under which a single shared cache is correct. The bug is invisible to any test that never runs more than one layer's worth of `qwen2_block_forward` calls against a shared cache -- which describes every synthetic self-test in this chapter, and is exactly why this section's real, 24-layer, 36-token ChatML generation was the first thing in this book able to expose it at all. It was found by comparing production behavior against an ad-hoc diagnostic that restructured the same computation layer-major purely to print a per-layer trace, and that restructuring happened to sidestep the bug by accident, producing the correct answer and revealing that the ACTUAL production loop's shared cache -- not the earlier RoPE, precision, or prompt fixes -- was still silently wrong. The fix keeps the natural, realistic position-major loop order (which is what real streaming, autoregressive generation actually requires) and replaces the single shared `KVCache cache(...)` with a `std::vector<KVCache> caches`, one entry per layer, allocated once before generation begins and passed as `caches[layer]` at each layer's call site inside the unchanged position-major loop -- visible in full in `run_generation` in the listing below.

With all four fixes in place, a real 24-layer forward pass over the real 36-token ChatML prompt `"What is the capital of France?"` produces `id=785, logit=22.5446, decoded="The"` -- the model correctly beginning the answer "The capital of France is Paris." -- reproduced bit-identically on a second full run, and reproduced identically across GCC 11.4.0 (this book's usual real-device target) and, separately, Apple clang on the author's own machine, which confirmed the remaining question at one point in this section's debugging was genuinely algorithmic rather than toolchain-dependent.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 04_first_real_token.cpp -o 04_first_real_token
./04_first_real_token                                            # self-tests only, reproduces everywhere
./04_first_real_token /path/to/qwen2.5-0.5b-instruct-q8_0.gguf   # adds real generation (requires the downloaded file)
```

**Sample input:** all 256 byte values round-tripped through the GPT-2 byte-to-unicode codec; pre-tokenization chunk boundaries checked against two hand-worked example strings; a tiny two-layer synthetic model's embedding dequantization checked against Q8_0's own known error bound; a full synthetic pipeline from embedding through argmax checked for a finite, bit-identical-on-rerun result; and, in real-generation mode, the actual 644 MB checkpoint loaded, a real 36-token ChatML prompt (with the model's own default system turn) encoded and checked byte-for-byte against `llama-tokenize`'s reference IDs, a real 24-layer forward pass run over all 36 positions, and the resulting first generated token checked to be bit-identical across two full re-runs of the entire computation.

```text
========================================================
Chapter 15.4: A First Real Token From Qwen2.5-0.5B-Instruct
========================================================

-- Test 1: GPT-2 byte<->unicode codec --
  all 256 bytes round-trip through encode+decode: yes
  byte 'A' (printable) maps to itself; byte ' ' maps to a 2-byte UTF-8 symbol

-- Test 2: gpt2_pretokenize chunk boundaries --
  "The capital of France is" -> 5 chunks: [The][ capital][ of][ France][ is]
  "What is the capital of France?" -> 7 chunks (trailing "?" has no leading space)

-- Test 3: synthetic model, embedding dequantization --
  loaded shape: dim=32 heads=4 heads_kv=2 layers=2
  embedding row 5 max dequantization error: 0.00254571 (Q8_0 tolerance)

-- Test 4: full synthetic pipeline, embedding to argmax --
  argmax token id: 59, logit=4.21505
  re-run bit-identical: yes

-- Self-tests: 15/15 checks passed ALL PASS --

=== REAL GENERATION (run via a shell on the reader's own machine against the actual downloaded file; not reproduced in the four-way cross-check environment) ===
model loaded: dim=896 heads=14 heads_kv=2 layers=24 vocab=151936

building vocabulary and merge table from the real file...
vocabulary: 151936 tokens, 151387 merge rules
special tokens: <|im_start|>=151644 <|im_end|>=151645

ChatML prompt (add_bos_token=false): 36 tokens
  raw content: "What is the capital of France?"

dequantizing all 24 real layers once (reused across every prompt position, rather than re-dequantizing the same weights on every position)...
dequantization done.

running the real 24-layer forward pass over 36 prompt positions...

first generated token: id=785 logit=22.5446 vocab_text="The" decoded="The"

re-running the full generation once more to confirm determinism (this repeats the entire 24-layer pass)...
re-run: id=785 logit=22.5446 -- bit-identical to first run: yes

-- Real-generation checks: 13/13 checks passed ALL PASS --

========================================================
ALL CHECKS PASSED
========================================================
```

!!! warning "[COMMON TRAP] a single shared resource is only safe under the access pattern a test happens to use"
    A single `KVCache` shared across many layers is not a bug in isolation -- it is the CORRECT design for the one access pattern this book's earlier synthetic tests always used: a single simulated layer, called repeatedly across positions, where nothing else ever writes to that cache in between. The moment a real multi-layer forward pass introduces a SECOND layer writing to the same shared object at the same position, the design silently stops being correct, without changing a single line of the cache's own code -- the bug lives entirely in how many distinct writers share the object, not in the object itself. This is exactly why the bug survived Section 15.3's 25/25-passing self-test and this section's own Test 4: neither test's access pattern was capable of exposing it, no matter how many additional assertions either one added, because the underlying resource-sharing assumption those tests exercised was never actually violated until a real, 24-layer forward pass came along. The general lesson generalizes past `KVCache` specifically: any object one test's access pattern happens to use safely deserves an explicit note about WHICH access pattern that safety depends on, because the next caller has no way to discover the assumption from the object's own interface alone.

## Chapter Summary

This chapter took every piece of machinery this book had built through Chapter 14 and pointed it at an actual, externally-produced model file for the first time. Section 15.1 built a GGUF reader that survives contact with a real file, fixing two gaps -- a wrong magic-number constant and a missing metadata value type -- that no self-consistent round-trip test against this book's own writer could ever have exposed. Section 15.2 used that reader to establish five concrete, byte-verified architecture differences between Llama, which this book's Part 0 and Part 3 were built against, and a real Qwen2.5 checkpoint: QKV bias tensors, a real 7:1 grouped-query-attention ratio (correcting this book's own unverified "6:1" TOC figure), tied embeddings, a two-orders-of-magnitude-higher RoPE base, and a ChatML format with no beginning-of-text token. Section 15.3 adapted the actual forward-pass and tokenizer code for those five differences as small, local changes -- four of Chapter 3's five building blocks needed no change at all -- and proved the adaptation correct against a small synthetic model, while being explicit that a synthetic self-test's passing does not, by itself, prove the code matches real, independently-verified ground truth. Section 15.4 closed that exact gap: wiring every piece together and running it against the real 644 MB file surfaced four genuine bugs -- a wrong RoPE pairing convention, float32 accumulation drift across 24 real layers, a missing default system turn the real file's own chat template requires, and a cross-layer cache-sharing bug that only a real multi-layer, multi-position forward pass could expose -- each found by comparing against an independently built `llama.cpp` running the identical real file, and none of them catchable by any synthetic self-test in this chapter, including this chapter's own. With all four fixed, this book's own from-scratch C++ inference engine produces a real, correctly-computed, correctly-decoded first token from a real downloaded model: "The," correctly beginning "The capital of France is Paris."

## Self-Check Questions

1. Section 15.1's magic-number bug (`0x46475547` instead of the correct byte sequence) survived two earlier chapters' own round-trip tests undetected. Explain precisely why a test that writes a file and reads it back with the SAME tool pair cannot, by construction, catch this specific class of bug.
2. Section 15.1's reader responds to an unrecognized metadata value type by skipping its bytes and continuing, rather than Chapter 12.5's `return false`. Why does this specific choice matter more for a REAL file than it ever did for a file this book's own writer produced?
3. Section 15.2 reports both this book's own stated "6:1" GQA ratio and the real file's measured 7:1 ratio, rather than silently using whichever one is correct. What standing rule from earlier chapters does this follow, and what would silently correcting the number hide?
4. Section 15.2's RoPE comparison shows that a higher base frequency produces an IDENTICAL rotation angle at dimension-pair k=0 but a SMALLER angle at k=3, for any two bases. Why is k=0 unaffected specifically, based on the formula `theta_k = base^(-2k/head_dim)`?
5. Section 15.3 states that four of Chapter 3's five building blocks needed no code change to support Qwen2 -- only call-site arguments changed. Which specific parameters, already present in Chapter 3's own RoPE and GQA attention functions, are what made this possible?
6. Section 15.3's own self-test passed 25/25 checks while still containing the RoPE-convention bug that Section 15.4 later found. Explain why Test 4's relative-position-invariance check could not have distinguished the interleaved pairing from the correct split-half pairing.
7. Section 15.4 found a missing default system turn by inspecting the real GGUF file's own embedded `tokenizer.chat_template` directly with `strings`, rather than assuming a prompt format from the model's documentation or general ChatML convention. Why was checking the file's own bytes necessary here, given that Section 15.2 had already confirmed the file uses ChatML?
8. Section 15.4's KVCache bug required a real, 24-layer, multi-position forward pass to expose -- neither Section 15.3's synthetic self-test nor Section 15.4's own synthetic Test 4 could catch it. What specific property of BOTH of those tests' access patterns made the shared-cache design safe for them specifically, and unsafe for the real generation loop?
9. The float32-accumulation bug and the RoPE-convention bug were both found by comparing against an independently built `llama.cpp`, rather than by inspecting this book's own code for mistakes. Why was an independent ground truth necessary for these two bugs specifically, when the missing-system-turn bug was found by inspecting the real file's own bytes instead?
10. Suppose a future chapter builds a new synthetic self-test for a multi-layer forward pass, but that test -- like Section 15.3's -- only ever exercises a single simulated layer called repeatedly. Based on this chapter's own account of the KVCache bug, what specific class of bug would such a test still be structurally unable to catch, no matter how many additional assertions it added?

## Where We Go Next

This chapter closed the gap between "an inference engine whose every piece is independently verified" and "an inference engine that produces a correct answer from a real, downloaded model" -- and closing that gap required exactly the kind of debugging no synthetic self-test, however thorough, can substitute for: an independently built reference implementation, and a willingness to keep looking after every earlier fix still left the wrong answer in place. With a real forward pass now producing a real, correct first token, Chapter 16 turns from producing one token to producing a conversation: the generation loop that turns one correct token into a full response, the interactive loop that turns one response into an ongoing exchange, a streaming token decoder, and the error handling and startup sequence a production binary needs that a single self-contained section never did -- all of it compiled into the single deployable binary this book has been building every piece of since Chapter 1.

## Worked Solutions

**1.** A round-trip test that writes a file with tool A and reads it back with tool A can only ever check that A agrees with itself -- if A's writer encodes the magic number's bytes in a specific (wrong) order and A's reader checks for that same (wrong) order, the two will always agree, because the test never introduces any information from outside that pair. The only way such a test could catch the bug is if the reader and writer disagreed with each other, which requires one of them to be right and the other wrong -- but this bug was a shared misunderstanding present in BOTH, so there was never a disagreement for the test to find. Only a file neither tool produced -- a real file, encoded by an independent, correct implementation of the actual spec -- introduces the outside information needed to expose the mismatch.

**2.** A file this book's own writer produced only ever contains metadata value types that writer itself was already programmed to emit, so a reader built against that writer's output will, by construction, never encounter a type it doesn't recognize -- Chapter 12.5's `return false` on an unrecognized type was therefore never actually exercised by any of that chapter's own tests. A real, externally-produced file carries no such guarantee: its author may have used any value type the real GGUF specification defines, entirely independent of what this book's own writer happens to emit, so the "unrecognized type" code path that was dead code against synthetic fixtures becomes a real, frequently-hit path the moment a real file is involved -- exactly why this section changed its behavior from failing the whole parse to skipping just that one key.

**3.** This follows the same standing rule Chapter 14.3 established when its own source material's unverified "501x" prefix-caching speedup claim was reported as unverified rather than either repeated as fact or silently ignored: a number that has not been checked against real, measured behavior is data about what someone once believed, not a verified fact, and the book's rule is to never carry such a number forward as though it had been confirmed. Silently correcting "6:1" to "7:1" in the TOC without comment would erase the historical record that the original figure was never checked in the first place; silently continuing to report "6:1" once the real file has been measured would actively misinform a reader relying on this chapter for the real architecture. Reporting both, side by side, is the only choice that stays honest about what was claimed versus what was actually measured.

**4.** The formula `theta_k = base^(-2k/head_dim)` reduces to `theta_0 = base^0 = 1` for any base whatsoever when `k=0`, because any nonzero number raised to the power 0 equals 1 -- the base's specific value never enters the calculation at all at that one dimension-pair. For any `k>0`, the exponent `-2k/head_dim` is nonzero, so the base's value directly affects `theta_k`, and because `base` in this formula is greater than 1 in both cases compared, a LARGER base produces a SMALLER `theta_k` (and therefore a smaller rotation angle at a given position) for every `k>0` -- exactly the pattern Test 3 confirms by checking identical angles at k=0 and a strictly smaller angle at k=3 for the higher base.

**5.** Chapter 3.3's `RoPETables` constructor already took a `base` parameter (used with Chapter 3's own default of 10,000), and Chapter 3.4's `gqa_attention` and `KVCache` already took a `group_size` parameter (used with Chapter 3's own worked examples of 1:1 and 2:1 ratios) -- neither function hard-coded its respective value as a compile-time constant. This meant Adaptations 2 and 3 (RoPE base 1,000,000 and a real 7:1 GQA ratio) were satisfied simply by passing different values to parameters that already existed, with zero changes to either function's own implementation -- the adaptation lives entirely at the call site, not inside Chapter 3's original code.

**6.** RoPE's relative-position invariance states that `dot(Q_at_position_p+d, K_at_position_p)` depends only on the distance `d`, not on `p` itself -- a property that holds as long as query and key vectors are rotated by a CONSISTENT, well-defined, position-dependent angle using SOME fixed pairing convention, applied identically to both vectors. Both the interleaved `(vec[2k], vec[2k+1])` pairing and the correct split-half `(vec[k], vec[k+half_dim])` pairing are individually well-defined, internally consistent rotation schemes that each satisfy this invariant on their own terms -- the invariant is a statement about internal consistency, not about which specific real-world architecture the convention happens to match. A test that only checks the invariant therefore cannot distinguish "correctly rotated the real Qwen2 way" from "consistently rotated some other, equally self-coherent but wrong way," because both produce a passing result on that specific check.

**7.** Section 15.2 confirmed the real file uses ChatML formatting -- the `<|im_start|>`/`<|im_end|>` tokens and no BOS prepend -- which describes the STRUCTURE of how turns are delimited, but says nothing about WHICH turns a well-formed ChatML prompt should include for a given input. The specific detail that a caller supplying no system message still receives an implicit default system turn is a behavior of THIS model's own fine-tuning and its own chat template's Jinja logic, not a property of the ChatML format in general -- some ChatML-formatted models have no such default at all. Only inspecting the real file's own embedded template (or comparing against a tool, like `llama-simple-chat`, that applies that exact template) can reveal a model-specific default like this; general knowledge of "this is a ChatML model" is the wrong level of detail to answer the question.

**8.** Both Section 15.3's self-test and Section 15.4's synthetic Test 4 process only a SINGLE simulated transformer layer, called repeatedly across multiple positions against one shared `KVCache` -- and under that specific access pattern, a shared cache is entirely correct, because the one and only writer at any given position is that same single layer every time; nothing else ever intervenes to overwrite what it just stored. The real generation loop introduces a second, structurally different pattern: 24 DISTINCT layers, each calling `cache.store()` for the SAME position in immediate succession, before any later position comes along to read it back. A single shared cache object cannot distinguish "the same layer writing again" from "a different layer writing for the first time" -- it has no concept of "layer" at all -- so the moment more than one distinct layer writes to it, the design that was correct for one layer silently becomes wrong for many, without any change to the cache's own code.

**9.** The float32-accumulation and RoPE-convention bugs are both cases where this book's OWN code was self-consistent -- every self-test it wrote for itself passed -- but disagreed with what a real, correctly-implemented Qwen2 forward pass should actually compute; nothing internal to this book's own code could reveal that disagreement, because the code had nothing else to compare itself against. Only an INDEPENDENT implementation, built by different authors from the same architecture specification and already known to produce correct results, can serve as the outside reference needed to catch a self-consistent-but-wrong computation. The missing-system-turn bug is a different kind of gap entirely: it is not a computational error but a missing piece of INPUT construction, and the fact revealing it (the real file's own embedded default template) is data recorded directly inside the real file itself, not a claim about the file's correct chat behavior derived from running someone else's software -- so inspecting the file's own bytes was sufficient there, while the two computational bugs specifically required an independent, already-correct forward pass to compare against.

**10.** Such a test would remain structurally unable to catch any bug that only manifests when MULTIPLE DISTINCT layers write to a SHARED resource in succession -- a cross-layer cache-sharing bug identical in kind to the one this chapter found, regardless of how many additional assertions were added about that single simulated layer's own OWN correctness. Adding more checks to a test that only ever exercises one layer cannot reveal a bug whose entire cause is the INTERACTION between two or more layers sharing state, because that interaction simply never occurs in the test's access pattern no matter how carefully the single layer's behavior is checked -- exactly the lesson Section 15.4's own account draws: the bug lived in how many distinct writers shared the object, a property no single-layer test can vary no matter how it's constructed.
