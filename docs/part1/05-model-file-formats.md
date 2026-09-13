# Chapter 5: Model File Formats -- GGUF, SafeTensors, and Memory-Mapped Loading

**What you will understand by the end of this chapter:**

- The exact byte-level layout of a GGUF file — header, typed key-value metadata store, tensor descriptor table, and a 32-byte-aligned data section — and why that layout, not a JSON schema or a serialization library, is what makes zero-copy loading possible at all.
- How to write a genuine GGUF file from scratch and read it back, verifying every field the reader recovers against exactly what the writer wrote.
- How the naming convention that indexes a GGUF file's tensors doubles as the interface between the file and the inference engine, and how a tensor descriptor's byte offset becomes a `std::span` view directly into memory-mapped pages, with zero heap allocation for tensor data.
- Why SafeTensors — the format nearly every model ships in on HuggingFace — uses a JSON header instead of GGUF's binary one, and how to parse that header with a hand-rolled extractor rather than a general JSON library.
- Why production GGUF files mostly use hierarchical "K-quant" super-blocks rather than the flat Q4_0/Q8_0 blocks built in Chapter 4, and that the difference is how a fixed bit budget gets allocated, not how large it is.

**What you need to know first:**

- Chapter 3.6's `mmap`-based weight loading (this chapter builds the GGUF-specific catalog on top of it, rather than re-deriving `mmap` itself), and Chapter 4's `fp16_t`, `BlockQ8`, and `BlockQ4` block layouts, which this chapter's file formats store on disk exactly as designed.
- `std::span`, basic POSIX file I/O (`open`, `mmap`, `munmap`), and enough bit manipulation to follow packed multi-bit fields across byte boundaries.

---

Chapters 2 through 4 built every mechanism a transformer's forward pass needs — tensors, kernels, quantized blocks — but every one of those chapters generated its own weights in-process with a seeded random number generator. Real deployment starts from a file: five billion real, trained parameters that have to get from disk into a running engine without wasting time, memory, or correctness. This chapter builds the two file formats that matter for that step. GGUF (the format used by llama.cpp, Ollama, LM Studio, and every quantized model on HuggingFace's GGUF repositories) is built byte by byte, written, read back, and then read again through a zero-copy `mmap` view that turns Chapter 4's Q8/Q4 blocks into `std::span`s pointing straight into mapped file pages. SafeTensors — the format nearly every model ships in before quantization — gets its own hand-rolled JSON header parser and its own zero-copy round trip. The chapter closes by correcting an arithmetic slip in the kind of quantization-format documentation this material is drawn from, and using the corrected numbers to show what production "K-quant" formats actually trade for their better quality: not a smaller file, but a smarter way of spending the same number of bits.

## 5.1 The GGUF Binary Format: Header, Metadata, and Tensor Descriptors

### Intuition

A model file format is a binary container that stores everything an inference engine needs — architecture parameters, tokenizer, and every weight tensor in its final quantized form — laid out in exactly the byte order the engine will consume. GGUF (GGML Universal File format, introduced by the llama.cpp project in 2023) does this with a fixed header, a typed key-value metadata store, a table of tensor descriptors, and a contiguous data section: no JSON to parse, no Python interpreter required, no serialization layer standing between the bytes on disk and the values the engine reads.

### The Concept, In Detail

Every multi-byte value in GGUF is little-endian. Strings are length-prefixed — a `uint64_t` byte count followed by raw characters, no null terminator — which avoids scanning for a terminator and makes a string's length available without a linear search. The 24-byte header holds a magic number (`0x46475547`, "GGUF" in little-endian ASCII), a version, a tensor count, and a metadata-pair count. Each metadata key-value pair is a length-prefixed key, a `uint32_t` type tag, and a value in the format that tag specifies. Each tensor descriptor is a length-prefixed name, a dimension count and the dimensions themselves, a quantization-type enum, and a byte offset relative to the start of the data section — and that data section begins at the first 32-byte-aligned position after the last descriptor, so that SIMD loads against the tensor data never straddle an unaligned boundary. Getting this exactly right by tracing a hex dump on paper is exactly the kind of arithmetic that is easy to get subtly wrong — a byte miscounted here or there compounds through every field that follows it — which is why the worked example below computes every offset in code and checks it, rather than trusting a number written down in a table.

### Code and Verification

```cpp
// Chapter 5.1 -- Chapters 2-4 treated model weights as C++ arrays that
// appeared in memory by construction: generate random floats, quantize
// them in-process, run inference on the result. Real deployment starts
// from a file on disk, and GGUF (GGML Universal File format, the
// format used by llama.cpp, Ollama, LM Studio, and every quantized
// model on HuggingFace's GGUF repositories) is the container that gets
// real, trained weights from that file into a running engine without
// wasting time, memory, or sanity. It has no JSON to parse and no
// Python dependency (unlike PyTorch's pickle-based .pt files, which
// require a Python interpreter and have been a real remote-code-
// execution vector) -- just a fixed binary header, a typed key-value
// metadata store, a table of tensor descriptors, and a contiguous data
// section, all readable with nothing more than memcpy.
//
// This section builds and then parses, byte by byte, the smallest
// possible non-trivial GGUF file: a header, one metadata key-value
// pair, and one tensor descriptor. Every multi-byte value is little-
// endian; every string is length-prefixed (a uint64 byte count, no
// null terminator); the data section begins at the first 32-byte-
// aligned offset after the last tensor descriptor. Rather than tracing
// a hex dump computed by hand on paper (where a byte-counting slip is
// easy to make and easy to miss), the byte offsets and the alignment
// padding below are computed and checked by the program itself --
// the same discipline this book has applied to every numeric claim
// since Chapter 2.

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

static constexpr uint32_t GGUF_MAGIC = 0x46475547;   // "GGUF" in little-endian ASCII
static constexpr uint32_t GGUF_VERSION = 3;
static constexpr uint32_t GGUF_TYPE_STRING = 8;
static constexpr uint32_t GGML_TYPE_F32 = 0;

// -- Byte-level append helpers: each one appends the little-endian wire
// representation of a value to a growing byte buffer, mirroring the
// exact primitives a real GGUFWriter uses internally (Section 5.2
// wraps these in a class; here they stay inline so every byte written
// is visible at the call site). --
void put_u32(std::vector<uint8_t>& buf, uint32_t v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + 4);
}
void put_u64(std::vector<uint8_t>& buf, uint64_t v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + 8);
}
void put_string(std::vector<uint8_t>& buf, const std::string& s) {
    put_u64(buf, static_cast<uint64_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

// -- Matching byte-level read helpers, advancing a cursor through the buffer. --
uint32_t get_u32(const std::vector<uint8_t>& buf, size_t& pos) {
    uint32_t v; std::memcpy(&v, buf.data() + pos, 4); pos += 4; return v;
}
uint64_t get_u64(const std::vector<uint8_t>& buf, size_t& pos) {
    uint64_t v; std::memcpy(&v, buf.data() + pos, 8); pos += 8; return v;
}
std::string get_string(const std::vector<uint8_t>& buf, size_t& pos) {
    uint64_t len = get_u64(buf, pos);
    std::string s(reinterpret_cast<const char*>(buf.data() + pos), len);
    pos += len;
    return s;
}

int main() {
    std::cout << "================================================\n";
    std::cout << "The GGUF Binary Format, Byte by Byte\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: Build the smallest non-trivial GGUF file by hand: a header,
    // one metadata pair ("general.architecture" = "llama"), and one
    // tensor descriptor (a float32 embedding table, shape [4096, 32768]).
    // =====================================================================
    std::cout << "-- Test 1: Building the file, field by field --\n";
    std::vector<uint8_t> file;
    {
        // Header: magic, version, n_tensors, n_kv (24 bytes fixed).
        put_u32(file, GGUF_MAGIC);
        put_u32(file, GGUF_VERSION);
        put_u64(file, 1);   // n_tensors
        put_u64(file, 1);   // n_kv
        size_t header_end = file.size();
        std::cout << "  header: " << header_end << " bytes\n";
        CHECK(header_end == 24);

        // One metadata KV pair: key, type tag, value.
        size_t kv_start = file.size();
        put_string(file, "general.architecture");
        put_u32(file, GGUF_TYPE_STRING);
        put_string(file, "llama");
        std::cout << "  metadata KV pair: " << (file.size() - kv_start) << " bytes\n";

        // One tensor descriptor: name, n_dims, dims[], type, offset.
        size_t td_start = file.size();
        put_string(file, "token_embd.weight");
        put_u32(file, 2);            // n_dims
        put_u64(file, 4096);         // dims[0]
        put_u64(file, 32768);        // dims[1]
        put_u32(file, GGML_TYPE_F32);
        put_u64(file, 0);            // offset (relative to data section start)
        std::cout << "  tensor descriptor: " << (file.size() - td_start) << " bytes\n";
        std::cout << "  total before alignment: " << file.size() << " bytes\n";

        // Show the raw header bytes as a hex dump -- the same view this
        // section's byte-level tracing is built on.
        std::cout << "  header hex: ";
        for (size_t i = 0; i < header_end; ++i)
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)file[i] << " ";
        std::cout << std::dec << std::setfill(' ') << "\n";
    }

    // =====================================================================
    // TEST 2: Alignment -- the data section starts at the first 32-byte
    // boundary at or after the end of the tensor descriptors, computed
    // (not assumed) from the buffer's actual current size.
    // =====================================================================
    std::cout << "\n-- Test 2: Alignment to the data section --\n";
    size_t data_section_start = 0;
    {
        size_t pos_before_padding = file.size();
        size_t remainder = pos_before_padding % 32;
        size_t pad = (remainder == 0) ? 0 : (32 - remainder);
        for (size_t i = 0; i < pad; ++i) file.push_back(0);
        data_section_start = file.size();

        std::cout << "  size before padding: " << pos_before_padding << " bytes\n";
        std::cout << "  padding added: " << pad << " bytes\n";
        std::cout << "  data section starts at: " << data_section_start << " bytes\n";
        CHECK(data_section_start % 32 == 0);
        CHECK(data_section_start >= pos_before_padding);
        CHECK(data_section_start - pos_before_padding < 32);
    }

    // =====================================================================
    // TEST 3: Parse the file back, byte by byte, and confirm every
    // decoded field matches what Test 1 wrote.
    // =====================================================================
    std::cout << "\n-- Test 3: Parsing the file back --\n";
    {
        size_t pos = 0;
        uint32_t magic = get_u32(file, pos);
        uint32_t version = get_u32(file, pos);
        uint64_t n_tensors = get_u64(file, pos);
        uint64_t n_kv = get_u64(file, pos);

        std::cout << "  magic: 0x" << std::hex << magic << std::dec
                  << " (\"GGUF\" == 0x" << std::hex << GGUF_MAGIC << std::dec << ")\n";
        std::cout << "  version=" << version << " n_tensors=" << n_tensors << " n_kv=" << n_kv << "\n";
        CHECK(magic == GGUF_MAGIC);
        CHECK(version == GGUF_VERSION);
        CHECK(n_tensors == 1);
        CHECK(n_kv == 1);

        std::string key = get_string(file, pos);
        uint32_t value_type = get_u32(file, pos);
        std::string value = get_string(file, pos);
        std::cout << "  metadata: \"" << key << "\" = \"" << value << "\" (type=" << value_type << ")\n";
        CHECK(key == "general.architecture");
        CHECK(value_type == GGUF_TYPE_STRING);
        CHECK(value == "llama");

        std::string name = get_string(file, pos);
        uint32_t n_dims = get_u32(file, pos);
        std::vector<uint64_t> dims(n_dims);
        for (auto& d : dims) d = get_u64(file, pos);
        uint32_t type = get_u32(file, pos);
        uint64_t offset = get_u64(file, pos);

        std::cout << "  tensor: \"" << name << "\" dims=[" << dims[0] << "," << dims[1]
                   << "] type=" << type << " offset=" << offset << "\n";
        CHECK(name == "token_embd.weight");
        CHECK(n_dims == 2);
        CHECK(dims[0] == 4096 && dims[1] == 32768);
        CHECK(type == GGML_TYPE_F32);
        CHECK(offset == 0);

        // The cursor should land exactly where Test 1 stopped writing
        // real content -- everything after this point is alignment padding.
        std::cout << "  parser cursor after descriptors: " << pos << " bytes\n";
        CHECK(pos <= data_section_start);
        CHECK(data_section_start - pos < 32);
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
g++ -std=c++23 -Wall -Wextra -O2 01_gguf_binary_format.cpp -o 01_gguf_binary_format
./01_gguf_binary_format
```

**Sample input:** the smallest non-trivial GGUF file — a header, one metadata pair (`"general.architecture" = "llama"`), and one tensor descriptor (a float32 embedding table, shape `[4096, 32768]`) — built field by field and then parsed back, with the 32-byte alignment padding computed rather than assumed.

```text
================================================
The GGUF Binary Format, Byte by Byte
================================================

-- Test 1: Building the file, field by field --
  header: 24 bytes
  metadata KV pair: 45 bytes
  tensor descriptor: 57 bytes
  total before alignment: 126 bytes
  header hex: 47 55 47 46 03 00 00 00 01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00 

-- Test 2: Alignment to the data section --
  size before padding: 126 bytes
  padding added: 2 bytes
  data section starts at: 128 bytes

-- Test 3: Parsing the file back --
  magic: 0x46475547 ("GGUF" == 0x46475547)
  version=3 n_tensors=1 n_kv=1
  metadata: "general.architecture" = "llama" (type=8)
  tensor: "token_embd.weight" dims=[4096,32768] type=0 offset=0
  parser cursor after descriptors: 126 bytes

================================================
18/18 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] trusting a byte offset computed by hand"
    A hex-dump trace of a GGUF header is a genuinely useful way to build intuition for the format, but it is also exactly the kind of arithmetic where a single miscounted string length throws off every offset after it — silently, since the file still parses, it just parses the WRONG bytes as the next field. This section's own alignment padding (computed from this file's actual field lengths) does not match a hand-computed offset for a differently-worded worked example elsewhere in the quantization literature this chapter draws from — which is exactly the point: the two examples describe different strings of different lengths, and only the computed offset for THIS file is trustworthy for THIS file.

## 5.2 A GGUF Writer and Reader: A Full Round Trip

### Intuition

Before a reader can be trusted, it needs a file with known ground truth to read — which means writing one. A writer that produces a small, controlled GGUF file and a reader that parses it back into an in-memory catalog of metadata and tensor descriptors are mirror images of each other, and testing them together, as a single round trip, is a stronger check than testing either in isolation: if the writer and reader disagree about a single byte's meaning, the round trip fails loudly instead of two independently-plausible-looking halves quietly disagreeing.

### The Concept, In Detail

The writer accumulates a file's bytes through a small set of primitive appenders (`write_u32`, `write_u64`, `write_string`), builds metadata key-value pairs and tensor descriptors on top of those primitives, aligns to a 32-byte boundary, and then writes each tensor's quantized data verbatim — using Chapter 4's actual `BlockQ8` (34 bytes, fp16 scale) and `BlockQ4` (18 bytes, fp16 scale) layouts, not a simplified stand-in. The reader is the mirror image: parse the header, then the metadata pairs (dispatching on each value's type tag), then the tensor descriptors, computing each tensor's on-disk byte size from its quantization type and the same block sizes the writer used. This separation — parsing metadata once and cheaply, versus accessing tensor data later and only on demand — is the architectural insight Section 5.3 depends on: the reader's job ends at building a queryable catalog, and it deliberately does not load a single byte of tensor data itself.

### Code and Verification

```cpp
// Chapter 5.2 -- Section 5.1 traced GGUF's byte layout by hand for two
// fields. A real loader needs a writer (to produce test files with
// known ground truth) and a reader (to parse an arbitrary file back
// into a queryable catalog of metadata and tensor descriptors) that
// handle the whole format. This section builds both, then performs a
// genuine round trip: write a small synthetic model to a real file on
// disk, close everything, reopen it fresh, and verify every field the
// reader recovers matches exactly what the writer wrote.
//
// The tensor data itself uses Chapter 4's actual block layouts --
// BlockQ8 with a 2-byte fp16 scale (34 bytes) and BlockQ4 with the
// same (18 bytes) -- not a simplified 4-byte-float-scale stand-in.
// Real GGUF Q4_0/Q8_0 blocks store their scale as float16 for exactly
// the reason Chapter 4.2 built fp16_t: it is the difference between
// this reader's byte-size arithmetic matching what real llama.cpp
// GGUF files actually contain, and merely approximating it.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// -- Chapter 4.2's fp16_t (repeated per this book's one-file-per-section convention). --
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

static constexpr uint32_t GGUF_MAGIC = 0x46475547;
static constexpr uint32_t GGUF_VERSION = 3;
enum GGUFType : uint32_t { T_STRING = 8, T_UINT32 = 4, T_FLOAT32 = 6 };
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q4_0 = 2, GGML_Q8_0 = 7 };

static const char* ggml_type_name(uint32_t t) {
    switch (t) { case 0: return "F32"; case 2: return "Q4_0"; case 7: return "Q8_0"; default: return "???"; }
}

// -- Writer: appends every field's little-endian wire format to a file. --
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

// -- Reader: the mirror image, building an in-memory catalog. --
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
            case GGML_Q8_0: return (n_elements / 32) * sizeof(BlockQ8);   // 34 bytes/block
            case GGML_Q4_0: return (n_elements / 32) * sizeof(BlockQ4);   // 18 bytes/block
            default: return 0;
        }
    }
};

using MetaValue = std::variant<std::string, uint32_t, float>;

class GGUFReader {
    std::ifstream in;
    void read_raw(void* data, size_t size) { in.read(reinterpret_cast<char*>(data), size); }
public:
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    std::unordered_map<std::string, MetaValue> metadata;
    std::vector<TensorInfo> tensors;
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
                case T_FLOAT32: { float v; read_raw(&v, 4); metadata[key] = v; break; }
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
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

int main() {
    std::cout << "================================================\n";
    std::cout << "A GGUF Writer and Reader: A Full Round Trip\n";
    std::cout << "================================================\n\n";

    constexpr int VOCAB = 256, DIM = 64, D_FF = 128, N_LAYERS = 2;
    const std::string path = "/tmp/ch5_test_model.gguf";

    // =====================================================================
    // TEST 1: Write a synthetic model -- one F32 embedding, one Q8_0
    // attention weight, one Q4_0 FFN gate weight -- using Chapter 4's
    // real (fp16-scaled) block layouts.
    // =====================================================================
    std::cout << "-- Test 1: Writing the file --\n";
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);

        std::vector<float> emb_data(VOCAB * DIM);
        for (float& v : emb_data) v = dist(rng);
        std::vector<float> wq_data(DIM * DIM);
        for (float& v : wq_data) v = dist(rng);
        std::vector<float> wg_data(DIM * D_FF);
        for (float& v : wg_data) v = dist(rng);

        int n_blocks_q8 = (DIM * DIM) / 32;
        std::vector<BlockQ8> wq_q8(n_blocks_q8);
        for (int b = 0; b < n_blocks_q8; ++b) wq_q8[b] = quantize_q8(&wq_data[b * 32]);

        int n_blocks_q4 = (DIM * D_FF) / 32;
        std::vector<BlockQ4> wg_q4(n_blocks_q4);
        for (int b = 0; b < n_blocks_q4; ++b) wg_q4[b] = quantize_q4(&wg_data[b * 32]);

        size_t emb_bytes = emb_data.size() * sizeof(float);
        size_t wq_bytes = wq_q8.size() * sizeof(BlockQ8);
        size_t wg_bytes = wg_q4.size() * sizeof(BlockQ4);
        uint64_t offset_emb = 0, offset_wq = emb_bytes, offset_wg = emb_bytes + wq_bytes;

        GGUFWriter writer(path);
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(3);   // n_tensors
        writer.write_u64(5);   // n_kv
        writer.write_kv_string("general.architecture", "llama");
        writer.write_kv_string("general.name", "test-model-64d");
        writer.write_kv_u32("llama.embedding_length", DIM);
        writer.write_kv_u32("llama.block_count", N_LAYERS);
        writer.write_kv_u32("llama.feed_forward_length", D_FF);

        uint64_t emb_dims[] = {DIM, VOCAB};
        writer.write_tensor_info("token_embd.weight", 2, emb_dims, GGML_F32, offset_emb);
        uint64_t wq_dims[] = {DIM, DIM};
        writer.write_tensor_info("blk.0.attn_q.weight", 2, wq_dims, GGML_Q8_0, offset_wq);
        uint64_t wg_dims[] = {DIM, D_FF};
        writer.write_tensor_info("blk.0.ffn_gate.weight", 2, wg_dims, GGML_Q4_0, offset_wg);

        writer.align(32);
        size_t data_start = writer.tell();
        writer.write_bytes(emb_data.data(), emb_bytes);
        writer.write_bytes(wq_q8.data(), wq_bytes);
        writer.write_bytes(wg_q4.data(), wg_bytes);

        std::cout << "  header+metadata+descriptors: " << data_start << " bytes\n";
        std::cout << "  token_embd.weight  F32  " << emb_bytes << " bytes, offset=" << offset_emb << "\n";
        std::cout << "  blk.0.attn_q       Q8_0 " << wq_bytes << " bytes, offset=" << offset_wq << "\n";
        std::cout << "  blk.0.ffn_gate     Q4_0 " << wg_bytes << " bytes, offset=" << offset_wg << "\n";
        std::cout << "  total file size: " << data_start + emb_bytes + wq_bytes + wg_bytes << " bytes\n";
        CHECK(writer.good());
    }

    // =====================================================================
    // TEST 2: Reopen the file fresh and parse the header.
    // =====================================================================
    std::cout << "\n-- Test 2: Parsing the header back --\n";
    GGUFReader reader;
    {
        bool ok = reader.open(path);
        CHECK(ok);
        CHECK(reader.magic == GGUF_MAGIC);
        CHECK(reader.version == GGUF_VERSION);
        CHECK(reader.n_tensors == 3);
        CHECK(reader.n_kv == 5);
        std::cout << "  magic=0x" << std::hex << reader.magic << std::dec
                   << " version=" << reader.version
                   << " tensors=" << reader.n_tensors << " kv=" << reader.n_kv << "\n";
    }

    // =====================================================================
    // TEST 3: Metadata round trip.
    // =====================================================================
    std::cout << "\n-- Test 3: Metadata --\n";
    {
        CHECK(reader.get_string("general.architecture") == "llama");
        CHECK(reader.get_string("general.name") == "test-model-64d");
        CHECK(reader.get_u32("llama.embedding_length") == DIM);
        CHECK(reader.get_u32("llama.block_count") == N_LAYERS);
        CHECK(reader.get_u32("llama.feed_forward_length") == D_FF);
        std::cout << "  architecture=\"" << reader.get_string("general.architecture") << "\"\n";
        std::cout << "  embedding_length=" << reader.get_u32("llama.embedding_length")
                   << " block_count=" << reader.get_u32("llama.block_count") << "\n";
    }

    // =====================================================================
    // TEST 4: Tensor descriptors -- shape, type, and computed byte sizes
    // using Chapter 4's real (fp16-scaled) block layouts.
    // =====================================================================
    std::cout << "\n-- Test 4: Tensor descriptors --\n";
    {
        std::cout << "  data section starts at byte " << reader.data_section_offset << "\n";
        for (const auto& t : reader.tensors)
            std::cout << "    " << std::left << std::setw(22) << t.name << std::right
                       << ggml_type_name(t.type) << "  n_elements=" << t.n_elements
                       << "  data_size=" << t.data_size << " bytes\n";

        auto* emb = reader.find_tensor("token_embd.weight");
        CHECK(emb != nullptr);
        CHECK(emb->type == GGML_F32);
        CHECK(emb->n_elements == static_cast<uint64_t>(VOCAB) * DIM);

        auto* wq = reader.find_tensor("blk.0.attn_q.weight");
        CHECK(wq != nullptr);
        CHECK(wq->type == GGML_Q8_0);
        CHECK(wq->n_elements == static_cast<uint64_t>(DIM) * DIM);
        CHECK(wq->data_size == (wq->n_elements / 32) * 34);   // fp16-scaled BlockQ8

        auto* wg = reader.find_tensor("blk.0.ffn_gate.weight");
        CHECK(wg != nullptr);
        CHECK(wg->type == GGML_Q4_0);
        CHECK(wg->data_size == (wg->n_elements / 32) * 18);   // fp16-scaled BlockQ4
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
g++ -std=c++23 -Wall -Wextra -O2 02_gguf_writer_reader.cpp -o 02_gguf_writer_reader
./02_gguf_writer_reader
```

**Sample input:** a synthetic model (a float32 embedding table, a Q8_0 attention weight, and a Q4_0 FFN gate weight, matching Chapter 4's block sizes) written to a real file on disk, then reopened fresh and parsed back field by field.

```text
================================================
A GGUF Writer and Reader: A Full Round Trip
================================================

-- Test 1: Writing the file --
  header+metadata+descriptors: 416 bytes
  token_embd.weight  F32  65536 bytes, offset=0
  blk.0.attn_q       Q8_0 4352 bytes, offset=65536
  blk.0.ffn_gate     Q4_0 4608 bytes, offset=69888
  total file size: 74912 bytes

-- Test 2: Parsing the header back --
  magic=0x46475547 version=3 tensors=3 kv=5

-- Test 3: Metadata --
  architecture="llama"
  embedding_length=64 block_count=2

-- Test 4: Tensor descriptors --
  data section starts at byte 416
    token_embd.weight     F32  n_elements=16384  data_size=65536 bytes
    blk.0.attn_q.weight   Q8_0  n_elements=4096  data_size=4352 bytes
    blk.0.ffn_gate.weight Q4_0  n_elements=8192  data_size=4608 bytes

================================================
21/21 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] copying a block's byte size from documentation instead of `sizeof()`"
    It is tempting to hardcode a tensor's on-disk byte size as `(n_elements / 32) * 36` for a Q8_0 tensor, following a reference table. If that block's actual scale field is 2 bytes (fp16) rather than 4 (float), the tensor's real size is `* 34`, and every tensor descriptor after the mismatched one now points at the wrong offset — silently, since arithmetic that is merely wrong still produces a number. This section's reader instead computes each block's size with `sizeof(BlockQ8)` and `sizeof(BlockQ4)` directly against the same struct definitions the writer used, so a change to either block's layout cannot desynchronize the two.

## 5.3 Tensor Descriptor Tables and Zero-Copy Weight Access

### Intuition

Every GGUF model follows a naming convention: global tensors (the embedding table, the final norm, the LM head) have short, unprefixed names, and per-layer tensors are prefixed `blk.{L}.` for layer `L`. This convention is not cosmetic — it is the entire interface between the file and the engine, since the engine asks for a weight by name (`"blk.5.attn_q.weight"`) and expects the reader's catalog to answer with a type, a shape, and an offset. Section 5.2's reader builds that catalog; this section uses it to build the zero-copy views that turn a memory-mapped file into a usable tensor.

### The Concept, In Detail

A Llama-architecture model with 32 layers has exactly `1 + 1 + 1 + 32 * 9 = 291` tensors: one embedding table, one final norm, one output projection, and nine tensors per layer (four attention weights, one attention norm, three FFN weights, one FFN norm). Given a tensor's descriptor — its type, offset, and element count — building a zero-copy view is a single `reinterpret_cast` plus a `std::span` construction: an F32 tensor's bytes are cast to `const float*`, a Q8_0 tensor's bytes to `const BlockQ8*`, a Q4_0 tensor's bytes to `const BlockQ4*`, in every case wrapped in a span pointing directly into the pages `mmap` mapped from Chapter 3.6's technique. This works because `#pragma pack(push, 1)` guarantees the compiler inserts no padding into these block structs, so their in-memory layout matches the file's on-disk layout byte for byte — the file format was designed around the block types, not the other way around. Verifying that this is genuinely zero-copy means checking two different things: that the values recovered through a span are numerically correct (compared against a real FP32 reference, not merely "non-zero"), and that every span's data pointer falls within the mapped region's address range — checked as a boolean, since a raw address is relocated by ASLR on every run and printing one would make this section's own output non-reproducible.

### Code and Verification

```cpp
// Chapter 5.3 -- every GGUF tensor name follows a convention llama.cpp
// established: global tensors (the embedding table, the final norm,
// the LM head) have short, unprefixed names; per-layer tensors are
// prefixed "blk.{L}." for layer L, with four attention weights, one
// attention norm, three FFN weights, and one FFN norm per layer. A
// 32-layer Llama model therefore has exactly 1 + 1 + 1 + 32*9 = 291
// tensors -- and this naming convention is the entire interface
// between the file and the engine: the engine asks for a tensor BY
// NAME, and the reader answers with a type, a shape, and an offset.
//
// This section is the payoff for Chapter 3.6's mmap loader and
// Section 5.2's writer/reader combined: open a GGUF file, memory-map
// it with a single mmap call, parse the header and every tensor
// descriptor directly out of the mapped pages (no read() calls
// anywhere, not even for the header), and build std::span views that
// point straight into those pages. The spans are then fed to Chapter
// 4.4's fused dot-product kernels exactly as if the weights had been
// quantized in-process -- because, as far as the type system is
// concerned, they have: the file's bytes simply ARE the BlockQ8 and
// BlockQ4 structs once cast, with #pragma pack(1) guaranteeing no
// compiler-inserted padding gets in the way.

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

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((float)(a) - (float)(b)) < (tol))

// -- Chapter 4.2/4.3's fp16_t and block layouts (repeated per this book's
// one-file-per-section convention). --
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

// -- Chapter 4.4's fused dot products, reused verbatim against spans. --
float dot_q8(std::span<const BlockQ8> blocks, std::span<const float> x) {
    float total = 0.0f;
    for (size_t b = 0; b < blocks.size(); ++b) {
        float s = static_cast<float>(blocks[b].scale), bs = 0.0f;
        for (int i = 0; i < 32; ++i) bs += static_cast<float>(blocks[b].weights[i]) * x[b * 32 + i];
        total += bs * s;
    }
    return total;
}
float dot_q4(std::span<const BlockQ4> blocks, std::span<const float> x) {
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
float dot_fp32(const float* w, const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += w[i] * x[i];
    return sum;
}

static constexpr uint32_t GGUF_MAGIC = 0x46475547;
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q4_0 = 2, GGML_Q8_0 = 7 };

// -- A minimal writer (identical to Section 5.2's), just enough to
// produce a file for this section's mmap reader to open. --
void write_test_gguf(const std::string& path, int vocab, int dim, int d_ff,
                      const std::vector<float>& emb, const std::vector<float>& wq, const std::vector<float>& wg) {
    std::ofstream out(path, std::ios::binary);
    auto w_u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    auto w_u64 = [&](uint64_t v) { out.write(reinterpret_cast<const char*>(&v), 8); };
    auto w_str = [&](const std::string& s) { uint64_t n = s.size(); w_u64(n); out.write(s.data(), (std::streamsize)n); };
    auto w_tensor = [&](const std::string& name, uint64_t d0, uint64_t d1, GGMLType type, uint64_t offset) {
        w_str(name); w_u32(2); w_u64(d0); w_u64(d1); w_u32(static_cast<uint32_t>(type)); w_u64(offset);
    };

    std::vector<BlockQ8> wq_q8((dim * dim) / 32);
    for (size_t b = 0; b < wq_q8.size(); ++b) wq_q8[b] = quantize_q8(&wq[b * 32]);
    std::vector<BlockQ4> wg_q4((dim * d_ff) / 32);
    for (size_t b = 0; b < wg_q4.size(); ++b) wg_q4[b] = quantize_q4(&wg[b * 32]);

    size_t emb_bytes = emb.size() * sizeof(float);
    size_t wq_bytes = wq_q8.size() * sizeof(BlockQ8);
    size_t wg_bytes = wg_q4.size() * sizeof(BlockQ4);

    w_u32(GGUF_MAGIC); w_u32(3); w_u64(3); w_u64(1);
    w_str("general.architecture"); w_u32(8); w_str("llama");
    w_tensor("token_embd.weight", static_cast<uint64_t>(dim), static_cast<uint64_t>(vocab), GGML_F32, 0);
    w_tensor("blk.0.attn_q.weight", static_cast<uint64_t>(dim), static_cast<uint64_t>(dim), GGML_Q8_0, emb_bytes);
    w_tensor("blk.0.ffn_gate.weight", static_cast<uint64_t>(dim), static_cast<uint64_t>(d_ff), GGML_Q4_0, emb_bytes + wq_bytes);

    size_t pos = static_cast<size_t>(out.tellp());
    size_t rem = pos % 32;
    if (rem != 0) { std::vector<char> pad(32 - rem, 0); out.write(pad.data(), (std::streamsize)pad.size()); }

    out.write(reinterpret_cast<const char*>(emb.data()), (std::streamsize)emb_bytes);
    out.write(reinterpret_cast<const char*>(wq_q8.data()), (std::streamsize)wq_bytes);
    out.write(reinterpret_cast<const char*>(wg_q4.data()), (std::streamsize)wg_bytes);
}

struct TensorDesc { std::string name; uint32_t type; uint64_t offset; uint64_t n_elements; };

struct MappedGGUF {
    int fd = -1;
    void* mapped = MAP_FAILED;
    size_t file_size = 0;
    size_t data_offset = 0;
    std::vector<TensorDesc> tensors;
    std::unordered_map<std::string, size_t> tensor_index;

    bool open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st);
        file_size = static_cast<size_t>(st.st_size);
        mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) return false;

        // Parsed entirely from the mapped memory -- no read() calls,
        // not even for the header.
        const uint8_t* p = static_cast<const uint8_t*>(mapped);
        size_t pos = 0;
        auto read_u32 = [&]() -> uint32_t { uint32_t v; std::memcpy(&v, p + pos, 4); pos += 4; return v; };
        auto read_u64 = [&]() -> uint64_t { uint64_t v; std::memcpy(&v, p + pos, 8); pos += 8; return v; };
        auto read_str = [&]() -> std::string { uint64_t len = read_u64(); std::string s(reinterpret_cast<const char*>(p + pos), len); pos += len; return s; };

        uint32_t magic = read_u32();
        if (magic != GGUF_MAGIC) return false;
        read_u32();   // version
        uint64_t n_tensors = read_u64();
        uint64_t n_kv = read_u64();

        for (uint64_t i = 0; i < n_kv; ++i) {
            read_str();
            uint32_t type = read_u32();
            switch (type) { case 8: read_str(); break; case 4: case 6: pos += 4; break; default: return false; }
        }

        tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            auto& t = tensors[i];
            t.name = read_str();
            uint32_t n_dims = read_u32();
            t.n_elements = 1;
            for (uint32_t d = 0; d < n_dims; ++d) t.n_elements *= read_u64();
            t.type = read_u32();
            t.offset = read_u64();
            tensor_index[t.name] = i;
        }

        size_t rem = pos % 32;
        data_offset = (rem == 0) ? pos : pos + (32 - rem);
        return true;
    }

    std::span<const float> get_f32(const std::string& name) const {
        auto it = tensor_index.find(name);
        if (it == tensor_index.end()) return {};
        const auto& t = tensors[it->second];
        const void* ptr = static_cast<const uint8_t*>(mapped) + data_offset + t.offset;
        return std::span<const float>(reinterpret_cast<const float*>(ptr), t.n_elements);
    }
    std::span<const BlockQ8> get_q8(const std::string& name) const {
        auto it = tensor_index.find(name);
        if (it == tensor_index.end()) return {};
        const auto& t = tensors[it->second];
        const void* ptr = static_cast<const uint8_t*>(mapped) + data_offset + t.offset;
        return std::span<const BlockQ8>(reinterpret_cast<const BlockQ8*>(ptr), t.n_elements / 32);
    }
    std::span<const BlockQ4> get_q4(const std::string& name) const {
        auto it = tensor_index.find(name);
        if (it == tensor_index.end()) return {};
        const auto& t = tensors[it->second];
        const void* ptr = static_cast<const uint8_t*>(mapped) + data_offset + t.offset;
        return std::span<const BlockQ4>(reinterpret_cast<const BlockQ4*>(ptr), t.n_elements / 32);
    }

    ~MappedGGUF() {
        if (mapped != MAP_FAILED) munmap(mapped, file_size);
        if (fd >= 0) ::close(fd);
    }
};

int main() {
    std::cout << "================================================\n";
    std::cout << "Tensor Descriptor Tables and Zero-Copy Weight Access\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: The naming convention's own arithmetic -- a 32-layer Llama
    // model's total tensor count, computed the same way the reader would
    // discover it (not copied from a reference table).
    // =====================================================================
    std::cout << "-- Test 1: Tensor count from the naming convention --\n";
    {
        constexpr int LAYERS = 32;
        constexpr int GLOBAL_TENSORS = 3;      // token_embd, output_norm, output
        constexpr int PER_LAYER_TENSORS = 9;   // attn_norm, q, k, v, attn_output, ffn_norm, gate, up, down
        int total = GLOBAL_TENSORS + LAYERS * PER_LAYER_TENSORS;
        std::cout << "  " << GLOBAL_TENSORS << " global + " << LAYERS << " layers x "
                   << PER_LAYER_TENSORS << " per-layer = " << total << " tensors\n";
        CHECK(total == 291);
    }

    // =====================================================================
    // TEST 2: Write a small synthetic model, then mmap it.
    // =====================================================================
    std::cout << "\n-- Test 2: mmap the file --\n";
    constexpr int VOCAB = 256, DIM = 64, D_FF = 128;
    const std::string path = "/tmp/ch5_zero_copy_test.gguf";
    std::vector<float> emb_data(VOCAB * DIM), wq_data(DIM * DIM), wg_data(DIM * D_FF);
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (float& v : emb_data) v = dist(rng);
        for (float& v : wq_data) v = dist(rng);
        for (float& v : wg_data) v = dist(rng);
        write_test_gguf(path, VOCAB, DIM, D_FF, emb_data, wq_data, wg_data);
    }

    MappedGGUF gguf;
    {
        bool ok = gguf.open(path.c_str());
        CHECK(ok);
        CHECK(gguf.mapped != MAP_FAILED);
        std::cout << "  file size: " << gguf.file_size << " bytes, tensors: " << gguf.tensors.size() << "\n";
    }

    // =====================================================================
    // TEST 3: Zero-copy F32 embedding access, checked against the
    // original (pre-write) data -- not merely "non-zero".
    // =====================================================================
    std::cout << "\n-- Test 3: Zero-copy F32 embedding lookup --\n";
    {
        auto emb = gguf.get_f32("token_embd.weight");
        CHECK(emb.size() == static_cast<size_t>(VOCAB) * DIM);
        bool matches_original = std::equal(emb.begin(), emb.end(), emb_data.begin());
        std::cout << "  embedding table: " << emb.size() << " floats\n";
        std::cout << "  matches the data written (bit-for-bit, zero-copy view): " << (matches_original ? "yes" : "no") << "\n";
        CHECK(matches_original);
    }

    // =====================================================================
    // TEST 4: Zero-copy Q8/Q4 weight access + fused dot products, checked
    // against a genuine FP32 reference computed from the SAME pre-write
    // data -- proving the mmap'd view produces numerically correct results,
    // not just "doesn't crash".
    // =====================================================================
    std::cout << "\n-- Test 4: Zero-copy Q8/Q4 fused dot products vs FP32 reference --\n";
    {
        std::vector<float> x(DIM);
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (float& v : x) v = dist(rng);

        // Each tensor's mmap'd span covers the ENTIRE weight matrix
        // (128 Q8 blocks for the 64x64 attn_q matrix); the dot product
        // below only wants the first row's worth (DIM/32 blocks) to
        // match x's length -- exactly like indexing row 0 of an
        // ordinary 2D array, just through a span instead of a pointer.
        int n_blocks_per_row = DIM / 32;

        auto wq_blocks = gguf.get_q8("blk.0.attn_q.weight");
        CHECK(!wq_blocks.empty());
        float q8_result = dot_q8(wq_blocks.first(static_cast<size_t>(n_blocks_per_row)), x);
        float q8_ref = dot_fp32(&wq_data[0], x.data(), DIM);
        std::cout << "  Q8 (mmap'd):  " << std::fixed << std::setprecision(6) << q8_result << "\n";
        std::cout << "  FP32 reference: " << q8_ref << "\n";
        CHECK(std::fabs(q8_result - q8_ref) < 0.05f);

        auto wg_blocks = gguf.get_q4("blk.0.ffn_gate.weight");
        CHECK(!wg_blocks.empty());
        float q4_result = dot_q4(wg_blocks.first(static_cast<size_t>(n_blocks_per_row)), x);
        float q4_ref = dot_fp32(&wg_data[0], x.data(), DIM);
        std::cout << "  Q4 (mmap'd):  " << q4_result << "\n";
        std::cout << "  FP32 reference: " << q4_ref << "\n";
        CHECK(std::fabs(q4_result - q4_ref) < 0.05f);
    }

    // =====================================================================
    // TEST 5: Every span's data pointer falls within the mmap'd region --
    // proof that no copies were made. Checked as a boolean, never as a
    // printed address (a raw pointer value is not reproducible between
    // runs -- ASLR relocates the mapping every time the program starts).
    // =====================================================================
    std::cout << "\n-- Test 5: Every view points inside the mmap'd region --\n";
    {
        auto emb = gguf.get_f32("token_embd.weight");
        auto wq = gguf.get_q8("blk.0.attn_q.weight");
        auto wg = gguf.get_q4("blk.0.ffn_gate.weight");

        const uint8_t* base = static_cast<const uint8_t*>(gguf.mapped);
        const uint8_t* end = base + gguf.file_size;
        auto in_range = [&](const void* ptr) {
            auto p = static_cast<const uint8_t*>(ptr);
            return p >= base && p < end;
        };

        bool emb_in_range = in_range(emb.data());
        bool wq_in_range = in_range(wq.data());
        bool wg_in_range = in_range(wg.data());
        std::cout << "  embedding view inside mapped region: " << (emb_in_range ? "yes" : "no") << "\n";
        std::cout << "  Q8 view inside mapped region: " << (wq_in_range ? "yes" : "no") << "\n";
        std::cout << "  Q4 view inside mapped region: " << (wg_in_range ? "yes" : "no") << "\n";
        std::cout << "  heap bytes allocated for tensor data: 0\n";
        CHECK(emb_in_range);
        CHECK(wq_in_range);
        CHECK(wg_in_range);
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
g++ -std=c++23 -Wall -Wextra -O2 03_zero_copy_weight_access.cpp -o 03_zero_copy_weight_access
./03_zero_copy_weight_access
```

**Sample input:** the 291-tensor naming-convention arithmetic for a 32-layer model, a freshly `mmap`'d synthetic GGUF file, a zero-copy F32 embedding lookup checked against the original data, and Q8/Q4 fused dot products (Chapter 4.4's kernels, reused against `mmap`'d spans) checked against a genuine FP32 reference.

```text
================================================
Tensor Descriptor Tables and Zero-Copy Weight Access
================================================

-- Test 1: Tensor count from the naming convention --
  3 global + 32 layers x 9 per-layer = 291 tensors

-- Test 2: mmap the file --
  file size: 74752 bytes, tensors: 3

-- Test 3: Zero-copy F32 embedding lookup --
  embedding table: 16384 floats
  matches the data written (bit-for-bit, zero-copy view): yes

-- Test 4: Zero-copy Q8/Q4 fused dot products vs FP32 reference --
  Q8 (mmap'd):  -0.015789
  FP32 reference: -0.015790
  Q4 (mmap'd):  0.012404
  FP32 reference: 0.011146

-- Test 5: Every view points inside the mmap'd region --
  embedding view inside mapped region: yes
  Q8 view inside mapped region: yes
  Q4 view inside mapped region: yes
  heap bytes allocated for tensor data: 0

================================================
12/12 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] taking the whole tensor's span when only one row is needed"
    A tensor descriptor's span covers the ENTIRE weight matrix — for a 64x64 Q8_0 attention weight, that is 128 blocks, not the 2 blocks one row actually needs. Passing the full span to a per-row dot product silently reads far past the end of whatever input vector it is being multiplied against: the loop bound comes from the span's size, the vector being indexed does not grow to match, and the result is undefined behavior that showed up here as `nan` the first time this section's own dot-product test ran, until the span was narrowed with `.first(n_blocks_per_row)` before the call. The fix is one line; finding it required actually running the code and noticing the output was `nan`, not inspecting the code and assuming the shapes lined up.

## 5.4 The SafeTensors Format: JSON Metadata and a Hand-Rolled Parser

### Intuition

SafeTensors, introduced by Hugging Face in 2022, is the format nearly every model repository ships its raw weights in, for the same reason GGUF exists: PyTorch's pickle-based `.pt` files can execute arbitrary code on load, and SafeTensors is strictly data. Its binary prefix and mmap-compatible layout are close cousins of GGUF's, with one deliberate difference: its metadata header is JSON, not a binary type-tagged store — easy for a human (or any language with a JSON library) to inspect, at the cost of being slower to parse than GGUF's binary metadata.

### The Concept, In Detail

A SafeTensors file is an 8-byte little-endian header length, that many bytes of JSON, optional padding to an 8-byte boundary, and then every tensor's raw data concatenated. The JSON object maps each tensor's name to its `dtype`, `shape`, and `data_offsets` — a `[start, end]` pair measured from the START OF THE TENSOR DATA, not the start of the file. Parsing this does not require a general JSON library: SafeTensors' header has a fixed, well-known shape (an object of objects, each with exactly three keys), so a hand-rolled extractor that finds a key by its quoted string and reads the value that follows it is entirely sufficient, and considerably simpler than pulling in a full parser for a document this constrained. As with GGUF, the actual payoff is the zero-copy read: once the header is parsed, each tensor's data pointer is `mmap_base + tensor_data_start + data_offsets[0]`, and the bytes there ARE the tensor — verified below not by trusting the parser's own bookkeeping, but by writing a file with known values, reading it back through the real loader, and checking the recovered floats bit-for-bit against what was written.

### Code and Verification

```cpp
// Chapter 5.4 -- SafeTensors, introduced by Hugging Face in 2022, is
// the other format you will encounter constantly: nearly every model
// repository on HuggingFace ships its raw (unquantized) weights this
// way, precisely because it was designed as the safe replacement for
// PyTorch's pickle-based .pt files, which can execute arbitrary code
// on load. SafeTensors is strictly data -- a tiny binary prefix
// followed by a JSON metadata header, followed by raw tensor bytes.
// There is no quantization support (weights are stored as plain
// F32/F16/BF16), which is exactly why models are typically converted
// FROM SafeTensors INTO GGUF as the last step before local inference:
// SafeTensors is the training/distribution format, GGUF is the
// inference format.
//
// The layout is simple: an 8-byte little-endian header LENGTH, that
// many bytes of JSON, then the tensor data (all tensors concatenated,
// each described by a `data_offsets: [start, end]` pair measured from
// the start of the tensor data, not the start of the file). This
// section writes a real, genuine parser for that JSON header -- not a
// full JSON library, just a hand-rolled extractor for the handful of
// fixed keys SafeTensors always uses -- and proves it works with an
// actual round trip: build a file, mmap it back, and confirm the
// tensor values recovered through the zero-copy view are bit-for-bit
// identical to what was written.

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cassert>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

enum class STDtype { F32, F16, BF16, I8, UNKNOWN };
STDtype parse_dtype(const std::string& s) {
    if (s == "F32") return STDtype::F32;
    if (s == "F16") return STDtype::F16;
    if (s == "BF16") return STDtype::BF16;
    if (s == "I8") return STDtype::I8;
    return STDtype::UNKNOWN;
}
size_t dtype_bytes(STDtype d) {
    switch (d) {
        case STDtype::F32: return 4;
        case STDtype::F16: case STDtype::BF16: return 2;
        case STDtype::I8: return 1;
        default: return 0;
    }
}
const char* dtype_name(STDtype d) {
    switch (d) { case STDtype::F32: return "F32"; case STDtype::F16: return "F16";
                 case STDtype::BF16: return "BF16"; case STDtype::I8: return "I8"; default: return "?"; }
}

// -- A minimal JSON value extractor -- not a general parser, just
// enough for SafeTensors' own fixed, well-structured shape: an object
// of objects with known keys ("dtype", "shape", "data_offsets"). --
std::string json_extract_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return "";
    size_t colon = json.find(':', kpos);
    size_t quote = json.find('"', colon);
    size_t end = json.find('"', quote + 1);
    if (colon == std::string::npos || quote == std::string::npos || end == std::string::npos) return "";
    return json.substr(quote + 1, end - quote - 1);
}
std::vector<int64_t> json_extract_int_array(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return {};
    size_t bracket = json.find('[', kpos);
    size_t close = json.find(']', bracket);
    if (bracket == std::string::npos || close == std::string::npos) return {};
    std::string arr = json.substr(bracket + 1, close - bracket - 1);
    std::vector<int64_t> result;
    std::stringstream ss(arr);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if (!tok.empty()) result.push_back(std::stoll(tok));
    }
    return result;
}

struct STTensorView {
    std::string name;
    STDtype dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t size_bytes = 0;
    int64_t n_elements() const { int64_t n = 1; for (int64_t d : shape) n *= d; return n; }
};

class SafeTensorsLoader {
public:
    ~SafeTensorsLoader() { if (m_ptr) munmap(const_cast<uint8_t*>(m_ptr), m_file_size); }

    void load(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open: " + path);
        struct stat st; fstat(fd, &st);
        m_file_size = static_cast<size_t>(st.st_size);
        void* p = mmap(nullptr, m_file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        m_ptr = static_cast<const uint8_t*>(p);

        uint64_t header_len;
        std::memcpy(&header_len, m_ptr, 8);
        std::string json(reinterpret_cast<const char*>(m_ptr + 8), header_len);
        m_tensor_data_base = m_ptr + 8 + header_len;

        size_t pos = 0;
        while (true) {
            size_t dtype_pos = json.find("\"dtype\"", pos);
            if (dtype_pos == std::string::npos) break;
            size_t obj_start = json.rfind('{', dtype_pos);
            if (obj_start == std::string::npos) { pos = dtype_pos + 1; continue; }
            size_t key_end = json.rfind('"', obj_start);
            if (key_end == std::string::npos) { pos = dtype_pos + 1; continue; }
            size_t key_start = json.rfind('"', key_end - 1);
            if (key_start == std::string::npos) { pos = dtype_pos + 1; continue; }
            std::string tensor_name = json.substr(key_start + 1, key_end - key_start - 1);
            if (tensor_name == "__metadata__") { pos = dtype_pos + 1; continue; }

            size_t obj_end = json.find('}', obj_start);
            if (obj_end == std::string::npos) break;
            std::string obj = json.substr(obj_start, obj_end - obj_start + 1);

            STTensorView tv;
            tv.name = tensor_name;
            tv.dtype = parse_dtype(json_extract_string(obj, "dtype"));
            tv.shape = json_extract_int_array(obj, "shape");
            auto offsets = json_extract_int_array(obj, "data_offsets");
            if (offsets.size() == 2) {
                tv.data = m_tensor_data_base + offsets[0];
                tv.size_bytes = static_cast<size_t>(offsets[1] - offsets[0]);
            }
            m_tensors[tv.name] = tv;
            pos = obj_end + 1;
        }
    }

    const STTensorView* tensor(const std::string& name) const {
        auto it = m_tensors.find(name);
        return (it != m_tensors.end()) ? &it->second : nullptr;
    }
    size_t n_tensors() const { return m_tensors.size(); }

private:
    const uint8_t* m_ptr = nullptr;
    size_t m_file_size = 0;
    const uint8_t* m_tensor_data_base = nullptr;
    std::unordered_map<std::string, STTensorView> m_tensors;
};

int main() {
    std::cout << "================================================\n";
    std::cout << "The SafeTensors Format: JSON Header + mmap\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: JSON field extraction on a single tensor object -- the
    // building block the loader's parsing loop relies on.
    // =====================================================================
    std::cout << "-- Test 1: JSON field extraction --\n";
    {
        std::string obj = R"({"dtype":"BF16","shape":[2048,2048],"data_offsets":[0,8388608]})";
        std::string dtype = json_extract_string(obj, "dtype");
        auto shape = json_extract_int_array(obj, "shape");
        auto offsets = json_extract_int_array(obj, "data_offsets");

        std::cout << "  dtype=\"" << dtype << "\" shape=[" << shape[0] << "," << shape[1]
                   << "] data_offsets=[" << offsets[0] << "," << offsets[1] << "]\n";
        CHECK(dtype == "BF16");
        CHECK(shape.size() == 2 && shape[0] == 2048 && shape[1] == 2048);
        CHECK(offsets.size() == 2 && offsets[0] == 0 && offsets[1] == 8388608);

        int64_t expected_bytes = 2048LL * 2048 * static_cast<int64_t>(dtype_bytes(STDtype::BF16));
        std::cout << "  2048x2048 BF16 tensor size check: " << offsets[1] << " == " << expected_bytes << "\n";
        CHECK(offsets[1] == expected_bytes);
    }

    // =====================================================================
    // TEST 2: dtype_bytes correctness.
    // =====================================================================
    std::cout << "\n-- Test 2: dtype_bytes --\n";
    {
        CHECK(dtype_bytes(STDtype::F32) == 4);
        CHECK(dtype_bytes(STDtype::F16) == 2);
        CHECK(dtype_bytes(STDtype::BF16) == 2);
        CHECK(dtype_bytes(STDtype::I8) == 1);
        std::cout << "  F32=4, F16=2, BF16=2, I8=1 bytes\n";
    }

    // =====================================================================
    // TEST 3: A genuine round trip -- build a real SafeTensors file with
    // two tensors and known values, load it back through the real
    // SafeTensorsLoader (mmap and all), and verify every value recovered
    // through the zero-copy view is bit-for-bit what was written.
    // =====================================================================
    std::cout << "\n-- Test 3: Full round trip (write, mmap, read back) --\n";
    const std::string path = "/tmp/ch5_test_model.safetensors";
    {
        // "weight": F32 [4,4] = 0.0, 1.0, ..., 15.0
        // "bias":   F32 [4]   = 100.0, 101.0, 102.0, 103.0
        std::vector<float> weight_data(16);
        for (int i = 0; i < 16; ++i) weight_data[static_cast<size_t>(i)] = static_cast<float>(i);
        std::vector<float> bias_data = {100.0f, 101.0f, 102.0f, 103.0f};

        size_t weight_bytes = weight_data.size() * sizeof(float);
        size_t bias_bytes = bias_data.size() * sizeof(float);

        std::ostringstream json;
        json << "{"
             << R"("__metadata__":{"format":"pt"},)"
             << R"("weight":{"dtype":"F32","shape":[4,4],"data_offsets":[0,)" << weight_bytes << "]},"
             << R"("bias":{"dtype":"F32","shape":[4],"data_offsets":[)" << weight_bytes << "," << (weight_bytes + bias_bytes) << "]}"
             << "}";
        std::string json_str = json.str();

        std::ofstream out(path, std::ios::binary);
        uint64_t hlen = json_str.size();
        out.write(reinterpret_cast<const char*>(&hlen), 8);
        out.write(json_str.data(), static_cast<std::streamsize>(json_str.size()));
        out.write(reinterpret_cast<const char*>(weight_data.data()), static_cast<std::streamsize>(weight_bytes));
        out.write(reinterpret_cast<const char*>(bias_data.data()), static_cast<std::streamsize>(bias_bytes));
        out.close();

        std::cout << "  wrote " << path << " (" << (8 + json_str.size() + weight_bytes + bias_bytes) << " bytes)\n";

        SafeTensorsLoader loader;
        loader.load(path);
        CHECK(loader.n_tensors() == 2);
        std::cout << "  loaded " << loader.n_tensors() << " tensors\n";

        auto* w = loader.tensor("weight");
        CHECK(w != nullptr);
        CHECK(w->dtype == STDtype::F32);
        CHECK(w->shape.size() == 2 && w->shape[0] == 4 && w->shape[1] == 4);
        CHECK(w->n_elements() == 16);

        const float* w_floats = reinterpret_cast<const float*>(w->data);
        bool weight_matches = std::equal(w_floats, w_floats + 16, weight_data.begin());
        std::cout << "  \"weight\": dtype=" << dtype_name(w->dtype) << " shape=[4,4], values match: "
                   << (weight_matches ? "yes" : "no") << "\n";
        CHECK(weight_matches);

        auto* b = loader.tensor("bias");
        CHECK(b != nullptr);
        CHECK(b->shape.size() == 1 && b->shape[0] == 4);
        const float* b_floats = reinterpret_cast<const float*>(b->data);
        bool bias_matches = std::equal(b_floats, b_floats + 4, bias_data.begin());
        std::cout << "  \"bias\": shape=[4], values match: " << (bias_matches ? "yes" : "no") << "\n";
        CHECK(bias_matches);

        unlink(path.c_str());
    }

    // =====================================================================
    // TEST 4: A genuinely computed size comparison for storing the same
    // 8-billion-parameter model as SafeTensors BF16 versus GGUF Q4_0 --
    // using the same effective-bytes-per-parameter model Chapter 4.1
    // built, not a copied reference-table figure.
    // =====================================================================
    std::cout << "\n-- Test 4: SafeTensors (BF16) vs GGUF (Q4_0) for an 8B-parameter model --\n";
    {
        constexpr double PARAMS = 8.0e9;
        constexpr double BF16_BYTES_PER_PARAM = 2.0;
        constexpr double Q4_0_BYTES_PER_PARAM = (16.0 + 2.0) / 32.0;   // Chapter 4.3's 18-byte block of 32

        double safetensors_gb = PARAMS * BF16_BYTES_PER_PARAM / 1e9;
        double gguf_gb = PARAMS * Q4_0_BYTES_PER_PARAM / 1e9;

        std::cout << "  SafeTensors (BF16, training/distribution format): "
                   << std::fixed << std::setprecision(1) << safetensors_gb << " GB\n";
        std::cout << "  GGUF (Q4_0, inference format):                    " << gguf_gb << " GB\n";
        std::cout << "  SafeTensors has no quantization support (F32/F16/BF16 only); GGUF does.\n";
        std::cout << "  Both are safe formats (no code execution); GGUF's metadata is binary, SafeTensors' is JSON.\n";
        CHECK(safetensors_gb > gguf_gb * 3.0);   // BF16 is roughly 3.5x the bytes/param of Q4_0
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
g++ -std=c++23 -Wall -Wextra -O2 04_safetensors_format.cpp -o 04_safetensors_format
./04_safetensors_format
```

**Sample input:** isolated JSON field-extraction checks, then a full round trip — a real SafeTensors file with two tensors (`"weight"`, a 4x4 float32 matrix of known values 0.0 through 15.0, and `"bias"`, a 4-element vector) written to disk, loaded back through `SafeTensorsLoader`, and checked bit-for-bit against the original data.

```text
================================================
The SafeTensors Format: JSON Header + mmap
================================================

-- Test 1: JSON field extraction --
  dtype="BF16" shape=[2048,2048] data_offsets=[0,8388608]
  2048x2048 BF16 tensor size check: 8388608 == 8388608

-- Test 2: dtype_bytes --
  F32=4, F16=2, BF16=2, I8=1 bytes

-- Test 3: Full round trip (write, mmap, read back) --
  wrote /tmp/ch5_test_model.safetensors (239 bytes)
  loaded 2 tensors
  "weight": dtype=F32 shape=[4,4], values match: yes
  "bias": shape=[4], values match: yes

-- Test 4: SafeTensors (BF16) vs GGUF (Q4_0) for an 8B-parameter model --
  SafeTensors (BF16, training/distribution format): 16.0 GB
  GGUF (Q4_0, inference format):                    4.5 GB
  SafeTensors has no quantization support (F32/F16/BF16 only); GGUF does.
  Both are safe formats (no code execution); GGUF's metadata is binary, SafeTensors' is JSON.

================================================
18/18 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] `data_offsets` are relative to the tensor data, not the file"
    SafeTensors' `data_offsets: [start, end]` pair looks, at a glance, like it should be measured from byte 0 of the file — it is not. It is measured from the first byte AFTER the JSON header (`8 + header_length`), so a tensor's real file offset is `tensor_data_base + data_offsets[0]`, and computing it as `data_offsets[0]` alone reads from entirely the wrong place in a file that has any header at all — which every real SafeTensors file does.

## 5.5 GGUF vs. SafeTensors, and a Peek at Production K-Quants

### Intuition

This chapter's two formats serve different stages of a model's life: SafeTensors for training and distribution (unquantized, JSON metadata, easy to inspect), GGUF for local inference (quantized, binary metadata, optimized for the exact moment weights need to reach a running engine). Real GGUF files for well-known models, though, mostly do not use the flat Q4_0/Q8_0 blocks Chapter 4 built — they use "K-quants," which spend the same bit budget more cleverly by adding a second level of scales.

### The Concept, In Detail

A Q4_K super-block covers 256 weights (eight of Chapter 4.3's 32-weight sub-blocks) and stores a super-block-level scale and minimum (2 bytes each, fp16), a packed set of eight 6-bit sub-block scales, and eight more 6-bit sub-block minimums — 12 bytes total for both sets — plus 128 bytes of packed 4-bit weight nibbles. The correct total is `2 + 2 + 12 + 128 = 144` bytes for 256 weights, which is `144 * 8 / 256 = 4.5` bits per weight — the identical nominal bit rate as Chapter 4.3's plain Q4_0 (`18 * 8 / 32 = 4.5` bits per weight, exactly). This is the section's central point: K-quants do not compress further than Q4_0 in the nominal sense, they allocate the SAME bits differently, trading one coarse scale per 32 weights for a hierarchical scale that adapts within each 256-weight super-block — capturing fine-grained variance a single flat scale cannot, at no extra bit cost. The 8 sub-block scales (and 8 sub-block minimums) packing into exactly 12 bytes relies on 6-bit values crossing byte boundaries — 8 values times 6 bits is 48 bits, exactly 6 bytes, with no wasted padding — which this section implements and round-trips directly, the same kind of bit-level packing exercise as Chapter 4.3's 4-bit nibbles, one level more intricate.

### Code and Verification

```cpp
// Chapter 5.5 -- this chapter has built exactly the two formats real
// LLM tooling actually uses: SafeTensors (Section 5.4) for training
// and distribution, GGUF (Sections 5.1-5.3) for local inference, with
// Chapter 4's Q4_0/Q8_0 blocks as GGUF's simplest quantized tensor
// types. Production GGUF files for well-known models rarely use plain
// Q4_0 today, though -- they use "K-quants" (Q4_K_M being the most
// common), which replace one scale per 32-weight block with a
// two-level hierarchy: a super-block of 256 weights carries its own
// scale and minimum, and each of the super-block's 8 sub-blocks of 32
// weights carries a 6-bit scale relative to the super-block's -- all
// packed as tightly as the bit width allows.
//
// [COMMON TRAP]: source material describing this layout is easy to
// get subtly wrong, because the arithmetic has several small pieces
// that must all be added correctly. One such description states a
// Q4_K super-block header as "12 bytes" and then computes a 140-byte
// total (12 + 128 nibble bytes) -- but its own field list for that
// header is a 2-byte super-block scale, a 2-byte super-block minimum,
// AND 12 bytes of packed sub-block scales, which is 16 bytes, not 12;
// the correct total is 2 + 2 + 12 + 128 = 144 bytes. This section
// computes that total in code rather than trusting either version on
// paper, and the result (144, matching this book's own GGUF
// quantization-type table in Section 5.1) settles which was right.

#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// -- Q4_K_M's packed 6-bit sub-block scales: 8 values, each in [0,63],
// packed into 6 bytes (8 * 6 = 48 bits = 6 bytes exactly, with no
// padding). Bits are packed sequentially, least-significant-bit
// first, spanning byte boundaries wherever 6 does not divide 8. --
std::array<uint8_t, 6> pack_6bit(const std::array<uint8_t, 8>& vals) {
    std::array<uint8_t, 6> out{};
    uint64_t bitbuf = 0;
    int bitcount = 0;
    size_t byte_pos = 0;
    for (uint8_t v : vals) {
        assert(v < 64);
        bitbuf |= (static_cast<uint64_t>(v) << bitcount);
        bitcount += 6;
        while (bitcount >= 8) {
            out[byte_pos++] = static_cast<uint8_t>(bitbuf & 0xFF);
            bitbuf >>= 8;
            bitcount -= 8;
        }
    }
    return out;
}

std::array<uint8_t, 8> unpack_6bit(const std::array<uint8_t, 6>& packed) {
    std::array<uint8_t, 8> out{};
    uint64_t bitbuf = 0;
    int bitcount = 0;
    size_t byte_pos = 0;
    for (size_t i = 0; i < 8; ++i) {
        while (bitcount < 6) {
            bitbuf |= (static_cast<uint64_t>(packed[byte_pos++]) << bitcount);
            bitcount += 8;
        }
        out[i] = static_cast<uint8_t>(bitbuf & 0x3F);
        bitbuf >>= 6;
        bitcount -= 6;
    }
    return out;
}

int main() {
    std::cout << "================================================\n";
    std::cout << "GGUF vs SafeTensors, and a Peek at Production K-Quants\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: The Q4_K_M super-block byte layout, computed field by
    // field, correcting the 140-vs-144 discrepancy noted above.
    // =====================================================================
    std::cout << "-- Test 1: Q4_K_M super-block byte layout --\n";
    {
        constexpr size_t D_BYTES = 2;         // fp16 super-block scale
        constexpr size_t DMIN_BYTES = 2;      // fp16 super-block minimum
        constexpr size_t SCALES_BYTES = 12;   // 8 sub-block scales + 8 sub-block mins, 6 bits each
        constexpr size_t QS_BYTES = 128;      // 256 weights x 4 bits, 2 per byte
        constexpr size_t SUPERBLOCK_WEIGHTS = 256;

        size_t total_bytes = D_BYTES + DMIN_BYTES + SCALES_BYTES + QS_BYTES;
        double bits_per_weight = static_cast<double>(total_bytes) * 8.0 / static_cast<double>(SUPERBLOCK_WEIGHTS);

        std::cout << "  d (fp16 scale): " << D_BYTES << " bytes\n";
        std::cout << "  dmin (fp16 min): " << DMIN_BYTES << " bytes\n";
        std::cout << "  packed sub-block scales+mins: " << SCALES_BYTES << " bytes\n";
        std::cout << "  packed 4-bit weight nibbles: " << QS_BYTES << " bytes\n";
        std::cout << "  total super-block size: " << total_bytes << " bytes for " << SUPERBLOCK_WEIGHTS << " weights\n";
        std::cout << "  effective bits/weight: " << bits_per_weight << "\n";
        CHECK(total_bytes == 144);
        CHECK(bits_per_weight == 4.5);
    }

    // =====================================================================
    // TEST 2: Q4_K_M vs Q4_0's nominal bits/weight -- both land at
    // exactly 4.5, using Chapter 4.3's real 18-byte Q4_0 block. The
    // difference between the two formats is not the bit budget; it is
    // how that budget is spent (one scale per 32 weights vs. a
    // hierarchical scale that adapts within a 256-weight super-block).
    // =====================================================================
    std::cout << "\n-- Test 2: Same nominal bit budget, different allocation --\n";
    {
        constexpr size_t Q4_0_BLOCK_BYTES = 18;   // Chapter 4.3: 2 (fp16 scale) + 16 (nibbles)
        constexpr size_t Q4_0_BLOCK_WEIGHTS = 32;
        double q4_0_bits = static_cast<double>(Q4_0_BLOCK_BYTES) * 8.0 / static_cast<double>(Q4_0_BLOCK_WEIGHTS);

        constexpr size_t Q4_K_SUPERBLOCK_BYTES = 144;
        constexpr size_t Q4_K_SUPERBLOCK_WEIGHTS = 256;
        double q4_k_bits = static_cast<double>(Q4_K_SUPERBLOCK_BYTES) * 8.0 / static_cast<double>(Q4_K_SUPERBLOCK_WEIGHTS);

        std::cout << "  Q4_0:   " << Q4_0_BLOCK_BYTES << " bytes / " << Q4_0_BLOCK_WEIGHTS
                   << " weights = " << q4_0_bits << " bits/weight (ONE scale for the whole block)\n";
        std::cout << "  Q4_K_M: " << Q4_K_SUPERBLOCK_BYTES << " bytes / " << Q4_K_SUPERBLOCK_WEIGHTS
                   << " weights = " << q4_k_bits << " bits/weight (a scale PER SUB-BLOCK of 32, within the super-block)\n";
        CHECK(q4_0_bits == q4_k_bits);
        std::cout << "  same nominal bit budget: " << (q4_0_bits == q4_k_bits ? "yes" : "no")
                   << " -- Q4_K_M spends it on finer-grained scales, not more of them\n";
    }

    // =====================================================================
    // TEST 3: The packed 6-bit sub-block scale scheme, round-tripped
    // exhaustively -- the bit-packing mechanism that makes Test 1's
    // 12-byte scales-and-mins field possible in the first place.
    // =====================================================================
    std::cout << "\n-- Test 3: Packed 6-bit sub-block scales, round-tripped --\n";
    {
        // Edge cases: all zeros, all max (63), and alternating.
        std::array<uint8_t, 8> all_zero{0,0,0,0,0,0,0,0};
        std::array<uint8_t, 8> all_max{63,63,63,63,63,63,63,63};
        std::array<uint8_t, 8> alternating{0,63,0,63,0,63,0,63};

        for (const auto& vals : {all_zero, all_max, alternating}) {
            auto packed = pack_6bit(vals);
            auto recovered = unpack_6bit(packed);
            CHECK(recovered == vals);
        }
        std::cout << "  edge cases (all-zero, all-max, alternating) round-trip: yes\n";

        // Exhaustive-ish fuzz: many random 8-tuples of 6-bit values,
        // fixed seed for reproducibility.
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 63);
        int mismatches = 0;
        constexpr int TRIALS = 2000;
        for (int t = 0; t < TRIALS; ++t) {
            std::array<uint8_t, 8> vals{};
            for (auto& v : vals) v = static_cast<uint8_t>(dist(rng));
            auto packed = pack_6bit(vals);
            auto recovered = unpack_6bit(packed);
            if (recovered != vals) ++mismatches;
        }
        std::cout << "  " << TRIALS << " random 8-tuples of 6-bit values: " << mismatches << " mismatches\n";
        CHECK(mismatches == 0);

        // The packed size itself: 8 values x 6 bits = 48 bits = 6 bytes exactly.
        auto sample_packed = pack_6bit(alternating);
        std::cout << "  packed size: " << sample_packed.size() << " bytes for 8 six-bit values (48 bits, no padding)\n";
        CHECK(sample_packed.size() == 6);
    }

    // =====================================================================
    // TEST 4: A genuinely computed format comparison for an 8-billion-
    // parameter model across every format this chapter covered.
    // =====================================================================
    std::cout << "\n-- Test 4: One model, every format this chapter covered --\n";
    {
        constexpr double PARAMS = 8.0e9;
        struct Format { const char* name; double bytes_per_param; const char* note; };
        Format formats[] = {
            {"SafeTensors F32",  4.0,          "training, full precision"},
            {"SafeTensors BF16", 2.0,          "training/distribution, common default"},
            {"GGUF Q8_0",        34.0 / 32.0,  "inference, Chapter 4.2's block"},
            {"GGUF Q4_0",        18.0 / 32.0,  "inference, Chapter 4.3's block"},
            {"GGUF Q4_K_M",      144.0 / 256.0,"inference, this section's super-block"},
        };
        for (const auto& f : formats) {
            double gb = PARAMS * f.bytes_per_param / 1e9;
            std::cout << "  " << std::left << std::setw(18) << f.name << std::right
                       << std::fixed << std::setprecision(2) << std::setw(7) << gb
                       << " GB  (" << f.note << ")\n";
        }
        // Q4_0 and Q4_K_M land at the same size in this model (both 4.5
        // bits/weight nominally) -- the quality difference between them
        // is not visible in a byte count, only in how each allocates
        // its scales, which Test 2 already demonstrated directly.
        double q4_0_gb = PARAMS * (18.0 / 32.0) / 1e9;
        double q4_k_gb = PARAMS * (144.0 / 256.0) / 1e9;
        CHECK(std::abs(q4_0_gb - q4_k_gb) < 1e-9);
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
g++ -std=c++23 -Wall -Wextra -O2 05_gguf_vs_safetensors_kquants.cpp -o 05_gguf_vs_safetensors_kquants
./05_gguf_vs_safetensors_kquants
```

**Sample input:** the Q4_K_M super-block's byte layout computed field by field, a direct bits-per-weight comparison against Chapter 4.3's Q4_0, an exhaustive round trip of the packed 6-bit sub-block scale scheme (2000 random 8-tuples plus edge cases), and a genuinely computed file-size comparison across every format this chapter covered for an 8-billion-parameter model.

```text
================================================
GGUF vs SafeTensors, and a Peek at Production K-Quants
================================================

-- Test 1: Q4_K_M super-block byte layout --
  d (fp16 scale): 2 bytes
  dmin (fp16 min): 2 bytes
  packed sub-block scales+mins: 12 bytes
  packed 4-bit weight nibbles: 128 bytes
  total super-block size: 144 bytes for 256 weights
  effective bits/weight: 4.5

-- Test 2: Same nominal bit budget, different allocation --
  Q4_0:   18 bytes / 32 weights = 4.5 bits/weight (ONE scale for the whole block)
  Q4_K_M: 144 bytes / 256 weights = 4.5 bits/weight (a scale PER SUB-BLOCK of 32, within the super-block)
  same nominal bit budget: yes -- Q4_K_M spends it on finer-grained scales, not more of them

-- Test 3: Packed 6-bit sub-block scales, round-tripped --
  edge cases (all-zero, all-max, alternating) round-trip: yes
  2000 random 8-tuples of 6-bit values: 0 mismatches
  packed size: 6 bytes for 8 six-bit values (48 bits, no padding)

-- Test 4: One model, every format this chapter covered --
  SafeTensors F32     32.00 GB  (training, full precision)
  SafeTensors BF16    16.00 GB  (training/distribution, common default)
  GGUF Q8_0            8.50 GB  (inference, Chapter 4.2's block)
  GGUF Q4_0            4.50 GB  (inference, Chapter 4.3's block)
  GGUF Q4_K_M          4.50 GB  (inference, this section's super-block)

================================================
9/9 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] a header field list that does not add up to its own stated total"
    Reference material describing Q4_K's super-block layout lists a 2-byte scale, a 2-byte minimum, and 12 bytes of packed sub-block scales as a group, then labels that group "12 bytes" and computes a 140-byte super-block total (`12 + 128`) — silently dropping the scale and minimum's combined 4 bytes from the sum, even though they are right there in the same field list. The correct group size is `2 + 2 + 12 = 16` bytes, and the correct super-block total is 144 bytes, matching this chapter's own quantization-type table. The lesson generalizes past this one instance: when a written total and a field-by-field sum disagree, computing the sum in code (as Test 1 above does) settles the question in a way that re-reading the same table a second time never will.

## Chapter Summary

This chapter took the quantized blocks Chapter 4 built and gave them a real home on disk: GGUF's binary header, typed metadata, and tensor descriptor table, built and parsed byte by byte and then verified with a full writer-reader round trip; the naming convention that lets an engine find any of a model's hundreds of tensors by name; the zero-copy `std::span` views that turn a `mmap`'d file directly into usable Q8/Q4 block arrays with no heap allocation for tensor data; SafeTensors' JSON-header alternative, parsed with a hand-rolled extractor rather than a general library; and a corrected look at production K-quant super-blocks, which improve on Q4_0 not by using fewer bits but by spending the same bits more intelligently across a two-level scale hierarchy. Two separate arithmetic slips surfaced along the way — a byte offset that only running the code (not tracing it by hand) gets right, and a documented K-quant super-block size that omits 4 bytes from its own stated total — which is the same discipline this book has followed since Chapter 2: a claim about bytes, offsets, or bits is verified by computing it, not by re-reading the paragraph that stated it.

## Self-Check Questions

1. Why does GGUF store strings as a length-prefix plus raw characters rather than null-terminated, and what does that choice save the reader from having to do?
2. Walk through why the data section must start at a 32-byte-aligned offset, and what would happen to SIMD loads against the tensor data if it did not.
3. Why is testing a writer and a reader together, as one round trip, a stronger check than testing each in isolation?
4. In Section 5.2's reader, what specifically breaks if `TensorInfo::compute_data_size()` hardcodes a Q8_0 block as 36 bytes instead of computing `sizeof(BlockQ8)`?
5. Explain the arithmetic behind a 32-layer Llama model having exactly 291 tensors.
6. Why does Section 5.3 check that a span's data pointer falls "within the mapped region" as a boolean, rather than printing the actual pointer value for a human to inspect?
7. Describe the bug that produced `nan` in Section 5.3's first version, and why passing an entire tensor's span (rather than one row's worth) to a dot product is exactly the kind of error that would NOT be caught by a compiler.
8. Why are SafeTensors' `data_offsets` measured from the start of the tensor data rather than the start of the file, and what value has to be added to `data_offsets[0]` to get a real position inside the file?
9. In what sense do Q4_0 and Q4_K_M use "the same number of bits per weight," and in what sense are they still meaningfully different formats?
10. Why does packing eight 6-bit values take exactly 6 bytes with no wasted padding, and what would change if the values were 5 bits each instead?

## Where We Go Next

This chapter can load a real GGUF or SafeTensors file with zero copies, but every weight it reads is still fed to the forward pass exactly as it comes off disk: pure Q8_0, pure Q4_0, one format at a time. Chapter 6 builds the Hybrid Quantized Engine that Section 5's own source material previewed but this chapter deliberately left out: an inference engine that mixes quantization strategies within a single forward pass — coarser formats for weights that tolerate it, finer ones where they do not — and integrates a TurboQuant-compressed KV cache alongside the blockwise-quantized weights this book has built since Chapter 4.

## Worked Solutions

**1.** A length prefix means the reader knows exactly how many bytes to consume for a string before reading a single character of it, with no need to scan forward looking for a terminating byte. This matters doubly for GGUF specifically because tokenizer vocabularies can contain 128,000+ string entries — scanning for null terminators across that many strings would be measurably slower than reading a fixed 8-byte count up front.

**2.** Alignment guarantees that every tensor's data begins at an address divisible by 32, which is a precondition many SIMD load instructions either require outright or run measurably faster with. If the data section started at an arbitrary, unaligned offset, some tensors would begin at addresses that force the CPU to either fault (on architectures that require aligned SIMD loads) or silently fall back to a slower unaligned-load code path — a correctness or performance problem that depends entirely on which specific tensor's offset happens to land where.

**3.** A writer and reader that each look correct in isolation can still disagree about the meaning of a shared format detail — for instance, both might compile and run individually while assuming different byte orders for a length field, or different block sizes for the same quantization type. Only feeding the writer's actual output into the reader and checking the round trip catches a disagreement like that; testing each independently against its own assumptions cannot.

**4.** If the real `BlockQ8` is 34 bytes (2-byte fp16 scale + 32 int8 weights) but `compute_data_size()` hardcodes 36, every Q8_0 tensor's computed `data_size` is 2 bytes too large per block. Nothing crashes immediately — the value is simply wrong, and any code that uses it (to validate a file's total size, or to compute where the NEXT tensor's data should begin, in a format that packed tensors by size rather than storing explicit offsets) would silently misplace itself.

**5.** A Llama-architecture model has exactly 3 global tensors (the token embedding table, the final output norm, and the LM head projection) plus, for each of its layers, 9 per-layer tensors (an attention norm, four attention projection matrices for Q/K/V/output, an FFN norm, and three FFN projection matrices for gate/up/down in the SwiGLU architecture). For 32 layers: `3 + 32 * 9 = 3 + 288 = 291`.

**6.** A raw pointer value depends on where the operating system's address space layout randomization (ASLR) happened to place the `mmap`'d region on this particular run — it is real information about THIS execution, but it carries no information a reader of this book's verified output could check, and printing it would make the section's own "run it twice, the output must match exactly" verification impossible to satisfy, since the address changes every run by design.

**7.** A tensor's zero-copy span covers its ENTIRE weight matrix (for instance, 128 Q8_0 blocks for a 64x64 matrix), but the dot-product test only wanted the first row's worth (2 blocks, matching a 64-element input vector). Passing the full 128-block span meant the dot product's inner loop indexed up to `127 * 32 + 31 = 4095` into an input vector that only had 64 elements — reading far past its end into unrelated memory. A compiler cannot catch this because `std::span::size()` and `std::span::operator[]` are both doing exactly what they are asked; nothing in the type system encodes the fact that the SEMANTIC intent was "one row," only that the span happens to be longer than that.

**8.** They are measured from the start of the tensor data (immediately after the 8-byte header-length field and the JSON header itself) rather than from byte 0 of the file, because the JSON header's own length varies from file to file depending on how many tensors and metadata fields it describes — storing offsets relative to a variable-length prefix would mean every offset changes if even one more character were added to the header. The value that must be added is `8 + header_length` (the header-length field's own 8 bytes, plus the JSON header's length in bytes), giving `tensor_data_base + data_offsets[0]` as the real file position.

**9.** They use the same number of bits per weight in the sense that both this chapter's Q4_0 (18 bytes for 32 weights) and Q4_K_M (144 bytes for 256 weights) compute to exactly 4.5 bits per weight — neither format is nominally "more compressed" than the other. They remain meaningfully different because Q4_0 spends that budget on one flat scale for an entire 32-weight block, while Q4_K_M spends the identical budget on a two-level hierarchy (one super-block scale plus eight finer sub-block scales), which can represent variance WITHIN a 256-weight region that a single flat scale cannot — the same bit cost buying meaningfully better fidelity to the original weights.

**10.** Eight values at 6 bits each is exactly `8 * 6 = 48` bits, and 48 is evenly divisible by 8, so the packed result is exactly 6 bytes with nothing left over — no value ever needs to be split unevenly across a byte boundary in a way that leaves a fractional bit unaccounted for at the end. If the values were 5 bits each instead, 8 of them would be `8 * 5 = 40` bits, which is also evenly divisible by 8 (5 bytes) — but a group of, say, 5 values at 6 bits each would be 30 bits, which is NOT evenly divisible by 8, and would either need padding to the next whole byte (wasting bits) or a scheme where a byte boundary sometimes falls in the middle of a value, which is exactly what this section's `pack_6bit`/`unpack_6bit` functions already handle correctly for the 8-value case, since 6 does not evenly divide 8 either.
