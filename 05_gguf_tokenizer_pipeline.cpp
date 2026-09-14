// 05_gguf_tokenizer_pipeline.cpp
// Chapter 12, Part 5 (capstone): a tokenizer is only useful for real
// inference if it can be loaded from the same GGUF file that carries the
// model's weights. Chapter 5's GGUFWriter/GGUFReader (02_gguf_writer_reader.cpp)
// already handles the scalar KV types a model's architecture metadata
// needs -- STRING(8), UINT32(4), FLOAT32(6) -- but a tokenizer's vocabulary
// does not fit in a scalar: it is a list of thousands of strings, one per
// token ID. GGUF represents that with KV type ARRAY(9): a key, the tag 9,
// then an element-type tag, a count, and that many elements back to back.
// This section extends Chapter 5's real, already-verified reader and
// writer with ARRAY support for STRING, INT32, and FLOAT32 elements --
// exactly the three element types real tokenizer metadata needs:
//   tokenizer.ggml.tokens      ARRAY of STRING   -- one string per token ID
//   tokenizer.ggml.token_type  ARRAY of INT32    -- 1=normal, 3=control, 6=byte
//   tokenizer.ggml.scores      ARRAY of FLOAT32  -- used by SentencePiece,
//                                                    ignored by this BPE
//                                                    tokenizer, still present
//                                                    for format compatibility
//   tokenizer.ggml.merges      ARRAY of STRING   -- "left right" per rule,
//                                                    ordered by priority
// plus tokenizer.ggml.bos_token_id / eos_token_id as ordinary UINT32 KVs.
//
// This file does not duplicate Chapter 5's tensor/quantization machinery
// (BlockQ8, BlockQ4, fp16_t) -- this capstone is about the tokenizer, so
// the test file it writes has zero tensors, only the KV metadata a real
// GGUF file's tokenizer section would carry.
//
// The capstone proves something stronger than "the bytes round-trip":
// Test 4 rebuilds a fully working Vocabulary + MergeTable + BPEEncoder
// (Section 12.1) and a TokenDecoder (Section 12.4) ENTIRELY from what
// GGUFReader reads back off disk -- no in-memory shortcut -- and Test 6
// runs Section 12.2's pre-tokenizer, that reconstructed encoder, and that
// reconstructed decoder together on ordinary text and confirms the
// original string comes back byte-for-byte. That is the whole tokenizer
// pipeline, loaded from a real file, doing a lossless round trip.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_gguf_tokenizer_pipeline.cpp -o 05_gguf_tokenizer_pipeline

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// ---------------------------------------------------------------------
// Chapter 5's GGUF constants, extended with INT32(5) and ARRAY(9) -- the
// two type tags Chapter 5's own KV loop never needed.
// ---------------------------------------------------------------------
static constexpr uint32_t GGUF_MAGIC = 0x46475547;
static constexpr uint32_t GGUF_VERSION = 3;
enum GGUFType : uint32_t { T_UINT32 = 4, T_INT32 = 5, T_FLOAT32 = 6, T_STRING = 8, T_ARRAY = 9 };

// -- Writer: Chapter 5's write_u32/write_u64/write_string/write_kv_string/
//    write_kv_u32/align/tell/good are reused verbatim. write_i32, write_f32,
//    and the three write_kv_array_* methods are new in this section. --
class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* data, size_t size) { out.write(reinterpret_cast<const char*>(data), size); pos += size; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_i32(int32_t v) { write_raw(&v, 4); }
    void write_f32(float v) { write_raw(&v, 4); }
    void write_string(const std::string& s) { uint64_t len = s.size(); write_raw(&len, 8); write_raw(s.data(), s.size()); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(T_STRING); write_string(v); }
    void write_kv_u32(const std::string& k, uint32_t v) { write_string(k); write_u32(T_UINT32); write_u32(v); }

    void write_kv_array_string(const std::string& k, const std::vector<std::string>& values) {
        write_string(k);
        write_u32(T_ARRAY);
        write_u32(T_STRING);
        write_u64(values.size());
        for (const auto& v : values) write_string(v);
    }
    void write_kv_array_i32(const std::string& k, const std::vector<int32_t>& values) {
        write_string(k);
        write_u32(T_ARRAY);
        write_u32(T_INT32);
        write_u64(values.size());
        for (int32_t v : values) write_i32(v);
    }
    void write_kv_array_f32(const std::string& k, const std::vector<float>& values) {
        write_string(k);
        write_u32(T_ARRAY);
        write_u32(T_FLOAT32);
        write_u64(values.size());
        for (float v : values) write_f32(v);
    }

    void align(size_t alignment) {
        size_t rem = pos % alignment;
        if (rem != 0) { std::vector<char> zeros(alignment - rem, 0); write_raw(zeros.data(), zeros.size()); }
    }
    size_t tell() const { return pos; }
    bool good() const { return out.good(); }
};

// -- Reader: Chapter 5's scalar cases (T_STRING/T_UINT32/T_FLOAT32) are
//    reused verbatim inside the same KV loop; the T_ARRAY case is new. --
using MetaValue = std::variant<std::string, uint32_t, float,
                                std::vector<std::string>, std::vector<int32_t>, std::vector<float>>;

class GGUFReader {
    std::ifstream in;
    void read_raw(void* data, size_t size) { in.read(reinterpret_cast<char*>(data), size); }
public:
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    std::unordered_map<std::string, MetaValue> metadata;

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
                case T_ARRAY: {
                    uint32_t elem_type; read_raw(&elem_type, 4);
                    uint64_t count; read_raw(&count, 8);
                    if (elem_type == T_STRING) {
                        std::vector<std::string> arr(count);
                        for (uint64_t j = 0; j < count; ++j) arr[j] = read_string();
                        metadata[key] = std::move(arr);
                    } else if (elem_type == T_INT32) {
                        std::vector<int32_t> arr(count);
                        for (uint64_t j = 0; j < count; ++j) read_raw(&arr[j], 4);
                        metadata[key] = std::move(arr);
                    } else if (elem_type == T_FLOAT32) {
                        std::vector<float> arr(count);
                        for (uint64_t j = 0; j < count; ++j) read_raw(&arr[j], 4);
                        metadata[key] = std::move(arr);
                    } else {
                        return false;  // element type this reader does not support
                    }
                    break;
                }
                default: return false;
            }
        }
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
    const std::vector<std::string>& get_string_array(const std::string& key) const {
        return std::get<std::vector<std::string>>(metadata.at(key));
    }
    const std::vector<int32_t>& get_i32_array(const std::string& key) const {
        return std::get<std::vector<int32_t>>(metadata.at(key));
    }
    const std::vector<float>& get_f32_array(const std::string& key) const {
        return std::get<std::vector<float>>(metadata.at(key));
    }
};

// ---------------------------------------------------------------------
// Section 12.1's Vocabulary / MergeTable / BPEEncoder, reused verbatim.
// ---------------------------------------------------------------------
struct MergeTable {
    std::unordered_map<std::string, int> priority_of;
    void add(const std::string& left, const std::string& right, int priority) {
        priority_of[left + " " + right] = priority;
    }
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

struct BPEEncoder {
    const MergeTable& merges;
    const Vocabulary& vocab;
    BPEEncoder(const MergeTable& m, const Vocabulary& v) : merges(m), vocab(v) {}
    std::vector<int> encode(const std::string& text) const {
        if (text.empty()) return {};
        std::vector<std::string> tokens;
        tokens.reserve(text.size());
        for (unsigned char c : text) tokens.push_back(std::string(1, static_cast<char>(c)));
        while (tokens.size() >= 2) {
            int best_priority = -1;
            int best_pos = -1;
            for (int i = 0; i < static_cast<int>(tokens.size()) - 1; ++i) {
                int p = merges.lookup(tokens[i], tokens[i + 1]);
                if (p >= 0 && (best_pos < 0 || p < best_priority)) {
                    best_priority = p;
                    best_pos = i;
                }
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
};

// ---------------------------------------------------------------------
// Section 12.2's pre-tokenizer, reused verbatim.
// ---------------------------------------------------------------------
enum CharClass { CLS_ALPHA, CLS_DIGIT, CLS_SPACE, CLS_PUNCT };

CharClass classify(char c) {
    unsigned char u = static_cast<unsigned char>(c);
    if (std::isalpha(u)) return CLS_ALPHA;
    if (std::isdigit(u)) return CLS_DIGIT;
    if (std::isspace(u)) return CLS_SPACE;
    return CLS_PUNCT;
}

std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<std::string> chunks;
    if (text.empty()) return chunks;
    size_t i = 0;
    while (i < text.size()) {
        std::string chunk;
        while (i < text.size() && classify(text[i]) == CLS_SPACE) {
            chunk += text[i++];
        }
        if (i >= text.size()) {
            if (!chunk.empty()) chunks.push_back(chunk);
            break;
        }
        CharClass cls = classify(text[i]);
        if (cls == CLS_ALPHA) {
            while (i < text.size() && classify(text[i]) == CLS_ALPHA) chunk += text[i++];
            chunks.push_back(chunk);
        } else if (cls == CLS_DIGIT) {
            while (i < text.size() && classify(text[i]) == CLS_DIGIT) chunk += text[i++];
            chunks.push_back(chunk);
        } else {
            chunk += text[i++];
            chunks.push_back(chunk);
        }
    }
    return chunks;
}

// ---------------------------------------------------------------------
// Section 12.4's token types and decoder, reused verbatim.
// ---------------------------------------------------------------------
enum TokenType : int32_t { TOKEN_NORMAL = 1, TOKEN_UNKNOWN = 2, TOKEN_CONTROL = 3, TOKEN_BYTE = 6 };

struct DecoderVocab {
    struct Entry { std::string text; TokenType type; };
    std::vector<Entry> entries;
    void add(const std::string& text, TokenType type) { entries.push_back({text, type}); }
    int size() const { return static_cast<int>(entries.size()); }
};

struct TokenDecoder {
    const DecoderVocab& vocab;
    bool show_special;
    TokenDecoder(const DecoderVocab& v, bool show_specials = false) : vocab(v), show_special(show_specials) {}
    std::string decode_token(int id) const {
        if (id < 0 || id >= vocab.size()) return "";
        const auto& entry = vocab.entries[id];
        switch (entry.type) {
            case TOKEN_NORMAL: return entry.text;
            case TOKEN_BYTE:   return (id < 256) ? std::string(1, static_cast<char>(id)) : entry.text;
            case TOKEN_CONTROL: return show_special ? entry.text : "";
            default: return "";
        }
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) result += decode_token(id);
        return result;
    }
    struct DecodeResult { std::string text; bool hit_eos; };
    DecodeResult decode_with_stop(const std::vector<int>& ids, int eos_id) const {
        DecodeResult result{"", false};
        for (int id : ids) {
            if (id == eos_id) { result.hit_eos = true; break; }
            result.text += decode_token(id);
        }
        return result;
    }
};

// ---------------------------------------------------------------------
// This section's own contribution: build the in-memory tokenizer once
// (same 270-token test vocabulary Sections 12.1/12.3 used, plus two
// control tokens this section adds for BOS/EOS), and a helper that
// splits a GGUF merge string ("left right") back into its two halves.
// ---------------------------------------------------------------------
struct Rule { std::string left, right, result; };
static const Rule RULES[] = {
    {"l", "o", "lo"}, {"lo", "w", "low"}, {"e", "s", "es"}, {"es", "t", "est"},
    {"e", "r", "er"}, {"low", "er", "lower"}, {"H", "e", "He"}, {"l", "l", "ll"},
    {"He", "ll", "Hell"}, {"Hell", "o", "Hello"}, {"w", "o", "wo"},
    {"wo", "r", "wor"}, {"l", "d", "ld"}, {"wor", "ld", "world"},
};

void split_merge_string(const std::string& s, std::string& left, std::string& right) {
    size_t sp = s.find(' ');
    left = s.substr(0, sp);
    right = s.substr(sp + 1);
}

int main() {
    const std::string path = "/tmp/ch12_tokenizer.gguf";

    std::cout << "========================================================\n";
    std::cout << "Chapter 12.5: A Complete Tokenizer Pipeline From a Real GGUF File\n";
    std::cout << "========================================================\n";

    // In-memory ground truth: 256 byte tokens + 14 merge-rule tokens
    // (IDs 256-269, Section 12.1's exact vocabulary) + 2 control tokens
    // this section adds for BOS/EOS (IDs 270-271).
    Vocabulary vocab;
    MergeTable merges;
    for (int i = 0; i < 256; ++i) vocab.add(std::string(1, static_cast<char>(i)));
    for (int i = 0; i < static_cast<int>(std::size(RULES)); ++i) {
        vocab.add(RULES[i].result);
        merges.add(RULES[i].left, RULES[i].right, i);
    }
    const int BOS_ID = vocab.add("<|begin_of_text|>");  // 270
    const int EOS_ID = vocab.add("<|end_of_text|>");    // 271
    BPEEncoder ground_truth_encoder(merges, vocab);

    std::vector<int32_t> token_types(vocab.size());
    for (int i = 0; i < vocab.size(); ++i) {
        if (i < 256) token_types[i] = TOKEN_BYTE;
        else if (i == BOS_ID || i == EOS_ID) token_types[i] = TOKEN_CONTROL;
        else token_types[i] = TOKEN_NORMAL;
    }
    std::vector<float> scores(vocab.size(), 0.0f);  // ignored by BPE, present for format compatibility
    std::vector<std::string> merge_strings;
    for (const auto& r : RULES) merge_strings.push_back(r.left + " " + r.right);

    std::cout << "\n-- Test 1: writing the file --\n";
    {
        GGUFWriter writer(path);
        writer.write_u32(GGUF_MAGIC);
        writer.write_u32(GGUF_VERSION);
        writer.write_u64(0);  // n_tensors: this capstone carries no tensor data
        writer.write_u64(7);  // n_kv
        writer.write_kv_string("general.architecture", "llama");
        writer.write_kv_array_string("tokenizer.ggml.tokens", vocab.id_to_text);
        writer.write_kv_array_i32("tokenizer.ggml.token_type", token_types);
        writer.write_kv_array_f32("tokenizer.ggml.scores", scores);
        writer.write_kv_array_string("tokenizer.ggml.merges", merge_strings);
        writer.write_kv_u32("tokenizer.ggml.bos_token_id", static_cast<uint32_t>(BOS_ID));
        writer.write_kv_u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(EOS_ID));
        std::cout << "  wrote " << vocab.size() << " tokens, " << merge_strings.size()
                  << " merge rules, " << writer.tell() << " bytes total\n";
        CHECK(writer.good());
    }

    std::cout << "\n-- Test 2: reopening the file fresh and parsing the header --\n";
    GGUFReader reader;
    {
        bool ok = reader.open(path);
        CHECK(ok);
        CHECK(reader.magic == GGUF_MAGIC);
        CHECK(reader.version == GGUF_VERSION);
        CHECK(reader.n_tensors == 0);
        CHECK(reader.n_kv == 7);
        std::cout << "  magic ok, version=" << reader.version
                   << ", n_tensors=" << reader.n_tensors << ", n_kv=" << reader.n_kv << "\n";
    }

    std::cout << "\n-- Test 3: array metadata matches exactly, element for element --\n";
    {
        CHECK(reader.get_string("general.architecture") == "llama");
        const auto& tokens_back = reader.get_string_array("tokenizer.ggml.tokens");
        const auto& types_back = reader.get_i32_array("tokenizer.ggml.token_type");
        const auto& scores_back = reader.get_f32_array("tokenizer.ggml.scores");
        const auto& merges_back = reader.get_string_array("tokenizer.ggml.merges");

        CHECK(tokens_back.size() == vocab.id_to_text.size());
        CHECK(tokens_back == vocab.id_to_text);
        CHECK(types_back.size() == token_types.size());
        CHECK(types_back == token_types);
        CHECK(scores_back.size() == scores.size());
        bool all_zero = true;
        for (float s : scores_back) if (s != 0.0f) all_zero = false;
        CHECK(all_zero);
        CHECK(merges_back.size() == merge_strings.size());
        CHECK(merges_back == merge_strings);
        CHECK(reader.get_u32("tokenizer.ggml.bos_token_id") == static_cast<uint32_t>(BOS_ID));
        CHECK(reader.get_u32("tokenizer.ggml.eos_token_id") == static_cast<uint32_t>(EOS_ID));

        std::cout << "  tokens=" << tokens_back.size() << " token_type=" << types_back.size()
                  << " scores=" << scores_back.size() << " merges=" << merges_back.size() << "\n";
        std::cout << "  bos_token_id=" << reader.get_u32("tokenizer.ggml.bos_token_id")
                   << " eos_token_id=" << reader.get_u32("tokenizer.ggml.eos_token_id") << "\n";
    }

    std::cout << "\n-- Test 4: a working encoder rebuilt entirely from the file --\n";
    Vocabulary loaded_vocab;
    MergeTable loaded_merges;
    {
        const auto& tokens_back = reader.get_string_array("tokenizer.ggml.tokens");
        for (const auto& t : tokens_back) loaded_vocab.add(t);  // sequential add reproduces the same IDs
        const auto& merges_back = reader.get_string_array("tokenizer.ggml.merges");
        for (size_t i = 0; i < merges_back.size(); ++i) {
            std::string left, right;
            split_merge_string(merges_back[i], left, right);
            loaded_merges.add(left, right, static_cast<int>(i));
        }
        BPEEncoder loaded_encoder(loaded_merges, loaded_vocab);

        // Section 12.1's own worked cases, re-run against the RECONSTRUCTED
        // tokenizer, compared against the in-memory ground truth encoder.
        auto check_matches = [&](const std::string& text) {
            auto expected = ground_truth_encoder.encode(text);
            auto got = loaded_encoder.encode(text);
            CHECK(got == expected);
            return got;
        };
        auto lowest_ids = check_matches("lowest");
        auto hello_ids = check_matches("Hello");
        std::cout << "  \"lowest\" -> " << lowest_ids.size() << " tokens (matches in-memory ground truth)\n";
        std::cout << "  \"Hello\" -> " << hello_ids.size() << " tokens (matches in-memory ground truth)\n";
        CHECK(hello_ids.size() == 3);  // ["He","l","lo"] -- priority, not opportunity, Section 12.1's own point
    }

    std::cout << "\n-- Test 5: a working decoder rebuilt entirely from the file --\n";
    DecoderVocab loaded_decoder_vocab;
    {
        const auto& tokens_back = reader.get_string_array("tokenizer.ggml.tokens");
        const auto& types_back = reader.get_i32_array("tokenizer.ggml.token_type");
        for (size_t i = 0; i < tokens_back.size(); ++i) {
            loaded_decoder_vocab.add(tokens_back[i], static_cast<TokenType>(types_back[i]));
        }
        TokenDecoder loaded_decoder(loaded_decoder_vocab);
        BPEEncoder loaded_encoder(loaded_merges, loaded_vocab);

        std::string text = "world";
        auto ids = loaded_encoder.encode(text);
        std::string decoded = loaded_decoder.decode(ids);
        CHECK(decoded == text);
        std::cout << "  \"" << text << "\" -> " << ids.size() << " token(s) -> \"" << decoded << "\" (exact round trip)\n";
    }

    std::cout << "\n-- Test 6: the full pipeline -- pretokenize + encode + decode -- on ordinary text --\n";
    {
        BPEEncoder loaded_encoder(loaded_merges, loaded_vocab);
        TokenDecoder loaded_decoder(loaded_decoder_vocab);

        std::string original = "the lowest world";
        auto chunks = pretokenize(original);
        std::vector<int> all_ids;
        for (const auto& chunk : chunks) {
            for (int id : loaded_encoder.encode(chunk)) all_ids.push_back(id);
        }
        std::string reassembled = loaded_decoder.decode(all_ids);
        CHECK(reassembled == original);
        std::cout << "  \"" << original << "\" -> " << chunks.size() << " pre-token chunks -> "
                  << all_ids.size() << " token IDs -> \"" << reassembled << "\"\n";
        std::cout << "  lossless: " << (reassembled == original ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Test 7: decode_with_stop using the BOS/EOS IDs read from the file --\n";
    {
        BPEEncoder loaded_encoder(loaded_merges, loaded_vocab);
        TokenDecoder loaded_decoder(loaded_decoder_vocab);
        uint32_t bos = reader.get_u32("tokenizer.ggml.bos_token_id");
        uint32_t eos = reader.get_u32("tokenizer.ggml.eos_token_id");

        std::vector<int> ids;
        ids.push_back(static_cast<int>(bos));
        for (int id : loaded_encoder.encode("lower")) ids.push_back(id);
        ids.push_back(static_cast<int>(eos));
        for (int id : loaded_encoder.encode("world")) ids.push_back(id);  // must be discarded

        auto result = loaded_decoder.decode_with_stop(ids, static_cast<int>(eos));
        CHECK(result.hit_eos);
        CHECK(result.text == "lower");  // BOS suppressed by default, "world" after EOS discarded
        std::cout << "  decoded up to eos_token_id: \"" << result.text << "\", hit_eos="
                  << (result.hit_eos ? "yes" : "no") << "\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
