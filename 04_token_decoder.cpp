// 04_token_decoder.cpp
// Chapter 12, Part 4: decoding reverses encoding -- given the sequence of
// token IDs a model produces, reconstruct the UTF-8 text they represent.
// This looks trivial (look up each ID, concatenate the results) but two
// details make it more than a table lookup.
//
// The first is byte-fallback tokens. Section 12.1 established that every
// token is either one of 256 base byte tokens or the concatenation of two
// earlier tokens via a merge rule -- the vocabulary is a tree rooted at
// individual bytes. Encoding falls back to a raw byte token whenever no
// merge rule applies; decoding must reverse that by emitting the actual
// byte value the token represents, not a literal escape-sequence string.
// This matters for multi-byte UTF-8: a codepoint like U+00E9 ("e" with an
// acute accent) is two bytes (0xC3, 0xA9) in UTF-8. If no merge rule ever
// joined those two bytes into one token, they arrive at decode time as two
// separate byte-fallback tokens; the decoder emits both raw bytes in order
// and the two bytes reassemble into one valid UTF-8 codepoint purely
// because they were emitted adjacently and in the right order. The
// decoder does not need to know anything about Unicode to get this right.
//
// The second is special/control tokens (this chapter's Section 12.3 kind:
// begin_of_text, eot_id, and so on). These carry no printable text a user
// should see; the decoder either suppresses them entirely or, in a debug
// mode, renders their placeholder text so a developer can see the control
// structure of a generated sequence. A control token can also serve as a
// stop signal: decode_with_stop below halts the moment it sees a chosen
// stop ID (typically end-of-text), discarding anything the model may have
// generated after it -- generation past EOS is not meaningful text.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_token_decoder.cpp -o 04_token_decoder

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// Matches the GGUF tokenizer.ggml.token_type convention this book's GGUF
// reader/writer will use in Section 12.5: 1=normal, 2=unknown, 3=control,
// 6=byte. (Values 4 and 5, unused here, are SentencePiece-specific types
// this BPE tokenizer does not produce.)
enum TokenType : int32_t {
    TOKEN_NORMAL  = 1,
    TOKEN_UNKNOWN = 2,
    TOKEN_CONTROL = 3,
    TOKEN_BYTE    = 6,
};

struct DecoderVocab {
    struct Entry {
        std::string text;
        TokenType type;
    };
    std::vector<Entry> entries;
    void add(const std::string& text, TokenType type) {
        entries.push_back({text, type});
    }
    int size() const { return static_cast<int>(entries.size()); }
};

// Converts token IDs to text.
//   - Normal tokens: emit the token's stored text directly.
//   - Byte tokens: for IDs 0-255, the ID itself IS the byte value -- emit
//     that raw byte, not the token's display text ("<0xC3>").
//   - Control tokens: suppressed by default; emitted as their display
//     text ("<|begin_of_text|>") only when show_special is set.
struct TokenDecoder {
    const DecoderVocab& vocab;
    bool show_special;
    TokenDecoder(const DecoderVocab& v, bool show_specials = false)
        : vocab(v), show_special(show_specials) {}

    std::string decode_token(int id) const {
        if (id < 0 || id >= vocab.size()) return "";
        const auto& entry = vocab.entries[id];
        switch (entry.type) {
            case TOKEN_NORMAL:
                return entry.text;
            case TOKEN_BYTE:
                if (id < 256) return std::string(1, static_cast<char>(id));
                return entry.text;
            case TOKEN_CONTROL:
                return show_special ? entry.text : "";
            default:
                return "";
        }
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) result += decode_token(id);
        return result;
    }

    struct DecodeResult {
        std::string text;
        bool hit_eos;
    };

    // Stops the instant `eos_id` is seen; anything after it is discarded,
    // since a model's output past its own end-of-text signal is not
    // meaningful generated text.
    DecodeResult decode_with_stop(const std::vector<int>& ids, int eos_id) const {
        DecodeResult result{"", false};
        for (int id : ids) {
            if (id == eos_id) {
                result.hit_eos = true;
                break;
            }
            result.text += decode_token(id);
        }
        return result;
    }
};

int main() {
    DecoderVocab vocab;
    for (int i = 0; i < 256; ++i) {
        char hex[16];
        std::snprintf(hex, sizeof(hex), "<0x%02X>", i);
        vocab.add(std::string(hex), TOKEN_BYTE);
    }
    vocab.add("Hello", TOKEN_NORMAL);       // ID 256
    vocab.add(" world", TOKEN_NORMAL);      // ID 257
    vocab.add("!", TOKEN_NORMAL);           // ID 258
    vocab.add(" how", TOKEN_NORMAL);        // ID 259
    vocab.add(" are", TOKEN_NORMAL);        // ID 260
    vocab.add(" you", TOKEN_NORMAL);        // ID 261
    vocab.add("<|begin_of_text|>", TOKEN_CONTROL);  // ID 262
    vocab.add("<|end_of_text|>", TOKEN_CONTROL);    // ID 263
    const int BOS_ID = 262, EOS_ID = 263;

    TokenDecoder decoder(vocab);

    std::cout << "========================================================\n";
    std::cout << "Chapter 12.4: The Token Decoder -- IDs Back to UTF-8\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: basic decode of normal tokens --\n";
    {
        std::vector<int> ids = {256, 257, 258};
        std::string text = decoder.decode(ids);
        std::cout << "  [256, 257, 258] -> \"" << text << "\"\n";
        CHECK(text == "Hello world!");
    }

    std::cout << "\n-- Test 2: byte-fallback decode --\n";
    {
        std::vector<int> ids = {72, 105};  // 'H'=72, 'i'=105
        std::string text = decoder.decode(ids);
        std::cout << "  [72, 105] (byte tokens for 'H', 'i') -> \"" << text << "\"\n";
        CHECK(text == "Hi");
    }

    std::cout << "\n-- Test 3: control tokens suppressed by default, visible in debug mode --\n";
    {
        std::vector<int> ids = {BOS_ID, 256, 257, 258, EOS_ID};
        std::string text = decoder.decode(ids);
        std::cout << "  default:  \"" << text << "\"\n";
        CHECK(text == "Hello world!");

        TokenDecoder debug_decoder(vocab, /*show_specials=*/true);
        std::string debug_text = debug_decoder.decode(ids);
        std::cout << "  debug:    \"" << debug_text << "\"\n";
        CHECK(debug_text.find("<|begin_of_text|>") != std::string::npos);
        CHECK(debug_text.find("<|end_of_text|>") != std::string::npos);
    }

    std::cout << "\n-- Test 4: decode_with_stop discards anything after EOS --\n";
    {
        std::vector<int> ids = {256, 257, EOS_ID, 258};  // "!" comes after EOS
        auto result = decoder.decode_with_stop(ids, EOS_ID);
        std::cout << "  text: \"" << result.text << "\", hit_eos: " << (result.hit_eos ? "yes" : "no") << "\n";
        CHECK(result.text == "Hello world");
        CHECK(result.hit_eos == true);
    }

    std::cout << "\n-- Test 5: normal and byte tokens mixed in one sequence --\n";
    {
        std::vector<int> ids = {256, 33};  // "Hello" (normal) + '!' (byte)
        std::string text = decoder.decode(ids);
        std::cout << "  [256, 33] -> \"" << text << "\"\n";
        CHECK(text == "Hello!");
    }

    std::cout << "\n-- Test 6: a real multi-byte UTF-8 codepoint via two byte-fallback tokens --\n";
    {
        // U+00E9 ("e" with an acute accent, e.g. in "cafe") is encoded in
        // UTF-8 as the two bytes 0xC3 0xA9. If no merge rule ever joined
        // them into a single token, they decode as two separate byte
        // tokens (195 and 169) whose emitted bytes concatenate into one
        // valid UTF-8 codepoint -- the decoder does no Unicode-aware work
        // at all, it just emits bytes 195 and 169 in order.
        std::vector<int> ids = {'c', 'a', 'f', 0xC3, 0xA9};
        std::string text = decoder.decode(ids);
        std::string expected = "caf\xC3\xA9";  // "cafe" with a UTF-8 accented e
        CHECK(text == expected);
        CHECK(text.size() == 5);  // 3 ASCII bytes + 2 bytes forming one codepoint
        // Printed as hex, not as the raw character, so this chapter's own
        // output stays plain ASCII even while it exercises real UTF-8 bytes.
        std::cout << "  [99,97,102,195,169] -> byte sequence (hex):";
        for (unsigned char b : text) std::printf(" %02x", b);
        std::cout << "  (3 ASCII bytes + one 2-byte UTF-8 codepoint, U+00E9)\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
