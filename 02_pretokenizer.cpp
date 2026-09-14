// 02_pretokenizer.cpp
// Chapter 12, Part 2: before any BPE merge rule ever looks at a byte, the
// input text is split into coarse chunks by a pre-tokenizer. This step
// exists so BPE can never merge across a word boundary: without it, a
// greedy byte-pair merger would happily fuse the trailing "o" of "Hello"
// with the space that follows it, then fuse that with the "w" of "world",
// producing tokens that straddle two different words. Pre-tokenization
// draws the boundaries BPE is not allowed to cross; BPE then runs
// independently inside each chunk.
//
// The convention used by Llama 3, GPT-4, and most modern BPE tokenizers
// (via a regex pattern derived from the original GPT-2/tiktoken pattern)
// is: alphabetic runs are one chunk, with any leading whitespace ABSORBED
// into the chunk that follows it, not attached to the chunk before it.
// "the cat" therefore splits into ["the", " cat"], not ["the ", "cat"].
// This is not a cosmetic choice -- it means the token for "cat" occurring
// after a space is a genuinely different token from "cat" occurring at
// the start of a sentence, and the vocabulary must learn both. Digit runs
// are grouped the same way (with leading whitespace absorbed). Punctuation
// is never grouped: each punctuation character becomes its own one-
// character chunk, so "wow!!!" produces three separate "!" chunks rather
// than one "!!!" chunk. Whitespace that is not followed by anything else
// (trailing whitespace at the end of the input) is emitted as its own
// chunk, since there is no following word to absorb it.
//
// This file implements a simplified ASCII-only version of that scheme
// (alphabetic / digit / punctuation / whitespace character classes; no
// Unicode letter categories, no contraction-specific rules such as
// splitting "don't" into "don" + "'t"). A production tokenizer's regex
// handles those cases too, but the four-way character-class scan below
// is the same core idea a full Unicode-aware implementation extends.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_pretokenizer.cpp -o 02_pretokenizer

#include <cctype>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

enum CharClass { CLS_ALPHA, CLS_DIGIT, CLS_SPACE, CLS_PUNCT };

CharClass classify(char c) {
    unsigned char u = static_cast<unsigned char>(c);
    if (std::isalpha(u)) return CLS_ALPHA;
    if (std::isdigit(u)) return CLS_DIGIT;
    if (std::isspace(u)) return CLS_SPACE;
    return CLS_PUNCT;
}

// Splits text into chunks:
//   1. Leading whitespace is absorbed into the chunk that follows it.
//   2. An alphabetic run (plus any absorbed leading whitespace) is one chunk.
//   3. A digit run (plus any absorbed leading whitespace) is one chunk.
//   4. Each punctuation character is its own chunk (leading whitespace,
//      if any, is still absorbed into it).
//   5. Whitespace with nothing following it (end of input) is its own chunk.
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
            while (i < text.size() && classify(text[i]) == CLS_ALPHA) {
                chunk += text[i++];
            }
            chunks.push_back(chunk);
        } else if (cls == CLS_DIGIT) {
            while (i < text.size() && classify(text[i]) == CLS_DIGIT) {
                chunk += text[i++];
            }
            chunks.push_back(chunk);
        } else {
            chunk += text[i++];
            chunks.push_back(chunk);
        }
    }
    return chunks;
}

void print_chunks(const std::vector<std::string>& chunks) {
    std::cout << "  [";
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << chunks[i] << "\"";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 12.2: Pre-Tokenization -- Splitting Text Before BPE\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: \"Hello, world!\" --\n";
    {
        auto chunks = pretokenize("Hello, world!");
        print_chunks(chunks);
        CHECK(chunks.size() == 4);
        CHECK(chunks[0] == "Hello");
        CHECK(chunks[1] == ",");
        CHECK(chunks[2] == " world");
        CHECK(chunks[3] == "!");
    }

    std::cout << "\n-- Test 2: \"the cat sat\" (leading space attaches to the following word) --\n";
    {
        auto chunks = pretokenize("the cat sat");
        print_chunks(chunks);
        CHECK(chunks.size() == 3);
        CHECK(chunks[0] == "the");
        CHECK(chunks[1] == " cat");
        CHECK(chunks[2] == " sat");
    }

    std::cout << "\n-- Test 3: \"I have 42 cats\" (digit runs grouped) --\n";
    {
        auto chunks = pretokenize("I have 42 cats");
        print_chunks(chunks);
        CHECK(chunks.size() == 4);
        CHECK(chunks[0] == "I");
        CHECK(chunks[1] == " have");
        CHECK(chunks[2] == " 42");
        CHECK(chunks[3] == " cats");
    }

    std::cout << "\n-- Test 4: \"wow!!!\" (each punctuation mark is its own chunk) --\n";
    {
        auto chunks = pretokenize("wow!!!");
        print_chunks(chunks);
        CHECK(chunks.size() == 4);
        CHECK(chunks[0] == "wow");
        CHECK(chunks[1] == "!");
        CHECK(chunks[2] == "!");
        CHECK(chunks[3] == "!");
    }

    std::cout << "\n-- Test 5: edge cases --\n";
    {
        auto empty = pretokenize("");
        CHECK(empty.empty());
        std::cout << "  \"\" -> []\n";

        auto spaces = pretokenize("   ");
        CHECK(spaces.size() == 1);
        CHECK(spaces[0] == "   ");
        std::cout << "  \"   \" -> [\"   \"] (trailing whitespace with nothing to absorb into)\n";
    }

    std::cout << "\n-- Test 6: a chunk boundary is not a merge boundary --\n";
    {
        // "lowest" as a single alphabetic chunk is exactly the string
        // 01_bpe_merge_engine.cpp's BPEEncoder::encode consumes whole; this
        // test confirms pre-tokenization hands BPE one clean chunk per word,
        // never splitting a word mid-run and never merging two words into one.
        auto chunks = pretokenize("lowest wins");
        print_chunks(chunks);
        CHECK(chunks.size() == 2);
        CHECK(chunks[0] == "lowest");
        CHECK(chunks[1] == " wins");
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
