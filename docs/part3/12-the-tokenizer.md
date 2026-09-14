# Chapter 12: The Tokenizer -- BPE, SentencePiece, and the Vocabulary Pipeline

**What you will understand by the end of this chapter:**

- Byte-pair encoding's core algorithm: how an ordered merge table turns individual bytes into a hierarchical vocabulary, and why the merge with the LOWEST priority number among all currently-adjacent pairs fires first at encode time — a rule where merge PRIORITY, not merge OPPORTUNITY, decides the outcome, sometimes in ways that look surprising until traced by hand.
- Why BPE never runs directly on raw text: a pre-tokenizer splits text into word-sized chunks first, attaching leading whitespace to the FOLLOWING word rather than the preceding one, specifically so a merge rule can never fuse two different words together.
- How special tokens (beginning-of-text, end-of-text, role headers, end-of-turn) are inserted into a token sequence by integer ID at fixed structural positions defined by a chat template — never discovered by matching a substring of user-supplied text — and why that separation is a genuine security property, not an implementation detail.
- How decoding reverses encoding: byte-fallback tokens must emit their raw byte value rather than their display text, multi-byte UTF-8 codepoints fall out of that for free with no Unicode-aware code at all, and control tokens are either suppressed or used as a stop signal.
- How to extend Chapter 5's real, already-verified GGUF reader and writer with ARRAY-type support, and use it to load a complete, working tokenizer — vocabulary, merge table, special tokens, and all — from an actual file on disk, then run the whole pipeline on ordinary text and get back exactly what went in.

**What you need to know first:**

- Chapter 5's `GGUFWriter`/`GGUFReader` (Section 5.2), which this chapter extends with KV type ARRAY(9) rather than replaces — the scalar STRING/UINT32/FLOAT32 cases this chapter reuses are unchanged from that section.
- This is a purely single-threaded, string/vector/map-based chapter: no `-pthread`, no `std::mdspan`, no `-ffp-contract=off` for any file here — a deliberate simplification after three chapters (9, 10, 11) that all needed at least one of those.
- The general shape of a GGUF file's key-value metadata section from Chapter 5.1: a key string, a type tag, then a value in that type's wire format — this chapter's only new idea is that the value can itself be a whole array.

---

Every chapter so far has assumed tokens already exist as small integers flowing into an embedding table. This chapter builds the thing that produces them. Section 12.1 implements byte-pair encoding's actual merge loop — the algorithm that turns a stream of raw bytes into the hierarchical, always-representable vocabulary a real model was trained on — and traces a genuinely counterintuitive case by hand to show that merge PRIORITY, not merge OPPORTUNITY, decides the result. Section 12.2 builds the pre-tokenizer that runs before BPE ever sees a byte, so a merge rule can never accidentally fuse two words together. Section 12.3 turns to special tokens and the chat template that assembles a real multi-turn conversation, and makes concrete a security property this book has not needed to state before: special tokens are inserted by ID at fixed positions, never discovered by pattern-matching user text, so a user cannot inject a fake end-of-turn signal just by typing its literal string. Section 12.4 reverses the whole process with a token decoder, handling byte-fallback and control tokens correctly enough that a real multi-byte UTF-8 codepoint falls out of it with no Unicode-specific code at all. Section 12.5 closes the chapter by extending Chapter 5's real GGUF reader and writer with array support, writing an actual tokenizer vocabulary to a real file, and loading a complete, working tokenizer back out of it — proving the whole pipeline with a lossless round trip on ordinary text.

## 12.1 Byte-Pair Encoding: The Merge Algorithm

### Intuition

A BPE vocabulary is built once, offline, by repeatedly finding the most frequent adjacent pair of symbols in a training corpus and merging it into a new symbol, recording each merge in the order it was learned. That order becomes the vocabulary's most important property at encode time: it is not merely a training log, it is a priority list that determines, deterministically, which merge applies first whenever more than one is possible.

### The Concept, In Detail

Encoding starts from individual bytes — every one of the 256 possible byte values is always a valid token, which is what guarantees that ANY input, including bytes with no learned merge at all, is always representable with no "unknown token" failure mode. From there, encoding repeatedly scans every currently-adjacent pair of tokens, looks up each pair's merge priority (lower number means higher priority, since priorities are just index positions in the ordered merge table), and applies whichever applicable merge has the LOWEST priority number — not the pair that happens to occur first left-to-right, not the pair that would eventually lead to the longest possible token, but strictly the one with the highest-priority (lowest-numbered) rule. This process repeats until no adjacent pair has any applicable rule left. The result is that a vocabulary is hierarchical by construction: every token beyond the base 256 bytes is the concatenation of exactly two previously-existing tokens joined by one specific merge rule, and nothing else. A worked case makes the priority-over-opportunity rule concrete rather than abstract: given a small merge table where `(l,o)` has priority 0 and `(H,e)` has priority 6, encoding the literal string "Hello" does NOT produce a single "Hello" token even though a complete chain of rules to build it exists in the table (`H+e->He`, `l+l->ll`, `He+ll->Hell`, `Hell+o->Hello`). Instead, `(l,o)` — being strictly higher priority than `(H,e)` — fires first on the byte sequence `[H,e,l,l,o]`, consuming the second "l" and the trailing "o" into a "lo" token before "He" or "Hell" ever get the chance to form, leaving the final result as `["He", "l", "lo"]`. This is correct, deterministic BPE behavior, not a bug — the same priority rule that makes encoding fast and unambiguous also makes its output occasionally surprising to a reader who only checks whether a chain of rules COULD have produced a different, more familiar-looking result.

### Code and Verification

```cpp
// 01_bpe_merge_engine.cpp
// Chapter 12, Part 1: byte-pair encoding is the algorithm this book's
// tokenizer builds around, and its core loop is short enough to trace by
// hand. Training BPE (done once, offline, on a large corpus) repeatedly
// finds the most frequent adjacent pair of symbols and merges it into a
// new symbol; the OUTPUT of training is an ORDERED merge table, where
// order encodes priority (rule 0 was learned first and is applied first).
// Encoding, at inference time, never learns anything new -- it starts
// from individual bytes and greedily applies whichever rule in the
// (fixed, read-only) table has the highest priority among all pairs
// currently adjacent in the sequence, repeating until no rule applies.
//
// The vocabulary this book's tokenizer uses is hierarchical by
// construction: every token is either one of the 256 base byte tokens
// (always present, so any input is always representable -- "byte
// fallback") or the concatenation of exactly two previously-existing
// tokens, created by exactly one merge rule. This section builds and
// verifies the merge engine itself, deferring pre-tokenization (Section
// 12.2), special tokens (12.3), decoding (12.4), and loading a real
// vocabulary from a GGUF file (12.5) to their own sections.
//
// A genuinely instructive property this section verifies rather than
// assumes: merge PRIORITY, not merge OPPORTUNITY, decides the result.
// "Hello" contains both an (H,e) pair and an (l,o) pair with rules
// available; whichever rule has the LOWER priority number fires first,
// and that single choice can prevent a DIFFERENT, otherwise-available
// merge later from ever having the chance to apply, because the token it
// would have needed no longer exists in the sequence. This is not a bug
// to work around -- it is exactly why encoding is deterministic: the
// same merge table always resolves the same tie the same way.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_bpe_merge_engine.cpp -o 01_bpe_merge_engine

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// =========================================================================
// MERGE TABLE: an ordered list of pair rules, stored as a hash map from
// "left right" (the two token TEXTS, space-separated) to a priority rank
// -- lower rank means the rule was learned earlier and is applied first.
// Real GGUF files store this as an array of "left right" strings, in
// priority order (Section 12.5 loads exactly that format).
// =========================================================================
struct MergeTable {
    std::unordered_map<std::string, int> priority_of;

    void add(const std::string& left, const std::string& right, int priority) {
        priority_of[left + " " + right] = priority;
    }
    // Returns the rule's priority, or -1 if left+right has no merge rule.
    int lookup(const std::string& left, const std::string& right) const {
        auto it = priority_of.find(left + " " + right);
        return (it == priority_of.end()) ? -1 : it->second;
    }
    int size() const { return static_cast<int>(priority_of.size()); }
};

// =========================================================================
// VOCABULARY: every token the tokenizer knows, indexed by ID. Token ID is
// simply the token's position in the array it was added to -- the same
// convention `tokenizer.ggml.tokens` uses in a real GGUF file.
// =========================================================================
struct Vocabulary {
    std::vector<std::string> id_to_text;
    std::unordered_map<std::string, int> text_to_id;

    int add(const std::string& text) {
        int id = static_cast<int>(id_to_text.size());
        id_to_text.push_back(text);
        text_to_id[text] = id;
        return id;
    }
    // Returns the token's ID, or -1 if this exact text is not in the vocabulary.
    int lookup(const std::string& text) const {
        auto it = text_to_id.find(text);
        return (it == text_to_id.end()) ? -1 : it->second;
    }
    const std::string& text_of(int id) const { return id_to_text[id]; }
    int size() const { return static_cast<int>(id_to_text.size()); }
};

// =========================================================================
// THE MERGE ENGINE: given a plain string, split it into individual bytes,
// then repeatedly find the adjacent pair with the LOWEST priority number
// (= highest priority) among every pair the merge table has a rule for,
// and merge it -- an O(n^2) scan per encode, correct and fast enough for
// the short, pre-tokenized chunks Section 12.2 will hand it (a few dozen
// characters at most, never a whole document at once).
// =========================================================================
struct BPEEncoder {
    const MergeTable& merges;
    const Vocabulary& vocab;

    BPEEncoder(const MergeTable& m, const Vocabulary& v) : merges(m), vocab(v) {}

    std::vector<int> encode(const std::string& text) const {
        if (text.empty()) return {};

        // Step 1: start from individual bytes, each its own one-character token.
        std::vector<std::string> tokens;
        tokens.reserve(text.size());
        for (unsigned char c : text) tokens.push_back(std::string(1, static_cast<char>(c)));

        // Step 2: repeatedly apply the single highest-priority merge that
        // exists among all CURRENTLY adjacent pairs, until none does.
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
            if (best_pos < 0) break;   // no rule applies to any remaining pair

            tokens[best_pos] += tokens[best_pos + 1];
            tokens.erase(tokens.begin() + best_pos + 1);
        }

        // Step 3: look up each resulting token string's ID. Byte fallback:
        // this book's vocabulary always contains all 256 single bytes, so
        // a lookup miss here would mean an internal inconsistency, not a
        // real "unknown token" -- verified explicitly in Test 5 below.
        std::vector<int> ids;
        ids.reserve(tokens.size());
        for (const auto& t : tokens) {
            int id = vocab.lookup(t);
            ids.push_back(id);
        }
        return ids;
    }
};

// A small vocabulary and merge table, traceable by hand, matching this
// book's convention of building a tiny but structurally real fixture
// rather than a synthetic stand-in with no correspondence to a real
// tokenizer's shape.
void build_test_tokenizer(Vocabulary& vocab, MergeTable& merges) {
    for (int i = 0; i < 256; ++i) vocab.add(std::string(1, static_cast<char>(i)));

    struct Rule { std::string left, right, result; };
    const Rule rules[] = {
        {"l", "o", "lo"},       // 0
        {"lo", "w", "low"},     // 1
        {"e", "s", "es"},       // 2
        {"es", "t", "est"},     // 3
        {"e", "r", "er"},       // 4
        {"low", "er", "lower"}, // 5
        {"H", "e", "He"},       // 6
        {"l", "l", "ll"},       // 7
        {"He", "ll", "Hell"},   // 8
        {"Hell", "o", "Hello"}, // 9
        {"w", "o", "wo"},       // 10
        {"wo", "r", "wor"},     // 11
        {"l", "d", "ld"},       // 12
        {"wor", "ld", "world"}, // 13
    };
    for (int i = 0; i < static_cast<int>(std::size(rules)); ++i) {
        vocab.add(rules[i].result);
        merges.add(rules[i].left, rules[i].right, i);
    }
}

std::string ids_to_text(const std::vector<int>& ids, const Vocabulary& vocab) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(ids[i]) + " (\"" + vocab.text_of(ids[i]) + "\")";
    }
    return out + "]";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 12.1: Byte-Pair Encoding -- The Merge Algorithm\n";
    std::cout << "========================================================\n\n";

    Vocabulary vocab;
    MergeTable merges;
    build_test_tokenizer(vocab, merges);
    BPEEncoder encoder(merges, vocab);
    std::cout << "vocabulary: " << vocab.size() << " tokens (256 bytes + "
              << merges.size() << " merge rules)\n\n";

    // =====================================================================
    // TEST 1: "lowest" -> ["low", "est"], traced by hand in Section 12.1's
    // own prose: (l,o) fires first (priority 0), then (lo,w) (priority 1),
    // then (e,s) (priority 2), then (es,t) (priority 3); no rule exists
    // for (low, es), so the sequence stops at two tokens.
    // =====================================================================
    std::cout << "-- Test 1: \"lowest\" --\n";
    {
        auto ids = encoder.encode("lowest");
        std::cout << "  " << ids_to_text(ids, vocab) << "\n";
        CHECK(ids.size() == 2);
        CHECK(vocab.text_of(ids[0]) == "low");
        CHECK(vocab.text_of(ids[1]) == "est");
    }

    // =====================================================================
    // TEST 2: "lower" collapses to a single token -- every intermediate
    // merge this word needs (l+o, lo+w, e+r, low+er) exists in the table.
    // =====================================================================
    std::cout << "\n-- Test 2: \"lower\" --\n";
    {
        auto ids = encoder.encode("lower");
        std::cout << "  " << ids_to_text(ids, vocab) << "\n";
        CHECK(ids.size() == 1);
        CHECK(vocab.text_of(ids[0]) == "lower");
    }

    // =====================================================================
    // TEST 3: "world" also collapses to a single token via its own chain
    // of rules (w+o, wo+r, l+d, wor+ld).
    // =====================================================================
    std::cout << "\n-- Test 3: \"world\" --\n";
    {
        auto ids = encoder.encode("world");
        std::cout << "  " << ids_to_text(ids, vocab) << "\n";
        CHECK(ids.size() == 1);
        CHECK(vocab.text_of(ids[0]) == "world");
    }

    // =====================================================================
    // TEST 4: "Hello" does NOT collapse to a single token with this
    // table, even though a rule chain to "Hello" exists (H+e, l+l,
    // He+ll, Hell+o) -- because (l,o) has priority 0, strictly lower
    // than (H,e)'s priority 6 or (l,l)'s priority 7, and (l,o) fires
    // FIRST: [H,e,l,l,o] -> merge (l,o) at the last two positions ->
    // [H,e,l,lo]. That single merge consumes the second "l" that
    // "He+ll" would have needed, so "ll" can never form, and neither can
    // "Hell" or "Hello" -- only (H,e) still applies, giving [He,l,lo].
    // This is not a defect in the engine; it is the deterministic,
    // priority-driven behavior this book's decoder (Section 12.4) and
    // every real BPE tokenizer depend on.
    // =====================================================================
    std::cout << "\n-- Test 4: \"Hello\" (priority ordering, not opportunity, decides the result) --\n";
    {
        auto ids = encoder.encode("Hello");
        std::cout << "  " << ids_to_text(ids, vocab) << "\n";
        std::cout << "  (l,o) has priority 0, lower than (H,e)'s priority 6 -- it fires first and\n";
        std::cout << "  consumes the 'o' that (Hell,o) -> \"Hello\" would have needed.\n";
        CHECK(ids.size() == 3);
        CHECK(vocab.text_of(ids[0]) == "He");
        CHECK(vocab.text_of(ids[1]) == "l");
        CHECK(vocab.text_of(ids[2]) == "lo");
    }

    // =====================================================================
    // TEST 5: bytes with no applicable merge rule at all pass straight
    // through as their own byte tokens -- "byte fallback" in miniature.
    // Section 12.5's real vocabulary always contains all 256 bytes, so
    // this lookup can never fail; checked here directly.
    // =====================================================================
    std::cout << "\n-- Test 5: no rule applies -- bytes pass through unmerged --\n";
    {
        auto ids = encoder.encode("xyz");
        std::cout << "  \"xyz\" -> " << ids_to_text(ids, vocab) << "\n";
        CHECK(ids.size() == 3);
        CHECK(ids[0] == static_cast<int>('x'));
        CHECK(ids[1] == static_cast<int>('y'));
        CHECK(ids[2] == static_cast<int>('z'));
    }

    // =====================================================================
    // TEST 6: empty input encodes to an empty ID sequence.
    // =====================================================================
    std::cout << "\n-- Test 6: empty input --\n";
    {
        auto ids = encoder.encode("");
        CHECK(ids.empty());
        std::cout << "  \"\" -> []\n";
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_bpe_merge_engine.cpp -o 01_bpe_merge_engine
./01_bpe_merge_engine
```

**Sample input:** a 270-token test vocabulary (256 base byte tokens plus 14 merge rules building up "lower" and "world" letter by letter) encoding six strings — "lowest", "lower", "world", "Hello", the never-merged "xyz", and an empty string — checked against hand-traced expected token sequences for each.

```text
========================================================
Chapter 12.1: Byte-Pair Encoding -- The Merge Algorithm
========================================================

vocabulary: 270 tokens (256 bytes + 14 merge rules)

-- Test 1: "lowest" --
  [257 ("low"), 259 ("est")]

-- Test 2: "lower" --
  [261 ("lower")]

-- Test 3: "world" --
  [269 ("world")]

-- Test 4: "Hello" (priority ordering, not opportunity, decides the result) --
  [262 ("He"), 108 ("l"), 256 ("lo")]
  (l,o) has priority 0, lower than (H,e)'s priority 6 -- it fires first and
  consumes the 'o' that (Hell,o) -> "Hello" would have needed.

-- Test 5: no rule applies -- bytes pass through unmerged --
  "xyz" -> [120 ("x"), 121 ("y"), 122 ("z")]

-- Test 6: empty input --
  "" -> []

========================================================
16/16 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming a complete merge chain to a token means encoding will produce that token"
    It is tempting to check whether a vocabulary CONTAINS a token like "Hello" — built from a real, valid chain of merge rules — and conclude that encoding the string "Hello" must produce it. The merge loop does not search for the longest reachable token or prefer chains that terminate in a token matching the whole input; at every step it applies whichever SINGLE applicable merge has the lowest priority number, with no lookahead at all. If some other pair in the current token sequence has a lower priority number than the next step of "Hello"'s own chain would need, that other pair merges first, and it can consume a byte "Hello"'s chain needed for itself — exactly what happens when `(l,o)`'s priority-0 rule consumes the second "l" and the "o" before `(He,ll)`'s priority-8 rule ever gets a turn. The vocabulary containing a token is necessary but never sufficient for encoding to produce it; only tracing the actual priority-ordered merge sequence tells you what a given input actually becomes.

## 12.2 Pre-Tokenization: Splitting Text Before BPE

### Intuition

If BPE ran directly on raw, unsplit text, nothing would stop a merge rule from fusing the last character of one word with the first character of the next — or with the space between them. A pre-tokenizer draws the boundaries BPE is never allowed to cross, splitting text into word-sized chunks before the merge loop ever runs, with BPE then operating independently inside each chunk.

### The Concept, In Detail

The convention this section implements — matching Llama 3, GPT-4, and most modern BPE tokenizers — classifies each character as alphabetic, digit, whitespace, or punctuation, and groups runs of the same class together with one deliberate asymmetry: leading whitespace is ABSORBED into the chunk that follows it, never attached to the chunk before it. "the cat" therefore splits into `["the", " cat"]`, not `["the ", "cat"]`. This is not cosmetic: it means the token the model learns for "cat" after a space is genuinely a different token from "cat" at the very start of input, and the vocabulary must — and does, in a real trained tokenizer — learn both separately. Digit runs are grouped the same way, absorbing leading whitespace exactly like alphabetic runs. Punctuation is never grouped at all: each individual punctuation character becomes its own one-character chunk, so "wow!!!" splits into four chunks, not two. Whitespace that has nothing left to absorb into (because it sits at the very end of the input) becomes its own trailing chunk instead of being silently dropped. A production tokenizer's real pre-tokenization pattern additionally handles full Unicode letter categories and specific contractions (splitting "don't" into "don" and "'t"); this section's simplified ASCII-only scanner covers the same four-way structural idea those richer rules extend.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_pretokenizer.cpp -o 02_pretokenizer
./02_pretokenizer
```

**Sample input:** six test sentences and edge cases — a sentence with punctuation, a three-word sentence checking leading-whitespace attachment, a sentence with a digit group, repeated punctuation, empty and whitespace-only input, and a sentence confirming pre-tokenization hands BPE one clean chunk per word.

```text
========================================================
Chapter 12.2: Pre-Tokenization -- Splitting Text Before BPE
========================================================

-- Test 1: "Hello, world!" --
  ["Hello", ",", " world", "!"]

-- Test 2: "the cat sat" (leading space attaches to the following word) --
  ["the", " cat", " sat"]

-- Test 3: "I have 42 cats" (digit runs grouped) --
  ["I", " have", " 42", " cats"]

-- Test 4: "wow!!!" (each punctuation mark is its own chunk) --
  ["wow", "!", "!", "!"]

-- Test 5: edge cases --
  "" -> []
  "   " -> ["   "] (trailing whitespace with nothing to absorb into)

-- Test 6: a chunk boundary is not a merge boundary --
  ["lowest", " wins"]

========================================================
25/25 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] attaching whitespace to the wrong side of a word"
    It is easy to write a pre-tokenizer that attaches trailing whitespace to the word BEFORE it — "the " then "cat" — because that is how splitting on whitespace naturally falls out of the most obvious scanning approach (consume a word, then consume the space that follows). Real tokenizers do the opposite on purpose: the space belongs to the word that FOLLOWS it. Getting this backwards does not merely produce a cosmetically different chunk boundary — it silently changes which token IDs BPE ends up producing for every word in a sentence except the first, because a token vocabulary trained on "word-with-leading-space" tokens has no matching entry for a "word-with-trailing-space" token, and the encoder falls back to a completely different (and much less efficient) tokenization for text that would otherwise have been a single common token.

## 12.3 Special Tokens and the Chat Template

### Intuition

Special tokens tell a model where a prompt begins, whose turn it is, and where a turn ends — structural scaffolding no amount of BPE merging can produce, because they are never present as ordinary text for BPE to encode in the first place. A chat template assembles them, by integer ID, around ordinary BPE-encoded text.

### The Concept, In Detail

A Llama-3-style conversation begins with a single begin-of-text token, followed by one CLOSED turn per message — a start-header token, the role name ("system", "user", or "assistant") encoded as perfectly ordinary text through the same `BPEEncoder` any other string would use, an end-header token, two literal newline characters, the message content (also ordinary BPE-encoded text), and an end-of-turn token — and ends with one OPEN final assistant header (start-header, "assistant", end-header, two newlines) that deliberately has no content and no end-of-turn token, because that missing continuation is exactly what the model is being asked to generate. The special-token IDs themselves — begin-of-text, end-of-text, start-header, end-header, end-of-turn — are inserted as raw integer constants at these fixed structural positions; nowhere in that assembly does any code inspect the CONTENTS of a message looking for a substring that matches a special token's name. This is a genuine security property, not an implementation detail: if a user's message literally contains the characters `<|end_of_text|>`, typed as ordinary text — whether by accident or as a deliberate attempt to inject a fake end-of-turn signal — the content-encoding path treats it exactly like any other text, running it through the same byte-fallback-guaranteed `BPEEncoder::encode` as everything else, and it can never become the actual integer ID that would end the sequence. There is no string-matching code path from user content to a special-token ID for an attacker to find in the first place.

### Code and Verification

```cpp
// 03_special_tokens_chat_template.cpp
// Chapter 12, Part 3: special tokens are the scaffolding that tells a model
// where a prompt begins, whose turn it is, and where a turn ends. They are
// never produced by the BPE merge loop -- they are inserted directly, by
// integer ID, by a chat template that wraps the user's raw text in the
// markup the model was trained on. This file reuses this chapter's own
// verified MergeTable / Vocabulary / BPEEncoder (Section 12.1) unchanged
// for encoding ordinary role and content text, and adds the special-token
// IDs and the ChatTemplateEncoder that assembles a full Llama-3-style
// multi-turn sequence around it.
//
// Real Llama 3 special token IDs (for context; this file does not download
// or verify against an actual model file -- that is deferred to a later
// chapter that loads a real GGUF vocabulary):
//   <|begin_of_text|>    = 128000  (BOS, appears exactly once, at position 0)
//   <|end_of_text|>      = 128001  (EOS)
//   <|start_header_id|>  = 128006  (opens a role header: "system"/"user"/"assistant")
//   <|end_header_id|>    = 128007  (closes a role header)
//   <|eot_id|>            = 128009  (end of one turn)
//
// A Llama 3 turn is: <|start_header_id|> + role + <|end_header_id|> +
// "\n\n" + content + <|eot_id|>. A conversation is BOS once, followed by
// one such turn per message, followed by an OPEN final assistant header
// (start_header_id + "assistant" + end_header_id + "\n\n") with no content
// and no eot_id -- that missing content is exactly what the model is asked
// to generate.
//
// [COMMON TRAP] this file makes concrete: the special-token IDs above are
// never looked up by matching a string. ChatTemplateEncoder inserts them
// as raw integer constants at fixed structural positions; the ONLY path
// user-supplied text ever takes is through BPEEncoder::encode(), which
// knows nothing about special tokens at all. So if a user's message
// literally contains the text "<|end_of_text|>" -- typed as ordinary
// characters, perhaps by an attacker trying to inject a fake end-of-turn
// signal -- it is BPE/byte-fallback encoded like any other text and can
// never become the integer 128001. Test 4 below demonstrates this by
// encoding that exact literal string as message content and checking
// that none of the resulting token IDs is any of the five special IDs.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_special_tokens_chat_template.cpp -o 03_special_tokens_chat_template

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// ---------------------------------------------------------------------
// Section 12.1's MergeTable / Vocabulary / BPEEncoder / build_test_tokenizer,
// reused verbatim (already compiled, run, and locked in 01_bpe_merge_engine.cpp).
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

void build_test_tokenizer(Vocabulary& vocab, MergeTable& merges) {
    for (int i = 0; i < 256; ++i) vocab.add(std::string(1, static_cast<char>(i)));
    struct Rule { std::string left, right, result; };
    const Rule rules[] = {
        {"l", "o", "lo"}, {"lo", "w", "low"}, {"e", "s", "es"}, {"es", "t", "est"},
        {"e", "r", "er"}, {"low", "er", "lower"}, {"H", "e", "He"}, {"l", "l", "ll"},
        {"He", "ll", "Hell"}, {"Hell", "o", "Hello"}, {"w", "o", "wo"},
        {"wo", "r", "wor"}, {"l", "d", "ld"}, {"wor", "ld", "world"},
    };
    for (int i = 0; i < static_cast<int>(std::size(rules)); ++i) {
        vocab.add(rules[i].result);
        merges.add(rules[i].left, rules[i].right, i);
    }
}

// ---------------------------------------------------------------------
// Special tokens and the chat template (new in this section).
// ---------------------------------------------------------------------
namespace special {
constexpr int32_t BEGIN_OF_TEXT   = 128000;
constexpr int32_t END_OF_TEXT     = 128001;
constexpr int32_t START_HEADER_ID = 128006;
constexpr int32_t END_HEADER_ID   = 128007;
constexpr int32_t EOT_ID          = 128009;
}  // namespace special

struct ChatMessage {
    std::string role;     // "system", "user", or "assistant"
    std::string content;
};

// Builds a Llama-3-style token ID sequence: BOS once, then one closed turn
// per message (header + role + header + "\n\n" + content + eot), then an
// OPEN final assistant header (no content, no eot) inviting generation.
// Special-token IDs are inserted directly as integer constants; role and
// content text are the ONLY things that ever pass through BPEEncoder --
// there is no code path anywhere in this function that maps a substring
// of `content` to a special-token ID by string comparison.
struct ChatTemplateEncoder {
    const BPEEncoder& encoder;
    explicit ChatTemplateEncoder(const BPEEncoder& enc) : encoder(enc) {}

    std::vector<int> encode_conversation(const std::vector<ChatMessage>& messages) const {
        std::vector<int> ids;
        ids.push_back(special::BEGIN_OF_TEXT);
        for (const auto& msg : messages) {
            append_turn(ids, msg.role, msg.content, /*close_turn=*/true);
        }
        // Final open assistant header: no content, no eot_id.
        append_turn(ids, "assistant", "", /*close_turn=*/false);
        return ids;
    }

private:
    void append_turn(std::vector<int>& ids, const std::string& role,
                      const std::string& content, bool close_turn) const {
        ids.push_back(special::START_HEADER_ID);
        for (int id : encoder.encode(role)) ids.push_back(id);
        ids.push_back(special::END_HEADER_ID);
        for (int id : encoder.encode("\n\n")) ids.push_back(id);
        if (close_turn) {
            for (int id : encoder.encode(content)) ids.push_back(id);
            ids.push_back(special::EOT_ID);
        }
    }
};

bool is_special_id(int id) {
    return id == special::BEGIN_OF_TEXT || id == special::END_OF_TEXT ||
           id == special::START_HEADER_ID || id == special::END_HEADER_ID ||
           id == special::EOT_ID;
}

int count_occurrences(const std::vector<int>& ids, int target) {
    int n = 0;
    for (int id : ids) if (id == target) ++n;
    return n;
}

int main() {
    Vocabulary vocab;
    MergeTable merges;
    build_test_tokenizer(vocab, merges);
    BPEEncoder encoder(merges, vocab);
    ChatTemplateEncoder chat(encoder);

    std::cout << "========================================================\n";
    std::cout << "Chapter 12.3: Special Tokens and the Chat Template\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: single-turn conversation (system + user) --\n";
    {
        std::vector<ChatMessage> conv = {
            {"system", "world"},
            {"user", "lower"},
        };
        auto ids = chat.encode_conversation(conv);
        std::cout << "  " << ids.size() << " token IDs total\n";
        CHECK(ids.front() == special::BEGIN_OF_TEXT);
        CHECK(count_occurrences(ids, special::BEGIN_OF_TEXT) == 1);
        CHECK(count_occurrences(ids, special::START_HEADER_ID) == 3);  // system, user, final assistant
        CHECK(count_occurrences(ids, special::END_HEADER_ID) == 3);
        CHECK(count_occurrences(ids, special::EOT_ID) == 2);           // system turn, user turn (not the open assistant turn)
        CHECK(ids.back() != special::EOT_ID);  // final assistant header is left open
    }

    std::cout << "\n-- Test 2: multi-turn conversation keeps exactly one BOS --\n";
    {
        std::vector<ChatMessage> conv = {
            {"system", "world"},
            {"user", "lowest"},
            {"assistant", "lower"},
            {"user", "world"},
        };
        auto ids = chat.encode_conversation(conv);
        CHECK(count_occurrences(ids, special::BEGIN_OF_TEXT) == 1);
        CHECK(count_occurrences(ids, special::EOT_ID) == 4);  // one per closed message
        CHECK(count_occurrences(ids, special::START_HEADER_ID) == 5);  // 4 messages + final open assistant
        std::cout << "  " << ids.size() << " token IDs, 1 BOS, 4 eot_id, 5 header pairs\n";
    }

    std::cout << "\n-- Test 3: role text is ordinary BPE, not a lookup table --\n";
    {
        // "assistant" is encoded exactly the way BPEEncoder::encode would
        // encode it as free-standing text -- there is no separate role enum.
        auto direct = encoder.encode("assistant");
        std::vector<ChatMessage> conv = {{"assistant", "world"}};
        auto ids = chat.encode_conversation(conv);
        // ids layout: [BOS, START_HEADER_ID, <direct role ids...>, END_HEADER_ID, ...]
        std::vector<int> extracted(ids.begin() + 2, ids.begin() + 2 + static_cast<int>(direct.size()));
        CHECK(extracted == direct);
        std::cout << "  role text \"assistant\" round-trips through the same encoder as any other text\n";
    }

    std::cout << "\n-- Test 4 [COMMON TRAP]: a literal special-token string in user content --\n";
    {
        // A user (or an attacker) types the literal characters
        // "<|end_of_text|>" as message content, hoping the encoder will
        // treat it as the real end-of-text control token (ID 128001).
        std::string injected = "<|end_of_text|>";
        std::vector<ChatMessage> conv = {{"user", injected}};
        auto ids = chat.encode_conversation(conv);

        // The full conversation's structure is unaffected by what the
        // injected text says: still exactly one BOS and exactly one
        // eot_id (closing this single user turn), no matter what the
        // attacker's string claims to be.
        CHECK(count_occurrences(ids, special::BEGIN_OF_TEXT) == 1);
        CHECK(count_occurrences(ids, special::EOT_ID) == 1);

        // Special IDs (>= 128000) legitimately appear at those fixed
        // structural positions; what must NEVER happen is the injected
        // text ITSELF producing one. Check that directly: encode the
        // content in isolation and confirm none of its ids are special.
        bool any_special_leaked = false;
        auto content_ids = encoder.encode(injected);
        for (int id : content_ids) {
            if (is_special_id(id)) any_special_leaked = true;
        }
        CHECK(!any_special_leaked);

        // And the content must round-trip back to the exact original bytes
        // via ordinary vocabulary lookup -- proving it was preserved as
        // text, not silently swallowed into a control token.
        std::string reconstructed;
        for (int id : content_ids) reconstructed += vocab.text_of(id);
        CHECK(reconstructed == injected);

        std::cout << "  content \"" << injected << "\" -> " << content_ids.size()
                  << " ordinary tokens, 0 special IDs, exact byte round-trip\n";
    }

    std::cout << "\n-- Test 5: empty system prompt still produces a well-formed turn --\n";
    {
        std::vector<ChatMessage> conv = {{"system", ""}};
        auto ids = chat.encode_conversation(conv);
        CHECK(count_occurrences(ids, special::EOT_ID) == 1);
        CHECK(ids.front() == special::BEGIN_OF_TEXT);
        std::cout << "  empty content still closes with eot_id (" << ids.size() << " ids total)\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_special_tokens_chat_template.cpp -o 03_special_tokens_chat_template
./03_special_tokens_chat_template
```

**Sample input:** a single-turn system+user conversation checking token-count structure; a four-message multi-turn conversation checking exactly one begin-of-text token appears no matter how many turns follow; role text encoded through the ordinary BPE path; a literal `<|end_of_text|>` string typed as user content, checked to confirm it produces zero special-token IDs and round-trips back to its exact original bytes; and an empty system prompt that still produces a well-formed, correctly closed turn.

```text
========================================================
Chapter 12.3: Special Tokens and the Chat Template
========================================================

-- Test 1: single-turn conversation (system + user) --
  35 token IDs total

-- Test 2: multi-turn conversation keeps exactly one BOS --
  60 token IDs, 1 BOS, 4 eot_id, 5 header pairs

-- Test 3: role text is ordinary BPE, not a lookup table --
  role text "assistant" round-trips through the same encoder as any other text

-- Test 4 [COMMON TRAP]: a literal special-token string in user content --
  content "<|end_of_text|>" -> 15 ordinary tokens, 0 special IDs, exact byte round-trip

-- Test 5: empty system prompt still produces a well-formed turn --
  empty content still closes with eot_id (25 ids total)

========================================================
16/16 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] trusting user-supplied text to carry control information"
    A chat template that — anywhere in its content-handling path — checks whether a user's message text matches a special token's string and, if so, treats it as that special token, has reintroduced exactly the vulnerability this section's design avoids: a user can now end a conversation early, forge a fake system turn, or otherwise manipulate the model's context simply by typing the right literal string. The fix is not sanitizing or escaping that string inside user content — it is never giving user content a code path to a special-token ID at all. This section's `ChatTemplateEncoder` inserts special-token IDs only as fixed integer constants at structural positions ITS OWN code controls, and routes every piece of user-supplied text, with no exception, through the ordinary `BPEEncoder` that has no notion of special tokens whatsoever. Structural separation, not input filtering, is what makes the injection impossible rather than merely unlikely.

## 12.4 The Token Decoder: IDs Back to UTF-8

### Intuition

Decoding reverses encoding: given the integer IDs a model produces, reconstruct the text they represent. This looks like a plain table lookup, but byte-fallback tokens and control tokens both need handling beyond "look up the string and concatenate."

### The Concept, In Detail

A normal token's decoded text is just its stored string. A byte-fallback token is different: for IDs 0 through 255, the ID itself IS the raw byte value the token represents, so decoding emits that single byte directly rather than a display string like "<0xC3>" that only exists for human-readable debugging output. This distinction matters most for multi-byte UTF-8: a codepoint like U+00E9 (an "e" with an acute accent) is two bytes, 0xC3 and 0xA9, in UTF-8. If no merge rule ever joined those two specific bytes into a single token — entirely plausible for a character that is rare in a small or specialized training corpus — they arrive at decode time as two separate byte-fallback tokens, IDs 195 and 169. The decoder does not need one line of Unicode-aware code to handle this correctly: it emits byte 195, then emits byte 169, and the two raw bytes it wrote, adjacent and in the right order, ARE a valid UTF-8 encoding of that codepoint, purely as a consequence of always emitting a byte token's actual byte value. Control tokens are handled by a third rule: by default they are suppressed entirely (never appearing in decoded text a user would see), though a decoder can optionally render their display text for debugging. A control token can also serve as a stop signal — `decode_with_stop` halts the instant it encounters a chosen ID (typically end-of-text) and discards everything the sequence contains after that point, since a model's output past its own end-of-text signal was never meant to be read as generated text at all.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_token_decoder.cpp -o 04_token_decoder
./04_token_decoder
```

**Sample input:** six tests — basic normal-token decoding, byte-fallback decoding of two ASCII bytes, control tokens suppressed by default and visible in a debug mode, `decode_with_stop` discarding text generated after an end-of-text token, normal and byte tokens mixed in one sequence, and a real two-byte UTF-8 codepoint (U+00E9) reconstructed from two separate byte-fallback tokens with no Unicode-specific code at all.

```text
========================================================
Chapter 12.4: The Token Decoder -- IDs Back to UTF-8
========================================================

-- Test 1: basic decode of normal tokens --
  [256, 257, 258] -> "Hello world!"

-- Test 2: byte-fallback decode --
  [72, 105] (byte tokens for 'H', 'i') -> "Hi"

-- Test 3: control tokens suppressed by default, visible in debug mode --
  default:  "Hello world!"
  debug:    "<|begin_of_text|>Hello world!<|end_of_text|>"

-- Test 4: decode_with_stop discards anything after EOS --
  text: "Hello world", hit_eos: yes

-- Test 5: normal and byte tokens mixed in one sequence --
  [256, 33] -> "Hello!"

-- Test 6: a real multi-byte UTF-8 codepoint via two byte-fallback tokens --
  [99,97,102,195,169] -> byte sequence (hex): 63 61 66 c3 a9  (3 ASCII bytes + one 2-byte UTF-8 codepoint, U+00E9)

========================================================
10/10 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] decoding a byte-fallback token's DISPLAY text instead of its byte value"
    A byte-fallback token's vocabulary entry is commonly given a human-readable display string like "<0xC3>" purely so a developer inspecting the vocabulary can tell what it is. It is a real and easy mistake to decode that token by emitting its STORED STRING — literally the seven characters "<0xC3>" — instead of the single raw byte 0xC3 it represents. The bug is invisible for plain ASCII text, where display strings and intended output rarely collide in an obviously broken way, but it corrupts every multi-byte UTF-8 sequence and every byte value a model actually needs decoded as raw output: instead of two bytes reassembling into one accented character, the user sees the literal text "<0xC3><0xA9>" where a single letter belonged. The fix this section relies on is a single, explicit special case: for byte-fallback tokens with IDs 0-255, the numeric ID itself is the byte to emit, and the display string exists for debugging output only, never for the actual decoded byte stream.

## 12.5 A Complete Tokenizer Pipeline Loaded from a Real GGUF File

### Intuition

A tokenizer is only as real as the file it can be loaded from. Chapter 5's GGUF reader and writer already handle a model's scalar architecture metadata; a vocabulary of thousands of strings needs GGUF's ARRAY type instead, and this section extends that same real reader and writer with exactly the array support a tokenizer's metadata requires.

### The Concept, In Detail

GGUF's KV type tag 9 marks an ARRAY value: after the ordinary key string and the type tag 9 itself, the wire format adds one more type tag identifying the ARRAY'S ELEMENT type, then a count, then that many elements back to back in that element type's own wire format. A real tokenizer's vocabulary needs exactly three element types: `tokenizer.ggml.tokens` is an ARRAY of STRING, one entry per token ID, in ID order; `tokenizer.ggml.token_type` is an ARRAY of INT32, classifying each token by the same convention this chapter's decoder already uses (1 for normal, 3 for control, 6 for byte-fallback); and `tokenizer.ggml.scores` is an ARRAY of FLOAT32 — a field SentencePiece-style tokenizers use for merge-probability weighting that this chapter's BPE tokenizer simply ignores, written as all zeros here purely so the file matches the format a real loader expects to find. `tokenizer.ggml.merges` is a fourth ARRAY of STRING, one "left right" space-separated pair per merge rule, in priority order — the exact same ordering this chapter's `MergeTable` has relied on internally since Section 12.1, just serialized. Two more fields, `tokenizer.ggml.bos_token_id` and `eos_token_id`, are ordinary scalar UINT32 KVs, unchanged from Chapter 5's original writer. This section's own verification found a real bug worth naming rather than quietly fixing: the file's declared KV count (`n_kv`) must exactly match the number of key-value pairs actually written, and getting it wrong by even one causes the reader to silently stop parsing one field early — every KV before the shortfall reads back correctly, and the reader reports success, but the last field written is never even attempted and its lookup quietly returns a default value with no error raised anywhere. With that extended reader and writer verified against each other, this section builds a complete, working `Vocabulary`, `MergeTable`, `BPEEncoder`, and `TokenDecoder` ENTIRELY from what the array-aware reader loads off disk — no in-memory shortcut — and runs Section 12.2's pre-tokenizer, that reconstructed encoder, and that reconstructed decoder together on an ordinary sentence, confirming the exact original text comes back out the other end.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_gguf_tokenizer_pipeline.cpp -o 05_gguf_tokenizer_pipeline
./05_gguf_tokenizer_pipeline
```

**Sample input:** a real 272-token GGUF file (Section 12.1's 270-token BPE vocabulary plus two control tokens for BOS/EOS), written with `tokenizer.ggml.tokens`/`token_type`/`scores`/`merges` arrays and BOS/EOS scalar IDs, reopened fresh, checked element-for-element against what was written, then used to rebuild a complete working encoder and decoder that reproduce Section 12.1's own worked cases exactly and round-trip the sentence "the lowest world" losslessly through pre-tokenization, BPE encoding, and decoding.

```text
========================================================
Chapter 12.5: A Complete Tokenizer Pipeline From a Real GGUF File
========================================================

-- Test 1: writing the file --
  wrote 272 tokens, 14 merge rules, 5189 bytes total

-- Test 2: reopening the file fresh and parsing the header --
  magic ok, version=3, n_tensors=0, n_kv=7

-- Test 3: array metadata matches exactly, element for element --
  tokens=272 token_type=272 scores=272 merges=14
  bos_token_id=270 eos_token_id=271

-- Test 4: a working encoder rebuilt entirely from the file --
  "lowest" -> 2 tokens (matches in-memory ground truth)
  "Hello" -> 3 tokens (matches in-memory ground truth)

-- Test 5: a working decoder rebuilt entirely from the file --
  "world" -> 1 token(s) -> "world" (exact round trip)

-- Test 6: the full pipeline -- pretokenize + encode + decode -- on ordinary text --
  "the lowest world" -> 3 pre-token chunks -> 8 token IDs -> "the lowest world"
  lossless: yes

-- Test 7: decode_with_stop using the BOS/EOS IDs read from the file --
  decoded up to eos_token_id: "lower", hit_eos=yes

========================================================
24/24 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] a wrong KV count that the reader never reports as an error"
    Writing a GGUF file's `n_kv` field as the number of key-value pairs the AUTHOR INTENDED to write, rather than the number the code actually ends up writing, is an easy off-by-one to introduce — especially when, as in this section's own first draft, a field gets added to the writer calls after the `n_kv` constant was already set. The reader has no way to detect the mismatch: it reads exactly `n_kv` key-value pairs, succeeds, and returns normally, having simply never attempted to read the trailing field the file's byte layout actually still contains. Every lookup for that missing field then returns its type's default value — 0 for a UINT32 — with no exception, no false return, and no diagnostic of any kind. This is precisely why this chapter's own discipline checks every written field against what comes back element-for-element rather than merely checking that `reader.open()` returned `true`: a reader reporting success is not the same claim as a reader having read everything the writer actually wrote.

## Chapter Summary

This chapter built the component every prior chapter's "token ID" assumed already existed: a complete tokenizer, from raw bytes to model-ready integers and back. Section 12.1 implemented byte-pair encoding's actual merge loop and traced a genuinely counterintuitive case by hand to establish that merge PRIORITY, not merge OPPORTUNITY, decides what a given input encodes to — a vocabulary containing a token is never sufficient proof that encoding will produce it. Section 12.2 built the pre-tokenizer that splits text into word-sized chunks before BPE ever runs, with the specific, deliberate convention of attaching leading whitespace to the following word rather than the preceding one. Section 12.3 assembled a real chat template around that encoder and established a genuine security property: special tokens are inserted by integer ID at fixed structural positions the template itself controls, never discovered by matching a substring of user-supplied text, so a literal special-token string typed as ordinary content can never become the real control token. Section 12.4 reversed the process with a token decoder correct enough that a real multi-byte UTF-8 codepoint falls out of plain byte-fallback handling with no Unicode-specific logic anywhere. Section 12.5 closed the chapter by extending Chapter 5's real GGUF reader and writer with array support, uncovering and documenting a genuine off-by-one hazard in a file's declared KV count along the way, and proved the whole pipeline by loading a complete, working tokenizer from an actual file and round-tripping ordinary text through it losslessly. Every future chapter's "token ID" now names something this book has actually built, verified, and can point to.

## Self-Check Questions

1. Section 12.1 defines a merge table where `(l,o)` has priority 0 and `(H,e)` has priority 6. Why does encoding "Hello" NOT produce a single "Hello" token, even though a complete chain of merge rules to build "Hello" exists in the table?
2. Why does starting BPE encoding from individual bytes (IDs 0-255) guarantee that any input, including text with no applicable merge rule at all, is always representable?
3. In Section 12.2's pre-tokenizer, why does "the cat" split into `["the", " cat"]` rather than `["the ", "cat"]`, and why does that specific choice matter for what the vocabulary has to learn?
4. Why is each individual punctuation character its own chunk in Section 12.2's pre-tokenizer, rather than grouping consecutive punctuation characters together the way alphabetic and digit runs are grouped?
5. Section 12.3's `ChatTemplateEncoder` inserts special-token IDs as raw integer constants rather than looking them up by matching text. Why does this design choice prevent a user from injecting a fake end-of-turn signal just by typing `<|end_of_text|>` as ordinary message content?
6. In Section 12.3, why does a conversation's final assistant turn deliberately have no content and no end-of-turn token, unlike every other closed turn in the conversation?
7. Section 12.4's decoder treats byte-fallback tokens with IDs 0-255 specially: it emits the raw byte the ID represents rather than the token's stored display string. Why does getting this backwards (emitting the display string instead) corrupt multi-byte UTF-8 output specifically, even though it might look harmless for plain ASCII?
8. Why does decoding two separate byte-fallback tokens (195 and 169) in sequence correctly reconstruct a valid two-byte UTF-8 codepoint, with no Unicode-aware code in the decoder at all?
9. Section 12.5 extends Chapter 5's GGUF reader with an ARRAY type. What are the three pieces of information the wire format for an ARRAY value must encode, beyond the ordinary key string and type tag every KV pair already has?
10. Section 12.5 discovered that writing the wrong `n_kv` count causes the reader to silently miss the last field written, with `reader.open()` still returning `true` and no exception raised anywhere. Why is this specific failure mode — a reader that reports success while having read less than what was written — more dangerous than a reader that fails outright, and what verification habit in this chapter is specifically designed to catch it?

## Where We Go Next

This chapter gave every future chapter a real tokenizer to assume: text goes in as a chunked, BPE-encoded, chat-templated sequence of integer IDs, and comes back out losslessly through a decoder that handles byte-fallback and control tokens correctly. Chapter 13 turns to what happens to those token IDs once a model has actually processed them: a KV cache that must grow across a long generation, get evicted intelligently when memory runs out, and stay correct across many concurrent requests sharing a single serving process — the state a production inference server has to manage that a single decode step, however well-parallelized, does not.

## Worked Solutions

**1.** Encoding scans every currently-adjacent pair at each step and applies whichever APPLICABLE pair has the lowest priority number, with no lookahead toward any particular target token. On the byte sequence `[H,e,l,l,o]`, the pair `(l,o)` has priority 0 — strictly lower (higher-priority) than `(H,e)`'s priority 6 — so `(l,o)` merges first, consuming the second "l" and the "o" into a "lo" token. That consumes exactly the "l" and "o" that "Hell"+"o" -> "Hello" (priority 9) would have needed as its own input, so by the time any rule building toward "Hello" could apply, the bytes it depended on are already gone. The final result is `["He","l","lo"]`: a complete chain to "Hello" existing in the table never mattered, because encoding never searches chains — it only ever applies the single highest-priority applicable merge at each step.

**2.** Every one of the 256 possible byte values is added to the vocabulary as a base token before any merge rule is ever considered, so any sequence of bytes — however unusual, however absent from the training data used to learn merge rules — can always be represented as, at minimum, its own sequence of individual byte tokens even if zero merge rules apply to it. This is what BPE's "byte fallback" guarantee actually means: there is no code path in the encoder that can fail to produce SOME valid token sequence for SOME input, because the base case (one token per byte) always exists and is always tried first before any merge is even attempted.

**3.** Attaching the leading space to the FOLLOWING word means the token for "cat" occurring after a space (" cat") is a completely different vocabulary entry from the token for "cat" occurring at the very start of input ("cat") or after punctuation. A real trained tokenizer's vocabulary contains BOTH as separate, independently-learned tokens, because the two contexts are common enough separately to each deserve their own single-token representation, and a model's learned embeddings for the two differ. Attaching whitespace to the preceding word instead ("the ") would instead need a vocabulary of word-plus-trailing-space tokens, which is not the convention any of the modern BPE tokenizers this section models were actually trained with — getting the side backwards produces chunks a real trained vocabulary was never built to recognize.

**4.** Punctuation characters are semantically and positionally independent of each other far more often than letters or digits are — three consecutive exclamation marks are three separate emphatic marks, not a single meaningful three-character symbol the way "cat" is a single meaningful word or "42" is a single meaningful number. Treating each punctuation character as its own chunk lets BPE's OWN merge rules decide, from real training data, whether or when specific punctuation sequences (like "..." or "!?") deserve to become single learned tokens, rather than the pre-tokenizer hard-coding that decision structurally the way it hard-codes "letters group together" for words.

**5.** The encoder never gives user-supplied content a code path to a special-token ID at all: `ChatTemplateEncoder::append_turn` inserts `START_HEADER_ID`, `END_HEADER_ID`, and `EOT_ID` as fixed integer constants at positions its OWN code controls, and routes message content through `BPEEncoder::encode` exclusively, a function that has no notion of special tokens and no string-comparison logic that could ever recognize `<|end_of_text|>` as anything other than fifteen ordinary characters to byte-fallback/BPE-encode like any other text. Because no code anywhere maps a substring of `content` to a special-token ID, there is no injection point for a user to exploit in the first place, regardless of what literal text they type.

**6.** The final assistant turn is left open — header opened, role name encoded, header closed, two newlines emitted, and then nothing further — because that missing content IS what the model is being asked to generate. Closing the turn with an end-of-turn token would tell the model its own response has already ended before it has produced a single token; leaving the sequence open at exactly that point is what invites the model to continue generating from there.

**7.** A byte-fallback token's stored display string (like "<0xC3>") exists purely so a human inspecting the vocabulary can identify what byte a given token ID represents — it is seven ASCII characters, not the one raw byte 0xC3 those characters describe. Emitting the display string instead of the byte value produces output that looks superficially plausible for isolated ASCII debugging but is catastrophically wrong for real content: two byte-fallback tokens meant to reassemble into one two-byte UTF-8 codepoint instead produce fourteen characters of literal "<0xC3><0xA9>" text, because the two display strings were concatenated instead of the two actual bytes 0xC3 and 0xA9 they were supposed to represent.

**8.** UTF-8 is defined so that a multi-byte codepoint's bytes, written out consecutively and in order, ARE its valid encoding — there is nothing else the format needs beyond the correct bytes appearing adjacently in the correct sequence. Since the decoder's byte-fallback case for IDs 0-255 emits exactly the raw byte value each ID represents, decoding token 195 then token 169 in sequence emits byte 0xC3 immediately followed by byte 0xA9, which is precisely the two-byte UTF-8 encoding of U+00E9 — the decoder produces correct UTF-8 purely as a side effect of correctly emitting raw bytes in order, without containing any code that knows what UTF-8 or Unicode even are.

**9.** Beyond the key string and the type tag 9 marking the value as an array, the wire format needs the array's ELEMENT type (so the reader knows whether each element is a string, an INT32, a FLOAT32, or another supported type), a COUNT (so the reader knows how many elements to read before the next KV pair's key string begins), and then that many elements, each in its own element type's ordinary wire format, written back to back with no additional framing between them.

**10.** A reader that fails outright on a malformed file is a bug a caller cannot miss — an error return or an exception forces the problem into the open immediately. A reader that reports success while having silently read fewer key-value pairs than the file's byte layout actually contains is far more dangerous because every check a caller might reasonably think to run (`reader.open()` returned `true`, the fields that WERE read all look correct) passes, and only a lookup for the specific field that got dropped reveals anything wrong — and that lookup returns a plausible-looking default value (0 for a UINT32) rather than an error, so a caller who does not happen to check that exact field never learns anything is missing at all. This chapter's discipline of checking every written field against what comes back ELEMENT FOR ELEMENT, rather than only checking that the file opened successfully, is specifically what catches this: Test 3's exhaustive per-array, per-scalar comparison is what actually surfaced the `n_kv` miscount during this section's own authoring, not the earlier, weaker check that `reader.open()` returned `true`.
