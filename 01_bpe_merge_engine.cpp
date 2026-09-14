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
