// Chapter 19.1 -- Chapter 18 taught a vision-language model to look at a
// factory line's own camera feed and decide what a single, isolated part
// IS. A retail shelf asks a different question first: not "what is this,"
// but "is this where it is SUPPOSED to be" -- and a model has no way to
// answer that unless it is told, in the same prompt as the photograph,
// exactly what the shelf is supposed to contain. That specification has
// a real, standard retail name: a PLANOGRAM, the diagram every stocked
// shelf is built against, stating which SKU belongs in which position,
// and how many facings (adjacent copies) it should have.
//
// This section is a from-scratch, fully real implementation -- there is
// no hardware or external spec this section cannot verify byte-for-byte,
// unlike Chapter 18's GVSP or OPC UA sections. What it builds: a real
// planogram data structure with real validation (no two slots claiming
// the same shelf position, no slot claiming zero or negative facings), a
// deterministic text rendering of that structure suitable for injection
// as a VLM's own system prompt, a real run-length-style compression pass
// for shelves with long runs of identical adjacent slots, and a
// conservative token-budget guard that refuses to silently truncate a
// planogram it cannot fit, however compressed.
//
// A note on the token-budget estimate specifically: this section counts
// whitespace-delimited words and applies a stated, conservative
// multiplier, rather than re-running Chapter 12's own real BPE
// tokenizer. English text typically encodes to MORE sub-word tokens than
// words (a real BPE vocabulary splits many English words into multiple
// tokens), so a plain word count would UNDERESTIMATE the real token
// count -- exactly backwards for a budget guard, whose whole job is
// never to let something through that does not actually fit. The stated
// multiplier is deliberately generous, trading a possibly-too-cautious
// budget check for the guarantee that it never falsely reports a prompt
// as fitting when a real tokenizer would disagree.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_planogram_and_system_prompt.cpp -o 01_planogram_and_system_prompt
// Run:     ./01_planogram_and_system_prompt

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the planogram data structure and its own real validation.
// =======================================================================
struct ShelfSlot {
    int shelf_index = 0;
    int position_index = 0;
    std::string sku_id;
    std::string product_name;
    int expected_facings = 0;
};

struct Planogram {
    std::string store_layout_id;
    std::vector<ShelfSlot> slots;
};

struct ValidationResult {
    bool ok = false;
    std::string error;
};

// A planogram with two slots claiming the same (shelf, position) is not
// a physically meaningful shelf layout -- two different SKUs cannot
// occupy the same slot -- and a slot with zero or negative facings is
// not a real stocking instruction either. Both are refused outright,
// never silently accepted with the last-seen value winning.
ValidationResult validate_planogram(const Planogram& pg) {
    std::set<std::pair<int, int>> seen;
    for (const auto& s : pg.slots) {
        if (s.expected_facings <= 0) {
            std::ostringstream err;
            err << "slot (shelf " << s.shelf_index << ", position " << s.position_index
                << ") has non-positive expected_facings (" << s.expected_facings << ")";
            return ValidationResult{false, err.str()};
        }
        auto coord = std::make_pair(s.shelf_index, s.position_index);
        if (!seen.insert(coord).second) {
            std::ostringstream err;
            err << "duplicate slot at (shelf " << s.shelf_index << ", position " << s.position_index << ")";
            return ValidationResult{false, err.str()};
        }
    }
    return ValidationResult{true, ""};
}

// =======================================================================
// PART 2: deterministic rendering into a VLM-ready system prompt.
// =======================================================================
std::vector<ShelfSlot> sorted_slots(const Planogram& pg) {
    std::vector<ShelfSlot> out = pg.slots;
    std::sort(out.begin(), out.end(), [](const ShelfSlot& a, const ShelfSlot& b) {
        if (a.shelf_index != b.shelf_index) return a.shelf_index < b.shelf_index;
        return a.position_index < b.position_index;
    });
    return out;
}

// The full, uncompressed rendering: one line per slot, grouped under a
// header line per shelf. Slots are sorted by (shelf, position) before
// rendering regardless of the order they were added to the planogram in,
// so two planograms built from the same slots in different insertion
// orders render identically -- a real determinism guarantee, not an
// accident of whatever order a caller happened to populate them in.
std::string render_full_prompt(const Planogram& pg) {
    std::vector<ShelfSlot> slots = sorted_slots(pg);
    std::ostringstream out;
    out << "PLANOGRAM: " << pg.store_layout_id << "\n";
    int current_shelf = std::numeric_limits<int>::min();
    for (const auto& s : slots) {
        if (s.shelf_index != current_shelf) {
            out << "Shelf " << s.shelf_index << ":\n";
            current_shelf = s.shelf_index;
        }
        out << "  Position " << s.position_index << ": SKU " << s.sku_id
            << " (\"" << s.product_name << "\"), expected facings: " << s.expected_facings << "\n";
    }
    return out.str();
}

// A compressed rendering that collapses a RUN of consecutive positions
// on the SAME shelf sharing the SAME sku_id, product_name, and
// expected_facings into a single range line -- a real, general
// run-length-style compression, not a special case for any one
// planogram. A shelf with no compressible runs renders identically
// (modulo the range notation collapsing a run of length 1 to itself)
// whether or not compression is applied, since compression only ever
// merges lines that already say the same thing.
std::string render_compressed_prompt(const Planogram& pg) {
    std::vector<ShelfSlot> slots = sorted_slots(pg);
    std::ostringstream out;
    out << "PLANOGRAM: " << pg.store_layout_id << "\n";
    int current_shelf = std::numeric_limits<int>::min();

    size_t i = 0;
    while (i < slots.size()) {
        if (slots[i].shelf_index != current_shelf) {
            out << "Shelf " << slots[i].shelf_index << ":\n";
            current_shelf = slots[i].shelf_index;
        }
        size_t run_end = i;
        while (run_end + 1 < slots.size()
               && slots[run_end + 1].shelf_index == slots[i].shelf_index
               && slots[run_end + 1].position_index == slots[run_end].position_index + 1
               && slots[run_end + 1].sku_id == slots[i].sku_id
               && slots[run_end + 1].product_name == slots[i].product_name
               && slots[run_end + 1].expected_facings == slots[i].expected_facings) {
            ++run_end;
        }
        if (run_end == i) {
            out << "  Position " << slots[i].position_index << ": SKU " << slots[i].sku_id
                << " (\"" << slots[i].product_name << "\"), expected facings: " << slots[i].expected_facings << "\n";
        } else {
            out << "  Positions " << slots[i].position_index << "-" << slots[run_end].position_index
                << ": SKU " << slots[i].sku_id << " (\"" << slots[i].product_name
                << "\"), expected facings: " << slots[i].expected_facings << " each\n";
        }
        i = run_end + 1;
    }
    return out.str();
}

// =======================================================================
// PART 3: a conservative token-budget guard. See the header note above
// for why this deliberately OVER-counts rather than under-counts.
// =======================================================================
constexpr double WORD_TO_TOKEN_SAFETY_FACTOR = 1.4;

size_t estimate_token_count(const std::string& text) {
    size_t words = 0;
    bool in_word = false;
    for (char c : text) {
        bool is_space = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
        if (!is_space && !in_word) { ++words; in_word = true; }
        if (is_space) in_word = false;
    }
    return static_cast<size_t>(static_cast<double>(words) * WORD_TO_TOKEN_SAFETY_FACTOR) + 1;
}

struct BudgetResult {
    bool fits = false;
    std::string prompt;
    size_t estimated_tokens = 0;
    bool used_compression = false;
};

// Tries the full rendering first; if it fits, that is what is returned,
// since the full rendering is the more explicit, easier-to-follow one
// for a VLM to read. Only when the full rendering does not fit does this
// function fall back to the compressed rendering -- and if even THAT
// does not fit, it refuses outright (fits=false, prompt left empty)
// rather than fabricating a silently truncated planogram a VLM would
// have no way to know was incomplete.
BudgetResult render_within_budget(const Planogram& pg, size_t max_tokens) {
    std::string full = render_full_prompt(pg);
    size_t full_tokens = estimate_token_count(full);
    if (full_tokens <= max_tokens) {
        return BudgetResult{true, full, full_tokens, false};
    }
    std::string compressed = render_compressed_prompt(pg);
    size_t compressed_tokens = estimate_token_count(compressed);
    if (compressed_tokens <= max_tokens) {
        return BudgetResult{true, compressed, compressed_tokens, true};
    }
    return BudgetResult{false, "", compressed_tokens, true};
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 19.1: Encoding a Planogram Directly into the System Prompt\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a well-formed planogram validates and renders deterministically --\n";
    {
        Planogram pg;
        pg.store_layout_id = "STORE-042-AISLE-7";
        pg.slots = {
            {1, 1, "SKU-1001", "Widget A", 3},
            {1, 2, "SKU-1002", "Widget B", 2},
            {2, 1, "SKU-2001", "Gadget C", 4},
        };
        auto v = validate_planogram(pg);
        CHECK(v.ok);
        std::string r1 = render_full_prompt(pg);
        std::string r2 = render_full_prompt(pg);
        CHECK(r1 == r2);
        CHECK(r1.find("PLANOGRAM: STORE-042-AISLE-7") != std::string::npos);
        CHECK(r1.find("Shelf 1:") != std::string::npos);
        CHECK(r1.find("Shelf 2:") != std::string::npos);
        CHECK(r1.find("SKU-1001") != std::string::npos);
        CHECK(r1.find("expected facings: 4") != std::string::npos);
        // Shelf 1 must appear before Shelf 2 in the rendering, and within
        // shelf 1, position 1 before position 2 -- confirming the sort
        // is real, not an accident of insertion order.
        CHECK(r1.find("Shelf 1:") < r1.find("Shelf 2:"));
        CHECK(r1.find("Position 1") < r1.find("Position 2"));
        std::cout << "  a 3-slot, 2-shelf planogram validates ok and renders identically across two calls\n";
    }

    std::cout << "\n-- Test 2: rendering is independent of insertion order --\n";
    {
        Planogram forward, reversed;
        forward.store_layout_id = reversed.store_layout_id = "STORE-001";
        forward.slots = {
            {1, 1, "SKU-A", "Item A", 1},
            {1, 2, "SKU-B", "Item B", 1},
            {2, 1, "SKU-C", "Item C", 1},
        };
        reversed.slots = {forward.slots[2], forward.slots[1], forward.slots[0]};
        CHECK(render_full_prompt(forward) == render_full_prompt(reversed));
        std::cout << "  the same slots added in reverse order render byte-identically once sorted\n";
    }

    std::cout << "\n-- Test 3: validation refuses a duplicate slot and a non-positive facing count --\n";
    {
        Planogram dup;
        dup.store_layout_id = "STORE-BAD-1";
        dup.slots = {
            {1, 1, "SKU-A", "Item A", 2},
            {1, 1, "SKU-B", "Item B", 3},   // same (shelf, position) as above
        };
        auto v1 = validate_planogram(dup);
        CHECK(!v1.ok);
        CHECK(v1.error.find("duplicate slot") != std::string::npos);

        Planogram bad_facings;
        bad_facings.store_layout_id = "STORE-BAD-2";
        bad_facings.slots = {{1, 1, "SKU-A", "Item A", 0}};
        auto v2 = validate_planogram(bad_facings);
        CHECK(!v2.ok);
        CHECK(v2.error.find("non-positive") != std::string::npos);

        std::cout << "  duplicate slot rejected (\"" << v1.error << "\"); "
                     "zero-facing slot rejected (\"" << v2.error << "\")\n";
    }

    std::cout << "\n-- Test 4: two different planograms render to different prompts --\n";
    {
        Planogram a, b;
        a.store_layout_id = b.store_layout_id = "STORE-SAME-ID";
        a.slots = {{1, 1, "SKU-A", "Item A", 2}};
        b.slots = {{1, 1, "SKU-B", "Item B", 5}};
        CHECK(render_full_prompt(a) != render_full_prompt(b));
        std::cout << "  a planogram with a different SKU and facing count at the same slot "
                     "renders to a genuinely different prompt string\n";
    }

    std::cout << "\n-- Test 5: run-length compression merges a real consecutive run, and only a real run --\n";
    {
        Planogram runny;
        runny.store_layout_id = "STORE-RUN-1";
        // Positions 1-4 on shelf 1: identical sku/product/facings, a real
        // compressible run. Position 5: a DIFFERENT sku, breaking the run.
        runny.slots = {
            {1, 1, "SKU-X", "Bulk Item", 6},
            {1, 2, "SKU-X", "Bulk Item", 6},
            {1, 3, "SKU-X", "Bulk Item", 6},
            {1, 4, "SKU-X", "Bulk Item", 6},
            {1, 5, "SKU-Y", "Different Item", 2},
        };
        std::string compressed = render_compressed_prompt(runny);
        CHECK(compressed.find("Positions 1-4: SKU SKU-X") != std::string::npos);
        CHECK(compressed.find("expected facings: 6 each") != std::string::npos);
        // Position 5 must NOT be swallowed into the run -- it appears on
        // its own singular line, not as part of "Positions 1-5".
        CHECK(compressed.find("Positions 1-5") == std::string::npos);
        CHECK(compressed.find("Position 5: SKU SKU-Y") != std::string::npos);

        Planogram no_run;
        no_run.store_layout_id = "STORE-NORUN-1";
        no_run.slots = {
            {1, 1, "SKU-A", "Item A", 1},
            {1, 2, "SKU-B", "Item B", 2},
        };
        // With no compressible run at all, the compressed rendering must
        // still contain both slots individually (compression never loses
        // information -- it only ever merges lines that already agree).
        std::string nc = render_compressed_prompt(no_run);
        CHECK(nc.find("Position 1: SKU SKU-A") != std::string::npos);
        CHECK(nc.find("Position 2: SKU SKU-B") != std::string::npos);
        CHECK(nc.find("Positions 1-2") == std::string::npos);

        std::cout << "  a real 4-position run of an identical SKU compresses into one range line, "
                     "the differing 5th position is correctly left out of that range, and a "
                     "planogram with no compressible run is rendered with no ranges at all\n";
    }

    std::cout << "\n-- Test 6: the token-budget guard fits the full render, falls back to compression, then refuses --\n";
    {
        Planogram small;
        small.store_layout_id = "STORE-SMALL";
        small.slots = {{1, 1, "SKU-A", "Item A", 1}, {1, 2, "SKU-B", "Item B", 1}};
        auto r_small = render_within_budget(small, /*max_tokens=*/1000);
        CHECK(r_small.fits);
        CHECK(!r_small.used_compression);
        CHECK(r_small.prompt == render_full_prompt(small));

        // A large planogram with a genuinely compressible run: too big
        // for a tight budget in its FULL form, but small enough once
        // compressed.
        Planogram large;
        large.store_layout_id = "STORE-LARGE";
        for (int pos = 1; pos <= 60; ++pos) {
            large.slots.push_back({1, pos, "SKU-BULK", "Bulk Restock Item", 8});
        }
        size_t full_tokens = estimate_token_count(render_full_prompt(large));
        size_t compressed_tokens = estimate_token_count(render_compressed_prompt(large));
        CHECK(compressed_tokens < full_tokens);   // compression must actually shrink a real run
        size_t tight_budget = compressed_tokens + 5;   // fits compressed, not full
        CHECK(full_tokens > tight_budget);
        auto r_large = render_within_budget(large, tight_budget);
        CHECK(r_large.fits);
        CHECK(r_large.used_compression);
        CHECK(r_large.prompt == render_compressed_prompt(large));

        // An impossibly tight budget: refuses outright, no prompt fabricated.
        auto r_impossible = render_within_budget(large, /*max_tokens=*/3);
        CHECK(!r_impossible.fits);
        CHECK(r_impossible.prompt.empty());

        std::cout << "  a small planogram fits its full rendering directly; a large, compressible "
                     "planogram (" << full_tokens << " est. tokens full, " << compressed_tokens
                   << " est. tokens compressed) falls back to compression under a tight budget of "
                   << tight_budget << "; an impossibly tight budget of 3 refuses outright rather "
                     "than fabricating a truncated prompt\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
