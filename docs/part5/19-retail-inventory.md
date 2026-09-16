# Chapter 19: Retail Inventory and Supply Chain Intelligence: Shelf Photography to Action

**What you will understand by the end of this chapter:**

- How to encode a real retail planogram -- the diagram stating which SKU belongs in which shelf slot and how many facings it should have -- into a deterministic, VLM-ready system prompt, with real validation, a real run-length compression pass for large stores, and a conservative token-budget guard that refuses to silently truncate a planogram it cannot fit.
- How to run Chapter 18's own real vision-language pipeline across a BATCH of shelf photographs, isolating one photo's own failure from the rest of the batch, and how to build a strict, from-scratch parser for the structured text a shelf-inspection prompt asks a VLM to answer with.
- How to model a REST integration into a retailer's own ERP/WMS systems as a from-scratch JSON serializer and a synthetic backend enforcing real HTTP status codes and real optimistic-concurrency conflict detection -- the discipline that keeps a live inventory record from suffering a silent lost update under concurrent writers.
- How to aggregate inventory snapshots honestly across many stores, excluding a store that did not report from a period's total rather than ever fabricating it as zero, and how to fit a real least-squares trend line to detect a stockout or overstock signal, gated by both a rate threshold and a projection horizon.
- How to reconcile a planogram against detections gathered over time into two genuinely different findings: an immediate misplacement, true the moment a single photo shows it, and a shrinkage suspicion, which this chapter's own detector refuses to raise until a stated number of CONSECUTIVE missing observations corroborates it.

**What you need to know first:**

- Chapter 18's complete vision-language pipeline -- Section 18.2's vision encoder and Section 18.3's fusion into the unchanged Qwen2 decoder -- which Section 19.2 runs, self-contained, across a real batch of photos rather than the single photo Chapter 18 ever tested it on.
- Chapter 18.4's asymmetric confidence-threshold reasoning and Chapter 18.5's interface-level protocol abstraction, both of which this chapter's own Sections 19.3 and 19.5 apply again in a genuinely different domain -- REST/ERP integration and evidence-based shrinkage detection, respectively.
- This chapter is more tractable end to end than Chapter 18: every section here is a real, fully verifiable, from-scratch implementation with no external spec this book cannot check byte-for-byte, and no section needed a reduced cross-architecture check the way Chapter 18.4's SQLite dependency once did.

---

Chapter 18 taught a vision-language model to inspect a single manufactured part in isolation and decide, from what it saw, whether the part was good. A retail shelf asks a related but genuinely different question: not "what is this," but "is this where it is supposed to be, and if it is missing, has it been missing long enough to matter." This chapter builds every piece that question requires, in the order a real retail deployment would need them -- encoding what SHOULD be on a shelf, observing what actually is there across a batch of real photographs, reporting both into the ERP and WMS systems a retailer already runs its business on, aggregating that reporting honestly across every store in a chain, and finally reconciling expectation against observation into the two real findings -- misplacement and shrinkage -- a store operator can actually act on.

## 19.1 Encoding a Planogram Directly into the System Prompt

### Intuition

A vision-language model looking at a shelf photograph has no way to know what SHOULD be there unless it is told, in the very same prompt as the photograph, exactly what a real retail planogram specifies: which SKU belongs in which shelf slot, and how many adjacent facings it should have.

### The Concept, In Detail

`Planogram` and `ShelfSlot` are a direct, real encoding of that retail concept -- a store layout id and a flat list of slots, each stating a shelf index, a position index, a SKU id and product name, and an expected facing count. `validate_planogram` refuses, rather than silently accepting, the two ways such a structure can fail to describe a physically meaningful shelf: two different slots claiming the identical (shelf, position) coordinate, which cannot both be true of one real shelf at once, and a slot claiming zero or negative facings, which is not a real stocking instruction at all.

`render_full_prompt` renders a planogram deterministically by first sorting every slot by (shelf, position) regardless of the order it was added in -- Test 2 confirms directly that the same three slots, added in forward and reversed order, render to byte-identical text -- because a system prompt's own content should depend on what a planogram SAYS, never on the incidental order a caller happened to populate it in.

A large store's own planogram can run long enough to threaten a real context-window budget, and this section takes that threat seriously rather than assuming it away. `render_compressed_prompt` applies a real run-length-style compression, merging a genuine RUN of consecutive positions on the same shelf sharing an identical SKU, product name, and facing count into a single range line -- Test 5 confirms this merges exactly the real compressible run in a synthetic planogram and, just as importantly, does NOT swallow an adjacent position whose SKU genuinely differs into that same range. `estimate_token_count` deliberately OVER-counts rather than under-counts, applying a stated conservative multiplier to a plain whitespace word count rather than re-running Chapter 12's own real BPE tokenizer -- because English text typically encodes to MORE sub-word tokens than words, and a budget guard whose whole job is refusing what does not actually fit must never be the more optimistic estimate of the two. `render_within_budget` tries the full rendering first, falls back to the compressed rendering only if the full one does not fit, and refuses outright -- returning no prompt at all -- if even the compressed rendering exceeds the stated budget, rather than fabricating a silently truncated planogram a VLM would have no way to know was incomplete.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_planogram_and_system_prompt.cpp -o 01_planogram_and_system_prompt
./01_planogram_and_system_prompt
```

**Sample input:** a well-formed planogram checked to validate and render deterministically and independently of insertion order; a duplicate slot and a non-positive facing count each checked to be refused with a specific, distinguishing error; two different planograms checked to render to genuinely different prompts; a real 4-position compressible run checked to merge into one range line without swallowing an adjacent differing position; and the token-budget guard checked to fit a small planogram's full rendering directly, fall back to compression for a large but compressible one under a tight budget, and refuse outright under an impossibly tight one.

```text
========================================================
Chapter 19.1: Encoding a Planogram Directly into the System Prompt
========================================================

-- Test 1: a well-formed planogram validates and renders deterministically --
  a 3-slot, 2-shelf planogram validates ok and renders identically across two calls

-- Test 2: rendering is independent of insertion order --
  the same slots added in reverse order render byte-identically once sorted

-- Test 3: validation refuses a duplicate slot and a non-positive facing count --
  duplicate slot rejected ("duplicate slot at (shelf 1, position 1)"); zero-facing slot rejected ("slot (shelf 1, position 1) has non-positive expected_facings (0)")

-- Test 4: two different planograms render to different prompts --
  a planogram with a different SKU and facing count at the same slot renders to a genuinely different prompt string

-- Test 5: run-length compression merges a real consecutive run, and only a real run --
  a real 4-position run of an identical SKU compresses into one range line, the differing 5th position is correctly left out of that range, and a planogram with no compressible run is rendered with no ranges at all

-- Test 6: the token-budget guard fits the full render, falls back to compression, then refuses --
  a small planogram fits its full rendering directly; a large, compressible planogram (846 est. tokens full, 22 est. tokens compressed) falls back to compression under a tight budget of 27; an impossibly tight budget of 3 refuses outright rather than fabricating a truncated prompt

32/32 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] a token-budget estimate that under-counts is worse than no estimate at all"
    It is tempting to estimate a prompt's token cost by counting whitespace-delimited words directly, since that is the cheapest computation available without re-running a real tokenizer. Real BPE tokenizers typically split many English words into MULTIPLE sub-word tokens, so a plain word count systematically UNDER-estimates the real cost -- exactly backwards for a guard whose entire purpose is refusing a prompt that will not actually fit. This section's own `estimate_token_count` applies a stated, deliberately generous multiplier for exactly this reason: a budget guard that is sometimes too cautious wastes nothing but a little context-window headroom, while a budget guard that ever under-counts can let through a prompt a real model would truncate mid-instruction, with no warning to either the caller or the model that anything was cut off.

## 19.2 A Batch Shelf-Photo Processor

### Intuition

Section 19.1 built the prompt; this section runs the actual inspection. A retail batch job processes many shelf photographs in one run, and this section's own real, stated discipline is that a single bad photo must never take the rest of the batch down with it.

### The Concept, In Detail

`process_batch` runs Chapter 18's own real, unchanged vision encoder and multimodal fusion across every photo in a batch, against the SAME planogram-derived token template -- a real retail batch job inspects many photos against one shared prompt, not a bespoke one per photo. Each photo's own outcome, success or a specific kind of failure (a fusion count mismatch, KV-capacity exhaustion, or a NaN), is recorded into its own `BatchPhotoResult` rather than raised as an exception that would abort the whole run. Test 2 proves this isolation is real: a deliberately mis-sized photo among two well-formed ones trips `build_fused_embeddings`'s own count-mismatch refusal from Chapter 18.3, and the batch correctly reports 2 successes and exactly 1 isolated failure, naming the specific photo that failed and why -- the other two photos' own results are completely unaffected.

This section's own second, separate concern is a real, strict parser for the structured text a shelf-inspection prompt asks a VLM to answer with: one detection per line, in a stated `SKU=...;SHELF=...;POSITION=...;CONFIDENCE=...` format. `parse_detection_line` is tested against REPRESENTATIVE EXAMPLE completions, not against genuine output from the real forward pass Test 1 and Test 2 already verified -- and that gap is stated honestly rather than glossed over. This book's own from-scratch model weights are random and untrained, exactly as they have been since Chapter 15's own synthetic self-tests; only a real, fine-tuned checkpoint would ever cause the real forward pass to emit genuinely structured text. What this section CAN and does verify for real, without depending on trained weights at all, is that the real multimodal pipeline itself runs correctly across a batch, and that the parser downstream of whatever a real deployed model eventually says is strict and correct: `std::from_chars` is used throughout, matching this book's own standing preference for checked return values over exceptions, and Test 4 confirms all eight real ways a detection line can be malformed -- a missing field, a duplicate field, an unexpected extra field, a non-integer SHELF, a non-numeric CONFIDENCE, an out-of-range CONFIDENCE, an empty SKU, and a field with no `=` at all -- are each refused with a specific, distinguishing error.

### Code and Verification

```cpp
// Chapter 19.2 -- Section 19.1 built a planogram and rendered it into a
// VLM-ready system prompt. This section runs the actual inspection: a
// batch of shelf photographs, each one pushed through Chapter 18's own
// real vision-language pipeline -- Section 18.2's vision encoder and
// Section 18.3's fusion into the unchanged Qwen2 decoder, repeated here
// self-contained exactly as every other file in this book repeats the
// shared machinery it depends on -- to prove that pipeline runs
// end-to-end, deterministically, and without NaNs, across a real batch
// of photos, not just the single photo Chapter 18 ever tested it on.
//
// A retail batch job's own real, stated constraint: a single bad photo
// -- corrupt, wrong dimensions, or one whose visual-token count somehow
// does not match its own text template -- must never take down the rest
// of the batch. `process_batch` isolates each photo's own failure into
// its own `BatchPhotoResult`, continuing to the next photo rather than
// aborting the run, exactly the same fail-LOUD-but-not-fail-STOP
// distinction Chapter 18.1's `FrameReassembler` already drew between a
// dropped packet (reported honestly) and a whole frame acquisition
// pipeline crashing over it.
//
// A second, separate concern this section builds: a real, strict parser
// for the structured text a shelf-inspection prompt asks a VLM to
// respond with -- one detection per line, in a stated `SKU=...;SHELF=...;
// POSITION=...;CONFIDENCE=...` format. This parser is tested against
// REPRESENTATIVE EXAMPLE completions, not against genuine output from
// the real forward pass run above, and that gap is stated honestly
// rather than glossed over: this book's own from-scratch model weights
// are random and untrained, exactly as they have been since Chapter 15's
// synthetic self-tests, and only a real, fine-tuned checkpoint would
// ever cause the real forward pass to emit genuinely structured text.
// What Section 19.2 CAN and does verify for real is the two things that
// do not depend on trained weights at all: that the real multimodal
// pipeline itself runs correctly across a batch, and that the parser
// downstream of whatever a real deployed model eventually says is
// strict, correct, and fails loudly on anything malformed.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_batch_shelf_photo_processor.cpp -o 02_batch_shelf_photo_processor
// Run:     ./02_batch_shelf_photo_processor

#include <mdspan/mdspan.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
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

// =======================================================================
// PART 1: Section 15.1's GGUF reader/writer (Section 15.4's own copy).
// =======================================================================
enum GGUFValueType : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12,
};
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q8_0 = 8 };
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
    uint32_t get_u32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0 : std::get<uint32_t>(it->second);
    }
    float get_f32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0.0f : std::get<float>(it->second);
    }
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

// =======================================================================
// PART 2: Chapter 4.2's fp16_t/BlockQ8 (Section 15.4's own copy).
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
// PART 3: memory-mapped file + dequantization (Section 15.4's own copy).
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
    ~MappedFile() { if (data) ::munmap(data, size); if (fd >= 0) ::close(fd); }
};
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
// PART 4: Section 15.3's adapted transformer block (Section 15.4's own
// copy).
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
        cos_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        sin_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * theta;
                cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::cos(angle);
                sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::sin(angle);
            }
        }
    }
    float cos_at(int pos, int k) const { return cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
    float sin_at(int pos, int k) const { return sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
};
void apply_rope(std::span<float> vec, int pos, const RoPETables& tables) {
    const int half_dim = tables.half_dim;
    for (int k = 0; k < half_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + half_dim)];
        float c = tables.cos_at(pos, k), s = tables.sin_at(pos, k);
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + half_dim)] = x1 * s + x2 * c;
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
        K.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
        V.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
    }
    View k_view() { return View(K.data(), n_heads_kv, max_seq_len, head_dim); }
    View v_view() { return View(V.data(), n_heads_kv, max_seq_len, head_dim); }
    auto k_at(int h, int t) { return std::submdspan(k_view(), h, t, std::full_extent); }
    auto v_at(int h, int t) { return std::submdspan(v_view(), h, t, std::full_extent); }
    void store(int h, int t, std::span<const float> k, std::span<const float> v) {
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[static_cast<size_t>(i)]; vslice[i] = v[static_cast<size_t>(i)]; }
    }
};
void gqa_attention(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                    int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(seq_len));
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, static_cast<size_t>(head_dim));
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            double d = 0.0;
            for (int i = 0; i < head_dim; ++i) d += static_cast<double>(q[i]) * static_cast<double>(k[i]);
            scores[static_cast<size_t>(t)] = static_cast<float>(d) * scale;
        }
        softmax_inplace(std::span<float>(scores.data(), static_cast<size_t>(seq_len)));
        std::vector<double> acc(static_cast<size_t>(head_dim), 0.0);
        for (int t = 0; t < seq_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            double w = scores[static_cast<size_t>(t)];
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
    std::vector<float> normed(static_cast<size_t>(shape.dim)), q(static_cast<size_t>(shape.q_dim())),
        k(static_cast<size_t>(shape.kv_dim())), v(static_cast<size_t>(shape.kv_dim()));
    std::vector<float> attn_out(static_cast<size_t>(shape.q_dim())), proj_out(static_cast<size_t>(shape.dim));
    rms_norm(normed, x, w.attn_norm);
    linear_with_bias(q, normed, w.Wq, w.bq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.q_dim()));
    linear_with_bias(k, normed, w.Wk, w.bk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    linear_with_bias(v, normed, w.Wv, w.bv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    for (int h = 0; h < shape.n_heads; ++h)
        apply_rope(std::span<float>(q.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        apply_rope(std::span<float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        cache.store(h, pos, std::span<const float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                             std::span<const float>(v.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)));
    gqa_attention(q, cache, attn_out, pos + 1, shape.n_heads, shape.group_size());
    matmul(proj_out, attn_out, w.Wo, static_cast<size_t>(shape.q_dim()), static_cast<size_t>(shape.dim));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += proj_out[static_cast<size_t>(i)];
    std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
    rms_norm(normed2, x, w.ffn_norm);
    swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += ffn_out[static_cast<size_t>(i)];
}
struct QwenModel {
    MappedFile mf;
    GGUFReader r;
    QwenShape shape{};
    RoPETables* rope = nullptr;
    bool load(const std::string& path) {
        if (!mf.open(path)) return false;
        if (!r.open(path)) return false;
        shape.dim = static_cast<int>(r.get_u32("qwen2.embedding_length"));
        shape.n_heads = static_cast<int>(r.get_u32("qwen2.attention.head_count"));
        shape.n_heads_kv = static_cast<int>(r.get_u32("qwen2.attention.head_count_kv"));
        shape.head_dim = shape.dim / shape.n_heads;
        shape.d_ff = static_cast<int>(r.get_u32("qwen2.feed_forward_length"));
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
                             + (static_cast<uint64_t>(token_id) * static_cast<uint64_t>(shape.dim) / 32) * sizeof(BlockQ8);
        std::vector<float> out(static_cast<size_t>(shape.dim));
        dequantize_row_q8(mf, row_offset, static_cast<size_t>(shape.dim), out);
        return out;
    }
};

// =======================================================================
// PART 1: raw-buffer preprocessing. Section 18.1's FrameResult::pixels
// is already exactly this input -- an HxWxC row-major byte buffer, no
// codec involved anywhere in this pipeline.
// =======================================================================
struct RawImage {
    uint32_t width = 0, height = 0, channels = 0;
    std::vector<uint8_t> pixels;   // row-major, HxWxC, matching FrameResult::pixels exactly
};

// Nearest-neighbor resize to a target size that is an exact multiple of
// the ViT's own patch size. Real deployments use a higher-quality
// filter; nearest-neighbor is this section's own stated simplification,
// chosen because it is exactly reproducible in integer arithmetic across
// every architecture this book locks against, with no floating-point
// rounding difference to chase.
RawImage resize_nearest(const RawImage& src, uint32_t dst_w, uint32_t dst_h) {
    RawImage dst;
    dst.width = dst_w; dst.height = dst_h; dst.channels = src.channels;
    dst.pixels.resize(static_cast<size_t>(dst_w) * dst_h * src.channels);
    for (uint32_t y = 0; y < dst_h; ++y) {
        uint32_t sy = std::min(src.height - 1, (y * src.height) / dst_h);
        for (uint32_t x = 0; x < dst_w; ++x) {
            uint32_t sx = std::min(src.width - 1, (x * src.width) / dst_w);
            const uint8_t* sp = &src.pixels[(static_cast<size_t>(sy) * src.width + sx) * src.channels];
            uint8_t* dp = &dst.pixels[(static_cast<size_t>(y) * dst_w + x) * src.channels];
            std::memcpy(dp, sp, src.channels);
        }
    }
    return dst;
}

// CLIP/SigLIP-style per-channel normalization constants -- stated, not
// derived: a real deployment would use whichever mean/std the specific
// checkpoint's own preprocessor_config.json declares.
struct NormStats { float mean[3] = {0.481f, 0.458f, 0.408f}; float std[3] = {0.269f, 0.261f, 0.276f}; };

// One patch's flattened, normalized pixel data: patch_size * patch_size
// * channels floats, in row-major (row, then col, then channel) order.
std::vector<float> extract_patch(const RawImage& img, uint32_t patch_row, uint32_t patch_col,
                                  uint32_t patch_size, const NormStats& norm) {
    std::vector<float> out(static_cast<size_t>(patch_size) * patch_size * img.channels);
    size_t idx = 0;
    for (uint32_t py = 0; py < patch_size; ++py) {
        uint32_t y = patch_row * patch_size + py;
        for (uint32_t px = 0; px < patch_size; ++px) {
            uint32_t x = patch_col * patch_size + px;
            const uint8_t* sp = &img.pixels[(static_cast<size_t>(y) * img.width + x) * img.channels];
            for (uint32_t c = 0; c < img.channels; ++c) {
                float v = static_cast<float>(sp[c]) / 255.0f;
                out[idx++] = (v - norm.mean[c % 3]) / norm.std[c % 3];
            }
        }
    }
    return out;
}

// Patchifies the WHOLE image into a row-major grid of flattened patches:
// patches[row * grid_w + col] is that (row, col) patch's own flattened,
// normalized pixel vector.
std::vector<std::vector<float>> patchify(const RawImage& img, uint32_t patch_size, const NormStats& norm,
                                          uint32_t& grid_h, uint32_t& grid_w) {
    grid_h = img.height / patch_size;
    grid_w = img.width / patch_size;
    std::vector<std::vector<float>> patches(static_cast<size_t>(grid_h) * grid_w);
    for (uint32_t r = 0; r < grid_h; ++r)
        for (uint32_t c = 0; c < grid_w; ++c)
            patches[static_cast<size_t>(r) * grid_w + c] = extract_patch(img, r, c, patch_size, norm);
    return patches;
}


// =======================================================================
// PART 3: two-dimensional rotary position encoding. A text token has one
// position; a patch has TWO (its row and its column), and Qwen2-VL's own
// real fix is splitting each attention head's dimension in half, rotating
// the FIRST half by the patch's row and the SECOND half by its column --
// the same 1D rotate-half mechanism this book has used since Chapter 9,
// applied twice, to two different coordinates, over two disjoint slices
// of the same vector.
// =======================================================================
struct RoPE2DTables {
    std::vector<float> cos_row, sin_row, cos_col, sin_col;
    int quarter_dim;   // half_dim (per axis) is head_dim/2; each axis's own rotate-half pairs cover head_dim/4
    RoPE2DTables(int max_coord, int head_dim, float base) : quarter_dim(head_dim / 4) {
        cos_row.resize(static_cast<size_t>(max_coord) * quarter_dim);
        sin_row.resize(static_cast<size_t>(max_coord) * quarter_dim);
        cos_col.resize(static_cast<size_t>(max_coord) * quarter_dim);
        sin_col.resize(static_cast<size_t>(max_coord) * quarter_dim);
        for (int coord = 0; coord < max_coord; ++coord) {
            for (int k = 0; k < quarter_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim / 2));
                float angle = static_cast<float>(coord) * theta;
                size_t idx = static_cast<size_t>(coord) * quarter_dim + k;
                cos_row[idx] = std::cos(angle); sin_row[idx] = std::sin(angle);
                cos_col[idx] = std::cos(angle); sin_col[idx] = std::sin(angle);
            }
        }
    }
};
// Rotates a length-`2*half`-element slice using the standard rotate-half
// pairing (index k paired with index k+half), exactly Section 9's own
// `apply_rope` -- factored out so both the row-half and the col-half of
// a patch's query/key vector can call the identical primitive.
void rotate_half_inplace(std::span<float> vec, std::span<const float> cos_tab, std::span<const float> sin_tab,
                          int coord, int quarter_dim) {
    for (int k = 0; k < quarter_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + quarter_dim)];
        float c = cos_tab[static_cast<size_t>(coord) * quarter_dim + k];
        float s = sin_tab[static_cast<size_t>(coord) * quarter_dim + k];
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + quarter_dim)] = x1 * s + x2 * c;
    }
}
// Applies 2D RoPE to one attention head's full head_dim vector: the
// FIRST half rotated by `row` (using its own rotate-half pairing over
// that half's two quarters), the SECOND half rotated by `col`.
void apply_rope2d(std::span<float> head_vec, int row, int col, const RoPE2DTables& t) {
    const int half = static_cast<int>(head_vec.size()) / 2;
    rotate_half_inplace(head_vec.subspan(0, static_cast<size_t>(half)), t.cos_row, t.sin_row, row, t.quarter_dim);
    rotate_half_inplace(head_vec.subspan(static_cast<size_t>(half), static_cast<size_t>(half)), t.cos_col, t.sin_col, col, t.quarter_dim);
}

// =======================================================================
// PART 4: the vision transformer block itself -- RMSNorm, QKV
// projection, 2D RoPE, FULL (non-causal, no KV cache) bidirectional
// attention over every patch, output projection, a residual, a second
// RMSNorm, the SwiGLU FFN, and a second residual.
// =======================================================================
struct ViTShape { int dim, n_heads, head_dim, d_ff; int qkv_dim() const { return n_heads * head_dim; } };
struct ViTBlockWeights {
    std::vector<float> attn_norm, Wq, Wk, Wv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};

// One full self-attention pass over ALL patches -- unlike the text
// decoder's causal, one-position-at-a-time attention, every patch here
// attends to every other patch in a single call, because an image has
// no "future" to mask and no cache to build incrementally.
void vit_full_attention(const std::vector<std::vector<float>>& q, const std::vector<std::vector<float>>& k,
                         const std::vector<std::vector<float>>& v, std::vector<std::vector<float>>& out,
                         int n_heads, int head_dim) {
    const int n = static_cast<int>(q.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < n; ++i) {
            std::vector<float> scores(static_cast<size_t>(n));
            std::span<const float> qi(q[static_cast<size_t>(i)].data() + h * head_dim, static_cast<size_t>(head_dim));
            for (int j = 0; j < n; ++j) {
                std::span<const float> kj(k[static_cast<size_t>(j)].data() + h * head_dim, static_cast<size_t>(head_dim));
                double d = 0.0;
                for (int c = 0; c < head_dim; ++c) d += static_cast<double>(qi[static_cast<size_t>(c)]) * kj[static_cast<size_t>(c)];
                scores[static_cast<size_t>(j)] = static_cast<float>(d) * scale;
            }
            softmax_inplace(scores);
            float* o = out[static_cast<size_t>(i)].data() + h * head_dim;
            for (int c = 0; c < head_dim; ++c) o[c] = 0.0f;
            for (int j = 0; j < n; ++j) {
                std::span<const float> vj(v[static_cast<size_t>(j)].data() + h * head_dim, static_cast<size_t>(head_dim));
                float w = scores[static_cast<size_t>(j)];
                for (int c = 0; c < head_dim; ++c) o[c] += w * vj[static_cast<size_t>(c)];
            }
        }
    }
}

void vit_block_forward(std::vector<std::vector<float>>& x, const ViTShape& shape, const ViTBlockWeights& w,
                        const std::vector<int>& rows, const std::vector<int>& cols, const RoPE2DTables& rope) {
    const int n = static_cast<int>(x.size());
    std::vector<std::vector<float>> q(static_cast<size_t>(n)), k(static_cast<size_t>(n)), v(static_cast<size_t>(n)), attn_out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        std::vector<float> normed(static_cast<size_t>(shape.dim));
        rms_norm(normed, x[static_cast<size_t>(i)], w.attn_norm);
        q[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        k[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        v[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        attn_out[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.qkv_dim()));
        matmul(q[static_cast<size_t>(i)], normed, w.Wq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        matmul(k[static_cast<size_t>(i)], normed, w.Wk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        matmul(v[static_cast<size_t>(i)], normed, w.Wv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.qkv_dim()));
        for (int h = 0; h < shape.n_heads; ++h) {
            apply_rope2d(std::span<float>(q[static_cast<size_t>(i)].data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                         rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)], rope);
            apply_rope2d(std::span<float>(k[static_cast<size_t>(i)].data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                         rows[static_cast<size_t>(i)], cols[static_cast<size_t>(i)], rope);
        }
    }
    vit_full_attention(q, k, v, attn_out, shape.n_heads, shape.head_dim);
    for (int i = 0; i < n; ++i) {
        std::vector<float> proj(static_cast<size_t>(shape.dim));
        matmul(proj, attn_out[static_cast<size_t>(i)], w.Wo, static_cast<size_t>(shape.qkv_dim()), static_cast<size_t>(shape.dim));
        for (int d = 0; d < shape.dim; ++d) x[static_cast<size_t>(i)][static_cast<size_t>(d)] += proj[static_cast<size_t>(d)];
        std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
        rms_norm(normed2, x[static_cast<size_t>(i)], w.ffn_norm);
        swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
        for (int d = 0; d < shape.dim; ++d) x[static_cast<size_t>(i)][static_cast<size_t>(d)] += ffn_out[static_cast<size_t>(d)];
    }
}

// =======================================================================
// PART 5: the 2x2 spatial patch merger. Four SPATIALLY ADJACENT patches
// -- (2r,2c), (2r,2c+1), (2r+1,2c), (2r+1,2c+1) -- are concatenated and
// projected down to the LLM's own embedding dimension, cutting the
// visual token count by 4x. Grouping by spatial adjacency, not by
// row-major list order, matters: for any grid wider than two patches,
// four row-major-consecutive patches are the START OF ONE ROW, not a 2x2
// neighborhood, and merging them would blend unrelated regions of the
// image into one token.
// =======================================================================
struct MergerWeights { std::vector<float> W1, b1, W2, b2; int hidden_dim; };

// Returns the flat patch-list indices of the 4 patches spatially
// adjacent to merged block (br, bc), in a fixed, testable order:
// top-left, top-right, bottom-left, bottom-right. Factored out on its
// own specifically so this section's own self-tests can check the
// SPATIAL grouping directly, independent of the matmul math around it.
std::array<int, 4> block_patch_indices(int br, int bc, int grid_w) {
    const int r0 = 2 * br, r1 = 2 * br + 1, c0 = 2 * bc, c1 = 2 * bc + 1;
    return {r0 * grid_w + c0, r0 * grid_w + c1, r1 * grid_w + c0, r1 * grid_w + c1};
}

std::vector<std::vector<float>> merge_all_2x2(const std::vector<std::vector<float>>& patches, int grid_h, int grid_w,
                                               int vit_dim, const MergerWeights& mw, int llm_dim) {
    const int out_h = grid_h / 2, out_w = grid_w / 2;
    std::vector<std::vector<float>> merged(static_cast<size_t>(out_h) * out_w);
    for (int br = 0; br < out_h; ++br) {
        for (int bc = 0; bc < out_w; ++bc) {
            std::vector<float> concat(static_cast<size_t>(4 * vit_dim));
            auto idx = block_patch_indices(br, bc, grid_w);
            for (int slot = 0; slot < 4; ++slot) {
                const auto& p = patches[static_cast<size_t>(idx[static_cast<size_t>(slot)])];
                std::copy(p.begin(), p.end(), concat.begin() + slot * vit_dim);
            }
            std::vector<float> h1(static_cast<size_t>(mw.hidden_dim));
            matmul(h1, concat, mw.W1, static_cast<size_t>(4 * vit_dim), static_cast<size_t>(mw.hidden_dim));
            for (int i = 0; i < mw.hidden_dim; ++i) h1[static_cast<size_t>(i)] = silu(h1[static_cast<size_t>(i)] + mw.b1[static_cast<size_t>(i)]);
            std::vector<float> out(static_cast<size_t>(llm_dim));
            matmul(out, h1, mw.W2, static_cast<size_t>(mw.hidden_dim), static_cast<size_t>(llm_dim));
            for (int i = 0; i < llm_dim; ++i) out[static_cast<size_t>(i)] += mw.b2[static_cast<size_t>(i)];
            merged[static_cast<size_t>(br) * out_w + bc] = std::move(out);
        }
    }
    return merged;
}


// =======================================================================
// PART 6 (new): the fusion itself. `decode_step_with_embedding` is
// Section 16.3's own `decode_step`, factored so the "how do I get this
// position's starting vector" question is answered by the CALLER --
// `decode_step` (below) answers it the original way (look up a real
// token id); `build_fused_embeddings` answers it a second way (splice in
// an already-computed visual token) for exactly the positions a caller
// marks as image positions.
// =======================================================================
struct DecodeStepResult { std::vector<float> hidden; int nan_at_layer = -1; };
DecodeStepResult decode_step_with_embedding(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                                              std::vector<KVCache>& caches, std::vector<float> x, int pos) {
    DecodeStepResult res;
    for (int layer = 0; layer < model.n_layers(); ++layer) {
        qwen2_block_forward(x, model.shape, layers[static_cast<size_t>(layer)], caches[static_cast<size_t>(layer)], pos, *model.rope);
        bool has_nan = false;
        for (float v : x) if (std::isnan(v)) { has_nan = true; break; }
        if (has_nan) { res.nan_at_layer = layer; break; }
    }
    res.hidden = std::move(x);
    return res;
}
DecodeStepResult decode_step(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                               std::vector<KVCache>& caches, int token_id, int pos) {
    return decode_step_with_embedding(model, layers, caches, model.embedding(token_id), pos);
}

// A sentinel token id reserved (by this section's own synthetic
// vocabulary, exactly the way real Qwen2-VL reserves `<|image_pad|>`'s
// own real id) to mark "a visual token belongs here" inside an otherwise
// ordinary text token-id sequence.
constexpr int IMAGE_PLACEHOLDER_ID = -1;

struct FusionResult {
    bool ok = false;
    std::string error;
    std::vector<std::vector<float>> embeddings;   // one per position, text-looked-up or visual-injected
};

// Walks `token_ids` in order, replacing every `IMAGE_PLACEHOLDER_ID`
// with the NEXT unused visual token from `visual_tokens`, in sequence --
// and refuses, rather than silently truncating or zero-padding, if the
// placeholder count and the visual token count do not match exactly.
// A mismatch here is exactly the kind of silent-misalignment bug this
// book has refused to let slide since Chapter 15's own KVCache-sharing
// bug: every later position's own attention would still run and produce
// SOME finite number, with nothing about the output signaling that the
// image tokens landed in the wrong places, or that some were reused, or
// dropped.
FusionResult build_fused_embeddings(const QwenModel& model, const std::vector<int>& token_ids,
                                     const std::vector<std::vector<float>>& visual_tokens) {
    FusionResult res;
    size_t used = 0;
    res.embeddings.reserve(token_ids.size());
    for (int tid : token_ids) {
        if (tid == IMAGE_PLACEHOLDER_ID) {
            if (used >= visual_tokens.size()) {
                res.error = "more image placeholders than visual tokens";
                return res;
            }
            res.embeddings.push_back(visual_tokens[used++]);
        } else {
            res.embeddings.push_back(model.embedding(tid));
        }
    }
    if (used != visual_tokens.size()) {
        res.error = "fewer image placeholders than visual tokens (" + std::to_string(used) +
                    " used, " + std::to_string(visual_tokens.size()) + " provided)";
        return res;
    }
    res.ok = true;
    return res;
}

struct MultimodalPrefillResult { std::vector<float> hidden; bool exceeded_capacity = false; bool fusion_failed = false;
                                  std::string fusion_error; int nan_at_layer = -1; };
MultimodalPrefillResult prefill_multimodal(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                                            std::vector<KVCache>& caches, const std::vector<int>& token_ids,
                                            const std::vector<std::vector<float>>& visual_tokens,
                                            int start_pos, int kv_capacity) {
    MultimodalPrefillResult res;
    auto fused = build_fused_embeddings(model, token_ids, visual_tokens);
    if (!fused.ok) { res.fusion_failed = true; res.fusion_error = fused.error; return res; }
    for (size_t i = 0; i < fused.embeddings.size(); ++i) {
        int pos = start_pos + static_cast<int>(i);
        if (pos >= kv_capacity) { res.exceeded_capacity = true; break; }
        auto step = decode_step_with_embedding(model, layers, caches, fused.embeddings[i], pos);
        res.hidden = std::move(step.hidden);
        if (step.nan_at_layer >= 0) { res.nan_at_layer = step.nan_at_layer; break; }
    }
    return res;
}

// Runs Section 18.2's own encoder end to end -- patchify, embed, N ViT
// blocks, merge -- and returns the resulting visual tokens, projected to
// the LLM's own embedding dimension so `build_fused_embeddings` can drop
// them straight into a text sequence with no further adaptation.
struct VisionEncoderWeights {
    std::vector<float> W_patch_embed;
    std::vector<ViTBlockWeights> layers;
    MergerWeights merger;
};
std::vector<std::vector<float>> run_vision_encoder(const RawImage& image, int patch_size, const NormStats& norm,
                                                    const ViTShape& shape, const VisionEncoderWeights& w,
                                                    int llm_dim, uint32_t& out_grid_h, uint32_t& out_grid_w) {
    uint32_t grid_h = 0, grid_w = 0;
    auto raw_patches = patchify(image, static_cast<uint32_t>(patch_size), norm, grid_h, grid_w);
    const int n_patches = static_cast<int>(raw_patches.size());
    const size_t patch_vec_len = raw_patches[0].size();

    std::vector<std::vector<float>> x(static_cast<size_t>(n_patches));
    std::vector<int> rows(static_cast<size_t>(n_patches)), cols(static_cast<size_t>(n_patches));
    for (int i = 0; i < n_patches; ++i) {
        x[static_cast<size_t>(i)].resize(static_cast<size_t>(shape.dim));
        matmul(x[static_cast<size_t>(i)], raw_patches[static_cast<size_t>(i)], w.W_patch_embed, patch_vec_len, static_cast<size_t>(shape.dim));
        rows[static_cast<size_t>(i)] = i / static_cast<int>(grid_w);
        cols[static_cast<size_t>(i)] = i % static_cast<int>(grid_w);
    }
    RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, shape.head_dim, 10000.0f);
    for (const auto& l : w.layers) vit_block_forward(x, shape, l, rows, cols, rope);

    out_grid_h = grid_h; out_grid_w = grid_w;
    return merge_all_2x2(x, static_cast<int>(grid_h), static_cast<int>(grid_w), shape.dim, w.merger, llm_dim);
}


// =======================================================================
// PART 7 (new): batch processing over multiple shelf photos, and a
// strict parser for a VLM's own structured detection-line output.
// =======================================================================
struct ShelfPhoto {
    std::string photo_id;
    RawImage image;
};

struct BatchPhotoResult {
    std::string photo_id;
    bool ok = false;
    std::string error;
    std::vector<float> hidden;   // final fused hidden state, only meaningful when ok
};

struct BatchRunResult {
    std::vector<BatchPhotoResult> results;
    size_t succeeded = 0;
    size_t failed = 0;
};

// Runs Chapter 18's own real vision-language pipeline over every photo
// in `photos`, in order, against the SAME `token_template` (a real batch
// job inspects many photos against the same planogram-derived prompt,
// not a different one per photo). Each photo's own outcome -- success,
// a fusion count mismatch, KV-capacity exhaustion, or a NaN -- is
// recorded into its own `BatchPhotoResult`; nothing about one photo's
// failure stops the loop from continuing to the next photo, exactly the
// fail-loud-but-not-fail-STOP distinction this book has drawn since
// Section 18.1's own dropped-packet handling.
BatchRunResult process_batch(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                              const std::vector<ShelfPhoto>& photos, const std::vector<int>& token_template,
                              int patch_size, const NormStats& norm, const ViTShape& vit_shape,
                              const VisionEncoderWeights& vision_weights, int kv_capacity) {
    BatchRunResult batch;
    for (const auto& photo : photos) {
        BatchPhotoResult r;
        r.photo_id = photo.photo_id;

        uint32_t gh = 0, gw = 0;
        auto visual_tokens = run_vision_encoder(photo.image, patch_size, norm, vit_shape, vision_weights,
                                                 model.shape.dim, gh, gw);
        std::vector<KVCache> caches;
        for (int l = 0; l < model.n_layers(); ++l) {
            caches.emplace_back(model.shape.n_heads_kv, kv_capacity, model.shape.head_dim);
        }
        auto step = prefill_multimodal(model, layers, caches, token_template, visual_tokens, 0, kv_capacity);
        if (step.fusion_failed) {
            r.error = "fusion failed: " + step.fusion_error;
        } else if (step.exceeded_capacity) {
            r.error = "exceeded KV capacity";
        } else if (step.nan_at_layer >= 0) {
            r.error = "NaN detected at layer " + std::to_string(step.nan_at_layer);
        } else {
            r.ok = true;
            r.hidden = std::move(step.hidden);
        }

        if (r.ok) ++batch.succeeded; else ++batch.failed;
        batch.results.push_back(std::move(r));
    }
    return batch;
}

// A single parsed detection, in the stated structured-text format a
// shelf-inspection prompt asks a real VLM to respond with:
// "SKU=<id>;SHELF=<int>;POSITION=<int>;CONFIDENCE=<float in [0,1]>".
struct ParsedDetection {
    std::string sku_id;
    int shelf_index = 0;
    int position_index = 0;
    double confidence = 0.0;
};

struct ParseResult {
    bool ok = false;
    std::string error;
    ParsedDetection detection;
};

// A strict, real parser -- every one of the four required fields must be
// present exactly once, SHELF and POSITION must be valid integers,
// CONFIDENCE must be a valid real number inside [0, 1], and no
// unrecognized extra field is silently ignored. `std::from_chars` is
// used throughout rather than the exception-throwing `std::stoi`/
// `std::stod` family, matching this book's own standing preference for
// real, checked return values over exceptions as its own control-flow
// mechanism -- exactly the reasoning behind Chapter 5's own GGUF reader
// reporting `.good()` rather than throwing on a malformed file.
ParseResult parse_detection_line(const std::string& line) {
    ParseResult res;
    std::unordered_map<std::string, std::string> fields;
    std::istringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ';')) {
        auto eq = field.find('=');
        if (eq == std::string::npos) {
            res.error = "malformed field (no '=' found): \"" + field + "\"";
            return res;
        }
        std::string key = field.substr(0, eq);
        std::string value = field.substr(eq + 1);
        if (fields.count(key)) {
            res.error = "duplicate field: " + key;
            return res;
        }
        fields[key] = value;
    }

    for (const char* required : {"SKU", "SHELF", "POSITION", "CONFIDENCE"}) {
        if (!fields.count(required)) {
            res.error = std::string("missing required field: ") + required;
            return res;
        }
    }
    if (fields.size() != 4) {
        res.error = "unexpected extra field(s) present (expected exactly SKU, SHELF, POSITION, CONFIDENCE)";
        return res;
    }
    if (fields["SKU"].empty()) {
        res.error = "SKU field is empty";
        return res;
    }

    int shelf = 0;
    const std::string& shelf_str = fields["SHELF"];
    auto shelf_parsed = std::from_chars(shelf_str.data(), shelf_str.data() + shelf_str.size(), shelf);
    if (shelf_parsed.ec != std::errc() || shelf_parsed.ptr != shelf_str.data() + shelf_str.size()) {
        res.error = "SHELF is not a valid integer: \"" + shelf_str + "\"";
        return res;
    }

    int position = 0;
    const std::string& pos_str = fields["POSITION"];
    auto pos_parsed = std::from_chars(pos_str.data(), pos_str.data() + pos_str.size(), position);
    if (pos_parsed.ec != std::errc() || pos_parsed.ptr != pos_str.data() + pos_str.size()) {
        res.error = "POSITION is not a valid integer: \"" + pos_str + "\"";
        return res;
    }

    double confidence = 0.0;
    const std::string& conf_str = fields["CONFIDENCE"];
    auto conf_parsed = std::from_chars(conf_str.data(), conf_str.data() + conf_str.size(), confidence);
    if (conf_parsed.ec != std::errc() || conf_parsed.ptr != conf_str.data() + conf_str.size()) {
        res.error = "CONFIDENCE is not a valid number: \"" + conf_str + "\"";
        return res;
    }
    if (confidence < 0.0 || confidence > 1.0) {
        res.error = "CONFIDENCE out of the valid [0, 1] range: " + conf_str;
        return res;
    }

    res.ok = true;
    res.detection = ParsedDetection{fields["SKU"], shelf, position, confidence};
    return res;
}

// =======================================================================
// PART 8: self-tests, reusing Section 18.3's own synthetic-model and
// synthetic-image construction pattern so every check here is
// deterministic and needs neither a real checkpoint nor a real camera.
// =======================================================================
std::vector<float> rand_vec(std::mt19937& rng, size_t n) {
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 19.2: A Batch Shelf-Photo Processor\n";
    std::cout << "========================================================\n";

    // ---- synthetic Qwen2-shaped decoder, exactly Section 18.3's own shape ----
    constexpr int S_DIM = 32, S_HEADS = 4, S_HEADS_KV = 2, S_HEAD_DIM = 8, S_FF = 64, S_LAYERS = 2, S_VOCAB = 64;
    const std::string synth_path = "/tmp/ch19_2_synthetic_model.gguf";
    auto write_synthetic_model = [&](unsigned seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.3f);
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
        auto rand_v = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
        add_q8("token_embd.weight", {S_DIM, S_VOCAB}, rand_v(static_cast<size_t>(S_DIM) * S_VOCAB));
        add_f32("output_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
        for (int layer = 0; layer < S_LAYERS; ++layer) {
            std::string p = "blk." + std::to_string(layer) + ".";
            add_f32(p + "attn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "attn_q.weight", {S_DIM, S_HEADS * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS * S_HEAD_DIM));
            add_f32(p + "attn_q.bias", {static_cast<uint64_t>(S_HEADS * S_HEAD_DIM)}, rand_v(S_HEADS * S_HEAD_DIM));
            add_q8(p + "attn_k.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_k.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_v(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_v.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_v(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_v.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_v(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_output.weight", {S_HEADS * S_HEAD_DIM, S_DIM}, rand_v(static_cast<size_t>(S_HEADS) * S_HEAD_DIM * S_DIM));
            add_f32(p + "ffn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "ffn_gate.weight", {S_DIM, S_FF}, rand_v(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_up.weight", {S_DIM, S_FF}, rand_v(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_down.weight", {S_FF, S_DIM}, rand_v(static_cast<size_t>(S_FF) * S_DIM));
        }
        GGUFWriter w(synth_path);
        w.write_magic(); w.write_u32(3);
        w.write_u64(pending.size()); w.write_u64(7);   // 7 kv pairs written below -- Section 16.3's own lesson
        w.write_kv_string("general.architecture", "qwen2");
        w.write_kv_u32("qwen2.block_count", S_LAYERS);
        w.write_kv_u32("qwen2.embedding_length", S_DIM);
        w.write_kv_u32("qwen2.attention.head_count", S_HEADS);
        w.write_kv_u32("qwen2.attention.head_count_kv", S_HEADS_KV);
        w.write_kv_u32("qwen2.feed_forward_length", S_FF);
        w.write_kv_f32("qwen2.rope.freq_base", 10000.0f);
        uint64_t off = 0;
        for (auto& p : pending) { w.write_tensor_info(p.name, p.dims, p.type, off); off += p.bytes.size(); }
        w.align(32);
        for (auto& p : pending) w.write_bytes(p.bytes.data(), p.bytes.size());
        return w.good();
    };
    CHECK(write_synthetic_model(7));
    QwenModel model;
    CHECK(model.load(synth_path));
    std::vector<QwenBlockWeights> layers;
    for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));

    // ---- synthetic ViT-shaped vision encoder, exactly Section 18.3's own shape ----
    constexpr int PATCH = 14, CH = 3, VIT_DIM = 16, V_HEADS = 2, V_HEAD_DIM = 8, V_FF = 24, V_LAYERS = 1;
    NormStats norm;
    auto make_vision_weights = [&](unsigned seed) {
        std::mt19937 rng(seed);
        VisionEncoderWeights vw;
        vw.W_patch_embed = rand_vec(rng, static_cast<size_t>(VIT_DIM) * (PATCH * PATCH * CH));
        vw.layers.resize(V_LAYERS);
        for (auto& l : vw.layers) {
            l.attn_norm.assign(VIT_DIM, 1.0f);
            l.Wq = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wk = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wv = rand_vec(rng, static_cast<size_t>(V_HEADS * V_HEAD_DIM) * VIT_DIM);
            l.Wo = rand_vec(rng, static_cast<size_t>(VIT_DIM) * (V_HEADS * V_HEAD_DIM));
            l.ffn_norm.assign(VIT_DIM, 1.0f);
            l.Wgate = rand_vec(rng, static_cast<size_t>(V_FF) * VIT_DIM);
            l.Wup = rand_vec(rng, static_cast<size_t>(V_FF) * VIT_DIM);
            l.Wdown = rand_vec(rng, static_cast<size_t>(VIT_DIM) * V_FF);
        }
        vw.merger.hidden_dim = 20;
        vw.merger.W1 = rand_vec(rng, static_cast<size_t>(vw.merger.hidden_dim) * (4 * VIT_DIM));
        vw.merger.b1 = rand_vec(rng, static_cast<size_t>(vw.merger.hidden_dim));
        vw.merger.W2 = rand_vec(rng, static_cast<size_t>(S_DIM) * vw.merger.hidden_dim);
        vw.merger.b2 = rand_vec(rng, static_cast<size_t>(S_DIM));
        return vw;
    };
    VisionEncoderWeights vision_weights = make_vision_weights(123);
    ViTShape vit_shape{VIT_DIM, V_HEADS, V_HEAD_DIM, V_FF};

    // 56x56 with PATCH=14 -> a 4x4 patch grid -> a 2x2 merge -> 4 visual
    // tokens, matching the 4-placeholder template below exactly.
    auto make_image = [&](unsigned seed, int size) {
        RawImage img; img.width = size; img.height = size; img.channels = CH;
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> byte_dist(0, 255);
        img.pixels.resize(static_cast<size_t>(img.width) * img.height * CH);
        for (auto& b : img.pixels) b = static_cast<uint8_t>(byte_dist(rng));
        return img;
    };

    std::vector<int> token_template = {5, 6, IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID,
                                        IMAGE_PLACEHOLDER_ID, IMAGE_PLACEHOLDER_ID, 7, 8};

    std::cout << "\n-- Test 1: a batch of well-formed photos runs end-to-end, finite and deterministic --\n";
    {
        constexpr int KV_CAP = 32;
        std::vector<ShelfPhoto> photos = {
            {"photo-001", make_image(10, 56)},
            {"photo-002", make_image(20, 56)},
            {"photo-003", make_image(30, 56)},
        };
        auto run1 = process_batch(model, layers, photos, token_template, PATCH, norm, vit_shape, vision_weights, KV_CAP);
        auto run2 = process_batch(model, layers, photos, token_template, PATCH, norm, vit_shape, vision_weights, KV_CAP);
        CHECK(run1.succeeded == 3);
        CHECK(run1.failed == 0);
        bool all_finite = true;
        for (const auto& r : run1.results) {
            if (!r.ok) { all_finite = false; continue; }
            for (float v : r.hidden) if (!std::isfinite(v)) all_finite = false;
        }
        CHECK(all_finite);
        bool deterministic = (run1.results.size() == run2.results.size());
        for (size_t i = 0; deterministic && i < run1.results.size(); ++i) {
            deterministic = (run1.results[i].ok == run2.results[i].ok) && (run1.results[i].hidden == run2.results[i].hidden);
        }
        CHECK(deterministic);
        std::cout << "  " << run1.succeeded << "/" << photos.size() << " photos succeeded, all finite: "
                   << (all_finite ? "yes" : "no") << ", identical across two independent batch runs: "
                   << (deterministic ? "yes" : "no") << "\n";
    }

    std::cout << "\n-- Test 2: one malformed photo among several good ones fails in isolation --\n";
    {
        constexpr int KV_CAP = 32;
        // photo-bad is 84x84 -- a 6x6 patch grid, a 3x3 (9-token) merge --
        // which does not match the template's 4 image placeholders, and
        // must trip build_fused_embeddings's own count-mismatch refusal
        // rather than silently truncating or crashing the whole batch.
        std::vector<ShelfPhoto> photos = {
            {"photo-001", make_image(10, 56)},
            {"photo-bad", make_image(40, 84)},
            {"photo-003", make_image(30, 56)},
        };
        auto run = process_batch(model, layers, photos, token_template, PATCH, norm, vit_shape, vision_weights, KV_CAP);
        CHECK(run.succeeded == 2);
        CHECK(run.failed == 1);
        CHECK(run.results.size() == 3);
        CHECK(run.results[0].ok);
        CHECK(!run.results[1].ok);
        CHECK(run.results[1].photo_id == "photo-bad");
        CHECK(run.results[1].error.find("fusion failed") != std::string::npos);
        CHECK(run.results[2].ok);
        std::cout << "  batch of 3: \"" << run.results[0].photo_id << "\" ok, \"" << run.results[1].photo_id
                   << "\" correctly isolated as failed (\"" << run.results[1].error << "\"), \""
                   << run.results[2].photo_id << "\" still ok -- one bad photo did not take down the batch\n";
    }

    std::cout << "\n-- Test 3: the structured-output parser accepts well-formed detection lines --\n";
    {
        auto r1 = parse_detection_line("SKU=10001;SHELF=1;POSITION=2;CONFIDENCE=0.93");
        CHECK(r1.ok);
        CHECK(r1.detection.sku_id == "10001" && r1.detection.shelf_index == 1 &&
              r1.detection.position_index == 2 && std::abs(r1.detection.confidence - 0.93) < 1e-9);

        auto r2 = parse_detection_line("SKU=SNACK-77;SHELF=3;POSITION=0;CONFIDENCE=1");
        CHECK(r2.ok);
        CHECK(r2.detection.confidence == 1.0);

        auto r3 = parse_detection_line("SHELF=2;SKU=A1;CONFIDENCE=0.0;POSITION=5");   // field order does not matter
        CHECK(r3.ok);
        CHECK(r3.detection.shelf_index == 2 && r3.detection.confidence == 0.0);

        std::cout << "  three well-formed lines (varying field order, integer and boundary confidence "
                     "values 0.0/1.0) all parsed correctly: SKU=" << r1.detection.sku_id
                   << " shelf=" << r1.detection.shelf_index << " position=" << r1.detection.position_index
                   << " confidence=" << r1.detection.confidence << "\n";
    }

    std::cout << "\n-- Test 4: the structured-output parser rejects every real kind of malformed line --\n";
    {
        auto missing = parse_detection_line("SKU=10001;SHELF=1;POSITION=2");
        CHECK(!missing.ok);
        CHECK(missing.error.find("missing required field") != std::string::npos);

        auto dup = parse_detection_line("SKU=10001;SKU=10002;SHELF=1;POSITION=2;CONFIDENCE=0.5");
        CHECK(!dup.ok);
        CHECK(dup.error.find("duplicate field") != std::string::npos);

        auto extra = parse_detection_line("SKU=10001;SHELF=1;POSITION=2;CONFIDENCE=0.5;EXTRA=1");
        CHECK(!extra.ok);
        CHECK(extra.error.find("extra field") != std::string::npos);

        auto bad_int = parse_detection_line("SKU=10001;SHELF=one;POSITION=2;CONFIDENCE=0.5");
        CHECK(!bad_int.ok);
        CHECK(bad_int.error.find("SHELF") != std::string::npos);

        auto bad_conf_format = parse_detection_line("SKU=10001;SHELF=1;POSITION=2;CONFIDENCE=high");
        CHECK(!bad_conf_format.ok);
        CHECK(bad_conf_format.error.find("CONFIDENCE") != std::string::npos);

        auto bad_conf_range = parse_detection_line("SKU=10001;SHELF=1;POSITION=2;CONFIDENCE=1.5");
        CHECK(!bad_conf_range.ok);
        CHECK(bad_conf_range.error.find("range") != std::string::npos);

        auto empty_sku = parse_detection_line("SKU=;SHELF=1;POSITION=2;CONFIDENCE=0.5");
        CHECK(!empty_sku.ok);
        CHECK(empty_sku.error.find("SKU") != std::string::npos);

        auto no_equals = parse_detection_line("SKU=10001;GARBAGE;POSITION=2;CONFIDENCE=0.5");
        CHECK(!no_equals.ok);
        CHECK(no_equals.error.find("malformed field") != std::string::npos);

        std::cout << "  all 8 malformed cases correctly rejected: missing field, duplicate field, "
                     "unexpected extra field, non-integer SHELF, non-numeric CONFIDENCE, out-of-range "
                     "CONFIDENCE, empty SKU, and a field with no '=' at all\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_batch_shelf_photo_processor.cpp -o 02_batch_shelf_photo_processor
./02_batch_shelf_photo_processor
```

**Sample input:** a batch of three well-formed synthetic photos run end-to-end through the real vision-language pipeline, checked for finiteness and determinism across two independent batch runs; a batch with one deliberately mis-sized photo among two good ones, checked to isolate exactly that photo's failure while the other two still succeed; the structured-output parser checked against three well-formed lines (varying field order and boundary confidence values); and the parser checked to reject all eight real kinds of malformed input with specific, distinguishing errors.

```text
========================================================
Chapter 19.2: A Batch Shelf-Photo Processor
========================================================

-- Test 1: a batch of well-formed photos runs end-to-end, finite and deterministic --
  3/3 photos succeeded, all finite: yes, identical across two independent batch runs: yes

-- Test 2: one malformed photo among several good ones fails in isolation --
  batch of 3: "photo-001" ok, "photo-bad" correctly isolated as failed ("fusion failed: fewer image placeholders than visual tokens (4 used, 9 provided)"), "photo-003" still ok -- one bad photo did not take down the batch

-- Test 3: the structured-output parser accepts well-formed detection lines --
  three well-formed lines (varying field order, integer and boundary confidence values 0.0/1.0) all parsed correctly: SKU=10001 shelf=1 position=2 confidence=0.93

-- Test 4: the structured-output parser rejects every real kind of malformed line --
  all 8 malformed cases correctly rejected: missing field, duplicate field, unexpected extra field, non-integer SHELF, non-numeric CONFIDENCE, out-of-range CONFIDENCE, empty SKU, and a field with no '=' at all

36/36 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] one bad photo aborting an entire batch turns a small, isolated problem into a large one"
    A batch loop that lets one photo's exception propagate out of the whole function trades a small, specific, easily-diagnosed problem (photo `photo-bad` failed with a fusion count mismatch) for a large, vague one (the ENTIRE batch of photos this run was supposed to inspect produced no results at all). On a real production line running this batch job on a fixed schedule, that difference is the difference between one shelf going un-reconciled today and the ENTIRE STORE going un-reconciled today, from a single photo that was probably just corrupted or mis-captured. `process_batch`'s own per-photo isolation, tested directly in Test 2, is what keeps a real, specific failure from ever escalating into an unrelated, unnecessary one.

## 19.3 REST Integration into ERP/WMS Systems

### Intuition

Sections 19.1 and 19.2 can now tell what a shelf should hold and what a photograph shows it actually holds. None of that reaches a retailer's own business systems until it crosses a real interface nearly every modern retail backend actually exposes: a REST API exchanging JSON over HTTP's own real methods and status codes.

### The Concept, In Detail

This section is stated as an INTERFACE-LEVEL abstraction, the same honest voice Chapter 18.5 applied to OPC UA: there is no real ERP or WMS endpoint reachable from this environment, so this section builds its own small, from-scratch JSON serializer -- with real string escaping, verified in Test 1 against a string containing a newline, a quote, and a backslash all at once -- and a `SyntheticErpBackend` implementing real REST semantics standing in for a genuine system. What this section does not claim is byte-for-byte compliance with any one real vendor's own actual API shape; what it teaches faithfully is the general discipline every one of those real systems' own integrations still has to get right.

That discipline centers on optimistic concurrency. A real inventory record is being updated by this chapter's own reconciliation job, by cashiers ringing up sales, and by other stores' own systems, all at once -- a PUT that blindly overwrites whatever is currently stored risks a genuine LOST UPDATE, where two concurrent writers each believe they are applying the authoritative value and whichever lands second silently erases the other. `put_stock_level`'s own `expected_version` parameter is the same real idea as an HTTP `If-Match` header carrying an ETag: Test 4 confirms a PUT whose caller does not currently hold the record's own latest version is refused with a real `409 Conflict`, and -- just as important as the refusal itself -- that the record's stored value is left completely untouched by the rejected write, never partially applied. Test 3 draws the finer distinction this discipline depends on: PUTting the identical quantity a second time, using the version the FIRST PUT actually returned, succeeds and leaves the business-observable quantity unchanged, even though the bookkeeping version counter itself still advances -- version numbers are not business state, only the mechanism that protects it. Test 6 draws the matching contrast on the POST side: `create_shrinkage_ticket` is a real, intentionally NON-idempotent operation, and two calls with byte-identical inputs correctly produce two different ticket ids, because each POST genuinely creates a new resource rather than updating an existing one.

### Code and Verification

```cpp
// Chapter 19.3 -- Section 19.2's batch processor can now tell, for a
// single photo, whether a shelf slot looks correctly stocked. None of
// that is useful to an actual retailer until it reaches the systems that
// already run the store: an ERP tracking on-hand inventory, a WMS
// tracking warehouse stock, both reachable over the same real interface
// nearly every modern retail backend actually exposes -- a REST API
// exchanging JSON over HTTP's own real methods and status codes.
//
// This section is stated as an INTERFACE-LEVEL abstraction, in the same
// honest voice Chapter 18.5 applied to OPC UA: there is no real ERP or
// WMS endpoint reachable from this environment to integrate against, so
// this section builds its own small, from-scratch JSON serializer and a
// synthetic backend implementing REAL REST semantics -- real HTTP status
// codes, real optimistic-concurrency conflict detection, real malformed-
// request rejection -- standing in for a genuine ERP/WMS system. What
// this section does NOT claim is compliance with any one real vendor's
// actual API shape (SAP, NetSuite, and a dozen others each expose their
// own real, incompatible endpoint conventions); what it DOES teach
// faithfully is the general REST discipline every one of those real
// systems' own integrations still has to get right.
//
// The one piece of REST discipline this section centers on is
// optimistic concurrency: a real inventory record updated by an ERP
// integration is also being updated by cashiers ringing up sales, other
// stores' own systems, and human stock takes, all at once. A PUT that
// blindly overwrites whatever is currently stored risks a genuine LOST
// UPDATE -- two concurrent writers each believe they are applying the
// authoritative new value, and whichever one's PUT lands second silently
// erases the other's. This section's own `expected_version` parameter
// (the same real idea as an HTTP `If-Match` header carrying an ETag) is
// how `SyntheticErpBackend::put_stock_level` refuses that silent
// overwrite: a PUT whose caller does not currently hold the record's own
// latest version is rejected with a real `409 Conflict`, never quietly
// applied over a value the caller never actually saw.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_erp_wms_rest_integration.cpp -o 03_erp_wms_rest_integration
// Run:     ./03_erp_wms_rest_integration

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: a small, from-scratch JSON value type and serializer -- the
// real subset this section's own request and response bodies need
// (null, bool, integer, double, string, and a string-keyed object with
// insertion order preserved for deterministic output), with real string
// escaping so a SKU id or a ticket description containing a quote,
// backslash, or newline still serializes to valid JSON.
// =======================================================================
enum class JsonType { Null, Bool, Int, Double, String, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    bool b = false;
    int64_t i = 0;
    double d = 0.0;
    std::string s;
    std::vector<std::pair<std::string, JsonValue>> obj;

    static JsonValue make_bool(bool v) { JsonValue j; j.type = JsonType::Bool; j.b = v; return j; }
    static JsonValue make_int(int64_t v) { JsonValue j; j.type = JsonType::Int; j.i = v; return j; }
    static JsonValue make_double(double v) { JsonValue j; j.type = JsonType::Double; j.d = v; return j; }
    static JsonValue make_string(std::string v) { JsonValue j; j.type = JsonType::String; j.s = std::move(v); return j; }
    static JsonValue make_object(std::vector<std::pair<std::string, JsonValue>> fields) {
        JsonValue j; j.type = JsonType::Object; j.obj = std::move(fields); return j;
    }
};

std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    std::ostringstream esc;
                    esc << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                    out += esc.str();
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string serialize_json(const JsonValue& v) {
    switch (v.type) {
        case JsonType::Null: return "null";
        case JsonType::Bool: return v.b ? "true" : "false";
        case JsonType::Int: return std::to_string(v.i);
        case JsonType::Double: {
            std::ostringstream out;
            out << v.d;
            return out.str();
        }
        case JsonType::String: return "\"" + escape_json_string(v.s) + "\"";
        case JsonType::Object: {
            std::ostringstream out;
            out << "{";
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) out << ",";
                out << "\"" << escape_json_string(v.obj[i].first) << "\":" << serialize_json(v.obj[i].second);
            }
            out << "}";
            return out.str();
        }
    }
    return "null";
}

// =======================================================================
// PART 2: real HTTP method and status modeling -- real, standard status
// codes (200, 201, 400, 404, 409), not invented placeholders.
// =======================================================================
enum class HttpStatus : int {
    OK = 200,
    Created = 201,
    BadRequest = 400,
    NotFound = 404,
    Conflict = 409,
};

struct HttpResponse {
    HttpStatus status;
    JsonValue body;
};

// =======================================================================
// PART 3: SyntheticErpBackend -- a synthetic REST endpoint implementing
// real REST semantics, standing in for a genuine ERP/WMS system this
// environment has no way to reach. See the header note above for why
// optimistic concurrency (expected_version) is this section's own real
// centerpiece.
// =======================================================================
struct StockRecord {
    int64_t quantity = 0;
    uint64_t version = 0;
};

class SyntheticErpBackend {
public:
    HttpResponse get_stock_level(const std::string& store_id, const std::string& sku_id) const {
        auto it = records_.find(key(store_id, sku_id));
        if (it == records_.end()) {
            return HttpResponse{HttpStatus::NotFound,
                                 JsonValue::make_object({{"error", JsonValue::make_string("no stock record for this store/sku")}})};
        }
        return HttpResponse{HttpStatus::OK, JsonValue::make_object({
            {"store_id", JsonValue::make_string(store_id)},
            {"sku_id", JsonValue::make_string(sku_id)},
            {"quantity", JsonValue::make_int(it->second.quantity)},
            {"version", JsonValue::make_int(static_cast<int64_t>(it->second.version))},
        })};
    }

    // Sets the ABSOLUTE quantity for a (store, sku) pair. `expected_version`
    // is the caller's own belief about the record's CURRENT version --
    // 0 for "I believe no record exists yet." A caller whose belief does
    // not match the record's real current version is refused with a real
    // 409 Conflict rather than silently overwriting a value it never
    // actually observed; the stored record is left completely untouched
    // by a rejected write. A malformed request -- an empty sku_id, or a
    // negative quantity, which no real on-hand inventory count can be --
    // is refused with a real 400 Bad Request, also leaving any existing
    // record untouched.
    HttpResponse put_stock_level(const std::string& store_id, const std::string& sku_id,
                                  int64_t quantity, uint64_t expected_version) {
        if (sku_id.empty()) {
            return HttpResponse{HttpStatus::BadRequest,
                                 JsonValue::make_object({{"error", JsonValue::make_string("sku_id must not be empty")}})};
        }
        if (quantity < 0) {
            return HttpResponse{HttpStatus::BadRequest,
                                 JsonValue::make_object({{"error", JsonValue::make_string("quantity must not be negative")}})};
        }
        const std::string k = key(store_id, sku_id);
        auto it = records_.find(k);
        uint64_t current_version = (it == records_.end()) ? 0 : it->second.version;
        if (expected_version != current_version) {
            return HttpResponse{HttpStatus::Conflict, JsonValue::make_object({
                {"error", JsonValue::make_string("version mismatch -- record was modified since it was last read")},
                {"current_version", JsonValue::make_int(static_cast<int64_t>(current_version))},
            })};
        }
        StockRecord rec{quantity, current_version + 1};
        records_[k] = rec;
        return HttpResponse{HttpStatus::OK, JsonValue::make_object({
            {"store_id", JsonValue::make_string(store_id)},
            {"sku_id", JsonValue::make_string(sku_id)},
            {"quantity", JsonValue::make_int(quantity)},
            {"version", JsonValue::make_int(static_cast<int64_t>(rec.version))},
        })};
    }

    // Always creates a genuinely NEW ticket -- unlike put_stock_level,
    // there is no meaningful "expected version" for a POST that creates
    // a new resource each time it is called, which is exactly the real
    // REST distinction between an idempotent PUT and a non-idempotent
    // POST this section's own self-tests check directly.
    HttpResponse create_shrinkage_ticket(const std::string& store_id, const std::string& sku_id,
                                          const std::string& description) {
        if (description.empty()) {
            return HttpResponse{HttpStatus::BadRequest,
                                 JsonValue::make_object({{"error", JsonValue::make_string("description must not be empty")}})};
        }
        ++next_ticket_seq_;
        std::ostringstream ticket_id;
        ticket_id << "TCKT-" << std::setw(6) << std::setfill('0') << next_ticket_seq_;
        tickets_.push_back(Ticket{ticket_id.str(), store_id, sku_id, description});
        return HttpResponse{HttpStatus::Created, JsonValue::make_object({
            {"ticket_id", JsonValue::make_string(ticket_id.str())},
            {"store_id", JsonValue::make_string(store_id)},
            {"sku_id", JsonValue::make_string(sku_id)},
        })};
    }

    size_t ticket_count() const { return tickets_.size(); }

private:
    struct Ticket { std::string ticket_id, store_id, sku_id, description; };
    static std::string key(const std::string& store_id, const std::string& sku_id) { return store_id + "|" + sku_id; }

    std::unordered_map<std::string, StockRecord> records_;
    std::vector<Ticket> tickets_;
    uint64_t next_ticket_seq_ = 0;
};

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 19.3: REST Integration into ERP/WMS Systems\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the JSON serializer escapes real special characters and preserves field order --\n";
    {
        std::string tricky = "line1\nquote:\"back\\slash";
        std::string escaped = escape_json_string(tricky);
        CHECK(escaped.find('\n') == std::string::npos);   // the raw newline must not survive unescaped
        CHECK(escaped.find("\\n") != std::string::npos);
        CHECK(escaped.find("\\\"") != std::string::npos);
        CHECK(escaped.find("\\\\") != std::string::npos);

        JsonValue obj = JsonValue::make_object({
            {"a", JsonValue::make_int(1)},
            {"b", JsonValue::make_string("x")},
            {"c", JsonValue::make_bool(true)},
        });
        std::string rendered = serialize_json(obj);
        CHECK(rendered == "{\"a\":1,\"b\":\"x\",\"c\":true}");
        std::cout << "  a string with a newline, a quote, and a backslash escapes to \"" << escaped
                   << "\"; a 3-field object serializes to the exact expected string with field order preserved: "
                   << rendered << "\n";
    }

    std::cout << "\n-- Test 2: GET on an unknown record is a real 404, and a first PUT creates it --\n";
    {
        SyntheticErpBackend backend;
        auto miss = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(miss.status == HttpStatus::NotFound);

        auto created = backend.put_stock_level("store-042", "SKU-1001", 50, /*expected_version=*/0);
        CHECK(created.status == HttpStatus::OK);
        auto found = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(found.status == HttpStatus::OK);
        CHECK(found.body.obj[2].second.i == 50);   // quantity field
        CHECK(found.body.obj[3].second.i == 1);    // version field, now 1

        std::cout << "  GET before any record exists returns 404; a PUT with expected_version=0 "
                     "correctly creates the record at version 1 with quantity 50\n";
    }

    std::cout << "\n-- Test 3: setting the same quantity again with the current version is idempotent in effect --\n";
    {
        SyntheticErpBackend backend;
        backend.put_stock_level("store-042", "SKU-1001", 50, 0);   // version becomes 1
        auto repeat = backend.put_stock_level("store-042", "SKU-1001", 50, 1);   // same quantity, correct current version
        CHECK(repeat.status == HttpStatus::OK);
        auto after = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(after.body.obj[2].second.i == 50);   // the business-observable quantity is unchanged
        CHECK(after.body.obj[3].second.i == 2);    // the bookkeeping version still advances -- it is not business state
        std::cout << "  PUTting the same quantity (50) a second time, using the version the first PUT "
                     "returned, succeeds and leaves the observable quantity unchanged at 50 -- the "
                     "version counter itself advances to 2, but that is bookkeeping, not the business "
                     "value a retailer's own inventory dashboard would show\n";
    }

    std::cout << "\n-- Test 4: a stale expected_version is refused with 409, and never applied --\n";
    {
        SyntheticErpBackend backend;
        backend.put_stock_level("store-042", "SKU-1001", 50, 0);   // version becomes 1
        auto stale = backend.put_stock_level("store-042", "SKU-1001", 999, /*expected_version=*/0);   // stale: real version is 1
        CHECK(stale.status == HttpStatus::Conflict);
        auto after = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(after.body.obj[2].second.i == 50);   // the stale write's quantity (999) must NOT have landed
        CHECK(after.body.obj[3].second.i == 1);    // version must still be 1, not bumped by the rejected write
        std::cout << "  a PUT with a stale expected_version (0, when the record's real version is "
                     "already 1) is correctly rejected with 409 Conflict, and the record's stored "
                     "quantity remains 50 -- the rejected write's own quantity (999) never landed, "
                     "which is exactly the lost-update this section's optimistic concurrency exists "
                     "to prevent\n";
    }

    std::cout << "\n-- Test 5: malformed requests are refused with 400, and never change stored state --\n";
    {
        SyntheticErpBackend backend;
        backend.put_stock_level("store-042", "SKU-1001", 50, 0);

        auto empty_sku = backend.put_stock_level("store-042", "", 10, 0);
        CHECK(empty_sku.status == HttpStatus::BadRequest);

        auto negative_qty = backend.put_stock_level("store-042", "SKU-1001", -5, 1);
        CHECK(negative_qty.status == HttpStatus::BadRequest);

        auto after = backend.get_stock_level("store-042", "SKU-1001");
        CHECK(after.body.obj[2].second.i == 50);   // untouched by either rejected request
        CHECK(after.body.obj[3].second.i == 1);

        std::cout << "  an empty sku_id and a negative quantity are both refused with 400 Bad Request, "
                     "and the existing record's quantity (50) and version (1) are completely "
                     "unaffected by either rejected request\n";
    }

    std::cout << "\n-- Test 6: ticket creation is a real, non-idempotent POST -- each call makes a new ticket --\n";
    {
        SyntheticErpBackend backend;
        auto t1 = backend.create_shrinkage_ticket("store-042", "SKU-1001", "3 consecutive empty-slot observations");
        CHECK(t1.status == HttpStatus::Created);
        auto t2 = backend.create_shrinkage_ticket("store-042", "SKU-1001", "3 consecutive empty-slot observations");
        CHECK(t2.status == HttpStatus::Created);
        CHECK(t1.body.obj[0].second.s != t2.body.obj[0].second.s);   // two DIFFERENT ticket ids from identical inputs
        CHECK(backend.ticket_count() == 2);

        auto bad_ticket = backend.create_shrinkage_ticket("store-042", "SKU-1001", "");
        CHECK(bad_ticket.status == HttpStatus::BadRequest);
        CHECK(backend.ticket_count() == 2);   // the rejected request created no ticket

        std::cout << "  two POSTs with IDENTICAL inputs produce two DIFFERENT ticket ids (\""
                   << t1.body.obj[0].second.s << "\" and \"" << t2.body.obj[0].second.s
                   << "\") -- unlike put_stock_level's idempotent PUT, a POST is not expected to "
                     "collapse repeated calls into one outcome; an empty description is refused with "
                     "400 and correctly creates no ticket at all\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_erp_wms_rest_integration.cpp -o 03_erp_wms_rest_integration
./03_erp_wms_rest_integration
```

**Sample input:** the JSON serializer checked against a string with a newline, a quote, and a backslash, and against a real 3-field object's exact expected serialization with field order preserved; a GET on an unknown record checked to return 404 and a first PUT checked to create it at version 1; repeating the same PUT with the current version checked to leave the observable quantity unchanged; a stale-version PUT checked to be refused with 409 while leaving the stored record completely untouched; an empty SKU and a negative quantity each checked to be refused with 400 without side effects; and two identical-input POSTs to `create_shrinkage_ticket` checked to produce two different ticket ids.

```text
========================================================
Chapter 19.3: REST Integration into ERP/WMS Systems
========================================================

-- Test 1: the JSON serializer escapes real special characters and preserves field order --
  a string with a newline, a quote, and a backslash escapes to "line1\nquote:\"back\\slash"; a 3-field object serializes to the exact expected string with field order preserved: {"a":1,"b":"x","c":true}

-- Test 2: GET on an unknown record is a real 404, and a first PUT creates it --
  GET before any record exists returns 404; a PUT with expected_version=0 correctly creates the record at version 1 with quantity 50

-- Test 3: setting the same quantity again with the current version is idempotent in effect --
  PUTting the same quantity (50) a second time, using the version the first PUT returned, succeeds and leaves the observable quantity unchanged at 50 -- the version counter itself advances to 2, but that is bookkeeping, not the business value a retailer's own inventory dashboard would show

-- Test 4: a stale expected_version is refused with 409, and never applied --
  a PUT with a stale expected_version (0, when the record's real version is already 1) is correctly rejected with 409 Conflict, and the record's stored quantity remains 50 -- the rejected write's own quantity (999) never landed, which is exactly the lost-update this section's optimistic concurrency exists to prevent

-- Test 5: malformed requests are refused with 400, and never change stored state --
  an empty sku_id and a negative quantity are both refused with 400 Bad Request, and the existing record's quantity (50) and version (1) are completely unaffected by either rejected request

-- Test 6: ticket creation is a real, non-idempotent POST -- each call makes a new ticket --
  two POSTs with IDENTICAL inputs produce two DIFFERENT ticket ids ("TCKT-000001" and "TCKT-000002") -- unlike put_stock_level's idempotent PUT, a POST is not expected to collapse repeated calls into one outcome; an empty description is refused with 400 and correctly creates no ticket at all

26/26 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a PUT's version counter as though it were part of the business data"
    It is tempting to read `put_stock_level`'s returned version advancing from 1 to 2 as a sign that "something changed," when Test 3 deliberately constructs the case where the underlying, business-relevant quantity did NOT change at all -- the same value (50) was written twice, each time with the correct current version. The version counter's own real job is protecting against a LOST UPDATE, not describing the business state itself; a caller (or a dashboard built on top of this backend) that conflates "version incremented" with "inventory actually changed" would report a false change event on every single idempotent re-confirmation a real reconciliation job performs, drowning genuine change events in noise generated by nothing more than the concurrency-control mechanism doing its job correctly.

## 19.4 Multi-Store Aggregation and Trend Detection

### Intuition

A single store's own inventory record answers "how much of this SKU does store 42 have right now." A retail chain needs a genuinely different question answered across every store at once: is this SKU trending toward a stockout, and by when -- and answering that honestly requires knowing which stores' numbers are even reflected in the total.

### The Concept, In Detail

Every timestamp in this section is a caller-supplied discrete PERIOD INDEX, never real wall-clock time -- this book's own standing discipline against locking real, machine-specific timing into output this book's cross-architecture check compares byte-for-byte, applied here to a business reporting period instead of a hardware clock tick. `aggregate_by_sku` sums on-hand quantity across every store that actually reported a given period, and this section's own real discipline is what it does with a store that did NOT report: Test 2 confirms that store is excluded entirely from both the total and the reporting count, and named explicitly in `stores_missing`, rather than ever being folded into the total as an assumed zero. A store's silence is a fact about DATA COMPLETENESS, and treating it as a fact about INVENTORY would fabricate a stockout signal for a store that may simply have missed a scheduled upload.

`detect_trend` fits a real ordinary-least-squares regression line to a (period, quantity) time series -- Test 3 confirms the fitted slope and intercept against a perfectly linear synthetic decline are exact, hand-computable values, and that the resulting projected zero-inventory period matches a hand-computed zero-crossing precisely. A trend is only flagged StockoutRisk when it clears BOTH a rate threshold (declining at least a stated amount per period) AND a projection-horizon check (the regression line's own zero-crossing must fall within a stated number of future periods) -- Test 6 proves both gates are doing real, independent work by constructing a series that clears the rate threshold by a wide margin (a steep decline of 50 units per period) but starts from such a large quantity that its real zero-crossing sits thousands of periods away, far beyond any actionable horizon, and is correctly left Stable rather than raising an alert about a shortage that is not actually imminent.

### Code and Verification

```cpp
// Chapter 19.4 -- Section 19.3 taught one store's own inventory record
// how to update itself safely against concurrent writers. A retailer
// with hundreds of stores needs a second, genuinely different question
// answered: not "what is store 42's own count of SKU-1001 right now,"
// but "across every store, is SKU-1001 trending toward a stockout, and
// by when." This section builds that aggregation and trend-detection
// layer entirely from scratch -- real summation, a real least-squares
// linear regression, and a real, stated discipline against ever
// fabricating a number for a store that simply did not report.
//
// Every timestamp in this section is a caller-supplied discrete PERIOD
// INDEX, never real wall-clock time -- the same standing discipline this
// book has applied to every timing-adjacent value it has ever needed to
// keep deterministic and cross-architecture-comparable, from Chapter
// 10's thread scheduling to Chapter 18.1's trigger controller.
//
// The one real correctness discipline this section centers on: a store
// that did not report a period's snapshot is NOT the same fact as a
// store reporting zero on-hand inventory, and treating the two as
// interchangeable would fabricate a stockout signal for a store that may
// simply have missed a scheduled upload. `aggregate_by_sku` reports
// exactly which expected stores are missing from a given period,
// excludes them from that period's total rather than assuming zero, and
// reports how many stores' worth of real data the total actually
// reflects -- so a caller reading the aggregate can tell "42 units,
// fully reported" apart from "42 units, only 2 of 5 stores checked in."
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_multistore_aggregation_and_trends.cpp -o 04_multistore_aggregation_and_trends
// Run:     ./04_multistore_aggregation_and_trends

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: snapshots and honest, gap-aware aggregation.
// =======================================================================
struct StoreSnapshot {
    std::string store_id;
    std::string sku_id;
    int period = 0;                  // a discrete, caller-defined period index -- never real time
    int64_t on_hand_quantity = 0;
};

struct AggregateResult {
    int64_t total_quantity = 0;
    int stores_reporting = 0;
    std::vector<std::string> stores_missing;   // sorted, for deterministic output
};

// Sums `on_hand_quantity` across every snapshot matching `sku_id` and
// `period`, but ONLY for stores in `expected_stores` -- any expected
// store with no matching snapshot for this exact period is reported in
// `stores_missing` and excluded entirely from `total_quantity`, never
// silently treated as reporting zero.
AggregateResult aggregate_by_sku(const std::vector<StoreSnapshot>& snapshots, const std::string& sku_id,
                                  int period, const std::vector<std::string>& expected_stores) {
    AggregateResult res;
    std::set<std::string> reported;
    for (const auto& s : snapshots) {
        if (s.sku_id != sku_id || s.period != period) continue;
        if (!reported.insert(s.store_id).second) continue;   // a duplicate snapshot for the same store+period is not double-counted
        res.total_quantity += s.on_hand_quantity;
        ++res.stores_reporting;
    }
    for (const auto& store : expected_stores) {
        if (!reported.count(store)) res.stores_missing.push_back(store);
    }
    std::sort(res.stores_missing.begin(), res.stores_missing.end());
    return res;
}

// =======================================================================
// PART 2: real least-squares linear regression over a (period, quantity)
// time series, and the two real trend signals it powers.
// =======================================================================
enum class TrendSignal { Stable, StockoutRisk, Overstock };

std::string to_string(TrendSignal s) {
    switch (s) {
        case TrendSignal::Stable: return "STABLE";
        case TrendSignal::StockoutRisk: return "STOCKOUT_RISK";
        case TrendSignal::Overstock: return "OVERSTOCK";
    }
    return "UNKNOWN";
}

struct TrendThresholds {
    double stockout_slope_per_period = -2.0;    // flag only if declining at least this fast
    double overstock_slope_per_period = 5.0;    // flag only if growing at least this fast
    int projection_horizon_periods = 20;        // a stockout must be projected within this many periods to be actionable
};

struct TrendResult {
    TrendSignal signal = TrendSignal::Stable;
    double slope = 0.0;
    double intercept = 0.0;
    std::optional<int> projected_zero_period;
};

// A real ordinary-least-squares fit of quantity as a linear function of
// period: slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx), intercept from the
// mean point. The series is sorted by period internally so a caller does
// not have to pre-sort it, and needs at least two distinct periods --
// a single data point has no trend to fit at all.
TrendResult detect_trend(std::vector<std::pair<int, int64_t>> series, const TrendThresholds& t) {
    TrendResult res;
    std::sort(series.begin(), series.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const size_t n = series.size();
    if (n < 2) return res;   // Stable by default -- nothing to fit

    double sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (const auto& [period, qty] : series) {
        double x = static_cast<double>(period);
        double y = static_cast<double>(qty);
        sx += x; sy += y; sxy += x * y; sxx += x * x;
    }
    double n_d = static_cast<double>(n);
    double denom = n_d * sxx - sx * sx;
    if (denom == 0.0) return res;   // every period identical -- degenerate, treat as Stable rather than dividing by zero

    double slope = (n_d * sxy - sx * sy) / denom;
    double mean_x = sx / n_d, mean_y = sy / n_d;
    double intercept = mean_y - slope * mean_x;
    res.slope = slope;
    res.intercept = intercept;

    int last_period = series.back().first;

    if (slope <= t.stockout_slope_per_period) {
        // The regression line's own zero-crossing: intercept + slope*period = 0.
        double zero_period_d = -intercept / slope;
        int zero_period = static_cast<int>(std::ceil(zero_period_d));
        if (zero_period >= last_period && zero_period <= last_period + t.projection_horizon_periods) {
            res.signal = TrendSignal::StockoutRisk;
            res.projected_zero_period = zero_period;
            return res;
        }
        // Declining fast enough by rate, but the projected zero-crossing
        // falls outside the actionable horizon -- not flagged. A decline
        // that will not matter for years is not the same alert as one
        // that will matter next week, even at an identical per-period
        // rate on a much larger current quantity.
        return res;
    }
    if (slope >= t.overstock_slope_per_period) {
        res.signal = TrendSignal::Overstock;
        return res;
    }
    return res;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 19.4: Multi-Store Aggregation and Trend Detection\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: aggregation across fully-reporting stores matches a hand-computed total --\n";
    {
        std::vector<StoreSnapshot> snaps = {
            {"store-A", "SKU-1", 3, 100},
            {"store-B", "SKU-1", 3, 150},
            {"store-C", "SKU-1", 3, 75},
            {"store-A", "SKU-1", 2, 999},   // a different period -- must not be included
        };
        auto r = aggregate_by_sku(snaps, "SKU-1", 3, {"store-A", "store-B", "store-C"});
        CHECK(r.total_quantity == 325);
        CHECK(r.stores_reporting == 3);
        CHECK(r.stores_missing.empty());
        std::cout << "  3 stores at period 3: 100 + 150 + 75 = " << r.total_quantity
                   << ", all " << r.stores_reporting << " expected stores reporting, none missing\n";
    }

    std::cout << "\n-- Test 2: a store that did not report is excluded, never assumed to be zero --\n";
    {
        std::vector<StoreSnapshot> snaps = {
            {"store-A", "SKU-1", 5, 100},
            {"store-B", "SKU-1", 5, 150},
            // store-C has no snapshot at all for period 5
        };
        auto r = aggregate_by_sku(snaps, "SKU-1", 5, {"store-A", "store-B", "store-C"});
        CHECK(r.total_quantity == 250);          // NOT 250 + 0 == still 250, but for the RIGHT reason: store-C excluded, not counted as zero
        CHECK(r.stores_reporting == 2);
        CHECK(r.stores_missing.size() == 1 && r.stores_missing[0] == "store-C");
        std::cout << "  store-C never reported period 5: total is " << r.total_quantity
                   << " from " << r.stores_reporting << " reporting stores, with store-C correctly "
                     "listed as missing rather than silently folded into the total as a zero\n";
    }

    std::cout << "\n-- Test 3: a perfectly linear decline is flagged StockoutRisk with a hand-verifiable projection --\n";
    {
        std::vector<std::pair<int, int64_t>> series = {{0, 100}, {1, 90}, {2, 80}, {3, 70}, {4, 60}, {5, 50}};
        TrendThresholds t;   // stockout_slope_per_period = -2.0, horizon = 20
        auto r = detect_trend(series, t);
        CHECK(r.signal == TrendSignal::StockoutRisk);
        CHECK(std::abs(r.slope - (-10.0)) < 1e-9);       // exact for this perfectly linear series
        CHECK(std::abs(r.intercept - 100.0) < 1e-9);
        CHECK(r.projected_zero_period.has_value());
        CHECK(*r.projected_zero_period == 10);           // 100 - 10*period == 0 at period 10, hand-computed
        std::cout << "  a perfectly linear decline of 10 units/period (100 down to 50 over periods 0-5) "
                     "fits a regression slope of " << r.slope << " and intercept " << r.intercept
                   << ", correctly flagged StockoutRisk with a projected zero-inventory period of "
                   << *r.projected_zero_period << " -- matching the hand-computed zero-crossing exactly\n";
    }

    std::cout << "\n-- Test 4: a flat, noisy series is correctly left Stable --\n";
    {
        std::vector<std::pair<int, int64_t>> series = {{0, 100}, {1, 101}, {2, 99}, {3, 100}, {4, 100}, {5, 101}};
        auto r = detect_trend(series, TrendThresholds{});
        CHECK(r.signal == TrendSignal::Stable);
        std::cout << "  a series hovering around 100 units with small noise fits a near-zero slope ("
                   << r.slope << ") and is correctly left STABLE\n";
    }

    std::cout << "\n-- Test 5: a growing series is flagged Overstock --\n";
    {
        std::vector<std::pair<int, int64_t>> series = {{0, 50}, {1, 60}, {2, 70}, {3, 80}, {4, 90}, {5, 100}};
        auto r = detect_trend(series, TrendThresholds{});
        CHECK(r.signal == TrendSignal::Overstock);
        CHECK(std::abs(r.slope - 10.0) < 1e-9);
        std::cout << "  a perfectly linear growth of 10 units/period fits slope " << r.slope
                   << " and is correctly flagged OVERSTOCK\n";
    }

    std::cout << "\n-- Test 6: a decline fast enough by RATE but projected far beyond the horizon is not flagged --\n";
    {
        // Slope here is a steep -50/period (comfortably past the -2.0
        // rate threshold), but the starting quantity is enormous, so the
        // real zero-crossing sits thousands of periods away -- far
        // beyond any actionable horizon.
        std::vector<std::pair<int, int64_t>> series;
        int64_t start = 1'000'000;
        for (int p = 0; p <= 5; ++p) series.push_back({p, start - 50 * p});
        TrendThresholds t;   // projection_horizon_periods = 20
        auto r = detect_trend(series, t);
        CHECK(r.signal != TrendSignal::StockoutRisk);
        CHECK(r.signal == TrendSignal::Stable);
        double implied_zero = -r.intercept / r.slope;
        CHECK(implied_zero > 5 + t.projection_horizon_periods);   // confirm the real zero-crossing IS far beyond the horizon
        std::cout << "  a steep -50/period decline (slope " << r.slope << ", comfortably past the rate "
                     "threshold) starting from " << start << " units projects to zero at period "
                   << implied_zero << ", far beyond the " << t.projection_horizon_periods
                   << "-period horizon -- correctly left STABLE rather than raising an alert about a "
                     "shortage that is not actually imminent\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_multistore_aggregation_and_trends.cpp -o 04_multistore_aggregation_and_trends
./04_multistore_aggregation_and_trends
```

**Sample input:** aggregation across three fully-reporting stores checked against a hand-computed total; a store that did not report a period checked to be excluded from that period's total and named in `stores_missing`, never assumed to be zero; a perfectly linear decline checked against an exact, hand-computable regression slope, intercept, and zero-crossing projection; a flat noisy series checked to remain Stable; a perfectly linear growth checked to be flagged Overstock; and a steep-by-rate decline from an enormous starting quantity checked to remain Stable because its real projected zero-crossing falls far beyond the stated horizon.

```text
========================================================
Chapter 19.4: Multi-Store Aggregation and Trend Detection
========================================================

-- Test 1: aggregation across fully-reporting stores matches a hand-computed total --
  3 stores at period 3: 100 + 150 + 75 = 325, all 3 expected stores reporting, none missing

-- Test 2: a store that did not report is excluded, never assumed to be zero --
  store-C never reported period 5: total is 250 from 2 reporting stores, with store-C correctly listed as missing rather than silently folded into the total as a zero

-- Test 3: a perfectly linear decline is flagged StockoutRisk with a hand-verifiable projection --
  a perfectly linear decline of 10 units/period (100 down to 50 over periods 0-5) fits a regression slope of -10 and intercept 100, correctly flagged StockoutRisk with a projected zero-inventory period of 10 -- matching the hand-computed zero-crossing exactly

-- Test 4: a flat, noisy series is correctly left Stable --
  a series hovering around 100 units with small noise fits a near-zero slope (0.0857143) and is correctly left STABLE

-- Test 5: a growing series is flagged Overstock --
  a perfectly linear growth of 10 units/period fits slope 10 and is correctly flagged OVERSTOCK

-- Test 6: a decline fast enough by RATE but projected far beyond the horizon is not flagged --
  a steep -50/period decline (slope -50, comfortably past the rate threshold) starting from 1000000 units projects to zero at period 20000, far beyond the 20-period horizon -- correctly left STABLE rather than raising an alert about a shortage that is not actually imminent

17/17 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] folding a non-reporting store into an aggregate as an implicit zero"
    Summing on-hand quantity across "every store in my list" and simply not adding anything for a store with no snapshot LOOKS identical, in the resulting total, to correctly excluding that store -- until the very next question asked of that total is "how many units of this SKU exist across the chain," at which point a missing store's real, unknown, possibly-substantial inventory has silently become zero in every downstream calculation. `aggregate_by_sku`'s own `stores_missing` field exists specifically so a caller can distinguish "3 units total, all 3 stores reporting" from "3 units total, only 1 of 3 stores reporting" -- two totals that happen to share a number but mean completely different things about how much of the chain's real inventory that number actually reflects.

## 19.5 Shrinkage and Misplacement Detection from Ordinary Shelf Photographs

### Intuition

This section closes the loop the chapter opened: Section 19.1's planogram states what SHOULD be on a shelf, Section 19.2's batch processor and parser produce what a VLM actually observed there, and this section reconciles the two into the findings a store operator genuinely acts on.

### The Concept, In Detail

A misplacement is immediate: if a slot's own detection names a DIFFERENT SKU than the planogram expects, that is a fact about ONE photo, true the moment it is observed, and Test 1 confirms `ShrinkageDetector::reconcile` flags it on the very first call, with no corroboration required. A missing item is a genuinely different kind of claim. A single empty-looking photo could mean real shrinkage, or it could mean a customer mid-reach, a restocking cart blocking the shelf, or an ordinary brief gap between deliveries -- and reporting shrinkage off one such photo would be exactly the kind of overconfident claim this book has refused to make since its own earliest treatment of numerical precision. `ShrinkageDetector` instead tracks, per slot, how many CONSECUTIVE reconciliations in a row have found it empty, and only raises `SHRINKAGE_SUSPECTED` once a stated threshold is crossed -- Test 2 confirms a single miss is correctly left `EMPTY_OBSERVED`, and Test 3 confirms three consecutive misses correctly cross a threshold of three.

The counter's own reset behavior is this section's own sharpest correctness point, and Test 4 is built specifically to prove it: two misses, followed by the item genuinely reappearing, followed by two MORE misses, must land at a streak of 2 (still below threshold) rather than 4 -- a detector that merely PAUSED the count across the reappearance, rather than genuinely RESETTING it, would treat two unrelated two-miss runs as one continuous shrinkage signal the real data never supported. Test 5's own multi-period reconciliation report ties every piece of this chapter together into the artifact a store operator would actually receive: across five simulated periods, exactly one real misplacement event and exactly one slot ever crossing the shrinkage threshold, each attributable to a specific shelf position and a specific run of evidence. This section's own scope ends at producing that finding -- a real deployment would report each `SHRINKAGE_SUSPECTED` slot by calling Section 19.3's own `create_shrinkage_ticket` against the real ERP/WMS backend, closing the chapter's own loop from a shelf photograph all the way to an actionable ticket in the systems a retailer already runs.

### Code and Verification

```cpp
// Chapter 19.5 -- This section closes the loop this chapter opened:
// Section 19.1's planogram states what SHOULD be on a shelf, Section
// 19.2's batch processor and its structured-output parser produce what
// a VLM actually OBSERVED there, and this section reconciles the two
// into the two real findings a store operator actually acts on --
// MISPLACEMENT and SHRINKAGE -- which are genuinely different claims
// requiring genuinely different evidence.
//
// A misplacement is immediate: if a slot's own detection names a
// DIFFERENT sku than the planogram expects, that is a fact about THIS
// ONE photo, true the moment it is observed, and needs no corroboration
// from any other photo to report. A missing item is not the same kind
// of claim. A single photo showing an empty slot could mean real
// shrinkage, or it could mean a customer is mid-reach, a restocking cart
// is blocking the shelf, or the photo caught a brief, ordinary moment of
// bare shelf between two deliveries -- and reporting a shrinkage alert
// off ONE empty-looking photo would be exactly the kind of overconfident
// claim this book has refused to make since Chapter 15's own honest
// treatment of numerical precision. `ShrinkageDetector` tracks, per
// slot, how many CONSECUTIVE reconciliations in a row have found it
// empty, resets that count the instant the item reappears, and only
// raises SHRINKAGE_SUSPECTED once a stated threshold of consecutive
// misses is crossed -- corroboration over time standing in for the
// single asymmetric-confidence-threshold reasoning Section 18.4 already
// applied to a single photo's own disposition, now applied across a
// SEQUENCE of photos instead of a single one.
//
// This section's own scope ends at producing that finding. A real
// deployment would report each SHRINKAGE_SUSPECTED slot by calling
// Section 19.3's own `create_shrinkage_ticket` against the real ERP/WMS
// backend -- this section does not repeat that REST machinery here,
// since nothing about ITS OWN real subject, reconciling a planogram
// against detections gathered over time, depends on it.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_shrinkage_and_misplacement_detection.cpp -o 05_shrinkage_and_misplacement_detection
// Run:     ./05_shrinkage_and_misplacement_detection

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the planogram slot and parsed-detection shapes, exactly
// Sections 19.1 and 19.2's own structures, repeated here self-contained.
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

struct ParsedDetection {
    std::string sku_id;
    int shelf_index = 0;
    int position_index = 0;
    double confidence = 0.0;
};

struct SlotKey {
    int shelf_index;
    int position_index;
    bool operator==(const SlotKey& o) const { return shelf_index == o.shelf_index && position_index == o.position_index; }
};
struct SlotKeyHash {
    size_t operator()(const SlotKey& k) const {
        return std::hash<int>{}(k.shelf_index) * 31 + std::hash<int>{}(k.position_index);
    }
};

// =======================================================================
// PART 2: reconciliation -- immediate misplacement, and shrinkage that
// requires corroboration across consecutive reconciliations.
// =======================================================================
enum class SlotStatus { OK, MISPLACED, EMPTY_OBSERVED, SHRINKAGE_SUSPECTED };

std::string to_string(SlotStatus s) {
    switch (s) {
        case SlotStatus::OK: return "OK";
        case SlotStatus::MISPLACED: return "MISPLACED";
        case SlotStatus::EMPTY_OBSERVED: return "EMPTY_OBSERVED";
        case SlotStatus::SHRINKAGE_SUSPECTED: return "SHRINKAGE_SUSPECTED";
    }
    return "UNKNOWN";
}

struct SlotFinding {
    int shelf_index = 0;
    int position_index = 0;
    std::string expected_sku;
    std::string observed_sku;   // empty when no detection landed on this slot at all
    SlotStatus status = SlotStatus::OK;
};

class ShrinkageDetector {
public:
    explicit ShrinkageDetector(int consecutive_empty_threshold) : threshold_(consecutive_empty_threshold) {}

    // Reconciles ONE period's worth of detections against the planogram.
    // Must be called once per period, in chronological order, for the
    // per-slot consecutive-empty counters to mean anything real -- this
    // is real, caller-managed state, not a pure function, because
    // shrinkage is fundamentally a claim about a SEQUENCE of
    // observations, not any single one.
    std::vector<SlotFinding> reconcile(const Planogram& pg, const std::vector<ParsedDetection>& detections) {
        std::unordered_map<SlotKey, ParsedDetection, SlotKeyHash> by_slot;
        for (const auto& d : detections) {
            by_slot[SlotKey{d.shelf_index, d.position_index}] = d;
        }

        std::vector<SlotFinding> findings;
        findings.reserve(pg.slots.size());
        for (const auto& slot : pg.slots) {
            SlotKey key{slot.shelf_index, slot.position_index};
            SlotFinding f;
            f.shelf_index = slot.shelf_index;
            f.position_index = slot.position_index;
            f.expected_sku = slot.sku_id;

            auto it = by_slot.find(key);
            if (it == by_slot.end()) {
                int& streak = empty_streak_[key];
                ++streak;
                f.observed_sku = "";
                f.status = (streak >= threshold_) ? SlotStatus::SHRINKAGE_SUSPECTED : SlotStatus::EMPTY_OBSERVED;
            } else {
                empty_streak_[key] = 0;   // the item is visibly present again -- the streak resets, it does not merely pause
                f.observed_sku = it->second.sku_id;
                f.status = (it->second.sku_id == slot.sku_id) ? SlotStatus::OK : SlotStatus::MISPLACED;
            }
            findings.push_back(f);
        }
        return findings;
    }

    int current_streak(int shelf_index, int position_index) const {
        auto it = empty_streak_.find(SlotKey{shelf_index, position_index});
        return (it == empty_streak_.end()) ? 0 : it->second;
    }

private:
    int threshold_;
    std::unordered_map<SlotKey, int, SlotKeyHash> empty_streak_;
};

// A real reconciliation report summarizing a full multi-period run: the
// distinct slots that were EVER flagged SHRINKAGE_SUSPECTED across the
// run, and the total count of individual MISPLACED events (a slot
// misplaced in three different periods counts as three real events,
// since each is an independent observation a store associate would have
// had three separate chances to correct).
struct ReconciliationReport {
    std::vector<SlotKey> shrinkage_suspected_slots;
    int misplaced_event_count = 0;
};

ReconciliationReport summarize(const std::vector<std::vector<SlotFinding>>& all_periods) {
    ReconciliationReport report;
    std::unordered_map<SlotKey, bool, SlotKeyHash> seen_shrinkage;
    for (const auto& period : all_periods) {
        for (const auto& f : period) {
            if (f.status == SlotStatus::SHRINKAGE_SUSPECTED) {
                SlotKey key{f.shelf_index, f.position_index};
                if (!seen_shrinkage[key]) {
                    seen_shrinkage[key] = true;
                    report.shrinkage_suspected_slots.push_back(key);
                }
            } else if (f.status == SlotStatus::MISPLACED) {
                ++report.misplaced_event_count;
            }
        }
    }
    return report;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 19.5: Shrinkage and Misplacement Detection from Ordinary Shelf Photographs\n";
    std::cout << "========================================================\n";

    Planogram pg;
    pg.store_layout_id = "STORE-777";
    pg.slots = {
        {1, 1, "SKU-A", "Item A", 3},
        {1, 2, "SKU-B", "Item B", 2},
    };

    std::cout << "\n-- Test 1: a misplacement is flagged immediately, on the very first observation --\n";
    {
        ShrinkageDetector det(3);
        std::vector<ParsedDetection> obs = {
            {"SKU-A", 1, 1, 0.9},
            {"SKU-WRONG", 1, 2, 0.95},   // slot (1,2) expects SKU-B
        };
        auto findings = det.reconcile(pg, obs);
        CHECK(findings.size() == 2);
        CHECK(findings[0].status == SlotStatus::OK);
        CHECK(findings[1].status == SlotStatus::MISPLACED);
        CHECK(findings[1].observed_sku == "SKU-WRONG" && findings[1].expected_sku == "SKU-B");
        std::cout << "  slot (1,1) matches its expected SKU-A: OK; slot (1,2) shows SKU-WRONG instead "
                     "of the expected SKU-B: flagged MISPLACED on the very first photo, no corroboration needed\n";
    }

    std::cout << "\n-- Test 2: a single missing observation is NOT enough to suspect shrinkage --\n";
    {
        ShrinkageDetector det(3);
        std::vector<ParsedDetection> obs = {{"SKU-A", 1, 1, 0.9}};   // slot (1,2) has no detection at all
        auto findings = det.reconcile(pg, obs);
        CHECK(findings[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(findings[1].status != SlotStatus::SHRINKAGE_SUSPECTED);
        CHECK(det.current_streak(1, 2) == 1);
        std::cout << "  slot (1,2) missing on its first observation: correctly EMPTY_OBSERVED, not yet "
                     "SHRINKAGE_SUSPECTED (streak=" << det.current_streak(1, 2) << ", threshold=3)\n";
    }

    std::cout << "\n-- Test 3: three CONSECUTIVE missing observations cross the threshold --\n";
    {
        ShrinkageDetector det(3);
        std::vector<ParsedDetection> present = {{"SKU-A", 1, 1, 0.9}};   // slot (1,2) always missing below
        auto f1 = det.reconcile(pg, present);
        auto f2 = det.reconcile(pg, present);
        auto f3 = det.reconcile(pg, present);
        CHECK(f1[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(f2[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(f3[1].status == SlotStatus::SHRINKAGE_SUSPECTED);
        std::cout << "  slot (1,2) missing across 3 consecutive reconciliations: EMPTY_OBSERVED, "
                     "EMPTY_OBSERVED, then correctly SHRINKAGE_SUSPECTED on crossing the threshold "
                     "on the 3rd\n";
    }

    std::cout << "\n-- Test 4: the item reappearing resets the streak -- it does not merely pause it --\n";
    {
        ShrinkageDetector det(3);
        std::vector<ParsedDetection> present_both = {{"SKU-A", 1, 1, 0.9}, {"SKU-B", 1, 2, 0.9}};
        std::vector<ParsedDetection> missing_b = {{"SKU-A", 1, 1, 0.9}};

        det.reconcile(pg, missing_b);   // streak(1,2) = 1
        det.reconcile(pg, missing_b);   // streak(1,2) = 2
        auto reappear = det.reconcile(pg, present_both);   // reappears -- streak resets to 0
        CHECK(reappear[1].status == SlotStatus::OK);
        int streak_after_reset = det.current_streak(1, 2);
        CHECK(streak_after_reset == 0);

        auto after1 = det.reconcile(pg, missing_b);   // streak = 1 again, NOT 3
        auto after2 = det.reconcile(pg, missing_b);   // streak = 2, still below threshold
        CHECK(after1[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(after2[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(after2[1].status != SlotStatus::SHRINKAGE_SUSPECTED);
        std::cout << "  after 2 misses, the item reappears (streak resets to " << streak_after_reset
                   << "); two MORE misses after that reappearance land at streak="
                   << det.current_streak(1, 2) << ", correctly still EMPTY_OBSERVED rather than "
                     "treating the two separate miss-runs as one continuous 4-miss shrinkage streak\n";
    }

    std::cout << "\n-- Test 5: a full multi-period reconciliation report is correct and complete --\n";
    {
        ShrinkageDetector det(3);
        std::vector<std::vector<SlotFinding>> all_periods;
        // Period 1: both slots fine.
        all_periods.push_back(det.reconcile(pg, {{"SKU-A", 1, 1, 0.9}, {"SKU-B", 1, 2, 0.9}}));
        // Period 2: slot (1,1) misplaced (a real, immediate event); slot (1,2) fine.
        all_periods.push_back(det.reconcile(pg, {{"SKU-WRONG", 1, 1, 0.9}, {"SKU-B", 1, 2, 0.9}}));
        // Period 3: slot (1,1) back to normal; slot (1,2) goes missing.
        all_periods.push_back(det.reconcile(pg, {{"SKU-A", 1, 1, 0.9}}));
        // Period 4: slot (1,2) still missing.
        all_periods.push_back(det.reconcile(pg, {{"SKU-A", 1, 1, 0.9}}));
        // Period 5: slot (1,2) still missing -- crosses the threshold of 3.
        all_periods.push_back(det.reconcile(pg, {{"SKU-A", 1, 1, 0.9}}));

        auto report = summarize(all_periods);
        CHECK(report.misplaced_event_count == 1);           // exactly the one period-2 event
        CHECK(report.shrinkage_suspected_slots.size() == 1);
        CHECK(report.shrinkage_suspected_slots[0].shelf_index == 1 && report.shrinkage_suspected_slots[0].position_index == 2);

        std::cout << "  across 5 periods: exactly " << report.misplaced_event_count
                   << " misplacement event (period 2's slot (1,1)) and exactly "
                   << report.shrinkage_suspected_slots.size() << " slot ever flagged shrinkage-suspected "
                     "(slot (1,2), after 3 consecutive missing periods 3 through 5) -- a complete, "
                     "accurate report a real store operator could act on directly\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_shrinkage_and_misplacement_detection.cpp -o 05_shrinkage_and_misplacement_detection
./05_shrinkage_and_misplacement_detection
```

**Sample input:** a misplaced slot checked to be flagged on the very first observation; a single missing observation checked to remain `EMPTY_OBSERVED` below threshold; three consecutive missing observations checked to cross a threshold of three; an item reappearing after two misses checked to reset the streak to zero, with two further misses afterward correctly still below threshold rather than continuing a stale count; and a full five-period reconciliation checked against a hand-verified report of exactly one misplacement event and exactly one shrinkage-suspected slot.

```text
========================================================
Chapter 19.5: Shrinkage and Misplacement Detection from Ordinary Shelf Photographs
========================================================

-- Test 1: a misplacement is flagged immediately, on the very first observation --
  slot (1,1) matches its expected SKU-A: OK; slot (1,2) shows SKU-WRONG instead of the expected SKU-B: flagged MISPLACED on the very first photo, no corroboration needed

-- Test 2: a single missing observation is NOT enough to suspect shrinkage --
  slot (1,2) missing on its first observation: correctly EMPTY_OBSERVED, not yet SHRINKAGE_SUSPECTED (streak=1, threshold=3)

-- Test 3: three CONSECUTIVE missing observations cross the threshold --
  slot (1,2) missing across 3 consecutive reconciliations: EMPTY_OBSERVED, EMPTY_OBSERVED, then correctly SHRINKAGE_SUSPECTED on crossing the threshold on the 3rd

-- Test 4: the item reappearing resets the streak -- it does not merely pause it --
  after 2 misses, the item reappears (streak resets to 0); two MORE misses after that reappearance land at streak=2, correctly still EMPTY_OBSERVED rather than treating the two separate miss-runs as one continuous 4-miss shrinkage streak

-- Test 5: a full multi-period reconciliation report is correct and complete --
  across 5 periods: exactly 1 misplacement event (period 2's slot (1,1)) and exactly 1 slot ever flagged shrinkage-suspected (slot (1,2), after 3 consecutive missing periods 3 through 5) -- a complete, accurate report a real store operator could act on directly

18/18 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] pausing a streak counter across a reappearance instead of genuinely resetting it"
    A detector that only decrements or freezes its consecutive-miss counter when an item reappears, rather than resetting it to zero, would treat an item that goes missing for two periods, comes back for one photo, and then goes missing for two MORE periods as continuous evidence of the SAME four-period shrinkage event -- when the real, honest interpretation is two separate two-period gaps, each individually below whatever threshold a retailer considers actionable. Test 4's own construction proves `ShrinkageDetector` gets this right by checking the counter's exact value immediately after the reappearance (zero) and again after two further misses (two, not four) -- the single number that would silently reveal a "pause instead of reset" bug if the reset logic were wrong.

## Chapter Summary

This chapter took Chapter 18's real vision-language pipeline into a genuinely different deployment target: a retail shelf, where what should be there has to be stated as explicitly as what a camera actually shows. Section 19.1 built a real, validated planogram structure, a deterministic rendering with run-length compression for large stores, and a conservative token-budget guard that refuses to fabricate a silently truncated prompt. Section 19.2 ran Chapter 18's own real pipeline across a batch of photographs with per-photo failure isolation, and built a strict structured-output parser tested honestly against representative example completions rather than genuine output this book's own untrained synthetic weights could never produce. Section 19.3 modeled a REST integration into a retailer's ERP/WMS systems as a from-scratch JSON serializer and a synthetic backend enforcing real optimistic-concurrency conflict detection, distinguishing an idempotent PUT's bookkeeping version from its actual business state and contrasting it against a genuinely non-idempotent POST. Section 19.4 aggregated inventory honestly across many stores -- never fabricating a non-reporting store's contribution as zero -- and fit a real least-squares regression to detect a stockout or overstock trend, gated by both a rate threshold and a projection horizon so a slow decline on a large quantity is not mistaken for an imminent shortage. Section 19.5 closed the chapter's own loop, reconciling the planogram against detections gathered over time into an immediate misplacement finding and a genuinely more cautious, evidence-corroborated shrinkage finding, with a reset-on-reappearance discipline this section proved correct by hand.

## Self-Check Questions

1. Section 19.1's `estimate_token_count` applies a stated multiplier to a plain whitespace word count rather than counting words directly. Explain specifically why an UNDER-estimate would be the more dangerous error for a token-budget guard to make, compared to an OVER-estimate.
2. Section 19.1's `render_within_budget` tries the full rendering, then the compressed rendering, and only then refuses outright. Why does it never attempt a THIRD, more aggressive form of truncation instead of refusing?
3. Section 19.2's `process_batch` isolates each photo's own failure into its own `BatchPhotoResult`. Describe the specific, concrete difference in outcome between a batch job built this way and one where a single photo's fusion-count mismatch is allowed to raise an exception out of the whole batch function.
4. Section 19.2's own structured-output parser is tested against representative example completions rather than genuine output from the real forward pass its own Test 1 and Test 2 already verified. Explain exactly why this gap exists, and what Section 19.2 DOES verify for real despite it.
5. Section 19.3's Test 3 shows a PUT's own version counter advancing from 1 to 2 even though the business-observable quantity did not change. Explain why treating that version increment as evidence of a real inventory change would be a mistake.
6. Section 19.3's `create_shrinkage_ticket` is deliberately NOT idempotent, unlike `put_stock_level`. Explain the real difference between these two operations that justifies this design choice.
7. Section 19.4's `aggregate_by_sku` reports a `stores_missing` list rather than simply omitting a non-reporting store's contribution silently. What real downstream question becomes impossible to answer correctly without that list?
8. Section 19.4's Test 6 constructs a series with a steep, threshold-clearing decline rate that is nonetheless left `Stable`. Explain why BOTH a rate threshold and a projection-horizon check are necessary, using this test's own numbers.
9. Section 19.5 flags a misplacement immediately but requires multiple consecutive observations before suspecting shrinkage. Explain the real, substantive difference between these two claims that justifies treating them so differently.
10. Section 19.5's Test 4 checks the detector's exact streak value immediately after an item reappears and again after two further misses. What SPECIFIC bug would this test catch that a test only checking the FINAL status after all five reconciliations might miss?

## Where We Go Next

This chapter showed that the same real vision-language core built in Chapter 18 can move from an isolated manufacturing part to a full retail reconciliation loop -- planogram, batch inspection, ERP/WMS integration, honest multi-store aggregation, and evidence-based shrinkage detection -- by changing what surrounds that core, not the core itself. Retail's own real stakes are largely economic: a missed stockout costs a sale, a false shrinkage alert costs an afternoon of a store associate's time. The next domain this book turns to raises the stakes considerably. Chapter 20 takes this same real vision-language discipline into medical imaging triage, where a VLM's own output feeds a human radiologist's own workflow rather than an inventory system, and where this book's own honest-scope discipline has to extend into genuinely new territory: the regulatory boundaries such a system must respect, and a stated, honest account of exactly where a vision-language model's own interpretation stops being reliable enough to act on without a human in the loop.

## Worked Solutions

**1.** An under-estimate is the more dangerous error because it can cause the guard to report a prompt as fitting within budget when a real tokenizer would actually produce MORE tokens than the stated limit -- meaning a real deployment would send a prompt a real model then truncates mid-instruction, silently and without warning, potentially cutting off the planogram itself partway through a shelf's own listing. An over-estimate's only cost is refusing a prompt that might have technically fit, wasting a small amount of available context-window headroom -- a far cheaper mistake than silently shipping a broken prompt to a real deployed model.

**2.** A third, more aggressive truncation (dropping slots entirely, or abbreviating product names) would mean the rendered prompt no longer describes the ACTUAL planogram -- a VLM told an incomplete, silently-edited version of the shelf's own real expected contents could then reasonably compare a photograph against slots that were never actually communicated to it, generating a plausible-looking but factually ungrounded misplacement or shrinkage report. Refusing outright when even the compressed rendering does not fit is the honest alternative: it tells the CALLER clearly that this specific planogram cannot be safely summarized at this budget, rather than quietly handing the model a fabricated, incomplete picture of the shelf and letting a wrong answer look exactly like a right one.

**3.** With per-photo isolation, a batch job processing ten photos where one is corrupted or mis-captured produces nine real, usable inspection results and one specific, named, diagnosable failure -- a store operator or an automated reconciliation job downstream still gets nine-tenths of the day's real work done, and knows exactly which photo needs a re-take. Without isolation, that same single bad photo would cause the entire batch function to raise an exception and produce NO results at all for any of the ten photos, turning one small, specific, easily-explained problem (one photo failed) into a much larger, vaguer one (the whole store's reconciliation run failed today, for a reason that requires digging through logs to even identify).

**4.** The gap exists because this book's own from-scratch model weights, exactly as they have been since Chapter 15's own self-tests, are randomly initialized and never trained -- so a real forward pass through them produces mathematically real, finite, deterministic numbers, but has no reason to produce genuinely structured, meaningful text, since nothing about random weights was ever taught what that structure should look like. Only a real, fine-tuned checkpoint would cause the real forward pass to emit text a strict parser could meaningfully be tested against as GENUINE model output. What Section 19.2 verifies for real despite this: that the real vision-language pipeline itself (encoding, fusion, decoding) runs correctly and deterministically across a whole batch of photos without crashing, without NaNs, and with correct per-photo failure isolation -- and, completely separately, that the parser downstream of whatever text a real deployed model eventually produces is strict and handles every real malformed case correctly, using representative example strings rather than this book's own meaningless synthetic output.

**5.** The version counter's real job is protecting against a lost update under concurrent writers, not describing the business state itself -- Test 3 deliberately writes the SAME quantity (50) twice, each time with the correct current version, specifically to demonstrate that the version can advance (from 1 to 2) while the actual inventory count a store's own dashboard would display never changes at all. A caller or a monitoring system that treated every version increment as a real inventory change event would report a false "something changed" alert on every single idempotent re-confirmation a routine reconciliation job performs, which would very quickly drown any GENUINE change events in noise generated purely by the concurrency-control mechanism doing exactly what it is supposed to do.

**6.** `put_stock_level` sets an ABSOLUTE, addressable value (a specific SKU's stock level at a specific store) that a caller can meaningfully re-apply with the intent "make sure this ends up at this value" -- a PUT is naturally idempotent because "set X to 50" run twice ends in the identical state as running it once. `create_shrinkage_ticket` instead records a NEW EVENT (a distinct instance of suspected shrinkage, worth its own audit trail) each time it is called, and there is no meaningful sense in which two calls with identical inputs should collapse into "the same ticket" -- a real shrinkage investigation genuinely benefits from a separate ticket per detected event, even if two events happen to share the same store, SKU, and description text.

**7.** Without `stores_missing`, a caller cannot distinguish "the total reflects every store I expected to hear from" from "the total reflects only some of the stores I expected to hear from, and the rest simply have not reported yet" -- two situations that can produce the exact same total_quantity number while meaning completely different things about how much of the chain's real inventory that number actually accounts for. A trend-detection or restocking decision made on an incomplete aggregate, without knowing it was incomplete, could badly misjudge how urgent a real shortage is, or manufacture an apparent shortage from what is really just a reporting gap.

**8.** Test 6's series has a real, hand-computable slope of -50 per period -- comfortably past the stated rate threshold of -2.0, so a rate-threshold check ALONE would flag it as declining fast enough to worry about. But the series starts from 1,000,000 units, so its real, hand-computable zero-crossing sits roughly 20,000 periods away -- vastly beyond the stated 20-period projection horizon. A rate check alone would raise an alert about a shortage that will not actually happen for tens of thousands of periods, which is not an actionable warning; requiring BOTH the rate threshold AND the horizon check ensures a StockoutRisk signal means something genuinely urgent -- fast enough decline AND close enough in time -- rather than merely "declining somewhat quickly, whenever that eventually catches up."

**9.** A misplacement is a claim about what a single photograph directly shows RIGHT NOW -- if the detected SKU at a slot differs from the expected one, that mismatch is a complete, self-contained fact requiring no further evidence, exactly the way Chapter 18.1's own dropped-packet detection needed no corroboration beyond the one frame it was reported in. A missing item is a claim about the ABSENCE of something, which a single photograph cannot distinguish from several ordinary, non-shrinkage explanations (a customer mid-reach, a restocking cart temporarily blocking the view, a brief gap between deliveries) -- only a PATTERN of the same slot being empty across multiple, separately-captured observations makes a genuine shrinkage explanation more likely than these ordinary alternatives, which is exactly why the shrinkage claim requires corroboration that the misplacement claim does not.

**10.** A test that only checks the FINAL status after all five reconciliations in Test 4 could still pass even if the detector merely PAUSED its counter across the reappearance instead of genuinely resetting it to zero -- for instance, a buggy implementation that resumed counting from 2 (rather than 0) after the reappearance would reach a streak of 4 after the two further misses, which happens to still be `EMPTY_OBSERVED` at a threshold of 3 only by coincidence of this specific test's exact numbers, and a slightly different threshold or miss count could let that same bug slip through a final-status-only check undetected. Checking the exact intermediate streak VALUE immediately after the reset (confirming it is precisely 0, not merely "still below threshold") is the one assertion that would catch a "pause instead of reset" bug directly, regardless of how many further misses follow or what threshold is configured.
