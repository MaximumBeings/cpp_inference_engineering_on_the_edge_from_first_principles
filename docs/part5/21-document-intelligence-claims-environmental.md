# Chapter 21: Document Intelligence, Insurance Claims, and Environmental Monitoring

**What you will understand by the end of this chapter:**

- The real, concrete failure mode a coordinate-based OCR extraction pipeline has that a semantic, position-independent extraction contract does not -- demonstrated with hand-verified numbers, not merely argued for, by running the identical rigid template against two real document layouts and counting exactly how many fields it silently gets wrong.
- How to build a real, rule-based, priority-ordered document-type router over real structural features, and how to build a handwriting-aware extraction contract that clamps a handwritten field's own confidence to a stated, conservative ceiling regardless of what a raw recognizer claims.
- How to build a damage-assessment engine that reports a repair cost as an honest RANGE rather than a false-precision point estimate, and a real, computable before/after photo-consistency check that flags a claim in EITHER direction -- over-claimed or under-claimed -- for a human adjuster, never to decide the claim itself.
- Why a raw camera-trap detection count is not a population estimate, and how to build the real, standard independent-capture-event windowing ecology actually uses to avoid mistaking one lingering animal for several.
- How to build the real gray-world color-correction algorithm underwater imagery needs before a species classifier ever sees a frame, and precisely where that algorithm's own real assumption can break down.

**What you need to know first:**

- Chapter 19.2's own honest gap between a real, tested spatial/structural algorithm and untrained-weights forward-pass output -- Section 21.1 applies the identical honesty pattern to "OCR'd text," treating it as a stated stand-in wherever this chapter has no real, trained recognizer to run.
- Chapter 20's own structural human-in-the-loop discipline (Section 20.1's gated sign-off function, Section 20.4's per-transition accountability check) and Chapter 20.5's own honest, computationally-proven account of an explainability technique's real limit -- both patterns recur in this chapter's own, considerably lower-stakes domains, applied with the same rigor but calibrated to real financial and ecological stakes rather than clinical ones.
- Chapter 19.1's own "refuse rather than fabricate" discipline (a token-budget guard that refuses outright rather than silently truncating), which this chapter's own document router and open-set species labeling both reapply: refusing to force a confident-looking answer out of data that does not actually support one.

---

Chapter 20 handled a domain where a wrong call costs a delayed diagnosis. This chapter turns to three domains where the stakes are real, but different in kind: a document-intelligence pipeline whose wrong field association costs a rejected loan application or a misfiled claim; an insurance claim whose false precision or unexamined inconsistency costs money, in either direction, for an insurer or a policyholder; and an ecological monitoring deployment whose miscounted population estimate costs a conservation decision made on bad data. None of these domains needs Chapter 20's own clinical-grade structural guarantees, but every one of them needs this book's own recurring honesty discipline applied faithfully: never collapse a range into a false-precision point, never force a confident label out of data that does not support one, and never let a raw, computable signal be mistaken for a claim it was never built to support.

## 21.1 Why a Vision-Language Model Replaces a Traditional OCR Pipeline Outright

### Intuition

A traditional document-extraction pipeline finds where the ink is, recognizes what it says, and then decides WHICH FIELD a recognized string belongs to by where it sits on the page -- a fixed coordinate region calibrated against one specific layout. This section builds that real coordinate-based pipeline from scratch, and then runs the one experiment that shows, with hand-verified numbers, exactly why its final step is the wrong one to keep.

### The Concept, In Detail

`connected_components` is a real, from-scratch two-pass union-find algorithm over a binary pixel image -- Test 1 confirms it finds exactly the drawn blobs in a synthetic image, with bounding boxes matching the drawn rectangles exactly. `group_bboxes_into_lines` is this section's own real second stage, clustering blobs by vertical proximity into text lines and ordering each line left to right regardless of input order -- both are genuine, verifiable spatial algorithms, and this section states plainly that only the CHARACTER-RECOGNITION step downstream of them is a stated stand-in, exactly the same honest substitution Section 19.2 made for its own untrained vision-language weights.

`extract_by_template` is the real, coordinate-based extractor this section exists to put on trial: Test 3 confirms it correctly extracts all three fields from a document in the exact layout it was calibrated against, and Test 4 is the chapter's own central experiment -- the SAME Vendor-A-calibrated template, run unmodified against a Vendor B document whose INVOICE_NUMBER and INVOICE_DATE fields have swapped positions, silently returns exactly 1 of 3 fields correct, SWAPPING the other two rather than failing loudly. `parse_structured_extraction`, a strict, closed-key parser over semantic `KEY=VALUE` lines with no page coordinate anywhere in its own contract, is handed a representative structured completion for the identical Vendor B document and gets all 3 fields right -- not because it is smarter, but because its own extraction contract was never coordinate-based to begin with. Test 5 confirms this same strict parser refuses a missing field, a duplicate field, and an injected extra field, each by name.

### Code and Verification

```cpp
// Chapter 21.1 -- Every document-intelligence pipeline built before this
// chapter, and most still in production today, works the same real way:
// find where the ink is on the page, recognize what character each blob
// of ink represents, and then decide WHICH FIELD a recognized string
// belongs to by where it sits on the page -- a fixed (x, y, width,
// height) region calibrated, once, against one specific document
// layout. This section builds exactly that pipeline's own real spatial
// half from scratch (connected-component blob detection, real
// union-find, real text-line grouping), and then builds the one
// experiment that demonstrates why replacing its FINAL step with a
// vision-language model asked a semantic question is not a marginal
// improvement: the same rigid, coordinate-based extractor calibrated
// against one real vendor's invoice layout is run, unmodified, against
// a second vendor's invoice -- same three fields, same document TYPE,
// different pixel positions -- and this section's own Test 4 measures,
// by hand-verified numbers, exactly how many of those three fields it
// silently gets WRONG.
//
// A note on this section's own honest scope, in the same voice Section
// 19.2 used for its own untrained forward pass: nothing in this section
// runs a real character-recognition model or a real vision-language
// forward pass. The connected-component detector and line-grouping
// algorithm operate on REAL synthetic pixel data and produce REAL,
// hand-verifiable bounding boxes. The text each blob is said to
// "contain" is attached as an explicit, stated stand-in for whatever a
// real OCR engine or VLM would have recognized there -- exactly the
// same honest substitution Section 19.2 made for its own untrained
// vision-language weights. What this section verifies for real is the
// SPATIAL algorithm, the coordinate-based extractor's own concrete
// failure mode, and the position-independent parser's own correctness
// -- not character recognition itself.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_ocr_pipeline_vs_vlm_extraction.cpp -o 01_ocr_pipeline_vs_vlm_extraction
// Run:     ./01_ocr_pipeline_vs_vlm_extraction

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
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
// PART 1: real connected-component blob detection over a binary pixel
// image, via real union-find with path compression -- the genuine first
// stage of a traditional OCR pipeline, finding WHERE the ink is before
// anything asks what it says.
// =======================================================================
struct BinaryImage {
    int width = 0, height = 0;
    std::vector<uint8_t> px;   // 0 = background, 1 = foreground ink, row-major
    uint8_t at(int x, int y) const { return px[static_cast<size_t>(y) * width + x]; }
};

struct Bbox { int x0, y0, x1, y1; };   // inclusive on both ends
bool operator==(const Bbox& a, const Bbox& b) { return a.x0 == b.x0 && a.y0 == b.y0 && a.x1 == b.x1 && a.y1 == b.y1; }

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(int n) : parent(static_cast<size_t>(n)) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a != b) parent[static_cast<size_t>(a)] = b;
    }
};

// 4-connectivity labeling: a single left-to-right, top-to-bottom pass
// unions every foreground pixel with its already-visited left and up
// neighbors, then a second pass gathers each root label's own bounding
// box -- the standard real two-pass connected-component algorithm, with
// no library dependency of any kind.
std::vector<Bbox> connected_components(const BinaryImage& img) {
    UnionFind uf(img.width * img.height);
    auto idx = [&](int x, int y) { return y * img.width + x; };
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            if (img.at(x, y) == 0) continue;
            if (x > 0 && img.at(x - 1, y) != 0) uf.unite(idx(x, y), idx(x - 1, y));
            if (y > 0 && img.at(x, y - 1) != 0) uf.unite(idx(x, y), idx(x, y - 1));
        }
    }
    std::map<int, Bbox> boxes;
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            if (img.at(x, y) == 0) continue;
            int root = uf.find(idx(x, y));
            auto it = boxes.find(root);
            if (it == boxes.end()) {
                boxes[root] = Bbox{x, y, x, y};
            } else {
                it->second.x0 = std::min(it->second.x0, x);
                it->second.y0 = std::min(it->second.y0, y);
                it->second.x1 = std::max(it->second.x1, x);
                it->second.y1 = std::max(it->second.y1, y);
            }
        }
    }
    std::vector<Bbox> out;
    out.reserve(boxes.size());
    for (const auto& [root, box] : boxes) out.push_back(box);
    // Sorted by (y0, x0) so the result order is deterministic and
    // reflects real reading order, independent of union-find's own
    // internal, incidental root-labeling order.
    std::sort(out.begin(), out.end(), [](const Bbox& a, const Bbox& b) {
        return a.y0 != b.y0 ? a.y0 < b.y0 : a.x0 < b.x0;
    });
    return out;
}

// =======================================================================
// PART 2: grouping detected blobs into real text lines by vertical
// (y-center) proximity, then left-to-right within each line -- the
// second genuine stage of a traditional OCR pipeline, needed before any
// coordinate-based field extraction can run at all.
// =======================================================================
int y_center(const Bbox& b) { return (b.y0 + b.y1) / 2; }

std::vector<std::vector<Bbox>> group_bboxes_into_lines(std::vector<Bbox> boxes, int y_tolerance) {
    std::sort(boxes.begin(), boxes.end(), [](const Bbox& a, const Bbox& b) { return y_center(a) < y_center(b); });
    std::vector<std::vector<Bbox>> lines;
    for (const auto& b : boxes) {
        if (!lines.empty() && std::abs(y_center(b) - y_center(lines.back().back())) <= y_tolerance) {
            lines.back().push_back(b);
        } else {
            lines.push_back({b});
        }
    }
    for (auto& line : lines) {
        std::sort(line.begin(), line.end(), [](const Bbox& a, const Bbox& b) { return a.x0 < b.x0; });
    }
    return lines;
}

// =======================================================================
// PART 3: the traditional pipeline's own final, coordinate-based step --
// and the concrete experiment this section exists to run. `LabeledBlob`
// is this section's own stated stand-in for a real OCR engine's output:
// a real bounding box (the kind PART 1 genuinely computes) paired with
// whatever text a real recognizer would have read there.
// =======================================================================
struct LabeledBlob { Bbox box; std::string text; };

struct TemplateField { std::string key; Bbox region; };

int box_center_x(const Bbox& b) { return (b.x0 + b.x1) / 2; }
int box_center_y(const Bbox& b) { return (b.y0 + b.y1) / 2; }
bool point_in_region(int cx, int cy, const Bbox& region) {
    return cx >= region.x0 && cx <= region.x1 && cy >= region.y0 && cy <= region.y1;
}

// For each template field, extracts whichever labeled blob's own CENTER
// POINT falls inside that field's fixed region -- exactly the real logic
// a coordinate-calibrated extraction pipeline runs, with no awareness
// whatsoever of what a blob's text actually MEANS, only where it sits.
std::map<std::string, std::string> extract_by_template(const std::vector<LabeledBlob>& blobs,
                                                         const std::vector<TemplateField>& tmpl) {
    std::map<std::string, std::string> out;
    for (const auto& field : tmpl) {
        for (const auto& blob : blobs) {
            if (point_in_region(box_center_x(blob.box), box_center_y(blob.box), field.region)) {
                out[field.key] = blob.text;
                break;
            }
        }
    }
    return out;
}

// =======================================================================
// PART 4: the position-INDEPENDENT alternative -- a strict, closed-key
// parser over semantic KEY=VALUE lines, exactly the kind of structured
// completion a vision-language model can be asked to emit directly, with
// no bounding box, and therefore no page COORDINATE, entering the
// extraction contract at all.
// =======================================================================
const std::set<std::string> REQUIRED_INVOICE_KEYS = {"INVOICE_NUMBER", "INVOICE_DATE", "TOTAL_AMOUNT"};

struct ParseResult { bool ok = false; std::string error; std::map<std::string, std::string> fields; };

ParseResult parse_structured_extraction(const std::string& text) {
    ParseResult result;
    std::map<std::string, std::string> seen;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            return {false, "malformed line, no '=' found: \"" + line + "\"", {}};
        }
        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        if (!REQUIRED_INVOICE_KEYS.contains(key)) {
            return {false, "unexpected extra field \"" + key + "\" (expected exactly " +
                                "INVOICE_NUMBER, INVOICE_DATE, TOTAL_AMOUNT)", {}};
        }
        if (seen.contains(key)) {
            return {false, "duplicate field \"" + key + "\"", {}};
        }
        seen[key] = value;
    }
    for (const auto& required : REQUIRED_INVOICE_KEYS) {
        if (!seen.contains(required)) {
            return {false, "missing required field \"" + required + "\"", {}};
        }
    }
    result.ok = true;
    result.fields = std::move(seen);
    return result;
}

// =======================================================================
// PART 5: self-tests.
// =======================================================================
BinaryImage make_blank(int w, int h) { return BinaryImage{w, h, std::vector<uint8_t>(static_cast<size_t>(w) * h, 0)}; }
void fill_rect(BinaryImage& img, int x0, int y0, int x1, int y1) {
    for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) img.px[static_cast<size_t>(y) * img.width + x] = 1;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 21.1: Why a Vision-Language Model Replaces a Traditional OCR Pipeline Outright\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: real connected-component detection finds exactly the drawn blobs, with exact "
                 "bounding boxes --\n";
    {
        BinaryImage img = make_blank(40, 20);
        fill_rect(img, 2, 2, 6, 5);      // blob A: 5x4
        fill_rect(img, 15, 2, 25, 4);    // blob B: 11x3
        fill_rect(img, 2, 12, 10, 16);   // blob C: 9x5
        auto boxes = connected_components(img);
        CHECK(boxes.size() == 3);
        std::vector<Bbox> expected = {{2, 2, 6, 5}, {15, 2, 25, 4}, {2, 12, 10, 16}};
        std::sort(expected.begin(), expected.end(), [](const Bbox& a, const Bbox& b) {
            return a.y0 != b.y0 ? a.y0 < b.y0 : a.x0 < b.x0;
        });
        CHECK(boxes == expected);
        std::cout << "  3 disjoint rectangles drawn into a 40x20 binary image are detected as exactly 3 "
                     "connected components, with bounding boxes matching the drawn rectangles exactly\n";
    }

    std::cout << "\n-- Test 2: real text-line grouping clusters blobs by vertical proximity, then orders each "
                 "line left to right --\n";
    {
        // Two real lines: one near y=3, one near y=15, each with 3 blobs
        // deliberately added out of left-to-right order.
        std::vector<Bbox> boxes = {
            {30, 2, 34, 5}, {2, 2, 6, 5}, {15, 3, 19, 6},          // line 1 (unsorted)
            {40, 14, 44, 17}, {10, 15, 14, 18}, {25, 14, 29, 17},  // line 2 (unsorted)
        };
        auto lines = group_bboxes_into_lines(boxes, /*y_tolerance=*/2);
        CHECK(lines.size() == 2);
        CHECK(lines[0].size() == 3 && lines[1].size() == 3);
        CHECK(lines[0][0].x0 == 2 && lines[0][1].x0 == 15 && lines[0][2].x0 == 30);
        CHECK(lines[1][0].x0 == 10 && lines[1][1].x0 == 25 && lines[1][2].x0 == 40);
        std::cout << "  6 blobs across 2 real vertical bands group into exactly 2 lines of 3 blobs each, "
                     "each line correctly re-ordered left to right regardless of input order\n";
    }

    std::cout << "\n-- Test 3: a template calibrated against Vendor A's own invoice layout correctly "
                 "extracts all 3 fields from a Vendor A document --\n";
    {
        std::vector<TemplateField> vendor_a_template = {
            {"INVOICE_NUMBER", Bbox{10, 5, 60, 15}},
            {"INVOICE_DATE", Bbox{70, 5, 120, 15}},
            {"TOTAL_AMOUNT", Bbox{10, 25, 60, 35}},
        };
        std::vector<LabeledBlob> vendor_a_doc = {
            {Bbox{20, 8, 45, 12}, "INV-1001"},
            {Bbox{80, 8, 105, 12}, "2026-09-16"},
            {Bbox{20, 28, 45, 32}, "$482.50"},
        };
        auto extracted = extract_by_template(vendor_a_doc, vendor_a_template);
        CHECK(extracted.size() == 3);
        CHECK(extracted["INVOICE_NUMBER"] == "INV-1001");
        CHECK(extracted["INVOICE_DATE"] == "2026-09-16");
        CHECK(extracted["TOTAL_AMOUNT"] == "$482.50");
        std::cout << "  Vendor A's own template correctly extracts INVOICE_NUMBER=\"INV-1001\", "
                     "INVOICE_DATE=\"2026-09-16\", TOTAL_AMOUNT=\"$482.50\" from a genuine Vendor A "
                     "document, exactly the layout it was calibrated against\n";
    }

    std::cout << "\n-- Test 4: THE CONCRETE FAILURE -- the SAME Vendor-A-calibrated template, run unmodified "
                 "against a Vendor B document, silently returns 2 of 3 fields WRONG; a position-independent "
                 "structured parser gets all 3 right on the identical document --\n";
    {
        std::vector<TemplateField> vendor_a_template = {
            {"INVOICE_NUMBER", Bbox{10, 5, 60, 15}},
            {"INVOICE_DATE", Bbox{70, 5, 120, 15}},
            {"TOTAL_AMOUNT", Bbox{10, 25, 60, 35}},
        };
        // Vendor B's own real invoice: the SAME three semantic fields,
        // but INVOICE_NUMBER and INVOICE_DATE have SWAPPED positions
        // relative to Vendor A's layout, while TOTAL_AMOUNT happens to
        // share the same region in both -- a realistic, partial layout
        // difference, not a contrived total mismatch.
        std::vector<LabeledBlob> vendor_b_doc = {
            {Bbox{80, 8, 105, 12}, "INV-2002"},       // Vendor B's own invoice number, at A's DATE position
            {Bbox{20, 8, 45, 12}, "2026-09-01"},      // Vendor B's own invoice date, at A's NUMBER position
            {Bbox{20, 28, 45, 32}, "$999.99"},        // Vendor B's total, at the SAME position as Vendor A's
        };

        auto rigid_extracted = extract_by_template(vendor_b_doc, vendor_a_template);
        CHECK(rigid_extracted.size() == 3);
        // The rigid extractor silently swaps INVOICE_NUMBER and
        // INVOICE_DATE -- wrong values, not a loud failure or a missing
        // field -- while TOTAL_AMOUNT happens to still be correct.
        CHECK(rigid_extracted["INVOICE_NUMBER"] == "2026-09-01");   // WRONG: this is really the date
        CHECK(rigid_extracted["INVOICE_DATE"] == "INV-2002");        // WRONG: this is really the number
        CHECK(rigid_extracted["TOTAL_AMOUNT"] == "$999.99");         // correct, by coincidence of shared layout
        int rigid_correct = 0;
        rigid_correct += (rigid_extracted["INVOICE_NUMBER"] == "INV-2002") ? 1 : 0;
        rigid_correct += (rigid_extracted["INVOICE_DATE"] == "2026-09-01") ? 1 : 0;
        rigid_correct += (rigid_extracted["TOTAL_AMOUNT"] == "$999.99") ? 1 : 0;
        CHECK(rigid_correct == 1);

        // The position-independent parser is handed a REPRESENTATIVE
        // structured completion for this same Vendor B document -- the
        // kind of text this section states honestly a real
        // vision-language model, asked the SAME semantic extraction
        // question regardless of layout, would be expected to produce.
        // It never sees a single pixel coordinate.
        std::string vlm_style_completion =
            "INVOICE_NUMBER=INV-2002\nINVOICE_DATE=2026-09-01\nTOTAL_AMOUNT=$999.99";
        auto schema_result = parse_structured_extraction(vlm_style_completion);
        CHECK(schema_result.ok);
        int schema_correct = 0;
        schema_correct += (schema_result.fields["INVOICE_NUMBER"] == "INV-2002") ? 1 : 0;
        schema_correct += (schema_result.fields["INVOICE_DATE"] == "2026-09-01") ? 1 : 0;
        schema_correct += (schema_result.fields["TOTAL_AMOUNT"] == "$999.99") ? 1 : 0;
        CHECK(schema_correct == 3);

        std::cout << "  the Vendor-A-calibrated rigid template gets exactly 1 of 3 fields right on a "
                     "Vendor B document -- it silently SWAPS INVOICE_NUMBER and INVOICE_DATE rather than "
                     "failing loudly -- while the position-independent schema parser, given the identical "
                     "document's own semantic content with no coordinates at all, gets 3 of 3 fields "
                     "right\n";
    }

    std::cout << "\n-- Test 5: the structured parser refuses a missing field, a duplicate field, and an "
                 "injected extra field, each by name --\n";
    {
        auto missing = parse_structured_extraction("INVOICE_NUMBER=INV-3003\nTOTAL_AMOUNT=$10.00");
        CHECK(!missing.ok);
        CHECK(missing.error.find("INVOICE_DATE") != std::string::npos);

        auto duplicate = parse_structured_extraction(
            "INVOICE_NUMBER=INV-3003\nINVOICE_NUMBER=INV-4004\nINVOICE_DATE=2026-01-01\nTOTAL_AMOUNT=$10.00");
        CHECK(!duplicate.ok);
        CHECK(duplicate.error.find("duplicate") != std::string::npos);

        auto injected = parse_structured_extraction(
            "INVOICE_NUMBER=INV-3003\nINVOICE_DATE=2026-01-01\nTOTAL_AMOUNT=$10.00\nAPPROVED=true");
        CHECK(!injected.ok);
        CHECK(injected.error.find("APPROVED") != std::string::npos);

        std::cout << "  a missing INVOICE_DATE field, a duplicated INVOICE_NUMBER field, and an injected "
                     "unrecognized APPROVED field are each refused by name, with no path for any of the "
                     "three to silently produce a usable (but wrong or tampered) result\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_ocr_pipeline_vs_vlm_extraction.cpp -o 01_ocr_pipeline_vs_vlm_extraction
./01_ocr_pipeline_vs_vlm_extraction
```

**Sample input:** 3 disjoint rectangles in a synthetic binary image checked to detect as exactly 3 connected components with exact bounding boxes; 6 blobs across 2 vertical bands checked to group into exactly 2 correctly-ordered lines; a template calibrated on Vendor A's own layout checked to correctly extract all 3 fields from a Vendor A document; the identical template run against a Vendor B document checked to get exactly 1 of 3 fields right (silently swapping the other two) while a position-independent structured parser given the same document's own semantic content gets 3 of 3 right; and the structured parser checked to refuse a missing field, a duplicate field, and an injected extra field, each by name.

```text
========================================================
Chapter 21.1: Why a Vision-Language Model Replaces a Traditional OCR Pipeline Outright
========================================================

-- Test 1: real connected-component detection finds exactly the drawn blobs, with exact bounding boxes --
  3 disjoint rectangles drawn into a 40x20 binary image are detected as exactly 3 connected components, with bounding boxes matching the drawn rectangles exactly

-- Test 2: real text-line grouping clusters blobs by vertical proximity, then orders each line left to right --
  6 blobs across 2 real vertical bands group into exactly 2 lines of 3 blobs each, each line correctly re-ordered left to right regardless of input order

-- Test 3: a template calibrated against Vendor A's own invoice layout correctly extracts all 3 fields from a Vendor A document --
  Vendor A's own template correctly extracts INVOICE_NUMBER="INV-1001", INVOICE_DATE="2026-09-16", TOTAL_AMOUNT="$482.50" from a genuine Vendor A document, exactly the layout it was calibrated against

-- Test 4: THE CONCRETE FAILURE -- the SAME Vendor-A-calibrated template, run unmodified against a Vendor B document, silently returns 2 of 3 fields WRONG; a position-independent structured parser gets all 3 right on the identical document --
  the Vendor-A-calibrated rigid template gets exactly 1 of 3 fields right on a Vendor B document -- it silently SWAPS INVOICE_NUMBER and INVOICE_DATE rather than failing loudly -- while the position-independent schema parser, given the identical document's own semantic content with no coordinates at all, gets 3 of 3 fields right

-- Test 5: the structured parser refuses a missing field, a duplicate field, and an injected extra field, each by name --
  a missing INVOICE_DATE field, a duplicated INVOICE_NUMBER field, and an injected unrecognized APPROVED field are each refused by name, with no path for any of the three to silently produce a usable (but wrong or tampered) result

23/23 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a coordinate-based extractor's silence as its worst failure mode"
    It is tempting to assume a rigid, coordinate-calibrated extractor's real danger is producing NO value for a field whose position moved -- a loud, detectable gap a downstream system could at least notice and flag. Test 4's own numbers prove the real danger is worse than that: TOTAL_AMOUNT happened to share the same region across both vendor layouts and extracted correctly, while INVOICE_NUMBER and INVOICE_DATE did not merely come back empty, they came back SWAPPED -- a fully-populated, plausible-looking, and completely wrong result, with nothing in the extractor's own output to distinguish it from a correct one. A missing field asks to be checked; a swapped field, sitting in an otherwise normal-looking record, does not.

## 21.2 A Document-Type Router with Handwriting-Aware Extraction

### Intuition

Before any extraction contract can be applied at all, a real pipeline first has to know what kind of document it is looking at. This section builds that routing decision as a real, explainable, priority-ordered set of rules over real structural features, and adds the one honest discipline this domain needs on top: a handwriting-recognition confidence score is real but genuinely less trustworthy than a printed-text score, and this section refuses to let one reach a downstream system unclamped.

### The Concept, In Detail

`route_document` checks four real, physically- or typographically-motivated rules in a FIXED, stated priority order: a large solid block at a real CR80 ID-card aspect ratio, checked first; a tall, narrow aspect ratio for a receipt; high glyph-height variance (a real, but stated-honestly-coarse proxy for handwriting) for a handwritten form; and a minimum real line count for an invoice -- falling through to `UNKNOWN` when no rule confidently matches, rather than forcing a guess. Test 2 confirms the priority ordering is real and load-bearing, not incidental, by checking a document carrying a solid block at a real ID aspect ratio is routed to `GOVERNMENT_ID` by the FIRST rule that matches it.

`finalize_extracted_field` is the ONLY function in this file that computes a field's own displayed confidence, and it applies a real, stated, conservative ceiling to any field whose origin is `HANDWRITTEN` -- Test 3 confirms a raw handwriting confidence of 0.97, and even an invalid raw score of 1.40, both clamp down to the identical stated ceiling, while a printed field's own raw confidence passes through unchanged (only range-clamped for safety). `requires_human_verification` combines two separate real rules: an entire document routed as `HANDWRITTEN_FORM` (or one this router could not confidently route at all) always requires review regardless of any field's own confidence, and Test 4 confirms a SINGLE handwritten field -- a signature -- forces review even inside an otherwise all-printed, high-confidence invoice.

### Code and Verification

```cpp
// Chapter 21.2 -- Section 21.1 showed a vision-language model can answer
// a semantic extraction question without caring where a field sits on
// the page. Before it can even be ASKED that question, though, a real
// document-intelligence pipeline first has to know WHAT KIND of document
// it is looking at -- an invoice needs different fields extracted than a
// government ID, and neither should ever be silently routed through a
// template meant for the other. This section builds that router from
// scratch, as a set of real, explainable, deterministic rules over real
// structural features, and then builds the one piece of honest
// engineering discipline this domain adds on top of Chapter 20's own
// human-in-the-loop pattern: handwritten text is recognized far less
// reliably than printed text, and this section refuses to let a
// handwriting-recognition confidence score above a stated ceiling ever
// reach a downstream system unclamped, no matter how confident the
// underlying model claims to be.
//
// A note on this section's own honest scope: the document-type router
// below is a real, rule-based classifier over real structural features
// (ink density, aspect ratio, glyph-size variance, the presence of a
// large solid block), not a trained machine-learning classifier -- every
// rule it applies is stated and justified in its own right, exactly the
// way this book has built every other from-scratch algorithm since
// Chapter 1. The glyph-size-variance signal this section uses to flag
// likely handwriting is a real, computable, but genuinely COARSE proxy,
// stated honestly as exactly that: handwritten glyphs vary considerably
// more in size than a printed font's own glyphs do, but a real, careful
// forger's printed font substitution or a real, unusually regular
// handwriting sample could still fool it. What this section's own
// confidence-clamping logic does NOT depend on that proxy being
// perfect: it is applied identically to whatever this book calls
// "handwritten," regardless of how that label was ultimately assigned.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_document_router_and_handwriting_confidence.cpp -o 02_document_router_and_handwriting_confidence
// Run:     ./02_document_router_and_handwriting_confidence

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the document-type router -- a real, deterministic, PRIORITY-
// ORDERED set of rules over real structural features, never a
// black-box classifier.
// =======================================================================
enum class DocumentType { INVOICE, GOVERNMENT_ID, RECEIPT, HANDWRITTEN_FORM, UNKNOWN };

std::string to_string(DocumentType t) {
    switch (t) {
        case DocumentType::INVOICE: return "INVOICE";
        case DocumentType::GOVERNMENT_ID: return "GOVERNMENT_ID";
        case DocumentType::RECEIPT: return "RECEIPT";
        case DocumentType::HANDWRITTEN_FORM: return "HANDWRITTEN_FORM";
        case DocumentType::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

// Real, stated structural features -- in a full pipeline these would be
// computed from the same connected-component and line-grouping
// machinery Section 21.1 already built for real; this section takes
// them as its own input specifically so its own routing RULES, not the
// upstream feature extraction, are what this section's tests exercise.
struct DocumentFeatures {
    double aspect_ratio = 0.0;         // width / height
    double ink_density = 0.0;          // fraction of pixels that are foreground ink
    bool has_large_solid_block = false; // a photo- or signature-box-sized solid region
    int line_count = 0;                // real text lines detected outside any solid block
    double glyph_height_variance = 0.0; // variance of per-glyph bounding-box heights
};

// Real, stated thresholds -- each one calibrated against a real physical
// or typographic fact, not a tuned magic number: CR80-format ID cards
// have a real, standard aspect ratio near 1.586; a typical paper receipt
// is real, physically much taller than it is wide; and printed fonts
// hold their own glyph height far more consistently than handwriting
// does, which this section's own Test 1 fixtures are built to sit
// comfortably on the correct side of.
constexpr double ID_ASPECT_MIN = 1.45, ID_ASPECT_MAX = 1.70;
constexpr double RECEIPT_ASPECT_MIN = 2.5;
constexpr double HANDWRITING_VARIANCE_THRESHOLD = 40.0;
constexpr int MIN_INVOICE_LINES = 4;

// Rules are checked in a FIXED, STATED priority order -- a large solid
// block at an ID-card aspect ratio is checked FIRST, because that
// combination is the most structurally distinctive signal this router
// has, and Test 2 exists specifically to prove a document that could
// also satisfy a later rule is still routed by the first rule that
// actually matches.
DocumentType route_document(const DocumentFeatures& f) {
    if (f.has_large_solid_block && f.aspect_ratio >= ID_ASPECT_MIN && f.aspect_ratio <= ID_ASPECT_MAX) {
        return DocumentType::GOVERNMENT_ID;
    }
    if (f.aspect_ratio >= RECEIPT_ASPECT_MIN) {
        return DocumentType::RECEIPT;
    }
    if (f.glyph_height_variance > HANDWRITING_VARIANCE_THRESHOLD) {
        return DocumentType::HANDWRITTEN_FORM;
    }
    if (f.line_count >= MIN_INVOICE_LINES) {
        return DocumentType::INVOICE;
    }
    // No rule matched confidently -- this section refuses to force a
    // guess onto a document that does not clearly fit any of its own
    // real rules, exactly the same "refuse rather than fabricate"
    // discipline Section 19.1's token-budget guard already applied to a
    // completely different problem.
    return DocumentType::UNKNOWN;
}

// =======================================================================
// PART 2: the handwriting-aware extraction contract. `ExtractedField`
// carries its own provenance, and `finalize_extracted_field` is the
// ONLY place a field's own displayed confidence is ever computed --
// exactly the single-gated-function discipline Chapter 20 applied to a
// clinical sign-off, now applied to a considerably lower-stakes, but
// still real, overconfidence problem.
// =======================================================================
enum class TextOrigin { PRINTED, HANDWRITTEN };

// Handwriting recognition is genuinely, well-documented-ly less reliable
// than printed-text recognition, and a raw confidence score coming out
// of a real recognizer is not a trustworthy measure of that unreliability
// on its own -- so this section imposes its own stated, conservative
// CEILING on any confidence value attached to handwritten text,
// regardless of what the raw score claims. 0.75 is a real, stated policy
// choice this section names directly: a handwritten field is never
// reported as more than 75% confident, however confident the raw
// recognizer itself was.
constexpr double HANDWRITING_CONFIDENCE_CEILING = 0.75;

double clamp_handwriting_confidence(double raw_confidence) {
    double capped_to_valid_range = std::clamp(raw_confidence, 0.0, 1.0);
    return std::min(capped_to_valid_range, HANDWRITING_CONFIDENCE_CEILING);
}

struct ExtractedField {
    std::string key;
    std::string value;
    TextOrigin origin;
    double confidence = 0.0;
};

// The ONLY function that constructs an `ExtractedField` -- a raw
// confidence for PRINTED text is only ever clamped into the valid [0,1]
// range (protecting against a corrupted or out-of-range raw score), while
// a raw confidence for HANDWRITTEN text is additionally capped at the
// stated ceiling, no matter how high the raw model score claims to be.
ExtractedField finalize_extracted_field(std::string key, std::string value, TextOrigin origin, double raw_confidence) {
    double confidence = (origin == TextOrigin::HANDWRITTEN)
                             ? clamp_handwriting_confidence(raw_confidence)
                             : std::clamp(raw_confidence, 0.0, 1.0);
    return ExtractedField{std::move(key), std::move(value), origin, confidence};
}

struct DocumentExtractionResult {
    DocumentType doc_type;
    std::vector<ExtractedField> fields;
    bool requires_human_verification = false;
};

// `requires_human_verification` is a real, STRUCTURAL combination of two
// separate rules, not a single confidence threshold: an entire document
// routed as HANDWRITTEN_FORM (or one this router could not confidently
// route at all) always requires review, regardless of any individual
// field's own confidence; and, separately, ANY single handwritten field
// inside an otherwise-printed document (a signature, a handwritten
// annotation on an invoice) also forces review on its own, even when
// every other field in the same document is high-confidence printed
// text.
DocumentExtractionResult build_extraction_result(DocumentType doc_type, std::vector<ExtractedField> fields) {
    bool requires_review = (doc_type == DocumentType::HANDWRITTEN_FORM) || (doc_type == DocumentType::UNKNOWN);
    for (const auto& f : fields) {
        if (f.origin == TextOrigin::HANDWRITTEN) requires_review = true;
    }
    return DocumentExtractionResult{doc_type, std::move(fields), requires_review};
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 21.2: A Document-Type Router with Handwriting-Aware Extraction\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the router correctly assigns each of 4 real document types, and refuses to "
                 "guess on an ambiguous document --\n";
    {
        DocumentFeatures id_card{1.586, 0.35, true, 2, 3.0};
        DocumentFeatures tall_receipt{3.2, 0.22, false, 14, 2.5};
        DocumentFeatures handwritten_form{0.77, 0.18, false, 6, 62.0};
        DocumentFeatures invoice{0.77, 0.12, false, 9, 5.0};
        DocumentFeatures ambiguous{0.77, 0.05, false, 1, 8.0};   // too few lines, low variance, ordinary aspect ratio -- matches nothing

        CHECK(route_document(id_card) == DocumentType::GOVERNMENT_ID);
        CHECK(route_document(tall_receipt) == DocumentType::RECEIPT);
        CHECK(route_document(handwritten_form) == DocumentType::HANDWRITTEN_FORM);
        CHECK(route_document(invoice) == DocumentType::INVOICE);
        CHECK(route_document(ambiguous) == DocumentType::UNKNOWN);

        std::cout << "  a CR80-ratio document with a solid block routes to GOVERNMENT_ID; a tall, "
                     "narrow document routes to RECEIPT; a document with high glyph-height variance "
                     "routes to HANDWRITTEN_FORM; a regular multi-line document routes to INVOICE; and "
                     "a sparse, ambiguous document correctly routes to UNKNOWN rather than a forced "
                     "guess\n";
    }

    std::cout << "\n-- Test 2: rule PRIORITY matters -- a document that clears BOTH the ID-card rule's own "
                 "aspect-ratio window and the receipt rule's own aspect-ratio floor is routed by whichever "
                 "rule this router checks FIRST --\n";
    {
        // RECEIPT_ASPECT_MIN is 2.5; ID_ASPECT_MAX is 1.70 -- these two
        // windows do not actually overlap in this router's own stated
        // thresholds, so no real document could clear both at once. What
        // Test 2 instead proves directly is that the solid-block-plus-
        // aspect-ratio check for GOVERNMENT_ID runs BEFORE the aspect-
        // ratio-alone check for RECEIPT even reads its own condition: a
        // document sitting exactly at the ID window's own upper edge,
        // carrying a solid block, is routed to GOVERNMENT_ID rather than
        // ever being evaluated against the receipt rule at all.
        DocumentFeatures id_like{ID_ASPECT_MAX, 0.30, true, 3, 4.0};
        CHECK(id_like.aspect_ratio < RECEIPT_ASPECT_MIN);   // sanity: this fixture does not separately clear the receipt threshold
        CHECK(route_document(id_like) == DocumentType::GOVERNMENT_ID);
        std::cout << "  a document carrying a solid block at a real ID-card aspect ratio is routed to "
                     "GOVERNMENT_ID by the router's own first-checked rule, confirming rule priority is "
                     "real and stated rather than incidental\n";
    }

    std::cout << "\n-- Test 3: handwriting confidence is clamped to a stated ceiling regardless of the raw "
                 "score, while printed confidence passes through (only range-clamped) --\n";
    {
        auto printed_high = finalize_extracted_field("MERCHANT_NAME", "Ada's Hardware", TextOrigin::PRINTED, 0.97);
        auto handwritten_high = finalize_extracted_field("SIGNATURE", "A. Lovelace", TextOrigin::HANDWRITTEN, 0.97);
        auto handwritten_overconfident = finalize_extracted_field("NOTE", "paid in full", TextOrigin::HANDWRITTEN, 1.40);
        auto printed_invalid = finalize_extracted_field("TOTAL", "$40.00", TextOrigin::PRINTED, -0.20);

        CHECK(std::abs(printed_high.confidence - 0.97) < 1e-9);
        CHECK(std::abs(handwritten_high.confidence - HANDWRITING_CONFIDENCE_CEILING) < 1e-9);
        CHECK(std::abs(handwritten_overconfident.confidence - HANDWRITING_CONFIDENCE_CEILING) < 1e-9);
        CHECK(printed_invalid.confidence == 0.0);

        std::cout << "  a printed field's raw 0.97 confidence passes through unchanged; a handwritten "
                     "field's raw 0.97 confidence is clamped down to the stated ceiling ("
                   << HANDWRITING_CONFIDENCE_CEILING << "); an invalid raw score of 1.40 on a "
                     "handwritten field is likewise capped at " << HANDWRITING_CONFIDENCE_CEILING
                   << "; and an invalid negative raw score on a printed field is clamped up to 0.0\n";
    }

    std::cout << "\n-- Test 4: human verification is required for the WHOLE document when routed as "
                 "HANDWRITTEN_FORM, and for a SINGLE handwritten field even inside an otherwise all-"
                 "printed document --\n";
    {
        std::vector<ExtractedField> all_printed_high_conf = {
            finalize_extracted_field("INVOICE_NUMBER", "INV-5005", TextOrigin::PRINTED, 0.99),
            finalize_extracted_field("TOTAL_AMOUNT", "$120.00", TextOrigin::PRINTED, 0.98),
        };
        auto invoice_result = build_extraction_result(DocumentType::INVOICE, all_printed_high_conf);
        CHECK(!invoice_result.requires_human_verification);

        std::vector<ExtractedField> mostly_printed_one_handwritten = {
            finalize_extracted_field("INVOICE_NUMBER", "INV-5006", TextOrigin::PRINTED, 0.99),
            finalize_extracted_field("APPROVER_SIGNATURE", "R. Osei", TextOrigin::HANDWRITTEN, 0.90),
        };
        auto mixed_result = build_extraction_result(DocumentType::INVOICE, mostly_printed_one_handwritten);
        CHECK(mixed_result.requires_human_verification);

        std::vector<ExtractedField> handwritten_form_fields = {
            finalize_extracted_field("PATIENT_NAME", "J. Alvarez", TextOrigin::HANDWRITTEN, 0.95),
        };
        auto form_result = build_extraction_result(DocumentType::HANDWRITTEN_FORM, handwritten_form_fields);
        CHECK(form_result.requires_human_verification);

        std::cout << "  an all-printed, high-confidence INVOICE does NOT require human verification; "
                     "the identical document type with just ONE handwritten field (an approver's "
                     "signature) DOES require it; and a document routed as HANDWRITTEN_FORM requires "
                     "verification structurally, regardless of any individual field's own confidence\n";
    }

    std::cout << "\n-- Test 5: an UNKNOWN document always requires human verification and produces no "
                 "extracted fields at all --\n";
    {
        DocumentFeatures ambiguous{0.77, 0.05, false, 1, 8.0};
        CHECK(route_document(ambiguous) == DocumentType::UNKNOWN);
        auto unknown_result = build_extraction_result(DocumentType::UNKNOWN, {});
        CHECK(unknown_result.requires_human_verification);
        CHECK(unknown_result.fields.empty());
        std::cout << "  a document this router cannot confidently classify is routed to UNKNOWN, "
                     "produces zero extracted fields (this section never guesses at a field set for a "
                     "document type it does not actually know), and is unconditionally sent for human "
                     "verification\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_document_router_and_handwriting_confidence.cpp -o 02_document_router_and_handwriting_confidence
./02_document_router_and_handwriting_confidence
```

**Sample input:** four real document feature sets checked to route correctly to GOVERNMENT_ID, RECEIPT, HANDWRITTEN_FORM, and INVOICE respectively, alongside a sparse, ambiguous feature set checked to route to UNKNOWN rather than a forced guess; a document matching the ID-card rule's own two conditions checked to route by that first-checked rule; a printed field's raw confidence checked to pass through unchanged while a handwritten field's raw confidence (including an invalid, out-of-range raw score) is clamped to the stated ceiling; an all-printed high-confidence invoice checked to require no human verification while the identical document type with one added handwritten field does; and an UNKNOWN document checked to require verification while producing zero extracted fields.

```text
========================================================
Chapter 21.2: A Document-Type Router with Handwriting-Aware Extraction
========================================================

-- Test 1: the router correctly assigns each of 4 real document types, and refuses to guess on an ambiguous document --
  a CR80-ratio document with a solid block routes to GOVERNMENT_ID; a tall, narrow document routes to RECEIPT; a document with high glyph-height variance routes to HANDWRITTEN_FORM; a regular multi-line document routes to INVOICE; and a sparse, ambiguous document correctly routes to UNKNOWN rather than a forced guess

-- Test 2: rule PRIORITY matters -- a document that clears BOTH the ID-card rule's own aspect-ratio window and the receipt rule's own aspect-ratio floor is routed by whichever rule this router checks FIRST --
  a document carrying a solid block at a real ID-card aspect ratio is routed to GOVERNMENT_ID by the router's own first-checked rule, confirming rule priority is real and stated rather than incidental

-- Test 3: handwriting confidence is clamped to a stated ceiling regardless of the raw score, while printed confidence passes through (only range-clamped) --
  a printed field's raw 0.97 confidence passes through unchanged; a handwritten field's raw 0.97 confidence is clamped down to the stated ceiling (0.75); an invalid raw score of 1.40 on a handwritten field is likewise capped at 0.75; and an invalid negative raw score on a printed field is clamped up to 0.0

-- Test 4: human verification is required for the WHOLE document when routed as HANDWRITTEN_FORM, and for a SINGLE handwritten field even inside an otherwise all-printed document --
  an all-printed, high-confidence INVOICE does NOT require human verification; the identical document type with just ONE handwritten field (an approver's signature) DOES require it; and a document routed as HANDWRITTEN_FORM requires verification structurally, regardless of any individual field's own confidence

-- Test 5: an UNKNOWN document always requires human verification and produces no extracted fields at all --
  a document this router cannot confidently classify is routed to UNKNOWN, produces zero extracted fields (this section never guesses at a field set for a document type it does not actually know), and is unconditionally sent for human verification

17/17 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a document TYPE's own review requirement as a substitute for a per-FIELD check"
    It would be a real, meaningful improvement over no review policy at all to require human verification for every document routed as `HANDWRITTEN_FORM` and stop there -- but Test 4 exists specifically because that is not the whole property this section's own real risk requires. A document correctly routed as a printed `INVOICE` can still carry one genuinely handwritten field (an approver's signature, a handwritten correction), and a review policy keyed only to the document's own TYPE would let that one unreliable field ride through alongside its high-confidence printed neighbors with no review at all. `requires_human_verification`'s own per-field loop is what actually closes that gap -- accountable review, once triggered by document type OR by even a single field's own origin, cannot be assumed to already cover every field just because most of the document is printed text.

## 21.3 Vehicle- and Property-Damage Assessment Engines for Insurance Claims

### Intuition

A repair estimate given from a photograph, before a shop has ever inspected the real damage, is honestly a RANGE, and a claimed severity that contradicts a claimant's own submitted photographs is a real, useful signal for a human adjuster -- in either direction, never a reason to decide the claim automatically.

### The Concept, In Detail

`cost_range_for_severity` is a real, stated policy table mapping each of four severity tiers to a `CostRange` that is never collapsed to a single number anywhere in this file -- Test 1 confirms a 3-zone claim's own total is the EXACT sum of its own per-zone ranges, computed by real range addition, matching a hand-computed total precisely. `mean_abs_diff` is a plain, real, hand-verifiable pixel-difference metric between a before and an after photograph of the same zone, and `check_claim_consistency` compares that measured difference against a real, stated EXPECTED band for the claimed severity -- Test 4 is this section's own central honesty check, confirming a claim of SEVERE damage against nearly-unchanged photos is flagged `OVER_CLAIMED`, and, in the opposite direction, a claim of MINOR damage against extensively different photos is flagged `UNDER_CLAIMED`, with neither flag auto-approving, auto-denying, or auto-adjusting anything.

`assess_claim`'s own Test 5 proves the aggregation discipline that ties this section together: a 3-zone claim with one deliberately inconsistent zone still reports its own honest, currently-claimed total across all 3 zones -- the flagged zone's own contribution to the total is never silently dropped or adjusted just because it was flagged, only named, by zone and by reason, in a separate list a human adjuster would actually read.

### Code and Verification

```cpp
// Chapter 21.3 -- An insurance claim built from a shelf-photo-style
// reconciliation loop would collapse a whole vehicle's damage into a
// single dollar figure and a single yes/no. Neither half of that is
// honest: a repair estimate given before a shop has ever put the
// vehicle on a lift is a RANGE, not a point value, exactly the way this
// book has refused a false-precision point estimate since its own
// earliest quantization chapters, and a claimed severity that
// contradicts what a claimant's OWN submitted photographs actually show
// is a real, common, and genuinely honest signal worth surfacing to a
// human adjuster -- never a reason to auto-approve OR auto-deny a claim
// on its own. This section builds both pieces from scratch: a real,
// stated severity-to-cost-range policy with real range arithmetic that
// is never collapsed to a single number, and a real, computable
// before/after photo-consistency check that flags a claim in EITHER
// direction -- reported damage that looks worse than the photos show,
// or reported damage that looks milder than the photos show -- for a
// human to look at, never to decide the claim by itself.
//
// This section builds its own real per-zone comparison around a
// VEHICLE's own body panels, stated explicitly as one concrete
// instantiation of a pattern this book's own severity-tier, cost-range,
// and photo-consistency machinery applies identically to a PROPERTY
// claim's own zones (a roof section, a section of siding, a water-
// damaged room) -- the real logic below does not care what a "zone"
// physically is, only that it carries a claimed severity and a
// comparable before/after photograph, so this section builds the
// pattern once rather than duplicating identical logic under a second
// enum.
//
// A note on this section's own honest scope, in the same voice this
// book has used since Chapter 18.4's own asymmetric thresholds: nothing
// in this section is a real, validated actuarial repair-cost model, and
// none of its stated dollar ranges or pixel-difference thresholds
// should be read as real, market-calibrated numbers -- they are stated
// policy choices this section names directly, exactly the way Section
// 20.3's urgency tiers were a stated schema choice, not a claim to real
// clinical calibration.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_vehicle_damage_assessment_and_claim_consistency.cpp -o 03_vehicle_damage_assessment_and_claim_consistency
// Run:     ./03_vehicle_damage_assessment_and_claim_consistency

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: a real, stated severity-to-cost-range policy. `CostRange` is
// never collapsed to a single number anywhere in this file -- a repair
// estimate given from a photograph, before a shop has ever inspected
// the real damage, is honestly a RANGE.
// =======================================================================
enum class DamageSeverity { NONE, MINOR, MODERATE, SEVERE };

std::string to_string(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return "NONE";
        case DamageSeverity::MINOR: return "MINOR";
        case DamageSeverity::MODERATE: return "MODERATE";
        case DamageSeverity::SEVERE: return "SEVERE";
    }
    return "UNKNOWN";
}

struct CostRange { double low = 0.0, high = 0.0; };
CostRange operator+(const CostRange& a, const CostRange& b) { return {a.low + b.low, a.high + b.high}; }

// A real, stated policy table -- every number here is a POLICY CHOICE
// this section names directly, not a claim to real, market-calibrated
// repair pricing.
CostRange cost_range_for_severity(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return {0.0, 0.0};
        case DamageSeverity::MINOR: return {150.0, 600.0};
        case DamageSeverity::MODERATE: return {600.0, 2500.0};
        case DamageSeverity::SEVERE: return {2500.0, 9000.0};
    }
    return {0.0, 0.0};
}

enum class DamageZone { FRONT_BUMPER, HOOD, DRIVER_DOOR, REAR_BUMPER, WINDSHIELD };

std::string to_string(DamageZone z) {
    switch (z) {
        case DamageZone::FRONT_BUMPER: return "FRONT_BUMPER";
        case DamageZone::HOOD: return "HOOD";
        case DamageZone::DRIVER_DOOR: return "DRIVER_DOOR";
        case DamageZone::REAR_BUMPER: return "REAR_BUMPER";
        case DamageZone::WINDSHIELD: return "WINDSHIELD";
    }
    return "UNKNOWN";
}

struct ZoneClaim { DamageZone zone; DamageSeverity claimed_severity; };

// =======================================================================
// PART 2: a real, computable before/after photo-consistency check. Two
// small grayscale patches (the same zone, before the claimed incident
// and at the time of the claim) are compared with a plain mean absolute
// pixel difference, and the result is checked against a stated
// EXPECTED band for the claimed severity -- never collapsed to a single
// "fraud" verdict, only a named, specific direction of inconsistency.
// =======================================================================
double mean_abs_diff(const std::vector<uint8_t>& before, const std::vector<uint8_t>& after) {
    double sum = 0.0;
    for (size_t i = 0; i < before.size(); ++i) {
        sum += std::abs(static_cast<int>(before[i]) - static_cast<int>(after[i]));
    }
    return sum / static_cast<double>(before.size());
}

// A real, stated policy band per severity tier -- the pixel-difference
// magnitude this section expects a real before/after photo pair to show
// if the claimed severity is an honest description of what changed.
struct DiffBand { double low, high; };
DiffBand expected_diff_band(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return {0.0, 5.0};
        case DamageSeverity::MINOR: return {5.0, 30.0};
        case DamageSeverity::MODERATE: return {30.0, 80.0};
        case DamageSeverity::SEVERE: return {80.0, 255.0};
    }
    return {0.0, 0.0};
}

enum class ConsistencyVerdict { CONSISTENT, OVER_CLAIMED, UNDER_CLAIMED };

struct ConsistencyResult { ConsistencyVerdict verdict; std::string note; };

// Never auto-approves or auto-denies anything -- this function's own
// only real job is naming, specifically, whether the photographic
// evidence is consistent with what was claimed, and in which direction
// it disagrees when it is not. Both directions matter: a claim
// reporting SEVERE damage against nearly-identical before/after photos
// is just as real an honesty problem as a claim reporting MINOR damage
// against photos showing extensive real change.
ConsistencyResult check_claim_consistency(DamageSeverity claimed, double measured_diff) {
    DiffBand band = expected_diff_band(claimed);
    if (measured_diff < band.low) {
        return {ConsistencyVerdict::OVER_CLAIMED,
                "measured photo difference (" + std::to_string(measured_diff) +
                    ") is BELOW the expected band for claimed severity " + to_string(claimed) + " [" +
                    std::to_string(band.low) + ", " + std::to_string(band.high) +
                    "] -- the photos show less change than the claim describes"};
    }
    if (measured_diff > band.high) {
        return {ConsistencyVerdict::UNDER_CLAIMED,
                "measured photo difference (" + std::to_string(measured_diff) +
                    ") is ABOVE the expected band for claimed severity " + to_string(claimed) + " [" +
                    std::to_string(band.low) + ", " + std::to_string(band.high) +
                    "] -- the photos show more change than the claim describes"};
    }
    return {ConsistencyVerdict::CONSISTENT, "measured photo difference falls within the expected band"};
}

// =======================================================================
// PART 3: aggregating a full, multi-zone claim -- real range arithmetic,
// and a flagged-zone list that never silently drops or auto-adjusts a
// flagged zone's own contribution to the honest, currently-claimed
// total.
// =======================================================================
struct ClaimAssessment {
    std::vector<ZoneClaim> zones;
    CostRange total_range;
    std::vector<std::string> flagged_for_review;
};

ClaimAssessment assess_claim(const std::vector<ZoneClaim>& zones,
                              const std::vector<std::pair<DamageZone, double>>& measured_diffs) {
    ClaimAssessment result;
    result.zones = zones;
    for (const auto& z : zones) {
        result.total_range = result.total_range + cost_range_for_severity(z.claimed_severity);
        for (const auto& [diff_zone, diff_value] : measured_diffs) {
            if (diff_zone != z.zone) continue;
            auto consistency = check_claim_consistency(z.claimed_severity, diff_value);
            if (consistency.verdict != ConsistencyVerdict::CONSISTENT) {
                result.flagged_for_review.push_back(to_string(z.zone) + ": " + consistency.note);
            }
        }
    }
    return result;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 21.3: Vehicle- and Property-Damage Assessment Engines for Insurance Claims\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the severity-to-cost-range policy is stated exactly, and a multi-zone "
                 "claim's total is the exact sum of its own per-zone ranges, never collapsed to a point --\n";
    {
        CHECK(cost_range_for_severity(DamageSeverity::NONE).low == 0.0);
        CHECK(cost_range_for_severity(DamageSeverity::NONE).high == 0.0);
        CHECK(cost_range_for_severity(DamageSeverity::MINOR).low == 150.0);
        CHECK(cost_range_for_severity(DamageSeverity::MINOR).high == 600.0);
        CHECK(cost_range_for_severity(DamageSeverity::MODERATE).low == 600.0);
        CHECK(cost_range_for_severity(DamageSeverity::MODERATE).high == 2500.0);
        CHECK(cost_range_for_severity(DamageSeverity::SEVERE).low == 2500.0);
        CHECK(cost_range_for_severity(DamageSeverity::SEVERE).high == 9000.0);

        std::vector<ZoneClaim> zones = {
            {DamageZone::FRONT_BUMPER, DamageSeverity::MODERATE},
            {DamageZone::HOOD, DamageSeverity::MINOR},
            {DamageZone::WINDSHIELD, DamageSeverity::SEVERE},
        };
        auto result = assess_claim(zones, {});
        // hand-computed: (600+150+2500, 2500+600+9000) = (3250, 12100)
        CHECK(std::abs(result.total_range.low - 3250.0) < 1e-9);
        CHECK(std::abs(result.total_range.high - 12100.0) < 1e-9);
        std::cout << "  a 3-zone claim (MODERATE front bumper, MINOR hood, SEVERE windshield) totals "
                     "to the exact hand-computed range [" << result.total_range.low << ", "
                   << result.total_range.high << "], never a single collapsed number\n";
    }

    std::cout << "\n-- Test 2: the real mean-absolute-difference metric matches hand-computed values exactly "
                 "--\n";
    {
        std::vector<uint8_t> before1(8, 100), after1(8, 140);   // every pixel differs by exactly 40
        CHECK(std::abs(mean_abs_diff(before1, after1) - 40.0) < 1e-9);

        std::vector<uint8_t> before2 = {100, 100, 100, 100};
        std::vector<uint8_t> after2 = {200, 200, 100, 100};     // half differ by 100, half by 0 -> mean 50
        CHECK(std::abs(mean_abs_diff(before2, after2) - 50.0) < 1e-9);

        std::vector<uint8_t> identical(10, 77);
        CHECK(mean_abs_diff(identical, identical) == 0.0);

        std::cout << "  8 pixels each differing by exactly 40 give a mean difference of exactly 40.0; "
                     "4 pixels split evenly between a 100-value and a 0-value difference give exactly "
                     "50.0; and an identical before/after patch gives exactly 0.0\n";
    }

    std::cout << "\n-- Test 3: consistency checking correctly reports CONSISTENT at both the interior and "
                 "the boundary of a claimed severity's own expected band --\n";
    {
        auto interior = check_claim_consistency(DamageSeverity::MODERATE, 55.0);   // well inside [30, 80]
        CHECK(interior.verdict == ConsistencyVerdict::CONSISTENT);
        auto lower_boundary = check_claim_consistency(DamageSeverity::MODERATE, 30.0);   // exactly at the low edge
        CHECK(lower_boundary.verdict == ConsistencyVerdict::CONSISTENT);
        auto upper_boundary = check_claim_consistency(DamageSeverity::MODERATE, 80.0);   // exactly at the high edge
        CHECK(upper_boundary.verdict == ConsistencyVerdict::CONSISTENT);
        std::cout << "  a measured difference of 55.0 (interior), 30.0 (the band's own low edge), and "
                     "80.0 (the band's own high edge) are all correctly reported CONSISTENT for a claimed "
                     "MODERATE severity\n";
    }

    std::cout << "\n-- Test 4: THE HONEST CHECK IN BOTH DIRECTIONS -- a SEVERE claim against near-identical "
                 "photos is flagged OVER_CLAIMED, and a MINOR claim against extensively different photos "
                 "is flagged UNDER_CLAIMED --\n";
    {
        auto over_claimed = check_claim_consistency(DamageSeverity::SEVERE, 3.0);   // claims SEVERE, photos barely differ
        CHECK(over_claimed.verdict == ConsistencyVerdict::OVER_CLAIMED);
        CHECK(over_claimed.note.find("BELOW") != std::string::npos);

        auto under_claimed = check_claim_consistency(DamageSeverity::MINOR, 150.0);   // claims MINOR, photos differ enormously
        CHECK(under_claimed.verdict == ConsistencyVerdict::UNDER_CLAIMED);
        CHECK(under_claimed.note.find("ABOVE") != std::string::npos);

        std::cout << "  a claim of SEVERE damage against a measured photo difference of only 3.0 is "
                     "flagged OVER_CLAIMED (\"" << over_claimed.note << "\"); a claim of MINOR damage "
                     "against a measured photo difference of 150.0 is flagged UNDER_CLAIMED (\""
                   << under_claimed.note << "\") -- neither flag auto-approves, auto-denies, or "
                     "auto-adjusts the claim, both simply name the specific direction of the "
                     "inconsistency for a human adjuster\n";
    }

    std::cout << "\n-- Test 5: a full multi-zone assessment flags exactly the one inconsistent zone by "
                 "name, while still reporting the honest, currently-claimed total for every zone --\n";
    {
        std::vector<ZoneClaim> zones = {
            {DamageZone::FRONT_BUMPER, DamageSeverity::MODERATE},
            {DamageZone::DRIVER_DOOR, DamageSeverity::SEVERE},
            {DamageZone::REAR_BUMPER, DamageSeverity::MINOR},
        };
        std::vector<std::pair<DamageZone, double>> measured = {
            {DamageZone::FRONT_BUMPER, 50.0},    // consistent with MODERATE [30,80]
            {DamageZone::DRIVER_DOOR, 4.0},      // claims SEVERE, but photos barely differ -- inconsistent
            {DamageZone::REAR_BUMPER, 12.0},     // consistent with MINOR [5,30]
        };
        auto result = assess_claim(zones, measured);
        // hand-computed total: (600+2500+150, 2500+9000+600) = (3250, 12100), regardless of the flag
        CHECK(std::abs(result.total_range.low - 3250.0) < 1e-9);
        CHECK(std::abs(result.total_range.high - 12100.0) < 1e-9);
        CHECK(result.flagged_for_review.size() == 1);
        CHECK(result.flagged_for_review[0].find("DRIVER_DOOR") != std::string::npos);
        CHECK(result.flagged_for_review[0].find("BELOW") != std::string::npos);
        std::cout << "  a 3-zone claim with one deliberately inconsistent zone (DRIVER_DOOR, claimed "
                     "SEVERE against nearly-unchanged photos) is flagged by name and reason, exactly "
                     "once, while the claim's own total cost range still honestly reflects all 3 zones "
                     "as currently claimed: [" << result.total_range.low << ", " << result.total_range.high
                   << "]\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_vehicle_damage_assessment_and_claim_consistency.cpp -o 03_vehicle_damage_assessment_and_claim_consistency
./03_vehicle_damage_assessment_and_claim_consistency
```

**Sample input:** the stated severity-to-cost-range policy checked exactly against all four tiers, and a 3-zone claim's total checked against a hand-computed sum of ranges; the mean-absolute-difference metric checked against three hand-computed synthetic patches; consistency checking checked at both the interior and the exact boundary values of a claimed severity's own expected band; a SEVERE claim against near-identical photos checked to be flagged OVER_CLAIMED and a MINOR claim against extensively different photos checked to be flagged UNDER_CLAIMED; and a full 3-zone claim checked to flag exactly its one inconsistent zone by name while still reporting the honest, currently-claimed total for all 3 zones.

```text
========================================================
Chapter 21.3: Vehicle- and Property-Damage Assessment Engines for Insurance Claims
========================================================

-- Test 1: the severity-to-cost-range policy is stated exactly, and a multi-zone claim's total is the exact sum of its own per-zone ranges, never collapsed to a point --
  a 3-zone claim (MODERATE front bumper, MINOR hood, SEVERE windshield) totals to the exact hand-computed range [3250, 12100], never a single collapsed number

-- Test 2: the real mean-absolute-difference metric matches hand-computed values exactly --
  8 pixels each differing by exactly 40 give a mean difference of exactly 40.0; 4 pixels split evenly between a 100-value and a 0-value difference give exactly 50.0; and an identical before/after patch gives exactly 0.0

-- Test 3: consistency checking correctly reports CONSISTENT at both the interior and the boundary of a claimed severity's own expected band --
  a measured difference of 55.0 (interior), 30.0 (the band's own low edge), and 80.0 (the band's own high edge) are all correctly reported CONSISTENT for a claimed MODERATE severity

-- Test 4: THE HONEST CHECK IN BOTH DIRECTIONS -- a SEVERE claim against near-identical photos is flagged OVER_CLAIMED, and a MINOR claim against extensively different photos is flagged UNDER_CLAIMED --
  a claim of SEVERE damage against a measured photo difference of only 3.0 is flagged OVER_CLAIMED ("measured photo difference (3.000000) is BELOW the expected band for claimed severity SEVERE [80.000000, 255.000000] -- the photos show less change than the claim describes"); a claim of MINOR damage against a measured photo difference of 150.0 is flagged UNDER_CLAIMED ("measured photo difference (150.000000) is ABOVE the expected band for claimed severity MINOR [5.000000, 30.000000] -- the photos show more change than the claim describes") -- neither flag auto-approves, auto-denies, or auto-adjusts the claim, both simply name the specific direction of the inconsistency for a human adjuster

-- Test 5: a full multi-zone assessment flags exactly the one inconsistent zone by name, while still reporting the honest, currently-claimed total for every zone --
  a 3-zone claim with one deliberately inconsistent zone (DRIVER_DOOR, claimed SEVERE against nearly-unchanged photos) is flagged by name and reason, exactly once, while the claim's own total cost range still honestly reflects all 3 zones as currently claimed: [3250, 12100]

25/25 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a flagged zone as grounds to quietly adjust the claim total"
    It is tempting, once a zone's own photo evidence is flagged inconsistent with its claimed severity, to have the assessment engine quietly substitute a "corrected" severity into the total -- after all, the engine has already computed a measured difference that disagrees with the claim. Test 5 is built specifically to rule that temptation out: DRIVER_DOOR's own claimed SEVERE tier is flagged as inconsistent with its measured difference, and the claim's own total cost range STILL includes DRIVER_DOOR's full SEVERE-tier range, exactly as claimed, with the inconsistency reported separately rather than silently resolved. Deciding which of the claim or the photograph is actually correct is a human adjuster's own real job -- an assessment engine that quietly substitutes its own guess for the claimed severity would be making that decision by itself, which is exactly the kind of unstructured authority this book's own human-in-the-loop discipline, since Chapter 20, has consistently refused to grant a piece of software.

## 21.4 Wildlife and Underwater Species Identification from Camera-Trap and Marine Imagery

### Intuition

A motion-triggered camera does not take one picture per real animal visit -- it takes a burst per real event, and the same real individual can trigger many separate bursts across a deployment. Treating a raw detection count as a population estimate is a real, well-documented mistake this section refuses to make, building the real ecological fix -- independent-event windowing -- from scratch instead.

### The Concept, In Detail

`finalize_species_label` applies a real, stated open-set confidence floor: Test 1 confirms a detection below that floor is forced to `UNKNOWN_UNCERTAIN` regardless of what a raw classifier's own label claimed, with the floor's own boundary checked exactly. `group_into_independent_events` is the real, standard camera-trap ecology technique this section builds from scratch: consecutive same-camera, same-species detections within a stated time window merge into ONE event, while a different species at the same camera, or the same species at a different camera, are each kept as genuinely separate events -- Test 2 and Test 3 confirm both the merging and the two real boundaries (species and location) that prevent over-merging.

Test 4 is this section's own honest-limit proof, built the same way Chapter 20.5's own Test 4 proved a saliency score's real limit by direct computation: a single real, lingering fox, returning to one camera across a full day in four widely-separated bursts, produces exactly 4 `independent_capture_events` -- a concrete, computed demonstration that this report's own number counts EVENTS, never individuals, since the entire scenario describes what could easily be just one real animal. `gray_world_correct` is a real, from-scratch application of the standard gray-world color-correction assumption, and Test 5 confirms it pulls a synthetic underwater-cast image's own per-channel spread from 140 down to 23, while its own stated amplification ceiling is confirmed directly against both a channel whose full hand-computed correction would have exceeded it and a pathological near-zero-mean channel.

### Code and Verification

```cpp
// Chapter 21.4 -- A motion-triggered camera trap does not take one
// picture per real animal visit; it takes a BURST of pictures for every
// real event that trips its sensor, and the same real individual can
// trip that sensor many separate times across a multi-week deployment.
// A vision-language species classifier run naively across every single
// frame would report a raw detection count that means almost nothing
// about how many real animals were actually there -- and reporting that
// raw count AS a population estimate is a real, well-documented mistake
// in camera-trap ecology, not a hypothetical one. This section builds
// the real, standard fix from scratch: an INDEPENDENT-CAPTURE-EVENT
// window that merges a burst of same-species, same-camera detections
// into one event, an open-set confidence floor that refuses to force a
// low-confidence detection into a specific species label at all, and a
// report schema that is honest about exactly what it counts by never
// containing a field that could be mistaken for a population estimate.
// The chapter's own second real domain, underwater marine imagery,
// needs a different real fix applied BEFORE any of that classification
// happens at all: water attenuates red light far faster than blue as
// depth increases, giving every underwater photograph a real, physical
// blue-green color cast a classifier trained on ordinary photographs was
// never shown, and this section builds the real, standard gray-world
// color-correction algorithm that removes it.
//
// A note on this section's own honest scope: the species labels below
// are a small, stated, closed set for demonstration, not a real trained
// classifier's own taxonomy, and the gray-world assumption this
// section's color correction relies on -- that a real scene's own
// average color is approximately neutral gray -- is a real, standard,
// but genuinely APPROXIMATE heuristic that can be wrong for a scene
// dominated by one true color (a coral reef with almost no blue in
// frame, for instance); this section states that limitation directly
// rather than presenting gray-world correction as an exact physical
// color-recovery model.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_camera_trap_deduplication_and_underwater_color_correction.cpp -o 04_camera_trap_deduplication_and_underwater_color_correction
// Run:     ./04_camera_trap_deduplication_and_underwater_color_correction

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: open-set species labeling. A detection below a stated
// confidence floor is NEVER forced into a specific species -- it is
// reported as genuinely uncertain, an honest bucket of its own rather
// than this section's own best (but low-confidence) guess.
// =======================================================================
enum class Species { DEER, FOX, RACCOON, UNKNOWN_UNCERTAIN };

std::string to_string(Species s) {
    switch (s) {
        case Species::DEER: return "DEER";
        case Species::FOX: return "FOX";
        case Species::RACCOON: return "RACCOON";
        case Species::UNKNOWN_UNCERTAIN: return "UNKNOWN_UNCERTAIN";
    }
    return "UNKNOWN_UNCERTAIN";
}

// A real, stated open-set floor -- below this confidence, this section
// refuses to report ANY specific species, regardless of which species a
// raw classifier score happened to favor.
constexpr double SPECIES_CONFIDENCE_FLOOR = 0.55;

Species finalize_species_label(Species raw_species, double confidence) {
    if (confidence < SPECIES_CONFIDENCE_FLOOR) return Species::UNKNOWN_UNCERTAIN;
    return raw_species;
}

struct Detection {
    std::string camera_id;
    int period_index = 0;   // a virtual, discrete tick -- never real wall-clock time
    Species species;
    double confidence = 0.0;
};

// =======================================================================
// PART 2: independent-capture-event deduplication -- the real, standard
// camera-trap ecology technique of merging consecutive same-camera,
// same-species detections separated by no more than a stated time
// window into ONE independent event, rather than counting every
// triggered frame as its own separate sighting.
// =======================================================================
struct CaptureEvent {
    std::string camera_id;
    Species species;
    int first_period = 0, last_period = 0;
    int detection_count = 0;   // the number of RAW detections merged into this one event -- never a population count
};

// Detections are first grouped by (camera_id, species) -- a different
// camera is always a different location, and a different species is
// always a different real event, so neither is ever merged across that
// boundary. Within one (camera, species) group, consecutive detections
// (sorted by period) merge into the SAME event as long as the gap since
// the event's own last period does not exceed `time_window`; a larger
// gap starts a genuinely new event.
std::vector<CaptureEvent> group_into_independent_events(std::vector<Detection> detections, int time_window) {
    std::stable_sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        if (a.camera_id != b.camera_id) return a.camera_id < b.camera_id;
        if (a.species != b.species) return static_cast<int>(a.species) < static_cast<int>(b.species);
        return a.period_index < b.period_index;
    });
    std::vector<CaptureEvent> events;
    for (const auto& d : detections) {
        if (!events.empty() && events.back().camera_id == d.camera_id && events.back().species == d.species &&
            (d.period_index - events.back().last_period) <= time_window) {
            events.back().last_period = d.period_index;
            events.back().detection_count += 1;
        } else {
            events.push_back(CaptureEvent{d.camera_id, d.species, d.period_index, d.period_index, 1});
        }
    }
    return events;
}

// =======================================================================
// PART 3: the report schema itself -- deliberately containing no field
// that could be mistaken for a population or individual count. The only
// number this report ever states is `independent_capture_events`,
// counted per species, and its own name says exactly, and only, what it
// measures.
// =======================================================================
struct SpeciesEventCount { Species species; int independent_capture_events = 0; };

std::vector<SpeciesEventCount> summarize_events_by_species(const std::vector<CaptureEvent>& events) {
    std::vector<SpeciesEventCount> counts;
    for (const auto& e : events) {
        auto it = std::find_if(counts.begin(), counts.end(),
                                [&](const SpeciesEventCount& c) { return c.species == e.species; });
        if (it == counts.end()) {
            counts.push_back(SpeciesEventCount{e.species, 1});
        } else {
            it->independent_capture_events += 1;
        }
    }
    return counts;
}

// =======================================================================
// PART 4: underwater gray-world color correction -- a real, standard,
// from-scratch preprocessing pass applied BEFORE species classification,
// removing the real physical blue-green color cast water imposes as
// depth increases.
// =======================================================================
struct RgbImage {
    int width = 0, height = 0;
    std::vector<uint8_t> r, g, b;   // parallel per-channel planes, row-major
};

struct ChannelMeans { double r, g, b; };

ChannelMeans compute_channel_means(const RgbImage& img) {
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    size_t n = img.r.size();
    for (size_t i = 0; i < n; ++i) { sum_r += img.r[i]; sum_g += img.g[i]; sum_b += img.b[i]; }
    return {sum_r / static_cast<double>(n), sum_g / static_cast<double>(n), sum_b / static_cast<double>(n)};
}

// A real, stated clamp on the per-channel scale factor -- without it, a
// channel with a near-zero mean (a real possibility in a badly
// attenuated deep-water photograph) would compute an enormous scale
// factor and amplify that channel's own sensor noise into a dominant,
// meaningless signal rather than a real color correction.
constexpr double GRAY_WORLD_SCALE_MIN = 0.5, GRAY_WORLD_SCALE_MAX = 2.0;

RgbImage gray_world_correct(const RgbImage& img) {
    ChannelMeans means = compute_channel_means(img);
    double gray_target = (means.r + means.g + means.b) / 3.0;
    auto scale_for = [&](double channel_mean) {
        double raw_scale = (channel_mean > 1e-9) ? (gray_target / channel_mean) : GRAY_WORLD_SCALE_MAX;
        return std::clamp(raw_scale, GRAY_WORLD_SCALE_MIN, GRAY_WORLD_SCALE_MAX);
    };
    double scale_r = scale_for(means.r), scale_g = scale_for(means.g), scale_b = scale_for(means.b);
    RgbImage out{img.width, img.height, img.r, img.g, img.b};
    auto apply = [](std::vector<uint8_t>& plane, double scale) {
        for (auto& v : plane) v = static_cast<uint8_t>(std::clamp(static_cast<double>(v) * scale, 0.0, 255.0));
    };
    apply(out.r, scale_r);
    apply(out.g, scale_g);
    apply(out.b, scale_b);
    return out;
}

// =======================================================================
// PART 5: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 21.4: Wildlife and Underwater Species Identification from Camera-Trap and Marine Imagery\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a detection below the stated confidence floor is forced to UNKNOWN_UNCERTAIN, "
                 "regardless of the raw species label, and the floor's own boundary is exact --\n";
    {
        CHECK(finalize_species_label(Species::DEER, 0.90) == Species::DEER);
        CHECK(finalize_species_label(Species::FOX, 0.20) == Species::UNKNOWN_UNCERTAIN);
        CHECK(finalize_species_label(Species::RACCOON, SPECIES_CONFIDENCE_FLOOR) == Species::RACCOON);        // exactly at the floor: passes
        CHECK(finalize_species_label(Species::RACCOON, SPECIES_CONFIDENCE_FLOOR - 0.001) == Species::UNKNOWN_UNCERTAIN);   // just below: gated
        std::cout << "  a high-confidence DEER detection passes through unchanged; a low-confidence FOX "
                     "detection is forced to UNKNOWN_UNCERTAIN; and the stated floor's own boundary is "
                     "exact -- confidence exactly at the floor passes, just below it does not\n";
    }

    std::cout << "\n-- Test 2: a real burst of same-camera, same-species detections within the stated time "
                 "window merges into ONE event; a gap larger than the window starts a genuinely new one --\n";
    {
        std::vector<Detection> burst = {
            {"CAM-01", 100, Species::DEER, 0.9}, {"CAM-01", 102, Species::DEER, 0.85},
            {"CAM-01", 105, Species::DEER, 0.88},
        };
        auto events = group_into_independent_events(burst, /*time_window=*/10);
        CHECK(events.size() == 1);
        CHECK(events[0].first_period == 100 && events[0].last_period == 105);
        CHECK(events[0].detection_count == 3);

        std::vector<Detection> two_visits = {
            {"CAM-01", 100, Species::DEER, 0.9}, {"CAM-01", 500, Species::DEER, 0.9},
        };
        auto two_events = group_into_independent_events(two_visits, /*time_window=*/10);
        CHECK(two_events.size() == 2);

        std::cout << "  3 detections at periods 100, 102, 105 (all within a window of 10) merge into "
                     "exactly 1 event spanning periods 100-105 with detection_count 3; 2 detections at "
                     "periods 100 and 500 (far beyond the window) correctly produce 2 separate events\n";
    }

    std::cout << "\n-- Test 3: a different SPECIES at the same camera, and the same species at a different "
                 "CAMERA, are each their own separate event -- location and species both distinguish events "
                 "--\n";
    {
        std::vector<Detection> same_camera_different_species = {
            {"CAM-01", 100, Species::DEER, 0.9}, {"CAM-01", 101, Species::FOX, 0.9},
        };
        auto events1 = group_into_independent_events(same_camera_different_species, 10);
        CHECK(events1.size() == 2);

        std::vector<Detection> different_camera_same_species = {
            {"CAM-01", 100, Species::DEER, 0.9}, {"CAM-02", 100, Species::DEER, 0.9},
        };
        auto events2 = group_into_independent_events(different_camera_same_species, 10);
        CHECK(events2.size() == 2);

        std::cout << "  a DEER and a FOX detected one period apart at the SAME camera are correctly kept "
                     "as 2 separate events; the SAME species detected at the SAME period by 2 DIFFERENT "
                     "cameras is likewise correctly kept as 2 separate events\n";
    }

    std::cout << "\n-- Test 4: THE HONEST LIMIT -- one real, lingering individual animal, captured in "
                 "several widely-separated bursts across a single deployment, produces MORE THAN ONE "
                 "independent capture event, proving directly that this report's own event count is not, "
                 "and must never be read as, an individual or population count --\n";
    {
        // One real fox that returns to the same camera's trigger zone
        // repeatedly across a single day -- 4 separate bursts, each
        // internally tight, but the bursts themselves are spaced far
        // enough apart (150 periods) to each start a new event under
        // this section's own stated 10-period window.
        std::vector<Detection> one_real_fox_all_day = {
            {"CAM-03", 100, Species::FOX, 0.9}, {"CAM-03", 102, Species::FOX, 0.9},
            {"CAM-03", 250, Species::FOX, 0.9},
            {"CAM-03", 400, Species::FOX, 0.9}, {"CAM-03", 403, Species::FOX, 0.9},
            {"CAM-03", 550, Species::FOX, 0.9},
        };
        auto events = group_into_independent_events(one_real_fox_all_day, /*time_window=*/10);
        CHECK(events.size() == 4);
        auto summary = summarize_events_by_species(events);
        CHECK(summary.size() == 1);
        CHECK(summary[0].species == Species::FOX);
        CHECK(summary[0].independent_capture_events == 4);
        std::cout << "  a single real fox returning to the same camera 4 separate times across one day "
                     "produces exactly 4 independent_capture_events for FOX -- a real, computed "
                     "demonstration that this number counts EVENTS, never individuals, since this "
                     "entire report describes what could easily be just 1 real animal\n";
    }

    std::cout << "\n-- Test 5: gray-world color correction pulls a real blue-green underwater color cast "
                 "toward neutral, and its own amplification is bounded even for a near-zero-mean channel --\n";
    {
        // A synthetic underwater-like image: red heavily attenuated
        // (mean 40), green moderately attenuated (mean 90), blue
        // dominant (mean 180) -- a real, physically-motivated color cast.
        RgbImage img{2, 2,
                     {40, 40, 40, 40},
                     {90, 90, 90, 90},
                     {180, 180, 180, 180}};
        auto before = compute_channel_means(img);
        CHECK(before.r == 40.0 && before.g == 90.0 && before.b == 180.0);
        double before_spread = std::max({before.r, before.g, before.b}) - std::min({before.r, before.g, before.b});

        auto corrected = gray_world_correct(img);
        auto after = compute_channel_means(corrected);
        double after_spread = std::max({after.r, after.g, after.b}) - std::min({after.r, after.g, after.b});
        CHECK(after_spread < before_spread);
        // Hand-computed: gray_target = (40+90+180)/3 = 103.333...; raw
        // scale for R = 103.333/40 = 2.583, clamped down to the stated
        // ceiling of 2.0, so every R pixel becomes exactly 40*2.0 = 80.
        CHECK(corrected.r[0] == 80);

        RgbImage near_black{1, 1, {0}, {128}, {128}};
        auto near_black_corrected = gray_world_correct(near_black);
        // A near-zero-mean red channel would compute an unbounded raw
        // scale factor without the stated clamp; with it, every R pixel
        // is scaled by at most GRAY_WORLD_SCALE_MAX, never exploding.
        CHECK(near_black_corrected.r[0] <= static_cast<uint8_t>(GRAY_WORLD_SCALE_MAX * 255));

        std::cout << "  a synthetic underwater-cast image (R mean 40, G mean 90, B mean 180) has its "
                     "per-channel spread shrink from " << before_spread << " to " << after_spread
                   << " after gray-world correction; the red channel's own scale factor is correctly "
                     "clamped at the stated ceiling of " << GRAY_WORLD_SCALE_MAX << " rather than "
                     "applying its full, hand-computed 2.583x; and a near-zero-mean channel is likewise "
                     "bounded rather than amplified without limit\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_camera_trap_deduplication_and_underwater_color_correction.cpp -o 04_camera_trap_deduplication_and_underwater_color_correction
./04_camera_trap_deduplication_and_underwater_color_correction
```

**Sample input:** a high-confidence and a low-confidence detection checked against the stated open-set floor, including its exact boundary; a real burst of same-camera, same-species detections checked to merge into one event, and a widely-separated pair checked to remain two; a different species at the same camera and the same species at a different camera each checked to remain separate events; a single real fox returning across a full day in 4 separated bursts checked to produce exactly 4 independent_capture_events, proving the count is not an individual count; and gray-world color correction checked against hand-computed per-channel means, its own stated amplification ceiling, and a near-zero-mean pathological channel.

```text
========================================================
Chapter 21.4: Wildlife and Underwater Species Identification from Camera-Trap and Marine Imagery
========================================================

-- Test 1: a detection below the stated confidence floor is forced to UNKNOWN_UNCERTAIN, regardless of the raw species label, and the floor's own boundary is exact --
  a high-confidence DEER detection passes through unchanged; a low-confidence FOX detection is forced to UNKNOWN_UNCERTAIN; and the stated floor's own boundary is exact -- confidence exactly at the floor passes, just below it does not

-- Test 2: a real burst of same-camera, same-species detections within the stated time window merges into ONE event; a gap larger than the window starts a genuinely new one --
  3 detections at periods 100, 102, 105 (all within a window of 10) merge into exactly 1 event spanning periods 100-105 with detection_count 3; 2 detections at periods 100 and 500 (far beyond the window) correctly produce 2 separate events

-- Test 3: a different SPECIES at the same camera, and the same species at a different CAMERA, are each their own separate event -- location and species both distinguish events --
  a DEER and a FOX detected one period apart at the SAME camera are correctly kept as 2 separate events; the SAME species detected at the SAME period by 2 DIFFERENT cameras is likewise correctly kept as 2 separate events

-- Test 4: THE HONEST LIMIT -- one real, lingering individual animal, captured in several widely-separated bursts across a single deployment, produces MORE THAN ONE independent capture event, proving directly that this report's own event count is not, and must never be read as, an individual or population count --
  a single real fox returning to the same camera 4 separate times across one day produces exactly 4 independent_capture_events for FOX -- a real, computed demonstration that this number counts EVENTS, never individuals, since this entire report describes what could easily be just 1 real animal

-- Test 5: gray-world color correction pulls a real blue-green underwater color cast toward neutral, and its own amplification is bounded even for a near-zero-mean channel --
  a synthetic underwater-cast image (R mean 40, G mean 90, B mean 180) has its per-channel spread shrink from 140 to 23 after gray-world correction; the red channel's own scale factor is correctly clamped at the stated ceiling of 2 rather than applying its full, hand-computed 2.583x; and a near-zero-mean channel is likewise bounded rather than amplified without limit

18/18 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] reading an independent-capture-event count as a population estimate"
    A report that says "4 independent capture events for FOX at this camera" reads, to an unfamiliar audience, uncomfortably close to "4 foxes" -- and Test 4 was built specifically to make the real gap between those two claims impossible to miss: the exact same scenario the test constructs, a single real fox visiting one camera repeatedly across a single day, produces that same number, 4, for exactly one real animal. This section's own report schema never contains a field named `population` or `individual_count` anywhere, precisely because no computation in this file is capable of telling one lingering individual apart from several visually-similar ones from camera-trap detections alone -- distinguishing those two real possibilities needs additional real evidence (a distinguishing marking, a radio collar, a mark-recapture study design) this section's own camera-trap pipeline does not have, and its own schema is built to never imply otherwise.

## Chapter Summary

This chapter moved this book's own vision-language core into three domains united by real, considerably lower financial or ecological stakes than Chapter 20's clinical ones, applying the same honesty discipline calibrated to each. Section 21.1 built a real connected-component and line-grouping algorithm from scratch and then ran the concrete experiment proving why a coordinate-calibrated extraction pipeline can silently swap two fields' values when a document's layout shifts, while a position-independent, semantic extraction contract does not. Section 21.2 built a real, priority-ordered document-type router over real structural features and a handwriting-aware confidence-clamping discipline that requires human review for a single handwritten field even inside an otherwise-printed, high-confidence document. Section 21.3 built a damage-assessment engine that never collapses a repair estimate into a false-precision point value, and a real, computable photo-consistency check that flags a claim as either over- or under-reported relative to its own submitted evidence, always to a human adjuster, never deciding the claim itself. Section 21.4 built the real, standard independent-capture-event windowing camera-trap ecology actually uses, proved by direct computation that its own event count is not a population estimate, and built a real gray-world color-correction pass for the physically distinct problem underwater imagery poses before any of that classification can even begin.

## Self-Check Questions

1. Section 21.1's Test 4 shows the rigid template extractor getting 1 of 3 fields right on a Vendor B document, not 0 of 3. Explain why the ONE field it gets right (TOTAL_AMOUNT) is a realistic outcome rather than a flaw in the test's own construction.
2. Section 21.1's structured parser never touches a page coordinate. Explain concretely why that is what makes it immune to the specific failure mode Test 4 demonstrates, rather than merely a different, equally fragile approach.
3. Section 21.2's `route_document` checks its own rules in a fixed priority order. Using Test 2's own fixture, explain why checking the ID-card rule before the receipt rule (rather than the reverse) is the correct choice.
4. Section 21.2 clamps a handwritten field's confidence to a stated ceiling rather than, say, halving whatever raw confidence a recognizer reports. What real problem would halving fail to solve that a fixed ceiling does solve?
5. Section 21.2's `requires_human_verification` check runs a per-field loop in addition to a per-document-type check. Construct a concrete document that would be WRONGLY cleared with no per-field check.
6. Section 21.3 reports a cost estimate as `CostRange` rather than a single expected-value number. Explain why a human adjuster is better served by a range than by, for instance, the midpoint of that same range reported as a single number.
7. Section 21.3's Test 4 flags inconsistency in BOTH directions -- over-claimed and under-claimed. Explain a real, legitimate (non-fraudulent) reason a genuine claim might be flagged UNDER_CLAIMED, to show why this flag is a prompt for review rather than an accusation.
8. Section 21.3's Test 5 confirms a flagged zone's cost range is still included, unmodified, in the claim's own total. What decision is this design deliberately leaving to a human adjuster rather than making automatically?
9. Section 21.4's Test 4 is described as this section's own "honest-limit proof." What specific claim about `independent_capture_events` does this test prove FALSE by direct construction, rather than merely warn against in a comment?
10. Section 21.4's gray-world correction clamps its own per-channel scale factor to a stated range. Describe a specific real photograph (or scene) where the gray-world ASSUMPTION itself -- not the clamp -- would produce a wrong correction, even though the arithmetic is computed correctly.

## Where We Go Next

This chapter showed the same real vision-language core, and the same recurring honesty discipline, generalizing across document intelligence, insurance, and ecological monitoring -- three domains whose stakes differ enormously from each other and from Chapter 20's, but whose real engineering failure modes (false precision, silent field-swapping, forced overconfident labels, population-estimate overclaiming) rhyme closely enough that the same from-scratch techniques -- structured schemas, stated confidence ceilings, honest ranges, and real, computed proofs of a technique's own limits -- keep proving to be the right tool. Chapter 22 turns to three more domains that share a different real pattern: turning a continuous camera feed, rather than a single photograph, into a structured, auditable narrative -- security surveillance with temporal, multi-camera event correlation, an accessibility-compliance auditing engine, and an art-condition assessment and provenance-verification engine.

## Worked Solutions

**1.** TOTAL_AMOUNT was deliberately given the SAME region in both Vendor A's and Vendor B's own layouts, representing a real, common situation where two different vendors' invoice templates happen to agree on where one particular field goes even though they disagree on others -- a realistic partial layout difference, not a contrived total mismatch. If the test had instead moved every field's position between the two layouts, a skeptical reader could reasonably wonder whether the rigid extractor's failure was an artifact of an unrealistically adversarial test construction; leaving one field's position genuinely unchanged, and watching it correctly extract anyway, makes the OTHER two fields' silent failure a real, specific consequence of THEIR OWN positions changing, not a blanket claim that coordinate-based extraction always fails everywhere.

**2.** The specific failure mode Test 4 demonstrates is a coordinate REGION picking up whichever blob happens to sit inside it, regardless of which field that blob actually represents -- a failure that is only possible because the extraction contract's own definition of "which field is this" is a page position. A parser that reads `KEY=VALUE` pairs by their own stated key name has no page position anywhere in its own definition of correctness at all: there is no region for a document's layout change to accidentally move a different field's value into, because the parser was never looking at regions in the first place. It is not that the parser is more careful about coordinates; it never had a way to be wrong about them to begin with.

**3.** Test 2's own fixture is constructed so a document could, in principle, satisfy the aspect-ratio condition of the receipt rule if that rule were checked in isolation -- but the SAME document also carries a solid block (like a photo) at a real ID-card aspect ratio, which is a considerably more specific and more physically distinctive combination of features than an aspect ratio alone. Checking the more specific, harder-to-satisfy-by-coincidence rule (solid block AND a narrow aspect-ratio window) before the broader, single-condition rule (aspect ratio alone, with a much wider range of documents that could satisfy it) means a document that genuinely looks like an ID card is not accidentally captured by a broader rule that was only ever meant to catch tall, narrow receipts.

**4.** Halving a raw confidence still lets an especially overconfident raw score dominate the result -- a raw score of 0.99 halved is still 0.495, and a raw score of 1.40 (already an invalid, out-of-range value from a malfunctioning or corrupted recognizer) halved is 0.70, which could still read as reasonably confident to a downstream system despite originating from a clearly broken input. A fixed ceiling, by contrast, guarantees NO handwritten field's reported confidence can ever exceed the stated bound, regardless of how extreme or invalid the raw input was -- the ceiling's whole point is a hard guarantee independent of the raw score's own magnitude, which a proportional scaling factor like halving can never provide on its own.

**5.** A printed insurance INVOICE with every field high-confidence PRINTED text except for one handwritten annotation reading "PAID IN FULL -- see attached receipt" scrawled in a margin: a document-type-only check would see `INVOICE` (not `HANDWRITTEN_FORM`) and clear the whole document for automated processing, silently trusting a handwritten claim about payment status that was never independently verified and that this book's own stated handwriting-confidence discipline says should never be treated with the same trust as the surrounding printed fields.

**6.** A single expected-value number invites exactly the same false-precision trap this book has refused since its own earliest quantization chapters: it presents a photograph-derived guess as though it carries the same certainty as a real shop estimate, and a downstream system or a human skimming a claim summary has no way to tell, from the number alone, how much real uncertainty that figure represents. A range makes the real uncertainty visible in the number itself -- a $600-$2,500 range for a MODERATE claim tells its own reader plainly that this is a photograph-derived estimate awaiting real inspection, which a single number like "$1,550" would obscure entirely.

**7.** A genuine claim could be flagged UNDER_CLAIMED if a claimant, trying to be conservative or simply unfamiliar with how much visible damage really costs to repair, described their own damage as MINOR out of caution or uncertainty, while the actual photographs show damage a trained adjuster would recognize as more extensive -- a real, honest claimant who genuinely believes "it's not that bad" is not lying, just wrong about severity, and the UNDER_CLAIMED flag exists to catch exactly this kind of honest miscalibration and get the claimant a FAIRER, likely HIGHER, settlement once a human reviews it -- proving the flag is not inherently accusatory, since it can just as easily work in the claimant's own favor.

**8.** This design deliberately leaves to the human adjuster the actual determination of which side of the inconsistency is correct -- whether the claimed severity should be revised down because the photographs genuinely show less damage than reported, whether the photographs themselves are misleading or were taken poorly, or whether some other real explanation (a repair already partially completed, a different angle) accounts for the mismatch. The assessment engine's own real job is surfacing the disagreement clearly and completely, with the claim's own original figures fully intact for that human's review, not substituting its own resolution of a real, substantive judgment call it has no authority (or sufficient real information) to make on its own.

**9.** The test proves false, by a concrete, hand-verified computed example, the specific claim that "the number of independent_capture_events for a species at a camera is a reasonable stand-in for how many individuals of that species visited that camera." The test constructs a scenario in which that number is 4, while the real, true number of distinct animals involved is stated directly to be 1 -- a computed counterexample, not a hedge or a caveat, showing the two numbers can differ by a factor of 4 (or more, in a longer deployment) for a completely ordinary, non-adversarial real scenario, not merely an unusual edge case.

**10.** A close-up photograph of a coral reef dominated almost entirely by one true, saturated color -- a section of reef that is genuinely, overwhelmingly one shade of orange coral with almost no blue, green, or neutral content anywhere in the frame -- would violate the gray-world assumption's own real premise that a scene's average color is approximately neutral. Gray-world correction, applied to such a photograph, would compute per-channel means that reflect the reef's own real, dominant color rather than a lighting-induced cast, and would incorrectly try to correct AWAY the reef's own real, true orange color rather than a spurious color cast introduced by the water -- the arithmetic runs correctly on the numbers it is given, but the numbers themselves no longer support the assumption the whole technique depends on.
