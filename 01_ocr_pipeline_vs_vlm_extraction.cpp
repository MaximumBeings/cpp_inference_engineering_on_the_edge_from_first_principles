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
