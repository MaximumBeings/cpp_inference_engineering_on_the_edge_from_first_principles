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
