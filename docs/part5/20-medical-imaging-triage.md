# Chapter 20: Medical Imaging Triage: Preliminary Radiology Reads and Human-in-the-Loop Workflow

**What you will understand by the end of this chapter:**

- Why a clinical triage system needs its human sign-off requirement built as a STRUCTURAL, type-system-enforced property of its own state machine, rather than a documented policy step a caller could forget or route around -- and how to build exactly one function capable of finalizing a case, gated on both the correct state and a real, non-empty clinician identifier.
- How to parse a real, standard subset of the DICOM file format (Explicit VR Little Endian tag/VR/length encoding) from scratch, and how to apply the exact real DICOM linear VOI LUT windowing formula a radiology workstation itself uses to turn raw pixel samples into a viewable grayscale image.
- How to design a structured triage-report schema and a strict parser that is deliberately, structurally INCAPABLE of representing a diagnosis -- restricting a queue-priority field to exactly three literal tokens and refusing any injected extra field outright -- and how to attach one unconditional, unomittable disclaimer to every report this chapter's own code can produce.
- How to extend a single case's own human-in-the-loop discipline across an entire worklist QUEUE using real optimistic concurrency, so that two radiologists racing to claim the same study cannot both succeed, and so that only the radiologist accountable for a case can ever advance or finalize it.
- How to build a real, honest explainability technique -- occlusion-based saliency, reusing Chapter 18's own frozen vision encoder unmodified -- and, just as importantly, how to prove by direct computation the specific, real limit of what that technique can and cannot tell you.

**What you need to know first:**

- Chapter 18.2's complete vision encoder (patchification, 2D RoPE, bidirectional ViT blocks, the 2x2 spatial merger), which Section 20.5 reuses byte-for-byte frozen, with no changes to any of its own five parts.
- Chapter 18.4's asymmetric threshold-based auto-disposition and Chapter 19.3's optimistic-concurrency `expected_version` PUT, both of which this chapter deliberately contrasts against or reuses in a domain where the stakes are considerably higher: Section 20.1 builds a STRUCTURAL alternative to threshold-based auto-disposition, and Section 20.4 reapplies Section 19.3's own concurrency discipline to a claim on a worklist item rather than a warehouse quantity.
- This chapter is handled with the extra care every section states directly: nothing in this chapter is legal, regulatory, or medical advice, no section claims compliance with any jurisdiction's software-as-a-medical-device framework, and every fixture, image, and report anywhere in this chapter is synthetic -- no real patient data and no real medical image appears anywhere in this book.

---

Chapter 18 taught a vision-language model to inspect a manufactured part and decide, on its own, whether the part passed. Chapter 19 taught the same core to reconcile a shelf against a planogram and raise its own findings. Medical imaging triage cannot work either way. A system that reads a scan and suggests how urgently a human should look at it is assisting a clinical decision, never making one, and this chapter states why that difference is not a matter of policy wording but of what a piece of software is structurally CAPABLE of doing on its own. This chapter builds every real piece that discipline requires, in the order a real triage pipeline would need them -- a state machine that cannot reach a final disposition without a real clinician's own sign-off, a real parser for the DICOM files a scan actually arrives in, a report format structurally incapable of smuggling a diagnosis into a queue-priority field, a worklist queue that extends the same sign-off discipline across many radiologists and many cases at once, and, finally, an honest account of how far a real explainability technique can see into the model's own reasoning, and exactly where it stops.

## 20.1 A Structural Human-in-the-Loop Triage State Machine

### Intuition

Chapter 18.4's `classify` decided, entirely on its own, whether a manufactured part passed or failed, gated only by a confidence threshold. A system suggesting how urgently a radiologist should look at a scan cannot be allowed to finalize anything on its own, at any confidence level -- and this section builds that guarantee as a property of the TYPE itself, not a rule a caller has to remember to follow.

### The Concept, In Detail

`TriageCase` moves through a real, ordered state machine -- `RECEIVED`, `AI_TRIAGE_COMPLETE`, `PENDING_HUMAN_REVIEW`, and finally `FINALIZED` or `REJECTED` -- and its own `final_disposition_` field is PRIVATE, reachable from exactly one function in the entire class: `record_human_signoff`. That function refuses to run unless the case already sits in `PENDING_HUMAN_REVIEW`, reachable only via `submit_ai_triage` followed by `route_to_review` in that order, AND unless it is given a real, non-empty clinician identifier -- Test 3 confirms an otherwise-valid sign-off attempt with an empty `clinician_id` is refused just as loudly as a wrong state, and leaves the case completely unaffected, because an unattributed sign-off is not a real audit-trail entry at all. Test 2 confirms the ordering itself is enforced, not merely the final step: attempting to finalize a case that skipped `route_to_review` entirely, and attempting to route a case that skipped `submit_ai_triage` entirely, are each refused with a specific, distinguishing error.

`submit_ai_triage` records the AI's own suggested priority into a field kept completely separate from `final_disposition_`, so a caller can never confuse what the AI suggested with what was actually decided -- and the audit log itself marks that suggestion "advisory" explicitly, in the text of the log entry, not only in a comment. A human reviewer disagreeing with the AI's own routing entirely is a real, expected outcome rather than an error: `reject_ai_triage`, tested in Test 4, moves a case to `REJECTED` with `final_disposition_` never touched, and the rejecting clinician's identity recorded by name in the audit trail. Test 5 confirms the audit trail itself is complete and correctly ordered across a full happy-path run: all four real transitions appear in sequence, with the AI's own suggestion explicitly marked advisory and the final entry naming the actual clinician who finalized the case.

### Code and Verification

```cpp
// Chapter 20.1 -- Every earlier chapter in Part 5 eventually let some
// piece of software take a real, autonomous action on its own: Chapter
// 18.4's `classify` auto-accepted or auto-rejected a manufactured part,
// and Chapter 19.3's `put_stock_level` committed an inventory change
// with no human in the loop at all. A system that reads a medical image
// and suggests how urgently a human should look at it cannot work that
// way, and this section states plainly why, and builds the one thing
// that actually enforces it.
//
// A note on scope, stated as directly as this book states every other
// limitation of its own: this is NOT legal or regulatory advice, this
// section does not claim compliance with any specific jurisdiction's
// software-as-a-medical-device framework, and a real deployment of
// anything resembling this code would need real regulatory and legal
// review this book cannot substitute for. What this section states, in
// general terms every such framework this book is aware of converges
// on in some form, is a single real principle: software that assists a
// clinical prioritization decision, rather than a licensed clinician
// making it, is treated as fundamentally different from software that
// AUTONOMOUSLY issues a final clinical action -- and the difference the
// rest of this section cares about is not a policy document's wording,
// but whether a human sign-off is a REAL, STRUCTURAL requirement a
// system cannot be routed around, or merely a step a policy asks
// someone to remember to perform.
//
// `TriageCase` is built to make that requirement structural rather than
// advisory. Its own `final_disposition_` field is PRIVATE, and the ONLY
// function in this entire class that can ever set it -- `record_human_
// signoff` -- refuses to run unless the case is already sitting in the
// one state (`PENDING_HUMAN_REVIEW`) that only a completed AI triage and
// an explicit routing step can reach, and unless it is given a real,
// non-empty clinician identifier. There is no second path to
// `FINALIZED` anywhere in this file, accidental or otherwise -- the
// invariant this section cares about is enforced by the type itself,
// not by a caller remembering to check something first.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_human_in_the_loop_triage_state_machine.cpp -o 01_human_in_the_loop_triage_state_machine
// Run:     ./01_human_in_the_loop_triage_state_machine

#include <iostream>
#include <optional>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the case state machine itself.
// =======================================================================
enum class CaseState { RECEIVED, AI_TRIAGE_COMPLETE, PENDING_HUMAN_REVIEW, FINALIZED, REJECTED };

std::string to_string(CaseState s) {
    switch (s) {
        case CaseState::RECEIVED: return "RECEIVED";
        case CaseState::AI_TRIAGE_COMPLETE: return "AI_TRIAGE_COMPLETE";
        case CaseState::PENDING_HUMAN_REVIEW: return "PENDING_HUMAN_REVIEW";
        case CaseState::FINALIZED: return "FINALIZED";
        case CaseState::REJECTED: return "REJECTED";
    }
    return "UNKNOWN";
}

class TriageCase {
public:
    struct Result { bool ok = false; std::string error; };

    explicit TriageCase(std::string case_id) : case_id_(std::move(case_id)) {
        log("case received");
    }

    // Records an AI system's own SUGGESTED priority -- explicitly
    // advisory. Nothing about this call, or anything this class
    // exposes, lets that suggestion become `final_disposition_` on its
    // own; it is stored in a completely separate field precisely so a
    // caller can never confuse "what the AI suggested" with "what was
    // actually decided."
    Result submit_ai_triage(const std::string& suggested_priority) {
        if (state_ != CaseState::RECEIVED) {
            return fail("submit_ai_triage requires state RECEIVED, current state is " + to_string(state_));
        }
        ai_suggested_priority_ = suggested_priority;
        state_ = CaseState::AI_TRIAGE_COMPLETE;
        log("AI triage complete: suggested priority = \"" + suggested_priority + "\" (advisory only)");
        return ok_result();
    }

    Result route_to_review() {
        if (state_ != CaseState::AI_TRIAGE_COMPLETE) {
            return fail("route_to_review requires state AI_TRIAGE_COMPLETE, current state is " + to_string(state_));
        }
        state_ = CaseState::PENDING_HUMAN_REVIEW;
        log("routed to human review queue");
        return ok_result();
    }

    // The ONLY function anywhere in this class that can move a case to
    // FINALIZED, and the only one that can ever assign
    // `final_disposition_`. Requires the case to already be sitting in
    // PENDING_HUMAN_REVIEW (reachable only via the two steps above, in
    // order) AND a real, non-empty clinician identifier -- an empty
    // identifier is refused just as loudly as a wrong state, because a
    // real audit trail with no record of WHO reviewed a case is not a
    // real audit trail at all.
    Result record_human_signoff(const std::string& clinician_id, const std::string& final_disposition) {
        if (state_ != CaseState::PENDING_HUMAN_REVIEW) {
            return fail("record_human_signoff requires state PENDING_HUMAN_REVIEW, current state is " + to_string(state_));
        }
        if (clinician_id.empty()) {
            return fail("record_human_signoff requires a non-empty clinician_id");
        }
        if (final_disposition.empty()) {
            return fail("record_human_signoff requires a non-empty final_disposition");
        }
        reviewing_clinician_id_ = clinician_id;
        final_disposition_ = final_disposition;
        state_ = CaseState::FINALIZED;
        log("FINALIZED by clinician \"" + clinician_id + "\": " + final_disposition);
        return ok_result();
    }

    // A human reviewer disagreeing with the AI's own suggested routing
    // entirely is a real, expected outcome, not an error -- the case
    // moves to REJECTED, `final_disposition_` is never touched, and the
    // disagreement itself is recorded in the audit trail by name.
    Result reject_ai_triage(const std::string& clinician_id, const std::string& reason) {
        if (state_ != CaseState::PENDING_HUMAN_REVIEW) {
            return fail("reject_ai_triage requires state PENDING_HUMAN_REVIEW, current state is " + to_string(state_));
        }
        if (clinician_id.empty()) {
            return fail("reject_ai_triage requires a non-empty clinician_id");
        }
        state_ = CaseState::REJECTED;
        log("AI triage rejected by clinician \"" + clinician_id + "\": " + reason);
        return ok_result();
    }

    CaseState state() const { return state_; }
    const std::optional<std::string>& ai_suggested_priority() const { return ai_suggested_priority_; }
    const std::optional<std::string>& final_disposition() const { return final_disposition_; }
    const std::optional<std::string>& reviewing_clinician_id() const { return reviewing_clinician_id_; }
    const std::vector<std::string>& audit_log() const { return audit_log_; }

private:
    Result fail(std::string error) { return Result{false, std::move(error)}; }
    Result ok_result() { return Result{true, ""}; }
    void log(const std::string& entry) { audit_log_.push_back(entry); }

    std::string case_id_;
    CaseState state_ = CaseState::RECEIVED;
    std::optional<std::string> ai_suggested_priority_;
    std::optional<std::string> final_disposition_;
    std::optional<std::string> reviewing_clinician_id_;
    std::vector<std::string> audit_log_;
};

// =======================================================================
// PART 2: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.1: A Structural Human-in-the-Loop Triage State Machine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the full, correctly-ordered happy path reaches FINALIZED --\n";
    {
        TriageCase c("CASE-0001");
        auto r1 = c.submit_ai_triage("EXPEDITED");
        auto r2 = c.route_to_review();
        auto r3 = c.record_human_signoff("dr-alvarez", "confirmed expedited review, findings consistent with AI-suggested priority");
        CHECK(r1.ok && r2.ok && r3.ok);
        CHECK(c.state() == CaseState::FINALIZED);
        CHECK(c.ai_suggested_priority().value() == "EXPEDITED");
        CHECK(c.final_disposition().has_value());
        CHECK(c.reviewing_clinician_id().value() == "dr-alvarez");
        CHECK(c.audit_log().size() == 4);   // received, AI triage, routed, finalized
        std::cout << "  a case moves RECEIVED -> AI_TRIAGE_COMPLETE -> PENDING_HUMAN_REVIEW -> FINALIZED, "
                     "with the final disposition attributed to clinician \"" << *c.reviewing_clinician_id()
                   << "\" and a " << c.audit_log().size() << "-entry audit trail\n";
    }

    std::cout << "\n-- Test 2: skipping a step is refused, whichever step is skipped --\n";
    {
        TriageCase skip_review("CASE-0002");
        skip_review.submit_ai_triage("ROUTINE");
        // Attempting to finalize directly after AI triage, without ever
        // routing to a human review queue at all.
        auto r = skip_review.record_human_signoff("dr-nakamura", "reviewed");
        CHECK(!r.ok);
        CHECK(skip_review.state() == CaseState::AI_TRIAGE_COMPLETE);
        CHECK(!skip_review.final_disposition().has_value());

        TriageCase skip_ai("CASE-0003");
        // Attempting to route straight to human review with no AI
        // triage step having run at all.
        auto r2 = skip_ai.route_to_review();
        CHECK(!r2.ok);
        CHECK(skip_ai.state() == CaseState::RECEIVED);

        std::cout << "  finalizing a case that was never routed to review is refused (\"" << r.error
                   << "\"); routing a case that was never AI-triaged is refused (\"" << r2.error
                   << "\") -- the ordering itself is enforced, not just the final sign-off step\n";
    }

    std::cout << "\n-- Test 3: an empty clinician_id is refused even when the state is otherwise correct --\n";
    {
        TriageCase c("CASE-0004");
        c.submit_ai_triage("STAT");
        c.route_to_review();
        auto r = c.record_human_signoff("", "reviewed and confirmed");
        CHECK(!r.ok);
        CHECK(r.error.find("clinician_id") != std::string::npos);
        CHECK(c.state() == CaseState::PENDING_HUMAN_REVIEW);   // completely unaffected by the rejected call
        CHECK(!c.final_disposition().has_value());
        std::cout << "  a sign-off attempt with an empty clinician_id is refused (\"" << r.error
                   << "\"), and the case remains PENDING_HUMAN_REVIEW, completely unaffected -- an "
                     "unattributed sign-off is not a real audit trail entry, so it is not accepted at all\n";
    }

    std::cout << "\n-- Test 4: a clinician can reject the AI's own suggestion outright --\n";
    {
        TriageCase c("CASE-0005");
        c.submit_ai_triage("ROUTINE");
        c.route_to_review();
        auto r = c.reject_ai_triage("dr-osei", "imaging quality insufficient for the suggested priority tier, escalating manually");
        CHECK(r.ok);
        CHECK(c.state() == CaseState::REJECTED);
        CHECK(!c.final_disposition().has_value());   // rejection is not a disposition
        bool mentions_clinician = false;
        for (const auto& entry : c.audit_log()) if (entry.find("dr-osei") != std::string::npos) mentions_clinician = true;
        CHECK(mentions_clinician);
        std::cout << "  a human reviewer can reject the AI's own suggested routing entirely; the case "
                     "moves to REJECTED with no final_disposition ever set, and the audit log records "
                     "the reviewing clinician by name\n";
    }

    std::cout << "\n-- Test 5: the audit trail records every real transition, in order, attributing the human step --\n";
    {
        TriageCase c("CASE-0006");
        c.submit_ai_triage("EXPEDITED");
        c.route_to_review();
        c.record_human_signoff("dr-ferreira", "confirmed");
        const auto& log = c.audit_log();
        CHECK(log.size() == 4);
        CHECK(log[0].find("received") != std::string::npos);
        CHECK(log[1].find("AI triage") != std::string::npos && log[1].find("advisory") != std::string::npos);
        CHECK(log[2].find("routed") != std::string::npos);
        CHECK(log[3].find("FINALIZED") != std::string::npos && log[3].find("dr-ferreira") != std::string::npos);
        std::cout << "  all 4 real transitions appear in the audit log in the exact order they "
                     "occurred, with the AI's own suggestion explicitly marked advisory and the "
                     "final entry naming the human clinician who actually finalized the case\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_human_in_the_loop_triage_state_machine.cpp -o 01_human_in_the_loop_triage_state_machine
./01_human_in_the_loop_triage_state_machine
```

**Sample input:** a full, correctly-ordered happy path checked to reach `FINALIZED` with a 4-entry audit trail attributing the sign-off to a named clinician; finalizing a case that skipped review, and routing a case that skipped AI triage, each checked to be refused; an empty `clinician_id` checked to be refused even when the state is otherwise correct, leaving the case completely unaffected; a clinician rejecting the AI's own suggested routing outright, checked to leave `final_disposition_` unset while recording the rejecting clinician by name; and the full audit trail of a complete happy-path run checked against its exact expected 4 entries, in order.

```text
========================================================
Chapter 20.1: A Structural Human-in-the-Loop Triage State Machine
========================================================

-- Test 1: the full, correctly-ordered happy path reaches FINALIZED --
  a case moves RECEIVED -> AI_TRIAGE_COMPLETE -> PENDING_HUMAN_REVIEW -> FINALIZED, with the final disposition attributed to clinician "dr-alvarez" and a 4-entry audit trail

-- Test 2: skipping a step is refused, whichever step is skipped --
  finalizing a case that was never routed to review is refused ("record_human_signoff requires state PENDING_HUMAN_REVIEW, current state is AI_TRIAGE_COMPLETE"); routing a case that was never AI-triaged is refused ("route_to_review requires state AI_TRIAGE_COMPLETE, current state is RECEIVED") -- the ordering itself is enforced, not just the final sign-off step

-- Test 3: an empty clinician_id is refused even when the state is otherwise correct --
  a sign-off attempt with an empty clinician_id is refused ("record_human_signoff requires a non-empty clinician_id"), and the case remains PENDING_HUMAN_REVIEW, completely unaffected -- an unattributed sign-off is not a real audit trail entry, so it is not accepted at all

-- Test 4: a clinician can reject the AI's own suggestion outright --
  a human reviewer can reject the AI's own suggested routing entirely; the case moves to REJECTED with no final_disposition ever set, and the audit log records the reviewing clinician by name

-- Test 5: the audit trail records every real transition, in order, attributing the human step --
  all 4 real transitions appear in the audit log in the exact order they occurred, with the AI's own suggestion explicitly marked advisory and the final entry naming the human clinician who actually finalized the case

24/24 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] a threshold-based auto-disposition rule generalizing from Chapter 18.4 into a domain where it does not belong"
    Chapter 18.4's `classify` auto-accepted or auto-rejected a manufactured part once its own confidence score crossed a stated threshold, and that design was the right one for a defect-detection line where a wrong call costs a discarded part. Reapplying that same pattern here -- auto-finalizing a triage case whenever the AI's own suggested priority carries a high enough confidence score -- would be a serious, substantive error, not a matter of degree: a manufactured part can be re-inspected, but a clinical prioritization decision this section's own docstrings state plainly is not this book's to make cannot be un-made by adding more confidence-threshold precision. `TriageCase`'s own private `final_disposition_` field, reachable from exactly one gated function, is what makes the difference structural rather than a matter of choosing the right threshold value -- there is no threshold high enough that this section's own code would ever let a case reach `FINALIZED` without a real clinician's own attributed sign-off.

## 20.2 DICOM Ingestion and Windowing for Display

### Intuition

A shelf photograph in Chapter 19 arrived as an ordinary raw pixel buffer. A medical scan does not: it arrives as a real, standardized DICOM file carrying its own pixel data alongside real metadata -- dimensions, bit depth, and a WINDOW CENTER and WINDOW WIDTH the scanner or radiologist intends the image to be viewed through -- and that container has to be parsed for real before a single pixel reaches Chapter 18's own vision encoder.

### The Concept, In Detail

This section implements Explicit VR Little Endian, DICOM's single most common transfer syntax, and reads only the handful of real, standard data elements a windowed-extraction pipeline actually needs: Rows, Columns, BitsAllocated, WindowCenter, WindowWidth, and PixelData, each addressed by its own real standard tag. It states its own scope as plainly as this book has stated the scope of every other external format it has implemented a real subset of since GGUF in Chapter 5: no Implicit VR, no big-endian transfer syntax, no compressed pixel data, and 16-bit signed grayscale samples only -- correctly and verifiably implementing the real, specific slice of the standard this chapter's own windowing math depends on, not a claim to the whole of it.

`DicomReader::get_ds` correctly reverses two real encoding rules `DicomWriter` itself has to apply when producing a valid file: a DS (decimal string) value of odd length must be padded to an even length with a trailing space, which Test 5 confirms round-trips a padded `"500"` back to exactly `500.0`; and a DS element MAY carry multiple backslash-separated values, of which this section's own stated scope reads only the first, which Test 5 also confirms directly against a real three-valued `"1024\2048\4096"` string. Test 4 confirms the reader refuses a file with corrupted magic bytes outright, by name, rather than parsing corrupted input as if it were valid.

`apply_window` is the exact real DICOM linear VOI LUT formula (PS3.3, C.11.2.1.2): a sample at or below the lower clip boundary maps to the display minimum, a sample above the upper clip boundary maps to the display maximum, and everything between maps linearly with no discontinuity at either edge. Test 2 checks this formula against five hand-computed values at once -- both clip boundaries exactly, both clipped regions, and the formula's own true midpoint -- and Test 3 confirms the same formula applied through the real file-parsing path, against a synthetic file whose five raw samples were chosen specifically to span clipped-low, the lower boundary, a near-midpoint value, the upper boundary, and clipped-high, produces exactly the expected windowed byte at every one of the five positions.

### Code and Verification

```cpp
// Chapter 20.2 -- A shelf photograph from Chapter 19 arrives as an
// ordinary raw pixel buffer; a medical scan does not. It arrives as a
// real, standardized DICOM file -- Digital Imaging and Communications in
// Medicine -- carrying its own pixel data alongside real metadata (the
// image's own dimensions, how many bits each pixel occupies, and a
// WINDOW CENTER and WINDOW WIDTH the scanner or radiologist intends the
// image to be viewed through) inside one self-describing binary
// container. Before Section 20.1's own triage machinery -- or Section
// 18.2's vision encoder, unmodified since Chapter 18 -- can see a single
// pixel, that container has to be parsed for real, and the raw sample
// values it holds have to be converted into a real, viewable grayscale
// image using the exact windowing math a radiology workstation itself
// applies.
//
// A note on this section's own real, stated scope, in the same honest
// voice this book has used for every external format it has ever
// implemented a real subset of (GGUF in Chapter 5, GVSP in Chapter 18.1):
// this section implements Explicit VR Little Endian, the single most
// common DICOM transfer syntax, and reads only the handful of real,
// standard data elements a windowed-extraction pipeline actually needs
// (Rows, Columns, BitsAllocated, WindowCenter, WindowWidth, PixelData).
// It does not implement Implicit VR, big-endian transfer syntaxes, or
// compressed pixel data (JPEG, JPEG-LS, JPEG2000 transfer syntaxes),
// and it assumes 16-bit signed grayscale samples -- the common case for
// CT and MR pixel data expressed in Hounsfield-unit-like values -- not
// every real DICOM photometric interpretation. A real clinical PACS
// pipeline has to handle far more of the standard than this; this
// section implements, correctly and verifiably, the real, specific
// slice of it that this chapter's own windowing math depends on.
//
// The windowing formula itself is the exact linear VOI LUT function the
// real DICOM standard defines (PS3.3, C.11.2.1.2): a raw sample at or
// below `center - 0.5 - (width-1)/2` clips to the display minimum, a
// sample above `center - 0.5 + (width-1)/2` clips to the display
// maximum, and everything between maps linearly -- continuously, with
// no discontinuity at either boundary, which this section's own tests
// verify with hand-computable numbers rather than trusting the formula
// on faith.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_dicom_ingestion_and_windowing.cpp -o 02_dicom_ingestion_and_windowing
// Run:     ./02_dicom_ingestion_and_windowing

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
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
// PART 1: the real Explicit VR Little Endian element format.
// =======================================================================
struct Tag {
    uint16_t group;
    uint16_t element;
    bool operator==(const Tag& o) const { return group == o.group && element == o.element; }
};
struct TagHash {
    size_t operator()(const Tag& t) const { return (static_cast<size_t>(t.group) << 16) | t.element; }
};

// The real, standard tags this section's own pipeline needs.
constexpr Tag TAG_ROWS{0x0028, 0x0010};
constexpr Tag TAG_COLUMNS{0x0028, 0x0011};
constexpr Tag TAG_BITS_ALLOCATED{0x0028, 0x0100};
constexpr Tag TAG_WINDOW_CENTER{0x0028, 0x1050};
constexpr Tag TAG_WINDOW_WIDTH{0x0028, 0x1051};
constexpr Tag TAG_PIXEL_DATA{0x7FE0, 0x0010};

// The real DICOM rule: a VR from this set is encoded with 2 reserved
// bytes followed by a 4-byte length; every other (short-form) VR is
// encoded with a 2-byte length directly. This is the real baseline set
// -- a handful of newer VRs from later DICOM editions are not included,
// consistent with this section's own stated scope above.
bool is_long_form_vr(const std::string& vr) {
    static const std::set<std::string> long_form = {"OB", "OW", "OF", "SQ", "UT", "UN"};
    return long_form.count(vr) != 0;
}

struct DataElement {
    Tag tag{};
    std::string vr;
    std::vector<uint8_t> value;
};

// =======================================================================
// PART 2: DicomWriter -- builds a real, byte-accurate synthetic DICOM
// file for this section's own self-tests. Nothing about this class is
// part of the real ingestion pipeline; it exists so the reader below can
// be tested against bytes this section's own code fully controls and
// can verify by construction, exactly as Chapter 5's GGUFWriter did for
// Chapter 15's real GGUF reader.
// =======================================================================
class DicomWriter {
public:
    explicit DicomWriter(const std::string& path) : out_(path, std::ios::binary) {}

    void write_preamble_and_magic() {
        std::vector<uint8_t> preamble(128, 0);
        out_.write(reinterpret_cast<const char*>(preamble.data()), static_cast<std::streamsize>(preamble.size()));
        out_.write("DICM", 4);
    }

    void write_corrupt_magic() {
        std::vector<uint8_t> preamble(128, 0);
        out_.write(reinterpret_cast<const char*>(preamble.data()), static_cast<std::streamsize>(preamble.size()));
        out_.write("XXXX", 4);
    }

    // A real string-VR value is padded to even length with a trailing
    // space per the DICOM standard's own real rule; this writer applies
    // that rule itself so the reader's own trimming can be tested
    // against a genuinely padded value, not one this test fixture
    // conveniently avoided needing to pad in the first place.
    void write_element(Tag tag, const std::string& vr, std::vector<uint8_t> value) {
        if (value.size() % 2 != 0) value.push_back(static_cast<uint8_t>(' '));
        write_u16(tag.group);
        write_u16(tag.element);
        out_.write(vr.data(), 2);
        if (is_long_form_vr(vr)) {
            write_u16(0);   // reserved
            write_u32(static_cast<uint32_t>(value.size()));
        } else {
            write_u16(static_cast<uint16_t>(value.size()));
        }
        if (!value.empty()) out_.write(reinterpret_cast<const char*>(value.data()), static_cast<std::streamsize>(value.size()));
    }

    void write_element_us(Tag tag, uint16_t v) {
        std::vector<uint8_t> bytes(2);
        std::memcpy(bytes.data(), &v, 2);
        write_element(tag, "US", bytes);
    }

    void write_element_ds(Tag tag, const std::string& decimal_string) {
        write_element(tag, "DS", std::vector<uint8_t>(decimal_string.begin(), decimal_string.end()));
    }

    void write_pixel_data_16bit(const std::vector<int16_t>& samples) {
        std::vector<uint8_t> bytes(samples.size() * 2);
        std::memcpy(bytes.data(), samples.data(), bytes.size());
        write_element(TAG_PIXEL_DATA, "OW", bytes);
    }

    bool good() const { return out_.good(); }

    // Explicitly flushes and closes the underlying file. A test fixture
    // that only relies on the destructor to close the file risks reading
    // it back before that destructor has actually run -- this section's
    // own tests call `close()` explicitly, right after writing every
    // element, specifically to avoid that real, easy-to-miss ordering bug.
    void close() { out_.close(); }

private:
    void write_u16(uint16_t v) { out_.write(reinterpret_cast<const char*>(&v), 2); }
    void write_u32(uint32_t v) { out_.write(reinterpret_cast<const char*>(&v), 4); }
    std::ofstream out_;
};

// =======================================================================
// PART 3: DicomReader -- the real ingestion code under test.
// =======================================================================
class DicomReader {
public:
    bool open(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) { last_error_ = "could not open file: " + path; return false; }
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return parse(bytes);
    }

    std::optional<uint16_t> get_us(Tag tag) const {
        auto it = elements_.find(tag);
        if (it == elements_.end() || it->second.value.size() < 2) return std::nullopt;
        uint16_t v = 0;
        std::memcpy(&v, it->second.value.data(), 2);
        return v;
    }

    // DS (Decimal String) values may be multi-valued, backslash-
    // separated -- this section's own stated scope reads only the
    // FIRST value, which is sufficient for a single-frame window
    // center/width and stated honestly as not handling genuinely
    // per-frame multi-valued windowing.
    std::optional<double> get_ds(Tag tag) const {
        auto it = elements_.find(tag);
        if (it == elements_.end()) return std::nullopt;
        std::string raw(it->second.value.begin(), it->second.value.end());
        size_t backslash = raw.find('\\');
        std::string first = (backslash == std::string::npos) ? raw : raw.substr(0, backslash);
        // Trim the real DICOM padding characters (trailing space or NUL).
        while (!first.empty() && (first.back() == ' ' || first.back() == '\0')) first.pop_back();
        while (!first.empty() && (first.front() == ' ')) first.erase(first.begin());
        if (first.empty()) return std::nullopt;
        double result = 0.0;
        auto parsed = std::from_chars(first.data(), first.data() + first.size(), result);
        if (parsed.ec != std::errc() || parsed.ptr != first.data() + first.size()) return std::nullopt;
        return result;
    }

    const std::vector<uint8_t>* get_raw(Tag tag) const {
        auto it = elements_.find(tag);
        return (it == elements_.end()) ? nullptr : &it->second.value;
    }

    const std::string& last_error() const { return last_error_; }

private:
    bool parse(const std::vector<uint8_t>& bytes) {
        if (bytes.size() < 132) { last_error_ = "file too small to contain a preamble and DICM magic"; return false; }
        if (std::string(bytes.begin() + 128, bytes.begin() + 132) != "DICM") {
            last_error_ = "missing 'DICM' magic at offset 128 -- not a valid DICOM file (or an unsupported transfer syntax)";
            return false;
        }
        size_t pos = 132;
        while (pos + 8 <= bytes.size()) {
            Tag tag{};
            std::memcpy(&tag.group, &bytes[pos], 2);
            std::memcpy(&tag.element, &bytes[pos + 2], 2);
            pos += 4;
            std::string vr(bytes.begin() + static_cast<long>(pos), bytes.begin() + static_cast<long>(pos) + 2);
            pos += 2;

            uint32_t length = 0;
            if (is_long_form_vr(vr)) {
                if (pos + 6 > bytes.size()) { last_error_ = "truncated long-form VR header"; return false; }
                pos += 2;   // reserved
                std::memcpy(&length, &bytes[pos], 4);
                pos += 4;
            } else {
                if (pos + 2 > bytes.size()) { last_error_ = "truncated short-form VR header"; return false; }
                uint16_t short_len = 0;
                std::memcpy(&short_len, &bytes[pos], 2);
                length = short_len;
                pos += 2;
            }
            if (pos + length > bytes.size()) { last_error_ = "element value runs past end of file"; return false; }

            DataElement el;
            el.tag = tag;
            el.vr = vr;
            el.value.assign(bytes.begin() + static_cast<long>(pos), bytes.begin() + static_cast<long>(pos) + length);
            elements_[tag] = std::move(el);
            pos += length;
        }
        return true;
    }

    std::unordered_map<Tag, DataElement, TagHash> elements_;
    std::string last_error_;
};

// =======================================================================
// PART 4: real DICOM linear VOI LUT windowing (PS3.3 C.11.2.1.2), and
// a real windowed-image extractor built on top of the reader above.
// =======================================================================
double apply_window(double x, double center, double width, double ymin = 0.0, double ymax = 255.0) {
    double lower = center - 0.5 - (width - 1.0) / 2.0;
    double upper = center - 0.5 + (width - 1.0) / 2.0;
    if (x <= lower) return ymin;
    if (x > upper) return ymax;
    return ((x - (center - 0.5)) / (width - 1.0) + 0.5) * (ymax - ymin) + ymin;
}

struct WindowedImage {
    bool ok = false;
    std::string error;
    uint16_t rows = 0;
    uint16_t columns = 0;
    std::vector<uint8_t> pixels;   // row-major, one byte per pixel
};

WindowedImage extract_windowed_image(const DicomReader& reader) {
    WindowedImage img;
    auto rows = reader.get_us(TAG_ROWS);
    auto cols = reader.get_us(TAG_COLUMNS);
    auto bits = reader.get_us(TAG_BITS_ALLOCATED);
    auto center = reader.get_ds(TAG_WINDOW_CENTER);
    auto width = reader.get_ds(TAG_WINDOW_WIDTH);
    const auto* pixel_data = reader.get_raw(TAG_PIXEL_DATA);

    if (!rows || !cols || !bits || !center || !width || !pixel_data) {
        img.error = "missing one or more required elements (Rows, Columns, BitsAllocated, WindowCenter, WindowWidth, PixelData)";
        return img;
    }
    if (*bits != 16) {
        img.error = "unsupported BitsAllocated (" + std::to_string(*bits) + ") -- this section only implements 16-bit grayscale";
        return img;
    }
    size_t expected_samples = static_cast<size_t>(*rows) * static_cast<size_t>(*cols);
    if (pixel_data->size() != expected_samples * 2) {
        img.error = "PixelData size does not match Rows*Columns*2 bytes -- refusing to extract from a mismatched buffer";
        return img;
    }

    img.rows = *rows;
    img.columns = *cols;
    img.pixels.resize(expected_samples);
    for (size_t i = 0; i < expected_samples; ++i) {
        int16_t raw = 0;
        std::memcpy(&raw, pixel_data->data() + i * 2, 2);
        double windowed = apply_window(static_cast<double>(raw), *center, *width);
        windowed = std::clamp(windowed, 0.0, 255.0);
        img.pixels[i] = static_cast<uint8_t>(std::lround(windowed));
    }
    img.ok = true;
    return img;
}

// =======================================================================
// PART 5: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.2: DICOM Ingestion and Windowed Image Extraction\n";
    std::cout << "========================================================\n";

    const std::string synth_path = "/tmp/ch20_2_synthetic.dcm";

    std::cout << "\n-- Test 1: a real synthetic DICOM file round-trips its own metadata exactly --\n";
    {
        DicomWriter w(synth_path);
        w.write_preamble_and_magic();
        w.write_element_us(TAG_ROWS, 4);
        w.write_element_us(TAG_COLUMNS, 4);
        w.write_element_us(TAG_BITS_ALLOCATED, 16);
        w.write_element_ds(TAG_WINDOW_CENTER, "2048");
        w.write_element_ds(TAG_WINDOW_WIDTH, "4096");
        std::vector<int16_t> samples(16, 0);
        w.write_pixel_data_16bit(samples);
        w.close();
        CHECK(w.good());

        DicomReader r;
        CHECK(r.open(synth_path));
        CHECK(r.get_us(TAG_ROWS).value() == 4);
        CHECK(r.get_us(TAG_COLUMNS).value() == 4);
        CHECK(r.get_us(TAG_BITS_ALLOCATED).value() == 16);
        CHECK(r.get_ds(TAG_WINDOW_CENTER).value() == 2048.0);
        CHECK(r.get_ds(TAG_WINDOW_WIDTH).value() == 4096.0);
        std::cout << "  a synthetic 4x4, 16-bit DICOM file round-trips Rows, Columns, BitsAllocated, "
                     "WindowCenter, and WindowWidth exactly as written\n";
    }

    std::cout << "\n-- Test 2: the real linear VOI LUT windowing formula matches hand-computed values exactly --\n";
    {
        // center=2048, width=4096 -> lower bound = 0.0, upper bound = 4095.0 exactly.
        CHECK(apply_window(0.0, 2048, 4096) == 0.0);        // exactly at the lower clip boundary
        CHECK(apply_window(-100.0, 2048, 4096) == 0.0);     // well below -- clipped
        CHECK(std::abs(apply_window(2047.5, 2048, 4096) - 127.5) < 1e-9);   // the formula's own true midpoint
        CHECK(std::abs(apply_window(4095.0, 2048, 4096) - 255.0) < 1e-9);   // exactly at the upper clip boundary
        CHECK(apply_window(5000.0, 2048, 4096) == 255.0);   // well above -- clipped
        std::cout << "  window(center=2048, width=4096): x=0 -> 0 (lower boundary), x=-100 -> 0 "
                     "(clipped), x=2047.5 -> 127.5 (the formula's true midpoint), x=4095 -> 255 "
                     "(upper boundary), x=5000 -> 255 (clipped) -- all matching hand-computed values exactly\n";
    }

    std::cout << "\n-- Test 3: extracting a windowed image from a real file produces exactly the expected bytes --\n";
    {
        DicomWriter w(synth_path);
        w.write_preamble_and_magic();
        w.write_element_us(TAG_ROWS, 1);
        w.write_element_us(TAG_COLUMNS, 5);
        w.write_element_us(TAG_BITS_ALLOCATED, 16);
        w.write_element_ds(TAG_WINDOW_CENTER, "2048");
        w.write_element_ds(TAG_WINDOW_WIDTH, "4096");
        // Five known raw samples spanning clip-low, near-low, near-mid, near-high, clip-high.
        std::vector<int16_t> samples = {-100, 0, 2048, 4095, 5000};
        w.write_pixel_data_16bit(samples);
        w.close();
        CHECK(w.good());

        DicomReader r;
        CHECK(r.open(synth_path));
        auto img = extract_windowed_image(r);
        CHECK(img.ok);
        CHECK(img.rows == 1 && img.columns == 5);
        CHECK(img.pixels.size() == 5);
        CHECK(img.pixels[0] == 0);     // -100: clipped low
        CHECK(img.pixels[1] == 0);     // 0: exactly the lower boundary
        CHECK(img.pixels[4] == 255);   // 5000: clipped high
        CHECK(img.pixels[3] == 255);   // 4095: exactly the upper boundary
        // pixel[2] (raw=2048) rounds from 127.53...  to 128.
        CHECK(img.pixels[2] == 128);
        std::cout << "  a 1x5 synthetic image with raw samples {-100, 0, 2048, 4095, 5000} extracts "
                     "to windowed bytes {" << static_cast<int>(img.pixels[0]) << ", "
                   << static_cast<int>(img.pixels[1]) << ", " << static_cast<int>(img.pixels[2]) << ", "
                   << static_cast<int>(img.pixels[3]) << ", " << static_cast<int>(img.pixels[4])
                   << "} -- clipped, boundary, and near-midpoint values all correct\n";
    }

    std::cout << "\n-- Test 4: a file with the wrong magic bytes is refused, not crashed on --\n";
    {
        DicomWriter w(synth_path);
        w.write_corrupt_magic();
        w.close();
        CHECK(w.good());
        DicomReader r;
        bool opened = r.open(synth_path);
        CHECK(!opened);
        CHECK(r.last_error().find("DICM") != std::string::npos);
        std::cout << "  a file with corrupted magic bytes at offset 128 is correctly refused (\""
                   << r.last_error() << "\") rather than parsed as if it were valid\n";
    }

    std::cout << "\n-- Test 5: a padded, odd-length DS value and a multi-valued DS both parse correctly --\n";
    {
        DicomWriter w(synth_path);
        w.write_preamble_and_magic();
        // "500" is 3 bytes -- an odd length the writer must pad to 4 with
        // a trailing space, exactly the real DICOM rule this test checks
        // the reader correctly reverses.
        w.write_element_ds(TAG_WINDOW_WIDTH, "500");
        // A real multi-valued DS: this section's own stated scope reads
        // only the first value.
        w.write_element_ds(TAG_WINDOW_CENTER, "1024\\2048\\4096");
        w.write_element_us(TAG_ROWS, 1);
        w.write_element_us(TAG_COLUMNS, 1);
        w.write_element_us(TAG_BITS_ALLOCATED, 16);
        w.write_pixel_data_16bit({0});
        w.close();
        CHECK(w.good());

        DicomReader r;
        CHECK(r.open(synth_path));
        CHECK(r.get_ds(TAG_WINDOW_WIDTH).value() == 500.0);
        CHECK(r.get_ds(TAG_WINDOW_CENTER).value() == 1024.0);   // first of the three backslash-separated values
        std::cout << "  a space-padded odd-length DS value (\"500\" padded to 4 bytes) correctly "
                     "parses to 500.0, and a multi-valued DS (\"1024\\2048\\4096\") correctly reads "
                     "only its first value, 1024.0\n";
    }

    std::remove(synth_path.c_str());

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_dicom_ingestion_and_windowing.cpp -o 02_dicom_ingestion_and_windowing
./02_dicom_ingestion_and_windowing
```

**Sample input:** a synthetic 4x4, 16-bit DICOM file checked to round-trip Rows, Columns, BitsAllocated, WindowCenter, and WindowWidth exactly as written; the linear VOI LUT formula checked against five hand-computed values spanning both clip boundaries, both clipped regions, and its own true midpoint; a 1x5 synthetic image with five raw samples chosen to span every one of those same cases, checked to extract to exactly the expected windowed bytes; a file with corrupted magic bytes checked to be refused by name rather than parsed; and a space-padded odd-length DS value together with a real multi-valued DS, each checked to parse correctly.

```text
========================================================
Chapter 20.2: DICOM Ingestion and Windowed Image Extraction
========================================================

-- Test 1: a real synthetic DICOM file round-trips its own metadata exactly --
  a synthetic 4x4, 16-bit DICOM file round-trips Rows, Columns, BitsAllocated, WindowCenter, and WindowWidth exactly as written

-- Test 2: the real linear VOI LUT windowing formula matches hand-computed values exactly --
  window(center=2048, width=4096): x=0 -> 0 (lower boundary), x=-100 -> 0 (clipped), x=2047.5 -> 127.5 (the formula's true midpoint), x=4095 -> 255 (upper boundary), x=5000 -> 255 (clipped) -- all matching hand-computed values exactly

-- Test 3: extracting a windowed image from a real file produces exactly the expected bytes --
  a 1x5 synthetic image with raw samples {-100, 0, 2048, 4095, 5000} extracts to windowed bytes {0, 0, 128, 255, 255} -- clipped, boundary, and near-midpoint values all correct

-- Test 4: a file with the wrong magic bytes is refused, not crashed on --
  a file with corrupted magic bytes at offset 128 is correctly refused ("missing 'DICM' magic at offset 128 -- not a valid DICOM file (or an unsupported transfer syntax)") rather than parsed as if it were valid

-- Test 5: a padded, odd-length DS value and a multi-valued DS both parse correctly --
  a space-padded odd-length DS value ("500" padded to 4 bytes) correctly parses to 500.0, and a multi-valued DS ("1024\2048\4096") correctly reads only its first value, 1024.0

29/29 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] constructing a DicomReader in the same scope as an unflushed DicomWriter"
    This section's own first working draft of Test 1 failed immediately with `std::bad_optional_access`, thrown from inside `DicomReader::open` on what should have been a valid, freshly-written file. The real cause was not a parsing bug at all: `DicomWriter`'s member `std::ofstream` had never been flushed or closed before a `DicomReader` was constructed and asked to read the exact same file path back within the same scope, so the writer's own destructor -- which would eventually close the file -- had not yet run, and the file `DicomReader::open` actually opened was zero bytes long. The fix was a real, explicit `close()` method on `DicomWriter`, called before every test constructs a reader against the same path. Two file handles pointed at the same path do not automatically observe each other's writes in the order the source code suggests; only a closed (or explicitly flushed) writer guarantees a reader opened afterward sees the complete file.

## 20.3 A Structured Triage-Reporting Prompt and Parser

### Intuition

Section 20.2 can now hand a real, windowed image to Chapter 18's vision-language pipeline. What that pipeline says back has to be constrained just as carefully as the state machine that gates its own finalization -- not merely by asking it nicely to avoid diagnosing, but by giving it a report SCHEMA that cannot represent a diagnosis even if a model's own text tried to write one into it.

### The Concept, In Detail

`render_triage_prompt` states its own constraint directly in the rendered prompt text itself -- "You are NOT diagnosing this study" and "URGENCY is a QUEUE PRIORITY for human review, not a diagnosis" -- but this section does not stop at asking politely. `parse_finding_line`'s own strict field-count check refuses ANY line carrying more than the exact four expected fields (FINDING, LOCATION, URGENCY, CONFIDENCE), which Test 2 confirms directly against a deliberately injected `"DIAGNOSIS=acute myocardial infarction"` fifth field -- refused outright, with no path for that field to survive into a `ParsedFinding` at all. The `Urgency` enum itself is restricted to exactly three literal tokens (`ROUTINE`, `EXPEDITED`, `STAT`), and Test 2 confirms both a generic invalid value (`"URGENT"`) and, more pointedly, a diagnosis-shaped string (`"PNEUMONIA"`) written directly into the URGENCY field are each refused by the identical check -- there is no way to smuggle a diagnosis into the one field the schema does allow free-form-adjacent content near, because that field's own valid values are a closed, exhaustively-checked set, not free text at all.

`build_triage_report` is the ONLY function in this file capable of constructing a `TriageReport`, and it unconditionally sets `disclaimer` to the exact fixed `MANDATORY_DISCLAIMER` string, with no parameter capable of changing or omitting it -- Test 5 confirms two reports built from completely different findings still carry byte-identical disclaimers. `overall_priority` is computed as the single MOST URGENT finding across the whole study via `urgency_rank`'s own max, never an average -- Test 4 confirms a study with two `ROUTINE` findings and one `STAT` finding correctly reports `STAT` as its overall priority, and that an empty findings list correctly defaults to `ROUTINE` rather than leaving the field in an undefined state.

### Code and Verification

```cpp
// Chapter 20.3 -- Section 20.2 produced a real, windowed, viewable image
// from a DICOM study. Section 20.1 built a state machine that refuses to
// let anything but a human clinician finalize a case. This section
// builds the piece that has to sit correctly between them: the prompt
// that tells a vision-language model exactly what kind of answer it is
// allowed to give, and a parser strict enough that even a model that
// tried to give a different kind of answer could not get it through.
//
// The one real design decision this section centers on is what the
// STRUCTURED OUTPUT FORMAT ITSELF is allowed to represent. `URGENCY` is
// a QUEUE PRIORITY -- which of three tiers a human reviewer's own
// worklist should treat a study as -- and the parser accepts exactly
// three literal values for it (`ROUTINE`, `EXPEDITED`, `STAT`) and
// nothing else. There is no `DIAGNOSIS` field anywhere in this format,
// and Test 2 confirms directly that a line attempting to smuggle one in
// -- either as an extra field, or by writing a diagnosis-shaped string
// into the `URGENCY` field itself -- is refused outright by the same
// strict parsing discipline Chapter 19.2 already applied to a retail
// detection line. This is the schema-level half of Section 20.1's own
// structural guarantee: even if a model's own free-text output somehow
// tried to assert a diagnosis, there is no path for that assertion to
// survive parsing and reach anything downstream as structured data. The
// same discipline governs the report as a whole: `build_triage_report`
// is the ONLY way to construct a `TriageReport`, and it unconditionally
// attaches one fixed, unmodifiable disclaimer string -- there is no
// parameter, overload, or code path anywhere in this file that produces
// a `TriageReport` without it.
//
// As in Section 20.1, this section makes no claim to represent any
// specific jurisdiction's regulatory requirements for how such a report
// must be worded or structured; the disclaimer text below is this
// book's own stated, illustrative example of the KIND of statement such
// a system would need to carry, not a claim that this exact wording
// satisfies any real requirement.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_triage_report_prompt_and_parser.cpp -o 03_triage_report_prompt_and_parser
// Run:     ./03_triage_report_prompt_and_parser

#include <algorithm>
#include <charconv>
#include <iostream>
#include <set>
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
// PART 1: study metadata and a deterministic, validated system prompt.
// =======================================================================
struct StudyMetadata {
    std::string study_id;
    std::string modality;     // one of a real, fixed, known set (see validate_study_metadata)
    std::string body_part;
    int rows = 0;
    int columns = 0;
};

struct ValidationResult { bool ok = false; std::string error; };

ValidationResult validate_study_metadata(const StudyMetadata& m) {
    static const std::set<std::string> known_modalities = {"CT", "MR", "CR", "US", "XA", "DX"};
    if (m.study_id.empty()) return {false, "study_id must not be empty"};
    if (!known_modalities.count(m.modality)) return {false, "unknown modality: \"" + m.modality + "\""};
    if (m.body_part.empty()) return {false, "body_part must not be empty"};
    if (m.rows <= 0 || m.columns <= 0) return {false, "rows and columns must both be positive"};
    return {true, ""};
}

// A fixed, real instructional block, stating what kind of answer this
// prompt permits and what structured format the answer must follow.
// Rendering is otherwise identical to Chapter 19.1's own deterministic
// planogram rendering: the same metadata always produces the exact same
// prompt text, byte for byte.
std::string render_triage_prompt(const StudyMetadata& m) {
    std::ostringstream out;
    out << "TRIAGE ASSISTANCE REQUEST\n";
    out << "You are assisting a radiologist's own review queue. You are NOT diagnosing this study.\n";
    out << "For each notable finding, respond with EXACTLY one line in this format:\n";
    out << "FINDING=<short description>;LOCATION=<anatomical location>;URGENCY=<ROUTINE|EXPEDITED|STAT>;CONFIDENCE=<0.0-1.0>\n";
    out << "URGENCY is a QUEUE PRIORITY for human review, not a diagnosis. Do not include any other field.\n";
    out << "If the study shows nothing notable, respond with no finding lines at all.\n";
    out << "Study: " << m.study_id << "\n";
    out << "Modality: " << m.modality << "\n";
    out << "Body part: " << m.body_part << "\n";
    out << "Image size: " << m.rows << "x" << m.columns << "\n";
    return out.str();
}

// =======================================================================
// PART 2: urgency tiers and the strict finding-line parser.
// =======================================================================
enum class Urgency { ROUTINE, EXPEDITED, STAT };

std::string to_string(Urgency u) {
    switch (u) {
        case Urgency::ROUTINE: return "ROUTINE";
        case Urgency::EXPEDITED: return "EXPEDITED";
        case Urgency::STAT: return "STAT";
    }
    return "UNKNOWN";
}

// Real ordering for aggregation: STAT is the most urgent, ROUTINE the
// least. Used only to pick the single most urgent tier across a study's
// own findings -- never to average or otherwise blend urgency values.
int urgency_rank(Urgency u) { return static_cast<int>(u); }

struct ParsedFinding {
    std::string finding;
    std::string location;
    Urgency urgency = Urgency::ROUTINE;
    double confidence = 0.0;
};

struct ParseResult {
    bool ok = false;
    std::string error;
    ParsedFinding finding;
};

// Strict by construction: exactly the four real fields this format
// defines, an URGENCY value that must be one of the three real literal
// tokens (never a free-text string that could carry a diagnosis), and a
// CONFIDENCE inside [0, 1]. `std::from_chars` throughout, matching this
// book's own standing preference for checked return values over
// exceptions.
ParseResult parse_finding_line(const std::string& line) {
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

    for (const char* required : {"FINDING", "LOCATION", "URGENCY", "CONFIDENCE"}) {
        if (!fields.count(required)) {
            res.error = std::string("missing required field: ") + required;
            return res;
        }
    }
    if (fields.size() != 4) {
        // This is the real structural refusal: an extra field -- for
        // instance a model-generated "DIAGNOSIS=..." -- is rejected
        // wholesale, not silently dropped or silently accepted alongside
        // the four real fields.
        res.error = "unexpected extra field(s) present (expected exactly FINDING, LOCATION, URGENCY, CONFIDENCE)";
        return res;
    }
    if (fields["FINDING"].empty()) { res.error = "FINDING field is empty"; return res; }
    if (fields["LOCATION"].empty()) { res.error = "LOCATION field is empty"; return res; }

    const std::string& urgency_str = fields["URGENCY"];
    Urgency urgency;
    if (urgency_str == "ROUTINE") urgency = Urgency::ROUTINE;
    else if (urgency_str == "EXPEDITED") urgency = Urgency::EXPEDITED;
    else if (urgency_str == "STAT") urgency = Urgency::STAT;
    else {
        res.error = "URGENCY must be one of ROUTINE, EXPEDITED, STAT (a queue priority, not a diagnosis) -- got \"" + urgency_str + "\"";
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
    res.finding = ParsedFinding{fields["FINDING"], fields["LOCATION"], urgency, confidence};
    return res;
}

// =======================================================================
// PART 3: TriageReport -- the ONLY way to attach findings to a fixed,
// unmodifiable disclaimer, and the ONLY way to compute a study's own
// overall queue priority.
// =======================================================================
constexpr const char* MANDATORY_DISCLAIMER =
    "PRELIMINARY AI-GENERATED TRIAGE SUGGESTION -- NOT A DIAGNOSIS -- "
    "REQUIRES REVIEW AND SIGN-OFF BY A LICENSED CLINICIAN BEFORE ANY CLINICAL ACTION";

struct TriageReport {
    std::string study_id;
    std::vector<ParsedFinding> findings;
    Urgency overall_priority = Urgency::ROUTINE;
    std::string disclaimer;
};

// A study with no findings at all defaults to ROUTINE -- an absence of
// notable findings is not itself an urgent signal. A study with any
// findings takes the SINGLE MOST URGENT tier among them, never an
// average and never merely the first one encountered, since a study
// containing even one STAT-tier finding belongs at the front of a real
// review queue regardless of how many ROUTINE findings sit alongside it.
TriageReport build_triage_report(const std::string& study_id, const std::vector<ParsedFinding>& findings) {
    TriageReport report;
    report.study_id = study_id;
    report.findings = findings;
    report.disclaimer = MANDATORY_DISCLAIMER;   // unconditional -- no parameter can omit or alter this
    Urgency worst = Urgency::ROUTINE;
    for (const auto& f : findings) {
        if (urgency_rank(f.urgency) > urgency_rank(worst)) worst = f.urgency;
    }
    report.overall_priority = worst;
    return report;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.3: A Structured Triage-Reporting Prompt and Parser\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: study metadata validates, and the rendered prompt is deterministic and complete --\n";
    {
        StudyMetadata m{"STUDY-2026-0091", "CT", "CHEST", 512, 512};
        auto v = validate_study_metadata(m);
        CHECK(v.ok);
        std::string p1 = render_triage_prompt(m);
        std::string p2 = render_triage_prompt(m);
        CHECK(p1 == p2);
        CHECK(p1.find("NOT diagnosing") != std::string::npos);
        CHECK(p1.find("QUEUE PRIORITY for human review, not a diagnosis") != std::string::npos);
        CHECK(p1.find("STUDY-2026-0091") != std::string::npos);
        CHECK(p1.find("512x512") != std::string::npos);

        StudyMetadata bad_modality{"STUDY-BAD", "LASER", "CHEST", 512, 512};
        auto v_bad = validate_study_metadata(bad_modality);
        CHECK(!v_bad.ok);
        CHECK(v_bad.error.find("modality") != std::string::npos);

        std::cout << "  a valid CT chest study validates and renders deterministically, with the "
                     "prompt explicitly stating it is not a diagnosis request and that URGENCY is a "
                     "queue priority; an unknown modality (\"LASER\") is correctly refused (\""
                   << v_bad.error << "\")\n";
    }

    std::cout << "\n-- Test 2: the parser accepts real queue-priority tiers and refuses every attempt to smuggle a diagnosis --\n";
    {
        auto r1 = parse_finding_line("FINDING=opacity;LOCATION=right lower lobe;URGENCY=EXPEDITED;CONFIDENCE=0.72");
        CHECK(r1.ok);
        CHECK(r1.finding.urgency == Urgency::EXPEDITED);

        auto r2 = parse_finding_line("FINDING=small nodule;LOCATION=left upper lobe;URGENCY=ROUTINE;CONFIDENCE=0.4");
        CHECK(r2.ok && r2.finding.urgency == Urgency::ROUTINE);

        auto r3 = parse_finding_line("FINDING=large pneumothorax;LOCATION=left hemithorax;URGENCY=STAT;CONFIDENCE=0.95");
        CHECK(r3.ok && r3.finding.urgency == Urgency::STAT);

        // A free-text urgency value -- not one of the three real
        // tokens -- is refused outright, even though "URGENT" looks
        // superficially like it could be a real tier.
        auto bad_urgency = parse_finding_line("FINDING=x;LOCATION=y;URGENCY=URGENT;CONFIDENCE=0.5");
        CHECK(!bad_urgency.ok);
        CHECK(bad_urgency.error.find("queue priority") != std::string::npos);

        // An attempted diagnosis written INTO the urgency field is
        // refused by the exact same check -- there is no special case
        // that treats a diagnosis-shaped string differently from any
        // other invalid token.
        auto smuggled_in_urgency = parse_finding_line("FINDING=x;LOCATION=y;URGENCY=PNEUMONIA;CONFIDENCE=0.9");
        CHECK(!smuggled_in_urgency.ok);

        // An attempted diagnosis as its OWN extra field is refused by
        // the strict field-count check -- the schema has no slot for it
        // to occupy even if a model tried to emit one.
        auto smuggled_extra_field = parse_finding_line(
            "FINDING=x;LOCATION=y;URGENCY=STAT;CONFIDENCE=0.9;DIAGNOSIS=acute myocardial infarction");
        CHECK(!smuggled_extra_field.ok);
        CHECK(smuggled_extra_field.error.find("extra field") != std::string::npos);

        std::cout << "  all three real queue-priority tiers (ROUTINE, EXPEDITED, STAT) parse "
                     "correctly; a free-text \"URGENT\" is refused (\"" << bad_urgency.error
                   << "\"); a diagnosis-shaped string written into URGENCY is refused by the same "
                     "check; and an explicit DIAGNOSIS=... field is refused outright (\""
                   << smuggled_extra_field.error << "\") -- there is no path for a diagnosis to "
                     "survive parsing as structured data\n";
    }

    std::cout << "\n-- Test 3: an out-of-range confidence and a missing field are both refused --\n";
    {
        auto bad_conf = parse_finding_line("FINDING=x;LOCATION=y;URGENCY=ROUTINE;CONFIDENCE=1.5");
        CHECK(!bad_conf.ok);
        auto missing = parse_finding_line("FINDING=x;LOCATION=y;URGENCY=ROUTINE");
        CHECK(!missing.ok);
        CHECK(missing.error.find("missing required field") != std::string::npos);
        std::cout << "  an out-of-range confidence (1.5) and a missing CONFIDENCE field are both "
                     "correctly refused\n";
    }

    std::cout << "\n-- Test 4: a study's overall priority is its SINGLE MOST URGENT finding, never an average --\n";
    {
        std::vector<ParsedFinding> mixed = {
            {"small nodule", "left upper lobe", Urgency::ROUTINE, 0.4},
            {"small nodule", "right upper lobe", Urgency::ROUTINE, 0.35},
            {"pneumothorax", "left hemithorax", Urgency::STAT, 0.9},
        };
        auto report = build_triage_report("STUDY-0002", mixed);
        CHECK(report.overall_priority == Urgency::STAT);
        CHECK(report.findings.size() == 3);

        auto empty_report = build_triage_report("STUDY-0003", {});
        CHECK(empty_report.overall_priority == Urgency::ROUTINE);
        CHECK(empty_report.findings.empty());

        std::cout << "  a study with two ROUTINE findings and one STAT finding correctly reports an "
                     "overall priority of " << to_string(report.overall_priority)
                   << " (the single most urgent tier present, not an average of the three); a study "
                     "with no findings at all correctly defaults to " << to_string(empty_report.overall_priority) << "\n";
    }

    std::cout << "\n-- Test 5: the mandatory disclaimer is unconditional, identical, and unomittable across reports --\n";
    {
        auto r1 = build_triage_report("STUDY-A", {});
        auto r2 = build_triage_report("STUDY-B", {{"x", "y", Urgency::STAT, 0.99}});
        CHECK(r1.disclaimer == r2.disclaimer);
        CHECK(r1.disclaimer == MANDATORY_DISCLAIMER);
        CHECK(r1.disclaimer.find("NOT A DIAGNOSIS") != std::string::npos);
        CHECK(r1.disclaimer.find("LICENSED CLINICIAN") != std::string::npos);
        std::cout << "  two reports with completely different findings still carry the exact same, "
                     "byte-identical disclaimer -- build_triage_report offers no parameter capable of "
                     "changing or omitting it\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_triage_report_prompt_and_parser.cpp -o 03_triage_report_prompt_and_parser
./03_triage_report_prompt_and_parser
```

**Sample input:** a valid CT chest study checked to validate and render a deterministic prompt explicitly stating it is not a diagnosis request, alongside an unknown modality checked to be refused; all three real queue-priority tiers checked to parse correctly, alongside a free-text `"URGENT"`, a diagnosis-shaped `"PNEUMONIA"` written into URGENCY, and an explicit injected `DIAGNOSIS=...` field, each checked to be refused; an out-of-range confidence and a missing field each checked to be refused; a study with mixed ROUTINE and STAT findings checked to report STAT as its overall priority, and an empty findings list checked to default to ROUTINE; and two reports with completely different findings checked to carry byte-identical, unconditional disclaimers.

```text
========================================================
Chapter 20.3: A Structured Triage-Reporting Prompt and Parser
========================================================

-- Test 1: study metadata validates, and the rendered prompt is deterministic and complete --
  a valid CT chest study validates and renders deterministically, with the prompt explicitly stating it is not a diagnosis request and that URGENCY is a queue priority; an unknown modality ("LASER") is correctly refused ("unknown modality: "LASER"")

-- Test 2: the parser accepts real queue-priority tiers and refuses every attempt to smuggle a diagnosis --
  all three real queue-priority tiers (ROUTINE, EXPEDITED, STAT) parse correctly; a free-text "URGENT" is refused ("URGENCY must be one of ROUTINE, EXPEDITED, STAT (a queue priority, not a diagnosis) -- got "URGENT""); a diagnosis-shaped string written into URGENCY is refused by the same check; and an explicit DIAGNOSIS=... field is refused outright ("unexpected extra field(s) present (expected exactly FINDING, LOCATION, URGENCY, CONFIDENCE)") -- there is no path for a diagnosis to survive parsing as structured data

-- Test 3: an out-of-range confidence and a missing field are both refused --
  an out-of-range confidence (1.5) and a missing CONFIDENCE field are both correctly refused

-- Test 4: a study's overall priority is its SINGLE MOST URGENT finding, never an average --
  a study with two ROUTINE findings and one STAT finding correctly reports an overall priority of STAT (the single most urgent tier present, not an average of the three); a study with no findings at all correctly defaults to ROUTINE

-- Test 5: the mandatory disclaimer is unconditional, identical, and unomittable across reports --
  two reports with completely different findings still carry the exact same, byte-identical disclaimer -- build_triage_report offers no parameter capable of changing or omitting it

28/28 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating 'ask the model not to diagnose' as sufficient on its own"
    A system prompt that tells a vision-language model not to diagnose is a real, worthwhile instruction, and `render_triage_prompt` states exactly that instruction directly in its own rendered text. But an instruction inside a prompt is advisory in exactly the same sense Section 20.1's AI-suggested priority is advisory -- nothing about a model FOLLOWING that instruction is structurally guaranteed the way `TriageCase`'s own sign-off gate is. This section's own real answer is not a better-worded instruction; it is a report schema `parse_finding_line` enforces at the PARSING layer, entirely independent of whether the underlying model happened to follow the prompt's own request. A diagnosis-shaped string written into the URGENCY field, or appended as an extra field entirely, is refused by a strict structural check, not by hoping the model read the instruction carefully -- exactly the same shift from "ask nicely" to "structurally cannot" that Section 20.1 already made for human sign-off.

## 20.4 PACS Worklist Integration with a Human-in-the-Loop Queue

### Intuition

Section 20.1 made one case's own human sign-off structural. A real PACS worklist has to enforce a second, equally real property across an entire QUEUE of cases at once: when two radiologists are both looking at the same worklist, at most one of them may actually be working a given case, and once someone signs a case off, nothing may silently overwrite that decision.

### The Concept, In Detail

This section states its own scope the same way Chapter 18.5 stated OPC UA's: it does not model any real PACS vendor's actual wire protocol, and instead builds the one piece worth building from scratch -- a worklist whose claim-and-sign-off discipline is enforced by the type itself. `WorklistEntry::claim` is a real compare-and-set on `state_ == UNASSIGNED`, reusing exactly the optimistic-concurrency idea Section 19.3's `put_stock_level` already applied to an inventory record, now protecting which human is accountable for a study instead of a warehouse quantity. Test 2 confirms this directly: a second radiologist's claim on an already-claimed accession number is refused, and the refusal names the FIRST radiologist by identity rather than silently overwriting the claim or failing generically.

Every transition after the initial claim requires not only the correct state but the SAME radiologist who is already accountable for the case -- Test 3 confirms a second radiologist can neither begin review nor finalize a case claimed by someone else, and that `finalize` separately refuses an empty report summary even from the correctly-accountable radiologist, for the same reason Section 20.1 refused an empty `clinician_id`. Test 4 confirms a `FINALIZED` case accepts no further transition of any kind -- not a reclaim, not a re-review, not a re-finalize, and not a rejection -- leaving its original report completely untouched, while a separately-rejected case correctly records its own rejection reason. `pending_by_priority` orders the still-unclaimed queue with `std::stable_sort`, specifically so that two `STAT` studies added minutes apart are never silently reordered on a later view of the same queue -- Test 1 confirms both the priority ordering and the FIFO tie-break within a single priority tier directly.

### Code and Verification

```cpp
// Chapter 20.4 -- Section 20.1 built a state machine that makes a single
// case's own human sign-off structural rather than advisory. A real
// picture archiving and communication system (PACS) worklist has to
// enforce a second, equally real property across an entire QUEUE of
// cases at once: when two radiologists are both looking at the same
// worklist, at most one of them may actually be working a given case,
// and once someone signs a case off, nothing -- not a network retry, not
// a second reviewer clicking "claim" a moment too late -- may silently
// stomp on that decision. Section 19.3's `put_stock_level` already
// solved a version of this for warehouse inventory with a real
// optimistic-concurrency check (`expected_version`, a real HTTP 409 on a
// stale write); this section reuses exactly that idea, but the
// resource being protected from a lost update is not an inventory
// count, it is which human is accountable for a patient's imaging study.
//
// A note on scope, stated as directly as every other section in this
// chapter states it: this is NOT legal or regulatory advice, this
// section does not model any real PACS vendor's actual wire protocol
// (DICOM's real Modality Worklist and Structured Report services are
// far larger than what is useful to build from scratch here), and
// nothing in this file should be read as a claim that a real clinical
// deployment could be built from this interface alone. What this
// section builds is the one piece worth building from scratch: a
// worklist whose claim-and-sign-off discipline is enforced by the type
// itself, exactly like Section 20.1's `TriageCase`, extended across a
// whole queue of cases with real optimistic concurrency and a real,
// append-only audit trail per case.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_pacs_worklist_integration.cpp -o 04_pacs_worklist_integration
// Run:     ./04_pacs_worklist_integration

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: queue priority. Repeated from Section 20.3's own definition --
// every file in this book stands alone -- with the same meaning: a
// worklist ordering hint, never a diagnosis, never averaged.
// =======================================================================
enum class Urgency { ROUTINE, EXPEDITED, STAT };

std::string to_string(Urgency u) {
    switch (u) {
        case Urgency::ROUTINE: return "ROUTINE";
        case Urgency::EXPEDITED: return "EXPEDITED";
        case Urgency::STAT: return "STAT";
    }
    return "UNKNOWN";
}

int urgency_rank(Urgency u) { return static_cast<int>(u); }

// =======================================================================
// PART 2: the per-case state machine.
// =======================================================================
enum class WorklistState { UNASSIGNED, ASSIGNED, IN_REVIEW, FINALIZED, REJECTED };

std::string to_string(WorklistState s) {
    switch (s) {
        case WorklistState::UNASSIGNED: return "UNASSIGNED";
        case WorklistState::ASSIGNED: return "ASSIGNED";
        case WorklistState::IN_REVIEW: return "IN_REVIEW";
        case WorklistState::FINALIZED: return "FINALIZED";
        case WorklistState::REJECTED: return "REJECTED";
    }
    return "UNKNOWN";
}

struct AuditEntry {
    int tick = 0;
    std::string actor;
    std::string action;
};

// `WorklistEntry` is one study sitting in the queue. Every field that
// matters for accountability -- who claimed it, what state it is in,
// the full history of who did what and when -- is private, and every
// transition is guarded by two checks together: the state must be the
// one, specific state that transition is allowed from, AND (for every
// transition after the initial claim) the caller must be the SAME
// radiologist who is already accountable for the case. A second
// radiologist cannot begin reviewing, finalize, or reject a case they
// never claimed, even if the first radiologist simply forgot to finish
// it -- exactly the kind of silent hand-off this section exists to rule
// out structurally rather than leave to policy.
class WorklistEntry {
public:
    struct Result { bool ok = false; std::string error; };

    WorklistEntry(std::string accession_number, std::string study_id, Urgency priority)
        : accession_number_(std::move(accession_number)),
          study_id_(std::move(study_id)),
          priority_(priority) {
        log(0, "system", "added to worklist, priority " + to_string(priority_));
    }

    // The one operation with real optimistic concurrency: it is a
    // compare-and-set on `state_ == UNASSIGNED`, not a blind write. Two
    // radiologists racing to claim the same accession number will not
    // both succeed -- the first call to reach this check wins, and the
    // second is refused with an explicit conflict message that names
    // who already holds the case, exactly as Section 19.3's PUT refused
    // a stale write with a named current version rather than silently
    // overwriting it.
    Result claim(const std::string& radiologist_id, int tick) {
        if (radiologist_id.empty()) {
            return fail("claim requires a non-empty radiologist_id");
        }
        if (state_ != WorklistState::UNASSIGNED) {
            return fail("conflict: accession " + accession_number_ + " is already " +
                         to_string(state_) + (claimed_by_ ? (" (claimed by \"" + *claimed_by_ + "\")") : ""));
        }
        claimed_by_ = radiologist_id;
        state_ = WorklistState::ASSIGNED;
        log(tick, radiologist_id, "claimed");
        return ok();
    }

    Result begin_review(const std::string& radiologist_id, int tick) {
        if (state_ != WorklistState::ASSIGNED) {
            return fail("begin_review requires state ASSIGNED, current state is " + to_string(state_));
        }
        if (!claimed_by_ || radiologist_id != *claimed_by_) {
            return fail("begin_review requires the radiologist who claimed the case (\"" +
                         claimed_by_.value_or("<none>") + "\"), not \"" + radiologist_id + "\"");
        }
        state_ = WorklistState::IN_REVIEW;
        log(tick, radiologist_id, "began review");
        return ok();
    }

    // The only path to FINALIZED. Requires IN_REVIEW, requires the same
    // radiologist who has been accountable for the case since `claim`,
    // and requires a real, non-empty report summary -- an empty summary
    // is refused just as loudly as the wrong reviewer, for the same
    // reason Section 20.1 refused an empty clinician_id: a finalized
    // case with nothing recorded about what was decided is not really
    // finalized.
    Result finalize(const std::string& radiologist_id, int tick, const std::string& report_summary) {
        if (state_ != WorklistState::IN_REVIEW) {
            return fail("finalize requires state IN_REVIEW, current state is " + to_string(state_));
        }
        if (!claimed_by_ || radiologist_id != *claimed_by_) {
            return fail("finalize requires the radiologist who claimed the case (\"" +
                         claimed_by_.value_or("<none>") + "\"), not \"" + radiologist_id + "\"");
        }
        if (report_summary.empty()) {
            return fail("finalize requires a non-empty report_summary");
        }
        report_summary_ = report_summary;
        state_ = WorklistState::FINALIZED;
        log(tick, radiologist_id, "FINALIZED: " + report_summary);
        return ok();
    }

    // A radiologist declining to read a case they claimed is a real,
    // expected outcome (equipment mismatch, wrong study routed, needs a
    // subspecialist) -- allowed from either ASSIGNED or IN_REVIEW, never
    // from FINALIZED, and always recorded by name and reason.
    Result reject(const std::string& radiologist_id, int tick, const std::string& reason) {
        if (state_ != WorklistState::ASSIGNED && state_ != WorklistState::IN_REVIEW) {
            return fail("reject requires state ASSIGNED or IN_REVIEW, current state is " + to_string(state_));
        }
        if (!claimed_by_ || radiologist_id != *claimed_by_) {
            return fail("reject requires the radiologist who claimed the case (\"" +
                         claimed_by_.value_or("<none>") + "\"), not \"" + radiologist_id + "\"");
        }
        state_ = WorklistState::REJECTED;
        log(tick, radiologist_id, "rejected: " + reason);
        return ok();
    }

    const std::string& accession_number() const { return accession_number_; }
    const std::string& study_id() const { return study_id_; }
    Urgency priority() const { return priority_; }
    WorklistState state() const { return state_; }
    const std::optional<std::string>& claimed_by() const { return claimed_by_; }
    const std::optional<std::string>& report_summary() const { return report_summary_; }
    const std::vector<AuditEntry>& audit_log() const { return audit_log_; }

private:
    Result fail(std::string error) { return Result{false, std::move(error)}; }
    Result ok() { return Result{true, ""}; }
    void log(int tick, std::string actor, std::string action) {
        audit_log_.push_back(AuditEntry{tick, std::move(actor), std::move(action)});
    }

    std::string accession_number_;
    std::string study_id_;
    Urgency priority_;
    WorklistState state_ = WorklistState::UNASSIGNED;
    std::optional<std::string> claimed_by_;
    std::optional<std::string> report_summary_;
    std::vector<AuditEntry> audit_log_;
};

// =======================================================================
// PART 3: the worklist itself -- a queue of `WorklistEntry` objects,
// insertion-ordered, with a priority view over the still-pending ones.
// =======================================================================
class PacsWorklist {
public:
    struct Result { bool ok = false; std::string error; };

    Result add_case(const std::string& accession_number, const std::string& study_id, Urgency priority) {
        if (accession_number.empty()) {
            return {false, "add_case requires a non-empty accession_number"};
        }
        if (find_index(accession_number) != -1) {
            return {false, "conflict: accession " + accession_number + " already exists on this worklist"};
        }
        entries_.emplace_back(accession_number, study_id, priority);
        return {true, ""};
    }

    WorklistEntry* find(const std::string& accession_number) {
        int idx = find_index(accession_number);
        return idx == -1 ? nullptr : &entries_[static_cast<size_t>(idx)];
    }

    // Pending cases (still UNASSIGNED), ordered STAT first, then
    // EXPEDITED, then ROUTINE -- and, within a single priority tier, in
    // the same order they were added. `std::stable_sort` is not an
    // arbitrary choice here: without it, two STAT studies added minutes
    // apart could be reordered arbitrarily every time this view is
    // recomputed, which is exactly the kind of queue-fairness bug a real
    // worklist cannot tolerate.
    std::vector<std::string> pending_by_priority() const {
        std::vector<const WorklistEntry*> pending;
        for (const auto& e : entries_) {
            if (e.state() == WorklistState::UNASSIGNED) pending.push_back(&e);
        }
        std::stable_sort(pending.begin(), pending.end(), [](const WorklistEntry* a, const WorklistEntry* b) {
            return urgency_rank(a->priority()) > urgency_rank(b->priority());
        });
        std::vector<std::string> result;
        result.reserve(pending.size());
        for (const auto* e : pending) result.push_back(e->accession_number());
        return result;
    }

    size_t size() const { return entries_.size(); }

private:
    int find_index(const std::string& accession_number) const {
        for (size_t i = 0; i < entries_.size(); i++) {
            if (entries_[i].accession_number() == accession_number) return static_cast<int>(i);
        }
        return -1;
    }

    std::vector<WorklistEntry> entries_;
};

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.4: PACS Worklist Integration with a Human-in-the-Loop Queue\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: pending cases surface STAT first, then EXPEDITED, then ROUTINE, "
                 "FIFO within a tier --\n";
    {
        PacsWorklist wl;
        auto r1 = wl.add_case("ACC-0001", "STUDY-CHEST-CT", Urgency::ROUTINE);
        auto r2 = wl.add_case("ACC-0002", "STUDY-HEAD-CT", Urgency::STAT);
        auto r3 = wl.add_case("ACC-0003", "STUDY-ABD-XR", Urgency::EXPEDITED);
        auto r4 = wl.add_case("ACC-0004", "STUDY-HEAD-CT-2", Urgency::STAT);
        auto dup = wl.add_case("ACC-0001", "STUDY-DUP", Urgency::STAT);
        CHECK(r1.ok && r2.ok && r3.ok && r4.ok);
        CHECK(!dup.ok);
        CHECK(wl.size() == 4);
        auto order = wl.pending_by_priority();
        std::vector<std::string> expected = {"ACC-0002", "ACC-0004", "ACC-0003", "ACC-0001"};
        CHECK(order == expected);
        std::cout << "  4 cases added (duplicate accession correctly refused); pending order is "
                     "ACC-0002, ACC-0004 (both STAT, insertion order preserved), ACC-0003 "
                     "(EXPEDITED), ACC-0001 (ROUTINE)\n";
    }

    std::cout << "\n-- Test 2: claiming is real optimistic concurrency -- the first radiologist "
                 "to claim a case wins, and a second claim is refused by name, not silently "
                 "overwritten --\n";
    {
        PacsWorklist wl;
        wl.add_case("ACC-0100", "STUDY-CT", Urgency::STAT);
        auto* e = wl.find("ACC-0100");
        auto first = e->claim("dr-alvarez", 1);
        auto second = e->claim("dr-nakamura", 2);
        auto empty_id = e->claim("", 3);
        CHECK(first.ok);
        CHECK(!second.ok);
        CHECK(second.error.find("dr-alvarez") != std::string::npos);
        CHECK(!empty_id.ok);
        CHECK(e->state() == WorklistState::ASSIGNED);
        CHECK(e->claimed_by().value() == "dr-alvarez");
        std::cout << "  dr-alvarez's claim succeeds; dr-nakamura's later claim on the same "
                     "accession is refused and the refusal names dr-alvarez as the current "
                     "holder; an empty radiologist_id is refused outright\n";
    }

    std::cout << "\n-- Test 3: only the radiologist who claimed a case may advance or finalize "
                 "it, and finalize refuses an empty report --\n";
    {
        PacsWorklist wl;
        wl.add_case("ACC-0200", "STUDY-MR", Urgency::EXPEDITED);
        auto* e = wl.find("ACC-0200");
        e->claim("dr-alvarez", 1);
        auto wrong_reviewer = e->begin_review("dr-nakamura", 2);
        auto right_reviewer = e->begin_review("dr-alvarez", 3);
        auto empty_report = e->finalize("dr-alvarez", 4, "");
        auto wrong_finalizer = e->finalize("dr-nakamura", 5, "no acute findings");
        auto real_finalize = e->finalize("dr-alvarez", 6, "no acute findings, follow-up in 6 months");
        CHECK(!wrong_reviewer.ok);
        CHECK(right_reviewer.ok);
        CHECK(!empty_report.ok);
        CHECK(!wrong_finalizer.ok);
        CHECK(real_finalize.ok);
        CHECK(e->state() == WorklistState::FINALIZED);
        CHECK(e->report_summary().value() == "no acute findings, follow-up in 6 months");
        std::cout << "  dr-nakamura cannot begin review or finalize a case claimed by "
                     "dr-alvarez; an empty report summary is refused even from the correct "
                     "reviewer; the real finalize by dr-alvarez succeeds\n";
    }

    std::cout << "\n-- Test 4: a finalized case accepts no further transitions of any kind, and "
                 "rejection is a real, distinct, always-attributed outcome --\n";
    {
        PacsWorklist wl;
        wl.add_case("ACC-0300", "STUDY-CT", Urgency::ROUTINE);
        auto* e1 = wl.find("ACC-0300");
        e1->claim("dr-alvarez", 1);
        e1->begin_review("dr-alvarez", 2);
        e1->finalize("dr-alvarez", 3, "no acute findings");
        auto reclaim = e1->claim("dr-nakamura", 4);
        auto rereview = e1->begin_review("dr-alvarez", 5);
        auto refinalize = e1->finalize("dr-alvarez", 6, "different finding entirely");
        auto reject_after_finalize = e1->reject("dr-alvarez", 7, "changed my mind");
        CHECK(!reclaim.ok && !rereview.ok && !refinalize.ok && !reject_after_finalize.ok);
        CHECK(e1->state() == WorklistState::FINALIZED);
        CHECK(e1->report_summary().value() == "no acute findings");

        wl.add_case("ACC-0301", "STUDY-XR", Urgency::STAT);
        auto* e2 = wl.find("ACC-0301");
        e2->claim("dr-nakamura", 10);
        auto rejection = e2->reject("dr-nakamura", 11, "wrong modality routed, needs re-acquisition");
        CHECK(rejection.ok);
        CHECK(e2->state() == WorklistState::REJECTED);
        std::cout << "  every attempt to reclaim, re-review, re-finalize, or reject a already-"
                     "FINALIZED case is refused and its original report is untouched; a "
                     "separate case is correctly moved to REJECTED with its reason recorded\n";
    }

    std::cout << "\n-- Test 5: the per-case audit trail is complete, ordered, and attributed --\n";
    {
        PacsWorklist wl;
        wl.add_case("ACC-0400", "STUDY-CT", Urgency::EXPEDITED);
        auto* e = wl.find("ACC-0400");
        e->claim("dr-alvarez", 5);
        e->begin_review("dr-alvarez", 6);
        e->finalize("dr-alvarez", 7, "no acute findings");
        const auto& log = e->audit_log();
        CHECK(log.size() == 4);
        CHECK(log[0].actor == "system" && log[0].action.find("added to worklist") != std::string::npos);
        CHECK(log[1].tick == 5 && log[1].actor == "dr-alvarez" && log[1].action == "claimed");
        CHECK(log[2].tick == 6 && log[2].actor == "dr-alvarez" && log[2].action == "began review");
        CHECK(log[3].tick == 7 && log[3].actor == "dr-alvarez" &&
              log[3].action.find("FINALIZED") != std::string::npos);
        std::cout << "  the audit trail for ACC-0400 records all 4 events in order, each with "
                     "the correct tick and actor: system add, dr-alvarez's claim, dr-alvarez's "
                     "review start, and dr-alvarez's finalization\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    if (g_passed == g_tests) {
        std::cout << "ALL CHECKS PASSED\n";
        return 0;
    }
    std::cout << "SOME CHECKS FAILED\n";
    return 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_pacs_worklist_integration.cpp -o 04_pacs_worklist_integration
./04_pacs_worklist_integration
```

**Sample input:** four cases at three different priorities checked to surface in STAT-then-EXPEDITED-then-ROUTINE order with FIFO tie-breaking within a tier, alongside a duplicate accession number checked to be refused; a first claim checked to succeed and a second claim on the same accession checked to be refused by naming the current holder, alongside an empty radiologist id checked to be refused; a case claimed by one radiologist checked to refuse review and finalization attempts from a different radiologist, and to refuse an empty report summary even from the correct one; a finalized case checked to refuse every further transition while a separate case's rejection is correctly recorded; and a full claim-review-finalize sequence checked against its exact expected 4-entry audit trail.

```text
========================================================
Chapter 20.4: PACS Worklist Integration with a Human-in-the-Loop Queue
========================================================

-- Test 1: pending cases surface STAT first, then EXPEDITED, then ROUTINE, FIFO within a tier --
  4 cases added (duplicate accession correctly refused); pending order is ACC-0002, ACC-0004 (both STAT, insertion order preserved), ACC-0003 (EXPEDITED), ACC-0001 (ROUTINE)

-- Test 2: claiming is real optimistic concurrency -- the first radiologist to claim a case wins, and a second claim is refused by name, not silently overwritten --
  dr-alvarez's claim succeeds; dr-nakamura's later claim on the same accession is refused and the refusal names dr-alvarez as the current holder; an empty radiologist_id is refused outright

-- Test 3: only the radiologist who claimed a case may advance or finalize it, and finalize refuses an empty report --
  dr-nakamura cannot begin review or finalize a case claimed by dr-alvarez; an empty report summary is refused even from the correct reviewer; the real finalize by dr-alvarez succeeds

-- Test 4: a finalized case accepts no further transitions of any kind, and rejection is a real, distinct, always-attributed outcome --
  every attempt to reclaim, re-review, re-finalize, or reject a already-FINALIZED case is refused and its original report is untouched; a separate case is correctly moved to REJECTED with its reason recorded

-- Test 5: the per-case audit trail is complete, ordered, and attributed --
  the audit trail for ACC-0400 records all 4 events in order, each with the correct tick and actor: system add, dr-alvarez's claim, dr-alvarez's review start, and dr-alvarez's finalization

27/27 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] letting 'claimed' alone stand in for 'accountable'"
    It is tempting to treat `claim` as the whole of this section's own safety property -- once a radiologist has claimed a case, the thinking goes, the hard part (preventing a double-claim) is done. Test 3 exists specifically because that is not the whole property: a SECOND radiologist could, in a system that checked only the current state and not WHO is claiming it, still begin reviewing or even finalize a case someone else claimed, simply by knowing its accession number and finding it sitting in `ASSIGNED` or `IN_REVIEW`. Every one of `begin_review`, `finalize`, and `reject` in this section separately checks `radiologist_id == *claimed_by_`, not merely the state -- accountability, once established by a successful claim, has to be checked again at every single subsequent transition, not assumed to still hold just because the state looks right.

## 20.5 Occlusion-Based Saliency and the Honest Limits of Explainability

### Intuition

Every earlier section in this chapter built a real structural guarantee about WHO may act and WHEN. None of that answers a different, harder question a real deployment will be asked: why did the model suggest this study was urgent at all? This section builds one real, honest answer, and then builds the proof of exactly what that answer cannot tell you.

### The Concept, In Detail

The technique itself reuses Chapter 18.2's own vision encoder completely unmodified -- Parts 1 through 5 of this section's own file are byte-for-byte the same patchification, 2D RoPE, bidirectional ViT blocks, and 2x2 merger Chapter 18.2 already verified, because a saliency score computed against a DIFFERENT network would not actually mean anything about the one this chapter cares about. `occlusion_saliency_map` masks one patch at a time -- replacing it with the dataset's own per-channel MEAN color, which Test 5 confirms is exactly the zero vector after Part 1's own normalization, deliberately never black, since a black patch would introduce an artificial dark edge that is itself a new signal the model could react to -- reruns the frozen encoder, and measures how far the resulting whole-image representation moved via a plain L2 distance. Test 3 confirms this genuinely localizes WHERE the model is sensitive to input content: a single deliberately anomalous patch in an otherwise-uniform synthetic image scores strictly higher than every one of the fifteen ordinary background patches.

Test 4 is this section's own most important result, built specifically to prove the technique's real limit by direct computation rather than by assertion: two DIFFERENT anomalous patches, occluded separately from the same baseline image, move the whole-image representation by a comparable magnitude (a ratio of roughly 2x between the two scores, checked directly to stay under a stated factor of 5) -- but their own displacement VECTORS, compared by cosine similarity, point in substantially different directions in the model's own hidden space (measured well below a stated 0.9 similarity threshold). A scalar saliency score answers "how much did the representation move," and this section's own numbers prove directly that two very different underlying causes can produce a similar-looking answer to that question while meaning something completely different about WHAT changed and WHY -- which is exactly the honest limit this section states in its own opening comment and then proves computationally rather than merely asserting.

### Code and Verification

```cpp
// Chapter 20.5 -- Every earlier section in this chapter built a real
// structural guarantee: Section 20.1 made human sign-off unavoidable,
// Section 20.4 made the same true across a whole worklist queue. None of
// that answers a different, harder question a real deployment will be
// asked: WHY did the model suggest this study was urgent at all? This
// section builds one real, honest answer -- occlusion-based saliency,
// reusing Chapter 18.2's own vision encoder frozen exactly as written --
// and then, just as carefully, builds the proof of what that answer
// cannot tell you.
//
// The technique itself is genuinely simple and genuinely real: mask one
// patch out of the image, rerun the SAME frozen encoder, and measure how
// far the resulting whole-image representation moved. A patch whose
// removal barely moves that representation was not doing much work in
// the model's own computation; a patch whose removal moves it a long
// way was. That is a real, computable fact about this specific frozen
// network's own sensitivity to its own input -- not a guess, not a
// post-hoc story invented to sound plausible.
//
// What it is NOT, and what this section states as plainly as every
// other limitation this book has ever stated: a displacement score
// tells you THAT the representation moved and roughly how far, never
// WHY, never in what SEMANTIC direction, and never whether the model's
// own suggested priority was moved for a clinically sound reason or a
// spurious one. Test 4 below builds a direct, computable proof of
// exactly that gap: two differently-located anomalies can move the
// representation by comparable amounts while moving it in almost
// entirely different directions in the model's own hidden space -- proof
// that the scalar score alone cannot distinguish "the same kind of
// shift happened twice" from "two completely different things happened
// to look similarly important." A saliency map is a real, useful WHERE.
// It is never a WHY, and this section builds no code that pretends
// otherwise.
//
// A note on scope, in this chapter's own recurring voice: this is NOT
// legal or regulatory advice, and nothing here claims that occlusion
// saliency (or any other explainability technique) satisfies any
// specific jurisdiction's transparency or explainability requirement for
// software assisting a clinical decision -- that determination needs
// real regulatory and legal review this book cannot substitute for.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off 05_occlusion_saliency_and_explainability_limits.cpp -o 05_occlusion_saliency_and_explainability_limits
// Run:     ./05_occlusion_saliency_and_explainability_limits

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: raw-buffer preprocessing. Repeated verbatim from Chapter
// 18.2's own Section 18.2 -- this file's own encoder must be BYTE-FOR-
// BYTE the same frozen network Section 18.2 already verified, or a
// saliency score computed against it would not actually mean anything.
// =======================================================================
struct RawImage {
    uint32_t width = 0, height = 0, channels = 0;
    std::vector<uint8_t> pixels;   // row-major, HxWxC, matching FrameResult::pixels exactly
};

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
// PART 2: this book's own RMSNorm/matmul/SwiGLU, repeated verbatim.
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
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
}

// =======================================================================
// PART 3: two-dimensional rotary position encoding, repeated verbatim.
// =======================================================================
struct RoPE2DTables {
    std::vector<float> cos_row, sin_row, cos_col, sin_col;
    int quarter_dim;
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
void apply_rope2d(std::span<float> head_vec, int row, int col, const RoPE2DTables& t) {
    const int half = static_cast<int>(head_vec.size()) / 2;
    rotate_half_inplace(head_vec.subspan(0, static_cast<size_t>(half)), t.cos_row, t.sin_row, row, t.quarter_dim);
    rotate_half_inplace(head_vec.subspan(static_cast<size_t>(half), static_cast<size_t>(half)), t.cos_col, t.sin_col, col, t.quarter_dim);
}

// =======================================================================
// PART 4: the vision transformer block itself, repeated verbatim.
// =======================================================================
struct ViTShape { int dim, n_heads, head_dim, d_ff; int qkv_dim() const { return n_heads * head_dim; } };
struct ViTBlockWeights {
    std::vector<float> attn_norm, Wq, Wk, Wv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};

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
// PART 5: the 2x2 spatial patch merger, repeated verbatim.
// =======================================================================
struct MergerWeights { std::vector<float> W1, b1, W2, b2; int hidden_dim; };

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
// PART 6: occlusion-based saliency, built on top of the frozen PARTS
// 1-5 above with no changes to any of them.
// =======================================================================

// A patch is "occluded" by replacing it with the per-channel MEAN color
// -- which, after PART 1's own normalization, is EXACTLY the zero vector
// (v == mean => (v - mean) / std == 0). Occluding with black instead
// would introduce a large, artificial dark-patch edge of its own, which
// is itself a strong signal the model could react to; occluding with the
// dataset's own mean is the smallest, most neutral edit this section can
// make to remove a patch's content without inserting a new, unrelated
// one in its place.
std::vector<float> mean_occluded_patch(size_t patch_vec_len) {
    return std::vector<float>(patch_vec_len, 0.0f);
}

struct EncoderWeights {
    std::vector<float> W_embed;
    std::vector<ViTBlockWeights> layers;
    MergerWeights merger;
    int vit_dim = 0, n_heads = 0, head_dim = 0, d_ff = 0, llm_dim = 0;
};

// Mean-pools the merged visual tokens down to ONE fixed-length vector: a
// single whole-image representation whose length never changes no
// matter which (if any) patch was occluded, so any two runs' outputs can
// always be compared with plain L2 distance.
std::vector<float> mean_pool(const std::vector<std::vector<float>>& merged) {
    const size_t dim = merged.empty() ? 0 : merged[0].size();
    std::vector<float> out(dim, 0.0f);
    for (const auto& tok : merged) for (size_t i = 0; i < dim; ++i) out[i] += tok[i];
    if (!merged.empty()) for (float& v : out) v /= static_cast<float>(merged.size());
    return out;
}

double l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq);
}

// Runs the frozen PART 1-5 pipeline end to end on an already-patchified
// image, with `occluded_patch` (if set) replaced by the neutral
// mean-color vector before embedding, and returns the whole-image
// mean-pooled representation.
std::vector<float> run_encoder_pooled(const std::vector<std::vector<float>>& raw_patches, uint32_t grid_h,
                                       uint32_t grid_w, const EncoderWeights& ew, const RoPE2DTables& rope,
                                       std::optional<int> occluded_patch) {
    const int n_patches = static_cast<int>(raw_patches.size());
    const size_t patch_vec_len = raw_patches[0].size();
    std::vector<std::vector<float>> x(static_cast<size_t>(n_patches));
    std::vector<int> rows(static_cast<size_t>(n_patches)), cols(static_cast<size_t>(n_patches));
    for (int i = 0; i < n_patches; ++i) {
        std::vector<float> occluded_buf;
        const std::vector<float>* src = &raw_patches[static_cast<size_t>(i)];
        if (occluded_patch && *occluded_patch == i) {
            occluded_buf = mean_occluded_patch(patch_vec_len);
            src = &occluded_buf;
        }
        x[static_cast<size_t>(i)].resize(static_cast<size_t>(ew.vit_dim));
        matmul(x[static_cast<size_t>(i)], *src, ew.W_embed, patch_vec_len, static_cast<size_t>(ew.vit_dim));
        rows[static_cast<size_t>(i)] = i / static_cast<int>(grid_w);
        cols[static_cast<size_t>(i)] = i % static_cast<int>(grid_w);
    }
    ViTShape shape{ew.vit_dim, ew.n_heads, ew.head_dim, ew.d_ff};
    for (const auto& l : ew.layers) vit_block_forward(x, shape, l, rows, cols, rope);
    auto merged = merge_all_2x2(x, static_cast<int>(grid_h), static_cast<int>(grid_w), ew.vit_dim, ew.merger, ew.llm_dim);
    return mean_pool(merged);
}

// The saliency map itself: one non-negative displacement score per
// patch. score[i] == l2_distance(baseline, output-with-patch-i-
// occluded) -- large where removing that patch moved the whole-image
// representation a long way, near zero where it barely moved it at all.
std::vector<double> occlusion_saliency_map(const std::vector<std::vector<float>>& raw_patches, uint32_t grid_h,
                                            uint32_t grid_w, const EncoderWeights& ew, const RoPE2DTables& rope) {
    auto baseline = run_encoder_pooled(raw_patches, grid_h, grid_w, ew, rope, std::nullopt);
    std::vector<double> scores(raw_patches.size());
    for (size_t i = 0; i < raw_patches.size(); ++i) {
        auto occluded_output = run_encoder_pooled(raw_patches, grid_h, grid_w, ew, rope, static_cast<int>(i));
        scores[i] = l2_distance(baseline, occluded_output);
    }
    return scores;
}

// The raw per-patch DISPLACEMENT VECTOR, kept separate from the scalar
// score above specifically so this section's own Test 4 can compare the
// DIRECTION two different occlusions move the representation in, not
// merely how far.
std::vector<float> occlusion_displacement_vector(const std::vector<float>& baseline,
                                                  const std::vector<float>& occluded_output) {
    std::vector<float> d(baseline.size());
    for (size_t i = 0; i < baseline.size(); ++i) d[i] = occluded_output[i] - baseline[i];
    return d;
}

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// =======================================================================
// PART 7: self-tests.
// =======================================================================
std::vector<float> rand_vec(std::mt19937& rng, size_t n) {
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

EncoderWeights build_weights(int vit_dim, int n_heads, int head_dim, int d_ff, int n_layers,
                              int merger_hidden, int llm_dim, size_t patch_vec_len) {
    EncoderWeights ew;
    ew.vit_dim = vit_dim; ew.n_heads = n_heads; ew.head_dim = head_dim; ew.d_ff = d_ff; ew.llm_dim = llm_dim;
    std::mt19937 w_rng(42);
    ew.W_embed = rand_vec(w_rng, static_cast<size_t>(vit_dim) * patch_vec_len);
    ew.layers.resize(static_cast<size_t>(n_layers));
    for (auto& l : ew.layers) {
        l.attn_norm.assign(static_cast<size_t>(vit_dim), 1.0f);
        l.Wq = rand_vec(w_rng, static_cast<size_t>(n_heads * head_dim) * vit_dim);
        l.Wk = rand_vec(w_rng, static_cast<size_t>(n_heads * head_dim) * vit_dim);
        l.Wv = rand_vec(w_rng, static_cast<size_t>(n_heads * head_dim) * vit_dim);
        l.Wo = rand_vec(w_rng, static_cast<size_t>(vit_dim) * (n_heads * head_dim));
        l.ffn_norm.assign(static_cast<size_t>(vit_dim), 1.0f);
        l.Wgate = rand_vec(w_rng, static_cast<size_t>(d_ff) * vit_dim);
        l.Wup = rand_vec(w_rng, static_cast<size_t>(d_ff) * vit_dim);
        l.Wdown = rand_vec(w_rng, static_cast<size_t>(vit_dim) * d_ff);
    }
    std::mt19937 m_rng(99);
    ew.merger.hidden_dim = merger_hidden;
    ew.merger.W1 = rand_vec(m_rng, static_cast<size_t>(merger_hidden) * (4 * vit_dim));
    ew.merger.b1 = rand_vec(m_rng, static_cast<size_t>(merger_hidden));
    ew.merger.W2 = rand_vec(m_rng, static_cast<size_t>(llm_dim) * merger_hidden);
    ew.merger.b2 = rand_vec(m_rng, static_cast<size_t>(llm_dim));
    return ew;
}

// Fills every patch of a 4x4-grid, 56x56 synthetic image with `bg_value`,
// then overwrites the patches at `anomaly_indices` with their paired
// values from `anomaly_values` -- letting the tests below build images
// with a controlled number of deliberately anomalous regions.
RawImage make_grid_image(uint32_t patch, uint32_t grid, uint8_t bg_value,
                          const std::vector<int>& anomaly_indices, const std::vector<uint8_t>& anomaly_values) {
    RawImage img;
    img.width = patch * grid; img.height = patch * grid; img.channels = 3;
    img.pixels.assign(static_cast<size_t>(img.width) * img.height * img.channels, bg_value);
    for (size_t a = 0; a < anomaly_indices.size(); ++a) {
        int idx = anomaly_indices[a];
        uint32_t pr = static_cast<uint32_t>(idx) / grid, pc = static_cast<uint32_t>(idx) % grid;
        for (uint32_t py = 0; py < patch; ++py) {
            uint32_t y = pr * patch + py;
            for (uint32_t px = 0; px < patch; ++px) {
                uint32_t x = pc * patch + px;
                uint8_t* p = &img.pixels[(static_cast<size_t>(y) * img.width + x) * img.channels];
                p[0] = p[1] = p[2] = anomaly_values[a];
            }
        }
    }
    return img;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 20.5: Occlusion-Based Saliency and the Honest Limits of Explainability\n";
    std::cout << "========================================================\n";

    constexpr uint32_t PATCH = 14, GRID = 4;
    constexpr int VIT_DIM = 24, N_HEADS = 2, HEAD_DIM = 8, D_FF = 32, N_LAYERS = 2;
    constexpr int LLM_DIM = 32, MERGER_HIDDEN = 40;

    NormStats norm;

    std::cout << "\n-- Test 1: the frozen encoder's whole-image representation is deterministic --\n";
    {
        RawImage img = make_grid_image(PATCH, GRID, 128, {}, {});
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(img, PATCH, norm, grid_h, grid_w);
        auto ew = build_weights(VIT_DIM, N_HEADS, HEAD_DIM, D_FF, N_LAYERS, MERGER_HIDDEN, LLM_DIM, patches[0].size());
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);
        auto out1 = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, std::nullopt);
        auto out2 = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, std::nullopt);
        CHECK(out1 == out2);
        CHECK(out1.size() == static_cast<size_t>(LLM_DIM));
        std::cout << "  two runs of the unmodified " << grid_h << "x" << grid_w
                   << "-patch image through the frozen encoder produce byte-identical "
                   << out1.size() << "-dim pooled representations\n";
    }

    std::cout << "\n-- Test 2: the saliency map has one non-negative score per patch, and is not "
                 "trivially all-zero --\n";
    {
        RawImage img = make_grid_image(PATCH, GRID, 100, {5, 9}, {30, 220});
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(img, PATCH, norm, grid_h, grid_w);
        auto ew = build_weights(VIT_DIM, N_HEADS, HEAD_DIM, D_FF, N_LAYERS, MERGER_HIDDEN, LLM_DIM, patches[0].size());
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);
        auto scores = occlusion_saliency_map(patches, grid_h, grid_w, ew, rope);
        CHECK(scores.size() == 16);
        bool all_non_negative = true, any_nontrivial = false;
        for (double s : scores) {
            if (s < 0.0) all_non_negative = false;
            if (s > 1e-6) any_nontrivial = true;
        }
        CHECK(all_non_negative);
        CHECK(any_nontrivial);
        std::cout << "  16 patches produce 16 non-negative displacement scores, and at least "
                     "one is clearly nonzero -- occlusion genuinely moves the representation\n";
    }

    std::cout << "\n-- Test 3: occluding a deliberately anomalous patch moves the representation "
                 "further than occluding an ordinary background patch --\n";
    {
        constexpr int ANOMALY_IDX = 6;   // row 1, col 2 of a 4x4 grid
        RawImage img = make_grid_image(PATCH, GRID, 128, {ANOMALY_IDX}, {12});
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(img, PATCH, norm, grid_h, grid_w);
        auto ew = build_weights(VIT_DIM, N_HEADS, HEAD_DIM, D_FF, N_LAYERS, MERGER_HIDDEN, LLM_DIM, patches[0].size());
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);
        auto scores = occlusion_saliency_map(patches, grid_h, grid_w, ew, rope);
        double max_background = 0.0;
        for (int i = 0; i < 16; ++i) if (i != ANOMALY_IDX) max_background = std::max(max_background, scores[static_cast<size_t>(i)]);
        CHECK(scores[static_cast<size_t>(ANOMALY_IDX)] > max_background);
        std::cout << "  the anomalous patch's own score (" << scores[static_cast<size_t>(ANOMALY_IDX)]
                   << ") exceeds the highest score among all 15 ordinary background patches ("
                   << max_background << ") -- occlusion saliency correctly localizes WHERE the "
                      "output is sensitive to input content\n";
    }

    std::cout << "\n-- Test 4: THE HONEST LIMIT -- two different anomalies can move the "
                 "representation by comparable magnitudes while moving it in substantially "
                 "different directions, so the scalar score alone cannot tell you WHY --\n";
    {
        constexpr int ANOMALY_A = 3, ANOMALY_B = 12;
        RawImage img = make_grid_image(PATCH, GRID, 128, {ANOMALY_A, ANOMALY_B}, {12, 230});
        uint32_t grid_h = 0, grid_w = 0;
        auto patches = patchify(img, PATCH, norm, grid_h, grid_w);
        auto ew = build_weights(VIT_DIM, N_HEADS, HEAD_DIM, D_FF, N_LAYERS, MERGER_HIDDEN, LLM_DIM, patches[0].size());
        RoPE2DTables rope(static_cast<int>(std::max(grid_h, grid_w)) + 1, HEAD_DIM, 10000.0f);

        auto baseline = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, std::nullopt);
        auto out_a = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, ANOMALY_A);
        auto out_b = run_encoder_pooled(patches, grid_h, grid_w, ew, rope, ANOMALY_B);
        double score_a = l2_distance(baseline, out_a);
        double score_b = l2_distance(baseline, out_b);
        auto disp_a = occlusion_displacement_vector(baseline, out_a);
        auto disp_b = occlusion_displacement_vector(baseline, out_b);
        double cos_sim = cosine_similarity(disp_a, disp_b);

        // "Comparable magnitude" is checked directly, not assumed: the
        // larger score is required to be within a factor of 5 of the
        // smaller one, so this test would fail loudly if the two
        // anomalies had turned out wildly mismatched in importance
        // rather than the deliberately similar disruption this section
        // built them to cause.
        double ratio = std::max(score_a, score_b) / std::min(score_a, score_b);
        CHECK(ratio < 5.0);
        // The actual claim this test exists to prove: direction is NOT
        // preserved just because magnitude is comparable. A cosine
        // similarity anywhere near 1.0 would mean both anomalies moved
        // the representation the same way, which would undermine the
        // entire point -- so this section checks it is well below that.
        CHECK(cos_sim < 0.9);
        std::cout << "  occluding patch " << ANOMALY_A << " moves the representation by "
                   << score_a << "; occluding patch " << ANOMALY_B << " moves it by " << score_b
                   << " (ratio " << ratio << ", comparable magnitude); but the two displacement "
                      "vectors have cosine similarity " << cos_sim << " -- pointing in "
                      "substantially different directions in the model's own hidden space. "
                      "Magnitude alone cannot tell these two, very different, causes apart\n";
    }

    std::cout << "\n-- Test 5: occlusion really does replace a patch with the dataset's own "
                 "per-channel mean color, not black --\n";
    {
        // mean_occluded_patch itself is exactly zero by construction --
        // checked directly here, not just asserted in a comment.
        auto occ = mean_occluded_patch(PATCH * PATCH * 3);
        bool all_exactly_zero = true;
        for (float v : occ) if (v != 0.0f) all_exactly_zero = false;
        CHECK(all_exactly_zero);

        // And a real patch filled with the ROUNDED mean pixel value in
        // every channel normalizes to (approximately) that same zero
        // vector -- proving "occlude with the mean" and "occlude with
        // mean_occluded_patch" are the same real edit, not two
        // unrelated ideas that happen to share a name.
        uint8_t mean_r = static_cast<uint8_t>(std::lround(norm.mean[0] * 255.0f));
        uint8_t mean_g = static_cast<uint8_t>(std::lround(norm.mean[1] * 255.0f));
        uint8_t mean_b = static_cast<uint8_t>(std::lround(norm.mean[2] * 255.0f));
        RawImage img; img.width = PATCH; img.height = PATCH; img.channels = 3;
        img.pixels.resize(static_cast<size_t>(PATCH) * PATCH * 3);
        for (size_t i = 0; i < img.pixels.size(); i += 3) {
            img.pixels[i] = mean_r; img.pixels[i + 1] = mean_g; img.pixels[i + 2] = mean_b;
        }
        auto mean_patch = extract_patch(img, 0, 0, PATCH, norm);
        bool all_near_zero = true;
        for (float v : mean_patch) if (std::fabs(v) > 0.01f) all_near_zero = false;
        CHECK(all_near_zero);
        std::cout << "  mean_occluded_patch() is exactly the zero vector, and a real patch "
                     "filled with the rounded per-channel mean pixel value normalizes to "
                     "within 0.01 of that same zero vector -- confirming occlusion removes a "
                     "patch's content by substituting the dataset's own neutral color, never "
                     "an artificial black edge\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off 05_occlusion_saliency_and_explainability_limits.cpp -o 05_occlusion_saliency_and_explainability_limits
./05_occlusion_saliency_and_explainability_limits
```

**Sample input:** an unmodified synthetic image run twice through the frozen encoder, checked to produce a byte-identical pooled representation; a 16-patch saliency map checked for correct length, non-negativity, and genuine non-triviality; a single deliberately anomalous patch checked to score strictly higher than the highest-scoring ordinary background patch; two differently-located anomalies checked to move the representation by a comparable magnitude (ratio under 5x) while their displacement vectors point in substantially different directions (cosine similarity under 0.9); and the occlusion baseline itself checked to be exactly the zero vector, matching a real patch filled with the rounded per-channel mean pixel value to within floating-point tolerance.

```text
========================================================
Chapter 20.5: Occlusion-Based Saliency and the Honest Limits of Explainability
========================================================

-- Test 1: the frozen encoder's whole-image representation is deterministic --
  two runs of the unmodified 4x4-patch image through the frozen encoder produce byte-identical 32-dim pooled representations

-- Test 2: the saliency map has one non-negative score per patch, and is not trivially all-zero --
  16 patches produce 16 non-negative displacement scores, and at least one is clearly nonzero -- occlusion genuinely moves the representation

-- Test 3: occluding a deliberately anomalous patch moves the representation further than occluding an ordinary background patch --
  the anomalous patch's own score (85.9369) exceeds the highest score among all 15 ordinary background patches (20.9086) -- occlusion saliency correctly localizes WHERE the output is sensitive to input content

-- Test 4: THE HONEST LIMIT -- two different anomalies can move the representation by comparable magnitudes while moving it in substantially different directions, so the scalar score alone cannot tell you WHY --
  occluding patch 3 moves the representation by 85.9904; occluding patch 12 moves it by 42.3685 (ratio 2.02958, comparable magnitude); but the two displacement vectors have cosine similarity 0.214527 -- pointing in substantially different directions in the model's own hidden space. Magnitude alone cannot tell these two, very different, causes apart

-- Test 5: occlusion really does replace a patch with the dataset's own per-channel mean color, not black --
  mean_occluded_patch() is exactly the zero vector, and a real patch filled with the rounded per-channel mean pixel value normalizes to within 0.01 of that same zero vector -- confirming occlusion removes a patch's content by substituting the dataset's own neutral color, never an artificial black edge

10/10 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] mistaking a saliency score's MAGNITUDE for an explanation of WHY"
    A saliency map that highlights "this region mattered most" is genuinely useful for deciding WHERE a radiologist's own attention might be best spent first, and this section's Test 3 proves that use is real. It is a substantively different, and considerably more dangerous, claim to treat two similarly-scored regions as having mattered for the SAME REASON, or to treat a high score as evidence the model's own suggestion was clinically well-founded rather than driven by an artifact the model happened to be sensitive to. Test 4's own two anomalies were constructed to be genuinely different causes -- different locations, different pixel values -- and still landed within a factor of 2 of each other in raw magnitude, while their real displacement vectors shared a cosine similarity of roughly 0.2, nowhere near the same direction. A displacement score is a real, computable WHERE. It was never a WHY, and this section builds no code, anywhere, that would let a caller mistake one for the other.

## Chapter Summary

This chapter took the human-in-the-loop discipline this book has treated carefully since Chapter 18.4's own threshold-based auto-disposition and rebuilt it, deliberately, as something considerably stronger: a STRUCTURAL guarantee a system is incapable of routing around, rather than a policy a caller has to remember to follow. Section 20.1 built `TriageCase`, whose only path to a final disposition runs through one gated function requiring both the correct state and a real, non-empty clinician identifier. Section 20.2 implemented a real, standard subset of the DICOM file format and the exact real DICOM linear VOI LUT windowing formula, verified against hand-computed values at every clip boundary and at the formula's own true midpoint, and found a genuine resource-lifetime bug -- an unflushed writer stream -- during its own construction. Section 20.3 built a triage-report schema and parser structurally incapable of representing a diagnosis, restricting a queue-priority field to three literal tokens and refusing any injected extra field outright, with one unconditional disclaimer attached by the only function capable of constructing a report at all. Section 20.4 extended that same sign-off discipline across an entire worklist queue using real optimistic concurrency, so that a claim on a case behaves exactly as safely under concurrent radiologists as Section 19.3's own inventory PUT behaves under concurrent writers. Section 20.5 closed the chapter with a real, honest explainability technique built on Chapter 18's own frozen vision encoder, and proved by direct computation -- not by assertion -- the specific, real limit of what a saliency score can and cannot tell a radiologist about why a model said what it said.

## Self-Check Questions

1. Section 20.1's `final_disposition_` field is private, reachable from exactly one function. Explain specifically why a documented POLICY requiring human sign-off before finalization would be a meaningfully weaker guarantee than what this section actually built, even if every caller in practice always followed that policy correctly.
2. Section 20.1's Test 2 checks that skipping EITHER `route_to_review` or the AI-triage step is refused, not only that skipping the final sign-off step is refused. Why does enforcing the full ordering matter, beyond simply gating the last step?
3. Section 20.2 needed an explicit `close()` method added to `DicomWriter` mid-construction. Explain the real bug this fixed, and why a `DicomReader` opened immediately after writing the same path could observe a zero-byte file despite the writer code appearing to have already written real data to it.
4. Section 20.2's windowing formula clips a raw sample to a display minimum or maximum outside a stated range, and maps linearly between. What would go visually wrong with a displayed image if a real implementation used a discontinuous (step) function at the clip boundaries instead of the exact linear VOI LUT formula this section implements?
5. Section 20.3's `parse_finding_line` refuses an injected `DIAGNOSIS=...` field with the same check that refuses a merely malformed line. Why is a strict field-COUNT check, rather than a check for specific forbidden field names, the more robust way to make that refusal general?
6. Section 20.3's `overall_priority` is computed as the single most urgent finding via a max operation, never an average. Construct a concrete two-finding example where averaging urgency ranks would produce a meaningfully worse (less safe) outcome than taking the max.
7. Section 20.4's `claim` alone is not sufficient to prevent an unaccountable radiologist from advancing a case, which is why `begin_review`, `finalize`, and `reject` each separately check `radiologist_id == *claimed_by_`. Why isn't checking the state alone (e.g., `state_ == ASSIGNED`) sufficient once a claim has already succeeded?
8. Section 20.4's `pending_by_priority` uses `std::stable_sort` rather than `std::sort`. Explain the specific, observable bug an ordinary unstable sort could introduce across two calls to this same function, using two STAT-priority cases as your example.
9. Section 20.5 occludes a patch with the dataset's own per-channel mean color rather than black. Explain specifically what a black-patch occlusion baseline would risk introducing that a mean-color occlusion does not.
10. Section 20.5's Test 4 checks BOTH that two anomalies' saliency scores are comparable in magnitude AND that their displacement vectors have low cosine similarity. Explain why checking only the magnitude comparison, without the cosine-similarity check, would fail to prove this section's own central claim about the limits of explainability.

## Where We Go Next

This chapter pushed this book's own human-in-the-loop discipline from a single gated function (Section 20.1) to an entire worklist queue (Section 20.4), and closed with an honest, computationally-proven account of what a real explainability technique can and cannot promise a human reviewer. The next several chapters return this book's own vision-language core to lower-stakes domains -- document intelligence, security and accessibility, trust and counterfeit detection, natural-language photo editing, and personal cameras -- where the same discipline this chapter built still applies, but where the cost of a wrong call is measured in inconvenience or expense rather than in the kind of stakes this chapter took the time to handle carefully. Chapter 21 takes this book's own vision-language pipeline into document intelligence: insurance claims, environmental compliance filings, and the real, structured extraction problem a page of dense, real-world paperwork poses to a model that has, until now, only ever looked at a shelf or a manufacturing line.

## Worked Solutions

**1.** A documented policy is only ever as reliable as the humans and processes that remember to follow it every single time, under every kind of pressure (a busy shift, an unfamiliar caller, a rushed integration) -- it can be forgotten, misread, or bypassed by a caller who simply calls a lower-level function directly. `TriageCase`'s own structural guarantee does not depend on anyone remembering anything: there is no second code path anywhere in the class, accidental or otherwise, that can ever set `final_disposition_` without passing through `record_human_signoff`'s own two checks. The difference is not about whether people are careful; it is about whether an uncareful moment -- a bug, a rushed integration, a caller who never read the policy at all -- is even CAPABLE of producing an unauthorized finalization, and with this section's own design, it categorically is not.

**2.** If only the final sign-off step were gated, a caller could still construct a case that reached `PENDING_HUMAN_REVIEW` through some other, unintended path -- skipping the AI-triage step entirely, for instance -- and a reviewer signing off on such a case would be attesting to having reviewed an AI suggestion that was never actually produced, corrupting the audit trail's own claim about what actually happened. Enforcing the FULL ordering means every field this book's own audit trail records (that an AI triage occurred, that it was explicitly routed for review, that a named clinician signed off) is guaranteed to correspond to something that genuinely happened in that order, not merely that the LAST step happened to be performed by a human.

**3.** The real bug was a resource-lifetime issue, not a logic error in the parsing code itself: `DicomWriter`'s member `std::ofstream` buffers its writes and only guarantees they reach the actual file on disk when the stream is flushed or closed, which normally happens automatically when the `ofstream`'s own destructor runs at the end of its scope. Constructing a `DicomReader` and calling `open()` on the SAME path while the `DicomWriter` object was still alive (and therefore its destructor had not yet run) meant the reader could observe the file exactly as the filesystem currently saw it -- often still zero bytes, with all of the writer's own buffered data not yet flushed to disk. The fix, an explicit `close()` method called before constructing the reader, makes the flush happen at a known point in the code rather than relying on destructor timing that the source code's own visual order does not guarantee.

**4.** A discontinuous step function at either clip boundary would create a visible, artificial hard edge in the displayed image at exactly the raw sample value where the step occurs -- two adjacent pixels differing in raw value by only 1 unit, straddling that boundary, would display as two dramatically different brightness levels instead of two nearly-identical ones, even though the underlying anatomy they represent is continuous. A radiologist viewing such an image could easily mistake that artificial discontinuity for a real anatomical edge or boundary that does not actually exist in the tissue being imaged -- exactly the kind of fabricated visual feature the real linear VOI LUT formula's own smooth, continuous mapping is specifically designed to avoid.

**5.** A check for specific forbidden field names (refusing anything literally named `DIAGNOSIS`, for instance) only catches the exact names someone thought to forbid in advance, and a differently-named field carrying the same kind of content (`IMPRESSION=...`, `ASSESSMENT=...`, or any other name not on that specific list) would sail through unrefused. A strict field-COUNT check refuses ANY line that does not have EXACTLY the four expected fields, regardless of what any extra field happens to be named -- it does not need to anticipate every name someone might try, because it refuses the sheer presence of an unexpected fifth field on structural grounds alone, which is a strictly more general and more robust defense than an ever-growing denylist of specific names.

**6.** Consider a study with one `STAT` finding (rank 2) and one `ROUTINE` finding (rank 0). The max-based approach correctly reports the study's own overall priority as `STAT` -- exactly the outcome a real triage queue needs, since the study genuinely does contain a finding requiring urgent review, however calm the other finding in the same study may be. Averaging the two ranks would produce something between `ROUTINE` and `EXPEDITED` (rank 1, "EXPEDITED," if rounded, or a genuinely intermediate non-integer value if not), which would route a study containing a real STAT-level finding into a LESS urgent review queue than it deserves -- exactly the kind of averaged-away urgent signal a real clinical triage system cannot afford to produce, since the calm finding does nothing to make the urgent one less urgent.

**7.** Checking only `state_ == ASSIGNED` would permit ANY caller who happens to know a case's own accession number -- not only the radiologist who actually claimed it -- to call `begin_review` or `finalize` on that case, since the state alone carries no information about WHO is accountable for it. Two different radiologists working from the same shared worklist could each attempt to advance the identical case, and a state-only check would let either of them succeed, silently overwriting or duplicating the other's own work with no record of which radiologist's actions should actually count. Checking `radiologist_id == *claimed_by_` at every subsequent transition is what actually enforces "the SAME accountable person throughout," rather than merely "someone, anyone, at the right moment in the state machine."

**8.** With an ordinary unstable sort, two STAT-priority cases added at different times could be returned in one order on one call to `pending_by_priority` and in the OPPOSITE order on a later call, purely because an unstable sort makes no guarantee about the relative order of elements it considers equal (both are priority `STAT`) -- even though neither case's own data changed between the two calls. A radiologist's worklist view could then show case A above case B one moment and B above A the next, with no real event having occurred to justify the reordering, which is exactly the kind of confusing, unexplainable queue behavior `std::stable_sort`'s own guarantee (elements considered equal keep their original relative order) rules out entirely.

**9.** A black occlusion patch has a raw pixel value of 0 in every channel, which is very unlikely to be anywhere near the image's own actual per-channel mean -- so replacing a patch with black does not merely "remove" that patch's own content, it also introduces a NEW, artificially large edge at the boundary between the black patch and its real neighboring patches, an edge the model may react to simply because sharp edges are salient to a vision encoder, regardless of whether the ORIGINAL patch's content was actually important. A mean-color occlusion, by contrast, is (in the normalized space Part 1 uses) exactly the zero vector -- the smallest, most neutral edit available, since it introduces no new edge of its own, meaning any resulting change in the model's output can be attributed more confidently to the removal of that patch's own real content rather than to an artifact the occlusion method itself introduced.

**10.** Checking only that the two scores are of comparable magnitude would be equally consistent with a much less interesting (and less honest) possible outcome: that the two anomalies, despite being different in location and pixel value, happened to move the representation in ROUGHLY THE SAME DIRECTION as well as by a similar amount -- in which case a defender of the technique could reasonably argue the saliency score, while scalar, was still capturing something like "the same kind of importance" both times. The cosine-similarity check is what actually rules that alternative explanation out: a similarity of roughly 0.2, far from 1.0, demonstrates directly that the two comparable-magnitude scores correspond to the representation moving in substantially DIFFERENT directions in the model's own hidden space -- proving the magnitude number alone genuinely cannot distinguish "the same kind of change happened twice" from "two different things happened to look similarly sized," which is the specific, real limit this section set out to prove rather than merely assert.
