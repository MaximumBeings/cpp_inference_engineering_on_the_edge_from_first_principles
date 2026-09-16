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
