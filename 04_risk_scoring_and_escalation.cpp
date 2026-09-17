// Chapter 26.4 -- This chapter's own capstone: a real, from-scratch,
// fully auditable risk-scoring engine that combines Section 26.1's MICR
// checksum and signature-comparison signals, Section 26.2's duplicate-
// invoice signal, and Section 26.3's amount-in-words cross-check into a
// single weighted score and a three-way disposition -- AUTO_CLEAR,
// ESCALATE_TO_REVIEW, or AUTO_REJECT. The central discipline, identical
// to Chapter 20's own TriageCase state machine and Chapter 23's own
// never-suppress-a-named-flag rule: the aggregate score decides the
// DISPOSITION, but every single contributing signal is named individually
// in an audit trail no aggregate number can hide, and one specific,
// high-severity red flag (a failed MICR checksum) can force escalation
// on its own, regardless of how low every other signal's score is.
//
// A note on this section's own honest scope: `assess_risk` produces a
// real, deterministic, fully-explained DISPOSITION from a fixed set of
// stated signals and weights -- it is a real screening step, not a fraud
// verdict. AUTO_CLEAR means "no stated signal in this engine's own scope
// fired," not "this transaction is definitely legitimate" -- the same
// honest structural-pass distinction Chapter 23.4's own luxury-goods
// screening drew for its own best possible outcome.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_risk_scoring_and_escalation.cpp -o 04_risk_scoring_and_escalation
// Run:     ./04_risk_scoring_and_escalation

#include <algorithm>
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
// A real, auditable risk signal: a name, a point weight, and whether it
// fired -- never a bare number with no way to trace back which real
// check produced it.
// =======================================================================
struct RiskSignal {
    std::string name;
    int weight = 0;
    bool fired = false;
};

enum class Disposition { AUTO_CLEAR, ESCALATE_TO_REVIEW, AUTO_REJECT };

std::string disposition_name(Disposition d) {
    switch (d) {
        case Disposition::AUTO_CLEAR: return "AUTO_CLEAR";
        case Disposition::ESCALATE_TO_REVIEW: return "ESCALATE_TO_REVIEW";
        case Disposition::AUTO_REJECT: return "AUTO_REJECT";
    }
    return "UNKNOWN";
}

struct RiskAssessment {
    int total_score = 0;
    Disposition disposition = Disposition::AUTO_CLEAR;
    std::vector<std::string> fired_signal_names;  // the full, named audit trail
    bool forced_by_micr_failure = false;
};

// A real, stated scoring policy: below ESCALATE_THRESHOLD, auto-clear;
// at or above ESCALATE_THRESHOLD but below REJECT_THRESHOLD, escalate to
// a human reviewer; at or above REJECT_THRESHOLD, auto-reject. A failed
// MICR checksum -- structurally inconsistent with any real bank's own
// routing number -- forces at least ESCALATE_TO_REVIEW regardless of
// every other signal's own score, since it alone indicates the check
// image itself may not be genuine.
constexpr int ESCALATE_THRESHOLD = 30;
constexpr int REJECT_THRESHOLD = 70;

RiskAssessment assess_risk(const std::vector<RiskSignal>& signals, bool micr_checksum_failed) {
    RiskAssessment result;
    for (const RiskSignal& s : signals) {
        if (s.fired) {
            result.total_score += s.weight;
            result.fired_signal_names.push_back(s.name);
        }
    }
    if (result.total_score >= REJECT_THRESHOLD) {
        result.disposition = Disposition::AUTO_REJECT;
    } else if (result.total_score >= ESCALATE_THRESHOLD) {
        result.disposition = Disposition::ESCALATE_TO_REVIEW;
    } else {
        result.disposition = Disposition::AUTO_CLEAR;
    }
    if (micr_checksum_failed && result.disposition == Disposition::AUTO_CLEAR) {
        result.disposition = Disposition::ESCALATE_TO_REVIEW;
        result.forced_by_micr_failure = true;
    }
    return result;
}

void print_assessment(const std::string& label, const RiskAssessment& a) {
    std::cout << label << ": score = " << a.total_score << ", disposition = "
              << disposition_name(a.disposition);
    if (a.forced_by_micr_failure) std::cout << " (forced by MICR checksum failure)";
    std::cout << ", fired signals: [";
    for (std::size_t i = 0; i < a.fired_signal_names.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << a.fired_signal_names[i];
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.4: Risk Scoring and Escalation\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: a clean transaction -- every signal from every prior
    // section passes -- is auto-cleared with an empty, honestly-empty
    // audit trail, not a suppressed one. --
    std::vector<RiskSignal> clean_signals = {
        {"duplicate_invoice", 40, false},
        {"amount_words_mismatch", 50, false},
        {"signature_mismatch", 35, false},
    };
    RiskAssessment a1 = assess_risk(clean_signals, false);
    print_assessment("-- Test 1 (clean transaction)", a1);
    CHECK(a1.total_score == 0);
    CHECK(a1.disposition == Disposition::AUTO_CLEAR);
    CHECK(a1.fired_signal_names.empty());

    // -- Test 2: a single, moderate-severity red flag (Section 26.2's
    // duplicate-invoice signal alone) crosses the escalate threshold but
    // not the reject threshold -- routed to a human reviewer, not
    // auto-rejected outright on one signal alone. --
    std::vector<RiskSignal> single_flag = {
        {"duplicate_invoice", 40, true},
        {"amount_words_mismatch", 50, false},
        {"signature_mismatch", 35, false},
    };
    RiskAssessment a2 = assess_risk(single_flag, false);
    print_assessment("-- Test 2 (one moderate flag)", a2);
    CHECK(a2.total_score == 40);
    CHECK(a2.disposition == Disposition::ESCALATE_TO_REVIEW);
    CHECK(a2.fired_signal_names.size() == 1);
    CHECK(a2.fired_signal_names[0] == "duplicate_invoice");

    // -- Test 3: two red flags together cross the reject threshold, and
    // the audit trail names BOTH contributing signals -- the aggregate
    // score never hides which specific checks actually fired. --
    std::vector<RiskSignal> two_flags = {
        {"duplicate_invoice", 40, true},
        {"amount_words_mismatch", 50, true},
        {"signature_mismatch", 35, false},
    };
    RiskAssessment a3 = assess_risk(two_flags, false);
    print_assessment("-- Test 3 (two flags, crosses reject threshold)", a3);
    CHECK(a3.total_score == 90);
    CHECK(a3.disposition == Disposition::AUTO_REJECT);
    CHECK(a3.fired_signal_names.size() == 2);

    // -- Test 4: a failed MICR checksum, on its own, with every OTHER
    // signal clean and a total score of 0 -- still forces at least
    // ESCALATE_TO_REVIEW, confirming one specific high-severity signal
    // is never outvoted by an otherwise-clean aggregate score, the
    // identical discipline Chapter 23.4's own hard-failing weight check
    // applied. --
    RiskAssessment a4 = assess_risk(clean_signals, true);
    print_assessment("-- Test 4 (MICR checksum failure alone, all other signals clean)", a4);
    CHECK(a4.total_score == 0);
    CHECK(a4.disposition == Disposition::ESCALATE_TO_REVIEW);
    CHECK(a4.forced_by_micr_failure);

    // -- Test 5: a failed MICR checksum combined with signals that
    // ALREADY cross the reject threshold on their own does not downgrade
    // the disposition -- the forcing rule only ever escalates an
    // AUTO_CLEAR upward, it never overrides an already-stricter
    // AUTO_REJECT with a milder ESCALATE_TO_REVIEW. --
    RiskAssessment a5 = assess_risk(two_flags, true);
    print_assessment("-- Test 5 (MICR checksum failure PLUS signals that already reject)", a5);
    CHECK(a5.disposition == Disposition::AUTO_REJECT);
    CHECK(!a5.forced_by_micr_failure);

    // -- Test 6: a score exactly AT the escalate threshold (30) is
    // escalated, not auto-cleared -- the boundary itself is checked, not
    // just a value comfortably below it. --
    std::vector<RiskSignal> boundary_signals = {
        {"minor_signal", 30, true},
    };
    RiskAssessment a6 = assess_risk(boundary_signals, false);
    print_assessment("-- Test 6 (score exactly at the escalate boundary of 30)", a6);
    CHECK(a6.total_score == 30);
    CHECK(a6.disposition == Disposition::ESCALATE_TO_REVIEW);

    // -- Test 7: a score exactly AT the reject threshold (70) is
    // rejected, not merely escalated -- the reject boundary is likewise
    // checked exactly. --
    std::vector<RiskSignal> reject_boundary_signals = {
        {"duplicate_invoice", 40, true},
        {"signature_mismatch", 30, true},
    };
    RiskAssessment a7 = assess_risk(reject_boundary_signals, false);
    print_assessment("-- Test 7 (score exactly at the reject boundary of 70)", a7);
    CHECK(a7.total_score == 70);
    CHECK(a7.disposition == Disposition::AUTO_REJECT);

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
