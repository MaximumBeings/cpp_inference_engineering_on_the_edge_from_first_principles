// Chapter 28.4 -- This chapter's own capstone: a real, fully auditable
// claims-disposition engine that reuses Section 28.3's own cost-range
// and per-zone fraud-likelihood machinery unchanged, and routes an
// entire claim to one of exactly three real dispositions based on the
// SINGLE WORST zone across the whole claim -- reapplying this book's own
// recurring never-suppress-a-named-flag discipline one more time, in
// the same direction Chapter 26.4 applied it (toward more scrutiny, not
// less), but with a real, deliberate ethical boundary this domain
// requires that neither Chapter 26.4 nor Chapter 27.4 needed: THIS
// engine has no "deny" or "reject" disposition at all. A suspected
// insurance fraud signal is a reason to route a claim to a real human
// Special Investigations Unit reviewer, never a reason for a machine to
// deny a real claimant's own payment automatically -- a false machine
// denial has real, serious consequences for a genuine claimant, exactly
// the reason real insurance regulation in most real jurisdictions
// requires a human decision before any fraud-suspected claim is denied.
//
// A note on this section's own honest scope: `ClaimDisposition` decides
// only WHERE a claim is ROUTED, never whether it is paid, and the
// claim's own total cost range is reported identically regardless of
// disposition -- routing a claim to a human reviewer does not, and must
// not, silently withhold or alter the claimant's own currently-stated
// estimate. This section's own COMMON TRAP box returns to exactly this
// distinction.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_claims_disposition_engine.cpp -o 04_claims_disposition_engine
// Run:     ./04_claims_disposition_engine

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
// Section 28.3's own cost-range and fraud-likelihood machinery, reused
// completely UNCHANGED.
// =======================================================================
enum class DamageSeverity { NONE, MINOR, MODERATE, SEVERE };

struct CostRange { double low = 0.0, high = 0.0; };
CostRange operator+(const CostRange& a, const CostRange& b) { return {a.low + b.low, a.high + b.high}; }

CostRange cost_range_for_severity(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return {0.0, 0.0};
        case DamageSeverity::MINOR: return {150.0, 600.0};
        case DamageSeverity::MODERATE: return {600.0, 2500.0};
        case DamageSeverity::SEVERE: return {2500.0, 9000.0};
    }
    return {0.0, 0.0};
}

struct FraudSignals {
    bool duplicate_photo = false;
    bool timestamp_anomaly = false;
    bool photo_inconsistent = false;
};

enum class FraudLikelihood { LOW, MEDIUM, HIGH };

std::string to_string(FraudLikelihood f) {
    switch (f) {
        case FraudLikelihood::LOW: return "LOW";
        case FraudLikelihood::MEDIUM: return "MEDIUM";
        case FraudLikelihood::HIGH: return "HIGH";
    }
    return "UNKNOWN";
}

FraudLikelihood assess_fraud_likelihood(const FraudSignals& s) {
    int count = (s.duplicate_photo ? 1 : 0) + (s.timestamp_anomaly ? 1 : 0) + (s.photo_inconsistent ? 1 : 0);
    if (count >= 2) return FraudLikelihood::HIGH;
    if (count == 1) return FraudLikelihood::MEDIUM;
    return FraudLikelihood::LOW;
}

struct ZoneClaim {
    std::string zone_name;
    DamageSeverity claimed_severity;
    FraudSignals fraud_signals;
};

// =======================================================================
// NEW in this section: a claim-level disposition, driven by the SINGLE
// WORST zone's own fraud-likelihood level across the entire claim --
// never diluted by averaging against otherwise-clean zones -- with
// exactly three possible outcomes, NONE of which is a denial.
// =======================================================================
enum class ClaimDisposition { PROCESS_NORMALLY, FLAG_FOR_SIU_REVIEW, HOLD_PENDING_VERIFICATION };

std::string to_string(ClaimDisposition d) {
    switch (d) {
        case ClaimDisposition::PROCESS_NORMALLY: return "PROCESS_NORMALLY";
        case ClaimDisposition::FLAG_FOR_SIU_REVIEW: return "FLAG_FOR_SIU_REVIEW";
        case ClaimDisposition::HOLD_PENDING_VERIFICATION: return "HOLD_PENDING_VERIFICATION";
    }
    return "UNKNOWN";
}

struct ClaimResult {
    CostRange total_range;
    ClaimDisposition disposition;
    std::vector<std::string> reasons;  // the full, named audit trail
};

// The claim's own total cost range is summed exactly as Section 28.3's
// own `assess_claim` did, completely independent of the disposition
// logic below it. The disposition itself is driven by the single WORST
// per-zone fraud level found anywhere in the claim -- a HIGH level in
// even one zone routes the ENTIRE claim to HOLD_PENDING_VERIFICATION,
// regardless of how many other zones are completely clean.
ClaimResult assess_claim_disposition(const std::vector<ZoneClaim>& zones) {
    ClaimResult result;
    FraudLikelihood worst = FraudLikelihood::LOW;
    for (const auto& z : zones) {
        result.total_range = result.total_range + cost_range_for_severity(z.claimed_severity);
        FraudLikelihood level = assess_fraud_likelihood(z.fraud_signals);
        if (level != FraudLikelihood::LOW) {
            result.reasons.push_back(z.zone_name + ": fraud likelihood " + to_string(level));
        }
        if (level == FraudLikelihood::HIGH) worst = FraudLikelihood::HIGH;
        else if (level == FraudLikelihood::MEDIUM && worst != FraudLikelihood::HIGH) worst = FraudLikelihood::MEDIUM;
    }
    if (worst == FraudLikelihood::HIGH) {
        result.disposition = ClaimDisposition::HOLD_PENDING_VERIFICATION;
    } else if (worst == FraudLikelihood::MEDIUM) {
        result.disposition = ClaimDisposition::FLAG_FOR_SIU_REVIEW;
    } else {
        result.disposition = ClaimDisposition::PROCESS_NORMALLY;
    }
    return result;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 28.4: The Claims Disposition Engine\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: an entirely clean, multi-zone claim -- processes
    // normally, with the exact hand-computed total cost range. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"hood", DamageSeverity::MINOR, {}},
        };
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 1: clean 2-zone claim -- disposition=" << to_string(r.disposition)
                  << ", total=[" << r.total_range.low << ", " << r.total_range.high << "] --\n";
        CHECK(r.disposition == ClaimDisposition::PROCESS_NORMALLY);
        CHECK(std::abs(r.total_range.low - 750.0) < 1e-9);
        CHECK(std::abs(r.total_range.high - 3100.0) < 1e-9);
        CHECK(r.reasons.empty());
    }

    // -- Test 2: one MEDIUM zone among otherwise-clean zones -- the
    // whole claim is flagged for review, never diluted by the clean
    // zones around it. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"driver_door", DamageSeverity::MINOR, {true, false, false}},
            {"rear_bumper", DamageSeverity::MINOR, {}},
        };
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 2: one MEDIUM zone among two clean zones -- disposition="
                  << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == ClaimDisposition::FLAG_FOR_SIU_REVIEW);
        CHECK(r.reasons.size() == 1);
    }

    // -- Test 3: one HIGH zone among otherwise clean/medium zones -- the
    // whole claim is held pending verification, the worst zone wins. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"driver_door", DamageSeverity::MINOR, {true, false, false}},
            {"windshield", DamageSeverity::SEVERE, {true, true, false}},
        };
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 3: one HIGH zone among a clean and a MEDIUM zone -- disposition="
                  << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == ClaimDisposition::HOLD_PENDING_VERIFICATION);
        CHECK(r.reasons.size() == 2);
    }

    // -- Test 4 (central): the IDENTICAL zone severities as Test 3
    // produce the EXACT SAME total cost range whether or not any fraud
    // signal is attached -- a direct paired comparison proving the
    // disposition logic never touches the claim's own cost estimate. --
    {
        std::vector<ZoneClaim> flagged_zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"driver_door", DamageSeverity::MINOR, {true, false, false}},
            {"windshield", DamageSeverity::SEVERE, {true, true, false}},
        };
        std::vector<ZoneClaim> clean_zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"driver_door", DamageSeverity::MINOR, {}},
            {"windshield", DamageSeverity::SEVERE, {}},
        };
        auto flagged = assess_claim_disposition(flagged_zones);
        auto clean = assess_claim_disposition(clean_zones);
        std::cout << "-- Test 4: HOLD-triggering total=[" << flagged.total_range.low << ", "
                  << flagged.total_range.high << "], fully-clean total=[" << clean.total_range.low << ", "
                  << clean.total_range.high << "] -- identical, disposition differs (" << to_string(flagged.disposition)
                  << " vs. " << to_string(clean.disposition) << ") --\n";
        CHECK(std::abs(flagged.total_range.low - clean.total_range.low) < 1e-9);
        CHECK(std::abs(flagged.total_range.high - clean.total_range.high) < 1e-9);
        CHECK(flagged.disposition != clean.disposition);
    }

    // -- Test 5 (structural never-auto-deny guarantee): every one of the
    // three real dispositions this engine can ever produce is checked by
    // name to contain neither "DENY" nor "REJECT" -- a structural
    // property of the enum's own three values, not an incidental fact
    // about this section's own test data. --
    {
        std::vector<ClaimDisposition> all = {ClaimDisposition::PROCESS_NORMALLY,
                                              ClaimDisposition::FLAG_FOR_SIU_REVIEW,
                                              ClaimDisposition::HOLD_PENDING_VERIFICATION};
        bool any_denial_wording = false;
        for (auto d : all) {
            std::string name = to_string(d);
            if (name.find("DENY") != std::string::npos || name.find("REJECT") != std::string::npos) {
                any_denial_wording = true;
            }
        }
        std::cout << "-- Test 5: all 3 real dispositions checked for denial wording -- found: "
                  << (any_denial_wording ? "YES (WRONG)" : "NONE (correct)") << " --\n";
        CHECK(all.size() == 3);
        CHECK(!any_denial_wording);
    }

    // -- Test 6: the disposition's own audit trail names the EXACT
    // zone(s) responsible for a HOLD_PENDING_VERIFICATION disposition
    // and their own fraud level, not merely an aggregate flag count. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"windshield", DamageSeverity::SEVERE, {true, true, true}},
        };
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 6: audit trail -- ";
        for (const auto& reason : r.reasons) std::cout << "[" << reason << "] ";
        std::cout << "--\n";
        CHECK(r.reasons.size() == 1);
        CHECK(r.reasons[0].find("windshield") != std::string::npos);
        CHECK(r.reasons[0].find("HIGH") != std::string::npos);
        CHECK(r.reasons[0].find("front_bumper") == std::string::npos);
    }

    // -- Test 7: the exact MEDIUM/HIGH signal-count boundary carried
    // through to the claim level -- exactly 2 signals on one zone drives
    // HOLD_PENDING_VERIFICATION, confirmed precisely at that boundary. --
    {
        std::vector<ZoneClaim> zones = {{"roof", DamageSeverity::MODERATE, {false, true, true}}};
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 7: single zone, exactly 2 fraud signals -- disposition="
                  << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == ClaimDisposition::HOLD_PENDING_VERIFICATION);
    }

    // -- Test 8: a full multi-zone, multi-signal claim's own total cost
    // range checked against an exact hand computation, tying this
    // capstone back to Chapter 21.3's own Test 1 and Test 5 numerically. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"hood", DamageSeverity::MINOR, {true, false, false}},
            {"windshield", DamageSeverity::SEVERE, {}},
        };
        auto r = assess_claim_disposition(zones);
        // hand-computed: (600+150+2500, 2500+600+9000) = (3250, 12100)
        std::cout << "-- Test 8: 3-zone claim total=[" << r.total_range.low << ", " << r.total_range.high
                  << "], disposition=" << to_string(r.disposition) << " --\n";
        CHECK(std::abs(r.total_range.low - 3250.0) < 1e-9);
        CHECK(std::abs(r.total_range.high - 12100.0) < 1e-9);
        CHECK(r.disposition == ClaimDisposition::FLAG_FOR_SIU_REVIEW);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
