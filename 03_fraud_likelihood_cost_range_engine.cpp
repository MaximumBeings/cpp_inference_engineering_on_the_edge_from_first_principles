// Chapter 28.3 -- This section directly extends Chapter 21.3's own real
// severity-to-cost-range engine and before/after photo-consistency check
// -- both reused completely UNCHANGED below -- with a new real per-zone
// fraud-likelihood signal built from Section 28.1's duplicate-photo flag,
// Section 28.2's timestamp-anomaly flag, and Chapter 21.3's own existing
// photo-consistency verdict. The central discipline this section adds:
// a zone's own COST ESTIMATE and its own FRAUD LIKELIHOOD are computed
// completely independently of each other -- attaching a fraud signal to
// a zone must never silently shrink, zero out, or otherwise adjust that
// zone's own currently-claimed cost contribution, exactly the same
// never-auto-adjust discipline Chapter 21.3's own Test 5 already
// established for a single flagged zone.
//
// A note on this section's own honest scope: `assess_fraud_likelihood`
// is a real, simple, fully transparent COUNT of how many named real
// signals fired for a zone -- LOW, MEDIUM, or HIGH describes how many
// independent real checks disagreed with the claim, never a calibrated
// real probability of fraud. This section's own COMMON TRAP box returns
// to exactly this distinction.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_fraud_likelihood_cost_range_engine.cpp -o 03_fraud_likelihood_cost_range_engine
// Run:     ./03_fraud_likelihood_cost_range_engine

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
// Chapter 21.3's own real severity-to-cost-range policy, reused
// completely UNCHANGED -- generalized here to a plain zone NAME string,
// exactly as Chapter 21.3's own introduction already stated this
// machinery applies identically to a vehicle panel or a property zone.
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

CostRange cost_range_for_severity(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return {0.0, 0.0};
        case DamageSeverity::MINOR: return {150.0, 600.0};
        case DamageSeverity::MODERATE: return {600.0, 2500.0};
        case DamageSeverity::SEVERE: return {2500.0, 9000.0};
    }
    return {0.0, 0.0};
}

// =======================================================================
// Chapter 21.3's own real before/after photo-consistency check, reused
// completely UNCHANGED.
// =======================================================================
double mean_abs_diff(const std::vector<std::uint8_t>& before, const std::vector<std::uint8_t>& after) {
    double sum = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i) {
        sum += std::abs(static_cast<int>(before[i]) - static_cast<int>(after[i]));
    }
    return sum / static_cast<double>(before.size());
}

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

ConsistencyVerdict check_claim_consistency(DamageSeverity claimed, double measured_diff) {
    DiffBand band = expected_diff_band(claimed);
    if (measured_diff < band.low) return ConsistencyVerdict::OVER_CLAIMED;
    if (measured_diff > band.high) return ConsistencyVerdict::UNDER_CLAIMED;
    return ConsistencyVerdict::CONSISTENT;
}

// =======================================================================
// NEW in this section: a per-zone bundle of three independent real
// fraud signals -- Section 28.1's duplicate-photo flag, Section 28.2's
// timestamp-anomaly flag, and Chapter 21.3's own photo-consistency
// verdict above -- combined into a transparent, named fraud-likelihood
// LEVEL, never a single opaque probability.
// =======================================================================
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

// A real, fully transparent count: zero named signals is LOW, exactly
// one is MEDIUM, two or more is HIGH. Every one of the three signals is
// weighted identically -- this function counts WHICH real checks
// disagreed with the claim, not how "serious" any one of them is.
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

struct ZoneFraudReport {
    std::string zone_name;
    FraudLikelihood level;
};

struct ClaimAssessment {
    CostRange total_range;
    std::vector<ZoneFraudReport> per_zone_fraud;
};

// The claim's own total cost range is summed EXACTLY as Chapter 21.3's
// own `assess_claim` already did -- this loop never reads a zone's own
// FraudSignals when computing total_range, by construction, not merely
// by convention.
ClaimAssessment assess_claim(const std::vector<ZoneClaim>& zones) {
    ClaimAssessment result;
    for (const auto& z : zones) {
        result.total_range = result.total_range + cost_range_for_severity(z.claimed_severity);
        result.per_zone_fraud.push_back({z.zone_name, assess_fraud_likelihood(z.fraud_signals)});
    }
    return result;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 28.3: Extending the Cost-Range Engine with a Fraud-Likelihood Signal\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: Chapter 21.3's own severity-to-cost-range policy,
    // reused unchanged, still holds exactly. --
    {
        std::cout << "-- Test 1: Chapter 21.3's own cost-range policy, reused unchanged --\n";
        CHECK(cost_range_for_severity(DamageSeverity::MINOR).low == 150.0);
        CHECK(cost_range_for_severity(DamageSeverity::MINOR).high == 600.0);
        CHECK(cost_range_for_severity(DamageSeverity::MODERATE).low == 600.0);
        CHECK(cost_range_for_severity(DamageSeverity::SEVERE).high == 9000.0);
    }

    // -- Test 2: zero fraud signals fired -- LOW. --
    {
        auto level = assess_fraud_likelihood({false, false, false});
        std::cout << "-- Test 2: zero signals fired -- level=" << to_string(level) << " --\n";
        CHECK(level == FraudLikelihood::LOW);
    }

    // -- Test 3: exactly one signal fired, tried in turn for each of the
    // three independent signal types -- each alone is MEDIUM, confirming
    // all three are weighted identically. --
    {
        auto only_duplicate = assess_fraud_likelihood({true, false, false});
        auto only_timestamp = assess_fraud_likelihood({false, true, false});
        auto only_inconsistent = assess_fraud_likelihood({false, false, true});
        std::cout << "-- Test 3: exactly one signal fired -- duplicate-only=" << to_string(only_duplicate)
                  << ", timestamp-only=" << to_string(only_timestamp) << ", inconsistent-only="
                  << to_string(only_inconsistent) << " --\n";
        CHECK(only_duplicate == FraudLikelihood::MEDIUM);
        CHECK(only_timestamp == FraudLikelihood::MEDIUM);
        CHECK(only_inconsistent == FraudLikelihood::MEDIUM);
    }

    // -- Test 4: exactly two signals fired -- the MEDIUM/HIGH boundary,
    // confirmed exactly at the count of 2. --
    {
        auto level = assess_fraud_likelihood({true, true, false});
        std::cout << "-- Test 4: exactly two signals fired (duplicate + timestamp) -- level="
                  << to_string(level) << " --\n";
        CHECK(level == FraudLikelihood::HIGH);
    }

    // -- Test 5: all three signals fired at once. --
    {
        auto level = assess_fraud_likelihood({true, true, true});
        std::cout << "-- Test 5: all three signals fired -- level=" << to_string(level) << " --\n";
        CHECK(level == FraudLikelihood::HIGH);
    }

    // -- Test 6 (central): a two-zone claim, one clean and one with all
    // three fraud signals firing, must produce EXACTLY the same
    // total_range as the identical severities with zero fraud signals
    // attached -- a direct, paired comparison proving the cost estimate
    // is never touched by the fraud-likelihood computation. --
    {
        std::vector<ZoneClaim> clean_zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"hood", DamageSeverity::SEVERE, {}},
        };
        std::vector<ZoneClaim> flagged_zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"hood", DamageSeverity::SEVERE, {true, true, true}},
        };
        auto clean_result = assess_claim(clean_zones);
        auto flagged_result = assess_claim(flagged_zones);
        std::cout << "-- Test 6: clean total=[" << clean_result.total_range.low << ", "
                  << clean_result.total_range.high << "], all-signals-firing total=["
                  << flagged_result.total_range.low << ", " << flagged_result.total_range.high
                  << "] -- identical --\n";
        CHECK(std::abs(clean_result.total_range.low - flagged_result.total_range.low) < 1e-9);
        CHECK(std::abs(clean_result.total_range.high - flagged_result.total_range.high) < 1e-9);
        CHECK(std::abs(flagged_result.total_range.low - 3100.0) < 1e-9);   // 600 + 2500
        CHECK(std::abs(flagged_result.total_range.high - 11500.0) < 1e-9);  // 2500 + 9000
        CHECK(flagged_result.per_zone_fraud[1].level == FraudLikelihood::HIGH);
    }

    // -- Test 7: a full multi-zone claim's own per-zone fraud report
    // names EACH zone by its own name and its own individual level, not
    // merely an aggregate count. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MINOR, {}},
            {"driver_door", DamageSeverity::SEVERE, {true, false, true}},
            {"rear_bumper", DamageSeverity::MODERATE, {false, true, false}},
        };
        auto result = assess_claim(zones);
        std::cout << "-- Test 7: 3-zone claim -- ";
        for (const auto& r : result.per_zone_fraud) std::cout << r.zone_name << "=" << to_string(r.level) << " ";
        std::cout << "--\n";
        CHECK(result.per_zone_fraud.size() == 3);
        CHECK(result.per_zone_fraud[0].zone_name == "front_bumper");
        CHECK(result.per_zone_fraud[0].level == FraudLikelihood::LOW);
        CHECK(result.per_zone_fraud[1].zone_name == "driver_door");
        CHECK(result.per_zone_fraud[1].level == FraudLikelihood::HIGH);
        CHECK(result.per_zone_fraud[2].zone_name == "rear_bumper");
        CHECK(result.per_zone_fraud[2].level == FraudLikelihood::MEDIUM);
    }

    // -- Test 8: closing the full loop -- Chapter 21.3's own reused
    // `mean_abs_diff` and `check_claim_consistency` compute a REAL
    // photo_inconsistent flag from actual before/after pixel data
    // (rather than a hand-set boolean), which then flows into
    // `assess_fraud_likelihood` exactly like any other signal, all
    // while the zone's own total cost contribution stays untouched. --
    {
        std::vector<std::uint8_t> before(16, 100), after(16, 102);  // nearly identical photos
        double measured_diff = mean_abs_diff(before, after);
        auto verdict = check_claim_consistency(DamageSeverity::SEVERE, measured_diff);
        FraudSignals signals{false, false, verdict != ConsistencyVerdict::CONSISTENT};
        std::vector<ZoneClaim> zones = {{"quarter_panel", DamageSeverity::SEVERE, signals}};
        auto result = assess_claim(zones);
        std::cout << "-- Test 8: claimed SEVERE against a measured photo difference of " << measured_diff
                  << " -- photo_inconsistent=" << (signals.photo_inconsistent ? "true" : "false")
                  << ", fraud level=" << to_string(result.per_zone_fraud[0].level) << ", total=["
                  << result.total_range.low << ", " << result.total_range.high << "] --\n";
        CHECK(verdict == ConsistencyVerdict::OVER_CLAIMED);
        CHECK(signals.photo_inconsistent);
        CHECK(result.per_zone_fraud[0].level == FraudLikelihood::MEDIUM);
        CHECK(std::abs(result.total_range.low - 2500.0) < 1e-9);
        CHECK(std::abs(result.total_range.high - 9000.0) < 1e-9);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
