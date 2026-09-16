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
