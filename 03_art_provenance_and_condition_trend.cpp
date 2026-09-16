// Chapter 22.3 -- An art-authentication engine has two genuinely
// different real jobs, and this section refuses to blur them into one.
// The first is a real, structural question with a definite yes-or-no
// answer: does a stated chain of ownership even make CHRONOLOGICAL
// sense -- no gap in custody, no two owners claiming the same period,
// exactly one current owner, and never an owner who supposedly disposed
// of a work before they ever acquired it. The second is a real,
// numerical question that only ever produces a TREND, never a verdict:
// is a work's own condition changing over time, and how fast -- answered
// with the identical real ordinary-least-squares regression Section
// 19.4 already built for a retail inventory's own stockout trend,
// applied here to a conservator's own periodic condition scores
// instead. Neither of these functions ever renders a verdict on
// authenticity or fraud; each one names a specific, checkable
// inconsistency or a specific, computed rate, and leaves what it means
// to a human investigator or conservator, exactly the way this book's
// own human-in-the-loop discipline has required since Chapter 20.
//
// A note on this section's own honest scope: a gap or an overlap in a
// stated provenance chain is a real, useful red flag worth surfacing,
// but it is not, by itself, proof of fraud -- a real gap can also result
// from an honestly incomplete historical record, and this section's own
// `validate_provenance_chain` reports every inconsistency it finds by
// name and by exact size, never as an accusation.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_art_provenance_and_condition_trend.cpp -o 03_art_provenance_and_condition_trend
// Run:     ./03_art_provenance_and_condition_trend

#include <algorithm>
#include <cmath>
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
// PART 1: provenance-chain validation -- a real, structural check of
// chronological consistency, never a fraud verdict.
// =======================================================================
struct ProvenanceRecord {
    std::string owner;
    int acquired_period = 0;
    std::optional<int> disposed_period;   // nullopt means "still the current owner"
};

struct ProvenanceValidation {
    bool valid = true;
    std::vector<std::string> issues;
};

ProvenanceValidation validate_provenance_chain(std::vector<ProvenanceRecord> records) {
    ProvenanceValidation result;
    std::stable_sort(records.begin(), records.end(), [](const ProvenanceRecord& a, const ProvenanceRecord& b) {
        return a.acquired_period < b.acquired_period;
    });

    // Rule 1: no owner may have disposed of the work before (or at the
    // same period as) they acquired it.
    for (const auto& r : records) {
        if (r.disposed_period.has_value() && *r.disposed_period <= r.acquired_period) {
            result.issues.push_back(r.owner + ": disposal period (" + std::to_string(*r.disposed_period) +
                                     ") does not come after their own acquisition period (" +
                                     std::to_string(r.acquired_period) + ")");
        }
    }

    // Rule 2: exactly one record may be open-ended (the current owner),
    // and it must be the chronologically LAST record in the chain.
    std::vector<std::string> open_ended_owners;
    for (const auto& r : records) {
        if (!r.disposed_period.has_value()) open_ended_owners.push_back(r.owner);
    }
    if (open_ended_owners.empty()) {
        result.issues.push_back("no current owner recorded -- every owner in this chain has a recorded disposal period");
    } else if (open_ended_owners.size() > 1) {
        std::string names;
        for (size_t i = 0; i < open_ended_owners.size(); ++i) {
            if (i > 0) names += ", ";
            names += open_ended_owners[i];
        }
        result.issues.push_back("more than one owner recorded with no disposal period (cannot both be the "
                                 "current owner): " + names);
    } else if (records.back().owner != open_ended_owners.front()) {
        result.issues.push_back(open_ended_owners.front() +
                                 " is recorded with no disposal period but is not the chronologically last "
                                 "owner in this chain");
    }

    // Rule 3: consecutive owners' own custody periods must be exactly
    // contiguous -- no gap, no overlap.
    for (size_t i = 0; i + 1 < records.size(); ++i) {
        const auto& current = records[i];
        const auto& next = records[i + 1];
        if (!current.disposed_period.has_value()) continue;   // an open-ended record has no "next" to compare
        if (*current.disposed_period < next.acquired_period) {
            int gap = next.acquired_period - *current.disposed_period;
            result.issues.push_back("gap of " + std::to_string(gap) + " period(s) in custody between " +
                                     current.owner + " (disposed at " + std::to_string(*current.disposed_period) +
                                     ") and " + next.owner + " (acquired at " + std::to_string(next.acquired_period) + ")");
        } else if (*current.disposed_period > next.acquired_period) {
            int overlap = *current.disposed_period - next.acquired_period;
            result.issues.push_back("overlap of " + std::to_string(overlap) + " period(s) in custody between " +
                                     current.owner + " (disposed at " + std::to_string(*current.disposed_period) +
                                     ") and " + next.owner + " (acquired at " + std::to_string(next.acquired_period) + ")");
        }
    }

    result.valid = result.issues.empty();
    return result;
}

// =======================================================================
// PART 2: condition-score trend detection -- the identical real
// ordinary-least-squares regression Section 19.4 built for a retail
// inventory trend, applied here to a conservator's own periodic
// condition scores.
// =======================================================================
struct ConditionReading { int period = 0; double score = 0.0; };
struct TrendFit { double slope = 0.0, intercept = 0.0; };

TrendFit fit_condition_trend(const std::vector<ConditionReading>& readings) {
    double n = static_cast<double>(readings.size());
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (const auto& r : readings) {
        double x = static_cast<double>(r.period), y = r.score;
        sum_x += x; sum_y += y; sum_xy += x * y; sum_xx += x * x;
    }
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;
    return TrendFit{slope, intercept};
}

enum class ConditionTrendVerdict { STABLE, ACCELERATING_DETERIORATION, IMPROVING };

// Real, stated rate thresholds -- a condition score's own natural
// period-to-period noise should never, on its own, cross either bound;
// only a real, sustained trend should.
constexpr double DETERIORATION_RATE_THRESHOLD = -0.5;
constexpr double IMPROVEMENT_RATE_THRESHOLD = 0.5;

ConditionTrendVerdict classify_condition_trend(double slope) {
    if (slope <= DETERIORATION_RATE_THRESHOLD) return ConditionTrendVerdict::ACCELERATING_DETERIORATION;
    if (slope >= IMPROVEMENT_RATE_THRESHOLD) return ConditionTrendVerdict::IMPROVING;
    return ConditionTrendVerdict::STABLE;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 22.3: An Art-Condition Assessment and Provenance-Verification Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a fully contiguous, chronologically consistent 3-owner chain validates with "
                 "zero issues --\n";
    {
        std::vector<ProvenanceRecord> chain = {
            {"Estate of A. Ferreira", 0, 100},
            {"Kingsford Gallery", 100, 250},
            {"Private Collection (current)", 250, std::nullopt},
        };
        auto result = validate_provenance_chain(chain);
        CHECK(result.valid);
        CHECK(result.issues.empty());
        std::cout << "  3 owners with exactly contiguous custody periods (0-100, 100-250, 250-present) "
                     "validate with zero issues\n";
    }

    std::cout << "\n-- Test 2: a real gap and a real overlap in custody are each flagged with the exact "
                 "hand-computed size --\n";
    {
        std::vector<ProvenanceRecord> gapped_chain = {
            {"Estate of A. Ferreira", 0, 100},
            {"Kingsford Gallery", 120, 300},    // gap of 20 periods
            {"Private Collection (current)", 300, std::nullopt},
        };
        auto gap_result = validate_provenance_chain(gapped_chain);
        CHECK(!gap_result.valid);
        CHECK(gap_result.issues.size() == 1);
        CHECK(gap_result.issues[0].find("gap of 20") != std::string::npos);

        std::vector<ProvenanceRecord> overlapping_chain = {
            {"Estate of A. Ferreira", 0, 150},
            {"Kingsford Gallery", 100, 300},    // overlap of 50 periods
            {"Private Collection (current)", 300, std::nullopt},
        };
        auto overlap_result = validate_provenance_chain(overlapping_chain);
        CHECK(!overlap_result.valid);
        CHECK(overlap_result.issues.size() == 1);
        CHECK(overlap_result.issues[0].find("overlap of 50") != std::string::npos);

        std::cout << "  a 20-period gap between Ferreira's disposal and Kingsford's acquisition is flagged "
                     "with the exact size (\"gap of 20\"); a 50-period overlap between the same two "
                     "records in a second chain is flagged with the exact size (\"overlap of 50\")\n";
    }

    std::cout << "\n-- Test 3: an owner disposing before they ever acquired is flagged, and a chain with no "
                 "current owner at all is flagged --\n";
    {
        std::vector<ProvenanceRecord> impossible_chain = {
            {"Kingsford Gallery", 100, 50},   // disposed BEFORE acquiring -- impossible
        };
        auto result1 = validate_provenance_chain(impossible_chain);
        CHECK(!result1.valid);
        bool found_impossible = false;
        for (const auto& issue : result1.issues) {
            if (issue.find("does not come after") != std::string::npos) found_impossible = true;
        }
        CHECK(found_impossible);

        std::vector<ProvenanceRecord> no_current_owner = {
            {"Estate of A. Ferreira", 0, 100},
            {"Kingsford Gallery", 100, 250},   // every record has a disposal date -- no one currently owns it
        };
        auto result2 = validate_provenance_chain(no_current_owner);
        CHECK(!result2.valid);
        bool found_no_owner = false;
        for (const auto& issue : result2.issues) {
            if (issue.find("no current owner recorded") != std::string::npos) found_no_owner = true;
        }
        CHECK(found_no_owner);

        std::cout << "  Kingsford Gallery's own recorded disposal at period 50, before their own recorded "
                     "acquisition at period 100, is flagged as impossible; a separate chain where every "
                     "single owner has a disposal date recorded (no one currently owns the work) is "
                     "flagged as missing a current owner\n";
    }

    std::cout << "\n-- Test 4: two simultaneously open-ended owners are flagged, and an open-ended owner "
                 "who is NOT the chronologically last record is flagged --\n";
    {
        std::vector<ProvenanceRecord> two_current_owners = {
            {"Estate of A. Ferreira", 0, std::nullopt},
            {"Kingsford Gallery", 100, std::nullopt},
        };
        auto result1 = validate_provenance_chain(two_current_owners);
        CHECK(!result1.valid);
        bool found_two_current = false;
        for (const auto& issue : result1.issues) {
            if (issue.find("more than one owner recorded with no disposal period") != std::string::npos) found_two_current = true;
        }
        CHECK(found_two_current);

        std::vector<ProvenanceRecord> stale_current_owner = {
            {"Estate of A. Ferreira", 0, std::nullopt},   // claims to still be the current owner...
            {"Kingsford Gallery", 100, 250},              // ...but a LATER transaction is recorded regardless
        };
        auto result2 = validate_provenance_chain(stale_current_owner);
        CHECK(!result2.valid);
        bool found_stale = false;
        for (const auto& issue : result2.issues) {
            if (issue.find("is not the chronologically last owner") != std::string::npos) found_stale = true;
        }
        CHECK(found_stale);

        std::cout << "  a chain with 2 owners BOTH recorded with no disposal period is flagged (they "
                     "cannot both currently own the work); a chain where Ferreira's own record claims no "
                     "disposal period while a later Kingsford transaction is recorded anyway is flagged "
                     "as a stale, inconsistent current-owner claim\n";
    }

    std::cout << "\n-- Test 5: real least-squares condition-trend fitting matches hand-computed slope and "
                 "intercept exactly, and correctly classifies deterioration, stability, and improvement --\n";
    {
        std::vector<ConditionReading> declining = {{0, 100.0}, {10, 90.0}, {20, 80.0}, {30, 70.0}};
        auto decline_fit = fit_condition_trend(declining);
        // hand-computed: slope = -1.0 exactly, intercept = 100.0 exactly
        CHECK(std::abs(decline_fit.slope - (-1.0)) < 1e-9);
        CHECK(std::abs(decline_fit.intercept - 100.0) < 1e-9);
        CHECK(classify_condition_trend(decline_fit.slope) == ConditionTrendVerdict::ACCELERATING_DETERIORATION);

        std::vector<ConditionReading> flat_noisy = {{0, 95.0}, {10, 96.0}, {20, 94.0}, {30, 95.0}};
        auto flat_fit = fit_condition_trend(flat_noisy);
        // hand-computed: slope = -40 / 2000 = -0.02
        CHECK(std::abs(flat_fit.slope - (-0.02)) < 1e-9);
        CHECK(classify_condition_trend(flat_fit.slope) == ConditionTrendVerdict::STABLE);

        std::vector<ConditionReading> improving = {{0, 60.0}, {10, 68.0}, {20, 76.0}, {30, 84.0}};
        auto improve_fit = fit_condition_trend(improving);
        // hand-computed: slope = 1600 / 2000 = 0.8 exactly
        CHECK(std::abs(improve_fit.slope - 0.8) < 1e-9);
        CHECK(classify_condition_trend(improve_fit.slope) == ConditionTrendVerdict::IMPROVING);

        std::cout << "  a perfectly linear decline of 10 points per 10 periods fits to an exact slope of "
                   << decline_fit.slope << " (classified ACCELERATING_DETERIORATION); a flat, mildly noisy "
                     "series fits to a near-zero slope of " << flat_fit.slope << " (classified STABLE); a "
                     "perfectly linear improvement (consistent with a real conservation treatment) fits to "
                     "an exact slope of " << improve_fit.slope << " (classified IMPROVING)\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
