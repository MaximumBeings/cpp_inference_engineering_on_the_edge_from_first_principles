// Chapter 29.4 -- This chapter's own capstone, and the FOURTH and final
// financial-industry chapter this book set out to build: a real,
// auditable transaction-monitoring engine that combines Section 29.1's
// structuring signal, Section 29.2's velocity signal, and Section 29.3's
// impossible-travel signal into one per-account disposition. The central
// discipline reapplies this book's own recurring never-suppress-a-
// named-flag structure one final time, with a real, deliberate legal
// boundary specific to this domain: under the real US Bank Secrecy Act,
// a Suspicious Activity Report (SAR) can only ever be FILED by a human
// compliance officer at a regulated institution -- no automated system
// may file one on its own. This engine's own strongest disposition is
// therefore named, precisely, `FILE_SAR_RECOMMENDED`, never
// `FILE_SAR` -- a real, legally meaningful distinction, not a stylistic
// choice.
//
// A note on this section's own honest scope: exactly like Chapter 28.4's
// own claims-disposition engine, this engine decides only where an
// account is ROUTED for human attention, never any account-level action
// (freezing funds, closing an account, blocking a card) on its own.
// This section's own COMMON TRAP box returns to exactly this
// distinction, one final time for this book's four-chapter run through
// real financial-industry deployments.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_transaction_monitoring_disposition_engine.cpp -o 04_transaction_monitoring_disposition_engine
// Run:     ./04_transaction_monitoring_disposition_engine

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

// A bundle of ALREADY-COMPUTED per-account signals from Sections
// 29.1-29.3 -- the same "named, pre-computed signal" convention this
// book has used in every capstone since Chapter 26.4's own RiskSignal.
struct AccountSignals {
    std::string account_id;
    bool structuring = false;
    bool velocity_anomaly = false;
    bool impossible_travel = false;
};

enum class MonitoringDisposition { NO_ACTION, INTERNAL_REVIEW, FILE_SAR_RECOMMENDED };

std::string to_string(MonitoringDisposition d) {
    switch (d) {
        case MonitoringDisposition::NO_ACTION: return "NO_ACTION";
        case MonitoringDisposition::INTERNAL_REVIEW: return "INTERNAL_REVIEW";
        case MonitoringDisposition::FILE_SAR_RECOMMENDED: return "FILE_SAR_RECOMMENDED";
    }
    return "UNKNOWN";
}

struct MonitoringResult {
    std::string account_id;
    MonitoringDisposition disposition;
    std::vector<std::string> reasons;  // the full, named audit trail
};

// Zero named signals: NO_ACTION. Exactly one: INTERNAL_REVIEW -- worth a
// human glance, not yet a real pattern. Two or more INDEPENDENT signal
// types firing together on the SAME account is this engine's own
// strongest real disposition, FILE_SAR_RECOMMENDED -- still only ever a
// recommendation, since only a human compliance officer may actually
// file a real SAR under real law.
MonitoringResult assess_account(const AccountSignals& s) {
    MonitoringResult r;
    r.account_id = s.account_id;
    int count = 0;
    if (s.structuring) { r.reasons.push_back("structuring pattern detected (Section 29.1)"); count++; }
    if (s.velocity_anomaly) { r.reasons.push_back("transaction velocity anomaly detected (Section 29.2)"); count++; }
    if (s.impossible_travel) { r.reasons.push_back("impossible travel pattern detected (Section 29.3)"); count++; }

    if (count >= 2) {
        r.disposition = MonitoringDisposition::FILE_SAR_RECOMMENDED;
    } else if (count == 1) {
        r.disposition = MonitoringDisposition::INTERNAL_REVIEW;
    } else {
        r.disposition = MonitoringDisposition::NO_ACTION;
    }
    return r;
}

std::vector<MonitoringResult> assess_accounts(const std::vector<AccountSignals>& accounts) {
    std::vector<MonitoringResult> results;
    for (const auto& a : accounts) results.push_back(assess_account(a));
    return results;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 29.4: The Transaction Monitoring Disposition Engine\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: a completely clean account -- no action, empty audit
    // trail. --
    {
        auto r = assess_account({"ACC-1", false, false, false});
        std::cout << "-- Test 1: clean account -- disposition=" << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == MonitoringDisposition::NO_ACTION);
        CHECK(r.reasons.empty());
    }

    // -- Test 2: exactly one signal fired, tried in turn for each of the
    // three independent signal types -- each alone is INTERNAL_REVIEW,
    // confirming all three are weighted identically. --
    {
        auto structuring_only = assess_account({"ACC-2A", true, false, false});
        auto velocity_only = assess_account({"ACC-2B", false, true, false});
        auto travel_only = assess_account({"ACC-2C", false, false, true});
        std::cout << "-- Test 2: exactly one signal fired -- structuring-only="
                  << to_string(structuring_only.disposition) << ", velocity-only="
                  << to_string(velocity_only.disposition) << ", travel-only=" << to_string(travel_only.disposition)
                  << " --\n";
        CHECK(structuring_only.disposition == MonitoringDisposition::INTERNAL_REVIEW);
        CHECK(velocity_only.disposition == MonitoringDisposition::INTERNAL_REVIEW);
        CHECK(travel_only.disposition == MonitoringDisposition::INTERNAL_REVIEW);
    }

    // -- Test 3: exactly two signals fired -- the boundary between
    // INTERNAL_REVIEW and this engine's own strongest disposition. --
    {
        auto r = assess_account({"ACC-3", true, true, false});
        std::cout << "-- Test 3: exactly two signals fired (structuring + velocity) -- disposition="
                  << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == MonitoringDisposition::FILE_SAR_RECOMMENDED);
        CHECK(r.reasons.size() == 2);
    }

    // -- Test 4: all three signals fired at once -- still
    // FILE_SAR_RECOMMENDED, this engine's own disposition has no fourth,
    // even stronger tier beyond it. --
    {
        auto r = assess_account({"ACC-4", true, true, true});
        std::cout << "-- Test 4: all three signals fired -- disposition=" << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == MonitoringDisposition::FILE_SAR_RECOMMENDED);
        CHECK(r.reasons.size() == 3);
    }

    // -- Test 5 (structural never-auto-file guarantee): this engine's
    // own strongest disposition name is checked, precisely, to end in
    // "RECOMMENDED" rather than stating a filing has occurred -- a real,
    // legally meaningful distinction, not incidental wording. --
    {
        std::string strongest = to_string(MonitoringDisposition::FILE_SAR_RECOMMENDED);
        bool ends_in_recommended = strongest.size() >= 11 && strongest.substr(strongest.size() - 11) == "RECOMMENDED";
        std::cout << "-- Test 5: strongest disposition name = \"" << strongest << "\" -- ends in RECOMMENDED: "
                  << (ends_in_recommended ? "YES (correct)" : "NO (WRONG)") << " --\n";
        CHECK(ends_in_recommended);
        CHECK(strongest != "FILE_SAR");
    }

    // -- Test 6: the disposition's own audit trail names the EXACT
    // signals responsible, not merely an aggregate count. --
    {
        auto r = assess_account({"ACC-6", true, false, true});
        std::cout << "-- Test 6: audit trail -- ";
        for (const auto& reason : r.reasons) std::cout << "[" << reason << "] ";
        std::cout << "--\n";
        CHECK(r.reasons.size() == 2);
        CHECK(r.reasons[0].find("structuring") != std::string::npos);
        CHECK(r.reasons[1].find("impossible travel") != std::string::npos);
        CHECK(r.reasons[0].find("velocity") == std::string::npos);
    }

    // -- Test 7: a full multi-account batch -- each account's own
    // disposition is computed completely independently of every other
    // account in the batch, the same real, sequence-aware but
    // per-account-isolated discipline every section in this chapter has
    // used since Section 29.1's own multi-account structuring test. --
    {
        std::vector<AccountSignals> accounts = {
            {"ACC-7A", false, false, false},
            {"ACC-7B", true, false, false},
            {"ACC-7C", true, true, false},
        };
        auto results = assess_accounts(accounts);
        std::cout << "-- Test 7: 3-account batch -- ";
        for (const auto& r : results) std::cout << r.account_id << "=" << to_string(r.disposition) << " ";
        std::cout << "--\n";
        CHECK(results.size() == 3);
        CHECK(results[0].disposition == MonitoringDisposition::NO_ACTION);
        CHECK(results[1].disposition == MonitoringDisposition::INTERNAL_REVIEW);
        CHECK(results[2].disposition == MonitoringDisposition::FILE_SAR_RECOMMENDED);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
