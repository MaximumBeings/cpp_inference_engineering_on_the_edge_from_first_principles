// Chapter 29.1 -- Every prior finance chapter in this book evaluated ONE
// document, ONE order-book snapshot, or ONE claim at a time. This
// chapter's own real constraint is different: catching a real, well-
// documented anti-money-laundering pattern requires looking at a
// SEQUENCE of transactions together, since no single transaction in the
// pattern looks suspicious on its own.
//
// "Structuring" (sometimes called "smurfing") is a real, specifically
// named federal offense under the US Bank Secrecy Act: deliberately
// breaking a large cash transaction into several smaller ones, each kept
// under the real $10,000 Currency Transaction Report (CTR) threshold
// (31 CFR 1010.311), specifically to avoid the reporting the full amount
// would trigger. No individual transaction in the pattern is suspicious
// by itself -- $4,000 is an ordinary cash transaction -- the pattern is
// only visible across several transactions from the SAME account within
// a real, short window of time.
//
// A note on this section's own honest scope, precise about the real
// regulation it models: the real CTR threshold applies to a cash
// transaction "in excess of" $10,000 -- STRICTLY more than $10,000, not
// $10,000 exactly -- and this section's own boundary tests confirm that
// precise real legal distinction directly, the same care this book has
// given every other real, published threshold since Chapter 26.1's ABA
// checksum. This section's own COMMON TRAP box returns to a related
// scope limitation.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_aml_structuring_detection.cpp -o 01_aml_structuring_detection
// Run:     ./01_aml_structuring_detection

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// A single real cash transaction -- `day` is a whole-number day count
// since an arbitrary epoch, the same stated-integer-timeline convention
// Section 28.2 already used for a photo's own capture day.
struct Transaction {
    std::string account_id;
    int day;
    double amount;
};

struct StructuringResult {
    bool flagged = false;
    double window_sum = 0.0;
    int transaction_count = 0;
    int window_start_day = 0;
};

// Scans every possible rolling window of `window_days` (inclusive on
// both ends) within ONE account's own sorted transaction history. A
// window is flagged only when BOTH real conditions hold at once: every
// individual transaction inside it stays strictly under the CTR
// threshold (the whole point of structuring is staying under it), AND
// the window's own sum is STRICTLY GREATER than the threshold -- the
// exact real regulatory boundary, not a rounded approximation of it.
StructuringResult detect_structuring(const std::vector<Transaction>& account_history, int window_days,
                                      double ctr_threshold) {
    for (std::size_t i = 0; i < account_history.size(); ++i) {
        int window_start = account_history[i].day;
        int window_end = window_start + window_days - 1;
        double sum = 0.0;
        int count = 0;
        bool any_at_or_over_threshold = false;
        for (std::size_t j = i; j < account_history.size() && account_history[j].day <= window_end; ++j) {
            sum += account_history[j].amount;
            count++;
            if (account_history[j].amount >= ctr_threshold) any_at_or_over_threshold = true;
        }
        if (!any_at_or_over_threshold && sum > ctr_threshold) {
            return {true, sum, count, window_start};
        }
    }
    return {false, 0.0, 0, 0};
}

// Groups a mixed batch of transactions by account and applies
// `detect_structuring` independently to each account's own history --
// one account's own pattern of deposits never contaminates another's.
std::vector<std::string> flag_structuring_accounts(std::vector<Transaction> all_transactions, int window_days,
                                                     double ctr_threshold) {
    std::map<std::string, std::vector<Transaction>> by_account;
    for (const auto& t : all_transactions) by_account[t.account_id].push_back(t);
    std::vector<std::string> flagged;
    for (auto& [account_id, history] : by_account) {
        std::sort(history.begin(), history.end(), [](const auto& a, const auto& b) { return a.day < b.day; });
        if (detect_structuring(history, window_days, ctr_threshold).flagged) flagged.push_back(account_id);
    }
    return flagged;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 29.1: AML Structuring (Smurfing) Detection\n";
    std::cout << "========================================================\n\n";

    const double CTR_THRESHOLD = 10000.0;
    const int WINDOW_DAYS = 3;

    // -- Test 1: a single transaction well over the real CTR threshold
    // is NOT flagged as structuring -- it would trigger a direct CTR on
    // its own, which is the system working correctly, not evasion. --
    {
        std::vector<Transaction> history = {{"ACC-1", 1, 15000.0}};
        auto r = detect_structuring(history, WINDOW_DAYS, CTR_THRESHOLD);
        std::cout << "-- Test 1: a single $15,000 transaction -- flagged=" << (r.flagged ? "YES" : "NO")
                  << " (a direct CTR case, not structuring) --\n";
        CHECK(!r.flagged);
    }

    // -- Test 2: three $4,000 transactions within the 3-day window --
    // each individually under threshold, but summing well past it. --
    {
        std::vector<Transaction> history = {{"ACC-2", 1, 4000.0}, {"ACC-2", 2, 4000.0}, {"ACC-2", 3, 4000.0}};
        auto r = detect_structuring(history, WINDOW_DAYS, CTR_THRESHOLD);
        std::cout << "-- Test 2: three $4,000 transactions across days 1-3 -- flagged="
                  << (r.flagged ? "YES" : "NO") << ", window_sum=$" << r.window_sum << ", count="
                  << r.transaction_count << " --\n";
        CHECK(r.flagged);
        CHECK(std::abs(r.window_sum - 12000.0) < 1e-9);
        CHECK(r.transaction_count == 3);
    }

    // -- Test 3: three $2,000 transactions within the same window --
    // genuinely small, legitimate transactions that never approach the
    // threshold at all. --
    {
        std::vector<Transaction> history = {{"ACC-3", 1, 2000.0}, {"ACC-3", 2, 2000.0}, {"ACC-3", 3, 2000.0}};
        auto r = detect_structuring(history, WINDOW_DAYS, CTR_THRESHOLD);
        std::cout << "-- Test 3: three $2,000 transactions across days 1-3 -- flagged="
                  << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(!r.flagged);
    }

    // -- Test 4: the identical three $4,000 amounts as Test 2, but
    // spread across 9 days instead of 3 -- no single 3-day window ever
    // contains more than one of them, so the pattern is correctly not
    // flagged, confirming the check is genuinely a ROLLING window, not a
    // whole-history sum. --
    {
        std::vector<Transaction> history = {{"ACC-4", 1, 4000.0}, {"ACC-4", 5, 4000.0}, {"ACC-4", 9, 4000.0}};
        auto r = detect_structuring(history, WINDOW_DAYS, CTR_THRESHOLD);
        std::cout << "-- Test 4: the same three $4,000 amounts spread across days 1, 5, 9 -- flagged="
                  << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(!r.flagged);
    }

    // -- Test 5: the real regulatory boundary itself -- two transactions
    // summing to EXACTLY $10,000.00 are NOT flagged, matching the real
    // legal threshold of "in excess of" $10,000, not "at or above" it. --
    {
        std::vector<Transaction> history = {{"ACC-5", 1, 5000.0}, {"ACC-5", 2, 5000.0}};
        auto r = detect_structuring(history, WINDOW_DAYS, CTR_THRESHOLD);
        std::cout << "-- Test 5: two transactions summing to exactly $10,000.00 -- flagged="
                  << (r.flagged ? "YES" : "NO") << " (matches the real 'in excess of' legal standard) --\n";
        CHECK(!r.flagged);
    }

    // -- Test 6: one cent past that same boundary -- now genuinely in
    // excess of the threshold, correctly flagged. --
    {
        std::vector<Transaction> history = {{"ACC-6", 1, 5000.0}, {"ACC-6", 2, 5000.01}};
        auto r = detect_structuring(history, WINDOW_DAYS, CTR_THRESHOLD);
        std::cout << "-- Test 6: two transactions summing to $10,000.01 -- flagged="
                  << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(r.flagged);
    }

    // -- Test 7: a mixed batch spanning two accounts -- only the account
    // actually exhibiting the pattern is named, never the clean one. --
    {
        std::vector<Transaction> mixed = {
            {"ACC-7A", 1, 4000.0}, {"ACC-7A", 2, 4000.0}, {"ACC-7A", 3, 4000.0},
            {"ACC-7B", 1, 500.0}, {"ACC-7B", 2, 500.0},
        };
        auto flagged = flag_structuring_accounts(mixed, WINDOW_DAYS, CTR_THRESHOLD);
        std::cout << "-- Test 7: two accounts, one structuring, one clean -- flagged=[";
        for (std::size_t i = 0; i < flagged.size(); i++) std::cout << (i ? ", " : "") << flagged[i];
        std::cout << "] --\n";
        CHECK(flagged.size() == 1);
        CHECK(flagged[0] == "ACC-7A");
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
