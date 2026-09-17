// Chapter 29.2 -- A second, independent real AML and card-fraud signal
// has nothing to do with WHICH amounts moved, only how often. A real
// account suddenly transacting far more frequently than any ordinary
// customer would -- several withdrawals in rapid succession, at
// different terminals -- is a well-documented real signal for a
// compromised card or account being drained quickly before it can be
// blocked, independent of whether any single amount looks unusual.
//
// Exactly like Section 27.2's own real tick-to-trade deadline check,
// every timing value in this file is a stated, labeled integer-minute
// parameter -- never a value read from a wall clock -- so this section's
// own locked self-test output is identical no matter when or how fast
// the machine running it actually is.
//
// A note on this section's own honest scope: this section counts
// TRANSACTION FREQUENCY only, deliberately independent of Section 29.1's
// own transaction-AMOUNT check -- a burst of small, ordinary purchases
// and a burst of large withdrawals are equally suspicious from a pure
// velocity standpoint, which is exactly why this section's own COMMON
// TRAP box warns against treating it as a substitute for Section 29.1's
// own amount-based check, or the reverse.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_transaction_velocity_anomaly_detection.cpp -o 02_transaction_velocity_anomaly_detection
// Run:     ./02_transaction_velocity_anomaly_detection

#include <algorithm>
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

// A stated capture minute, in place of a real wall-clock timestamp --
// the same reduction Section 27.2 already applied to microseconds and
// Section 28.2 already applied to whole days.
struct TimedTransaction {
    std::string account_id;
    int minute;
    std::string terminal_id;
};

struct VelocityResult {
    bool flagged = false;
    int max_count_in_window = 0;
    int window_start_minute = 0;
};

// Scans every possible rolling window of `window_minutes` (inclusive on
// both ends) within one account's own sorted transaction history,
// tracking the LARGEST count found in any window. A window is flagged
// only when its own count is STRICTLY GREATER than the stated maximum --
// exactly `max_count` transactions in a window is still within bounds,
// the same strict-inequality convention Section 27.2's own deadline
// check already established.
VelocityResult detect_velocity_anomaly(const std::vector<TimedTransaction>& account_history, int window_minutes,
                                        int max_count) {
    VelocityResult result;
    for (std::size_t i = 0; i < account_history.size(); ++i) {
        int window_start = account_history[i].minute;
        int window_end = window_start + window_minutes;
        int count = 0;
        for (std::size_t j = i; j < account_history.size() && account_history[j].minute <= window_end; ++j) {
            count++;
        }
        if (count > result.max_count_in_window) {
            result.max_count_in_window = count;
            result.window_start_minute = window_start;
        }
    }
    result.flagged = result.max_count_in_window > max_count;
    return result;
}

std::vector<std::string> flag_velocity_accounts(std::vector<TimedTransaction> all_transactions, int window_minutes,
                                                  int max_count) {
    std::map<std::string, std::vector<TimedTransaction>> by_account;
    for (const auto& t : all_transactions) by_account[t.account_id].push_back(t);
    std::vector<std::string> flagged;
    for (auto& [account_id, history] : by_account) {
        std::sort(history.begin(), history.end(), [](const auto& a, const auto& b) { return a.minute < b.minute; });
        if (detect_velocity_anomaly(history, window_minutes, max_count).flagged) flagged.push_back(account_id);
    }
    return flagged;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 29.2: Transaction Velocity Anomaly Detection\n";
    std::cout << "========================================================\n\n";

    const int WINDOW_MINUTES = 30;
    const int MAX_COUNT = 3;

    // -- Test 1: two ordinary transactions 20 minutes apart -- never
    // more than 2 in any window, well within bounds. --
    {
        std::vector<TimedTransaction> history = {{"ACC-1", 0, "T-A"}, {"ACC-1", 20, "T-B"}};
        auto r = detect_velocity_anomaly(history, WINDOW_MINUTES, MAX_COUNT);
        std::cout << "-- Test 1: 2 transactions, 20 minutes apart -- flagged=" << (r.flagged ? "YES" : "NO")
                  << ", max_count_in_window=" << r.max_count_in_window << " --\n";
        CHECK(!r.flagged);
        CHECK(r.max_count_in_window == 2);
    }

    // -- Test 2: a real burst -- 4 transactions within a 10-minute span,
    // well past the stated maximum of 3 within a 30-minute window. --
    {
        std::vector<TimedTransaction> history = {
            {"ACC-2", 0, "T-A"}, {"ACC-2", 3, "T-B"}, {"ACC-2", 6, "T-C"}, {"ACC-2", 9, "T-D"},
        };
        auto r = detect_velocity_anomaly(history, WINDOW_MINUTES, MAX_COUNT);
        std::cout << "-- Test 2: 4 transactions within 10 minutes -- flagged=" << (r.flagged ? "YES" : "NO")
                  << ", max_count_in_window=" << r.max_count_in_window << " --\n";
        CHECK(r.flagged);
        CHECK(r.max_count_in_window == 4);
    }

    // -- Test 3: the exact boundary -- exactly 3 transactions within the
    // 30-minute window is still within stated bounds, not flagged. --
    {
        std::vector<TimedTransaction> history = {{"ACC-3", 0, "T-A"}, {"ACC-3", 10, "T-B"}, {"ACC-3", 20, "T-C"}};
        auto r = detect_velocity_anomaly(history, WINDOW_MINUTES, MAX_COUNT);
        std::cout << "-- Test 3: exactly 3 transactions within 30 minutes -- flagged="
                  << (r.flagged ? "YES" : "NO") << ", max_count_in_window=" << r.max_count_in_window << " --\n";
        CHECK(!r.flagged);
        CHECK(r.max_count_in_window == 3);
    }

    // -- Test 4: one transaction past that same boundary -- 4
    // transactions within the identical 30-minute window, now flagged. --
    {
        std::vector<TimedTransaction> history = {
            {"ACC-4", 0, "T-A"}, {"ACC-4", 10, "T-B"}, {"ACC-4", 20, "T-C"}, {"ACC-4", 30, "T-D"},
        };
        auto r = detect_velocity_anomaly(history, WINDOW_MINUTES, MAX_COUNT);
        std::cout << "-- Test 4: 4 transactions within 30 minutes -- flagged=" << (r.flagged ? "YES" : "NO")
                  << ", max_count_in_window=" << r.max_count_in_window << " --\n";
        CHECK(r.flagged);
        CHECK(r.max_count_in_window == 4);
    }

    // -- Test 5: 4 transactions total, but properly spread across 45
    // minutes at a steady 15-minute pace -- no single 30-minute window
    // ever contains more than 3, confirming a high total count over a
    // long enough span is not, by itself, a velocity anomaly. --
    {
        std::vector<TimedTransaction> history = {
            {"ACC-5", 0, "T-A"}, {"ACC-5", 15, "T-B"}, {"ACC-5", 30, "T-C"}, {"ACC-5", 45, "T-D"},
        };
        auto r = detect_velocity_anomaly(history, WINDOW_MINUTES, MAX_COUNT);
        std::cout << "-- Test 5: 4 transactions steadily spread across 45 minutes -- flagged="
                  << (r.flagged ? "YES" : "NO") << ", max_count_in_window=" << r.max_count_in_window << " --\n";
        CHECK(!r.flagged);
        CHECK(r.max_count_in_window == 3);
    }

    // -- Test 6: a mixed batch across two accounts -- only the bursting
    // account is named. --
    {
        std::vector<TimedTransaction> mixed = {
            {"ACC-6A", 0, "T-A"}, {"ACC-6A", 3, "T-B"}, {"ACC-6A", 6, "T-C"}, {"ACC-6A", 9, "T-D"},
            {"ACC-6B", 0, "T-A"}, {"ACC-6B", 25, "T-B"},
        };
        auto flagged = flag_velocity_accounts(mixed, WINDOW_MINUTES, MAX_COUNT);
        std::cout << "-- Test 6: two accounts, one bursting, one normal -- flagged=[";
        for (std::size_t i = 0; i < flagged.size(); i++) std::cout << (i ? ", " : "") << flagged[i];
        std::cout << "] --\n";
        CHECK(flagged.size() == 1);
        CHECK(flagged[0] == "ACC-6A");
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
