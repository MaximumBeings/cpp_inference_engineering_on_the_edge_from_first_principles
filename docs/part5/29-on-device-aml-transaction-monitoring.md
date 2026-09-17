# Chapter 29: On-Device AML Transaction Anomaly Monitoring at ATMs and POS Terminals

**What you will understand by the end of this chapter:**

- How to detect "structuring" (smurfing) -- a real, specifically named federal offense under the US Bank Secrecy Act -- by scanning a rolling window across a SEQUENCE of transactions, since no individual transaction in the pattern is suspicious on its own.
- How to build a real transaction-velocity anomaly check, reapplying Section 27.2's own "stated parameters, never a wall clock" discipline to a frequency-based signal instead of a deadline.
- How to build a real "impossible travel" check from the real, published Haversine great-circle-distance formula, catching a required travel speed that could not possibly be genuine, while deliberately staying generous enough never to flag a real traveler.
- How to combine three independent, sequence-based signals into one per-account disposition, closing this book's four-chapter run through real financial-industry deployments with a final, real, legally precise never-suppress-a-flag discipline: a Suspicious Activity Report can only ever be RECOMMENDED by this engine, never filed by it.

**What you need to know first:**

- Section 27.2's own discipline of deriving timing behavior entirely from stated, labeled parameters rather than a wall clock is reapplied in Section 29.2, this time to a transaction-frequency window instead of a processing deadline.
- Section 28.2's own stated-integer-day-count convention for a claim's own timeline is reapplied in Section 29.1, this time to a rolling structuring-detection window.
- Chapter 26.1's own real, published ABA checksum and Chapter 27.1's own real, published microprice formula are joined in this chapter by a third real, published formula: the Haversine great-circle-distance formula, used in Section 29.3.
- This chapter's own central shift from every prior finance chapter: every check in this chapter evaluates a SEQUENCE of transactions from the same account together, since no single transaction in any of this chapter's three real patterns is suspicious in isolation.

---

Chapter 28 turned to a claim's own photographic and temporal evidence. This chapter closes this book's four-chapter run through real financial-industry deployments with a genuinely different real constraint once more: catching a pattern that only becomes visible across a SEQUENCE of transactions from the same account, where no single transaction in the sequence is suspicious by itself. Each of this chapter's four sections builds one real, independent, sequence-based AML signal, and closes with a capstone that combines all three into one auditable, per-account disposition -- never an automated filing, since real law reserves that decision for a human.

## 29.1 AML Structuring (Smurfing) Detection

### Intuition

"Structuring" is a real, specifically named offense under the US Bank Secrecy Act: deliberately breaking a large cash transaction into several smaller ones, each kept under the real $10,000 Currency Transaction Report threshold, specifically to avoid the reporting the full amount would trigger. No individual transaction in the pattern looks suspicious -- the pattern is only visible across several transactions from the same account within a real, short window of time.

### The Concept, In Detail

`detect_structuring` scans every possible rolling window within one account's own sorted transaction history, flagging a window only when BOTH real conditions hold: every individual transaction inside it stays strictly under the real CTR threshold, and the window's own sum is strictly greater than it. Tests 5 and 6 confirm this section's own precise real legal boundary: the actual US regulation applies to a transaction "in excess of" $10,000 -- strictly more, not $10,000 exactly -- and this section's own code matches that real distinction precisely rather than rounding it away. Test 4 confirms the check is a genuine ROLLING window, not a whole-history sum: the identical three amounts that trigger a flag when clustered within the window do not trigger one when spread across nine days instead.

Test 7 confirms `flag_structuring_accounts`'s own per-account independence: a mixed batch spanning two accounts flags only the one actually exhibiting the pattern, by name.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_aml_structuring_detection.cpp -o 01_aml_structuring_detection
./01_aml_structuring_detection
```

**Sample input:** a single transaction well over the real CTR threshold, checked to be a direct CTR case rather than structuring; three transactions each under the threshold but summing well past it within the stated window, checked to be flagged; the identical pattern of small transactions with a sum safely under the threshold, checked as legitimate; the identical flagged amounts spread across a much longer span, checked as NOT flagged since no single window contains enough of them; the exact real regulatory boundary of $10,000.00 exactly, and one cent past it; and a mixed two-account batch checked to flag only the account actually exhibiting the pattern.

```text
========================================================
Chapter 29.1: AML Structuring (Smurfing) Detection
========================================================

-- Test 1: a single $15,000 transaction -- flagged=NO (a direct CTR case, not structuring) --
-- Test 2: three $4,000 transactions across days 1-3 -- flagged=YES, window_sum=$12000, count=3 --
-- Test 3: three $2,000 transactions across days 1-3 -- flagged=NO --
-- Test 4: the same three $4,000 amounts spread across days 1, 5, 9 -- flagged=NO --
-- Test 5: two transactions summing to exactly $10,000.00 -- flagged=NO (matches the real 'in excess of' legal standard) --
-- Test 6: two transactions summing to $10,000.01 -- flagged=YES --
-- Test 7: two accounts, one structuring, one clean -- flagged=[ACC-7A] --

10/10 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating this section's own function as a complete real AML structuring detector"
    A real, deployed structuring detector considers many additional real signals this section's own narrow, illustrative function does not: transactions across MULTIPLE accounts controlled by the same real person (a well-documented refinement real structuring schemes use to spread transactions even further), transactions at multiple different branches or ATMs specifically chosen to avoid a single teller's own suspicion, and a real employee's own trained judgment about a customer's stated purpose for a transaction. `detect_structuring` catches exactly one real, narrow, and well-documented pattern -- multiple transactions from the SAME account, each individually under the threshold, summing past it within a short window -- and should be understood as one real signal among many a real compliance program would need, not a complete real detection system on its own.

## 29.2 Transaction Velocity Anomaly Detection

### Intuition

A second, independent real signal has nothing to do with WHICH amounts moved, only how often. A real account suddenly transacting far more frequently than any ordinary customer would is a well-documented real signal for a compromised card or account being drained quickly, independent of whether any single amount looks unusual.

### The Concept, In Detail

`detect_velocity_anomaly` reapplies Section 27.2's own "stated parameters, never a wall clock" discipline to a frequency count instead of a deadline: every timing value in this file is a stated integer minute, and the function scans every possible rolling window in one account's own sorted history, flagging only when a window's own transaction count is strictly greater than a stated maximum. Tests 3 and 4 confirm the exact boundary of that maximum: exactly the stated count within a window is still within bounds, one more is not. Test 5 confirms a genuinely high total transaction count, properly spread out over a long enough span, is correctly NOT a velocity anomaly on its own -- the check is about a real BURST, not a real total.

Test 6 confirms `flag_velocity_accounts`'s own per-account independence, the same discipline Section 29.1 already established for a different signal entirely.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_transaction_velocity_anomaly_detection.cpp -o 02_transaction_velocity_anomaly_detection
./02_transaction_velocity_anomaly_detection
```

**Sample input:** two ordinary transactions well spaced apart, checked as normal; a real burst of four transactions within a short span, checked to be flagged with the exact maximum count found; the exact stated maximum count within the window, and one transaction past it, each checked at the precise boundary; four transactions properly spread across a longer span at a steady pace, checked as NOT a velocity anomaly despite the same total count as the flagged burst; and a mixed two-account batch checked to flag only the actually bursting account.

```text
========================================================
Chapter 29.2: Transaction Velocity Anomaly Detection
========================================================

-- Test 1: 2 transactions, 20 minutes apart -- flagged=NO, max_count_in_window=2 --
-- Test 2: 4 transactions within 10 minutes -- flagged=YES, max_count_in_window=4 --
-- Test 3: exactly 3 transactions within 30 minutes -- flagged=NO, max_count_in_window=3 --
-- Test 4: 4 transactions within 30 minutes -- flagged=YES, max_count_in_window=4 --
-- Test 5: 4 transactions steadily spread across 45 minutes -- flagged=NO, max_count_in_window=3 --
-- Test 6: two accounts, one bursting, one normal -- flagged=[ACC-6A] --

12/12 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a velocity anomaly as interchangeable with Section 29.1's own structuring signal"
    Section 29.1's structuring check and this section's own velocity check are deliberately independent: a burst of several small, ordinary purchases at different terminals is a real velocity anomaly with no structuring signal attached to it at all (no amount ever approaches the real CTR threshold), while a slow, patient structuring scheme spread across exactly the stated window with only two or three transactions total triggers Section 29.1's own check without ever coming close to this section's own frequency threshold. Treating either signal as a stand-in for the other -- assuming a clean velocity check means an account cannot be structuring, or the reverse -- would miss real, well-documented patterns that only this chapter's OTHER independent signal is actually built to catch.

## 29.3 Impossible Travel Geographic Detection

### Intuition

A third, independent real signal asks a different question again: could the same physical person plausibly have been at both of two transaction locations, given how little time passed between them? "Impossible travel" is a real, well-documented technique used across both AML and account-security fraud detection, built on a deliberately generous real bound -- fast enough to cover an actual commercial flight -- so a genuine cross-country traveler is never flagged.

### The Concept, In Detail

`check_travel_pair` computes the real Haversine great-circle distance between two consecutive transaction locations -- a real, published formula, exactly the same "real, not invented for this book" standard Chapter 26.1's ABA checksum and Chapter 27.1's microprice formula already met -- divides by the real elapsed time, and flags only when the required speed strictly exceeds a stated, deliberately generous maximum of 900 km/h, the real cruising speed of a typical commercial jet. Test 2 confirms a genuinely plausible cross-country flight in 6 real hours is correctly NOT flagged, while Test 3 confirms the identical distance in 30 minutes is a real, genuine impossibility. Tests 4 and 5 confirm the exact speed boundary, constructed algorithmically from the real distance itself rather than a fragile hardcoded value, so the boundary check remains exact regardless of which two real cities it uses.

Test 6 confirms this check is genuinely about SPEED, not raw distance: a much shorter real distance (London to Paris) still triggers a flag when the elapsed time is short enough, and Test 7 confirms the same per-account independence this chapter's other two sections already established.

### Code and Verification

```cpp
// Chapter 29.3 -- A third, independent real AML and card-fraud signal
// asks a different question again: could the SAME physical person
// plausibly have been at both of two transaction locations, given how
// little time passed between them? "Impossible travel" is a real, well-
// documented technique used across both AML transaction monitoring and
// account-security fraud detection: compute the real great-circle
// distance between two consecutive transaction locations, divide by the
// real elapsed time, and compare the required speed against a
// deliberately GENEROUS real bound -- fast enough to cover an actual
// commercial flight, so a genuine cross-country traveler is never
// flagged. Only a required speed beyond even that generous bound is
// truly IMPOSSIBLE, not merely unusual, which is exactly what this
// section's own name promises and nothing more.
//
// The real distance calculation is the Haversine formula -- a real,
// published great-circle-distance formula that has computed distances
// between two points on a sphere since well before any digital computer
// existed, exactly the same "real, published formula, not invented for
// this book" standard Chapter 27.1's own microprice formula and Chapter
// 26.1's own ABA checksum already met.
//
// A note on this section's own honest scope: MAX_PLAUSIBLE_SPEED_KMH is
// stated at 900 km/h -- the real cruising speed of a typical commercial
// jet -- specifically chosen to be generous enough that no genuine
// traveler is ever flagged. This section's own COMMON TRAP box returns
// to exactly what this generosity trades away.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_impossible_travel_geographic_detection.cpp -o 03_impossible_travel_geographic_detection
// Run:     ./03_impossible_travel_geographic_detection

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numbers>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
static bool near_eq(double a, double b, double eps) { return std::fabs(a - b) < eps; }
#define CHECK_NEAR(a, b, eps) CHECK(near_eq((a), (b), (eps)))

// A transaction location, at a stated capture minute (a double, since
// this section's own real speed calculation genuinely needs continuous
// time, not a whole-minute reduction the way Section 29.2's frequency
// count did).
struct GeoTransaction {
    std::string account_id;
    double minute;
    double lat_deg;
    double lon_deg;
};

// The real, published Haversine great-circle-distance formula. R is
// Earth's own real mean radius in kilometers.
double haversine_distance_km(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
    constexpr double R = 6371.0;
    double phi1 = lat1_deg * std::numbers::pi / 180.0;
    double phi2 = lat2_deg * std::numbers::pi / 180.0;
    double dphi = (lat2_deg - lat1_deg) * std::numbers::pi / 180.0;
    double dlambda = (lon2_deg - lon1_deg) * std::numbers::pi / 180.0;
    double a = std::sin(dphi / 2.0) * std::sin(dphi / 2.0) +
               std::cos(phi1) * std::cos(phi2) * std::sin(dlambda / 2.0) * std::sin(dlambda / 2.0);
    double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
    return R * c;
}

struct TravelResult {
    bool flagged = false;
    double distance_km = 0.0;
    double required_speed_kmh = 0.0;
};

// Checks ONE consecutive pair of transactions from the same account.
// Flagged only when the required speed to cover the real distance in
// the real elapsed time is STRICTLY GREATER than the stated maximum
// plausible speed -- exactly at that speed is still, if barely, real.
TravelResult check_travel_pair(const GeoTransaction& earlier, const GeoTransaction& later,
                                double max_speed_kmh) {
    TravelResult r;
    r.distance_km = haversine_distance_km(earlier.lat_deg, earlier.lon_deg, later.lat_deg, later.lon_deg);
    double elapsed_hours = (later.minute - earlier.minute) / 60.0;
    r.required_speed_kmh = (elapsed_hours > 0.0) ? r.distance_km / elapsed_hours : 0.0;
    r.flagged = r.required_speed_kmh > max_speed_kmh;
    return r;
}

// Scans every consecutive pair in one account's own sorted transaction
// history, returning the first pair found to require an impossible
// speed.
TravelResult detect_impossible_travel(const std::vector<GeoTransaction>& account_history, double max_speed_kmh) {
    for (std::size_t i = 0; i + 1 < account_history.size(); ++i) {
        auto r = check_travel_pair(account_history[i], account_history[i + 1], max_speed_kmh);
        if (r.flagged) return r;
    }
    return {};
}

std::vector<std::string> flag_impossible_travel_accounts(std::vector<GeoTransaction> all_transactions,
                                                           double max_speed_kmh) {
    std::map<std::string, std::vector<GeoTransaction>> by_account;
    for (const auto& t : all_transactions) by_account[t.account_id].push_back(t);
    std::vector<std::string> flagged;
    for (auto& [account_id, history] : by_account) {
        std::sort(history.begin(), history.end(), [](const auto& a, const auto& b) { return a.minute < b.minute; });
        if (detect_impossible_travel(history, max_speed_kmh).flagged) flagged.push_back(account_id);
    }
    return flagged;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 29.3: Impossible Travel Geographic Detection\n";
    std::cout << "========================================================\n\n";

    const double MAX_PLAUSIBLE_SPEED_KMH = 900.0;
    // Two real, publicly known city coordinates.
    const double NYC_LAT = 40.7128, NYC_LON = -74.0060;
    const double LA_LAT = 34.0522, LA_LON = -118.2437;
    const double LONDON_LAT = 51.5074, LONDON_LON = -0.1278;
    const double PARIS_LAT = 48.8566, PARIS_LON = 2.3522;

    double dist_nyc_la = haversine_distance_km(NYC_LAT, NYC_LON, LA_LAT, LA_LON);
    double dist_london_paris = haversine_distance_km(LONDON_LAT, LONDON_LON, PARIS_LAT, PARIS_LON);
    std::cout << "(reference) real great-circle distance New York <-> Los Angeles: " << dist_nyc_la
              << " km; London <-> Paris: " << dist_london_paris << " km\n\n";

    // -- Test 1: the identical location, one minute apart -- zero
    // distance means zero required speed, never flagged regardless of
    // how little time passed. --
    {
        GeoTransaction a{"ACC-1", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-1", 1.0, NYC_LAT, NYC_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 1: identical location, 1 minute apart -- distance=" << r.distance_km
                  << " km, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(!r.flagged);
        CHECK_NEAR(r.distance_km, 0.0, 1e-9);
    }

    // -- Test 2: New York to Los Angeles in 6 real hours -- a genuinely
    // plausible cross-country flight -- not flagged. --
    {
        GeoTransaction a{"ACC-2", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-2", 360.0, LA_LAT, LA_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 2: NYC to LA in 6 hours -- required_speed=" << r.required_speed_kmh
                  << " km/h, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(!r.flagged);
        CHECK_NEAR(r.required_speed_kmh, dist_nyc_la / 6.0, 1e-6);
    }

    // -- Test 3: the identical NYC-to-LA distance, but in 30 minutes --
    // a real, genuine impossibility. --
    {
        GeoTransaction a{"ACC-3", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-3", 30.0, LA_LAT, LA_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 3: NYC to LA in 30 minutes -- required_speed=" << r.required_speed_kmh
                  << " km/h, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(r.flagged);
        CHECK(r.required_speed_kmh > MAX_PLAUSIBLE_SPEED_KMH);
    }

    // -- Test 4: the exact boundary -- elapsed time constructed so the
    // required speed lands EXACTLY on the stated maximum, confirmed not
    // flagged (an inclusive boundary, the same convention this book has
    // used for every real threshold since Chapter 26.1). --
    {
        double boundary_hours = dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH;
        GeoTransaction a{"ACC-4", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-4", boundary_hours * 60.0, LA_LAT, LA_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 4: NYC to LA in exactly " << (boundary_hours * 60.0)
                  << " minutes -- required_speed=" << r.required_speed_kmh << " km/h (max="
                  << MAX_PLAUSIBLE_SPEED_KMH << "), flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(!r.flagged);
        CHECK_NEAR(r.required_speed_kmh, MAX_PLAUSIBLE_SPEED_KMH, 1e-6);
    }

    // -- Test 5: one minute less than that exact boundary -- now
    // genuinely over the stated maximum, correctly flagged. --
    {
        double boundary_hours = dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH;
        GeoTransaction a{"ACC-5", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-5", boundary_hours * 60.0 - 1.0, LA_LAT, LA_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 5: one minute less than that boundary -- required_speed=" << r.required_speed_kmh
                  << " km/h, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(r.flagged);
        CHECK(r.required_speed_kmh > MAX_PLAUSIBLE_SPEED_KMH);
    }

    // -- Test 6: a much SHORTER real distance (London to Paris) but a
    // very short elapsed time -- flagged too, confirming this check is
    // genuinely about SPEED, not raw distance. --
    {
        GeoTransaction a{"ACC-6", 0.0, LONDON_LAT, LONDON_LON};
        GeoTransaction b{"ACC-6", 10.0, PARIS_LAT, PARIS_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 6: London to Paris (" << dist_london_paris << " km) in 10 minutes -- required_speed="
                  << r.required_speed_kmh << " km/h, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(r.flagged);
    }

    // -- Test 7: a mixed batch across two accounts -- only the account
    // exhibiting impossible travel is named. --
    {
        std::vector<GeoTransaction> mixed = {
            {"ACC-7A", 0.0, NYC_LAT, NYC_LON}, {"ACC-7A", 30.0, LA_LAT, LA_LON},
            {"ACC-7B", 0.0, LONDON_LAT, LONDON_LON}, {"ACC-7B", 360.0, PARIS_LAT, PARIS_LON},
        };
        auto flagged = flag_impossible_travel_accounts(mixed, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 7: two accounts, one impossible-travel pattern, one normal -- flagged=[";
        for (std::size_t i = 0; i < flagged.size(); i++) std::cout << (i ? ", " : "") << flagged[i];
        std::cout << "] --\n";
        CHECK(flagged.size() == 1);
        CHECK(flagged[0] == "ACC-7A");
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_impossible_travel_geographic_detection.cpp -o 03_impossible_travel_geographic_detection
./03_impossible_travel_geographic_detection
```

**Sample input:** the identical location one minute apart, checked to require zero speed; a real cross-country distance covered in a genuinely plausible 6 hours, checked as not flagged; the identical distance covered in 30 minutes, checked as a real impossibility; the exact speed boundary constructed algorithmically from the real distance, and one minute less than it, each checked precisely; a much shorter real distance covered in a very short time, checked to still be flagged on speed alone; and a mixed two-account batch checked to flag only the account exhibiting impossible travel.

```text
========================================================
Chapter 29.3: Impossible Travel Geographic Detection
========================================================

(reference) real great-circle distance New York <-> Los Angeles: 3935.75 km; London <-> Paris: 343.556 km

-- Test 1: identical location, 1 minute apart -- distance=0 km, flagged=NO --
-- Test 2: NYC to LA in 6 hours -- required_speed=655.958 km/h, flagged=NO --
-- Test 3: NYC to LA in 30 minutes -- required_speed=7871.49 km/h, flagged=YES --
-- Test 4: NYC to LA in exactly 262.383 minutes -- required_speed=900 km/h (max=900), flagged=NO --
-- Test 5: one minute less than that boundary -- required_speed=903.443 km/h, flagged=YES --
-- Test 6: London to Paris (343.556 km) in 10 minutes -- required_speed=2061.34 km/h, flagged=YES --
-- Test 7: two accounts, one impossible-travel pattern, one normal -- flagged=[ACC-7A] --

13/13 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating MAX_PLAUSIBLE_SPEED_KMH's own generosity as a weakness rather than a deliberate design choice"
    A tighter, less generous speed threshold would catch MORE real anomalies, but at a real, serious cost: a genuine customer who legitimately flew from one real city to another between two ordinary transactions would be flagged as fraudulent, purely for traveling normally. This section's own stated 900 km/h bound is deliberately set close to the real physical limit of ordinary human travel specifically so that ONLY a genuinely impossible required speed -- one no real commercial transportation could achieve -- is ever flagged, exactly the same "real, useful signal without an unacceptable false-positive cost" design tradeoff Section 26.1's own signature-comparison tolerance and Section 28.1's own duplicate-photo threshold already made in their own respective domains.

## 29.4 The Transaction Monitoring Disposition Engine

### Intuition

This chapter's capstone, and the fourth and final financial-industry chapter this book set out to build, combines Section 29.1's structuring signal, Section 29.2's velocity signal, and Section 29.3's impossible-travel signal into one per-account disposition -- with a real, legally precise boundary specific to this domain: under real US law, a Suspicious Activity Report can only ever be filed by a human compliance officer, never by an automated system.

### The Concept, In Detail

`assess_account` counts how many of the three independent signals fired for one account -- Test 2 confirms all three are weighted identically, each alone producing `INTERNAL_REVIEW`. Test 3 confirms the exact boundary at a count of two signals, driving this engine's own strongest disposition, `FILE_SAR_RECOMMENDED`. Test 5 is this section's own central, real legal precision check: the strongest disposition's own name is checked, precisely, to end in "RECOMMENDED" rather than stating a filing has actually occurred -- a real, legally meaningful distinction this engine's own naming enforces structurally, not merely by convention.

Test 6 confirms the disposition's own audit trail names the exact signals responsible, and Test 7 confirms every account in a mixed batch is assessed completely independently, closing this chapter's own recurring per-account-isolation discipline one final time.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_transaction_monitoring_disposition_engine.cpp -o 04_transaction_monitoring_disposition_engine
./04_transaction_monitoring_disposition_engine
```

**Sample input:** a completely clean account, checked to require no action; each of the three independent signal types fired alone, checked to produce the identical intermediate disposition; exactly two signals fired together, checked at the exact boundary of this engine's own strongest disposition; all three signals fired at once, checked to remain at that same strongest tier; the strongest disposition's own name checked, precisely, for the real legal distinction between a recommendation and an actual filing; a disposition's own audit trail checked to name the exact contributing signals; and a full multi-account batch checked for complete per-account independence.

```text
========================================================
Chapter 29.4: The Transaction Monitoring Disposition Engine
========================================================

-- Test 1: clean account -- disposition=NO_ACTION --
-- Test 2: exactly one signal fired -- structuring-only=INTERNAL_REVIEW, velocity-only=INTERNAL_REVIEW, travel-only=INTERNAL_REVIEW --
-- Test 3: exactly two signals fired (structuring + velocity) -- disposition=FILE_SAR_RECOMMENDED --
-- Test 4: all three signals fired -- disposition=FILE_SAR_RECOMMENDED --
-- Test 5: strongest disposition name = "FILE_SAR_RECOMMENDED" -- ends in RECOMMENDED: YES (correct) --
-- Test 6: audit trail -- [structuring pattern detected (Section 29.1)] [impossible travel pattern detected (Section 29.3)] --
-- Test 7: 3-account batch -- ACC-7A=NO_ACTION ACC-7B=INTERNAL_REVIEW ACC-7C=FILE_SAR_RECOMMENDED --

19/19 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating FILE_SAR_RECOMMENDED as equivalent to a SAR having actually been filed"
    `MonitoringDisposition::FILE_SAR_RECOMMENDED` means only that TWO OR MORE independent real signals fired for this account, which real compliance practice treats as strong enough evidence to warrant a human compliance officer's own real review and, potentially, a real SAR filing -- it does not mean a SAR has been filed, and this engine has no code path that could ever file one automatically, since real US law reserves that decision for a human at a regulated institution. A downstream system that treated this disposition as equivalent to an actual completed filing would misrepresent this engine's own real legal role, exactly the same category of harm Chapter 28.4's own COMMON TRAP box already named for conflating a claim routed to human review with a claim that had already been denied.

## Chapter Summary

This chapter closed this book's four-chapter run through real financial-industry deployments with a genuinely different real constraint: catching a pattern visible only across a sequence of transactions, never in any single one alone. Section 29.1 built a real structuring detector matching the exact legal boundary of the US Bank Secrecy Act's own CTR threshold. Section 29.2 reapplied Section 27.2's own stated-parameter timing discipline to a transaction-frequency burst instead of a deadline. Section 29.3 built a real "impossible travel" check from the real, published Haversine formula, deliberately generous enough to never flag a genuine traveler. Section 29.4 closed the chapter, and this book's four-chapter finance arc, by combining all three signals into one auditable disposition with a final, real legal precision: this engine can only ever recommend a Suspicious Activity Report, never file one, since real law reserves that decision for a human.

## Self-Check Questions

1. Section 29.1's `detect_structuring` requires every individual transaction in a flagged window to stay strictly under the CTR threshold. Explain what real, specific pattern this function would fail to catch if this individual-transaction check were removed, using only the window-sum condition.
2. Section 29.1's own COMMON TRAP box names a real refinement this section's own function does not implement: transactions spread across multiple accounts controlled by the same person. Explain concretely what additional real data this book's own `Transaction` struct would need to gain before that refinement could be implemented at all.
3. Section 29.2's Test 5 constructs 4 transactions spread across 45 minutes at a steady 15-minute pace, confirming this is NOT a velocity anomaly. Explain why this specific test is a stronger check of `detect_velocity_anomaly`'s own correctness than a test with only 2 well-spaced transactions would be.
4. Section 29.2's own COMMON TRAP box explains that a structuring pattern and a velocity anomaly are independent signals. Construct one concrete, realistic transaction sequence that would trigger Section 29.1's own structuring check while triggering NEITHER Section 29.2's velocity check nor Section 29.3's impossible-travel check.
5. Section 29.3's Test 4 constructs its own boundary using `dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH` directly, rather than a hardcoded number of minutes. Explain why this algorithmic construction makes the test more robust than picking a specific round number of minutes and checking it against the stated 900 km/h constant.
6. Section 29.3's own COMMON TRAP box explains the real tradeoff behind choosing a generous 900 km/h threshold. Describe one concrete real scenario in which a MORE generous threshold (say, 1200 km/h) would cause this section's own check to miss a real, genuinely impossible pattern that the stated 900 km/h threshold would have caught.
7. Section 29.4's `assess_account` counts fired signals using a simple integer count, exactly like Section 28.3's own `assess_fraud_likelihood`. Identify one structural similarity and one structural difference between these two functions' own real designs, referring to their actual code.
8. Section 29.4's own introduction explains a real, legally grounded reason its strongest disposition is named `FILE_SAR_RECOMMENDED` rather than `FILE_SAR`. Explain why Chapter 26.4's own `AUTO_REJECT` disposition did not need this same naming caution for its own domain.
9. Across Sections 29.1 through 29.3, each section's own detection function operates on ONE account's own transaction history at a time, with a separate aggregator function handling multiple accounts. Explain one concrete real advantage of keeping these two responsibilities in separate functions, rather than combining per-account detection and multi-account aggregation into a single function.
10. This chapter's own introduction states its central shift from every prior finance chapter: evaluating a SEQUENCE of transactions rather than any single one in isolation. Identify the ONE specific design choice, present in all three of Sections 29.1, 29.2, and 29.3, that makes this sequence-based evaluation possible, and explain why Chapters 26 and 28's own per-document and per-claim checks did not need it.

## Where We Go Next

This chapter closed this book's four-chapter run through real financial-industry deployments -- check and invoice fraud, low-latency market signals, insurance claims, and now sequence-based AML monitoring -- each grounded in a real, well-documented pattern from its own domain, and each ending in an auditable disposition that names its own reasoning rather than hiding behind a single opaque score. With all four now complete, Part 5 itself is complete: twelve real, deployed case-study systems spanning industrial inspection, retail, medical imaging, document intelligence, security and authenticity, natural language photo editing, personal cameras, and now four real financial-industry deployments. The next step revisits Chapter 32's own closing count of real edge deployments across this entire book, now that Part 5's own final real total is settled.

## Worked Solutions

**1.** Without the individual-transaction check, `detect_structuring` would also flag a SINGLE large, ordinary transaction whose own amount happens to exceed the threshold together with a few small, unrelated transactions in the same window -- for instance, a genuine $12,000 transaction (which already, correctly, triggers a real CTR directly and reports itself) alongside a completely unrelated $50 purchase in the same window would sum to $12,050, past the threshold, but this is not structuring at all, since the $12,000 transaction was never hidden below the threshold in the first place. The individual-transaction check exists specifically to isolate the real pattern this function targets: amounts DELIBERATELY kept under the threshold, not merely a window whose sum happens to cross it for any reason at all.

**2.** The `Transaction` struct would need a field identifying the REAL PERSON behind an account, independent of the account_id itself -- for instance, a real customer identifier (a legal name, a real government-issued ID number, or an internal customer-relationship identifier) shared across every account that same real person controls. Without this field, `detect_structuring`'s own per-account grouping in `flag_structuring_accounts` has no way to recognize that two different `account_id` values might belong to the same real person spreading a structuring pattern across both of them -- the function would need to group by this new customer-identity field instead of (or in addition to) `account_id` to catch that specific real refinement.

**3.** A test with only 2 well-spaced transactions could pass `detect_velocity_anomaly` correctly even under a subtly WRONG implementation -- for instance, one that only checks the count of the very FIRST window found, rather than scanning every possible rolling window for the true maximum -- since with only 2 transactions there is little room for such a bug to produce a visibly wrong answer. Test 5's own 4 transactions at a steady pace force a CORRECT implementation to check MULTIPLE overlapping windows (starting at minute 0, 15, 30, and 45) and confirm the maximum count across every one of them is still within bounds, which would catch a bug that only checked one window, or that failed to correctly recompute the count for each new rolling window start.

**4.** Three transactions of $4,000 each, all from account ACC-X, all made at the SAME single ATM location (so no geographic movement occurs at all, keeping Section 29.3's own impossible-travel check clean), spread exactly 1 day apart across days 1, 2, and 3 -- comfortably within Section 29.1's own 3-day structuring window (summing to $12,000, correctly flagged) but far too infrequent (only 3 transactions across 3 full days) to approach Section 29.2's own velocity threshold of more than 3 transactions within a single 30-MINUTE window, since the transactions here are separated by entire days, not minutes.

**5.** A hardcoded round number of minutes checked against the stated 900 km/h constant would only remain a valid boundary test for THIS SPECIFIC real distance (New York to Los Angeles) -- if a future edit to this section changed which two cities Test 4 used, or adjusted the stated `MAX_PLAUSIBLE_SPEED_KMH` value for any reason, a hardcoded minute value would silently stop testing the actual boundary at all, landing at some arbitrary point either safely inside or past it without anyone noticing until the test's own assertions started failing for the wrong reason. Constructing `boundary_hours` directly from `dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH` guarantees the test always lands EXACTLY on the true boundary implied by whatever real distance and real speed constant the file currently uses, remaining a true boundary test even if either value changes later.

**6.** Consider a genuinely fraudulent pattern where a compromised card is used in New York and, exactly 3 hours later, used again in London -- a real distance of roughly 5,570 km, requiring a real speed of about 1,857 km/h, which is well beyond what any real commercial passenger transportation could achieve (commercial supersonic passenger flight ended with Concorde's retirement in 2003) and would be correctly flagged as impossible under the stated 900 km/h threshold. Raising the threshold to 1,200 km/h would still catch this specific example, but a slightly less extreme version of the identical fraud pattern -- say, a required speed of exactly 1,050 km/h, still an entirely real impossibility for any genuine traveler -- would slip through undetected under the more generous 1,200 km/h bound, while the stated 900 km/h threshold would have caught it correctly.

**7.** Both functions share the identical structural approach: count how many independent, boolean, pre-named signals fired, and map that count onto an ordered set of outcomes via simple integer comparisons (`count >= 2`, `count == 1`), never weighting any one signal more heavily than another. The two functions differ in what they map that count ONTO: Section 28.3's `assess_fraud_likelihood` returns a `FraudLikelihood` LEVEL (LOW/MEDIUM/HIGH) describing a single zone's own suspicion, feeding into a SEPARATE disposition function (Section 28.4's own `assess_claim_disposition`) one level higher, while Section 29.4's `assess_account` maps its count DIRECTLY onto the final, actionable `MonitoringDisposition` in one single function -- there is no separate zone-level/claim-level split in this chapter, since Section 29.4's own three signals are already computed at the account level Section 29.4 itself operates on.

**8.** Chapter 26.4's own `AUTO_REJECT` disposition governs a bank's own internal decision about whether to CLEAR a specific check for processing -- a real, ordinary business decision a bank's own automated system is legally and operationally permitted to make about its own transaction processing. Section 29.4's own `FILE_SAR_RECOMMENDED`, by contrast, concerns a specific REGULATORY FILING (a Suspicious Activity Report) that real US law under the Bank Secrecy Act explicitly reserves for a human decision-maker at a regulated institution -- no amount of business judgment delegated to an automated check-clearing system changes who is legally permitted to file a SAR, which is exactly why this specific disposition name needed the extra legal precision that an internal processing decision like `AUTO_REJECT` never required.

**9.** Keeping per-account detection (`detect_structuring`, `detect_velocity_anomaly`, `detect_impossible_travel`) and multi-account aggregation (`flag_structuring_accounts`, `flag_velocity_accounts`, `flag_impossible_travel_accounts`) as separate functions lets each per-account function be tested, and reasoned about, in complete isolation -- exactly as this section's own Tests 1 through 6 in each of the first three sections do, using a single account's own history directly, with no grouping or sorting logic involved at all. A combined function would force every single-account test to also exercise the grouping and sorting logic every time, making it harder to tell whether a test failure came from the real detection LOGIC itself or from the unrelated bookkeeping of splitting a mixed batch by account -- the same general "test the core logic separately from the batch-orchestration wrapper around it" principle this book's own capstone engines have followed since Chapter 26.4.

**10.** The one specific design choice present in all three of Sections 29.1, 29.2, and 29.3 is that each section's own core detection function (`detect_structuring`, `detect_velocity_anomaly`, `detect_impossible_travel`) takes a WHOLE VECTOR of an account's own transaction history as its input, rather than a single transaction -- and internally scans across multiple entries in that vector (a rolling window, or consecutive pairs) to compute its own result. Chapter 26's own MICR/invoice/amount checks and Chapter 28's own duplicate-photo/timestamp checks each operate on ONE document, ONE photo, or ONE claim's own data at a time, since every real pattern those two chapters target (a forged checksum, a reused photo, a mismatched amount) is genuinely visible within a SINGLE item's own data -- this chapter's own three real AML patterns, by contrast, are only real patterns AT ALL when considered across multiple transactions together, which is exactly why accepting a whole sequence as input, rather than one transaction, is the one design choice this chapter's sequence-based evaluation could not do without.
