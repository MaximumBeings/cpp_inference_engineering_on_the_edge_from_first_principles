// Chapter 23.2 -- A receipt, extracted by the same kind of OCR/VLM
// pipeline this book built in Chapter 21.1, can carry small per-character
// extraction errors; a bank or card network's own transaction record, by
// contrast, carries the true merchant name as recorded by the payment
// network itself. Reconciling the two real records requires a real fuzzy
// string-matching algorithm -- this section builds the classic Levenshtein
// edit-distance algorithm from scratch -- plus two further real matching
// rules: a settlement date can only fall ON OR AFTER a purchase date,
// never before, and a settled amount may legitimately exceed a receipt's
// own printed subtotal by an added tip, but only up to a real, stated cap.
//
// A note on this section's own honest scope: this section's own fuzzy
// merchant-name match is built specifically for OCR-INTRODUCED per-
// character noise (a misread digit or letter), not for the different
// real problem of payment-processor descriptor mangling (a transaction
// descriptor like "SQ *STARBUCKS COFFEE" for a receipt's own "Starbucks")
// -- a different real matching technique a production reconciliation
// system would need in addition to, not instead of, this one.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_receipt_transaction_reconciliation.cpp -o 02_receipt_transaction_reconciliation
// Run:     ./02_receipt_transaction_reconciliation

#include <algorithm>
#include <cctype>
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
// PART 1: the real, classic Levenshtein edit-distance algorithm, built
// from scratch with a plain O(n*m) dynamic-programming table.
// =======================================================================
int levenshtein_distance(const std::string& a, const std::string& b) {
    std::size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 0; i <= n; i++) dp[i][0] = static_cast<int>(i);
    for (std::size_t j = 0; j <= m; j++) dp[0][j] = static_cast<int>(j);
    for (std::size_t i = 1; i <= n; i++) {
        for (std::size_t j = 1; j <= m; j++) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost});
        }
    }
    return dp[n][m];
}

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

// A real, stated policy choice: an OCR'd merchant name may differ from
// its own true merchant name by up to this fraction of its own length --
// enough to absorb a single misread character on a short name, without
// absorbing a genuinely different merchant.
constexpr double MERCHANT_MATCH_MAX_EDIT_RATIO = 0.34;

bool merchant_names_fuzzy_match(const std::string& ocr_name, const std::string& true_name) {
    std::string a = to_upper(ocr_name), b = to_upper(true_name);
    int dist = levenshtein_distance(a, b);
    std::size_t max_len = std::max(a.size(), b.size());
    if (max_len == 0) return true;
    double ratio = static_cast<double>(dist) / static_cast<double>(max_len);
    return ratio <= MERCHANT_MATCH_MAX_EDIT_RATIO;
}

// =======================================================================
// PART 2: real amount matching -- exact, or tip-adjusted up to a real,
// stated cap -- and real settlement-date-window matching.
// =======================================================================
constexpr double AMOUNT_EXACT_EPSILON = 0.005;
constexpr double MAX_TIP_FRACTION = 0.25;
constexpr int SETTLEMENT_WINDOW_MAX_DAYS = 3;

bool amounts_match(double receipt_amount, double txn_amount) {
    if (std::abs(txn_amount - receipt_amount) < AMOUNT_EXACT_EPSILON) return true;
    if (txn_amount > receipt_amount) {
        double tip = txn_amount - receipt_amount;
        return tip <= receipt_amount * MAX_TIP_FRACTION + AMOUNT_EXACT_EPSILON;
    }
    return false;
}

// Settlement can only occur ON OR AFTER the purchase's own real period --
// a transaction dated before its own receipt's purchase period can never
// be the same real event, regardless of how small that gap is.
bool within_settlement_window(int receipt_period, int txn_period) {
    int delta = txn_period - receipt_period;
    return delta >= 0 && delta <= SETTLEMENT_WINDOW_MAX_DAYS;
}

// =======================================================================
// PART 3: the reconciliation engine -- honestly reports AMBIGUOUS, naming
// every fitting candidate, exactly the same discipline Section 22.1's own
// multi-camera correlation applied to a temporally ambiguous re-
// identification.
// =======================================================================
struct Receipt {
    std::string receipt_id;
    std::string merchant_name_ocr;
    double amount = 0.0;
    int purchase_period = 0;
};

struct Transaction {
    std::string transaction_id;
    std::string merchant_name;
    double amount = 0.0;
    int settlement_period = 0;
};

enum class ReconciliationOutcome { MATCHED, AMBIGUOUS, MISSING_TRANSACTION };

struct ReconciliationResult {
    ReconciliationOutcome outcome;
    std::vector<std::string> candidate_transaction_ids;
};

ReconciliationResult reconcile_receipt(const Receipt& r, const std::vector<Transaction>& transactions) {
    std::vector<std::string> candidates;
    for (const auto& t : transactions) {
        if (within_settlement_window(r.purchase_period, t.settlement_period) &&
            amounts_match(r.amount, t.amount) &&
            merchant_names_fuzzy_match(r.merchant_name_ocr, t.merchant_name)) {
            candidates.push_back(t.transaction_id);
        }
    }
    if (candidates.empty()) return {ReconciliationOutcome::MISSING_TRANSACTION, {}};
    if (candidates.size() == 1) return {ReconciliationOutcome::MATCHED, candidates};
    return {ReconciliationOutcome::AMBIGUOUS, candidates};
}

// A transaction never claimed by any receipt's own unique match is a real,
// useful signal for an expense audit: a charge with no corresponding
// receipt on file at all.
std::vector<std::string> find_transactions_missing_receipts(const std::vector<Transaction>& transactions,
                                                              const std::vector<ReconciliationResult>& results) {
    std::vector<std::string> claimed;
    for (const auto& result : results) {
        if (result.outcome == ReconciliationOutcome::MATCHED) {
            claimed.push_back(result.candidate_transaction_ids[0]);
        }
    }
    std::vector<std::string> missing;
    for (const auto& t : transactions) {
        if (std::find(claimed.begin(), claimed.end(), t.transaction_id) == claimed.end()) {
            missing.push_back(t.transaction_id);
        }
    }
    return missing;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 23.2: A Receipt-to-Transaction Reconciliation Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real Levenshtein algorithm matches its own famous exact reference "
                 "value (KITTEN -> SITTING is distance 3), alongside identical and fully-disjoint strings --\n";
    {
        CHECK(levenshtein_distance("KITTEN", "SITTING") == 3);
        CHECK(levenshtein_distance("STARBUCKS", "STARBUCKS") == 0);
        CHECK(levenshtein_distance("", "ABC") == 3);
        std::cout << "  KITTEN -> SITTING computes to exactly 3, matching this algorithm's own famous "
                     "reference value; an identical pair computes to 0; an empty string against a "
                     "3-character string computes to exactly 3 (three real insertions)\n";
    }

    std::cout << "\n-- Test 2: a single-character OCR misread is fuzzy-matched to its own true merchant "
                 "name, while a genuinely different merchant is not --\n";
    {
        CHECK(merchant_names_fuzzy_match("WALGREEN5", "WALGREENS"));
        CHECK(!merchant_names_fuzzy_match("WALGREENS", "CVS PHARMACY"));
        std::cout << "  \"WALGREEN5\" (a real OCR digit/letter confusion of the trailing S) fuzzy-matches "
                     "its own true name \"WALGREENS\" at an edit ratio of 1/9; \"WALGREENS\" does not "
                     "match the genuinely different merchant \"CVS PHARMACY\"\n";
    }

    std::cout << "\n-- Test 3: amount matching accepts an added tip up to its own real, stated cap, and "
                 "rejects one beyond it, exactly at the boundary --\n";
    {
        CHECK(amounts_match(40.00, 40.00));
        CHECK(amounts_match(40.00, 46.00));
        CHECK(amounts_match(40.00, 50.00));
        CHECK(!amounts_match(40.00, 50.01));
        CHECK(!amounts_match(40.00, 39.00));
        std::cout << "  a $40.00 receipt matches an exact $40.00 settlement, a 15% tip ($46.00), and "
                     "exactly its own real 25% tip cap ($50.00); it does NOT match $50.01 (one cent over "
                     "the cap) or a $39.00 settlement below its own printed amount\n";
    }

    std::cout << "\n-- Test 4: the settlement-date window is exact at its own real boundary, and a "
                 "settlement dated before its own purchase never matches regardless of gap size --\n";
    {
        CHECK(within_settlement_window(100, 100));
        CHECK(within_settlement_window(100, 103));
        CHECK(!within_settlement_window(100, 104));
        CHECK(!within_settlement_window(100, 99));
        CHECK(!within_settlement_window(100, 50));
        std::cout << "  a same-period settlement and a settlement exactly 3 periods later both match; "
                     "one 4 periods later does not, and one dated even 1 period before its own purchase "
                     "(let alone 50) never matches\n";
    }

    std::cout << "\n-- Test 5: a receipt with exactly one fitting transaction is reported MATCHED, naming "
                 "that transaction --\n";
    {
        Receipt r{"R-1", "TARGEI", 52.30, 200};
        std::vector<Transaction> pool = {
            {"T-1", "TARGET", 52.30, 201},
            {"T-2", "WALGREENS", 12.00, 201},
        };
        auto result = reconcile_receipt(r, pool);
        CHECK(result.outcome == ReconciliationOutcome::MATCHED);
        CHECK(result.candidate_transaction_ids.size() == 1);
        CHECK(result.candidate_transaction_ids[0] == "T-1");
        std::cout << "  an OCR'd receipt reading \"TARGEI\" for $52.30 matches exactly one real "
                     "transaction (T-1, \"TARGET\", $52.30) out of a 2-transaction pool\n";
    }

    std::cout << "\n-- Test 6: two transactions that BOTH fit every real matching rule are honestly "
                 "reported AMBIGUOUS, naming both candidates, rather than silently picking one --\n";
    {
        Receipt r{"R-2", "STARBUCKS", 12.00, 300};
        std::vector<Transaction> pool = {
            {"T-10", "STARBUCKS", 13.50, 300},
            {"T-11", "STARBUCKS", 13.50, 301},
        };
        auto result = reconcile_receipt(r, pool);
        CHECK(result.outcome == ReconciliationOutcome::AMBIGUOUS);
        CHECK(result.candidate_transaction_ids.size() == 2);
        std::cout << "  a $12.00 Starbucks receipt with a real, plausible tip finds TWO transactions "
                     "(T-10 same-day, T-11 one day later) both matching on amount, merchant, and window "
                     "-- reported AMBIGUOUS, naming both, rather than guessing which one is correct\n";
    }

    std::cout << "\n-- Test 7: a receipt with no fitting transaction anywhere in the pool is reported "
                 "MISSING_TRANSACTION --\n";
    {
        Receipt r{"R-3", "OFFICE SUPPLY CO", 200.00, 400};
        std::vector<Transaction> pool = {
            {"T-20", "OFFICE SUPPLY CO", 45.00, 400},
        };
        auto result = reconcile_receipt(r, pool);
        CHECK(result.outcome == ReconciliationOutcome::MISSING_TRANSACTION);
        CHECK(result.candidate_transaction_ids.empty());
        std::cout << "  a $200.00 receipt against a pool containing only a $45.00 transaction from the "
                     "same real merchant finds no amount-matching candidate and is reported "
                     "MISSING_TRANSACTION\n";
    }

    std::cout << "\n-- Test 8: a transaction never claimed by any receipt's own unique match is flagged "
                 "MISSING a receipt, a real, useful signal for an expense audit --\n";
    {
        std::vector<Receipt> receipts = {
            {"R-4", "TARGET", 52.30, 500},
        };
        std::vector<Transaction> pool = {
            {"T-30", "TARGET", 52.30, 500},
            {"T-31", "UNKNOWN VENDOR LLC", 890.00, 500},
        };
        std::vector<ReconciliationResult> results;
        for (const auto& r : receipts) results.push_back(reconcile_receipt(r, pool));
        auto missing = find_transactions_missing_receipts(pool, results);
        CHECK(missing.size() == 1);
        CHECK(missing[0] == "T-31");
        std::cout << "  of a 2-transaction pool, T-30 is claimed by its own matching receipt while T-31, "
                     "an $890.00 charge with no receipt on file at all, is flagged as missing a receipt\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
