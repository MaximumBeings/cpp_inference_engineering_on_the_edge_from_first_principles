// Chapter 26.2 -- A well-known real invoice-fraud pattern is not a forged
// invoice at all, but the SAME real invoice submitted twice for payment,
// with the invoice number altered by a single character specifically to
// dodge a naive exact-match duplicate check -- "INV-10422" resubmitted as
// "INV-10423" a week later, same vendor, same amount, hoping an
// overworked accounts-payable clerk (or a naive exact-string dedup pass)
// never notices. This section builds a real, from-scratch duplicate-
// invoice detector using the identical real Levenshtein edit-distance
// algorithm Chapter 23.2 built for OCR-noisy receipt reconciliation,
// applied here to a different real signal: an invoice number that is
// SUSPICIOUSLY CLOSE to, but not identical to, a previously paid
// invoice's own number, from the same vendor, for the same amount.
//
// A note on this section's own honest scope: `flag_duplicate` identifies
// a STRUCTURAL pattern -- a near-identical invoice number combined with
// an identical vendor and amount -- that is highly correlated with real
// double-billing fraud. It does not, and cannot, prove intent; a vendor
// that genuinely issues two separate real invoices for the same amount
// in the same week (a recurring service charge, for instance) would also
// match this pattern and should still be reviewed by a person, not
// auto-rejected -- exactly the same honest structural-pass discipline
// Chapter 23.4's own luxury-goods screening applied.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_invoice_duplicate_detection.cpp -o 02_invoice_duplicate_detection
// Run:     ./02_invoice_duplicate_detection

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
// PART 1: the real, classic Levenshtein edit-distance algorithm -- the
// identical implementation Chapter 23.2 built for receipt reconciliation.
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

// =======================================================================
// PART 2: a real, from-scratch duplicate-invoice detector combining
// three independent real signals -- exact vendor match, exact amount
// match, and a NEAR (but not identical) invoice number.
// =======================================================================
struct Invoice {
    std::string vendor;
    std::string invoice_number;
    double amount = 0.0;
};

constexpr double AMOUNT_EXACT_EPSILON = 0.005;
// A real, stated policy: an invoice number within this many single-
// character edits of a previously paid invoice number -- but NOT
// identical, which a plain exact-match dedup pass would already catch --
// is close enough to be the same real digits with one deliberately
// altered character.
constexpr int SUSPICIOUS_MAX_EDIT_DISTANCE = 2;

struct DuplicateFlag {
    bool flagged = false;
    std::string matched_invoice_number;
    int edit_distance = -1;
};

DuplicateFlag flag_duplicate(const Invoice& candidate, const std::vector<Invoice>& paid_history) {
    for (const Invoice& paid : paid_history) {
        if (to_upper(paid.vendor) != to_upper(candidate.vendor)) continue;
        if (std::abs(paid.amount - candidate.amount) >= AMOUNT_EXACT_EPSILON) continue;
        if (paid.invoice_number == candidate.invoice_number) continue;  // exact match: a plain dedup pass already catches this
        int dist = levenshtein_distance(to_upper(paid.invoice_number), to_upper(candidate.invoice_number));
        if (dist > 0 && dist <= SUSPICIOUS_MAX_EDIT_DISTANCE) {
            return DuplicateFlag{true, paid.invoice_number, dist};
        }
    }
    return DuplicateFlag{};
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.2: Invoice Duplicate Detection\n";
    std::cout << "========================================================\n\n";

    std::vector<Invoice> paid_history = {
        {"Acme Office Supply", "INV-10422", 1284.50},
        {"Northbridge Logistics", "PO-88213", 6120.00},
        {"Acme Office Supply", "INV-10517", 340.00},
    };

    // -- Test 1: the classic KITTEN-to-SITTING reference value, confirming
    // this is the real, unmodified Levenshtein algorithm. --
    int kitten_dist = levenshtein_distance("KITTEN", "SITTING");
    std::cout << "-- Test 1: Levenshtein(\"KITTEN\", \"SITTING\") = " << kitten_dist
              << " (the algorithm's own famous reference value) --\n";
    CHECK(kitten_dist == 3);

    // -- Test 2: an invoice number one character off from a real, already-
    // paid invoice, same vendor, same amount -- flagged as a likely
    // duplicate, naming the exact invoice number it matches and the exact
    // edit distance. --
    Invoice resubmit{"Acme Office Supply", "INV-10423", 1284.50};
    DuplicateFlag flag1 = flag_duplicate(resubmit, paid_history);
    std::cout << "-- Test 2: candidate " << resubmit.invoice_number << " ($" << resubmit.amount
              << ", " << resubmit.vendor << ") vs. paid history -- flagged: "
              << (flag1.flagged ? "YES" : "NO");
    if (flag1.flagged) {
        std::cout << ", matches paid invoice " << flag1.matched_invoice_number
                   << " at edit distance " << flag1.edit_distance;
    }
    std::cout << " --\n";
    CHECK(flag1.flagged);
    CHECK(flag1.matched_invoice_number == "INV-10422");
    CHECK(flag1.edit_distance == 1);

    // -- Test 3: a genuinely different invoice number for the same vendor
    // and amount -- far more than the stated edit-distance threshold from
    // any paid invoice -- is correctly NOT flagged. --
    Invoice different_number{"Acme Office Supply", "INV-99871", 1284.50};
    DuplicateFlag flag2 = flag_duplicate(different_number, paid_history);
    std::cout << "-- Test 3: candidate " << different_number.invoice_number
              << " ($" << different_number.amount << ", " << different_number.vendor
              << ") vs. paid history -- flagged: " << (flag2.flagged ? "YES (WRONG)" : "NO (correctly not flagged)") << " --\n";
    CHECK(!flag2.flagged);

    // -- Test 4: an exact re-submission of the identical invoice number --
    // the case a plain exact-match dedup pass already catches on its own
    // -- is correctly excluded from THIS detector's own near-match logic,
    // since its own job is the gap a naive exact-match check misses. --
    Invoice exact_resubmit{"Acme Office Supply", "INV-10422", 1284.50};
    DuplicateFlag flag3 = flag_duplicate(exact_resubmit, paid_history);
    std::cout << "-- Test 4: candidate " << exact_resubmit.invoice_number
              << " (an EXACT resubmission, already caught by a plain dedup pass) -- flagged by THIS near-match detector: "
              << (flag3.flagged ? "YES (WRONG)" : "NO (correctly out of this detector's own scope)") << " --\n";
    CHECK(!flag3.flagged);

    // -- Test 5: the same near-match invoice number pattern from a
    // DIFFERENT vendor is correctly not flagged -- a coincidental invoice-
    // number collision across unrelated vendors is not evidence of
    // double-billing. --
    Invoice different_vendor{"Northbridge Logistics", "INV-10423", 1284.50};
    DuplicateFlag flag4 = flag_duplicate(different_vendor, paid_history);
    std::cout << "-- Test 5: candidate " << different_vendor.invoice_number
              << " from a DIFFERENT vendor (" << different_vendor.vendor
              << ") at the identical amount -- flagged: "
              << (flag4.flagged ? "YES (WRONG)" : "NO (correctly not flagged)") << " --\n";
    CHECK(!flag4.flagged);

    // -- Test 6: the identical near-match invoice-number pattern but at a
    // genuinely different amount is correctly not flagged -- this
    // detector's own three signals (vendor, amount, near-match number)
    // are all required together, not any one alone. --
    Invoice different_amount{"Acme Office Supply", "INV-10423", 55.00};
    DuplicateFlag flag5 = flag_duplicate(different_amount, paid_history);
    std::cout << "-- Test 6: candidate " << different_amount.invoice_number
              << " at a genuinely different amount ($" << different_amount.amount
              << " vs. the paid $1284.50) -- flagged: "
              << (flag5.flagged ? "YES (WRONG)" : "NO (correctly not flagged)") << " --\n";
    CHECK(!flag5.flagged);

    // -- Test 6: an invoice number exactly AT the stated edit-distance
    // boundary (distance 2) is still flagged; the exact boundary itself
    // is checked, not just a value comfortably inside it. --
    Invoice boundary{"Acme Office Supply", "INV-10499", 1284.50};
    int boundary_dist = levenshtein_distance("INV-10422", "INV-10499");
    DuplicateFlag flag6 = flag_duplicate(boundary, paid_history);
    std::cout << "-- Test 7: candidate " << boundary.invoice_number
              << " at hand-computed edit distance " << boundary_dist
              << " from the paid invoice (the stated boundary of " << SUSPICIOUS_MAX_EDIT_DISTANCE
              << ") -- flagged: " << (flag6.flagged ? "YES" : "NO") << " --\n";
    CHECK(boundary_dist == 2);
    CHECK(flag6.flagged);

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
