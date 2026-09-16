// Chapter 23.3 -- An automated expense-report policy engine is a real,
// checkable rules engine over a stated, real corporate travel policy: a
// per-category spending cap, a per-mile reimbursement rate, a blanket
// non-reimbursable category, and a receipt-required threshold. This
// section's own central honesty discipline, consistent with Section
// 21.3's own `CostRange` and this book's own recurring refusal to let a
// flagged discrepancy quietly resolve itself: the engine reports both the
// employee's own CLAIMED amount and the policy's own computed APPROVED
// amount side by side for every line item, and names the exact, specific
// rule behind every discrepancy between them -- it never silently
// substitutes one number for the other without saying so.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_expense_report_policy_enforcement.cpp -o 03_expense_report_policy_enforcement
// Run:     ./03_expense_report_policy_enforcement

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
// PART 1: the stated real policy constants and the per-city-tier lodging
// cap table.
// =======================================================================
enum class CityTier { TIER_1, TIER_2, TIER_3 };

// A real, stated per-diem-tier lookup: a major-metro tier-1 city's own
// real cost of lodging is not the same real number as a tier-3 city's,
// and this policy's own cap reflects that rather than applying one flat
// number everywhere.
double lodging_cap_for_tier(CityTier tier) {
    switch (tier) {
        case CityTier::TIER_1: return 350.0;
        case CityTier::TIER_2: return 200.0;
        case CityTier::TIER_3: return 120.0;
    }
    return 0.0;  // unreachable: CityTier's own 3 enumerators are fully handled above.
}

constexpr double MEALS_DAILY_CAP = 75.0;
constexpr double MILEAGE_RATE_PER_MILE = 0.67;
constexpr double RECEIPT_REQUIRED_THRESHOLD = 25.0;
constexpr double AMOUNT_EPSILON = 0.005;

// =======================================================================
// PART 2: the line item, the named violations, and the per-item
// evaluation.
// =======================================================================
enum class ExpenseCategory { MEALS, LODGING, MILEAGE, ALCOHOL, OFFICE_SUPPLIES };

enum class PolicyViolation {
    OVER_CATEGORY_CAP,
    ALCOHOL_NOT_REIMBURSABLE,
    MISSING_REQUIRED_RECEIPT,
    MILEAGE_CALCULATION_MISMATCH,
};

struct ExpenseLineItem {
    std::string item_id;
    ExpenseCategory category;
    double claimed_amount = 0.0;
    double miles = 0.0;
    CityTier city_tier = CityTier::TIER_3;
    bool has_receipt = false;
};

struct ViolationEntry {
    PolicyViolation violation;
    std::string detail;
};

struct LineItemEvaluation {
    std::string item_id;
    double claimed_amount = 0.0;
    double approved_amount = 0.0;
    std::vector<ViolationEntry> violations;
};

LineItemEvaluation evaluate_line_item(const ExpenseLineItem& item) {
    LineItemEvaluation eval;
    eval.item_id = item.item_id;
    eval.claimed_amount = item.claimed_amount;

    switch (item.category) {
        case ExpenseCategory::ALCOHOL: {
            eval.approved_amount = 0.0;
            eval.violations.push_back(
                {PolicyViolation::ALCOHOL_NOT_REIMBURSABLE,
                 "claimed $" + std::to_string(item.claimed_amount) +
                     " for an alcohol line item, which this policy never reimburses regardless of amount"});
            return eval;  // alcohol is fully excluded; the receipt-required check never applies to it.
        }
        case ExpenseCategory::MEALS: {
            double cap = MEALS_DAILY_CAP;
            eval.approved_amount = std::min(item.claimed_amount, cap);
            if (item.claimed_amount > cap + AMOUNT_EPSILON) {
                eval.violations.push_back(
                    {PolicyViolation::OVER_CATEGORY_CAP,
                     "claimed $" + std::to_string(item.claimed_amount) + " exceeds the real $" +
                         std::to_string(cap) + " daily meals cap by exactly $" +
                         std::to_string(item.claimed_amount - cap)});
            }
            break;
        }
        case ExpenseCategory::LODGING: {
            double cap = lodging_cap_for_tier(item.city_tier);
            eval.approved_amount = std::min(item.claimed_amount, cap);
            if (item.claimed_amount > cap + AMOUNT_EPSILON) {
                eval.violations.push_back(
                    {PolicyViolation::OVER_CATEGORY_CAP,
                     "claimed $" + std::to_string(item.claimed_amount) + " exceeds the real $" +
                         std::to_string(cap) +
                         " per-night lodging cap for this city's own stated tier by exactly $" +
                         std::to_string(item.claimed_amount - cap)});
            }
            break;
        }
        case ExpenseCategory::MILEAGE: {
            double expected = item.miles * MILEAGE_RATE_PER_MILE;
            eval.approved_amount = expected;
            if (std::abs(item.claimed_amount - expected) > AMOUNT_EPSILON) {
                eval.violations.push_back(
                    {PolicyViolation::MILEAGE_CALCULATION_MISMATCH,
                     "claimed $" + std::to_string(item.claimed_amount) + " does not match " +
                         std::to_string(item.miles) + " miles at the real $" +
                         std::to_string(MILEAGE_RATE_PER_MILE) + "/mile rate ($" + std::to_string(expected) +
                         " expected)"});
            }
            return eval;  // mileage is verified against a trip log, not a receipt, per this policy's own
                          // stated scope; the receipt-required check below never applies to it.
        }
        case ExpenseCategory::OFFICE_SUPPLIES: {
            eval.approved_amount = item.claimed_amount;
            break;
        }
    }

    if (item.claimed_amount > RECEIPT_REQUIRED_THRESHOLD + AMOUNT_EPSILON && !item.has_receipt) {
        eval.violations.push_back(
            {PolicyViolation::MISSING_REQUIRED_RECEIPT,
             "claimed $" + std::to_string(item.claimed_amount) +
                 " has no receipt on file, above the real $" + std::to_string(RECEIPT_REQUIRED_THRESHOLD) +
                 " requirement threshold"});
    }

    return eval;
}

// =======================================================================
// PART 3: the full-report aggregation -- the claimed and approved totals
// are each a real, exact sum of the per-item figures, and every
// violation stays attributed to its own specific item.
// =======================================================================
struct ExpenseReportSummary {
    double total_claimed = 0.0;
    double total_approved = 0.0;
    std::vector<LineItemEvaluation> line_items;
};

ExpenseReportSummary evaluate_expense_report(const std::vector<ExpenseLineItem>& items) {
    ExpenseReportSummary summary;
    for (const auto& item : items) {
        auto eval = evaluate_line_item(item);
        summary.total_claimed += eval.claimed_amount;
        summary.total_approved += eval.approved_amount;
        summary.line_items.push_back(eval);
    }
    return summary;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 23.3: An Automated Expense-Report Policy Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a meal claim under and exactly at the real daily cap is fully approved with "
                 "zero violations; one over the cap is flagged with the exact computed overage --\n";
    {
        ExpenseLineItem under{"E-1", ExpenseCategory::MEALS, 50.0, 0, CityTier::TIER_3, true};
        ExpenseLineItem at_cap{"E-2", ExpenseCategory::MEALS, 75.0, 0, CityTier::TIER_3, true};
        ExpenseLineItem over{"E-3", ExpenseCategory::MEALS, 90.0, 0, CityTier::TIER_3, true};
        auto eval_under = evaluate_line_item(under);
        auto eval_at_cap = evaluate_line_item(at_cap);
        auto eval_over = evaluate_line_item(over);
        CHECK(std::abs(eval_under.approved_amount - 50.0) < 1e-9);
        CHECK(eval_under.violations.empty());
        CHECK(std::abs(eval_at_cap.approved_amount - 75.0) < 1e-9);
        CHECK(eval_at_cap.violations.empty());
        CHECK(std::abs(eval_over.approved_amount - 75.0) < 1e-9);
        CHECK(eval_over.violations.size() == 1);
        CHECK(eval_over.violations[0].violation == PolicyViolation::OVER_CATEGORY_CAP);
        CHECK(eval_over.violations[0].detail.find("15") != std::string::npos);
        std::cout << "  a $50.00 and an exactly-$75.00 meal claim are both fully approved with zero "
                     "violations; a $90.00 claim is capped to $75.00 approved and flagged with the "
                     "exact $15.00 overage\n";
    }

    std::cout << "\n-- Test 2: lodging's own real per-city-tier cap is applied correctly for all 3 tiers, "
                 "and an over-cap claim in the cheapest tier is still flagged against ITS OWN tier's cap, "
                 "not a different tier's --\n";
    {
        ExpenseLineItem tier1{"E-4", ExpenseCategory::LODGING, 340.0, 0, CityTier::TIER_1, true};
        ExpenseLineItem tier2{"E-5", ExpenseCategory::LODGING, 190.0, 0, CityTier::TIER_2, true};
        ExpenseLineItem tier3_over{"E-6", ExpenseCategory::LODGING, 150.0, 0, CityTier::TIER_3, true};
        auto eval1 = evaluate_line_item(tier1);
        auto eval2 = evaluate_line_item(tier2);
        auto eval3 = evaluate_line_item(tier3_over);
        CHECK(eval1.violations.empty());
        CHECK(std::abs(eval1.approved_amount - 340.0) < 1e-9);
        CHECK(eval2.violations.empty());
        CHECK(std::abs(eval2.approved_amount - 190.0) < 1e-9);
        CHECK(eval3.violations.size() == 1);
        CHECK(std::abs(eval3.approved_amount - 120.0) < 1e-9);
        CHECK(eval3.violations[0].detail.find("30") != std::string::npos);
        std::cout << "  a $340 tier-1 stay and a $190 tier-2 stay both pass under their own real caps "
                     "($350 and $200); a $150 tier-3 stay is capped to $120.00 approved and flagged with "
                     "the exact $30.00 overage against ITS OWN tier-3 cap\n";
    }

    std::cout << "\n-- Test 3: a mileage claim matching the real rate exactly is approved with zero "
                 "violations; a claim that does not match its own computed value is flagged naming both "
                 "figures --\n";
    {
        ExpenseLineItem correct{"E-7", ExpenseCategory::MILEAGE, 67.0, 100.0, CityTier::TIER_3, false};
        ExpenseLineItem wrong{"E-8", ExpenseCategory::MILEAGE, 70.0, 100.0, CityTier::TIER_3, false};
        auto eval_correct = evaluate_line_item(correct);
        auto eval_wrong = evaluate_line_item(wrong);
        CHECK(std::abs(eval_correct.approved_amount - 67.0) < 1e-9);
        CHECK(eval_correct.violations.empty());
        CHECK(std::abs(eval_wrong.approved_amount - 67.0) < 1e-9);
        CHECK(eval_wrong.violations.size() == 1);
        CHECK(eval_wrong.violations[0].violation == PolicyViolation::MILEAGE_CALCULATION_MISMATCH);
        std::cout << "  100 miles at the real $0.67/mile rate computes to exactly $67.00: a claim of "
                     "$67.00 passes with zero violations, while a claim of $70.00 for the same 100 miles "
                     "is approved at the correctly computed $67.00 and flagged, naming both the claimed "
                     "and the expected figure\n";
    }

    std::cout << "\n-- Test 4: an alcohol line item is approved at $0.00 regardless of its claimed amount, "
                 "and never triggers the receipt-required check --\n";
    {
        ExpenseLineItem wine{"E-9", ExpenseCategory::ALCOHOL, 45.0, 0, CityTier::TIER_3, false};
        auto eval = evaluate_line_item(wine);
        CHECK(std::abs(eval.approved_amount - 0.0) < 1e-9);
        CHECK(eval.violations.size() == 1);
        CHECK(eval.violations[0].violation == PolicyViolation::ALCOHOL_NOT_REIMBURSABLE);
        std::cout << "  a $45.00 alcohol claim with no receipt on file is approved at exactly $0.00, "
                     "flagged only for ALCOHOL_NOT_REIMBURSABLE -- never additionally flagged for a "
                     "missing receipt, since a fully excluded category has nothing left to require one "
                     "for\n";
    }

    std::cout << "\n-- Test 5: the receipt-required threshold is exact at its own real boundary -- claims "
                 "at or below it never require a receipt, and claims above it do --\n";
    {
        ExpenseLineItem at_threshold{"E-10", ExpenseCategory::OFFICE_SUPPLIES, 25.0, 0, CityTier::TIER_3, false};
        ExpenseLineItem above_threshold{"E-11", ExpenseCategory::OFFICE_SUPPLIES, 25.01, 0, CityTier::TIER_3, false};
        auto eval_at = evaluate_line_item(at_threshold);
        auto eval_above = evaluate_line_item(above_threshold);
        CHECK(eval_at.violations.empty());
        CHECK(eval_above.violations.size() == 1);
        CHECK(eval_above.violations[0].violation == PolicyViolation::MISSING_REQUIRED_RECEIPT);
        std::cout << "  a $25.00 office-supply claim with no receipt requires none and is approved with "
                     "zero violations; a $25.01 claim with no receipt is flagged MISSING_REQUIRED_RECEIPT\n";
    }

    std::cout << "\n-- Test 6: a full multi-item report computes its own claimed and approved totals as "
                 "exact sums, and attributes every violation to its own specific line item, never "
                 "silently dropping a flagged item's own claimed contribution to the total --\n";
    {
        std::vector<ExpenseLineItem> items = {
            {"E-20", ExpenseCategory::MEALS, 90.0, 0, CityTier::TIER_3, true},        // over cap by 15
            {"E-21", ExpenseCategory::LODGING, 340.0, 0, CityTier::TIER_1, true},     // fine
            {"E-22", ExpenseCategory::MILEAGE, 67.0, 100.0, CityTier::TIER_3, false}, // fine
            {"E-23", ExpenseCategory::ALCOHOL, 30.0, 0, CityTier::TIER_3, false},     // excluded
        };
        auto summary = evaluate_expense_report(items);
        double expected_claimed = 90.0 + 340.0 + 67.0 + 30.0;
        double expected_approved = 75.0 + 340.0 + 67.0 + 0.0;
        CHECK(std::abs(summary.total_claimed - expected_claimed) < 1e-9);
        CHECK(std::abs(summary.total_approved - expected_approved) < 1e-9);
        CHECK(summary.line_items.size() == 4);
        int total_violations = 0;
        for (const auto& li : summary.line_items) total_violations += static_cast<int>(li.violations.size());
        CHECK(total_violations == 2);  // E-20's over-cap, E-23's alcohol exclusion
        CHECK(summary.line_items[0].item_id == "E-20" && summary.line_items[0].violations.size() == 1);
        CHECK(summary.line_items[3].item_id == "E-23" && summary.line_items[3].violations.size() == 1);
        std::cout << "  a 4-item report totals $" << expected_claimed << " claimed against $"
                   << expected_approved << " approved -- the $90 meal claim's full $90 still counts "
                     "toward the claimed total even though only $75 is approved, and the excluded $30 "
                     "alcohol item still counts toward the claimed total even though $0 is approved -- "
                     "the report's own claimed figure is never silently reduced, only the approved "
                     "figure and the named violation tell the real story\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
