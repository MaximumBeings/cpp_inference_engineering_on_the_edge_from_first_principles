// Chapter 25.4 -- A camera-based nutrition tracker's own privacy-first
// architecture means it cannot depend on a live USDA FoodData Central
// API call for every meal: every function in this section runs
// entirely on-device, against a small, stated, offline reference
// subset of real, approximate USDA FoodData Central per-100g values,
// with no image or meal data ever needing to leave the device. Portion
// size is estimated from a real reference-object scale (a standard
// dinner plate's own known real diameter) and reported honestly as a
// range, never a fabricated single-gram point estimate -- this book's
// own established CostRange discipline (Section 21.3), reapplied here
// to a genuinely uncertain real-world measurement.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_camera_based_nutrition_tracking.cpp -o 04_camera_based_nutrition_tracking
// Run:     ./04_camera_based_nutrition_tracking

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: a small, closed, offline reference subset of real,
// approximate USDA FoodData Central per-100g values. Stated and
// approximate, deliberately -- this is a small offline subset for a
// privacy-first device, never the live API, and this section never
// claims lab-grade precision for either these reference values or the
// portion estimates built on top of them.
// =======================================================================
enum class FoodItem {
    APPLE_RAW, BANANA_RAW, CHICKEN_BREAST_COOKED, WHITE_RICE_COOKED, BROCCOLI_RAW, EGG_LARGE_COOKED,
};

const std::vector<FoodItem> ALL_FOOD_ITEMS = {
    FoodItem::APPLE_RAW, FoodItem::BANANA_RAW, FoodItem::CHICKEN_BREAST_COOKED,
    FoodItem::WHITE_RICE_COOKED, FoodItem::BROCCOLI_RAW, FoodItem::EGG_LARGE_COOKED,
};

struct NutrientProfile {
    double calories_per_100g = 0.0;
    double protein_g_per_100g = 0.0;
    double carbs_g_per_100g = 0.0;
    double fat_g_per_100g = 0.0;
    double density_g_per_cm3 = 0.0;      // for portion-mass estimation
    double assumed_thickness_cm = 0.0;   // a stated, illustrative average-serving assumption
};

NutrientProfile nutrient_profile_for(FoodItem food) {
    switch (food) {
        case FoodItem::APPLE_RAW:             return {52.0, 0.3, 13.8, 0.2, 0.6, 2.0};
        case FoodItem::BANANA_RAW:            return {89.0, 1.1, 22.8, 0.3, 0.5, 2.0};
        case FoodItem::CHICKEN_BREAST_COOKED: return {165.0, 31.0, 0.0, 3.6, 1.0, 1.5};
        case FoodItem::WHITE_RICE_COOKED:     return {130.0, 2.7, 28.0, 0.3, 0.9, 2.5};
        case FoodItem::BROCCOLI_RAW:          return {34.0, 2.8, 6.6, 0.4, 0.35, 3.0};
        case FoodItem::EGG_LARGE_COOKED:      return {155.0, 13.0, 1.1, 11.0, 1.03, 2.5};
    }
    return {};
}

// =======================================================================
// PART 2: real portion-size estimation from a real reference object --
// a standard dinner plate of known real diameter -- reported honestly
// as a range rather than a single fabricated gram figure.
// =======================================================================
constexpr double REFERENCE_PLATE_DIAMETER_CM = 26.0;
constexpr double THICKNESS_UNCERTAINTY_FRACTION = 0.20;

struct PortionEstimate {
    double low_g = 0.0, mid_g = 0.0, high_g = 0.0;
};

PortionEstimate estimate_portion_range(double food_area_px, double plate_diameter_px, FoodItem food) {
    double plate_area_cm2 = std::numbers::pi * std::pow(REFERENCE_PLATE_DIAMETER_CM / 2.0, 2);
    double plate_area_px = std::numbers::pi * std::pow(plate_diameter_px / 2.0, 2);
    double scale_cm2_per_px2 = plate_area_cm2 / plate_area_px;
    double food_area_cm2 = food_area_px * scale_cm2_per_px2;

    NutrientProfile profile = nutrient_profile_for(food);
    double mass_mid_g = food_area_cm2 * profile.assumed_thickness_cm * profile.density_g_per_cm3;
    return {
        mass_mid_g * (1.0 - THICKNESS_UNCERTAINTY_FRACTION),
        mass_mid_g,
        mass_mid_g * (1.0 + THICKNESS_UNCERTAINTY_FRACTION),
    };
}

// =======================================================================
// PART 3: daily nutrient tally over human-confirmed log entries -- the
// portion ESTIMATE above is a range the user reviews and confirms
// before logging, exactly this book's own established human-in-the-loop
// discipline (Sections 20.1 and 20.4), which is what lets the tally
// itself stay exact and deterministic rather than needing to propagate
// a range through every downstream sum.
// =======================================================================
struct FoodLogEntry {
    FoodItem food;
    double confirmed_grams = 0.0;
};

struct NutrientTotals {
    double calories = 0.0, protein_g = 0.0, carbs_g = 0.0, fat_g = 0.0;
};

NutrientTotals daily_nutrient_tally(const std::vector<FoodLogEntry>& entries) {
    NutrientTotals totals;
    for (const auto& entry : entries) {
        NutrientProfile p = nutrient_profile_for(entry.food);
        double factor = entry.confirmed_grams / 100.0;
        totals.calories += p.calories_per_100g * factor;
        totals.protein_g += p.protein_g_per_100g * factor;
        totals.carbs_g += p.carbs_g_per_100g * factor;
        totals.fat_g += p.fat_g_per_100g * factor;
    }
    return totals;
}

enum class IntakeClassification { BELOW_RANGE, WITHIN_RANGE, ABOVE_RANGE };

struct DailyTargetRange {
    double calories_min = 0.0, calories_max = 0.0;
};

IntakeClassification classify_daily_intake(double total_calories, DailyTargetRange range) {
    if (total_calories < range.calories_min) return IntakeClassification::BELOW_RANGE;
    if (total_calories > range.calories_max) return IntakeClassification::ABOVE_RANGE;
    return IntakeClassification::WITHIN_RANGE;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 25.4: Camera-Based Nutrition Tracking via a USDA-Style Offline Reference\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the offline reference subset's own stated real values are exactly what "
                 "this section commits to for a representative sample --\n";
    {
        CHECK(near(nutrient_profile_for(FoodItem::APPLE_RAW).calories_per_100g, 52.0));
        CHECK(near(nutrient_profile_for(FoodItem::CHICKEN_BREAST_COOKED).calories_per_100g, 165.0));
        CHECK(near(nutrient_profile_for(FoodItem::CHICKEN_BREAST_COOKED).protein_g_per_100g, 31.0));
        CHECK(near(nutrient_profile_for(FoodItem::EGG_LARGE_COOKED).fat_g_per_100g, 11.0));
        std::cout << "  apple (raw) is exactly 52 kcal per 100g; chicken breast (cooked) is exactly "
                     "165 kcal and exactly 31.0g protein per 100g; egg (large, cooked) is exactly "
                     "11.0g fat per 100g -- this section's own stated reference values, checked "
                     "directly\n";
    }

    std::cout << "\n-- Test 2: every food in the closed vocabulary has a real, non-degenerate reference "
                 "entry -- an exhaustive check, not a hand-picked sample --\n";
    {
        int checked = 0;
        for (FoodItem food : ALL_FOOD_ITEMS) {
            NutrientProfile p = nutrient_profile_for(food);
            CHECK(p.calories_per_100g > 0.0);
            CHECK(p.density_g_per_cm3 > 0.0);
            CHECK(p.assumed_thickness_cm > 0.0);
            checked++;
        }
        CHECK(checked == 6);
        std::cout << "  all " << checked << " foods in the closed vocabulary have a real, positive "
                     "calorie count, density, and assumed serving thickness -- no food silently falls "
                     "through to a missing or zeroed-out reference entry\n";
    }

    std::cout << "\n-- Test 3: real portion-mass estimation from a real reference-plate scale matches "
                 "an exact, hand-computed value --\n";
    {
        // A 2600px plate diameter against the real 26cm reference plate is exactly a 100px/cm scale,
        // so a real (26/2600)^2 = 0.0001 cm^2/px^2 area scale, independent of pi.
        auto chicken = estimate_portion_range(800000.0, 2600.0, FoodItem::CHICKEN_BREAST_COOKED);
        // 800000px^2 * 0.0001 = 80 cm^2; 80 * 1.5cm thickness * 1.0 g/cm^3 density = 120.0g mid estimate.
        CHECK(near(chicken.mid_g, 120.0));
        CHECK(near(chicken.low_g, 96.0));   // 120 * 0.8
        CHECK(near(chicken.high_g, 144.0)); // 120 * 1.2

        auto banana = estimate_portion_range(500000.0, 2600.0, FoodItem::BANANA_RAW);
        // 500000px^2 * 0.0001 = 50 cm^2; 50 * 2.0cm * 0.5 g/cm^3 = 50.0g mid estimate.
        CHECK(near(banana.mid_g, 50.0));
        CHECK(near(banana.low_g, 40.0));
        CHECK(near(banana.high_g, 60.0));
        std::cout << "  an 80cm^2 real chicken-breast area at a real 2600px reference-plate diameter "
                     "estimates to exactly 120.0g at the midpoint, with an honest [96.0g, 144.0g] real "
                     "range; a 50cm^2 real banana area under the identical scale estimates to exactly "
                     "50.0g at the midpoint, with an honest [40.0g, 60.0g] range\n";
    }

    std::cout << "\n-- Test 4: the real low/mid/high ordering of a portion estimate holds across every "
                 "food in the closed vocabulary, not merely the hand-picked examples above --\n";
    {
        int checked = 0;
        for (FoodItem food : ALL_FOOD_ITEMS) {
            auto est = estimate_portion_range(600000.0, 2600.0, food);
            CHECK(est.low_g < est.mid_g);
            CHECK(est.mid_g < est.high_g);
            checked++;
        }
        CHECK(checked == 6);
        std::cout << "  across all 6 foods in the closed vocabulary, the real low estimate is strictly "
                     "less than the real mid estimate, which is strictly less than the real high "
                     "estimate, with zero exceptions -- this section never collapses its own honest "
                     "range into a single value anywhere in the table\n";
    }

    std::cout << "\n-- Test 5: a daily tally over several real, human-confirmed log entries sums to an "
                 "exact, hand-computed total across every real macro --\n";
    {
        std::vector<FoodLogEntry> entries = {
            {FoodItem::CHICKEN_BREAST_COOKED, 150.0},
            {FoodItem::WHITE_RICE_COOKED, 200.0},
            {FoodItem::BROCCOLI_RAW, 100.0},
        };
        auto totals = daily_nutrient_tally(entries);
        // chicken 150g: 247.5 kcal, 46.5g protein, 0g carbs, 5.4g fat
        // rice 200g:    260.0 kcal, 5.4g protein, 56.0g carbs, 0.6g fat
        // broccoli 100g: 34.0 kcal, 2.8g protein, 6.6g carbs, 0.4g fat
        CHECK(near(totals.calories, 541.5));
        CHECK(near(totals.protein_g, 54.7));
        CHECK(near(totals.carbs_g, 62.6));
        CHECK(near(totals.fat_g, 6.4));
        std::cout << "  150g chicken breast + 200g white rice + 100g broccoli, all human-confirmed "
                     "gram amounts, tally to exactly 541.5 kcal, 54.7g protein, 62.6g carbs, and 6.4g "
                     "fat -- matching a real hand computation across every entry and every macro\n";
    }

    std::cout << "\n-- Test 6: daily-intake classification against a real stated target range is "
                 "correct on both sides and inclusive at each real boundary --\n";
    {
        DailyTargetRange range{1800.0, 2200.0};
        CHECK(classify_daily_intake(1500.0, range) == IntakeClassification::BELOW_RANGE);
        CHECK(classify_daily_intake(2000.0, range) == IntakeClassification::WITHIN_RANGE);
        CHECK(classify_daily_intake(2500.0, range) == IntakeClassification::ABOVE_RANGE);
        CHECK(classify_daily_intake(1800.0, range) == IntakeClassification::WITHIN_RANGE);  // lower bound, inclusive
        CHECK(classify_daily_intake(2200.0, range) == IntakeClassification::WITHIN_RANGE);  // upper bound, inclusive
        CHECK(classify_daily_intake(1799.9, range) == IntakeClassification::BELOW_RANGE);
        CHECK(classify_daily_intake(2200.1, range) == IntakeClassification::ABOVE_RANGE);
        std::cout << "  against a real stated 1800-2200 kcal target range: 1500 kcal classifies "
                     "BELOW_RANGE, 2000 kcal WITHIN_RANGE, 2500 kcal ABOVE_RANGE, and both real "
                     "boundary values themselves, 1800 and 2200 kcal, correctly classify WITHIN_RANGE "
                     "-- with 1799.9 and 2200.1 immediately outside on either side, confirming the "
                     "boundary is genuinely inclusive rather than accidentally so\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
