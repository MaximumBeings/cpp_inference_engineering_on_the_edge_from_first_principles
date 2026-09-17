// Chapter 24.3 -- Section 24.1's own closed-vocabulary edit plan and
// Section 24.2's own real cv::Mat engine are joined by a real,
// deterministic RESOLUTION TABLE: an explicit, tabulated mapping from
// every (intent, strength) pair in the closed vocabulary to a single
// concrete numeric parameter, plus a real, general property this table
// can be checked against on its own -- a STRONGER strength must always
// produce a LARGER real magnitude of change than a MODERATE or SUBTLE one
// for every intent where strength actually applies. This section also
// builds a real conflict detector that catches a directly self-
// contradictory plan (asking to both increase and decrease the same
// property, or to fully desaturate an image while also asking to
// increase its saturation) before it ever reaches Section 24.2's own
// engine.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_request_to_operation_mapping_table.cpp -o 03_request_to_operation_mapping_table
// Run:     ./03_request_to_operation_mapping_table

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
// PART 1: the closed edit-intent and strength vocabularies, repeated
// here per this book's own self-contained-file convention.
// =======================================================================
enum class EditIntent {
    BRIGHTNESS_INCREASE, BRIGHTNESS_DECREASE,
    CONTRAST_INCREASE, CONTRAST_DECREASE,
    WARMTH_INCREASE, WARMTH_DECREASE,
    SATURATION_INCREASE, SATURATION_DECREASE, DESATURATE_FULL,
    SHARPEN, DENOISE, CROP_CENTER_SQUARE,
};

enum class Strength { SUBTLE, MODERATE, STRONG };

struct EditPlanEntry {
    EditIntent intent;
    Strength strength;
};

struct EditPlan {
    std::vector<EditPlanEntry> entries;
};

const std::vector<EditIntent> ALL_INTENTS = {
    EditIntent::BRIGHTNESS_INCREASE, EditIntent::BRIGHTNESS_DECREASE,
    EditIntent::CONTRAST_INCREASE, EditIntent::CONTRAST_DECREASE,
    EditIntent::WARMTH_INCREASE, EditIntent::WARMTH_DECREASE,
    EditIntent::SATURATION_INCREASE, EditIntent::SATURATION_DECREASE, EditIntent::DESATURATE_FULL,
    EditIntent::SHARPEN, EditIntent::DENOISE, EditIntent::CROP_CENTER_SQUARE,
};

std::string intent_name(EditIntent intent) {
    switch (intent) {
        case EditIntent::BRIGHTNESS_INCREASE: return "BRIGHTNESS_INCREASE";
        case EditIntent::BRIGHTNESS_DECREASE: return "BRIGHTNESS_DECREASE";
        case EditIntent::CONTRAST_INCREASE: return "CONTRAST_INCREASE";
        case EditIntent::CONTRAST_DECREASE: return "CONTRAST_DECREASE";
        case EditIntent::WARMTH_INCREASE: return "WARMTH_INCREASE";
        case EditIntent::WARMTH_DECREASE: return "WARMTH_DECREASE";
        case EditIntent::SATURATION_INCREASE: return "SATURATION_INCREASE";
        case EditIntent::SATURATION_DECREASE: return "SATURATION_DECREASE";
        case EditIntent::DESATURATE_FULL: return "DESATURATE_FULL";
        case EditIntent::SHARPEN: return "SHARPEN";
        case EditIntent::DENOISE: return "DENOISE";
        case EditIntent::CROP_CENTER_SQUARE: return "CROP_CENTER_SQUARE";
    }
    return "UNKNOWN";
}

// =======================================================================
// PART 2: the real resolution table -- every (intent, strength) pair
// resolves to exactly one concrete, real numeric parameter, matching
// Section 24.2's own real cv::Mat engine constants exactly.
// =======================================================================
struct OperationParams {
    std::string op_name;
    double value = 0.0;
    bool strength_invariant = false;
};

OperationParams resolve_operation_params(EditIntent intent, Strength strength) {
    switch (intent) {
        case EditIntent::BRIGHTNESS_INCREASE:
            return {"BRIGHTNESS_BETA",
                    (strength == Strength::SUBTLE) ? 15.0 : (strength == Strength::MODERATE) ? 30.0 : 50.0, false};
        case EditIntent::BRIGHTNESS_DECREASE:
            return {"BRIGHTNESS_BETA",
                    -((strength == Strength::SUBTLE) ? 15.0 : (strength == Strength::MODERATE) ? 30.0 : 50.0), false};
        case EditIntent::CONTRAST_INCREASE:
            return {"CONTRAST_ALPHA",
                    (strength == Strength::SUBTLE) ? 1.10 : (strength == Strength::MODERATE) ? 1.25 : 1.50, false};
        case EditIntent::CONTRAST_DECREASE:
            return {"CONTRAST_ALPHA",
                    (strength == Strength::SUBTLE) ? 0.90 : (strength == Strength::MODERATE) ? 0.75 : 0.60, false};
        case EditIntent::WARMTH_INCREASE:
            return {"WARMTH_SHIFT",
                    (strength == Strength::SUBTLE) ? 8.0 : (strength == Strength::MODERATE) ? 16.0 : 28.0, false};
        case EditIntent::WARMTH_DECREASE:
            return {"WARMTH_SHIFT",
                    -((strength == Strength::SUBTLE) ? 8.0 : (strength == Strength::MODERATE) ? 16.0 : 28.0), false};
        case EditIntent::SATURATION_INCREASE:
            return {"SATURATION_SCALE",
                    (strength == Strength::SUBTLE) ? 1.15 : (strength == Strength::MODERATE) ? 1.35 : 1.60, false};
        case EditIntent::SATURATION_DECREASE:
            return {"SATURATION_SCALE",
                    (strength == Strength::SUBTLE) ? 0.85 : (strength == Strength::MODERATE) ? 0.65 : 0.40, false};
        case EditIntent::DESATURATE_FULL:
            // A real, absolute operation: "fully" admits no partial degree, so this entry is the same
            // regardless of whatever strength token accompanied it.
            return {"SATURATION_SCALE", 0.0, true};
        case EditIntent::SHARPEN:
            return {"SHARPEN_K",
                    (strength == Strength::SUBTLE) ? 0.5 : (strength == Strength::MODERATE) ? 1.0 : 1.75, false};
        case EditIntent::DENOISE:
            return {"DENOISE_KERNEL_SIZE",
                    (strength == Strength::SUBTLE) ? 3.0 : (strength == Strength::MODERATE) ? 5.0 : 7.0, false};
        case EditIntent::CROP_CENTER_SQUARE:
            // A real, structural operation with no tunable degree at all.
            return {"CROP_TO_CENTER_SQUARE", 0.0, true};
    }
    return {"UNKNOWN", 0.0, true};
}

// The real "no-op" value for each op family -- the point a parameter
// would sit at if the requested edit had zero real effect. Real
// magnitude of change is distance from this neutral point, not the raw
// parameter value itself, since a DECREASE op's own alpha or scale value
// gets SMALLER (moving away from 1.0) as its own real effect gets
// stronger.
double neutral_value_for_op(const std::string& op_name) {
    if (op_name == "CONTRAST_ALPHA" || op_name == "SATURATION_SCALE") return 1.0;
    return 0.0;
}

// =======================================================================
// PART 3: the real conflict detector.
// =======================================================================
struct ConflictEntry {
    EditIntent a, b;
};

const std::vector<std::pair<EditIntent, EditIntent>> OPPOSING_PAIRS = {
    {EditIntent::BRIGHTNESS_INCREASE, EditIntent::BRIGHTNESS_DECREASE},
    {EditIntent::CONTRAST_INCREASE, EditIntent::CONTRAST_DECREASE},
    {EditIntent::WARMTH_INCREASE, EditIntent::WARMTH_DECREASE},
    {EditIntent::SATURATION_INCREASE, EditIntent::SATURATION_DECREASE},
    {EditIntent::SATURATION_INCREASE, EditIntent::DESATURATE_FULL},
    {EditIntent::SATURATION_DECREASE, EditIntent::DESATURATE_FULL},
};

bool plan_has_intent(const EditPlan& plan, EditIntent intent) {
    return std::any_of(plan.entries.begin(), plan.entries.end(),
                        [&](const EditPlanEntry& e) { return e.intent == intent; });
}

std::vector<ConflictEntry> detect_conflicting_intents(const EditPlan& plan) {
    std::vector<ConflictEntry> conflicts;
    for (const auto& [a, b] : OPPOSING_PAIRS) {
        if (plan_has_intent(plan, a) && plan_has_intent(plan, b)) {
            conflicts.push_back({a, b});
        }
    }
    return conflicts;
}

// Resolving a plan into concrete operations is only meaningful once it
// is known to be conflict-free -- this function does not itself check,
// leaving that decision to its caller, exactly as Section 21.3's own
// cap-checking function left the decision of what to DO about a flagged
// discrepancy to a human reviewer rather than making it silently here.
std::vector<OperationParams> resolve_plan_to_operations(const EditPlan& plan) {
    std::vector<OperationParams> ops;
    for (const auto& entry : plan.entries) {
        ops.push_back(resolve_operation_params(entry.intent, entry.strength));
    }
    return ops;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 24.3: A Request-to-Operation Mapping Table\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the resolution table's own real values match hand-computed constants "
                 "exactly for a representative sample across every op family --\n";
    {
        CHECK(resolve_operation_params(EditIntent::BRIGHTNESS_INCREASE, Strength::SUBTLE).value == 15.0);
        CHECK(resolve_operation_params(EditIntent::CONTRAST_DECREASE, Strength::STRONG).value == 0.60);
        CHECK(resolve_operation_params(EditIntent::WARMTH_DECREASE, Strength::MODERATE).value == -16.0);
        CHECK(resolve_operation_params(EditIntent::SATURATION_INCREASE, Strength::STRONG).value == 1.60);
        CHECK(resolve_operation_params(EditIntent::SHARPEN, Strength::STRONG).value == 1.75);
        CHECK(resolve_operation_params(EditIntent::DENOISE, Strength::SUBTLE).value == 3.0);
        std::cout << "  BRIGHTNESS_INCREASE/SUBTLE resolves to exactly 15.0; CONTRAST_DECREASE/STRONG to "
                     "exactly 0.60; WARMTH_DECREASE/MODERATE to exactly -16.0; SATURATION_INCREASE/STRONG "
                     "to exactly 1.60; SHARPEN/STRONG to exactly 1.75; DENOISE/SUBTLE to exactly 3.0\n";
    }

    std::cout << "\n-- Test 2: DESATURATE_FULL and CROP_CENTER_SQUARE resolve to the identical real value "
                 "at all 3 strength levels -- their own stated real absoluteness, verified directly --\n";
    {
        auto d1 = resolve_operation_params(EditIntent::DESATURATE_FULL, Strength::SUBTLE);
        auto d2 = resolve_operation_params(EditIntent::DESATURATE_FULL, Strength::MODERATE);
        auto d3 = resolve_operation_params(EditIntent::DESATURATE_FULL, Strength::STRONG);
        CHECK(d1.strength_invariant && d2.strength_invariant && d3.strength_invariant);
        CHECK(d1.value == d2.value && d2.value == d3.value);
        auto c1 = resolve_operation_params(EditIntent::CROP_CENTER_SQUARE, Strength::SUBTLE);
        auto c3 = resolve_operation_params(EditIntent::CROP_CENTER_SQUARE, Strength::STRONG);
        CHECK(c1.strength_invariant && c3.strength_invariant);
        CHECK(c1.value == c3.value);
        std::cout << "  DESATURATE_FULL resolves to the identical value regardless of whether it is "
                     "paired with SUBTLE, MODERATE, or STRONG; CROP_CENTER_SQUARE does the same -- both "
                     "correctly marked strength_invariant in the table itself\n";
    }

    std::cout << "\n-- Test 3: a real, general table-integrity property -- STRONG always produces a "
                 "strictly larger real magnitude of change than MODERATE, which is strictly larger than "
                 "SUBTLE -- holds across every strength-sensitive intent in the entire table --\n";
    {
        int checked_intents = 0;
        for (EditIntent intent : ALL_INTENTS) {
            auto subtle = resolve_operation_params(intent, Strength::SUBTLE);
            if (subtle.strength_invariant) continue;
            auto moderate = resolve_operation_params(intent, Strength::MODERATE);
            auto strong = resolve_operation_params(intent, Strength::STRONG);
            double neutral = neutral_value_for_op(subtle.op_name);
            double mag_subtle = std::abs(subtle.value - neutral);
            double mag_moderate = std::abs(moderate.value - neutral);
            double mag_strong = std::abs(strong.value - neutral);
            CHECK(mag_subtle < mag_moderate);
            CHECK(mag_moderate < mag_strong);
            checked_intents++;
        }
        CHECK(checked_intents == 10);  // all 12 intents minus the 2 strength-invariant ones
        std::cout << "  across all " << checked_intents << " strength-sensitive intents in the table "
                     "(every intent except DESATURATE_FULL and CROP_CENTER_SQUARE), each intent's own "
                     "real distance from its own neutral value strictly increases from SUBTLE to "
                     "MODERATE to STRONG, with zero exceptions\n";
    }

    std::cout << "\n-- Test 4: directly opposing intents present in the same plan are flagged, while a "
                 "plan combining unrelated, non-conflicting intents is not --\n";
    {
        EditPlan brightness_conflict{{
            {EditIntent::BRIGHTNESS_INCREASE, Strength::SUBTLE},
            {EditIntent::BRIGHTNESS_DECREASE, Strength::MODERATE},
        }};
        EditPlan saturation_conflict{{
            {EditIntent::SATURATION_INCREASE, Strength::STRONG},
            {EditIntent::DESATURATE_FULL, Strength::SUBTLE},
        }};
        EditPlan clean_plan{{
            {EditIntent::BRIGHTNESS_INCREASE, Strength::SUBTLE},
            {EditIntent::SHARPEN, Strength::MODERATE},
            {EditIntent::CROP_CENTER_SQUARE, Strength::SUBTLE},
        }};
        auto conflicts1 = detect_conflicting_intents(brightness_conflict);
        auto conflicts2 = detect_conflicting_intents(saturation_conflict);
        auto conflicts3 = detect_conflicting_intents(clean_plan);
        CHECK(conflicts1.size() == 1);
        CHECK(conflicts2.size() == 1);
        CHECK(conflicts3.empty());
        std::cout << "  BRIGHTNESS_INCREASE + BRIGHTNESS_DECREASE together are flagged as 1 conflict; "
                     "SATURATION_INCREASE + DESATURATE_FULL together are flagged as 1 conflict; a "
                     "3-intent plan combining BRIGHTNESS_INCREASE, SHARPEN, and CROP_CENTER_SQUARE -- "
                     "none of which oppose each other -- is flagged with zero conflicts\n";
    }

    std::cout << "\n-- Test 5: a conflict-free plan resolves to its own exact, real sequence of concrete "
                 "operation parameters, in the same order as the plan's own entries --\n";
    {
        EditPlan plan{{
            {EditIntent::CONTRAST_INCREASE, Strength::SUBTLE},
            {EditIntent::SATURATION_INCREASE, Strength::SUBTLE},
            {EditIntent::SHARPEN, Strength::SUBTLE},
        }};
        CHECK(detect_conflicting_intents(plan).empty());
        auto ops = resolve_plan_to_operations(plan);
        CHECK(ops.size() == 3);
        CHECK(ops[0].op_name == "CONTRAST_ALPHA" && ops[0].value == 1.10);
        CHECK(ops[1].op_name == "SATURATION_SCALE" && ops[1].value == 1.15);
        CHECK(ops[2].op_name == "SHARPEN_K" && ops[2].value == 0.5);
        std::cout << "  this section's own \"make it look better\" default plan, confirmed conflict-free, "
                     "resolves to exactly 3 concrete operations in order: CONTRAST_ALPHA=1.10, "
                     "SATURATION_SCALE=1.15, SHARPEN_K=0.5 -- the identical real values Section 24.2's "
                     "own engine uses for these same 3 intents at SUBTLE strength\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
