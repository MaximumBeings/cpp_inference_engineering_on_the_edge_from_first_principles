// Chapter 24.4 -- A real photo-editing conversation is rarely one
// request; it is a request followed by "a bit more," "actually undo
// that," or "no, make it cooler instead." This section builds a real,
// structural conversation-state engine over Section 24.1's own edit
// plan: a strength can be nudged up (capped at a real STRONG ceiling,
// never silently wrapping) or down (an already-SUBTLE edit nudged down
// is removed entirely, not left unchanged), an entire refinement can be
// undone via a real history stack, and -- this section's own central
// discipline -- a follow-up request that directly opposes an intent
// ALREADY in the current plan is treated as the user changing their
// mind, and REPLACES the old intent, rather than holding both
// simultaneously the way Section 24.3's own conflict detector would
// correctly flag in a single, freshly-parsed plan with no conversation
// history behind it.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_iterative_refinement_conversation.cpp -o 04_iterative_refinement_conversation
// Run:     ./04_iterative_refinement_conversation

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
    bool operator==(const EditPlanEntry& other) const {
        return intent == other.intent && strength == other.strength;
    }
};

struct EditPlan {
    std::vector<EditPlanEntry> entries;
    bool operator==(const EditPlan& other) const { return entries == other.entries; }
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

bool find_opposite_intent(EditIntent intent, EditIntent& opposite_out) {
    for (const auto& [a, b] : OPPOSING_PAIRS) {
        if (a == intent) { opposite_out = b; return true; }
        if (b == intent) { opposite_out = a; return true; }
    }
    return false;
}

Strength step_up(Strength s) {
    return (s == Strength::SUBTLE) ? Strength::MODERATE : Strength::STRONG;
}
Strength step_down(Strength s) {
    return (s == Strength::STRONG) ? Strength::MODERATE : Strength::SUBTLE;
}
bool is_at_max(Strength s) { return s == Strength::STRONG; }
bool is_at_min(Strength s) { return s == Strength::SUBTLE; }

// =======================================================================
// PART 2: the conversation state -- a current plan and a real history
// stack of prior plan states, enabling a real undo.
// =======================================================================
struct EditSession {
    EditPlan current;
    std::vector<EditPlan> history;
};

enum class RefinementAction { INCREASE_STRENGTH, DECREASE_STRENGTH, UNDO_LAST, ADD_INTENT, REMOVE_INTENT };

struct RefinementRequest {
    RefinementAction action;
    EditIntent target_intent = EditIntent::SHARPEN;
    Strength new_strength = Strength::SUBTLE;
};

enum class RefinementOutcome {
    APPLIED,
    CAPPED_AT_MAX,
    REMOVED_AT_MIN,
    REPLACED_CONFLICTING_INTENT,
    NOTHING_TO_UNDO,
    INTENT_NOT_PRESENT,
};

struct RefinementResult {
    RefinementOutcome outcome;
    EditPlan new_plan;
    std::string detail;
};

// =======================================================================
// PART 3: the refinement engine.
// =======================================================================
RefinementResult apply_refinement(EditSession& session, const RefinementRequest& req) {
    RefinementResult result;
    EditPlan& plan = session.current;

    switch (req.action) {
        case RefinementAction::UNDO_LAST: {
            if (session.history.empty()) {
                result.outcome = RefinementOutcome::NOTHING_TO_UNDO;
                result.new_plan = plan;
                result.detail = "no prior plan state to undo to";
                return result;
            }
            EditPlan previous = session.history.back();
            session.history.pop_back();
            session.current = previous;
            result.outcome = RefinementOutcome::APPLIED;
            result.new_plan = session.current;
            result.detail = "reverted to the prior plan state";
            return result;
        }
        case RefinementAction::INCREASE_STRENGTH: {
            auto it = std::find_if(plan.entries.begin(), plan.entries.end(),
                                    [&](const EditPlanEntry& e) { return e.intent == req.target_intent; });
            if (it == plan.entries.end()) {
                result.outcome = RefinementOutcome::INTENT_NOT_PRESENT;
                result.new_plan = plan;
                result.detail = "intent is not present in the current plan; nothing to increase";
                return result;
            }
            if (is_at_max(it->strength)) {
                result.outcome = RefinementOutcome::CAPPED_AT_MAX;
                result.new_plan = plan;
                result.detail = "already at the real maximum STRONG strength; no further increase applied";
                return result;
            }
            session.history.push_back(plan);
            it->strength = step_up(it->strength);
            result.outcome = RefinementOutcome::APPLIED;
            result.new_plan = plan;
            result.detail = "strength increased by one real level";
            return result;
        }
        case RefinementAction::DECREASE_STRENGTH: {
            auto it = std::find_if(plan.entries.begin(), plan.entries.end(),
                                    [&](const EditPlanEntry& e) { return e.intent == req.target_intent; });
            if (it == plan.entries.end()) {
                result.outcome = RefinementOutcome::INTENT_NOT_PRESENT;
                result.new_plan = plan;
                result.detail = "intent is not present in the current plan; nothing to decrease";
                return result;
            }
            session.history.push_back(plan);
            if (is_at_min(it->strength)) {
                plan.entries.erase(it);
                result.outcome = RefinementOutcome::REMOVED_AT_MIN;
                result.new_plan = plan;
                result.detail =
                    "already at the real minimum SUBTLE strength; the intent was removed entirely "
                    "rather than left unchanged";
                return result;
            }
            it->strength = step_down(it->strength);
            result.outcome = RefinementOutcome::APPLIED;
            result.new_plan = plan;
            result.detail = "strength decreased by one real level";
            return result;
        }
        case RefinementAction::ADD_INTENT: {
            EditIntent opposite;
            bool has_opposite = find_opposite_intent(req.target_intent, opposite) && plan_has_intent(plan, opposite);
            session.history.push_back(plan);
            if (has_opposite) {
                plan.entries.erase(std::remove_if(plan.entries.begin(), plan.entries.end(),
                                                   [&](const EditPlanEntry& e) { return e.intent == opposite; }),
                                    plan.entries.end());
                plan.entries.push_back({req.target_intent, req.new_strength});
                result.outcome = RefinementOutcome::REPLACED_CONFLICTING_INTENT;
                result.new_plan = plan;
                result.detail =
                    "replaced the existing opposing intent with this new request, rather than holding "
                    "both simultaneously";
                return result;
            }
            auto it = std::find_if(plan.entries.begin(), plan.entries.end(),
                                    [&](const EditPlanEntry& e) { return e.intent == req.target_intent; });
            if (it != plan.entries.end()) {
                it->strength = req.new_strength;
            } else {
                plan.entries.push_back({req.target_intent, req.new_strength});
            }
            result.outcome = RefinementOutcome::APPLIED;
            result.new_plan = plan;
            result.detail = "added to the current plan";
            return result;
        }
        case RefinementAction::REMOVE_INTENT: {
            auto it = std::find_if(plan.entries.begin(), plan.entries.end(),
                                    [&](const EditPlanEntry& e) { return e.intent == req.target_intent; });
            if (it == plan.entries.end()) {
                result.outcome = RefinementOutcome::INTENT_NOT_PRESENT;
                result.new_plan = plan;
                result.detail = "intent is not present in the current plan; nothing to remove";
                return result;
            }
            session.history.push_back(plan);
            plan.entries.erase(it);
            result.outcome = RefinementOutcome::APPLIED;
            result.new_plan = plan;
            result.detail = "removed from the current plan";
            return result;
        }
    }
    return result;  // unreachable: every RefinementAction enumerator is handled above.
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 24.4: Iterative Refinement Through Conversation\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: repeated \"more\" requests step strength up one real level at a time, "
                 "capping at STRONG rather than wrapping or erroring --\n";
    {
        EditSession session{EditPlan{{{EditIntent::SHARPEN, Strength::SUBTLE}}}, {}};
        auto r1 = apply_refinement(session, {RefinementAction::INCREASE_STRENGTH, EditIntent::SHARPEN, {}});
        CHECK(r1.outcome == RefinementOutcome::APPLIED && r1.new_plan.entries[0].strength == Strength::MODERATE);
        auto r2 = apply_refinement(session, {RefinementAction::INCREASE_STRENGTH, EditIntent::SHARPEN, {}});
        CHECK(r2.outcome == RefinementOutcome::APPLIED && r2.new_plan.entries[0].strength == Strength::STRONG);
        auto r3 = apply_refinement(session, {RefinementAction::INCREASE_STRENGTH, EditIntent::SHARPEN, {}});
        CHECK(r3.outcome == RefinementOutcome::CAPPED_AT_MAX);
        CHECK(session.current.entries[0].strength == Strength::STRONG);
        std::cout << "  SHARPEN steps SUBTLE -> MODERATE -> STRONG across 2 real \"more\" requests; a "
                     "3rd \"more\" request is reported CAPPED_AT_MAX, leaving the plan at STRONG rather "
                     "than silently wrapping back to SUBTLE or erroring\n";
    }

    std::cout << "\n-- Test 2: a \"less\" request on an already-SUBTLE intent removes it entirely rather "
                 "than leaving it unchanged --\n";
    {
        EditSession session{EditPlan{{{EditIntent::BRIGHTNESS_INCREASE, Strength::MODERATE}}}, {}};
        auto r1 = apply_refinement(session, {RefinementAction::DECREASE_STRENGTH, EditIntent::BRIGHTNESS_INCREASE, {}});
        CHECK(r1.outcome == RefinementOutcome::APPLIED && session.current.entries[0].strength == Strength::SUBTLE);
        auto r2 = apply_refinement(session, {RefinementAction::DECREASE_STRENGTH, EditIntent::BRIGHTNESS_INCREASE, {}});
        CHECK(r2.outcome == RefinementOutcome::REMOVED_AT_MIN);
        CHECK(!plan_has_intent(session.current, EditIntent::BRIGHTNESS_INCREASE));
        std::cout << "  BRIGHTNESS_INCREASE steps MODERATE -> SUBTLE on the first \"less\" request; a "
                     "second \"less\" request on an already-SUBTLE edit removes BRIGHTNESS_INCREASE from "
                     "the plan entirely -- \"less\" than the smallest real amount means none at all\n";
    }

    std::cout << "\n-- Test 3: undo reverts exactly one refinement at a time through a real history stack, "
                 "and undoing with an empty history is honestly refused --\n";
    {
        EditSession session{EditPlan{{{EditIntent::SHARPEN, Strength::SUBTLE}}}, {}};
        EditPlan original = session.current;
        apply_refinement(session, {RefinementAction::INCREASE_STRENGTH, EditIntent::SHARPEN, {}});
        apply_refinement(session, {RefinementAction::INCREASE_STRENGTH, EditIntent::SHARPEN, {}});
        CHECK(session.current.entries[0].strength == Strength::STRONG);
        auto undo1 = apply_refinement(session, {RefinementAction::UNDO_LAST, {}, {}});
        CHECK(undo1.outcome == RefinementOutcome::APPLIED);
        CHECK(session.current.entries[0].strength == Strength::MODERATE);
        auto undo2 = apply_refinement(session, {RefinementAction::UNDO_LAST, {}, {}});
        CHECK(session.current == original);
        auto undo3 = apply_refinement(session, {RefinementAction::UNDO_LAST, {}, {}});
        CHECK(undo3.outcome == RefinementOutcome::NOTHING_TO_UNDO);
        CHECK(session.current == original);
        (void)undo2;
        std::cout << "  after 2 real \"more\" requests (SUBTLE -> MODERATE -> STRONG), one undo reverts "
                     "exactly to MODERATE, a second undo reverts exactly to the original SUBTLE plan, and "
                     "a third undo with nothing left in history is honestly reported NOTHING_TO_UNDO, "
                     "leaving the plan unchanged rather than erroring or reverting further\n";
    }

    std::cout << "\n-- Test 4: adding a genuinely new, non-conflicting intent is simply applied --\n";
    {
        EditSession session{EditPlan{{{EditIntent::SHARPEN, Strength::SUBTLE}}}, {}};
        auto result = apply_refinement(session, {RefinementAction::ADD_INTENT, EditIntent::DENOISE, Strength::MODERATE});
        CHECK(result.outcome == RefinementOutcome::APPLIED);
        CHECK(plan_has_intent(session.current, EditIntent::DENOISE));
        CHECK(plan_has_intent(session.current, EditIntent::SHARPEN));
        CHECK(session.current.entries.size() == 2);
        std::cout << "  adding DENOISE at MODERATE to a plan that already has SHARPEN produces a clean "
                     "2-intent plan with both present -- no conflict, so nothing is replaced\n";
    }

    std::cout << "\n-- Test 5: a follow-up request that directly opposes an intent already in the plan "
                 "REPLACES it -- this section's own central discipline, distinct from Section 24.3's own "
                 "flag-only conflict detector --\n";
    {
        EditSession session{EditPlan{{{EditIntent::WARMTH_INCREASE, Strength::MODERATE}}}, {}};
        auto result = apply_refinement(session, {RefinementAction::ADD_INTENT, EditIntent::WARMTH_DECREASE, Strength::STRONG});
        CHECK(result.outcome == RefinementOutcome::REPLACED_CONFLICTING_INTENT);
        CHECK(plan_has_intent(session.current, EditIntent::WARMTH_DECREASE));
        CHECK(!plan_has_intent(session.current, EditIntent::WARMTH_INCREASE));
        CHECK(session.current.entries.size() == 1);
        std::cout << "  a plan already containing WARMTH_INCREASE at MODERATE, given a follow-up request "
                     "for WARMTH_DECREASE at STRONG, ends up containing ONLY WARMTH_DECREASE -- the "
                     "user's own follow-up is treated as changing their mind, not as a second, "
                     "simultaneously-held contradictory instruction\n";
    }

    std::cout << "\n-- Test 6: a full multi-turn conversation produces the exact real final plan across a "
                 "sequence of refinements, including one that replaces a conflicting intent and one that "
                 "undoes a prior step --\n";
    {
        EditSession session{EditPlan{{
                                 {EditIntent::CONTRAST_INCREASE, Strength::SUBTLE},
                                 {EditIntent::SATURATION_INCREASE, Strength::SUBTLE},
                                 {EditIntent::SHARPEN, Strength::SUBTLE},
                             }},
                             {}};
        apply_refinement(session, {RefinementAction::INCREASE_STRENGTH, EditIntent::SHARPEN, {}});
        apply_refinement(session, {RefinementAction::ADD_INTENT, EditIntent::WARMTH_INCREASE, Strength::MODERATE});
        auto replace_result =
            apply_refinement(session, {RefinementAction::ADD_INTENT, EditIntent::WARMTH_DECREASE, Strength::STRONG});
        CHECK(replace_result.outcome == RefinementOutcome::REPLACED_CONFLICTING_INTENT);
        CHECK(session.current.entries.size() == 4);
        auto undo_result = apply_refinement(session, {RefinementAction::UNDO_LAST, {}, {}});
        CHECK(undo_result.outcome == RefinementOutcome::APPLIED);
        CHECK(plan_has_intent(session.current, EditIntent::WARMTH_INCREASE));
        CHECK(!plan_has_intent(session.current, EditIntent::WARMTH_DECREASE));
        CHECK(session.current.entries.size() == 4);
        bool sharpen_still_moderate = false;
        for (const auto& e : session.current.entries) {
            if (e.intent == EditIntent::SHARPEN && e.strength == Strength::MODERATE) sharpen_still_moderate = true;
        }
        CHECK(sharpen_still_moderate);
        std::cout << "  starting from the default 3-intent plan: SHARPEN is bumped to MODERATE, "
                     "WARMTH_INCREASE is added, WARMTH_DECREASE then replaces it (4 intents total); "
                     "undoing that last step restores WARMTH_INCREASE while correctly leaving the "
                     "earlier SHARPEN bump at MODERATE untouched -- undo reverts exactly one real "
                     "conversational turn, not the whole session\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
