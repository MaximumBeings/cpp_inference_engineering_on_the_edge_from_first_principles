// Chapter 24.1 -- "Make it look better" is not an instruction a real
// image-processing pipeline can execute: it names no specific operation,
// no direction, and no strength. This section builds the same real
// discipline this book applied to Chapter 20.3's own triage-report
// schema: a stated system prompt that forces a vision-language model to
// translate ANY natural-language edit request, however vague, into a
// CLOSED, small vocabulary of named edit intents and strengths -- never a
// free-form description of "enhancements" -- plus a strict, structural
// parser that rejects anything the model outputs outside that closed
// vocabulary, naming the exact reason for every rejection.
//
// A note on this section's own honest scope: this section's own stated
// system prompt (below) is a real, specific contract a real deployment
// would send to a real vision-language model; this section, like Section
// 19.2 and Section 21.1 before it, treats the MODEL'S OWN completion text
// as a stated stand-in wherever no real, trained model is being run, and
// verifies only the real, from-scratch parsing and validation logic that
// would sit downstream of it.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_edit_interpretation_prompt_and_plan_parser.cpp -o 01_edit_interpretation_prompt_and_plan_parser
// Run:     ./01_edit_interpretation_prompt_and_plan_parser

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the stated system prompt -- a real, specific contract, not a
// vague instruction to "improve" a photo.
// =======================================================================
const std::string EDIT_INTERPRETATION_SYSTEM_PROMPT = R"PROMPT(
You are a photo-edit interpretation system. Given a user's natural-
language request about a photograph, output ONE edit intent per line, in
the exact format:

INTENT=<intent>;STRENGTH=<strength>

<intent> must be exactly one of: BRIGHTNESS_INCREASE, BRIGHTNESS_DECREASE,
CONTRAST_INCREASE, CONTRAST_DECREASE, WARMTH_INCREASE, WARMTH_DECREASE,
SATURATION_INCREASE, SATURATION_DECREASE, DESATURATE_FULL, SHARPEN,
DENOISE, CROP_CENTER_SQUARE.

<strength> must be exactly one of: SUBTLE, MODERATE, STRONG.

Never output free-form text, a description of the edit, or any token
outside these two closed lists. A request naming no specific edit (for
example, "make it look better") maps to exactly this conservative default,
and nothing more aggressive:

INTENT=CONTRAST_INCREASE;STRENGTH=SUBTLE
INTENT=SATURATION_INCREASE;STRENGTH=SUBTLE
INTENT=SHARPEN;STRENGTH=SUBTLE
)PROMPT";

// =======================================================================
// PART 2: the closed edit-intent and strength vocabularies.
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

const std::vector<std::pair<std::string, EditIntent>> INTENT_NAMES = {
    {"BRIGHTNESS_INCREASE", EditIntent::BRIGHTNESS_INCREASE},
    {"BRIGHTNESS_DECREASE", EditIntent::BRIGHTNESS_DECREASE},
    {"CONTRAST_INCREASE", EditIntent::CONTRAST_INCREASE},
    {"CONTRAST_DECREASE", EditIntent::CONTRAST_DECREASE},
    {"WARMTH_INCREASE", EditIntent::WARMTH_INCREASE},
    {"WARMTH_DECREASE", EditIntent::WARMTH_DECREASE},
    {"SATURATION_INCREASE", EditIntent::SATURATION_INCREASE},
    {"SATURATION_DECREASE", EditIntent::SATURATION_DECREASE},
    {"DESATURATE_FULL", EditIntent::DESATURATE_FULL},
    {"SHARPEN", EditIntent::SHARPEN},
    {"DENOISE", EditIntent::DENOISE},
    {"CROP_CENTER_SQUARE", EditIntent::CROP_CENTER_SQUARE},
};

const std::vector<std::pair<std::string, Strength>> STRENGTH_NAMES = {
    {"SUBTLE", Strength::SUBTLE},
    {"MODERATE", Strength::MODERATE},
    {"STRONG", Strength::STRONG},
};

bool lookup_intent(const std::string& name, EditIntent& out) {
    for (const auto& [n, v] : INTENT_NAMES) {
        if (n == name) { out = v; return true; }
    }
    return false;
}

bool lookup_strength(const std::string& name, Strength& out) {
    for (const auto& [n, v] : STRENGTH_NAMES) {
        if (n == name) { out = v; return true; }
    }
    return false;
}

std::string intent_name(EditIntent intent) {
    for (const auto& [n, v] : INTENT_NAMES) if (v == intent) return n;
    return "UNKNOWN";
}

// =======================================================================
// PART 3: the strict structural parser -- every rejection names the
// exact, specific reason, exactly the same discipline as Section 20.3's
// own triage-report parser and Section 21.1's own structured extraction
// parser.
// =======================================================================
struct ParseResult {
    bool valid = false;
    EditPlan plan;
    std::string error;
};

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) parts.push_back(item);
    return parts;
}

std::string trim(const std::string& s) {
    std::size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

ParseResult parse_edit_plan(const std::string& text) {
    ParseResult result;
    std::vector<EditIntent> seen;

    for (const auto& raw_line : split(text, '\n')) {
        std::string line = trim(raw_line);
        if (line.empty()) continue;

        auto fields = split(line, ';');
        if (fields.size() != 2) {
            result.error = "malformed line \"" + line + "\": expected exactly 2 fields (INTENT=...;STRENGTH=...)";
            return result;
        }

        auto intent_field = split(trim(fields[0]), '=');
        auto strength_field = split(trim(fields[1]), '=');
        if (intent_field.size() != 2 || trim(intent_field[0]) != "INTENT" ||
            strength_field.size() != 2 || trim(strength_field[0]) != "STRENGTH") {
            result.error = "malformed line \"" + line + "\": expected the exact field order INTENT=...;STRENGTH=...";
            return result;
        }

        std::string intent_str = trim(intent_field[1]);
        std::string strength_str = trim(strength_field[1]);

        EditIntent intent;
        if (!lookup_intent(intent_str, intent)) {
            result.error = "unknown edit intent \"" + intent_str + "\" is not in this system's own closed vocabulary";
            return result;
        }

        Strength strength;
        if (!lookup_strength(strength_str, strength)) {
            result.error = "unknown strength \"" + strength_str + "\" is not in this system's own closed vocabulary";
            return result;
        }

        for (EditIntent s : seen) {
            if (s == intent) {
                result.error = "duplicate intent \"" + intent_str + "\" appears more than once in the same plan";
                return result;
            }
        }

        seen.push_back(intent);
        result.plan.entries.push_back({intent, strength});
    }

    result.valid = true;
    return result;
}

// A real, stated conservative default this section's own system prompt
// commits to for any vague request naming no specific edit.
EditPlan default_plan_for_vague_request() {
    return EditPlan{{
        {EditIntent::CONTRAST_INCREASE, Strength::SUBTLE},
        {EditIntent::SATURATION_INCREASE, Strength::SUBTLE},
        {EditIntent::SHARPEN, Strength::SUBTLE},
    }};
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 24.1: An Edit-Interpretation Prompt and Structured Plan Parser\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a valid multi-intent completion parses to exactly its own real entries, in "
                 "order, and an empty request parses to a valid, empty plan --\n";
    {
        std::string text = "INTENT=BRIGHTNESS_INCREASE;STRENGTH=MODERATE\nINTENT=SHARPEN;STRENGTH=STRONG";
        auto result = parse_edit_plan(text);
        CHECK(result.valid);
        CHECK(result.plan.entries.size() == 2);
        CHECK(result.plan.entries[0].intent == EditIntent::BRIGHTNESS_INCREASE);
        CHECK(result.plan.entries[0].strength == Strength::MODERATE);
        CHECK(result.plan.entries[1].intent == EditIntent::SHARPEN);
        CHECK(result.plan.entries[1].strength == Strength::STRONG);

        auto empty_result = parse_edit_plan("");
        CHECK(empty_result.valid);
        CHECK(empty_result.plan.entries.empty());
        std::cout << "  a 2-line completion parses to exactly 2 entries in order (BRIGHTNESS_INCREASE at "
                     "MODERATE, SHARPEN at STRONG); an empty request parses to a valid plan with zero "
                     "entries rather than an error\n";
    }

    std::cout << "\n-- Test 2: an intent token outside the closed vocabulary is rejected, naming the "
                 "specific unrecognized token --\n";
    {
        auto result = parse_edit_plan("INTENT=REMOVE_BACKGROUND;STRENGTH=STRONG");
        CHECK(!result.valid);
        CHECK(result.error.find("REMOVE_BACKGROUND") != std::string::npos);
        std::cout << "  \"REMOVE_BACKGROUND,\" a real edit a user might plausibly request but which is "
                     "not in this section's own stated closed vocabulary, is rejected by name\n";
    }

    std::cout << "\n-- Test 3: a strength token outside the closed vocabulary is rejected, naming the "
                 "specific unrecognized token --\n";
    {
        auto result = parse_edit_plan("INTENT=SHARPEN;STRENGTH=EXTREME");
        CHECK(!result.valid);
        CHECK(result.error.find("EXTREME") != std::string::npos);
        std::cout << "  \"EXTREME,\" a plausible-sounding but non-existent strength level, is rejected "
                     "by name\n";
    }

    std::cout << "\n-- Test 4: the same intent appearing twice in one plan is rejected, naming that "
                 "specific intent --\n";
    {
        auto result = parse_edit_plan(
            "INTENT=BRIGHTNESS_INCREASE;STRENGTH=SUBTLE\nINTENT=BRIGHTNESS_INCREASE;STRENGTH=STRONG");
        CHECK(!result.valid);
        CHECK(result.error.find("BRIGHTNESS_INCREASE") != std::string::npos);
        std::cout << "  BRIGHTNESS_INCREASE appearing at both SUBTLE and STRONG in the same plan is "
                     "rejected as a duplicate, naming the specific intent rather than silently keeping "
                     "the last one\n";
    }

    std::cout << "\n-- Test 5: a structurally malformed line is rejected, naming the exact line --\n";
    {
        auto missing_field = parse_edit_plan("INTENT=SHARPEN");
        auto wrong_order = parse_edit_plan("STRENGTH=SUBTLE;INTENT=SHARPEN");
        CHECK(!missing_field.valid);
        CHECK(missing_field.error.find("INTENT=SHARPEN") != std::string::npos);
        CHECK(!wrong_order.valid);
        std::cout << "  a line missing its own required STRENGTH field, and a line with the two fields "
                     "in the wrong order, are both rejected, each naming the specific malformed line\n";
    }

    std::cout << "\n-- Test 6: this section's own stated default plan for a vague request parses to "
                 "exactly the conservative 3-entry plan the system prompt itself commits to --\n";
    {
        std::string vague_default_text =
            "INTENT=CONTRAST_INCREASE;STRENGTH=SUBTLE\n"
            "INTENT=SATURATION_INCREASE;STRENGTH=SUBTLE\n"
            "INTENT=SHARPEN;STRENGTH=SUBTLE";
        auto result = parse_edit_plan(vague_default_text);
        CHECK(result.valid);
        CHECK(result.plan == default_plan_for_vague_request());
        CHECK(EDIT_INTERPRETATION_SYSTEM_PROMPT.find(vague_default_text) != std::string::npos);
        std::cout << "  the exact text this section's own system prompt commits to for a vague \"make "
                     "it look better\" request parses to exactly the stated conservative default plan "
                     "(" << intent_name(default_plan_for_vague_request().entries[0].intent) << ", " <<
                     intent_name(default_plan_for_vague_request().entries[1].intent) << ", " <<
                     intent_name(default_plan_for_vague_request().entries[2].intent) << ", all SUBTLE) "
                     "-- never a more aggressive, unrequested combination\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
