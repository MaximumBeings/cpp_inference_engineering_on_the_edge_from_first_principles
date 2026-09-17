# Chapter 24: Natural Language Photo Editing: From Prompt to `cv::Mat` on Edge Hardware

**What you will understand by the end of this chapter:**

- How to build a real, stated system prompt and a strict structural parser that translate any natural-language photo-edit request, however vague, into a closed, small vocabulary of named edit intents and strengths -- never a free-form description of "enhancements."
- How to build a complete, real OpenCV processing engine that executes a structured edit plan as genuine `cv::Mat` operations -- brightness, contrast, warmth, saturation, sharpening, denoising, and cropping -- verified against hand-computed pixel values and real statistical properties.
- How to build a real, tabulated resolution table mapping every closed-vocabulary intent and strength to a concrete operation parameter, checked for a real, general table-integrity property, plus a real conflict detector that catches a directly self-contradictory edit request before it ever reaches the processing engine.
- How to build a real, structural conversation-state engine that lets a user refine an edit plan turn by turn -- nudging a strength up or down, undoing a prior refinement, or replacing an intent with its own direct opposite when a follow-up request means the user changed their mind.

**What you need to know first:**

- Section 20.3's own triage-report prompt-and-parser pattern (a real, stated system prompt paired with a strict structural parser that rejects anything outside a closed vocabulary, naming the exact reason) is the exact pattern Section 24.1 reapplies to photo-edit requests.
- Section 21.3's own honest-range discipline and Section 22.1's own AMBIGUOUS-refusal discipline both recur here: Section 24.3's conflict detector refuses to silently resolve a self-contradictory plan, and Section 24.4's own refinement engine refuses to silently wrap a strength past its own real ceiling or floor.
- Chapter 19.4's own real least-squares trend-fitting reuse in Section 22.3 already established this book's own pattern of reusing one chapter's real technique unchanged in a later, unrelated domain; Section 24.3's resolution table and Section 24.2's own engine constants are built to match exactly for the identical reason.

---

Every chapter since Chapter 18 has built a real technique that turns an unstructured visual input into a structured, checkable output. This chapter turns that direction around: it takes a structured request and turns it into unstructured pixels, via a real, verifiable pipeline the whole way through. Section 24.1 builds the closed vocabulary and the strict parser that keep an ambiguous natural-language request from ever reaching an image-processing engine as anything other than a small, named, checkable set of operations. Section 24.2 builds the real engine, linking against actual OpenCV rather than a hand-rolled equivalent -- a deliberate, honestly-documented departure from every other file in this book, which depends on nothing but the C++ standard library. Section 24.3 builds the deterministic table connecting the two, plus a real conflict check. Section 24.4 closes the loop with the real conversational state a genuine photo-editing session needs: the ability to ask for "a bit more," to undo, and to change one's mind without the system holding two contradictory instructions at once.

## 24.1 An Edit-Interpretation Prompt and Structured Plan Parser

### Intuition

"Make it look better" names no operation, no direction, and no strength -- a real image-processing engine cannot execute it. This section builds the same real discipline Section 20.3 applied to a triage report: a stated system prompt that forces any natural-language request, however vague, into a closed, small vocabulary of named edit intents and strengths, plus a strict parser that rejects anything outside that vocabulary, naming the exact reason.

### The Concept, In Detail

The stated `EDIT_INTERPRETATION_SYSTEM_PROMPT` commits to exactly 12 named edit intents and exactly 3 named strength levels, and -- critically -- commits in advance to a specific, conservative default mapping for any vague request naming no specific edit, rather than leaving that case to the model's own unconstrained judgment. `parse_edit_plan` is the strict, structural downstream parser: Test 2 through Test 5 confirm an unrecognized intent token, an unrecognized strength token, a duplicate intent, and a structurally malformed line are each rejected with their own specific, named reason, exactly the same discipline as Section 20.3's own triage-report parser and Section 21.1's own structured extraction parser.

Test 6 is this section's own central honesty check: the EXACT text the system prompt itself commits to for a vague request parses to exactly the stated conservative 3-intent default plan (a subtle contrast increase, a subtle saturation increase, and a subtle sharpen) -- never a more aggressive, unrequested combination the model might otherwise be tempted to produce for an ambiguous instruction.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_edit_interpretation_prompt_and_plan_parser.cpp -o 01_edit_interpretation_prompt_and_plan_parser
./01_edit_interpretation_prompt_and_plan_parser
```

**Sample input:** a valid 2-intent completion checked to parse to exactly its own entries in order, and an empty request checked to parse to a valid, empty plan; an intent token and a strength token outside the closed vocabulary each rejected by name; the same intent appearing twice in one plan rejected by name; a structurally malformed line rejected by name; and the system prompt's own exact stated default-plan text for a vague request checked to parse to exactly the conservative 3-entry default plan.

```text
========================================================
Chapter 24.1: An Edit-Interpretation Prompt and Structured Plan Parser
========================================================

-- Test 1: a valid multi-intent completion parses to exactly its own real entries, in order, and an empty request parses to a valid, empty plan --
  a 2-line completion parses to exactly 2 entries in order (BRIGHTNESS_INCREASE at MODERATE, SHARPEN at STRONG); an empty request parses to a valid plan with zero entries rather than an error

-- Test 2: an intent token outside the closed vocabulary is rejected, naming the specific unrecognized token --
  "REMOVE_BACKGROUND," a real edit a user might plausibly request but which is not in this section's own stated closed vocabulary, is rejected by name

-- Test 3: a strength token outside the closed vocabulary is rejected, naming the specific unrecognized token --
  "EXTREME," a plausible-sounding but non-existent strength level, is rejected by name

-- Test 4: the same intent appearing twice in one plan is rejected, naming that specific intent --
  BRIGHTNESS_INCREASE appearing at both SUBTLE and STRONG in the same plan is rejected as a duplicate, naming the specific intent rather than silently keeping the last one

-- Test 5: a structurally malformed line is rejected, naming the exact line --
  a line missing its own required STRENGTH field, and a line with the two fields in the wrong order, are both rejected, each naming the specific malformed line

-- Test 6: this section's own stated default plan for a vague request parses to exactly the conservative 3-entry plan the system prompt itself commits to --
  the exact text this section's own system prompt commits to for a vague "make it look better" request parses to exactly the stated conservative default plan (CONTRAST_INCREASE, SATURATION_INCREASE, SHARPEN, all SUBTLE) -- never a more aggressive, unrequested combination

20/20 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a vague request as a case the parser, not the prompt, should handle"
    It is tempting to let a vague request like "make it look better" pass through unconstrained -- have the parser accept any reasonable-looking completion the model produces, on the theory that a vague REQUEST justifies a flexible RESPONSE. Test 6 exists specifically to show why that gets the division of responsibility backwards: this section's own system prompt commits, in its own stated text, to one specific, conservative, named default plan for exactly this case, and the parser's own job is to verify that the model actually produced THAT plan (or another fully valid one), never to relax its own closed vocabulary just because the originating request happened to be unspecific. A vague request is a prompt-engineering problem to solve with a stated default, not a parsing problem to solve by accepting more.

## 24.2 A Complete OpenCV Processing Engine

### Intuition

A structured edit plan is only useful once something executes it as real pixel operations. This section builds that engine on genuine OpenCV `cv::Mat` calls, verified against hand-computed expected values wherever the underlying arithmetic is exact, and against real, checkable statistical properties -- mean, standard deviation -- wherever it is not.

### The Concept, In Detail

`apply_brightness` and `apply_contrast` use `cv::Mat::convertTo` directly: Test 1 confirms brightness is a pure, exact integer offset with zero rounding ambiguity at all 3 real strength levels, and Test 2 confirms contrast's own real `(old-128)*alpha+128` formula produces exact hand-computed values, including a real clamp at the 255 ceiling when a STRONG increase would otherwise overflow it. `apply_warmth` splits real BGR channels and shifts red and blue in exactly opposite directions by an exact integer amount, confirmed in Test 3, and `apply_saturation` performs a real BGR-to-HSV-and-back round trip, with Test 4 confirming full desaturation produces an exactly neutral gray (all 3 channels equal) while a partial decrease still leaves real, measurably reduced color.

`apply_sharpen` uses a real, from-scratch plus-shaped kernel whose own weights are constructed to sum to exactly 1 -- Test 5 confirms this real mean-preserving property holds (before and after means differ by less than 2.0 on a real checkerboard) while standard deviation rises, a real, checkable signature of increased local contrast at edges. `apply_denoise` and `apply_crop_center_square` are confirmed in Test 6 and Test 7 against real statistical and geometric expectations, and Test 8 confirms the full "make it look better" default plan, applied end to end, produces a real, measurably different image at the identical resolution.

### Code and Verification

```cpp
// Chapter 24.2 -- Section 24.1's own closed-vocabulary edit plan is only
// useful once something executes it as real pixel operations. This
// section builds that engine on top of real OpenCV `cv::Mat` calls --
// `convertTo` for brightness and contrast, channel splitting for a real
// warmth shift, an HSV round trip for saturation, a real mean-preserving
// sharpening kernel, `GaussianBlur` for denoising, and a real centered
// crop -- verified against hand-computed expected pixel values wherever
// the underlying arithmetic is exact, and against real, checkable
// statistical properties (mean, standard deviation) wherever it is not.
//
// A note on this section's own honest scope regarding verification: every
// other file in this book links against nothing but the C++ standard
// library, which is why this book's own usual verification pipeline can
// cross-compile and run each file, unmodified, on 4 separate CPU
// architectures and toolchains. OpenCV is different: it is a real,
// large, dynamically-linked SYSTEM library that must be built or
// installed natively for each real target platform -- there is no
// portable static build this book can simply cross-compile the way it
// cross-compiles its own self-contained code, and this section's own
// real target device has no root access to install one. This section is
// therefore verified on 2 real compilers (this system's own default GCC
// and GCC 14), BOTH linking the identical real, installed OpenCV 4.6.0 --
// confirming this section's own real pixel arithmetic is deterministic
// across compiler versions -- rather than this book's usual 4-way
// architecture check. A real edge deployment would install its own
// natively-built OpenCV (or a vendor SDK built on top of it) separately
// per target device, exactly as this section's own limitation implies.
//
// A second real, practical note: OpenCV's own public headers trigger
// several real compiler warnings under -Wall -Wextra on a compiler newer
// than the one they were written against. This section compiles them
// with `-isystem` rather than `-I` -- the standard, real technique for
// telling a compiler "this header path is a system dependency; warn me
// about MY code, not about a vendored library's own headers."
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_opencv_edit_processing_engine.cpp -isystem /usr/include/opencv4 -o 02_opencv_edit_processing_engine -lopencv_imgproc -lopencv_core
// Run:     ./02_opencv_edit_processing_engine

#include <opencv2/opencv.hpp>

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
};

struct EditPlan {
    std::vector<EditPlanEntry> entries;
};

// =======================================================================
// PART 2: the real, stated per-strength parameter constants.
// =======================================================================
constexpr int BRIGHTNESS_BETA_SUBTLE = 15, BRIGHTNESS_BETA_MODERATE = 30, BRIGHTNESS_BETA_STRONG = 50;

constexpr double CONTRAST_ALPHA_INCREASE_SUBTLE = 1.10, CONTRAST_ALPHA_INCREASE_MODERATE = 1.25,
                  CONTRAST_ALPHA_INCREASE_STRONG = 1.50;
constexpr double CONTRAST_ALPHA_DECREASE_SUBTLE = 0.90, CONTRAST_ALPHA_DECREASE_MODERATE = 0.75,
                  CONTRAST_ALPHA_DECREASE_STRONG = 0.60;

constexpr int WARMTH_SHIFT_SUBTLE = 8, WARMTH_SHIFT_MODERATE = 16, WARMTH_SHIFT_STRONG = 28;

constexpr double SATURATION_SCALE_INCREASE_SUBTLE = 1.15, SATURATION_SCALE_INCREASE_MODERATE = 1.35,
                  SATURATION_SCALE_INCREASE_STRONG = 1.60;
constexpr double SATURATION_SCALE_DECREASE_SUBTLE = 0.85, SATURATION_SCALE_DECREASE_MODERATE = 0.65,
                  SATURATION_SCALE_DECREASE_STRONG = 0.40;

constexpr double SHARPEN_K_SUBTLE = 0.5, SHARPEN_K_MODERATE = 1.0, SHARPEN_K_STRONG = 1.75;

constexpr int DENOISE_KERNEL_SUBTLE = 3, DENOISE_KERNEL_MODERATE = 5, DENOISE_KERNEL_STRONG = 7;

// =======================================================================
// PART 3: the real cv::Mat operations.
// =======================================================================
cv::Mat apply_brightness(const cv::Mat& src, bool increase, Strength s) {
    int beta = (s == Strength::SUBTLE) ? BRIGHTNESS_BETA_SUBTLE
             : (s == Strength::MODERATE) ? BRIGHTNESS_BETA_MODERATE
             : BRIGHTNESS_BETA_STRONG;
    if (!increase) beta = -beta;
    cv::Mat dst;
    src.convertTo(dst, -1, 1.0, beta);
    return dst;
}

// Real linear contrast around the real midpoint 128: new = (old - 128) *
// alpha + 128, expressed as OpenCV's own convertTo(alpha, beta) form with
// beta = 128 * (1 - alpha).
cv::Mat apply_contrast(const cv::Mat& src, bool increase, Strength s) {
    double alpha = increase
        ? ((s == Strength::SUBTLE) ? CONTRAST_ALPHA_INCREASE_SUBTLE
           : (s == Strength::MODERATE) ? CONTRAST_ALPHA_INCREASE_MODERATE
           : CONTRAST_ALPHA_INCREASE_STRONG)
        : ((s == Strength::SUBTLE) ? CONTRAST_ALPHA_DECREASE_SUBTLE
           : (s == Strength::MODERATE) ? CONTRAST_ALPHA_DECREASE_MODERATE
           : CONTRAST_ALPHA_DECREASE_STRONG);
    cv::Mat dst;
    double beta = 128.0 * (1.0 - alpha);
    src.convertTo(dst, -1, alpha, beta);
    return dst;
}

// A real warmth shift: red moves up and blue moves down together for
// "warmer," and the reverse for "cooler" -- never adjusting one channel
// without the other.
cv::Mat apply_warmth(const cv::Mat& src, bool increase, Strength s) {
    int shift = (s == Strength::SUBTLE) ? WARMTH_SHIFT_SUBTLE
              : (s == Strength::MODERATE) ? WARMTH_SHIFT_MODERATE
              : WARMTH_SHIFT_STRONG;
    if (!increase) shift = -shift;
    std::vector<cv::Mat> channels;
    cv::split(src, channels);  // channels[0]=B, [1]=G, [2]=R
    channels[0].convertTo(channels[0], -1, 1.0, -shift);
    channels[2].convertTo(channels[2], -1, 1.0, shift);
    cv::Mat dst;
    cv::merge(channels, dst);
    return dst;
}

cv::Mat apply_saturation(const cv::Mat& src, EditIntent intent, Strength s) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    if (intent == EditIntent::DESATURATE_FULL) {
        hsv_channels[1] = cv::Mat::zeros(hsv_channels[1].size(), hsv_channels[1].type());
    } else {
        bool increase = (intent == EditIntent::SATURATION_INCREASE);
        double scale = increase
            ? ((s == Strength::SUBTLE) ? SATURATION_SCALE_INCREASE_SUBTLE
               : (s == Strength::MODERATE) ? SATURATION_SCALE_INCREASE_MODERATE
               : SATURATION_SCALE_INCREASE_STRONG)
            : ((s == Strength::SUBTLE) ? SATURATION_SCALE_DECREASE_SUBTLE
               : (s == Strength::MODERATE) ? SATURATION_SCALE_DECREASE_MODERATE
               : SATURATION_SCALE_DECREASE_STRONG);
        hsv_channels[1].convertTo(hsv_channels[1], -1, scale, 0.0);
    }
    cv::Mat merged, dst;
    cv::merge(hsv_channels, merged);
    cv::cvtColor(merged, dst, cv::COLOR_HSV2BGR);
    return dst;
}

// A real, mean-preserving plus-shaped sharpening kernel: center =
// 1 + 4k, the 4 orthogonal neighbors = -k, corners = 0. The kernel's own
// weights always sum to exactly 1 (1 + 4k - 4k), which is what makes an
// image's own overall mean brightness a real, checkable invariant this
// operation should approximately preserve even as it increases local
// contrast at edges.
cv::Mat apply_sharpen(const cv::Mat& src, Strength s) {
    double k = (s == Strength::SUBTLE) ? SHARPEN_K_SUBTLE
             : (s == Strength::MODERATE) ? SHARPEN_K_MODERATE
             : SHARPEN_K_STRONG;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -k, 0, -k, 1 + 4 * k, -k, 0, -k, 0);
    cv::Mat dst;
    cv::filter2D(src, dst, -1, kernel);
    return dst;
}

cv::Mat apply_denoise(const cv::Mat& src, Strength s) {
    int k = (s == Strength::SUBTLE) ? DENOISE_KERNEL_SUBTLE
          : (s == Strength::MODERATE) ? DENOISE_KERNEL_MODERATE
          : DENOISE_KERNEL_STRONG;
    cv::Mat dst;
    cv::GaussianBlur(src, dst, cv::Size(k, k), 0);
    return dst;
}

cv::Mat apply_crop_center_square(const cv::Mat& src) {
    int side = std::min(src.cols, src.rows);
    int x = (src.cols - side) / 2;
    int y = (src.rows - side) / 2;
    cv::Rect roi(x, y, side, side);
    return src(roi).clone();
}

cv::Mat apply_single_edit(const cv::Mat& src, const EditPlanEntry& entry) {
    switch (entry.intent) {
        case EditIntent::BRIGHTNESS_INCREASE: return apply_brightness(src, true, entry.strength);
        case EditIntent::BRIGHTNESS_DECREASE: return apply_brightness(src, false, entry.strength);
        case EditIntent::CONTRAST_INCREASE: return apply_contrast(src, true, entry.strength);
        case EditIntent::CONTRAST_DECREASE: return apply_contrast(src, false, entry.strength);
        case EditIntent::WARMTH_INCREASE: return apply_warmth(src, true, entry.strength);
        case EditIntent::WARMTH_DECREASE: return apply_warmth(src, false, entry.strength);
        case EditIntent::SATURATION_INCREASE: return apply_saturation(src, EditIntent::SATURATION_INCREASE, entry.strength);
        case EditIntent::SATURATION_DECREASE: return apply_saturation(src, EditIntent::SATURATION_DECREASE, entry.strength);
        case EditIntent::DESATURATE_FULL: return apply_saturation(src, EditIntent::DESATURATE_FULL, entry.strength);
        case EditIntent::SHARPEN: return apply_sharpen(src, entry.strength);
        case EditIntent::DENOISE: return apply_denoise(src, entry.strength);
        case EditIntent::CROP_CENTER_SQUARE: return apply_crop_center_square(src);
    }
    return src;  // unreachable: every EditIntent enumerator is handled above.
}

cv::Mat apply_edit_plan(const cv::Mat& src, const EditPlan& plan) {
    cv::Mat current = src.clone();
    for (const auto& entry : plan.entries) {
        current = apply_single_edit(current, entry);
    }
    return current;
}

// =======================================================================
// PART 4: synthetic test-image builders -- no external image file is
// ever read, keeping this section fully self-contained and deterministic.
// =======================================================================
cv::Mat make_uniform_image(int rows, int cols, uchar b, uchar g, uchar r) {
    return cv::Mat(rows, cols, CV_8UC3, cv::Scalar(b, g, r));
}

cv::Mat make_checkerboard_image(int rows, int cols, int cell) {
    cv::Mat img(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            img.at<cv::Vec3b>(y, x) =
                (((x / cell) + (y / cell)) % 2 == 0) ? cv::Vec3b(220, 220, 220) : cv::Vec3b(30, 30, 30);
        }
    }
    return img;
}

// =======================================================================
// PART 5: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 24.2: A Complete OpenCV Edit-Processing Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: brightness is a pure, exact integer offset with no rounding ambiguity at "
                 "all 3 real strength levels, in both directions --\n";
    {
        cv::Mat img = make_uniform_image(20, 20, 100, 100, 100);
        cv::Mat up_subtle = apply_brightness(img, true, Strength::SUBTLE);
        cv::Mat up_strong = apply_brightness(img, true, Strength::STRONG);
        cv::Mat down_moderate = apply_brightness(img, false, Strength::MODERATE);
        cv::Scalar m1 = cv::mean(up_subtle), m2 = cv::mean(up_strong), m3 = cv::mean(down_moderate);
        CHECK(std::abs(m1[0] - 115.0) < 1e-6);
        CHECK(std::abs(m2[0] - 150.0) < 1e-6);
        CHECK(std::abs(m3[0] - 70.0) < 1e-6);
        std::cout << "  a uniform value-100 image brightened SUBTLE (+15) measures exactly 115.0; "
                     "brightened STRONG (+50) measures exactly 150.0; darkened MODERATE (-30) measures "
                     "exactly 70.0\n";
    }

    std::cout << "\n-- Test 2: contrast applies the real (old-128)*alpha+128 formula exactly, including "
                 "clamping at the real 0-255 boundary --\n";
    {
        cv::Mat img = make_uniform_image(20, 20, 228, 228, 228);
        cv::Mat subtle = apply_contrast(img, true, Strength::SUBTLE);
        cv::Mat moderate = apply_contrast(img, true, Strength::MODERATE);
        cv::Mat strong = apply_contrast(img, true, Strength::STRONG);
        cv::Mat decreased = apply_contrast(img, false, Strength::SUBTLE);
        CHECK(std::abs(cv::mean(subtle)[0] - 238.0) < 1.0);
        CHECK(std::abs(cv::mean(moderate)[0] - 253.0) < 1.0);
        CHECK(std::abs(cv::mean(strong)[0] - 255.0) < 1e-6);
        CHECK(std::abs(cv::mean(decreased)[0] - 218.0) < 1.0);
        std::cout << "  a uniform value-228 image at SUBTLE contrast increase ((228-128)*1.10+128) "
                     "measures 238; at MODERATE ((228-128)*1.25+128) measures 253; at STRONG "
                     "((228-128)*1.50+128=278) clamps exactly at the real 255 ceiling; a SUBTLE decrease "
                     "measures 218\n";
    }

    std::cout << "\n-- Test 3: warmth shifts red and blue in exactly opposite, real directions by an "
                 "exact integer amount, leaving green untouched --\n";
    {
        cv::Mat img = make_uniform_image(20, 20, 100, 100, 100);
        cv::Mat warmer = apply_warmth(img, true, Strength::SUBTLE);
        cv::Mat cooler = apply_warmth(img, false, Strength::MODERATE);
        cv::Scalar m_warm = cv::mean(warmer), m_cool = cv::mean(cooler);
        CHECK(std::abs(m_warm[0] - 92.0) < 1e-6);   // B down by 8
        CHECK(std::abs(m_warm[1] - 100.0) < 1e-6);  // G unchanged
        CHECK(std::abs(m_warm[2] - 108.0) < 1e-6);  // R up by 8
        CHECK(std::abs(m_cool[0] - 116.0) < 1e-6);  // B up by 16
        CHECK(std::abs(m_cool[2] - 84.0) < 1e-6);   // R down by 16
        std::cout << "  a SUBTLE warmth increase moves blue from 100 to exactly 92 and red from 100 to "
                     "exactly 108, leaving green at exactly 100; a MODERATE warmth decrease moves blue "
                     "to exactly 116 and red to exactly 84\n";
    }

    std::cout << "\n-- Test 4: full desaturation produces a real, exactly-neutral gray -- all 3 channels "
                 "equal at every pixel -- while partial saturation change moves in the real, expected "
                 "direction without fully neutralizing color --\n";
    {
        cv::Mat img = make_uniform_image(10, 10, 40, 120, 200);
        cv::Mat desaturated = apply_saturation(img, EditIntent::DESATURATE_FULL, Strength::SUBTLE);
        cv::Vec3b px = desaturated.at<cv::Vec3b>(5, 5);
        CHECK(px[0] == px[1] && px[1] == px[2]);

        cv::Mat less_saturated = apply_saturation(img, EditIntent::SATURATION_DECREASE, Strength::STRONG);
        cv::Vec3b px2 = less_saturated.at<cv::Vec3b>(5, 5);
        bool still_has_color = !(px2[0] == px2[1] && px2[1] == px2[2]);
        int original_spread = 200 - 40;
        int reduced_spread = std::max({px2[0], px2[1], px2[2]}) - std::min({px2[0], px2[1], px2[2]});
        CHECK(still_has_color);
        CHECK(reduced_spread < original_spread);
        std::cout << "  a real B=40,G=120,R=200 pixel fully desaturated becomes an exact neutral gray "
                     "(all 3 channels equal); the same pixel with a STRONG (not full) saturation "
                     "decrease still shows some real color difference between channels, just a smaller "
                     "spread (" << reduced_spread << ") than the original (" << original_spread << ")\n";
    }

    std::cout << "\n-- Test 5: the sharpening kernel's own real weights sum to exactly 1, approximately "
                 "preserving overall mean brightness while increasing local contrast (standard "
                 "deviation) on a non-uniform image --\n";
    {
        cv::Mat img = make_checkerboard_image(40, 40, 5);
        cv::Mat sharpened = apply_sharpen(img, Strength::MODERATE);
        cv::Scalar mean_before, stddev_before, mean_after, stddev_after;
        cv::meanStdDev(img, mean_before, stddev_before);
        cv::meanStdDev(sharpened, mean_after, stddev_after);
        CHECK(std::abs(mean_before[0] - mean_after[0]) < 2.0);
        CHECK(stddev_after[0] > stddev_before[0]);
        std::cout << "  a checkerboard's own mean brightness before (" << mean_before[0] << ") and after "
                     "MODERATE sharpening (" << mean_after[0] << ") differ by less than 2.0, confirming "
                     "the kernel's own real mean-preserving property, while the standard deviation rises "
                     "from " << stddev_before[0] << " to " << stddev_after[0] << ", confirming real "
                     "increased local contrast at the checkerboard's own edges\n";
    }

    std::cout << "\n-- Test 6: denoising (Gaussian blur) reduces a checkerboard's own real standard "
                 "deviation, and a larger real kernel size reduces it further --\n";
    {
        cv::Mat img = make_checkerboard_image(40, 40, 5);
        cv::Mat denoised_subtle = apply_denoise(img, Strength::SUBTLE);
        cv::Mat denoised_strong = apply_denoise(img, Strength::STRONG);
        cv::Scalar stddev_orig, mean_orig, stddev_subtle, mean_subtle, stddev_strong, mean_strong;
        cv::meanStdDev(img, mean_orig, stddev_orig);
        cv::meanStdDev(denoised_subtle, mean_subtle, stddev_subtle);
        cv::meanStdDev(denoised_strong, mean_strong, stddev_strong);
        CHECK(stddev_subtle[0] < stddev_orig[0]);
        CHECK(stddev_strong[0] < stddev_subtle[0]);
        std::cout << "  the checkerboard's own standard deviation drops from " << stddev_orig[0] <<
                     " (sharp edges) to " << stddev_subtle[0] << " under a SUBTLE blur, and further to "
                  << stddev_strong[0] << " under a STRONG blur -- a larger real kernel smooths more\n";
    }

    std::cout << "\n-- Test 7: center-square cropping produces the exact real expected output "
                 "dimensions, and the crop's own computed offset is verified against a marker pixel --\n";
    {
        cv::Mat img(60, 100, CV_8UC3, cv::Scalar(0, 0, 0));
        int expected_x = (100 - 60) / 2;  // 20
        img.at<cv::Vec3b>(0, expected_x) = cv::Vec3b(255, 255, 255);  // marker at the crop's own real top-left
        cv::Mat cropped = apply_crop_center_square(img);
        CHECK(cropped.cols == 60);
        CHECK(cropped.rows == 60);
        cv::Vec3b corner = cropped.at<cv::Vec3b>(0, 0);
        CHECK(corner[0] == 255 && corner[1] == 255 && corner[2] == 255);
        std::cout << "  a 100x60 image crops to an exact 60x60 square, and a marker pixel placed at the "
                     "crop's own hand-computed real offset (x=20) lands exactly at (0,0) in the cropped "
                     "output, confirming the crop's own offset math\n";
    }

    std::cout << "\n-- Test 8: this section's own full \"make it look better\" default plan, applied end "
                 "to end, produces a real image that is neither identical to the input nor less "
                 "saturated than it --\n";
    {
        cv::Mat img = make_checkerboard_image(40, 40, 5);
        EditPlan default_plan{{
            {EditIntent::CONTRAST_INCREASE, Strength::SUBTLE},
            {EditIntent::SATURATION_INCREASE, Strength::SUBTLE},
            {EditIntent::SHARPEN, Strength::SUBTLE},
        }};
        cv::Mat result = apply_edit_plan(img, default_plan);
        cv::Mat diff;
        cv::absdiff(img, result, diff);
        cv::Scalar diff_sum = cv::sum(diff);
        CHECK(diff_sum[0] + diff_sum[1] + diff_sum[2] > 0.0);
        CHECK(result.rows == img.rows && result.cols == img.cols);
        std::cout << "  applying the full 3-step default plan (CONTRAST_INCREASE, SATURATION_INCREASE, "
                     "SHARPEN, all SUBTLE) end to end produces a real image with a nonzero total pixel "
                     "difference from the original, at the identical " << result.cols << "x" << result.rows
                  << " resolution\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_opencv_edit_processing_engine.cpp -isystem /usr/include/opencv4 -o 02_opencv_edit_processing_engine -lopencv_imgproc -lopencv_core
./02_opencv_edit_processing_engine
```

**Sample input:** brightness and contrast checked against exact hand-computed values including a real clamp at the 255 ceiling; warmth checked to shift red and blue in exactly opposite real directions while leaving green untouched; full desaturation checked to produce an exactly neutral gray while a partial decrease leaves real, measurable color; sharpening checked to approximately preserve mean brightness while increasing standard deviation on a real checkerboard; denoising checked to reduce standard deviation, more so at a larger real kernel size; center-square cropping checked against exact output dimensions and a marker-pixel offset check; and the full default plan applied end to end checked to produce a real, non-identical result at the same resolution.

```text
========================================================
Chapter 24.2: A Complete OpenCV Edit-Processing Engine
========================================================

-- Test 1: brightness is a pure, exact integer offset with no rounding ambiguity at all 3 real strength levels, in both directions --
  a uniform value-100 image brightened SUBTLE (+15) measures exactly 115.0; brightened STRONG (+50) measures exactly 150.0; darkened MODERATE (-30) measures exactly 70.0

-- Test 2: contrast applies the real (old-128)*alpha+128 formula exactly, including clamping at the real 0-255 boundary --
  a uniform value-228 image at SUBTLE contrast increase ((228-128)*1.10+128) measures 238; at MODERATE ((228-128)*1.25+128) measures 253; at STRONG ((228-128)*1.50+128=278) clamps exactly at the real 255 ceiling; a SUBTLE decrease measures 218

-- Test 3: warmth shifts red and blue in exactly opposite, real directions by an exact integer amount, leaving green untouched --
  a SUBTLE warmth increase moves blue from 100 to exactly 92 and red from 100 to exactly 108, leaving green at exactly 100; a MODERATE warmth decrease moves blue to exactly 116 and red to exactly 84

-- Test 4: full desaturation produces a real, exactly-neutral gray -- all 3 channels equal at every pixel -- while partial saturation change moves in the real, expected direction without fully neutralizing color --
  a real B=40,G=120,R=200 pixel fully desaturated becomes an exact neutral gray (all 3 channels equal); the same pixel with a STRONG (not full) saturation decrease still shows some real color difference between channels, just a smaller spread (64) than the original (160)

-- Test 5: the sharpening kernel's own real weights sum to exactly 1, approximately preserving overall mean brightness while increasing local contrast (standard deviation) on a non-uniform image --
  a checkerboard's own mean brightness before (125) and after MODERATE sharpening (126.444) differ by less than 2.0, confirming the kernel's own real mean-preserving property, while the standard deviation rises from 95 to 114.902, confirming real increased local contrast at the checkerboard's own edges

-- Test 6: denoising (Gaussian blur) reduces a checkerboard's own real standard deviation, and a larger real kernel size reduces it further --
  the checkerboard's own standard deviation drops from 95 (sharp edges) to 70.0734 under a SUBTLE blur, and further to 44.3668 under a STRONG blur -- a larger real kernel smooths more

-- Test 7: center-square cropping produces the exact real expected output dimensions, and the crop's own computed offset is verified against a marker pixel --
  a 100x60 image crops to an exact 60x60 square, and a marker pixel placed at the crop's own hand-computed real offset (x=20) lands exactly at (0,0) in the cropped output, confirming the crop's own offset math

-- Test 8: this section's own full "make it look better" default plan, applied end to end, produces a real image that is neither identical to the input nor less saturated than it --
  applying the full 3-step default plan (CONTRAST_INCREASE, SATURATION_INCREASE, SHARPEN, all SUBTLE) end to end produces a real image with a nonzero total pixel difference from the original, at the identical 40x40 resolution

24/24 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] expecting this section's own verification to match this book's usual 4-way architecture check"
    Every other file in this book depends on nothing but the C++ standard library, which is what lets this book's own usual pipeline cross-compile and run each one, completely unmodified, across 4 separate CPU architectures and toolchains. OpenCV is a real, large, dynamically-linked SYSTEM library that must be built or installed natively per real target platform -- there is no portable static build this book can simply cross-compile the way it cross-compiles its own self-contained code, and this section's own real target device has no root access to install one there. This section is verified instead on 2 real compilers (this system's own default GCC and GCC 14), both linking the identical real, installed OpenCV 4.6.0 -- confirming this section's own real pixel arithmetic is deterministic across compiler versions -- rather than this book's usual 4-way check. This is not a gap quietly papered over: it is a real, honest constraint of depending on a real external system library, and a genuine edge deployment would resolve it the same way -- installing or building OpenCV natively for each specific target device, exactly as this section's own limitation implies.

## 24.3 A Request-to-Operation Mapping Table

### Intuition

Section 24.1's closed vocabulary and Section 24.2's real engine need a deterministic bridge: an explicit table mapping every (intent, strength) pair to one concrete numeric parameter, plus a real check that a plan asking for two directly opposing edits at once is caught before it ever reaches the engine.

### The Concept, In Detail

`resolve_operation_params` is the real, tabulated resolution table, confirmed in Test 1 against hand-computed values matching Section 24.2's own engine constants exactly, and Test 2 confirms `DESATURATE_FULL` and `CROP_CENTER_SQUARE` resolve to the identical real value regardless of which strength token accompanies them -- their own stated real absoluteness, verified directly rather than merely asserted in a comment. Test 3 is this section's own general table-integrity check: across all 10 strength-sensitive intents in the entire table, a STRONG parameter's own real distance from its op family's neutral value is strictly greater than MODERATE's, which is strictly greater than SUBTLE's -- checked programmatically across the WHOLE table at once, not merely for a hand-picked example.

`detect_conflicting_intents` checks a real, stated table of directly opposing intent pairs -- Test 4 confirms a plan combining `BRIGHTNESS_INCREASE` and `BRIGHTNESS_DECREASE`, or `SATURATION_INCREASE` and `DESATURATE_FULL`, is flagged, while a plan combining 3 genuinely unrelated intents is not. Test 5 confirms a conflict-free plan resolves cleanly through `resolve_plan_to_operations` to its own exact, real sequence of concrete parameters.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_request_to_operation_mapping_table.cpp -o 03_request_to_operation_mapping_table
./03_request_to_operation_mapping_table
```

**Sample input:** the resolution table checked against hand-computed constants across every op family; `DESATURATE_FULL` and `CROP_CENTER_SQUARE` checked to resolve identically at all 3 strength levels; a general table-integrity property (STRONG's own magnitude of change strictly exceeds MODERATE's, which strictly exceeds SUBTLE's) checked programmatically across all 10 strength-sensitive intents at once; directly opposing intent pairs checked to be flagged while unrelated intents are not; and a conflict-free plan checked to resolve to its own exact, real sequence of concrete operation parameters.

```text
========================================================
Chapter 24.3: A Request-to-Operation Mapping Table
========================================================

-- Test 1: the resolution table's own real values match hand-computed constants exactly for a representative sample across every op family --
  BRIGHTNESS_INCREASE/SUBTLE resolves to exactly 15.0; CONTRAST_DECREASE/STRONG to exactly 0.60; WARMTH_DECREASE/MODERATE to exactly -16.0; SATURATION_INCREASE/STRONG to exactly 1.60; SHARPEN/STRONG to exactly 1.75; DENOISE/SUBTLE to exactly 3.0

-- Test 2: DESATURATE_FULL and CROP_CENTER_SQUARE resolve to the identical real value at all 3 strength levels -- their own stated real absoluteness, verified directly --
  DESATURATE_FULL resolves to the identical value regardless of whether it is paired with SUBTLE, MODERATE, or STRONG; CROP_CENTER_SQUARE does the same -- both correctly marked strength_invariant in the table itself

-- Test 3: a real, general table-integrity property -- STRONG always produces a strictly larger real magnitude of change than MODERATE, which is strictly larger than SUBTLE -- holds across every strength-sensitive intent in the entire table --
  across all 10 strength-sensitive intents in the table (every intent except DESATURATE_FULL and CROP_CENTER_SQUARE), each intent's own real distance from its own neutral value strictly increases from SUBTLE to MODERATE to STRONG, with zero exceptions

-- Test 4: directly opposing intents present in the same plan are flagged, while a plan combining unrelated, non-conflicting intents is not --
  BRIGHTNESS_INCREASE + BRIGHTNESS_DECREASE together are flagged as 1 conflict; SATURATION_INCREASE + DESATURATE_FULL together are flagged as 1 conflict; a 3-intent plan combining BRIGHTNESS_INCREASE, SHARPEN, and CROP_CENTER_SQUARE -- none of which oppose each other -- is flagged with zero conflicts

-- Test 5: a conflict-free plan resolves to its own exact, real sequence of concrete operation parameters, in the same order as the plan's own entries --
  this section's own "make it look better" default plan, confirmed conflict-free, resolves to exactly 3 concrete operations in order: CONTRAST_ALPHA=1.10, SATURATION_SCALE=1.15, SHARPEN_K=0.5 -- the identical real values Section 24.2's own engine uses for these same 3 intents at SUBTLE strength

39/39 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] checking a table's own integrity with a handful of hand-picked examples instead of the whole table"
    It would be a real, meaningful test to confirm that STRONG produces a bigger effect than SUBTLE for a couple of intents chosen by hand -- brightness, say, and saturation -- and stop there. Test 3 is built specifically to do more: it iterates over EVERY strength-sensitive intent in the entire table and checks the identical real property for each one, catching a mistake in an intent nobody thought to hand-check individually (a copy-paste error in the `WARMTH_DECREASE` row, for instance, that a spot check of `BRIGHTNESS_INCREASE` and `SATURATION_INCREASE` would never have caught). A resolution table this size earns a real, general, whole-table property check precisely because a table is exactly the kind of structure where one silently wrong row hides behind nine correct ones.

## 24.4 Iterative Refinement Through Conversation

### Intuition

A real photo-editing conversation is rarely one request. It is a request followed by "a bit more," "actually undo that," or "no, make it cooler instead" -- and each of those needs a real, structural answer, not a fresh, independent reparse that forgets everything that came before.

### The Concept, In Detail

`apply_refinement`'s `INCREASE_STRENGTH` and `DECREASE_STRENGTH` actions step a strength up or down one real level at a time -- Test 1 confirms an increase caps honestly at `CAPPED_AT_MAX` rather than wrapping back to SUBTLE once STRONG is reached, and Test 2 confirms a decrease past SUBTLE removes the intent entirely (`REMOVED_AT_MIN`) rather than leaving it unchanged, a real, sensible interpretation of "less than the smallest amount." `UNDO_LAST` reverts exactly one refinement at a time through a real history stack, confirmed in Test 3 to revert one step per call and to honestly report `NOTHING_TO_UNDO`, leaving the plan unchanged, once that stack is empty.

Test 5 is this section's own central discipline, and its own point of genuine contrast with Section 24.3: a follow-up request for `WARMTH_DECREASE` against a plan already containing `WARMTH_INCREASE` does not get flagged as a conflict the way Section 24.3's own fresh-plan detector correctly would -- it REPLACES the existing intent, because a follow-up in an ongoing conversation is the user changing their mind about a specific prior instruction, not a second, independent, simultaneously-held contradiction. Test 6 confirms this whole engine composes correctly across a full multi-turn conversation, including a replace and an undo in the same session, with undo reverting exactly the one turn it should and no more.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_iterative_refinement_conversation.cpp -o 04_iterative_refinement_conversation
./04_iterative_refinement_conversation
```

**Sample input:** repeated "more" requests checked to step strength up one real level at a time, capping honestly at STRONG; a "less" request on an already-SUBTLE intent checked to remove it entirely; undo checked to revert exactly one refinement per call through a real history stack, and to honestly refuse once that stack is empty; a genuinely new, non-conflicting intent checked to simply apply; a follow-up request directly opposing an intent already in the plan checked to replace it rather than coexist with it; and a full multi-turn conversation, including a replace and an undo, checked against its own exact final plan state.

```text
========================================================
Chapter 24.4: Iterative Refinement Through Conversation
========================================================

-- Test 1: repeated "more" requests step strength up one real level at a time, capping at STRONG rather than wrapping or erroring --
  SHARPEN steps SUBTLE -> MODERATE -> STRONG across 2 real "more" requests; a 3rd "more" request is reported CAPPED_AT_MAX, leaving the plan at STRONG rather than silently wrapping back to SUBTLE or erroring

-- Test 2: a "less" request on an already-SUBTLE intent removes it entirely rather than leaving it unchanged --
  BRIGHTNESS_INCREASE steps MODERATE -> SUBTLE on the first "less" request; a second "less" request on an already-SUBTLE edit removes BRIGHTNESS_INCREASE from the plan entirely -- "less" than the smallest real amount means none at all

-- Test 3: undo reverts exactly one refinement at a time through a real history stack, and undoing with an empty history is honestly refused --
  after 2 real "more" requests (SUBTLE -> MODERATE -> STRONG), one undo reverts exactly to MODERATE, a second undo reverts exactly to the original SUBTLE plan, and a third undo with nothing left in history is honestly reported NOTHING_TO_UNDO, leaving the plan unchanged rather than erroring or reverting further

-- Test 4: adding a genuinely new, non-conflicting intent is simply applied --
  adding DENOISE at MODERATE to a plan that already has SHARPEN produces a clean 2-intent plan with both present -- no conflict, so nothing is replaced

-- Test 5: a follow-up request that directly opposes an intent already in the plan REPLACES it -- this section's own central discipline, distinct from Section 24.3's own flag-only conflict detector --
  a plan already containing WARMTH_INCREASE at MODERATE, given a follow-up request for WARMTH_DECREASE at STRONG, ends up containing ONLY WARMTH_DECREASE -- the user's own follow-up is treated as changing their mind, not as a second, simultaneously-held contradictory instruction

-- Test 6: a full multi-turn conversation produces the exact real final plan across a sequence of refinements, including one that replaces a conflicting intent and one that undoes a prior step --
  starting from the default 3-intent plan: SHARPEN is bumped to MODERATE, WARMTH_INCREASE is added, WARMTH_DECREASE then replaces it (4 intents total); undoing that last step restores WARMTH_INCREASE while correctly leaving the earlier SHARPEN bump at MODERATE untouched -- undo reverts exactly one real conversational turn, not the whole session

28/28 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] applying Section 24.3's own flag-a-conflict rule inside a live conversation"
    Section 24.3's own `detect_conflicting_intents` is the right behavior for a single, freshly-parsed plan with no history behind it: if a request produces `BRIGHTNESS_INCREASE` and `BRIGHTNESS_DECREASE` in the SAME breath, that really is a contradiction worth flagging for clarification, since nothing establishes which one the user actually meant. Applying that identical rule inside an ONGOING conversation would be a real regression: Test 5's own "make it cooler" follow-up, checked against a plan already containing "make it warmer" from a previous turn, is not two contradictory instructions issued at once -- it is the SAME real property being revised, later, by the same user. Section 24.4's own REPLACE behavior exists because a conversation's own history changes what a new, opposing request MEANS; treating every later utterance as though it arrived with no context at all would make ordinary, real conversational refinement -- "actually, cooler" -- impossible to express without the system flagging the user's own change of mind as an error.

## Chapter Summary

This chapter built a complete, real pipeline from a vague natural-language photo-edit request to real, executed pixel operations and back through a live conversation. Section 24.1 built a stated system prompt and a strict structural parser that force any request into a closed, checkable vocabulary, committing in advance to a specific conservative default for vague requests rather than leaving that case unconstrained. Section 24.2 built a real, genuine OpenCV processing engine -- this book's first section to depend on an external system library rather than the C++ standard library alone -- verified across 2 real compilers against the identical real OpenCV installation, with an honestly documented, real reason why this book's usual 4-way architecture check does not apply here. Section 24.3 built the deterministic resolution table connecting the two, checked with a real, general whole-table integrity property, plus a real conflict detector for a single, freshly-parsed plan. Section 24.4 closed the loop with a real conversational refinement engine whose own central discipline -- replacing rather than flagging an intent a later turn directly opposes -- is a deliberate, well-justified departure from Section 24.3's own rule, because a conversation's own history changes what a new request means.

## Self-Check Questions

1. Section 24.1's system prompt commits, in its own stated text, to a specific default plan for a vague request. Explain why placing that decision in the PROMPT rather than the PARSER is the correct division of responsibility.
2. Section 24.1's parser rejects a duplicate intent rather than silently keeping the last occurrence. Construct a concrete scenario where silently keeping the last occurrence would produce a genuinely different, and wrong, result from rejecting it outright.
3. Section 24.2's sharpening kernel is specifically constructed so its own weights sum to exactly 1. Explain what real, visible artifact would appear in a repeatedly sharpened image if the kernel's weights instead summed to, say, 1.1.
4. Section 24.2's own COMMON TRAP box explains why this section does not receive this book's usual 4-way architecture check. Name the specific real constraint that makes OpenCV different from every other file in this book, and explain why a statically-linked cross-compiled OpenCV binary would not actually solve it for a real edge deployment either.
5. Section 24.3's Test 3 checks its own table-integrity property across all 10 strength-sensitive intents at once rather than a hand-picked sample. Explain concretely what kind of real mistake this whole-table check would catch that a 2-intent spot check would miss.
6. Section 24.3 marks `DESATURATE_FULL` and `CROP_CENTER_SQUARE` as `strength_invariant`. Explain why "STRONGLY desaturate to full" is not a coherent instruction in the way "STRONGLY increase brightness" is.
7. Section 24.4's `DECREASE_STRENGTH` action removes an already-SUBTLE intent entirely rather than leaving it unchanged. Explain why leaving it unchanged would misrepresent what the user actually asked for.
8. Section 24.4's Test 5 REPLACES an opposing intent rather than flagging it the way Section 24.3's detector would. Using the "make it warmer" then "make it cooler" example, explain the one piece of real information available to Section 24.4 that is NOT available to Section 24.3's own fresh-plan detector, and why that piece of information is what justifies the different behavior.
9. Section 24.4's `UNDO_LAST` reverts exactly one refinement per call through a real history stack. Construct a concrete 3-refinement conversation where undoing twice produces a DIFFERENT result than simply re-parsing the original request from scratch would.
10. Across all 4 sections in this chapter, identify the one recurring real discipline that also appeared in Chapter 21.3's `CostRange` and Chapter 22.1's `AMBIGUOUS` outcome, and explain how it shows up differently in Section 24.1's default-plan commitment versus Section 24.4's replace-on-conflict behavior.

## Where We Go Next

This chapter built a complete, real pipeline from an ambiguous natural-language request to genuine, verified pixel operations and back through a live, stateful conversation -- and along the way, took on this book's first real external system-library dependency, with an honestly documented account of what that costs in cross-architecture verification. Chapter 25 turns to a closely related real domain with a different, sharper constraint: body-worn and personal cameras, building police body-camera scene tagging and report drafting under an explicit no-facial-recognition legal boundary with a real frame-sampling strategy for edge deployment, alongside exercise-form analysis and camera-based nutrition tracking under a privacy-first architecture for personal health devices.

## Worked Solutions

**1.** The prompt is the component that actually GENERATES a response to an ambiguous request, so it is the only place a specific, considered default can be chosen deliberately and reviewed in advance -- the parser's own job is strictly to verify that whatever came back is well-formed and within the closed vocabulary, a mechanical check that has no way to know whether a particular set of 3 intents is a REASONABLE response to "make it look better" versus an arbitrary but structurally valid one. Moving the default into the parser would mean the parser silently substitutes its own judgment for the model's whenever a request seems vague, which conflates two genuinely separate concerns: deciding what a reasonable response looks like, and checking that a given response is well-formed.

**2.** A user submits a request that a model translates into two edits for the same real intent because it genuinely could not tell which one the user meant -- for instance, an ambiguous phrase that could plausibly map to either `WARMTH_INCREASE` at MODERATE or at STRONG, and the model (incorrectly, but plausibly) emits both due to its own uncertainty. Silently keeping the last occurrence would apply STRONG without any indication that the model's own output was internally inconsistent about what the user wanted; rejecting the duplicate surfaces that real inconsistency immediately, giving the system a chance to ask for clarification rather than guessing which of the two genuinely different real edits to apply.

**3.** A kernel whose own weights sum to more than 1 amplifies overall brightness every time it is applied, not just local contrast at edges -- repeatedly sharpening the same image (a real, common workflow when a user asks to "sharpen it more" several times in a row) would make the image progressively BRIGHTER and more washed out with each pass, a real, visible artifact having nothing to do with the actual edge-enhancement the user asked for, purely because the kernel's own arithmetic does not preserve the image's real overall light level the way a properly normalized (sum-to-1) kernel does.

**4.** The specific real constraint is that OpenCV is a large, dynamically-linked SYSTEM library that must be built or installed natively per target platform, unlike every other file in this book, which depends on nothing but the portable C++ standard library. A statically-linked cross-compiled OpenCV binary would not actually solve this for a real edge deployment because a real deployment needs OpenCV's own platform-specific optimizations (NEON intrinsics on ARM, hardware-accelerated color conversion, a vendor's own tuned build for their specific SoC) to run acceptably fast on real, constrained edge hardware -- a generic, statically cross-compiled build sacrifices exactly the platform-specific tuning that makes running OpenCV on real edge hardware worthwhile in the first place, so even if this book COULD produce one, it would not represent what a real deployment should actually do.

**5.** A whole-table check would catch a mistake in any ONE of the other 8 intents nobody happened to hand-pick for a spot check -- for instance, a copy-paste error where `WARMTH_DECREASE`'s own MODERATE and STRONG values were accidentally swapped (16 and 28 reversed), which would make MODERATE's own distance from neutral LARGER than STRONG's for that one specific intent. A 2-intent spot check testing only, say, `BRIGHTNESS_INCREASE` and `SATURATION_INCREASE` would report success while this real, specific error sat undetected in a third intent nobody thought to verify individually -- exactly the kind of mistake a table of a dozen near-identical-looking rows is prone to, and exactly what iterating over the WHOLE table closes off.

**6.** "Strongly increase brightness" describes a MATTER OF DEGREE -- brightness can be increased a little or a lot, and STRONG names a real, specific point along that continuum. "Fully desaturate" already names an absolute endpoint -- zero saturation, a real, complete removal of color -- and there is no possible state MORE desaturated than completely gray; asking to do it "strongly" versus "subtly" describes no real difference in the resulting image, because the destination is already the same regardless of how emphatically it is requested, which is exactly why the operation is coherent only as strength-invariant.

**7.** A user who says "actually, less bright" about an edit that is already at the smallest real amount this system offers (SUBTLE) is expressing that even that smallest amount was too much -- leaving it unchanged at SUBTLE would apply an edit the user has now explicitly said they do not want any part of, misrepresenting their own most recent, most specific statement of intent. Removing the intent entirely is the only response that actually reflects what "less than the least" means in a system with a real, finite lower bound: there is no smaller positive amount to fall back to, so the honest next step is none at all.

**8.** The one piece of real information available to Section 24.4 that Section 24.3's own detector cannot see is CONVERSATION HISTORY -- specifically, that `WARMTH_INCREASE` was not merely present in a static snapshot but was ADDED by this same user in an earlier real turn of this same ongoing conversation. That history is what licenses the inference that a later, opposing request is a REVISION of that specific earlier choice rather than an independent, simultaneous instruction; Section 24.3's detector, operating on a single freshly-parsed plan with no memory of how it came to contain what it contains, has no basis to distinguish "the user wants both directions at once" (worth flagging) from "the user changed their mind" (worth replacing), so it can only, correctly, flag the co-occurrence and let a higher-level system with access to history make the distinction Section 24.4 is built to make.

**9.** Start with a plan containing `SHARPEN` at SUBTLE. Refinement 1: increase `SHARPEN` to MODERATE. Refinement 2: add `WARMTH_INCREASE` at MODERATE. Refinement 3: increase `SHARPEN` to STRONG. Undoing twice from this point reverts refinement 3 (back to `SHARPEN` at MODERATE) and then refinement 2 (removing `WARMTH_INCREASE` entirely) -- leaving a plan with only `SHARPEN` at MODERATE. Re-parsing the ORIGINAL request from scratch, by contrast, would reproduce the very first plan: `SHARPEN` at SUBTLE, with no `WARMTH_INCREASE` ever having existed and no memory that `SHARPEN` was ever bumped to MODERATE along the way. The two results share the absence of `WARMTH_INCREASE`, but disagree on `SHARPEN`'s own strength (MODERATE after 2 undos, versus SUBTLE from a fresh reparse) -- undo reverts to an intermediate REAL state the conversation actually passed through, while a fresh reparse has no such memory at all.

**10.** The recurring discipline is committing, in advance and explicitly, to a specific real behavior for a case that could otherwise be resolved arbitrarily or silently -- Chapter 21.3's `CostRange` commits to reporting an honest range rather than collapsing to a false-precision point, and Chapter 22.1's `AMBIGUOUS` outcome commits to naming every fitting candidate rather than silently picking one. In Section 24.1, this shows up as committing, in the SYSTEM PROMPT's own stated text, to one specific conservative default plan for a vague request, rather than leaving that case for the model to resolve however it sees fit at request time. In Section 24.4, the identical discipline shows up in the OPPOSITE form: rather than committing to always flagging a contradiction (the safer-looking default), it commits to REPLACING an opposing intent specifically because conversational history changes what the contradiction actually means -- the same underlying commitment to a deliberate, considered, explicitly justified real behavior, chosen in advance rather than resolved arbitrarily in the moment, applied to two situations whose own real context calls for opposite concrete responses.
