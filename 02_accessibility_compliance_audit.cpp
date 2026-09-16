// Chapter 22.2 -- An accessibility audit is one of the few domains this
// book has reached where the "structured extraction" a vision-language
// model produces from a screenshot can be checked against a real,
// EXACT, publicly standardized formula rather than a stated policy
// choice of this book's own invention. The Web Content Accessibility
// Guidelines define color contrast, minimum touch-target size, and
// text-alternative requirements in precise, checkable terms, and this
// section builds the real math -- relative luminance, contrast ratio,
// and the AA/AAA pass thresholds -- from scratch, verified against a
// real, famous reference value (pure black text on a pure white
// background contrasts at exactly 21:1) rather than an invented
// tolerance this book chose for itself.
//
// A note on this section's own honest scope: this section implements
// WCAG 2.x's own contrast-ratio formula and a small, real subset of its
// success criteria (1.4.3 Contrast Minimum, 2.5.5 Target Size, 1.1.1
// Non-text Content) exactly as the specification defines them, not the
// whole of WCAG, and every violation this section's own audit reports
// names the SPECIFIC success criterion it failed -- never a vague
// "accessibility issue" -- exactly the same named-reason-for-refusal
// discipline this book has applied to every other refusal since
// Chapter 19.1's own token-budget guard.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_accessibility_compliance_audit.cpp -o 02_accessibility_compliance_audit
// Run:     ./02_accessibility_compliance_audit

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
// PART 1: the real WCAG 2.x relative-luminance and contrast-ratio
// formulas, implemented exactly as the specification defines them.
// =======================================================================
struct RgbColor { int r = 0, g = 0, b = 0; };

double srgb_channel_to_linear(int channel_0_255) {
    double c = static_cast<double>(channel_0_255) / 255.0;
    if (c <= 0.03928) return c / 12.92;
    return std::pow((c + 0.055) / 1.055, 2.4);
}

// The exact WCAG relative-luminance formula: a weighted sum of the
// linearized channels, weighted by human luminance perception (green
// contributes far more than blue).
double relative_luminance(const RgbColor& c) {
    double r_lin = srgb_channel_to_linear(c.r);
    double g_lin = srgb_channel_to_linear(c.g);
    double b_lin = srgb_channel_to_linear(c.b);
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;
}

// The exact WCAG contrast-ratio formula: (lighter + 0.05) / (darker +
// 0.05), with the LIGHTER of the two luminances always in the
// numerator -- getting this ordering backwards is a real, common
// implementation bug that silently produces a contrast ratio below 1.0
// for exactly half of all real color pairs, so this function computes
// the max/min itself rather than trusting a caller to pass them in the
// right order.
double contrast_ratio(const RgbColor& a, const RgbColor& b) {
    double la = relative_luminance(a), lb = relative_luminance(b);
    double lighter = std::max(la, lb), darker = std::min(la, lb);
    return (lighter + 0.05) / (darker + 0.05);
}

// =======================================================================
// PART 2: WCAG AA/AAA classification -- a real, stated pair of
// thresholds per text size, applied to a plain numeric ratio so this
// section's own classification logic can be tested independently of
// how that ratio was computed.
// =======================================================================
enum class TextSize { NORMAL, LARGE };

struct WcagClassification { bool meets_aa = false; bool meets_aaa = false; };

WcagClassification classify_contrast(double ratio, TextSize size) {
    double aa_threshold = (size == TextSize::NORMAL) ? 4.5 : 3.0;
    double aaa_threshold = (size == TextSize::NORMAL) ? 7.0 : 4.5;
    return WcagClassification{ratio >= aa_threshold, ratio >= aaa_threshold};
}

// =======================================================================
// PART 3: the audit engine -- real, named, per-criterion checks over a
// UI element, and a report that never contains a vague, unattributed
// violation.
// =======================================================================
struct UiElement {
    std::string id;
    RgbColor foreground, background;
    TextSize text_size = TextSize::NORMAL;
    int width = 0, height = 0;
    bool is_interactive = false;
    bool is_image = false;
    std::string alt_text;
};

// A real, stated policy choice: this section's own audit enforces the
// AA bar (the level most real accessibility regulations, including the
// ADA and Section 508, actually cite as their own minimum), not the
// stricter, optional AAA bar -- an element failing AAA but passing AA is
// not reported as a violation at all.
constexpr int MIN_TOUCH_TARGET_PX = 44;

std::vector<std::string> audit_element(const UiElement& e) {
    std::vector<std::string> violations;
    double ratio = contrast_ratio(e.foreground, e.background);
    auto classification = classify_contrast(ratio, e.text_size);
    if (!classification.meets_aa) {
        violations.push_back(e.id + ": 1.4.3 Contrast (Minimum) -- measured ratio " + std::to_string(ratio) +
                              ":1 is below the required AA threshold");
    }
    if (e.is_interactive && (e.width < MIN_TOUCH_TARGET_PX || e.height < MIN_TOUCH_TARGET_PX)) {
        violations.push_back(e.id + ": 2.5.5 Target Size -- " + std::to_string(e.width) + "x" +
                              std::to_string(e.height) + "px is below the required " +
                              std::to_string(MIN_TOUCH_TARGET_PX) + "x" + std::to_string(MIN_TOUCH_TARGET_PX) + "px minimum");
    }
    if (e.is_image && e.alt_text.empty()) {
        violations.push_back(e.id + ": 1.1.1 Non-text Content -- image element has no text alternative");
    }
    return violations;
}

std::vector<std::string> audit_elements(const std::vector<UiElement>& elements) {
    std::vector<std::string> all_violations;
    for (const auto& e : elements) {
        auto v = audit_element(e);
        all_violations.insert(all_violations.end(), v.begin(), v.end());
    }
    return all_violations;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 22.2: An Accessibility-Compliance Auditing Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real WCAG contrast formula matches its own famous exact reference value "
                 "-- pure black on pure white contrasts at exactly 21:1 --\n";
    {
        RgbColor black{0, 0, 0}, white{255, 255, 255};
        double ratio = contrast_ratio(black, white);
        CHECK(std::abs(ratio - 21.0) < 1e-9);
        double self_ratio = contrast_ratio(white, white);
        CHECK(std::abs(self_ratio - 1.0) < 1e-9);
        std::cout << "  black text on a white background computes to a contrast ratio of " << ratio
                   << ":1, matching WCAG's own famous exact reference value; white on white computes to "
                     "exactly 1:1 (no contrast at all)\n";
    }

    std::cout << "\n-- Test 2: contrast_ratio is order-independent -- the same real bug of forgetting to "
                 "take the lighter/darker max/min would silently break this --\n";
    {
        RgbColor light_gray{200, 200, 200}, dark_gray{50, 50, 50};
        double forward = contrast_ratio(light_gray, dark_gray);
        double reversed = contrast_ratio(dark_gray, light_gray);
        CHECK(forward == reversed);
        CHECK(forward > 1.0);
        std::cout << "  contrast_ratio(light, dark) and contrast_ratio(dark, light) both compute to the "
                     "identical value (" << forward << ":1), confirming the lighter/darker ordering is "
                     "handled internally rather than assumed from argument order\n";
    }

    std::cout << "\n-- Test 3: AA and AAA classification thresholds are exact at their own stated boundary "
                 "values, for both text sizes --\n";
    {
        CHECK(classify_contrast(4.5, TextSize::NORMAL).meets_aa);
        CHECK(!classify_contrast(4.499, TextSize::NORMAL).meets_aa);
        CHECK(classify_contrast(3.0, TextSize::LARGE).meets_aa);
        CHECK(!classify_contrast(2.999, TextSize::LARGE).meets_aa);
        CHECK(classify_contrast(7.0, TextSize::NORMAL).meets_aaa);
        CHECK(!classify_contrast(6.999, TextSize::NORMAL).meets_aaa);
        CHECK(classify_contrast(4.5, TextSize::LARGE).meets_aaa);
        CHECK(!classify_contrast(4.499, TextSize::LARGE).meets_aaa);
        std::cout << "  normal text passes AA at exactly 4.5:1 and fails just below it; large text passes "
                     "AA at exactly 3.0:1; normal text passes AAA at exactly 7.0:1; large text passes AAA "
                     "at exactly 4.5:1 -- every stated threshold is exact, not approximate\n";
    }

    std::cout << "\n-- Test 4: a fully compliant element produces zero violations, and a non-compliant "
                 "element is flagged with all 3 real, specific, named criteria --\n";
    {
        UiElement compliant{"btn-save", RgbColor{0, 0, 0}, RgbColor{255, 255, 255}, TextSize::NORMAL,
                             48, 48, true, false, ""};
        auto compliant_violations = audit_element(compliant);
        CHECK(compliant_violations.empty());

        UiElement noncompliant{"icon-delete", RgbColor{180, 180, 180}, RgbColor{200, 200, 200}, TextSize::NORMAL,
                                30, 30, true, true, ""};
        auto bad_violations = audit_element(noncompliant);
        CHECK(bad_violations.size() == 3);
        bool has_contrast = false, has_target = false, has_alt = false;
        for (const auto& v : bad_violations) {
            if (v.find("1.4.3") != std::string::npos) has_contrast = true;
            if (v.find("2.5.5") != std::string::npos) has_target = true;
            if (v.find("1.1.1") != std::string::npos) has_alt = true;
        }
        CHECK(has_contrast && has_target && has_alt);
        std::cout << "  a fully compliant 48x48 black-on-white button produces zero violations; a "
                     "30x30 low-contrast icon with no alt text is flagged with all 3 real, specific "
                     "criteria at once: 1.4.3 Contrast, 2.5.5 Target Size, and 1.1.1 Non-text Content\n";
    }

    std::cout << "\n-- Test 5: a full multi-element audit report attributes every violation to its own "
                 "specific element, with no false positives on the compliant elements sharing the report --\n";
    {
        UiElement good_text{"label-total", RgbColor{20, 20, 20}, RgbColor{255, 255, 255}, TextSize::NORMAL,
                             0, 0, false, false, ""};
        UiElement good_button{"btn-confirm", RgbColor{255, 255, 255}, RgbColor{0, 90, 0}, TextSize::LARGE,
                               60, 60, true, false, ""};
        UiElement bad_icon{"icon-delete", RgbColor{180, 180, 180}, RgbColor{200, 200, 200}, TextSize::NORMAL,
                            30, 30, true, true, ""};
        auto report = audit_elements({good_text, good_button, bad_icon});
        CHECK(report.size() == 3);   // all 3 from icon-delete alone
        for (const auto& v : report) {
            CHECK(v.find("icon-delete") != std::string::npos);
        }
        std::cout << "  a 3-element report (a compliant label, a compliant large button, and the "
                     "non-compliant icon from Test 4) produces exactly 3 violations, all attributed to "
                     "icon-delete -- the two genuinely compliant elements contribute zero false "
                     "positives\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
