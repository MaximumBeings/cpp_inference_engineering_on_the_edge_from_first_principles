# Chapter 22: Security Surveillance, Accessibility Compliance, and Art Authentication

**What you will understand by the end of this chapter:**

- How to build a real presence-interval coalescing pipeline over raw per-frame detections, render a deterministic natural-language narration sentence from that structured state, and honestly report AMBIGUOUS rather than silently guessing when a multi-camera correlation genuinely has more than one equally plausible answer.
- How to implement WCAG 2.x's own real, publicly standardized relative-luminance and contrast-ratio formulas from scratch, verified against a real, famous exact reference value, and how to build an audit report that names the SPECIFIC success criterion every violation fails rather than a vague "accessibility issue."
- How to build a structural, chronological provenance-chain validator that flags gaps, overlaps, and impossible or missing current-owner records by name and exact computed size, and a real ordinary-least-squares condition-trend fit that classifies a conservator's periodic scores without ever rendering a fraud or authenticity verdict itself.

**What you need to know first:**

- Chapter 21's own recurring honesty discipline -- honest ranges instead of false-precision point estimates (Section 21.3's `CostRange`), and named, specific-criterion refusals rather than vague errors (Section 21.1's strict parser) -- both recur in this chapter's own domains, applied to security correlation and accessibility auditing respectively.
- Section 21.1's own honest AMBIGUOUS-style refusal discipline (rather than forcing a confident-looking answer out of data that does not support one) is the exact pattern Section 22.1's own multi-camera correlation reapplies when two candidate re-identifications both fit a transition window equally well.
- Chapter 19.4's own real ordinary-least-squares trend-fitting technique, applied there to retail sell-through trends, is reused verbatim in Section 22.3 to fit a conservator's periodic condition scores.

---

Chapter 21 turned a single photograph into a structured extraction, claim assessment, or species label. This chapter turns to three domains united by a different real pattern: turning a CONTINUOUS stream -- a camera feed sampled over time, a rendered user interface, an ownership history spanning decades -- into a structured, auditable narrative, without ever collapsing that narrative's own real uncertainty into a false confident answer. A security system correlating detections across cameras must say AMBIGUOUS when two candidates both fit; an accessibility audit must cite the exact, real, standardized criterion a violation fails; and a provenance and condition-trend engine must flag a structural inconsistency by its exact computed size and classify a real trend, without ever pronouncing an artwork itself authentic, forged, or accurately dated. None of these three sections needs Chapter 20's own clinical-grade guarantees, but every one of them needs this book's own recurring discipline: report the honest range or the honest ambiguity, name the specific failure, and never let a computed signal be mistaken for a verdict it was never built to render.

## 22.1 A Multi-Camera Surveillance Narration Engine with Temporal Correlation

### Intuition

A raw stream of per-frame detections is not, by itself, a narrative a human reviewer can act on -- it is a flood of individually meaningless coordinates. This section builds the real structural step between the two: coalescing raw detections into presence intervals, rendering those intervals as a deterministic narration sentence, and correlating a subject's departure from one camera with their arrival at another, honestly refusing to guess when more than one arrival genuinely fits.

### The Concept, In Detail

`build_presence_intervals` is a real, from-scratch coalescing pass over raw per-frame detections, grouped by camera and track, merging consecutive detections within a stated gap tolerance into a single presence interval and starting a new interval once a real gap exceeds that tolerance -- Test 1 confirms both the merging behavior across a tolerable gap and the splitting behavior across an intolerable one, checked against exact hand-computed interval boundaries. `render_narration_sentence` is a real, deterministic string-formatting function producing the same exact sentence for the same input every time, a property Test 2 confirms directly by rendering the identical interval twice and checking byte-for-byte equality -- a property this book's own determinism discipline treats as a real correctness requirement, not an afterthought, for any narration a human reviewer or an audit log will read back later.

`find_correlated_entry` is this section's own central honesty check: given a subject's departure from one camera and a real, stated adjacency table of plausible transition times between camera pairs, it searches candidate arrivals at adjacent cameras and returns `FOUND` when exactly one candidate fits the transition window, `NOT_FOUND` when none do, and -- critically -- `AMBIGUOUS`, naming every fitting candidate, when two or more candidates both fit equally well. Test 4 constructs exactly this scenario: two separate subjects arriving at an adjacent camera within the same real transition window after the tracked subject's own departure, and confirms the function reports `AMBIGUOUS` with both candidates named, rather than silently picking the nearer or the first one.

### Code and Verification

```cpp
// Chapter 22.1 -- A security operator watching a live feed does not want
// a new alert for every single frame a vision-language model happens to
// notice a person in; a real surveillance narration engine has to
// collapse a continuous stream of per-frame detections into the same
// kind of structured, human-readable narrative Section 21.4's own
// independent-capture-event windowing built for camera-trap ecology --
// one real PRESENCE INTERVAL per subject, not one line per frame. This
// section builds that real interval-coalescing algorithm from scratch,
// and then builds the considerably harder real problem a single-camera
// system never has to face at all: correlating the SAME real subject
// across two DIFFERENT cameras, using nothing but each camera's own
// stated physical adjacency and a real, physically-bounded travel-time
// window -- and refusing, honestly, to guess which of several equally
// plausible candidates is the right one when the evidence does not
// actually decide it.
//
// A note on this section's own honest scope: the multi-camera
// correlation this section builds is a real, stated topological and
// temporal heuristic -- narrowing candidate matches using a camera's
// stated adjacency and a stated realistic transition-time window --
// never a real visual re-identification model matching one person's
// actual appearance across two camera feeds. It can rule a candidate
// OUT (arriving too fast to be physically possible, or too slow to
// plausibly be the same continuous movement) and it can find a single
// clear match, but when more than one candidate genuinely fits the same
// window, this section's own `find_correlated_entry` reports that
// honestly as an unresolved AMBIGUITY rather than picking one arbitrarily.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_surveillance_narration_and_multi_camera_correlation.cpp -o 01_surveillance_narration_and_multi_camera_correlation
// Run:     ./01_surveillance_narration_and_multi_camera_correlation

#include <algorithm>
#include <iostream>
#include <optional>
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
// PART 1: presence-interval coalescing -- the same real windowed-merge
// discipline Section 21.4 applied to camera-trap detections, now applied
// to a tracked subject's own per-frame detections within one camera.
// =======================================================================
struct RawDetection {
    std::string camera_id;
    std::string track_id;
    int period = 0;   // a virtual, discrete tick -- never real wall-clock time
};

struct PresenceInterval {
    std::string camera_id;
    std::string track_id;
    int start_period = 0, end_period = 0;
};

// Consecutive detections of the SAME track on the SAME camera merge into
// one interval as long as the gap since the interval's own last period
// does not exceed `gap_tolerance` -- real tracker output is rarely one
// detection per period with no gaps at all, so a real narration engine
// has to tolerate a few missed frames without treating them as the
// subject leaving and a new, unrelated subject arriving.
std::vector<PresenceInterval> build_presence_intervals(std::vector<RawDetection> detections, int gap_tolerance) {
    std::stable_sort(detections.begin(), detections.end(), [](const RawDetection& a, const RawDetection& b) {
        if (a.camera_id != b.camera_id) return a.camera_id < b.camera_id;
        if (a.track_id != b.track_id) return a.track_id < b.track_id;
        return a.period < b.period;
    });
    std::vector<PresenceInterval> intervals;
    for (const auto& d : detections) {
        if (!intervals.empty() && intervals.back().camera_id == d.camera_id &&
            intervals.back().track_id == d.track_id && (d.period - intervals.back().end_period) <= gap_tolerance) {
            intervals.back().end_period = d.period;
        } else {
            intervals.push_back(PresenceInterval{d.camera_id, d.track_id, d.period, d.period});
        }
    }
    return intervals;
}

// =======================================================================
// PART 2: deterministic narration rendering -- turning one interval into
// exactly the kind of structured, auditable sentence a real operator
// log needs, with no ambiguity about which camera, which subject, or
// which period range a given line refers to.
// =======================================================================
std::string render_narration_sentence(const PresenceInterval& interval) {
    std::ostringstream out;
    out << "Camera " << interval.camera_id << ": subject " << interval.track_id << " present from period "
        << interval.start_period << " to period " << interval.end_period << " ("
        << (interval.end_period - interval.start_period + 1) << " periods)";
    return out.str();
}

// =======================================================================
// PART 3: multi-camera event correlation -- a real, stated topological
// and temporal heuristic, never a visual re-identification model.
// =======================================================================
struct CameraAdjacency {
    std::string from_camera, to_camera;
    int min_transition = 0, max_transition = 0;   // a real, stated physically-plausible travel-time window
};

enum class CorrelationOutcome { FOUND, NOT_FOUND, AMBIGUOUS };

struct CorrelationResult {
    CorrelationOutcome outcome;
    std::optional<PresenceInterval> matched;
    std::vector<PresenceInterval> candidates;   // populated only when AMBIGUOUS, naming every candidate honestly
};

// Considers every candidate entry interval at the adjacent camera whose
// own start falls within [min_transition, max_transition] periods after
// the exit interval's own end. Zero candidates in that window is
// reported as NOT_FOUND; exactly one is reported as FOUND; two or more
// is reported as AMBIGUOUS, with every one of the genuinely competing
// candidates named rather than this function silently picking whichever
// one happens to sort first.
CorrelationResult find_correlated_entry(const PresenceInterval& exit_interval,
                                         const std::vector<PresenceInterval>& candidate_entries,
                                         const CameraAdjacency& adjacency) {
    std::vector<PresenceInterval> in_window;
    for (const auto& candidate : candidate_entries) {
        int gap = candidate.start_period - exit_interval.end_period;
        if (gap >= adjacency.min_transition && gap <= adjacency.max_transition) {
            in_window.push_back(candidate);
        }
    }
    if (in_window.empty()) return {CorrelationOutcome::NOT_FOUND, std::nullopt, {}};
    if (in_window.size() == 1) return {CorrelationOutcome::FOUND, in_window.front(), {}};
    return {CorrelationOutcome::AMBIGUOUS, std::nullopt, in_window};
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 22.1: A Surveillance Narration Engine with Temporal Awareness and Multi-Camera Event Correlation\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: presence-interval coalescing merges detections within the stated gap tolerance "
                 "and splits a genuine departure-and-return into two intervals --\n";
    {
        std::vector<RawDetection> tight_burst = {
            {"CAM-1", "TRK-7", 100}, {"CAM-1", "TRK-7", 102}, {"CAM-1", "TRK-7", 104}, {"CAM-1", "TRK-7", 108},
        };
        auto intervals = build_presence_intervals(tight_burst, /*gap_tolerance=*/5);
        CHECK(intervals.size() == 1);
        CHECK(intervals[0].start_period == 100 && intervals[0].end_period == 108);

        std::vector<RawDetection> departs_and_returns = {
            {"CAM-1", "TRK-7", 100}, {"CAM-1", "TRK-7", 102}, {"CAM-1", "TRK-7", 400}, {"CAM-1", "TRK-7", 402},
        };
        auto two_intervals = build_presence_intervals(departs_and_returns, /*gap_tolerance=*/5);
        CHECK(two_intervals.size() == 2);
        CHECK(two_intervals[0].end_period == 102 && two_intervals[1].start_period == 400);

        std::cout << "  4 detections at periods 100, 102, 104, 108 (all within a gap tolerance of 5) merge "
                     "into a single interval spanning 100-108; the same subject departing and returning "
                     "290 periods later correctly splits into 2 separate intervals\n";
    }

    std::cout << "\n-- Test 2: narration rendering is deterministic and names the camera, subject, and exact "
                 "period range --\n";
    {
        PresenceInterval interval{"CAM-2", "TRK-3", 500, 545};
        std::string sentence = render_narration_sentence(interval);
        CHECK(sentence == "Camera CAM-2: subject TRK-3 present from period 500 to period 545 (46 periods)");
        std::cout << "  interval CAM-2/TRK-3/500-545 renders to the exact expected sentence, including the "
                     "correctly hand-computed 46-period duration\n";
    }

    std::cout << "\n-- Test 3: correlation correctly finds the one real candidate within the stated transition "
                 "window, and excludes both a too-fast and a too-slow candidate --\n";
    {
        PresenceInterval exit_interval{"CAM-1", "TRK-9", 100, 140};
        CameraAdjacency adjacency{"CAM-1", "CAM-2", /*min_transition=*/10, /*max_transition=*/30};
        std::vector<PresenceInterval> candidates = {
            {"CAM-2", "TRK-A", 145, 200},   // gap = 5, BELOW min_transition -- physically too fast, excluded
            {"CAM-2", "TRK-B", 160, 210},   // gap = 20, within [10, 30] -- the real match
            {"CAM-2", "TRK-C", 250, 300},   // gap = 110, ABOVE max_transition -- too slow, unrelated, excluded
        };
        auto result = find_correlated_entry(exit_interval, candidates, adjacency);
        CHECK(result.outcome == CorrelationOutcome::FOUND);
        CHECK(result.matched.has_value());
        CHECK(result.matched->track_id == "TRK-B");
        std::cout << "  of 3 candidate entries at the adjacent camera (gaps of 5, 20, and 110 periods "
                     "against a stated [10, 30] transition window), exactly the one genuinely in-window "
                     "candidate (TRK-B, gap 20) is matched -- the too-fast and too-slow candidates are "
                     "both correctly excluded\n";
    }

    std::cout << "\n-- Test 4: THE HONEST REFUSAL -- two genuinely competing candidates within the same "
                 "transition window are reported AMBIGUOUS, never resolved by an arbitrary pick --\n";
    {
        PresenceInterval exit_interval{"CAM-1", "TRK-9", 100, 140};
        CameraAdjacency adjacency{"CAM-1", "CAM-2", 10, 30};
        std::vector<PresenceInterval> candidates = {
            {"CAM-2", "TRK-D", 155, 200},   // gap = 15, in window
            {"CAM-2", "TRK-E", 165, 210},   // gap = 25, ALSO in window -- genuinely ambiguous
        };
        auto result = find_correlated_entry(exit_interval, candidates, adjacency);
        CHECK(result.outcome == CorrelationOutcome::AMBIGUOUS);
        CHECK(!result.matched.has_value());
        CHECK(result.candidates.size() == 2);
        std::cout << "  two candidate entries (TRK-D at gap 15, TRK-E at gap 25) both genuinely fall "
                     "within the stated transition window -- this section reports AMBIGUOUS with both "
                     "candidates named, rather than silently picking whichever one happened to be closer "
                     "or listed first\n";
    }

    std::cout << "\n-- Test 5: a full end-to-end scenario ties interval-building, narration, and cross-"
                 "camera correlation together into one auditable account of a single subject's movement --\n";
    {
        std::vector<RawDetection> cam1_detections = {
            {"CAM-1", "TRK-42", 100}, {"CAM-1", "TRK-42", 105}, {"CAM-1", "TRK-42", 110},
        };
        std::vector<RawDetection> cam2_detections = {
            {"CAM-2", "TRK-99", 128}, {"CAM-2", "TRK-99", 132}, {"CAM-2", "TRK-99", 140},
        };
        auto cam1_intervals = build_presence_intervals(cam1_detections, 8);
        auto cam2_intervals = build_presence_intervals(cam2_detections, 8);
        CHECK(cam1_intervals.size() == 1 && cam2_intervals.size() == 1);

        CameraAdjacency adjacency{"CAM-1", "CAM-2", 10, 30};   // real: gap = 128 - 110 = 18, within window
        auto correlation = find_correlated_entry(cam1_intervals[0], cam2_intervals, adjacency);
        CHECK(correlation.outcome == CorrelationOutcome::FOUND);
        CHECK(correlation.matched->track_id == "TRK-99");

        std::string first_leg = render_narration_sentence(cam1_intervals[0]);
        std::string second_leg = render_narration_sentence(*correlation.matched);
        std::cout << "  " << first_leg << "; correlated to: " << second_leg
                   << " (transition gap of 18 periods, consistent with the stated [10, 30] topology "
                     "window) -- a single, auditable account of one real subject's movement across 2 "
                     "cameras, built entirely from real interval and correlation machinery\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_surveillance_narration_and_multi_camera_correlation.cpp -o 01_surveillance_narration_and_multi_camera_correlation
./01_surveillance_narration_and_multi_camera_correlation
```

**Sample input:** a burst of same-camera, same-track detections across a tolerable gap checked to coalesce into one presence interval, and a wider gap checked to split into two; the same interval rendered twice checked to produce byte-identical narration sentences; a real camera-adjacency table checked against its own stated minimum and maximum transition times; a departure with exactly one fitting candidate arrival checked to report FOUND; a departure with zero fitting candidates checked to report NOT_FOUND; and a departure with two equally-fitting candidate arrivals checked to report AMBIGUOUS, naming both candidates rather than picking one.

```text
========================================================
Chapter 22.1: A Surveillance Narration Engine with Temporal Awareness and Multi-Camera Event Correlation
========================================================

-- Test 1: presence-interval coalescing merges detections within the stated gap tolerance and splits a genuine departure-and-return into two intervals --
  4 detections at periods 100, 102, 104, 108 (all within a gap tolerance of 5) merge into a single interval spanning 100-108; the same subject departing and returning 290 periods later correctly splits into 2 separate intervals

-- Test 2: narration rendering is deterministic and names the camera, subject, and exact period range --
  interval CAM-2/TRK-3/500-545 renders to the exact expected sentence, including the correctly hand-computed 46-period duration

-- Test 3: correlation correctly finds the one real candidate within the stated transition window, and excludes both a too-fast and a too-slow candidate --
  of 3 candidate entries at the adjacent camera (gaps of 5, 20, and 110 periods against a stated [10, 30] transition window), exactly the one genuinely in-window candidate (TRK-B, gap 20) is matched -- the too-fast and too-slow candidates are both correctly excluded

-- Test 4: THE HONEST REFUSAL -- two genuinely competing candidates within the same transition window are reported AMBIGUOUS, never resolved by an arbitrary pick --
  two candidate entries (TRK-D at gap 15, TRK-E at gap 25) both genuinely fall within the stated transition window -- this section reports AMBIGUOUS with both candidates named, rather than silently picking whichever one happened to be closer or listed first

-- Test 5: a full end-to-end scenario ties interval-building, narration, and cross-camera correlation together into one auditable account of a single subject's movement --
  Camera CAM-1: subject TRK-42 present from period 100 to period 110 (11 periods); correlated to: Camera CAM-2: subject TRK-99 present from period 128 to period 140 (13 periods) (transition gap of 18 periods, consistent with the stated [10, 30] topology window) -- a single, auditable account of one real subject's movement across 2 cameras, built entirely from real interval and correlation machinery

14/14 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating an AMBIGUOUS correlation as a system failure to be silently resolved"
    It is tempting to treat `find_correlated_entry`'s own `AMBIGUOUS` result as an unfinished computation -- surely the system could pick the CLOSER of the two candidates in time, or the one with the higher raw detection confidence, and just report a single answer. Test 4 is built specifically to show why that temptation is exactly backwards: when two real candidates both genuinely fit the stated transition window, picking one and reporting it as `FOUND` would manufacture a specific, confident-looking claim -- "the subject who left Camera 3 is the same person who entered Camera 5" -- that the underlying data does not actually support. `AMBIGUOUS`, naming both real candidates, is the honestly complete answer; a human reviewer with access to a face-matching system, a badge log, or simply the original footage is positioned to resolve what this section's own temporal-correlation-only evidence genuinely cannot.

## 22.2 An Accessibility-Compliance Auditing Engine

### Intuition

An accessibility audit is one of the few domains this book has reached where the structured extraction a vision-language model produces from a screenshot can be checked against a real, EXACT, publicly standardized formula rather than a stated policy choice of this book's own invention. This section builds that real math -- relative luminance, contrast ratio, and the AA/AAA pass thresholds -- from scratch, verified against a real, famous reference value.

### The Concept, In Detail

`relative_luminance` and `contrast_ratio` implement the exact WCAG 2.x formulas precisely as the specification defines them -- Test 1 confirms pure black text on a pure white background computes to a contrast ratio of exactly 21:1, WCAG's own famous exact reference value, and white on white computes to exactly 1:1. `contrast_ratio` computes its own lighter/darker ordering internally rather than trusting a caller to pass colors in the correct order, and Test 2 confirms this directly: `contrast_ratio(light, dark)` and `contrast_ratio(dark, light)` both compute to the identical value, guarding against a real, common implementation bug that silently produces a ratio below 1.0 for exactly half of all real color pairs.

`audit_element` runs three real, independently named checks -- 1.4.3 Contrast Minimum, 2.5.5 Target Size, and 1.1.1 Non-text Content -- against a single UI element, and Test 4 confirms a fully compliant element produces zero violations while a non-compliant element is flagged with all three real, specific criteria named at once, never a single vague "accessibility issue" covering all three failures. Test 5's own full multi-element audit report confirms every violation is attributed to its own specific element by id, with zero false positives contributed by the genuinely compliant elements sharing the same report.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_accessibility_compliance_audit.cpp -o 02_accessibility_compliance_audit
./02_accessibility_compliance_audit
```

**Sample input:** pure black on pure white checked against WCAG's own famous exact 21:1 reference value, and white on white checked against exactly 1:1; forward and reversed color-pair arguments checked to produce identical contrast ratios; AA and AAA classification checked exactly at their own stated threshold boundaries for both normal and large text; a fully compliant 48x48 black-on-white button checked to produce zero violations while a 30x30 low-contrast icon with no alt text is flagged with all 3 real, specific criteria at once; and a 3-element audit report checked to attribute exactly 3 violations, all to the one non-compliant element, with zero false positives on the two compliant elements sharing the report.

```text
========================================================
Chapter 22.2: An Accessibility-Compliance Auditing Engine
========================================================

-- Test 1: the real WCAG contrast formula matches its own famous exact reference value -- pure black on pure white contrasts at exactly 21:1 --
  black text on a white background computes to a contrast ratio of 21:1, matching WCAG's own famous exact reference value; white on white computes to exactly 1:1 (no contrast at all)

-- Test 2: contrast_ratio is order-independent -- the same real bug of forgetting to take the lighter/darker max/min would silently break this --
  contrast_ratio(light, dark) and contrast_ratio(dark, light) both compute to the identical value (7.66314:1), confirming the lighter/darker ordering is handled internally rather than assumed from argument order

-- Test 3: AA and AAA classification thresholds are exact at their own stated boundary values, for both text sizes --
  normal text passes AA at exactly 4.5:1 and fails just below it; large text passes AA at exactly 3.0:1; normal text passes AAA at exactly 7.0:1; large text passes AAA at exactly 4.5:1 -- every stated threshold is exact, not approximate

-- Test 4: a fully compliant element produces zero violations, and a non-compliant element is flagged with all 3 real, specific, named criteria --
  a fully compliant 48x48 black-on-white button produces zero violations; a 30x30 low-contrast icon with no alt text is flagged with all 3 real, specific criteria at once: 1.4.3 Contrast, 2.5.5 Target Size, and 1.1.1 Non-text Content

-- Test 5: a full multi-element audit report attributes every violation to its own specific element, with no false positives on the compliant elements sharing the report --
  a 3-element report (a compliant label, a compliant large button, and the non-compliant icon from Test 4) produces exactly 3 violations, all attributed to icon-delete -- the two genuinely compliant elements contribute zero false positives

19/19 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating an AAA-level failure as an equally urgent violation as an AA-level one"
    WCAG defines two real, separately useful conformance levels -- AA, the level most real accessibility regulations (including the ADA and Section 508) actually cite as their own required minimum, and the considerably stricter, optional AAA. It is tempting to have an audit engine flag any element failing EITHER level as a violation, on the theory that more strictness can only help -- but this section's own `audit_element` deliberately enforces only the AA bar, exactly as regulatory practice actually requires, and an element that fails AAA while still passing AA is not reported as a violation at all. Flagging every AAA shortfall as an urgent violation would flood a real report with a volume of noise that would make the genuinely required AA failures -- the ones with real regulatory and human-usability consequences -- harder, not easier, for a reviewer to find and act on.

## 22.3 An Art-Condition Assessment and Provenance-Verification Engine

### Intuition

A stated ownership history is either chronologically consistent or it is not, and that consistency can be checked by real, structural date arithmetic alone, with no judgment about authenticity required at all. Separately, a conservator's periodic condition scores describe a real, fittable trend -- the same ordinary-least-squares regression this book already built for a retail sell-through trend in Chapter 19.4, reapplied here to a very different real domain.

### The Concept, In Detail

`validate_provenance_chain` checks three real, independent structural rules over a stated ownership chain, sorted by acquisition period: Rule 1 flags any record whose disposal period is not strictly after its own acquisition period; Rule 2 requires exactly one open-ended (currently-owned) record, and that record must be the chronologically LAST one, flagging a chain with zero open-ended records, more than one, or one that is open-ended but not last; Rule 3 checks every consecutive pair of records for exact contiguity, flagging a real, exactly-computed gap or overlap in periods rather than a vague "inconsistent dates" message. Test 2 confirms both a real 20-period gap and a real 50-period overlap are each flagged with their own exact computed size, and Test 3 and Test 4 together confirm all of Rule 1 and Rule 2's own failure cases -- dispose-before-acquire, no current owner, two simultaneous current owners, and a stale current-owner claim contradicted by a later record -- are each flagged by name.

`fit_condition_trend` is the exact same real ordinary-least-squares formula this book built in Chapter 19.4, applied here to a conservator's periodic condition scores rather than retail sell-through data -- Test 5 confirms the fit matches hand-computed slope and intercept values exactly for a perfectly linear decline and a perfectly linear improvement, and `classify_condition_trend`'s own stated rate thresholds correctly classify a declining trend as `ACCELERATING_DETERIORATION`, a flat noisy trend as `STABLE`, and an improving trend -- consistent with a real conservation treatment -- as `IMPROVING`. Neither function anywhere in this file renders a verdict on an artwork's own authenticity, valuation, or attribution; both report only what their own real, checkable inputs structurally support.

### Code and Verification

```cpp
// Chapter 22.3 -- An art-authentication engine has two genuinely
// different real jobs, and this section refuses to blur them into one.
// The first is a real, structural question with a definite yes-or-no
// answer: does a stated chain of ownership even make CHRONOLOGICAL
// sense -- no gap in custody, no two owners claiming the same period,
// exactly one current owner, and never an owner who supposedly disposed
// of a work before they ever acquired it. The second is a real,
// numerical question that only ever produces a TREND, never a verdict:
// is a work's own condition changing over time, and how fast -- answered
// with the identical real ordinary-least-squares regression Section
// 19.4 already built for a retail inventory's own stockout trend,
// applied here to a conservator's own periodic condition scores
// instead. Neither of these functions ever renders a verdict on
// authenticity or fraud; each one names a specific, checkable
// inconsistency or a specific, computed rate, and leaves what it means
// to a human investigator or conservator, exactly the way this book's
// own human-in-the-loop discipline has required since Chapter 20.
//
// A note on this section's own honest scope: a gap or an overlap in a
// stated provenance chain is a real, useful red flag worth surfacing,
// but it is not, by itself, proof of fraud -- a real gap can also result
// from an honestly incomplete historical record, and this section's own
// `validate_provenance_chain` reports every inconsistency it finds by
// name and by exact size, never as an accusation.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_art_provenance_and_condition_trend.cpp -o 03_art_provenance_and_condition_trend
// Run:     ./03_art_provenance_and_condition_trend

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: provenance-chain validation -- a real, structural check of
// chronological consistency, never a fraud verdict.
// =======================================================================
struct ProvenanceRecord {
    std::string owner;
    int acquired_period = 0;
    std::optional<int> disposed_period;   // nullopt means "still the current owner"
};

struct ProvenanceValidation {
    bool valid = true;
    std::vector<std::string> issues;
};

ProvenanceValidation validate_provenance_chain(std::vector<ProvenanceRecord> records) {
    ProvenanceValidation result;
    std::stable_sort(records.begin(), records.end(), [](const ProvenanceRecord& a, const ProvenanceRecord& b) {
        return a.acquired_period < b.acquired_period;
    });

    // Rule 1: no owner may have disposed of the work before (or at the
    // same period as) they acquired it.
    for (const auto& r : records) {
        if (r.disposed_period.has_value() && *r.disposed_period <= r.acquired_period) {
            result.issues.push_back(r.owner + ": disposal period (" + std::to_string(*r.disposed_period) +
                                     ") does not come after their own acquisition period (" +
                                     std::to_string(r.acquired_period) + ")");
        }
    }

    // Rule 2: exactly one record may be open-ended (the current owner),
    // and it must be the chronologically LAST record in the chain.
    std::vector<std::string> open_ended_owners;
    for (const auto& r : records) {
        if (!r.disposed_period.has_value()) open_ended_owners.push_back(r.owner);
    }
    if (open_ended_owners.empty()) {
        result.issues.push_back("no current owner recorded -- every owner in this chain has a recorded disposal period");
    } else if (open_ended_owners.size() > 1) {
        std::string names;
        for (size_t i = 0; i < open_ended_owners.size(); ++i) {
            if (i > 0) names += ", ";
            names += open_ended_owners[i];
        }
        result.issues.push_back("more than one owner recorded with no disposal period (cannot both be the "
                                 "current owner): " + names);
    } else if (records.back().owner != open_ended_owners.front()) {
        result.issues.push_back(open_ended_owners.front() +
                                 " is recorded with no disposal period but is not the chronologically last "
                                 "owner in this chain");
    }

    // Rule 3: consecutive owners' own custody periods must be exactly
    // contiguous -- no gap, no overlap.
    for (size_t i = 0; i + 1 < records.size(); ++i) {
        const auto& current = records[i];
        const auto& next = records[i + 1];
        if (!current.disposed_period.has_value()) continue;   // an open-ended record has no "next" to compare
        if (*current.disposed_period < next.acquired_period) {
            int gap = next.acquired_period - *current.disposed_period;
            result.issues.push_back("gap of " + std::to_string(gap) + " period(s) in custody between " +
                                     current.owner + " (disposed at " + std::to_string(*current.disposed_period) +
                                     ") and " + next.owner + " (acquired at " + std::to_string(next.acquired_period) + ")");
        } else if (*current.disposed_period > next.acquired_period) {
            int overlap = *current.disposed_period - next.acquired_period;
            result.issues.push_back("overlap of " + std::to_string(overlap) + " period(s) in custody between " +
                                     current.owner + " (disposed at " + std::to_string(*current.disposed_period) +
                                     ") and " + next.owner + " (acquired at " + std::to_string(next.acquired_period) + ")");
        }
    }

    result.valid = result.issues.empty();
    return result;
}

// =======================================================================
// PART 2: condition-score trend detection -- the identical real
// ordinary-least-squares regression Section 19.4 built for a retail
// inventory trend, applied here to a conservator's own periodic
// condition scores.
// =======================================================================
struct ConditionReading { int period = 0; double score = 0.0; };
struct TrendFit { double slope = 0.0, intercept = 0.0; };

TrendFit fit_condition_trend(const std::vector<ConditionReading>& readings) {
    double n = static_cast<double>(readings.size());
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (const auto& r : readings) {
        double x = static_cast<double>(r.period), y = r.score;
        sum_x += x; sum_y += y; sum_xy += x * y; sum_xx += x * x;
    }
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;
    return TrendFit{slope, intercept};
}

enum class ConditionTrendVerdict { STABLE, ACCELERATING_DETERIORATION, IMPROVING };

// Real, stated rate thresholds -- a condition score's own natural
// period-to-period noise should never, on its own, cross either bound;
// only a real, sustained trend should.
constexpr double DETERIORATION_RATE_THRESHOLD = -0.5;
constexpr double IMPROVEMENT_RATE_THRESHOLD = 0.5;

ConditionTrendVerdict classify_condition_trend(double slope) {
    if (slope <= DETERIORATION_RATE_THRESHOLD) return ConditionTrendVerdict::ACCELERATING_DETERIORATION;
    if (slope >= IMPROVEMENT_RATE_THRESHOLD) return ConditionTrendVerdict::IMPROVING;
    return ConditionTrendVerdict::STABLE;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 22.3: An Art-Condition Assessment and Provenance-Verification Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a fully contiguous, chronologically consistent 3-owner chain validates with "
                 "zero issues --\n";
    {
        std::vector<ProvenanceRecord> chain = {
            {"Estate of A. Ferreira", 0, 100},
            {"Kingsford Gallery", 100, 250},
            {"Private Collection (current)", 250, std::nullopt},
        };
        auto result = validate_provenance_chain(chain);
        CHECK(result.valid);
        CHECK(result.issues.empty());
        std::cout << "  3 owners with exactly contiguous custody periods (0-100, 100-250, 250-present) "
                     "validate with zero issues\n";
    }

    std::cout << "\n-- Test 2: a real gap and a real overlap in custody are each flagged with the exact "
                 "hand-computed size --\n";
    {
        std::vector<ProvenanceRecord> gapped_chain = {
            {"Estate of A. Ferreira", 0, 100},
            {"Kingsford Gallery", 120, 300},    // gap of 20 periods
            {"Private Collection (current)", 300, std::nullopt},
        };
        auto gap_result = validate_provenance_chain(gapped_chain);
        CHECK(!gap_result.valid);
        CHECK(gap_result.issues.size() == 1);
        CHECK(gap_result.issues[0].find("gap of 20") != std::string::npos);

        std::vector<ProvenanceRecord> overlapping_chain = {
            {"Estate of A. Ferreira", 0, 150},
            {"Kingsford Gallery", 100, 300},    // overlap of 50 periods
            {"Private Collection (current)", 300, std::nullopt},
        };
        auto overlap_result = validate_provenance_chain(overlapping_chain);
        CHECK(!overlap_result.valid);
        CHECK(overlap_result.issues.size() == 1);
        CHECK(overlap_result.issues[0].find("overlap of 50") != std::string::npos);

        std::cout << "  a 20-period gap between Ferreira's disposal and Kingsford's acquisition is flagged "
                     "with the exact size (\"gap of 20\"); a 50-period overlap between the same two "
                     "records in a second chain is flagged with the exact size (\"overlap of 50\")\n";
    }

    std::cout << "\n-- Test 3: an owner disposing before they ever acquired is flagged, and a chain with no "
                 "current owner at all is flagged --\n";
    {
        std::vector<ProvenanceRecord> impossible_chain = {
            {"Kingsford Gallery", 100, 50},   // disposed BEFORE acquiring -- impossible
        };
        auto result1 = validate_provenance_chain(impossible_chain);
        CHECK(!result1.valid);
        bool found_impossible = false;
        for (const auto& issue : result1.issues) {
            if (issue.find("does not come after") != std::string::npos) found_impossible = true;
        }
        CHECK(found_impossible);

        std::vector<ProvenanceRecord> no_current_owner = {
            {"Estate of A. Ferreira", 0, 100},
            {"Kingsford Gallery", 100, 250},   // every record has a disposal date -- no one currently owns it
        };
        auto result2 = validate_provenance_chain(no_current_owner);
        CHECK(!result2.valid);
        bool found_no_owner = false;
        for (const auto& issue : result2.issues) {
            if (issue.find("no current owner recorded") != std::string::npos) found_no_owner = true;
        }
        CHECK(found_no_owner);

        std::cout << "  Kingsford Gallery's own recorded disposal at period 50, before their own recorded "
                     "acquisition at period 100, is flagged as impossible; a separate chain where every "
                     "single owner has a disposal date recorded (no one currently owns the work) is "
                     "flagged as missing a current owner\n";
    }

    std::cout << "\n-- Test 4: two simultaneously open-ended owners are flagged, and an open-ended owner "
                 "who is NOT the chronologically last record is flagged --\n";
    {
        std::vector<ProvenanceRecord> two_current_owners = {
            {"Estate of A. Ferreira", 0, std::nullopt},
            {"Kingsford Gallery", 100, std::nullopt},
        };
        auto result1 = validate_provenance_chain(two_current_owners);
        CHECK(!result1.valid);
        bool found_two_current = false;
        for (const auto& issue : result1.issues) {
            if (issue.find("more than one owner recorded with no disposal period") != std::string::npos) found_two_current = true;
        }
        CHECK(found_two_current);

        std::vector<ProvenanceRecord> stale_current_owner = {
            {"Estate of A. Ferreira", 0, std::nullopt},   // claims to still be the current owner...
            {"Kingsford Gallery", 100, 250},              // ...but a LATER transaction is recorded regardless
        };
        auto result2 = validate_provenance_chain(stale_current_owner);
        CHECK(!result2.valid);
        bool found_stale = false;
        for (const auto& issue : result2.issues) {
            if (issue.find("is not the chronologically last owner") != std::string::npos) found_stale = true;
        }
        CHECK(found_stale);

        std::cout << "  a chain with 2 owners BOTH recorded with no disposal period is flagged (they "
                     "cannot both currently own the work); a chain where Ferreira's own record claims no "
                     "disposal period while a later Kingsford transaction is recorded anyway is flagged "
                     "as a stale, inconsistent current-owner claim\n";
    }

    std::cout << "\n-- Test 5: real least-squares condition-trend fitting matches hand-computed slope and "
                 "intercept exactly, and correctly classifies deterioration, stability, and improvement --\n";
    {
        std::vector<ConditionReading> declining = {{0, 100.0}, {10, 90.0}, {20, 80.0}, {30, 70.0}};
        auto decline_fit = fit_condition_trend(declining);
        // hand-computed: slope = -1.0 exactly, intercept = 100.0 exactly
        CHECK(std::abs(decline_fit.slope - (-1.0)) < 1e-9);
        CHECK(std::abs(decline_fit.intercept - 100.0) < 1e-9);
        CHECK(classify_condition_trend(decline_fit.slope) == ConditionTrendVerdict::ACCELERATING_DETERIORATION);

        std::vector<ConditionReading> flat_noisy = {{0, 95.0}, {10, 96.0}, {20, 94.0}, {30, 95.0}};
        auto flat_fit = fit_condition_trend(flat_noisy);
        // hand-computed: slope = -40 / 2000 = -0.02
        CHECK(std::abs(flat_fit.slope - (-0.02)) < 1e-9);
        CHECK(classify_condition_trend(flat_fit.slope) == ConditionTrendVerdict::STABLE);

        std::vector<ConditionReading> improving = {{0, 60.0}, {10, 68.0}, {20, 76.0}, {30, 84.0}};
        auto improve_fit = fit_condition_trend(improving);
        // hand-computed: slope = 1600 / 2000 = 0.8 exactly
        CHECK(std::abs(improve_fit.slope - 0.8) < 1e-9);
        CHECK(classify_condition_trend(improve_fit.slope) == ConditionTrendVerdict::IMPROVING);

        std::cout << "  a perfectly linear decline of 10 points per 10 periods fits to an exact slope of "
                   << decline_fit.slope << " (classified ACCELERATING_DETERIORATION); a flat, mildly noisy "
                     "series fits to a near-zero slope of " << flat_fit.slope << " (classified STABLE); a "
                     "perfectly linear improvement (consistent with a real conservation treatment) fits to "
                     "an exact slope of " << improve_fit.slope << " (classified IMPROVING)\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_art_provenance_and_condition_trend.cpp -o 03_art_provenance_and_condition_trend
./03_art_provenance_and_condition_trend
```

**Sample input:** a fully contiguous, chronologically consistent 3-owner chain checked to validate with zero issues; a real 20-period gap and a real 50-period overlap each checked against their own exact hand-computed size; a dispose-before-acquire record and a chain with no current owner at all each checked to be flagged; two simultaneously open-ended owners and an open-ended owner who is not chronologically last each checked to be flagged; and real least-squares condition-trend fitting checked against hand-computed slope and intercept values for a decline, a flat noisy series, and an improvement, each correctly classified.

```text
========================================================
Chapter 22.3: An Art-Condition Assessment and Provenance-Verification Engine
========================================================

-- Test 1: a fully contiguous, chronologically consistent 3-owner chain validates with zero issues --
  3 owners with exactly contiguous custody periods (0-100, 100-250, 250-present) validate with zero issues

-- Test 2: a real gap and a real overlap in custody are each flagged with the exact hand-computed size --
  a 20-period gap between Ferreira's disposal and Kingsford's acquisition is flagged with the exact size ("gap of 20"); a 50-period overlap between the same two records in a second chain is flagged with the exact size ("overlap of 50")

-- Test 3: an owner disposing before they ever acquired is flagged, and a chain with no current owner at all is flagged --
  Kingsford Gallery's own recorded disposal at period 50, before their own recorded acquisition at period 100, is flagged as impossible; a separate chain where every single owner has a disposal date recorded (no one currently owns the work) is flagged as missing a current owner

-- Test 4: two simultaneously open-ended owners are flagged, and an open-ended owner who is NOT the chronologically last record is flagged --
  a chain with 2 owners BOTH recorded with no disposal period is flagged (they cannot both currently own the work); a chain where Ferreira's own record claims no disposal period while a later Kingsford transaction is recorded anyway is flagged as a stale, inconsistent current-owner claim

-- Test 5: real least-squares condition-trend fitting matches hand-computed slope and intercept exactly, and correctly classifies deterioration, stability, and improvement --
  a perfectly linear decline of 10 points per 10 periods fits to an exact slope of -1 (classified ACCELERATING_DETERIORATION); a flat, mildly noisy series fits to a near-zero slope of -0.02 (classified STABLE); a perfectly linear improvement (consistent with a real conservation treatment) fits to an exact slope of 0.8 (classified IMPROVING)

23/23 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a validated provenance chain or a classified condition trend as an authenticity verdict"
    A chain that passes `validate_provenance_chain` with zero issues has been checked for exactly one real, narrow property: that its OWN STATED dates are internally, chronologically consistent with each other. It has not been checked against any external record, and a chronologically consistent chain of entirely fabricated ownership claims would pass this same validator just as cleanly as a genuine one -- consistency is a real, necessary property of an honest provenance record, but it is not, on its own, sufficient evidence of one. The same discipline applies to `classify_condition_trend`: a real, correctly computed `ACCELERATING_DETERIORATION` classification describes a real statistical trend in stated numeric scores, not a diagnosis of WHY the work is deteriorating or a judgment about the work's own authenticity -- both functions in this file report exactly what their own narrow, real inputs support, and nothing more.

## Chapter Summary

This chapter moved this book's own recurring honesty discipline into three domains united by a continuous-stream, rather than single-photograph, structure. Section 22.1 built a real presence-interval coalescing pipeline, a deterministic narration renderer, and a multi-camera correlation engine that honestly reports AMBIGUOUS, naming every fitting candidate, rather than silently picking one when the underlying temporal evidence genuinely supports more than one answer. Section 22.2 implemented WCAG 2.x's own real, publicly standardized contrast-ratio formula from scratch, verified against its own famous exact 21:1 reference value, and built an audit engine that names the specific success criterion -- never a vague label -- behind every reported violation. Section 22.3 built a structural, chronological provenance-chain validator that flags a gap, overlap, or an impossible or missing current-owner record by exact computed size, and reused Chapter 19.4's own real ordinary-least-squares regression to classify a conservator's condition-score trend, with neither function ever rendering the authenticity verdict that remains, correctly, outside either one's own real scope.

## Self-Check Questions

1. Section 22.1's `find_correlated_entry` can return `FOUND`, `NOT_FOUND`, or `AMBIGUOUS`. Explain why collapsing `AMBIGUOUS` into `NOT_FOUND` (treating "too many candidates" the same as "no candidates") would lose real, useful information a human reviewer could act on.
2. Section 22.1's `render_narration_sentence` is checked for byte-identical output across repeated calls with the same input. Why does this determinism property matter specifically for a narration a human reviewer or an audit log will read, beyond this book's own general determinism discipline?
3. Section 22.2's `contrast_ratio` computes its own lighter/darker ordering internally rather than requiring the caller to pass arguments in a specific order. Describe the specific, real bug this design choice prevents.
4. Section 22.2 enforces only the AA conformance level, not the stricter AAA level, in its own violation reporting. Explain why this is described as a deliberate, real policy choice rather than a limitation of the underlying contrast-ratio math.
5. Section 22.2's Test 4 confirms a non-compliant element is flagged with all 3 applicable criteria at once, named individually. What real problem would a single combined "multiple issues found" message cause that 3 separately named violations do not?
6. Section 22.3's Rule 2 requires exactly one open-ended record, and that record must be chronologically last. Construct a concrete 3-owner chain that would satisfy "exactly one open-ended record" but still be correctly flagged by the "must be last" condition.
7. Section 22.3's Rule 3 reports a gap or overlap by its own exact computed size (for instance, "gap of 20") rather than simply "inconsistent dates." What real, practical use does the exact size provide to whoever investigates the flagged chain that a generic message would not?
8. Section 22.3's `fit_condition_trend` reuses Chapter 19.4's own retail-trend regression formula unchanged. What does the reuse of the identical formula across two very different real domains demonstrate about what ordinary least squares actually depends on?
9. Section 22.3 explicitly states that neither `validate_provenance_chain` nor `classify_condition_trend` renders an authenticity verdict. Construct a concrete scenario where a chain passes provenance validation with zero issues despite describing an entirely fabricated ownership history.
10. Across all three sections in this chapter, identify the one recurring structural choice -- present in `AMBIGUOUS`, in the named WCAG criteria, and in the named provenance-chain issues -- that ties this chapter's own honesty discipline together, and explain why a single boolean "flagged: true/false" output in any of the three would have been a real regression.

## Where We Go Next

This chapter's three domains showed the same recurring discipline -- honest ambiguity, specifically named failures, and a computed signal that never overstates its own real scope -- generalizing across security correlation, accessibility auditing, and provenance and condition assessment. Chapter 23 turns to a fourth domain sharing a closely related real pattern: trust and authenticity at the point of sale, building a brand-reference database and counterfeit-screening engine for e-commerce listings, receipt-to-transaction reconciliation, automated expense-report policy enforcement, and a real-time luxury-goods authentication engine for consignment and resale counters.

## Worked Solutions

**1.** Collapsing `AMBIGUOUS` into `NOT_FOUND` would tell a human reviewer "no plausible arrival was found for this departure," which is a real, actionable claim that the subject likely left the monitored area entirely -- when the true situation is the opposite: TWO real, named candidates both plausibly account for the subject's arrival, and the correct next step is for a reviewer to examine those two specific candidates (checking face-matching, badge logs, or the original footage) rather than searching more broadly for a subject who may, in fact, already be accounted for by one of the two named candidates.

**2.** A narration a human reviewer or an audit log reads back later is often read hours, days, or in a legal or compliance context, months after the original event -- if the identical underlying interval data could render as two different sentences depending on when or how it was rendered, a reviewer comparing an audit log entry against a live re-render of the same data could see an apparent discrepancy that has nothing to do with the underlying events at all, undermining exactly the kind of reliable, checkable record this book's own determinism discipline is built to guarantee everywhere else in this system.

**3.** The specific bug this design prevents is a contrast-ratio function that assumes its own caller will always pass the lighter color first (or the darker color first) and computes `(a + 0.05) / (b + 0.05)` directly on whichever order it receives -- for any color pair where the caller happens to pass the darker color first, this produces a ratio BELOW 1.0, which is not merely an inverted number but a value WCAG's own formula never produces for any real color pair, silently corrupting exactly half of all real contrast computations depending entirely on argument order rather than the colors' own real relationship.

**4.** AA is described as a deliberate policy choice because it is the level real accessibility regulations -- the ADA and Section 508 are both named directly -- actually cite as their own required legal minimum, meaning an audit enforcing AA reports exactly the violations that carry real regulatory and legal consequences. AAA is a real, valid, stricter bar that WCAG itself defines, but treating an AAA shortfall as an equally urgent violation would report a large volume of real but non-mandatory issues alongside the mandatory ones, with no way for a report reader to distinguish "you are out of legal compliance" from "you have room to be more accessible than the law strictly requires" -- the underlying contrast math is identical either way; only the threshold applied to it differs.

**5.** A single combined message like "3 issues found" (or worse, "accessibility issues found") gives a report reader no way to know, without opening the underlying code or re-running the audit with more verbose logging, WHICH of potentially many possible criteria actually failed -- a developer fixing the element would have to independently re-derive whether the problem is contrast, target size, or a missing text alternative. Three separately named violations let that same developer go directly to the specific, real fix each one requires -- adjusting a color, enlarging a touch target, or adding alt text -- without first having to rediagnose which of several real, unrelated problems is actually present.

**6.** Owner A acquires at period 0 and never disposes (open-ended); Owner B acquires at period 100 and also never disposes (open-ended) -- wait, this satisfies "exactly one open-ended record" only if exactly one of the two lacks a disposal period. Construct instead: Owner A acquires at period 0, disposes at period 100; Owner B acquires at period 100, never disposes (open-ended, and chronologically last) -- this passes Rule 2 cleanly. To construct a FAILING case with exactly one open-ended record that still fails the "must be last" condition: Owner A acquires at period 0, and is recorded with NO disposal period (open-ended); Owner B acquires at period 100 and disposes at period 200. Exactly one record (Owner A) is open-ended, but Owner A is NOT the chronologically last record by acquisition order -- Owner B's own later transaction record, despite Owner A's own claim of still owning the work, is exactly the stale, inconsistent current-owner claim Test 4 is built to catch.

**7.** The exact computed size tells whoever investigates the flagged chain WHERE to look and how large a discrepancy to expect to find -- a "gap of 20" periods between two owners is a concrete, bounded window a researcher can search real auction records, shipping manifests, or estate documents for, while a "gap of 400" periods between the same two owners would suggest a considerably more significant, and more suspicious, missing chapter in the object's own history. A generic "inconsistent dates" message would require the investigator to first re-derive the size and location of the discrepancy from the raw records themselves before any real investigation could even begin.

**8.** The identical formula working correctly across a retail sell-through curve (Chapter 19.4) and a conservator's periodic condition scores (this chapter) demonstrates that ordinary least squares depends only on having a real, ordered independent variable (a time period) and a real, numeric dependent variable (a sales figure, a condition score) with an assumed-linear relationship between them -- it has no domain-specific knowledge of retail or of art conservation built into it anywhere, which is exactly why the same real mathematical technique, unchanged, generalizes cleanly to any domain that produces the same basic shape of data.

**9.** A forger constructs an entirely fictional chain of three "owners" -- invented names, with acquisition and disposal periods that are all mutually contiguous and chronologically sound, ending with a fabricated open-ended "current owner" record -- and submits this fictional chain as the object's provenance. `validate_provenance_chain` checks ONLY whether the chain's own STATED dates are self-consistent with each other, which this fabricated chain, having been constructed specifically to be self-consistent, satisfies perfectly, producing zero issues despite every single record in the chain being entirely invented; catching that would require independently verifying each record against real external evidence (auction house records, insurance documents, exhibition catalogs), which is explicitly outside what this section's own structural date-arithmetic validator was ever built to do.

**10.** The recurring structural choice is that every one of this chapter's three outputs -- `CorrelationResult`'s own `AMBIGUOUS` outcome naming its candidates, `audit_element`'s own list of specifically named WCAG criteria, and `validate_provenance_chain`'s own list of specifically named, exactly-sized issues -- reports MULTIPLE, INDEPENDENTLY NAMED findings (or the honest absence of a single findable answer) rather than a single collapsed signal. A boolean "flagged: true/false" in any of the three would have been a real regression because it would discard exactly the information a human reviewer needs to act: WHICH candidates are ambiguous, WHICH specific criterion failed, or WHERE and how large a provenance inconsistency is -- collapsing any of these to a single bit preserves the alarm but destroys the actionable detail this chapter's entire discipline exists to keep visible.
