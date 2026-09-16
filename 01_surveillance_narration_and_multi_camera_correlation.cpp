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
