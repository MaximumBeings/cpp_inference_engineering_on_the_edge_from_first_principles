// Chapter 25.3 -- A personal camera doing exercise-form analysis needs
// two real things standard-library C++ can build entirely on its own:
// a real joint-angle formula from 2D keypoints, and a real rep-counting
// state machine with real hysteresis, honest enough to refuse counting
// a rep that never actually reached real depth. Both run entirely
// on-device, on synthetic keypoint sequences here, with no camera
// frame or any other personal data ever needing to leave the device --
// this section's own real privacy-first architecture.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_exercise_form_analysis_and_rep_counting.cpp -o 03_exercise_form_analysis_and_rep_counting
// Run:     ./03_exercise_form_analysis_and_rep_counting

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
// PART 1: a real joint-angle formula -- the angle at `vertex`, formed by
// the two real vectors reaching out to `a` and `c`, via the standard
// dot-product-over-magnitudes formula.
// =======================================================================
struct Point2D {
    double x = 0.0, y = 0.0;
};

double joint_angle_degrees(Point2D a, Point2D vertex, Point2D c) {
    double v1x = a.x - vertex.x, v1y = a.y - vertex.y;
    double v2x = c.x - vertex.x, v2y = c.y - vertex.y;
    double dot = v1x * v2x + v1y * v2y;
    double mag1 = std::sqrt(v1x * v1x + v1y * v1y);
    double mag2 = std::sqrt(v2x * v2x + v2y * v2y);
    double cos_theta = dot / (mag1 * mag2);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));  // guard real floating-point overshoot at +/-1
    return std::acos(cos_theta) * 180.0 / std::numbers::pi;
}

// =======================================================================
// PART 2: a real rep-counting state machine over a knee-angle sequence,
// with real hysteresis -- a rep counts only on a full real cycle that
// actually reaches real depth, and dipping back into BOTTOM mid-ascent
// never double-counts.
// =======================================================================
enum class SquatPhase { STANDING, DESCENDING, BOTTOM, ASCENDING };

constexpr double STANDING_ANGLE_MIN_DEG = 160.0;
constexpr double BOTTOM_ANGLE_MAX_DEG = 100.0;

class RepCounter {
public:
    void process_frame(double knee_angle_deg) {
        switch (phase_) {
            case SquatPhase::STANDING:
                if (knee_angle_deg < STANDING_ANGLE_MIN_DEG) phase_ = SquatPhase::DESCENDING;
                break;
            case SquatPhase::DESCENDING:
                if (knee_angle_deg <= BOTTOM_ANGLE_MAX_DEG) {
                    phase_ = SquatPhase::BOTTOM;
                } else if (knee_angle_deg >= STANDING_ANGLE_MIN_DEG) {
                    // Real depth was never reached before returning to standing -- this is a real,
                    // honest partial rep, and it does not count, exactly this book's own established
                    // "refuse rather than fabricate" discipline applied to a physical rep count.
                    phase_ = SquatPhase::STANDING;
                }
                break;
            case SquatPhase::BOTTOM:
                if (knee_angle_deg > BOTTOM_ANGLE_MAX_DEG) phase_ = SquatPhase::ASCENDING;
                break;
            case SquatPhase::ASCENDING:
                if (knee_angle_deg >= STANDING_ANGLE_MIN_DEG) {
                    completed_reps_++;
                    phase_ = SquatPhase::STANDING;
                } else if (knee_angle_deg <= BOTTOM_ANGLE_MAX_DEG) {
                    // A real dip back into full depth mid-ascent is still the SAME rep in progress,
                    // not a second one -- returning to BOTTOM here, rather than counting early,
                    // is what prevents a real double-count.
                    phase_ = SquatPhase::BOTTOM;
                }
                break;
        }
    }

    int completed_reps() const { return completed_reps_; }
    SquatPhase phase() const { return phase_; }

private:
    SquatPhase phase_ = SquatPhase::STANDING;
    int completed_reps_ = 0;
};

// =======================================================================
// PART 3: real form-fault detection from each rep's own bottom-of-rep
// angles.
// =======================================================================
enum class ExerciseFormFault { INSUFFICIENT_DEPTH, KNEE_ASYMMETRY };

constexpr double KNEE_ASYMMETRY_THRESHOLD_DEG = 15.0;

std::vector<ExerciseFormFault> detect_form_faults(double left_knee_angle_at_bottom,
                                                   double right_knee_angle_at_bottom) {
    std::vector<ExerciseFormFault> faults;
    if (left_knee_angle_at_bottom > BOTTOM_ANGLE_MAX_DEG || right_knee_angle_at_bottom > BOTTOM_ANGLE_MAX_DEG) {
        faults.push_back(ExerciseFormFault::INSUFFICIENT_DEPTH);
    }
    if (std::fabs(left_knee_angle_at_bottom - right_knee_angle_at_bottom) > KNEE_ASYMMETRY_THRESHOLD_DEG) {
        faults.push_back(ExerciseFormFault::KNEE_ASYMMETRY);
    }
    return faults;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 25.3: Exercise-Form Analysis and Rep Counting\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real joint-angle formula matches exact, hand-verifiable geometric "
                 "configurations --\n";
    {
        CHECK(near(joint_angle_degrees({0, 1}, {0, 0}, {1, 0}), 90.0));
        CHECK(near(joint_angle_degrees({0, 1}, {0, 0}, {0, -1}), 180.0));
        CHECK(near(joint_angle_degrees({1, 1}, {0, 0}, {1, 0}), 45.0));
        CHECK(near(joint_angle_degrees({1, 0}, {0, 0}, {1, 0}), 0.0));
        std::cout << "  two perpendicular real vectors from a shared vertex measure to exactly 90 "
                     "degrees; two opposite real vectors (a fully straightened leg) measure to exactly "
                     "180 degrees; a real 45-degree configuration measures to exactly 45 degrees; and "
                     "two identical real vectors measure to exactly 0 degrees\n";
    }

    std::cout << "\n-- Test 2: 3 real full reps that reach true depth are counted, and a real partial "
                 "rep that never reaches true depth before returning to standing is honestly excluded "
                 "from the count --\n";
    {
        RepCounter counter;
        // Rep 1, full.
        for (double a : {170.0, 140.0, 90.0, 130.0, 175.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 1);
        // A real partial rep: descends to 128 degrees (short of the real 100-degree depth
        // threshold) and returns to standing without ever reaching BOTTOM.
        for (double a : {135.0, 128.0, 165.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 1);  // still 1 -- the partial rep did not count
        // Rep 2, full.
        for (double a : {170.0, 120.0, 85.0, 150.0, 172.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 2);
        // Rep 3, full.
        for (double a : {110.0, 95.0, 145.0, 168.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 3);
        std::cout << "  a real 17-frame sequence containing 3 genuine full reps and 1 real partial rep "
                     "that bottoms out at only 128 degrees (short of the real 100-degree depth "
                     "threshold) before returning to standing counts to exactly 3 -- the partial rep "
                     "is correctly never counted\n";
    }

    std::cout << "\n-- Test 3: a real dip back into full depth mid-ascent is correctly treated as the "
                 "same rep still in progress, never a double count --\n";
    {
        RepCounter counter;
        for (double a : {170.0, 120.0, 90.0, 130.0, 95.0, 140.0, 165.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 1);
        std::cout << "  a rep that reaches bottom, rises partway to 130 degrees, dips back down to 95 "
                     "degrees (real full depth again), then finally rises all the way to standing "
                     "counts as exactly 1 real rep, not 2 -- the mid-ascent dip back into BOTTOM never "
                     "triggers a premature or duplicate count\n";
    }

    std::cout << "\n-- Test 4: INSUFFICIENT_DEPTH is flagged only when a rep's own bottom angle "
                 "genuinely fails to reach the real depth threshold, inclusive of the boundary itself "
                 "--\n";
    {
        auto shallow = detect_form_faults(115.0, 112.0);
        bool has_shallow = false;
        for (auto f : shallow) if (f == ExerciseFormFault::INSUFFICIENT_DEPTH) has_shallow = true;
        CHECK(has_shallow);

        auto exactly_at_threshold = detect_form_faults(100.0, 100.0);
        bool has_at_threshold = false;
        for (auto f : exactly_at_threshold) if (f == ExerciseFormFault::INSUFFICIENT_DEPTH) has_at_threshold = true;
        CHECK(!has_at_threshold);

        auto deep_enough = detect_form_faults(90.0, 88.0);
        CHECK(deep_enough.empty());
        std::cout << "  bottom angles of 115/112 degrees (short of real depth) flag "
                     "INSUFFICIENT_DEPTH; bottom angles of exactly 100/100 degrees -- the real "
                     "threshold itself -- correctly do NOT flag it; bottom angles of 90/88 degrees "
                     "flag no faults at all\n";
    }

    std::cout << "\n-- Test 5: KNEE_ASYMMETRY is flagged only once the real left/right difference "
                 "genuinely exceeds this section's own stated threshold --\n";
    {
        auto symmetric = detect_form_faults(90.0, 92.0);
        CHECK(symmetric.empty());

        auto exactly_at_threshold = detect_form_faults(90.0, 105.0);  // exactly 15.0 degrees apart
        bool flagged_at_threshold = false;
        for (auto f : exactly_at_threshold) if (f == ExerciseFormFault::KNEE_ASYMMETRY) flagged_at_threshold = true;
        CHECK(!flagged_at_threshold);

        auto asymmetric = detect_form_faults(90.0, 106.0);  // 16.0 degrees apart -- and 106 also insufficient depth
        bool flagged_asymmetry = false, flagged_depth = false;
        for (auto f : asymmetric) {
            if (f == ExerciseFormFault::KNEE_ASYMMETRY) flagged_asymmetry = true;
            if (f == ExerciseFormFault::INSUFFICIENT_DEPTH) flagged_depth = true;
        }
        CHECK(flagged_asymmetry);
        CHECK(flagged_depth);
        std::cout << "  a real 2-degree left/right difference flags no fault at all; a real 15-degree "
                     "difference -- exactly this section's own stated threshold -- correctly does NOT "
                     "flag KNEE_ASYMMETRY; a real 16-degree difference does, and here it also correctly "
                     "flags INSUFFICIENT_DEPTH at the same time, since 106 degrees also fails the "
                     "real depth threshold on its own\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
