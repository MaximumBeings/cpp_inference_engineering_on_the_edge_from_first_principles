// Chapter 26.2 -- Softmax turns raw logits into real probabilities, and
// its own textbook definition, exp(x_i) / sum(exp(x_j)), is exactly
// correct mathematically and dangerously wrong to implement literally:
// a single large real logit overflows a real floating-point exponential
// long before the actual probability computation is finished. This
// section reproduces that real failure on purpose, derives the real
// algebraic identity that fixes it (softmax is shift-invariant), and
// shows the identical bug reappears -- in an even more dangerous, silent
// form -- in the closely related log-sum-exp computation real LLM
// log-likelihood scoring depends on.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_numerically_stable_softmax_and_log_sum_exp.cpp -o 02_numerically_stable_softmax_and_log_sum_exp
// Run:     ./02_numerically_stable_softmax_and_log_sum_exp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: naive softmax, implemented in real 32-bit float -- the
// precision real production kernels actually run -- and left exactly as
// literally as the textbook formula reads.
// =======================================================================
std::vector<float> naive_softmax_f32(const std::vector<float>& x) {
    std::vector<float> exps(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) {
        exps[i] = std::exp(x[i]);
        sum += exps[i];
    }
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); i++) out[i] = exps[i] / sum;
    return out;
}

// =======================================================================
// PART 2: the real, shift-invariant fix. softmax(x) is mathematically
// identical to softmax(x - c) for ANY real constant c, because the
// constant factor exp(-c) introduced in every term of both the
// numerator and the denominator cancels exactly. Choosing c = max(x)
// guarantees the largest real shifted value is exactly 0 (exp(0) = 1,
// no overflow), and every other shifted value is <= 0 (exp of a
// non-positive number can only underflow toward 0, never overflow).
// =======================================================================
std::vector<float> stable_softmax_f32(const std::vector<float>& x) {
    float max_x = *std::max_element(x.begin(), x.end());
    std::vector<float> exps(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) {
        exps[i] = std::exp(x[i] - max_x);
        sum += exps[i];
    }
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); i++) out[i] = exps[i] / sum;
    return out;
}

// =======================================================================
// PART 3: the identical real bug, in the closely related log-sum-exp
// computation -- and here, in double precision, its own failure mode is
// even more dangerous: it never produces a NaN a caller might notice.
// It silently returns +infinity as though that were a real, valid
// log-probability.
// =======================================================================
double naive_log_sum_exp(const std::vector<double>& x) {
    double sum = 0.0;
    for (double v : x) sum += std::exp(v);
    return std::log(sum);
}

double stable_log_sum_exp(const std::vector<double>& x) {
    double max_x = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (double v : x) sum += std::exp(v - max_x);
    return max_x + std::log(sum);
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.2: Numerically Stable Softmax and Log-Sum-Exp\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a single real large logit makes naive_softmax_f32 produce a genuine NaN "
                 "for exactly the entry that should carry nearly all the real probability mass --\n";
    {
        std::vector<float> x = {1.0f, 2.0f, 100.0f};
        auto result = naive_softmax_f32(x);
        CHECK(std::isnan(result[2]));   // exp(100.0f) overflows float32 to +inf; inf/inf = NaN
        CHECK(!std::isnan(result[0]));  // the small entries underflow to a plain, if misleadingly exact, 0.0f
        CHECK(!std::isnan(result[1]));
        std::cout << "  naive_softmax_f32({1.0, 2.0, 100.0}) produces a genuine NaN for index 2 -- the "
                     "one entry that should hold almost the entire real probability mass -- because "
                     "exp(100.0f) overflows float32 to +inf, and inf/inf is undefined\n";
    }

    std::cout << "\n-- Test 2: stable_softmax_f32 on the identical real input produces a valid real "
                 "probability distribution -- no NaN anywhere, and the values sum to exactly 1.0 --\n";
    {
        std::vector<float> x = {1.0f, 2.0f, 100.0f};
        auto result = stable_softmax_f32(x);
        for (float v : result) CHECK(!std::isnan(v) && !std::isinf(v));
        CHECK(result[2] > 0.999f);  // the dominant real logit correctly claims nearly all the probability
        float sum = result[0] + result[1] + result[2];
        CHECK(near(static_cast<double>(sum), 1.0, 1e-5));
        std::cout << "  stable_softmax_f32 on the identical real input produces a fully valid "
                     "distribution: index 2 correctly carries over 99.9% of the real probability mass, "
                     "and all three values sum to exactly 1.0 -- the exact same real input naive_softmax "
                     "could not process at all\n";
    }

    std::cout << "\n-- Test 3: on a real, safe-magnitude input with no overflow risk, naive and stable "
                 "softmax agree -- confirming they really are the identical real mathematical function, "
                 "differing only in numerical robustness, never in what they compute --\n";
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        auto naive = naive_softmax_f32(x);
        auto stable = stable_softmax_f32(x);
        for (size_t i = 0; i < x.size(); i++) CHECK(near(static_cast<double>(naive[i]), static_cast<double>(stable[i]), 1e-5));
        double sum = naive[0] + naive[1] + naive[2];
        CHECK(near(sum, 1.0, 1e-5));
        std::cout << "  on the real, safe input {1.0, 2.0, 3.0} (well within float32's real dynamic "
                     "range), naive and stable softmax agree to within 1e-5 at every index -- the shift "
                     "by max(x) really does leave the mathematical result unchanged\n";
    }

    std::cout << "\n-- Test 4: naive_log_sum_exp on a real, safe input matches stable_log_sum_exp "
                 "exactly, confirming the identical shift-invariant identity in the closely related "
                 "log-sum-exp formulation --\n";
    {
        std::vector<double> x = {1.0, 2.0, 3.0};
        double naive = naive_log_sum_exp(x);
        double stable = stable_log_sum_exp(x);
        CHECK(near(naive, stable, 1e-9));
        std::cout << "  on {1.0, 2.0, 3.0}, naive_log_sum_exp and stable_log_sum_exp agree to within "
                     "1e-9 -- log(sum(exp(x))) and max(x) + log(sum(exp(x - max(x)))) really are the "
                     "identical real function\n";
    }

    std::cout << "\n-- Test 5: naive_log_sum_exp on a real large-magnitude input silently returns +inf "
                 "-- not a NaN, not a crash, just a plainly WRONG finite-valued answer masquerading as "
                 "a real number -- while stable_log_sum_exp on the identical input returns the correct, "
                 "genuinely finite real value --\n";
    {
        std::vector<double> x = {1000.0, 1.0, 2.0};
        double naive = naive_log_sum_exp(x);
        double stable = stable_log_sum_exp(x);
        CHECK(std::isinf(naive));
        CHECK(!std::isinf(stable) && !std::isnan(stable));
        CHECK(stable >= 1000.0 && stable < 1000.001);  // the true value is 1000 plus a vanishingly small correction
        std::cout << "  naive_log_sum_exp({1000.0, 1.0, 2.0}) silently returns +inf, because exp(1000.0) "
                     "overflows even double precision -- a caller who never separately checked would "
                     "treat +inf as a legitimate log-probability; stable_log_sum_exp on the identical "
                     "input correctly returns approximately 1000.0, the real, genuinely finite answer, "
                     "confirming the naive result was not merely imprecise but categorically wrong\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
