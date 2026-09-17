// Chapter 27.4 -- Real numerical debugging tools for catching the bugs that
// only appear at serving scale. Part 1 builds a real NaN/Inf-propagation
// tracer: given a real sequence of per-layer activation snapshots, it finds
// the layer where corruption FIRST appeared, not merely the layer where it
// happens to still be visible -- because once a real NaN exists, ordinary
// arithmetic propagates it forward through every downstream layer
// automatically, making "where is a NaN visible" a much weaker question
// than "where did it start." Part 2 builds a real floating-point
// drift-detection tool: real float32 addition is not associative, so
// continuous batching's own real, traffic-dependent regrouping of sequences
// into different batches across different runs can produce genuinely
// different (though both individually valid) reduction results -- this
// section quantifies that real drift directly rather than assuming it away.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_nan_propagation_tracing_and_fp_drift_detection.cpp -o 04_nan_propagation_tracing_and_fp_drift_detection
// Run:     ./04_nan_propagation_tracing_and_fp_drift_detection

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }
bool nearf(float a, float b, float eps = 1e-6f) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: real NaN/Inf-propagation tracing. Given a real sequence of
// per-layer activation snapshots (one vector<double> per layer, in real
// forward-pass order), find the FIRST layer at which any element is
// non-finite -- the real point of origin, which is what an engineer
// actually needs to know to fix the bug, as opposed to merely knowing that
// a NaN exists somewhere downstream.
// =======================================================================
bool has_non_finite(const std::vector<double>& layer) {
    for (double v : layer) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

std::optional<int> first_nan_layer(const std::vector<std::vector<double>>& layers) {
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (has_non_finite(layers[i])) return static_cast<int>(i);
    }
    return std::nullopt;
}

std::optional<std::size_t> first_non_finite_index(const std::vector<double>& layer) {
    for (std::size_t i = 0; i < layer.size(); ++i) {
        if (std::isnan(layer[i]) || std::isinf(layer[i])) return i;
    }
    return std::nullopt;
}

// A real, minimal simulation of how a NaN actually originates and then
// propagates: layer 2's own per-example normalization divides by
// (max_abs - max_abs), a genuine 0/0 whenever every element in that example
// shares the identical max_abs -- producing one real NaN at that layer.
// Layers 3 and 4 are then computed by ordinary elementwise arithmetic on
// the PREVIOUS layer's own output, exactly how a real forward pass would --
// so the NaN propagates forward automatically, without being reintroduced.
std::vector<std::vector<double>> build_realistic_propagation_trace() {
    std::vector<std::vector<double>> layers;
    layers.push_back({1.0, 2.0, 3.0});                 // layer 0: clean
    layers.push_back({2.0, 3.0, 4.0});                 // layer 1: clean

    // layer 2: a broken per-example normalization -- divide each element by
    // (max_abs - max_abs), which is genuinely 0.0 here since every element
    // in this toy example equals the same max_abs of 4.0.
    double max_abs = 4.0;
    double denom = max_abs - max_abs;  // == 0.0, a real division-by-zero setup
    std::vector<double> layer2;
    for (double v : layers[1]) layer2.push_back(v / denom);  // 2/0=inf, 3/0=inf, 4/0=inf -- but element 0 is special below
    layer2[0] = 0.0 / denom;  // 0.0 / 0.0 is a genuine NaN, not +-inf
    layers.push_back(layer2);

    // layer 3: ordinary elementwise arithmetic on layer 2's own output --
    // arithmetic with a NaN operand produces a NaN automatically, and
    // arithmetic with a real +-inf operand stays non-finite too.
    std::vector<double> layer3;
    for (double v : layers[2]) layer3.push_back(v * 0.5 + 1.0);
    layers.push_back(layer3);

    // layer 4: the same real propagation continues one layer further.
    std::vector<double> layer4;
    for (double v : layers[3]) layer4.push_back(v - 1.0);
    layers.push_back(layer4);

    return layers;
}

// =======================================================================
// PART 2: real floating-point drift detection via reduction order. Real
// float32 addition is not associative -- summing the identical real values
// in a different order can produce a genuinely different result, because
// each individual addition rounds to the nearest representable float32.
// This matters directly for continuous batching: which sequences a given
// real step groups together (Section 27.3) can change which order a
// reduction (a sum over a batch dimension, say) is actually computed in
// across different runs of the identical logical computation.
// =======================================================================
float sum_huge_first(float huge, const std::vector<float>& small_values) {
    float s = huge;
    for (float v : small_values) s = s + v;
    return s;
}

float sum_small_first_then_huge(float huge, const std::vector<float>& small_values) {
    float s = 0.0f;
    for (float v : small_values) s = s + v;
    return s + huge;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 27.4: Numerical Debugging Tools -- NaN-Propagation Tracing and FP Drift Detection\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a real 5-layer forward-pass trace in which corruption genuinely ORIGINATES at "
                 "layer 2 (a real division-by-zero in a broken normalization step) and then propagates "
                 "forward, unmodified in cause, through layers 3 and 4 purely via ordinary arithmetic -- "
                 "first_nan_layer correctly reports the real ORIGIN, layer 2, not merely one of the 3 "
                 "layers where corruption happens to still be visible --\n";
    {
        auto layers = build_realistic_propagation_trace();
        CHECK(layers.size() == 5);
        CHECK(!has_non_finite(layers[0]));
        CHECK(!has_non_finite(layers[1]));
        CHECK(has_non_finite(layers[2]));
        CHECK(has_non_finite(layers[3]));
        CHECK(has_non_finite(layers[4]));

        auto origin = first_nan_layer(layers);
        CHECK(origin.has_value());
        CHECK(*origin == 2);

        auto bad_index = first_non_finite_index(layers[2]);
        CHECK(bad_index.has_value());
        CHECK(*bad_index == 0);
        CHECK(std::isnan(layers[2][0]));
        CHECK(std::isinf(layers[2][1]));
        CHECK(std::isinf(layers[2][2]));

        std::cout << "  layers 0 and 1 are fully finite; layer 2's own broken normalization produces a "
                     "genuine NaN at index 0 (0.0 / 0.0) and genuine +inf at indices 1 and 2 (nonzero / "
                     "0.0); layers 3 and 4 are both non-finite too, purely because ordinary elementwise "
                     "arithmetic on a non-finite input stays non-finite -- first_nan_layer correctly "
                     "reports layer 2 as the real point of origin\n";
    }

    std::cout << "\n-- Test 2: checking ONLY the final layer for non-finite values -- the naive approach a "
                 "system with no real tracing tool might fall back on -- correctly detects THAT a real "
                 "problem exists, but provides none of first_nan_layer's own actionable information about "
                 "WHERE it actually started, which is the entire real point of building a tracer at all --\n";
    {
        auto layers = build_realistic_propagation_trace();
        bool final_layer_broken = has_non_finite(layers.back());
        CHECK(final_layer_broken);

        // The naive "check the last layer" approach cannot distinguish this real trace, where
        // corruption started at layer 2, from one where it started at layer 3 or layer 4 instead --
        // both would show an identical "yes, the final layer is broken" verdict.
        std::vector<std::vector<double>> alternate_trace = {
            {1.0, 2.0, 3.0},
            {2.0, 3.0, 4.0},
            {1.0, 1.0, 1.0},                       // layer 2 clean this time
            {std::nan(""), 1.0, 1.0},               // corruption instead originates at layer 3
            {std::nan(""), 1.0, 1.0},               // and simply propagates to layer 4
        };
        CHECK(has_non_finite(alternate_trace.back()) == final_layer_broken);  // identical naive verdict
        CHECK(*first_nan_layer(alternate_trace) == 3);                        // but a genuinely different real origin
        std::cout << "  both this section's own real trace (origin at layer 2) and a genuinely different "
                     "alternate trace (origin at layer 3) show an IDENTICAL \"yes\" verdict under a naive "
                     "final-layer-only check -- first_nan_layer is what actually distinguishes them, "
                     "correctly reporting layer 2 for the first trace and layer 3 for the second\n";
    }

    std::cout << "\n-- Test 3: real float32 reduction order genuinely changes the result -- summing a huge "
                 "value first, then ten real small values, loses all 10 of them to rounding, while summing "
                 "the identical 10 small values together FIRST, then adding the huge value, preserves most "
                 "of their real contribution -- a real, measurable, nonzero 8.0-unit drift between two "
                 "equally valid reduction orders over the identical real input values --\n";
    {
        float huge = 1e8f;
        std::vector<float> small_values(10, 1.0f);

        float order_a = sum_huge_first(huge, small_values);
        float order_b = sum_small_first_then_huge(huge, small_values);

        CHECK(nearf(order_a, 100000000.0f));
        CHECK(nearf(order_b, 100000008.0f));

        float drift = order_a - order_b;
        CHECK(nearf(drift, -8.0f));
        CHECK(!nearf(order_a, order_b));  // genuinely, measurably different -- not merely a rounding artifact

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  order A (huge first, then each 1.0f added sequentially) rounds to exactly "
                  << order_a << ", silently losing all 10 real additions -- each individual 1.0f is too "
                     "small to move a float32 near 1e8 to the next representable value; order B (the 10 "
                     "small values summed together first, THEN added to the huge value) rounds to "
                  << order_b << " instead, preserving most of their real combined contribution; the real "
                     "drift between these two equally valid reduction orders is exactly " << drift << "\n";
        std::cout << std::defaultfloat << std::setprecision(6);
    }

    std::cout << "\n-- Test 4: comparing both real reduction orders against the true double-precision value "
                 "confirms neither float32 order is exactly correct, but summing small values first is "
                 "genuinely, measurably closer to the truth than summing the huge value first -- a real, "
                 "actionable finding for a serving system whose own continuous-batching schedule changes "
                 "which order a batched reduction is actually computed in from run to run --\n";
    {
        float huge = 1e8f;
        std::vector<float> small_values(10, 1.0f);
        double true_value = 1e8 + 10.0;

        float order_a = sum_huge_first(huge, small_values);
        float order_b = sum_small_first_then_huge(huge, small_values);

        double error_a = std::fabs(static_cast<double>(order_a) - true_value);
        double error_b = std::fabs(static_cast<double>(order_b) - true_value);

        CHECK(near(error_a, 10.0));
        CHECK(near(error_b, 2.0));
        CHECK(error_b < error_a);

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  against the true double-precision value of " << true_value << ", order A's own "
                     "real error is " << error_a << " while order B's own real error is only " << error_b
                  << " -- summing small values together before combining them with a much larger value is "
                     "a real, general strategy for reducing accumulated floating-point error, not merely "
                     "an artifact of this section's own specific chosen numbers\n";
        std::cout << std::defaultfloat << std::setprecision(6);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
