// Appendix D.3 -- Latency-Budget Arithmetic.
//
// This file reuses two real formulas this book already derived and
// verified, unchanged, and combines them for the first time against a
// REAL, officially cited hardware number:
//
//   - Chapter 4.1's own memory-wall formula:
//       tokens_per_sec = bandwidth / bytes_read_per_token
//     Chapter 4.1's own worked table reported "14.2 tok/s" for a Q4_0
//     Llama-3-8B-shaped model (~4.5 GB) at a stated illustrative 64 GB/s.
//     Test 1 below reproduces that same rounded figure from the same
//     formula and the same approximate inputs.
//
//   - Chapter 27.2's own deadline-enforcement pattern: accumulate named
//     stage costs in order, and refuse the instant the running total
//     strictly exceeds budget, naming the exact stage responsible rather
//     than merely reporting a final over-budget total.
//
// Test 2 combines the first formula with Appendix D.1's own real,
// verified Raspberry Pi 5 bandwidth (17.068 GB/s, not a stated
// illustrative number) against that same ~4.5 GB model size, producing a
// genuinely new number this book has not computed before: a real,
// concrete tokens-per-second estimate for a real, named, cheap edge
// board, grounded in nothing but formulas and figures this book already
// established and cited.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 d3_latency_budget_calculator.cpp -o d3_latency_budget_calculator
// Run:     ./d3_latency_budget_calculator

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

// Chapter 4.1's own formula, reused verbatim.
double tokens_per_sec(double bandwidth_bytes_per_sec, double bytes_read_per_token) {
    return bandwidth_bytes_per_sec / bytes_read_per_token;
}

// Chapter 27.2's own deadline-enforcement pattern, generalized to any
// named sequence of stage costs against any latency budget.
struct Stage {
    std::string name;
    double cost_us;
};

struct BudgetCheckResult {
    bool fits;
    double total_us;
    std::string first_stage_over_budget;  // empty if it fits
};

BudgetCheckResult check_latency_budget(const std::vector<Stage>& stages, double budget_us) {
    double total = 0.0;
    for (const auto& s : stages) {
        total += s.cost_us;
        if (total > budget_us) {  // strict >, Chapter 27.2's own boundary convention
            return {false, total, s.name};
        }
    }
    return {true, total, ""};
}

int main() {
    std::cout << "===================================================\n";
    std::cout << "Appendix D.3: Latency-Budget Arithmetic\n";
    std::cout << "===================================================\n\n";

    // -- Test 1: Chapter 4.1's own worked example reproduced -- a stated,
    // illustrative 64 GB/s against a ~4.5 GB Q4_0 model rounds to the
    // same 14.2 tok/s the chapter itself reported. --
    {
        double bandwidth = 64e9;          // Chapter 4.1's own stated illustrative bandwidth
        double model_bytes = 4.5e9;       // Chapter 4.1's own approximate Q4_0 model-size figure
        double tps = tokens_per_sec(bandwidth, model_bytes);
        double rounded = std::round(tps * 10.0) / 10.0;
        std::cout << "-- Test 1: 64 GB/s (Chapter 4.1's stated figure), 4.5 GB Q4_0 model -- "
                  << tps << " tok/s (rounds to " << rounded << ") --\n";
        CHECK(rounded == 14.2);
    }

    // -- Test 2: the SAME formula and SAME approximate model size, now
    // against Appendix D.1's own real, verified Raspberry Pi 5 bandwidth
    // (17.068 GB/s) rather than a stated illustrative number -- a
    // genuinely new, real, concrete estimate for a real, named board. --
    {
        double bandwidth = 17.068e9;      // Appendix D.1's own verified Raspberry Pi 5 figure
        double model_bytes = 4.5e9;
        double tps = tokens_per_sec(bandwidth, model_bytes);
        std::cout << "-- Test 2: 17.068 GB/s (Appendix D.1's real Raspberry Pi 5 figure), "
                  << "4.5 GB Q4_0 model -- " << tps << " tok/s --\n";
        CHECK(tps > 3.7 && tps < 3.85);
    }

    // -- Test 3: check_latency_budget on a real, comfortably-fitting
    // sequence of named stages -- reports true and the exact total. --
    {
        std::vector<Stage> stages = {
            {"tokenize", 40.0}, {"embed_lookup", 15.0}, {"forward_pass", 300.0}, {"sample", 5.0},
        };
        auto r = check_latency_budget(stages, 500.0);
        std::cout << "-- Test 3: stages totaling 360us against a 500us budget -- fits=" << r.fits
                  << ", total=" << r.total_us << "us --\n";
        CHECK(r.fits);
        CHECK(r.total_us == 360.0);
        CHECK(r.first_stage_over_budget.empty());
    }

    // -- Test 4: the same stages against a tighter budget -- refuses at
    // the exact stage responsible, not merely reporting the final total. --
    {
        std::vector<Stage> stages = {
            {"tokenize", 40.0}, {"embed_lookup", 15.0}, {"forward_pass", 300.0}, {"sample", 5.0},
        };
        auto r = check_latency_budget(stages, 340.0);
        std::cout << "-- Test 4: same stages against a 340us budget -- fits=" << r.fits
                  << ", breaches at=\"" << r.first_stage_over_budget << "\", total_at_breach="
                  << r.total_us << "us --\n";
        CHECK(!r.fits);
        CHECK(r.first_stage_over_budget == "forward_pass");
        CHECK(r.total_us == 355.0);
    }

    // -- Test 5: the boundary itself -- a total exactly equal to budget
    // fits (strict > only, never >=), the same convention Chapter 27.2
    // established and tested explicitly. --
    {
        std::vector<Stage> stages = {{"only_stage", 100.0}};
        auto at_budget = check_latency_budget(stages, 100.0);
        auto one_over = check_latency_budget(stages, 99.999999);
        std::cout << "-- Test 5: exactly-at-budget fits=" << at_budget.fits
                  << ", one-unit-under-budget fits=" << one_over.fits << " --\n";
        CHECK(at_budget.fits);
        CHECK(!one_over.fits);
    }

    // -- Test 6: tying Test 2's real Raspberry Pi 5 estimate to a stated
    // deployment SLA -- does 3.79ish tok/s clear a real "at least 3
    // tokens/sec" target? --
    {
        double bandwidth = 17.068e9;
        double model_bytes = 4.5e9;
        double tps = tokens_per_sec(bandwidth, model_bytes);
        double target_tps = 3.0;
        std::cout << "-- Test 6: real Raspberry Pi 5 estimate (" << tps << " tok/s) vs. a "
                  << target_tps << " tok/s target -- meets target: " << (tps >= target_tps ? "YES" : "NO") << " --\n";
        CHECK(tps >= target_tps);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
