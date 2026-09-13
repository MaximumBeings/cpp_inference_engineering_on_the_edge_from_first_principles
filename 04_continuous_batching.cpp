// 04_continuous_batching.cpp
// Chapter 8, Part 4: continuous batching turns many independent decode
// requests' single-token GEMVs into one batched GEMM -- mathematically
// the EXACT SAME memory-bound-to-compute-bound transition Section 8.3
// analyzed for a single request's prefill, just triggered by stacking
// unrelated requests' one token each into the batch dimension instead of
// stacking one request's many prompt tokens. This file reuses Section
// 8.3's own cost formula unchanged (with "seq" reinterpreted as "batch
// size B") to find the batch-size crossover, then asks the question
// prefill's analysis didn't need to: what happens to PER-REQUEST latency,
// not just aggregate throughput, on both sides of that crossover.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_continuous_batching.cpp -o out04

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

// =========================================================================
// Section 8.3's cost formula, unchanged. There "seq" was the number of
// prompt-token columns sharing one weight read; here it is the number of
// DIFFERENT REQUESTS' single decode tokens sharing that same weight read
// -- the scheduler that stacks them is what "continuous batching" means.
// =========================================================================
struct Cost {
    long long flops = 0;
    long long bytes = 0;
    void flop(long long n = 1) { flops += n; }
    void access(long long n_bytes) { bytes += n_bytes; }
    double arithmetic_intensity() const { return static_cast<double>(flops) / static_cast<double>(bytes); }
};
Cost linear_fp32_cost(long long n_out, long long n_in, long long batch) {
    Cost cost;
    cost.access(n_out * n_in * 4);     // W, read once, shared across the whole batch
    cost.access(n_in * batch * 4);     // X, one column per request in the batch
    cost.access(n_out * batch * 4);    // Y, one column per request in the batch
    cost.flop(2 * n_out * n_in * batch);
    return cost;
}
struct Roofline {
    double peak_compute_gflops, peak_bandwidth_gbps;
    double achievable_gflops(double ai) const { return std::min(peak_compute_gflops, ai * peak_bandwidth_gbps); }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 8.4: Continuous Batching -- the Same Crossover, a Different Question\n";
    std::cout << "========================================================\n\n";

    constexpr double PEAK_COMPUTE_GFLOPS = 896.0, PEAK_BANDWIDTH_GBPS = 51.2;  // Section 8.1's derived roofline
    constexpr double RIDGE_POINT = PEAK_COMPUTE_GFLOPS / PEAK_BANDWIDTH_GBPS;   // 17.5 FLOPs/byte
    constexpr long long N_OUT = 14336, N_IN = 4096;  // the same FFN projection Section 8.3 used
    Roofline roof{PEAK_COMPUTE_GFLOPS, PEAK_BANDWIDTH_GBPS};

    // A batch's predicted processing time, in milliseconds, from the
    // roofline model: total FLOPs divided by the achievable rate at this
    // batch's own arithmetic intensity. Every request in the batch is
    // served SYNCHRONOUSLY -- nobody's next token is ready until the
    // whole batched GEMM finishes -- so this is also each request's
    // per-token latency for that round.
    auto batch_time_ms = [&](long long B) {
        Cost c = linear_fp32_cost(N_OUT, N_IN, B);
        double gflops = static_cast<double>(c.flops) / 1e9;
        double achievable = roof.achievable_gflops(c.arithmetic_intensity());
        return (gflops / achievable) * 1000.0;  // seconds -> ms
    };

    // =====================================================================
    // TEST 1: below the crossover, batching more requests together barely
    // changes the batch's total processing time, because the weight read
    // -- not the batch size -- is what the time is dominated by.
    // =====================================================================
    std::cout << "-- Test 1: batch latency is nearly flat while memory-bound --\n";
    {
        double t1 = batch_time_ms(1);
        double t8 = batch_time_ms(8);
        std::cout << "  Batch time at B=1: " << std::fixed << std::setprecision(4) << t1 << " ms\n";
        std::cout << "  Batch time at B=8: " << t8 << " ms  (" << std::setprecision(2) << (t8 / t1) << "x of B=1)\n";
        std::cout << "  8x more requests served for well under 8x the time -- almost free,\n";
        std::cout << "  because the dominant cost (reading the weight matrix once) didn't change.\n";
        CHECK(t8 / t1 < 2.0);  // far less than the naive "8 requests = 8x time" assumption
        CHECK(t8 > t1);        // but not literally free either -- X/Y bytes and FLOPs did grow
    }

    // =====================================================================
    // TEST 2: find B*, the batch size at which this GEMM crosses from
    // memory-bound to compute-bound -- by search, reusing Section 8.3's
    // own crossover-finding method verbatim.
    // =====================================================================
    std::cout << "\n-- Test 2: batch-size crossover B* --\n";
    long long b_star = -1;
    {
        for (long long b = 1; b <= 100000; ++b) {
            if (linear_fp32_cost(N_OUT, N_IN, b).arithmetic_intensity() >= RIDGE_POINT) { b_star = b; break; }
        }
        std::cout << "  B* = " << b_star << " (this FFN projection becomes compute-bound at batch size " << b_star << ")\n";
        CHECK(b_star > 0);
        CHECK_NEAR(b_star, 36, 2);  // same shape as Section 8.3's crossover_fp32 -- batch size plays seq's role
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): assuming more batching always helps
    // throughput proportionally. Past B*, the batch is compute-bound, so
    // batch time grows LINEARLY with B (no more free lunch from shared
    // weight reads) -- aggregate throughput plateaus at a fixed
    // tokens/second ceiling, while each request's own per-token latency
    // keeps getting WORSE the more requests are piled into the batch.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: past B*, more batching raises latency without raising throughput --\n";
    {
        long long b_small = b_star / 2;       // still memory-bound
        long long b_big = b_star * 8;         // deep in the compute-bound regime

        double t_small = batch_time_ms(b_small);
        double t_big = batch_time_ms(b_big);
        double throughput_small = static_cast<double>(b_small) / t_small;  // requests served per ms
        double throughput_big = static_cast<double>(b_big) / t_big;

        std::cout << "  B=" << b_small << " (memory-bound):  batch time=" << std::setprecision(4) << t_small
                  << " ms, throughput=" << std::setprecision(2) << throughput_small << " tokens/ms\n";
        std::cout << "  B=" << b_big << " (compute-bound):  batch time=" << t_big
                  << " ms, throughput=" << throughput_big << " tokens/ms\n";
        std::cout << "  Batch time grew " << std::setprecision(2) << (t_big / t_small) << "x while batch size grew "
                  << (static_cast<double>(b_big) / b_small) << "x -- roughly proportionally, once compute-bound.\n";
        std::cout << "  Throughput grew only " << (throughput_big / throughput_small)
                  << "x for a " << (static_cast<double>(b_big) / b_small) << "x bigger batch --\n";
        std::cout << "  every request now waits longer per token, for a throughput gain far short of\n";
        std::cout << "  proportional. Past B*, adding requests to a batch trades latency for very\n";
        std::cout << "  little additional throughput, the mirror image of Test 1's near-free batching.\n";

        // Throughput growth is far short of the batch-size growth ratio (proportional
        // growth would be the naive, wrong assumption this trap corrects).
        double batch_growth = static_cast<double>(b_big) / static_cast<double>(b_small);
        double throughput_growth = throughput_big / throughput_small;
        CHECK(throughput_growth < batch_growth * 0.5);
        // Per-token latency (== batch time, since service is synchronous) got worse, not better.
        CHECK(t_big > t_small * (batch_growth * 0.5));  // grew at least roughly proportionally to batch size
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
