// 01_latency_budget_and_amdahls_law.cpp
// Chapter 10, Part 1: before writing a single std::thread, derive WHY a
// single core cannot hit a realistic decode-time latency target at all --
// reusing Chapter 8's own CpuSpec and roofline formulas unchanged, at the
// exact FFN dimensions Chapter 8.3 and 8.5 already used -- and then derive
// the ceiling that adding MORE cores runs into. Amdahl's law is not a new
// architectural fact the way peak compute and peak bandwidth were; it is a
// direct algebraic consequence of one premise: any real per-token step has
// SOME part that does not parallelize across cores (sampling, detokenizing,
// scheduling), and that part alone puts a hard ceiling on how much total
// speedup any number of additional cores can ever buy.
//
// Every number in this file is either a stated architectural or workload
// parameter (labeled as such, exactly like Chapter 8.1's illustrative CPU
// and Chapter 8.5's illustrative model dimensions) or arithmetic derived
// from those parameters. No thread is created in this file -- Amdahl's law
// is pure arithmetic over a serial fraction and a core count, and this
// section's job is to derive that arithmetic before Section 10.2 spends any
// of it on an actual std::thread.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_latency_budget_and_amdahls_law.cpp -o 01_latency_budget_and_amdahls_law

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <limits>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

// =========================================================================
// Chapter 8.1's CpuSpec and Roofline, reused verbatim -- this section
// derives nothing new about a single core's peak compute or bandwidth; it
// only asks how long a REAL decode step's FLOPs take against that ceiling.
// =========================================================================
struct CpuSpec {
    int cores;
    double sustained_ghz;
    int simd_lanes_fp32;
    int fma_ports;
    double flops_per_cycle_per_core() const {
        return static_cast<double>(simd_lanes_fp32) * fma_ports * 2.0;
    }
    double peak_gflops() const {
        return cores * sustained_ghz * flops_per_cycle_per_core();
    }
};

struct Roofline {
    double peak_compute_gflops;
    double peak_bandwidth_gbps;
    double achievable_gflops(double arithmetic_intensity) const {
        return std::min(peak_compute_gflops, arithmetic_intensity * peak_bandwidth_gbps);
    }
};

// =========================================================================
// Amdahl's law: if a fraction `serial_fraction` of a task's work cannot be
// parallelized at all, and the remaining `1 - serial_fraction` splits
// perfectly evenly across `n` cores with zero coordination overhead (the
// most GENEROUS possible assumption about the parallel part), the total
// time relative to one core is `serial_fraction + (1 - serial_fraction)/n`,
// and speedup is the reciprocal of that.
// =========================================================================
struct AmdahlModel {
    double serial_fraction;  // in [0, 1]: the fraction of per-token work that never parallelizes

    double relative_time(double n_cores) const {
        return serial_fraction + (1.0 - serial_fraction) / n_cores;
    }
    double speedup(double n_cores) const {
        return 1.0 / relative_time(n_cores);
    }
    // The speedup no number of cores can ever exceed, however large --
    // the limit of speedup(n) as n -> infinity is 1 / serial_fraction,
    // since the parallel term (1 - serial_fraction)/n vanishes.
    double speedup_ceiling() const {
        return 1.0 / serial_fraction;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 10.1: The Latency Budget and Amdahl's Ceiling\n";
    std::cout << "========================================================\n\n";

    // Chapter 8.1's own illustrative 8-core AVX2 machine, unchanged.
    CpuSpec cpu{.cores = 8, .sustained_ghz = 3.5, .simd_lanes_fp32 = 8, .fma_ports = 2};
    Roofline roof{cpu.peak_gflops(), /*peak_bandwidth_gbps=*/51.2};  // Chapter 8.1's own derived bandwidth

    // =====================================================================
    // TEST 1: a single core of this machine cannot hit a realistic
    // interactive decode latency target -- derived from Chapter 8.5's own
    // full-decode-step FLOP total (every QKVO and FFN projection, plus
    // attention, at DIM=4096, N_HEADS_Q=32, N_HEADS_KV=8, D_FF=14336,
    // cache_len=2048), run through Chapter 8.1's own achievable-throughput
    // formula at that step's own measured, memory-bound arithmetic
    // intensity -- not a new, separately invented workload.
    // =====================================================================
    std::cout << "-- Test 1: one core's decode-step time against a real latency budget --\n";
    {
        // Chapter 8.5's own instrumented Cost counter, run against a real
        // GQA kernel at DIM=4096, N_HEADS_Q=32, N_HEADS_KV=8, D_FF=14336,
        // cache_len=2048, measured a full decode step (every QKVO and FFN
        // projection, GQA attention, both RMSNorms) at exactly
        // FLOPs=470466888, Bytes=890151168 -- reused here verbatim as
        // Chapter 8.5's own locked, verified total, not re-derived.
        constexpr double DECODE_STEP_FLOPS_TOTAL = 470466888.0;
        constexpr double DECODE_STEP_GFLOPS_TOTAL = DECODE_STEP_FLOPS_TOTAL / 1e9;
        // Chapter 8.5's own measured arithmetic intensity at that exact
        // shape (470466888 / 890151168), confirmed there to be memory-bound
        // against a ridge point of 17.50 -- reused here exactly, not
        // Chapter 8.1's separate representative AI=0.5 illustration.
        constexpr double DECODE_AI = 470466888.0 / 890151168.0;

        double achievable = roof.achievable_gflops(DECODE_AI);
        double single_core_achievable = achievable / cpu.cores;  // one core's share of the machine's peak
        double single_core_time_ms = (DECODE_STEP_GFLOPS_TOTAL / single_core_achievable) * 1000.0;

        constexpr double TARGET_TOKENS_PER_SEC = 20.0;  // a stated, labeled interactive-latency target
        double budget_ms_per_token = 1000.0 / TARGET_TOKENS_PER_SEC;

        std::cout << "  One decode step: " << std::fixed << std::setprecision(3)
                  << DECODE_STEP_GFLOPS_TOTAL << " GFLOP (Chapter 8.5's own full-layer total)\n";
        std::cout << "  One core's achievable throughput at AI=" << std::setprecision(4) << DECODE_AI << ": "
                  << std::setprecision(2) << single_core_achievable << " GFLOP/s\n";
        std::cout << "  One core's time for one decode step: " << std::setprecision(3)
                  << single_core_time_ms << " ms\n";
        std::cout << "  Latency budget at " << TARGET_TOKENS_PER_SEC << " tokens/sec: "
                  << budget_ms_per_token << " ms/token\n";
        std::cout << "  Single core is " << std::setprecision(1)
                  << (single_core_time_ms / budget_ms_per_token) << "x over budget.\n";

        CHECK(single_core_time_ms > budget_ms_per_token);
        CHECK_NEAR(budget_ms_per_token, 50.0, 1e-9);
    }

    // =====================================================================
    // TEST 2: Amdahl's law's basic algebraic properties -- speedup(1) is
    // exactly 1 regardless of serial fraction, and speedup is strictly
    // increasing in core count for any serial fraction less than 1.
    // =====================================================================
    std::cout << "\n-- Test 2: Amdahl's law's basic properties --\n";
    {
        // A stated, labeled per-token serial fraction: the part of a
        // decode step -- sampling the next token, updating the KV cache
        // write cursor, detokenizing -- that runs on exactly one core no
        // matter how many are available, illustrative in the same sense
        // Chapter 8.5's own dimensions were labeled illustrative.
        AmdahlModel model{.serial_fraction = 0.05};

        CHECK_NEAR(model.speedup(1.0), 1.0, 1e-9);

        double prev = model.speedup(1.0);
        bool strictly_increasing = true;
        for (double n : {2.0, 4.0, 8.0, 16.0, 32.0}) {
            double s = model.speedup(n);
            std::cout << "  speedup(" << std::setprecision(0) << n << " cores) = "
                      << std::setprecision(3) << s << "x\n";
            if (!(s > prev)) strictly_increasing = false;
            prev = s;
        }
        CHECK(strictly_increasing);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: assuming N cores buys N times the speedup.
    // With a serial fraction of just 5%, 8 cores buys roughly 5.9x, not
    // 8x -- and speedup does not merely fall short, it CONVERGES to a
    // hard ceiling of 1/serial_fraction as core count grows without
    // bound, so throwing arbitrarily many cores at the problem is
    // provably bounded before a single thread is ever created.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: assuming N cores means N times the speedup --\n";
    {
        AmdahlModel model{.serial_fraction = 0.05};

        double naive_expectation_at_8 = 8.0;  // "8 cores should mean 8x"
        double actual_at_8 = model.speedup(8.0);
        std::cout << "  serial_fraction = " << model.serial_fraction << " (5% of a decode step never parallelizes)\n";
        std::cout << "  Naive expectation at 8 cores: " << naive_expectation_at_8 << "x\n";
        std::cout << "  Actual Amdahl speedup at 8 cores: " << std::setprecision(3) << actual_at_8 << "x\n";
        CHECK(actual_at_8 < naive_expectation_at_8);
        CHECK(actual_at_8 > 5.5 && actual_at_8 < 6.0);

        double ceiling = model.speedup_ceiling();
        double at_1000 = model.speedup(1000.0);
        double at_1e9 = model.speedup(1.0e9);
        std::cout << "  Speedup ceiling (n -> infinity): 1 / " << model.serial_fraction
                  << " = " << std::setprecision(1) << ceiling << "x\n";
        std::cout << "  speedup(1000 cores)        = " << std::setprecision(4) << at_1000 << "x\n";
        std::cout << "  speedup(1,000,000,000 cores) = " << at_1e9 << "x -- still short of the ceiling, never past it\n";

        CHECK_NEAR(ceiling, 20.0, 1e-9);
        CHECK(at_1000 < ceiling);
        CHECK(at_1e9 < ceiling);
        CHECK(at_1e9 > at_1000);  // still strictly increasing, just converging
        // No finite core count reaches the ceiling exactly, since
        // (1 - serial_fraction)/n is strictly positive for any finite n.
        CHECK_NEAR(model.relative_time(1.0e9) - model.serial_fraction, (1.0 - model.serial_fraction) / 1.0e9, 1e-15);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
