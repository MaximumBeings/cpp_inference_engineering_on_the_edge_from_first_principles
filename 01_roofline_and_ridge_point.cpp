// 01_roofline_and_ridge_point.cpp
// Chapter 8, Part 1: derive the roofline model from first principles --
// not by quoting a manufacturer's headline GFLOPS number, but by building
// peak compute and peak memory bandwidth up from stated architectural
// parameters (core count, clock speed, SIMD width, FMA throughput; memory
// channels, transfer rate, bus width), then using the resulting ridge
// point to predict how fast a kernel of a given arithmetic intensity can
// actually run.
//
// Every number in this file is either a stated architectural parameter
// (labeled as such, exactly like Chapter 5.3's illustrative model
// dimensions) or arithmetic derived from those parameters -- nothing is
// measured by timing code on this machine, because a shared cloud sandbox
// gives no reproducible wall-clock number to lock a chapter's output
// against. The roofline FORMULA itself is exact arithmetic and is what
// gets verified here.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_roofline_and_ridge_point.cpp -o out01

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
// PEAK COMPUTE: derived from stated architectural parameters, not quoted
// =========================================================================
// One fused-multiply-add (FMA) does 2 FLOPs (one multiply, one add). An
// AVX2 register holds 8 packed FP32 lanes. Since Haswell (2013), x86
// cores that support AVX2+FMA can issue FMA instructions on two execution
// ports per cycle. So peak FP32 FLOPs per core per cycle is:
//   lanes_per_register * FMA_ports_per_cycle * FLOPs_per_FMA
struct CpuSpec {
    int cores;
    double sustained_ghz;   // all-core clock under sustained AVX2 load
    int simd_lanes_fp32;    // 8 for AVX2 (256-bit / 32-bit), 4 for SSE
    int fma_ports;          // 2 on mainstream Haswell-and-later x86 cores

    double flops_per_cycle_per_core() const {
        return static_cast<double>(simd_lanes_fp32) * fma_ports * 2.0;  // *2: FMA = mul+add
    }
    // Peak FP32 GFLOP/s across all cores.
    double peak_gflops() const {
        return cores * sustained_ghz * flops_per_cycle_per_core();
    }
};

// =========================================================================
// PEAK BANDWIDTH: derived from stated memory parameters, not quoted
// =========================================================================
// DDR memory transfers 8 bytes per cycle per 64-bit channel, at the
// stated transfer rate (megatransfers/second). Multi-channel memory
// multiplies this by the channel count.
struct MemorySpec {
    int channels;
    double transfer_rate_mts;  // e.g. 3200 for DDR4-3200 (million transfers/sec)
    int bytes_per_transfer;    // 8 bytes for a 64-bit channel

    // Peak bandwidth in GB/s.
    double peak_gbps() const {
        return channels * transfer_rate_mts * bytes_per_transfer / 1000.0;
    }
};

// =========================================================================
// THE ROOFLINE FORMULA
// =========================================================================
// Given a kernel's arithmetic intensity (FLOPs per byte moved) and a
// machine's peak compute and peak bandwidth, the roofline model predicts
// the best achievable throughput:
//   achievable_GFLOPS = min(peak_compute, arithmetic_intensity * peak_bandwidth)
// The ridge point (peak_compute / peak_bandwidth, in FLOPs/byte) is where
// the two terms cross: kernels to its left are memory-bound (bandwidth
// caps them), kernels to its right are compute-bound (peak FLOPs caps
// them).
struct Roofline {
    double peak_compute_gflops;
    double peak_bandwidth_gbps;

    double ridge_point() const { return peak_compute_gflops / peak_bandwidth_gbps; }

    double achievable_gflops(double arithmetic_intensity) const {
        return std::min(peak_compute_gflops, arithmetic_intensity * peak_bandwidth_gbps);
    }
    bool is_memory_bound(double arithmetic_intensity) const { return arithmetic_intensity < ridge_point(); }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 8.1: The Roofline Model, Derived From First Principles\n";
    std::cout << "========================================================\n\n";

    // Stated, illustrative desktop-class parameters (same convention as
    // Chapter 5.3's illustrative model dimensions: labeled inputs, not
    // measurements). An 8-core AVX2 chip with dual-channel DDR4-3200.
    CpuSpec cpu{.cores = 8, .sustained_ghz = 3.5, .simd_lanes_fp32 = 8, .fma_ports = 2};
    MemorySpec mem{.channels = 2, .transfer_rate_mts = 3200.0, .bytes_per_transfer = 8};
    Roofline roof{cpu.peak_gflops(), mem.peak_gbps()};

    // =====================================================================
    // TEST 1: Peak compute and peak bandwidth derive to sane, checkable
    // values from the stated architectural inputs.
    // =====================================================================
    std::cout << "-- Test 1: Deriving peak compute and peak bandwidth --\n";
    {
        double flops_per_cycle = cpu.flops_per_cycle_per_core();
        std::cout << "  FLOPs/cycle/core = " << cpu.simd_lanes_fp32 << " lanes * " << cpu.fma_ports
                  << " FMA ports * 2 FLOPs/FMA = " << flops_per_cycle << "\n";
        CHECK_NEAR(flops_per_cycle, 32.0, 1e-9);

        std::cout << "  Peak compute = " << cpu.cores << " cores * " << cpu.sustained_ghz
                  << " GHz * " << flops_per_cycle << " FLOPs/cycle = "
                  << std::fixed << std::setprecision(1) << roof.peak_compute_gflops << " GFLOP/s\n";
        CHECK_NEAR(roof.peak_compute_gflops, 8 * 3.5 * 32.0, 1e-6);

        std::cout << "  Peak bandwidth = " << mem.channels << " channels * " << mem.transfer_rate_mts
                  << " MT/s * " << mem.bytes_per_transfer << " bytes = "
                  << roof.peak_bandwidth_gbps << " GB/s\n";
        CHECK_NEAR(roof.peak_bandwidth_gbps, 2 * 3200.0 * 8 / 1000.0, 1e-6);

        std::cout << "  Ridge point = " << std::setprecision(2) << roof.ridge_point() << " FLOPs/byte\n";
        CHECK_NEAR(roof.ridge_point(), roof.peak_compute_gflops / roof.peak_bandwidth_gbps, 1e-9);
    }

    // =====================================================================
    // TEST 2: The roofline formula behaves correctly on both sides of the
    // ridge point -- linear growth below it, a hard ceiling above it.
    // =====================================================================
    std::cout << "\n-- Test 2: Achievable throughput on both sides of the ridge --\n";
    {
        double ridge = roof.ridge_point();
        double below = ridge * 0.5, above = ridge * 4.0;

        double achievable_below = roof.achievable_gflops(below);
        double achievable_above = roof.achievable_gflops(above);

        std::cout << "  Ridge point: " << std::setprecision(2) << ridge << " FLOPs/byte\n";
        std::cout << "  AI = " << below << " (below ridge) -> achievable = " << achievable_below
                  << " GFLOP/s (memory-bound: " << (roof.is_memory_bound(below) ? "yes" : "no") << ")\n";
        std::cout << "  AI = " << above << " (above ridge) -> achievable = " << achievable_above
                  << " GFLOP/s (memory-bound: " << (roof.is_memory_bound(above) ? "yes" : "no") << ")\n";

        // Below the ridge, achievable throughput is exactly AI * bandwidth.
        CHECK_NEAR(achievable_below, below * roof.peak_bandwidth_gbps, 1e-6);
        CHECK(roof.is_memory_bound(below));
        // Above the ridge, achievable throughput is capped at peak compute,
        // regardless of how much higher the AI climbs.
        CHECK_NEAR(achievable_above, roof.peak_compute_gflops, 1e-6);
        CHECK(!roof.is_memory_bound(above));

        // Doubling AI further above the ridge changes nothing.
        CHECK_NEAR(roof.achievable_gflops(above * 2.0), roof.peak_compute_gflops, 1e-6);
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): assuming a faster CPU always helps.
    // For a memory-bound kernel, doubling peak COMPUTE changes nothing --
    // the kernel was never limited by compute in the first place.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: a faster CPU does nothing for a memory-bound kernel --\n";
    {
        double decode_ai = 0.5;  // representative decode-time GEMV AI, derived in Section 8.2

        Roofline slow_cpu{roof.peak_compute_gflops, roof.peak_bandwidth_gbps};
        Roofline fast_cpu{roof.peak_compute_gflops * 4.0, roof.peak_bandwidth_gbps};  // 4x more FLOPs, same memory

        double achievable_slow = slow_cpu.achievable_gflops(decode_ai);
        double achievable_fast = fast_cpu.achievable_gflops(decode_ai);

        std::cout << "  Kernel AI = " << decode_ai << " FLOPs/byte (a decode-time GEMV)\n";
        std::cout << "  Slow CPU (" << std::setprecision(0) << slow_cpu.peak_compute_gflops
                  << " GFLOP/s peak): achievable = " << std::setprecision(2) << achievable_slow << " GFLOP/s\n";
        std::cout << "  Fast CPU (" << std::setprecision(0) << fast_cpu.peak_compute_gflops
                  << " GFLOP/s peak, 4x more compute): achievable = " << std::setprecision(2)
                  << achievable_fast << " GFLOP/s\n";
        std::cout << "  Quadrupling peak compute changed the achievable throughput by "
                  << std::setprecision(4) << (achievable_fast - achievable_slow) << " GFLOP/s --\n";
        std::cout << "  effectively nothing, because bandwidth was the bottleneck all along.\n";

        CHECK_NEAR(achievable_slow, achievable_fast, 1e-9);
        CHECK(fast_cpu.peak_compute_gflops > slow_cpu.peak_compute_gflops * 3.9);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
