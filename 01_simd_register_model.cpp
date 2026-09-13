// 01_simd_register_model.cpp
// Chapter 9, Part 1: before writing a single intrinsic, derive what a
// SIMD register actually buys a kernel -- exactly the way Chapter 8.1
// derived peak compute from stated architectural parameters rather than
// quoting a marketing number. A SIMD register is just a wider lane
// count: AVX2's 256-bit register holds 8 packed FP32 lanes (or 32 packed
// int8 lanes); Arm NEON's 128-bit register holds 4 packed FP32 lanes (or
// 16 packed int8 lanes). Reusing Chapter 8.1's own CpuSpec formula with
// only the lane count changed shows exactly how much of a chip's peak
// FLOP rate SIMD width alone is responsible for -- and, per Chapter 8's
// own COMMON TRAP, exactly why that width buys nothing for a kernel that
// is memory-bound rather than compute-bound.
//
// This file contains no architecture-specific intrinsics -- it is pure,
// portable arithmetic over labeled register-width parameters -- so it
// compiles and produces the identical, deterministic output on x86_64
// and on Arm alike. Sections 9.2-9.5 are where the actual AVX2 and NEON
// instructions appear.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_simd_register_model.cpp -o 01_simd_register_model

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
// Chapter 8.1's CpuSpec, unchanged, except simd_lanes_fp32 is now the
// number that actually distinguishes an AVX2 core from a NEON core: a
// SIMD register's bit width divided by 32 (bits per FP32 lane).
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

// Lanes of a given element width that fit in one SIMD register of a
// stated bit width -- register capacity is fixed in BITS, so a smaller
// element type always yields proportionally more lanes.
constexpr int lanes_per_register(int register_bits, int element_bits) {
    return register_bits / element_bits;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 9.1: The SIMD Register Model, Derived From Register Width\n";
    std::cout << "========================================================\n\n";

    // Stated register widths -- these are architectural facts, not
    // measurements: AVX2 registers (ymm0-15) are 256 bits; Arm NEON
    // registers (v0-31) are 128 bits. AVX-512, where available, would be
    // 512 bits, following the identical formula below.
    constexpr int AVX2_REGISTER_BITS = 256;
    constexpr int NEON_REGISTER_BITS = 128;
    constexpr int FP32_BITS = 32, INT8_BITS = 8, INT16_BITS = 16;

    // =====================================================================
    // TEST 1: lane counts derive directly from register width divided by
    // element width -- no lookup table required.
    // =====================================================================
    std::cout << "-- Test 1: lane counts, derived from register bits / element bits --\n";
    {
        int avx2_fp32_lanes = lanes_per_register(AVX2_REGISTER_BITS, FP32_BITS);
        int avx2_int8_lanes = lanes_per_register(AVX2_REGISTER_BITS, INT8_BITS);
        int neon_fp32_lanes = lanes_per_register(NEON_REGISTER_BITS, FP32_BITS);
        int neon_int8_lanes = lanes_per_register(NEON_REGISTER_BITS, INT8_BITS);

        std::cout << "  AVX2 (256-bit ymm):  " << avx2_fp32_lanes << " x fp32 lanes, "
                  << avx2_int8_lanes << " x int8 lanes\n";
        std::cout << "  NEON (128-bit v-reg): " << neon_fp32_lanes << " x fp32 lanes, "
                  << neon_int8_lanes << " x int8 lanes\n";

        CHECK(avx2_fp32_lanes == 8);
        CHECK(avx2_int8_lanes == 32);
        CHECK(neon_fp32_lanes == 4);
        CHECK(neon_int8_lanes == 16);
        // A register's lane count for a HALF-width element is always
        // exactly double its lane count for the full-width element --
        // this is arithmetic, not an empirical coincidence.
        CHECK(lanes_per_register(AVX2_REGISTER_BITS, INT16_BITS) == avx2_fp32_lanes * 2);
    }

    // =====================================================================
    // TEST 2: reusing Chapter 8.1's peak-compute formula unchanged, with
    // only simd_lanes_fp32 swapped, shows exactly how much of a chip's
    // peak FLOP rate its SIMD width alone accounts for.
    // =====================================================================
    std::cout << "\n-- Test 2: Chapter 8.1's peak-compute formula, AVX2 vs. NEON lane counts --\n";
    {
        // Same core count and clock speed for both -- isolating lane
        // count as the only variable, exactly as Chapter 8.1 isolated
        // peak compute from peak bandwidth.
        CpuSpec avx2_cpu{.cores = 8, .sustained_ghz = 3.5, .simd_lanes_fp32 = 8, .fma_ports = 2};
        CpuSpec neon_cpu{.cores = 8, .sustained_ghz = 3.5, .simd_lanes_fp32 = 4, .fma_ports = 2};

        std::cout << "  Same core count (8) and clock (3.5 GHz), AVX2 lanes=8 vs. NEON lanes=4:\n";
        std::cout << "  AVX2 peak compute: " << std::fixed << std::setprecision(1)
                  << avx2_cpu.peak_gflops() << " GFLOP/s\n";
        std::cout << "  NEON peak compute: " << neon_cpu.peak_gflops() << " GFLOP/s\n";
        std::cout << "  Ratio: " << std::setprecision(2)
                  << (avx2_cpu.peak_gflops() / neon_cpu.peak_gflops()) << "x\n";

        // Doubling the lane count exactly doubles peak compute -- the
        // formula is linear in simd_lanes_fp32, nothing more subtle.
        CHECK_NEAR(avx2_cpu.peak_gflops(), neon_cpu.peak_gflops() * 2.0, 1e-9);
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): a wider SIMD register buys nothing for a
    // memory-bound kernel -- the identical lesson Chapter 8.1 taught
    // about peak compute in general, now specifically about the part of
    // peak compute that SIMD width controls.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: wider SIMD registers do nothing for a memory-bound kernel --\n";
    {
        constexpr double PEAK_BANDWIDTH_GBPS = 51.2;  // Chapter 8.1's derived bandwidth, reused
        auto achievable = [&](double peak_compute, double ai) { return std::min(peak_compute, ai * PEAK_BANDWIDTH_GBPS); };

        CpuSpec avx2_cpu{.cores = 8, .sustained_ghz = 3.5, .simd_lanes_fp32 = 8, .fma_ports = 2};
        CpuSpec neon_cpu{.cores = 8, .sustained_ghz = 3.5, .simd_lanes_fp32 = 4, .fma_ports = 2};

        constexpr double DECODE_AI = 0.5;  // Chapter 8.1's own representative decode-time GEMV
        double avx2_achievable = achievable(avx2_cpu.peak_gflops(), DECODE_AI);
        double neon_achievable = achievable(neon_cpu.peak_gflops(), DECODE_AI);

        std::cout << "  Decode-time kernel, AI=" << DECODE_AI << " FLOPs/byte (memory-bound on both):\n";
        std::cout << "  AVX2 (8 fp32 lanes) achievable: " << std::setprecision(2) << avx2_achievable << " GFLOP/s\n";
        std::cout << "  NEON (4 fp32 lanes) achievable: " << neon_achievable << " GFLOP/s\n";
        std::cout << "  Doubling the SIMD lane count changed achievable throughput by "
                  << std::setprecision(4) << (avx2_achievable - neon_achievable) << " GFLOP/s --\n";
        std::cout << "  nothing, for exactly the reason Chapter 8.1 gave: bandwidth, not the width\n";
        std::cout << "  of the register doing the arithmetic, is what a memory-bound kernel waits on.\n";

        CHECK_NEAR(avx2_achievable, neon_achievable, 1e-9);
        CHECK(avx2_cpu.peak_gflops() > neon_cpu.peak_gflops());  // the compute capacity really did double
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
