// Appendix D.1 -- Computing Theoretical Peak Bandwidth From a Real Spec Sheet.
//
// This reuses Chapter 8.1's own peak-bandwidth formula verbatim:
//   peak_bandwidth = channels * transfer_rate * bytes_per_transfer
// and checks it against one real, officially published edge-silicon spec
// this book can verify component-by-component: the Raspberry Pi 5's own
// BCM2712 SoC. Raspberry Pi's own official documentation states the part
// as "LPDDR4X-4267" (4267 MT/s) over a "32-bit LPDDR4X memory interface"
// that "provides up to 17 GB/s of memory bandwidth" -- one channel, a
// 32-bit (4-byte) bus, and a stated transfer rate are everything the
// formula needs, and multiplying them out reproduces the vendor's own
// rounded headline figure.
//
// Two other real edge/consumer platforms are cited in this appendix's own
// prose alongside this file (Jetson Orin Nano's 68 GB/s over a 128-bit
// LPDDR5 interface, and the Apple M4 family's 120/273/546 GB/s unified
// memory figures) but are NOT re-derived here, honestly: their vendors do
// not publish the same channel-count-times-transfer-rate breakdown
// Raspberry Pi's own documentation does, so this file only asserts what
// it can actually recompute from independently stated components.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 d1_peak_bandwidth_formula.cpp -o d1_peak_bandwidth_formula
// Run:     ./d1_peak_bandwidth_formula

#include <cmath>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// Chapter 8.1's own formula, reused verbatim. transfer_rate_mts is in
// mega-transfers per second; bytes_per_transfer is the bus width in bytes
// (32 bits = 4 bytes, 128 bits = 16 bytes, and so on). The result is in
// decimal GB/s (10^9 bytes/s), matching how vendors publish this figure.
double peak_bandwidth_gbps(double channels, double transfer_rate_mts, double bytes_per_transfer) {
    // MT/s * bytes/transfer = MB/s (10^6 bytes/s); dividing by 1000 gives GB/s.
    return channels * transfer_rate_mts * bytes_per_transfer / 1000.0;
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "Appendix D.1: Computing Theoretical Peak Bandwidth\n";
    std::cout << "=================================================\n\n";

    // -- Test 1: the Raspberry Pi 5's own real, officially documented
    // BCM2712 memory interface -- one 32-bit LPDDR4X-4267 channel. --
    {
        double channels = 1.0;
        double transfer_rate_mts = 4267.0;   // Raspberry Pi's own official product brief
        double bytes_per_transfer = 4.0;     // a 32-bit bus, per Raspberry Pi's own documentation

        double bandwidth = peak_bandwidth_gbps(channels, transfer_rate_mts, bytes_per_transfer);
        std::cout << "-- Test 1: Raspberry Pi 5 (BCM2712) -- channels=" << channels
                  << ", transfer_rate=" << transfer_rate_mts << " MT/s, bus_width="
                  << (bytes_per_transfer * 8) << "-bit -- computed peak=" << bandwidth << " GB/s --\n";

        CHECK(std::abs(bandwidth - 17.068) < 1e-9);
        // Raspberry Pi's own documentation rounds this down to "17 GB/s" --
        // confirm the formula's own result rounds to the identical headline figure.
        CHECK(std::floor(bandwidth) == 17.0);
    }

    // -- Test 2: the formula is a pure multiplication -- doubling any one
    // of the three real components exactly doubles the result, regardless
    // of which component changes. This is the same real property Chapter
    // 8.1 itself relied on: peak bandwidth has no term that saturates or
    // interacts nonlinearly with the others. --
    {
        double base = peak_bandwidth_gbps(1.0, 4267.0, 4.0);
        double double_channels = peak_bandwidth_gbps(2.0, 4267.0, 4.0);
        double double_rate = peak_bandwidth_gbps(1.0, 4267.0 * 2.0, 4.0);
        double double_width = peak_bandwidth_gbps(1.0, 4267.0, 8.0);
        std::cout << "-- Test 2: linearity -- base=" << base << ", 2x channels=" << double_channels
                  << ", 2x transfer_rate=" << double_rate << ", 2x bus_width=" << double_width << " --\n";
        CHECK(std::abs(double_channels - 2.0 * base) < 1e-9);
        CHECK(std::abs(double_rate - 2.0 * base) < 1e-9);
        CHECK(std::abs(double_width - 2.0 * base) < 1e-9);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
