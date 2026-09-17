// Chapter 30.4 -- Chapter 4 implemented affine quantization as a working
// tool; this section treats it as what it actually is: a real affine
// map, quantize(x) = round(x / scale) + zero_point, whose only
// non-linear step is the round itself. That single fact is what
// produces quantization's own real, provable error bound (never more
// than scale/2 away from the original value), and its own real,
// checkable failure to compose: re-quantizing an already-dequantized
// value at a coarser scale is NOT the same real operation as quantizing
// the original value directly at that coarser scale -- a genuine
// non-associativity with direct consequences for any system, like
// Chapter 14's own streaming re-quantization, that quantizes more than
// once.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_quantization_as_affine_algebra.cpp -o 04_quantization_as_affine_algebra
// Run:     ./04_quantization_as_affine_algebra

#include <cmath>
#include <cstdint>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: the real affine map itself -- the standard optimal scale for
// mapping a real [min_val, max_val] range onto an unsigned integer grid
// of a stated bit width, plus its own forward (quantize) and inverse
// (dequantize) affine transformations.
// =======================================================================
double optimal_scale(double min_val, double max_val, int bits) {
    double levels = std::pow(2.0, bits) - 1.0;  // e.g. 255 real distinct integer codes for 8 bits
    return (max_val - min_val) / levels;
}

int64_t quantize(double x, double min_val, double scale) {
    return static_cast<int64_t>(std::llround((x - min_val) / scale));
}

double dequantize(int64_t q, double min_val, double scale) {
    return min_val + static_cast<double>(q) * scale;
}

// =======================================================================
// PART 2: the real, provable error bound. Because round() never moves a
// value by more than half a real quantization step, no real x in
// [min_val, max_val] can ever land farther than scale/2 from its own
// dequantized reconstruction.
// =======================================================================
double quantization_error(double x, double min_val, double scale) {
    return std::fabs(x - dequantize(quantize(x, min_val, scale), min_val, scale));
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 30.4: Quantization as Affine Algebra\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real optimal scale formula, and an exact real round-trip at both "
                 "endpoints of the quantization range --\n";
    {
        double min_val = -1.0, max_val = 1.0;
        double scale = optimal_scale(min_val, max_val, 8);
        CHECK(near(scale, 2.0 / 255.0));

        int64_t q_min = quantize(min_val, min_val, scale);
        int64_t q_max = quantize(max_val, min_val, scale);
        CHECK(q_min == 0);
        CHECK(q_max == 255);
        CHECK(near(dequantize(q_min, min_val, scale), min_val, 1e-9));
        CHECK(near(dequantize(q_max, min_val, scale), max_val, 1e-9));
        std::cout << "  an 8-bit real quantization of [-1.0, 1.0] has a real scale of exactly 2.0/255; "
                     "the real range's own two endpoints quantize to exactly integer codes 0 and 255, "
                     "and dequantizing each reproduces the exact original endpoint\n";
    }

    std::cout << "\n-- Test 2: the real, general error bound -- no real value anywhere in the "
                 "quantization range is ever reconstructed more than scale/2 away from itself, checked "
                 "across many sample points, not merely a hand-picked few --\n";
    {
        double min_val = -3.0, max_val = 5.0;
        double scale = optimal_scale(min_val, max_val, 8);
        int violations = 0;
        int checked = 0;
        for (int i = 0; i <= 1000; i++) {
            double x = min_val + (max_val - min_val) * (static_cast<double>(i) / 1000.0);
            double err = quantization_error(x, min_val, scale);
            if (err > scale / 2.0 + 1e-9) violations++;
            checked++;
        }
        CHECK(violations == 0);
        CHECK(checked == 1001);
        std::cout << "  across " << checked << " real sample points evenly spanning [-3.0, 5.0], not a "
                     "single one is ever reconstructed more than scale/2 away from its own true value -- "
                     "the real error bound holds with zero exceptions\n";
    }

    std::cout << "\n-- Test 3: the real error bound from Test 2 is TIGHT, not merely generous -- a real "
                 "value constructed exactly at a quantization bin's own real midpoint is reconstructed "
                 "with an error genuinely close to the full scale/2 bound, not comfortably inside it --\n";
    {
        double min_val = -3.0, max_val = 5.0;
        double scale = optimal_scale(min_val, max_val, 8);
        // A real value sitting exactly halfway between two adjacent real quantization levels.
        double x = min_val + scale * 10.5;
        double err = quantization_error(x, min_val, scale);
        CHECK(err > scale / 2.0 - 1e-6);
        CHECK(err <= scale / 2.0 + 1e-6);
        std::cout << "  a real value placed exactly at the midpoint between two adjacent quantization "
                     "levels reconstructs with an error of " << err << ", within a millionth of the "
                     "theoretical scale/2 bound of " << (scale / 2.0) << " -- confirming the bound from "
                     "Test 2 is the real, tight worst case, not an artificially loose estimate\n";
    }

    std::cout << "\n-- Test 4: re-quantization does not commute -- quantizing a real value once at a "
                 "coarse scale, versus quantizing it finely, dequantizing, then re-quantizing coarsely, "
                 "can produce two genuinely DIFFERENT real integer codes, a direct real consequence for "
                 "any system (like Chapter 14's own streaming re-quantization) that quantizes more than "
                 "once --\n";
    {
        double min_val = 0.0, max_val = 10.0;
        double fine_scale = optimal_scale(min_val, max_val, 7);   // a real, fine-grained scale
        double coarse_scale = optimal_scale(min_val, max_val, 4); // a real, much coarser scale

        double x = 1.0 / 3.0;  // a real value chosen so the two real paths land on different final codes

        // Path A: quantize directly at the coarse scale.
        int64_t direct_coarse = quantize(x, min_val, coarse_scale);

        // Path B: quantize finely first, dequantize back to a real float, THEN quantize at the coarse scale.
        int64_t fine_code = quantize(x, min_val, fine_scale);
        double roundtripped = dequantize(fine_code, min_val, fine_scale);
        int64_t via_fine_then_coarse = quantize(roundtripped, min_val, coarse_scale);

        CHECK(direct_coarse != via_fine_then_coarse);
        std::cout << "  quantizing " << x << " directly at a coarse 4-bit scale yields code "
                  << direct_coarse << "; quantizing the SAME real value finely at 7 bits first, "
                     "dequantizing it back to a real float, and THEN quantizing that at the identical "
                     "coarse 4-bit scale yields a genuinely different code, " << via_fine_then_coarse
                  << " -- re-quantization is a real, checkable non-associative operation, not merely a "
                     "theoretical curiosity\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
