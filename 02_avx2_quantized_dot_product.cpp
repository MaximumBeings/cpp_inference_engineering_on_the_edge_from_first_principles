// 02_avx2_quantized_dot_product.cpp
// Chapter 9, Part 2: every Q8_0 and Q4_0 block dot product in this book
// so far -- Chapter 4's fused_dot_product, Chapter 8.3's linear layer
// cost model -- has been a plain scalar loop. This section writes the
// real AVX2 instructions a production inference engine actually issues
// to compute that same dot product eight elements at a time, and checks
// the vectorized result against the scalar reference this book has used
// throughout, not merely against itself.
//
// This file requires an x86_64 CPU with AVX2 and FMA support to compile
// its intrinsics meaningfully and to run at all; it is not portable to
// Arm (Section 9.3 is Arm NEON's equivalent of this section).
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -mavx2 -mfma 02_avx2_quantized_dot_product.cpp -o 02_avx2_quantized_dot_product

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
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
// Chapter 4/8's fp16_t, BlockQ8, and BlockQ4 (reused verbatim) -- the
// exact byte layouts a real AVX2 kernel has to unpack.
// =========================================================================
struct fp16_t {
    uint16_t bits = 0;
    fp16_t() = default;
    fp16_t(float f) { bits = encode(f); }
    operator float() const { return decode(bits); }
    static uint16_t encode(float f) {
        uint32_t x; std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;
        if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        if (exp <= 0)  return static_cast<uint16_t>(sign);
        uint32_t mant10 = mant >> 13;
        uint32_t remainder = mant & 0x1FFFu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            mant10 += 1;
            if (mant10 == 0x400u) { mant10 = 0; exp += 1; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant10);
    }
    static float decode(uint16_t h) {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
        uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
        uint32_t fbits;
        if (exp == 0) fbits = sign;
        else if (exp == 31) fbits = sign | 0x7F800000u | (mant << 13);
        else fbits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        float f; std::memcpy(&f, &fbits, 4); return f;
    }
};
#pragma pack(push, 1)
struct BlockQ8 { fp16_t scale; int8_t weights[32]; };
struct BlockQ4 { fp16_t scale; uint8_t nibbles[16]; };
#pragma pack(pop)

BlockQ8 quantize_q8(const float* data) {
    BlockQ8 b;
    float alpha = 0.0f;
    for (int i = 0; i < 32; ++i) alpha = std::max(alpha, std::fabs(data[i]));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); std::memset(b.weights, 0, 32); return b; }
    b.scale = fp16_t(alpha / 127.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 32; ++i)
        b.weights[i] = static_cast<int8_t>(std::clamp(std::round(data[i] * inv), -127.0f, 127.0f));
    return b;
}
BlockQ4 quantize_q4(const float* data) {
    BlockQ4 b;
    float alpha = 0.0f;
    for (int i = 0; i < 32; ++i) alpha = std::max(alpha, std::fabs(data[i]));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); for (auto& n : b.nibbles) n = 0x88; return b; }
    b.scale = fp16_t(alpha / 7.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 16; ++i) {
        int lo = static_cast<int>(std::clamp(std::round(data[2 * i] * inv), -8.0f, 7.0f));
        int hi = static_cast<int>(std::clamp(std::round(data[2 * i + 1] * inv), -8.0f, 7.0f));
        b.nibbles[i] = static_cast<uint8_t>((static_cast<uint8_t>(hi + 8) << 4) | static_cast<uint8_t>(lo + 8));
    }
    return b;
}

// =========================================================================
// SCALAR REFERENCE -- exactly the loop this book has used since Chapter
// 4, unchanged. This is the ground truth every vectorized version below
// is checked against.
// =========================================================================
float scalar_dot_q8(const BlockQ8& b, const float* x) {
    float s = static_cast<float>(b.scale), acc = 0.0f;
    for (int i = 0; i < 32; ++i) acc += static_cast<float>(b.weights[i]) * s * x[i];
    return acc;
}
float scalar_dot_q4(const BlockQ4& b, const float* x) {
    float s = static_cast<float>(b.scale), acc = 0.0f;
    for (int i = 0; i < 16; ++i) {
        uint8_t p = b.nibbles[i];
        acc += (static_cast<float>(static_cast<int>(p & 0xF) - 8) * s) * x[2 * i];
        acc += (static_cast<float>(static_cast<int>(p >> 4) - 8) * s) * x[2 * i + 1];
    }
    return acc;
}

// Horizontal sum of one AVX2 __m256 register's 8 lanes down to a scalar.
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 sum2 = _mm_hadd_ps(sum4, sum4);
    __m128 sum1 = _mm_hadd_ps(sum2, sum2);
    return _mm_cvtss_f32(sum1);
}

// =========================================================================
// AVX2 Q8_0 dot product: 8 int8 weights widened to int32, converted to
// float, scaled, and FMA'd against 8 activations -- four such rounds
// cover all 32 elements of one block.
// =========================================================================
float avx2_dot_q8(const BlockQ8& b, const float* x) {
    __m256 vscale = _mm256_set1_ps(static_cast<float>(b.scale));
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < 32; i += 8) {
        __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b.weights + i));  // 8 packed int8
        __m256i w32 = _mm256_cvtepi8_epi32(w8);   // sign-extend to 8 x int32
        __m256 wf = _mm256_cvtepi32_ps(w32);      // convert to 8 x float
        __m256 xv = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(wf, vscale), xv, acc);
    }
    return hsum256_ps(acc);
}

// =========================================================================
// AVX2 Q4_0 dot product: the packed nibbles must be unpacked into 32
// individual signed int8 values BEFORE the same widen-convert-FMA
// pipeline applies. The unpack itself is real SIMD work, not a scalar
// preprocessing step smuggled in before "the real" vectorized part.
// =========================================================================
float avx2_dot_q4(const BlockQ4& b, const float* x) {
    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b.nibbles));  // 16 packed bytes
    __m128i mask0f = _mm_set1_epi8(0x0F);
    __m128i bias8 = _mm_set1_epi8(8);

    __m128i lo_nib = _mm_and_si128(packed, mask0f);                         // 16 low nibbles, one per byte
    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(packed, 4), mask0f);      // 16 high nibbles, one per byte
    lo_nib = _mm_sub_epi8(lo_nib, bias8);   // remove the +8 bias -> signed [-8, 7]
    hi_nib = _mm_sub_epi8(hi_nib, bias8);

    // Interleave lo[i], hi[i] back into original element order:
    // data[2i] = lo[i], data[2i+1] = hi[i] -- exactly what quantize_q4 wrote.
    __m128i elems_0_15 = _mm_unpacklo_epi8(lo_nib, hi_nib);   // elements 0..15
    __m128i elems_16_31 = _mm_unpackhi_epi8(lo_nib, hi_nib);  // elements 16..31

    __m256 vscale = _mm256_set1_ps(static_cast<float>(b.scale));
    __m256 acc = _mm256_setzero_ps();
    __m128i chunks[4] = {
        elems_0_15,                                   // elements 0..7 in the low 8 bytes
        _mm_srli_si128(elems_0_15, 8),                 // elements 8..15
        elems_16_31,                                   // elements 16..23
        _mm_srli_si128(elems_16_31, 8),                // elements 24..31
    };
    for (int c = 0; c < 4; ++c) {
        __m256i w32 = _mm256_cvtepi8_epi32(chunks[c]);
        __m256 wf = _mm256_cvtepi32_ps(w32);
        __m256 xv = _mm256_loadu_ps(x + c * 8);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(wf, vscale), xv, acc);
    }
    return hsum256_ps(acc);
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 9.2: AVX2 Integer Dot Products -- Q8_0 and Q4_0, Vectorized\n";
    std::cout << "========================================================\n\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    // =====================================================================
    // TEST 1: AVX2 Q8_0 dot product matches the scalar reference across
    // many random blocks.
    // =====================================================================
    std::cout << "-- Test 1: AVX2 Q8_0 dot product vs. scalar reference, 200 random blocks --\n";
    {
        float max_abs_err = 0.0f, max_rel_err = 0.0f;
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<float> data(32), x(32);
            for (auto& v : data) v = dist(rng);
            for (auto& v : x) v = dist(rng);
            BlockQ8 b = quantize_q8(data.data());

            float scalar_result = scalar_dot_q8(b, x.data());
            float avx2_result = avx2_dot_q8(b, x.data());
            float abs_err = std::fabs(scalar_result - avx2_result);
            max_abs_err = std::max(max_abs_err, abs_err);
            max_rel_err = std::max(max_rel_err, abs_err / std::max(1.0f, std::fabs(scalar_result)));
        }
        std::cout << "  Max absolute error over 200 trials: " << std::scientific << std::setprecision(2)
                  << max_abs_err << std::defaultfloat << std::setprecision(6) << "\n";
        std::cout << "  (Nonzero only because AVX2's tree-reduction horizontal sum adds the same 32\n";
        std::cout << "  products in a different order than the scalar loop's strictly sequential sum --\n";
        std::cout << "  floating-point addition is not associative, so a different order can differ in\n";
        std::cout << "  its last few bits even though every individual multiply is identical.)\n";
        CHECK(max_abs_err < 1e-3f);
        CHECK(max_rel_err < 1e-4f);
    }

    // =====================================================================
    // TEST 2: AVX2 Q4_0 dot product -- including the nibble-unpack step --
    // matches the scalar reference across many random blocks.
    // =====================================================================
    std::cout << "\n-- Test 2: AVX2 Q4_0 dot product (with SIMD nibble unpacking) vs. scalar, 200 trials --\n";
    {
        float max_abs_err = 0.0f;
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<float> data(32), x(32);
            for (auto& v : data) v = dist(rng);
            for (auto& v : x) v = dist(rng);
            BlockQ4 b = quantize_q4(data.data());

            float scalar_result = scalar_dot_q4(b, x.data());
            float avx2_result = avx2_dot_q4(b, x.data());
            max_abs_err = std::max(max_abs_err, std::fabs(scalar_result - avx2_result));
        }
        std::cout << "  Max absolute error over 200 trials: " << std::scientific << std::setprecision(2)
                  << max_abs_err << std::defaultfloat << std::setprecision(6) << "\n";
        CHECK(max_abs_err < 1e-3f);
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): sign-extending BEFORE removing the +8 bias
    // vs. after -- these are not interchangeable, because the raw nibble
    // is UNSIGNED (0..15) and only becomes meaningful once the bias is
    // removed, whereas sign-extending a value that is still in [0, 15]
    // as if it were already signed produces a completely different
    // (always non-negative) number.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: sign-extending a nibble before removing its bias --\n";
    {
        std::vector<float> data(32);
        for (int i = 0; i < 32; ++i) data[i] = static_cast<float>((i % 15) - 7) * 0.3f;
        BlockQ4 b = quantize_q4(data.data());

        // Correct: extract raw nibble (0..15, unsigned), THEN subtract 8
        // to land in the signed range [-8, 7] the format actually uses.
        uint8_t raw_lo = b.nibbles[0] & 0x0F;
        int correct_value = static_cast<int>(raw_lo) - 8;

        // Buggy: treat the raw nibble as if it were already a signed
        // 4-bit two's-complement value and sign-extend it directly (as
        // int8_t) without ever subtracting the bias -- an easy mistake
        // when porting from a format that genuinely does store signed
        // nibbles directly.
        int8_t buggy_signed = static_cast<int8_t>(raw_lo);  // raw_lo is 0..15, so this never goes negative
        int buggy_value = static_cast<int>(buggy_signed);

        std::cout << "  Raw nibble bits: " << static_cast<int>(raw_lo) << " (unsigned, range [0, 15])\n";
        std::cout << "  Correct (subtract bias 8 first): " << correct_value << " (range [-8, 7])\n";
        std::cout << "  Buggy (sign-extend the raw bits directly): " << buggy_value
                  << " -- identical to the raw bits whenever they are already < 8,\n";
        std::cout << "  and never negative regardless of what the block actually encoded.\n";

        CHECK(correct_value >= -8 && correct_value <= 7);
        // The buggy path is wrong specifically whenever the true value
        // was meant to be negative (raw nibble < 8): it reports the
        // unsigned raw bits instead of ever going negative.
        CHECK(buggy_value == static_cast<int>(raw_lo));
        CHECK(buggy_value >= 0);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
