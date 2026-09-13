// 03_neon_quantized_dot_product.cpp
// Chapter 9, Part 3: Arm NEON's equivalent of Section 9.2 -- the same
// Q8_0 and Q4_0 block dot products, now written against NEON's 128-bit,
// 4-lane-fp32 register model instead of AVX2's 256-bit, 8-lane one.
// Every intrinsic here has a direct AVX2 analogue from Section 9.2
// (vld1q_s8 <-> _mm256_loadu_si256, vmovl_s8/vmovl_s16 widening <->
// _mm256_cvtepi8_epi32, vfmaq_f32 <-> _mm256_fmadd_ps), which is the
// point: the two instruction sets solve the identical problem with
// differently-shaped tools, not with different ideas.
//
// This file requires an Arm CPU (or an aarch64 cross-compile run under
// emulation) to compile and run its NEON intrinsics; it is not portable
// to x86_64 (Section 9.2 is AVX2's equivalent of this section). Locked
// here via cross-compilation with aarch64-linux-gnu-g++ and execution
// under qemu-aarch64 user-mode emulation, and separately confirmed by
// running natively on real Arm hardware -- this book's own policy
// against fabricated numbers applies as much to "did this NEON code
// even run" as it does to timing, so nothing here is asserted without
// having actually executed on an aarch64 target one way or the other.
//
// Compile (native Arm): g++ -std=c++23 -Wall -Wextra -O2 03_neon_quantized_dot_product.cpp -o 03_neon_quantized_dot_product
// Compile (cross, x86 host): aarch64-linux-gnu-g++ -std=c++23 -Wall -Wextra -O2 03_neon_quantized_dot_product.cpp -o 03_neon_quantized_dot_product_arm
// Run under emulation:       qemu-aarch64 -L /usr/aarch64-linux-gnu 03_neon_quantized_dot_product_arm

#include <arm_neon.h>
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

// =========================================================================
// Chapter 4/8/9's fp16_t, BlockQ8, and BlockQ4 (reused verbatim)
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

// Widen 16 packed int8 lanes into four int32x4_t chunks (elements
// [0..3], [4..7], [8..11], [12..15]) -- NEON has no single instruction
// that widens int8 all the way to int32 in one step, so this goes
// through int16 first, exactly as the AVX2 version goes through
// _mm256_cvtepi8_epi32 in one step because x86's widening convert can
// jump two widths at once where NEON's cannot.
static inline void widen16_to_int32x4(int8x16_t v, int32x4_t out[4]) {
    int16x8_t lo16 = vmovl_s8(vget_low_s8(v));
    int16x8_t hi16 = vmovl_s8(vget_high_s8(v));
    out[0] = vmovl_s16(vget_low_s16(lo16));
    out[1] = vmovl_s16(vget_high_s16(lo16));
    out[2] = vmovl_s16(vget_low_s16(hi16));
    out[3] = vmovl_s16(vget_high_s16(hi16));
}

// =========================================================================
// NEON Q8_0 dot product: 16 int8 weights loaded and widened at a time,
// processed four elements (one NEON fp32 lane group) per FMA.
// =========================================================================
float neon_dot_q8(const BlockQ8& b, const float* x) {
    float32x4_t vscale = vdupq_n_f32(static_cast<float>(b.scale));
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int base = 0; base < 32; base += 16) {
        int8x16_t w8 = vld1q_s8(b.weights + base);
        int32x4_t chunks[4];
        widen16_to_int32x4(w8, chunks);
        for (int c = 0; c < 4; ++c) {
            float32x4_t wf = vcvtq_f32_s32(chunks[c]);
            float32x4_t xv = vld1q_f32(x + base + c * 4);
            acc = vfmaq_f32(acc, vmulq_f32(wf, vscale), xv);
        }
    }
    return vaddvq_f32(acc);  // horizontal sum across all 4 lanes
}

// =========================================================================
// NEON Q4_0 dot product: the same mask/shift/bias/interleave unpack as
// Section 9.2's AVX2 version, using NEON's vzip1q/vzip2q_s8 in place of
// _mm_unpacklo_epi8/_mm_unpackhi_epi8 -- the identical interleaving
// operation, named differently.
// =========================================================================
float neon_dot_q4(const BlockQ4& b, const float* x) {
    uint8x16_t packed = vld1q_u8(b.nibbles);
    uint8x16_t mask0f = vdupq_n_u8(0x0F);
    uint8x16_t bias8 = vdupq_n_u8(8);

    uint8x16_t lo_nib = vandq_u8(packed, mask0f);
    uint8x16_t hi_nib = vandq_u8(vshrq_n_u8(packed, 4), mask0f);
    // Subtracting 8 from a small unsigned value (0..15) and reinterpreting
    // the resulting bit pattern as signed gives the correct value modulo
    // 256 -- the same trick Section 9.2's _mm_sub_epi8 relies on, since
    // two's-complement subtraction does not care whether the operands are
    // "meant" to be signed or unsigned.
    int8x16_t lo_signed = vreinterpretq_s8_u8(vsubq_u8(lo_nib, bias8));
    int8x16_t hi_signed = vreinterpretq_s8_u8(vsubq_u8(hi_nib, bias8));

    // Interleave lo[i], hi[i] back into original element order:
    // data[2i] = lo[i], data[2i+1] = hi[i].
    int8x16_t elems_0_15 = vzip1q_s8(lo_signed, hi_signed);
    int8x16_t elems_16_31 = vzip2q_s8(lo_signed, hi_signed);

    float32x4_t vscale = vdupq_n_f32(static_cast<float>(b.scale));
    float32x4_t acc = vdupq_n_f32(0.0f);

    int32x4_t chunks_lo[4], chunks_hi[4];
    widen16_to_int32x4(elems_0_15, chunks_lo);
    widen16_to_int32x4(elems_16_31, chunks_hi);
    for (int c = 0; c < 4; ++c) {
        float32x4_t wf = vcvtq_f32_s32(chunks_lo[c]);
        float32x4_t xv = vld1q_f32(x + c * 4);
        acc = vfmaq_f32(acc, vmulq_f32(wf, vscale), xv);
    }
    for (int c = 0; c < 4; ++c) {
        float32x4_t wf = vcvtq_f32_s32(chunks_hi[c]);
        float32x4_t xv = vld1q_f32(x + 16 + c * 4);
        acc = vfmaq_f32(acc, vmulq_f32(wf, vscale), xv);
    }
    return vaddvq_f32(acc);
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 9.3: Arm NEON -- The Mobile/Edge Path\n";
    std::cout << "========================================================\n\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    // =====================================================================
    // TEST 1: NEON Q8_0 dot product vs. scalar reference.
    // =====================================================================
    std::cout << "-- Test 1: NEON Q8_0 dot product vs. scalar reference, 200 random blocks --\n";
    {
        float max_abs_err = 0.0f;
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<float> data(32), x(32);
            for (auto& v : data) v = dist(rng);
            for (auto& v : x) v = dist(rng);
            BlockQ8 b = quantize_q8(data.data());

            float scalar_result = scalar_dot_q8(b, x.data());
            float neon_result = neon_dot_q8(b, x.data());
            max_abs_err = std::max(max_abs_err, std::fabs(scalar_result - neon_result));
        }
        std::cout << "  Max absolute error over 200 trials: " << std::scientific << std::setprecision(2)
                  << max_abs_err << std::defaultfloat << std::setprecision(6) << "\n";
        CHECK(max_abs_err < 1e-3f);
    }

    // =====================================================================
    // TEST 2: NEON Q4_0 dot product (with SIMD nibble unpacking) vs. scalar.
    // =====================================================================
    std::cout << "\n-- Test 2: NEON Q4_0 dot product (with SIMD nibble unpacking) vs. scalar, 200 trials --\n";
    {
        float max_abs_err = 0.0f;
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<float> data(32), x(32);
            for (auto& v : data) v = dist(rng);
            for (auto& v : x) v = dist(rng);
            BlockQ4 b = quantize_q4(data.data());

            float scalar_result = scalar_dot_q4(b, x.data());
            float neon_result = neon_dot_q4(b, x.data());
            max_abs_err = std::max(max_abs_err, std::fabs(scalar_result - neon_result));
        }
        std::cout << "  Max absolute error over 200 trials: " << std::scientific << std::setprecision(2)
                  << max_abs_err << std::defaultfloat << std::setprecision(6) << "\n";
        CHECK(max_abs_err < 1e-3f);
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): assuming NEON and AVX2 intrinsics map one to
    // one. NEON has no single "widen int8 straight to int32" instruction
    // the way AVX2's _mm256_cvtepi8_epi32 does -- it only widens one
    // power-of-two step at a time (int8 -> int16 via vmovl_s8, then
    // int16 -> int32 via vmovl_s16), so a direct one-line port of the
    // AVX2 kernel does not compile, and the workaround is not optional
    // boilerplate but a genuinely different number of steps.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: NEON has no int8-to-int32 widen in one step --\n";
    {
        int8x8_t some_int8 = vdup_n_s8(5);
        int16x8_t widened_to_16 = vmovl_s8(some_int8);   // one step: int8x8 -> int16x8
        int32x4_t widened_to_32 = vmovl_s16(vget_low_s16(widened_to_16));  // a SECOND step: int16x4 -> int32x4

        int16_t check16[8]; vst1q_s16(check16, widened_to_16);
        int32_t check32[4]; vst1q_s32(check32, widened_to_32);

        std::cout << "  int8 value 5, widened to int16: " << check16[0]
                  << "  (one vmovl_s8 call, doubles the lane width from 8 to 16 bits)\n";
        std::cout << "  the same value, widened again to int32: " << check32[0]
                  << "  (a SECOND vmovl_s16 call was required -- there is no vmovl_s8-to-s32)\n";
        std::cout << "  AVX2's _mm256_cvtepi8_epi32 does both of these steps in a single instruction;\n";
        std::cout << "  porting AVX2 intrinsic-for-intrinsic to NEON silently drops a required step\n";
        std::cout << "  rather than merely renaming one, which is why Sections 9.2 and 9.3 use a\n";
        std::cout << "  visibly different number of widening calls for the identical logical operation.\n";

        CHECK(check16[0] == 5);
        CHECK(check32[0] == 5);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
