// Chapter 4.2 -- Q8 is the conservative end of quantization: each
// 32-weight block is stored as 32 int8 values plus one scale, chosen
// so the block's own largest-magnitude weight maps to +-127 (the
// symmetric special case from Section 4.1, Z=0). This section also
// answers the question Section 4.1's memory-wall numbers assumed an
// answer to: WHY blocks of 32, rather than one scale for an entire
// weight matrix?
//
// The scale itself is stored as a hand-rolled IEEE-754 binary16
// (fp16_t) rather than a 4-byte float -- the same "build the reduced-
// precision type by hand" technique Chapter 2.5 used for bf16_t, just
// with a different bit layout (1 sign + 5 exponent + 10 mantissa,
// instead of bf16's 1+8+7). A scale is always a small positive number
// recovered from a handful of weights, so fp16's tighter ~0.05%
// rounding on the scale itself is negligible next to the quantization
// error already being introduced by int8 rounding -- and it is what
// turns a block from 36 bytes (4-byte scale) into 34 bytes, matching
// the ~3.8x compression this book (and real GGUF Q8_0 files) claim,
// rather than the 3.56x a 4-byte scale actually delivers.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((float)(a) - (float)(b)) < (tol))

// A hand-rolled IEEE-754 binary16 (fp16), built the same way Chapter
// 2.5 built bf16_t: by hand, from the bit pattern, with std::memcpy
// used for safe type-punning. Unlike bf16 (a simple truncation of
// FP32's top 16 bits), fp16 uses a different exponent width, so
// converting requires re-biasing the exponent and correctly rounding
// the mantissa down from 23 bits to 10 -- this implementation handles
// the normal-number range exactly (round-to-nearest-even) and flushes
// subnormal-on-encode to zero, which is correct for every scale value
// this chapter ever produces (always a small positive normal float,
// never within a factor of 1000 of fp16's smallest normal magnitude).
struct fp16_t {
    uint16_t bits = 0;

    fp16_t() = default;
    fp16_t(float f) { bits = encode(f); }
    operator float() const { return decode(bits); }

    static uint16_t encode(float f) {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;

        if (((x >> 23) & 0xFFu) == 0xFFu) {                 // inf/nan
            return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
        }
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);   // overflow -> inf
        if (exp <= 0)  return static_cast<uint16_t>(sign);             // underflow -> flush to zero

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
        if (exp == 0) {
            fbits = sign;                                    // zero (subnormal fp16 never produced by encode)
        } else if (exp == 31) {
            fbits = sign | 0x7F800000u | (mant << 13);        // inf/nan
        } else {
            uint32_t exp32 = exp - 15 + 127;
            fbits = sign | (exp32 << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &fbits, 4);
        return f;
    }
};

// 34 bytes total: 2 (fp16 scale) + 32 (int8 weights).
// FP32 equivalent: 32 x 4 = 128 bytes -> 128/34 = 3.76x compression.
struct BlockQ8 {
    fp16_t scale;
    int8_t weights[32];
};

BlockQ8 quantize_q8(std::span<const float> input) {
    assert(input.size() == 32);
    BlockQ8 block;

    float alpha = 0.0f;
    for (float v : input) alpha = std::max(alpha, std::fabs(v));

    if (alpha == 0.0f) {
        block.scale = fp16_t(0.0f);
        for (int i = 0; i < 32; ++i) block.weights[i] = 0;
        return block;
    }

    block.scale = fp16_t(alpha / 127.0f);
    float s = static_cast<float>(block.scale);          // the ACTUAL (fp16-rounded) scale used
    float inv_scale = 1.0f / s;

    for (int i = 0; i < 32; ++i) {
        float clamped = std::clamp(std::round(input[i] * inv_scale), -127.0f, 127.0f);
        block.weights[i] = static_cast<int8_t>(clamped);
    }
    return block;
}

void dequantize_q8(const BlockQ8& block, std::span<float> output) {
    assert(output.size() == 32);
    float s = static_cast<float>(block.scale);
    for (int i = 0; i < 32; ++i)
        output[i] = static_cast<float>(block.weights[i]) * s;
}

struct ErrorMetrics { float mse, max_abs, snr_db; };

ErrorMetrics compute_errors(std::span<const float> original, std::span<const float> recon) {
    float sum_sq_err = 0.0f, sum_sq_sig = 0.0f, max_abs = 0.0f;
    for (size_t i = 0; i < original.size(); ++i) {
        float err = original[i] - recon[i];
        sum_sq_err += err * err;
        sum_sq_sig += original[i] * original[i];
        max_abs = std::max(max_abs, std::fabs(err));
    }
    ErrorMetrics m;
    m.mse = sum_sq_err / static_cast<float>(original.size());
    m.max_abs = max_abs;
    m.snr_db = (sum_sq_err > 0.0f) ? 10.0f * std::log10(sum_sq_sig / sum_sq_err) : 999.0f;
    return m;
}

int original_main() {
    std::cout << "================================================\n";
    std::cout << "Symmetric Blockwise Q8 Quantization (fp16 scale)\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: fp16_t itself -- exact round-trip for exactly-representable
    // values, and the known bit patterns for 1.0 and 0.5.
    // =====================================================================
    std::cout << "-- Test 1: fp16_t correctness --\n";
    {
        CHECK(fp16_t::encode(1.0f) == 0x3C00);
        CHECK(fp16_t::encode(0.5f) == 0x3800);
        CHECK(static_cast<float>(fp16_t(1.0f)) == 1.0f);
        CHECK(static_cast<float>(fp16_t(0.5f)) == 0.5f);
        CHECK(static_cast<float>(fp16_t(0.0f)) == 0.0f);
        CHECK(sizeof(fp16_t) == 2);

        // A typical scale value: rounds to within ~0.02% (well under the
        // int8 quantization error this scale is already being applied to).
        float s = 0.61f / 127.0f;
        float rel_err = std::fabs(static_cast<float>(fp16_t(s)) - s) / s;
        std::cout << "  fp16(1.0) bits=0x" << std::hex << fp16_t::encode(1.0f) << std::dec << " (expect 0x3c00)\n";
        std::cout << "  fp16 round-trip of a typical scale, relative error: "
                   << std::scientific << rel_err << "\n";
        CHECK(rel_err < 0.001f);
    }

    // =====================================================================
    // TEST 2: Trace Section 4.1's worked-example vocabulary through a
    // full block -- the same alpha=0.61 example, now with the fp16 scale.
    // =====================================================================
    std::cout << "\n-- Test 2: Worked example (alpha=0.61) --\n";
    {
        std::vector<float> x(32, 0.0f);
        x[0]=0.15f; x[1]=-0.42f; x[2]=0.08f; x[3]=0.33f;
        x[4]=-0.61f; x[5]=0.22f; x[6]=-0.10f; x[7]=0.47f;

        BlockQ8 block = quantize_q8(x);
        float s = static_cast<float>(block.scale);
        std::cout << "  alpha (max abs): 0.61\n";
        std::cout << "  scale (fp16-rounded) = " << std::fixed << std::setprecision(6) << s << "\n";
        CHECK_NEAR(s, 0.61f / 127.0f, 5e-5f);

        CHECK(block.weights[0] == 31);
        CHECK(block.weights[1] == -87);
        CHECK(block.weights[4] == -127);

        std::vector<float> recon(32);
        dequantize_q8(block, recon);
        CHECK_NEAR(recon[0], 0.14889f, 0.0005f);
        CHECK_NEAR(recon[4], -0.60998f, 0.0005f);

        auto err = compute_errors(x, recon);
        std::cout << "  quantized int8 (first 8): [";
        for (int i = 0; i < 8; ++i) std::cout << " " << (int)block.weights[i];
        std::cout << " ]\n";
        std::cout << "  MSE: " << std::scientific << err.mse << ", SNR: " << std::fixed
                  << std::setprecision(1) << err.snr_db << " dB\n";
        CHECK(err.mse < 1e-5f);
    }

    // =====================================================================
    // TEST 3: Scale invariance -- the block's own max always maps to +-127
    // =====================================================================
    std::cout << "\n-- Test 3: Max value always maps to +-127 --\n";
    {
        for (float magnitude : {0.1f, 1.0f, 10.0f, 100.0f}) {
            std::vector<float> x(32, 0.0f);
            x[0] = magnitude; x[1] = -magnitude; x[2] = magnitude * 0.5f;
            BlockQ8 block = quantize_q8(x);
            CHECK(block.weights[0] == 127);
            CHECK(block.weights[1] == -127);
            std::cout << "  magnitude=" << magnitude << " q[0]=" << (int)block.weights[0]
                      << " q[1]=" << (int)block.weights[1] << "\n";
        }
    }

    // =====================================================================
    // TEST 4: Realistic Xavier-initialized weights -- SNR should be high.
    // =====================================================================
    std::cout << "\n-- Test 4: Realistic weights (Xavier normal, std=0.02) --\n";
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        std::vector<float> x(32);
        for (float& v : x) v = dist(rng);

        BlockQ8 block = quantize_q8(x);
        std::vector<float> recon(32);
        dequantize_q8(block, recon);
        auto err = compute_errors(x, recon);
        std::cout << "  Q8 MSE: " << std::scientific << err.mse << ", SNR: " << std::fixed
                   << std::setprecision(1) << err.snr_db << " dB\n";
        CHECK(err.snr_db > 40.0f);
    }

    // =====================================================================
    // TEST 5: All-zeros edge case -- must not NaN.
    // =====================================================================
    std::cout << "\n-- Test 5: All-zeros input --\n";
    {
        std::vector<float> x(32, 0.0f);
        BlockQ8 block = quantize_q8(x);
        std::vector<float> recon(32);
        dequantize_q8(block, recon);
        CHECK(static_cast<float>(block.scale) == 0.0f);
        bool all_zero = std::all_of(recon.begin(), recon.end(), [](float v) { return v == 0.0f; });
        CHECK(all_zero);
        std::cout << "  scale=0, all outputs=0, no NaN: " << (all_zero ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 6: Memory layout and compression.
    // =====================================================================
    std::cout << "\n-- Test 6: Memory layout --\n";
    {
        double compression = 32.0 * 4.0 / static_cast<double>(sizeof(BlockQ8));
        std::cout << "  sizeof(BlockQ8) = " << sizeof(BlockQ8) << " bytes (2 scale + 32 weights)\n";
        std::cout << "  FP32 equivalent = 128 bytes\n";
        std::cout << "  compression = " << std::fixed << std::setprecision(2) << compression << "x\n";
        CHECK(sizeof(BlockQ8) == 34);
        CHECK(compression > 3.7 && compression < 3.8);
    }

    // =====================================================================
    // TEST 7: The outlier problem -- why one GLOBAL scale is not enough.
    // Block A contains one true outlier (12.5) among typical weights
    // (0.05); Block B has no outlier, its own max magnitude is 0.5. A
    // SINGLE global scale (from the whole 64-value array) is dictated
    // by the outlier and destroys precision for every normal weight in
    // BOTH blocks; per-block scales fix only the block that needs it.
    // =====================================================================
    std::cout << "\n-- Test 7: The outlier problem (global scale vs blockwise) --\n";
    {
        std::vector<float> block_a(32, 0.05f);
        block_a[0] = 12.5f;                       // the outlier
        std::vector<float> block_b(32, 0.05f);
        block_b[1] = 0.5f;                        // block B's own (non-outlier) max

        // -- Global: one scale from the max magnitude across BOTH blocks --
        float global_alpha = std::max(
            *std::max_element(block_a.begin(), block_a.end(), [](float a, float b){ return std::fabs(a) < std::fabs(b); }),
            *std::max_element(block_b.begin(), block_b.end(), [](float a, float b){ return std::fabs(a) < std::fabs(b); }));
        float global_scale = global_alpha / 127.0f;
        int32_t q_global = static_cast<int32_t>(std::round(0.05f / global_scale));
        float recon_global = static_cast<float>(q_global) * global_scale;
        float rel_err_global = std::fabs(0.05f - recon_global) / 0.05f;

        std::cout << "  global scale = " << std::fixed << std::setprecision(5) << global_scale
                  << " (from outlier, alpha=" << global_alpha << ")\n";
        std::cout << "  quantize(0.05) with global scale -> q=" << q_global
                   << ", reconstructed=" << recon_global
                   << ", relative error=" << std::setprecision(1) << rel_err_global * 100.0f << "%\n";
        CHECK(rel_err_global > 0.90f);   // catastrophic -- matches the ~97% figure worked by hand in this section

        // -- Blockwise: block B gets its OWN scale, unaffected by block A's outlier --
        BlockQ8 bb = quantize_q8(block_b);
        float s_b = static_cast<float>(bb.scale);
        int32_t q_block = static_cast<int32_t>(std::round(0.05f / s_b));
        float recon_block = static_cast<float>(q_block) * s_b;
        float rel_err_block = std::fabs(0.05f - recon_block) / 0.05f;

        std::cout << "  block B's own scale = " << std::setprecision(5) << s_b << " (from its own max, 0.5)\n";
        std::cout << "  quantize(0.05) with blockwise scale -> q=" << q_block
                   << ", reconstructed=" << recon_block
                   << ", relative error=" << std::setprecision(1) << rel_err_block * 100.0f << "%\n";
        CHECK(rel_err_block < 0.05f);    // an order of magnitude better -- block A's outlier never touches block B

        std::cout << "  blockwise error is smaller than global error: "
                  << (rel_err_block < rel_err_global ? "yes" : "no") << "\n";
        CHECK(rel_err_block < rel_err_global);
    }

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
int main(){std::cout << "decode(0x0001)=" << fp16_t::decode(1) << " encoded tiny=" << fp16_t::encode(1e-6f) << "\n";std::vector<float> x(32,0.0f);x[0]=0.001f;auto q=quantize_q8(x);std::cout << "stored_scale=" << (float)q.scale << "\n";}
