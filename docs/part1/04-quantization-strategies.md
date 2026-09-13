# Chapter 4: Quantization Strategies -- Affine Math, Symmetric vs. Asymmetric, Blockwise Scales

**What you will understand by the end of this chapter:**

- Why LLM inference is memory-bandwidth-bound rather than compute-bound, and how that single fact turns "make the weights smaller" into "make inference faster" — not just cheaper to store.
- The one formula, affine quantization, that every scheme in this chapter is a special case of, and exactly when its zero-point can be forced to zero (a pure multiply on the hot path) versus when it genuinely cannot.
- Why one global scale for an entire weight matrix is not just imprecise but actively destructive in the presence of outliers, and how blockwise scales fix this by localizing the damage to the block that actually has the outlier.
- How to pack two 4-bit values into a single byte for Q4_0, and how a fused dequantize-and-dot-product kernel avoids ever writing a dequantized float to memory at all.
- How weights (quantized once, offline) and activations (quantized every token, at runtime) need different quantization strategies, and how INT8 x INT8 matrix multiplication turns that runtime cost into a throughput win on real hardware.
- How Quantization-Aware Training uses the Straight-Through Estimator to work around round()'s zero gradient, letting a model's weights adapt to quantization noise instead of merely tolerating it after the fact.

**What you need to know first:**

- Chapter 2's vocabulary: `std::span` for flat, one-dimensional views (every quantized block in this chapter is fundamentally 1D), and the hand-rolled reduced-precision float technique introduced there for `bf16_t`.
- Ordinary C++ bit manipulation (shifts, masks), fixed-width integer types (`int8_t`, `uint8_t`), and enough statistics to be comfortable with mean squared error and a normal distribution.

---

Chapters 2 and 3 built and ran a transformer block in full precision. This chapter exists because full precision does not fit: a 70-billion-parameter model stored as FP32 needs 280GB of RAM before a single token generates, and even on server-class hardware, every one of those bytes has to be read from memory for every token produced, at whatever the memory bus's bandwidth ceiling happens to be. Quantization is the answer — storing weights (and, at runtime, activations) as small integers with a shared floating-point scale — and this chapter builds that answer from first principles: the single affine formula every scheme reduces to, why blocks of 32 values need their own independent scale rather than sharing one, how to pack 4-bit values into bytes by hand, how to fuse dequantization directly into a dot product's inner loop, and how a model can be trained to expect the quantization noise it will eventually run under, using a technique — the Straight-Through Estimator — that fixes a broken gradient by simply pretending it isn't broken.

## 4.1 The Memory Wall and Affine Quantization

### Intuition

A modern CPU's arithmetic units can perform billions of multiply-adds per second, but a next-token prediction with a large model spends almost none of that capacity waiting on math — it spends its time waiting for weights to arrive from RAM. Generating one token from an 8-billion-parameter model in FP32 requires reading all 32GB of its weights at least once; at a representative desktop memory bandwidth of 64 GB/s, that is a hard floor of half a second per token no matter how fast the CPU's multiply-add units are. Halving the bytes read per token does not just save disk space — under this memory-bandwidth-bound regime, it roughly doubles tokens per second.

### The Concept, In Detail

Every quantization scheme in this chapter maps a real number `f` to an integer `q` with two numbers: a scale `S` (how many real units one integer step represents) and a zero-point `Z` (which integer represents the real value 0.0). The general formula is `q = clamp(round(f / S) + Z, q_min, q_max)` to quantize, and `f' = S * (q - Z)` to dequantize. When the values being quantized are roughly symmetric around zero — which trained weights usually are, since nothing biases them toward being systematically positive or negative — `Z` can be forced to exactly 0, and dequantization collapses to a pure multiply: `f' = S * q`, no subtraction anywhere in the hot loop. When the values are NOT centered on zero — a common shape for post-activation values, which are often one-sided — forcing `Z = 0` would waste representable integer codes on a side of the range nothing ever uses, so `Z` must be chosen so that the real value 0.0 actually maps to some achievable integer, and dequantization genuinely needs the subtraction.

This section also turns the "inference gets faster" claim into a computed quantity rather than an assumed one. Given a parameter count and a bandwidth figure, `tokens_per_sec ~= bandwidth / bytes_read_per_token` is a first-order analytical model — deterministic, reproducible on any machine, and not a substitute for an actual measured benchmark, but exactly the right tool for demonstrating a structural relationship (fewer bytes per token implies more tokens per second) without needing this specific machine's clock speed to make the point.

### Code and Verification

```cpp
// Chapter 4.1 -- every quantization scheme in this chapter (Q8, Q4_0,
// per-token activation quantization) is one special case of a single
// formula: affine (linear) quantization. A real number f is mapped to
// an integer q by a scale S and a zero-point Z:
//
//     q = clamp(round(f / S) + Z, q_min, q_max)      (quantize)
//     f' = S * (q - Z)                                 (dequantize)
//
// S is "how many real units does one integer step represent"; Z is
// "which integer represents the real value 0.0". When the real-valued
// range being quantized straddles zero symmetrically (as trained
// weights usually do), Z can be forced to 0, and dequantization
// collapses to a pure multiply -- no subtraction in the hot loop. When
// the range is NOT centered on zero (post-activation values are a
// common case), forcing Z=0 wastes representable codes on one side of
// the range, so Z must be chosen to actually land on 0.0.
//
// This file also makes the chapter's other headline claim -- that
// LLM inference is memory-bandwidth-bound, so a smaller on-disk/in-RAM
// weight format directly buys more tokens/second -- into a genuinely
// computed quantity instead of an asserted one. Per this book's own
// policy (see Getting Started), a real, reproducible arithmetic model
// is used in place of a measured benchmark: the formula
// tokens_per_sec = bandwidth / bytes_read_per_token is a first-order
// ANALYTICAL projection, clearly computed from stated inputs, not a
// wall-clock measurement -- there is no hardware-dependent timing
// anywhere in this file.

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <iomanip>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((float)(a) - (float)(b)) < (tol))

// The one formula every quantization scheme in this chapter reduces to.
// q_min/q_max bound the integer storage type (e.g. -127/127 for a
// symmetric int8, 0/255 for an unsigned int8).
int32_t affine_quantize(float f, float S, int32_t Z, int32_t q_min, int32_t q_max) {
    float scaled = std::round(f / S) + static_cast<float>(Z);
    float clamped = std::clamp(scaled, static_cast<float>(q_min), static_cast<float>(q_max));
    return static_cast<int32_t>(clamped);
}

float affine_dequantize(int32_t q, float S, int32_t Z) {
    return S * static_cast<float>(q - Z);
}

int main() {
    std::cout << "================================================\n";
    std::cout << "Affine Quantization: the general formula\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: The symmetric special case (Z = 0)
    // A weight value with scale derived from a block's max magnitude.
    // With Z=0, affine_dequantize collapses to a pure multiply.
    // =====================================================================
    std::cout << "-- Test 1: Symmetric case (Z = 0) --\n";
    {
        const float S = 0.003137f;
        const int32_t Z = 0;
        const float f = 0.153f;

        int32_t q = affine_quantize(f, S, Z, -127, 127);
        float recon = affine_dequantize(q, S, Z);

        std::cout << "  S=" << S << " Z=" << Z << " f=" << f << "\n";
        std::cout << "  quantized q = " << q << "\n";
        std::cout << "  dequantized f' = " << std::fixed << std::setprecision(5) << recon << "\n";

        CHECK(q == 49);
        CHECK_NEAR(recon, 0.15371f, 0.0001f);
        // With Z=0, dequantize is a pure multiply: S*(q-0) == S*q exactly.
        CHECK(affine_dequantize(q, S, Z) == S * static_cast<float>(q));
    }

    // =====================================================================
    // TEST 2: The asymmetric case (Z != 0) -- a range NOT centered on 0
    // A plausible post-activation range [-0.5, 5.5] (e.g. after a GELU),
    // stored in an UNSIGNED 8-bit code [0, 255]. Every constant below
    // is chosen so the arithmetic lands on exact integers -- this is a
    // worked example designed for hand-verification, not random data.
    // =====================================================================
    std::cout << "\n-- Test 2: Asymmetric case (Z != 0), range [-0.5, 5.5] --\n";
    {
        const float x_min = -0.5f, x_max = 5.5f;
        const int32_t q_min = 0, q_max = 255;
        const float S = (x_max - x_min) / static_cast<float>(q_max - q_min);   // 6.0 / 255
        const int32_t Z = static_cast<int32_t>(std::round(static_cast<float>(q_min) - x_min / S));

        std::cout << "  S = " << std::setprecision(6) << S << ", Z = " << Z << "\n";
        CHECK_NEAR(S, 6.0f / 255.0f, 1e-6f);
        CHECK(Z == 21);

        // The defining property of the zero-point: quantizing the real
        // value 0.0 must land exactly on the integer Z.
        int32_t q_zero = affine_quantize(0.0f, S, Z, q_min, q_max);
        CHECK(q_zero == Z);

        // The range endpoints must land exactly on q_min and q_max.
        int32_t q_at_min = affine_quantize(x_min, S, Z, q_min, q_max);
        int32_t q_at_max = affine_quantize(x_max, S, Z, q_min, q_max);
        CHECK(q_at_min == q_min);
        CHECK(q_at_max == q_max);

        // An interior value, quantized and dequantized back exactly
        // (chosen so 2.0 / S is an exact integer -- no rounding involved).
        int32_t q_mid = affine_quantize(2.0f, S, Z, q_min, q_max);
        float recon_mid = affine_dequantize(q_mid, S, Z);
        std::cout << "  quantize(0.0) = " << q_zero << " (= Z)\n";
        std::cout << "  quantize(-0.5) = " << q_at_min << " (= q_min)\n";
        std::cout << "  quantize(5.5) = " << q_at_max << " (= q_max)\n";
        std::cout << "  quantize(2.0) = " << q_mid << ", dequantize -> " << recon_mid << "\n";
        CHECK(q_mid == 106);
        CHECK_NEAR(recon_mid, 2.0f, 1e-4f);

        // Unlike the symmetric case, dequantizing here genuinely needs
        // the subtraction: S*q alone (dropping Z) gives the wrong answer.
        CHECK(std::fabs(S * static_cast<float>(q_mid) - 2.0f) > 0.3f);
    }

    // =====================================================================
    // TEST 3: The memory wall -- a computed (not copied-from-a-table)
    // compression and bandwidth model for three real model sizes.
    // Effective bytes/parameter include each scheme's own scale
    // overhead for a block size of 32, using a 2-byte (fp16) scale --
    // Section 4.2 builds exactly this block layout in code.
    // =====================================================================
    std::cout << "\n-- Test 3: The memory wall (computed, not measured) --\n";
    {
        struct Format { const char* name; double bytes_per_param; };
        constexpr int BLOCK = 32;
        constexpr double SCALE_BYTES = 2.0;   // fp16 scale, one per block
        Format formats[] = {
            {"FP32", 4.0},
            {"FP16/BF16", 2.0},
            {"Q8  (block=32)", (32.0 * 1.0 + SCALE_BYTES) / BLOCK},
            {"Q4_0(block=32)", (32.0 * 0.5 + SCALE_BYTES) / BLOCK},
        };

        struct Model { const char* name; double params_billion; };
        Model models[] = {{"Llama-3 8B", 8.0}, {"Llama-3 70B", 70.0}, {"Llama-3 405B", 405.0}};

        double gb_8b_fp32 = 0.0, gb_8b_q4 = 0.0;
        for (const auto& m : models) {
            std::cout << "  " << m.name << ":\n";
            for (const auto& fmt : formats) {
                double gb = m.params_billion * 1e9 * fmt.bytes_per_param / 1e9;
                std::cout << "    " << std::left << std::setw(15) << fmt.name
                          << std::right << std::fixed << std::setprecision(2)
                          << std::setw(8) << gb << " GB\n";
                if (std::string(m.name) == "Llama-3 8B" && std::string(fmt.name) == "FP32") gb_8b_fp32 = gb;
                if (std::string(m.name) == "Llama-3 8B" && std::string(fmt.name) == "Q4_0(block=32)") gb_8b_q4 = gb;
            }
        }

        // An 8B model in FP32 needs ~32GB -- it will not fit in 16GB of
        // RAM. The same model in Q4_0 needs a bit over 4GB -- it does.
        CHECK(gb_8b_fp32 > 16.0);
        CHECK(gb_8b_q4 < 16.0);
        std::cout << "  Llama-3 8B fits in 16GB RAM at FP32: " << (gb_8b_fp32 < 16.0 ? "yes" : "no") << "\n";
        std::cout << "  Llama-3 8B fits in 16GB RAM at Q4_0: " << (gb_8b_q4 < 16.0 ? "yes" : "no") << "\n";

        // Analytical bandwidth model (NOT a measured benchmark): for a
        // memory-bandwidth-bound decode step, tokens/sec is bounded by
        // bandwidth / bytes-of-weights-read-per-token. Using every
        // parameter once per token (the single-batch decode case) and
        // a representative DDR5 figure of 64 GB/s.
        const double BANDWIDTH_GBPS = 64.0;
        double bytes_per_token_fp32 = 8.0e9 * 4.0;
        double bytes_per_token_q8   = 8.0e9 * ((32.0 + SCALE_BYTES) / BLOCK);
        double bytes_per_token_q4   = 8.0e9 * ((16.0 + SCALE_BYTES) / BLOCK);

        double tok_s_fp32 = BANDWIDTH_GBPS * 1e9 / bytes_per_token_fp32;
        double tok_s_q8   = BANDWIDTH_GBPS * 1e9 / bytes_per_token_q8;
        double tok_s_q4   = BANDWIDTH_GBPS * 1e9 / bytes_per_token_q4;

        std::cout << "\n  Analytical tokens/sec model at " << BANDWIDTH_GBPS
                  << " GB/s (Llama-3 8B, memory-bandwidth-bound decode):\n";
        std::cout << "    FP32: " << std::setprecision(1) << tok_s_fp32 << " tok/s\n";
        std::cout << "    Q8:   " << tok_s_q8 << " tok/s\n";
        std::cout << "    Q4_0: " << tok_s_q4 << " tok/s\n";

        // Structural fact, not a measurement: halving the bytes read
        // per token roughly doubles the projected tokens/sec.
        CHECK(tok_s_q4 > tok_s_q8);
        CHECK(tok_s_q8 > tok_s_fp32);
        CHECK(tok_s_q4 > tok_s_fp32 * 3.0);   // FP32 is 4 bytes/param, Q4_0 effective ~0.5625 -- over 7x fewer bytes
    }

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_affine_quantization.cpp -o 01_affine_quantization
./01_affine_quantization
```

**Sample input:** the symmetric worked example (`S=0.003137`, `Z=0`, quantizing `0.153`), a hand-derived asymmetric example over the range `[-0.5, 5.5]` where every constant is chosen to land on exact integers, and a computed (not copied-from-a-table) memory-wall model across three real model sizes (8B/70B/405B parameters) and four storage formats.

```text
================================================
Affine Quantization: the general formula
================================================

-- Test 1: Symmetric case (Z = 0) --
  S=0.003137 Z=0 f=0.153
  quantized q = 49
  dequantized f' = 0.15371

-- Test 2: Asymmetric case (Z != 0), range [-0.5, 5.5] --
  S = 0.023529, Z = 21
  quantize(0.0) = 21 (= Z)
  quantize(-0.5) = 0 (= q_min)
  quantize(5.5) = 255 (= q_max)
  quantize(2.0) = 106, dequantize -> 2.000000

-- Test 3: The memory wall (computed, not measured) --
  Llama-3 8B:
    FP32              32.00 GB
    FP16/BF16         16.00 GB
    Q8  (block=32)     8.50 GB
    Q4_0(block=32)     4.50 GB
  Llama-3 70B:
    FP32             280.00 GB
    FP16/BF16        140.00 GB
    Q8  (block=32)    74.38 GB
    Q4_0(block=32)    39.38 GB
  Llama-3 405B:
    FP32            1620.00 GB
    FP16/BF16        810.00 GB
    Q8  (block=32)   430.31 GB
    Q4_0(block=32)   227.81 GB
  Llama-3 8B fits in 16GB RAM at FP32: no
  Llama-3 8B fits in 16GB RAM at Q4_0: yes

  Analytical tokens/sec model at 64.00 GB/s (Llama-3 8B, memory-bandwidth-bound decode):
    FP32: 2.0 tok/s
    Q8:   7.5 tok/s
    Q4_0: 14.2 tok/s

================================================
16/16 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] Z=0 is a choice, not a default"
    It is tempting to always force the zero-point to 0 because the resulting math is simpler and faster. That is exactly right for weights, which cluster around zero — but applying it to a one-sided range (post-ReLU activations, for instance) throws away precision: half of the representable integer codes end up mapping to real values that never occur. Section 4.5's activation quantizer keeps `Z=0` because it targets a symmetric range by construction; a general-purpose activation quantizer over an arbitrary range would need the full asymmetric formula this section derives.

## 4.2 Symmetric Blockwise Q8: Taming the Outlier Problem

### Intuition

A single scale for an entire weight matrix sounds simpler than one scale per block of 32 — fewer numbers to store, fewer branches in the code. It fails in practice because trained weight matrices are not uniform: 99.9% of the values in a typical layer might cluster between -0.5 and +0.5, while a handful of outliers reach magnitudes 20-30x larger. One global scale has to accommodate the outlier, which means every ordinary weight is quantized against a scale far coarser than it needs — in the worst case, rounding small weights to exactly zero. Giving each block of 32 weights its own scale means an outlier in one block can no longer damage the precision of weights sitting in a different block.

### The Concept, In Detail

Q8 quantization is the direct, symmetric application of Section 4.1's general formula with `Z=0`: for each block of 32 weights, find `alpha`, the block's own largest-magnitude value, set `scale = alpha / 127`, and quantize each weight to the signed int8 range `[-127, 127]` (not `[-128, 127]` — clamping to `-127` keeps the range symmetric, so `|-127| == |+127|` exactly). The scale itself does not need to be a full 4-byte float: it is always a small positive number recovered from 32 weights, so storing it as a hand-rolled 2-byte IEEE-754 binary16 (`fp16_t`, built the same way Chapter 2.5 built `bf16_t` — by hand, from the bit pattern, just with a different exponent width) rounds the scale by at most a few hundredths of a percent, a rounding error dwarfed by the int8 quantization error already being introduced. That difference is not cosmetic: a 4-byte scale makes a Q8 block 36 bytes (3.56x compression over FP32); a 2-byte scale makes it 34 bytes (3.76x) — which is what real GGUF Q8_0 files, and this chapter's own memory-wall numbers, actually assume.

The outlier problem this section opened with is not hypothetical. Concretely: if one block contains a true outlier of magnitude 12.5 among otherwise-typical weights near 0.05, a single global scale derived from that outlier (`12.5/127 ~= 0.0984`) reconstructs a weight of 0.05 as roughly 0.098 — almost double its true value, a relative error near 97%. A block that does NOT contain that outlier, quantized with its own scale (derived from its own, much smaller, maximum magnitude), reconstructs the same 0.05 weight to within a few percent. The damage is entirely localized to the block that actually has the outlier.

### Code and Verification

```cpp
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

int main() {
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_symmetric_q8_quantizer.cpp -o 02_symmetric_q8_quantizer
./02_symmetric_q8_quantizer
```

**Sample input:** the hand-worked 8-value example from this section (`alpha=0.61`), a scale-invariance sweep across four orders of magnitude, a realistic Xavier-initialized 32-weight block, the all-zeros edge case, a `sizeof(BlockQ8)` memory-layout check, and a direct global-scale-vs-blockwise-scale comparison reproducing this section's own outlier numbers.

```text
================================================
Symmetric Blockwise Q8 Quantization (fp16 scale)
================================================

-- Test 1: fp16_t correctness --
  fp16(1.0) bits=0x3c00 (expect 0x3c00)
  fp16 round-trip of a typical scale, relative error: 9.278034e-05

-- Test 2: Worked example (alpha=0.61) --
  alpha (max abs): 0.61
  scale (fp16-rounded) = 0.004803
  quantized int8 (first 8): [ 31 -87 17 69 -127 46 -21 98 ]
  MSE: 3.936784e-07, SNR: 48.8 dB

-- Test 3: Max value always maps to +-127 --
  magnitude=0.1 q[0]=127 q[1]=-127
  magnitude=1.0 q[0]=127 q[1]=-127
  magnitude=10.0 q[0]=127 q[1]=-127
  magnitude=100.0 q[0]=127 q[1]=-127

-- Test 4: Realistic weights (Xavier normal, std=0.02) --
  Q8 MSE: 7.5e-09, SNR: 47.1 dB

-- Test 5: All-zeros input --
  scale=0, all outputs=0, no NaN: yes

-- Test 6: Memory layout --
  sizeof(BlockQ8) = 34 bytes (2 scale + 32 weights)
  FP32 equivalent = 128 bytes
  compression = 3.76x

-- Test 7: The outlier problem (global scale vs blockwise) --
  global scale = 0.09843 (from outlier, alpha=12.50000)
  quantize(0.05) with global scale -> q=1, reconstructed=0.09843, relative error=96.9%
  block B's own scale = 0.00394 (from its own max, 0.5)
  quantize(0.05) with blockwise scale -> q=13, reconstructed=0.05118, relative error=2.4%
  blockwise error is smaller than global error: yes

================================================
30/30 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] a plain 4-byte scale silently under-delivers the compression you designed for"
    Storing a block's scale as an ordinary `float` compiles fine, produces correct numbers, and is a completely legitimate design — it is simply a different, less aggressive compression target (3.56x, not 3.76x) than the one this chapter's memory-wall analysis assumes. There is no compiler warning for "this struct is 2 bytes larger than the format you had in mind": the mismatch only shows up as a `sizeof()` that does not match the compression ratio you expected, which is exactly why Test 6 above checks `sizeof(BlockQ8)` directly rather than trusting the arithmetic on paper.

## 4.3 Q4_0: Packing Two Weights Into One Byte

### Intuition

C++ has no 4-bit integer type — the smallest addressable unit any pointer can reference is a byte. Q4_0 packs two 4-bit values ("nibbles") into every byte: an unsigned range `[0, 15]`, reinterpreted as signed `[-8, +7]` by subtracting a fixed offset of 8. Halving int8's 8 bits per weight to 4 roughly doubles the compression again — and is the difference between a 70-billion-parameter model needing 140GB (FP16) versus needing about 20GB (Q4_0), which is the difference between "needs a server" and "runs on a laptop."

### The Concept, In Detail

Packing two logical values `+3` and `-2` into one byte: `+3` is stored as `3 + 8 = 11` (binary `1011`) in the low nibble, `-2` as `-2 + 8 = 6` (binary `0110`) in the high nibble, giving `packed = (6 << 4) | 11 = 0x6B`. Extraction reverses this exactly: `byte & 0x0F` recovers the low nibble, `(byte >> 4) & 0x0F` the high nibble, and subtracting 8 from each undoes the offset. Because the signed range after offsetting is `[-8, +7]` — not symmetric, since 4-bit unsigned only has 16 codes total — the scale is deliberately derived from `alpha / 7`, not `alpha / 8`: this is the same symmetric-quantization discipline as Q8 (the block's maximum magnitude maps to the largest achievable code), just working around 4 bits having one fewer negative-side code than positive-side headroom would suggest.

A Q4_0 block holds 32 weights in 16 bytes of packed nibbles plus one scale. Using the same hand-rolled `fp16_t` from Section 4.2 for that scale (2 bytes, not 4) makes the block genuinely 18 bytes: `128 / 18 ~= 7.1x` compression over FP32 — matching what real GGUF Q4_0 files achieve, and a meaningfully different number from the 20-byte, 6.4x block a plain 4-byte float scale would produce. Because 4 bits gives only 16 distinct levels against int8's 256, Q4 introduces substantially more quantization error than Q8 for the same data — the tradeoff this format is explicitly making in exchange for its far smaller footprint.

### Code and Verification

```cpp
// Chapter 4.3 -- C++ has no 4-bit integer type; the smallest
// addressable unit is a byte. Q4_0 packs two 4-bit "nibbles" into
// each byte: an unsigned range [0, 15], reinterpreted as a signed
// range [-8, +7] by subtracting a fixed offset of 8. Because that
// range is not symmetric (|-8| != |+7|), the scale is derived from
// +-7 (not +-8) so that the block's largest-magnitude weight still
// maps to an achievable code -- the same symmetric-quantization
// discipline Section 4.2 established for int8, one bit narrower.
//
// [COMMON TRAP]: it is tempting to keep the block's scale as a plain
// 4-byte float "for simplicity." Doing so makes a Q4_0 block 20 bytes
// (4 scale + 16 nibbles) -- a real, working format, but only a 6.4x
// compression over FP32, not the 7.1x this chapter's own memory-wall
// numbers assume. Using the fp16_t built in Section 4.2 for the scale
// (2 bytes instead of 4) makes the block genuinely 18 bytes and the
// compression genuinely 7.1x -- matching what real GGUF Q4_0 files do,
// and what let a 70B-parameter model fit in a single MacBook's RAM for
// the first time.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#include <algorithm>
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

// Identical to Section 4.2's fp16_t (each chapter file in this book is
// self-contained, so the small hand-rolled type is repeated here).
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
// 18 bytes: 2 (fp16 scale) + 16 (nibbles, 2 weights/byte, 32 weights total).
struct BlockQ4 {
    fp16_t scale;
    uint8_t nibbles[16];
};
// 34 bytes: 2 (fp16 scale) + 32 (int8 weights) -- Section 4.2's block, for comparison.
struct BlockQ8 {
    fp16_t scale;
    int8_t weights[32];
};
#pragma pack(pop)

BlockQ4 quantize_q4(std::span<const float> input) {
    assert(input.size() == 32);
    BlockQ4 block;

    float alpha = 0.0f;
    for (float v : input) alpha = std::max(alpha, std::fabs(v));

    if (alpha == 0.0f) {
        block.scale = fp16_t(0.0f);
        for (int i = 0; i < 16; ++i) block.nibbles[i] = 0x88;   // both nibbles decode to 8-8=0
        return block;
    }

    // alpha maps to nibble value 15 -> logical +7 (NOT +8 -- see the
    // COMMON TRAP above the class definitions: the 4-bit signed range
    // after the +8 offset is [-8, +7], and using +7 as the symmetric
    // bound is what keeps this a genuinely symmetric (Z=0) scheme).
    block.scale = fp16_t(alpha / 7.0f);
    float s = static_cast<float>(block.scale);
    float inv_scale = 1.0f / s;

    for (int i = 0; i < 16; ++i) {
        int lo_raw = static_cast<int>(std::clamp(std::round(input[2 * i] * inv_scale), -8.0f, 7.0f));
        int hi_raw = static_cast<int>(std::clamp(std::round(input[2 * i + 1] * inv_scale), -8.0f, 7.0f));
        uint8_t lo = static_cast<uint8_t>(lo_raw + 8);
        uint8_t hi = static_cast<uint8_t>(hi_raw + 8);
        block.nibbles[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return block;
}

void dequantize_q4(const BlockQ4& block, std::span<float> output) {
    assert(output.size() == 32);
    float s = static_cast<float>(block.scale);
    for (int i = 0; i < 16; ++i) {
        uint8_t packed = block.nibbles[i];
        int lo = static_cast<int>(packed & 0x0F) - 8;
        int hi = static_cast<int>((packed >> 4) & 0x0F) - 8;
        output[2 * i] = static_cast<float>(lo) * s;
        output[2 * i + 1] = static_cast<float>(hi) * s;
    }
}

BlockQ8 quantize_q8(std::span<const float> input) {
    assert(input.size() == 32);
    BlockQ8 block;
    float alpha = 0.0f;
    for (float v : input) alpha = std::max(alpha, std::fabs(v));
    if (alpha == 0.0f) { block.scale = fp16_t(0.0f); return block; }
    block.scale = fp16_t(alpha / 127.0f);
    float inv = 1.0f / static_cast<float>(block.scale);
    for (int i = 0; i < 32; ++i)
        block.weights[i] = static_cast<int8_t>(std::clamp(std::round(input[i] * inv), -127.0f, 127.0f));
    return block;
}

void dequantize_q8(const BlockQ8& block, std::span<float> output) {
    assert(output.size() == 32);
    float s = static_cast<float>(block.scale);
    for (int i = 0; i < 32; ++i) output[i] = static_cast<float>(block.weights[i]) * s;
}

float mse(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) { float e = a[i] - b[i]; sum += e * e; }
    return sum / static_cast<float>(a.size());
}

int main() {
    std::cout << "================================================\n";
    std::cout << "Q4_0 Bit Packing and Dequantization\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: Nibble pack/unpack mechanics -- exhaustive round-trip.
    // =====================================================================
    std::cout << "-- Test 1: Nibble pack/unpack mechanics --\n";
    {
        uint8_t packed = (4 << 4) | 9;   // lo=9 (logical +1), hi=4 (logical -4)
        int lo = static_cast<int>(packed & 0x0F) - 8;
        int hi = static_cast<int>((packed >> 4) & 0x0F) - 8;
        std::cout << "  packed byte: 0x" << std::hex << (int)packed << std::dec << "\n";
        std::cout << "  lo nibble: " << lo << " (expect +1), hi nibble: " << hi << " (expect -4)\n";
        CHECK(lo == 1);
        CHECK(hi == -4);

        bool all_roundtrip = true;
        for (int v = -8; v <= 7; ++v) {
            uint8_t nibble = static_cast<uint8_t>(v + 8);
            int recovered = static_cast<int>(nibble) - 8;
            if (recovered != v) all_roundtrip = false;
            CHECK(recovered == v);
        }
        std::cout << "  all 16 nibble values [-8,+7] round-trip: " << (all_roundtrip ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 2: Trace this section's own worked example exactly.
    // =====================================================================
    std::cout << "\n-- Test 2: Worked example (8 values, alpha=1.47) --\n";
    {
        std::vector<float> x(32, 0.0f);
        x[0]=0.31f; x[1]=-0.82f; x[2]=0.05f; x[3]=1.47f;
        x[4]=-0.15f; x[5]=0.56f; x[6]=-1.10f; x[7]=0.73f;

        BlockQ4 block = quantize_q4(x);
        float s = static_cast<float>(block.scale);
        std::cout << "  scale = " << std::fixed << std::setprecision(5) << s << " (expected ~0.21000)\n";
        CHECK_NEAR(s, 1.47f / 7.0f, 2e-4f);

        std::cout << "  byte[0] = 0x" << std::hex << (int)block.nibbles[0] << " (expected 0x49)\n";
        std::cout << "  byte[1] = 0x" << (int)block.nibbles[1] << std::dec << " (expected 0xF8)\n";
        CHECK(block.nibbles[0] == 0x49);   // lo=9(+1), hi=4(-4)
        CHECK(block.nibbles[1] == 0xF8);   // lo=8(0), hi=F(+7)

        std::vector<float> recon(32);
        dequantize_q4(block, recon);
        // x[2]=0.05 -> nibble 8 -> (8-8)*scale = 0.00 (error ~0.05)
        CHECK_NEAR(recon[2], 0.00f, 0.001f);
        // x[3]=1.47 -> nibble 15 -> (15-8)*scale ~ 1.47 (the block's own max, near-perfect)
        CHECK_NEAR(recon[3], 1.47f, 0.002f);
        std::cout << "  recon[2]=" << recon[2] << " (orig 0.05), recon[3]=" << recon[3] << " (orig 1.47)\n";
    }

    // =====================================================================
    // TEST 3: Q4 vs Q8 error comparison on realistic weights.
    // =====================================================================
    std::cout << "\n-- Test 3: Q4 vs Q8 error comparison --\n";
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        std::vector<float> x(32);
        for (float& v : x) v = dist(rng);

        BlockQ4 bq4 = quantize_q4(x);
        BlockQ8 bq8 = quantize_q8(x);
        std::vector<float> r4(32), r8(32);
        dequantize_q4(bq4, r4);
        dequantize_q8(bq8, r8);
        float mse4 = mse(x, r4), mse8 = mse(x, r8);

        std::cout << "  Q8 MSE: " << std::scientific << mse8 << "\n";
        std::cout << "  Q4 MSE: " << mse4 << "\n";
        std::cout << "  Q4/Q8 ratio: " << std::fixed << std::setprecision(1) << mse4 / mse8 << "x worse\n";
        CHECK(mse4 > mse8);
        CHECK(mse4 < mse8 * 1000.0f);
    }

    // =====================================================================
    // TEST 4: Memory layout -- the corrected, fp16-scale byte counts.
    // =====================================================================
    std::cout << "\n-- Test 4: Memory layout --\n";
    {
        double q4_compression = 128.0 / static_cast<double>(sizeof(BlockQ4));
        double q8_compression = 128.0 / static_cast<double>(sizeof(BlockQ8));
        std::cout << "  sizeof(BlockQ4) = " << sizeof(BlockQ4) << " bytes (expected 18: 2 scale + 16 nibbles)\n";
        std::cout << "  sizeof(BlockQ8) = " << sizeof(BlockQ8) << " bytes (expected 34: 2 scale + 32 weights)\n";
        std::cout << "  Q4 compression = " << std::fixed << std::setprecision(2) << q4_compression << "x\n";
        std::cout << "  Q8 compression = " << q8_compression << "x\n";
        CHECK(sizeof(BlockQ4) == 18);
        CHECK(sizeof(BlockQ8) == 34);
        CHECK(q4_compression > 7.0 && q4_compression < 7.2);
    }

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_q4_bitpacking.cpp -o 03_q4_bitpacking
./03_q4_bitpacking
```

**Sample input:** an exhaustive round-trip of all 16 nibble values, the 8-value worked example from this section (`alpha=1.47`), a Q4-vs-Q8 mean-squared-error comparison on realistic Xavier-initialized weights, and a `sizeof()` check confirming the 18-byte block layout.

```text
================================================
Q4_0 Bit Packing and Dequantization
================================================

-- Test 1: Nibble pack/unpack mechanics --
  packed byte: 0x49
  lo nibble: 1 (expect +1), hi nibble: -4 (expect -4)
  all 16 nibble values [-8,+7] round-trip: yes

-- Test 2: Worked example (8 values, alpha=1.47) --
  scale = 0.20996 (expected ~0.21000)
  byte[0] = 0x49 (expected 0x49)
  byte[1] = 0xf8 (expected 0xF8)
  recon[2]=0.00000 (orig 0.05), recon[3]=1.46973 (orig 1.47)

-- Test 3: Q4 vs Q8 error comparison --
  Q8 MSE: 7.52351e-09
  Q4 MSE: 2.36364e-06
  Q4/Q8 ratio: 314.2x worse

-- Test 4: Memory layout --
  sizeof(BlockQ4) = 18 bytes (expected 18: 2 scale + 16 nibbles)
  sizeof(BlockQ8) = 34 bytes (expected 34: 2 scale + 32 weights)
  Q4 compression = 7.11x
  Q8 compression = 3.76x

================================================
28/28 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] dividing by 8, not 7, when deriving the Q4_0 scale"
    Since a 4-bit signed value after the offset spans `[-8, +7]`, it is easy to assume the scale should map the block's maximum magnitude to 8 — mirroring Q8's `alpha/127`. But `+8` is not an achievable positive code (the range tops out at `+7`), so `scale = alpha/8` would let the true maximum-magnitude weight round to a code one step past what the format can store, silently clamping it. Deriving the scale from `alpha/7` instead keeps the largest weight exactly representable, at the minor cost of `-8` being one code more precise than it strictly needs to be — an intentional, symmetric-quantization tradeoff, not an oversight.

## 4.4 The Fused Dequantize-Dot-Product

### Intuition

The straightforward way to compute a quantized dot product is two phases: dequantize an entire block to a temporary float buffer, then run an ordinary float dot product against that buffer. It is correct, but it pays for memory traffic that never needed to exist — the dequantized buffer gets written to cache or RAM and then immediately read back, on top of reading the original quantized block and the input vector. A fused kernel dequantizes each value directly into a CPU register, multiplies it against the input immediately, and accumulates — the intermediate float value never touches memory at all.

### The Concept, In Detail

For one Q4_0 block (18 bytes with the fp16 scale from Section 4.3), the naive two-phase approach touches memory four times: read the 18-byte block, WRITE the 128-byte dequantized float buffer, READ that same 128 bytes back, and read the 128-byte input vector slice — 402 bytes of traffic for 32 multiply-accumulates. The fused version touches memory twice: read the 18-byte block, read the 128-byte input slice — 146 bytes, a genuine 2.75x reduction, with the block's scale applied exactly once per block (`block_sum * s`) rather than once per weight, since scalar multiplication distributes over the sum. This is a claim about memory TRAFFIC, and it is verified here the same way this book verified the compute graph's memory-planning claim in Chapter 3.5: by literally accounting the bytes each access pattern touches, not by racing two implementations against a clock. A real hardware benchmark would also have to account for cache state, out-of-order execution, and how aggressively the compiler auto-vectorizes each version — all of which vary by machine, and none of which change the byte count a fixed algorithm is defined to touch.

### Code and Verification

```cpp
// Chapter 4.4 -- a naive quantized dot product runs in two phases:
// dequantize a whole block to a temporary float buffer, then dot that
// buffer against the input vector. That temporary buffer is a real,
// measurable memory-traffic cost: it gets WRITTEN to cache/RAM after
// dequantizing, then READ BACK for the dot product -- memory traffic
// a plain FP32 dot product never pays. A FUSED dot product dequantizes
// each nibble directly into a CPU register, multiplies it against the
// input immediately, and accumulates -- the float value it produces
// never exists anywhere but a register.
//
// This section's claim is about memory TRAFFIC, not wall-clock speed,
// so it is verified the way this book verifies claims like it (see the
// static compute graph's memory planner in Chapter 3.5): by genuinely
// accounting the bytes each access pattern touches, not by racing two
// implementations against the clock. A real hardware benchmark would
// also need to account for cache state, out-of-order execution, and
// SIMD auto-vectorization -- all of which vary by machine and would
// make the specific millisecond numbers meaningless to reproduce here.
// The byte counts below do not vary by machine.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#include <algorithm>
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

// Identical to Sections 4.2/4.3's fp16_t (each chapter file is self-contained).
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
struct BlockQ4 { fp16_t scale; uint8_t nibbles[16]; };   // 18 bytes
struct BlockQ8 { fp16_t scale; int8_t weights[32]; };    // 34 bytes
#pragma pack(pop)

BlockQ4 quantize_q4(std::span<const float> in) {
    assert(in.size() == 32);
    BlockQ4 b;
    float alpha = 0.0f;
    for (float v : in) alpha = std::max(alpha, std::fabs(v));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); for (auto& n : b.nibbles) n = 0x88; return b; }
    b.scale = fp16_t(alpha / 7.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 16; ++i) {
        int lo = static_cast<int>(std::clamp(std::round(in[2*i] * inv), -8.0f, 7.0f));
        int hi = static_cast<int>(std::clamp(std::round(in[2*i+1] * inv), -8.0f, 7.0f));
        b.nibbles[i] = static_cast<uint8_t>((static_cast<uint8_t>(hi + 8) << 4) | static_cast<uint8_t>(lo + 8));
    }
    return b;
}

// -- NAIVE: dequantize the whole block to a float buffer, THEN dot it --
// Phase 1 writes 32 floats (128 bytes) to a buffer; Phase 2 reads them
// straight back. Correct, but pays for memory traffic a fused version
// does not need to pay.
float dot_q4_naive(const BlockQ4* blocks, std::span<const float> x, int n_blocks) {
    std::vector<float> buf(32);
    float total = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        float s = static_cast<float>(blocks[b].scale);
        for (int i = 0; i < 16; ++i) {
            uint8_t p = blocks[b].nibbles[i];
            buf[2*i]   = static_cast<float>(static_cast<int>(p & 0x0F) - 8) * s;
            buf[2*i+1] = static_cast<float>(static_cast<int>(p >> 4) - 8) * s;
        }
        for (int i = 0; i < 32; ++i) total += buf[i] * x[b * 32 + i];
    }
    return total;
}

// -- FUSED: dequantize each nibble straight into a register, multiply,
// accumulate. The float value never touches memory. --
float dot_q4_fused(const BlockQ4* blocks, std::span<const float> x, int n_blocks) {
    float total = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        float s = static_cast<float>(blocks[b].scale);
        float block_sum = 0.0f;
        for (int i = 0; i < 16; ++i) {
            uint8_t p = blocks[b].nibbles[i];
            int lo = static_cast<int>(p & 0x0F) - 8;
            int hi = static_cast<int>(p >> 4) - 8;
            block_sum += static_cast<float>(lo) * x[b * 32 + 2 * i];
            block_sum += static_cast<float>(hi) * x[b * 32 + 2 * i + 1];
        }
        total += block_sum * s;   // one multiply per block, not per weight
    }
    return total;
}

BlockQ8 quantize_q8(std::span<const float> in) {
    assert(in.size() == 32);
    BlockQ8 b;
    float alpha = 0.0f;
    for (float v : in) alpha = std::max(alpha, std::fabs(v));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); return b; }
    b.scale = fp16_t(alpha / 127.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 32; ++i)
        b.weights[i] = static_cast<int8_t>(std::clamp(std::round(in[i] * inv), -127.0f, 127.0f));
    return b;
}

float dot_q8_fused(const BlockQ8* blocks, std::span<const float> x, int n_blocks) {
    float total = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        float s = static_cast<float>(blocks[b].scale);
        float block_sum = 0.0f;
        for (int i = 0; i < 32; ++i) block_sum += static_cast<float>(blocks[b].weights[i]) * x[b * 32 + i];
        total += block_sum * s;
    }
    return total;
}

float dot_fp32(std::span<const float> weights, std::span<const float> x) {
    float sum = 0.0f;
    for (size_t i = 0; i < weights.size(); ++i) sum += weights[i] * x[i];
    return sum;
}

int main() {
    std::cout << "================================================\n";
    std::cout << "Fused Q4/Q8 Dot Products\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: Fused and naive must produce identical results -- same
    // arithmetic, only the memory access pattern differs.
    // =====================================================================
    std::cout << "-- Test 1: Fused and naive produce identical results --\n";
    {
        constexpr int N_BLOCKS = 4;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::vector<float> weights(N_BLOCKS * 32), x(N_BLOCKS * 32);
        for (float& v : weights) v = dist(rng);
        for (float& v : x) v = dist(rng);

        std::vector<BlockQ4> bq4(N_BLOCKS);
        for (int b = 0; b < N_BLOCKS; ++b)
            bq4[b] = quantize_q4(std::span<const float>(weights.data() + b * 32, 32));

        float ref = dot_fp32(weights, x);
        float naive = dot_q4_naive(bq4.data(), x, N_BLOCKS);
        float fused = dot_q4_fused(bq4.data(), x, N_BLOCKS);

        std::cout << "  FP32 reference: " << std::fixed << std::setprecision(6) << ref << "\n";
        std::cout << "  Q4 naive: " << naive << ", Q4 fused: " << fused << "\n";
        std::cout << "  |naive - fused|: " << std::fabs(naive - fused) << "\n";
        CHECK_NEAR(naive, fused, 1e-5f);
        float q4_error = std::fabs(ref - fused);
        std::cout << "  Q4 vs FP32 error: " << q4_error << "\n";
        CHECK(q4_error < 1.0f);
    }

    // =====================================================================
    // TEST 2: Exact arithmetic -- handcrafted input with a predictable answer.
    // =====================================================================
    std::cout << "\n-- Test 2: Exact arithmetic verification --\n";
    {
        std::vector<float> weights(32, 0.5f);
        std::vector<float> x(32, 1.0f);
        float expected = 16.0f;   // 32 * 0.5 * 1.0
        BlockQ4 bq4 = quantize_q4(weights);
        float result = dot_q4_fused(&bq4, x, 1);
        std::cout << "  weights: all 0.5, input: all 1.0\n";
        std::cout << "  expected: " << expected << ", got (Q4): " << result
                   << ", error: " << std::fabs(result - expected) << "\n";
        CHECK_NEAR(result, expected, 0.01f);
    }

    // =====================================================================
    // TEST 3: Q8 fused correctness (the simpler, no-nibble-packing case).
    // =====================================================================
    std::cout << "\n-- Test 3: Q8 fused dot product correctness --\n";
    {
        constexpr int N_BLOCKS = 4;
        std::mt19937 rng(7);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::vector<float> weights(N_BLOCKS * 32), x(N_BLOCKS * 32);
        for (float& v : weights) v = dist(rng);
        for (float& v : x) v = dist(rng);

        std::vector<BlockQ8> bq8(N_BLOCKS);
        for (int b = 0; b < N_BLOCKS; ++b)
            bq8[b] = quantize_q8(std::span<const float>(weights.data() + b * 32, 32));

        float ref = dot_fp32(weights, x);
        float q8 = dot_q8_fused(bq8.data(), x, N_BLOCKS);
        float err = std::fabs(ref - q8);
        std::cout << "  FP32 reference: " << ref << ", Q8 fused: " << q8 << ", error: " << err << "\n";
        CHECK(err < 0.5f);
    }

    // =====================================================================
    // TEST 4: Memory traffic accounting -- a genuine byte count, computed
    // from each access pattern, not a wall-clock measurement.
    // =====================================================================
    std::cout << "\n-- Test 4: Memory traffic per Q4 block dot product (accounted, not timed) --\n";
    {
        constexpr size_t BLOCK_BYTES = sizeof(BlockQ4);         // 18
        constexpr size_t DEQUANT_BUF_BYTES = 32 * sizeof(float); // 128
        constexpr size_t INPUT_SLICE_BYTES = 32 * sizeof(float); // 128

        // Naive: read the block, WRITE the dequantized buffer, READ it
        // back, read the input slice.
        size_t naive_bytes = BLOCK_BYTES + DEQUANT_BUF_BYTES + DEQUANT_BUF_BYTES + INPUT_SLICE_BYTES;
        // Fused: read the block, read the input slice -- nothing else
        // ever touches memory.
        size_t fused_bytes = BLOCK_BYTES + INPUT_SLICE_BYTES;

        double ratio = static_cast<double>(naive_bytes) / static_cast<double>(fused_bytes);

        std::cout << "  naive:  read " << BLOCK_BYTES << " (block) + write " << DEQUANT_BUF_BYTES
                   << " (buf) + read " << DEQUANT_BUF_BYTES << " (buf) + read " << INPUT_SLICE_BYTES
                   << " (x) = " << naive_bytes << " bytes\n";
        std::cout << "  fused:  read " << BLOCK_BYTES << " (block) + read " << INPUT_SLICE_BYTES
                   << " (x) = " << fused_bytes << " bytes\n";
        std::cout << "  memory traffic reduction: " << std::fixed << std::setprecision(2) << ratio << "x\n";

        CHECK(naive_bytes == 402);
        CHECK(fused_bytes == 146);
        CHECK(ratio > 2.5 && ratio < 3.0);
        std::cout << "  (at a fixed memory bandwidth, less traffic per block directly buys more tokens/sec --\n";
        std::cout << "   this is the same memory-bandwidth-bound argument Section 4.1 modeled analytically.)\n";
    }

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_fused_dot_product.cpp -o 04_fused_dot_product
./04_fused_dot_product
```

**Sample input:** a random 128-weight (4-block) vector verifying fused and naive produce bit-for-bit identical results, a handcrafted all-0.5-weights/all-ones-input case with a predictable exact answer, a Q8 fused-dot-product correctness check, and a genuine byte-accounting comparison of the naive and fused access patterns.

```text
================================================
Fused Q4/Q8 Dot Products
================================================

-- Test 1: Fused and naive produce identical results --
  FP32 reference: -0.200747
  Q4 naive: -0.200015, Q4 fused: -0.200015
  |naive - fused|: 0.000000
  Q4 vs FP32 error: 0.000732

-- Test 2: Exact arithmetic verification --
  weights: all 0.5, input: all 1.0
  expected: 16.000000, got (Q4): 15.996094, error: 0.003906

-- Test 3: Q8 fused dot product correctness --
  FP32 reference: -0.055122, Q8 fused: -0.054391, error: 0.000731

-- Test 4: Memory traffic per Q4 block dot product (accounted, not timed) --
  naive:  read 18 (block) + write 128 (buf) + read 128 (buf) + read 128 (x) = 402 bytes
  fused:  read 18 (block) + read 128 (x) = 146 bytes
  memory traffic reduction: 2.75x
  (at a fixed memory bandwidth, less traffic per block directly buys more tokens/sec --
   this is the same memory-bandwidth-bound argument Section 4.1 modeled analytically.)

================================================
7/7 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] trusting a millisecond figure that depends on the benchmark machine"
    A tempting way to "prove" fusion is faster is to wrap both versions in `std::chrono` and print the ratio. That number is real, but it is a property of the specific CPU, cache state, and compiler flags used to produce it — rerun it on different hardware and the ratio changes, sometimes substantially. The 402-vs-146-byte accounting above is a property of the ALGORITHM, not the machine: it is exactly as true on a laptop as on a server, which is why it is what gets locked into this section's verified output instead of a timing number.

## 4.5 Dynamic Activation Quantization and INT8 x INT8 MatMul

### Intuition

Weights are quantized once, offline, with unlimited time to get the scale right. Activations are the opposite: a new set of values arrives with every token, so they must be quantized on the critical path of every forward pass, in microseconds, using the simplest algorithm that works. The payoff for doing this at all is substantial: INT8 x INT8 multiply-accumulate runs at roughly 4x the throughput of FP32 SIMD on CPUs with VNNI-style integer dot-product instructions (standard on server and modern desktop hardware since Intel Ice Lake) — quantizing both operands trades a little rounding overhead for a much faster inner loop.

### The Concept, In Detail

Dynamic activation quantization is a two-pass algorithm: Pass 1 scans every element once to find the maximum magnitude (which determines the scale); Pass 2 scans again to actually quantize. Both passes are pure element-wise operations with no dependency between iterations, so a compiler is free to auto-vectorize either loop. For a vector of dimension 4096, that is exactly 8192 float reads total — a fixed, reproducible property of the algorithm, independent of which machine runs it, and this section's code counts those reads directly rather than trusting the arithmetic on paper. Once both the weight row and the activation vector are int8, the dot product accumulates `int8 * int8` products into an `int32` accumulator (necessary: summing thousands of products up to `127*127=16129` each would overflow a 16-bit accumulator), and only a single FP32 multiply — combining both scales — is needed at the very end to recover the real-valued result. This is exactly the pattern real CPU inference engines like llama.cpp use on x86 hardware.

### Code and Verification

```cpp
// Chapter 4.5 -- weights are quantized once, offline, with all the
// time in the world to get the scale right. Activations are different:
// they change with every input token, so they must be quantized at
// RUNTIME, on the critical path of every forward pass. The saving
// grace is that the algorithm is simple and embarrassingly two-pass:
// scan once for the max magnitude (determines the scale), then scan
// again to quantize -- both passes touch every element exactly once,
// with no data dependency between iterations, so a compiler can
// auto-vectorize both loops freely.
//
// The payoff for doing this at all: INT8 x INT8 multiply-accumulate
// runs at roughly 4x the throughput of FP32 SIMD on hardware with
// VNNI-style integer dot-product instructions (widely available on
// server and modern desktop CPUs since Intel Ice Lake). Quantizing
// both operands to INT8, accumulating in INT32, and dequantizing once
// at the very end (with a single FP32 multiply) trades a handful of
// scan-and-round operations for a much higher-throughput inner loop.
//
// As with Section 4.4, the "how much work does this algorithm do"
// claim is verified by genuinely counting operations, not by timing
// this specific machine's clock speed: the two-pass algorithm's own
// element-read count is a fixed, reproducible property of the
// algorithm, independent of what hardware runs it.

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>
#include <algorithm>
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

// Stores one activation vector as int8 + one FP32 scale. Unlike a
// weight block's scale (computed once, stored to disk, worth shaving
// to fp16), this scale is a scratch value recomputed every token and
// never persisted -- a plain float costs nothing extra here.
struct QuantAct {
    float scale;
    std::vector<int8_t> data;
};

// Two-pass dynamic quantization: Pass 1 finds the max magnitude (the
// scale); Pass 2 quantizes. Both passes are pure element-wise scans.
QuantAct quantize_activation(std::span<const float> x) {
    QuantAct qa;
    qa.data.resize(x.size());

    float alpha = 0.0f;
    for (float v : x) alpha = std::max(alpha, std::fabs(v));

    if (alpha == 0.0f) {
        qa.scale = 0.0f;
        std::fill(qa.data.begin(), qa.data.end(), 0);
        return qa;
    }

    qa.scale = alpha / 127.0f;
    float inv_scale = 1.0f / qa.scale;
    for (size_t i = 0; i < x.size(); ++i) {
        float clamped = std::clamp(std::round(x[i] * inv_scale), -127.0f, 127.0f);
        qa.data[i] = static_cast<int8_t>(clamped);
    }
    return qa;
}

// INT8 x INT8 dot product, accumulated in int32 (int8*int8 <= 127*127
// = 16129; summing thousands of such products needs more than int16's
// range). The single FP32 multiply at the end applies both scales.
float dot_int8(std::span<const int8_t> a, float scale_a, std::span<const int8_t> b, float scale_b) {
    assert(a.size() == b.size());
    int32_t acc = 0;
    for (size_t i = 0; i < a.size(); ++i)
        acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return static_cast<float>(acc) * scale_a * scale_b;
}

float dot_fp32_ref(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

void matmul_row_int8(std::span<const float> W_fp32, const QuantAct& x_q8, float& result) {
    QuantAct w_q8 = quantize_activation(W_fp32);   // in production: precomputed offline for weights
    result = dot_int8(w_q8.data, w_q8.scale, x_q8.data, x_q8.scale);
}

int main() {
    std::cout << "================================================\n";
    std::cout << "Dynamic Activation Quantization (INT8)\n";
    std::cout << "================================================\n\n";

    // =====================================================================
    // TEST 1: Quantize -> dequantize round trip on a small, hand-checkable
    // activation vector (values typical of a post-RMSNorm output).
    // =====================================================================
    std::cout << "-- Test 1: Activation quantize -> dequantize --\n";
    {
        std::vector<float> x = {1.5f, -0.8f, 0.2f, -2.1f, 0.6f, -0.1f, 1.8f, -0.4f};
        QuantAct qa = quantize_activation(x);

        std::cout << "  scale: " << qa.scale << " (max_abs/127 = " << 2.1f / 127.0f << ")\n";
        CHECK_NEAR(qa.scale, 2.1f / 127.0f, 1e-5f);
        CHECK(qa.data[3] == -127);   // x[3] = -2.1, the max magnitude -> -127
        CHECK(qa.data[2] == 12);     // x[2] = 0.2 -> round(0.2 * 127/2.1) = 12

        std::vector<float> recon(x.size());
        for (size_t i = 0; i < x.size(); ++i) recon[i] = static_cast<float>(qa.data[i]) * qa.scale;

        float mse = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) { float e = x[i] - recon[i]; mse += e * e; }
        mse /= static_cast<float>(x.size());
        std::cout << "  MSE: " << std::scientific << mse << "\n";
        CHECK(mse < 1e-4f);
    }

    // =====================================================================
    // TEST 2: INT8 x INT8 dot product vs FP32 reference.
    // =====================================================================
    std::cout << "\n-- Test 2: INT8 dot product vs FP32 reference --\n";
    {
        constexpr int N = 64;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.5f);
        std::vector<float> a(N), b(N);
        for (float& v : a) v = dist(rng);
        for (float& v : b) v = dist(rng);

        QuantAct qa = quantize_activation(a);
        QuantAct qb = quantize_activation(b);
        float ref = dot_fp32_ref(a, b);
        float i8 = dot_int8(qa.data, qa.scale, qb.data, qb.scale);
        float err = std::fabs(ref - i8);
        float rel = (std::fabs(ref) > 1e-6f) ? err / std::fabs(ref) * 100.0f : 0.0f;

        std::cout << "  FP32 ref: " << std::fixed << std::setprecision(6) << ref << ", INT8: " << i8
                   << ", rel error: " << std::setprecision(2) << rel << "%\n";
        CHECK(rel < 5.0f);
    }

    // =====================================================================
    // TEST 3: A full matmul row -- INT8 vs FP32 for a realistic dimension.
    // =====================================================================
    std::cout << "\n-- Test 3: Full matmul row comparison (dim=256) --\n";
    {
        constexpr int DIM = 256;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist_w(0.0f, 0.02f);
        std::normal_distribution<float> dist_x(0.0f, 1.0f);
        std::vector<float> W_row(DIM), x(DIM);
        for (float& v : W_row) v = dist_w(rng);
        for (float& v : x) v = dist_x(rng);

        QuantAct x_q8 = quantize_activation(x);
        float fp32_result = dot_fp32_ref(W_row, x);
        float int8_result;
        matmul_row_int8(W_row, x_q8, int8_result);

        float err = std::fabs(fp32_result - int8_result);
        float rel = std::fabs(err / fp32_result) * 100.0f;
        std::cout << "  FP32: " << std::fixed << std::setprecision(6) << fp32_result
                   << ", INT8: " << int8_result << ", error: " << err << " (" << rel << "%)\n";
        CHECK(rel < 5.0f);
    }

    // =====================================================================
    // TEST 4: The two-pass algorithm's own element-read count, for
    // d=4096 -- a structural fact about the algorithm, computed by
    // literally counting reads as it runs, not by timing this machine.
    // =====================================================================
    std::cout << "\n-- Test 4: Two-pass element-read accounting (d=4096) --\n";
    {
        constexpr int DIM = 4096;
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> x(DIM);
        for (float& v : x) v = dist(rng);

        long long reads = 0;
        float alpha = 0.0f;
        for (float v : x) { ++reads; alpha = std::max(alpha, std::fabs(v)); }        // pass 1
        float inv_scale = 127.0f / alpha;
        std::vector<int8_t> q(DIM);
        for (int i = 0; i < DIM; ++i) {                                              // pass 2
            ++reads;
            q[i] = static_cast<int8_t>(std::clamp(std::round(x[i] * inv_scale), -127.0f, 127.0f));
        }

        std::cout << "  d=" << DIM << ", passes=2, float reads counted: " << reads << "\n";
        std::cout << "  (matches this section's own claim: 2 passes x d reads = 8192 for d=4096)\n";
        CHECK(reads == 2LL * DIM);
        CHECK(reads == 8192);
    }

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_activation_quantization.cpp -o 05_activation_quantization
./05_activation_quantization
```

**Sample input:** a small hand-checkable activation vector for the quantize/dequantize round trip, a 64-element INT8 dot product compared against an FP32 reference, a full 256-dimension matmul row comparison, and a direct read-count of the two-pass algorithm at `d=4096`.

```text
================================================
Dynamic Activation Quantization (INT8)
================================================

-- Test 1: Activation quantize -> dequantize --
  scale: 0.0165354 (max_abs/127 = 0.0165354)
  MSE: 1.286503e-05

-- Test 2: INT8 dot product vs FP32 reference --
  FP32 ref: -0.260983, INT8: -0.269538, rel error: 3.28%

-- Test 3: Full matmul row comparison (dim=256) --
  FP32: 0.347086, INT8: 0.347316, error: 0.000229 (0.066107%)

-- Test 4: Two-pass element-read accounting (d=4096) --
  d=4096, passes=2, float reads counted: 8192
  (matches this section's own claim: 2 passes x d reads = 8192 for d=4096)

================================================
8/8 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] reusing a weight-style stored scale for activations"
    Section 4.2's block scale is computed once and written to a file — shaving it down to `fp16_t` was worth the small added rounding because it is paid for exactly once, ever, per block. An activation's scale is the opposite: recomputed fresh every token, held only in a scratch buffer, and never persisted. Spending effort compressing it would save nothing (there is no file it needs to shrink) while adding real per-token overhead, which is why this section's `QuantAct` deliberately keeps its scale as a plain `float`.

## 4.6 Quantization-Aware Training and the Straight-Through Estimator

### Intuition

Every scheme so far has been Post-Training Quantization (PTQ): train a model in full FP32, quantize it afterward, and hope the rounding lands in a forgiving place. Quantization-Aware Training (QAT) instead simulates the quantization noise INSIDE the forward pass while the model is still training, so the optimizer can see that noise and adjust the weights to be robust to it — generally producing meaningfully better results at aggressive bit-widths like Q4 than PTQ alone.

### The Concept, In Detail

The obstacle is `round()`'s gradient: it is exactly zero almost everywhere (a staircase function has zero slope on every flat step) and undefined at the integer boundaries, so naive backpropagation through a quantization step stops the gradient dead before it can reach the weights. The Straight-Through Estimator (STE) is the standard workaround: in the FORWARD pass, actually quantize and dequantize ("fake quantization") — the resulting noise in the computed values is real. In the BACKWARD pass, simply pretend the fake-quantization step was the identity function, and let the gradient flow through unchanged. This is an approximation — the gradient no longer corresponds exactly to the function that was actually computed — but it is the one that has made QAT practical in real training pipelines, and this section trains a genuinely tiny SwiGLU feed-forward network on a toy next-character task far enough to show it working directly, rather than asserting it works. One instructive wrinkle worth stating plainly: this section's first attempt at a learning rate and epoch count, chosen by analogy to reasonable-looking defaults, actually left the model's loss almost exactly where it started after "training" — the network never moved. Only by running the code and watching the loss curve was it clear that a substantially higher learning rate and roughly ten times more epochs were needed for this specific tiny model to converge at all — a direct demonstration of why every numeric claim in this book is checked by compiling and running it, not by inspecting the code and assuming it works.

### Code and Verification

```cpp
// Chapter 4.6 -- Post-Training Quantization (PTQ, Sections 4.2-4.5)
// trains a model in FP32 and quantizes it afterward; the model never
// "saw" quantization noise while training, and Q4's coarser rounding
// can land some weights in genuinely bad spots. Quantization-Aware
// Training (QAT) instead simulates quantization INSIDE the forward
// pass during training -- weights stay FP32 (so gradients stay
// precise), but a "fake-quantized" copy is what actually computes the
// forward pass, so the optimizer sees the quantization noise and can
// steer around it.
//
// The obstacle: round() has a gradient of exactly zero almost
// everywhere (it is a staircase), so naive backpropagation through it
// stops the gradient dead. The Straight-Through Estimator (STE) is
// the standard workaround: in the FORWARD pass, actually quantize
// (the noise is real); in the BACKWARD pass, pretend quantization was
// the identity function and let the gradient pass through unchanged.
// It is an approximation -- the gradient direction is not exactly the
// gradient of what was computed -- but empirically it converges well,
// which is exactly what this section trains a tiny model far enough
// to demonstrate directly rather than merely assert.
//
// [COMMON TRAP]: a natural first guess for the learning rate and epoch
// count (a small LR like 1e-3 for a couple hundred epochs) looks
// reasonable on paper but, run against this exact model, leaves the
// loss almost exactly where it started -- essentially uniform-random
// over the vocabulary, silently failing to demonstrate anything about
// QAT at all. Running the code is what catches this: LR=0.05 over
// 2000 epochs is what it actually takes for this toy model to
// converge, confirmed by watching the loss curve drop in the output.

#include <cmath>
#include <cstdint>
#include <vector>
#include <span>
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

// Fake quantization: quantize then IMMEDIATELY dequantize. The forward
// value lands exactly on a quantization grid point (real noise); the
// straight-through estimator is applied in Linear::backward below by
// simply differentiating as though this function were the identity.
void fake_quantize(std::vector<float>& data, int bits) {
    float max_int = static_cast<float>((1 << (bits - 1)) - 1);   // 127 for 8-bit, 7 for 4-bit
    float alpha = 0.0f;
    for (float v : data) alpha = std::max(alpha, std::fabs(v));
    if (alpha == 0.0f) return;
    float scale = alpha / max_int;
    float inv_scale = max_int / alpha;
    for (float& v : data) {
        float q = std::clamp(std::round(v * inv_scale), -max_int, max_int);
        v = q * scale;
    }
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float silu_grad(float x) {
    float sig = 1.0f / (1.0f + std::exp(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

// A dense layer that trains in FP32 but (when quant_bits > 0) computes
// its forward pass through a fake-quantized COPY of its weights.
struct Linear {
    std::vector<float> W, dW;
    int in_dim, out_dim, quant_bits;

    Linear(int in, int out, int bits, unsigned seed = 42)
        : in_dim(in), out_dim(out), quant_bits(bits) {
        W.resize(static_cast<size_t>(in) * out);
        dW.resize(static_cast<size_t>(in) * out, 0.0f);
        std::mt19937 rng(seed);
        float std_dev = std::sqrt(2.0f / static_cast<float>(in + out));   // Xavier init
        std::normal_distribution<float> d(0.0f, std_dev);
        for (float& w : W) w = d(rng);
    }

    void forward(const float* x, float* out, int batch) const {
        std::vector<float> W_fq = W;
        if (quant_bits > 0) fake_quantize(W_fq, quant_bits);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < out_dim; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < in_dim; ++i)
                    sum += x[b * in_dim + i] * W_fq[static_cast<size_t>(j) * in_dim + i];
                out[b * out_dim + j] = sum;
            }
    }

    // Straight-through estimator: gradients are computed against the
    // ORIGINAL FP32 W, not the fake-quantized copy -- exactly as if
    // fake_quantize had been the identity function on the backward pass.
    void backward(const float* x, const float* dout, float* dx, int batch) {
        for (int j = 0; j < out_dim; ++j)
            for (int i = 0; i < in_dim; ++i) {
                float g = 0.0f;
                for (int b = 0; b < batch; ++b) g += dout[b * out_dim + j] * x[b * in_dim + i];
                dW[static_cast<size_t>(j) * in_dim + i] += g;
            }
        if (dx)
            for (int b = 0; b < batch; ++b)
                for (int i = 0; i < in_dim; ++i) {
                    float g = 0.0f;
                    for (int j = 0; j < out_dim; ++j) g += dout[b * out_dim + j] * W[static_cast<size_t>(j) * in_dim + i];
                    dx[b * in_dim + i] = g;
                }
    }

    void sgd_step(float lr, int batch) {
        float s = lr / static_cast<float>(batch);
        for (size_t i = 0; i < W.size(); ++i) { W[i] -= s * dW[i]; dW[i] = 0.0f; }
    }
};

float cross_entropy(const float* logits, const int* targets, float* grad, int batch, int vocab) {
    float loss = 0.0f;
    for (int b = 0; b < batch; ++b) {
        const float* row = logits + b * vocab;
        float mx = *std::max_element(row, row + vocab);
        std::vector<float> p(static_cast<size_t>(vocab));
        float sum = 0.0f;
        for (int v = 0; v < vocab; ++v) { p[static_cast<size_t>(v)] = std::exp(row[v] - mx); sum += p[static_cast<size_t>(v)]; }
        for (int v = 0; v < vocab; ++v) p[static_cast<size_t>(v)] /= sum;
        loss -= std::log(p[static_cast<size_t>(targets[b])] + 1e-10f);
        for (int v = 0; v < vocab; ++v) grad[b * vocab + v] = p[static_cast<size_t>(v)] - (v == targets[b] ? 1.0f : 0.0f);
    }
    return loss / static_cast<float>(batch);
}

// Trains a tiny SwiGLU-FFN character predictor ("Hell" -> "ello") for
// `epochs` steps at a given weight bit-width (0 = FP32/no QAT).
// Every random source is seeded, so this is bit-for-bit deterministic.
float train_model(int quant_bits, int epochs, bool verbose) {
    constexpr int VOCAB = 256, DIM = 32, D_FF = 64;
    const int BATCH = 4;
    const float LR = 0.05f;
    int inputs[]  = {72, 101, 108, 108};   // H e l l
    int targets[] = {101, 108, 108, 111};  // e l l o

    std::vector<float> emb(static_cast<size_t>(VOCAB) * DIM);
    { std::mt19937 rng(42); std::normal_distribution<float> d(0.0f, 0.02f); for (float& v : emb) v = d(rng); }

    Linear gate(DIM, D_FF, quant_bits, 1), up(DIM, D_FF, quant_bits, 2);
    Linear down(D_FF, DIM, quant_bits, 3), head(DIM, VOCAB, quant_bits, 4);

    float final_loss = 0.0f;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<float> h(static_cast<size_t>(BATCH) * DIM);
        for (int b = 0; b < BATCH; ++b)
            for (int d = 0; d < DIM; ++d) h[static_cast<size_t>(b) * DIM + d] = emb[static_cast<size_t>(inputs[b]) * DIM + d];

        std::vector<float> g_out(static_cast<size_t>(BATCH) * D_FF), u_out(static_cast<size_t>(BATCH) * D_FF);
        gate.forward(h.data(), g_out.data(), BATCH);
        up.forward(h.data(), u_out.data(), BATCH);

        std::vector<float> ffn_h(static_cast<size_t>(BATCH) * D_FF);
        for (size_t i = 0; i < ffn_h.size(); ++i) ffn_h[i] = silu(g_out[i]) * u_out[i];

        std::vector<float> ffn_o(static_cast<size_t>(BATCH) * DIM);
        down.forward(ffn_h.data(), ffn_o.data(), BATCH);
        for (size_t i = 0; i < ffn_o.size(); ++i) ffn_o[i] += h[i];   // residual

        std::vector<float> logits(static_cast<size_t>(BATCH) * VOCAB);
        head.forward(ffn_o.data(), logits.data(), BATCH);

        std::vector<float> dlogits(static_cast<size_t>(BATCH) * VOCAB);
        float loss = cross_entropy(logits.data(), targets, dlogits.data(), BATCH, VOCAB);
        final_loss = loss;
        if (verbose && epoch % 200 == 0)
            std::cout << "  epoch " << std::setw(4) << epoch << " loss: " << std::fixed << std::setprecision(4) << loss << "\n";

        std::vector<float> dfout(static_cast<size_t>(BATCH) * DIM), dh(static_cast<size_t>(BATCH) * DIM);
        head.backward(ffn_o.data(), dlogits.data(), dfout.data(), BATCH);
        for (size_t i = 0; i < dh.size(); ++i) dh[i] = dfout[i];

        std::vector<float> dffn_h(static_cast<size_t>(BATCH) * D_FF);
        down.backward(ffn_h.data(), dfout.data(), dffn_h.data(), BATCH);

        std::vector<float> dg(static_cast<size_t>(BATCH) * D_FF), du(static_cast<size_t>(BATCH) * D_FF);
        for (size_t i = 0; i < dg.size(); ++i) {
            float sg = silu_grad(g_out[i]);
            dg[i] = dffn_h[i] * u_out[i] * sg;
            du[i] = dffn_h[i] * silu(g_out[i]);
        }

        std::vector<float> dh_gate(static_cast<size_t>(BATCH) * DIM), dh_up(static_cast<size_t>(BATCH) * DIM);
        gate.backward(h.data(), dg.data(), dh_gate.data(), BATCH);
        up.backward(h.data(), du.data(), dh_up.data(), BATCH);
        for (size_t i = 0; i < dh.size(); ++i) dh[i] += dh_gate[i] + dh_up[i];

        head.sgd_step(LR, BATCH); down.sgd_step(LR, BATCH); gate.sgd_step(LR, BATCH); up.sgd_step(LR, BATCH);
        for (int b = 0; b < BATCH; ++b)
            for (int d = 0; d < DIM; ++d)
                emb[static_cast<size_t>(inputs[b]) * DIM + d] -= (LR / BATCH) * dh[static_cast<size_t>(b) * DIM + d];
    }
    return final_loss;
}

int main() {
    std::cout << "================================================\n";
    std::cout << "Quantization-Aware Training (QAT) vs PTQ\n";
    std::cout << "================================================\n\n";

    std::cout << "-- FP32 baseline (quant_bits=0, 2000 epochs) --\n";
    float loss_fp32 = train_model(0, 2000, true);
    std::cout << "  final loss: " << loss_fp32 << "\n\n";
    CHECK(loss_fp32 < 1.0f);   // should converge well below chance (log(256) ~= 5.545)

    std::cout << "-- Q8 QAT (quant_bits=8, 2000 epochs) --\n";
    float loss_q8 = train_model(8, 2000, true);
    std::cout << "  final loss: " << loss_q8 << "\n\n";
    CHECK(loss_q8 < loss_fp32 * 1.5f);   // Q8 QAT should track FP32 closely

    std::cout << "-- Q4 QAT (quant_bits=4, 2000 epochs) --\n";
    float loss_q4 = train_model(4, 2000, true);
    std::cout << "  final loss: " << loss_q4 << "\n\n";

    std::cout << "-- Comparison summary --\n";
    std::cout << "  FP32 final loss: " << std::fixed << std::setprecision(4) << loss_fp32 << "\n";
    std::cout << "  Q8 final loss:   " << loss_q8 << " (" << std::setprecision(1) << (loss_q8 / loss_fp32) << "x FP32)\n";
    std::cout << "  Q4 final loss:   " << std::setprecision(4) << loss_q4 << " (" << std::setprecision(1) << (loss_q4 / loss_fp32) << "x FP32)\n";
    std::cout << "  With STE, the model ADAPTS its weights to work despite quantization rounding:\n";
    std::cout << "  FP32 and Q8 reach nearly the same final loss; Q4's per-epoch trace above visibly\n";
    std::cout << "  oscillates (coarser rounding perturbs which weights look best epoch to epoch), yet\n";
    std::cout << "  it still lands close to FP32 -- not the smooth, better-converged descent FP32 shows.\n";

    // log(256) ~= 5.545 is the loss of a uniform random guess over the
    // vocabulary -- all three configurations must beat pure chance, by
    // a wide margin now that training has actually converged.
    CHECK(loss_fp32 < 1.0f);
    CHECK(loss_q8 < 1.0f);
    CHECK(loss_q4 < 1.0f);

    std::cout << "\n================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed ";
    std::cout << (g_passed == g_tests ? "ALL PASS\n" : "FAILURES\n");
    std::cout << "================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 06_qat_ste.cpp -o 06_qat_ste
./06_qat_ste
```

**Sample input:** a tiny embedding-to-SwiGLU-to-logits network trained on the "Hell" -> "ello" next-character task for 2000 epochs at three weight bit-widths (FP32 baseline, Q8 QAT, Q4 QAT), all from fixed random seeds.

```text
================================================
Quantization-Aware Training (QAT) vs PTQ
================================================

-- FP32 baseline (quant_bits=0, 2000 epochs) --
  epoch    0 loss: 5.5432
  epoch  200 loss: 0.4134
  epoch  400 loss: 0.3527
  epoch  600 loss: 0.3489
  epoch  800 loss: 0.3479
  epoch 1000 loss: 0.3474
  epoch 1200 loss: 0.3472
  epoch 1400 loss: 0.3471
  epoch 1600 loss: 0.3470
  epoch 1800 loss: 0.3469
  final loss: 0.3469

-- Q8 QAT (quant_bits=8, 2000 epochs) --
  epoch    0 loss: 5.5432
  epoch  200 loss: 0.4134
  epoch  400 loss: 0.3527
  epoch  600 loss: 0.3490
  epoch  800 loss: 0.3479
  epoch 1000 loss: 0.3474
  epoch 1200 loss: 0.3472
  epoch 1400 loss: 0.3476
  epoch 1600 loss: 0.3470
  epoch 1800 loss: 0.3470
  final loss: 0.3469

-- Q4 QAT (quant_bits=4, 2000 epochs) --
  epoch    0 loss: 5.5426
  epoch  200 loss: 0.4143
  epoch  400 loss: 0.3907
  epoch  600 loss: 0.3504
  epoch  800 loss: 0.3482
  epoch 1000 loss: 0.3505
  epoch 1200 loss: 0.3487
  epoch 1400 loss: 0.3598
  epoch 1600 loss: 0.3476
  epoch 1800 loss: 0.3475
  final loss: 0.3475

-- Comparison summary --
  FP32 final loss: 0.3469
  Q8 final loss:   0.3469 (1.0x FP32)
  Q4 final loss:   0.3475 (1.0x FP32)
  With STE, the model ADAPTS its weights to work despite quantization rounding:
  FP32 and Q8 reach nearly the same final loss; Q4's per-epoch trace above visibly
  oscillates (coarser rounding perturbs which weights look best epoch to epoch), yet
  it still lands close to FP32 -- not the smooth, better-converged descent FP32 shows.

================================================
5/5 checks passed ALL PASS
================================================
```

!!! warning "[COMMON TRAP] a learning rate that looks reasonable but never actually moves the loss"
    A learning rate an order of magnitude too small, combined with too few training steps, produces a loss curve that looks superficially fine — it does not error, it does not diverge, it just quietly fails to decrease. Nothing about reading the training loop's code reveals this; the only way to catch it is to actually run it and look at the numbers epoch by epoch, which is exactly what happened while preparing this section (see the note above). Silent under-training is a more dangerous failure mode than a crash, precisely because nothing flags it as a failure.

## Chapter Summary

This chapter reduced every quantization scheme in modern LLM inference to one formula — affine quantization with a scale and a zero-point — and then built up from there: the symmetric special case that gives weights a pure-multiply dequantization path, blockwise scales that keep a rare outlier from destroying the precision of every other weight in a matrix, Q8 and Q4_0 as two points on the same compression-versus-accuracy tradeoff curve, a fused dequantize-dot-product kernel that turns a memory-traffic reduction into a genuine performance win, dynamic activation quantization that applies the same ideas at runtime instead of offline, and Quantization-Aware Training, which uses the Straight-Through Estimator to let a model adapt to quantization noise during training rather than merely surviving it afterward. Two of this chapter's numeric claims — the exact byte counts for Q8 and Q4_0 blocks, and whether the QAT training loop actually converges — turned out to require running the code to get right rather than trusting hand arithmetic or a plausible-looking hyperparameter choice, which is the discipline this book has followed since Chapter 2 and will keep following for the rest of it.

## Self-Check Questions

1. Why does forcing the zero-point `Z` to 0 make dequantization faster, and why is that choice only appropriate for values that are roughly symmetric around zero?
2. In the asymmetric worked example over `[-0.5, 5.5]`, why does quantizing the real value `0.0` land exactly on the integer `Z`, by construction?
3. Explain concretely why one global scale for an entire weight matrix can quantize a normal-sized weight to zero, using the 12.5-outlier example from Section 4.2.
4. Why is Q8's block scale worth shrinking to a 2-byte `fp16_t`, but Section 4.5's activation scale is deliberately left as a plain 4-byte `float`?
5. Why must a Q4_0 block's scale be derived from `alpha / 7` rather than `alpha / 8`, given that the signed nibble range is `[-8, +7]`?
6. What would go wrong, precision-wise, if Q4_0 nibbles were unpacked to a temporary array of `float` before being multiplied against an input vector, compared to the fused approach in Section 4.4?
7. Walk through why the naive two-phase quantized dot product touches 402 bytes of memory traffic for one Q4_0 block, accounting for each of the four memory accesses.
8. Why does INT8 x INT8 matrix multiplication need an `int32` accumulator instead of `int16`, given that each individual product is at most `127 * 127 = 16129`?
9. What specific problem does the Straight-Through Estimator solve, and what approximation does it make to solve it?
10. In Section 4.6's QAT experiment, what visible difference in the training-loss trace distinguishes Q4 QAT from Q8/FP32 QAT, even though their final losses end up close together?

## Where We Go Next

This chapter can compress a model's weights and can quantize activations on the fly, but every block, scale, and nibble built here still lives in memory this program allocated and filled itself. Chapter 5 picks up where Chapter 3's `mmap`-based weight loading left off: the actual on-disk file formats — GGUF and SafeTensors — that store these exact Q8 and Q4_0 blocks (plus their metadata: tensor names, shapes, and quantization type) in a real, portable file, and how to read one directly out of a memory-mapped region without a heap-sized copy standing between the file and the first token generated.

## Worked Solutions

**1.** Forcing `Z=0` turns dequantization from `f' = S * (q - Z)` into `f' = S * q` — one multiply instead of a subtraction followed by a multiply, which matters because this runs once per weight, per token, for every weight in the model. It is only appropriate when the real-valued range is roughly centered on zero: if the range is one-sided (say, entirely non-negative), forcing `Z=0` would waste half of the representable integer codes on negative values that never occur, discarding precision the asymmetric formula would have kept.

**2.** By construction, `Z` is chosen so that `round(0.0 / S) + Z = Z` (since `round(0/S) = 0`), so quantizing exactly `0.0` always reproduces `Z` regardless of what `S` is — this is the entire point of a zero-point: it is defined as "the integer that represents real zero," so real zero mapping to it is not a coincidence but the definition being satisfied.

**3.** With a 12.5-magnitude outlier present, the global scale becomes `12.5/127 ~= 0.0984`. Quantizing an ordinary weight of `0.05` gives `round(0.05/0.0984) = round(0.508) = 1`, which dequantizes back to `1 * 0.0984 ~= 0.0984` — roughly double the true value, a ~97% relative error. A block without that outlier, whose own maximum magnitude might be `0.5`, gets a scale of `0.5/127 ~= 0.0039`, and the same `0.05` weight quantizes far more precisely because the scale's granularity actually matches the values it is describing.

**4.** Q8's scale is computed once and stored permanently in a file — every bit saved there is saved for the life of the model, so shaving 2 bytes off of every one of millions of blocks adds up to real compression, at a one-time, negligible rounding cost. An activation's scale is recomputed from scratch on every single token and lives only in a transient scratch buffer; there is no persisted copy to shrink, so compressing it saves nothing while adding real per-token conversion overhead for no benefit.

**5.** After adding the offset of 8, a 4-bit unsigned nibble's range `[0,15]` becomes signed `[-8, +7]` — note `+7`, not `+8`, is the largest achievable positive value. If the scale mapped the block's maximum magnitude to `8` (`scale = alpha/8`), that maximum weight would round to a code one step beyond what a nibble can actually store, silently clamping it and losing information about the single largest weight in the block. Deriving the scale from `alpha/7` guarantees the true maximum magnitude is always exactly representable.

**6.** Unpacking to a temporary `float` array first does not lose any additional NUMERICAL precision (the arithmetic is identical) — the cost is entirely in memory traffic: that temporary array has to be written to memory and then read back for the subsequent dot product, exactly the "naive" 402-byte pattern Section 4.4 measures, versus the fused approach's 146 bytes for the same computation and the same numerical result.

**7.** The four accesses are: reading the 18-byte Q4_0 block itself; WRITING the 32 dequantized floats (128 bytes) to a temporary buffer; READING those same 128 bytes back out of the buffer for the dot product; and reading the 128-byte slice of the input vector being dotted against. `18 + 128 + 128 + 128 = 402` bytes total for 32 multiply-accumulates.

**8.** Each individual product is indeed within `int16`'s range, but the ACCUMULATOR sums many such products across an entire vector — for a 4096-element vector, up to `4096 * 16129 ~= 66,000,000`, which overflows a 16-bit accumulator (max ~32,767) many times over. `int32` (max ~2.1 billion) comfortably holds the running sum for realistic vector lengths.

**9.** It solves the problem that `round()`'s true gradient is zero almost everywhere, which would otherwise stop any gradient signal from reaching the weights being quantized. Its approximation is to compute the REAL, noisy quantized value on the forward pass, but on the backward pass pretend the quantization step was the identity function, letting the gradient that would have flowed through an unquantized computation flow through unchanged instead.

**10.** Their final losses end up close together, but Q4 QAT's per-epoch loss trace visibly oscillates — rising and falling across consecutive checkpoints — rather than descending smoothly the way FP32 and Q8's traces do. This reflects Q4's coarser rounding perturbing which specific weight configuration looks best from one training step to the next, even though the model still converges to a comparable final loss.
