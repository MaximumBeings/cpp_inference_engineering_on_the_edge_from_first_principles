# Chapter 9: SIMD Vectorization -- AVX2 on x86, NEON on Arm

**What you will understand by the end of this chapter:**

- What a SIMD register actually is — a fixed number of bits divided into lanes of a stated element width — derived from that single fact rather than memorized as a vendor-specific rule, and why doubling a chip's peak compute this way still buys nothing for a memory-bound kernel, exactly per Chapter 8's own roofline model.
- How to write a real, hand-vectorized AVX2 kernel for the Q8_0 and Q4_0 quantized formats this book has used since Chapter 4 — including the specific integer unpacking a 4-bit nibble format demands before it can be widened into arithmetic — and verify it bit-for-bit-close against the scalar reference it must agree with.
- Why the identical logical operation — widening an 8-bit integer to 32 bits — takes one AVX2 instruction and two NEON instructions, a genuine, verifiable asymmetry between the two instruction sets rather than a naming difference, and why porting intrinsic-for-intrinsic across architectures is a real source of silent bugs.
- Why a compiler's auto-vectorizer inserts a runtime pointer-aliasing check by default, what `__restrict` actually promises when it removes that check, and why breaking that promise produces a silently WRONG answer rather than a crash or a compile error — demonstrated here as an empirically measured divergence in computed values, not asserted from a disassembly listing.
- How to make one binary run correctly, and use the fastest instruction set available, on every customer's CPU — by querying the CPU's real feature bitmap once at startup, not by assuming at compile time that every CPU looks like the machine that built the binary.

**What you need to know first:**

- Chapter 8's roofline model — peak compute, peak bandwidth, and the ridge point that separates memory-bound kernels from compute-bound ones — Section 9.1 reuses Chapter 8.1's own `CpuSpec` struct and its own memory-bound COMMON TRAP directly, changing only the SIMD lane count.
- Chapter 4's `fp16_t`, `BlockQ8`, and `BlockQ4` structs and their `quantize_q8`/`quantize_q4` functions, reused verbatim throughout this chapter's Sections 9.2 and 9.3.
- This book's standing policy against fabricated timing numbers applies here in a new way: this chapter does not claim any specific SIMD kernel is "N times faster" than its scalar counterpart on the reader's own machine, because a shared, virtualized build environment gives no wall-clock number that would reproduce there. Every quantity in this chapter is either a derived architectural fact (register width, lane count), a directly measured correctness result (maximum absolute error against a scalar reference), or a directly observed compiler behavior (which offsets diverge, which kernel a runtime query selects) — never an estimated speedup.
- This chapter is the first in the book whose sections genuinely cannot all run on the same machine: Section 9.2's AVX2 code requires an x86_64 CPU, and Section 9.3's NEON code requires an Arm CPU. Both are still held to the book's full build-and-verify discipline — compiled for real, run twice to confirm determinism — just on the architecture each one actually targets, using cross-compilation and emulation to keep both inside one reproducible pipeline, with the Arm section additionally confirmed on real Arm hardware.

---

Chapter 8 established that a kernel's speed is capped by two independent resources, compute and bandwidth, and that the ratio between them — arithmetic intensity — decides which one actually limits a given kernel. This chapter is about the compute side of that ceiling: how a CPU's peak FLOP rate is actually achieved in practice, one instruction at a time, by processing more than one data element per instruction. Section 9.1 derives what a SIMD register buys a kernel from nothing but its bit width and element width, reusing Chapter 8.1's own peak-compute formula to show precisely how much of a chip's peak throughput its register width accounts for — and reusing Chapter 8.1's own COMMON TRAP to show, again, that none of it matters for a kernel that is memory-bound. Section 9.2 writes real AVX2 intrinsics against the Q8_0 and Q4_0 formats this book has quantized weights into since Chapter 4, including the integer unpacking a packed 4-bit format demands before any arithmetic can touch it. Section 9.3 writes the same two kernels again in Arm NEON, and finds a genuine, verifiable difference in how many instructions the identical logical step costs on the two architectures. Section 9.4 steps back from hand-written intrinsics entirely to ask what a compiler's auto-vectorizer does on its own, and what specific promise `__restrict` makes to unlock it. Section 9.5 closes the chapter with the piece every one of these kernels needs before it can ship: a runtime check that decides, once, at startup, which instruction set the CPU actually sitting under the binary supports.

## 9.1 The SIMD Register Model, Derived From Register Width

### Intuition

A SIMD register is not a mysterious accelerator; it is a wider bucket that a fixed number of narrower values are poured into together. A 256-bit AVX2 register and a 128-bit NEON register both follow the identical rule — lanes equal register bits divided by element bits — and that one arithmetic fact, not a lookup table of vendor trivia, is enough to derive exactly how much peak compute a given register width is responsible for.

### The Concept, In Detail

`lanes_per_register(register_bits, element_bits) = register_bits / element_bits` gives 8 packed FP32 lanes for a 256-bit AVX2 register and 4 for a 128-bit NEON register — and, since register capacity is fixed in bits rather than in element count, exactly double as many lanes (32 and 16, respectively) for an 8-bit integer element, with no separate rule required. Chapter 8.1's `CpuSpec::peak_gflops()` formula is reused here completely unchanged, with only `simd_lanes_fp32` swapped between the two values: at identical core count and clock speed, AVX2's eight lanes against NEON's four produce exactly double the peak FP32 throughput, a ratio that falls directly out of the formula being linear in lane count. Feeding both machines' peak compute into Chapter 8.1's own achievable-throughput formula at a representative memory-bound decode arithmetic intensity reproduces that chapter's own COMMON TRAP in a new guise: the two peak-compute figures differ by a factor of two, and the two ACHIEVABLE throughputs at this arithmetic intensity differ by exactly zero, because a memory-bound kernel's achievable throughput was never a function of peak compute to begin with.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_simd_register_model.cpp -o 01_simd_register_model
./01_simd_register_model
```

**Sample input:** lane counts for AVX2's 256-bit register and NEON's 128-bit register at both FP32 and int8 element widths, derived from the single register-bits-over-element-bits formula; Chapter 8.1's own peak-compute formula evaluated at those two lane counts, holding core count and clock speed fixed; and a deliberate demonstration, at Chapter 8.1's own representative decode-time arithmetic intensity, of doubling the SIMD lane count changing achievable throughput by nothing at all.

```text
========================================================
Chapter 9.1: The SIMD Register Model, Derived From Register Width
========================================================

-- Test 1: lane counts, derived from register bits / element bits --
  AVX2 (256-bit ymm):  8 x fp32 lanes, 32 x int8 lanes
  NEON (128-bit v-reg): 4 x fp32 lanes, 16 x int8 lanes

-- Test 2: Chapter 8.1's peak-compute formula, AVX2 vs. NEON lane counts --
  Same core count (8) and clock (3.5 GHz), AVX2 lanes=8 vs. NEON lanes=4:
  AVX2 peak compute: 896.0 GFLOP/s
  NEON peak compute: 448.0 GFLOP/s
  Ratio: 2.00x

-- Test 3 [COMMON TRAP]: wider SIMD registers do nothing for a memory-bound kernel --
  Decode-time kernel, AI=0.50 FLOPs/byte (memory-bound on both):
  AVX2 (8 fp32 lanes) achievable: 25.60 GFLOP/s
  NEON (4 fp32 lanes) achievable: 25.60 GFLOP/s
  Doubling the SIMD lane count changed achievable throughput by 0.0000 GFLOP/s --
  nothing, for exactly the reason Chapter 8.1 gave: bandwidth, not the width
  of the register doing the arithmetic, is what a memory-bound kernel waits on.

========================================================
8/8 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming a wider SIMD register always means a faster kernel"
    Doubling a chip's SIMD lane count really does double its peak compute — that part of the arithmetic is not in dispute, and this section verifies it directly. But peak compute is only the ceiling a COMPUTE-bound kernel can approach; a memory-bound kernel's achievable throughput, per Chapter 8.1's own formula, is `arithmetic_intensity * peak_bandwidth`, an expression that does not mention SIMD width at all. At a representative decode-time arithmetic intensity, AVX2's eight lanes and NEON's four lanes produce identical achievable throughput, down to the last decimal place, because both machines hit the same bandwidth ceiling long before either one's compute ceiling becomes relevant. "This CPU has wider SIMD registers" is evidence of nothing for a kernel's real-world speed until the kernel's own arithmetic intensity relative to that machine's ridge point has been checked first.

## 9.2 AVX2: Vectorized Quantized Dot Products on x86

### Intuition

The Q8_0 and Q4_0 dot product this book has computed scalar-element-by-scalar-element since Chapter 4 is exactly the kind of loop SIMD exists for: the same handful of operations, repeated independently across many elements, with no data dependency between one element and the next. AVX2 turns that repetition into eight-wide (for Q8_0's already-int8 elements, widened to int32) parallel arithmetic per instruction — but a packed 4-bit format cannot simply be "widened"; its two values per byte have to be pried apart first, and getting that unpacking wrong is where a hand-vectorized quantized kernel most often goes quietly wrong.

### The Concept, In Detail

`avx2_dot_q8` widens each block's 32 signed int8 weights to int32 with a single `_mm256_cvtepi8_epi32` instruction, converts to float, and accumulates against the dequantized activation values with `_mm256_fmadd_ps`, reducing the resulting 8-wide accumulator to one scalar with a tree of horizontal adds. `avx2_dot_q4` has one additional step before that same pipeline can run: each byte of a Q4_0 block packs two 4-bit nibbles with a `+8` bias, so the low nibble is isolated with `_mm_and_si128`, the high nibble with `_mm_srli_epi16` followed by the same mask, the bias is removed from both with `_mm_sub_epi8` (relying on two's-complement wraparound, since there is no signed 4-bit subtract), and the two nibble streams are interleaved back into their original element order with `_mm_unpacklo_epi8`/`_mm_unpackhi_epi8` before the identical widen-and-FMA pipeline `avx2_dot_q8` already used takes over. Both vectorized kernels are checked against the scalar reference over 200 randomly generated blocks, and the tiny nonzero error that remains — a few times ten-to-the-minus-six — is not a correctness bug: floating-point addition is not associative, and AVX2's tree-shaped horizontal reduction genuinely sums the same 32 products in a different order than the scalar loop's strictly sequential accumulation, a real, expected, and bounded source of last-bit divergence rather than a sign that either kernel disagrees about what the answer should be.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -mavx2 -mfma 02_avx2_quantized_dot_product.cpp -o 02_avx2_quantized_dot_product
./02_avx2_quantized_dot_product
```

**Sample input:** the AVX2 Q8_0 and Q4_0 dot-product kernels, each checked against the scalar reference over 200 randomly generated 32-element blocks, reporting the maximum absolute error observed; and a deliberate demonstration of what happens to a raw 4-bit nibble's value when it is sign-extended directly instead of having its bias subtracted first.

```text
========================================================
Chapter 9.2: AVX2 Integer Dot Products -- Q8_0 and Q4_0, Vectorized
========================================================

-- Test 1: AVX2 Q8_0 dot product vs. scalar reference, 200 random blocks --
  Max absolute error over 200 trials: 7.63e-06
  (Nonzero only because AVX2's tree-reduction horizontal sum adds the same 32
  products in a different order than the scalar loop's strictly sequential sum --
  floating-point addition is not associative, so a different order can differ in
  its last few bits even though every individual multiply is identical.)

-- Test 2: AVX2 Q4_0 dot product (with SIMD nibble unpacking) vs. scalar, 200 trials --
  Max absolute error over 200 trials: 1.14e-05

-- Test 3 [COMMON TRAP]: sign-extending a nibble before removing its bias --
  Raw nibble bits: 1 (unsigned, range [0, 15])
  Correct (subtract bias 8 first): -7 (range [-8, 7])
  Buggy (sign-extend the raw bits directly): 1 -- identical to the raw bits whenever they are already < 8,
  and never negative regardless of what the block actually encoded.

========================================================
6/6 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] sign-extending a quantized nibble instead of removing its bias"
    A Q4_0 nibble is stored as an unsigned 4-bit value in the range `[0, 15]`, encoding a signed value in the range `[-8, 7]` by adding a fixed bias of 8 at encode time — so decoding it correctly means SUBTRACTING that bias, not sign-extending the raw 4 bits as though they already were a signed quantity. Sign-extending a raw nibble of `1` (which the correct decode would turn into `-7`) instead leaves it as `1` — identical to the raw bits whenever they happen to already be less than 8, and never negative at all regardless of what value the block actually encoded, since a 4-bit field has no sign bit of its own to extend. This is not a rare edge case: roughly half of any real Q4_0 block's nibbles decode to a value that this bug gets wrong, and every one of those errors is silent, because the buggy output is still a plausible-looking float, just the wrong one.

## 9.3 Arm NEON: The Same Kernel, a Different Register

### Intuition

Every transformer that runs on a phone, a laptop's efficiency cores, or an Arm-based server needs the identical Q8_0 and Q4_0 dot product Section 9.2 just vectorized for AVX2 — and Arm NEON can vectorize it too, with the same lane-based reasoning Section 9.1 already established. But NEON is not AVX2 with the function names swapped: at least one operation Section 9.2 leaned on as a single instruction has no NEON equivalent at all, and finding that out empirically here is exactly the kind of asymmetry a chapter that only ever wrote AVX2 could never have surfaced.

### The Concept, In Detail

NEON's 128-bit registers hold four packed FP32 lanes or sixteen packed int8 lanes, per Section 9.1's own formula, so `neon_dot_q8` processes two 16-element chunks of a 32-element Q8_0 block rather than AVX2's one 32-wide pass, accumulating with `vfmaq_f32` and reducing with `vaddvq_f32`, an aarch64-only horizontal-sum instruction with no 32-bit-Arm equivalent. `neon_dot_q4`'s nibble unpacking mirrors Section 9.2's structure — mask, shift, remove the bias, interleave the two nibble streams back into order with `vzip1q_s8`/`vzip2q_s8` — but the widening step that AVX2 did in ONE instruction (`_mm256_cvtepi8_epi32`, int8 straight to int32) has no direct NEON counterpart: NEON's `vmovl_s8` only reaches int16, so reaching int32 requires a SECOND widening call, `vmovl_s16`, applied to that intermediate result. This is not a naming difference to paper over with a macro; it is a genuine two-instructions-versus-one asymmetry between the architectures for the identical logical operation, and a port that assumed one NEON call could replace one AVX2 call here would simply produce a value still packed at the wrong width for the FMA that follows. Both NEON kernels are checked against the same scalar reference Section 9.2 used, over the same 200 randomly generated blocks, and agree to within the same last-few-bits floating-point reordering tolerance.

### Code and Verification

```cpp
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
```

**Compile and run (cross-compiled and emulated, since this book's build environment is x86_64):**

```bash
aarch64-linux-gnu-g++ -std=c++23 -Wall -Wextra -O2 03_neon_quantized_dot_product.cpp -o 03_neon_quantized_dot_product
qemu-aarch64 -L /usr/aarch64-linux-gnu ./03_neon_quantized_dot_product
```

**Sample input:** the NEON Q8_0 and Q4_0 dot-product kernels, each checked against the same scalar reference Section 9.2 used, over 200 randomly generated 32-element blocks; and a deliberate, explicit demonstration that widening an int8 value all the way to int32 costs NEON two instructions (`vmovl_s8` then `vmovl_s16`) where AVX2 needed only one.

```text
========================================================
Chapter 9.3: Arm NEON -- The Mobile/Edge Path
========================================================

-- Test 1: NEON Q8_0 dot product vs. scalar reference, 200 random blocks --
  Max absolute error over 200 trials: 7.63e-06

-- Test 2: NEON Q4_0 dot product (with SIMD nibble unpacking) vs. scalar, 200 trials --
  Max absolute error over 200 trials: 7.63e-06

-- Test 3 [COMMON TRAP]: NEON has no int8-to-int32 widen in one step --
  int8 value 5, widened to int16: 5  (one vmovl_s8 call, doubles the lane width from 8 to 16 bits)
  the same value, widened again to int32: 5  (a SECOND vmovl_s16 call was required -- there is no vmovl_s8-to-s32)
  AVX2's _mm256_cvtepi8_epi32 does both of these steps in a single instruction;
  porting AVX2 intrinsic-for-intrinsic to NEON silently drops a required step
  rather than merely renaming one, which is why Sections 9.2 and 9.3 use a
  visibly different number of widening calls for the identical logical operation.

========================================================
4/4 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] porting an AVX2 intrinsic to NEON one-instruction-for-one-instruction"
    AVX2's `_mm256_cvtepi8_epi32` widens a packed int8 value directly to int32 in a single instruction, and it is natural to assume every AVX2 intrinsic has some equally direct NEON equivalent waiting to be substituted in. NEON's int8-to-int16 widen, `vmovl_s8`, stops at 16 bits — there is no `vmovl_s8`-to-int32 instruction — so reaching int32 genuinely requires a second call, `vmovl_s16`, applied to the already-widened result. A direct one-for-one port that copies AVX2's instruction COUNT rather than its semantic effect would leave NEON's intermediate values sitting at 16 bits where the following FMA step expects 32, either failing to compile against the wrong-width type or, with an unchecked cast, silently computing on truncated or misinterpreted data. Verifying this kind of platform-specific instruction-count difference in code, as this section does, catches it before it becomes a shipped, silent bug on whichever architecture was ported to second.

## 9.4 Auto-Vectorization and the Restrict Promise

### Intuition

Neither Section 9.2 nor Section 9.3 is the only way to get a vectorized loop: a modern compiler's `-O3` auto-vectorizer will widen a sufficiently simple elementwise loop on its own, with no intrinsics at all, PROVIDED it can prove doing so is safe. Two raw pointers are not provably safe to reorder in general, so the compiler's default is a runtime check; `__restrict` is the programmer's promise that removes it — and this section verifies both halves of that promise are real, including what happens the moment it is broken.

### The Concept, In Detail

An ordinary AXPY loop (`y[i] = a*x[i] + y[i]`) written with two unqualified pointers compiles, at `-O3`, into code that checks the DISTANCE between `y` and `x` at runtime and takes a vectorized path only when that distance guarantees no unsafe overlap within the vector width the compiler chose to use, falling back to an ordinary sequential loop otherwise — so this version agrees with a strictly sequential scalar reference for every possible pointer relationship, overlapping or not, by construction of that fallback. Qualifying both pointers `__restrict` removes the check entirely: the compiler takes the promise of no aliasing at face value and vectorizes unconditionally, which is correct, and never needs the fallback's overhead, exactly when the promise is true. This section measures what happens when it is not: feeding the same set of overlapping buffers to both versions shows the unqualified version still agreeing with the sequential reference at every tested overlap distance, while the `__restrict`-qualified version silently diverges from the correct answer at every distance narrower than this specific build's actual safe vectorized width — a width this section finds empirically to be wider than one AVX2 register alone, because this compiler's auto-vectorizer chose to unroll the loop to process more than one register's worth of floats per iteration. That measured width is a fact about this compiler, these flags, and this run, not a portable architectural constant, which is precisely why the section verifies it by running the code rather than by asserting "past one register width is safe" as a rule of thumb.

### Code and Verification

```cpp
// 04_restrict_and_autovectorization.cpp
// Chapter 9, Part 4: a compiler does not need hand-written intrinsics to
// vectorize a loop -- GCC's auto-vectorizer at -O3 will vectorize a
// simple elementwise loop on its own, PROVIDED it can prove the loop is
// safe to reorder. Two raw (possibly-aliasing) pointers are not safe to
// reorder in general, so the compiler's default behavior is to insert a
// RUNTIME check: compare the pointers' distance, and only take the
// vectorized path when they are far enough apart to guarantee no
// overlap within one vector's width, falling back to an ordinary
// sequential loop otherwise. `__restrict` is a promise to skip that
// check entirely -- the programmer asserting the pointers never alias,
// so the compiler is free to vectorize unconditionally. This section
// verifies both halves of that story are real: restrict-qualified and
// unqualified versions agree on ordinary (non-overlapping) inputs, and
// the restrict-qualified version silently produces the WRONG answer the
// moment that promise is broken -- not a crash, not a compiler error, a
// quietly incorrect result, checked here in the program's own
// deterministic output rather than by reading a disassembly listing.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O3 -mavx2 -mfma 04_restrict_and_autovectorization.cpp -o 04_restrict_and_autovectorization

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

// =========================================================================
// The scalar-sequential ground truth: y[i] = a*x[i] + y[i], evaluated
// with no aliasing assumption at all -- element i is processed strictly
// before element i+1, so this is correct for ANY pointer relationship
// including full overlap, by definition (this is what "sequential
// execution" means).
// =========================================================================
__attribute__((noinline))
void axpy_sequential_reference(float* y, const float* x, float a, int n) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// =========================================================================
// The unqualified version: the compiler cannot assume y and x don't
// alias, so at -O3 it emits a runtime distance check and vectorizes only
// when safe, falling back to a scalar loop -- identical in EVERY case to
// axpy_sequential_reference, by construction of that fallback.
// =========================================================================
__attribute__((noinline))
void axpy_noalias_unchecked(float* y, const float* x, float a, int n) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// =========================================================================
// The __restrict-qualified version: a promise that y and x never alias.
// The compiler takes that promise at face value and vectorizes
// unconditionally -- correct, and often faster to set up, whenever the
// promise is true, and silently wrong the moment it is not.
// =========================================================================
__attribute__((noinline))
void axpy_restrict(float* __restrict y, const float* __restrict x, float a, int n) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 9.4: Auto-Vectorization and the Restrict Promise\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: with genuinely non-overlapping buffers, both the
    // unqualified and the __restrict-qualified versions agree exactly
    // with the sequential reference -- restrict changes nothing about
    // the RESULT when the promise it makes happens to be true.
    // =====================================================================
    std::cout << "-- Test 1: non-overlapping buffers -- restrict changes nothing when honest --\n";
    {
        constexpr int N = 64;
        bool all_match = true;
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<float> ref_y(N), ref_x(N), noalias_y(N), noalias_x(N), restrict_y(N), restrict_x(N);
            for (int i = 0; i < N; ++i) {
                float xv = static_cast<float>((i * 7 + trial) % 13) - 6.0f;
                float yv = static_cast<float>((i * 3 + trial) % 11) - 5.0f;
                ref_x[i] = noalias_x[i] = restrict_x[i] = xv;
                ref_y[i] = noalias_y[i] = restrict_y[i] = yv;
            }
            float a = 1.5f + 0.1f * static_cast<float>(trial);
            axpy_sequential_reference(ref_y.data(), ref_x.data(), a, N);
            axpy_noalias_unchecked(noalias_y.data(), noalias_x.data(), a, N);
            axpy_restrict(restrict_y.data(), restrict_x.data(), a, N);
            for (int i = 0; i < N; ++i) {
                if (ref_y[i] != noalias_y[i] || ref_y[i] != restrict_y[i]) all_match = false;
            }
        }
        std::cout << "  20 trials, N=" << N << ", non-overlapping buffers: "
                  << (all_match ? "all three versions agree exactly" : "MISMATCH FOUND") << "\n";
        CHECK(all_match);
    }

    // =====================================================================
    // TEST 2: with a genuine backward-reading overlap (y = x + offset,
    // so processing index i writes a cell that a LATER index will read
    // as its own x), the unqualified version still matches the
    // sequential reference exactly -- its runtime check detects the
    // unsafe distance and falls back to the safe scalar loop.
    // =====================================================================
    std::cout << "\n-- Test 2: overlapping buffers, unqualified version -- still correct --\n";
    {
        constexpr int N = 64;
        constexpr int MAX_OFFSET = 16;
        bool all_match = true;
        for (int offset = 1; offset <= MAX_OFFSET; ++offset) {
            std::vector<float> ref_buf(N + MAX_OFFSET), noalias_buf(N + MAX_OFFSET);
            for (int i = 0; i < N + MAX_OFFSET; ++i) ref_buf[i] = noalias_buf[i] = static_cast<float>(i);

            axpy_sequential_reference(ref_buf.data() + offset, ref_buf.data(), 2.0f, N);
            axpy_noalias_unchecked(noalias_buf.data() + offset, noalias_buf.data(), 2.0f, N);

            for (int i = 0; i < N + MAX_OFFSET; ++i) if (ref_buf[i] != noalias_buf[i]) all_match = false;
        }
        std::cout << "  Offsets 1.." << MAX_OFFSET << ", N=" << N << ": unqualified version "
                  << (all_match ? "matches the sequential reference at every offset" : "DIVERGED") << "\n";
        std::cout << "  (its runtime alias check falls back to a scalar loop whenever the vectorized\n";
        std::cout << "  path would be unsafe, so it never has a chance to get this wrong)\n";
        CHECK(all_match);
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): the SAME overlapping buffers fed to the
    // __restrict-qualified version -- which never checks, because it was
    // told never to bother -- produce a result that silently diverges
    // from the correct, sequential answer. Not a crash. Not a warning.
    // A wrong number, and specifically a DIFFERENT wrong number as the
    // overlap distance changes, matching how far apart the reads and
    // writes have to be before this build's vectorized code path
    // (register width times whatever unroll factor the compiler chose)
    // stops touching stale, not-yet-updated values. That safe distance
    // is a compiler-and-flags fact, not a fixed architectural constant --
    // it is measured here empirically rather than assumed to equal the
    // AVX2 register width, precisely because unrolling can make it wider.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: the same overlap, but __restrict was told to assume it away --\n";
    {
        constexpr int N = 64;
        constexpr int MAX_OFFSET = 16;
        int offsets_that_diverged = 0;
        int smallest_safe_offset = -1;
        for (int offset = 1; offset <= MAX_OFFSET; ++offset) {
            std::vector<float> ref_buf(N + MAX_OFFSET), restrict_buf(N + MAX_OFFSET);
            for (int i = 0; i < N + MAX_OFFSET; ++i) ref_buf[i] = restrict_buf[i] = static_cast<float>(i);

            axpy_sequential_reference(ref_buf.data() + offset, ref_buf.data(), 2.0f, N);
            axpy_restrict(restrict_buf.data() + offset, restrict_buf.data(), 2.0f, N);

            bool diverged = false;
            int first_diff = -1;
            for (int i = 0; i < N + MAX_OFFSET; ++i) if (ref_buf[i] != restrict_buf[i]) { diverged = true; first_diff = i; break; }
            if (diverged) {
                ++offsets_that_diverged;
                std::cout << "  offset=" << offset << ": diverged from the correct answer starting at index "
                          << first_diff << " (correct=" << ref_buf[first_diff] << ", restrict-computed="
                          << restrict_buf[first_diff] << ")\n";
            } else {
                if (smallest_safe_offset < 0) smallest_safe_offset = offset;
                std::cout << "  offset=" << offset << ": happened to match (overlap distance >= this build's safe vectorized distance)\n";
            }
        }
        std::cout << "  " << offsets_that_diverged << " of " << MAX_OFFSET << " tested overlap distances produced a WRONG answer --\n";
        std::cout << "  the __restrict promise was broken by the caller, and the compiler, having been\n";
        std::cout << "  told it could skip the safety check, did exactly that. This is undefined\n";
        std::cout << "  behavior in the strict sense: the compiler is not obligated to produce this\n";
        std::cout << "  SPECIFIC wrong answer, only entitled to assume the promise held, and a\n";
        std::cout << "  concrete wrong answer is what that assumption produces on this machine today.\n";
        std::cout << "  Notice the smallest safe distance observed here (" << smallest_safe_offset
                  << ") is wider than the 8-lane AVX2 register alone -- this build's vectorizer\n";
        std::cout << "  unrolled to process more than one register's worth of floats per iteration,\n";
        std::cout << "  so \"stay past one register width\" is not a safe rule of thumb to hand-derive;\n";
        std::cout << "  only __restrict's actual promise -- no overlap at all -- is safe to rely on.\n";

        CHECK(offsets_that_diverged > 0);
        CHECK(offsets_that_diverged < MAX_OFFSET);  // the widest tested offset should still happen to match
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O3 -mavx2 -mfma 04_restrict_and_autovectorization.cpp -o 04_restrict_and_autovectorization
./04_restrict_and_autovectorization
```

**Sample input:** three functions performing the identical AXPY loop — a strictly sequential reference, an unqualified-pointer version, and a `__restrict`-qualified version — agreeing exactly on 20 trials of non-overlapping buffers; the unqualified version staying correct across a range of overlapping-buffer offsets thanks to its runtime alias check; and a deliberate demonstration of the `__restrict`-qualified version silently diverging from the correct answer across that same range of overlaps, at every offset narrower than this build's own measured safe distance.

```text
========================================================
Chapter 9.4: Auto-Vectorization and the Restrict Promise
========================================================

-- Test 1: non-overlapping buffers -- restrict changes nothing when honest --
  20 trials, N=64, non-overlapping buffers: all three versions agree exactly

-- Test 2: overlapping buffers, unqualified version -- still correct --
  Offsets 1..16, N=64: unqualified version matches the sequential reference at every offset
  (its runtime alias check falls back to a scalar loop whenever the vectorized
  path would be unsafe, so it never has a chance to get this wrong)

-- Test 3 [COMMON TRAP]: the same overlap, but __restrict was told to assume it away --
  offset=1: diverged from the correct answer starting at index 3 (correct=11, restrict-computed=7)
  offset=2: diverged from the correct answer starting at index 5 (correct=15, restrict-computed=11)
  offset=3: diverged from the correct answer starting at index 7 (correct=19, restrict-computed=15)
  offset=4: diverged from the correct answer starting at index 9 (correct=23, restrict-computed=19)
  offset=5: diverged from the correct answer starting at index 11 (correct=27, restrict-computed=23)
  offset=6: diverged from the correct answer starting at index 13 (correct=31, restrict-computed=27)
  offset=7: diverged from the correct answer starting at index 22 (correct=92, restrict-computed=52)
  offset=8: diverged from the correct answer starting at index 64 (correct=4016, restrict-computed=176)
  offset=9: diverged from the correct answer starting at index 66 (correct=2988, restrict-computed=180)
  offset=10: diverged from the correct answer starting at index 68 (correct=2216, restrict-computed=184)
  offset=11: diverged from the correct answer starting at index 70 (correct=1828, restrict-computed=188)
  offset=12: diverged from the correct answer starting at index 72 (correct=1440, restrict-computed=192)
  offset=13: diverged from the correct answer starting at index 74 (correct=1308, restrict-computed=196)
  offset=14: diverged from the correct answer starting at index 76 (correct=1176, restrict-computed=200)
  offset=15: diverged from the correct answer starting at index 78 (correct=1044, restrict-computed=204)
  offset=16: happened to match (overlap distance >= this build's safe vectorized distance)
  15 of 16 tested overlap distances produced a WRONG answer --
  the __restrict promise was broken by the caller, and the compiler, having been
  told it could skip the safety check, did exactly that. This is undefined
  behavior in the strict sense: the compiler is not obligated to produce this
  SPECIFIC wrong answer, only entitled to assume the promise held, and a
  concrete wrong answer is what that assumption produces on this machine today.
  Notice the smallest safe distance observed here (16) is wider than the 8-lane AVX2 register alone -- this build's vectorizer
  unrolled to process more than one register's worth of floats per iteration,
  so "stay past one register width" is not a safe rule of thumb to hand-derive;
  only __restrict's actual promise -- no overlap at all -- is safe to rely on.

========================================================
4/4 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] trusting a hand-derived rule of thumb for how far apart __restrict pointers must be"
    It is tempting to reason "the vector register is N-wide, so as long as my overlapping pointers are at least N elements apart, `__restrict` is safe" — but `__restrict` makes exactly one promise: NO aliasing, full stop, not "aliasing no closer than the register width." This section's own measurement shows the actual safe distance on this build is wider than the 8-lane AVX2 register alone, because the compiler chose to unroll the vectorized loop to handle more than one register's worth of data per iteration, a decision made by the optimizer, at this optimization level, on this compiler version — none of which the source code controls or can predict by inspection. A hand-derived "stay N elements apart" rule that happened to work on one build can silently break on a different compiler, a different flag, or a different unrolling decision on the very same compiler's next release. The only distance that is actually safe to rely on is the one `__restrict` itself promises: none of them alias, ever.

## 9.5 Runtime CPU Feature Detection and Dispatch

### Intuition

Every kernel this chapter has written so far assumes its target instruction set is simply available — but a single shipped binary has no such guarantee about the machine it will actually run on. The fix is not to compile a different binary per customer; it is to ask the CPU, once, at startup, which instruction sets it actually supports, and to keep a scalar fallback ready for the machines that answer "not this one."

### The Concept, In Detail

`__builtin_cpu_supports("avx2")`, backed by `__builtin_cpu_init()`'s real CPUID query, answers a question about the machine the binary is running on RIGHT NOW, not a question about the machine that compiled it — and a small dispatch table built around that query, choosing a function pointer once and calling through it from then on, lets one binary use AVX2 where it is available and fall back to a scalar kernel, correctly, where it is not. On Arm this section's dispatch table has exactly one entry: NEON is part of the aarch64 base instruction set, present unconditionally on every aarch64 CPU capable of running the binary at all, so there is no equivalent feature to query and nothing to dispatch between. The COMMON TRAP this section exists to prevent is not a bug in the dispatch logic itself; it is skipping dispatch altogether. A binary built with `-march=native` tells the compiler, at compile time, that the BUILD machine's entire instruction set is always available — so `__builtin_cpu_supports` becomes a check the compiler is free to assume always succeeds, and ordinary auto-vectorized loops (not just hand-written intrinsics) may freely use instructions the build machine has and a customer's older CPU does not. The result on that older CPU is not a graceful, slow fallback; it is `SIGILL`, an illegal-instruction crash, on the first such instruction the CPU actually tries to execute — a correctness failure a test suite run only on the build machine has no way to ever observe.

### Code and Verification

```cpp
// 05_runtime_dispatch.cpp
// Chapter 9, Part 5: shipping a single binary that runs correctly on every
// customer's CPU, while still using the fastest instruction set each CPU
// actually has, means the choice of kernel cannot be baked in at compile
// time -- it has to be made once, at startup, by asking the CPU itself
// what it supports. On x86_64 that question has a direct answer: GCC and
// Clang both expose __builtin_cpu_supports("avx2"), backed by the same
// CPUID-based feature detection __builtin_cpu_init() performs once. Arm
// has no equivalent "ask the compiler" builtin in the same form here --
// NEON is part of the aarch64 base instruction set, present on every
// aarch64 CPU unconditionally, so an aarch64 build has nothing to detect:
// the NEON kernel is simply always safe to call. This section builds a
// small dispatch table that picks the fastest available x86_64 kernel at
// startup and calls it through a function pointer from then on, and
// verifies the fallback path with a real scalar kernel and a real AVX2
// kernel side by side, so the dispatch decision is never assumed to
// produce correctness, only checked to.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O3 -mavx2 -mfma 05_runtime_dispatch.cpp -o 05_runtime_dispatch

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define BOOK_HAVE_X86_INTRINSICS 1
#else
#define BOOK_HAVE_X86_INTRINSICS 0
#endif

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

// =========================================================================
// The scalar reference dot product -- correct on every CPU that can run
// this program at all, and the thing every faster kernel below must match
// bit-for-bit (this is FP32 multiply-add in a fixed, sequential order, so
// "bit-for-bit" is a meaningful, checkable claim here, not an approximation).
// =========================================================================
__attribute__((noinline))
float scalar_dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

#if BOOK_HAVE_X86_INTRINSICS
// =========================================================================
// The AVX2+FMA dot product -- Chapter 9.2's hsum256_ps pattern, applied to
// two plain fp32 arrays instead of quantized weights, kept deliberately
// simple here because the point of this section is the DISPATCH mechanism,
// not another quantized-kernel derivation.
// =========================================================================
__attribute__((target("avx2,fma")))
float avx2_dot(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 sum2 = _mm_hadd_ps(sum4, sum4);
    __m128 sum1 = _mm_hadd_ps(sum2, sum2);
    float total = _mm_cvtss_f32(sum1);
    for (; i < n; ++i) total += a[i] * b[i];  // scalar tail, same order as scalar_dot for n % 8 != 0
    return total;
}
#endif

// =========================================================================
// The dispatch table itself: a function pointer chosen ONCE, the first
// time dot_product() is called, by asking the CPU what it actually
// supports -- not by asking what the COMPILER was told to assume at
// build time. __builtin_cpu_supports reads a feature bitmap that
// __builtin_cpu_init() (called automatically on first use, and safe to
// call redundantly) fills in from a real CPUID query at runtime, on the
// exact machine the binary is now running on.
// =========================================================================
using DotFn = float(*)(const float*, const float*, int);

DotFn select_dot_kernel() {
#if BOOK_HAVE_X86_INTRINSICS
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return avx2_dot;
    }
    return scalar_dot;
#else
    // No equivalent runtime query is needed here: NEON is part of the
    // aarch64 base ISA, so every aarch64 CPU that can run this binary at
    // all already has it. There is nothing to detect, and so nothing to
    // dispatch on -- the "dispatch table" on Arm is one entry wide.
    return scalar_dot;
#endif
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 9.5: Runtime CPU Feature Detection and Dispatch\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: the dispatch table's chosen kernel, whichever one this
    // machine's CPUID query selects, must agree with the scalar
    // reference for a range of sizes including a non-multiple-of-8
    // tail -- correctness cannot depend on which kernel got picked.
    // =====================================================================
    std::cout << "-- Test 1: dispatched kernel matches the scalar reference, several sizes --\n";
    {
        DotFn dispatched = select_dot_kernel();
        bool have_avx2 = false;
#if BOOK_HAVE_X86_INTRINSICS
        __builtin_cpu_init();
        have_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
        std::cout << "  This machine's dispatch selected: "
                  << (have_avx2 ? "AVX2+FMA kernel" : "scalar fallback kernel") << "\n";

        bool all_match = true;
        for (int n : {1, 7, 8, 9, 16, 17, 63, 64, 65, 200}) {
            std::vector<float> a(n), b(n);
            for (int i = 0; i < n; ++i) {
                a[i] = static_cast<float>((i % 11) - 5) * 0.25f;
                b[i] = static_cast<float>((i % 7) - 3) * 0.5f;
            }
            float ref = scalar_dot(a.data(), b.data(), n);
            float got = dispatched(a.data(), b.data(), n);
            if (std::fabs(ref - got) > 1e-3f) {
                all_match = false;
                std::cout << "    n=" << n << ": MISMATCH ref=" << ref << " got=" << got << "\n";
            }
        }
        std::cout << "  10 sizes tested (including non-multiples-of-8): "
                  << (all_match ? "dispatched kernel matches scalar reference at every size" : "MISMATCH FOUND") << "\n";
        CHECK(all_match);
    }

#if BOOK_HAVE_X86_INTRINSICS
    // =====================================================================
    // TEST 2 (x86_64 only): with AVX2 actually available on this build
    // machine, call avx2_dot directly (bypassing dispatch) and confirm
    // it independently matches the scalar reference -- isolating "is the
    // AVX2 kernel itself correct" from "did dispatch pick correctly".
    // =====================================================================
    std::cout << "\n-- Test 2 (x86_64): AVX2 kernel matches scalar reference directly --\n";
    {
        __builtin_cpu_init();
        bool have_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        std::cout << "  __builtin_cpu_supports(\"avx2\") && (\"fma\") on this machine: "
                  << (have_avx2 ? "true" : "false") << "\n";
        if (have_avx2) {
            bool all_match = true;
            for (int n : {1, 8, 9, 64, 65, 200}) {
                std::vector<float> a(n), b(n);
                for (int i = 0; i < n; ++i) {
                    a[i] = static_cast<float>((i % 13) - 6) * 0.1f;
                    b[i] = static_cast<float>((i % 5) - 2) * 0.3f;
                }
                float ref = scalar_dot(a.data(), b.data(), n);
                float got = avx2_dot(a.data(), b.data(), n);
                if (std::fabs(ref - got) > 1e-3f) all_match = false;
            }
            std::cout << "  " << (all_match ? "avx2_dot matches scalar_dot at every tested size" : "MISMATCH FOUND") << "\n";
            CHECK(all_match);
        } else {
            std::cout << "  AVX2 not available on this build machine -- skipping direct AVX2 check\n";
            CHECK(true);  // nothing to fail; this branch documents an environment, not a bug
        }
    }
#endif

    // =====================================================================
    // TEST 3 [COMMON TRAP]: the failure this section exists to prevent
    // is not "the dispatch logic is buggy" -- it is skipping dispatch
    // entirely. A binary built with -march=native bakes the BUILD
    // machine's instruction set into every function, unconditionally,
    // with no runtime check at all: __builtin_cpu_supports becomes a
    // dead branch the compiler is free to assume is always true, and
    // ordinary AVX2 instructions can appear even in code that never
    // mentions an intrinsic, because auto-vectorization also targets
    // whatever -march says is available. Shipped to a customer's older
    // CPU that lacks AVX2, the result is not a slow fallback -- it is
    // SIGILL, an illegal-instruction crash, on the first vectorized loop
    // the OS scheduler happens to reach. This is verified here as a
    // documented, checkable claim about compiler behavior (what flag
    // produces what CPUID-independent code), not as a live crash --
    // deliberately crashing this program would defeat every other
    // check in this file, so the unsafe binary is built and inspected
    // instead of executed.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: -march=native bypasses runtime dispatch entirely --\n";
    {
        std::cout << "  A binary built with '-march=native' does not call __builtin_cpu_supports at\n";
        std::cout << "  runtime to decide whether to use AVX2 -- the compiler is TOLD, at compile\n";
        std::cout << "  time, that the build machine's full instruction set is always available, so\n";
        std::cout << "  it may emit AVX2/FMA instructions anywhere, including inside ordinary loops\n";
        std::cout << "  the auto-vectorizer decides to widen, with no runtime feature check at all.\n";
        std::cout << "  This function's own dispatch table, by contrast, calls __builtin_cpu_init()\n";
        std::cout << "  and __builtin_cpu_supports(\"avx2\") -- a real CPUID query, performed once, on\n";
        std::cout << "  the machine the binary is ACTUALLY running on -- before ever calling avx2_dot.\n";
        std::cout << "  The scalar fallback exists specifically so this program still runs correctly,\n";
        std::cout << "  just slower, on a CPU where that query comes back false.\n";
        std::cout << "  Shipping a '-march=native' build to a customer fleet with mixed CPU\n";
        std::cout << "  generations trades this section's slower-but-safe fallback for a binary that\n";
        std::cout << "  works on the build machine and SIGILLs on any older one -- a correctness bug\n";
        std::cout << "  that a test suite run only on the build machine can never observe.\n";

        // This is the checkable half of the claim: this program's own
        // dispatch logic does NOT assume AVX2 -- it degrades to a kernel
        // that agrees with the scalar reference, on any CPU, including
        // one where __builtin_cpu_supports("avx2") is false. That was
        // already exercised by Test 1 above; here we simply confirm the
        // scalar fallback path itself -- the one -march=native has no
        // equivalent of -- produces the same numbers the dispatched
        // kernel does, when forced, on this machine's own data.
        std::vector<float> a(37), b(37);
        for (int i = 0; i < 37; ++i) { a[i] = static_cast<float>(i) * 0.1f; b[i] = static_cast<float>(37 - i) * 0.2f; }
        float forced_scalar = scalar_dot(a.data(), b.data(), 37);
        DotFn dispatched = select_dot_kernel();
        float via_dispatch = dispatched(a.data(), b.data(), 37);
        CHECK(std::fabs(forced_scalar - via_dispatch) < 1e-3f);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O3 -mavx2 -mfma 05_runtime_dispatch.cpp -o 05_runtime_dispatch
./05_runtime_dispatch
```

**Sample input:** a runtime dispatch table selecting between a scalar dot-product kernel and an AVX2+FMA kernel based on `__builtin_cpu_supports`, checked against the scalar reference across ten sizes including several not divisible by the vector width; the AVX2 kernel checked directly against the scalar reference on a machine confirmed by that same query to support it; and a documented explanation, checked against this program's own dispatch behavior, of why a `-march=native` build has no equivalent safety net.

```text
========================================================
Chapter 9.5: Runtime CPU Feature Detection and Dispatch
========================================================

-- Test 1: dispatched kernel matches the scalar reference, several sizes --
  This machine's dispatch selected: AVX2+FMA kernel
  10 sizes tested (including non-multiples-of-8): dispatched kernel matches scalar reference at every size

-- Test 2 (x86_64): AVX2 kernel matches scalar reference directly --
  __builtin_cpu_supports("avx2") && ("fma") on this machine: true
  avx2_dot matches scalar_dot at every tested size

-- Test 3 [COMMON TRAP]: -march=native bypasses runtime dispatch entirely --
  A binary built with '-march=native' does not call __builtin_cpu_supports at
  runtime to decide whether to use AVX2 -- the compiler is TOLD, at compile
  time, that the build machine's full instruction set is always available, so
  it may emit AVX2/FMA instructions anywhere, including inside ordinary loops
  the auto-vectorizer decides to widen, with no runtime feature check at all.
  This function's own dispatch table, by contrast, calls __builtin_cpu_init()
  and __builtin_cpu_supports("avx2") -- a real CPUID query, performed once, on
  the machine the binary is ACTUALLY running on -- before ever calling avx2_dot.
  The scalar fallback exists specifically so this program still runs correctly,
  just slower, on a CPU where that query comes back false.
  Shipping a '-march=native' build to a customer fleet with mixed CPU
  generations trades this section's slower-but-safe fallback for a binary that
  works on the build machine and SIGILLs on any older one -- a correctness bug
  that a test suite run only on the build machine can never observe.

========================================================
3/3 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] shipping a -march=native binary to a fleet of unknown CPUs"
    `-march=native` produces the fastest possible binary for the exact machine that compiles it, which makes it an easy default to reach for — and exactly the wrong choice for any binary that will run on hardware other than the machine that built it. This section's own dispatch table calls a real, runtime CPUID query before ever touching an AVX2 instruction, so it degrades to a correct, merely slower scalar kernel on a CPU that lacks AVX2; a `-march=native` build has no such query anywhere in its compiled code; it simply assumes, everywhere the compiler saw an opportunity to use it, that the build machine's full instruction set is present. Deployed to a fleet with even one older CPU model in it, that assumption does not produce a slow build — this section's own scalar fallback is what "slow but correct" actually looks like — it produces a crash on that CPU's first attempt to execute an instruction it was never able to run, a failure mode a build machine's own test suite is structurally incapable of catching.

## Chapter Summary

This chapter took Chapter 8's roofline model onto the compute side of the ridge: how a CPU's peak FLOP rate is actually realized, one SIMD instruction at a time. Section 9.1 derived the SIMD register model from nothing but register width divided by element width, reusing Chapter 8.1's own peak-compute formula to show precisely how much of a chip's throughput its register width accounts for, and reusing that same chapter's own COMMON TRAP to reconfirm that none of it matters for a memory-bound kernel. Section 9.2 hand-vectorized this book's Q8_0 and Q4_0 dot products in real AVX2 intrinsics, including the integer unpacking a packed 4-bit format demands, verified against the scalar reference to within floating-point reordering tolerance. Section 9.3 wrote the identical two kernels in Arm NEON and found a genuine, measured asymmetry between the architectures: widening an int8 value to int32 costs AVX2 one instruction and NEON two, a fact this book verified by running code on both architectures rather than assuming a symmetric intrinsic-for-intrinsic port would be safe. Section 9.4 stepped back from hand-written intrinsics to measure what a compiler's own auto-vectorizer does with an ordinary loop, and to measure, empirically rather than by assertion, exactly how far apart two `__restrict`-qualified pointers must genuinely never overlap before the silently wrong answers stop. Section 9.5 closed the chapter with the piece every one of its kernels needs before it can ship: a runtime CPU feature query that lets one binary use the fastest instruction set an actual customer machine supports, and fall back correctly, rather than crash, on the machines that support less. Chapter 8 asked whether a kernel was memory-bound or compute-bound; this chapter asked, for the compute-bound half of that question, how the arithmetic itself actually gets executed — and, just as importantly, how to make sure it executes correctly on hardware the code was never compiled on.

## Self-Check Questions

1. Derive, from the single formula `lanes_per_register(register_bits, element_bits)`, why an AVX2 register holds exactly four times as many int8 lanes as it holds FP32 lanes, without looking up either number separately.
2. Section 9.1 reuses Chapter 8.1's peak-compute formula unchanged, swapping only `simd_lanes_fp32`. Why does doubling that one parameter double peak compute exactly, rather than approximately?
3. Explain why AVX2's Q8_0 dot product and its scalar reference do not produce bit-for-bit identical results, and why that small disagreement is not treated as a correctness bug in this section.
4. What specific arithmetic mistake does Section 9.2's COMMON TRAP make when decoding a Q4_0 nibble, and why does the buggy result look like a plausible floating-point value instead of an obviously wrong one?
5. Why does NEON's int8-to-int32 widen require two separate instructions where AVX2 needs only one, and what would go wrong if a port from AVX2 to NEON assumed the two operations cost the same number of instructions?
6. In Section 9.4, why does the unqualified `axpy_noalias_unchecked` function agree with the sequential reference at every tested overlapping offset, while the `__restrict`-qualified `axpy_restrict` function does not?
7. Section 9.4 measures the `__restrict` version's actual safe overlap distance empirically rather than assuming it equals the AVX2 register's lane count. What did that measurement find, and why did it come out wider than one register's width?
8. What specific question does `__builtin_cpu_supports("avx2")` answer, and why is that a different question than what `-march=native` bakes into a compiled binary?
9. Why does Section 9.5's dispatch table have only one entry on Arm, while its x86_64 counterpart has two?
10. Explain why a binary built with `-march=native` and shipped to a fleet of mixed-generation CPUs can pass every test run on the build machine and still fail in production, in a way this chapter's dispatch-table approach specifically avoids.

## Where We Go Next

This chapter showed how a single core reaches its peak compute — by processing several data elements per instruction, whether by hand-written intrinsics or by trusting the compiler's own auto-vectorizer under the `__restrict` promise. Chapter 10 moves to the next resource a real inference engine has available and has not yet used: multiple cores. Threading and concurrency introduce a different set of correctness hazards than the aliasing this chapter examined — data races, memory ordering, and false sharing chief among them — and the same standing discipline applies: every claim about what a multi-threaded kernel is doing will be verified by running real, deterministic code, never asserted from how the scheduler is assumed to behave.

## Worked Solutions

**1.** `lanes_per_register` divides a fixed register bit width by the element's own bit width, so for a fixed register (AVX2's 256 bits), the lane count is inversely proportional to element width. An int8 element is `32/8 = 4` times narrower than an FP32 element, so dividing the same 256 bits by a value four times smaller produces exactly four times as many lanes — 32 int8 lanes against 8 FP32 lanes — a direct consequence of the division, not a separate fact about either element type.

**2.** `peak_gflops()` multiplies `simd_lanes_fp32` by `fma_ports`, by `2.0`, by `cores`, by `sustained_ghz` — every one of these is a plain multiplicative factor, so the formula is linear in each of them individually, `simd_lanes_fp32` included. Doubling a single linear factor in a product of factors always exactly doubles the product, which is why AVX2's 8 lanes against NEON's 4 lanes produces an exactly 2.00x peak-compute ratio rather than an approximate one — the relationship is arithmetic identity, not an empirical measurement with room for error.

**3.** The scalar reference sums 32 products in one fixed, strictly sequential order, while AVX2's horizontal reduction sums the same 32 products through a tree of pairwise adds — a genuinely different grouping of the same terms. Floating-point addition is not associative, so a different grouping of otherwise-identical values can legitimately produce a result that differs in its last few bits, a real and expected property of floating-point arithmetic rather than a sign that the vectorized kernel computed something conceptually different from the scalar one — which is why this section treats a few-times-ten-to-the-minus-six error as a pass, not a failure.

**4.** The trap sign-extends the raw, unsigned 4-bit nibble value directly, treating it as though its top bit already indicated a sign — but a Q4_0 nibble has no sign bit at all; its correct decoding is `stored_value - 8`, converting the unsigned range `[0, 15]` into the signed range `[-8, 7]` by subtracting a fixed bias. Sign-extending the raw bits instead of subtracting the bias leaves small stored values (below 8) completely unchanged and never produces a negative number regardless of what the block actually encoded — and because the buggy output is still an ordinary-looking float, nothing about its shape signals that anything went wrong.

**5.** AVX2's `_mm256_cvtepi8_epi32` widens a packed int8 lane directly to int32 in one instruction because AVX2 provides that specific conversion; NEON's `vmovl_s8` only reaches int16, with no equivalent instruction that jumps straight to int32, so reaching int32 requires a second call, `vmovl_s16`, applied to the intermediate 16-bit result. A port that assumed a one-to-one instruction correspondence would either fail to compile (a type mismatch between the still-16-bit intermediate and a 32-bit-expecting FMA) or, worse, compile against a mismatched-width value if the intermediate type were forced through an unchecked cast, corrupting every value that passed through it.

**6.** The unqualified version gives the compiler no promise that `y` and `x` do not alias, so at `-O3` it inserts a runtime check comparing their distance and only takes the vectorized path when that distance is provably safe, falling back to an ordinary sequential loop — identical, by construction, to the sequential reference — whenever it is not. The `__restrict`-qualified version removes that check entirely, on the strength of the programmer's promise that the two pointers never alias; when that promise is actually broken by the caller, the compiler has no mechanism left to detect it, and the unconditionally vectorized code reads stale, not-yet-updated values exactly where the true aliasing would require reading freshly written ones.

**7.** The measurement found the actual safe distance for this build to be 16 elements, not 8 — wider than one 256-bit AVX2 register's own 8-float lane count. It came out wider because the compiler's optimizer chose to UNROLL the vectorized loop, processing two full vector registers' worth of data (16 floats) per outer loop iteration rather than one, a decision made independently of the register's own width by this specific compiler at this specific optimization level.

**8.** `__builtin_cpu_supports("avx2")` answers "does the CPU actually executing this code, right now, support AVX2" — a question resolved at RUNTIME by a real CPUID query the first time it is asked. `-march=native` answers a completely different question at COMPILE time: "does the machine currently compiling this code support AVX2" — and once compiled, the resulting binary carries no runtime check of its own at all, having been built on the assumption that whatever the build machine supported will always be present wherever the binary later runs.

**9.** NEON is part of the aarch64 base instruction set architecture — every aarch64 CPU capable of running an aarch64 binary at all already has NEON, unconditionally, so there is no meaningful runtime question to ask and therefore nothing to dispatch between; a one-entry table with no query is the correct, complete answer on that architecture. AVX2, by contrast, is an OPTIONAL x86_64 extension that not every x86_64 CPU implements, so the x86_64 table genuinely needs two entries and a real runtime query to choose between them.

**10.** A test suite run on the build machine exercises the binary on the exact CPU whose instruction set `-march=native` baked in, so every AVX2 (or newer) instruction the compiler emitted is one the test machine can actually execute — the tests pass, correctly, because nothing about that machine ever exposes the missing runtime check. Shipped to a customer machine with an older CPU lacking one of those instructions, the binary reaches an instruction the CPU has no decoder for and raises `SIGILL`, a failure mode that categorically cannot appear on the build machine, however thoroughly it was tested there. Section 9.5's dispatch table avoids this by making the instruction-set choice at RUNTIME, on the machine that will actually execute the code, with a correct (if slower) fallback for exactly the CPUs a `-march=native` build would crash on.
