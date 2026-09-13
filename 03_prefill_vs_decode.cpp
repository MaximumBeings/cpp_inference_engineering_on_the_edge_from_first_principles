// 03_prefill_vs_decode.cpp
// Chapter 8, Part 3: prefill and decode use the exact same weight
// matrices but are, arithmetically, different problems. Prefill
// multiplies a weight matrix by MANY token columns at once (a GEMM);
// decode multiplies the same matrix by a SINGLE column (a GEMV). This
// file measures both, in FP32 and in Chapter 4's Q4_0 format, and finds
// the actual sequence length at which a kernel crosses from memory-bound
// to compute-bound -- rather than asserting a single "prefill is compute
// bound, decode is memory bound" rule that turns out to depend on how
// long the sequence is.
//
// The weight matrix, and the input/output activations, are genuinely
// two-dimensional -- exactly the case Chapter 2 built std::mdspan for --
// so this file views them through std::mdspan rather than through bare
// row/col integer arithmetic over a flat buffer. A small, real GEMM
// kernel runs over these views at Chapter 4-6's illustrative small scale
// (DIM=64, D_FF=256) so its FLOP count is a genuine consequence of code
// that actually executes, the same discipline Section 8.2 used. The
// BYTE count stays a stated modeling convention (each weight element
// touched once, reused across every column of the batch) -- exactly as
// disclosed below -- because how many times a byte is actually re-fetched
// from DRAM depends on cache residency and tiling, not on loop syntax;
// this file cross-checks that convention against the real kernel's own
// measured bytes at matching shapes before relying on it at the full
// FFN projection size (14336 x 4096), which is never materialized as an
// actual 235 MB buffer just to prove arithmetic already checked correct.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -I_vendor_mdspan/include 03_prefill_vs_decode.cpp -o out03

#include <mdspan/mdspan.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <span>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

// =========================================================================
// CHAPTER 4's fp16_t and BlockQ4 (reused verbatim -- exact byte layout,
// 18 bytes/block: a 2-byte fp16 scale plus 16 bytes of packed nibbles)
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
struct BlockQ4 { fp16_t scale; uint8_t nibbles[16]; };  // 32 values, 18 bytes total
#pragma pack(pop)

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
float dequant_q4_elem(const BlockQ4& b, int j) {
    // j in [0, 32): which of the block's 32 packed values.
    uint8_t p = b.nibbles[j / 2];
    int nib = (j % 2 == 0) ? (p & 0xF) : (p >> 4);
    return static_cast<float>(nib - 8) * static_cast<float>(b.scale);
}

// =========================================================================
// COST COUNTER (Section 8.2's convention, reused)
// =========================================================================
struct Cost {
    long long flops = 0;
    long long bytes = 0;
    void flop(long long n = 1) { flops += n; }
    void access(long long n_bytes) { bytes += n_bytes; }
    double arithmetic_intensity() const { return static_cast<double>(flops) / static_cast<double>(bytes); }
};

// =========================================================================
// REAL, SMALL-SCALE KERNELS -- genuine std::mdspan views over genuine
// backing buffers, at Chapter 4-6's illustrative DIM=64, D_FF=256 scale.
// Loop order is (out, in, seq): each weight element W[o,i] is touched
// exactly once per (o,i) pair and its FMA is applied to every column of
// the batch before moving on -- the real access pattern a tiled GEMM
// uses to keep a weight row resident across the whole column sweep, and
// exactly why GEMM's arithmetic intensity rises with batch size while
// GEMV's does not. FLOPs are counted as the arithmetic actually runs;
// bytes are counted once per element touched, matching that reuse.
// =========================================================================
using Matrix = std::mdspan<float, std::dextents<size_t, 2>>;
using Q4Matrix = std::mdspan<BlockQ4, std::dextents<size_t, 2>>;

void linear_fp32_real(Matrix W, Matrix X, Matrix Y, Cost& cost) {
    size_t n_out = W.extent(0), n_in = W.extent(1), seq = X.extent(1);
    for (size_t s = 0; s < seq; ++s) for (size_t o = 0; o < n_out; ++o) Y[o, s] = 0.0f;
    for (size_t o = 0; o < n_out; ++o) {
        for (size_t i = 0; i < n_in; ++i) {
            float w = W[o, i];
            cost.access(4);  // W[o,i] read once, reused across every column below
            for (size_t s = 0; s < seq; ++s) {
                Y[o, s] += w * X[i, s];
                cost.flop(2);  // one FMA
            }
        }
    }
    cost.access(static_cast<long long>(n_in) * seq * 4);   // X, read once per element
    cost.access(static_cast<long long>(n_out) * seq * 4);  // Y, written once per element
}

void linear_q4_real(Q4Matrix Wq, Matrix X, Matrix Y, Cost& cost) {
    size_t n_out = Wq.extent(0), nblocks = Wq.extent(1), seq = X.extent(1);
    for (size_t s = 0; s < seq; ++s) for (size_t o = 0; o < n_out; ++o) Y[o, s] = 0.0f;
    for (size_t o = 0; o < n_out; ++o) {
        for (size_t nb = 0; nb < nblocks; ++nb) {
            const BlockQ4& blk = Wq[o, nb];
            cost.access(sizeof(BlockQ4));  // whole packed block read once
            for (int j = 0; j < 32; ++j) {
                float w = dequant_q4_elem(blk, j);
                cost.flop(1);  // dequant: nibble-to-signed-int-to-float, counted as one extra op
                size_t i = nb * 32 + j;
                for (size_t s = 0; s < seq; ++s) {
                    Y[o, s] += w * X[i, s];
                    cost.flop(2);  // one FMA
                }
            }
        }
    }
    cost.access(static_cast<long long>(Wq.extent(1)) * 32 * seq * 4);  // X, read once per element
    cost.access(static_cast<long long>(n_out) * seq * 4);              // Y, written once per element
}

// =========================================================================
// CLOSED-FORM COST FORMULAS -- the same accounting as the two real
// kernels above, but as pure arithmetic over shape parameters, so Test 2
// can search seq_len = 1..100000 and Test 3 can use the FFN projection's
// real dimensions (14336 x 4096) without allocating either buffer.
// Test 0 below proves these formulas exactly match the real kernels'
// measured Cost at identical shapes before they are trusted at a scale
// no longer worth materializing.
// =========================================================================
Cost linear_fp32_cost(long long n_out, long long n_in, long long seq) {
    Cost cost;
    cost.access(n_out * n_in * 4);   // W, read once
    cost.access(n_in * seq * 4);     // X, read once per element
    cost.access(n_out * seq * 4);    // Y, written once per element
    cost.flop(2 * n_out * n_in * seq);
    return cost;
}
Cost linear_q4_cost(long long n_out, long long n_in, long long seq) {
    Cost cost;
    long long nblocks_per_row = (n_in + 31) / 32;
    cost.access(n_out * nblocks_per_row * static_cast<long long>(sizeof(BlockQ4)));
    cost.access(n_in * seq * 4);
    cost.access(n_out * seq * 4);
    // Dequantizing a weight element (nibble -> signed int -> float) happens
    // ONCE per weight, exactly like reading it -- the dequantized value is
    // then reused across every column of the batch, same as the FP32 case.
    cost.flop(n_out * n_in);           // one dequant op per weight element
    cost.flop(2 * n_out * n_in * seq); // one FMA per (out, in, col)
    return cost;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 8.3: Prefill (GEMM) vs. Decode (GEMV)\n";
    std::cout << "========================================================\n\n";

    constexpr double RIDGE_POINT = 896.0 / 51.2;  // Section 8.1's derived ridge, 17.5 FLOPs/byte
    constexpr int DIM = 64, D_FF = 256;  // Chapter 4-6's illustrative small-model scale, reused here

    // =====================================================================
    // TEST 0: the real, mdspan-viewed kernel and the closed-form formula
    // agree exactly at matching shapes -- so the formula used at full FFN
    // scale below is not an unverified assertion.
    // =====================================================================
    std::cout << "-- Test 0: real mdspan-based kernel vs. closed-form formula, same shape --\n";
    {
        constexpr int SEQ = 5;
        std::vector<float> w_data(static_cast<size_t>(D_FF) * DIM), x_data(static_cast<size_t>(DIM) * SEQ),
                            y_data(static_cast<size_t>(D_FF) * SEQ);
        for (size_t i = 0; i < w_data.size(); ++i) w_data[i] = (static_cast<float>(i % 13) - 6.0f) * 0.05f;
        for (size_t i = 0; i < x_data.size(); ++i) x_data[i] = (static_cast<float>(i % 9) - 4.0f) * 0.1f;

        Matrix W(w_data.data(), D_FF, DIM), X(x_data.data(), DIM, SEQ), Y(y_data.data(), D_FF, SEQ);
        Cost real_cost;
        linear_fp32_real(W, X, Y, real_cost);
        Cost formula_cost = linear_fp32_cost(D_FF, DIM, SEQ);

        std::cout << "  Real kernel:    FLOPs=" << real_cost.flops << " Bytes=" << real_cost.bytes << "\n";
        std::cout << "  Closed formula: FLOPs=" << formula_cost.flops << " Bytes=" << formula_cost.bytes << "\n";
        CHECK(real_cost.flops == formula_cost.flops);
        CHECK(real_cost.bytes == formula_cost.bytes);

        // Correctness: check Y[0,0] against an independently computed dot product.
        double expect_y00 = 0.0;
        for (int i = 0; i < DIM; ++i) expect_y00 += static_cast<double>(W[0, i]) * X[i, 0];
        CHECK_NEAR(static_cast<double>(Y[0, 0]), expect_y00, 1e-4);

        // Same cross-check for the Q4 path: quantize W row-by-row into BlockQ4.
        long long nblocks = (DIM + 31) / 32;
        std::vector<BlockQ4> wq_data(static_cast<size_t>(D_FF) * nblocks);
        Q4Matrix Wq(wq_data.data(), D_FF, static_cast<size_t>(nblocks));
        for (int o = 0; o < D_FF; ++o)
            for (long long nb = 0; nb < nblocks; ++nb)
                Wq[o, nb] = quantize_q4(&w_data[static_cast<size_t>(o) * DIM + nb * 32]);

        std::vector<float> y2_data(static_cast<size_t>(D_FF) * SEQ);
        Matrix Y2(y2_data.data(), D_FF, SEQ);
        Cost real_q4_cost;
        linear_q4_real(Wq, X, Y2, real_q4_cost);
        Cost formula_q4_cost = linear_q4_cost(D_FF, DIM, SEQ);

        std::cout << "  Real Q4 kernel: FLOPs=" << real_q4_cost.flops << " Bytes=" << real_q4_cost.bytes << "\n";
        std::cout << "  Q4  formula:    FLOPs=" << formula_q4_cost.flops << " Bytes=" << formula_q4_cost.bytes << "\n";
        CHECK(real_q4_cost.flops == formula_q4_cost.flops);
        CHECK(real_q4_cost.bytes == formula_q4_cost.bytes);
    }

    // =====================================================================
    // TEST 1: Decode (SEQ=1) -- FP32 vs Q4, measured from real struct
    // sizes via the now-verified formula, not assumed.
    // =====================================================================
    constexpr long long N_OUT = 14336, N_IN = 4096;  // one FFN gate/up projection, D_FF=4*DIM convention
    std::cout << "\n-- Test 1: Decode (SEQ=1), one FFN projection (" << N_OUT << " x " << N_IN << ") --\n";
    {
        Cost fp32 = linear_fp32_cost(N_OUT, N_IN, 1);
        Cost q4 = linear_q4_cost(N_OUT, N_IN, 1);

        std::cout << "  FP32: " << fp32.bytes / (1024 * 1024) << " MB read, AI = "
                  << std::fixed << std::setprecision(4) << fp32.arithmetic_intensity() << " FLOPs/byte\n";
        std::cout << "  Q4:   " << q4.bytes / (1024 * 1024) << " MB read, AI = "
                  << q4.arithmetic_intensity() << " FLOPs/byte\n";
        double measured_ratio = static_cast<double>(fp32.bytes) / static_cast<double>(q4.bytes);
        std::cout << "  Measured byte-size ratio FP32:Q4 = " << std::setprecision(2) << measured_ratio << "x\n";
        std::cout << "  (Not the ~4x a naive 'Q4 means 4 bits vs. 32 bits' guess would predict -- a\n";
        std::cout << "  32-value BlockQ4 packs 4-bit nibbles PLUS a 2-byte fp16 scale into 18 bytes,\n";
        std::cout << "  which compresses harder than a flat 4-bit-per-value estimate, since the scale\n";
        std::cout << "  overhead is small relative to the 32 values it covers.)\n";

        CHECK(fp32.arithmetic_intensity() < RIDGE_POINT);
        CHECK(q4.arithmetic_intensity() < RIDGE_POINT);   // still memory-bound at SEQ=1
        CHECK(q4.arithmetic_intensity() > fp32.arithmetic_intensity());  // but less so
        // Real ratio from the struct's actual byte layout: 128 bytes/block (32 fp32) vs.
        // 18 bytes/block (BlockQ4) = 7.11x -- measured here, not assumed to be 4x.
        CHECK(measured_ratio > 7.0 && measured_ratio < 7.2);
    }

    // =====================================================================
    // TEST 2: Find the actual crossover SEQ where each format becomes
    // compute-bound, by searching rather than assuming a formula.
    // =====================================================================
    std::cout << "\n-- Test 2: Crossover sequence length (memory-bound -> compute-bound) --\n";
    long long crossover_fp32 = -1, crossover_q4 = -1;
    {
        for (long long seq = 1; seq <= 100000 && (crossover_fp32 < 0 || crossover_q4 < 0); ++seq) {
            if (crossover_fp32 < 0 && linear_fp32_cost(N_OUT, N_IN, seq).arithmetic_intensity() >= RIDGE_POINT)
                crossover_fp32 = seq;
            if (crossover_q4 < 0 && linear_q4_cost(N_OUT, N_IN, seq).arithmetic_intensity() >= RIDGE_POINT)
                crossover_q4 = seq;
        }
        std::cout << "  FP32 becomes compute-bound at seq_len = " << crossover_fp32 << "\n";
        std::cout << "  Q4   becomes compute-bound at seq_len = " << crossover_q4 << "\n";
        std::cout << "  Q4's smaller weight footprint means fewer batched tokens are needed\n";
        std::cout << "  before the same GEMM kernel crosses into the compute-bound regime.\n";
        CHECK(crossover_fp32 > 0 && crossover_q4 > 0);
        CHECK(crossover_q4 < crossover_fp32);  // Q4 crosses over sooner
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): assuming quantization's throughput benefit
    // holds at ANY sequence length. Once both formats are comfortably
    // past the ridge point, achievable throughput is capped at the SAME
    // peak compute for both -- the byte-count advantage that helped at
    // low AI stops mattering once bytes are no longer the bottleneck.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: quantization's speedup vanishes deep in the compute-bound regime --\n";
    {
        constexpr double PEAK_COMPUTE_GFLOPS = 896.0;
        constexpr double PEAK_BANDWIDTH_GBPS = 51.2;
        auto achievable = [&](double ai) { return std::min(PEAK_COMPUTE_GFLOPS, ai * PEAK_BANDWIDTH_GBPS); };

        long long large_seq = std::max(crossover_fp32, crossover_q4) * 4;  // well past both crossovers
        Cost fp32_large = linear_fp32_cost(N_OUT, N_IN, large_seq);
        Cost q4_large = linear_q4_cost(N_OUT, N_IN, large_seq);

        double ach_fp32 = achievable(fp32_large.arithmetic_intensity());
        double ach_q4 = achievable(q4_large.arithmetic_intensity());

        std::cout << "  At seq_len = " << large_seq << " (well past both crossovers):\n";
        std::cout << "    FP32 AI = " << std::setprecision(2) << fp32_large.arithmetic_intensity()
                  << "  -> achievable = " << ach_fp32 << " GFLOP/s\n";
        std::cout << "    Q4   AI = " << q4_large.arithmetic_intensity()
                  << "  -> achievable = " << ach_q4 << " GFLOP/s\n";
        std::cout << "  Both are capped at peak compute -- quantization's byte-count advantage\n";
        std::cout << "  bought nothing here, because bytes stopped being the bottleneck long ago.\n";
        std::cout << "  The commonly cited '4x speedup from quantization' is a DECODE-regime claim;\n";
        std::cout << "  it does not carry over unchanged into a large, already compute-bound prefill.\n";

        CHECK_NEAR(ach_fp32, PEAK_COMPUTE_GFLOPS, 1e-6);
        CHECK_NEAR(ach_q4, PEAK_COMPUTE_GFLOPS, 1e-6);
        CHECK_NEAR(ach_fp32, ach_q4, 1e-6);  // identical -- the format stopped mattering
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
