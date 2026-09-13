# Chapter 8: The Memory Wall and the Roofline Model -- Why Arithmetic Is Rarely the Bottleneck

**What you will understand by the end of this chapter:**

- Why "how many FLOPs does this kernel do" is the wrong first question to ask about performance, and "how many bytes does it move, relative to those FLOPs" is the right one — derived here as arithmetic intensity, not asserted as folklore.
- How to derive a machine's peak compute and peak memory bandwidth from stated architectural parameters (core count, clock speed, SIMD width; memory channels, transfer rate) rather than quoting an unverified marketing number, and how those two peaks combine into a ridge point that separates memory-bound kernels from compute-bound ones.
- Why three real transformer kernels — RMSNorm, softmax, and RoPE — are all memory-bound by a wide margin, measured by instrumenting the actual kernel code to count every FLOP and every byte as it runs, not by looking the answer up.
- Why prefill (a GEMM, many tokens at once) and decode (a GEMV, one token at a time) are the same weight matrix and the same arithmetic, sitting on opposite sides of the same ridge point — and why quantization's speedup at one sequence length can vanish entirely at another.
- Why continuous batching's throughput gain is the exact same GEMV-to-GEMM transition, viewed from the scheduler's side instead of the sequence's — and why that gain comes with a latency cost past the same crossover point.
- Why, in a full decode step, the part of the cost that visibly grows with context length (attention against the KV cache) is not the part that actually dominates the bytes moved at realistic context lengths — the weight matrices, read fresh on every single step regardless of context, are.

**What you need to know first:**

- Chapter 3's std::mdspan vocabulary, for viewing a genuinely two- or three-dimensional structure — a weight matrix, a KV cache — as more than a flat buffer with hand-computed offsets; Sections 8.3 and 8.5 use it directly.
- Chapter 3.4's GQA attention and KV cache, and Chapters 2/4's Q4_0 block format, both reused verbatim in this chapter's later sections rather than redefined.
- This book's standing policy against fabricated timing numbers applies with particular force here: this chapter never runs a wall-clock benchmark, because a shared, virtualized build environment gives no number that would be reproducible on a rerun or on a reader's own machine. Every quantity in this chapter is either a stated, labeled architectural parameter or an exact, deterministic count of operations and bytes — FLOPs and bytes are integers a real kernel actually produces, not estimates a clock has to guess at.

---

Every kernel in this book so far has been judged by what it computes and how accurately. This chapter asks a different question: given a kernel that is already correct, what determines how FAST it can possibly run on real hardware — not "how fast did it run just now," which a shared sandbox cannot answer reproducibly, but "what is the hard ceiling imposed by the machine's own arithmetic and memory limits." Section 8.1 builds that ceiling, the roofline model, from first principles: peak compute and peak memory bandwidth, derived from stated architectural parameters, combining into a ridge point that separates kernels whose speed is capped by arithmetic from kernels whose speed is capped by how fast bytes can be fetched from memory. Section 8.2 measures where three real transformer kernels actually land relative to that ridge, by instrumenting them to count their own FLOPs and bytes as they run. Section 8.3 applies the same framework to the single most consequential distinction in an inference engine's performance profile — prefill versus decode — and finds the exact sequence length at which one becomes the other. Section 8.4 asks the same question from a serving system's point of view: continuous batching turns many decode requests' single-token GEMVs into one batched GEMM, the identical transition Section 8.3 analyzed, and this section asks what that means for latency, not just throughput. Section 8.5 closes the chapter with a full transformer layer's roofline profile, combining every kernel this book has built — RMSNorm, QKVO projections, GQA attention against a real KV cache, and the FFN — into one decode step, and asks which part of that step's memory traffic actually dominates.

## 8.1 The Roofline Model, Derived From First Principles

### Intuition

A kernel's speed is capped by two entirely different resources: how many arithmetic operations the processor can perform per second, and how many bytes it can move between memory and the processor per second. Whichever of the two a given kernel demands more of, relative to what it does with each byte it touches, is the one that actually limits it — and that ratio, FLOPs per byte, is a property of the KERNEL, while the two peaks are properties of the MACHINE. Neither number is worth much on its own; the useful question is always how they compare for one specific kernel on one specific machine.

### The Concept, In Detail

Peak compute is derived, not quoted: a core that supports AVX2 and fused multiply-add can retire `lanes_per_register * fma_ports_per_cycle * 2` FLOPs per cycle (the `2` because one FMA does a multiply and an add), and multiplying by core count and sustained clock speed gives the machine's peak FP32 GFLOP/s. Peak bandwidth is derived the same way: `channels * transfer_rate * bytes_per_transfer` gives peak GB/s for a stated memory configuration. Dividing the two gives the ridge point, in FLOPs per byte — the arithmetic intensity at which a kernel's demand for compute and its demand for bandwidth are exactly balanced. A kernel below the ridge point is memory-bound: however fast its arithmetic could run in isolation, it spends more time waiting on bytes than the processor spends computing on them, so its achievable throughput is `arithmetic_intensity * peak_bandwidth` — linear in arithmetic intensity, capped by bandwidth. A kernel above the ridge point is compute-bound: bytes arrive faster than the processor can consume them, so achievable throughput flattens at `peak_compute` regardless of how much higher arithmetic intensity climbs. This produces the chapter's first COMMON TRAP directly: a faster processor — more peak compute — does nothing whatsoever for a kernel that is memory-bound, because compute was never the limiting resource to begin with.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_roofline_and_ridge_point.cpp -o 01_roofline_and_ridge_point
./01_roofline_and_ridge_point
```

**Sample input:** an 8-core, AVX2, dual-channel DDR4-3200 machine's peak compute and peak bandwidth, derived from those stated parameters and cross-checked against the same arithmetic done independently; the achievable-throughput formula evaluated on both sides of the resulting ridge point, checking linear growth below it and a hard ceiling above it; and a deliberate demonstration of quadrupling peak compute for a memory-bound kernel.

```text
========================================================
Chapter 8.1: The Roofline Model, Derived From First Principles
========================================================

-- Test 1: Deriving peak compute and peak bandwidth --
  FLOPs/cycle/core = 8 lanes * 2 FMA ports * 2 FLOPs/FMA = 32
  Peak compute = 8 cores * 3.5 GHz * 32 FLOPs/cycle = 896.0 GFLOP/s
  Peak bandwidth = 2 channels * 3200.0 MT/s * 8 bytes = 51.2 GB/s
  Ridge point = 17.50 FLOPs/byte

-- Test 2: Achievable throughput on both sides of the ridge --
  Ridge point: 17.50 FLOPs/byte
  AI = 8.75 (below ridge) -> achievable = 448.00 GFLOP/s (memory-bound: yes)
  AI = 70.00 (above ridge) -> achievable = 896.00 GFLOP/s (memory-bound: no)

-- Test 3 [COMMON TRAP]: a faster CPU does nothing for a memory-bound kernel --
  Kernel AI = 0.50 FLOPs/byte (a decode-time GEMV)
  Slow CPU (896 GFLOP/s peak): achievable = 25.60 GFLOP/s
  Fast CPU (3584 GFLOP/s peak, 4x more compute): achievable = 25.60 GFLOP/s
  Quadrupling peak compute changed the achievable throughput by 0.0000 GFLOP/s --
  effectively nothing, because bandwidth was the bottleneck all along.

========================================================
11/11 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming a faster CPU always helps"
    Peak compute and peak bandwidth are independent resources, and a kernel is only ever limited by whichever one it demands more of. For a memory-bound kernel — one whose arithmetic intensity sits below the ridge point — the achievable-throughput formula reduces to `arithmetic_intensity * peak_bandwidth`, which does not contain the peak-compute term at all. Quadrupling peak compute for such a kernel changes its achievable throughput by exactly zero, not approximately zero: bandwidth was the entire bottleneck, and no amount of additional arithmetic capacity relieves a constraint the kernel was never bound by. "Upgrade to a machine with more FLOPs" is sound advice for a compute-bound kernel and completely useless advice for a memory-bound one, and the roofline model is what tells the two situations apart before any hardware purchase is made.

## 8.2 Measuring Arithmetic Intensity in Real Kernels

### Intuition

Section 8.1's ridge point is only useful once a kernel's own arithmetic intensity is known — and that number should come from instrumenting the kernel's actual control flow, not from a hand-derived formula that could quietly drift out of sync with what the code really does. RMSNorm, softmax, and RoPE are three of the cheapest, most frequently executed kernels in a transformer's forward pass, and all three turn out to sit far on the memory-bound side of any realistic ridge point — for the same underlying reason in each case.

### The Concept, In Detail

A `Cost` counter threaded through each kernel increments its FLOP and byte totals as the kernel's real loops execute, so the resulting arithmetic-intensity number is a direct consequence of the code, not a separately asserted claim about it. RMSNorm reads a vector and a weight vector, does a handful of FLOPs per element (a square, a running sum, a scale, a multiply), and writes the result — a small, fixed amount of arithmetic per byte moved, regardless of dimension, so it lands memory-bound at any realistic size. Softmax makes three full passes over its input (find the max, exponentiate and sum, divide by the sum), each pass moving as many bytes as the vector has elements, for an arithmetic cost per element that a transcendental function's real cost (counted here at a stated, labeled convention of eight FLOPs per `exp`, not treated as free) still does not come close to matching. RoPE rotates pairs of dimensions using precomputed cosine and sine tables — cheap per-pair arithmetic (six FLOPs) against the pair's own four bytes, plus the often-forgotten cost of reading the tables themselves. All three land far below any realistic ridge point, which is exactly why they are typically fused into neighboring kernels in a real engine rather than optimized in isolation — optimizing a memory-bound kernel's arithmetic buys almost nothing, per Section 8.1's own lesson.

### Code and Verification

```cpp
// 02_kernel_arithmetic_intensity.cpp
// Chapter 8, Part 2: measure the arithmetic intensity of three real
// transformer kernels -- RMSNorm, softmax, and RoPE -- not by looking up
// a table, but by instrumenting the actual kernel code to count every
// floating-point operation and every byte read or written as it runs.
// The resulting FLOPs/byte ratio is then classified against Section
// 8.1's derived roofline.
//
// Counting operations (integers, exact) rather than timing wall-clock
// nanoseconds is what keeps this file's output reproducible bit-for-bit
// across compilers and machines -- there is no floating-point summation
// order or scheduling noise involved in an operation COUNT.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_kernel_arithmetic_intensity.cpp -o out02

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
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
// COST COUNTER: every kernel below increments this as it actually runs,
// so the FLOP and byte counts are a direct consequence of the real
// control flow, not a separately-asserted formula that could drift out
// of sync with the implementation.
// =========================================================================
struct Cost {
    long long flops = 0;
    long long bytes = 0;
    void flop(long long n = 1) { flops += n; }
    void read(long long n_elems, int elem_bytes = 4) { bytes += n_elems * elem_bytes; }
    void write(long long n_elems, int elem_bytes = 4) { bytes += n_elems * elem_bytes; }
    double arithmetic_intensity() const { return static_cast<double>(flops) / static_cast<double>(bytes); }
};

// A transcendental function (exp, in softmax) is not one hardware FLOP,
// but nor is it free. This constant is a stated modeling convention --
// a commonly cited rough cost for a vectorized exp -- used consistently
// wherever exp is counted below.
constexpr long long EXP_FLOP_COST = 8;

// =========================================================================
// KERNEL 1: RMSNorm -- out[i] = x[i] / rms(x) * w[i]
// =========================================================================
void rms_norm(std::span<const float> x, std::span<const float> w, std::span<float> out, Cost& cost) {
    int D = static_cast<int>(x.size());
    cost.read(D);  // read x
    cost.read(D);  // read w

    float sum_sq = 0.0f;
    for (int i = 0; i < D; ++i) { sum_sq += x[i] * x[i]; cost.flop(2); }  // 1 mul + 1 add
    float mean_sq = sum_sq / static_cast<float>(D);
    cost.flop(1);
    float rms = 1.0f / std::sqrt(mean_sq + 1e-6f);
    cost.flop(3);  // add-eps, sqrt, reciprocal (counted as 3 FLOPs by convention)

    for (int i = 0; i < D; ++i) {
        out[i] = x[i] * rms * w[i];
        cost.flop(2);  // 2 multiplies
    }
    cost.write(D);  // write out
}

// =========================================================================
// KERNEL 2: softmax, in place
// =========================================================================
void softmax_inplace(std::span<float> x, Cost& cost) {
    int D = static_cast<int>(x.size());
    cost.read(D);  // pass 1: find max

    float mx = x[0];
    for (int i = 1; i < D; ++i) { if (x[i] > mx) mx = x[i]; cost.flop(1); }  // compare

    cost.read(D);   // pass 2: subtract max and exp
    float sum = 0.0f;
    for (int i = 0; i < D; ++i) {
        x[i] = std::exp(x[i] - mx);
        cost.flop(1 + EXP_FLOP_COST);  // subtract + exp
        sum += x[i];
        cost.flop(1);  // running sum add
    }
    cost.write(D);  // the exp'd values just written back into x

    cost.read(D);   // pass 3: divide by sum
    for (int i = 0; i < D; ++i) { x[i] /= sum; cost.flop(1); }
    cost.write(D);
}

// =========================================================================
// KERNEL 3: RoPE (rotary position embedding), applied to one query vector
// =========================================================================
// Rotates each consecutive pair of dimensions by a position-dependent
// angle, using precomputed cos/sin tables (one entry per pair).
void apply_rope(std::span<float> q, std::span<const float> cos_table, std::span<const float> sin_table, Cost& cost) {
    int D = static_cast<int>(q.size());
    int pairs = D / 2;
    cost.read(D);       // read q
    cost.read(pairs);   // read cos table
    cost.read(pairs);   // read sin table

    for (int p = 0; p < pairs; ++p) {
        float x0 = q[2 * p], x1 = q[2 * p + 1];
        float c = cos_table[p], s = sin_table[p];
        q[2 * p]     = x0 * c - x1 * s;
        q[2 * p + 1] = x0 * s + x1 * c;
        cost.flop(6);  // 4 multiplies + 2 add/sub
    }
    cost.write(D);  // write q back, in place
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 8.2: Arithmetic Intensity of Real Transformer Kernels\n";
    std::cout << "========================================================\n\n";

    // Section 8.1's derived roofline, reused for classification here.
    constexpr double PEAK_COMPUTE_GFLOPS = 896.0;
    constexpr double PEAK_BANDWIDTH_GBPS = 51.2;
    constexpr double RIDGE_POINT = PEAK_COMPUTE_GFLOPS / PEAK_BANDWIDTH_GBPS;  // 17.5 FLOPs/byte

    constexpr int DIM = 4096;

    // =====================================================================
    // TEST 1: RMSNorm's measured arithmetic intensity
    // =====================================================================
    std::cout << "-- Test 1: RMSNorm (dim=" << DIM << ") --\n";
    {
        std::vector<float> x(DIM), w(DIM, 1.0f), out(DIM);
        for (int i = 0; i < DIM; ++i) x[i] = static_cast<float>((i % 7) - 3) * 0.1f;
        Cost cost;
        rms_norm(x, w, out, cost);

        double ai = cost.arithmetic_intensity();
        std::cout << "  FLOPs: " << cost.flops << "   Bytes: " << cost.bytes
                  << "   AI: " << std::fixed << std::setprecision(4) << ai << " FLOPs/byte\n";
        std::cout << "  Ridge point: " << std::setprecision(2) << RIDGE_POINT << " -> "
                  << (ai < RIDGE_POINT ? "MEMORY-BOUND" : "COMPUTE-BOUND") << "\n";
        CHECK(cost.flops == 4LL * DIM + 4);  // 2*D (sum-of-squares) + 4 + 2*D (scale)
        CHECK(cost.bytes == 3LL * DIM * 4);  // read x, read w, write out (D floats each)
        CHECK(ai < RIDGE_POINT);  // RMSNorm should be sharply memory-bound
    }

    // =====================================================================
    // TEST 2: Softmax's measured arithmetic intensity
    // =====================================================================
    std::cout << "\n-- Test 2: Softmax (seq_len=2048) --\n";
    {
        constexpr int SEQ = 2048;
        std::vector<float> scores(SEQ);
        for (int i = 0; i < SEQ; ++i) scores[i] = static_cast<float>((i % 11) - 5) * 0.05f;
        Cost cost;
        softmax_inplace(scores, cost);

        double ai = cost.arithmetic_intensity();
        std::cout << "  FLOPs: " << cost.flops << "   Bytes: " << cost.bytes
                  << "   AI: " << std::setprecision(4) << ai << " FLOPs/byte\n";
        std::cout << "  Ridge point: " << std::setprecision(2) << RIDGE_POINT << " -> "
                  << (ai < RIDGE_POINT ? "MEMORY-BOUND" : "COMPUTE-BOUND") << "\n";
        // sanity: output is a valid probability distribution
        float total = 0.0f;
        for (float v : scores) total += v;
        CHECK_NEAR(total, 1.0f, 1e-4);
        CHECK(ai < RIDGE_POINT);  // softmax's three memory passes dominate its cheap arithmetic
    }

    // =====================================================================
    // TEST 3: RoPE's measured arithmetic intensity
    // =====================================================================
    std::cout << "\n-- Test 3: RoPE (head_dim=128) --\n";
    {
        constexpr int HEAD_DIM = 128;
        std::vector<float> q(HEAD_DIM), cos_t(HEAD_DIM / 2), sin_t(HEAD_DIM / 2);
        for (int i = 0; i < HEAD_DIM; ++i) q[i] = static_cast<float>((i % 5) - 2) * 0.2f;
        for (int p = 0; p < HEAD_DIM / 2; ++p) {
            float angle = static_cast<float>(p) * 0.01f;
            cos_t[p] = std::cos(angle);
            sin_t[p] = std::sin(angle);
        }
        float norm_before = 0.0f;
        for (float v : q) norm_before += v * v;

        Cost cost;
        apply_rope(q, cos_t, sin_t, cost);

        float norm_after = 0.0f;
        for (float v : q) norm_after += v * v;

        double ai = cost.arithmetic_intensity();
        std::cout << "  FLOPs: " << cost.flops << "   Bytes: " << cost.bytes
                  << "   AI: " << std::setprecision(4) << ai << " FLOPs/byte\n";
        std::cout << "  Ridge point: " << std::setprecision(2) << RIDGE_POINT << " -> "
                  << (ai < RIDGE_POINT ? "MEMORY-BOUND" : "COMPUTE-BOUND") << "\n";
        // RoPE is a rotation: it must preserve each pair's norm exactly.
        CHECK_NEAR(norm_before, norm_after, 1e-3);
        CHECK(ai < RIDGE_POINT);
    }

    // =====================================================================
    // TEST 4 (COMMON TRAP): forgetting to count a real memory access
    // inflates the measured arithmetic intensity, and can flip a kernel's
    // classification across the ridge point.
    // =====================================================================
    std::cout << "\n-- Test 4 [COMMON TRAP]: incomplete byte counting inflates AI --\n";
    {
        constexpr int HEAD_DIM = 128;
        std::vector<float> q(HEAD_DIM), cos_t(HEAD_DIM / 2), sin_t(HEAD_DIM / 2);
        for (int i = 0; i < HEAD_DIM; ++i) q[i] = static_cast<float>((i % 5) - 2) * 0.2f;
        for (int p = 0; p < HEAD_DIM / 2; ++p) {
            float angle = static_cast<float>(p) * 0.01f;
            cos_t[p] = std::cos(angle);
            sin_t[p] = std::sin(angle);
        }

        Cost correct;
        apply_rope(q, cos_t, sin_t, correct);  // counts q read+write AND both tables

        // Buggy accounting: someone counts only the vector being rotated
        // (read + write) and forgets that the cos/sin tables are ALSO
        // real memory traffic -- an easy mistake, since the tables are
        // "just a lookup," not the vector being transformed.
        long long buggy_bytes = 4LL * HEAD_DIM /* read q */ + 4LL * HEAD_DIM /* write q */;
        double buggy_ai = static_cast<double>(correct.flops) / static_cast<double>(buggy_bytes);
        double correct_ai = correct.arithmetic_intensity();

        std::cout << "  Correct byte count (q + cos table + sin table, read and write): "
                  << correct.bytes << " bytes -> AI = " << std::fixed << std::setprecision(4) << correct_ai << "\n";
        std::cout << "  Buggy byte count (forgets the cos/sin table reads):            "
                  << buggy_bytes << " bytes -> AI = " << buggy_ai << "\n";
        std::cout << "  Both land on the memory-bound side of this machine's ridge point ("
                  << std::setprecision(2) << RIDGE_POINT << "), so the classification does not\n";
        std::cout << "  flip here -- but the buggy count overstates AI by "
                  << std::setprecision(1) << (buggy_ai / correct_ai - 1.0) * 100.0
                  << "%, and for a kernel whose true AI sits close to the ridge, an\n";
        std::cout << "  overstatement of this size is exactly enough to misclassify it.\n";
        CHECK(buggy_ai > correct_ai);
        CHECK(buggy_bytes < correct.bytes);
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
g++ -std=c++23 -Wall -Wextra -O2 02_kernel_arithmetic_intensity.cpp -o 02_kernel_arithmetic_intensity
./02_kernel_arithmetic_intensity
```

**Sample input:** RMSNorm at dimension 4096, softmax at sequence length 2048, and RoPE at head dimension 128, each instrumented and classified against Section 8.1's derived ridge point; and a deliberate demonstration of forgetting to count RoPE's cosine/sine table reads as real memory traffic.

```text
========================================================
Chapter 8.2: Arithmetic Intensity of Real Transformer Kernels
========================================================

-- Test 1: RMSNorm (dim=4096) --
  FLOPs: 16388   Bytes: 49152   AI: 0.3334 FLOPs/byte
  Ridge point: 17.50 -> MEMORY-BOUND

-- Test 2: Softmax (seq_len=2048) --
  FLOPs: 24575   Bytes: 40960   AI: 0.6000 FLOPs/byte
  Ridge point: 17.50 -> MEMORY-BOUND

-- Test 3: RoPE (head_dim=128) --
  FLOPs: 384   Bytes: 1536   AI: 0.2500 FLOPs/byte
  Ridge point: 17.50 -> MEMORY-BOUND

-- Test 4 [COMMON TRAP]: incomplete byte counting inflates AI --
  Correct byte count (q + cos table + sin table, read and write): 1536 bytes -> AI = 0.2500
  Buggy byte count (forgets the cos/sin table reads):            1024 bytes -> AI = 0.3750
  Both land on the memory-bound side of this machine's ridge point (17.50), so the classification does not
  flip here -- but the buggy count overstates AI by 50.0%, and for a kernel whose true AI sits close to the ridge, an
  overstatement of this size is exactly enough to misclassify it.

========================================================
9/9 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] forgetting a real memory access when counting bytes"
    The cosine and sine tables RoPE reads are easy to leave out of a byte count — they are "just a lookup," not the vector actually being transformed — but a lookup table read from memory is exactly as real a byte of traffic as the vector itself, especially when, as here, it is not reused across many calls but re-read for each one. Omitting it does not produce an obviously broken number; it produces an arithmetic-intensity figure that is quietly too high, because the same FLOP count is now being divided by too few bytes. At head dimension 128 this inflates the measured intensity by fifty percent — not enough to flip RoPE's classification here, since it is memory-bound by a wide margin either way, but for any kernel whose true arithmetic intensity sits close to the ridge point, an overstatement of this size is exactly enough to misclassify it as compute-bound when it is not.

## 8.3 Prefill vs. Decode: The Same Weights, Different Arithmetic

### Intuition

Prefill and decode use the identical weight matrix for the identical linear projection, and yet one is routinely described as compute-bound and the other as memory-bound. The difference is not the weights or the arithmetic — it is how many columns of input share that one weight read before it is done with. A GEMM (many token columns at once) and a GEMV (one column) are the same operation at two different points on the very same ridge point Section 8.1 derived, and the sequence length at which one becomes the other is a number this section finds by search, not by assumption.

### The Concept, In Detail

A linear layer's weight matrix, read once, can be reused across every column of a batch before the next matrix's weights are needed — so a batch of `seq` columns pays the weight-read cost exactly once while paying the FLOP cost `seq` times, and arithmetic intensity climbs linearly with `seq` until it crosses the ridge point. At `seq=1` (decode), that reuse buys nothing: the entire weight matrix is read to produce a single output column, which is why decode is memory-bound almost everywhere in practice. Quantizing the weight matrix to Q4_0 raises arithmetic intensity at any given `seq` (fewer bytes for the same FLOPs), which shifts the crossover to a SMALLER sequence length — but once a sequence is already long enough to be deep in the compute-bound regime for both formats, both are capped at the same peak-compute ceiling, and quantization's byte-count advantage stops mattering entirely. The weight matrix and the batch of activations are genuinely two-dimensional objects, viewed here through std::mdspan rather than through hand-computed row/column offsets, with a small real GEMM kernel run at reduced, illustrative dimensions specifically so its FLOP and byte counts can be checked against the closed-form formula this section then applies, unmaterialized, at the FFN projection's real (14336 x 4096) size.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_prefill_vs_decode.cpp -o 03_prefill_vs_decode
./03_prefill_vs_decode
```

**Sample input:** a real, std::mdspan-viewed FP32 and Q4_0 GEMM kernel at small illustrative dimensions, cross-checked against the closed-form cost formula it verifies; decode-time (seq=1) FP32-vs-Q4 byte and arithmetic-intensity comparison at the FFN projection's real 14336x4096 size; a search for the exact sequence length at which each format crosses from memory-bound to compute-bound; and a deliberate demonstration of quantization's throughput advantage at a sequence length well past both crossovers.

```text
========================================================
Chapter 8.3: Prefill (GEMM) vs. Decode (GEMV)
========================================================

-- Test 0: real mdspan-based kernel vs. closed-form formula, same shape --
  Real kernel:    FLOPs=163840 Bytes=71936
  Closed formula: FLOPs=163840 Bytes=71936
  Real Q4 kernel: FLOPs=180224 Bytes=15616
  Q4  formula:    FLOPs=180224 Bytes=15616

-- Test 1: Decode (SEQ=1), one FFN projection (14336 x 4096) --
  FP32: 224 MB read, AI = 0.4998 FLOPs/byte
  Q4:   31 MB read, AI = 5.3215 FLOPs/byte
  Measured byte-size ratio FP32:Q4 = 7.10x
  (Not the ~4x a naive 'Q4 means 4 bits vs. 32 bits' guess would predict -- a
  32-value BlockQ4 packs 4-bit nibbles PLUS a 2-byte fp16 scale into 18 bytes,
  which compresses harder than a flat 4-bit-per-value estimate, since the scale
  overhead is small relative to the 32 values it covers.)

-- Test 2: Crossover sequence length (memory-bound -> compute-bound) --
  FP32 becomes compute-bound at seq_len = 36
  Q4   becomes compute-bound at seq_len = 5
  Q4's smaller weight footprint means fewer batched tokens are needed
  before the same GEMM kernel crosses into the compute-bound regime.

-- Test 3 [COMMON TRAP]: quantization's speedup vanishes deep in the compute-bound regime --
  At seq_len = 144 (well past both crossovers):
    FP32 AI = 68.89  -> achievable = 896.00 GFLOP/s
    Q4   AI = 388.80  -> achievable = 896.00 GFLOP/s
  Both are capped at peak compute -- quantization's byte-count advantage
  bought nothing here, because bytes stopped being the bottleneck long ago.
  The commonly cited '4x speedup from quantization' is a DECODE-regime claim;
  it does not carry over unchanged into a large, already compute-bound prefill.

========================================================
14/14 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming quantization's speedup holds at any sequence length"
    "Quantization gives roughly a 4x speedup" is a claim about the DECODE regime, where reading four times fewer weight bytes for the same arithmetic directly translates into a proportionally faster memory-bound kernel. Once the sequence length grows past both formats' compute-bound crossover, that logic no longer applies: both FP32 and Q4 are capped at the same peak-compute ceiling, and the achievable-throughput formula does not depend on arithmetic intensity at all once a kernel is compute-bound. Quantizing the weights of an already deeply compute-bound GEMM changes memory traffic that was never the bottleneck to begin with — exactly Section 8.1's trap, in a new form. The speedup is real and large at decode time; it is measured here to be exactly zero at a sufficiently long, already compute-bound sequence length, and the honest claim is "quantization speeds up decode," not "quantization speeds up inference" unconditionally.

## 8.4 Continuous Batching: the Same Crossover, a Different Question

### Intuition

A production inference server rarely runs one request's GEMV at a time. Continuous batching stacks many DIFFERENT requests' single decode tokens into one batched matrix multiply, sharing the same weight read across all of them — which is mathematically the identical GEMV-to-GEMM transition Section 8.3 analyzed for one request's own prompt tokens, just triggered by the scheduler instead of by prompt length. Reusing that section's own formula unchanged, with `seq` reinterpreted as batch size, this section asks the question prefill's analysis did not need to: what happens to each individual REQUEST's latency, not just the batch's aggregate throughput, on both sides of the same crossover.

### The Concept, In Detail

Below the batch-size crossover, a batch is memory-bound, and its total processing time is dominated by the one shared weight read — so doubling, or even multiplying by eight, the number of requests sharing that batch barely changes how long the batch takes to process, because the dominant cost did not change. Every request in a synchronously-served batch waits for the WHOLE batch to finish before its own next token is ready, so that near-flat batch time is also, directly, each request's own per-token latency — batching in this regime is close to free. Past the crossover, the batch is compute-bound, and batch time grows roughly linearly with batch size, since the shared-weight-read discount no longer has anywhere left to apply: FLOPs, not bytes, are now the limit, and FLOPs scale with every additional request. Aggregate throughput (tokens served per unit time) keeps growing on both sides of the crossover, but far more slowly past it than before — while each request's own latency, tied directly to the now-linearly-growing batch time, gets steadily worse the more requests are piled into an already compute-bound batch. This is the same shape as Section 8.1's and 8.3's traps, applied to a scheduling decision rather than a hardware or format choice.

### Code and Verification

```cpp
// 04_continuous_batching.cpp
// Chapter 8, Part 4: continuous batching turns many independent decode
// requests' single-token GEMVs into one batched GEMM -- mathematically
// the EXACT SAME memory-bound-to-compute-bound transition Section 8.3
// analyzed for a single request's prefill, just triggered by stacking
// unrelated requests' one token each into the batch dimension instead of
// stacking one request's many prompt tokens. This file reuses Section
// 8.3's own cost formula unchanged (with "seq" reinterpreted as "batch
// size B") to find the batch-size crossover, then asks the question
// prefill's analysis didn't need to: what happens to PER-REQUEST latency,
// not just aggregate throughput, on both sides of that crossover.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_continuous_batching.cpp -o out04

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
// Section 8.3's cost formula, unchanged. There "seq" was the number of
// prompt-token columns sharing one weight read; here it is the number of
// DIFFERENT REQUESTS' single decode tokens sharing that same weight read
// -- the scheduler that stacks them is what "continuous batching" means.
// =========================================================================
struct Cost {
    long long flops = 0;
    long long bytes = 0;
    void flop(long long n = 1) { flops += n; }
    void access(long long n_bytes) { bytes += n_bytes; }
    double arithmetic_intensity() const { return static_cast<double>(flops) / static_cast<double>(bytes); }
};
Cost linear_fp32_cost(long long n_out, long long n_in, long long batch) {
    Cost cost;
    cost.access(n_out * n_in * 4);     // W, read once, shared across the whole batch
    cost.access(n_in * batch * 4);     // X, one column per request in the batch
    cost.access(n_out * batch * 4);    // Y, one column per request in the batch
    cost.flop(2 * n_out * n_in * batch);
    return cost;
}
struct Roofline {
    double peak_compute_gflops, peak_bandwidth_gbps;
    double achievable_gflops(double ai) const { return std::min(peak_compute_gflops, ai * peak_bandwidth_gbps); }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 8.4: Continuous Batching -- the Same Crossover, a Different Question\n";
    std::cout << "========================================================\n\n";

    constexpr double PEAK_COMPUTE_GFLOPS = 896.0, PEAK_BANDWIDTH_GBPS = 51.2;  // Section 8.1's derived roofline
    constexpr double RIDGE_POINT = PEAK_COMPUTE_GFLOPS / PEAK_BANDWIDTH_GBPS;   // 17.5 FLOPs/byte
    constexpr long long N_OUT = 14336, N_IN = 4096;  // the same FFN projection Section 8.3 used
    Roofline roof{PEAK_COMPUTE_GFLOPS, PEAK_BANDWIDTH_GBPS};

    // A batch's predicted processing time, in milliseconds, from the
    // roofline model: total FLOPs divided by the achievable rate at this
    // batch's own arithmetic intensity. Every request in the batch is
    // served SYNCHRONOUSLY -- nobody's next token is ready until the
    // whole batched GEMM finishes -- so this is also each request's
    // per-token latency for that round.
    auto batch_time_ms = [&](long long B) {
        Cost c = linear_fp32_cost(N_OUT, N_IN, B);
        double gflops = static_cast<double>(c.flops) / 1e9;
        double achievable = roof.achievable_gflops(c.arithmetic_intensity());
        return (gflops / achievable) * 1000.0;  // seconds -> ms
    };

    // =====================================================================
    // TEST 1: below the crossover, batching more requests together barely
    // changes the batch's total processing time, because the weight read
    // -- not the batch size -- is what the time is dominated by.
    // =====================================================================
    std::cout << "-- Test 1: batch latency is nearly flat while memory-bound --\n";
    {
        double t1 = batch_time_ms(1);
        double t8 = batch_time_ms(8);
        std::cout << "  Batch time at B=1: " << std::fixed << std::setprecision(4) << t1 << " ms\n";
        std::cout << "  Batch time at B=8: " << t8 << " ms  (" << std::setprecision(2) << (t8 / t1) << "x of B=1)\n";
        std::cout << "  8x more requests served for well under 8x the time -- almost free,\n";
        std::cout << "  because the dominant cost (reading the weight matrix once) didn't change.\n";
        CHECK(t8 / t1 < 2.0);  // far less than the naive "8 requests = 8x time" assumption
        CHECK(t8 > t1);        // but not literally free either -- X/Y bytes and FLOPs did grow
    }

    // =====================================================================
    // TEST 2: find B*, the batch size at which this GEMM crosses from
    // memory-bound to compute-bound -- by search, reusing Section 8.3's
    // own crossover-finding method verbatim.
    // =====================================================================
    std::cout << "\n-- Test 2: batch-size crossover B* --\n";
    long long b_star = -1;
    {
        for (long long b = 1; b <= 100000; ++b) {
            if (linear_fp32_cost(N_OUT, N_IN, b).arithmetic_intensity() >= RIDGE_POINT) { b_star = b; break; }
        }
        std::cout << "  B* = " << b_star << " (this FFN projection becomes compute-bound at batch size " << b_star << ")\n";
        CHECK(b_star > 0);
        CHECK_NEAR(b_star, 36, 2);  // same shape as Section 8.3's crossover_fp32 -- batch size plays seq's role
    }

    // =====================================================================
    // TEST 3 (COMMON TRAP): assuming more batching always helps
    // throughput proportionally. Past B*, the batch is compute-bound, so
    // batch time grows LINEARLY with B (no more free lunch from shared
    // weight reads) -- aggregate throughput plateaus at a fixed
    // tokens/second ceiling, while each request's own per-token latency
    // keeps getting WORSE the more requests are piled into the batch.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: past B*, more batching raises latency without raising throughput --\n";
    {
        long long b_small = b_star / 2;       // still memory-bound
        long long b_big = b_star * 8;         // deep in the compute-bound regime

        double t_small = batch_time_ms(b_small);
        double t_big = batch_time_ms(b_big);
        double throughput_small = static_cast<double>(b_small) / t_small;  // requests served per ms
        double throughput_big = static_cast<double>(b_big) / t_big;

        std::cout << "  B=" << b_small << " (memory-bound):  batch time=" << std::setprecision(4) << t_small
                  << " ms, throughput=" << std::setprecision(2) << throughput_small << " tokens/ms\n";
        std::cout << "  B=" << b_big << " (compute-bound):  batch time=" << t_big
                  << " ms, throughput=" << throughput_big << " tokens/ms\n";
        std::cout << "  Batch time grew " << std::setprecision(2) << (t_big / t_small) << "x while batch size grew "
                  << (static_cast<double>(b_big) / b_small) << "x -- roughly proportionally, once compute-bound.\n";
        std::cout << "  Throughput grew only " << (throughput_big / throughput_small)
                  << "x for a " << (static_cast<double>(b_big) / b_small) << "x bigger batch --\n";
        std::cout << "  every request now waits longer per token, for a throughput gain far short of\n";
        std::cout << "  proportional. Past B*, adding requests to a batch trades latency for very\n";
        std::cout << "  little additional throughput, the mirror image of Test 1's near-free batching.\n";

        // Throughput growth is far short of the batch-size growth ratio (proportional
        // growth would be the naive, wrong assumption this trap corrects).
        double batch_growth = static_cast<double>(b_big) / static_cast<double>(b_small);
        double throughput_growth = throughput_big / throughput_small;
        CHECK(throughput_growth < batch_growth * 0.5);
        // Per-token latency (== batch time, since service is synchronous) got worse, not better.
        CHECK(t_big > t_small * (batch_growth * 0.5));  // grew at least roughly proportionally to batch size
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
g++ -std=c++23 -Wall -Wextra -O2 04_continuous_batching.cpp -o 04_continuous_batching
./04_continuous_batching
```

**Sample input:** batch processing time at batch size 1 versus batch size 8, well below the crossover; a search for the batch-size crossover B* of the same FFN projection Section 8.3 used; and a deliberate comparison of aggregate throughput and per-request latency at a batch size well past B*, against a batch size still below it.

```text
========================================================
Chapter 8.4: Continuous Batching -- the Same Crossover, a Different Question
========================================================

-- Test 1: batch latency is nearly flat while memory-bound --
  Batch time at B=1: 4.5890 ms
  Batch time at B=8: 4.5990 ms  (1.00x of B=1)
  8x more requests served for well under 8x the time -- almost free,
  because the dominant cost (reading the weight matrix once) didn't change.

-- Test 2: batch-size crossover B* --
  B* = 36 (this FFN projection becomes compute-bound at batch size 36)

-- Test 3 [COMMON TRAP]: past B*, more batching raises latency without raising throughput --
  B=18 (memory-bound):  batch time=4.6134 ms, throughput=3.90 tokens/ms
  B=288 (compute-bound):  batch time=37.75 ms, throughput=7.63 tokens/ms
  Batch time grew 8.18x while batch size grew 16.00x -- roughly proportionally, once compute-bound.
  Throughput grew only 1.96x for a 16.00x bigger batch --
  every request now waits longer per token, for a throughput gain far short of
  proportional. Past B*, adding requests to a batch trades latency for very
  little additional throughput, the mirror image of Test 1's near-free batching.

========================================================
6/6 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming more batching always helps throughput proportionally"
    Below the batch-size crossover, adding requests to a batch is close to free — batch time barely grows, so throughput grows almost proportionally with batch size, and it is tempting to extrapolate that relationship indefinitely. Past the crossover, the batch is compute-bound, and batch time grows roughly in step with batch size instead of staying flat — so throughput growth falls sharply behind batch-size growth, while every request's own per-token latency, tied to that now-growing batch time, gets measurably worse. A scheduler that keeps adding requests to an already compute-bound batch is not buying the throughput gain it bought earlier; it is trading request latency for a shrinking throughput return, the same trade a memory-bound-to-compute-bound crossover always forces once bytes stop being the bottleneck.

## 8.5 A Full Layer's Roofline Profile

### Intuition

A single decode step touches far more than one kernel: two RMSNorms, four attention projections, attention itself against a growing KV cache, and three FFN projections. It is tempting to assume that because attention's cost visibly GROWS with context length, it must be the part that dominates a long-context decode step's memory traffic. Summing every kernel's own cost formula from this chapter shows that, at realistic context lengths, it is not — the weight matrices, read completely fresh on every single step regardless of how long the conversation has been, are far larger.

### The Concept, In Detail

Every projection's cost reuses Section 8.3's verified linear-layer formula directly. Attention's cost is new to this section but built to match, term for term, a real GQA attention kernel: each KV head's cached rows are read once and reused by every query head in its group — the same "read once, reuse across the group" shape Section 8.3 used for weight matrices — while the score dot products, softmax, and weighted sum are real per-element arithmetic paid by every query head, verified here against Chapter 3.4's own std::mdspan-viewed KVCache and gqa_attention, reused verbatim with a cost counter threaded through it. Summing all of these for one decode step at a realistic context length confirms the whole step lands memory-bound, consistent with Section 8.1's own choice of a representative decode arithmetic intensity. Separating the step's bytes into the part that is FIXED (the QKVO and FFN weight matrices, read in full on every step regardless of context length) and the part that GROWS with context (the KV cache read) shows the fixed part dominating by more than an order of magnitude at a two-thousand-token context, and finds the context length at which the two would actually cross to be far beyond most deployed context windows.

### Code and Verification

```cpp
// 05_full_layer_roofline_profile.cpp
// Chapter 8, Part 5 (capstone): profile one full transformer layer's
// decode step -- both RMSNorms, the four attention projections (Q, K, V,
// O), attention itself against a GQA KV cache, and the three FFN
// projections (gate, up, down) -- by summing the SAME cost formulas this
// chapter has already built and, in Section 8.3's case, already verified
// against a real kernel: linear_fp32_cost for every projection, and a
// new attention-cost formula verified here the same way, against a real,
// std::mdspan-viewed GQA attention kernel reused verbatim from Chapter
// 3.4's KVCache.
//
// The question this file answers: of everything a decode step touches,
// what actually dominates the bytes moved -- the weights, or the KV
// cache that grows with every generated token?
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 05_full_layer_roofline_profile.cpp -o out05

#include <mdspan/mdspan.hpp>
#include <cmath>
#include <cstdint>
#include <span>
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
// COST COUNTER (Sections 8.2/8.3's convention, reused)
// =========================================================================
struct Cost {
    long long flops = 0;
    long long bytes = 0;
    void flop(long long n = 1) { flops += n; }
    void access(long long n_bytes) { bytes += n_bytes; }
    double arithmetic_intensity() const { return static_cast<double>(flops) / static_cast<double>(bytes); }
};
Cost operator+(Cost a, const Cost& b) { a.flops += b.flops; a.bytes += b.bytes; return a; }
Cost& operator+=(Cost& a, const Cost& b) { a.flops += b.flops; a.bytes += b.bytes; return a; }

// Section 8.3's verified linear-layer formula, unchanged.
Cost linear_fp32_cost(long long n_out, long long n_in, long long seq) {
    Cost cost;
    cost.access(n_out * n_in * 4);
    cost.access(n_in * seq * 4);
    cost.access(n_out * seq * 4);
    cost.flop(2 * n_out * n_in * seq);
    return cost;
}
// Section 8.2's verified RMSNorm formula, generalized from a fixed DIM
// to a parameter (the derivation is identical; only the symbol changes).
constexpr long long EXP_FLOP_COST = 8;
Cost rmsnorm_cost(long long dim) {
    Cost c; c.flop(4 * dim + 4); c.access(3 * dim * 4); return c;
}

// =========================================================================
// CLOSED-FORM ATTENTION COST: one new query token, against `cache_len`
// already-cached positions, for n_heads_q query heads sharing n_heads_kv
// KV heads (GQA). Each KV head's cache rows are read ONCE (bytes) and
// reused by every query head in its group -- the same "read once, reuse
// across the group" shape Section 8.3 used for weight matrices -- while
// the score dot products, softmax, and weighted sum are real per-element
// arithmetic paid by every query head. This formula is written to match,
// term for term, what the real kernel below actually counts as it runs
// (Test 0 checks the two agree exactly), rather than being independently
// asserted.
// =========================================================================
Cost attention_cost_formula(long long n_heads_q, long long n_heads_kv, long long head_dim, long long cache_len) {
    Cost c;
    c.access(n_heads_kv * 2 * cache_len * head_dim * 4);      // K and V, one read each, per KV head
    c.flop(n_heads_q * cache_len * head_dim * 2);             // Q.K^T score dot products, all query heads
    c.flop(n_heads_q * cache_len * head_dim * 2);             // weighted sum over V, all query heads
    c.flop(n_heads_q * cache_len * (2 + EXP_FLOP_COST));      // softmax: subtract+exp, then scale, per element
    c.access(n_heads_q * 8 * cache_len);                      // softmax's own read-then-write of the score buffer
    return c;
}

// =========================================================================
// REAL, SMALL-SCALE VALIDATION: Chapter 3.4's KVCache and gqa_attention,
// reused verbatim (the exact struct and function, mdspan/submdspan and
// all), with a Cost counter threaded through so the closed-form formula
// above can be checked against bytes and FLOPs a real kernel actually
// produced -- not merely asserted.
// =========================================================================
void softmax_inplace_counted(std::span<float> scores, Cost& cost) {
    int len = static_cast<int>(scores.size());
    cost.access(4LL * len);
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; cost.flop(1 + EXP_FLOP_COST); }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) { s *= inv_sum; cost.flop(1); }
    cost.access(4LL * len);
}

struct KVCache {
    std::vector<float> K, V;
    int n_heads_kv, max_seq_len, head_dim;
    using View = std::mdspan<float, std::dextents<size_t, 3>>;

    KVCache(int nh, int seq, int hd) : n_heads_kv(nh), max_seq_len(seq), head_dim(hd) {
        K.assign(static_cast<size_t>(nh) * seq * hd, 0.0f);
        V.assign(static_cast<size_t>(nh) * seq * hd, 0.0f);
    }
    View k_view() { return View(K.data(), n_heads_kv, max_seq_len, head_dim); }
    View v_view() { return View(V.data(), n_heads_kv, max_seq_len, head_dim); }
    auto k_at(int h, int t) { return std::submdspan(k_view(), h, t, std::full_extent); }
    auto v_at(int h, int t) { return std::submdspan(v_view(), h, t, std::full_extent); }
    void store(int h, int t, std::span<const float> k, std::span<const float> v) {
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[i]; vslice[i] = v[i]; }
    }
};

// Chapter 3.4's gqa_attention, reused verbatim, with one addition: a
// Cost& thread that counts each K/V slot exactly once per KV head (read
// once, reused by every query head in its group) and every FLOP as the
// real dot products and weighted sums execute.
void gqa_attention_counted(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                            int cache_len, int n_heads_q, int group_size, Cost& cost) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(cache_len);
    int last_kv_h_counted = -1;

    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, head_dim);

        // This KV head's cache rows are read once per kv_h -- reused by
        // every query head in its group -- exactly the shape
        // attention_cost_formula charges.
        bool first_in_group = (kv_h != last_kv_h_counted);
        if (first_in_group) { cost.access(static_cast<long long>(cache_len) * head_dim * 4); last_kv_h_counted = kv_h; }

        for (int t = 0; t < cache_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            float d = 0.0f;
            for (int i = 0; i < head_dim; ++i) { d += q[i] * k[i]; cost.flop(2); }
            scores[t] = d * scale;
        }
        softmax_inplace_counted(std::span<float>(scores.data(), cache_len), cost);

        float* out = output.data() + h * head_dim;
        std::fill(out, out + head_dim, 0.0f);
        if (first_in_group) cost.access(static_cast<long long>(cache_len) * head_dim * 4);  // V, same reuse
        for (int t = 0; t < cache_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            float w = scores[t];
            for (int i = 0; i < head_dim; ++i) { out[i] += w * v[i]; cost.flop(2); }
        }
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 8.5: A Full Layer's Roofline Profile\n";
    std::cout << "========================================================\n\n";

    constexpr double PEAK_COMPUTE_GFLOPS = 896.0, PEAK_BANDWIDTH_GBPS = 51.2;
    constexpr double RIDGE_POINT = PEAK_COMPUTE_GFLOPS / PEAK_BANDWIDTH_GBPS;

    // =====================================================================
    // TEST 0: the closed-form attention formula matches a real,
    // mdspan-viewed GQA kernel's measured Cost, at small dimensions.
    // =====================================================================
    std::cout << "-- Test 0: closed-form attention cost vs. a real GQA kernel, same shape --\n";
    {
        constexpr int HEAD_DIM = 8, N_Q = 4, N_KV = 2, GROUP = 2, CACHE_LEN = 6;
        KVCache cache(N_KV, CACHE_LEN, HEAD_DIM);
        std::vector<float> k(HEAD_DIM), v(HEAD_DIM);
        for (int h = 0; h < N_KV; ++h) {
            for (int t = 0; t < CACHE_LEN; ++t) {
                for (int i = 0; i < HEAD_DIM; ++i) { k[i] = 0.01f * ((h * 37 + t * 11 + i) % 13 - 6); v[i] = 0.02f * ((h * 19 + t * 7 + i) % 9 - 4); }
                cache.store(h, t, k, v);
            }
        }
        std::vector<float> q(N_Q * HEAD_DIM), output(N_Q * HEAD_DIM);
        for (size_t i = 0; i < q.size(); ++i) q[i] = 0.05f * (static_cast<float>(i % 7) - 3.0f);

        Cost real_cost;
        gqa_attention_counted(q, cache, output, CACHE_LEN, N_Q, GROUP, real_cost);
        Cost formula_cost = attention_cost_formula(N_Q, N_KV, HEAD_DIM, CACHE_LEN);

        std::cout << "  Real GQA kernel: FLOPs=" << real_cost.flops << " Bytes=" << real_cost.bytes << "\n";
        std::cout << "  Closed formula:  FLOPs=" << formula_cost.flops << " Bytes=" << formula_cost.bytes << "\n";
        CHECK(real_cost.flops == formula_cost.flops);
        CHECK(real_cost.bytes == formula_cost.bytes);
    }

    // =====================================================================
    // Real illustrative model dimensions (Section 8.3's FFN convention,
    // extended to a full layer): DIM=4096, 32 query heads sharing 8 KV
    // heads (GQA group=4, the Llama-3-8B-class ratio), D_FF=14336.
    // =====================================================================
    constexpr long long DIM = 4096, N_HEADS_Q = 32, N_HEADS_KV = 8, HEAD_DIM = DIM / N_HEADS_Q, D_FF = 14336;
    constexpr long long KV_DIM = N_HEADS_KV * HEAD_DIM;

    auto decode_step_cost = [&](long long cache_len_before) {
        Cost c;
        c += rmsnorm_cost(DIM); c += rmsnorm_cost(DIM);                              // attn_norm, ffn_norm
        c += linear_fp32_cost(DIM, DIM, 1);      // W_q
        c += linear_fp32_cost(KV_DIM, DIM, 1);   // W_k
        c += linear_fp32_cost(KV_DIM, DIM, 1);   // W_v
        c += linear_fp32_cost(DIM, DIM, 1);      // W_o
        c += attention_cost_formula(N_HEADS_Q, N_HEADS_KV, HEAD_DIM, cache_len_before + 1);  // +1: this step's own new token
        c += linear_fp32_cost(D_FF, DIM, 1);     // W_gate
        c += linear_fp32_cost(D_FF, DIM, 1);     // W_up
        c += linear_fp32_cost(DIM, D_FF, 1);     // W_down
        return c;
    };
    // Bytes belonging only to the four QKVO projections and three FFN
    // projections -- the part of a decode step that does NOT depend on
    // how long the conversation so far has been.
    auto fixed_weight_bytes = [&]() {
        Cost c;
        c += linear_fp32_cost(DIM, DIM, 1);
        c += linear_fp32_cost(KV_DIM, DIM, 1);
        c += linear_fp32_cost(KV_DIM, DIM, 1);
        c += linear_fp32_cost(DIM, DIM, 1);
        c += linear_fp32_cost(D_FF, DIM, 1);
        c += linear_fp32_cost(D_FF, DIM, 1);
        c += linear_fp32_cost(DIM, D_FF, 1);
        return c.bytes;
    };

    // =====================================================================
    // TEST 1: at a realistic context length, classify the whole decode
    // step against the roofline, and check it lands memory-bound (as
    // Section 8.1's Test 3 assumed when it picked AI=0.5 as "a
    // representative decode-time GEMV").
    // =====================================================================
    std::cout << "\n-- Test 1: one decode step at cache_len=2048, classified against the roofline --\n";
    {
        constexpr long long L = 2048;
        Cost step = decode_step_cost(L);
        double ai = step.arithmetic_intensity();
        std::cout << "  FLOPs=" << step.flops << "  Bytes=" << step.bytes
                  << "  AI=" << std::fixed << std::setprecision(4) << ai << " FLOPs/byte\n";
        std::cout << "  Ridge point=" << std::setprecision(2) << RIDGE_POINT << " -> "
                  << (ai < RIDGE_POINT ? "MEMORY-BOUND" : "COMPUTE-BOUND") << "\n";
        CHECK(ai < RIDGE_POINT);
    }

    // =====================================================================
    // TEST 2 (COMMON TRAP): the KV cache read is the part of a decode
    // step that grows with context length, so it is tempting to assume
    // it dominates a long-context decode step's cost. At realistic
    // context lengths it does not -- the QKVO and FFN weight matrices,
    // read completely fresh on EVERY step regardless of context length,
    // are far larger.
    // =====================================================================
    std::cout << "\n-- Test 2 [COMMON TRAP]: attention does not dominate decode bytes at realistic context lengths --\n";
    {
        constexpr long long L = 2048;
        long long weight_bytes = fixed_weight_bytes();
        // Bytes attributable to K+V cache reads alone: one read of each
        // cached K and V slot, per KV head (shared across its group of
        // query heads, exactly as attention_cost_formula charges it).
        long long kv_cache_bytes = N_HEADS_KV * 2 * L * HEAD_DIM * 4;

        std::cout << "  Fixed QKVO+FFN weight bytes (every step, any context length): "
                  << weight_bytes / (1024 * 1024) << " MB\n";
        std::cout << "  KV-cache read bytes at context length " << L << ": "
                  << kv_cache_bytes / 1024 << " KB\n";
        std::cout << "  Weight bytes exceed KV-cache bytes by " << std::setprecision(1)
                  << (static_cast<double>(weight_bytes) / static_cast<double>(kv_cache_bytes)) << "x at this context length.\n";

        long long crossover_L = weight_bytes / (N_HEADS_KV * 2 * HEAD_DIM * 4);
        std::cout << "  The KV cache would need to reach " << crossover_L
                  << " tokens of context before its read bytes alone caught up to the fixed\n";
        std::cout << "  per-step weight-read cost -- far beyond a " << L << "-token context, and beyond\n";
        std::cout << "  most deployed context windows. Attention's cost is the part that VISIBLY\n";
        std::cout << "  grows with context, which is exactly why it gets blamed first -- but the\n";
        std::cout << "  fixed weight-read cost, paid again every single step regardless of context,\n";
        std::cout << "  is what a decode step's memory traffic is actually dominated by until the\n";
        std::cout << "  context grows far longer than most conversations ever do.\n";

        CHECK(weight_bytes > kv_cache_bytes * 10);       // weight reads dominate by an order of magnitude here
        CHECK(crossover_L > 50000);                       // and only stop dominating at a very long context
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
g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 05_full_layer_roofline_profile.cpp -o 05_full_layer_roofline_profile
./05_full_layer_roofline_profile
```

**Sample input:** the closed-form attention cost formula checked against a real, std::mdspan-viewed GQA kernel reused from Chapter 3.4, at small dimensions; a full decode step at DIM=4096, 32 query heads sharing 8 KV heads (GQA group 4), D_FF=14336, and context length 2048, classified against the roofline; and a deliberate comparison of the step's fixed weight-read bytes against its context-dependent KV-cache-read bytes at that same context length.

```text
========================================================
Chapter 8.5: A Full Layer's Roofline Profile
========================================================

-- Test 0: closed-form attention cost vs. a real GQA kernel, same shape --
  Real GQA kernel: FLOPs=1008 Bytes=960
  Closed formula:  FLOPs=1008 Bytes=960

-- Test 1: one decode step at cache_len=2048, classified against the roofline --
  FLOPs=470466888  Bytes=890151168  AI=0.5285 FLOPs/byte
  Ridge point=17.50 -> MEMORY-BOUND

-- Test 2 [COMMON TRAP]: attention does not dominate decode bytes at realistic context lengths --
  Fixed QKVO+FFN weight bytes (every step, any context length): 832 MB
  KV-cache read bytes at context length 2048: 16384 KB
  Weight bytes exceed KV-cache bytes by 52.0x at this context length.
  The KV cache would need to reach 106536 tokens of context before its read bytes alone caught up to the fixed
  per-step weight-read cost -- far beyond a 2048-token context, and beyond
  most deployed context windows. Attention's cost is the part that VISIBLY
  grows with context, which is exactly why it gets blamed first -- but the
  fixed weight-read cost, paid again every single step regardless of context,
  is what a decode step's memory traffic is actually dominated by until the
  context grows far longer than most conversations ever do.

========================================================
5/5 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming attention dominates a long-context decode step"
    Attention's KV-cache read is the part of a decode step's cost that visibly grows with context length, which makes it the natural first place to look when a long-context deployment is slow — but "the part that grows" and "the part that dominates" are not the same claim. At a two-thousand-token context, the fixed QKVO-and-FFN weight-read cost, paid again in full on every single decode step no matter how long the conversation has been, outweighs the KV-cache read by more than fifty times, because the weight matrices are simply that much larger than one step's worth of cached keys and values. The KV-cache read only catches up to the fixed weight cost at a context length measured in the hundreds of thousands of tokens — far beyond most deployed context windows. Profiling effort spent shrinking attention's footprint at ordinary context lengths addresses the part of the bottleneck that is easiest to see, not the part that is actually largest.

## Chapter Summary

This chapter built the roofline model from first principles — peak compute and peak bandwidth derived from stated architectural parameters, not quoted from a data sheet, combining into a ridge point that separates memory-bound kernels from compute-bound ones — and then applied it, section by section, to increasingly complete pieces of a real inference engine. Section 8.1 showed that a faster processor does nothing for a memory-bound kernel, because the two peaks are independent resources and a kernel is only ever limited by whichever one it demands more of. Section 8.2 measured three real transformer kernels' own arithmetic intensity by instrumenting their actual control flow, finding RMSNorm, softmax, and RoPE all memory-bound by a wide margin, and showed how easily an incomplete byte count inflates that measurement. Section 8.3 showed that prefill and decode are the identical weight matrix and arithmetic sitting on opposite sides of the same ridge point, found the exact sequence length where one becomes the other, and showed quantization's speedup vanishing once a sequence is deep enough into the compute-bound regime. Section 8.4 showed that continuous batching is the identical crossover viewed from a scheduler's perspective, and that the throughput gains it provides below the crossover become a latency cost above it. Section 8.5 closed the chapter by summing every kernel this book has built into one full decode step's roofline profile, and showed that the part of the step's cost that visibly grows with context — attention against the KV cache — is not the part that dominates its memory traffic at realistic context lengths; the weight matrices, read fresh on every step regardless of context, are. Part 1 asked what a model's numbers should be represented as; Part 2, which this chapter opens, asks how fast an engine can actually move those numbers through a CPU, and the roofline model built here is the framework every remaining optimization decision in Part 2 will be checked against.

## Self-Check Questions

1. Why does quadrupling a machine's peak compute change a memory-bound kernel's achievable throughput by exactly zero, not merely by a small amount?
2. Explain, in terms of the achievable-throughput formula, why a kernel's classification as memory-bound or compute-bound depends on both the kernel's own arithmetic intensity and the machine's ridge point, not on the kernel alone.
3. Section 8.2 counts a transcendental function like `exp` as costing a stated eight FLOPs, not one. Why does treating it as free (one FLOP, or zero) distort a kernel's measured arithmetic intensity, and in which direction?
4. What specifically was forgotten in Section 8.2's COMMON TRAP, and why does omitting it inflate arithmetic intensity rather than deflate it?
5. Why are prefill and decode described as "the same weight matrix, different arithmetic" rather than as two fundamentally different kernels?
6. Explain why quantizing a weight matrix shifts a GEMM's memory-bound-to-compute-bound crossover to a SMALLER sequence length, and why that same quantization stops helping at all once the sequence is comfortably past the crossover on both sides.
7. In Section 8.4, why is a batch's total processing time also each individual request's own per-token latency, and what does that fact depend on about how the batch is served?
8. Explain why continuous batching's throughput benefit shrinks, rather than continuing proportionally, once a batch's size passes the compute-bound crossover.
9. In Section 8.5, why is the KV-cache read described as "the part that visibly grows with context" while the QKVO/FFN weight reads are described as "fixed," and why does that distinction matter for where a long-context deployment's actual bottleneck is likely to be?
10. Section 8.3's real GEMM kernel runs at small, illustrative dimensions rather than at the FFN projection's real 14336x4096 size. What is that small kernel actually used to verify, and why does the chapter trust the closed-form formula at the full size as a result?

## Where We Go Next

This chapter established the framework — the roofline model — that every remaining optimization in this book will be checked against: is a given kernel memory-bound or compute-bound, and does a proposed change actually address the resource that is limiting it. Chapter 9 puts that framework to work on the compute side of the ridge, covering SIMD vectorization — AVX2 on x86 and NEON on Arm — the mechanism by which a compute-bound kernel's peak FLOP rate is actually achieved in practice, and, just as importantly, the mechanism that determines how large a MEMORY-bound kernel's peak bandwidth figure really is once real cache and prefetch behavior are accounted for.

## Worked Solutions

**1.** The achievable-throughput formula for a memory-bound kernel is `arithmetic_intensity * peak_bandwidth` — an expression that does not contain the peak-compute term at all, because a memory-bound kernel's arithmetic intensity, by definition, sits below the ridge point where compute would first become the limit. Quadrupling peak compute changes a quantity that this formula never depended on in the first place, so the change is not merely small; it is exactly zero, because bandwidth was the entire bottleneck and no additional arithmetic capacity relieves a constraint that was never binding.

**2.** Whether a kernel is memory-bound or compute-bound is decided by comparing the kernel's own arithmetic intensity against the RATIO of the machine's two peaks (the ridge point), not against either peak in isolation. The identical kernel, with the identical arithmetic intensity, can be memory-bound on one machine and compute-bound on another if the two machines have different ridge points — so "is this kernel memory-bound" is never a question about the kernel alone; it is a question about the kernel running on a SPECIFIC machine.

**3.** Treating `exp` as one FLOP (or as free) understates the true arithmetic cost of every element softmax touches, which understates the numerator of the arithmetic-intensity ratio (FLOPs) while leaving the denominator (bytes) unchanged. This makes the kernel look MORE memory-bound than it actually is — its true arithmetic intensity is higher than the undercounted figure suggests, because the real hardware cost of a vectorized `exp` is measured, in practice, to be several times a single FLOP, not one.

**4.** Section 8.2's trap forgot to count the bytes read from the cosine and sine lookup tables RoPE depends on, counting only the rotated vector's own read and write. Omitting a real memory access removes bytes from the denominator of the arithmetic-intensity ratio while leaving the FLOP numerator unchanged, which makes the computed ratio LARGER than the true one — an inflation, not a deflation, and specifically dangerous for any kernel whose true intensity sits close to the ridge point, since an inflated number is exactly what could push a genuinely memory-bound kernel across the line into an apparent, incorrect compute-bound classification.

**5.** Prefill and decode invoke the identical weight matrix through the identical linear-algebra operation — a matrix multiplied by a batch of columns — differing only in how many columns that batch contains (many prompt tokens for prefill, one generated token for decode). Because the weight matrix is read once and reused across every column in the batch, the SAME kernel's arithmetic intensity rises with batch size; prefill and decode are two points on that single curve, not two different kernels with two different formulas.

**6.** Quantizing the weight matrix reduces the bytes the weight-read term of the cost model contributes without changing the FLOP count at all, which raises arithmetic intensity at every sequence length — including smaller ones — so the sequence length at which that raised intensity first reaches the ridge point is smaller than it was for the unquantized format. Once a sequence length is comfortably past BOTH formats' crossover, however, both are compute-bound, and the achievable-throughput formula in that regime is simply `peak_compute`, a constant that does not depend on arithmetic intensity (and therefore not on the format's byte count) at all — so quantization's byte-count advantage, which only mattered while bytes were the bottleneck, stops producing any speedup whatsoever.

**7.** A batch of decode requests served synchronously means no request's next token is available until the entire batch's GEMM has finished computing every request's output column — there is no mechanism by which one request's token could be ready earlier than another's within the same batch. Because of this, the roofline-predicted time to process the whole batch IS the time every individual request must wait for its own next token, so "batch processing time" and "per-request latency" are the same measured quantity, not two quantities that happen to correlate.

**8.** Below the compute-bound crossover, batch processing time grows far more slowly than batch size (the shared weight read dominates and barely changes), so throughput — requests served per unit time — grows almost as fast as batch size does. Past the crossover, the shared-weight-read discount has nowhere further to apply, since FLOPs, not bytes, are now the limiting resource, and FLOPs scale directly with the number of requests in the batch — so batch time now grows roughly in proportion to batch size, and throughput's growth falls correspondingly far behind the batch size's own growth, rather than continuing at the earlier, near-proportional rate.

**9.** The KV cache accumulates one more cached position with every token generated, so the number of bytes attention must read from it grows with the length of the conversation so far; the QKVO and FFN weight matrices are a fixed size set once at model-load time and are read in their entirety on every decode step regardless of how many tokens have been generated. This distinction matters because it is tempting to assume that whatever grows with context length must eventually dominate the cost — and it eventually does, but only at a context length this section shows to be far longer than most real conversations reach, meaning the FIXED weight-read cost is the actual dominant term across the range of context lengths most deployments will ever see.

**10.** The small kernel is used to verify that the closed-form cost FORMULA — the arithmetic used to predict FLOPs and bytes without materializing a buffer — exactly matches what a real, running kernel measures at an identical shape, both in total FLOPs and in total bytes. Because the formula is pure arithmetic over shape parameters (n_out, n_in, seq), a match at one shape is a match at every shape the same formula is evaluated at — so once the small-scale check passes, the chapter can trust the same formula applied to the FFN projection's real, much larger dimensions without needing to allocate and run a kernel over a buffer of that full size just to re-confirm arithmetic already shown to be correct.
