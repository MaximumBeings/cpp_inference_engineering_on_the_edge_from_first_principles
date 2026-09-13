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
