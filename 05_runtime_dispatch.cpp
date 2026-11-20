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
// Compile: g++ -std=c++23 -Wall -Wextra -O3 05_runtime_dispatch.cpp -o 05_runtime_dispatch

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
// within the stated tolerance: SIMD reduction order and fused operations
// can differ from the scalar reference in their final rounding bits.
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
    // TEST 3: the fallback is compiled for the baseline ISA. Global
    // -mavx2/-mfma or -march=native could permit unsupported instructions
    // outside the guarded target-attributed function, even if CPUID checks
    // remain in the binary. Selecting scalar_dot does not undo such flags.
    // A same-host run checks arithmetic, not execution on an older CPU.
    std::cout << "\n-- Test 3: baseline compilation and specialized dispatch --\n";
    {
        std::cout << "  Compile the baseline translation unit without -mavx2/-mfma or -march=native.\n";
        std::cout << "  Only avx2_dot has target(\"avx2,fma\"); dispatch checks both features.\n";
        std::cout << "  Global ISA flags may introduce unsupported instructions outside that function.\n";
        std::cout << "  A forced scalar arithmetic check is not proof of execution on a non-AVX2 CPU.\n";
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
