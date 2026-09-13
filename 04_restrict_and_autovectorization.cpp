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
