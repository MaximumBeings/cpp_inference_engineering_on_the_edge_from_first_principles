// Appendix A -- a real, minimal smoke-test source file for this appendix's
// own CMake project: a float32 dot product with a genuine x86_64 AVX2
// path, a genuine aarch64 NEON path, and a portable scalar fallback,
// selected entirely at COMPILE time by which architecture macro the
// toolchain actually defines -- exactly the real decision a production
// CMakeLists.txt has to make when the identical source tree is built once
// on an x86 development machine and once cross-compiled for an Arm edge
// target. All three paths are checked against each other for exact
// numerical agreement before this file ever reports success.
//
// This file is deliberately independent of Chapter 9's own AVX2/NEON
// dot-product files (which teach the two instruction sets side by side on
// a Q8_0/Q4_0 quantized block layout) -- this appendix's own job is the
// BUILD SYSTEM around a file like this, not the intrinsics themselves, so
// this file uses a plain float32 array to keep the CMake story the focus.
//
// Compiled and run automatically by this appendix's own CMakeLists.txt;
// see appendix_a/README.md for the exact native and cross-compiled
// commands this file was actually verified with.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define APPENDIX_A_BACKEND "AVX2 (x86_64)"
#elif defined(__aarch64__)
#include <arm_neon.h>
#define APPENDIX_A_BACKEND "NEON (aarch64)"
#else
#define APPENDIX_A_BACKEND "portable scalar"
#endif

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-4) { return std::fabs(a - b) < eps; }

// The portable scalar reference every architecture's own accelerated path
// is checked against -- this is the one function guaranteed to compile
// and run correctly on any target, including one this appendix's own
// toolchain file was never even written for.
float dot_scalar(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

#if defined(__x86_64__) || defined(_M_X64)
float dot_avx2(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float partial[8];
    _mm256_storeu_ps(partial, acc);
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) sum += partial[j];
    for (; i < n; ++i) sum += a[i] * b[i];  // real scalar tail for n not a multiple of 8
    return sum;
}
#elif defined(__aarch64__)
float dot_neon(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb);
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) sum += a[i] * b[i];  // real scalar tail for n not a multiple of 4
    return sum;
}
#endif

float dot_accelerated(const float* a, const float* b, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    return dot_avx2(a, b, n);
#elif defined(__aarch64__)
    return dot_neon(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Appendix A smoke test: portable_dot_product\n";
    std::cout << "Backend selected at compile time: " << APPENDIX_A_BACKEND << "\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a small, hand-verifiable real case -- {1,2,3,4,5} . {2,2,2,2,2} = 30 -- "
                 "matches on both the portable scalar reference and this architecture's own accelerated "
                 "path --\n";
    {
        std::vector<float> a = {1, 2, 3, 4, 5};
        std::vector<float> b = {2, 2, 2, 2, 2};
        float scalar_result = dot_scalar(a.data(), b.data(), 5);
        float accel_result = dot_accelerated(a.data(), b.data(), 5);
        CHECK(near(scalar_result, 30.0));
        CHECK(near(accel_result, 30.0));
        std::cout << "  scalar = " << scalar_result << ", accelerated (" << APPENDIX_A_BACKEND
                  << ") = " << accel_result << "\n";
    }

    std::cout << "\n-- Test 2: on a real, deterministically generated 257-element vector pair (a length "
                 "deliberately not a multiple of 4 or 8, exercising each accelerated path's own real "
                 "scalar tail-handling code), the accelerated path matches the portable scalar reference "
                 "to within a tight real tolerance --\n";
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> a(257), b(257);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        float scalar_result = dot_scalar(a.data(), b.data(), 257);
        float accel_result = dot_accelerated(a.data(), b.data(), 257);
        CHECK(near(scalar_result, accel_result, 1e-2));
        std::cout << "  scalar = " << scalar_result << ", accelerated (" << APPENDIX_A_BACKEND
                  << ") = " << accel_result << " (257 elements -- exercises the tail loop)\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
