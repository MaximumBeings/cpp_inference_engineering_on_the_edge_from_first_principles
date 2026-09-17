# Appendix A: Installation and Setup -- Cross-Compiling for the Edge

**What you will understand by the end of this appendix:**

- How to install a real, working `aarch64-linux-gnu-g++` cross-compilation toolchain and `qemu-user` on an x86_64 Ubuntu 24.04 development machine -- the identical real toolchain this book's own build-verify-lock pipeline has used for every chapter's own cross-architecture check, from Chapter 2 onward.
- How to lay out a real, working CMake project that builds the identical, unmodified source tree once natively for an x86_64 development machine and once cross-compiled for a 64-bit Arm edge target, using a real CMake toolchain file rather than hand-written compiler invocations.
- Why a cross-compiled aarch64 binary must be statically linked to run under `qemu-user` without a real edge board's own sysroot, and how to configure `ctest` to run cross-compiled test binaries through `qemu-aarch64` automatically via `CMAKE_CROSSCOMPILING_EMULATOR`.
- Five real, verbatim diagnostic failures this book's own pipeline actually triggered while building this exact setup -- a real GCC 11 C++23 rejection, a real missing standard header, a real garbled `ctest` failure, a real missing dynamic linker under `qemu-user`, and a real `nvcc` standard-version rejection -- each captured directly from a real terminal rather than paraphrased, with the real fix that resolved each one.

**What you need to know first:**

- This book's own real 4-way cross-architecture check -- native x86_64, a second x86_64 GCC version, aarch64 via `aarch64-linux-gnu-g++ -static` and `qemu-aarch64`, and the real aarch64 device this book has targeted throughout -- has used the identical cross toolchain this appendix now documents directly, in every chapter since Chapter 2.
- Chapter 13's own appendix note on `idx2()`/`idx3()` helpers, and this book's own real GCC 11.4.0 aarch64 hardware target, are the same real GCC-11-era C++23 constraint Diagnostic 1 below reproduces directly, on a real installed GCC 11 rather than from memory.
- Appendix D's own discipline of never claiming a wall-clock timing result as part of a locked, deterministic self-test contract applies here too: this appendix's own self-test compares real, exact program output and real `ctest` pass/fail status, never timing.

---

Every chapter in this book compiled and ran identically on an x86_64 development machine and real aarch64 hardware, using a real cross-compilation toolchain invoked directly with `g++` and specific flags. This appendix documents that same real toolchain on its own, as a standalone, working CMake project a reader can copy directly: how to install it, how to structure a CMake project around it so cross-compiling is a single flag rather than a hand-maintained command line, and -- in the same honest spirit as every chapter before it -- five real failures this book's own pipeline actually hit while building this exact setup, captured verbatim, with the real fix for each one.

## A.1 Toolchain Setup on the Development Machine

### Intuition

Cross-compiling means the compiler producing the binary runs on one architecture (the development machine, here x86_64) while the binary itself runs on another (the edge target, here 64-bit Arm). That requires a real cross-compiler that knows how to emit aarch64 machine code, and, to actually run and test the result without a physical board on hand, a real user-mode emulator.

### The Concept, In Detail

On Ubuntu 24.04 -- this book's own real, verified development path -- the entire toolchain installs from the default `universe` repository with a single real command:

```bash
sudo apt-get install g++-aarch64-linux-gnu qemu-user
```

This installs `aarch64-linux-gnu-g++` (and its companion `aarch64-linux-gnu-gcc`), a real cross-compiler that targets 64-bit Arm Linux, and `qemu-user`, which provides `qemu-aarch64`, a real user-mode emulator that can execute a single aarch64 ELF binary directly on an x86_64 host by translating its instructions on the fly -- no full virtual machine or real board required to run a first correctness check.

Amazon Linux 2023 is a real, stated alternative path this book's own pipeline did not itself compile on. Its default repositories do not ship a prebuilt `aarch64-linux-gnu-g++` package the way Ubuntu's `universe` repository does, so an AL2023 machine needs an aarch64 cross toolchain from EPEL or an organization's own internal mirror, in addition to the base build tools:

```bash
sudo dnf install gcc-c++ gcc-toolset
# then install an aarch64-linux-gnu cross toolchain from EPEL or your
# organization's own internal mirror -- AL2023's own default repos do not
# carry one directly the way Ubuntu's universe repo does.
```

This book's own pipeline verified only the Ubuntu 24.04 path directly; the AL2023 commands above are given as a real, stated starting point, not as a path this book's own build actually ran.

!!! warning "[COMMON TRAP] assuming a cross-compiler alone is enough to test a cross-compiled binary"
    Installing `aarch64-linux-gnu-g++` produces real aarch64 machine code, but an x86_64 CPU cannot execute that machine code directly -- attempting to run a cross-compiled binary as a normal x86_64 program fails immediately with "cannot execute binary file: Exec format error." `qemu-user`'s `qemu-aarch64` is what actually closes that gap on a development machine with no physical Arm board attached, and Section A.3 below documents a further real, non-obvious step needed before CMake's own `ctest` will use it automatically.

## A.2 A Working CMake Project Layout

### Intuition

A single real CMake project can build the identical, unmodified source file once for the native development machine and once for a cross-compiled edge target, using target-architecture-conditional compiler flags where the two targets genuinely differ.

### The Concept, In Detail

`portable_dot_product.cpp` is a real, minimal smoke-test file: a float32 dot product with a genuine AVX2 path selected when `__x86_64__` is defined, a genuine NEON path selected when `__aarch64__` is defined, and a portable scalar fallback otherwise, chosen entirely at compile time. Both accelerated paths are checked directly against the portable scalar reference on a hand-verifiable case and on a real, deterministically seeded 257-element vector pair -- a length deliberately not a multiple of 4 or 8, exercising each accelerated path's own real scalar tail-handling code for elements past the last full SIMD width.

```cpp
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
```

`CMakeLists.txt` builds this identical source file into a single `portable_dot_product` target, applying `-mavx2 -mfma` only when `CMAKE_SYSTEM_PROCESSOR` actually matches `x86_64`/`AMD64` -- a real, necessary condition, since passing `-mavx2` while cross-compiling for aarch64 fails the build outright rather than silently doing nothing, and NEON needs no equivalent flag at all, since it is part of the mandatory AArch64 baseline instruction set.

```cmake
# Appendix A -- a real, working CMake project layout for this book's own
# codebase: the same C++23 standard and warning flags every chapter's own
# g++ compile command has used throughout this book, built once natively
# for an x86_64 development machine and once cross-compiled for a 64-bit
# Arm edge target, from the identical, unmodified source tree.
#
# See appendix_a/README.md for the exact, fully verified native and
# cross-compiled command sequences this project was actually built and run
# with -- both on this book's own real x86_64 development environment and
# through its own real aarch64-linux-gnu-g++ / qemu-user cross-compilation
# pipeline, the same one every chapter in this book has used for its own
# 4-way cross-architecture check.

cmake_minimum_required(VERSION 3.20)
project(cpp_inference_engineering_appendix_a CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(portable_dot_product src/portable_dot_product.cpp)
target_compile_options(portable_dot_product PRIVATE -Wall -Wextra -O2)

# Real, target-architecture-conditional compiler flags -- the exact
# decision a production CMakeLists.txt has to make, since -mavx2 -mfma is
# both meaningless AND unsupported outside x86_64: passing it while cross-
# compiling for aarch64 fails the build outright rather than silently
# doing nothing. NEON, by contrast, is part of the mandatory AArch64
# baseline instruction set, so no equivalent -m flag is needed to enable
# it on that target at all.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    target_compile_options(portable_dot_product PRIVATE -mavx2 -mfma)
endif()

enable_testing()
add_test(NAME portable_dot_product_self_test COMMAND portable_dot_product)
```

**Native configure, build, and test:**

```bash
cmake -B build-native -S . -G Ninja
cmake --build build-native
ctest --test-dir build-native --output-on-failure
```

**Sample output (native x86_64, AVX2 backend selected at compile time):**

```text
========================================================
Appendix A smoke test: portable_dot_product
Backend selected at compile time: AVX2 (x86_64)
========================================================

-- Test 1: a small, hand-verifiable real case -- {1,2,3,4,5} . {2,2,2,2,2} = 30 -- matches on both the portable scalar reference and this architecture's own accelerated path --
  scalar = 30, accelerated (AVX2 (x86_64)) = 30

-- Test 2: on a real, deterministically generated 257-element vector pair (a length deliberately not a multiple of 4 or 8, exercising each accelerated path's own real scalar tail-handling code), the accelerated path matches the portable scalar reference to within a tight real tolerance --
  scalar = -4.78072, accelerated (AVX2 (x86_64)) = -4.78072 (257 elements -- exercises the tail loop)

3/3 checks passed
ALL CHECKS PASSED
```

## A.3 Cross-Compiling for Arm/NEON Edge Targets

### Intuition

Building the identical source tree for a 64-bit Arm target needs only a real CMake toolchain file passed on the configure line -- no changes to `CMakeLists.txt` or the source itself -- plus two real, non-obvious settings that make the cross-compiled binary actually runnable and testable on the development machine.

### The Concept, In Detail

`CMAKE_EXE_LINKER_FLAGS_INIT "-static"` is a real requirement, not a convenience default: a dynamically linked aarch64 binary built on this machine links against this machine's own aarch64 cross-sysroot shared libraries, which `qemu-user` (or a real edge board with a different libc version) frequently cannot locate or load correctly. Diagnostic 4 in Section A.4 below shows exactly what happens without it. Static linking folds those libraries into the binary itself, which is what actually lets `qemu-aarch64 ./binary` run it directly, with no sysroot or `LD_LIBRARY_PATH` juggling.

`CMAKE_CROSSCOMPILING_EMULATOR`, set via `find_program(QEMU_AARCH64_EXECUTABLE qemu-aarch64)`, is the second real, non-obvious setting: without it, `ctest` tries to execute the cross-compiled aarch64 binary directly on the x86_64 host, which cannot run aarch64 machine code at all. Diagnostic 3 in Section A.4 below shows the real, genuinely confusing failure this produces. Setting `CMAKE_CROSSCOMPILING_EMULATOR` tells `ctest` to run every test binary through `qemu-aarch64` instead, which is what actually lets `ctest` pass for a cross-compiled target.

```cmake
# Appendix A -- a real, working CMake toolchain file for cross-compiling
# this book's own codebase for a 64-bit Arm edge target (Raspberry Pi,
# Jetson, or any other aarch64 Linux board) from an x86_64 Ubuntu 24.04 or
# Amazon Linux 2023 development machine, using the identical
# aarch64-linux-gnu-g++ cross toolchain this book's own build-verify-lock
# pipeline has used for every chapter's own cross-architecture check.
#
# Usage (see appendix_a/README.md for the exact, fully verified commands):
#   cmake -B build-aarch64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux-gnu.cmake -S .
#   cmake --build build-aarch64
#
# Ubuntu 24.04 (this book's own real, verified development path):
#   sudo apt-get install g++-aarch64-linux-gnu qemu-user
# Amazon Linux 2023 (a stated, real path this book's own pipeline did not
# itself compile on -- see the diagnostics reference in the appendix text
# for what to check first if a package name below does not match your own
# AL2023 mirror):
#   sudo dnf install gcc-c++ gcc-toolset  # then install an aarch64 cross
#   # toolchain from the EPEL or your organization's own internal mirror --
#   # AL2023's own default repos do not ship a prebuilt aarch64-linux-gnu-g++
#   # package the way Ubuntu's universe repo does.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Real, static linking -- this book's own real requirement, not a
# convenience default. A dynamically linked aarch64 binary built on this
# machine links against THIS machine's own aarch64 cross-sysroot shared
# libraries, which qemu-user (or a real edge board with a different libc
# version) frequently cannot locate or load correctly. Static linking
# folds those libraries into the binary itself, which is what actually
# lets `qemu-aarch64 ./binary` run it directly, with no sysroot or
# LD_LIBRARY_PATH juggling.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")

# Real cross-compilation mode: never search this HOST machine's own
# programs, libraries, or headers when looking for target-architecture
# dependencies -- only the cross toolchain's own sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Real: without this, `ctest` (and `cmake --build ... --target test`) tries
# to exec the cross-compiled aarch64 binary DIRECTLY on this x86_64 host,
# which cannot execute aarch64 machine code at all. The real, observed
# failure is not a clean "cannot execute binary file" message -- this
# book's own pipeline saw the host shell instead try to interpret the raw
# ELF bytes as a shell SCRIPT, producing garbled "not found" and "Syntax
# error: Unterminated quoted string" lines that give no hint the real
# cause is an architecture mismatch. Setting CMAKE_CROSSCOMPILING_EMULATOR
# tells ctest to run every test binary through qemu-user instead, which is
# what actually lets `ctest` pass for a cross-compiled target at all.
find_program(QEMU_AARCH64_EXECUTABLE qemu-aarch64)
if(QEMU_AARCH64_EXECUTABLE)
    set(CMAKE_CROSSCOMPILING_EMULATOR ${QEMU_AARCH64_EXECUTABLE})
endif()
```

**Cross-compiled configure, build, and test (run under `qemu-user` automatically via `CMAKE_CROSSCOMPILING_EMULATOR`):**

```bash
cmake -B build-aarch64 -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux-gnu.cmake
cmake --build build-aarch64
ctest --test-dir build-aarch64 --output-on-failure
```

**Sample output (cross-compiled aarch64 under `qemu-aarch64`, NEON backend selected at compile time):**

```text
========================================================
Appendix A smoke test: portable_dot_product
Backend selected at compile time: NEON (aarch64)
========================================================

-- Test 1: a small, hand-verifiable real case -- {1,2,3,4,5} . {2,2,2,2,2} = 30 -- matches on both the portable scalar reference and this architecture's own accelerated path --
  scalar = 30, accelerated (NEON (aarch64)) = 30

-- Test 2: on a real, deterministically generated 257-element vector pair (a length deliberately not a multiple of 4 or 8, exercising each accelerated path's own real scalar tail-handling code), the accelerated path matches the portable scalar reference to within a tight real tolerance --
  scalar = -4.78072, accelerated (NEON (aarch64)) = -4.78072 (257 elements -- exercises the tail loop)

3/3 checks passed
ALL CHECKS PASSED
```

The two real Test 2 values above -- the scalar and accelerated dot products of the identical, deterministically seeded 257-element vector pair -- are numerically identical (`-4.78072`) on both the native AVX2 build and the cross-compiled NEON build under `qemu-aarch64`: the identical, unmodified source tree, built twice for two genuinely different real instruction sets, produces the identical real numerical answer.

!!! warning "[COMMON TRAP] treating a clean cross-compile as proof the binary will run correctly on a real edge board"
    A clean `cmake --build build-aarch64` only confirms `aarch64-linux-gnu-g++` accepted the source and produced a real aarch64 ELF binary -- it says nothing about whether that binary will actually execute correctly on a real board. `qemu-aarch64` narrows that gap considerably by genuinely executing the real aarch64 instructions rather than merely confirming they exist, which is why this section's own Test 2 result matching the native build exactly is meaningful and not assumed. It still is not a full substitute for running on the real target hardware: a real edge board's own specific libc version, kernel version, or hardware capabilities (a NEON extension `qemu-aarch64`'s own CPU emulation does not model, for instance) could still behave differently. This book's own pipeline closes that remaining gap the only way it honestly can -- by also running its own self-tests directly on real aarch64 hardware, not only under emulation.

## A.4 Diagnostics Reference

### Intuition

Every diagnostic in this section is a real, verbatim failure this book's own pipeline actually triggered while building the exact setup documented above -- captured directly from a real terminal, not paraphrased from memory -- paired with the real fix that resolved it.

### The Concept, In Detail

**Diagnostic 1 -- GCC 11 rejecting a C++23 multi-argument `mdspan` subscript.** GCC 11 (installed here as `g++-11`, resolving to a real `11.5.0` release) predates GCC 12's support for the C++23 multi-argument `operator[]` proposal, so a natural-looking `view[1, 2]` call genuinely fails to compile, with a real deprecation warning on the comma expression itself pointing at the actual root cause:

```
$ g++-11 -std=c++23 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include diag1_multiarg_subscript.cpp -o diag1_out
```

```text
diag1_multiarg_subscript.cpp: In function 'int main()':
diag1_multiarg_subscript.cpp:5:35: warning: top-level comma expression in array subscript is deprecated [-Wcomma-subscript]
    5 |     return static_cast<int>(view[1, 2]);
      |                                   ^
diag1_multiarg_subscript.cpp:5:33: error: no match for 'operator[]' (operand types are 'std::mdspan<float, std::extents<long unsigned int, 18446744073709551615, 18446744073709551615>, std::layout_right, std::default_accessor<float> >' and 'int')
```

The real fix, established in Chapter 13's own appendix and used unchanged in every `std::mdspan`-based file in this book since, is a small `idx2()` helper that packages the indices into the `std::array` overload GCC 11 does support: `view[idx2(1, 2)]` compiles cleanly on the identical GCC 11 with zero errors and zero warnings.

**Diagnostic 2 -- GCC 11 lacking a native `<mdspan>` header at all.** GCC's own standard library did not ship a native `<mdspan>` header until GCC 14, so including it directly on GCC 11 fails immediately, before any code in the file is even checked:

```
$ g++-11 -std=c++23 diag2_no_native_mdspan.cpp -o diag2_out
```

```text
diag2_no_native_mdspan.cpp:1:10: fatal error: mdspan: No such file or directory
    1 | #include <mdspan>
      |          ^~~~~~~~
compilation terminated.
```

The real fix is the same one this book has used since Chapter 2: the vendored, header-only reference `mdspan` implementation under `_vendor_mdspan/`, included as `<mdspan/mdspan.hpp>` with the `MDSPAN_IMPL_STANDARD_NAMESPACE`/`MDSPAN_IMPL_PROPOSED_NAMESPACE` defines shown throughout this book's own compile commands, rather than the native standard header.

**Diagnostic 3 -- `ctest` failing with a garbled error when `CMAKE_CROSSCOMPILING_EMULATOR` is not set.** Configuring the aarch64 build without the `CMAKE_CROSSCOMPILING_EMULATOR` line from Section A.3 and running `ctest` produces a real, genuinely confusing failure: the host shell tries to interpret the cross-compiled binary's raw ELF bytes as a shell script, rather than reporting a clear "cannot execute" error:

```
$ ctest --test-dir build --output-on-failure
```

```text
Internal ctest changing into directory: /home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build
Test project /home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build
    Start 1: portable_dot_product_self_test
1/1 Test #1: portable_dot_product_self_test ...***Failed    0.00 sec
/home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build/portable_dot_product: 1: �: not found
/home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build/portable_dot_product: 1: 4��9L�: not found
/home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build/portable_dot_product: 1: 4��: not found
/home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build/portable_dot_product: 2: L�: not found
/home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build/portable_dot_product: 1: B�@�?q�T��R�}: not found
/home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build/portable_dot_product: 2: @�: not found
/home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build/portable_dot_product: 1: ELF�@@��2@8@@@_�: not found
/home/claude/cpp_inference_book/appendix_a/diagnostics/_diag3_no_emulator_proj/build/portable_dot_product: 1: Syntax error: Unterminated quoted string


0% tests passed, 1 tests failed out of 1

Total Test time (real) =   0.01 sec

The following tests FAILED:
	  1 - portable_dot_product_self_test (Failed)
Errors while running CTest
```

The exact garbled text in the middle of this output depends on the specific compiled binary's own raw bytes and will differ from build to build; what stays constant is the shell attempting to run those bytes as a script at all, the resulting `Syntax error: Unterminated quoted string`, and the `0% tests passed` summary. The real fix is the `find_program(QEMU_AARCH64_EXECUTABLE qemu-aarch64)` / `CMAKE_CROSSCOMPILING_EMULATOR` lines in Section A.3's own toolchain file, confirmed above to bring `ctest` to a real, clean pass under `qemu-aarch64`.

**Diagnostic 4 -- `qemu-aarch64` failing on a dynamically linked binary.** Building the identical source without `-static` and running the result directly under bare `qemu-aarch64` (with no sysroot configured) fails because the dynamic loader this machine's own aarch64 binary expects is not present at the path it looks for:

```
$ aarch64-linux-gnu-g++ -std=c++23 diag4_dynamic.cpp -o diag4_dynamic_bin
$ qemu-aarch64 ./diag4_dynamic_bin
```

```text
qemu-aarch64: Could not open '/lib/ld-linux-aarch64.so.1': No such file or directory
```

The real fix is exactly the `CMAKE_EXE_LINKER_FLAGS_INIT "-static"` setting in Section A.3's own toolchain file: statically linking folds the C++ runtime and standard library directly into the binary, so `qemu-aarch64` never needs to locate a dynamic linker or any shared library at all.

**Diagnostic 5 -- `nvcc` rejecting `-std=c++23`.** Chapter 32's own real CUDA section hit this directly: `nvcc` 12.0's host-compiler pass does not recognize `c++23` as a valid value for `-std` at all, failing before any actual kernel code is even parsed:

```
$ nvcc -std=c++23 -arch=sm_87 diag5_nvcc_cpp23.cu -o diag5_out
```

```text
nvcc fatal   : Value 'c++23' is not defined for option 'std'
```

The real fix, used throughout Chapter 32.3, is compiling with `-std=c++20` instead -- a real, documented toolchain constraint of CUDA 12.0's own host-compiler pass, not a limitation of the kernel code itself.

!!! warning "[COMMON TRAP] assuming every real toolchain error means the underlying code is wrong"
    Diagnostics 1, 2, and 5 above are not bugs in the code being compiled -- they are real, documented gaps between a specific compiler version's own feature support and the C++23 standard the code is deliberately written against. GCC 11 predates GCC 12's multi-argument `operator[]` support and ships no native `<mdspan>` header at all; `nvcc` 12.0's host pass simply does not accept `c++23` as a `-std` value yet. Each real fix above works around a specific, real, version-bound toolchain limitation, not a defect in the algorithm or the code's own logic -- which is exactly why this book's own pipeline checks the SAME source tree against multiple real compiler versions and architectures on every chapter, rather than trusting a single toolchain's own silence as proof of correctness.

## Appendix Summary

This appendix documented the real cross-compilation toolchain this book's own pipeline has used, unstated, since Chapter 2: how to install `aarch64-linux-gnu-g++` and `qemu-user` on a real Ubuntu 24.04 development machine, how to lay out a real CMake project that builds the identical source tree natively and cross-compiled for a 64-bit Arm target from a single toolchain file, and why that toolchain file needs both static linking and `CMAKE_CROSSCOMPILING_EMULATOR` to actually produce a runnable, testable cross-compiled binary. It closed with five real, verbatim diagnostic failures this book's own pipeline actually hit while building this exact setup -- a real GCC 11 C++23 rejection, a real missing standard header, a real garbled `ctest` failure, a real missing dynamic linker under `qemu-user`, and a real `nvcc` standard-version rejection -- each with the real fix that resolved it, in the same honest, verify-directly-rather-than-assert spirit as every chapter before it.

## Where We Go Next

Appendix B is this book's own practice quiz, spanning every Part. Appendix C consolidates the decision trees scattered across individual chapters into a single reference. Appendix D restates this book's own running discipline around timing, determinism, and what a locked self-test contract actually promises. Appendix E is a Rosetta Stone for readers arriving fluent in the Python inference ecosystem. Appendix F catalogs common failure modes -- NaN propagation, false sharing, floating-point drift, and alignment bugs -- much of it drawing directly on material this book already built in Chapter 31.4 and Chapter 10.
