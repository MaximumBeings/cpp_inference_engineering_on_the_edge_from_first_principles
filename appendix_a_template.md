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

@@CODE_SRC@@

`CMakeLists.txt` builds this identical source file into a single `portable_dot_product` target, applying `-mavx2 -mfma` only when `CMAKE_SYSTEM_PROCESSOR` actually matches `x86_64`/`AMD64` -- a real, necessary condition, since passing `-mavx2` while cross-compiling for aarch64 fails the build outright rather than silently doing nothing, and NEON needs no equivalent flag at all, since it is part of the mandatory AArch64 baseline instruction set.

@@CODE_CMAKE@@

**Native configure, build, and test:**

```bash
cmake -B build-native -S . -G Ninja
cmake --build build-native
ctest --test-dir build-native --output-on-failure
```

**Sample output (native x86_64, AVX2 backend selected at compile time):**

@@OUT_NATIVE@@

## A.3 Cross-Compiling for Arm/NEON Edge Targets

### Intuition

Building the identical source tree for a 64-bit Arm target needs only a real CMake toolchain file passed on the configure line -- no changes to `CMakeLists.txt` or the source itself -- plus two real, non-obvious settings that make the cross-compiled binary actually runnable and testable on the development machine.

### The Concept, In Detail

`CMAKE_EXE_LINKER_FLAGS_INIT "-static"` is a real requirement, not a convenience default: a dynamically linked aarch64 binary built on this machine links against this machine's own aarch64 cross-sysroot shared libraries, which `qemu-user` (or a real edge board with a different libc version) frequently cannot locate or load correctly. Diagnostic 4 in Section A.4 below shows exactly what happens without it. Static linking folds those libraries into the binary itself, which is what actually lets `qemu-aarch64 ./binary` run it directly, with no sysroot or `LD_LIBRARY_PATH` juggling.

`CMAKE_CROSSCOMPILING_EMULATOR`, set via `find_program(QEMU_AARCH64_EXECUTABLE qemu-aarch64)`, is the second real, non-obvious setting: without it, `ctest` tries to execute the cross-compiled aarch64 binary directly on the x86_64 host, which cannot run aarch64 machine code at all. Diagnostic 3 in Section A.4 below shows the real, genuinely confusing failure this produces. Setting `CMAKE_CROSSCOMPILING_EMULATOR` tells `ctest` to run every test binary through `qemu-aarch64` instead, which is what actually lets `ctest` pass for a cross-compiled target.

@@CODE_TOOLCHAIN@@

**Cross-compiled configure, build, and test (run under `qemu-user` automatically via `CMAKE_CROSSCOMPILING_EMULATOR`):**

```bash
cmake -B build-aarch64 -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux-gnu.cmake
cmake --build build-aarch64
ctest --test-dir build-aarch64 --output-on-failure
```

**Sample output (cross-compiled aarch64 under `qemu-aarch64`, NEON backend selected at compile time):**

@@OUT_AARCH64@@

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

@@DIAG1@@

The real fix, established in Chapter 13's own appendix and used unchanged in every `std::mdspan`-based file in this book since, is a small `idx2()` helper that packages the indices into the `std::array` overload GCC 11 does support: `view[idx2(1, 2)]` compiles cleanly on the identical GCC 11 with zero errors and zero warnings.

**Diagnostic 2 -- GCC 11 lacking a native `<mdspan>` header at all.** GCC's own standard library did not ship a native `<mdspan>` header until GCC 14, so including it directly on GCC 11 fails immediately, before any code in the file is even checked:

```
$ g++-11 -std=c++23 diag2_no_native_mdspan.cpp -o diag2_out
```

@@DIAG2@@

The real fix is the same one this book has used since Chapter 2: the vendored, header-only reference `mdspan` implementation under `_vendor_mdspan/`, included as `<mdspan/mdspan.hpp>` with the `MDSPAN_IMPL_STANDARD_NAMESPACE`/`MDSPAN_IMPL_PROPOSED_NAMESPACE` defines shown throughout this book's own compile commands, rather than the native standard header.

**Diagnostic 3 -- `ctest` failing with a garbled error when `CMAKE_CROSSCOMPILING_EMULATOR` is not set.** Configuring the aarch64 build without the `CMAKE_CROSSCOMPILING_EMULATOR` line from Section A.3 and running `ctest` produces a real, genuinely confusing failure: the host shell tries to interpret the cross-compiled binary's raw ELF bytes as a shell script, rather than reporting a clear "cannot execute" error:

```
$ ctest --test-dir build --output-on-failure
```

@@DIAG3@@

The exact garbled text in the middle of this output depends on the specific compiled binary's own raw bytes and will differ from build to build; what stays constant is the shell attempting to run those bytes as a script at all, the resulting `Syntax error: Unterminated quoted string`, and the `0% tests passed` summary. The real fix is the `find_program(QEMU_AARCH64_EXECUTABLE qemu-aarch64)` / `CMAKE_CROSSCOMPILING_EMULATOR` lines in Section A.3's own toolchain file, confirmed above to bring `ctest` to a real, clean pass under `qemu-aarch64`.

**Diagnostic 4 -- `qemu-aarch64` failing on a dynamically linked binary.** Building the identical source without `-static` and running the result directly under bare `qemu-aarch64` (with no sysroot configured) fails because the dynamic loader this machine's own aarch64 binary expects is not present at the path it looks for:

```
$ aarch64-linux-gnu-g++ -std=c++23 diag4_dynamic.cpp -o diag4_dynamic_bin
$ qemu-aarch64 ./diag4_dynamic_bin
```

@@DIAG4@@

The real fix is exactly the `CMAKE_EXE_LINKER_FLAGS_INIT "-static"` setting in Section A.3's own toolchain file: statically linking folds the C++ runtime and standard library directly into the binary, so `qemu-aarch64` never needs to locate a dynamic linker or any shared library at all.

**Diagnostic 5 -- `nvcc` rejecting `-std=c++23`.** Chapter 28's own real CUDA section hit this directly: `nvcc` 12.0's host-compiler pass does not recognize `c++23` as a valid value for `-std` at all, failing before any actual kernel code is even parsed:

```
$ nvcc -std=c++23 -arch=sm_87 diag5_nvcc_cpp23.cu -o diag5_out
```

@@DIAG5@@

The real fix, used throughout Chapter 28.3, is compiling with `-std=c++20` instead -- a real, documented toolchain constraint of CUDA 12.0's own host-compiler pass, not a limitation of the kernel code itself.

!!! warning "[COMMON TRAP] assuming every real toolchain error means the underlying code is wrong"
    Diagnostics 1, 2, and 5 above are not bugs in the code being compiled -- they are real, documented gaps between a specific compiler version's own feature support and the C++23 standard the code is deliberately written against. GCC 11 predates GCC 12's multi-argument `operator[]` support and ships no native `<mdspan>` header at all; `nvcc` 12.0's host pass simply does not accept `c++23` as a `-std` value yet. Each real fix above works around a specific, real, version-bound toolchain limitation, not a defect in the algorithm or the code's own logic -- which is exactly why this book's own pipeline checks the SAME source tree against multiple real compiler versions and architectures on every chapter, rather than trusting a single toolchain's own silence as proof of correctness.

## Appendix Summary

This appendix documented the real cross-compilation toolchain this book's own pipeline has used, unstated, since Chapter 2: how to install `aarch64-linux-gnu-g++` and `qemu-user` on a real Ubuntu 24.04 development machine, how to lay out a real CMake project that builds the identical source tree natively and cross-compiled for a 64-bit Arm target from a single toolchain file, and why that toolchain file needs both static linking and `CMAKE_CROSSCOMPILING_EMULATOR` to actually produce a runnable, testable cross-compiled binary. It closed with five real, verbatim diagnostic failures this book's own pipeline actually hit while building this exact setup -- a real GCC 11 C++23 rejection, a real missing standard header, a real garbled `ctest` failure, a real missing dynamic linker under `qemu-user`, and a real `nvcc` standard-version rejection -- each with the real fix that resolved it, in the same honest, verify-directly-rather-than-assert spirit as every chapter before it.

## Where We Go Next

Appendix B is this book's own practice quiz, spanning every Part. Appendix C consolidates the decision trees scattered across individual chapters into a single reference. Appendix D restates this book's own running discipline around timing, determinism, and what a locked self-test contract actually promises. Appendix E is a Rosetta Stone for readers arriving fluent in the Python inference ecosystem. Appendix F catalogs common failure modes -- NaN propagation, false sharing, floating-point drift, and alignment bugs -- much of it drawing directly on material this book already built in Chapter 27.4 and Chapter 10.
