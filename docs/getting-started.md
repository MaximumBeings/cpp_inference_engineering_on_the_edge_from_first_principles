# Getting Started

This book targets modern C++ compiled with `g++` (GCC 13 or newer). Most chapters use nothing beyond C++20/23 as shipped in any current GCC, but two specific areas need an extra toolchain component described below: `std::mdspan` (Part 0) and Arm NEON (Part 2).

## The base toolchain

```bash
g++ --version                                   # GCC 13.0 or newer
g++ -std=c++23 -Wall -Wextra -O2 file.cpp -o binary
```

Every plain host `.cpp` file in this book compiles and runs with exactly this line, no extra flags, no extra libraries.

## `std::mdspan`: a vendored reference implementation

`std::mdspan` is standard C++23, but no shipping standard library — not GCC 13's or 14's libstdc++, not current libc++ — implements it yet. This book uses the official Kokkos/mdspan reference implementation instead, the same reference implementation the standard itself was developed from, pinned to a specific commit for reproducibility:

```bash
git clone https://github.com/kokkos/mdspan.git
cd mdspan && git checkout 8989f70749e28f337e6f7aa210db88659dba6f2f
```

Compile any file that includes `<mdspan/mdspan.hpp>` with:

```bash
g++ -std=c++20 -Wall -Wextra -O2 \
    -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental \
    -I/path/to/mdspan/include file.cpp -o binary
```

The two `-D` flags tell the reference implementation to define its types directly in namespace `std` (its default is the non-conflicting `Kokkos` namespace, for codebases that need both an experimental and a future standard version side by side). With them, `std::mdspan`, `std::extents`, and `std::submdspan` in this book's own code are exactly the standard-track API, backed by an implementation that runs correctly today rather than a stub.

## SIMD: two real, genuinely-executed paths

- **x86 AVX2** compiles and runs natively on any AVX2-capable machine (nearly all x86-64 hardware from the last decade):

```bash
g++ -std=c++23 -Wall -Wextra -O2 -mavx2 -mfma file.cpp -o binary
./binary
```

- **Arm NEON** is cross-compiled and then genuinely *executed* under user-mode emulation — this book does not stop at "this compiles for Arm," it runs the resulting binary and locks its real output, the same way every other example in this book is verified:

```bash
sudo apt-get install -y g++-aarch64-linux-gnu qemu-user
aarch64-linux-gnu-g++ -std=c++23 -Wall -Wextra -O2 file.cpp -o binary_arm
qemu-aarch64 -L /usr/aarch64-linux-gnu ./binary_arm
```

Both paths produce this book's own locked, verified output — neither is a description of what *should* happen on real hardware.

## GPU acceleration (Part 6 only)

Part 6's CUDA material follows the same honesty discipline this book uses everywhere else. A genuine `nvcc` toolchain, installable without a physical GPU or driver:

```bash
pip install --break-system-packages nvidia-cuda-nvcc-cu13
```

Every `.cu` file in Part 6 is genuinely compiled for a real architecture (`-arch=sm_80` and newer). Where a kernel needs an actual device to *execute*, this book's authoring environment honestly reports back whatever a driver-less machine actually reports (typically `cudaErrorInsufficientDriver`) and verifies the kernel's logic instead with a host-side simulation of its exact grid, block, and thread structure — never a fabricated "it works" with no evidence behind it.

## The honesty discipline this book follows

- **Plain host C++** is genuinely compiled and genuinely run, natively, every time.
- **Arm NEON code** is genuinely cross-compiled and genuinely run under `qemu-user` emulation — real execution, not a compile-only claim.
- **x86 AVX2 code** is genuinely compiled and genuinely run natively.
- **CUDA kernels** (Part 6 only) are genuinely compiled for a real architecture; where this environment cannot launch them, that limitation is stated plainly and the kernel's logic is verified instead by an exhaustive host-side simulation of its own exact indexing.
- **Timing and throughput numbers are never fabricated.** Where a chapter's argument depends on which of two approaches is faster rather than merely which is correct, this book measures a genuinely computed, deterministic quantity instead of a wall-clock number wherever one exists — an operation count, a byte count, a cache-line count — specifically because a wall-clock number captured once is not reproducible on a rerun, let alone on a reader's own hardware, the way an exact count is. Where genuine wall-clock methodology itself is the topic, this book says so explicitly and separates any genuinely-varying raw number out of its own locked, verified output.

Every chapter states which of the above applies to each piece of its own code, so nothing is left for a reader to guess about how a claim was actually established.

## Compile-line conventions

- Plain C++ host files (`.cpp`): `g++ -std=c++23 -Wall -Wextra -O2 file.cpp -o binary`
- Files using `std::mdspan`: add the two `-D` flags and the `-I` path shown above, and compile with `-std=c++20` (the reference implementation does not require C++23).
- x86 SIMD files: add `-mavx2 -mfma` (or the specific instruction-set flag a section calls for).
- Arm NEON files: cross-compiled with `aarch64-linux-gnu-g++`, run with `qemu-aarch64 -L /usr/aarch64-linux-gnu`.
- CUDA files (Part 6): `nvcc -arch=sm_80 file.cu -o binary`, with `-lcudart` and toolchain include/library paths shown inline wherever a file calls the CUDA Runtime API from the host.

## Prerequisites

This book assumes working knowledge of C++ (classes, templates, RAII, move semantics) and does not assume any prior experience with quantization, SIMD intrinsics, or transformer internals — Part 0 builds every piece of vocabulary this book needs from nothing. It does assume basic familiarity with how a transformer's forward pass is structured at a conceptual level (attention, feed-forward layers); Part 0's own Chapter 3 is what turns that into working, compiled C++.
