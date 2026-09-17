# Appendix A demo project

This directory is the real, working CMake project Appendix A's own text
walks through. Every command below was actually run to produce the output
locked into `portable_dot_product.native.out.txt` and
`portable_dot_product.aarch64.out.txt`, and to produce the exact,
real diagnostic error text quoted in the appendix's own "Diagnostics
Reference" section.

## Native build (x86_64 development machine)

```bash
cmake -B build-native -S . -G Ninja
cmake --build build-native
ctest --test-dir build-native --output-on-failure
```

## Cross-compiled build (aarch64 edge target, run under qemu-user)

```bash
cmake -B build-aarch64 -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux-gnu.cmake
cmake --build build-aarch64
ctest --test-dir build-aarch64 --output-on-failure
```

`ctest` runs the cross-compiled binary automatically through `qemu-aarch64`
because the toolchain file sets `CMAKE_CROSSCOMPILING_EMULATOR` -- see the
comment in `cmake/toolchain-aarch64-linux-gnu.cmake` for the real, garbled
failure this project saw before that line was added.
