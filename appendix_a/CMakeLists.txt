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
