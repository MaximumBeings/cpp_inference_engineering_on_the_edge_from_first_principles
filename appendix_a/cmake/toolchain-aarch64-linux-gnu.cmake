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
