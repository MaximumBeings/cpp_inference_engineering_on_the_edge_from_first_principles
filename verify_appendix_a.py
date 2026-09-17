#!/usr/bin/env python3
"""Verify docs/appendix/a-installation-and-setup.md.

Unlike a numbered chapter's own single self-contained .cpp file per section,
this appendix's own "Code and Verification" material is a real, multi-file
CMake project (appendix_a/) plus five real, verbatim diagnostic failures
this book's own pipeline actually triggered while building it. This script
checks both: that the built markdown's own code/output blocks exactly match
the locked files on disk, AND that the real project and real diagnostics
are still genuinely reproducible right now, on this machine, rather than
merely trusting a previously captured transcript.

Section A.3's cross-compiled build and Diagnostics 1/2/5 need tools
(aarch64-linux-gnu-g++, qemu-aarch64, g++-11, nvcc) that may not be present
on every machine this script runs on (in particular, this book's own real
aarch64 device has no aarch64-linux-gnu-g++, g++-11, or nvcc, and running
its own native g++ IS the point there). Each such check is skipped with a
clear, explicit message rather than failing when its tool is absent --
the same documented-exception pattern Chapter 32's own verify script used
for nvcc.
"""
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Python itself injects LC_CTYPE=C.UTF-8 into subprocess environments by default (PEP 538/540
# locale coercion) even when the parent shell has no locale set at all. That silently switches
# g++-11's own diagnostic quoting from plain ASCII quotes to Unicode curly quotes ( vs ),
# which would make this script's own freshly reproduced Diagnostic 1/2 output disagree with the
# locked, real ASCII-quoted transcript this book's own pipeline actually captured from a plain
# shell. Forcing LC_ALL=C for these specific real compiler invocations keeps the comparison
# honest and reproducible rather than accidentally locale-dependent.
_C_LOCALE_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C", "LANGUAGE": ""}

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/appendix/a-installation-and-setup.md"
APPA = BASE / "appendix_a"
DIAG = APPA / "diagnostics"

EXPECTED_H1 = "# Appendix A: Installation and Setup -- Cross-Compiling for the Edge"
EXPECTED_H2 = [
    "## A.1 Toolchain Setup on the Development Machine",
    "## A.2 A Working CMake Project Layout",
    "## A.3 Cross-Compiling for Arm/NEON Edge Targets",
    "## A.4 Diagnostics Reference",
    "## Appendix Summary",
    "## Where We Go Next",
]

CODE_FILES = {
    "cpp": [APPA / "src/portable_dot_product.cpp"],
    "cmake": [APPA / "CMakeLists.txt", APPA / "cmake/toolchain-aarch64-linux-gnu.cmake"],
}
OUT_FILES = [
    APPA / "portable_dot_product.native.out.txt",
    APPA / "portable_dot_product.aarch64.out.txt",
    DIAG / "diag1_excerpt.txt",
    DIAG / "diag2_no_native_mdspan.stderr.txt",
    DIAG / "diag3_ctest_no_emulator.stdout.txt",
    DIAG / "diag4_qemu_missing_linker.stdout.txt",
    DIAG / "diag5_nvcc_cpp23.stderr.txt",
]

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"OK:   {msg}")


def _lines_outside_code_fences(text):
    """Headers (and the non-ASCII scan) must ignore '#'-comment lines inside cmake/bash
    fenced code blocks, which also start with '# ' but are not markdown headers at all."""
    out = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return out


def check_markdown_structure():
    if not MD_PATH.exists():
        fail(f"built markdown does not exist: {MD_PATH}")
        return None
    text = MD_PATH.read_text()
    prose_lines = _lines_outside_code_fences(text)

    h1_lines = [l for l in prose_lines if l.startswith("# ")]
    if len(h1_lines) != 1:
        fail(f"expected exactly 1 H1, found {len(h1_lines)}: {h1_lines}")
    elif h1_lines[0] != EXPECTED_H1:
        fail(f"H1 mismatch: {h1_lines[0]!r} != {EXPECTED_H1!r}")
    else:
        ok("H1 matches")

    h2_lines = [l for l in prose_lines if l.startswith("## ")]
    if h2_lines != EXPECTED_H2:
        fail(f"H2 sequence mismatch:\n  got:      {h2_lines}\n  expected: {EXPECTED_H2}")
    else:
        ok("H2 sequence matches (A.1-A.4 + Summary + Where We Go Next; no self-check "
           "section, since Appendix B is this book's own dedicated quiz)")

    # The em-dash is this book's own stylistic choice throughout; U+FFFD (the replacement
    # character) is a real, expected artifact of Diagnostic 3's own genuinely garbled,
    # non-UTF-8 binary bytes being displayed as text -- both are allowed here, nothing else.
    ALLOWED_NON_ASCII = {"—", "�"}
    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
            bad_chars[ch] = bad_chars.get(ch, 0) + 1
    if bad_chars:
        fail(f"non-ASCII characters found (other than em-dash / U+FFFD): {bad_chars}")
    else:
        ok("no disallowed non-ASCII characters")

    return text


def check_locked_blocks(text):
    fence_re = re.compile(r"```(\w+)\n(.*?)\n```", re.DOTALL)
    blocks = fence_re.findall(text)

    for lang, paths in CODE_FILES.items():
        got_blocks = [b for l, b in blocks if l == lang]
        if len(got_blocks) != len(paths):
            fail(f"expected exactly {len(paths)} {lang} code block(s), found {len(got_blocks)}")
            continue
        ok(f"found exactly {len(paths)} {lang} code block(s)")
        for path, got in zip(paths, got_blocks):
            locked = path.read_text().rstrip("\n")
            if got.rstrip("\n") != locked:
                fail(f"code block does not exactly match locked file {path.name}")
            else:
                ok(f"code block matches locked {path.name} exactly")

    text_blocks = [b for l, b in blocks if l == "text"]
    if len(text_blocks) != len(OUT_FILES):
        fail(f"expected exactly {len(OUT_FILES)} text output blocks, found {len(text_blocks)}")
        return
    ok(f"found exactly {len(OUT_FILES)} text output blocks")
    for path, got in zip(OUT_FILES, text_blocks):
        locked = path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
        if got.rstrip("\n") != locked:
            fail(f"output block does not exactly match locked file {path.name}")
        else:
            ok(f"output block matches locked {path.name} exactly")


class Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run(cmd, cwd=None, env=None):
    # Decoded manually with errors="replace": Diagnostic 3's own real ctest output contains
    # genuinely non-UTF-8 bytes (the shell trying to interpret raw ELF bytes as a script), and
    # subprocess's own text=True mode raises UnicodeDecodeError on that real content outright.
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True)
    return Result(r.returncode,
                  r.stdout.decode("utf-8", errors="replace"),
                  r.stderr.decode("utf-8", errors="replace"))


def check_cmake_project():
    """Genuinely re-configure, build, and test the real CMake project fresh
    (native, then cross-compiled if the toolchain is present), checking the
    program's own stdout against the locked output twice for determinism,
    and checking ctest itself reports a real, clean pass."""
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        fail("cmake and/or ninja not found -- cannot verify the real CMake project at all")
        return

    # Native build. Built OUTSIDE the repo tree (a fresh tempfile.mkdtemp() directory) rather
    # than under appendix_a/ itself: on a connected-folder mount with no delete permission (a
    # real, recurring characteristic of this book's own device), CMake's own try_compile scratch
    # cleanup fails outright with "Operation not permitted" the moment any file under the mount
    # needs to be removed. A build directory outside the mount sidesteps that entirely -- the
    # source is still read directly from the real, committed appendix_a/ tree via -S.
    native_build = Path(tempfile.mkdtemp(prefix="verify_appendix_a_native_"))
    r = run(["cmake", "-B", str(native_build), "-S", str(APPA), "-G", "Ninja"])
    if r.returncode != 0:
        fail(f"native cmake configure failed:\n{r.stderr}")
    else:
        r2 = run(["cmake", "--build", str(native_build)])
        if r2.returncode != 0:
            fail(f"native cmake build failed:\n{r2.stderr}")
        else:
            ok("native CMake project configures and builds cleanly")
            binp = native_build / "portable_dot_product"
            # The "native" build compiles for WHATEVER architecture this machine actually is --
            # on this book's own real aarch64 device that is genuinely aarch64/NEON, not x86_64/
            # AVX2, so the right locked comparison file depends on the real, detected host
            # architecture rather than always being the x86_64-captured "native" transcript.
            host_is_aarch64 = platform.machine() in ("aarch64", "arm64")
            locked_name = "portable_dot_product.aarch64.out.txt" if host_is_aarch64 else "portable_dot_product.native.out.txt"
            locked = (APPA / locked_name).read_text().rstrip("\n")
            r3 = run([str(binp)])
            r4 = run([str(binp)])
            if r3.stdout.rstrip("\n") != locked:
                fail(f"fresh native binary rerun does not match locked {locked_name} "
                     f"(this machine's own real architecture is {platform.machine()})")
            else:
                ok(f"fresh native binary rerun matches locked {locked_name} exactly "
                   f"(this machine's own real architecture is {platform.machine()})")
            if r3.stdout != r4.stdout:
                fail("native binary output is not deterministic across two runs")
            else:
                ok("native binary output confirmed deterministic across two runs")

            rt = run(["ctest", "--test-dir", str(native_build), "--output-on-failure"])
            if "100% tests passed" not in rt.stdout:
                fail(f"native ctest did not report a clean 100% pass:\n{rt.stdout}")
            else:
                ok("native ctest reports a real, clean 100% pass")
    shutil.rmtree(native_build, ignore_errors=True)

    # Cross-compiled build -- only if the real toolchain is present on this machine.
    if shutil.which("aarch64-linux-gnu-g++") is None or shutil.which("qemu-aarch64") is None:
        ok("aarch64-linux-gnu-g++ and/or qemu-aarch64 not found on this machine -- expected on "
           "this book's own real aarch64 device (there, the NATIVE build above already IS the "
           "real-hardware check); skipping the cross-compiled build check here.")
        return

    cross_build = Path(tempfile.mkdtemp(prefix="verify_appendix_a_aarch64_"))
    toolchain = APPA / "cmake/toolchain-aarch64-linux-gnu.cmake"
    r = run(["cmake", "-B", str(cross_build), "-S", str(APPA), "-G", "Ninja",
             f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"])
    if r.returncode != 0:
        fail(f"cross-compiled cmake configure failed:\n{r.stderr}")
    else:
        r2 = run(["cmake", "--build", str(cross_build)])
        if r2.returncode != 0:
            fail(f"cross-compiled cmake build failed:\n{r2.stderr}")
        else:
            ok("cross-compiled CMake project configures and builds cleanly")
            binp = cross_build / "portable_dot_product"
            locked = (APPA / "portable_dot_product.aarch64.out.txt").read_text().rstrip("\n")
            r3 = run(["qemu-aarch64", str(binp)])
            r4 = run(["qemu-aarch64", str(binp)])
            if r3.stdout.rstrip("\n") != locked:
                fail("fresh cross-compiled binary rerun (under qemu-aarch64) does not match "
                     "locked aarch64 output")
            else:
                ok("fresh cross-compiled binary rerun (under qemu-aarch64) matches locked "
                   "output exactly")
            if r3.stdout != r4.stdout:
                fail("cross-compiled binary output is not deterministic across two runs")
            else:
                ok("cross-compiled binary output confirmed deterministic across two runs")

            rt = run(["ctest", "--test-dir", str(cross_build), "--output-on-failure"])
            if "100% tests passed" not in rt.stdout:
                fail(f"cross-compiled ctest did not report a clean 100% pass "
                     f"(CMAKE_CROSSCOMPILING_EMULATOR should have run it under qemu-aarch64):"
                     f"\n{rt.stdout}")
            else:
                ok("cross-compiled ctest reports a real, clean 100% pass under qemu-aarch64 "
                   "(CMAKE_CROSSCOMPILING_EMULATOR is genuinely working)")
    shutil.rmtree(cross_build, ignore_errors=True)


def check_diagnostics():
    """Re-trigger each of the 5 real diagnostics fresh and confirm the failure (or fix)
    still reproduces, rather than only trusting a previously captured transcript."""

    # Diagnostic 1: GCC 11 rejecting a multi-arg mdspan subscript, and the idx2() fix.
    if shutil.which("g++-11") is None:
        ok("g++-11 not found on this machine -- expected on this book's own real aarch64 "
           "device, which has no g++-11 installed (its own native GCC 11.4.0 is what every "
           "OTHER chapter's real-device check already exercises); skipping Diagnostic 1's "
           "fresh recompilation here.")
    else:
        mdflags = ["-DMDSPAN_IMPL_STANDARD_NAMESPACE=std",
                   "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
                   f"-I{BASE / '_vendor_mdspan/include'}"]
        r = run(["g++-11", "-std=c++23"] + mdflags +
                ["diag1_multiarg_subscript.cpp", "-o", "/tmp/_verify_diag1"], cwd=DIAG, env=_C_LOCALE_ENV)
        excerpt = (DIAG / "diag1_excerpt.txt").read_text()
        if r.returncode == 0:
            fail("Diagnostic 1's broken multi-arg subscript file unexpectedly compiled cleanly")
        elif excerpt not in r.stderr:
            fail(f"Diagnostic 1's fresh g++-11 stderr no longer contains the locked excerpt:\n{r.stderr}")
        else:
            ok("Diagnostic 1 reproduces fresh: g++-11 still rejects the multi-arg mdspan "
               "subscript with the locked, real excerpt text")
        r2 = run(["g++-11", "-std=c++23"] + mdflags +
                  ["diag1_multiarg_subscript_fixed.cpp", "-o", "/tmp/_verify_diag1_fixed"], cwd=DIAG, env=_C_LOCALE_ENV)
        if r2.returncode != 0:
            fail(f"Diagnostic 1's own idx2() fix no longer compiles cleanly on g++-11:\n{r2.stderr}")
        else:
            ok("Diagnostic 1's real idx2() fix confirmed to compile cleanly on g++-11")

    # Diagnostic 2: GCC 11 lacking a native <mdspan> header.
    if shutil.which("g++-11") is None:
        ok("g++-11 not found -- skipping Diagnostic 2's fresh recompilation (see Diagnostic 1 above).")
    else:
        r = run(["g++-11", "-std=c++23", "diag2_no_native_mdspan.cpp",
                  "-o", "/tmp/_verify_diag2"], cwd=DIAG, env=_C_LOCALE_ENV)
        locked = (DIAG / "diag2_no_native_mdspan.stderr.txt").read_text()
        if r.returncode == 0:
            fail("Diagnostic 2's native <mdspan> include unexpectedly compiled cleanly on g++-11")
        elif r.stderr.strip() != locked.strip():
            fail(f"Diagnostic 2's fresh g++-11 stderr no longer matches the locked text:\n{r.stderr}")
        else:
            ok("Diagnostic 2 reproduces fresh, exactly matching the locked real g++-11 output")

    # Diagnostic 3: ctest without CMAKE_CROSSCOMPILING_EMULATOR -- checked via stable substrings,
    # since the garbled bytes in the middle of the real output depend on the exact build path
    # and the binary's own raw bytes (both real, both genuinely unpredictable in general, and
    # explicitly called out as such in the appendix text itself).
    if shutil.which("aarch64-linux-gnu-g++") is None or shutil.which("cmake") is None or shutil.which("ninja") is None:
        ok("aarch64-linux-gnu-g++ and/or cmake/ninja not found -- skipping Diagnostic 3's fresh "
           "reproduction (expected on this book's own real aarch64 device).")
    elif platform.machine() in ("aarch64", "arm64"):
        ok("this machine's own real architecture is already aarch64 -- a binary built with "
           "aarch64-linux-gnu-g++ is NOT actually cross-architecture here, so it runs directly "
           "with no exec-format mismatch at all, and this diagnostic's own failure mode "
           "(the host shell trying to run aarch64 machine code as a script) genuinely cannot "
           "occur. This is exactly why this book's own real aarch64 device is where the "
           "identical scenario needs the emulator in the FIRST PLACE only when the host is "
           "x86_64; skipping Diagnostic 3's own reproduction here rather than reporting a false "
           "failure for a mismatch that cannot exist on this host.")
    else:
        # Same reasoning as the native/cross builds above: built outside the mounted repo tree
        # so CMake's own scratch-file cleanup never hits this connected folder's no-delete quirk.
        proj = Path(tempfile.mkdtemp(prefix="verify_diag3_"))
        (proj / "cmake").mkdir(parents=True)
        (proj / "src").mkdir(parents=True)
        shutil.copy(APPA / "src/portable_dot_product.cpp", proj / "src")
        shutil.copy(APPA / "CMakeLists.txt", proj)
        (proj / "cmake/toolchain-no-emulator.cmake").write_text(
            "set(CMAKE_SYSTEM_NAME Linux)\n"
            "set(CMAKE_SYSTEM_PROCESSOR aarch64)\n"
            "set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)\n"
            "set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)\n"
            'set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")\n'
            "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n"
            "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)\n"
            "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)\n"
            "set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)\n"
            "# Deliberately no CMAKE_CROSSCOMPILING_EMULATOR.\n"
        )
        r = run(["cmake", "-B", "build", "-S", ".", "-G", "Ninja",
                  "-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-no-emulator.cmake"], cwd=proj)
        r2 = run(["cmake", "--build", "build"], cwd=proj)
        rt = run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=proj)
        combined = rt.stdout + rt.stderr
        before = "1/1 Test #1: portable_dot_product_self_test ...***Failed"
        after = "Syntax error: Unterminated quoted string"
        summary = "0% tests passed, 1 tests failed out of 1"
        if r.returncode != 0 or r2.returncode != 0:
            fail(f"Diagnostic 3's own setup (configure/build) unexpectedly failed:\n{r.stderr}\n{r2.stderr}")
        elif before in combined and after in combined and summary in combined:
            ok("Diagnostic 3 reproduces fresh: ctest without CMAKE_CROSSCOMPILING_EMULATOR still "
               "fails with the same real, stable garbled-execution signature")
        else:
            fail(f"Diagnostic 3's fresh ctest output no longer contains the expected stable "
                 f"failure signature:\n{combined}")
        shutil.rmtree(proj, ignore_errors=True)

    # Diagnostic 4: qemu-aarch64 on a dynamically linked binary.
    if shutil.which("aarch64-linux-gnu-g++") is None or shutil.which("qemu-aarch64") is None:
        ok("aarch64-linux-gnu-g++ and/or qemu-aarch64 not found -- skipping Diagnostic 4's fresh "
           "reproduction (expected on this book's own real aarch64 device).")
    else:
        r = run(["aarch64-linux-gnu-g++", "-std=c++23", str(DIAG / "diag4_dynamic.cpp"),
                  "-o", "/tmp/_verify_diag4_bin"])
        if r.returncode != 0:
            fail(f"Diagnostic 4's own dynamic-link compile unexpectedly failed:\n{r.stderr}")
        else:
            r2 = run(["qemu-aarch64", "/tmp/_verify_diag4_bin"])
            locked = (DIAG / "diag4_qemu_missing_linker.stdout.txt").read_text()
            combined = r2.stdout + r2.stderr
            if r2.returncode == 0:
                fail("Diagnostic 4's dynamically linked binary unexpectedly ran successfully under "
                     "bare qemu-aarch64")
            elif combined.strip() != locked.strip():
                fail(f"Diagnostic 4's fresh qemu-aarch64 output no longer matches the locked text:\n{combined}")
            else:
                ok("Diagnostic 4 reproduces fresh, exactly matching the locked real qemu-aarch64 error")

    # Diagnostic 5: nvcc rejecting -std=c++23.
    if shutil.which("nvcc") is None:
        ok("nvcc not found on this machine -- expected (no CUDA toolchain on macOS/Apple Silicon, "
           "the same documented exception Chapter 32's own verify script uses); skipping "
           "Diagnostic 5's fresh reproduction.")
    else:
        r = run(["nvcc", "-std=c++23", "-arch=sm_87", "diag5_nvcc_cpp23.cu",
                  "-o", "/tmp/_verify_diag5"], cwd=DIAG, env=_C_LOCALE_ENV)
        locked = (DIAG / "diag5_nvcc_cpp23.stderr.txt").read_text()
        if r.returncode == 0:
            fail("Diagnostic 5's -std=c++23 nvcc invocation unexpectedly succeeded")
        elif r.stderr.strip() != locked.strip():
            fail(f"Diagnostic 5's fresh nvcc stderr no longer matches the locked text:\n{r.stderr}")
        else:
            ok("Diagnostic 5 reproduces fresh, exactly matching the locked real nvcc error")


def main() -> int:
    text = check_markdown_structure()
    if text is not None:
        check_locked_blocks(text)
    check_cmake_project()
    check_diagnostics()

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
