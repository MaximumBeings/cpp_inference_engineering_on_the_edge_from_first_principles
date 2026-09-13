#!/usr/bin/env python3
"""Verify docs/part2/09-simd-vectorization-avx2-neon.md.

Chapter 9 is the first chapter whose files are not all buildable on one
architecture: Section 9.2 (AVX2) and Section 9.4 (-mavx2 -mfma) require an
x86_64 host; Section 9.3 (NEON) requires an aarch64 target, reached here by
cross-compiling with aarch64-linux-gnu-g++ and running under qemu-aarch64
when this script runs on an x86_64 host (the book's canonical build
environment), or natively when it happens to run on an aarch64 host.
Section 9.1 is fully portable. Section 9.5 compiles and passes its own
internal checks on either architecture, but its LOCKED, byte-exact output
was captured on x86_64 with AVX2 actually selected by its runtime dispatch;
on a host without AVX2 the same source correctly selects the scalar
fallback instead and produces a shorter, still-internally-passing, but
different transcript -- so this script only requires a byte-exact match
against the locked output on an x86_64 host, and otherwise only requires
that the freshly built binary itself reports ALL PASS.
"""
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part2/09-simd-vectorization-avx2-neon.md"

EXPECTED_H1 = "# Chapter 9: SIMD Vectorization -- AVX2 on x86, NEON on Arm"
EXPECTED_H2 = [
    "## 9.1 The SIMD Register Model, Derived From Register Width",
    "## 9.2 AVX2: Vectorized Quantized Dot Products on x86",
    "## 9.3 Arm NEON: The Same Kernel, a Different Register",
    "## 9.4 Auto-Vectorization and the Restrict Promise",
    "## 9.5 Runtime CPU Feature Detection and Dispatch",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

# idx -> (filename, arch requirement)
#   "any"      -- portable, compiles and must match locked output on any host
#   "x86_64"   -- requires -mavx2 -mfma; native-only; skipped (not failed) off x86_64
#   "aarch64"  -- requires arm_neon.h; cross-compiled + qemu on non-aarch64 hosts
#   "avx2_dispatch" -- portable source, but locked output assumes AVX2 was
#                      selected; byte-exact check only enforced on x86_64
FILES = {
    1: ("01_simd_register_model.cpp", "any"),
    2: ("02_avx2_quantized_dot_product.cpp", "x86_64"),
    3: ("03_neon_quantized_dot_product.cpp", "aarch64"),
    4: ("04_restrict_and_autovectorization.cpp", "x86_64"),
    5: ("05_runtime_dispatch.cpp", "avx2_dispatch"),
}
OUT_FILES = {
    1: "01_simd_register_model.out.txt",
    2: "02_avx2_quantized_dot_product.out.txt",
    3: "03_neon_quantized_dot_product.out.txt",
    4: "04_restrict_and_autovectorization.out.txt",
    5: "05_runtime_dispatch.out.txt",
}

HOST_ARCH = platform.machine()  # e.g. "x86_64" or "aarch64"

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"OK:   {msg}")


def skip(msg):
    print(f"SKIP: {msg}")


def compile_and_run(idx: int, fname: str, arch_req: str):
    """Returns (fresh_stdout_or_None, returncode_or_None, skipped: bool)."""
    src = BASE / fname
    binp = BASE / f"_verify_ch9_{idx:02d}_bin"

    if arch_req == "any":
        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2", str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            return None, None, False
        ok(f"compile of {fname} succeeded (g++ -std=c++23, portable)")
        r2 = subprocess.run([str(binp)], capture_output=True, text=True)
        binp.unlink(missing_ok=True)
        return r2.stdout.rstrip("\n"), r2.returncode, False

    if arch_req == "x86_64":
        if HOST_ARCH != "x86_64":
            skip(f"{fname} requires an x86_64 host (AVX2 intrinsics/-mavx2); this host is {HOST_ARCH}")
            return None, None, True
        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra"]
        cmd += ["-O3"] if fname.startswith("04_") else ["-O2"]
        cmd += ["-mavx2", "-mfma", str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            return None, None, False
        ok(f"compile of {fname} succeeded (g++ -std=c++23 -mavx2 -mfma, native x86_64)")
        r2 = subprocess.run([str(binp)], capture_output=True, text=True)
        binp.unlink(missing_ok=True)
        return r2.stdout.rstrip("\n"), r2.returncode, False

    if arch_req == "aarch64":
        if HOST_ARCH == "aarch64":
            cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2", str(src), "-o", str(binp)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                fail(f"compile of {fname} failed:\n{r.stderr}")
                return None, None, False
            ok(f"compile of {fname} succeeded (g++ -std=c++23, native aarch64)")
            r2 = subprocess.run([str(binp)], capture_output=True, text=True)
            binp.unlink(missing_ok=True)
            return r2.stdout.rstrip("\n"), r2.returncode, False
        else:
            cross_cc = shutil.which("aarch64-linux-gnu-g++")
            qemu = shutil.which("qemu-aarch64")
            if not cross_cc or not qemu:
                skip(f"{fname} requires aarch64-linux-gnu-g++ and qemu-aarch64 (cross toolchain); not found on this host")
                return None, None, True
            cmd = [cross_cc, "-std=c++23", "-Wall", "-Wextra", "-O2", str(src), "-o", str(binp)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                fail(f"cross-compile of {fname} failed:\n{r.stderr}")
                return None, None, False
            ok(f"cross-compile of {fname} succeeded (aarch64-linux-gnu-g++ -std=c++23)")
            qemu_cmd = [qemu, "-L", "/usr/aarch64-linux-gnu", str(binp)]
            r2 = subprocess.run(qemu_cmd, capture_output=True, text=True)
            binp.unlink(missing_ok=True)
            return r2.stdout.rstrip("\n"), r2.returncode, False

    if arch_req == "avx2_dispatch":
        if HOST_ARCH == "x86_64":
            cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O3", "-mavx2", "-mfma", str(src), "-o", str(binp)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                fail(f"compile of {fname} failed:\n{r.stderr}")
                return None, None, False
            ok(f"compile of {fname} succeeded (g++ -std=c++23 -mavx2 -mfma, native x86_64)")
            r2 = subprocess.run([str(binp)], capture_output=True, text=True)
            binp.unlink(missing_ok=True)
            return r2.stdout.rstrip("\n"), r2.returncode, False
        else:
            cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O3", str(src), "-o", str(binp)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                fail(f"compile of {fname} failed:\n{r.stderr}")
                return None, None, False
            ok(f"compile of {fname} succeeded (g++ -std=c++23, portable path -- this host lacks AVX2 dispatch)")
            r2 = subprocess.run([str(binp)], capture_output=True, text=True)
            binp.unlink(missing_ok=True)
            return r2.stdout.rstrip("\n"), r2.returncode, False

    fail(f"unknown arch requirement {arch_req!r} for {fname}")
    return None, None, False


def main() -> int:
    print(f"Host architecture: {HOST_ARCH}\n")

    if not MD_PATH.exists():
        fail(f"built markdown does not exist: {MD_PATH}")
        return report()

    text = MD_PATH.read_text()
    lines = text.splitlines()

    h1_lines = [l for l in lines if l.startswith("# ")]
    if len(h1_lines) != 1:
        fail(f"expected exactly 1 H1, found {len(h1_lines)}: {h1_lines}")
    elif h1_lines[0] != EXPECTED_H1:
        fail(f"H1 mismatch: {h1_lines[0]!r} != {EXPECTED_H1!r}")
    else:
        ok("H1 matches")

    h2_lines = [l for l in lines if l.startswith("## ")]
    if h2_lines != EXPECTED_H2:
        fail(f"H2 sequence mismatch:\n  got:      {h2_lines}\n  expected: {EXPECTED_H2}")
    else:
        ok("H2 sequence matches")

    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch != "—":
            bad_chars[ch] = bad_chars.get(ch, 0) + 1
    if bad_chars:
        fail(f"non-ASCII characters found (other than em-dash): {bad_chars}")
    else:
        ok("no disallowed non-ASCII characters")

    fence_re = re.compile(r"```(\w+)\n(.*?)\n```", re.DOTALL)
    blocks = fence_re.findall(text)
    cpp_blocks = [b for lang, b in blocks if lang == "cpp"]
    text_blocks = [b for lang, b in blocks if lang == "text"]
    bash_blocks = [b for lang, b in blocks if lang == "bash"]

    if len(bash_blocks) != 5:
        fail(f"expected exactly 5 bash compile/run blocks, found {len(bash_blocks)}")
    else:
        ok("found exactly 5 bash compile/run blocks")
        for idx in range(1, 6):
            fname = FILES[idx][0]
            if fname not in bash_blocks[idx - 1]:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")

    if len(cpp_blocks) != 5:
        fail(f"expected exactly 5 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 5 cpp code blocks")
        for idx in range(1, 6):
            fname = FILES[idx][0]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    if len(text_blocks) != 5:
        fail(f"expected exactly 5 text output blocks, found {len(text_blocks)}")
    else:
        ok("found exactly 5 text output blocks")
        for idx in range(1, 6):
            fname = OUT_FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = text_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"output block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"output block {idx} matches locked {fname} exactly")

    for idx, (fname, arch_req) in FILES.items():
        fresh, returncode, skipped = compile_and_run(idx, fname, arch_req)
        if skipped:
            continue
        if fresh is None:
            continue  # already reported as a failure by compile_and_run

        locked = (BASE / OUT_FILES[idx]).read_text().rstrip("\n")

        if arch_req == "avx2_dispatch" and HOST_ARCH != "x86_64":
            # Off x86_64, this source correctly takes the scalar-fallback
            # path and produces a different, shorter transcript than the
            # x86_64-canonical locked output -- so we only require that
            # THIS run's own internal checks passed, not a byte match.
            if "ALL PASS" in fresh:
                ok(f"fresh rerun of {fname} on {HOST_ARCH} reports ALL PASS (scalar-fallback path; "
                   f"locked output is the x86_64/AVX2-dispatch transcript, not compared here)")
            else:
                fail(f"fresh rerun of {fname} on {HOST_ARCH} did not report ALL PASS")
            if returncode != 0:
                fail(f"fresh rerun of {fname} exited non-zero ({returncode})")
            continue

        if fresh != locked:
            fail(f"fresh rerun of {fname} does not match locked output")
        else:
            ok(f"fresh rerun of {fname} matches locked output exactly (deterministic)")
        if returncode != 0:
            fail(f"fresh rerun of {fname} exited non-zero ({returncode})")

    sc_start = text.find("## Self-Check Questions")
    sc_end = text.find("## Where We Go Next")
    ws_start = text.find("## Worked Solutions")
    sc_section = text[sc_start:sc_end]
    ws_section = text[ws_start:]
    sc_count = len(re.findall(r"^\d+\.\s", sc_section, re.MULTILINE))
    ws_count = len(re.findall(r"^\*\*\d+\.\*\*", ws_section, re.MULTILINE))
    if sc_count == 0 or sc_count != ws_count:
        fail(f"self-check question count ({sc_count}) != worked solution count ({ws_count})")
    else:
        ok(f"{sc_count} self-check questions, {ws_count} worked solutions -- counts match")

    return report()


def report():
    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
