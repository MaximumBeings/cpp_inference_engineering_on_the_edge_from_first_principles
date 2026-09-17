#!/usr/bin/env python3
"""Verify docs/part6/26-mathematical-foundations.md.

All 6 sections in this chapter depend on nothing but the C++ standard
library, so all 6 get this book's usual, unconditional compile/run/
determinism check.
"""
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part6/26-mathematical-foundations.md"

EXPECTED_H1 = "# Chapter 26: Mathematical Foundations for Kernel Authors: FLOPs, the Roofline, and the Hessian"
EXPECTED_H2 = [
    "## 26.1 The Dot Product as Inference's Atomic Unit, and the Real GEMV/GEMM Crossover",
    "## 26.2 Numerically Stable Softmax: The Log-Sum-Exp Fix",
    "## 26.3 RoPE's Rotation Math",
    "## 26.4 Quantization as Affine Algebra: Real Error Bounds and the Optimal Scale",
    "## 26.5 The Hessian's Role in GPTQ: Second-Order Error Compensation",
    "## 26.6 A Full Roofline Analysis of a Transformer Layer",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover.cpp",
    2: "02_numerically_stable_softmax_and_log_sum_exp.cpp",
    3: "03_rope_rotation_math.cpp",
    4: "04_quantization_as_affine_algebra.cpp",
    5: "05_hessian_role_in_gptq.cpp",
    6: "06_full_transformer_layer_roofline_analysis.cpp",
}
OUT_FILES = {
    1: "01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover.out.txt",
    2: "02_numerically_stable_softmax_and_log_sum_exp.out.txt",
    3: "03_rope_rotation_math.out.txt",
    4: "04_quantization_as_affine_algebra.out.txt",
    5: "05_hessian_role_in_gptq.out.txt",
    6: "06_full_transformer_layer_roofline_analysis.out.txt",
}

COMPILE_FLAGS = ["-std=c++23", "-Wall", "-Wextra", "-O2"]

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"OK:   {msg}")


def main() -> int:
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
        ok("H2 sequence matches (6 numbered sections + summary/self-check/next/solutions)")

    ALLOWED_NON_ASCII = {"—"}
    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
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

    if len(bash_blocks) != 6:
        fail(f"expected exactly 6 bash compile/run blocks, found {len(bash_blocks)}")
    else:
        ok("found exactly 6 bash compile/run blocks")
        for idx in range(1, 7):
            fname = FILES[idx]
            block = bash_blocks[idx - 1]
            if fname not in block:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")
            if "-std=c++23" not in block:
                fail(f"bash block {idx} does not compile with -std=c++23")

    if len(cpp_blocks) != 6:
        fail(f"expected exactly 6 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 6 cpp code blocks")
        for idx in range(1, 7):
            fname = FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    if len(text_blocks) != 6:
        fail(f"expected exactly 6 text output blocks (one self-test output per section), found {len(text_blocks)}")
    else:
        ok("found exactly 6 text output blocks")
        for idx in range(1, 7):
            fname = OUT_FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = text_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"output block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"output block {idx} matches locked {fname} exactly")

    tmp_bins = []

    for idx, fname in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch26_{idx:02d}_bin"

        cmd = ["g++"] + COMPILE_FLAGS + [str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded ({' '.join(COMPILE_FLAGS)})")

        locked_expected = (BASE / OUT_FILES[idx]).read_text().rstrip("\n")

        r2 = subprocess.run([str(binp)], capture_output=True, text=True)
        fresh = r2.stdout.rstrip("\n")
        if fresh != locked_expected:
            fail(f"fresh self-test rerun of {fname} does not match the locked self-test output")
        else:
            ok(f"fresh self-test rerun of {fname} matches locked self-test output exactly (deterministic)")
        if r2.returncode != 0:
            fail(f"fresh rerun of {fname} exited non-zero ({r2.returncode})")

        r3 = subprocess.run([str(binp)], capture_output=True, text=True)
        fresh2 = r3.stdout.rstrip("\n")
        if fresh2 != fresh:
            fail(f"second rerun of {fname} does not match the first rerun (nondeterministic output!)")
        else:
            ok(f"second rerun of {fname} matches the first rerun exactly (confirmed deterministic)")

        tmp_bins.append(binp)

    for b in tmp_bins:
        try:
            b.unlink()
        except OSError:
            # Best-effort cleanup only: some environments (e.g. a connected-folder mount with no
            # delete permission granted) refuse the unlink outright. A leftover scratch binary here
            # does not affect anything this script has already checked above.
            pass

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
