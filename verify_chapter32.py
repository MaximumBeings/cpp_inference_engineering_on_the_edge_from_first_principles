#!/usr/bin/env python3
"""Verify Chapter 32 sources, captured output and executable examples.

CUDA runs compare repeatability on the current machine and require the
suite to succeed. GPU and no-device transcripts legitimately differ.
Set CUDA_ARCH=sm_80 for A100 or CUDA_ARCH=sm_87 for Jetson Orin.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part6/32-flash-attention-and-cuda-kernels.md"

EXPECTED_H1 = "# Chapter 32: Flash Attention and CUDA Kernels: Taking the Engine to the GPU"
EXPECTED_H2 = [
    "## 32.1 The O(N^2) Memory Wall and the Online-Softmax Fix",
    "## 32.2 A std::mdspan-Based Flash Attention Implementation and Benchmark",
    "## 32.3 A CUDA Production Engine and Its Own Kernel-Validation Suite",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

CPP_FILES = {
    1: "01_memory_wall_and_online_softmax.cpp",
    2: "02_mdspan_flash_attention_implementation_and_benchmark.cpp",
}
CUDA_FILE = "03_cuda_kernel_and_validation_suite.cu"

OUT_FILES = {
    1: "01_memory_wall_and_online_softmax.out.txt",
    2: "02_mdspan_flash_attention_implementation_and_benchmark.out.txt",
    3: "03_cuda_kernel_and_validation_suite.out.txt",
}

MDSPAN_FLAGS = [
    "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std",
    "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
    f"-I{BASE / '_vendor_mdspan/include'}",
]
COMPILE_FLAGS = {
    1: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    2: ["-std=c++23", "-Wall", "-Wextra", "-O2"] + MDSPAN_FLAGS,
}

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
        ok("H2 sequence matches (3 numbered sections + summary/self-check/next/solutions)")

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
    cuda_blocks = [b for lang, b in blocks if lang == "cuda"]
    text_blocks = [b for lang, b in blocks if lang == "text"]
    bash_blocks = [b for lang, b in blocks if lang == "bash"]

    if len(bash_blocks) != 3:
        fail(f"expected exactly 3 compile/run blocks, found {len(bash_blocks)}")
    else:
        ok("found exactly 3 compile/run blocks")
        for idx, fname in CPP_FILES.items():
            block = bash_blocks[idx - 1]
            if fname not in block:
                fail(f"compile/run block {idx} does not reference source file {fname}")
            else:
                ok(f"compile/run block {idx} references {fname}")
            if "-std=c++23" not in block:
                fail(f"compile/run block {idx} does not compile with -std=c++23")
        if CUDA_FILE not in bash_blocks[2]:
            fail(f"compile/run block 3 does not reference source file {CUDA_FILE}")
        else:
            ok(f"compile/run block 3 references {CUDA_FILE}")
        if "nvcc" not in bash_blocks[2] or "-std=c++20" not in bash_blocks[2]:
            fail("compile/run block 3 does not show the real nvcc -std=c++20 compile command")
        else:
            ok("compile/run block 3 shows the real nvcc -std=c++20 compile command")

    if len(cpp_blocks) != 2:
        fail(f"expected exactly 2 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 2 cpp code blocks")
        for idx, fname in CPP_FILES.items():
            locked = (BASE / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    if len(cuda_blocks) != 1:
        fail(f"expected exactly 1 cuda code block, found {len(cuda_blocks)}")
    else:
        locked = (BASE / CUDA_FILE).read_text().rstrip("\n")
        got = cuda_blocks[0].rstrip("\n")
        if got != locked:
            fail(f"cuda code block does not exactly match locked file {CUDA_FILE}")
        else:
            ok(f"cuda code block matches locked {CUDA_FILE} exactly")

    if len(text_blocks) != 3:
        fail(f"expected exactly 3 text output blocks (one self-test output per section), found {len(text_blocks)}")
    else:
        ok("found exactly 3 text output blocks")
        for idx in range(1, 4):
            fname = OUT_FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = text_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"output block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"output block {idx} matches locked {fname} exactly")

    tmp_bins = []

    # Sections 1 and 2: this book's usual unconditional g++ compile/run/determinism check.
    for idx, fname in CPP_FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch28_{idx:02d}_bin"

        cmd = ["g++"] + COMPILE_FLAGS[idx] + [str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded")

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

    # Section 3: a real, documented, reduced-verification exception. Compiled and run with a real
    # nvcc ONLY where one is actually present (this book's own cloud sandbox); this book's own real
    # aarch64 hardware is an Apple Silicon Mac, which has never had a CUDA toolchain available at
    # all, so this leg is expected to be skipped there -- not a failure, a documented structural fact.
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        ok("nvcc not found on this machine -- expected (no CUDA toolchain on macOS/Apple Silicon); "
           "skipping compile-and-run check for Section 3's own .cu file. Locked code/output block "
           "content was already checked above against the markdown.")
    else:
        src = BASE / CUDA_FILE
        binp = BASE / "_verify_ch28_03_bin"
        architecture = os.environ.get("CUDA_ARCH", "native")
        cmd = ["nvcc", "-std=c++20", f"-arch={architecture}", str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"nvcc compile of {CUDA_FILE} failed:\n{r.stderr}")
        else:
            ok(f"nvcc compile of {CUDA_FILE} succeeded for {architecture}")
            r2 = subprocess.run([str(binp)], capture_output=True, text=True)
            r3 = subprocess.run([str(binp)], capture_output=True, text=True)
            if r2.returncode != 0 or r3.returncode != 0:
                fail(f"CUDA validation exited non-zero:\n{r2.stderr}\n{r2.stdout}")
            elif "ALL CHECKS PASSED" not in r2.stdout:
                fail("CUDA validation did not report successful checks")
            elif r2.stdout != r3.stdout:
                fail("CUDA validation output changed across two runs")
            else:
                ok("CUDA validation passed twice with identical output on this machine")
                ok("Device checks executed" if "device comparisons executed" in r2.stdout
                   else "No-device execution completed; device comparisons were explicitly skipped")
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
