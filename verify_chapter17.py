#!/usr/bin/env python3
"""Verify docs/part4/17-benchmarking.md.

This chapter's honest-exception shape follows Chapter 16's precedent
directly. Sections 17.1 and 17.2 have NO real-file mode at all -- each is
pure synthetic self-test, verified exactly like every non-honest-exception
chapter's own file: full recompile, full rerun, full comparison against
the locked output, confirmed deterministic across two fresh runs.

Section 17.3 has a real-file mode, shaped like Section 16.4's: self-test
mode (no arguments) and real-file mode (`<model.gguf> [--pp N] [--tg N]`)
are two separate invocations of the same binary, not one run with an
appended block, because a real invocation's wall-clock throughput numbers
are exactly the one kind of output that MUST differ machine to machine --
locking them into this book's four-way byte-identical cross-check would
be locking in a claim no other reader's hardware could ever reproduce.
This script recompiles 03_llamacpp_comparison.cpp and reruns it with NO
arguments, comparing that output exactly against
03_llamacpp_comparison.out.txt (confirmed deterministic across two fresh
runs, same as every other section here). The real-run output
(03_llamacpp_comparison.real_run.out.txt) is verified only as data:
confirmed present in the built markdown byte for byte, and confirmed to
look like this section's own real llama.cpp comparison -- it is not
(cannot be) re-executed here, since that requires the actual ~644 MB
downloaded checkpoint, a from-source llama.cpp build, and real wall-clock
compute on a reader's own machine.

Section 1 compiles with plain flags (no transformer-block or mdspan
code); Sections 2 and 3 compile with -ffp-contract=off and the vendored
mdspan headers, exactly like Chapter 16's Sections 3 and 4.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part4/17-benchmarking.md"

EXPECTED_H1 = "# Chapter 17: Benchmarking Against llama.cpp and the Broader Ecosystem"
EXPECTED_H2 = [
    "## 17.1 What to Measure and Why",
    "## 17.2 A From-Scratch Benchmarking Harness",
    "## 17.3 An Honest Comparison Against llama.cpp",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_benchmark_stats_and_definitions.cpp",
    2: "02_benchmark_harness.cpp",
    3: "03_llamacpp_comparison.cpp",
}
OUT_FILES = {
    1: "01_benchmark_stats_and_definitions.out.txt",
    2: "02_benchmark_harness.out.txt",
    3: "03_llamacpp_comparison.out.txt",
}
REAL_OUT_FILE = "03_llamacpp_comparison.real_run.out.txt"
REAL_RUN_MARKER = "An Honest Comparison Against llama.cpp"

COMPILE_FLAGS = {
    1: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    2: ["-std=c++23", "-Wall", "-Wextra", "-O2", "-ffp-contract=off",
        "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std", "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
        "-I_vendor_mdspan/include"],
    3: ["-std=c++23", "-Wall", "-Wextra", "-O2", "-ffp-contract=off",
        "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std", "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
        "-I_vendor_mdspan/include"],
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
    text_blocks = [b for lang, b in blocks if lang == "text"]
    bash_blocks = [b for lang, b in blocks if lang == "bash"]

    if len(bash_blocks) != 3:
        fail(f"expected exactly 3 bash compile/run blocks, found {len(bash_blocks)}")
    else:
        ok("found exactly 3 bash compile/run blocks")
        for idx in range(1, 4):
            fname = FILES[idx]
            block = bash_blocks[idx - 1]
            if fname not in block:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")
            if "-std=c++23" not in block:
                fail(f"bash block {idx} does not compile with -std=c++23")
            needs_mdspan = idx in (2, 3)
            has_mdspan = "MDSPAN_IMPL_STANDARD_NAMESPACE" in block and "-ffp-contract=off" in block
            if needs_mdspan and not has_mdspan:
                fail(f"bash block {idx} is section {idx}, which requires -ffp-contract=off and mdspan flags")
            if not needs_mdspan and ("-ffp-contract=off" in block or "MDSPAN_IMPL_STANDARD_NAMESPACE" in block):
                fail(f"bash block {idx} unexpectedly compiles with mdspan/-ffp-contract flags "
                     f"(section {idx} has no transformer-block code requiring them)")
            if idx == 3:
                if ".gguf" not in block:
                    fail("bash block 3 is Section 17.3's real-file section but shows no real-file invocation")
                if "--pp" not in block or "--tg" not in block:
                    fail("bash block 3's real invocation should show this section's own --pp/--tg flags")

    if len(cpp_blocks) != 3:
        fail(f"expected exactly 3 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 3 cpp code blocks")
        for idx in range(1, 4):
            fname = FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    # 4 text blocks: one self-test output per section (3), plus Section
    # 17.3's separately-documented real-run output.
    if len(text_blocks) != 4:
        fail(f"expected exactly 4 text output blocks (3 self-test + 1 real-run), found {len(text_blocks)}")
    else:
        ok("found exactly 4 text output blocks")
        for idx in range(1, 4):
            fname = OUT_FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = text_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"output block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"output block {idx} matches locked {fname} exactly")

        real_locked = (BASE / REAL_OUT_FILE).read_text().rstrip("\n")
        real_got = text_blocks[3].rstrip("\n")
        if real_got != real_locked:
            fail(f"real-run output block does not exactly match documented file {REAL_OUT_FILE}")
        else:
            ok(f"real-run output block matches documented {REAL_OUT_FILE} exactly (verified as data, not re-executed)")
        if REAL_RUN_MARKER not in real_locked:
            fail(f"documented real-run output is missing its own banner {REAL_RUN_MARKER!r}")
        else:
            ok("documented real-run output contains its honest-exception real-run banner")
        if "vocab=151936" not in real_locked or "llama.cpp" not in real_locked.lower():
            fail("documented real-run output does not look like this section's own real comparison "
                 "against the real Qwen2.5-0.5B-Instruct checkpoint and a real llama.cpp build")
        else:
            ok("documented real-run output matches this section's own described real comparison")

    tmp_bins = []
    for idx, fname in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch17_{idx:02d}_bin"

        cmd = ["g++"] + COMPILE_FLAGS[idx] + [str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded ({' '.join(COMPILE_FLAGS[idx])})")

        locked_expected = (BASE / OUT_FILES[idx]).read_text().rstrip("\n")

        # Section 17.3's binary requires an explicit "no arguments" run to
        # hit self-test mode -- identical to every other section here,
        # since argc==1 is simply "no arguments" either way. Real mode
        # (argc>1) is Section 17.3's actual comparison contract and is
        # never invoked by this script.
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
        except FileNotFoundError:
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
