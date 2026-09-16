#!/usr/bin/env python3
"""Verify docs/part4/15-running-real-models.md.

This chapter introduces a genuinely new verification wrinkle: Sections
15.1, 15.2, and 15.4 each have a REAL-FILE mode (argv[1] = a path to the
actual ~644 MB downloaded Qwen2.5-0.5B-Instruct checkpoint) whose locked
output cannot be reproduced here, because that file is deliberately never
transferred into this build environment. For those three files, this
script recompiles and reruns each one with NO argument (self-test mode
only) and confirms that output against the SELF-TEST PORTION of the
locked .out.txt -- i.e. the locked file with its "=== REAL ... ==="
section excised, which is exactly what a no-argument run produces. It
also confirms self-test determinism across two fresh runs. The real-file
section of each locked .out.txt is verified only as data: it is checked
to be present in the built markdown byte for byte, matching what the
file's own account says was captured on the reader's real machine, but
it is not (cannot be) re-executed here.

Section 15.3 has no real-file mode at all -- it is pure synthetic
self-test, verified exactly like every earlier chapter's own file: full
recompile, full rerun, full comparison against the locked output.

Like Sections 15.3 and 15.4 themselves, files 3 and 4 compile with
-ffp-contract=off and the vendored mdspan headers (the same
cross-architecture floating-point and tensor-view requirements Chapter
11 first established); files 1 and 2 are simple GGUF-reader code with no
such requirement and compile with just -std=c++23 -Wall -Wextra -O2.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part4/15-running-real-models.md"

EXPECTED_H1 = "# Chapter 15: Running Real Models -- From HuggingFace to First Token with Qwen2.5"
EXPECTED_H2 = [
    "## 15.1 A Reader That Survives Contact With a Real File",
    "## 15.2 Five Ways a Real Qwen2 Checkpoint Differs From Llama",
    "## 15.3 Adapting the Forward Pass and Tokenizer for Qwen2",
    "## 15.4 A First Real Token From Qwen2.5-0.5B-Instruct",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_real_gguf_inspection.cpp",
    2: "02_llama_vs_qwen2_diff.cpp",
    3: "03_qwen2_transformer_block.cpp",
    4: "04_first_real_token.cpp",
}
OUT_FILES = {
    1: "01_real_gguf_inspection.out.txt",
    2: "02_llama_vs_qwen2_diff.out.txt",
    3: "03_qwen2_transformer_block.out.txt",
    4: "04_first_real_token.out.txt",
}
# Sections with a real-file honest-exception mode (argc/argv).
REAL_FILE_SECTIONS = {1, 2, 4}
REAL_MARKERS = {
    1: "\n=== REAL FILE INSPECTION",
    2: "\n=== REAL FILE VERIFICATION",
    4: "\n=== REAL GENERATION",
}
CLOSING_PATTERN = ("\n========================================================\n"
                    "ALL CHECKS PASSED\n"
                    "========================================================")

COMPILE_FLAGS = {
    1: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    2: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    3: ["-std=c++23", "-Wall", "-Wextra", "-O2", "-ffp-contract=off",
        "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std", "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
        "-I_vendor_mdspan/include"],
    4: ["-std=c++23", "-Wall", "-Wextra", "-O2", "-ffp-contract=off",
        "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std", "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
        "-I_vendor_mdspan/include"],
}

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"OK:   {msg}")


def expected_selftest_output(locked_text: str, idx: int) -> str:
    """For a real-file-mode section, derive what a no-argument run's full
    stdout should look like: everything up to (not including) the
    "=== REAL ..." marker, plus the final closing banner -- exactly what
    the source's own control flow produces when argc < 2, since the
    closing banner print is unconditional and the real-file block is the
    only thing skipped.
    """
    marker = REAL_MARKERS[idx]
    if marker not in locked_text:
        raise ValueError(f"locked output for section {idx} missing expected marker {marker!r}")
    if CLOSING_PATTERN not in locked_text:
        raise ValueError(f"locked output for section {idx} missing closing banner")
    before = locked_text[:locked_text.index(marker)]
    closing = locked_text[locked_text.rindex(CLOSING_PATTERN):]
    return before + closing


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
        ok("H2 sequence matches (4 numbered sections + summary/self-check/next/solutions)")

    # U+0120 is not stray formatting: it is GPT-2's own standard glyph
    # for the byte-level BPE "space" symbol, and Section 15.4's source
    # comment names it exactly this way because that is how every
    # GPT-2/tiktoken reference spells it. It is content this chapter
    # genuinely needs, not an em-dash- or smart-quote-style formatting
    # artifact -- allowed for exactly that reason, unlike earlier
    # chapters that never had domain content requiring a non-ASCII
    # glyph at all.
    ALLOWED_NON_ASCII = {"—", "Ġ"}
    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
            bad_chars[ch] = bad_chars.get(ch, 0) + 1
    if bad_chars:
        fail(f"non-ASCII characters found (other than em-dash and GPT-2's U+0120 space glyph): {bad_chars}")
    else:
        ok("no disallowed non-ASCII characters")

    fence_re = re.compile(r"```(\w+)\n(.*?)\n```", re.DOTALL)
    blocks = fence_re.findall(text)
    cpp_blocks = [b for lang, b in blocks if lang == "cpp"]
    text_blocks = [b for lang, b in blocks if lang == "text"]
    bash_blocks = [b for lang, b in blocks if lang == "bash"]

    if len(bash_blocks) != 4:
        fail(f"expected exactly 4 bash compile/run blocks, found {len(bash_blocks)}")
    else:
        ok("found exactly 4 bash compile/run blocks")
        for idx in range(1, 5):
            fname = FILES[idx]
            block = bash_blocks[idx - 1]
            if fname not in block:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")
            if "-std=c++23" not in block:
                fail(f"bash block {idx} does not compile with -std=c++23")
            needs_mdspan = idx in (3, 4)
            has_mdspan = "MDSPAN_IMPL_STANDARD_NAMESPACE" in block and "-ffp-contract=off" in block
            if needs_mdspan and not has_mdspan:
                fail(f"bash block {idx} is section {idx}, which requires -ffp-contract=off and mdspan flags")
            if not needs_mdspan and ("-ffp-contract=off" in block or "MDSPAN_IMPL_STANDARD_NAMESPACE" in block):
                fail(f"bash block {idx} unexpectedly compiles with mdspan/-ffp-contract flags "
                     f"(section {idx} is plain GGUF-reader code with no such requirement)")
            if idx in REAL_FILE_SECTIONS:
                if "qwen2.5-0.5b-instruct-q8_0.gguf" not in block and ".gguf" not in block:
                    fail(f"bash block {idx} is a real-file section but shows no real-file invocation")

    if len(cpp_blocks) != 4:
        fail(f"expected exactly 4 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 4 cpp code blocks")
        for idx in range(1, 5):
            fname = FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    if len(text_blocks) != 4:
        fail(f"expected exactly 4 text output blocks, found {len(text_blocks)}")
    else:
        ok("found exactly 4 text output blocks")
        for idx in range(1, 5):
            fname = OUT_FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = text_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"output block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"output block {idx} matches locked {fname} exactly")
            if idx in REAL_FILE_SECTIONS:
                marker = REAL_MARKERS[idx]
                if marker not in locked:
                    fail(f"locked output for section {idx} is missing its real-file marker {marker!r}")
                else:
                    ok(f"locked output for section {idx} contains its honest-exception real-file section (verified as data, not re-executed)")

    tmp_bins = []
    for idx, fname in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch15_{idx:02d}_bin"

        cmd = ["g++"] + COMPILE_FLAGS[idx] + [str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded ({' '.join(COMPILE_FLAGS[idx])})")

        locked_full = (BASE / OUT_FILES[idx]).read_text().rstrip("\n")
        if idx in REAL_FILE_SECTIONS:
            try:
                locked_expected = expected_selftest_output(locked_full, idx).rstrip("\n")
            except ValueError as e:
                fail(str(e))
                continue
        else:
            locked_expected = locked_full

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
