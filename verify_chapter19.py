#!/usr/bin/env python3
"""Verify docs/part5/19-retail-inventory.md.

Every section in this chapter (19.1 through 19.5) is pure, deterministic
self-test against synthetic fixtures -- no section has a separate
real-file mode, and unlike Chapter 18.4's SQLite dependency, no section
here has an external system library to be missing from the cross-
compilation sandbox, so this script recompiles and reruns all five
sections without any documented reduced-verification exception.

Sections 1, 3, 4, and 5 compile with plain flags. Section 2 alone reuses
Chapter 18's transformer-block and vision-encoder machinery, so it needs
-ffp-contract=off and the vendored mdspan headers, exactly like Chapter
18's own Section 3.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part5/19-retail-inventory.md"

EXPECTED_H1 = "# Chapter 19: Retail Inventory and Supply Chain Intelligence: Shelf Photography to Action"
EXPECTED_H2 = [
    "## 19.1 Encoding a Planogram Directly into the System Prompt",
    "## 19.2 A Batch Shelf-Photo Processor",
    "## 19.3 REST Integration into ERP/WMS Systems",
    "## 19.4 Multi-Store Aggregation and Trend Detection",
    "## 19.5 Shrinkage and Misplacement Detection from Ordinary Shelf Photographs",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_planogram_and_system_prompt.cpp",
    2: "02_batch_shelf_photo_processor.cpp",
    3: "03_erp_wms_rest_integration.cpp",
    4: "04_multistore_aggregation_and_trends.cpp",
    5: "05_shrinkage_and_misplacement_detection.cpp",
}
OUT_FILES = {
    1: "01_planogram_and_system_prompt.out.txt",
    2: "02_batch_shelf_photo_processor.out.txt",
    3: "03_erp_wms_rest_integration.out.txt",
    4: "04_multistore_aggregation_and_trends.out.txt",
    5: "05_shrinkage_and_misplacement_detection.out.txt",
}

COMPILE_FLAGS = {
    1: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    2: ["-std=c++23", "-Wall", "-Wextra", "-O2", "-ffp-contract=off",
        "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std", "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
        "-I_vendor_mdspan/include"],
    3: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    4: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    5: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
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
        ok("H2 sequence matches (5 numbered sections + summary/self-check/next/solutions)")

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

    if len(bash_blocks) != 5:
        fail(f"expected exactly 5 bash compile/run blocks, found {len(bash_blocks)}")
    else:
        ok("found exactly 5 bash compile/run blocks")
        for idx in range(1, 6):
            fname = FILES[idx]
            block = bash_blocks[idx - 1]
            if fname not in block:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")
            if "-std=c++23" not in block:
                fail(f"bash block {idx} does not compile with -std=c++23")
            needs_mdspan = idx == 2
            has_mdspan = "MDSPAN_IMPL_STANDARD_NAMESPACE" in block and "-ffp-contract=off" in block
            if needs_mdspan and not has_mdspan:
                fail(f"bash block {idx} is section {idx}, which requires -ffp-contract=off and mdspan flags")
            if not needs_mdspan and ("-ffp-contract=off" in block or "MDSPAN_IMPL_STANDARD_NAMESPACE" in block):
                fail(f"bash block {idx} unexpectedly compiles with mdspan/-ffp-contract flags "
                     f"(section {idx} has no transformer-block code requiring them)")

    if len(cpp_blocks) != 5:
        fail(f"expected exactly 5 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 5 cpp code blocks")
        for idx in range(1, 6):
            fname = FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    if len(text_blocks) != 5:
        fail(f"expected exactly 5 text output blocks (one self-test output per section), found {len(text_blocks)}")
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

    tmp_bins = []
    for idx, fname in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch19_{idx:02d}_bin"

        cmd = ["g++"] + COMPILE_FLAGS[idx] + [str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded ({' '.join(COMPILE_FLAGS[idx])})")

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
