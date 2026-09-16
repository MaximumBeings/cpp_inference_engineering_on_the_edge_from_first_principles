#!/usr/bin/env python3
"""Verify docs/part5/21-document-intelligence-claims-environmental.md.

Every section in this chapter (21.1 through 21.4) is pure, deterministic
self-test against synthetic fixtures and synthetic data only -- no section
here has an external system dependency missing from the cross-compilation
sandbox, so this script recompiles and reruns all four sections without
any documented reduced-verification exception. All four sections compile
with plain flags -- unlike Chapter 20.5, no section in this chapter
touches Chapter 18's vision-encoder machinery, so none needs
-ffp-contract=off or the vendored mdspan include path.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part5/21-document-intelligence-claims-environmental.md"

EXPECTED_H1 = "# Chapter 21: Document Intelligence, Insurance Claims, and Environmental Monitoring"
EXPECTED_H2 = [
    "## 21.1 Why a Vision-Language Model Replaces a Traditional OCR Pipeline Outright",
    "## 21.2 A Document-Type Router with Handwriting-Aware Extraction",
    "## 21.3 Vehicle- and Property-Damage Assessment Engines for Insurance Claims",
    "## 21.4 Wildlife and Underwater Species Identification from Camera-Trap and Marine Imagery",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_ocr_pipeline_vs_vlm_extraction.cpp",
    2: "02_document_router_and_handwriting_confidence.cpp",
    3: "03_vehicle_damage_assessment_and_claim_consistency.cpp",
    4: "04_camera_trap_deduplication_and_underwater_color_correction.cpp",
}
OUT_FILES = {
    1: "01_ocr_pipeline_vs_vlm_extraction.out.txt",
    2: "02_document_router_and_handwriting_confidence.out.txt",
    3: "03_vehicle_damage_assessment_and_claim_consistency.out.txt",
    4: "04_camera_trap_deduplication_and_underwater_color_correction.out.txt",
}

COMPILE_FLAGS = {
    1: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    2: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    3: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    4: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
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
        ok("H2 sequence matches (4 numbered sections + summary/self-check/next/solutions)")

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
            if "-ffp-contract=off" in block or "MDSPAN_IMPL_STANDARD_NAMESPACE" in block or "_vendor_mdspan" in block:
                fail(f"bash block {idx} unexpectedly references vision-encoder-only flags/includes "
                     f"(no section in this chapter touches Chapter 18's vision encoder)")

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
        fail(f"expected exactly 4 text output blocks (one self-test output per section), found {len(text_blocks)}")
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

    tmp_bins = []
    for idx, fname in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch21_{idx:02d}_bin"

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
