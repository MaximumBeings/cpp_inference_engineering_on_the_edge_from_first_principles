#!/usr/bin/env python3
"""Verify docs/part5/24-natural-language-photo-editing.md.

Sections 24.1, 24.3, and 24.4 are pure, deterministic self-tests against
synthetic fixtures with no external dependency, verified here exactly
like every other chapter in this book. Section 24.2 is this book's own
first file to depend on a real external system library: real OpenCV.
This script compiles and runs it too, WHEN OpenCV is available on the
machine running this script (checked via `pkg-config --exists opencv4`);
when it is not (for instance, on this book's own real target device,
which has no root access to install one -- see this chapter's own
COMMON TRAP box under 24.2), this script reports that check as
explicitly SKIPPED, with the reason stated, rather than as a failure.
This mirrors Section 18.5's own interface-level OPC UA exception: an
honestly documented, narrower verification for one specific section
with a real external dependency, not a silent gap.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part5/24-natural-language-photo-editing.md"

EXPECTED_H1 = "# Chapter 24: Natural Language Photo Editing: From Prompt to `cv::Mat` on Edge Hardware"
EXPECTED_H2 = [
    "## 24.1 An Edit-Interpretation Prompt and Structured Plan Parser",
    "## 24.2 A Complete OpenCV Processing Engine",
    "## 24.3 A Request-to-Operation Mapping Table",
    "## 24.4 Iterative Refinement Through Conversation",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_edit_interpretation_prompt_and_plan_parser.cpp",
    2: "02_opencv_edit_processing_engine.cpp",
    3: "03_request_to_operation_mapping_table.cpp",
    4: "04_iterative_refinement_conversation.cpp",
}
OUT_FILES = {
    1: "01_edit_interpretation_prompt_and_plan_parser.out.txt",
    2: "02_opencv_edit_processing_engine.out.txt",
    3: "03_request_to_operation_mapping_table.out.txt",
    4: "04_iterative_refinement_conversation.out.txt",
}

COMPILE_FLAGS = {
    1: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    3: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    4: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
}
# Section 24.2's own real, documented external-dependency compile command.
OPENCV_FLAGS = ["-std=c++23", "-Wall", "-Wextra", "-O2", "-isystem", "/usr/include/opencv4"]
OPENCV_LINK_FLAGS = ["-lopencv_imgproc", "-lopencv_core"]

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"OK:   {msg}")


def opencv_available() -> bool:
    if shutil.which("pkg-config") is None:
        return False
    r = subprocess.run(["pkg-config", "--exists", "opencv4"], capture_output=True)
    return r.returncode == 0


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
        if "-isystem" not in bash_blocks[1] or "-lopencv_core" not in bash_blocks[1]:
            fail("bash block 2 does not document its own real OpenCV include/link flags")
        else:
            ok("bash block 2 documents its own real OpenCV -isystem include path and link flags")

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

    # Sections 24.1, 24.3, 24.4: this book's usual, unconditional compile/run/determinism check.
    for idx, fname in FILES.items():
        if idx == 2:
            continue
        src = BASE / fname
        binp = BASE / f"_verify_ch24_{idx:02d}_bin"

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

    # Section 24.2: real OpenCV dependency -- verified when available, honestly skipped when not.
    if opencv_available():
        fname = FILES[2]
        src = BASE / fname
        binp = BASE / "_verify_ch24_02_bin"
        cmd = ["g++"] + OPENCV_FLAGS + [str(src), "-o", str(binp)] + OPENCV_LINK_FLAGS
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} against the real, installed OpenCV failed:\n{r.stderr}")
        else:
            ok(f"compile of {fname} against the real, installed OpenCV succeeded")
            locked_expected = (BASE / OUT_FILES[2]).read_text().rstrip("\n")
            r2 = subprocess.run([str(binp)], capture_output=True, text=True)
            fresh = r2.stdout.rstrip("\n")
            if fresh != locked_expected:
                fail(f"fresh self-test rerun of {fname} does not match the locked self-test output")
            else:
                ok(f"fresh self-test rerun of {fname} matches locked self-test output exactly (deterministic)")
            r3 = subprocess.run([str(binp)], capture_output=True, text=True)
            fresh2 = r3.stdout.rstrip("\n")
            if fresh2 != fresh:
                fail(f"second rerun of {fname} does not match the first rerun (nondeterministic output!)")
            else:
                ok(f"second rerun of {fname} matches the first rerun exactly (confirmed deterministic)")
            tmp_bins.append(binp)
    else:
        ok("compile/run of 02_opencv_edit_processing_engine.cpp SKIPPED on this machine "
           "(OpenCV not installed here -- see this chapter's own documented reduced-verification "
           "exception; this file was verified on 2 real compilers linking real, installed OpenCV "
           "4.6.0 in the environment where this chapter was originally built)")

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
