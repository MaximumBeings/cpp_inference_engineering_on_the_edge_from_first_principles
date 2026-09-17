#!/usr/bin/env python3
"""Verify docs/appendix/b-practice-quiz.md.

B.2/B.3 are prose (conceptual questions and answers) with no code to check.
B.4's three challenges each get this book's usual, unconditional compile/
run/determinism check with g++ -std=c++23, since none of them depend on
anything beyond the standard library.
"""
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/appendix/b-practice-quiz.md"
APPB = BASE / "appendix_b"

EXPECTED_H1 = "# Appendix B: Practice Quiz"
EXPECTED_H2 = [
    "## B.1 How to Use This Quiz",
    "## B.2 Conceptual Review Questions",
    "## B.3 Conceptual Review Answers",
    "## B.4 Predict-the-Output Challenges",
    "## Appendix Summary",
    "## Where We Go Next",
]
EXPECTED_H3 = [
    "### Challenge 1: The Softmax Overflow Trap",
    "### Challenge 2: Reduction Order and Float32 Non-Associativity",
    "### Challenge 3: The Re-Quantization Trap",
]

CPP_FILES = {
    1: "challenge1_softmax_overflow.cpp",
    2: "challenge2_reduction_order.cpp",
    3: "challenge3_requant_nonassoc.cpp",
}
OUT_FILES = {
    1: "challenge1_softmax_overflow.out.txt",
    2: "challenge2_reduction_order.out.txt",
    3: "challenge3_requant_nonassoc.out.txt",
}

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"OK:   {msg}")


def _lines_outside_code_fences(text):
    out = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return out


def main() -> int:
    if not MD_PATH.exists():
        fail(f"built markdown does not exist: {MD_PATH}")
        return report()

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
        ok("H2 sequence matches (B.1-B.4 + Summary + Where We Go Next)")

    h3_lines = [l for l in prose_lines if l.startswith("### ")]
    if h3_lines != EXPECTED_H3:
        fail(f"H3 (Challenge) sequence mismatch:\n  got:      {h3_lines}\n  expected: {EXPECTED_H3}")
    else:
        ok("H3 sequence matches all 3 Challenges")

    ALLOWED_NON_ASCII = {"—"}
    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
            bad_chars[ch] = bad_chars.get(ch, 0) + 1
    if bad_chars:
        fail(f"non-ASCII characters found (other than em-dash): {bad_chars}")
    else:
        ok("no disallowed non-ASCII characters")

    # 14 conceptual questions in B.2 must match 14 answers in B.3.
    b2_start = text.find("## B.2 Conceptual Review Questions")
    b3_start = text.find("## B.3 Conceptual Review Answers")
    b4_start = text.find("## B.4 Predict-the-Output Challenges")
    b2_section = text[b2_start:b3_start]
    b3_section = text[b3_start:b4_start]
    q_count = len(re.findall(r"^\d+\.\s", b2_section, re.MULTILINE))
    a_count = len(re.findall(r"^\*\*\d+\.\*\*", b3_section, re.MULTILINE))
    if q_count == 0 or q_count != a_count:
        fail(f"conceptual question count ({q_count}) != answer count ({a_count})")
    elif q_count != 14:
        fail(f"expected exactly 14 conceptual questions, found {q_count}")
    else:
        ok(f"{q_count} conceptual questions, {a_count} answers -- counts match, both equal 14 "
           "(2 per Part across all 7 Parts)")

    fence_re = re.compile(r"```(\w+)\n(.*?)\n```", re.DOTALL)
    blocks = fence_re.findall(text)
    cpp_blocks = [b for lang, b in blocks if lang == "cpp"]
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
            if "-std=c++23" not in block:
                fail(f"compile/run block {idx} does not compile with -std=c++23")
        ok("all 3 compile/run blocks reference their own source file and -std=c++23")

    if len(cpp_blocks) != 3:
        fail(f"expected exactly 3 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 3 cpp code blocks")
        for idx, fname in CPP_FILES.items():
            locked = (APPB / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    if len(text_blocks) != 3:
        fail(f"expected exactly 3 text output blocks, found {len(text_blocks)}")
    else:
        ok("found exactly 3 text output blocks")
        for idx, fname in OUT_FILES.items():
            locked = (APPB / fname).read_text().rstrip("\n")
            got = text_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"output block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"output block {idx} matches locked {fname} exactly")

    # Genuinely recompile and rerun each Challenge fresh, twice, for determinism -- this book's
    # usual unconditional check, since all 3 challenges are pure standard-library C++23 with no
    # architecture-specific intrinsics or external dependencies.
    tmp_bins = []
    for idx, fname in CPP_FILES.items():
        src = APPB / fname
        binp = APPB / f"_verify_appendix_b_{idx:02d}_bin"

        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2", str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded")

        locked_expected = (APPB / OUT_FILES[idx]).read_text().rstrip("\n")

        r2 = subprocess.run([str(binp)], capture_output=True, text=True)
        fresh = r2.stdout.rstrip("\n")
        if fresh != locked_expected:
            fail(f"fresh rerun of {fname} does not match the locked output")
        else:
            ok(f"fresh rerun of {fname} matches locked output exactly")
        if r2.returncode != 0:
            fail(f"fresh rerun of {fname} exited non-zero ({r2.returncode})")

        r3 = subprocess.run([str(binp)], capture_output=True, text=True)
        if r3.stdout != r2.stdout:
            fail(f"second rerun of {fname} does not match the first rerun (nondeterministic!)")
        else:
            ok(f"second rerun of {fname} matches the first rerun exactly (confirmed deterministic)")

        tmp_bins.append(binp)

    for b in tmp_bins:
        try:
            b.unlink()
        except OSError:
            # Best-effort cleanup: some environments (e.g. a connected-folder mount with no
            # delete permission granted) refuse the unlink. A leftover scratch binary here does
            # not affect anything this script has already checked above.
            pass

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
