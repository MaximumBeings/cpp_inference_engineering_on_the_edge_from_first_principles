#!/usr/bin/env python3
"""Verify docs/appendix/c-decision-tree-reference.md.

C.1/C.2/C.3 each restate a real, already-derived decision tree from
Chapters 4-14 (and Chapter 30's crossover formula) as one callable advisor
function. Each advisor is pure standard-library C++23 with no
architecture-specific intrinsics or external dependencies, so all three get
this book's usual unconditional compile/run/determinism check.
"""
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/appendix/c-decision-tree-reference.md"
APPC = BASE / "appendix_c"

EXPECTED_H1 = "# Appendix C: A Decision-Tree Reference: Quantization, Threading, and Cache Strategies at a Glance"
EXPECTED_H2 = [
    "## C.1 Choosing a Quantization Strategy",
    "## C.2 Choosing a Threading and Parallelization Strategy",
    "## C.3 Choosing a KV Cache Strategy",
    "## Appendix Summary",
    "## Where We Go Next",
]
EXPECTED_H3 = [
    "### Intuition",
    "### The Concept, In Detail",
    "### Intuition",
    "### The Concept, In Detail",
    "### Intuition",
    "### The Concept, In Detail",
]

CPP_FILES = {
    1: "c1_quantization_format_advisor.cpp",
    2: "c2_threading_strategy_advisor.cpp",
    3: "c3_kv_cache_strategy_advisor.cpp",
}
OUT_FILES = {
    1: "c1_quantization_format_advisor.out.txt",
    2: "c2_threading_strategy_advisor.out.txt",
    3: "c3_kv_cache_strategy_advisor.out.txt",
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
        ok("H2 sequence matches (C.1-C.3 + Summary + Where We Go Next)")

    h3_lines = [l for l in prose_lines if l.startswith("### ")]
    if h3_lines != EXPECTED_H3:
        fail(f"H3 sequence mismatch:\n  got:      {h3_lines}\n  expected: {EXPECTED_H3}")
    else:
        ok("H3 sequence matches (Intuition + The Concept, In Detail for each of C.1-C.3)")

    ALLOWED_NON_ASCII = {"—"}
    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
            bad_chars[ch] = bad_chars.get(ch, 0) + 1
    if bad_chars:
        fail(f"non-ASCII characters found (other than em-dash): {bad_chars}")
    else:
        ok("no disallowed non-ASCII characters")

    # Exactly 6 [COMMON TRAP] boxes: 2 each in C.1, C.2, and C.3.
    trap_count = len(re.findall(r'!!! warning "\[COMMON TRAP\]', text))
    if trap_count != 6:
        fail(f"expected exactly 6 [COMMON TRAP] boxes (2 per section), found {trap_count}")
    else:
        ok(f"found exactly {trap_count} [COMMON TRAP] boxes (2 per section)")

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
            locked = (APPC / fname).read_text().rstrip("\n")
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
            locked = (APPC / fname).read_text().rstrip("\n")
            got = text_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"output block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"output block {idx} matches locked {fname} exactly")

    # Genuinely recompile and rerun each advisor fresh, twice, for
    # determinism -- this book's usual unconditional check.
    tmp_bins = []
    for idx, fname in CPP_FILES.items():
        src = APPC / fname
        binp = APPC / f"_verify_appendix_c_{idx:02d}_bin"

        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2", str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded")

        locked_expected = (APPC / OUT_FILES[idx]).read_text().rstrip("\n")

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
            # delete permission granted) refuse the unlink -- doesn't affect anything checked above.
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
