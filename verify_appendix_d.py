#!/usr/bin/env python3
"""Verify docs/appendix/d-profiling-and-benchmarking.md.

D.1-D.3 each get this book's usual unconditional compile/run/determinism
check (pure standard-library C++23, no architecture-specific intrinsics).
D.4 is diagnostics-only (real, captured `perf` transcripts, like Appendix
A's own diagnostics) -- there is nothing to compile there. Unlike Appendix
B/C, this appendix's own bash fences are NOT all compile/run blocks: D.2
and D.4 each include extra illustrative bash snippets (a std::chrono
wrapper is fenced with no language tag on purpose, and D.4's two example
`perf` command lines are real bash fences that reference no locked source
file) -- so this script finds each file's own compile/run block by content
match rather than assuming bash blocks appear in a fixed 1:1 order.
"""
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/appendix/d-profiling-and-benchmarking.md"
APPD = BASE / "appendix_d"

EXPECTED_H1 = "# Appendix D: Profiling and Benchmarking on Edge Hardware"
EXPECTED_H2 = [
    "## D.1 Computing Theoretical Peak Bandwidth From a Real Spec Sheet",
    "## D.2 A Real Bandwidth Micro-Benchmark, and Why Its Own Timing Can Never Be Locked",
    "## D.3 Latency-Budget Arithmetic",
    "## D.4 Profiling With `perf`: The Flags That Actually Work, and Where They Genuinely Don't",
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
    "### Intuition",
    "### The Concept, In Detail",
]

CPP_FILES = {
    1: "d1_peak_bandwidth_formula.cpp",
    2: "d2_bandwidth_microbenchmark.cpp",
    3: "d3_latency_budget_calculator.cpp",
}
OUT_FILES = {
    1: "d1_peak_bandwidth_formula.out.txt",
    2: "d2_bandwidth_microbenchmark.out.txt",
    3: "d3_latency_budget_calculator.out.txt",
}
DIAG_FILES = {
    "diag1": APPD / "diagnostics/diag1_perf_hardware_events_unsupported.txt",
    "diag2": APPD / "diagnostics/diag2_perf_paranoid_denial_on_device.txt",
    "diag3": APPD / "diagnostics/diag3_real_timing_varies_cloud_sandbox.txt",
    "diag4": APPD / "diagnostics/diag4_real_timing_varies_on_device.txt",
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
        ok("H2 sequence matches (D.1-D.4 + Summary + Where We Go Next)")

    h3_lines = [l for l in prose_lines if l.startswith("### ")]
    if h3_lines != EXPECTED_H3:
        fail(f"H3 sequence mismatch:\n  got:      {h3_lines}\n  expected: {EXPECTED_H3}")
    else:
        ok("H3 sequence matches (Intuition + The Concept, In Detail for each of D.1-D.4)")

    ALLOWED_NON_ASCII = {"—"}
    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
            bad_chars[ch] = bad_chars.get(ch, 0) + 1
    if bad_chars:
        fail(f"non-ASCII characters found (other than em-dash): {bad_chars}")
    else:
        ok("no disallowed non-ASCII characters")

    trap_count = len(re.findall(r'!!! warning "\[COMMON TRAP\]', text))
    if trap_count != 2:
        fail(f"expected exactly 2 [COMMON TRAP] boxes, found {trap_count}")
    else:
        ok(f"found exactly {trap_count} [COMMON TRAP] boxes")

    footnote_count = len(re.findall(r"^\[\^\w+\]:", text, re.MULTILINE))
    if footnote_count != 3:
        fail(f"expected exactly 3 footnote definitions (real hardware citations), found {footnote_count}")
    else:
        ok(f"found exactly {footnote_count} footnote citations for the real hardware bandwidth figures")

    fence_re = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
    blocks = fence_re.findall(text)
    cpp_blocks = [b for lang, b in blocks if lang == "cpp"]
    text_blocks = [b for lang, b in blocks if lang == "text"]
    bash_blocks = [b for lang, b in blocks if lang == "bash"]

    if len(cpp_blocks) != 3:
        fail(f"expected exactly 3 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 3 cpp code blocks")
        for idx, fname in CPP_FILES.items():
            locked = (APPD / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    # D.4's own diagnostics + D.2's own timing transcripts are all fenced as
    # "text" blocks, alongside D.1-D.3's own 3 locked self-test outputs --
    # 3 self-test outputs + 4 diagnostic/timing transcripts = 7 total, in
    # this exact template order: D.1's OUT1, then D.2's DIAG3/DIAG4 (they
    # appear before D.2's own OUT2 in the template's own prose), then OUT2,
    # then D.3's OUT3, then D.4's DIAG1/DIAG2.
    EXPECTED_TEXT_BLOCK_ORDER = [
        ("out", 1), ("diag", "diag3"), ("diag", "diag4"), ("out", 2),
        ("out", 3), ("diag", "diag1"), ("diag", "diag2"),
    ]
    if len(text_blocks) != len(EXPECTED_TEXT_BLOCK_ORDER):
        fail(f"expected exactly {len(EXPECTED_TEXT_BLOCK_ORDER)} text blocks "
             f"(3 locked outputs + 4 diagnostics), found {len(text_blocks)}")
    else:
        ok(f"found exactly {len(text_blocks)} text blocks (3 locked self-test outputs + 4 real diagnostic transcripts)")
        for pos, (kind, key) in enumerate(EXPECTED_TEXT_BLOCK_ORDER):
            if kind == "out":
                locked = (APPD / OUT_FILES[key]).read_text().rstrip("\n")
                label = f"output block for {OUT_FILES[key]}"
            else:
                locked = DIAG_FILES[key].read_text().rstrip("\n")
                label = f"diagnostic block for {key}"
            got = text_blocks[pos].rstrip("\n")
            if got != locked:
                fail(f"{label} does not exactly match its locked file (position {pos})")
            else:
                ok(f"{label} matches its locked file exactly")

    # Each of the 3 compiled files must have its own compile/run bash block
    # SOMEWHERE among all bash fences (this appendix also has 2 extra,
    # illustrative `perf` command bash fences that reference no source file).
    for idx, fname in CPP_FILES.items():
        matches = [b for b in bash_blocks if fname in b and "-std=c++23" in b]
        if len(matches) != 1:
            fail(f"expected exactly 1 compile/run block referencing {fname}, found {len(matches)}")
        else:
            ok(f"found exactly 1 compile/run block referencing {fname}")

    if len(bash_blocks) != 5:
        fail(f"expected exactly 5 bash blocks (3 compile/run + 2 illustrative perf commands), found {len(bash_blocks)}")
    else:
        ok("found exactly 5 bash blocks (3 compile/run + D.4's 2 illustrative perf commands)")

    # Genuinely recompile and rerun each of D.1-D.3 fresh, twice, for
    # determinism -- this book's usual unconditional check.
    tmp_bins = []
    for idx, fname in CPP_FILES.items():
        src = APPD / fname
        binp = APPD / f"_verify_appendix_d_{idx:02d}_bin"

        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2", str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded")

        locked_expected = (APPD / OUT_FILES[idx]).read_text().rstrip("\n")

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
