#!/usr/bin/env python3
"""Verify docs/part2/11-parallelizing-the-forward-pass.md.

Every file in this chapter creates real std::thread objects (via each
file's own ParallelPool, generalized from Chapter 10.3's TensorThreadPool),
so every compile uses -pthread. Files that touch the mdspan-backed KVCache
(everything except Section 11.4's plain-array reduction) additionally need
the vendored mdspan flags Chapter 8 established.

Every file ALSO compiles with -ffp-contract=off, unconditionally, for a
reason this chapter discovered the hard way while authoring Section 11.2:
GCC's default -ffp-contract=fast lets the compiler fuse a multiply and an
add into a single hardware FMA wherever it judges doing so profitable, a
judgment made per call site (after inlining) that is NOT guaranteed to
treat two textually different call sites of "the same" arithmetic
identically -- and aarch64's baseline instruction set always has scalar
FMA available where this book's x86_64 baseline (no -mfma) does not,
so the exact same source produced bit-for-bit matching serial-vs-parallel
results on one architecture and NOT the other until -ffp-contract=off was
added. This script does not merely trust that flag was remembered in the
markdown's own bash blocks -- it checks for it explicitly, the same way
Chapter 10's script checked for -pthread.

As in Chapter 10, every file in this chapter is run TWICE per invocation
(not just once), given this chapter's continued exposure to real thread
scheduling.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part2/11-parallelizing-the-forward-pass.md"
MDSPAN_INCLUDE = BASE / "_vendor_mdspan/include"

EXPECTED_H1 = "# Chapter 11: Parallelizing the Forward Pass -- Row-Parallel, Head-Parallel, and Reproducible Across Thread Counts"
EXPECTED_H2 = [
    "## 11.1 Row-Parallel vs. Column-Parallel Matmul",
    "## 11.2 A Persistent Pool Across a Full Layer's Phases",
    "## 11.3 Head-Parallel Attention",
    "## 11.4 Deterministic Floating-Point Reduction, Independent of Thread Count",
    "## 11.5 A Fully Multi-Threaded Decode Step",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

# (filename, needs_mdspan_flags, needs_pthread)
# Every file in this chapter creates real threads; only Section 11.4's
# plain-array reduction has no mdspan-backed structure to view.
FILES = {
    1: ("01_row_vs_column_parallel_matmul.cpp", True, True),
    2: ("02_persistent_pool_full_layer.cpp", True, True),
    3: ("03_head_parallel_attention.cpp", True, True),
    4: ("04_deterministic_reduction.cpp", False, True),
    5: ("05_full_decode_step_multithreaded.cpp", True, True),
}
OUT_FILES = {
    1: "01_row_vs_column_parallel_matmul.out.txt",
    2: "02_persistent_pool_full_layer.out.txt",
    3: "03_head_parallel_attention.out.txt",
    4: "04_deterministic_reduction.out.txt",
    5: "05_full_decode_step_multithreaded.out.txt",
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
        ok("H2 sequence matches")

    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch != "—":
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
            fname, needs_mdspan, needs_pthread = FILES[idx]
            block = bash_blocks[idx - 1]
            if fname not in block:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")
            if needs_pthread and "-pthread" not in block:
                fail(f"bash block {idx} creates threads but does not compile with -pthread")
            elif needs_pthread:
                ok(f"bash block {idx} compiles with -pthread")
            if "-ffp-contract=off" not in block:
                fail(f"bash block {idx} does not compile with -ffp-contract=off (this chapter's cross-architecture fix)")
            else:
                ok(f"bash block {idx} compiles with -ffp-contract=off")
            if needs_mdspan and "MDSPAN_IMPL_STANDARD_NAMESPACE" not in block:
                fail(f"bash block {idx} needs mdspan flags but does not have them")
            elif needs_mdspan:
                ok(f"bash block {idx} compiles with mdspan flags")

    if len(cpp_blocks) != 5:
        fail(f"expected exactly 5 cpp code blocks, found {len(cpp_blocks)}")
    else:
        ok("found exactly 5 cpp code blocks")
        for idx in range(1, 6):
            fname = FILES[idx][0]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = cpp_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"code block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"code block {idx} matches locked {fname} exactly")

    if len(text_blocks) != 5:
        fail(f"expected exactly 5 text output blocks, found {len(text_blocks)}")
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

    if not (MDSPAN_INCLUDE / "mdspan" / "mdspan.hpp").exists():
        fail(f"vendored mdspan headers not found at {MDSPAN_INCLUDE}")

    tmp_bins = []
    for idx, (fname, needs_mdspan, needs_pthread) in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch11_{idx:02d}_bin"

        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2", "-ffp-contract=off"]
        if needs_pthread:
            cmd += ["-pthread"]
        if needs_mdspan:
            cmd += [
                "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std",
                "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
                f"-I{MDSPAN_INCLUDE}",
            ]
        cmd += [str(src), "-o", str(binp)]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded (g++ -std=c++23 -ffp-contract=off{' -pthread' if needs_pthread else ''}{' -mdspan-flags' if needs_mdspan else ''})")

        r2 = subprocess.run([str(binp)], capture_output=True, text=True)
        fresh = r2.stdout.rstrip("\n")
        locked = (BASE / OUT_FILES[idx]).read_text().rstrip("\n")
        if fresh != locked:
            fail(f"fresh rerun of {fname} does not match locked output")
        else:
            ok(f"fresh rerun of {fname} matches locked output exactly (deterministic)")
        if r2.returncode != 0:
            fail(f"fresh rerun of {fname} exited non-zero ({r2.returncode})")

        # A second real run, since this chapter's whole premise is real
        # thread scheduling -- run twice and diff, the same discipline
        # Chapter 10's files were held to.
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
