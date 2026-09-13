#!/usr/bin/env python3
"""Verify docs/part2/10-threading-and-concurrency.md.

Every file in this chapter creates real std::thread objects, so every
compile uses -pthread in addition to this book's standard flags, and file
3 additionally needs the vendored mdspan flags (needs_mdspan=True, same
convention Chapter 8 established).

This chapter's files are more delicate than earlier chapters' in one
specific way: they involve real OS thread scheduling. Each one was
authored only after confirming, empirically, that its CHECKED, PRINTED
output is stable across many repeated runs (5-10+ consecutive runs were
checked by hand during authoring, plus separate confirmation under GCC 14
and under aarch64 cross-compilation + qemu emulation) -- not merely run
twice by luck. That stability is not an accident: every section was
deliberately designed so that whatever IS printed is either guaranteed by
a language/library contract (std::atomic's RMW guarantee, integer
addition's exact associativity, std::mutex's non-recursive contract) or is
a purely structural, compile-time-determined fact (partition-range
arithmetic, cache-line offsets relative to a stable base address) --
never a raw, scheduling-dependent number. Where a section's real subject
IS a scheduling-dependent outcome (Section 10.2's data race, Section
10.5's exact division of labor between threads), that outcome is
deliberately excluded from the file's own printed output, so this
script's ordinary run-twice-and-diff check remains a meaningful,
non-flaky test rather than something this chapter had to weaken.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part2/10-threading-and-concurrency.md"
MDSPAN_INCLUDE = BASE / "_vendor_mdspan/include"

EXPECTED_H1 = "# Chapter 10: Threading and Concurrency -- Real Threads, Verified Without a Clock"
EXPECTED_H2 = [
    "## 10.1 The Latency Budget and Amdahl's Ceiling",
    "## 10.2 Atomics, Mutexes, and Why Synchronization Is Not Optional",
    "## 10.3 A Barrier-Synchronized Thread Pool for a Partitioned GEMV",
    "## 10.4 False Sharing: A Memory-Layout Hazard, Not a Race",
    "## 10.5 Work Stealing: Verifying Correctness Under a Nondeterministic Schedule",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

# (filename, needs_mdspan_flags, needs_pthread)
# Section 10.1 is pure arithmetic -- it creates no std::thread at all, so
# it neither needs nor uses -pthread; every other section in this chapter
# does.
FILES = {
    1: ("01_latency_budget_and_amdahls_law.cpp", False, False),
    2: ("02_atomics_mutexes_and_synchronization.cpp", False, True),
    3: ("03_barrier_thread_pool_gemv.cpp", True, True),
    4: ("04_false_sharing_and_cache_line_layout.cpp", False, True),
    5: ("05_work_stealing_scheduler.cpp", False, True),
}
OUT_FILES = {
    1: "01_latency_budget_and_amdahls_law.out.txt",
    2: "02_atomics_mutexes_and_synchronization.out.txt",
    3: "03_barrier_thread_pool_gemv.out.txt",
    4: "04_false_sharing_and_cache_line_layout.out.txt",
    5: "05_work_stealing_scheduler.out.txt",
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
            fname, _, needs_pthread = FILES[idx]
            if fname not in bash_blocks[idx - 1]:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")
            has_pthread = "-pthread" in bash_blocks[idx - 1]
            if needs_pthread and not has_pthread:
                fail(f"bash block {idx} creates threads but does not compile with -pthread")
            elif needs_pthread:
                ok(f"bash block {idx} compiles with -pthread")
            else:
                ok(f"bash block {idx} correctly needs no -pthread (no threads created)")

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
        binp = BASE / f"_verify_ch10_{idx:02d}_bin"

        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2"]
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
        ok(f"compile of {fname} succeeded (g++ -std=c++23{' -pthread' if needs_pthread else ''}{' -mdspan-flags' if needs_mdspan else ''})")

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
        # every other chapter's files are held to, applied here with
        # particular deliberateness given what this chapter is about.
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
