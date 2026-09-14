#!/usr/bin/env python3
"""Verify docs/part3/14-advanced-kv-cache-management.md.

Like Chapters 12 and 13, this chapter is pure single-threaded C++: no
-pthread, no std::mdspan, no -ffp-contract=off for any file. Each file
compiles with just -std=c++23 -Wall -Wextra -O2, and each binary is run
twice per invocation to confirm determinism. This chapter has 4 numbered
sections instead of the usual 5 -- a legitimate deviation precedented by
Chapters 03 and 04 already having 6 -- because the TOC's own chapter
description names exactly 3 new techniques plus a capstone.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part3/14-advanced-kv-cache-management.md"

EXPECTED_H1 = "# Chapter 14: Advanced KV Cache Management -- Sliding Windows, Streaming Re-Quantization, and Prefix Caching"
EXPECTED_H2 = [
    "## 14.1 Sliding-Window Attention as a Ring Buffer at Scale",
    "## 14.2 Streaming Re-Quantization: Tiered Precision as Entries Age",
    "## 14.3 Prefix Caching for Multi-Turn Conversations",
    "## 14.4 A Complete Cache Manager Combining Every Technique",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_sliding_window_cache.cpp",
    2: "02_streaming_requantization.cpp",
    3: "03_prefix_caching.cpp",
    4: "04_complete_cache_manager.cpp",
}
OUT_FILES = {
    1: "01_sliding_window_cache.out.txt",
    2: "02_streaming_requantization.out.txt",
    3: "03_prefix_caching.out.txt",
    4: "04_complete_cache_manager.out.txt",
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
        ok("H2 sequence matches (4 numbered sections, precedented by Chapters 03/04 deviating from 5)")

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
            for forbidden in ("-pthread", "-ffp-contract=off", "MDSPAN_IMPL_STANDARD_NAMESPACE"):
                if forbidden in block:
                    fail(f"bash block {idx} unexpectedly compiles with {forbidden} "
                         f"(this chapter is pure single-threaded data-structure code)")
            if "-std=c++23" not in block:
                fail(f"bash block {idx} does not compile with -std=c++23")

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

    tmp_bins = []
    for idx, fname in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch14_{idx:02d}_bin"

        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2", str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded (g++ -std=c++23 -Wall -Wextra -O2)")

        r2 = subprocess.run([str(binp)], capture_output=True, text=True)
        fresh = r2.stdout.rstrip("\n")
        locked = (BASE / OUT_FILES[idx]).read_text().rstrip("\n")
        if fresh != locked:
            fail(f"fresh rerun of {fname} does not match locked output")
        else:
            ok(f"fresh rerun of {fname} matches locked output exactly (deterministic)")
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
