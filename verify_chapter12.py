#!/usr/bin/env python3
"""Verify docs/part3/12-the-tokenizer.md.

This chapter is a deliberate simplification after three chapters (9, 10,
11) that each needed at least one of -pthread, mdspan flags, or
-ffp-contract=off: every file here is pure single-threaded string/vector/
map code, so no file's compile command needs any of those. Each file
compiles with just the book's standard -std=c++23 -Wall -Wextra -O2.

As in every prior chapter, each binary is run twice per invocation to
confirm determinism, even though nothing here touches real threads --
the discipline is applied uniformly rather than judged file by file.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part3/12-the-tokenizer.md"

EXPECTED_H1 = "# Chapter 12: The Tokenizer -- BPE, SentencePiece, and the Vocabulary Pipeline"
EXPECTED_H2 = [
    "## 12.1 Byte-Pair Encoding: The Merge Algorithm",
    "## 12.2 Pre-Tokenization: Splitting Text Before BPE",
    "## 12.3 Special Tokens and the Chat Template",
    "## 12.4 The Token Decoder: IDs Back to UTF-8",
    "## 12.5 A Complete Tokenizer Pipeline Loaded from a Real GGUF File",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_bpe_merge_engine.cpp",
    2: "02_pretokenizer.cpp",
    3: "03_special_tokens_chat_template.cpp",
    4: "04_token_decoder.cpp",
    5: "05_gguf_tokenizer_pipeline.cpp",
}
OUT_FILES = {
    1: "01_bpe_merge_engine.out.txt",
    2: "02_pretokenizer.out.txt",
    3: "03_special_tokens_chat_template.out.txt",
    4: "04_token_decoder.out.txt",
    5: "05_gguf_tokenizer_pipeline.out.txt",
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
            fname = FILES[idx]
            block = bash_blocks[idx - 1]
            if fname not in block:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")
            # This chapter needs none of the flags prior chapters required --
            # check that none of them crept in by copy-paste from Chapter 11.
            for forbidden in ("-pthread", "-ffp-contract=off", "MDSPAN_IMPL_STANDARD_NAMESPACE"):
                if forbidden in block:
                    fail(f"bash block {idx} unexpectedly compiles with {forbidden} "
                         f"(this chapter is pure single-threaded string/vector/map code)")
            if "-std=c++23" not in block:
                fail(f"bash block {idx} does not compile with -std=c++23")

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

    tmp_bins = []
    for idx, fname in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch12_{idx:02d}_bin"

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
