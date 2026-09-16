#!/usr/bin/env python3
"""Verify docs/part4/16-production-engine.md.

This chapter's honest-exception shape differs from Chapter 15's. Sections
16.1-16.3 have NO real-file mode at all -- each is pure synthetic
self-test, verified exactly like every non-honest-exception chapter's own
file: full recompile, full rerun, full comparison against the locked
output, confirmed deterministic across two fresh runs.

Section 16.4 has a real-file mode, but shaped differently from Sections
15.1, 15.2, and 15.4: those files ran self-tests UNCONDITIONALLY and then
optionally appended a real-generation section in the SAME run when given
a path, so their locked .out.txt could be split by excising the real
section's own marker. Section 16.4's real invocation IS this chapter's
actual CLI contract (`<model.gguf> -p "..." [options]`) and cannot
coexist with a no-argument self-test in one run -- so self-test mode
(argc==1) and real mode are two completely separate invocations, each
with its own separate output file. This script recompiles
04_production_engine.cpp and reruns it with NO arguments, comparing that
output exactly against 04_production_engine.out.txt (confirmed
deterministic across two fresh runs, same as every other section here).
The real-run output (04_production_engine.real_run.out.txt) is verified
only as data: confirmed present in the built markdown byte for byte, and
confirmed to look like what the real run's own banner and profiler
summary should look like -- it is not (cannot be) re-executed here,
since that requires the actual ~644 MB downloaded checkpoint and several
minutes of real compute on a reader's own machine.

Sections 1 and 2 compile with plain flags (no transformer-block or
mdspan code); Sections 3 and 4 compile with -ffp-contract=off and the
vendored mdspan headers, exactly like Chapter 15's Sections 3 and 4.
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part4/16-production-engine.md"

EXPECTED_H1 = "# Chapter 16: The Production Engine -- Integrating Everything into a Single Binary"
EXPECTED_H2 = [
    "## 16.1 Engine Configuration and the Command Line",
    "## 16.2 Sampling and a UTF-8-Safe Streaming Decoder",
    "## 16.3 The Generation Loop and the Conversation Loop",
    "## 16.4 The Complete Production Engine and Its Built-In Profiler",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

FILES = {
    1: "01_engine_config_and_cli.cpp",
    2: "02_sampling_and_streaming_decoder.cpp",
    3: "03_conversation_loop.cpp",
    4: "04_production_engine.cpp",
}
OUT_FILES = {
    1: "01_engine_config_and_cli.out.txt",
    2: "02_sampling_and_streaming_decoder.out.txt",
    3: "03_conversation_loop.out.txt",
    4: "04_production_engine.out.txt",
}
REAL_OUT_FILE = "04_production_engine.real_run.out.txt"
REAL_RUN_MARKER = "=== REAL PRODUCTION RUN"

COMPILE_FLAGS = {
    1: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    2: ["-std=c++23", "-Wall", "-Wextra", "-O2"],
    3: ["-std=c++23", "-Wall", "-Wextra", "-O2", "-ffp-contract=off",
        "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std", "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
        "-I_vendor_mdspan/include"],
    4: ["-std=c++23", "-Wall", "-Wextra", "-O2", "-ffp-contract=off",
        "-DMDSPAN_IMPL_STANDARD_NAMESPACE=std", "-DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental",
        "-I_vendor_mdspan/include"],
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

    # 'é' is not stray formatting: Section 16.2's own worked example is
    # specifically about a real GPT-2 byte-fallback tokenizer splitting
    # this exact character's two-byte UTF-8 encoding (0xC3 0xA9) across
    # two separate tokens, and the chapter's prose names the character
    # itself rather than describing it abstractly -- the same kind of
    # necessary domain content Chapter 15's own verify script allowed for
    # GPT-2's U+0120 space glyph, not a smart-quote- or em-dash-style
    # formatting artifact.
    ALLOWED_NON_ASCII = {"—", "é"}
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
            needs_mdspan = idx in (3, 4)
            has_mdspan = "MDSPAN_IMPL_STANDARD_NAMESPACE" in block and "-ffp-contract=off" in block
            if needs_mdspan and not has_mdspan:
                fail(f"bash block {idx} is section {idx}, which requires -ffp-contract=off and mdspan flags")
            if not needs_mdspan and ("-ffp-contract=off" in block or "MDSPAN_IMPL_STANDARD_NAMESPACE" in block):
                fail(f"bash block {idx} unexpectedly compiles with mdspan/-ffp-contract flags "
                     f"(section {idx} has no transformer-block code requiring them)")
            if idx == 4:
                if ".gguf" not in block:
                    fail("bash block 4 is Section 16.4's real-file section but shows no real-file invocation")
                if "--verbose" not in block or "-p " not in block:
                    fail("bash block 4's real invocation should show this chapter's own real CLI flags (-p, --verbose)")

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

    # 5 text blocks: one self-test output per section (4), plus Section
    # 16.4's separately-documented real-run output.
    if len(text_blocks) != 5:
        fail(f"expected exactly 5 text output blocks (4 self-test + 1 real-run), found {len(text_blocks)}")
    else:
        ok("found exactly 5 text output blocks")
        for idx in range(1, 5):
            fname = OUT_FILES[idx]
            locked = (BASE / fname).read_text().rstrip("\n")
            got = text_blocks[idx - 1].rstrip("\n")
            if got != locked:
                fail(f"output block {idx} does not exactly match locked file {fname}")
            else:
                ok(f"output block {idx} matches locked {fname} exactly")

        real_locked = (BASE / REAL_OUT_FILE).read_text().rstrip("\n")
        real_got = text_blocks[4].rstrip("\n")
        if real_got != real_locked:
            fail(f"real-run output block does not exactly match documented file {REAL_OUT_FILE}")
        else:
            ok(f"real-run output block matches documented {REAL_OUT_FILE} exactly (verified as data, not re-executed)")
        if REAL_RUN_MARKER not in real_locked:
            fail(f"documented real-run output is missing its own banner {REAL_RUN_MARKER!r}")
        else:
            ok("documented real-run output contains its honest-exception real-run banner")
        if "dim=896" not in real_locked or "capital of france" not in real_locked.lower():
            fail("documented real-run output does not look like this chapter's own real conversation "
                 "against the real Qwen2.5-0.5B-Instruct checkpoint")
        else:
            ok("documented real-run output matches this chapter's own described real conversation")

    tmp_bins = []
    for idx, fname in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch16_{idx:02d}_bin"

        cmd = ["g++"] + COMPILE_FLAGS[idx] + [str(src), "-o", str(binp)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded ({' '.join(COMPILE_FLAGS[idx])})")

        locked_expected = (BASE / OUT_FILES[idx]).read_text().rstrip("\n")

        # Section 16.4's binary requires an explicit "no arguments" run to
        # hit self-test mode -- identical to every other section here,
        # since argc==1 is simply "no arguments" either way. Real mode
        # (argc>1) is Section 16.4's actual CLI contract and is never
        # invoked by this script.
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
