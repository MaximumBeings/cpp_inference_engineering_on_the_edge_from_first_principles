#!/usr/bin/env python3
"""Verify docs/part1/07-turboquant.md."""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/part1/07-turboquant.md"

EXPECTED_H1 = "# Chapter 7: TurboQuant -- Online Vector Quantization with Near-Optimal Distortion"
EXPECTED_H2 = [
    "## 7.1 Random Rotation: Turning Worst-Case Vectors into Average-Case Ones",
    "## 7.2 Lloyd-Max Optimal Codebooks and the MSE Quantizer",
    "## 7.3 The Inner-Product Bias Problem and the QJL Fix",
    "## 7.4 Compressing the KV Cache with TurboQuant",
    "## 7.5 TurboQuant vs. Blockwise Quantization, and the Distortion Bounds",
    "## Chapter Summary",
    "## Self-Check Questions",
    "## Where We Go Next",
    "## Worked Solutions",
]

# (filename, needs_mdspan_flags) -- none of this chapter's files need mdspan.
FILES = {
    1: ("01_rotation_and_beta_distribution.cpp", False),
    2: ("02_lloyd_max_mse_quantizer.cpp", False),
    3: ("03_qjl_unbiased_inner_product.cpp", False),
    4: ("04_kv_cache_quantization.cpp", False),
    5: ("05_turboquant_vs_blockwise.cpp", False),
}
OUT_FILES = {
    1: "01_rotation_and_beta_distribution.out.txt",
    2: "02_lloyd_max_mse_quantizer.out.txt",
    3: "03_qjl_unbiased_inner_product.out.txt",
    4: "04_kv_cache_quantization.out.txt",
    5: "05_turboquant_vs_blockwise.out.txt",
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
            fname = FILES[idx][0]
            if fname not in bash_blocks[idx - 1]:
                fail(f"bash block {idx} does not reference source file {fname}")
            else:
                ok(f"bash block {idx} references {fname}")

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

    tmp_bins = []
    for idx, (fname, needs_mdspan) in FILES.items():
        src = BASE / fname
        binp = BASE / f"_verify_ch7_{idx:02d}_bin"

        cmd = ["g++", "-std=c++23", "-Wall", "-Wextra", "-O2"]
        cmd += [str(src), "-o", str(binp)]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"compile of {fname} failed:\n{r.stderr}")
            continue
        ok(f"compile of {fname} succeeded (g++ -std=c++23)")

        r2 = subprocess.run([str(binp)], capture_output=True, text=True)
        fresh = r2.stdout.rstrip("\n")
        locked = (BASE / OUT_FILES[idx]).read_text().rstrip("\n")
        if fresh != locked:
            fail(f"fresh rerun of {fname} does not match locked output")
        else:
            ok(f"fresh rerun of {fname} matches locked output exactly (deterministic)")
        if r2.returncode != 0:
            fail(f"fresh rerun of {fname} exited non-zero ({r2.returncode})")
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
