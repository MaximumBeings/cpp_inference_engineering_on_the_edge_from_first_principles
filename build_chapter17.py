#!/usr/bin/env python3
"""Build docs/part5/17-benchmarking.md from chapter17_template.md by
substituting @@CODEn@@ with each section's locked source file (fenced as
cpp), @@OUTn@@ with each section's locked self-test output file (fenced
as text), and -- for Section 17.3 only, which has a real-file mode shaped
like Chapter 16.4's (a separate invocation, not an appended block) --
@@OUT3REAL@@ with the documented real-run output, captured once on a
reader's own machine and never re-executed by this build.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter17_template.md"
OUTPUT = BASE / "docs/part4/17-benchmarking.md"

FILES = {
    1: "01_benchmark_stats_and_definitions.cpp",
    2: "02_benchmark_harness.cpp",
    3: "03_llamacpp_comparison.cpp",
}
OUT_FILES = {
    1: "01_benchmark_stats_and_definitions.out.txt",
    2: "02_benchmark_harness.out.txt",
    3: "03_llamacpp_comparison.out.txt",
}
REAL_OUT_FILE = "03_llamacpp_comparison.real_run.out.txt"


def fenced(content: str, lang: str) -> str:
    content = content.rstrip("\n")
    return f"```{lang}\n{content}\n```"


def main() -> int:
    template = TEMPLATE.read_text()

    for idx, fname in FILES.items():
        src_path = BASE / fname
        if not src_path.exists():
            print(f"ERROR: missing locked source file {fname}", file=sys.stderr)
            return 1
        marker = f"@@CODE{idx}@@"
        if marker not in template:
            print(f"ERROR: template missing {marker}", file=sys.stderr)
            return 1
        template = template.replace(marker, fenced(src_path.read_text(), "cpp"))

    for idx, fname in OUT_FILES.items():
        out_path = BASE / fname
        if not out_path.exists():
            print(f"ERROR: missing locked output file {fname}", file=sys.stderr)
            return 1
        marker = f"@@OUT{idx}@@"
        if marker not in template:
            print(f"ERROR: template missing {marker}", file=sys.stderr)
            return 1
        template = template.replace(marker, fenced(out_path.read_text(), "text"))

    real_out_path = BASE / REAL_OUT_FILE
    if not real_out_path.exists():
        print(f"ERROR: missing documented real-run output file {REAL_OUT_FILE}", file=sys.stderr)
        return 1
    if "@@OUT3REAL@@" not in template:
        print("ERROR: template missing @@OUT3REAL@@", file=sys.stderr)
        return 1
    template = template.replace("@@OUT3REAL@@", fenced(real_out_path.read_text(), "text"))

    remaining = re.findall(r"@@\w+@@", template)
    if remaining:
        print(f"ERROR: unsubstituted markers remain: {remaining}", file=sys.stderr)
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(template)
    print(f"Built {OUTPUT} ({len(template)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
