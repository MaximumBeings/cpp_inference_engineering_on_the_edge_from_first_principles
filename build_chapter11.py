#!/usr/bin/env python3
"""Build docs/part2/11-parallelizing-the-forward-pass.md from
chapter11_template.md by substituting @@CODEn@@ with each section's locked
source file (fenced as cpp) and @@OUTn@@ with each section's locked output
file (fenced as text).
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter11_template.md"
OUTPUT = BASE / "docs/part2/11-parallelizing-the-forward-pass.md"

FILES = {
    1: "01_row_vs_column_parallel_matmul.cpp",
    2: "02_persistent_pool_full_layer.cpp",
    3: "03_head_parallel_attention.cpp",
    4: "04_deterministic_reduction.cpp",
    5: "05_full_decode_step_multithreaded.cpp",
}
OUT_FILES = {
    1: "01_row_vs_column_parallel_matmul.out.txt",
    2: "02_persistent_pool_full_layer.out.txt",
    3: "03_head_parallel_attention.out.txt",
    4: "04_deterministic_reduction.out.txt",
    5: "05_full_decode_step_multithreaded.out.txt",
}


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
