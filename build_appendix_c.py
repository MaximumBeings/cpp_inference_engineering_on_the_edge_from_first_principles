#!/usr/bin/env python3
"""Build docs/appendix/c-decision-tree-reference.md from appendix_c_template.md
by substituting @@CODEn@@ with each section's locked advisor source file and
@@OUTn@@ with each section's locked, real captured self-test output.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "appendix_c_template.md"
OUTPUT = BASE / "docs/appendix/c-decision-tree-reference.md"
APPC = BASE / "appendix_c"

FILES = {
    1: "c1_quantization_format_advisor.cpp",
    2: "c2_threading_strategy_advisor.cpp",
    3: "c3_kv_cache_strategy_advisor.cpp",
}
OUT_FILES = {
    1: "c1_quantization_format_advisor.out.txt",
    2: "c2_threading_strategy_advisor.out.txt",
    3: "c3_kv_cache_strategy_advisor.out.txt",
}


def fenced(content: str, lang: str) -> str:
    content = content.rstrip("\n")
    return f"```{lang}\n{content}\n```"


def main() -> int:
    template = TEMPLATE.read_text()

    for idx, fname in FILES.items():
        src_path = APPC / fname
        if not src_path.exists():
            print(f"ERROR: missing locked source file {fname}", file=sys.stderr)
            return 1
        marker = f"@@CODE{idx}@@"
        if marker not in template:
            print(f"ERROR: template missing {marker}", file=sys.stderr)
            return 1
        template = template.replace(marker, fenced(src_path.read_text(), "cpp"))

    for idx, fname in OUT_FILES.items():
        out_path = APPC / fname
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
