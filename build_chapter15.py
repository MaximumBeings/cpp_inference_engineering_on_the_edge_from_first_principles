#!/usr/bin/env python3
"""Build docs/part4/15-running-real-models.md from chapter15_template.md
by substituting @@CODEn@@ with each section's locked source file (fenced
as cpp) and @@OUTn@@ with each section's locked output file (fenced as
text).
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter15_template.md"
OUTPUT = BASE / "docs/part4/15-running-real-models.md"

FILES = {
    1: "01_real_gguf_inspection.cpp",
    2: "02_llama_vs_qwen2_diff.cpp",
    3: "03_qwen2_transformer_block.cpp",
    4: "04_first_real_token.cpp",
}
OUT_FILES = {
    1: "01_real_gguf_inspection.out.txt",
    2: "02_llama_vs_qwen2_diff.out.txt",
    3: "03_qwen2_transformer_block.out.txt",
    4: "04_first_real_token.out.txt",
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
