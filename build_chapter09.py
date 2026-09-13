#!/usr/bin/env python3
"""Build docs/part2/09-simd-vectorization-avx2-neon.md from chapter09_template.md
by substituting @@CODEn@@ with each section's locked source file (fenced as
cpp) and @@OUTn@@ with each section's locked output file (fenced as text).
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter09_template.md"
OUTPUT = BASE / "docs/part2/09-simd-vectorization-avx2-neon.md"

FILES = {
    1: "01_simd_register_model.cpp",
    2: "02_avx2_quantized_dot_product.cpp",
    3: "03_neon_quantized_dot_product.cpp",
    4: "04_restrict_and_autovectorization.cpp",
    5: "05_runtime_dispatch.cpp",
}
OUT_FILES = {
    1: "01_simd_register_model.out.txt",
    2: "02_avx2_quantized_dot_product.out.txt",
    3: "03_neon_quantized_dot_product.out.txt",
    4: "04_restrict_and_autovectorization.out.txt",
    5: "05_runtime_dispatch.out.txt",
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
