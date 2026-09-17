#!/usr/bin/env python3
"""Build docs/part6/32-flash-attention-and-cuda-kernels.md from
chapter32_template.md by substituting @@CODEn@@ with each section's locked
source file (fenced as cpp, or cuda for Section 3's real .cu file) and
@@OUTn@@ with each section's locked self-test output file (fenced as text).
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter32_template.md"
OUTPUT = BASE / "docs/part6/32-flash-attention-and-cuda-kernels.md"

FILES = {
    1: ("01_memory_wall_and_online_softmax.cpp", "cpp"),
    2: ("02_mdspan_flash_attention_implementation_and_benchmark.cpp", "cpp"),
    3: ("03_cuda_kernel_and_validation_suite.cu", "cuda"),
}
OUT_FILES = {
    1: "01_memory_wall_and_online_softmax.out.txt",
    2: "02_mdspan_flash_attention_implementation_and_benchmark.out.txt",
    3: "03_cuda_kernel_and_validation_suite.out.txt",
}


def fenced(content: str, lang: str) -> str:
    content = content.rstrip("\n")
    return f"```{lang}\n{content}\n```"


def main() -> int:
    template = TEMPLATE.read_text()

    for idx, (fname, lang) in FILES.items():
        src_path = BASE / fname
        if not src_path.exists():
            print(f"ERROR: missing locked source file {fname}", file=sys.stderr)
            return 1
        marker = f"@@CODE{idx}@@"
        if marker not in template:
            print(f"ERROR: template missing {marker}", file=sys.stderr)
            return 1
        template = template.replace(marker, fenced(src_path.read_text(), lang))

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
