#!/usr/bin/env python3
"""Build docs/part6/26-mathematical-foundations.md from chapter26_template.md
by substituting @@CODEn@@ with each section's locked source file (fenced as
cpp) and @@OUTn@@ with each section's locked self-test output file (fenced
as text).
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter26_template.md"
OUTPUT = BASE / "docs/part6/26-mathematical-foundations.md"

FILES = {
    1: "01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover.cpp",
    2: "02_numerically_stable_softmax_and_log_sum_exp.cpp",
    3: "03_rope_rotation_math.cpp",
    4: "04_quantization_as_affine_algebra.cpp",
    5: "05_hessian_role_in_gptq.cpp",
    6: "06_full_transformer_layer_roofline_analysis.cpp",
}
OUT_FILES = {
    1: "01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover.out.txt",
    2: "02_numerically_stable_softmax_and_log_sum_exp.out.txt",
    3: "03_rope_rotation_math.out.txt",
    4: "04_quantization_as_affine_algebra.out.txt",
    5: "05_hessian_role_in_gptq.out.txt",
    6: "06_full_transformer_layer_roofline_analysis.out.txt",
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
