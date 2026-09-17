#!/usr/bin/env python3
"""Build docs/part5/24-natural-language-photo-editing.md from
chapter24_template.md by substituting @@CODEn@@ with each section's locked
source file (fenced as cpp) and @@OUTn@@ with each section's locked
self-test output file (fenced as text).

Section 24.2 is this book's own first file to depend on a real external
system library (OpenCV) rather than the C++ standard library alone; see
this chapter's own COMMON TRAP box under 24.2 and verify_chapter24.py's
own module docstring for what that changes about verification.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter24_template.md"
OUTPUT = BASE / "docs/part5/24-natural-language-photo-editing.md"

FILES = {
    1: "01_edit_interpretation_prompt_and_plan_parser.cpp",
    2: "02_opencv_edit_processing_engine.cpp",
    3: "03_request_to_operation_mapping_table.cpp",
    4: "04_iterative_refinement_conversation.cpp",
}
OUT_FILES = {
    1: "01_edit_interpretation_prompt_and_plan_parser.out.txt",
    2: "02_opencv_edit_processing_engine.out.txt",
    3: "03_request_to_operation_mapping_table.out.txt",
    4: "04_iterative_refinement_conversation.out.txt",
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
