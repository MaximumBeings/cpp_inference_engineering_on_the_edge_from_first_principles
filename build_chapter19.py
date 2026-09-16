#!/usr/bin/env python3
"""Build docs/part5/19-retail-inventory.md from chapter19_template.md by
substituting @@CODEn@@ with each section's locked source file (fenced as
cpp) and @@OUTn@@ with each section's locked self-test output file
(fenced as text).

Every section in this chapter is pure, deterministic self-test against
synthetic fixtures -- no section has a separate real-file mode, and (unlike
Chapter 18.4's SQLite dependency) no section here needed a reduced
cross-architecture check either.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter19_template.md"
OUTPUT = BASE / "docs/part5/19-retail-inventory.md"

FILES = {
    1: "01_planogram_and_system_prompt.cpp",
    2: "02_batch_shelf_photo_processor.cpp",
    3: "03_erp_wms_rest_integration.cpp",
    4: "04_multistore_aggregation_and_trends.cpp",
    5: "05_shrinkage_and_misplacement_detection.cpp",
}
OUT_FILES = {
    1: "01_planogram_and_system_prompt.out.txt",
    2: "02_batch_shelf_photo_processor.out.txt",
    3: "03_erp_wms_rest_integration.out.txt",
    4: "04_multistore_aggregation_and_trends.out.txt",
    5: "05_shrinkage_and_misplacement_detection.out.txt",
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
