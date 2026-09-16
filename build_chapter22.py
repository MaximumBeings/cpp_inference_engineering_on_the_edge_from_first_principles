#!/usr/bin/env python3
"""Build docs/part5/22-security-accessibility-art.md from
chapter22_template.md by substituting @@CODEn@@ with each section's locked
source file (fenced as cpp) and @@OUTn@@ with each section's locked
self-test output file (fenced as text).

Every section in this chapter is pure, deterministic self-test against
synthetic fixtures and synthetic data only -- no section here has an
external system dependency, so this script recompiles and reruns all three
sections without any documented reduced-verification exception.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter22_template.md"
OUTPUT = BASE / "docs/part5/22-security-accessibility-art.md"

FILES = {
    1: "01_surveillance_narration_and_multi_camera_correlation.cpp",
    2: "02_accessibility_compliance_audit.cpp",
    3: "03_art_provenance_and_condition_trend.cpp",
}
OUT_FILES = {
    1: "01_surveillance_narration_and_multi_camera_correlation.out.txt",
    2: "02_accessibility_compliance_audit.out.txt",
    3: "03_art_provenance_and_condition_trend.out.txt",
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
