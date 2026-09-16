#!/usr/bin/env python3
"""Build docs/part5/18-industrial-visual-inspection.md from
chapter18_template.md by substituting @@CODEn@@ with each section's
locked source file (fenced as cpp) and @@OUTn@@ with each section's
locked self-test output file (fenced as text).

Unlike Chapter 17, no section in this chapter has a separate real-file
mode -- every section here (18.1 through 18.5) is pure, deterministic
self-test against synthetic fixtures, run with no arguments, exactly
like Chapter 16's non-honest-exception sections. Section 18.4's own
cross-architecture verification gap (no aarch64 libsqlite3 available in
this book's own cross-compilation sandbox) is a verification-time
concern, not a build-time one, and is handled by verify_chapter18.py,
not here.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter18_template.md"
OUTPUT = BASE / "docs/part5/18-industrial-visual-inspection.md"

FILES = {
    1: "01_gige_frame_acquisition.cpp",
    2: "02_vision_encoder.cpp",
    3: "03_multimodal_fusion.cpp",
    4: "04_disposition_and_logging.cpp",
    5: "05_opcua_integration.cpp",
}
OUT_FILES = {
    1: "01_gige_frame_acquisition.out.txt",
    2: "02_vision_encoder.out.txt",
    3: "03_multimodal_fusion.out.txt",
    4: "04_disposition_and_logging.out.txt",
    5: "05_opcua_integration.out.txt",
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
