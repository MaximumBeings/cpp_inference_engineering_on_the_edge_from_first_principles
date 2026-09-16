#!/usr/bin/env python3
"""Build docs/part4/16-production-engine.md from chapter16_template.md by
substituting @@CODEn@@ with each section's locked source file (fenced as
cpp), @@OUTn@@ with each section's locked self-test output file (fenced
as text), and -- for Section 16.4 only, which has a real-file mode
shaped differently from Chapter 15's -- @@OUT4REAL@@ with the documented
real-run output, captured once on a reader's own machine and never
re-executed by this build.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter16_template.md"
OUTPUT = BASE / "docs/part4/16-production-engine.md"

FILES = {
    1: "01_engine_config_and_cli.cpp",
    2: "02_sampling_and_streaming_decoder.cpp",
    3: "03_conversation_loop.cpp",
    4: "04_production_engine.cpp",
}
OUT_FILES = {
    1: "01_engine_config_and_cli.out.txt",
    2: "02_sampling_and_streaming_decoder.out.txt",
    3: "03_conversation_loop.out.txt",
    4: "04_production_engine.out.txt",
}
REAL_OUT_FILE = "04_production_engine.real_run.out.txt"


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

    real_out_path = BASE / REAL_OUT_FILE
    if not real_out_path.exists():
        print(f"ERROR: missing documented real-run output file {REAL_OUT_FILE}", file=sys.stderr)
        return 1
    if "@@OUT4REAL@@" not in template:
        print("ERROR: template missing @@OUT4REAL@@", file=sys.stderr)
        return 1
    template = template.replace("@@OUT4REAL@@", fenced(real_out_path.read_text(), "text"))

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
