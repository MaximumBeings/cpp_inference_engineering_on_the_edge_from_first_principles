#!/usr/bin/env python3
"""Build docs/appendix/d-profiling-and-benchmarking.md from
appendix_d_template.md by substituting @@CODEn@@ with each section's locked
source file, @@OUTn@@ with each section's locked self-test output, and
@@DIAGn@@ with each real, captured diagnostic transcript.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "appendix_d_template.md"
OUTPUT = BASE / "docs/appendix/d-profiling-and-benchmarking.md"
APPD = BASE / "appendix_d"

FILES = {
    1: "d1_peak_bandwidth_formula.cpp",
    2: "d2_bandwidth_microbenchmark.cpp",
    3: "d3_latency_budget_calculator.cpp",
}
OUT_FILES = {
    1: "d1_peak_bandwidth_formula.out.txt",
    2: "d2_bandwidth_microbenchmark.out.txt",
    3: "d3_latency_budget_calculator.out.txt",
}
DIAG_FILES = {
    "@@DIAG1@@": APPD / "diagnostics/diag1_perf_hardware_events_unsupported.txt",
    "@@DIAG2@@": APPD / "diagnostics/diag2_perf_paranoid_denial_on_device.txt",
    "@@DIAG3@@": APPD / "diagnostics/diag3_real_timing_varies_cloud_sandbox.txt",
    "@@DIAG4@@": APPD / "diagnostics/diag4_real_timing_varies_on_device.txt",
}


def fenced(content: str, lang: str) -> str:
    content = content.rstrip("\n")
    return f"```{lang}\n{content}\n```"


def main() -> int:
    template = TEMPLATE.read_text()

    for idx, fname in FILES.items():
        src_path = APPD / fname
        if not src_path.exists():
            print(f"ERROR: missing locked source file {fname}", file=sys.stderr)
            return 1
        marker = f"@@CODE{idx}@@"
        if marker not in template:
            print(f"ERROR: template missing {marker}", file=sys.stderr)
            return 1
        template = template.replace(marker, fenced(src_path.read_text(), "cpp"))

    for idx, fname in OUT_FILES.items():
        out_path = APPD / fname
        if not out_path.exists():
            print(f"ERROR: missing locked output file {fname}", file=sys.stderr)
            return 1
        marker = f"@@OUT{idx}@@"
        if marker not in template:
            print(f"ERROR: template missing {marker}", file=sys.stderr)
            return 1
        template = template.replace(marker, fenced(out_path.read_text(), "text"))

    for marker, path in DIAG_FILES.items():
        if not path.exists():
            print(f"ERROR: missing diagnostic file {path}", file=sys.stderr)
            return 1
        if marker not in template:
            print(f"ERROR: template missing {marker}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8", errors="replace")
        template = template.replace(marker, fenced(text, "text"))

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
