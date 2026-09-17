#!/usr/bin/env python3
"""Build docs/part5/27-low-latency-market-signal-inference.md from
chapter27_template.md by substituting @@CODEn@@ with each section's locked
source file (fenced as cpp) and @@OUTn@@ with each section's locked
self-test output file (fenced as text).
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "chapter27_template.md"
OUTPUT = BASE / "docs/part5/27-low-latency-market-signal-inference.md"

FILES = {
    1: "01_order_book_microstructure_features.cpp",
    2: "02_tick_to_trade_deadline_enforcement.cpp",
    3: "03_financial_news_sentiment_scoring.cpp",
    4: "04_signal_fusion_and_trading_decision_engine.cpp",
}
OUT_FILES = {
    1: "01_order_book_microstructure_features.out.txt",
    2: "02_tick_to_trade_deadline_enforcement.out.txt",
    3: "03_financial_news_sentiment_scoring.out.txt",
    4: "04_signal_fusion_and_trading_decision_engine.out.txt",
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
