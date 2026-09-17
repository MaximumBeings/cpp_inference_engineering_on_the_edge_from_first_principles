#!/usr/bin/env python3
"""Build docs/appendix/e-rosetta-stone.md from appendix_e_template.md.

Unlike every other chapter/appendix build script, this one substitutes
nothing: Appendix E introduces no new compiled source or captured output of
its own (see its own opening paragraph) -- every C++ name it cites is
already locked, compiled, and published inside its own home chapter's own
docs/ page, and re-fencing that code here would create a second, driftable
copy of a claim this book already has exactly one locked copy of. This
script's only job is therefore to copy the template to its published
location and confirm no leftover substitution markers survived a copy/paste
from another appendix's build script.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "appendix_e_template.md"
OUTPUT = BASE / "docs/appendix/e-rosetta-stone.md"


def main() -> int:
    template = TEMPLATE.read_text()

    remaining = re.findall(r"@@\w+@@", template)
    if remaining:
        print(f"ERROR: unexpected substitution markers found: {remaining}", file=sys.stderr)
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(template)
    print(f"Built {OUTPUT} ({len(template)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
