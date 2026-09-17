#!/usr/bin/env python3
"""Build docs/appendix/a-installation-and-setup.md from appendix_a_template.md
by substituting @@CODE_*@@ with each locked source/config file and @@OUT_*@@ /
@@DIAG*@@ with each locked, real captured output/diagnostic file.
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "appendix_a_template.md"
OUTPUT = BASE / "docs/appendix/a-installation-and-setup.md"
APPA = BASE / "appendix_a"

CODE_FILES = {
    "@@CODE_SRC@@": (APPA / "src/portable_dot_product.cpp", "cpp"),
    "@@CODE_CMAKE@@": (APPA / "CMakeLists.txt", "cmake"),
    "@@CODE_TOOLCHAIN@@": (APPA / "cmake/toolchain-aarch64-linux-gnu.cmake", "cmake"),
}

OUT_FILES = {
    "@@OUT_NATIVE@@": APPA / "portable_dot_product.native.out.txt",
    "@@OUT_AARCH64@@": APPA / "portable_dot_product.aarch64.out.txt",
    "@@DIAG1@@": APPA / "diagnostics/diag1_excerpt.txt",
    "@@DIAG2@@": APPA / "diagnostics/diag2_no_native_mdspan.stderr.txt",
    "@@DIAG3@@": APPA / "diagnostics/diag3_ctest_no_emulator.stdout.txt",
    "@@DIAG4@@": APPA / "diagnostics/diag4_qemu_missing_linker.stdout.txt",
    "@@DIAG5@@": APPA / "diagnostics/diag5_nvcc_cpp23.stderr.txt",
}


def fenced(content: str, lang: str) -> str:
    content = content.rstrip("\n")
    return f"```{lang}\n{content}\n```"


def main() -> int:
    template = TEMPLATE.read_text()

    for marker, (path, lang) in CODE_FILES.items():
        if not path.exists():
            print(f"ERROR: missing locked source file {path}", file=sys.stderr)
            return 1
        if marker not in template:
            print(f"ERROR: template missing {marker}", file=sys.stderr)
            return 1
        template = template.replace(marker, fenced(path.read_text(), lang))

    for marker, path in OUT_FILES.items():
        if not path.exists():
            print(f"ERROR: missing locked output file {path}", file=sys.stderr)
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
