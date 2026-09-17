#!/usr/bin/env python3
"""Verify docs/appendix/e-rosetta-stone.md.

Appendix E introduces no new compiled code of its own -- every C++ name it
cites already has exactly one locked, compiled copy inside its own home
chapter's own docs/ page, and this appendix is not allowed to become a
second, driftable source of truth for any of them. So instead of
recompiling anything, this script does the check that actually matters for
a citation-heavy appendix: it greps each cited chapter's own already-built
markdown page for the exact filename or exact quoted fragment this
appendix's own prose claims lives there, so a citation that drifts out of
sync with its source (a renamed file, an edited quote, a chapter that no
longer says what this appendix says it says) fails loudly rather than
silently. It also checks, as a real negative assertion, that Chapter 31
genuinely never names vLLM or "iteration-level scheduling" -- the exact
honest-gap claim this appendix's own E.5 section makes about its own
source material.
"""
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "docs/appendix/e-rosetta-stone.md"

CH2 = BASE / "docs/part0/02-the-tensor-engine.md"
CH5 = BASE / "docs/part1/05-model-file-formats.md"
CH12 = BASE / "docs/part3/12-the-tokenizer.md"
CH13 = BASE / "docs/part3/13-the-kv-cache-manager.md"
CH14 = BASE / "docs/part3/14-advanced-kv-cache-management.md"
CH15 = BASE / "docs/part4/15-running-real-models.md"
CH16 = BASE / "docs/part4/16-production-engine.md"
CH31 = BASE / "docs/part6/31-continuous-batching-and-production-serving.md"

EXPECTED_H1 = "# Appendix E: From PyTorch and llama.cpp to C++: A Rosetta Stone"
EXPECTED_H2 = [
    "## E.1 Tensors: `std::mdspan` and GGUF/SafeTensors vs. `torch.Tensor` and `AutoModel.from_pretrained`",
    "## E.2 Tokenization: A From-Scratch BPE Pipeline vs. HuggingFace's `tokenizers`",
    "## E.3 KV Cache and PagedAttention: This Book's `BlockManager` vs. vLLM's Real PagedAttention and SGLang's RadixAttention",
    "## E.4 Model Loading and Serving: The Production Engine vs. a Real Python Serving Stack",
    "## E.5 Continuous Batching: This Book's Scheduler vs. Orca's Iteration-Level Scheduling and vLLM's Real Continuous Batching",
    "## Appendix Summary",
    "## Where We Go Next",
]
EXPECTED_H3 = [
    "### Intuition", "### The Concept, In Detail",
    "### Intuition", "### The Concept, In Detail",
    "### Intuition", "### The Concept, In Detail",
    "### Intuition", "### The Concept, In Detail",
    "### Intuition", "### The Concept, In Detail",
]

# (source chapter file, [exact substrings this appendix's own prose claims live there])
CITATION_CHECKS = [
    (CH2, [
        "01_mdspan_views_and_extents.cpp",
        "02_submdspan_attention_slicing.cpp",
        "04_arena_allocator.cpp",
        "05_mixed_precision_bf16.cpp",
    ]),
    (CH5, [
        "02_gguf_writer_reader.cpp",
        "03_zero_copy_weight_access.cpp",
        "04_safetensors_format.cpp",
    ]),
    (CH12, [
        "01_bpe_merge_engine.cpp",
        "02_pretokenizer.cpp",
        "03_special_tokens_chat_template.cpp",
        "04_token_decoder.cpp",
        "05_gguf_tokenizer_pipeline.cpp",
        "priority, not opportunity",
        "covers the same four-way structural idea those richer rules extend",
    ]),
    (CH13, [
        "02_paged_attention_block_manager.cpp",
        "03_ring_buffer_sink_tokens.cpp",
        "04_h2o_eviction.cpp",
        "PagedAttention (the technique vLLM popularized)",
        "16 is the size real systems like vLLM settle on",
    ]),
    (CH14, [
        "01_sliding_window_cache.cpp",
        "02_streaming_requantization.cpp",
        "03_prefix_caching.cpp",
        "simplified symmetric uniform quantizer at multiple",
        "rather than the full rotation-plus-codebook machinery of",
        'This section’s source material describes prefix caching only in prose'.replace("’", "'"),
    ]),
    (CH15, [
        "04_first_real_token.cpp",
        "01_real_gguf_inspection.cpp",
        "GGUF_VERSION = 3",
        "'G','U','G','F'",
    ]),
    (CH16, [
        "04_production_engine.cpp",
        "a flag that parses successfully is a promise, not a convenience",
    ]),
    (CH31, [
        "01_static_batching_problem_and_head_of_line_blocking.cpp",
        "02_prefill_decode_conflict_and_chunked_prefill.cpp",
        "03_continuous_batching_scheduler_from_scratch.cpp",
        "04_nan_propagation_tracing_and_fp_drift_detection.cpp",
    ]),
]

# The exact honest-gap claim E.5 makes about its own source chapter: Chapter 31
# never names vLLM and never uses the phrase "iteration-level scheduling".
CH31_ABSENT_TERMS = ["vLLM", "iteration-level scheduling"]

EXPECTED_FOOTNOTES = 6
EXPECTED_TRAPS = 4

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"OK:   {msg}")


def _lines_outside_code_fences(text):
    out = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return out


def main() -> int:
    if not MD_PATH.exists():
        fail(f"built markdown does not exist: {MD_PATH}")
        return report()

    text = MD_PATH.read_text()
    prose_lines = _lines_outside_code_fences(text)

    h1_lines = [l for l in prose_lines if l.startswith("# ")]
    if len(h1_lines) != 1:
        fail(f"expected exactly 1 H1, found {len(h1_lines)}: {h1_lines}")
    elif h1_lines[0] != EXPECTED_H1:
        fail(f"H1 mismatch: {h1_lines[0]!r} != {EXPECTED_H1!r}")
    else:
        ok("H1 matches")

    h2_lines = [l for l in prose_lines if l.startswith("## ")]
    if h2_lines != EXPECTED_H2:
        fail(f"H2 sequence mismatch:\n  got:      {h2_lines}\n  expected: {EXPECTED_H2}")
    else:
        ok("H2 sequence matches (E.1-E.5 + Summary + Where We Go Next)")

    h3_lines = [l for l in prose_lines if l.startswith("### ")]
    if h3_lines != EXPECTED_H3:
        fail(f"H3 sequence mismatch:\n  got:      {h3_lines}\n  expected: {EXPECTED_H3}")
    else:
        ok("H3 sequence matches (Intuition + The Concept, In Detail for each of E.1-E.5)")

    ALLOWED_NON_ASCII = {"—"}  # em-dash
    bad_chars = {}
    for ch in text:
        if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
            bad_chars[ch] = bad_chars.get(ch, 0) + 1
    if bad_chars:
        fail(f"non-ASCII characters found (other than em-dash): {bad_chars}")
    else:
        ok("no disallowed non-ASCII characters")

    trap_count = len(re.findall(r'!!! warning "\[COMMON TRAP\]', text))
    if trap_count != EXPECTED_TRAPS:
        fail(f"expected exactly {EXPECTED_TRAPS} [COMMON TRAP] boxes, found {trap_count}")
    else:
        ok(f"found exactly {trap_count} [COMMON TRAP] boxes")

    footnote_count = len(re.findall(r"^\[\^\w+\]:", text, re.MULTILINE))
    if footnote_count != EXPECTED_FOOTNOTES:
        fail(f"expected exactly {EXPECTED_FOOTNOTES} footnote definitions, found {footnote_count}")
    else:
        ok(f"found exactly {footnote_count} real external-source footnote citations")

    # No fenced cpp/text/bash code blocks at all -- this appendix is deliberately
    # prose + tables + citations, introducing no new compiled or captured content.
    fence_re = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
    blocks = fence_re.findall(text)
    code_like = [lang for lang, _ in blocks if lang in ("cpp", "bash", "text")]
    if code_like:
        fail(f"expected zero cpp/bash/text code blocks (this appendix cites, never re-fences, "
             f"existing locked code), found: {code_like}")
    else:
        ok("found zero cpp/bash/text code blocks, as expected for a citation-only appendix")

    for path, needles in CITATION_CHECKS:
        if not path.exists():
            fail(f"cited source chapter does not exist: {path}")
            continue
        src = path.read_text()
        for needle in needles:
            if needle not in src:
                fail(f"citation drift: {path.name} no longer contains {needle!r}")
            else:
                ok(f"citation confirmed: {path.name} contains {needle!r}")

    ch31_src = CH31.read_text() if CH31.exists() else ""
    for term in CH31_ABSENT_TERMS:
        if term in ch31_src:
            fail(f"E.5's own honest-gap claim is now false: {CH31.name} contains {term!r}")
        else:
            ok(f"confirmed absent as claimed: {CH31.name} still never mentions {term!r}")

    return report()


def report():
    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        return 1
    print("=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
