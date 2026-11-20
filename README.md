# C++ Inference Engineering on the Edge

The book and standalone C++ examples for building inference systems from first principles.

Read the published book at https://maximumbeings.github.io/cpp_inference_engineering_on_the_edge_from_first_principles/.

## Build the book

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-docs.txt
mkdocs build --strict
mkdocs serve
```

## Compile and verify examples

Follow [Getting Started](docs/getting-started.md) for compiler, platform, SQLite, CUDA, and pinned mdspan requirements. The verification scripts expect mdspan at `_vendor_mdspan/include`.

Each `build_chapterNN.py` embeds the chapter's standalone source and captured output into its Markdown template. Regenerate the corresponding chapter whenever an example changes.

```sh
python3 build_chapter16.py
python3 verify_chapter16.py
python3 tests/inference_regressions.py
```

Set `CXX=clang++` for the regression script when using Apple Clang. Historical transcripts and architecture-specific examples use the book's GCC/Linux environment. See [the audit report](AUDIT_REPORT.md) for the scope and remaining platform-specific validation.

## Regenerate illustrations

The book includes 31 chapter-opening graphics, five matching appendix openers, 140 chapter concept diagrams and 20 appendix concept diagrams (one per numbered section), and 38 overview/setup figures. All 234 SVGs share a deep-blue palette and a 760-unit canvas. They are accessible, responsive, and stored in `docs/images/`. Click a section figure to open it at full size.

```sh
python3 tools/build_diagrams.py
python3 tools/illustrations/build.py
mkdocs build --strict
python3 tools/illustrations/verify.py site
```

Section-specific content lives in `tools/illustrations/concepts.py`; the generated manifest records coverage. Both generators update chapter templates so rebuilding embedded code examples preserves the illustrations. `docs/stylesheets/figures.css` controls their presentation. Numerical examples and schematic bars are labeled separately from measured benchmarks.

## Technical review

See [the technical accuracy audit](TECHNICAL_ACCURACY_AUDIT.md) for open correctness findings, primary references and correction order. The machine-readable findings are in `audit/technical-findings-2026-11-20.json`.

Run `python3 tests/sample_output_checks.py` to verify that the expanded tensor samples match actual stdout and that no captured sample contains only pass-count lines.

Accuracy-fix validation records are in `validation/accuracy-fixes/`. Run `python3 tests/book_reference_checks.py` to check published chapter references and the corrected KV complexity explanation.
