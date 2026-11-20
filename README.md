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
