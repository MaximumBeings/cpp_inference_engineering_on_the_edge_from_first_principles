# Website and C++ audit

**Technical accuracy follow-up:** The [comprehensive technical review](TECHNICAL_ACCURACY_AUDIT.md) found 26 open issues beyond the execution checks documented below. A passing test run does not clear those semantic, interoperability, benchmark or standards findings.

Reviewed the published homepage and repository at commit `cff737b6cec58dd20f8ccba3ed98f22155c3faf1`. The fixes and validation evidence are included in this checkout.

## Fixed

| Finding | Correction |
| --- | --- |
| CLI parsers accept NaN, partial numbers such as `12junk`, invalid unsigned seeds, and unchecked integer overflow. | Use `std::from_chars`, require complete conversion, and reject non-finite floating-point values in both the standalone parser and production engine. |
| Repetition penalty applies again for every occurrence of the same token in history. | Penalize each distinct valid token once in both sampler implementations. |
| Threading example relies on a transitive declaration of `std::ref`. | Include `<functional>` directly and regenerate Chapter 10. |
| Camera example converts `2.0 * 255` to `uint8_t` in its self-check, outside the representable range. | Verify that a zero red channel stays zero and regenerate Chapter 21. |
| Getting Started creates `mdspan/` while verification scripts expect `_vendor_mdspan/`. | Align the clone, pinned checkout, and include directory with the scripts. |
| Setup claims all host files need no extra flags or libraries. | Document threading, SQLite, architecture, POSIX, compiler, and Linux memory-check requirements. |
| Setup makes an unqualified, time-sensitive claim that no shipping library supports mdspan. | State the requirement for the book's GCC 13/14 configuration and describe version-dependent support. |
| Homepage references a missing Chapter 1 and overstates the deployment examples. | Start the foundations reference at Chapter 2 and describe worked deployment examples and integration designs. |
| Website lacks a canonical base URL and repository/source links. | Set `site_url`, `repo_url`, and `edit_uri`. |
| Documentation build has no dependency specification or contributor instructions. | Add pinned documentation requirements and build/regeneration instructions. |

Generated chapter pages were rebuilt from the source files. Existing captured transcripts for the corrected standalone CLI, sampling, threading, and camera examples still match their fresh output.

## Validation

- Strict MkDocs build passes. An HTML crawl checks 39 generated pages and finds no broken relative file links or anchors.
- Compiled and ran 113 root C++ examples using Apple Clang on arm64, excluding architecture-specific requirements where detected. Initial results were 101 passing, two compile failures, six runtime failures, and four platform skips.
- The missing-header example and camera self-check now pass. The SQLite example passes with the documented macOS `-lsqlite3` flag.
- Targeted regression harnesses pass for malformed numeric arguments and repeated-history penalties in both standalone and production copies.
- The camera example passes UndefinedBehaviorSanitizer and float-cast-overflow checks.
- The production engine passes its 29 built-in checks using the pinned mdspan headers.
- Appendix E's existing citation verification passes.

## A100 and GCC/Linux follow-up

Validated on an NVIDIA A100-SXM4-40GB, driver 580.105.08, CUDA toolkit 12.8, and GCC 13.3. Logs are saved in `validation/a100/`.

The CUDA audit exposed additional defects:

- Device discovery self-tests incorrectly asserted that a physical GPU must be absent.
- The device test accepted both `PASSED` and `FAILED` comparison statuses.
- CUDA allocation, transfer, launch, and synchronization errors were unchecked.
- The numerical comparison could accept NaN because comparisons with NaN return false.
- Unsupported head dimensions could overrun the fixed per-thread register array.
- The verifier compared device execution with a transcript captured on a machine without a GPU and fixed compilation to Jetson's `sm_87`.

Corrected all of these, added 60 attention shapes covering partial key tiles and query blocks, and validated unsupported dimensions and non-finite comparisons. All 71 checks pass on the A100. Compute Sanitizer memcheck, racecheck, and synccheck report no errors. A separate run with CUDA devices hidden passes the host checks and explicitly skips device comparisons. These small correctness cases do not establish performance or arbitrary-input numerical accuracy.

The full chapter verification run completed 31 scripts, including the inference regression harness. Appendix A initially required unavailable build tools. With CMake, Ninja, the Arm cross-compiler and QEMU installed, its diagnostic also exposed an assumption that Arm execution without an explicit CMake emulator must fail. The instance registers QEMU through `binfmt_misc`, so that binary succeeds. The verifier now recognizes this valid configuration and the appendix explains it. The final appendix rerun passes. Chapter 9's explicit Arm-emulator run passes as well. Chapter 32's separate GPU verifier passes.

All 27 embedded C++ examples in Chapters 2–6 also compile and run on GCC/Linux. The prior Linux-only memory, stochastic transcript, and compiler-dependent demonstration failures observed on macOS pass in the canonical Linux environment.

## Remaining validation limits

- The real-model checkpoint was not downloaded for a fresh end-to-end Qwen generation or throughput benchmark.
- Physical Jetson execution was not tested. Arm execution used QEMU on x86-64.
- The original macOS differences remain documented as portability limits.
- Internal link checks cover generated pages, not every external reference. No full mathematical proof audit or browser accessibility certification was performed.

## Publication

The source branch contains the corrected examples, generated Markdown, and validation evidence. The GitHub Pages branch is built from the validated MkDocs configuration.
