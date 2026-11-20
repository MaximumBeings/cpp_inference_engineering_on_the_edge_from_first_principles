# Validation Status

The November 20, 2026 [technical accuracy audit](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/main/TECHNICAL_ACCURACY_AUDIT.md) identified 26 correctness and claim-scope issues. The correction pass fixes affected algorithms and constants and narrows overstated production, compatibility and conformance claims. Detailed current checks and remaining limits are recorded in [accuracy-fix validation](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/tree/main/validation/accuracy-fixes). The table below preserves the earlier execution baseline; it is not a claim that every platform was freshly rerun after these corrections.

The current regression checks cover binary16 subnormals and quantizer edge cases, published tensor stdout, CLI/sampling behavior, benchmark arithmetic and chapter references. A successful self-test does not prove arbitrary-file safety, external-format interoperability, production suitability or standards compliance.

| Area | Completed validation | Scope |
| --- | --- | --- |
| Chapters 2-6 | All 27 embedded C++ examples compile and run under GCC/Linux. | The examples' built-in checks, not arbitrary-input proofs. |
| Chapters 7-31 and Appendices A-E | Chapter and appendix verification suites pass in the canonical Linux environment. Targeted inference regressions also pass. | Some verifiers validate captured real-model output as data rather than downloading and rerunning the checkpoint. |
| Arm SIMD and setup | Cross-compiled Arm examples execute under QEMU, including an explicit Chapter 9 Arm run and Appendix A's CMake checks. | Emulation validates execution; it does not measure physical Arm-device performance. |
| Chapter 32 CUDA attention | NVIDIA A100-SXM4-40GB, CUDA 12.8, GCC 13.3. All 71 checks pass, including 60 attention shapes with partial tiles and query blocks. | Device output is compared with a CPU reference, with rejection of unsupported dimensions and non-finite comparisons. |
| CUDA diagnostics | Compute Sanitizer memcheck, racecheck and synccheck report no errors. A separate run with devices hidden checks the explicit no-device path. | Clean results apply to the exercised cases. A skipped device comparison is not a device-validation pass. |
| Documentation | Strict MkDocs builds and local image/link checks pass. | External references and complete mathematical proofs have not been exhaustively audited. |

## Validation evidence

- [Audit report and validation scope](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/main/AUDIT_REPORT.md)
- [GCC/Linux, Arm and A100 validation logs](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/tree/main/validation/a100)
- [A100 device transcript](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/main/validation/a100/cuda_device.txt)
- [Memory sanitizer](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/main/validation/a100/cuda_memcheck.txt), [race sanitizer](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/main/validation/a100/cuda_racecheck.txt), and [synchronization sanitizer](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/main/validation/a100/cuda_synccheck.txt)
- [Final Appendix A verification](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/main/validation/a100/verify_appendix_a_final.log), which supersedes its initial missing-build-tools failure.

## What the validation does not claim

Historical four-way cross-check descriptions in chapter narratives are not new hardware validation claims. The checked-in follow-up evidence establishes GCC/Linux, Arm execution under QEMU and the listed physical A100 tests; it does not independently substantiate every older physical-Arm assertion.

The canonical execution evidence is GCC/Linux. Some compiler- and random-distribution-dependent transcripts differ on macOS; those portability differences are documented in the audit report.

The fresh validation did not download the real-model checkpoint for a new end-to-end Qwen generation or throughput comparison. Existing real-model transcripts remain historical captures checked by the relevant verifiers. Physical Jetson execution was not tested. The deployment chapters include tested algorithms, state machines and integration examples; passing those tests does not establish production deployment, clinical validity, fraud-detection accuracy or trading performance.

Historical failure transcripts remain in the book where they explain a diagnostic. Similarly, examples that deliberately reject an input or skip a missing device retain that behavior. These are tested error-handling paths, not statements that the code has never been tested.
