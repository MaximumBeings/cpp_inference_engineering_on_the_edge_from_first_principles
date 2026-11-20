# C++ Inference Book Technical Accuracy Audit

**November 20, 2026. Review and correction record.** The original review found **26 confirmed issues: nine P1 and seventeen P2**. The correction pass below distinguishes implementation fixes from claims narrowed to the examples' actual scope. Passing self-tests does not establish general external-format compatibility, benchmark comparability or regulatory conformance.

P1 findings affect foundational correctness, interoperability, portability, evidence integrity or standards claims. P2 findings require correction before the affected explanation or example can support its current claim. The finding descriptions below preserve the original audited snapshot; consult the correction record for their current disposition. Scope corrections do not add the production capabilities that the original text overstated.

## Review coverage

Three agents reviewed numerical/model foundations (Chapters 2-8 and 30), C++/serving/toolchains (Chapters 9-17 and 31, Appendices A-E), and deployments/CUDA (Chapters 18-29 and 32). The primary review also checked validation evidence, prose cross-references and sample-output coverage. Findings use implementation inspection, focused local reproductions, the recorded GCC/Linux and A100 results, and primary documentation. Locations below refer to the audited source snapshot `a62ee74` unless a source comment was subsequently moved.

This is a comprehensive review across the published subject areas, not an exhaustive proof, certification, production deployment assessment, complete concurrency analysis or fresh benchmark campaign. Physical Jetson execution and a new real-model comparison were not performed. Explicitly disclosed teaching simplifications were distinguished from claims of standard compatibility or production behavior.

## Correction record

The correction pass changes the affected implementations, source embeds, explanations and captured stdout. Detailed per-finding resolutions and actual validation logs are in [validation/accuracy-fixes](validation/accuracy-fixes). Machine-readable dispositions are in [the findings ledger](audit/technical-findings-2026-11-20.json).

- Implementation corrections cover binary16 subnormals and safe scales, container constants, baseline dispatch, benchmark statistics, SwiGLU accounting, simultaneous-location handling and unknown observation coverage.
- Teaching simplifications are explicitly identified: the four-bit layout is not upstream Q4_0; GGUF readers remain trusted-input instructional readers; FNV is not evidentiary protection; limited accessibility/AML/schema checks do not establish conformance or legal conclusions.
- CUDA arithmetic is unchanged. Its documented contract is unscaled noncausal attention with tolerance comparisons and fixed score-tile storage, not a complete production decoder or fixed total memory. Existing A100 runs remain historical evidence, not a fresh GPU run for this correction pass.
- Historical model transcripts and unavailable platform runs are not relabeled as new measurements. See the validation record for what actually ran.

## Original findings

### TA-01 P1 Tiny valid quantization scales become zero

**Location:** [docs/part1/04-quantization-strategies.md line 421](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/docs/part1/04-quantization-strategies.md#L421).

The half helper flushes binary16 subnormals. A 32-value block containing 0.001 and zeros produces a zero Q8 scale and subsequently a NaN-to-int8 conversion. The sanitizer reproduction fails even though the published ordinary-value tests pass. Decoding binary16 0x0001 also incorrectly returns zero.

**Correction:** Implement subnormal conversion or explicitly use a documented representation with safe scale handling. Reject or handle zero/nonfinite scales before integer conversion, test tiny nonzero blocks, and inspect copied half helpers.

**Primary sources:** [NVIDIA binary16 definition](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____half.html).

### TA-02 P1 GGUF magic and a Q8 type identifier are incorrect

**Location:** [docs/part1/05-model-file-formats.md line 435](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/docs/part1/05-model-file-formats.md#L435).

The early writer/reader examples use 0x46475547, which serializes GUGF. The little-endian integer for GGUF is 0x46554747. GGML_Q8_0 is also assigned 7 instead of 8. Related constants recur in Chapter 6. A writer and reader sharing the same mistake do not establish interoperability.

**Correction:** Correct the constants and dispatch tables, check literal file bytes, and validate against independently generated upstream fixtures. If retaining a simplified container, name it a teaching format rather than GGUF.

**Primary sources:** [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [ggml type identifiers](https://github.com/ggml-org/ggml/blob/master/include/ggml.h).

### TA-03 P1 The teaching four-bit layout is not actual Q4_0

**Location:** [docs/part1/04-quantization-strategies.md line 762](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/docs/part1/04-quantization-strategies.md#L762).

The lesson insists on alpha/7 and pairs adjacent values in nibbles. Upstream Q4_0 uses the signed extremum divided by -8 and pairs coordinates j and j+16. The error propagates into later format and SIMD examples.

**Correction:** Implement the upstream scale and packing convention, or explicitly rename the teaching encoding and separate it from real GGUF Q4_0. Add independent byte-level fixtures.

**Primary sources:** [Q4_0 reference encoder](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c).

### TA-04 P1 Global AVX2 flags defeat the runtime fallback contract

**Location:** [05_runtime_dispatch.cpp line 19](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/05_runtime_dispatch.cpp#L19).

The dispatch example and verifier compile the entire translation unit with -mavx2 -mfma while promising execution on CPUs without those features. Unguarded ordinary code may then contain unsupported instructions.

**Correction:** Remove global ISA-enabling flags from the baseline translation unit. Keep specialization inside the existing target("avx2,fma") function or a separate object. Also narrow the claim that -march=native necessarily folds explicit feature queries away.

**Primary sources:** [GCC runtime feature detection](https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html).

### TA-05 P1 Historical benchmark rates are relabeled as current comparisons

**Location:** [03_llamacpp_comparison.cpp line 944](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/03_llamacpp_comparison.cpp#L944).

Fixed historical NEON llama.cpp rates are printed under arbitrary current --pp/--tg lengths, combined with a fresh local timing, and described as relative speed on this machine. The hardware, workload and statistic are not guaranteed to match.

**Correction:** Keep historical captures with immutable provenance and workload metadata. Require a matching fresh reference measurement before computing a current-machine ratio.

**Primary sources:** [llama-bench measurement fields](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).

### TA-06 P1 The real-file reader accepts unvalidated tensor ranges

**Location:** [01_real_gguf_inspection.cpp line 241](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/01_real_gguf_inspection.cpp#L241).

The reader can report success without proving every tensor range is inside the file. Stream-read success, descriptor/string bounds, dimension products and alignment are not comprehensively checked. Later model examples copy this reader.

**Correction:** Validate metadata reads and dimensions, supported types/version, alignment, checked size arithmetic and all mapped ranges before exposing views. Until then, describe the reader as restricted to trusted instructional inputs.

**Primary sources:** [GGUF layout contract](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

### TA-07 P1 A noncryptographic checksum is presented as evidentiary tamper protection

**Location:** [chapter25_template.md line 98](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/chapter25_template.md#L98).

The incident-report example calls an unprotected 32-bit FNV checksum tamper evidence. The checksum is over annotations rather than the original video. One mutation changing a checksum does not establish evidentiary integrity.

**Correction:** Label the demonstration as change detection. Evidence-integrity claims require cryptographic hashing of the relevant source and records, protected reference digests, and documented custody procedures.

**Primary sources:** [FNV is noncryptographic](https://datatracker.ietf.org/doc/html/rfc9923), [NIST digital evidence preservation](https://nvlpubs.nist.gov/nistpubs/ir/2022/NIST.IR.8387.pdf).

### TA-08 P1 The structuring rule is not the legal definition

**Location:** [chapter29_template.md line 53](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/chapter29_template.md#L53).

The chapter equates an aggregate above $10,000 and individually smaller payments with the exact definition of structuring, and calls other cases legitimate. Structuring depends on evasion purpose and is not limited to that aggregate or one institution/day.

**Correction:** Call the code a simplified alert heuristic. Distinguish CTR thresholds, institutional review, suspicious patterns and legal conclusions; do not infer legitimacy from a negative rule result.

**Primary sources:** [FFIEC structuring guidance](https://bsaaml.ffiec.gov/manual/Appendices/08).

### TA-09 P1 The accessibility checker confuses AA and AAA target size

**Location:** [02_accessibility_compliance_audit.cpp line 106](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/02_accessibility_compliance_audit.cpp#L106).

The checker claims AA-only coverage but enforces criterion 2.5.5 at 44 by 44 CSS pixels, which is AAA. WCAG 2.2 AA criterion 2.5.8 uses 24 by 24 or sufficient spacing, with exceptions.

**Correction:** Specify the WCAG version and conformance level, implement the selected rule and exceptions, and avoid claiming full conformance from these limited checks.

**Primary sources:** [W3C 2.5.5 AAA](https://www.w3.org/WAI/WCAG22/Understanding/target-size-enhanced.html), [W3C 2.5.8 AA](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html).

### TA-10 P2 SafeTensors support is understated

**Location:** [docs/part1/05-model-file-formats.md line 1531](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/docs/part1/05-model-file-formats.md#L1531).

The sample output says SafeTensors has no quantization support and supports only F32/F16/BF16. Its numeric types include integer representations and additional floating types. Container support is different from defining GGUF block semantics.

**Correction:** Describe which dtypes and representation metadata the example implements, and distinguish container storage from framework-specific quantization conventions.

**Primary sources:** [SafeTensors dtype definitions](https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs).

### TA-11 P2 Q4_K block size is confused with a mixed model recipe

**Location:** [docs/part1/05-model-file-formats.md line 1648](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/docs/part1/05-model-file-formats.md#L1648).

The 144-byte/256-value calculation describes Q4_K blocks. Q4_K_M is a mixed model recipe and may use other precisions for selected tensors.

**Correction:** Label the demonstrated block Q4_K and calculate whole-model size from its actual tensor types and metadata, rather than one block ratio.

**Primary sources:** [llama.cpp quantization recipes](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp).

### TA-12 P2 Aggregate layer intensity is treated as an exact bottleneck classifier

**Location:** [06_full_transformer_layer_roofline_analysis.cpp line 83](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/06_full_transformer_layer_roofline_analysis.cpp#L83).

Total FLOPs divided by selected total bytes cannot establish the bottleneck of serial unfused kernels. Memory-bound and compute-bound kernels can coexist, and the model omits some activation and KV traffic. Related reasoning appears in Chapter 8.

**Correction:** Label the aggregate as an optimistic weight-dominated model. Sum per-kernel time bounds and distinguish those bounds from measured performance.

**Primary sources:** [Original Roofline paper](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf).

### TA-13 P2 The full-layer model omits a SwiGLU projection

**Location:** [06_full_transformer_layer_roofline_analysis.cpp line 60](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/06_full_transformer_layer_roofline_analysis.cpp#L60).

The Chapter 30 capstone counts a two-matrix FFN although the book engine uses gate, up and down projections. For the stated tiny example, matching SwiGLU changes 576 FLOPs/512 weight bytes to 704/640 before ancillary operations and traffic.

**Correction:** Explicitly name a conventional two-projection FFN or count all three SwiGLU matrices and regenerate the calculations and output.

**Primary sources:** [GLU variants](https://arxiv.org/abs/2002.05202).

### TA-14 P2 The QJL formula omits residual magnitude

**Location:** [docs/part1/07-turboquant.md line 924](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/docs/part1/07-turboquant.md#L924).

The prose reconstruction formula omits the residual norm even though sign bits alone are invariant to positive rescaling. The implementation already includes that norm.

**Correction:** Include the residual norm, define the projection count consistently, and specify the zero-residual case. Preserve the already-correct norm handling in code.

**Primary sources:** [TurboQuant paper](https://arxiv.org/html/2504.19874v1).

### TA-15 P2 The GPTQ illustration does not preserve previously quantized weights

**Location:** [05_hessian_role_in_gptq.cpp line 103](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/05_hessian_role_in_gptq.cpp#L103).

The compensation loop updates every index except the current one despite promising updates only to not-yet-quantized weights. Repeated calls can move a previously quantized weight off its grid.

**Correction:** Label it a single-step compensation illustration, or implement an active set with the appropriate inverse/Cholesky updates and test several successive quantization steps.

**Primary sources:** [GPTQ reference implementation](https://github.com/IST-DASLab/gptq/blob/main/gptq.py).

### TA-16 P2 Compression commentary reverses scale overhead

**Location:** [03_prefill_vs_decode.cpp line 278](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/03_prefill_vs_decode.cpp#L278).

The stdout says four bits versus 32 implies roughly 4x compression and metadata strengthens compression. The raw ratio is 8x; adding a scale reduces it to about 128/18 = 7.11x.

**Correction:** Correct the explanation and captured stdout. The existing encoded-byte calculation can remain.

### TA-17 P2 Reciprocal median latency is mislabeled as throughput

**Location:** [01_benchmark_stats_and_definitions.cpp line 97](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/01_benchmark_stats_and_definitions.cpp#L97).

1000/median(step_ms) is a median-equivalent rate, not generally completed tokens divided by elapsed time. For step times [1,1,100] ms, it gives 1000 tokens/s while elapsed throughput is about 29.41 tokens/s.

**Correction:** Report count/sum(time) throughput and latency quantiles separately. Do not compare reciprocal-median rates with whole-run llama-bench rates.

**Primary sources:** [llama-bench statistics](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).

### TA-18 P2 The benchmark sampling step is a random-ID draw

**Location:** [02_benchmark_harness.cpp line 595](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/02_benchmark_harness.cpp#L595).

The comment describes vocabulary-sized argmax/sampling work, but pick(token_rng) draws one integer without scanning logits. Reported full-generation TTFT therefore excludes material work.

**Correction:** Run the actual sampler or label the timing as a forward-pass microbenchmark with token selection, decoding and transport excluded.

### TA-19 P2 Zero elapsed time suppresses an impossible-travel alert

**Location:** [03_impossible_travel_geographic_detection.cpp line 83](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/03_impossible_travel_geographic_detection.cpp#L83).

The code sets required speed to zero for elapsed time <= 0. A same-time New York/Los Angeles pair therefore receives no travel flag despite positive distance.

**Correction:** Handle positive distance at zero elapsed time explicitly, reject reversed timestamps, and test equal-time, reversed-time and same-location cases.

### TA-20 P2 Human-only SAR filing is stated as a blanket legal rule

**Location:** [04_transaction_monitoring_disposition_engine.cpp line 8](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/04_transaction_monitoring_disposition_engine.cpp#L8).

The text treats manual institutional review and electronic submission mechanics as the same legal requirement. FinCEN supports automated/batch transmission; the cited blanket prohibition on automated filing is unsupported.

**Correction:** State that this example does not implement SAR filing and uses a review policy. Separate accountable decision governance from the transmission method.

**Primary sources:** [FinCEN transmission methods](https://bsaefiling.fincen.gov/transmission-methods), [FinCEN secure data transfer](https://bsaefiling.fincen.gov/SDTM).

### TA-21 P2 The CUDA attention demonstration is unscaled and noncausal

**Location:** [03_cuda_kernel_and_validation_suite.cu line 68](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/03_cuda_kernel_and_validation_suite.cu#L68).

Both CPU reference and GPU omit 1/sqrt(head dimension) scaling and a causal mask while the chapter presents the result as a production decoder attention step. Shared assumptions allow both implementations to agree.

**Correction:** Prominently specify unscaled noncausal attention, or implement the intended scale and mask and compare with an independent definition.

**Primary sources:** [Scaled dot-product attention definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

### TA-22 P2 Fixed attention memory and exactness claims exceed the checks

**Location:** [chapter32_template.md line 15](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/chapter32_template.md#L15).

The fixed allocation demonstrated is the score tile, not all Q/K/V/output/state memory. Tolerance-based numerical comparisons also do not establish bitwise equality or proof for arbitrary inputs.

**Correction:** Say the implementation avoids the quadratic score matrix and uses fixed score-tile storage for fixed tile dimensions. State the tested tolerances and input scope.

**Primary sources:** [FlashAttention paper](https://arxiv.org/abs/2205.14135).

### TA-23 P2 Three field-name checks are not a complete privacy guarantee

**Location:** [01_bodycam_frame_sampling_and_scene_tagging.cpp line 144](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/01_bodycam_frame_sampling_and_scene_tagging.cpp#L144).

The concept checks only name, person_id and face_embedding. Other field names or recognition outside this result type are not ruled out. The book overstates the scope as structural prohibition of facial recognition.

**Correction:** Describe a narrow schema guard and its precise checked names. A broader privacy claim needs pipeline constraints and separate review.

### TA-24 P2 Missing observations become an affirmative inactivity claim

**Location:** [02_incident_report_drafting_and_chain_of_custody.cpp line 135](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/02_incident_report_drafting_and_chain_of_custody.cpp#L135).

Gaps between tags and empty observation collections are labeled No notable activity without evidence that the interval was observed and classified negative.

**Correction:** Use No tagged observations or Coverage unknown unless validated observation coverage supports an inactivity statement. Track unavailable footage separately.

### TA-25 P2 KV cache complexity mixes per-step and total work

**Location:** [chapter13_template.md line 53](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/chapter13_template.md#L53).

The sentence compares O(N^2) total generation with O(N) per step. These are different quantities. Cached dense attention still sums linear per-step work to quadratic total attention work across a growing sequence.

**Correction:** Compare per-step costs with per-step costs and give total-generation costs separately, including the roles of projections, attention and any fixed window.

**Primary sources:** [KV caching explanation](https://huggingface.co/docs/transformers/cache_explanation).

### TA-26 P2 The prose refers to missing chapters and appendices

**Location:** [docs/part0/02-the-tensor-engine.md line 25](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/a62ee74/docs/part0/02-the-tensor-engine.md#L25).

The published navigation starts at Chapter 2 and ends at Appendix E, but prose contains six references to an absent Chapter 1 or Appendix F, including Appendix A/B/E conclusions.

**Correction:** Replace these references with existing sections or add the promised material. Recheck prose references as well as clickable links.

## Additional provenance question

The repository repeatedly describes four-way validation including physical aarch64 hardware. The current checked-in follow-up evidence supports GCC/Linux, QEMU Arm execution and physical A100 execution. Those logs do not substantiate every older physical-Arm claim. This is an evidence gap, not a conclusion that historical runs never occurred. Preserve independently identifiable hardware/toolchain logs for those historical assertions, or narrow them to the execution records available.

## Reproduction evidence

The benign tiny-scale reproduction is saved as [fp16_tiny_scale.cpp](audit/evidence/fp16_tiny_scale.cpp), with actual [sanitizer output](audit/evidence/fp16_tiny_scale.stderr.txt). This is the original failing snapshot, retained as historical evidence; it is not the corrected implementation or a passing regression test. Current conversion and quantizer tests are in `tests/numerical_accuracy_regressions.py`.

```sh
clang++ -std=c++23 -O1 -fsanitize=undefined,float-cast-overflow \
  -fno-sanitize-recover=all audit/evidence/fp16_tiny_scale.cpp -o /tmp/edge-audit-fp16
/tmp/edge-audit-fp16
```

Agent review also reproduced the simultaneous-location alert failure and confirmed reader range validation gaps. No external service or production system was tested.

## Sample output improvements completed

The scan covered 162 fenced text blocks. Sections 2.1 and 2.2 were the two whose sample stdout contained only a pass count. Both now print actual shapes, strides, selected values and write-through behavior before their original test summaries. Section 2.2 also prints every selected head row, range-slice values, combined-slice values and whether the head view aliases the original allocation.

The updated samples were compiled and run, and their exact stdout was inserted into the chapter. `python3 tests/sample_output_checks.py` re-executes both examples and checks the published output. No summary-only sample blocks remain. Other captured output was not regenerated merely to make a known incorrect claim look polished. The subsequent correction pass updates affected claims and refreshes stdout where implementations changed; the resolution logs distinguish fresh runs from retained historical captures.

## Original correction plan

1. Correct the half helper and standard file/quantization contracts, including copied implementations and diagrams. Use independent format fixtures.
2. Correct baseline CPU dispatch and current-versus-historical benchmark behavior.
3. Correct accessibility, AML and evidence-integrity claims alongside their implementation contracts.
4. Correct numerical/performance explanations, attention scope and remaining application edge cases.
5. Regenerate affected source embeds and stdout only after those fixes, run independent regression checks, and update the public validation record with the actual corrected scope.

## Verification during this review

The new tensor stdout checks pass. The tiny-scale sanitizer reproduction confirms TA-01. Earlier recorded GCC/Linux and A100 logs are evidence of their listed runs. Current correction-pass validation is recorded separately under `validation/accuracy-fixes`. The changed tensor examples retain their original 18/18 and 16/16 checks in addition to the explanatory output.
