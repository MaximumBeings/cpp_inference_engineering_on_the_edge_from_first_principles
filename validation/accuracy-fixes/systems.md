# Systems technical accuracy resolutions

- TA-04 — Code and prose fixed. Runtime dispatcher translation unit compiles at baseline; AVX2/FMA flags apply only to its target-attributed function. Removed claim that native compilation always removes feature checks. Forced scalar tests are explicitly not proof of running on an older CPU.
- TA-05 — Code and prose fixed. Removed hardcoded llama.cpp rates and arbitrary-workload speed ratios. Current output reports local measurements only and requires a matched external run for comparison. Original real-run capture is retained and labeled historical, including its obsolete median-derived statistic; it was not rerun.
- TA-06 — Scope corrected, not hardened. Reader copies in chapters12/15–19 explicitly require trusted fixtures/checkpoints and disclose missing input-validation bounds. No claim is made that these instructional parsers safely accept untrusted files.
- TA-12 — Accounting/scope fixed. Aggregate intensity is an optimistic bound, not a classifier for each serial kernel. Added sum-of-per-kernel-bounds counterexample (aggregate10.1 versus serial20 in analytical seconds) and corresponding explanation.
- TA-13 — Math/code fixed. SwiGLU includes gate, up and down projections: tiny capstone704FLOPs/640weightbytes; assumptions and omitted costs explicit.
- TA-17 — Code/prose fixed in all three benchmark copies. Elapsed TPS uses count/summed latency (reciprocal mean for sequential steps); even median averages the central pair. Regression [1,1,100]ms yields29.4118tok/s, not1000.
- TA-18 — Scope corrected. Harness is labeled forward-pass microbenchmark; prefill-only time is not user-visible TTFT, and random workload token IDs are not a vocabulary sampler.

## Validation/provenance

Fresh Apple Clang arm64 compile/run: benchmark examples17/17 each; roofline capstone and chapter8 profile pass. Chapter09 verifier passes portable scalar fallback, explicitly skips x86-only kernels; chapters17 and30 verifiers pass compilation, two deterministic reruns, embedded source/stdout and self-check checks. See adjacent chapter09/17/30-systems.txt logs.

Linux peakRSS checks remain required on Linux. On macOS explicit unavailable(-1) is checked and no memory-growth measurement is fabricated. The chapter9 x86 transcript preserves its original architecture-specific results; its common explanatory Test3 block was updated from the fresh portable run, not presented as a fresh x86 execution. Lambda SSH connection timed out; no GPU/remote model rerun is claimed. No GPU was provisioned. TeachingQ4-versus-GGUF encoding boundary added to chapter9, coordinated with numerical audit.
