# C++ Inference Engineering on the Edge

*Building LLM and Vision-Language Inference Systems in C++23, From First Principles*

<!-- chapter-figure -->
<figure class="chapter-figure" markdown="1">

[![From first principles to a deployable edge engine. The whole book connects compact models, C++ kernels, serving state, hardware limits and practical applications.](images/index-overview.svg?v=8141a0c7c073)](images/index-overview.svg?v=8141a0c7c073)

  <figcaption>A complete path through Parts 0-6, with verification across every layer and fixed device budgets guiding the design.</figcaption>
</figure>
<!-- /chapter-figure -->

Most inference-engineering material is written from the data center outward: elastic memory, a GPU a network call away, and throughput measured across thousands of concurrent requests. This book starts from the opposite machine — a fixed amount of RAM that will never grow, a CPU that may or may not have a companion GPU, a battery and a thermal ceiling, and no network round-trip to fall back on when local compute runs short — and builds an inference engine that has to work under those constraints from the first line of code, not as an afterthought bolted on once a cloud-shaped design already exists.

This book builds its own tensor, quantization and threading foundations from Chapter 2. Its examples were revalidated on November 20, 2026 using GCC/Linux, Arm execution under QEMU, and a physical NVIDIA A100 for the CUDA attention kernel. The chapter and appendix verification suites pass, as do the 27 embedded C++ examples in Chapters 2-6. The A100 suite passes all 71 checks, including 60 device comparisons, with clean memory, race and synchronization sanitizer results. [Validation status](validation-status.md) records the evidence and the remaining scope limits, including fresh real-model benchmarks and physical Jetson execution.

The [technical accuracy audit](https://github.com/MaximumBeings/cpp_inference_engineering_on_the_edge_from_first_principles/blob/main/TECHNICAL_ACCURACY_AUDIT.md) records open corrections identified after the execution checks. This book remains under development; passing self-tests is not a substitute for independently checking its format, numerical and deployment claims.

## What this book covers

- **Part 0 — Foundations for Inference at the Edge.** Why edge inference is a fundamentally different problem from cloud inference, a zero-copy tensor engine built on `std::mdspan` with an arena allocator and mixed-precision support, and the computational graph (RMSNorm, SwiGLU, RoPE, attention) that every transformer forward pass is built from.
- **Part 1 — Shrinking the Model: Quantization for Constrained Memory.** Integer quantization from affine math to blockwise scales, the GGUF and SafeTensors file formats with genuine memory-mapped loading, a hybrid engine that mixes precisions by tensor class, and TurboQuant's near-optimal online vector quantization.
- **Part 2 — Performance on the Metal: SIMD, Cache, and Threads.** The roofline model and the prefill/decode split, AVX2 on x86 and NEON on Arm (both genuinely compiled and genuinely run, the Arm path under emulation), thread pools and work-stealing schedulers, and parallelizing a full transformer forward pass across cores.
- **Part 3 — Serving State: Tokenization and the KV Cache.** A from-scratch byte-pair-encoding tokenizer, paged attention and ring-buffer KV caches with smart eviction, and the sliding-window, streaming-requantization, and prefix-caching techniques that keep a long conversation inside a fixed memory budget.
- **Part 4 — From Model to Binary: Real Weights, Real Models, Production Engines.** Running a real, downloaded Qwen2.5 model to its first genuine token, a production engine integrated into a single deployable binary, and honest benchmarking against the wider inference-engine ecosystem.
- **Part 5 — Deploying Vision-Language Intelligence at the Edge.** Worked deployment examples and integration designs for vision-language inference — manufacturing-line defect detection, retail shelf intelligence, medical-imaging triage, document and insurance-claim processing, security and accessibility auditing, point-of-sale authenticity verification, natural-language photo editing, and body-worn cameras — each one carrying the edge constraints from Part 0 through every architecture decision.
- **Part 6 — Going Further: Mathematical Foundations and GPU Acceleration.** The mathematics a kernel author actually needs (FLOPs, the roofline, the Hessian), continuous-batching production serving architecture, and Flash Attention taken to a real CUDA implementation for the edge devices that do carry a small GPU.

## How to read this book

Every chapter follows the same shape: an intuition built from a concrete picture before any code, a background section with the real, complete implementation, one or more worked examples with genuinely computed output, a `[COMMON TRAP]` callout naming a specific, real mistake rather than a vague warning, a chapter summary, self-check questions, and worked solutions. Getting Started explains exactly which toolchain compiles and runs each kind of code in this book — native x86, emulated Arm, or (in Chapter 32) a GPU validated on an A100 — so nothing is left for a reader to guess about how a claim was actually established.
