# C++ Inference Engineering: Building LLM and Vision-Language Systems for the Edge

*A from-scratch, first-principles book on building inference engines in modern C++23 — with every design decision made from the constraints of edge hardware first: fixed memory budgets, no GPU (or a small one), thermal limits, and no network round-trip to fall back on.*

This table of contents follows the same shape as *Data Structures and Algorithms in CUDA C++*: numbered Parts, numbered chapters with a one-line description each, a sequential build-verify-lock discipline for every code example, and a lettered set of appendices closing the book. Every chapter below is built from real, working C++23 (and where GPU offload is discussed, real CUDA), compiled and run for real, never hand-waved.

A **Source Material Map** at the end shows exactly which of your existing drafted chapters (in `C++_Inference_Engineering_Book/`) each new chapter is built from, so you can see how much of the heavy lifting is already done.

---

## Part 0 — Foundations for Inference at the Edge

**1. Why Inference at the Edge Is a Different Problem**
Cloud inference optimizes for throughput across many concurrent requests with elastic memory and a GPU a phone call away; edge inference optimizes for one request at a time against a hard, fixed memory ceiling, a battery, and a thermal limit — this chapter sets the constraints (latency budget, memory budget, power budget, no-network assumption) that every later design decision in the book is judged against.

**2. The Tensor Engine: `std::mdspan`, Arena Allocators, and Mixed Precision**
Zero-copy tensor views with `std::mdspan`, cache-optimized row/column-major layout and stride, a bump-pointer arena allocator to eliminate heap fragmentation on memory-constrained devices, and a `std::variant`-based FP32/FP16/BF16 mixed-precision tensor type.

**3. The Computational Graph: RMSNorm, SwiGLU, RoPE, and Attention**
The three operations that make up a modern transformer block, built and numerically verified one at a time — RMSNorm, the SwiGLU feed-forward network, rotary positional embeddings, and grouped-query attention with a KV cache — assembled into a static, liveness-planned compute graph with `mmap`-based weight loading.

---

## Part 1 — Shrinking the Model: Quantization for Constrained Memory

**4. Quantization Strategies: Affine Math, Symmetric vs. Asymmetric, Blockwise Scales**
Integer quantization from first principles — the affine scale/zero-point scheme, symmetric vs. asymmetric quantization, why a single global scale fails and blockwise scales fix it, Q8 and Q4 bit-packing, fused dequantize-and-dot-product kernels, and quantization-aware training with the straight-through estimator.

**5. Model File Formats: GGUF, SafeTensors, and Memory-Mapped Loading**
The GGUF binary format byte by byte (magic number, key-value metadata, tensor descriptors, Q4_K_M block layout), a from-scratch GGUF reader and writer, the SafeTensors alternative, and `mmap`-based zero-copy weight access that never pages the whole model into RAM at once — the difference between a model that fits on an edge device and one that does not.

**6. The Hybrid Quantized Engine: Mixing Precisions for Different Jobs**
Not every tensor deserves the same quantization — a hybrid engine architecture that routes different weight classes through different quantizers, the resulting data flow, and a full hybrid quantized forward pass, wired end to end into one inference pipeline.

**7. TurboQuant: Online Vector Quantization with Near-Optimal Distortion**
A supplementary deep dive into vector (not scalar) quantization: random rotation, the Beta-distribution trick that beats per-block max scaling, Lloyd-Max optimal codebooks, the inner-product bias problem and its two-stage fix (QJL), and applying the whole technique to compress the KV cache specifically.

---

## Part 2 — Performance on the Metal: SIMD, Cache, and Threads

**8. The Memory Wall and the Roofline Model**
Why a 5%-CPU-utilization inference engine is the default outcome, not a bug — the roofline model, arithmetic intensity for each transformer kernel, and the prefill-vs-decode split (compute-bound GEMM vs. memory-bound GEMV) that shapes every optimization in this Part.

**9. SIMD Vectorization: AVX2 on x86, NEON on Arm**
The x86 AVX2 register model and integer-arithmetic instruction set, the ARM NEON register model for the mobile and embedded path this book cares about most, fused quantized dot products on both, the Fast Walsh-Hadamard Transform as a cheap stand-in for a dense random rotation, and runtime CPU-feature detection with a dispatch table so one binary runs correctly on both.

**10. Threading and Concurrency: Thread Pools, Work Stealing, and False Sharing**
Why AI threading is not web-server threading, the latency budget worked out in real numbers, a `std::barrier`-based thread pool and a spin-barrier hybrid for when it is not fast enough, false sharing traced to the cache-line level with a measured benchmark, a work-stealing scheduler for uneven per-core load, and thread affinity pinning.

**11. Parallelizing the Forward Pass: Multi-Head, Multi-Layer, Multi-Thread**
Row-parallel vs. column-parallel matmul and why row-parallel wins for single-token generation, a persistent thread pool driving a fully parallel transformer layer, the pipeline pattern for overlapping layers, deterministic floating-point reduction (so two runs of the same prompt give bit-identical output), and NUMA-aware memory placement.

---

## Part 3 — Serving State: Tokenization and the KV Cache

**12. The Tokenizer: BPE, SentencePiece, and the Vocabulary Pipeline**
Byte-pair encoding from first principles — the merge algorithm, a from-scratch BPE merge engine, regex-based pre-tokenization, special tokens and chat templates, and a token decoder that turns IDs back into correct UTF-8 — ending in a complete tokenizer pipeline loaded directly from a GGUF vocabulary.

**13. The KV Cache Manager: Paged Attention, Ring Buffers, and Smart Eviction**
The memory-wall arithmetic that makes the KV cache the real bottleneck on an edge device, PagedAttention's block-table indirection, a fixed-capacity ring buffer with the RoPE position-decoupling subtlety and sink-token protection, and the H2O heavy-hitter eviction policy that uses attention probability as a free importance signal.

**14. Advanced KV Cache Management: Sliding Windows, Streaming Re-Quantization, and Prefix Caching**
Sliding-window attention as a ring buffer at scale, a tiered cache architecture that re-quantizes older entries to a coarser precision as they age out of the working set, and prefix caching for multi-turn conversations — closing with a complete cache manager combining every technique behind one interface.

---

## Part 4 — From Model to Binary: Real Weights, Real Models, Production Engines

**15. Running Real Models: From HuggingFace to First Token with Qwen2.5**
Downloading and inspecting a real GGUF model, the specific architecture diff between Llama and Qwen2 (QKV bias, a 7:1 grouped-query attention ratio, tied embeddings, RoPE base frequency, ChatML), adapting the forward pass and tokenizer for those five differences, and generating a first real, correct token from a real downloaded model.

**16. The Production Engine: Integrating Everything into a Single Binary**
The contract a production inference binary must honor, the startup sequence and memory map, the generation loop and the interactive conversation loop, a streaming token decoder, error handling for the edge cases that only show up outside a notebook, and a built-in profiler — all of it compiled into one deployable binary.

**17. Benchmarking Against llama.cpp and the Broader Ecosystem**
What to measure and why (tokens/sec, time-to-first-token, memory high-water mark), a from-scratch benchmarking harness, and an honest, apples-to-apples comparison of this book's engine against llama.cpp on identical hardware and identical models.

---

## Part 5 — Deploying Vision-Language Intelligence at the Edge

*Every chapter in this Part runs a real vision-language model (Qwen2.5-VL) as the engine built in Parts 0-4, wired into a specific, physically-grounded deployment — a camera, a factory line, a point of sale — with the edge constraints (no cloud round-trip, fixed hardware, real-time deadlines) carried through every architecture decision.*

**18. Industrial Visual Inspection: Edge-Deployed Defect Detection on Manufacturing Lines**
GigE Vision frame acquisition with hardware triggering, an image preprocessing pipeline for the ViT encoder, the Qwen2.5-VL inference engine wired to a live camera feed, confidence-threshold disposition logic, defect logging to SQLite, and OPC UA integration into a real MES/SCADA line.

**19. Retail Inventory and Supply Chain Intelligence: Shelf Photography to Action**
Encoding a planogram directly into the system prompt, a batch shelf-photo processor, REST integration into ERP/WMS systems, multi-store aggregation and trend detection, and shrinkage and misplacement detection from ordinary shelf photographs.

**20. Medical Imaging Triage: Preliminary Radiology Reads and Human-in-the-Loop Workflow**
The regulatory framework and system-classification boundaries this kind of deployment must respect, DICOM ingestion and windowed image extraction, a structured triage-reporting prompt, PACS integration with a human-in-the-loop worklist, and an honest treatment of where VLM interpretation's explainability runs out.

**21. Document Intelligence, Insurance Claims, and Environmental Monitoring**
Why a VLM replaces a traditional OCR pipeline outright, a document-type router with handwriting-aware extraction, vehicle- and property-damage assessment engines for insurance claims, and wildlife and underwater species identification from camera-trap and marine imagery.

**22. Security Surveillance, Accessibility Compliance, and Art Authentication**
A surveillance narration engine with temporal awareness and multi-camera event correlation, an accessibility-compliance auditing engine, and an art-condition assessment and provenance-verification engine — three domains united by the same pattern: turning a continuous camera feed into a structured, auditable narrative.

**23. Trust and Authenticity at the Point of Sale: Counterfeits, Reconciliation, Expense Audits, and Luxury Goods**
A brand-reference database and counterfeit-screening engine for e-commerce listings with seller trust scoring, receipt-to-transaction reconciliation, automated expense-report policy enforcement, and a real-time luxury-goods authentication engine for consignment and resale counters.

**24. Natural Language Photo Editing: From Prompt to `cv::Mat` on Edge Hardware**
An edit-interpretation prompt that turns "make it look better" into a structured edit plan, a complete OpenCV processing engine that executes that plan as real `cv::Mat` operations, a request-to-operation mapping table, and iterative refinement through conversation.

**25. Body-Worn and Personal Cameras: Evidence Documentation and Health Tracking**
Police body-camera scene tagging and report drafting under an explicit no-facial-recognition legal boundary with frame-sampling strategy and edge deployment, alongside exercise-form analysis and camera-based nutrition tracking (via USDA FoodData Central) under a privacy-first architecture for personal health devices.

---

## Part 6 — Going Further: Mathematical Foundations and GPU Acceleration

**30. Mathematical Foundations for Kernel Authors: FLOPs, the Roofline, and the Hessian**
The dot product as inference's atomic unit, GEMM as the real bandwidth hog, why naive softmax overflows and the numerically stable fix, RoPE's rotation math, quantization as affine algebra, the Hessian's role in GPTQ, and FLOP counting for a full roofline analysis of a transformer layer.

**31. Continuous Batching and Production Serving Architecture**
The static-batching problem and why real serving systems abandon it, the prefill/decode conflict a scheduler must resolve, a continuous-batching scheduler built from scratch, and numerical debugging tools (NaN-propagation tracing, floating-point drift detection) for catching the bugs that only appear at serving scale.

**32. Flash Attention and CUDA Kernels: Taking the Engine to the GPU**
The O(N²) memory wall standard attention hits and the online-softmax trick that fixes it, a `std::mdspan`-based Flash Attention implementation and benchmark, and a CUDA production engine with its own kernel-validation suite for the edge devices — Jetson-class boards among them — that do carry a small GPU.

---

## Appendices

**A. Installation & Setup — Cross-Compiling for the Edge**
Toolchain setup on an x86 development machine (Amazon Linux 2023 and Ubuntu 24.04 paths), then cross-compiling the same codebase for ARM/NEON targets — Raspberry Pi, Jetson, and mobile — with a working CMake project layout and a diagnostics reference for the environment-specific failures that only show up off the dev machine.

**B. Practice Quiz**
Conceptual review questions and predict-the-output challenges spanning every Part, in the same format as the CUDA book's own quiz appendix.

**C. A Decision-Tree Reference: Quantization, Threading, and Cache Strategies at a Glance**
Every "Five Laws," decision tree, and quick-reference table scattered across the individual chapters, consolidated into one cross-referenced appendix — which quantization format for which constraint, which threading mechanism for which workload, which KV cache strategy for which context length.

**D. Profiling and Benchmarking on Edge Hardware**
Hardware bandwidth surveys across real edge silicon, latency-budget arithmetic, `perf`-based profiling with the flags that actually work on ARM, and this book's own honest-numbers discipline: deterministic operation counts locked and verified, genuinely variable wall-clock timing kept out of the locked contract.

**E. From PyTorch and llama.cpp to C++: A Rosetta Stone**
`torch.Tensor` and `AutoModel.from_pretrained` next to this book's own `mdspan` tensor and GGUF loader, vLLM's continuous batching next to this book's own scheduler, HuggingFace's tokenizer next to the from-scratch BPE pipeline — a translation guide for readers arriving fluent in the Python inference ecosystem.

**F. Common Failure Modes: NaN Propagation, False Sharing, Floating-Point Drift, and Alignment Bugs**
The failure modes that produce a correct-looking but silently wrong or silently slow inference engine — NaN propagation through a transformer stack, cache-line false sharing between threads, floating-point non-determinism across runs, and the memory-alignment bugs that only surface on ARM's stricter alignment rules.

---

## Source Material Map

Nearly every chapter above is built directly from a drafted companion guide already in `C++_Inference_Engineering_Book/` — duplicate and superseded drafts (multiple "Chapter 4," "Chapter 7," etc. across your two drafting passes) are resolved below to the single most complete version each new chapter draws from.

| New chapter | Built from your existing draft |
|---|---|
| 2. The Tensor Engine | *C++23 Tensor Engine — Complete Companion Guide, Version 3* (`mdspan_chapter_1_redo_v3.md`) |
| 3. The Computational Graph | *Chapter 2 Companion Guide — The Computational Graph, Version 2* |
| 4. Quantization Strategies | *Chapter 3 Companion Guide — Quantization Strategies, Version 2* |
| 5. Model File Formats | *Chapter 4 — GGUF File Loading & Hybrid Quantized Inference, II* + *Chapter 7 — GGUF, Real Weights & Benchmarking, Version 2* |
| 6. The Hybrid Quantized Engine | *Chapter 4 — GGUF File Loading & Hybrid Quantized Inference* (Hybrid Engine sections) |
| 7. TurboQuant | *Supplementary Chapter — TurboQuant, V2* |
| 8. The Memory Wall and the Roofline Model | *Chapter 4 Companion Guide — SIMD & Vectorization, Version 2* (Roofline sections) |
| 9. SIMD Vectorization | *Chapter 5 Companion Guide — SIMD Vectorisation: AVX2 and NEON Intrinsics* |
| 10. Threading and Concurrency | *Chapter 6 Companion Guide — Threading & Concurrency, Version 2* |
| 11. Parallelizing the Forward Pass | *Chapter 8 Companion Guide — Multi-Head, Multi-Layer, Multi-Thread* |
| 12. The Tokenizer | *Chapter 6 Companion Guide — The Tokenizer: BPE, SentencePiece* |
| 13. The KV Cache Manager | *Chapter 5 Companion Guide — The KV Cache Manager, Version 2* |
| 14. Advanced KV Cache Management | *Chapter 7 Companion Guide — KV Cache Management: Paging, Eviction, Streaming Quantisation* |
| 15. Running Real Models | *Chapter 10 Companion Guide — Running Real Models: HuggingFace to First Token, Qwen V2* |
| 16. The Production Engine | *Chapter 9 Companion Guide — The Production Engine* |
| 17. Benchmarking | *Chapter 7 — GGUF, Real Weights & Benchmarking* (Benchmarking sections) |
| 18. Industrial Visual Inspection | *Chapter 11A — Industrial Visual Inspection* |
| 19. Retail Inventory | *Chapter 11B — Retail Inventory and Supply Chain Intelligence* |
| 20. Medical Imaging Triage | *Chapter 11C — Medical Imaging Triage* |
| 21. Document Intelligence, Claims, Environmental | *Chapter 13 — Document Intelligence, Insurance Claims, and Environmental Monitoring* |
| 22. Security, Accessibility, Art | *Chapter 15 — Security Surveillance, Accessibility Compliance, and Art Authentication* |
| 23. Trust and Authenticity at Point of Sale | *Chapters 16A-D — Counterfeit Screening, Reconciliation, Expense Audit, Luxury Authentication* |
| 24. Natural Language Photo Editing | *Chapter 17 — Natural Language Photo Editing* (both drafts) |
| 25. Body-Worn and Personal Cameras | *Chapters 18A-B — Police Body Camera, Exercise/Nutrition Tracking* |
| 30. Mathematical Foundations | *Production Guide — From Mathematical Foundations to CUDA Kernels, Version 2* (math sections) |
| 31. Continuous Batching and Production Serving | *Production Guide* (serving architecture + numerical debugging sections) |
| 32. Flash Attention and CUDA Kernels | *Production Guide* (Flash Attention + CUDA sections) |

**New material this TOC adds** (not present in your drafts, written to give the book its edge-first spine): Chapter 1's edge-vs-cloud constraint framing, and Appendices A, C, D, E, and F, which consolidate material scattered across many chapters' own "Quick Reference" sections into dedicated, cross-referenced appendices in the CUDA book's style.
