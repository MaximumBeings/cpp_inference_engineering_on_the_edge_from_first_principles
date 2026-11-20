# Chapter 17: Benchmarking Against llama.cpp and the Broader Ecosystem

<!-- edge:opener:start -->
<figure class="chapter-figure" markdown="1">

[![Chapter 17: Measure the full request lifecycle](../images/chapter-openers/17.svg?v=175c39078322)](../images/chapter-openers/17.svg?v=175c39078322)

<figcaption>Chapter roadmap. Open the image for a full-size view.</figcaption>
</figure>
<!-- edge:opener:end -->

**What you will understand by the end of this chapter:**

- Why "how fast is it" is not one number but at least three genuinely different ones -- time-to-first-token, tokens-per-second, and time-per-output-token -- and why elapsed throughput must be distinguished from median latency.
- How to measure a process's own real peak memory use directly from the kernel (`/proc/self/status`'s `VmHWM`), rather than estimating it from a model's file size or parameter count.
- How to build a benchmarking harness that times this book's own REAL forward pass -- Chapter 16's own `decode_step` and `prefill`, unchanged -- rather than a simulated bandwidth model, including a warm-up window that excludes a memory-mapped file's real, one-time page-fault cost from steady-state statistics, and a KV-capacity guard that refuses to silently report statistics computed past a cache's own capacity.
- Why a synthetic model's own metadata can corrupt a correct forward pass in a way that has nothing to do with tensor bytes at all: a missing `rope.freq_base` key, defaulting to 0.0, drives this book's own RoPE formula straight into a `0 * infinity` NaN before a single real weight is even read.
- How to define a matching comparison protocol against llama.cpp, and why historical reference rates cannot support a current-machine speed ratio.

**What you need to know first:**

- Chapter 16's complete production engine, in particular Section 16.3's `decode_step` and `prefill` -- this chapter's own benchmarking harness calls these exact, already-verified functions, unchanged, rather than building a second timing path that would need its own independent trust.
- Chapter 15's real, correct first token and the actual downloaded checkpoint (`qwen2.5-0.5b-instruct-q8_0.gguf`) this chapter benchmarks -- the same file Chapters 15 and 16 have used throughout, so this chapter's own comparison is against the identical numerical result those chapters already confirmed correct.
- This chapter's own honest-exception shape follows Chapter 16's directly. Sections 17.1 and 17.2 have no real-file mode at all: each proves its own new machinery (statistics, timing definitions, a from-scratch harness) against a small synthetic model, exactly as Sections 16.1 through 16.3 did. Section 17.3, like Section 16.4, has a real-file mode that is a completely separate invocation from its own self-test mode, because a real run's wall-clock throughput is exactly the one kind of output that MUST differ from machine to machine -- locking it into this book's four-way byte-identical cross-check would be locking in a claim no other reader's hardware could ever reproduce.

---

!!! warning "Trusted instructional inputs only"
    The GGUF readers copied into this chapter assume the documented trusted fixtures and compatible checkpoints. They are not hardened arbitrary-file parsers: read success, metadata allocation limits, dimension products, alignment and every tensor byte range are not comprehensively validated. A successful `open()` does not establish that an arbitrary file is safe. Validate those properties before exposing mapped views in production, and do not use these examples on untrusted model files.


Chapter 16 integrated a forward pass, token selection and streaming. This chapter separates the questions a benchmark must answer: how much work was timed, how elapsed throughput differs from latency quantiles, and whether a reference run actually matches the current host and workload. The current program measures the local forward pass only. Historical real-model captures are preserved below with explicit limitations; they are not a live comparison service.

<!-- chapter-figure -->
<figure class="chapter-figure" markdown="1">

![Measure the full request lifecycle. Warm-up and the measured decode interval must be explicitly separated.](../images/17-overview.svg?v=84ae552d1ebb){: loading="lazy" width="760" }

  <figcaption>Report model, hardware, flags and measurement boundaries with every comparison.</figcaption>
</figure>
<!-- /chapter-figure -->

## 17.1 What to Measure and Why

<!-- edge:concept-17.1:start -->
<figure class="chapter-figure" markdown="1">

[![Latency and throughput answer different questions](../images/technical/ch17/17-01.svg?v=5ad7289922bf)](../images/technical/ch17/17-01.svg?v=5ad7289922bf)

<figcaption>Declare workload, token counts, batch size and timing boundaries for every reported number.</figcaption>
</figure>
<!-- edge:concept-17.1:end -->


### Intuition

"How fast is it" hides a real ambiguity. A user waiting for a chatbot to start responding cares about one thing; a user watching text stream in afterward cares about a different one; and neither of those is the same as "how much memory did this cost to run," which no earlier chapter in this book ever had reason to measure at all.

### The Concept, In Detail

Time-to-first-token (TTFT) covers the request path until the first output becomes available. A simplified in-process model can sum prefill and first-token selection, but queueing, tokenization, decoding and transport must be included when measuring a user-visible service. **Elapsed decode throughput** is measured tokens divided by their total elapsed decode time, equivalently `1000 / mean(step_ms)` for sequential one-token steps. **Mean TPOT** is that mean step time. Median/P95/P99 latencies are separate distribution summaries; reciprocal median latency is not elapsed throughput.

The median describes a typical step, while mean TPOT and elapsed throughput retain the effect of slow measured steps. For `[1,1,100]` milliseconds, median latency is 1 ms but elapsed throughput is `3/0.102`, about 29.41 tokens/s, not 1000 tokens/s. The statistics helper reports mean, median, P95, P99, min and max. For even-size samples, median is the average of the two central values.

A specific, real effect motivates one more definition: `is_warmup_step` marks the first several decode steps as excluded from every statistic above. This book's real forward pass has no JIT and no bytecode cache to warm up -- but the memory-mapped weight file genuinely does have a cold-start cost. The OS has not yet paged in every tensor's bytes on the very first few accesses to a freshly `mmap`ed file, so the first several real decode steps pay a real, one-time page-fault cost that a steady-state throughput claim should not be charged for. Excluding a warm-up window is not throwing away inconvenient data; it is separating a one-time cost from the recurring one the statistic is actually trying to characterize.

The one measurement this chapter needs that no earlier chapter did is memory. `peak_rss_kb` reads `/proc/self/status`'s `VmHWM` line directly from the kernel -- a REAL measurement of this specific process's actual high-water-mark memory use, not an estimate computed from a model's file size or parameter count. Producing one correct token, or one correct conversation, says nothing on its own about how much memory doing so actually cost; comparing against `llama.cpp` needs exactly this number, because a faster engine that also uses ten times the memory is not an unambiguous improvement. `peak_rss_kb` returns -1, a real "measurement unavailable here" outcome, on a platform without `/proc` -- distinct from, and never confused with, "measured zero."

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_benchmark_stats_and_definitions.cpp -o 01_benchmark_stats_and_definitions
./01_benchmark_stats_and_definitions
```

**Sample input:** known hand-computable latency arrays, including an uneven `[1,1,100]` ms example distinguishing elapsed throughput from reciprocal-median rate; empty input; a conceptual TTFT sum; a warm-up window; and a platform-specific check that Linux peak RSS increases after allocating and touching memory. Timing values measured on the host are informational rather than locked benchmark claims.

@@OUT1@@

!!! warning "[COMMON TRAP] reserving memory is not the same as using it"
    A bare `std::vector<uint8_t> grow(200'000'000)` reserves 200MB of address space, but on a system with lazy page allocation, the kernel is free to satisfy that reservation with pages it never actually backs with real memory until something writes to them -- `VmHWM` can stay completely unchanged after a `.resize()` alone, making a test that only reserves memory and checks for `VmHWM` growth silently measure nothing. The fix is not a bigger reservation; it is writing to every page (`for (size_t i = 0; i < GROW_BYTES; i += 4096) grow[i] = ...`), which forces the kernel to actually back each page with physical memory before `VmHWM` has anything real to report growing.

## 17.2 A From-Scratch Benchmarking Harness

<!-- edge:concept-17.2:start -->
<figure class="chapter-figure" markdown="1">

[![A benchmark must control its measurement scope](../images/technical/ch17/17-02.svg?v=a90f4a89f0e7)](../images/technical/ch17/17-02.svg?v=a90f4a89f0e7)

<figcaption>Asynchronous GPU work needs completion before timing ends. Include or exclude setup consistently.</figcaption>
</figure>
<!-- edge:concept-17.2:end -->


### Intuition

Section 17.1 defined what to measure. This section wires those definitions into a REAL benchmarking harness that runs Chapter 16's own `decode_step` and `prefill` -- not a simulated bandwidth model standing in for a forward pass, and not a placeholder assuming some other file will supply the real computation later. Benchmarking code that never runs the thing it claims to measure cannot actually measure it.

### The Concept, In Detail

This section's own tokenizer needs are minimal, almost to the point of having none: a benchmarking harness measures how fast a fixed number of forward-pass POSITIONS run, and does not need real text at all -- arbitrary token IDs within the model's own vocabulary range exercise the identical `decode_step` code path a real generated token would. This file therefore omits the GPT-2 byte codec, the BPE engine, and the entire sampling pipeline: none of them are on the path being measured, and including them would only give the timing loop more to account for without changing what it is actually trying to isolate.

`run_benchmark` is explicitly a **forward-pass microbenchmark**. It measures prefill-only latency and subsequent decode-step latency, not full-generation TTFT. The workload feeds random in-range token IDs into the forward pass. A random integer draw does not perform argmax or scan a vocabulary-sized probability distribution. Token selection, text decoding, streaming, request queueing and transport are excluded. Warm-up decode steps are excluded from the reported interval, and the capacity guard reports the actual number of measured steps.

`detect_slowdown` is this section's own thermal-throttle-style detection algorithm: it compares the median of a run's first quarter of per-step latencies against the median of its last quarter, and flags a slowdown when the back half is more than 15% slower than the front half. This section's own self-test checks this algorithm against a HAND-CONSTRUCTED, not measured, latency sequence -- fully deterministic and reproducible on every architecture, because the test is checking the DETECTION ALGORITHM's own logic, not real hardware behavior, and a synthetic sequence with a known, exact 40% back-half slowdown is a far more precise way to confirm that logic than waiting for a real machine to happen to throttle during a test run.

Building this harness surfaced a real bug, and it is worth tracing exactly, because nothing about it lives in tensor data at all. An early draft of this section's synthetic model writer omitted the `qwen2.rope.freq_base` metadata key entirely, reasoning that `get_f32`'s documented default of 0.0 for a missing key would be harmless, since this harness only measures timing and never checks numerical correctness. That reasoning was wrong, and this book's own NaN-detection machinery caught it immediately: `RoPETables` computes `theta = 1 / pow(base, 2k/head_dim)` for each dimension pair `k`, and for `base = 0.0`, every `k > 0` gives `pow(0, positive) = 0`, so `theta = 1/0 = +infinity`. At position 0, the very first angle computed is `0 * infinity` -- and IEEE 754 defines that product as NaN, not zero. That NaN then propagates through `cos`/`sin` into the query and key vectors of layer 0, and Chapter 16's own `nan_at_layer` detection correctly refused to report benchmark statistics computed downstream of it: `run_benchmark` returned `ok=false` on what should have been a routine timing run against a model this section's own writer had produced. The fix is the same lesson Section 16.3 already taught about `kv_count`, applied to a different field entirely: declaring a default "harmless" is not the same as tracing what the actual formula does with it. Writing the real `qwen2.rope.freq_base = 10000.0` -- the value every earlier chapter in this book already uses -- avoids the singularity at its source rather than working around its symptom.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_benchmark_harness.cpp -o 02_benchmark_harness
./02_benchmark_harness
```

**Sample input:** `run_benchmark`, run against a small synthetic model, checked for structural invariants that must hold regardless of how fast any given machine happens to run it -- correct prompt and decode-step counts, `min <= median <= p95 <= p99 <= max`, a positive tokens-per-second, and real, measured (not merely nonzero-by-construction) peak RSS; a deliberately undersized KV cache checked to stop decoding early rather than silently reporting statistics computed past its own capacity; and `detect_slowdown` checked against both a perfectly steady hand-constructed sequence (no slowdown reported) and one with a genuine, exact 40% back-half slowdown (correctly detected).

@@OUT2@@

!!! warning "[COMMON TRAP] a \"harmless\" default is only harmless if you trace what the formula does with it"
    `get_f32`'s documented behavior for a missing metadata key -- returning 0.0 rather than throwing -- is a reasonable default for a KEY THIS BOOK'S OWN READER CHOSE not to write. It is a different claim entirely to assume that default is SAFE for every formula that might read it, and `RoPETables`'s own `1 / pow(base, exponent)` is exactly the kind of formula where a "reasonable-looking" default of zero drives a division by zero into an infinity, and where multiplying that infinity by a position of exactly zero produces NaN rather than the zero a reader might naively expect "no rotation at position zero" to mean. The general lesson: a default value's safety is a property of the specific formula consuming it, not of the value in isolation -- and the only way to know is to trace the formula, not to assume "it's just zero" settles the question.

## 17.3 An Honest Comparison Against llama.cpp

<!-- edge:concept-17.3:start -->
<figure class="chapter-figure" markdown="1">

[![A comparison needs the same work on both sides](../images/technical/ch17/17-03.svg?v=68941899192e)](../images/technical/ch17/17-03.svg?v=68941899192e)

<figcaption>A schematic or analytical estimate must not be presented as a measured runtime comparison.</figcaption>
</figure>
<!-- edge:concept-17.3:end -->


### Intuition

Sections 17.1 and 17.2 built this book's own honest measuring stick: precise definitions and a harness that times this book's real forward pass. This section finally points that measuring stick at the question the whole chapter exists to answer: compared to `llama.cpp`, how good is the engine this book has built from scratch since Chapter 15?

### The Concept, In Detail

A valid engine comparison must align the host, model file/hash, precision, prompt/generation lengths, context, thread counts, batch settings, warm-up and timing boundaries. The current example measures this book's integrated single-threaded forward pass, but **does not measure llama.cpp**. It therefore prints no current relative-speed ratio. To compare, run both implementations under an explicitly matching protocol and use elapsed-throughput statistics for both.

Producing a real `llama.cpp` binary to compare against required rebuilding it from source, since Chapter 15's own cleanup deleted the copy that chapter had used. `llama-bench` -- `llama.cpp`'s own standardized throughput-measurement tool -- reports exactly the two numbers this chapter's own definitions need, `pp<N>` and `tg<N>`, at a stated batch size and thread count, formatted precisely enough to quote directly rather than parse out of a differently-shaped tool's output.

This section's own harness code deliberately does not shell out to that `llama-bench` binary at runtime. Two independent reasons rule that out, not one. First, this book has no `llama.cpp` source tree or built binary bundled with it -- this section's own file has nothing to link against or invoke unless a reader has separately rebuilt `llama.cpp` themselves, and a program whose self-tests only pass on a machine with a second project pre-installed would not be self-contained the way every other file in this book has been. Second, and more fundamentally, shelling out to a binary this book did not build and did not verify would not be measuring anything this book could stand behind with the same confidence as Chapter 15's own numerical cross-check against `llama-eval-callback`. What this section's real-file mode DOES do honestly is measure this book's own engine freshly, in real time, against the real file -- and print that fresh, real measurement immediately alongside `llama-bench` reference numbers captured once, documented, and never silently updated or cherry-picked.

Running that comparison surfaced a second real, honest finding, and -- following this book's own standing practice all the way back to Chapter 14.3's treatment of an unverified performance claim -- it is reported exactly as measured rather than smoothed over. Across two independent real runs on the real device, prompt processing measured SLOWER, per token, than token generation: roughly half the throughput, consistently, in both runs. This is initially counterintuitive, because token generation's positions sit strictly LATER in the sequence than prefill's, meaning each decode step's own attention pass scans a longer history than the prefill step immediately before it -- if anything, decode should be the more expensive phase per token, not the cheaper one. The most likely real explanation is a phenomenon this chapter's own old reference material's methodology notes already named as a factor worth tracking: CPU frequency scaling. A `schedutil`-style governor ramps clock speed up gradually in response to SUSTAINED load; prefill is this program's first sustained burst of heavy computation, run at whatever clock speed the governor had settled on beforehand, while decode continues immediately afterward, now benefiting from however far the governor had ramped up during prefill's own execution. This book's own engine has no code path that would produce this pattern on its own -- attention cost is a small fraction of a 24-layer, 896-dimension model's total per-token cost next to its matmuls, which are position-independent -- so a hardware-level explanation is the honest one to report, rather than inventing a software cause this measurement gives no evidence for. The practical lesson for this chapter's own methodology: a single-shot prefill measurement, unlike decode's own warm-up-excluded steady state, has no equivalent window to exclude a one-time ramp-up cost from, and a reader repeating this exact comparison on their own hardware should expect their own prefill-versus-decode ratio to vary for the same reason.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_llamacpp_comparison.cpp -o 03_llamacpp_comparison
./03_llamacpp_comparison                                                                          # self-tests only, reproduces everywhere
./03_llamacpp_comparison /path/to/qwen2.5-0.5b-instruct-q8_0.gguf --pp 32 --tg 32                  # real comparison run (requires the downloaded file)
```

**Sample input:** this section's own copy of the forward pass and `run_benchmark` re-verified against a synthetic model exactly as Section 17.2 already proved (structural invariants, KV-capacity exhaustion, `detect_slowdown`'s own logic), confirming this file's real-file mode is built on machinery this book has already independently checked; and, in real comparison mode, this book's own engine benchmarked fresh against the actual downloaded checkpoint at `llama-bench`'s own `pp32`/`tg32` convention, printed immediately alongside `llama-bench` reference numbers captured once on this book's own real device, at both one thread (the direct, same-thread-count comparison) and this machine's full core count (labeled honestly as a different comparison).

@@OUT3@@

**Sample real-run output** (executed once, on a reader's own machine, against the actual downloaded checkpoint and a real, from-source `llama.cpp` build; not reproduced in this book's four-way cross-check, because a real wall-clock throughput number is exactly the one kind of output that must differ machine to machine):

@@OUT3REAL@@

!!! warning "Historical captures are not current benchmark references"
    The real-run text below is retained as a historical capture of an earlier program. Its fixed NEON reference rates and reciprocal-median rate are not a validated current comparison. Do not relabel those rates for a different host or token count, and do not use its quoted speed ratio as current evidence. The corrected program reports only its fresh local forward-pass measurements and requires separately measured matching reference results before any ratio is meaningful.

## Chapter Summary

This chapter separates elapsed throughput from latency quantiles, demonstrates a forward-pass timing harness with an explicit scope, and explains the controls required for a comparison against another runtime. The current harness excludes sampling, decoding and transport, so its prefill-only latency is not service TTFT. Historical real-model captures remain labeled as historical data; their rates are never substituted for a fresh matching reference run.

## Self-Check Questions

1. What different information do median latency, mean TPOT and elapsed tokens/second provide?
2. For step times `[1,1,100]` milliseconds, calculate elapsed throughput and explain why reciprocal-median rate is misleading.
3. Why must a benchmark declare its warm-up policy instead of assuming a few excluded steps remove every cold-start effect?
4. Why does a missing RoPE base of zero generate non-finite values for positive frequency exponents?
5. Why must a benchmark report the number of steps actually executed when the KV-capacity guard stops it early?
6. Why does the corrected program omit a ratio against its previously hardcoded llama.cpp reference rates?
7. Which conditions should match before comparing the book engine with llama.cpp, and why should different thread counts stay separate?
8. Why is the random-ID forward-pass harness not a measurement of full-generation TTFT?

## Where We Go Next

This chapter gives the integrated engine a scoped measurement path and defines what evidence is required before comparing implementations. Kernel-only timing, full generation latency and end-to-end service latency are different measurements. Part 5 uses these distinctions when discussing vision-language integration examples.

## Worked Solutions

**1.** Median describes a typical individual step; mean TPOT includes all measured step times; elapsed throughput is completed tokens divided by their total measured duration. A low median can coexist with poor elapsed throughput when a minority of steps are slow.

**2.** Total time is 102 ms for three tokens, so throughput is `3000/102`, about 29.41 tokens/s. Median latency is 1 ms; its reciprocal would suggest 1000 tokens/s and ignore the slow step.

**3.** Page residency, frequency scaling, cache state and other initialization effects depend on the workload and system. Excluding declared warm-up work defines a measurement interval; it does not prove every transient effect disappeared.

**4.** With a positive exponent, `pow(0, exponent)` is zero, so its reciprocal is infinity. At position zero the angle becomes `0 * infinity`, which is NaN. Metadata validation should reject an invalid base before creating RoPE tables.

**5.** A rate must use the actual completed work and its corresponding duration. Reporting the requested count after early termination fabricates work that did not run.

**6.** Those rates belong to a historical device, model, workload and statistic. A new local measurement cannot inherit their provenance; changing token-count labels does not remeasure the reference. A current ratio needs fresh matching reference evidence.

**7.** Match hardware, model/hash and precision, token/context lengths, thread counts, batch settings, warm-up and timing scope. Different thread counts change parallel resources as well as implementation, so their rates answer a different question.

**8.** Drawing a random integer does not scan logits, run the sampler or decode and stream text. The harness isolates prefill and decode forward passes. Full-generation or service TTFT requires timing the corresponding additional stages.
