# Appendix C: A Decision-Tree Reference: Quantization, Threading, and Cache Strategies at a Glance

This appendix builds nothing new. Every format, every partitioning rule, every cache strategy named below was already derived and verified somewhere in Parts 1 through 3 (and, for the threading crossover formula, Part 6) -- this appendix's only job is to pull those real, scattered conclusions into one place a reader can consult while making a real decision, with an exact pointer back to where each one was built. Each of the three sections below closes with a small, real, compiled advisor program that turns its own decision tree into an executable function returning both the recommendation and the specific chapter fact behind it -- restating a decision as code, exactly this book's own standing discipline for every other real claim, rather than leaving it as a diagram a reader has to trust on faith.

---

## C.1 Choosing a Quantization Strategy

### Intuition

Chapters 4, 6, and 7 built five real formats -- F32, Q8_0, Q4_0, dynamic per-token INT8, and TurboQuant -- and none of them is simply "better." Each one trades away a different thing: Q8_0 and Q4_0 trade reconstruction accuracy for file size on data that will be calibrated once, offline, with unlimited time; dynamic INT8 trades a persisted, shrinkable scale for the ability to quantize data that does not exist until the current token is being generated; TurboQuant trades a per-vector rotation cost for the ability to compress data with no calibration pass available at all. The real question this section answers is never "which format is best" -- it is "what kind of data is this, and what does that kind of data actually need."

### The Concept, In Detail

The decision tree below reproduces exactly the reasoning Chapters 4.5, 6.1, 6.3, and 7.4-7.5 already worked through, just walked in the order a reader would actually ask the questions:

```
Is this tensor a KV cache entry (online, never calibrated ahead of time)?
  YES -> TurboQuant, MSE mode                                   (Chapter 7.4)
         -- codebooks depend only on d and bits, not on the data itself,
            so there is nothing to calibrate; MSE mode (not the unbiased
            QJL "Prod" mode) wins at the KV cache's typical 3-4 bit range.

  NO -> Is it an activation, quantized fresh on every single token?
          YES -> Dynamic INT8, plain float scale                 (Chapter 4.5)
                 -- nothing is ever written to a file, so there is no
                    persisted scale worth shrinking to fp16.

          NO -> It is a static weight, quantized once, offline. Which role?
                  NORM      -> F32                                (Chapter 6.3)
                               -- element count is negligible at any scale.
                  EMBEDDING -> Q8_0                                (Chapter 6.3)
                               -- element count (2 * VOCAB * DIM) can dwarf
                                  every other category combined.
                  ATTENTION -> Q8_0                                (Chapter 6.1)
                               -- feeds a discrete softmax-argmax decision
                                  that a given error size flips at a
                                  measurable, non-negligible rate.
                  FFN       -> Q4_0                                (Chapter 6.1 / 6.3)
                               -- the same-order-of-magnitude error stays a
                                  continuous, bounded output error, and FFN
                                  matrices outnumber attention matrices
                                  roughly 3:1 per layer.
```

| Tensor category | When it's quantized | Real format | Established in |
|---|---|---|---|
| Norm weights | Once, offline | F32 | Chapter 6.3 |
| Embedding table | Once, offline | Q8_0 | Chapter 6.3 |
| Attention (Q, K, V, O) | Once, offline | Q8_0 | Chapter 6.1 |
| FFN (gate, up, down) | Once, offline | Q4_0 | Chapter 6.1 / 6.3 |
| Activations | Fresh, every token | Dynamic INT8 | Chapter 4.5 |
| KV cache entries | Online, uncalibrated | TurboQuant, MSE mode | Chapter 7.4 / 7.5 |

!!! warning "[COMMON TRAP] treating TurboQuant as a drop-in replacement for blockwise weight quantization"
    Chapter 7.5 was explicit about this: TurboQuant's own `O(d^2)` per-vector rotation cost "does not belong inside a fused SIMD matmul inner loop the way a per-block integer scale does." Static weights are quantized once with unlimited calibration time -- exactly the case Q8_0 and Q4_0 already handle well. TurboQuant's real advantage is specifically for online, uncalibrated data like a live KV cache, not a substitute for offline weight formats.

!!! warning "[COMMON TRAP] treating 'small' and 'sensitive' as one budget category"
    Chapter 6.3's own real trap: norm weights and embedding tables both sound like they deserve careful treatment, but only one of them is actually small in element count. An embedding table's size scales with vocabulary and can exceed every other tensor category combined at real vocabulary sizes -- check that a population is actually small before budgeting for it as though it were.

```bash
g++ -std=c++23 -Wall -Wextra -O2 c1_quantization_format_advisor.cpp -o c1_quantization_format_advisor
./c1_quantization_format_advisor
```

@@CODE1@@

@@OUT1@@

---

## C.2 Choosing a Threading and Parallelization Strategy

### Intuition

Chapters 8 through 11 answer three genuinely separate questions that are easy to collapse into one: whether a given operation is memory-bound or compute-bound at all (which decides whether quantization even helps), whether adding more threads is still worth doing (which Amdahl's law bounds, not intuition), and which axis to actually split an operation across (which decides whether the parallel result is correct and reproducible at all, not merely fast). None of these three questions has a single "use more threads" answer -- each has its own real formula or rule this book already derived and verified.

### The Concept, In Detail

```
Step 1 -- Is this linear layer memory-bound or compute-bound at your batch size?

    crossover_batch_size(ridge_point) = 2 * ridge_point           (Chapter 30.1)
    ridge_point = peak_compute_flops_per_sec / peak_bandwidth_bytes_per_sec

    batch_size <  crossover -> memory-bound:
                                shrinking BYTES (quantization) is the real lever.
    batch_size >= crossover -> compute-bound:
                                shrinking FLOPs (a faster kernel) is the real lever;
                                quantization's byte-count advantage stops mattering
                                once both formats hit the same peak-compute ceiling
                                (Chapter 8.3).

Step 2 -- Is adding more threads to this workload still worth it?

    amdahl_speedup(serial_fraction, n) = 1 / (serial_fraction + (1 - serial_fraction) / n)
    amdahl_ceiling(serial_fraction)    = 1 / serial_fraction        (Chapter 10.1)

    Compare the speedup at your candidate thread count against the ceiling --
    a workload's own serial fraction sets a real limit no thread count can
    ever cross, and most of the benefit arrives long before that limit does.

Step 3 -- Which axis do you actually partition THIS operation across?

    A matmul's OUTPUT rows        -> row-parallel                  (Chapter 10.3 / 11.1)
                                      bit-for-bit identical to serial, no combine step.
    GQA attention, across heads   -> group-aligned head-parallel   (Chapter 11.3)
                                      whole KV groups per thread, never raw head index.
    A cross-element reduction     -> fixed-block reduction         (Chapter 11.4 / 11.5)
                                      block boundaries independent of thread count.
```

| Question | Real formula / rule | Established in |
|---|---|---|
| Memory-bound or compute-bound? | `crossover_batch_size = 2 * ridge_point` | Chapter 30.1 |
| Worth adding more threads? | `amdahl_ceiling = 1 / serial_fraction` | Chapter 10.1 |
| Partitioning a matmul | Row-parallel over output rows | Chapter 10.3 / 11.1 |
| Partitioning GQA attention | Group-aligned, by whole KV group | Chapter 11.3 |
| Partitioning a reduction | Fixed-block, independent of thread count | Chapter 11.4 / 11.5 |

!!! warning "[COMMON TRAP] assuming N cores buys N times the speedup"
    Chapter 10.1's own worked example: an 8-core machine on a workload with just a 5% serial fraction gets roughly 5.93x, not 8x -- and no thread count on that same workload can ever exceed the true ceiling of 20x. The ceiling is set entirely by the serial fraction, not by how many cores are thrown at the problem.

!!! warning "[COMMON TRAP] assuming a bigger model changes GEMV's own memory-bound verdict"
    Chapter 30.1's own finding: a bigger model adds proportionally more FLOPs AND proportionally more bytes, so a linear layer's arithmetic intensity barely moves with model size. The only real lever that moves arithmetic intensity for a linear layer is batch size (continuous batching), never the model's own parameter count.

```bash
g++ -std=c++23 -Wall -Wextra -O2 c2_threading_strategy_advisor.cpp -o c2_threading_strategy_advisor
./c2_threading_strategy_advisor
```

@@CODE2@@

@@OUT2@@

---

## C.3 Choosing a KV Cache Strategy

### Intuition

Chapters 13 and 14 built five real KV cache techniques, and the temptation is to treat the most recently built one as the strict upgrade over everything before it. It isn't. A plain ring buffer is the right answer when the maximum context length is already known and small; a paged, block-table allocator earns its own bookkeeping overhead only once that length becomes genuinely unpredictable; sliding-window attention is free correctness for a model actually trained with one and only an empirical approximation otherwise; H2O eviction only becomes necessary once paging alone can no longer keep every token resident; and prefix caching is not a competing cache design at all, but a layer on top of whichever paged cache is already in use.

### The Concept, In Detail

```
Is this a multi-turn conversation, reusing shared history across turns?
  YES -> Prefix caching, layered on a paged cache            (Chapter 14.3 / 14.4)
         -- credits only tokens the runtime cache still actually holds,
            never token-ID equality alone.

  NO (or once the above is handled) -- was the model explicitly trained
  with a fixed attention window (e.g. Mistral 7B's own 4096-token window)?
  YES -> Sliding window                                       (Chapter 14.1)
         -- exactly what the model learned to expect, not an approximation
            of full attention, for THIS model.
  NO  -> What is the real context-length regime?
           Short, known, and small enough to reserve up front
             -> Plain ring buffer                              (Chapter 13.3)
                -- write-slot and RoPE-position counters kept separate.
           Long and genuinely unpredictable
             -> Paged, block-table allocator                   (Chapter 13.2)
                -- fixed-size power-of-two blocks plus a free list.
           Very long, memory-constrained even with paging
             -> Paged allocator + H2O importance eviction      (Chapter 13.4)
                -- evicts the least-useful token, with a recent-token
                   exemption, instead of merely the oldest.
```

| Real scenario | Recommended strategy | Established in |
|---|---|---|
| Multi-turn conversation, shared history | Prefix caching on a paged cache | Chapter 14.3 / 14.4 |
| Model trained with a fixed attention window | Sliding window | Chapter 14.1 |
| Short, known, bounded context | Plain ring buffer | Chapter 13.3 |
| Long, unpredictably growing context | Paged, block-table allocator | Chapter 13.2 |
| Very long, memory-constrained context | Paged allocator + H2O eviction | Chapter 13.4 |

Chapter 13.1's own exact bytes-per-token formula is worth having at hand for sizing any of the above: for each layer, each GQA KV head (not query head), per cached token, 2 vectors (K and V) of `head_dim` elements each. A Llama-3-8B-shaped configuration (32 layers, 8 KV heads, head_dim 128, FP16) costs exactly 128 KiB per token -- negligible at a few hundred tokens, but exactly 16 GiB at a 128K-token (2^17) context, which is why the regime genuinely changes as context length grows, not merely the convenience of which technique to reach for.

!!! warning "[COMMON TRAP] treating sliding window as free correctness for a model that wasn't trained with one"
    Chapter 14.1's own real distinction: for a model trained with unrestricted attention, a sliding window is only an empirical approximation that "tends to work well in practice... because attention weight empirically falls off sharply with distance" -- not a guarantee. A task needing precise recall of something outside the window can fail silently, with no crash or warning at all.

!!! warning "[COMMON TRAP] crediting prefix caching for tokens the cache no longer actually holds"
    Chapter 14.4's own worked example: a 100-token history can match a new turn's prefix token-for-token while a tight eviction policy has already evicted all but 20 of those entries -- so only 20 tokens are actually reusable, not 100. Prefix-caching logic must check actual cache membership, never token-ID equality alone.

```bash
g++ -std=c++23 -Wall -Wextra -O2 c3_kv_cache_strategy_advisor.cpp -o c3_kv_cache_strategy_advisor
./c3_kv_cache_strategy_advisor
```

@@CODE3@@

@@OUT3@@

---

## Appendix Summary

None of the three advisors in this appendix introduced a new technique, a new formula, or a new number. Each one is Chapters 4 through 14 (and Chapter 30's crossover formula) restated as a callable function, because a decision tree a reader has to hold in their head while making a real choice is worth less than one they can call and get a cited answer back from. The three real lessons underneath all three sections are the same lesson wearing three different hats: a format, a partitioning axis, or a cache technique is never simply better in the abstract -- it is only ever the right answer to a specific, real question about the data (is it calibrated once or generated per-token?), the workload (is it memory-bound or compute-bound at this batch size?), or the deployment (is the context length known, growing, or already too large to keep entirely resident?).

## Where We Go Next

Appendix D turns from choosing a strategy to measuring whether a real deployment is actually getting what a chosen strategy promised: hardware bandwidth surveys across real edge silicon, latency-budget arithmetic, `perf`-based profiling with the flags that actually work on Arm, and this book's own honest-numbers discipline applied one more time -- deterministic operation counts locked and verified, genuinely variable wall-clock timing kept explicitly out of the locked contract.
