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

```cpp
// Appendix C.1 -- A Quantization Format Advisor.
//
// This file does not introduce a single new quantization technique. It
// turns the real, already-derived decision criteria from Chapter 4
// (Quantization Strategies), Chapter 6 (The Hybrid Quantized Engine), and
// Chapter 7 (TurboQuant) into one callable function that reproduces those
// chapters' own conclusions -- restated as executable code, not just prose,
// exactly this book's own standing discipline for every other real claim.
//
// The four real facts this advisor encodes:
//   1. Chapter 6.3: norm weights are genuinely negligible in element count
//      regardless of scale, so there is no real memory reason to compress
//      them -- keep them F32.
//   2. Chapter 6.1: attention feeds a discrete softmax-argmax decision that
//      a given error size flips at a measurable, non-negligible rate, while
//      an embedding table's own element count (2 * VOCAB * DIM) can exceed
//      every other tensor category combined at real vocabulary sizes
//      (Chapter 6.3) -- both get the more careful Q8_0 format.
//   3. Chapter 6.1: an FFN matrix's same-order-of-magnitude rounding error
//      stays a continuous, bounded output error (no discrete decision it
//      can flip), and FFN matrices outnumber attention matrices roughly
//      3:1 per layer -- Q4_0 buys the largest absolute memory savings
//      exactly where precision is least fragile.
//   4. Chapter 4.5: activations are quantized fresh every token on the
//      critical path, never persisted to a file, so there is nothing to
//      shrink a stored scale for -- dynamic INT8 keeps a plain float scale.
//      Chapter 7.4/7.5: a KV cache entry is online, uncalibrated data with
//      no offline calibration pass available, exactly the case TurboQuant's
//      data-oblivious codebooks were built for; its MSE mode (not the
//      unbiased QJL "Prod" mode) is the right choice at the KV cache's
//      typical 3-4 bit range, where the inner-product bias QJL corrects
//      for is too small to be worth the extra residual stage.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 c1_quantization_format_advisor.cpp -o c1_quantization_format_advisor
// Run:     ./c1_quantization_format_advisor

#include <iostream>
#include <string>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// Reused verbatim from Chapter 6.2's own WeightRole enum.
enum class WeightRole { NORM, EMBEDDING, ATTENTION, FFN };

// What kind of tensor this is, at the level Chapter 4/7 distinguish it:
// a static weight (quantized once, offline, unlimited calibration time),
// an online activation (quantized fresh every token), or a KV cache entry
// (online, uncalibrated, no offline pass possible).
enum class StorageKind { STATIC_WEIGHT, ONLINE_ACTIVATION, KV_CACHE_ENTRY };

enum class RecommendedFormat { F32, Q8_0, Q4_0, DYNAMIC_INT8, TURBOQUANT_MSE };

std::string to_string(RecommendedFormat f) {
    switch (f) {
        case RecommendedFormat::F32: return "F32";
        case RecommendedFormat::Q8_0: return "Q8_0";
        case RecommendedFormat::Q4_0: return "Q4_0";
        case RecommendedFormat::DYNAMIC_INT8: return "DYNAMIC_INT8";
        case RecommendedFormat::TURBOQUANT_MSE: return "TURBOQUANT_MSE";
    }
    return "UNKNOWN";
}

struct FormatRecommendation {
    RecommendedFormat format;
    std::string reason;
};

FormatRecommendation recommend_quantization_format(StorageKind kind, WeightRole role) {
    if (kind == StorageKind::KV_CACHE_ENTRY) {
        return {RecommendedFormat::TURBOQUANT_MSE,
                "Chapter 7.4: online, uncalibrated cache data with no offline pass available -- "
                "MSE mode wins at the KV cache's typical 3-4 bit range without QJL's extra residual stage"};
    }
    if (kind == StorageKind::ONLINE_ACTIVATION) {
        return {RecommendedFormat::DYNAMIC_INT8,
                "Chapter 4.5: requantized fresh every token on the critical path -- nothing persisted "
                "to a file, so there is no stored scale worth shrinking to fp16"};
    }
    // StorageKind::STATIC_WEIGHT
    switch (role) {
        case WeightRole::NORM:
            return {RecommendedFormat::F32,
                    "Chapter 6.3: norm element count is negligible regardless of scale -- nothing real to save"};
        case WeightRole::EMBEDDING:
            return {RecommendedFormat::Q8_0,
                    "Chapter 6.3: embedding element count (2*VOCAB*DIM) can dwarf every other tensor "
                    "category at real vocabulary sizes -- the careful format is worth affording here"};
        case WeightRole::ATTENTION:
            return {RecommendedFormat::Q8_0,
                    "Chapter 6.1: attention feeds a discrete softmax-argmax decision a given error size "
                    "flips at a measurable, non-negligible rate"};
        case WeightRole::FFN:
            return {RecommendedFormat::Q4_0,
                    "Chapter 6.1: an FFN gate's rounding error stays a continuous, bounded output error, "
                    "and FFN matrices outnumber attention matrices roughly 3:1 per layer -- Q4_0's larger "
                    "absolute savings land exactly where precision is least fragile"};
    }
    return {RecommendedFormat::F32, "unreachable"};
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "Appendix C.1: The Quantization Format Advisor\n";
    std::cout << "====================================================\n\n";

    // -- Tests 1-4: the four real static-weight roles reproduce exactly
    // Chapter 6.2's own per-tensor policy (NORM->F32, EMBEDDING->Q8_0,
    // ATTENTION->Q8_0, FFN->Q4_0). --
    {
        auto norm = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::NORM);
        std::cout << "-- Test 1: static NORM weight -- format=" << to_string(norm.format) << " --\n";
        CHECK(norm.format == RecommendedFormat::F32);
    }
    {
        auto embed = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::EMBEDDING);
        std::cout << "-- Test 2: static EMBEDDING weight -- format=" << to_string(embed.format) << " --\n";
        CHECK(embed.format == RecommendedFormat::Q8_0);
    }
    {
        auto attn = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::ATTENTION);
        std::cout << "-- Test 3: static ATTENTION weight -- format=" << to_string(attn.format) << " --\n";
        CHECK(attn.format == RecommendedFormat::Q8_0);
    }
    {
        auto ffn = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::FFN);
        std::cout << "-- Test 4: static FFN weight -- format=" << to_string(ffn.format) << " --\n";
        CHECK(ffn.format == RecommendedFormat::Q4_0);
    }

    // -- Test 5: an online activation, regardless of which role it feeds,
    // always gets DYNAMIC_INT8 -- the STORAGE KIND decides here, not the
    // role. --
    {
        auto act = recommend_quantization_format(StorageKind::ONLINE_ACTIVATION, WeightRole::FFN);
        std::cout << "-- Test 5: online activation -- format=" << to_string(act.format) << " --\n";
        CHECK(act.format == RecommendedFormat::DYNAMIC_INT8);
        CHECK(act.reason.find("fresh every token") != std::string::npos);
    }

    // -- Test 6: a KV cache entry always gets TURBOQUANT_MSE -- again the
    // STORAGE KIND decides, not the role (a KV cache entry is always
    // attention-side data, but that is not why it gets this answer). --
    {
        auto kv = recommend_quantization_format(StorageKind::KV_CACHE_ENTRY, WeightRole::ATTENTION);
        std::cout << "-- Test 6: KV cache entry -- format=" << to_string(kv.format) << " --\n";
        CHECK(kv.format == RecommendedFormat::TURBOQUANT_MSE);
        CHECK(kv.reason.find("3-4 bit") != std::string::npos);
    }

    // -- Test 7: a small, hand-traceable memory-policy comparison,
    // reproducing Chapter 6.3's own real structural finding on a FRESH,
    // smaller illustrative model (not a reproduction of Chapter 6.3's own
    // specific multi-billion-parameter figures): a "naive" hybrid policy
    // that protects norm AND embedding at F32 together differs from the
    // "corrected" policy this advisor actually recommends (F32 norm only,
    // Q8_0 embedding) in EXACTLY one place -- the embedding format -- so
    // embedding accounts for the ENTIRE excess between them, by
    // construction, exactly as Chapter 6.3 found. Q8_0 costs 34 bytes per
    // 32-element block (Chapter 4.2) and Q4_0 costs 18 bytes per 32-element
    // block (Chapter 4.3); this advisor uses those same real per-block
    // byte counts as a per-element average, ignoring block-boundary
    // rounding, which is accurate enough for a comparison at this scale. --
    {
        constexpr double DIM = 8, N_LAYERS = 2, D_FF = 32, VOCAB = 50;
        constexpr double BYTES_PER_ELEM_F32 = 4.0;
        constexpr double BYTES_PER_ELEM_Q8_0 = 34.0 / 32.0;   // Chapter 4.2's own block size
        constexpr double BYTES_PER_ELEM_Q4_0 = 18.0 / 32.0;   // Chapter 4.3's own block size

        const double norm_elems = N_LAYERS * 2 * DIM + DIM;          // Chapter 6.3's own formula
        const double embed_elems = 2 * VOCAB * DIM;                  // Chapter 6.3's own formula
        const double attn_elems = N_LAYERS * 4 * DIM * DIM;          // Q, K, V, O per layer
        const double ffn_elems = N_LAYERS * 3 * DIM * D_FF;          // gate, up, down per layer

        auto norm_pick = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::NORM);
        auto embed_pick = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::EMBEDDING);
        auto attn_pick = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::ATTENTION);
        auto ffn_pick = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::FFN);
        CHECK(norm_pick.format == RecommendedFormat::F32);
        CHECK(embed_pick.format == RecommendedFormat::Q8_0);
        CHECK(attn_pick.format == RecommendedFormat::Q8_0);
        CHECK(ffn_pick.format == RecommendedFormat::Q4_0);

        const double norm_bytes = norm_elems * BYTES_PER_ELEM_F32;
        const double attn_bytes = attn_elems * BYTES_PER_ELEM_Q8_0;
        const double ffn_bytes = ffn_elems * BYTES_PER_ELEM_Q4_0;

        const double naive_hybrid_embed_bytes = embed_elems * BYTES_PER_ELEM_F32;  // the mistake: F32
        const double corrected_hybrid_embed_bytes = embed_elems * BYTES_PER_ELEM_Q8_0;  // this advisor's pick

        const double naive_hybrid_total = norm_bytes + naive_hybrid_embed_bytes + attn_bytes + ffn_bytes;
        const double corrected_hybrid_total = norm_bytes + corrected_hybrid_embed_bytes + attn_bytes + ffn_bytes;
        const double uniform_q4_0_total =
            (norm_elems + embed_elems + attn_elems + ffn_elems) * BYTES_PER_ELEM_Q4_0;
        const double uniform_q8_0_total =
            (norm_elems + embed_elems + attn_elems + ffn_elems) * BYTES_PER_ELEM_Q8_0;

        const double excess = naive_hybrid_total - corrected_hybrid_total;
        const double embedding_only_diff = naive_hybrid_embed_bytes - corrected_hybrid_embed_bytes;

        std::cout << "-- Test 7: small illustrative model (DIM=" << DIM << ", N_LAYERS=" << N_LAYERS
                  << ", D_FF=" << D_FF << ", VOCAB=" << VOCAB << ") --\n";
        std::cout << "   uniform Q4_0 total    = " << uniform_q4_0_total << " bytes\n";
        std::cout << "   uniform Q8_0 total    = " << uniform_q8_0_total << " bytes\n";
        std::cout << "   naive hybrid total    = " << naive_hybrid_total << " bytes (embedding kept F32)\n";
        std::cout << "   corrected hybrid total= " << corrected_hybrid_total << " bytes (embedding at Q8_0)\n";
        std::cout << "   excess (naive - corrected) = " << excess
                  << ", embedding-only difference = " << embedding_only_diff << "\n";

        // Corrected hybrid must sit strictly between the two uniform
        // policies, exactly as Chapter 6.3 found.
        CHECK(corrected_hybrid_total > uniform_q4_0_total);
        CHECK(corrected_hybrid_total < uniform_q8_0_total);
        // The naive-vs-corrected policies differ in exactly one place --
        // the embedding format -- so the embedding-only difference must
        // account for the WHOLE excess, not merely most of it.
        CHECK(excess == embedding_only_diff);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

```text
====================================================
Appendix C.1: The Quantization Format Advisor
====================================================

-- Test 1: static NORM weight -- format=F32 --
-- Test 2: static EMBEDDING weight -- format=Q8_0 --
-- Test 3: static ATTENTION weight -- format=Q8_0 --
-- Test 4: static FFN weight -- format=Q4_0 --
-- Test 5: online activation -- format=DYNAMIC_INT8 --
-- Test 6: KV cache entry -- format=TURBOQUANT_MSE --
-- Test 7: small illustrative model (DIM=8, N_LAYERS=2, D_FF=32, VOCAB=50) --
   uniform Q4_0 total    = 1624.5 bytes
   uniform Q8_0 total    = 3068.5 bytes
   naive hybrid total    = 4768 bytes (embedding kept F32)
   corrected hybrid total= 2418 bytes (embedding at Q8_0)
   excess (naive - corrected) = 2350, embedding-only difference = 2350

15/15 checks passed
ALL CHECKS PASSED
```

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

```cpp
// Appendix C.2 -- A Threading and Parallelization Strategy Advisor.
//
// The real, hard-won lesson across Chapters 8-11 is not "should I use
// threads" -- it is "which partition axis keeps the result correct and
// reproducible, and how many threads are actually worth adding." This file
// restates those chapters' own real formulas and partitioning rules as one
// callable advisor, changing none of them:
//
//   - Chapter 30.1's own crossover-batch-size formula, reused verbatim,
//     decides whether a linear layer at a given batch size is memory-bound
//     or compute-bound -- which in turn decides whether shrinking bytes
//     (quantization) or shrinking FLOPs (a faster kernel) is the lever
//     that actually helps.
//   - Chapter 10.1's own Amdahl's-law speedup and ceiling formulas, reused
//     verbatim, decide how many threads are still worth adding before a
//     workload's own serial fraction dominates the result.
//   - Chapter 10.3/11.1's own row-parallel-over-output-rows rule, Chapter
//     11.3's own group-aligned head-parallel rule, and Chapter 11.4/11.5's
//     own fixed-block-reduction rule decide WHICH axis to partition an
//     operation across, each one keeping this book's own bit-for-bit or
//     thread-count-independent reproducibility guarantee that a naive
//     partition of the same operation would silently break.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 c2_threading_strategy_advisor.cpp -o c2_threading_strategy_advisor
// Run:     ./c2_threading_strategy_advisor

#include <cmath>
#include <iostream>
#include <string>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// -- Part 1: Chapter 30.1's own crossover-batch-size formula, reused
// verbatim (AI(M) = M/2 for a weight-dominated GEMM, solved against the
// machine's own ridge point). --
enum class ComputeRegime { MEMORY_BOUND, COMPUTE_BOUND };

double crossover_batch_size(double ridge_point) {
    return 2.0 * ridge_point;  // Chapter 30.1: solving AI(M) = M/2 = ridge_point for M
}

ComputeRegime classify_batch(double batch_size, double ridge_point) {
    return batch_size < crossover_batch_size(ridge_point) ? ComputeRegime::MEMORY_BOUND
                                                            : ComputeRegime::COMPUTE_BOUND;
}

// -- Part 2: Chapter 10.1's own Amdahl's-law formulas, reused verbatim. --
double amdahl_speedup(double serial_fraction, int n_threads) {
    double total_relative_time = serial_fraction + (1.0 - serial_fraction) / n_threads;
    return 1.0 / total_relative_time;
}

double amdahl_ceiling(double serial_fraction) {
    return 1.0 / serial_fraction;  // the limit as n_threads -> infinity
}

// -- Part 3: which axis to partition an operation across. --
enum class OperationKind { MATMUL_ROW_OUTPUT, ATTENTION_HEADS_GQA, ELEMENTWISE_REDUCTION };
enum class PartitionStrategy { ROW_PARALLEL, GROUP_ALIGNED_HEAD_PARALLEL, FIXED_BLOCK_REDUCTION };

std::string to_string(PartitionStrategy s) {
    switch (s) {
        case PartitionStrategy::ROW_PARALLEL: return "ROW_PARALLEL";
        case PartitionStrategy::GROUP_ALIGNED_HEAD_PARALLEL: return "GROUP_ALIGNED_HEAD_PARALLEL";
        case PartitionStrategy::FIXED_BLOCK_REDUCTION: return "FIXED_BLOCK_REDUCTION";
    }
    return "UNKNOWN";
}

struct ThreadingRecommendation {
    PartitionStrategy strategy;
    std::string reason;
};

ThreadingRecommendation recommend_partition_strategy(OperationKind kind) {
    switch (kind) {
        case OperationKind::MATMUL_ROW_OUTPUT:
            return {PartitionStrategy::ROW_PARALLEL,
                    "Chapter 10.3/11.1: partitioning disjoint OUTPUT rows needs no combination step "
                    "and reproduces the serial result bit-for-bit; never combine a column-parallel "
                    "split's partial sums with a shared atomic -- well-defined is not the same as "
                    "reproducible run-to-run"};
        case OperationKind::ATTENTION_HEADS_GQA:
            return {PartitionStrategy::GROUP_ALIGNED_HEAD_PARALLEL,
                    "Chapter 11.3: partitioning query heads by raw index can split a single KV-group "
                    "across two threads, forcing that KV head's cache rows to be read redundantly by "
                    "more than one thread -- assign whole KV groups to threads instead"};
        case OperationKind::ELEMENTWISE_REDUCTION:
            return {PartitionStrategy::FIXED_BLOCK_REDUCTION,
                    "Chapter 11.4/11.5: thread-indexed chunking of a reduction like RMSNorm produces a "
                    "different bit pattern for every thread count tested; fixed-size blocks independent "
                    "of thread count are required, and this one reduction's own reproducibility gates "
                    "the entire pipeline's -- not merely its own phase's"};
    }
    return {PartitionStrategy::ROW_PARALLEL, "unreachable"};
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "Appendix C.2: The Threading and Parallelization Strategy Advisor\n";
    std::cout << "=========================================================\n\n";

    // -- Test 1: Chapter 30.1's own worked example reproduced exactly --
    // a machine with 1000 GFLOP/s peak compute and 25 GB/s peak bandwidth
    // has ridge point 40.0 FLOPs/byte, so crossover batch size is 80.0:
    // batch 79 is still memory-bound, batch 81 is already compute-bound. --
    {
        double ridge_point = 1000.0 / 25.0;  // GFLOPs/s / GB/s = FLOPs/byte
        double crossover = crossover_batch_size(ridge_point);
        std::cout << "-- Test 1: ridge_point=" << ridge_point << " -- crossover_batch_size=" << crossover << " --\n";
        CHECK(ridge_point == 40.0);
        CHECK(crossover == 80.0);
        CHECK(classify_batch(79.0, ridge_point) == ComputeRegime::MEMORY_BOUND);
        CHECK(classify_batch(81.0, ridge_point) == ComputeRegime::COMPUTE_BOUND);
    }

    // -- Test 2: Chapter 10.1's own worked example reproduced exactly --
    // a 5% serial fraction gives roughly 5.93x speedup at 8 threads, far
    // short of the true, unreachable ceiling of 20x. --
    {
        double serial_fraction = 0.05;
        double speedup_8 = amdahl_speedup(serial_fraction, 8);
        double ceiling = amdahl_ceiling(serial_fraction);
        std::cout << "-- Test 2: serial_fraction=" << serial_fraction << " -- speedup@8=" << speedup_8
                  << ", ceiling=" << ceiling << " --\n";
        CHECK(ceiling == 20.0);
        CHECK(std::abs(speedup_8 - 5.925925925925926) < 1e-9);
        CHECK(speedup_8 < ceiling);
    }

    // -- Test 3: the ceiling is a true limit -- even a millon threads on
    // the same 5% serial fraction never reaches it, only approaches it. --
    {
        double serial_fraction = 0.05;
        double speedup_huge = amdahl_speedup(serial_fraction, 1000000);
        double ceiling = amdahl_ceiling(serial_fraction);
        std::cout << "-- Test 3: speedup@1000000 threads=" << speedup_huge << " (ceiling=" << ceiling << ") --\n";
        CHECK(speedup_huge < ceiling);
        CHECK(speedup_huge > 19.99);
    }

    // -- Tests 4-6: the three real partition rules, reproduced exactly. --
    {
        auto matmul = recommend_partition_strategy(OperationKind::MATMUL_ROW_OUTPUT);
        std::cout << "-- Test 4: matmul output rows -- strategy=" << to_string(matmul.strategy) << " --\n";
        CHECK(matmul.strategy == PartitionStrategy::ROW_PARALLEL);
    }
    {
        auto attn = recommend_partition_strategy(OperationKind::ATTENTION_HEADS_GQA);
        std::cout << "-- Test 5: GQA attention heads -- strategy=" << to_string(attn.strategy) << " --\n";
        CHECK(attn.strategy == PartitionStrategy::GROUP_ALIGNED_HEAD_PARALLEL);
        CHECK(attn.reason.find("redundant") != std::string::npos);
    }
    {
        auto reduce = recommend_partition_strategy(OperationKind::ELEMENTWISE_REDUCTION);
        std::cout << "-- Test 6: elementwise reduction (e.g. RMSNorm) -- strategy=" << to_string(reduce.strategy) << " --\n";
        CHECK(reduce.strategy == PartitionStrategy::FIXED_BLOCK_REDUCTION);
        CHECK(reduce.reason.find("thread count") != std::string::npos);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

```text
=========================================================
Appendix C.2: The Threading and Parallelization Strategy Advisor
=========================================================

-- Test 1: ridge_point=40 -- crossover_batch_size=80 --
-- Test 2: serial_fraction=0.05 -- speedup@8=5.92593, ceiling=20 --
-- Test 3: speedup@1000000 threads=19.9996 (ceiling=20) --
-- Test 4: matmul output rows -- strategy=ROW_PARALLEL --
-- Test 5: GQA attention heads -- strategy=GROUP_ALIGNED_HEAD_PARALLEL --
-- Test 6: elementwise reduction (e.g. RMSNorm) -- strategy=FIXED_BLOCK_REDUCTION --

14/14 checks passed
ALL CHECKS PASSED
```

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

```cpp
// Appendix C.3 -- A KV Cache Strategy Advisor.
//
// Chapters 13 and 14 built five real, separately-verified KV cache
// techniques: a plain ring buffer with position/slot decoupling (13.3),
// a paged, block-table-indirected allocator (13.2), H2O attention-weighted
// eviction (13.4), sliding-window attention (14.1), streaming
// re-quantization of aging entries (14.2), and prefix caching across
// conversation turns (14.3/14.4). None of them is a universal replacement
// for the others -- each solves a real, different constraint. This file
// restates the real decision criteria Chapters 13-14 actually established
// for choosing among them, as one callable advisor, plus Chapter 13.1's
// own exact KV-cache-bytes-per-token formula reused verbatim.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 c3_kv_cache_strategy_advisor.cpp -o c3_kv_cache_strategy_advisor
// Run:     ./c3_kv_cache_strategy_advisor

#include <cstdint>
#include <iostream>
#include <string>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// How the context length itself is expected to behave -- known and small
// enough to reserve up front, genuinely unpredictable and growing, or so
// long that even a paged allocator cannot keep every token resident.
enum class ContextRegime { SHORT_BOUNDED, LONG_GROWING, VERY_LONG_MEMORY_CONSTRAINED };

enum class CacheStrategy {
    PLAIN_RING_BUFFER,
    PAGED_BLOCK_TABLE,
    SLIDING_WINDOW,
    PAGED_WITH_H2O_EVICTION,
    PREFIX_CACHING_ON_PAGED,
};

std::string to_string(CacheStrategy s) {
    switch (s) {
        case CacheStrategy::PLAIN_RING_BUFFER: return "PLAIN_RING_BUFFER";
        case CacheStrategy::PAGED_BLOCK_TABLE: return "PAGED_BLOCK_TABLE";
        case CacheStrategy::SLIDING_WINDOW: return "SLIDING_WINDOW";
        case CacheStrategy::PAGED_WITH_H2O_EVICTION: return "PAGED_WITH_H2O_EVICTION";
        case CacheStrategy::PREFIX_CACHING_ON_PAGED: return "PREFIX_CACHING_ON_PAGED";
    }
    return "UNKNOWN";
}

struct CacheRecommendation {
    CacheStrategy strategy;
    std::string reason;
};

// model_trained_with_fixed_window: Chapter 14.1's own real distinction --
// a model explicitly trained with a fixed attention window (its own
// example: Mistral 7B's 4096-token window) loses nothing from a
// same-sized window at inference time, because the window is exactly what
// it learned to expect, not an approximation of full attention.
//
// multi_turn_conversation: Chapter 14.3/14.4's own scope -- prefix
// caching is a cross-TURN optimization layered on top of a paged cache,
// not a replacement for how any single turn's own entries are stored or
// evicted.
CacheRecommendation recommend_kv_cache_strategy(ContextRegime regime,
                                                 bool model_trained_with_fixed_window,
                                                 bool multi_turn_conversation) {
    if (multi_turn_conversation) {
        return {CacheStrategy::PREFIX_CACHING_ON_PAGED,
                "Chapter 14.3/14.4: reuses shared history across turns via the block table, but only "
                "credits tokens the runtime cache still actually holds -- never token-ID equality alone "
                "-- so it must sit on top of a paged, block-addressed cache to begin with"};
    }
    if (model_trained_with_fixed_window) {
        return {CacheStrategy::SLIDING_WINDOW,
                "Chapter 14.1: a model explicitly trained with a fixed attention window loses nothing "
                "from a same-sized ring-buffer window -- it is exactly what the model learned to expect, "
                "not an approximation of full attention"};
    }
    switch (regime) {
        case ContextRegime::SHORT_BOUNDED:
            return {CacheStrategy::PLAIN_RING_BUFFER,
                    "Chapter 13.3: once the maximum context length is known and small enough to reserve "
                    "up front, a single fixed-capacity ring buffer with the write-slot and RoPE-position "
                    "counters kept separate is sufficient"};
        case ContextRegime::LONG_GROWING:
            return {CacheStrategy::PAGED_BLOCK_TABLE,
                    "Chapter 13.2: fixed-size power-of-two blocks plus a free list avoid both "
                    "worst-case-reservation waste and reallocate-and-copy stalls as a genuinely "
                    "unpredictable context grows"};
        case ContextRegime::VERY_LONG_MEMORY_CONSTRAINED:
            return {CacheStrategy::PAGED_WITH_H2O_EVICTION,
                    "Chapter 13.4: once paging alone cannot keep every token resident, H2O's attention-"
                    "probability importance score evicts the least-useful token instead of merely the "
                    "oldest, with a recent-token exemption so brand-new context is never evicted before "
                    "it has had a chance to prove itself important"};
    }
    return {CacheStrategy::PLAIN_RING_BUFFER, "unreachable"};
}

// Chapter 13.1's own exact formula, reused verbatim: for each layer, each
// GQA KV head (not query head), per cached token, 2 vectors (K, V) of
// head_dim elements each.
uint64_t kv_cache_bytes_per_token(uint64_t n_layers, uint64_t n_kv_heads, uint64_t head_dim,
                                   uint64_t bytes_per_element) {
    return n_layers * n_kv_heads * head_dim * 2 * bytes_per_element;
}

int main() {
    std::cout << "===================================================\n";
    std::cout << "Appendix C.3: The KV Cache Strategy Advisor\n";
    std::cout << "===================================================\n\n";

    // -- Test 1: a multi-turn conversation always gets prefix caching,
    // regardless of context regime or window training -- it is a layer
    // ON TOP of the underlying cache, not a competitor to it. --
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::SHORT_BOUNDED, true, true);
        std::cout << "-- Test 1: multi-turn conversation -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::PREFIX_CACHING_ON_PAGED);
    }

    // -- Test 2: a model trained with a fixed attention window gets
    // sliding window, as long as it is not also a multi-turn scenario. --
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::LONG_GROWING, true, false);
        std::cout << "-- Test 2: trained-with-fixed-window model -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::SLIDING_WINDOW);
    }

    // -- Tests 3-5: the three real context-regime defaults, with neither
    // of the two overriding conditions set. --
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::SHORT_BOUNDED, false, false);
        std::cout << "-- Test 3: short, bounded context -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::PLAIN_RING_BUFFER);
    }
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::LONG_GROWING, false, false);
        std::cout << "-- Test 4: long, unpredictably growing context -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::PAGED_BLOCK_TABLE);
    }
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::VERY_LONG_MEMORY_CONSTRAINED, false, false);
        std::cout << "-- Test 5: very long, memory-constrained context -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::PAGED_WITH_H2O_EVICTION);
        CHECK(r.reason.find("H2O") != std::string::npos);
    }

    // -- Test 6: Chapter 13.1's own worked example reproduced exactly --
    // a Llama-3-8B-shaped config (32 layers, 8 GQA KV heads, head_dim 128,
    // FP16) costs exactly 128 KiB (131072 bytes) per cached token. --
    {
        uint64_t bytes = kv_cache_bytes_per_token(32, 8, 128, 2);
        std::cout << "-- Test 6: Llama-3-8B-shaped config -- bytes_per_token=" << bytes << " --\n";
        CHECK(bytes == 131072ULL);
        CHECK(bytes == 128ULL * 1024ULL);  // exactly 128 KiB, not merely "about" 128 KB
    }

    // -- Test 7: Chapter 13.1's own "16 GB at 128K tokens" claim
    // reproduced exactly -- at a context length of exactly 131072 tokens
    // (2^17, the real meaning of "128K" tokens), the same per-token cost
    // from Test 6 totals exactly 16 GiB, not merely "about" 16 GB. --
    {
        uint64_t bytes_per_token = kv_cache_bytes_per_token(32, 8, 128, 2);
        uint64_t context_length = 131072ULL;  // 128K tokens = 2^17
        uint64_t total_bytes = bytes_per_token * context_length;
        double total_gib = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "-- Test 7: total KV cache at 128K-token context -- total_bytes=" << total_bytes
                  << ", total_gib=" << total_gib << " --\n";
        CHECK(total_bytes == 17179869184ULL);
        CHECK(total_gib == 16.0);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

```text
===================================================
Appendix C.3: The KV Cache Strategy Advisor
===================================================

-- Test 1: multi-turn conversation -- strategy=PREFIX_CACHING_ON_PAGED --
-- Test 2: trained-with-fixed-window model -- strategy=SLIDING_WINDOW --
-- Test 3: short, bounded context -- strategy=PLAIN_RING_BUFFER --
-- Test 4: long, unpredictably growing context -- strategy=PAGED_BLOCK_TABLE --
-- Test 5: very long, memory-constrained context -- strategy=PAGED_WITH_H2O_EVICTION --
-- Test 6: Llama-3-8B-shaped config -- bytes_per_token=131072 --
-- Test 7: total KV cache at 128K-token context -- total_bytes=17179869184, total_gib=16 --

10/10 checks passed
ALL CHECKS PASSED
```

---

## Appendix Summary

None of the three advisors in this appendix introduced a new technique, a new formula, or a new number. Each one is Chapters 4 through 14 (and Chapter 30's crossover formula) restated as a callable function, because a decision tree a reader has to hold in their head while making a real choice is worth less than one they can call and get a cited answer back from. The three real lessons underneath all three sections are the same lesson wearing three different hats: a format, a partitioning axis, or a cache technique is never simply better in the abstract -- it is only ever the right answer to a specific, real question about the data (is it calibrated once or generated per-token?), the workload (is it memory-bound or compute-bound at this batch size?), or the deployment (is the context length known, growing, or already too large to keep entirely resident?).

## Where We Go Next

Appendix D turns from choosing a strategy to measuring whether a real deployment is actually getting what a chosen strategy promised: hardware bandwidth surveys across real edge silicon, latency-budget arithmetic, `perf`-based profiling with the flags that actually work on Arm, and this book's own honest-numbers discipline applied one more time -- deterministic operation counts locked and verified, genuinely variable wall-clock timing kept explicitly out of the locked contract.
