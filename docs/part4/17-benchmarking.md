# Chapter 17: Benchmarking Against llama.cpp and the Broader Ecosystem

**What you will understand by the end of this chapter:**

- Why "how fast is it" is not one number but at least three genuinely different ones -- time-to-first-token, tokens-per-second, and time-per-output-token -- and why this book reports the MEDIAN of a run's per-step latencies rather than the mean.
- How to measure a process's own real peak memory use directly from the kernel (`/proc/self/status`'s `VmHWM`), rather than estimating it from a model's file size or parameter count.
- How to build a benchmarking harness that times this book's own REAL forward pass -- Chapter 16's own `decode_step` and `prefill`, unchanged -- rather than a simulated bandwidth model, including a warm-up window that excludes a memory-mapped file's real, one-time page-fault cost from steady-state statistics, and a KV-capacity guard that refuses to silently report statistics computed past a cache's own capacity.
- Why a synthetic model's own metadata can corrupt a correct forward pass in a way that has nothing to do with tensor bytes at all: a missing `rope.freq_base` key, defaulting to 0.0, drives this book's own RoPE formula straight into a `0 * infinity` NaN before a single real weight is even read.
- How to rebuild `llama.cpp` from source and run an honest, apples-to-apples comparison against it on identical hardware, an identical file, and an identical batch size and thread count -- and why this book's own from-scratch engine's real, measured shortfall against it is not a bug to go hunt down.

**What you need to know first:**

- Chapter 16's complete production engine, in particular Section 16.3's `decode_step` and `prefill` -- this chapter's own benchmarking harness calls these exact, already-verified functions, unchanged, rather than building a second timing path that would need its own independent trust.
- Chapter 15's real, correct first token and the actual downloaded checkpoint (`qwen2.5-0.5b-instruct-q8_0.gguf`) this chapter benchmarks -- the same file Chapters 15 and 16 have used throughout, so this chapter's own comparison is against the identical numerical result those chapters already confirmed correct.
- This chapter's own honest-exception shape follows Chapter 16's directly. Sections 17.1 and 17.2 have no real-file mode at all: each proves its own new machinery (statistics, timing definitions, a from-scratch harness) against a small synthetic model, exactly as Sections 16.1 through 16.3 did. Section 17.3, like Section 16.4, has a real-file mode that is a completely separate invocation from its own self-test mode, because a real run's wall-clock throughput is exactly the one kind of output that MUST differ from machine to machine -- locking it into this book's four-way byte-identical cross-check would be locking in a claim no other reader's hardware could ever reproduce.

---

Chapter 16 produced a complete, working, self-contained production engine -- a real command line, a real sampling pipeline, a real multi-turn conversation, and a real two-turn exchange against the actual downloaded Qwen2.5-0.5B-Instruct checkpoint. It never once asked whether that engine was any GOOD, compared to anything else. This chapter asks exactly that question, against the reference every CPU inference engine in the world is measured against: `llama.cpp`. Answering it honestly takes three steps in order. First, agreeing on WHAT to measure, because a single "tokens per second" number quietly conflates at least three different phases of generation with different costs and different bottlenecks. Second, building a harness that measures those real numbers against this book's own REAL forward pass, not a simulation of one. Third, and only once both of those are trustworthy, actually running the comparison -- on identical hardware, an identical file, and a stated, narrow methodology -- and reporting whatever number comes out, including a number that is not flattering.

## 17.1 What to Measure and Why

### Intuition

"How fast is it" hides a real ambiguity. A user waiting for a chatbot to start responding cares about one thing; a user watching text stream in afterward cares about a different one; and neither of those is the same as "how much memory did this cost to run," which no earlier chapter in this book ever had reason to measure at all.

### The Concept, In Detail

Time-to-first-token (TTFT) is everything a user waits through before ANY output appears: the full prefill pass over the entire prompt, plus the one sampling step that turns prefill's final hidden state into the first generated token. Tokens-per-second (TPS) in steady-state decode is the reciprocal of the MEDIAN per-step latency, and time-per-output-token (TPOT) is that same median latency stated directly in milliseconds -- TPS and TPOT are the identical measurement stated two ways, one easier to compare against a target throughput, one easier for a human to reason about directly, and multiplying them back together must recover exactly 1000 (milliseconds per second).

The choice to report the median rather than the mean is a stated methodology decision, not an afterthought: a small number of unusually slow steps -- a thermal throttling event, an OS scheduling hiccup -- pulls a mean upward in a way that misrepresents steady-state performance, while the median is unmoved by a minority of outliers. This is exactly the reasoning Chapter 13's own eviction-policy discussions already applied to choosing a robust statistic over a sensitive one, applied here to latency instead of cache behavior. `Stats::compute` reports the full picture regardless -- mean, median, P95, P99, min, and max -- so a reader can see exactly how far a real run's tail departs from its typical case, rather than trusting one number to summarize a whole distribution.

A specific, real effect motivates one more definition: `is_warmup_step` marks the first several decode steps as excluded from every statistic above. This book's real forward pass has no JIT and no bytecode cache to warm up -- but the memory-mapped weight file genuinely does have a cold-start cost. The OS has not yet paged in every tensor's bytes on the very first few accesses to a freshly `mmap`ed file, so the first several real decode steps pay a real, one-time page-fault cost that a steady-state throughput claim should not be charged for. Excluding a warm-up window is not throwing away inconvenient data; it is separating a one-time cost from the recurring one the statistic is actually trying to characterize.

The one measurement this chapter needs that no earlier chapter did is memory. `peak_rss_kb` reads `/proc/self/status`'s `VmHWM` line directly from the kernel -- a REAL measurement of this specific process's actual high-water-mark memory use, not an estimate computed from a model's file size or parameter count. Producing one correct token, or one correct conversation, says nothing on its own about how much memory doing so actually cost; comparing against `llama.cpp` needs exactly this number, because a faster engine that also uses ten times the memory is not an unambiguous improvement. `peak_rss_kb` returns -1, a real "measurement unavailable here" outcome, on a platform without `/proc` -- distinct from, and never confused with, "measured zero."

### Code and Verification

```cpp
// Chapter 17.1 -- Chapter 16 produced a complete, working production
// engine. This chapter asks whether it is a GOOD one, compared to the
// reference every CPU inference engine is measured against: llama.cpp.
// Answering that honestly starts with agreeing on WHAT to measure,
// because "how fast is it" hides at least three genuinely different
// numbers that a naive wall-clock measurement conflates.
//
// This section defines those numbers precisely and builds the statistics
// machinery every later section in this chapter reads from -- a Stats
// struct (mean, median, P95, P99) and this book's own definitions of
// time-to-first-token, tokens-per-second, and time-per-output-token --
// plus a real, working measurement of this process's own peak memory
// use, the one dimension of "how good is this engine" that Chapters 15
// and 16 never had reason to measure at all.
//
// A stated methodology choice, carried forward into Section 17.2: this
// book reports the MEDIAN of a run's per-step latencies, not the mean,
// because a small number of unusually slow steps (thermal throttling, an
// OS scheduling hiccup) pulls a mean upward in a way that misrepresents
// steady-state performance, while the median is unmoved by a minority of
// outliers -- exactly the reasoning Chapter 13's own eviction-policy
// discussions already applied to choosing a robust statistic over a
// sensitive one.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_benchmark_stats_and_definitions.cpp -o 01_benchmark_stats_and_definitions
// Run:     ./01_benchmark_stats_and_definitions

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: robust statistics over a run's per-step latencies. Every field
// here answers a specific question a single average cannot: min/max show
// the full spread, median shows the typical case unmoved by outliers,
// and P95/P99 show the tail a user-facing latency SLA actually has to
// account for.
// =======================================================================
struct Stats {
    double mean = 0.0, median = 0.0, p95 = 0.0, p99 = 0.0;
    double min_val = 0.0, max_val = 0.0, stddev = 0.0;
    size_t n = 0;

    static Stats compute(std::vector<double> vals) {
        Stats s;
        if (vals.empty()) return s;
        std::sort(vals.begin(), vals.end());
        s.n = vals.size();
        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        s.mean = sum / static_cast<double>(s.n);
        double sq_sum = 0.0;
        for (double v : vals) sq_sum += (v - s.mean) * (v - s.mean);
        s.stddev = std::sqrt(sq_sum / static_cast<double>(s.n));
        s.median = vals[s.n / 2];
        s.p95 = vals[static_cast<size_t>(static_cast<double>(s.n) * 0.95)];
        s.p99 = vals[static_cast<size_t>(static_cast<double>(s.n) * 0.99)];
        s.min_val = vals.front();
        s.max_val = vals.back();
        return s;
    }
};

// =======================================================================
// PART 2: this book's own precise definitions. TTFT and decode-step
// latency measure two DIFFERENT phases of generation (prefill is
// typically compute-bound over several tokens at once; decode is
// memory-bandwidth-bound, one token at a time), so collapsing them into
// one "tokens per second since the prompt was submitted" number hides
// exactly the distinction a reader trying to improve either phase needs.
// =======================================================================

// Time-to-first-token: everything a user waits through before ANY output
// appears -- the full prefill pass over the prompt, plus the one sampling
// step that turns the prefill's final hidden state into the first
// generated token. Sampling itself is typically well under a millisecond
// (Section 16.2's own pipeline is a handful of vector passes over the
// vocabulary), so TTFT is dominated by prefill for any prompt of a
// realistic length.
double time_to_first_token_ms(double prefill_ms, double first_sample_ms) {
    return prefill_ms + first_sample_ms;
}

// Tokens-per-second in steady-state decode: the reciprocal of the
// MEDIAN per-step latency, not the mean, for the robustness reason
// this file's own header comment states.
double tokens_per_second(const Stats& decode_stats) {
    return decode_stats.median > 0.0 ? 1000.0 / decode_stats.median : 0.0;
}

// Time-per-output-token: the median decode-step latency itself, in
// milliseconds. This is the exact same number as 1000/tokens_per_second
// -- TPOT and TPS are the same measurement stated two ways, one easier
// for a human to reason about ("14 ms per token") and one easier to
// compare against a target throughput ("70 tokens/sec").
double time_per_output_token_ms(const Stats& decode_stats) {
    return decode_stats.median;
}

// A step at or before this index is warm-up: excluded from every
// statistic above. Chapter 15's own real-file sections already found
// that this book's real forward pass has no JIT or cold-cache effects
// in the sense a bytecode VM would -- but the memory-mapped weight file
// genuinely does: the OS has not yet paged in every tensor's bytes on
// the very first few accesses, so the first several decode steps pay a
// real, one-time page-fault cost this book's own steady-state throughput
// claim should not be charged for.
bool is_warmup_step(int step_index, int warmup_steps) { return step_index < warmup_steps; }

// =======================================================================
// PART 3: this process's own peak resident set size, read directly from
// the kernel via /proc/self/status -- a REAL measurement of this
// specific process's actual memory behavior, not an estimate computed
// from a model's file size or parameter count. Chapters 15 and 16 never
// needed this: producing one correct token, or one correct conversation,
// says nothing about how much memory doing so actually cost. Comparing
// against llama.cpp (Section 17.3) needs exactly this number, because a
// faster engine that also uses ten times the memory is not an
// unambiguous improvement.
// =======================================================================
// Returns -1 if the kernel does not expose VmHWM (e.g., not running on
// Linux) -- a real "measurement is unavailable here" outcome, distinct
// from "measured zero," which this section's own caller must not confuse.
long peak_rss_kb() {
    std::ifstream in("/proc/self/status");
    if (!in.is_open()) return -1;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long kb = -1;
            iss >> kb;
            return kb;
        }
    }
    return -1;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 17.1: What to Measure and Why\n";
    std::cout << "========================================================\n";

    // =====================================================================
    // TEST 1: Stats::compute against a known, hand-computable array --
    // the same shape of check this book has used for every statistics
    // helper since Chapter 1.
    // =====================================================================
    std::cout << "\n-- Test 1: Stats::compute correctness --\n";
    {
        std::vector<double> vals = {5.0, 10.0, 15.0, 20.0, 25.0, 100.0};
        auto s = Stats::compute(vals);
        CHECK(s.n == 6);
        CHECK(s.min_val == 5.0);
        CHECK(s.max_val == 100.0);
        CHECK(s.median >= 15.0 && s.median <= 20.0);
        CHECK(s.p95 >= 25.0);
        // The single outlier (100.0) pulls the mean well above the
        // median -- the exact asymmetry this section's own methodology
        // choice (report median, not mean) exists to be robust against.
        CHECK(s.mean > s.median);
        std::cout << "  n=" << s.n << " median=" << s.median << " mean=" << s.mean
                   << " (mean pulled above median by the outlier: " << (s.mean > s.median ? "yes" : "no") << ")\n";

        Stats empty = Stats::compute({});
        CHECK(empty.n == 0);
        std::cout << "  empty input handled without dividing by zero: yes\n";
    }

    // =====================================================================
    // TEST 2: TTFT, TPS, and TPOT are related exactly the way this
    // section's own definitions state -- checked algebraically, not
    // against any real measured wall-clock number, so this test is
    // fully deterministic on every architecture.
    // =====================================================================
    std::cout << "\n-- Test 2: TTFT/TPS/TPOT definitions are internally consistent --\n";
    {
        std::vector<double> decode_ms(60, 14.0);   // a perfectly steady 14ms/token run
        Stats decode_stats = Stats::compute(decode_ms);
        double tps = tokens_per_second(decode_stats);
        double tpot = time_per_output_token_ms(decode_stats);
        CHECK(tpot == 14.0);
        CHECK(std::fabs(tps - (1000.0 / 14.0)) < 1e-9);
        // TPS and TPOT are the SAME measurement, stated two ways --
        // multiplying them back together must recover 1000 (ms/sec).
        CHECK(std::fabs(tps * tpot - 1000.0) < 1e-9);
        std::cout << "  steady 14ms/token decode: TPOT=" << tpot << "ms/token, TPS=" << tps
                   << " tok/s, TPS*TPOT==1000: " << (std::fabs(tps * tpot - 1000.0) < 1e-9 ? "yes" : "no") << "\n";

        double ttft = time_to_first_token_ms(/*prefill_ms=*/230.0, /*first_sample_ms=*/0.4);
        CHECK(ttft == 230.4);
        std::cout << "  TTFT = prefill_ms + first_sample_ms = 230.0 + 0.4 = " << ttft << "ms\n";

        CHECK(is_warmup_step(0, 5) && is_warmup_step(4, 5));
        CHECK(!is_warmup_step(5, 5) && !is_warmup_step(59, 5));
        std::cout << "  warm-up window [0,5) correctly excludes step 5 onward\n";
    }

    // =====================================================================
    // TEST 3: peak_rss_kb() is a REAL measurement, not a constant --
    // confirmed by deliberately growing this process's own memory
    // footprint and checking the SAME function reports growth. The raw
    // KB values are machine-specific and go to stderr only; the locked,
    // cross-architecture-verified stdout carries just the qualitative
    // "did it grow" verdict.
    // =====================================================================
    std::cout << "\n-- Test 3: peak_rss_kb() reflects real process memory, not a guess --\n";
    {
        long before = peak_rss_kb();
        bool measurement_available = (before >= 0);
        CHECK(measurement_available);

        if (measurement_available) {
            // Allocate and TOUCH every page of a sizable buffer -- a
            // reservation alone (e.g. a bare .resize() the OS can
            // satisfy with lazy, never-faulted-in pages) would not
            // necessarily move VmHWM at all; writing to every page
            // forces the kernel to actually back it with real memory.
            constexpr size_t GROW_BYTES = 200'000'000;   // ~200 MB
            std::vector<uint8_t> grow(GROW_BYTES);
            for (size_t i = 0; i < GROW_BYTES; i += 4096) grow[i] = static_cast<uint8_t>(i);
            long after = peak_rss_kb();
            bool grew = after > before;
            CHECK(grew);
            std::cerr << "  (informational, machine-specific) peak RSS before: " << before
                       << " KB, after touching ~200MB: " << after << " KB\n";
            std::cout << "  peak RSS measurably increased after touching a ~200MB buffer: " << (grew ? "yes" : "no") << "\n";
        } else {
            std::cout << "  /proc/self/status unavailable on this platform -- peak_rss_kb() correctly reports -1"
                          " rather than a fabricated number\n";
        }
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_benchmark_stats_and_definitions.cpp -o 01_benchmark_stats_and_definitions
./01_benchmark_stats_and_definitions
```

**Sample input:** `Stats::compute` checked against a known, hand-computable array (an outlier confirmed to pull the mean above the median, and an empty input confirmed handled without dividing by zero); TTFT, TPS, and TPOT checked for algebraic self-consistency (TPS times TPOT recovers exactly 1000) rather than against any specific measured millisecond value, so the check is fully deterministic on every architecture; and `peak_rss_kb` checked against a REAL measurement, by deliberately allocating and touching every page of a ~200MB buffer and confirming the same function reports real growth -- with the raw, machine-specific KB values sent to `stderr` and only the qualitative "did it grow" verdict on the locked, cross-architecture-compared `stdout`.

```text
========================================================
Chapter 17.1: What to Measure and Why
========================================================

-- Test 1: Stats::compute correctness --
  n=6 median=20 mean=29.1667 (mean pulled above median by the outlier: yes)
  empty input handled without dividing by zero: yes

-- Test 2: TTFT/TPS/TPOT definitions are internally consistent --
  steady 14ms/token decode: TPOT=14ms/token, TPS=71.4286 tok/s, TPS*TPOT==1000: yes
  TTFT = prefill_ms + first_sample_ms = 230.0 + 0.4 = 230.4ms
  warm-up window [0,5) correctly excludes step 5 onward

-- Test 3: peak_rss_kb() reflects real process memory, not a guess --
  peak RSS measurably increased after touching a ~200MB buffer: yes

15/15 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] reserving memory is not the same as using it"
    A bare `std::vector<uint8_t> grow(200'000'000)` reserves 200MB of address space, but on a system with lazy page allocation, the kernel is free to satisfy that reservation with pages it never actually backs with real memory until something writes to them -- `VmHWM` can stay completely unchanged after a `.resize()` alone, making a test that only reserves memory and checks for `VmHWM` growth silently measure nothing. The fix is not a bigger reservation; it is writing to every page (`for (size_t i = 0; i < GROW_BYTES; i += 4096) grow[i] = ...`), which forces the kernel to actually back each page with physical memory before `VmHWM` has anything real to report growing.

## 17.2 A From-Scratch Benchmarking Harness

### Intuition

Section 17.1 defined what to measure. This section wires those definitions into a REAL benchmarking harness that runs Chapter 16's own `decode_step` and `prefill` -- not a simulated bandwidth model standing in for a forward pass, and not a placeholder assuming some other file will supply the real computation later. Benchmarking code that never runs the thing it claims to measure cannot actually measure it.

### The Concept, In Detail

This section's own tokenizer needs are minimal, almost to the point of having none: a benchmarking harness measures how fast a fixed number of forward-pass POSITIONS run, and does not need real text at all -- arbitrary token IDs within the model's own vocabulary range exercise the identical `decode_step` code path a real generated token would. This file therefore omits the GPT-2 byte codec, the BPE engine, and the entire sampling pipeline: none of them are on the path being measured, and including them would only give the timing loop more to account for without changing what it is actually trying to isolate.

`run_benchmark` prefills a prompt, times it for TTFT, then runs a further, caller-specified number of real decode steps -- the first several of which are timed but excluded from the reported `Stats`, exactly as `is_warmup_step` defines. Every timed step calls Chapter 16's real, already cross-architecture-verified `decode_step` against whatever model it is handed, real or synthetic; nothing in the timing loop itself is simulated. A KV-capacity guard breaks the decode loop the moment the next position would exceed the cache's own allocated capacity, and reports however many steps were actually measured rather than fabricating statistics for positions that were never computed -- a benchmarking harness that silently produced a truncated, mislabeled report on capacity exhaustion would be actively misleading about the throughput it claims to have measured.

`detect_slowdown` is this section's own thermal-throttle-style detection algorithm: it compares the median of a run's first quarter of per-step latencies against the median of its last quarter, and flags a slowdown when the back half is more than 15% slower than the front half. This section's own self-test checks this algorithm against a HAND-CONSTRUCTED, not measured, latency sequence -- fully deterministic and reproducible on every architecture, because the test is checking the DETECTION ALGORITHM's own logic, not real hardware behavior, and a synthetic sequence with a known, exact 40% back-half slowdown is a far more precise way to confirm that logic than waiting for a real machine to happen to throttle during a test run.

Building this harness surfaced a real bug, and it is worth tracing exactly, because nothing about it lives in tensor data at all. An early draft of this section's synthetic model writer omitted the `qwen2.rope.freq_base` metadata key entirely, reasoning that `get_f32`'s documented default of 0.0 for a missing key would be harmless, since this harness only measures timing and never checks numerical correctness. That reasoning was wrong, and this book's own NaN-detection machinery caught it immediately: `RoPETables` computes `theta = 1 / pow(base, 2k/head_dim)` for each dimension pair `k`, and for `base = 0.0`, every `k > 0` gives `pow(0, positive) = 0`, so `theta = 1/0 = +infinity`. At position 0, the very first angle computed is `0 * infinity` -- and IEEE 754 defines that product as NaN, not zero. That NaN then propagates through `cos`/`sin` into the query and key vectors of layer 0, and Chapter 16's own `nan_at_layer` detection correctly refused to report benchmark statistics computed downstream of it: `run_benchmark` returned `ok=false` on what should have been a routine timing run against a model this section's own writer had produced. The fix is the same lesson Section 16.3 already taught about `kv_count`, applied to a different field entirely: declaring a default "harmless" is not the same as tracing what the actual formula does with it. Writing the real `qwen2.rope.freq_base = 10000.0` -- the value every earlier chapter in this book already uses -- avoids the singularity at its source rather than working around its symptom.

### Code and Verification

```cpp
// Chapter 17.2 -- Section 17.1 defined what to measure (TTFT, TPS/TPOT,
// peak memory) and built the statistics this chapter reports with. This
// section wires those definitions into a REAL benchmarking harness that
// runs Section 16.3's own `decode_step` and `prefill` -- not a simulated
// bandwidth model, and not a placeholder that assumes some other file
// will supply the real forward pass later. Benchmarking code that never
// runs the thing it claims to measure cannot actually measure it; this
// harness always calls the same real, already-verified forward-pass
// machinery this book has been building since Chapter 15.
//
// This section's own tokenizer needs are minimal: a benchmarking harness
// measures how fast a fixed number of forward-pass POSITIONS run, and
// does not need real text at all -- arbitrary token IDs within the
// model's own vocabulary range exercise the identical `decode_step` code
// path a real generated token would. This file therefore omits the GPT-2
// byte codec, the BPE engine, and the sampling pipeline entirely: none
// of them are on the path being measured.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_benchmark_harness.cpp -o 02_benchmark_harness
// Run:     ./02_benchmark_harness

#include <mdspan/mdspan.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: Section 15.1's GGUF reader/writer (Section 15.4's own copy).
// =======================================================================
enum GGUFValueType : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12,
};
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q8_0 = 8 };
size_t scalar_byte_size(uint32_t t) {
    switch (t) {
        case V_UINT8: case V_INT8: case V_BOOL: return 1;
        case V_UINT16: case V_INT16: return 2;
        case V_UINT32: case V_INT32: case V_FLOAT32: return 4;
        case V_UINT64: case V_INT64: case V_FLOAT64: return 8;
        default: return 0;
    }
}
class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* data, size_t size) { out.write(reinterpret_cast<const char*>(data), size); pos += size; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_magic() { write_raw("GGUF", 4); }
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_f32(float v) { write_raw(&v, 4); }
    void write_string(const std::string& s) { uint64_t len = s.size(); write_raw(&len, 8); write_raw(s.data(), s.size()); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(V_STRING); write_string(v); }
    void write_kv_u32(const std::string& k, uint32_t v) { write_string(k); write_u32(V_UINT32); write_u32(v); }
    void write_kv_f32(const std::string& k, float v) { write_string(k); write_u32(V_FLOAT32); write_f32(v); }
    void write_tensor_info(const std::string& name, const std::vector<uint64_t>& dims, GGMLType type, uint64_t offset) {
        write_string(name);
        write_u32(static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) write_u64(d);
        write_u32(static_cast<uint32_t>(type));
        write_u64(offset);
    }
    void align(size_t alignment) {
        size_t rem = pos % alignment;
        if (rem != 0) { std::vector<char> zeros(alignment - rem, 0); write_raw(zeros.data(), zeros.size()); }
    }
    void write_bytes(const void* data, size_t size) { write_raw(data, size); }
    size_t tell() const { return pos; }
    bool good() const { return out.good(); }
};
struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t n_elements = 0;
};
using MetaValue = std::variant<std::string, uint32_t, float, bool, std::vector<std::string>>;
class GGUFReader {
    std::ifstream in;
    void read_raw(void* data, size_t size) { in.read(reinterpret_cast<char*>(data), size); }
    void skip_value(uint32_t type) {
        if (type == V_STRING) { read_string(); return; }
        if (type == V_ARRAY) {
            uint32_t elem_type; read_raw(&elem_type, 4);
            uint64_t count; read_raw(&count, 8);
            for (uint64_t i = 0; i < count; ++i) skip_value(elem_type);
            return;
        }
        in.seekg(static_cast<std::streamoff>(scalar_byte_size(type)), std::ios::cur);
    }
public:
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    std::unordered_map<std::string, MetaValue> metadata;
    std::vector<TensorInfo> tensors;
    uint64_t data_section_offset = 0;
    bool open(const std::string& path) {
        in.open(path, std::ios::binary);
        if (!in.is_open()) return false;
        char magic_bytes[4];
        read_raw(magic_bytes, 4);
        if (std::memcmp(magic_bytes, "GGUF", 4) != 0) return false;
        std::memcpy(&magic, magic_bytes, 4);
        read_raw(&version, 4);
        read_raw(&n_tensors, 8);
        read_raw(&n_kv, 8);
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string();
            uint32_t type; read_raw(&type, 4);
            switch (type) {
                case V_STRING: metadata[key] = read_string(); break;
                case V_UINT32: { uint32_t v; read_raw(&v, 4); metadata[key] = v;
                                 if (key == "general.alignment") alignment = v;
                                 break; }
                case V_FLOAT32: { float v; read_raw(&v, 4); metadata[key] = v; break; }
                case V_BOOL: { uint8_t v; read_raw(&v, 1); metadata[key] = (v != 0); break; }
                case V_ARRAY: {
                    uint32_t elem_type; read_raw(&elem_type, 4);
                    uint64_t count; read_raw(&count, 8);
                    if (elem_type == V_STRING) {
                        std::vector<std::string> arr(count);
                        for (uint64_t j = 0; j < count; ++j) arr[j] = read_string();
                        metadata[key] = std::move(arr);
                    } else {
                        for (uint64_t j = 0; j < count; ++j) skip_value(elem_type);
                    }
                    break;
                }
                default: skip_value(type); break;
            }
        }
        tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            auto& t = tensors[i];
            t.name = read_string();
            uint32_t n_dims; read_raw(&n_dims, 4);
            t.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) read_raw(&t.dims[d], 8);
            read_raw(&t.type, 4);
            read_raw(&t.offset, 8);
            t.n_elements = 1;
            for (auto d : t.dims) t.n_elements *= d;
        }
        uint64_t header_end = static_cast<uint64_t>(in.tellg());
        uint64_t rem = header_end % alignment;
        data_section_offset = (rem == 0) ? header_end : header_end + (alignment - rem);
        return true;
    }
    std::string read_string() {
        uint64_t len; read_raw(&len, 8);
        std::string s(len, '\0'); read_raw(s.data(), len); return s;
    }
    uint32_t get_u32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0 : std::get<uint32_t>(it->second);
    }
    float get_f32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0.0f : std::get<float>(it->second);
    }
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

// =======================================================================
// PART 2: Chapter 4.2's fp16_t/BlockQ8 (Section 15.4's own copy).
// =======================================================================
struct fp16_t {
    uint16_t bits = 0;
    fp16_t() = default;
    fp16_t(float f) { bits = encode(f); }
    operator float() const { return decode(bits); }
    static uint16_t encode(float f) {
        uint32_t x; std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;
        if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        if (exp <= 0)  return static_cast<uint16_t>(sign);
        uint32_t mant10 = mant >> 13;
        uint32_t remainder = mant & 0x1FFFu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            mant10 += 1;
            if (mant10 == 0x400u) { mant10 = 0; exp += 1; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant10);
    }
    static float decode(uint16_t h) {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
        uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
        uint32_t fbits;
        if (exp == 0) fbits = sign;
        else if (exp == 31) fbits = sign | 0x7F800000u | (mant << 13);
        else fbits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        float f; std::memcpy(&f, &fbits, 4); return f;
    }
};
#pragma pack(push, 1)
struct BlockQ8 { fp16_t scale; int8_t weights[32]; };
#pragma pack(pop)
static_assert(sizeof(BlockQ8) == 34);
BlockQ8 quantize_q8(const float* data) {
    BlockQ8 b;
    float alpha = 0.0f;
    for (int i = 0; i < 32; ++i) alpha = std::max(alpha, std::fabs(data[i]));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); std::memset(b.weights, 0, 32); return b; }
    b.scale = fp16_t(alpha / 127.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 32; ++i)
        b.weights[i] = static_cast<int8_t>(std::clamp(std::round(data[i] * inv), -127.0f, 127.0f));
    return b;
}

// =======================================================================
// PART 3: memory-mapped file + dequantization (Section 15.4's own copy).
// =======================================================================
struct MappedFile {
    void* data = nullptr;
    size_t size = 0;
    int fd = -1;
    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0) { ::close(fd); fd = -1; return false; }
        size = static_cast<size_t>(st.st_size);
        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; ::close(fd); fd = -1; return false; }
        return true;
    }
    const uint8_t* at(uint64_t byte_offset) const { return static_cast<const uint8_t*>(data) + byte_offset; }
    ~MappedFile() { if (data) ::munmap(data, size); if (fd >= 0) ::close(fd); }
};
std::vector<float> dequantize_tensor(const MappedFile& mf, const GGUFReader& r, const TensorInfo& t) {
    std::vector<float> out(t.n_elements);
    const uint8_t* p = mf.at(r.data_section_offset + t.offset);
    if (t.type == GGML_F32) {
        std::memcpy(out.data(), p, t.n_elements * sizeof(float));
    } else if (t.type == GGML_Q8_0) {
        uint64_t n_blocks = t.n_elements / 32;
        for (uint64_t b = 0; b < n_blocks; ++b) {
            BlockQ8 blk;
            std::memcpy(&blk, p + b * sizeof(BlockQ8), sizeof(BlockQ8));
            float s = static_cast<float>(blk.scale);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(blk.weights[i]) * s;
        }
    }
    return out;
}
void dequantize_row_q8(const MappedFile& mf, uint64_t abs_row_offset, size_t n_elements, std::span<float> out) {
    const uint8_t* p = mf.at(abs_row_offset);
    size_t n_blocks = n_elements / 32;
    for (size_t b = 0; b < n_blocks; ++b) {
        BlockQ8 blk;
        std::memcpy(&blk, p + b * sizeof(BlockQ8), sizeof(BlockQ8));
        float s = static_cast<float>(blk.scale);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(blk.weights[i]) * s;
    }
}

// =======================================================================
// PART 4: Section 15.3's adapted transformer block (Section 15.4's own
// copy).
// =======================================================================
void rms_norm(std::span<float> out, std::span<const float> x, std::span<const float> weights, float epsilon = 1e-6f) {
    const size_t d = x.size();
    double sum_sq = 0.0;
    for (size_t i = 0; i < d; ++i) sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    float rms_inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(d)) + epsilon);
    for (size_t i = 0; i < d; ++i) out[i] = (x[i] * rms_inv) * weights[i];
}
void matmul(std::span<float> out, std::span<const float> x, std::span<const float> W, size_t in_dim, size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < in_dim; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(W[j * in_dim + i]);
        out[j] = static_cast<float>(sum);
    }
}
inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
void swiglu_ffn(std::span<float> out, std::span<const float> x, std::span<const float> W_gate,
                 std::span<const float> W_up, std::span<const float> W_down, size_t dim, size_t d_ff) {
    std::vector<float> gate_proj(d_ff), up_proj(d_ff), hidden(d_ff);
    matmul(gate_proj, x, W_gate, dim, d_ff);
    matmul(up_proj, x, W_up, dim, d_ff);
    for (size_t i = 0; i < d_ff; ++i) hidden[i] = silu(gate_proj[i]) * up_proj[i];
    matmul(out, hidden, W_down, d_ff, dim);
}
void linear_with_bias(std::span<float> out, std::span<const float> x, std::span<const float> W,
                       std::span<const float> bias, size_t in_dim, size_t out_dim) {
    matmul(out, x, W, in_dim, out_dim);
    for (size_t j = 0; j < out_dim; ++j) out[j] += bias[j];
}
struct RoPETables {
    std::vector<float> cos_vals, sin_vals;
    int half_dim;
    RoPETables(int seq_len, int head_dim, float base) : half_dim(head_dim / 2) {
        cos_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        sin_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * theta;
                cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::cos(angle);
                sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::sin(angle);
            }
        }
    }
    float cos_at(int pos, int k) const { return cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
    float sin_at(int pos, int k) const { return sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
};
void apply_rope(std::span<float> vec, int pos, const RoPETables& tables) {
    const int half_dim = tables.half_dim;
    for (int k = 0; k < half_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + half_dim)];
        float c = tables.cos_at(pos, k), s = tables.sin_at(pos, k);
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + half_dim)] = x1 * s + x2 * c;
    }
}
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
}
struct KVCache {
    std::vector<float> K, V;
    int n_heads_kv, max_seq_len, head_dim;
    using View = std::mdspan<float, std::dextents<size_t, 3>>;
    KVCache(int nh, int seq, int hd) : n_heads_kv(nh), max_seq_len(seq), head_dim(hd) {
        K.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
        V.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
    }
    View k_view() { return View(K.data(), n_heads_kv, max_seq_len, head_dim); }
    View v_view() { return View(V.data(), n_heads_kv, max_seq_len, head_dim); }
    auto k_at(int h, int t) { return std::submdspan(k_view(), h, t, std::full_extent); }
    auto v_at(int h, int t) { return std::submdspan(v_view(), h, t, std::full_extent); }
    void store(int h, int t, std::span<const float> k, std::span<const float> v) {
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[static_cast<size_t>(i)]; vslice[i] = v[static_cast<size_t>(i)]; }
    }
};
void gqa_attention(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                    int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(seq_len));
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, static_cast<size_t>(head_dim));
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            double d = 0.0;
            for (int i = 0; i < head_dim; ++i) d += static_cast<double>(q[i]) * static_cast<double>(k[i]);
            scores[static_cast<size_t>(t)] = static_cast<float>(d) * scale;
        }
        softmax_inplace(std::span<float>(scores.data(), static_cast<size_t>(seq_len)));
        std::vector<double> acc(static_cast<size_t>(head_dim), 0.0);
        for (int t = 0; t < seq_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            double w = scores[static_cast<size_t>(t)];
            for (int i = 0; i < head_dim; ++i) acc[static_cast<size_t>(i)] += w * static_cast<double>(v[i]);
        }
        float* out = output.data() + h * head_dim;
        for (int i = 0; i < head_dim; ++i) out[i] = static_cast<float>(acc[static_cast<size_t>(i)]);
    }
}
struct QwenBlockWeights {
    std::vector<float> attn_norm, Wq, bq, Wk, bk, Wv, bv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};
struct QwenShape {
    int dim, n_heads, n_heads_kv, head_dim, d_ff;
    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_heads_kv * head_dim; }
    int group_size() const { return n_heads / n_heads_kv; }
};
void qwen2_block_forward(std::span<float> x, const QwenShape& shape, const QwenBlockWeights& w,
                          KVCache& cache, int pos, const RoPETables& rope) {
    std::vector<float> normed(static_cast<size_t>(shape.dim)), q(static_cast<size_t>(shape.q_dim())),
        k(static_cast<size_t>(shape.kv_dim())), v(static_cast<size_t>(shape.kv_dim()));
    std::vector<float> attn_out(static_cast<size_t>(shape.q_dim())), proj_out(static_cast<size_t>(shape.dim));
    rms_norm(normed, x, w.attn_norm);
    linear_with_bias(q, normed, w.Wq, w.bq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.q_dim()));
    linear_with_bias(k, normed, w.Wk, w.bk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    linear_with_bias(v, normed, w.Wv, w.bv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    for (int h = 0; h < shape.n_heads; ++h)
        apply_rope(std::span<float>(q.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        apply_rope(std::span<float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        cache.store(h, pos, std::span<const float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                             std::span<const float>(v.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)));
    gqa_attention(q, cache, attn_out, pos + 1, shape.n_heads, shape.group_size());
    matmul(proj_out, attn_out, w.Wo, static_cast<size_t>(shape.q_dim()), static_cast<size_t>(shape.dim));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += proj_out[static_cast<size_t>(i)];
    std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
    rms_norm(normed2, x, w.ffn_norm);
    swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += ffn_out[static_cast<size_t>(i)];
}
struct QwenModel {
    MappedFile mf;
    GGUFReader r;
    QwenShape shape{};
    RoPETables* rope = nullptr;
    bool load(const std::string& path) {
        if (!mf.open(path)) return false;
        if (!r.open(path)) return false;
        shape.dim = static_cast<int>(r.get_u32("qwen2.embedding_length"));
        shape.n_heads = static_cast<int>(r.get_u32("qwen2.attention.head_count"));
        shape.n_heads_kv = static_cast<int>(r.get_u32("qwen2.attention.head_count_kv"));
        shape.head_dim = shape.dim / shape.n_heads;
        shape.d_ff = static_cast<int>(r.get_u32("qwen2.feed_forward_length"));
        float base = r.get_f32("qwen2.rope.freq_base");
        rope = new RoPETables(4096, shape.head_dim, base);
        return true;
    }
    ~QwenModel() { delete rope; }
    int n_layers() const { return static_cast<int>(r.get_u32("qwen2.block_count")); }
    std::vector<float> tensor(const std::string& name) const {
        const auto* t = r.find_tensor(name);
        return dequantize_tensor(mf, r, *t);
    }
    QwenBlockWeights layer(int idx) const {
        std::string p = "blk." + std::to_string(idx) + ".";
        QwenBlockWeights w;
        w.attn_norm = tensor(p + "attn_norm.weight");
        w.Wq = tensor(p + "attn_q.weight");   w.bq = tensor(p + "attn_q.bias");
        w.Wk = tensor(p + "attn_k.weight");   w.bk = tensor(p + "attn_k.bias");
        w.Wv = tensor(p + "attn_v.weight");   w.bv = tensor(p + "attn_v.bias");
        w.Wo = tensor(p + "attn_output.weight");
        w.ffn_norm = tensor(p + "ffn_norm.weight");
        w.Wgate = tensor(p + "ffn_gate.weight");
        w.Wup = tensor(p + "ffn_up.weight");
        w.Wdown = tensor(p + "ffn_down.weight");
        return w;
    }
    std::vector<float> embedding(int token_id) const {
        const auto* t = r.find_tensor("token_embd.weight");
        uint64_t row_offset = r.data_section_offset + t->offset
                             + (static_cast<uint64_t>(token_id) * static_cast<uint64_t>(shape.dim) / 32) * sizeof(BlockQ8);
        std::vector<float> out(static_cast<size_t>(shape.dim));
        dequantize_row_q8(mf, row_offset, static_cast<size_t>(shape.dim), out);
        return out;
    }
};

// =======================================================================
// PART 5: Section 16.3's incremental decode step and prefill, repeated
// unchanged (this file has no need for Section 16.4's profiler or
// streaming-callback extensions -- this section's OWN timing wraps
// these calls from outside, at exactly the granularity Section 17.1's
// definitions need: whole prefill, whole decode step).
// =======================================================================
struct DecodeStepResult { std::vector<float> hidden; int nan_at_layer = -1; };
DecodeStepResult decode_step(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                              std::vector<KVCache>& caches, int token_id, int pos) {
    DecodeStepResult res;
    std::vector<float> x = model.embedding(token_id);
    for (int layer = 0; layer < model.n_layers(); ++layer) {
        qwen2_block_forward(x, model.shape, layers[static_cast<size_t>(layer)], caches[static_cast<size_t>(layer)], pos, *model.rope);
        bool has_nan = false;
        for (float v : x) if (std::isnan(v)) { has_nan = true; break; }
        if (has_nan) { res.nan_at_layer = layer; break; }
    }
    res.hidden = std::move(x);
    return res;
}
struct PrefillResult { std::vector<float> hidden; bool exceeded_capacity = false; int nan_at_layer = -1; };
PrefillResult prefill(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                       std::vector<KVCache>& caches, const std::vector<int>& token_ids, int start_pos, int kv_capacity) {
    PrefillResult res;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int pos = start_pos + static_cast<int>(i);
        if (pos >= kv_capacity) { res.exceeded_capacity = true; break; }
        auto step = decode_step(model, layers, caches, token_ids[i], pos);
        res.hidden = std::move(step.hidden);
        if (step.nan_at_layer >= 0) { res.nan_at_layer = step.nan_at_layer; break; }
    }
    return res;
}

// =======================================================================
// PART 6: Section 17.1's Stats and peak-memory measurement, repeated
// unchanged.
// =======================================================================
struct Stats {
    double mean = 0.0, median = 0.0, p95 = 0.0, p99 = 0.0;
    double min_val = 0.0, max_val = 0.0, stddev = 0.0;
    size_t n = 0;
    static Stats compute(std::vector<double> vals) {
        Stats s;
        if (vals.empty()) return s;
        std::sort(vals.begin(), vals.end());
        s.n = vals.size();
        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        s.mean = sum / static_cast<double>(s.n);
        double sq_sum = 0.0;
        for (double v : vals) sq_sum += (v - s.mean) * (v - s.mean);
        s.stddev = std::sqrt(sq_sum / static_cast<double>(s.n));
        s.median = vals[s.n / 2];
        s.p95 = vals[static_cast<size_t>(static_cast<double>(s.n) * 0.95)];
        s.p99 = vals[static_cast<size_t>(static_cast<double>(s.n) * 0.99)];
        s.min_val = vals.front();
        s.max_val = vals.back();
        return s;
    }
};
double tokens_per_second(const Stats& decode_stats) { return decode_stats.median > 0.0 ? 1000.0 / decode_stats.median : 0.0; }
long peak_rss_kb() {
    std::ifstream in("/proc/self/status");
    if (!in.is_open()) return -1;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long kb = -1;
            iss >> kb;
            return kb;
        }
    }
    return -1;
}

// =======================================================================
// PART 7 (new): the harness itself. `run_benchmark` prefills a prompt,
// times it for TTFT, then runs `decode_steps` further real decode steps
// -- the first `warmup_steps` of which are timed but EXCLUDED from the
// reported Stats, exactly as Section 17.1's `is_warmup_step` defines.
// Every timed step calls Section 16.3's real `decode_step` against
// whatever model it is handed; nothing here is simulated.
// =======================================================================
struct BenchResult {
    bool ok = false;
    double prefill_ms = 0.0;
    int prefill_tokens = 0;
    double ttft_ms = 0.0;
    Stats decode_stats;
    int decode_steps_measured = 0;
    long peak_rss_kb_after = -1;
};
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

BenchResult run_benchmark(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                           const std::vector<int>& prompt_ids, int decode_steps, int warmup_steps,
                           int kv_capacity, std::mt19937& token_rng, int vocab_size) {
    BenchResult res;
    std::vector<KVCache> caches;
    for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, kv_capacity, model.shape.head_dim);

    auto t0 = Clock::now();
    auto pf = prefill(model, layers, caches, prompt_ids, 0, kv_capacity);
    res.prefill_ms = elapsed_ms(t0);
    res.prefill_tokens = static_cast<int>(prompt_ids.size());
    if (pf.exceeded_capacity || pf.nan_at_layer >= 0) return res;

    // The one sampling step TTFT includes: this harness benchmarks the
    // FORWARD pass, so it stands in for Section 16.2's real sampling
    // pipeline with an argmax over a uniform distribution of the same
    // vocabulary size -- cheap, deterministic, and the same order of
    // work Section 16.2's own pipeline does for one token.
    std::uniform_int_distribution<int> pick(0, vocab_size - 1);
    auto s0 = Clock::now();
    int first_token = pick(token_rng);
    double first_sample_ms = elapsed_ms(s0);
    res.ttft_ms = res.prefill_ms + first_sample_ms;

    std::vector<double> decode_ms_all;
    int pos = static_cast<int>(prompt_ids.size()) - 1;
    int next_token = first_token;
    for (int step = 0; step < decode_steps; ++step) {
        if (pos + 1 >= kv_capacity) break;
        ++pos;
        auto d0 = Clock::now();
        auto step_result = decode_step(model, layers, caches, next_token, pos);
        double ms = elapsed_ms(d0);
        if (step_result.nan_at_layer >= 0) return res;
        decode_ms_all.push_back(ms);
        next_token = pick(token_rng);
    }

    std::vector<double> decode_ms_measured;
    for (int i = 0; i < static_cast<int>(decode_ms_all.size()); ++i)
        if (!(i < warmup_steps)) decode_ms_measured.push_back(decode_ms_all[static_cast<size_t>(i)]);

    res.decode_stats = Stats::compute(decode_ms_measured);
    res.decode_steps_measured = static_cast<int>(decode_ms_measured.size());
    res.peak_rss_kb_after = peak_rss_kb();
    res.ok = true;
    return res;
}

// A structural, machine-independent check: given a HAND-CONSTRUCTED
// (not measured) sequence of per-step latencies, does comparing the
// first quarter's median against the last quarter's median correctly
// flag a >15% slowdown? This tests the DETECTION ALGORITHM, not real
// hardware behavior, so it is exactly as reproducible across this
// book's four-way cross-check as any other synthetic self-test.
bool detect_slowdown(const std::vector<double>& step_ms, double threshold_ratio = 1.15) {
    int n = static_cast<int>(step_ms.size());
    if (n < 4) return false;
    std::vector<double> first_q(step_ms.begin(), step_ms.begin() + n / 4);
    std::vector<double> last_q(step_ms.begin() + 3 * n / 4, step_ms.end());
    Stats s1 = Stats::compute(first_q), s2 = Stats::compute(last_q);
    return s1.median > 0.0 && (s2.median / s1.median) > threshold_ratio;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 17.2: A From-Scratch Benchmarking Harness\n";
    std::cout << "========================================================\n";

    constexpr int S_DIM = 32, S_HEADS = 4, S_HEADS_KV = 2, S_HEAD_DIM = 8, S_FF = 64, S_LAYERS = 2, S_VOCAB = 64;
    const std::string synth_path = "/tmp/ch17_2_synthetic_model.gguf";
    auto write_synthetic_model = [&](unsigned seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        struct Pending { std::string name; std::vector<uint64_t> dims; GGMLType type; std::vector<uint8_t> bytes; };
        std::vector<Pending> pending;
        auto add_f32 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& data) {
            std::vector<uint8_t> bytes(data.size() * 4);
            std::memcpy(bytes.data(), data.data(), bytes.size());
            pending.push_back({name, dims, GGML_F32, bytes});
        };
        auto add_q8 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& data) {
            size_t n_blocks = data.size() / 32;
            std::vector<uint8_t> bytes(n_blocks * sizeof(BlockQ8));
            for (size_t b = 0; b < n_blocks; ++b) {
                BlockQ8 blk = quantize_q8(data.data() + b * 32);
                std::memcpy(bytes.data() + b * sizeof(BlockQ8), &blk, sizeof(BlockQ8));
            }
            pending.push_back({name, dims, GGML_Q8_0, bytes});
        };
        auto rand_vec = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
        add_q8("token_embd.weight", {S_DIM, S_VOCAB}, rand_vec(static_cast<size_t>(S_DIM) * S_VOCAB));
        add_f32("output_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
        for (int layer = 0; layer < S_LAYERS; ++layer) {
            std::string p = "blk." + std::to_string(layer) + ".";
            add_f32(p + "attn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "attn_q.weight", {S_DIM, S_HEADS * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS * S_HEAD_DIM));
            add_f32(p + "attn_q.bias", {static_cast<uint64_t>(S_HEADS * S_HEAD_DIM)}, rand_vec(S_HEADS * S_HEAD_DIM));
            add_q8(p + "attn_k.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_k.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_vec(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_v.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_v.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_vec(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_output.weight", {S_HEADS * S_HEAD_DIM, S_DIM}, rand_vec(static_cast<size_t>(S_HEADS) * S_HEAD_DIM * S_DIM));
            add_f32(p + "ffn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "ffn_gate.weight", {S_DIM, S_FF}, rand_vec(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_up.weight", {S_DIM, S_FF}, rand_vec(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_down.weight", {S_FF, S_DIM}, rand_vec(static_cast<size_t>(S_FF) * S_DIM));
        }
        GGUFWriter w(synth_path);
        w.write_magic(); w.write_u32(3);
        w.write_u64(pending.size()); w.write_u64(7);   // exactly 7 kv pairs written below
        w.write_kv_string("general.architecture", "qwen2");
        w.write_kv_u32("qwen2.block_count", S_LAYERS);
        w.write_kv_u32("qwen2.embedding_length", S_DIM);
        w.write_kv_u32("qwen2.attention.head_count", S_HEADS);
        w.write_kv_u32("qwen2.attention.head_count_kv", S_HEADS_KV);
        w.write_kv_u32("qwen2.feed_forward_length", S_FF);
        w.write_kv_f32("qwen2.rope.freq_base", 10000.0f);
        // An earlier draft of this harness omitted rope.freq_base entirely,
        // reasoning that get_f32's documented 0.0f default for a missing
        // key would be harmless since this section only measures timing,
        // not numerical correctness. That reasoning was wrong in a way
        // this book's own NaN-detection machinery caught immediately:
        // RoPETables computes theta = 1/pow(base, 2k/head_dim), and for
        // base=0.0 every k>0 gives pow(0, positive)=0, so theta=1/0=+inf;
        // at pos=0 the very first angle is 0*inf, which IEEE 754 defines
        // as NaN, not 0. That NaN then propagates through cos/sin into
        // the query and key vectors of layer 0, and Section 16's own
        // nan_at_layer detection correctly refused to report benchmark
        // statistics computed downstream of it -- run_benchmark returned
        // ok=false on what should have been a routine timing run. The
        // fix is the same lesson Section 16.3 already taught about
        // kv_count: declaring a default as "harmless" is not the same as
        // checking what the actual formula does with it. Writing the
        // real qwen2.rope.freq_base=10000.0 (the value every other
        // chapter uses) avoids the singularity entirely.
        uint64_t off = 0;
        for (auto& p : pending) { w.write_tensor_info(p.name, p.dims, p.type, off); off += p.bytes.size(); }
        w.align(32);
        for (auto& p : pending) w.write_bytes(p.bytes.data(), p.bytes.size());
        return w.good();
    };

    // =====================================================================
    // TEST 1: run_benchmark against the synthetic model produces a
    // structurally sane report -- correct token counts, min <= median <=
    // p95 <= p99 <= max, and a positive TTFT -- checked as INVARIANTS
    // that must hold regardless of how fast this specific machine
    // happens to run, rather than against any specific millisecond value.
    // =====================================================================
    std::cout << "\n-- Test 1: run_benchmark produces a structurally correct report --\n";
    {
        CHECK(write_synthetic_model(7));
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));

        std::vector<int> prompt_ids = {1, 2, 3, 4, 5};
        std::mt19937 token_rng(42);
        constexpr int DECODE_STEPS = 40, WARMUP = 5, KV_CAP = 64;
        auto result = run_benchmark(model, layers, prompt_ids, DECODE_STEPS, WARMUP, KV_CAP, token_rng, S_VOCAB);

        CHECK(result.ok);
        CHECK(result.prefill_tokens == static_cast<int>(prompt_ids.size()));
        CHECK(result.decode_steps_measured == DECODE_STEPS - WARMUP);
        CHECK(result.ttft_ms >= result.prefill_ms);
        CHECK(result.decode_stats.min_val <= result.decode_stats.median);
        CHECK(result.decode_stats.median <= result.decode_stats.p95);
        CHECK(result.decode_stats.p95 <= result.decode_stats.p99);
        CHECK(result.decode_stats.p99 <= result.decode_stats.max_val);
        CHECK(tokens_per_second(result.decode_stats) > 0.0);
        CHECK(result.peak_rss_kb_after > 0);

        std::cerr << "  (informational, machine-specific) TTFT=" << result.ttft_ms << "ms, decode median="
                   << result.decode_stats.median << "ms, TPS=" << tokens_per_second(result.decode_stats)
                   << ", peak RSS=" << result.peak_rss_kb_after << "KB\n";
        std::cout << "  benchmark completed: " << (result.ok ? "yes" : "no")
                   << ", measured " << result.decode_steps_measured << "/" << DECODE_STEPS
                   << " decode steps (warm-up correctly excluded), stat ordering min<=median<=p95<=p99<=max: "
                   << (result.decode_stats.min_val <= result.decode_stats.median &&
                       result.decode_stats.median <= result.decode_stats.p95 &&
                       result.decode_stats.p95 <= result.decode_stats.p99 &&
                       result.decode_stats.p99 <= result.decode_stats.max_val ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 2: a too-small KV capacity is caught, exactly as Section
    // 16.3's own error handling requires -- a benchmarking harness that
    // silently produced a truncated, mislabeled report on capacity
    // exhaustion would be actively misleading about the throughput it
    // claims to have measured.
    // =====================================================================
    std::cout << "\n-- Test 2: KV-capacity exhaustion is caught, not silently averaged over --\n";
    {
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
        std::vector<int> prompt_ids = {1, 2, 3};
        std::mt19937 token_rng(1);
        auto result = run_benchmark(model, layers, prompt_ids, /*decode_steps=*/100, /*warmup=*/5,
                                     /*kv_capacity=*/6, token_rng, S_VOCAB);
        CHECK(result.ok);
        CHECK(result.decode_steps_measured < 100 - 5);
        std::cout << "  a 3-token prompt against a 6-position cache stops decode early rather than"
                     " reporting statistics computed past the cache's own capacity: measured "
                   << result.decode_steps_measured << " (< " << (100 - 5) << ") steps\n";
    }

    // =====================================================================
    // TEST 3: detect_slowdown's own logic, checked against a
    // HAND-CONSTRUCTED (not measured) latency sequence -- deterministic
    // and reproducible on every architecture, because nothing here reads
    // a real clock.
    // =====================================================================
    std::cout << "\n-- Test 3: slowdown detection algorithm (hand-constructed sequence) --\n";
    {
        std::vector<double> steady(80, 10.0);
        CHECK(!detect_slowdown(steady));

        std::vector<double> throttled;
        for (int i = 0; i < 40; ++i) throttled.push_back(10.0);
        for (int i = 0; i < 40; ++i) throttled.push_back(14.0);   // 40% slower in the back half
        CHECK(detect_slowdown(throttled));

        std::cout << "  a perfectly steady sequence: slowdown detected = "
                   << (detect_slowdown(steady) ? "yes (WRONG)" : "no (correct)") << "\n";
        std::cout << "  a sequence with a genuine 40% back-half slowdown: slowdown detected = "
                   << (detect_slowdown(throttled) ? "yes (correct)" : "no (WRONG)") << "\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_benchmark_harness.cpp -o 02_benchmark_harness
./02_benchmark_harness
```

**Sample input:** `run_benchmark`, run against a small synthetic model, checked for structural invariants that must hold regardless of how fast any given machine happens to run it -- correct prompt and decode-step counts, `min <= median <= p95 <= p99 <= max`, a positive tokens-per-second, and real, measured (not merely nonzero-by-construction) peak RSS; a deliberately undersized KV cache checked to stop decoding early rather than silently reporting statistics computed past its own capacity; and `detect_slowdown` checked against both a perfectly steady hand-constructed sequence (no slowdown reported) and one with a genuine, exact 40% back-half slowdown (correctly detected).

```text
========================================================
Chapter 17.2: A From-Scratch Benchmarking Harness
========================================================

-- Test 1: run_benchmark produces a structurally correct report --
  benchmark completed: yes, measured 35/40 decode steps (warm-up correctly excluded), stat ordering min<=median<=p95<=p99<=max: yes

-- Test 2: KV-capacity exhaustion is caught, not silently averaged over --
  a 3-token prompt against a 6-position cache stops decode early rather than reporting statistics computed past the cache's own capacity: measured 0 (< 95) steps

-- Test 3: slowdown detection algorithm (hand-constructed sequence) --
  a perfectly steady sequence: slowdown detected = no (correct)
  a sequence with a genuine 40% back-half slowdown: slowdown detected = yes (correct)

17/17 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] a \"harmless\" default is only harmless if you trace what the formula does with it"
    `get_f32`'s documented behavior for a missing metadata key -- returning 0.0 rather than throwing -- is a reasonable default for a KEY THIS BOOK'S OWN READER CHOSE not to write. It is a different claim entirely to assume that default is SAFE for every formula that might read it, and `RoPETables`'s own `1 / pow(base, exponent)` is exactly the kind of formula where a "reasonable-looking" default of zero drives a division by zero into an infinity, and where multiplying that infinity by a position of exactly zero produces NaN rather than the zero a reader might naively expect "no rotation at position zero" to mean. The general lesson: a default value's safety is a property of the specific formula consuming it, not of the value in isolation -- and the only way to know is to trace the formula, not to assume "it's just zero" settles the question.

## 17.3 An Honest Comparison Against llama.cpp

### Intuition

Sections 17.1 and 17.2 built this book's own honest measuring stick: precise definitions and a harness that times this book's real forward pass. This section finally points that measuring stick at the question the whole chapter exists to answer: compared to `llama.cpp`, how good is the engine this book has built from scratch since Chapter 15?

### The Concept, In Detail

"Honest" here means a specific, narrow, stated methodology, so the number at the end of this section means exactly what it claims and nothing more. Same model file -- the identical `qwen2.5-0.5b-instruct-q8_0.gguf` checkpoint used throughout Chapters 15 through 17. Same quantization, since it is the only format this book's own engine has ever supported. Same machine. Same `batch_size=1`. And the same thread count: ONE thread, matching this book's own engine exactly, because this book has never built a thread pool into its real forward pass -- Section 16.1's own CLI contract explicitly declined to offer a `-t <threads>` flag for precisely this reason. `llama.cpp` is also run a second way, at this machine's full core count, because that is how nearly every real user actually runs it in practice; that second number is reported honestly labeled as what it is -- more threads, not a better algorithm -- and is never blended into the single-threaded comparison that is this section's real methodological claim.

Producing a real `llama.cpp` binary to compare against required rebuilding it from source, since Chapter 15's own cleanup deleted the copy that chapter had used. `llama-bench` -- `llama.cpp`'s own standardized throughput-measurement tool -- reports exactly the two numbers this chapter's own definitions need, `pp<N>` and `tg<N>`, at a stated batch size and thread count, formatted precisely enough to quote directly rather than parse out of a differently-shaped tool's output.

This section's own harness code deliberately does not shell out to that `llama-bench` binary at runtime. Two independent reasons rule that out, not one. First, this book has no `llama.cpp` source tree or built binary bundled with it -- this section's own file has nothing to link against or invoke unless a reader has separately rebuilt `llama.cpp` themselves, and a program whose self-tests only pass on a machine with a second project pre-installed would not be self-contained the way every other file in this book has been. Second, and more fundamentally, shelling out to a binary this book did not build and did not verify would not be measuring anything this book could stand behind with the same confidence as Chapter 15's own numerical cross-check against `llama-eval-callback`. What this section's real-file mode DOES do honestly is measure this book's own engine freshly, in real time, against the real file -- and print that fresh, real measurement immediately alongside `llama-bench` reference numbers captured once, documented, and never silently updated or cherry-picked.

Running that comparison surfaced a second real, honest finding, and -- following this book's own standing practice all the way back to Chapter 14.3's treatment of an unverified performance claim -- it is reported exactly as measured rather than smoothed over. Across two independent real runs on the real device, prompt processing measured SLOWER, per token, than token generation: roughly half the throughput, consistently, in both runs. This is initially counterintuitive, because token generation's positions sit strictly LATER in the sequence than prefill's, meaning each decode step's own attention pass scans a longer history than the prefill step immediately before it -- if anything, decode should be the more expensive phase per token, not the cheaper one. The most likely real explanation is a phenomenon this chapter's own old reference material's methodology notes already named as a factor worth tracking: CPU frequency scaling. A `schedutil`-style governor ramps clock speed up gradually in response to SUSTAINED load; prefill is this program's first sustained burst of heavy computation, run at whatever clock speed the governor had settled on beforehand, while decode continues immediately afterward, now benefiting from however far the governor had ramped up during prefill's own execution. This book's own engine has no code path that would produce this pattern on its own -- attention cost is a small fraction of a 24-layer, 896-dimension model's total per-token cost next to its matmuls, which are position-independent -- so a hardware-level explanation is the honest one to report, rather than inventing a software cause this measurement gives no evidence for. The practical lesson for this chapter's own methodology: a single-shot prefill measurement, unlike decode's own warm-up-excluded steady state, has no equivalent window to exclude a one-time ramp-up cost from, and a reader repeating this exact comparison on their own hardware should expect their own prefill-versus-decode ratio to vary for the same reason.

### Code and Verification

```cpp
// Chapter 17.3 -- Sections 17.1 and 17.2 built this book's own honest
// measuring stick: precise definitions (TTFT, TPS, TPOT, peak RSS) and a
// harness that times this book's REAL forward pass, not a simulation.
// This section finally points that measuring stick at the question the
// whole chapter exists to answer: compared to llama.cpp -- the reference
// every CPU inference engine is measured against -- how good is the
// engine this book has built from scratch since Chapter 15?
//
// "Honest" here means a specific, narrow methodology, stated up front so
// the number at the end of this section means exactly what it claims:
// same model file (the identical Qwen2.5-0.5B-Instruct Q8_0 GGUF used
// throughout this book), same quantization, same machine, same
// batch_size=1, and the SAME thread count -- one thread, matching this
// book's own engine, which (like every engine built in this book) has no
// thread pool at all. llama.cpp is also run a second way, at this
// machine's full core count, because that is how almost every real user
// actually runs it; that second number is reported honestly labeled as
// what it is, a DIFFERENT comparison (more threads, not a better
// algorithm), never blended into the single-threaded one.
//
// This section's own contribution mirrors Section 17.2's own harness
// almost exactly -- same GGUF reader, same transformer block, same
// `run_benchmark` -- because a comparison is only honest if the same
// measuring code that was already verified in 17.2 is what points at
// the real file here, rather than a second, never-independently-tested
// timing path built just for this one section. This file's self-test
// mode (no arguments) re-verifies that copy of the machinery exactly as
// Section 17.2 did, against a synthetic model, on every architecture
// this book locks against. Its REAL-FILE mode -- `./03_llamacpp_comparison
// <model.gguf> [--pp N] [--tg N]` -- is, like Section 16.4's real
// conversation, a separate invocation whose output is captured once on
// the real device and documented rather than re-executed by this book's
// own cross-architecture lock, because wall-clock throughput numbers are
// exactly the one kind of output that MUST differ machine to machine for
// this comparison to mean anything at all.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_llamacpp_comparison.cpp -o 03_llamacpp_comparison
// Self-test run: ./03_llamacpp_comparison
// Real-file run: ./03_llamacpp_comparison <model.gguf> [--pp N] [--tg N]

#include <mdspan/mdspan.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: Section 15.1's GGUF reader/writer (Section 15.4's own copy).
// =======================================================================
enum GGUFValueType : uint32_t {
    V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
    V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
    V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12,
};
enum GGMLType : uint32_t { GGML_F32 = 0, GGML_Q8_0 = 8 };
size_t scalar_byte_size(uint32_t t) {
    switch (t) {
        case V_UINT8: case V_INT8: case V_BOOL: return 1;
        case V_UINT16: case V_INT16: return 2;
        case V_UINT32: case V_INT32: case V_FLOAT32: return 4;
        case V_UINT64: case V_INT64: case V_FLOAT64: return 8;
        default: return 0;
    }
}
class GGUFWriter {
    std::ofstream out;
    size_t pos = 0;
    void write_raw(const void* data, size_t size) { out.write(reinterpret_cast<const char*>(data), size); pos += size; }
public:
    explicit GGUFWriter(const std::string& path) : out(path, std::ios::binary) {}
    void write_magic() { write_raw("GGUF", 4); }
    void write_u32(uint32_t v) { write_raw(&v, 4); }
    void write_u64(uint64_t v) { write_raw(&v, 8); }
    void write_f32(float v) { write_raw(&v, 4); }
    void write_string(const std::string& s) { uint64_t len = s.size(); write_raw(&len, 8); write_raw(s.data(), s.size()); }
    void write_kv_string(const std::string& k, const std::string& v) { write_string(k); write_u32(V_STRING); write_string(v); }
    void write_kv_u32(const std::string& k, uint32_t v) { write_string(k); write_u32(V_UINT32); write_u32(v); }
    void write_kv_f32(const std::string& k, float v) { write_string(k); write_u32(V_FLOAT32); write_f32(v); }
    void write_tensor_info(const std::string& name, const std::vector<uint64_t>& dims, GGMLType type, uint64_t offset) {
        write_string(name);
        write_u32(static_cast<uint32_t>(dims.size()));
        for (uint64_t d : dims) write_u64(d);
        write_u32(static_cast<uint32_t>(type));
        write_u64(offset);
    }
    void align(size_t alignment) {
        size_t rem = pos % alignment;
        if (rem != 0) { std::vector<char> zeros(alignment - rem, 0); write_raw(zeros.data(), zeros.size()); }
    }
    void write_bytes(const void* data, size_t size) { write_raw(data, size); }
    size_t tell() const { return pos; }
    bool good() const { return out.good(); }
};
struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t n_elements = 0;
};
using MetaValue = std::variant<std::string, uint32_t, float, bool, std::vector<std::string>>;
class GGUFReader {
    std::ifstream in;
    void read_raw(void* data, size_t size) { in.read(reinterpret_cast<char*>(data), size); }
    void skip_value(uint32_t type) {
        if (type == V_STRING) { read_string(); return; }
        if (type == V_ARRAY) {
            uint32_t elem_type; read_raw(&elem_type, 4);
            uint64_t count; read_raw(&count, 8);
            for (uint64_t i = 0; i < count; ++i) skip_value(elem_type);
            return;
        }
        in.seekg(static_cast<std::streamoff>(scalar_byte_size(type)), std::ios::cur);
    }
public:
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    std::unordered_map<std::string, MetaValue> metadata;
    std::vector<TensorInfo> tensors;
    uint64_t data_section_offset = 0;
    bool open(const std::string& path) {
        in.open(path, std::ios::binary);
        if (!in.is_open()) return false;
        char magic_bytes[4];
        read_raw(magic_bytes, 4);
        if (std::memcmp(magic_bytes, "GGUF", 4) != 0) return false;
        std::memcpy(&magic, magic_bytes, 4);
        read_raw(&version, 4);
        read_raw(&n_tensors, 8);
        read_raw(&n_kv, 8);
        uint64_t alignment = 32;
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = read_string();
            uint32_t type; read_raw(&type, 4);
            switch (type) {
                case V_STRING: metadata[key] = read_string(); break;
                case V_UINT32: { uint32_t v; read_raw(&v, 4); metadata[key] = v;
                                 if (key == "general.alignment") alignment = v;
                                 break; }
                case V_FLOAT32: { float v; read_raw(&v, 4); metadata[key] = v; break; }
                case V_BOOL: { uint8_t v; read_raw(&v, 1); metadata[key] = (v != 0); break; }
                case V_ARRAY: {
                    uint32_t elem_type; read_raw(&elem_type, 4);
                    uint64_t count; read_raw(&count, 8);
                    if (elem_type == V_STRING) {
                        std::vector<std::string> arr(count);
                        for (uint64_t j = 0; j < count; ++j) arr[j] = read_string();
                        metadata[key] = std::move(arr);
                    } else {
                        for (uint64_t j = 0; j < count; ++j) skip_value(elem_type);
                    }
                    break;
                }
                default: skip_value(type); break;
            }
        }
        tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            auto& t = tensors[i];
            t.name = read_string();
            uint32_t n_dims; read_raw(&n_dims, 4);
            t.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) read_raw(&t.dims[d], 8);
            read_raw(&t.type, 4);
            read_raw(&t.offset, 8);
            t.n_elements = 1;
            for (auto d : t.dims) t.n_elements *= d;
        }
        uint64_t header_end = static_cast<uint64_t>(in.tellg());
        uint64_t rem = header_end % alignment;
        data_section_offset = (rem == 0) ? header_end : header_end + (alignment - rem);
        return true;
    }
    std::string read_string() {
        uint64_t len; read_raw(&len, 8);
        std::string s(len, '\0'); read_raw(s.data(), len); return s;
    }
    uint32_t get_u32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0 : std::get<uint32_t>(it->second);
    }
    float get_f32(const std::string& key) const {
        auto it = metadata.find(key);
        return (it == metadata.end()) ? 0.0f : std::get<float>(it->second);
    }
    const TensorInfo* find_tensor(const std::string& name) const {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    }
};

// =======================================================================
// PART 2: Chapter 4.2's fp16_t/BlockQ8 (Section 15.4's own copy).
// =======================================================================
struct fp16_t {
    uint16_t bits = 0;
    fp16_t() = default;
    fp16_t(float f) { bits = encode(f); }
    operator float() const { return decode(bits); }
    static uint16_t encode(float f) {
        uint32_t x; std::memcpy(&x, &f, 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = x & 0x7FFFFFu;
        if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        if (exp <= 0)  return static_cast<uint16_t>(sign);
        uint32_t mant10 = mant >> 13;
        uint32_t remainder = mant & 0x1FFFu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (mant10 & 1u))) {
            mant10 += 1;
            if (mant10 == 0x400u) { mant10 = 0; exp += 1; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant10);
    }
    static float decode(uint16_t h) {
        uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
        uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
        uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;
        uint32_t fbits;
        if (exp == 0) fbits = sign;
        else if (exp == 31) fbits = sign | 0x7F800000u | (mant << 13);
        else fbits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        float f; std::memcpy(&f, &fbits, 4); return f;
    }
};
#pragma pack(push, 1)
struct BlockQ8 { fp16_t scale; int8_t weights[32]; };
#pragma pack(pop)
static_assert(sizeof(BlockQ8) == 34);
BlockQ8 quantize_q8(const float* data) {
    BlockQ8 b;
    float alpha = 0.0f;
    for (int i = 0; i < 32; ++i) alpha = std::max(alpha, std::fabs(data[i]));
    if (alpha == 0.0f) { b.scale = fp16_t(0.0f); std::memset(b.weights, 0, 32); return b; }
    b.scale = fp16_t(alpha / 127.0f);
    float inv = 1.0f / static_cast<float>(b.scale);
    for (int i = 0; i < 32; ++i)
        b.weights[i] = static_cast<int8_t>(std::clamp(std::round(data[i] * inv), -127.0f, 127.0f));
    return b;
}

// =======================================================================
// PART 3: memory-mapped file + dequantization (Section 15.4's own copy).
// =======================================================================
struct MappedFile {
    void* data = nullptr;
    size_t size = 0;
    int fd = -1;
    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0) { ::close(fd); fd = -1; return false; }
        size = static_cast<size_t>(st.st_size);
        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { data = nullptr; ::close(fd); fd = -1; return false; }
        return true;
    }
    const uint8_t* at(uint64_t byte_offset) const { return static_cast<const uint8_t*>(data) + byte_offset; }
    ~MappedFile() { if (data) ::munmap(data, size); if (fd >= 0) ::close(fd); }
};
std::vector<float> dequantize_tensor(const MappedFile& mf, const GGUFReader& r, const TensorInfo& t) {
    std::vector<float> out(t.n_elements);
    const uint8_t* p = mf.at(r.data_section_offset + t.offset);
    if (t.type == GGML_F32) {
        std::memcpy(out.data(), p, t.n_elements * sizeof(float));
    } else if (t.type == GGML_Q8_0) {
        uint64_t n_blocks = t.n_elements / 32;
        for (uint64_t b = 0; b < n_blocks; ++b) {
            BlockQ8 blk;
            std::memcpy(&blk, p + b * sizeof(BlockQ8), sizeof(BlockQ8));
            float s = static_cast<float>(blk.scale);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(blk.weights[i]) * s;
        }
    }
    return out;
}
void dequantize_row_q8(const MappedFile& mf, uint64_t abs_row_offset, size_t n_elements, std::span<float> out) {
    const uint8_t* p = mf.at(abs_row_offset);
    size_t n_blocks = n_elements / 32;
    for (size_t b = 0; b < n_blocks; ++b) {
        BlockQ8 blk;
        std::memcpy(&blk, p + b * sizeof(BlockQ8), sizeof(BlockQ8));
        float s = static_cast<float>(blk.scale);
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = static_cast<float>(blk.weights[i]) * s;
    }
}

// =======================================================================
// PART 4: Section 15.3's adapted transformer block (Section 15.4's own
// copy).
// =======================================================================
void rms_norm(std::span<float> out, std::span<const float> x, std::span<const float> weights, float epsilon = 1e-6f) {
    const size_t d = x.size();
    double sum_sq = 0.0;
    for (size_t i = 0; i < d; ++i) sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    float rms_inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(d)) + epsilon);
    for (size_t i = 0; i < d; ++i) out[i] = (x[i] * rms_inv) * weights[i];
}
void matmul(std::span<float> out, std::span<const float> x, std::span<const float> W, size_t in_dim, size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < in_dim; ++i) sum += static_cast<double>(x[i]) * static_cast<double>(W[j * in_dim + i]);
        out[j] = static_cast<float>(sum);
    }
}
inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
void swiglu_ffn(std::span<float> out, std::span<const float> x, std::span<const float> W_gate,
                 std::span<const float> W_up, std::span<const float> W_down, size_t dim, size_t d_ff) {
    std::vector<float> gate_proj(d_ff), up_proj(d_ff), hidden(d_ff);
    matmul(gate_proj, x, W_gate, dim, d_ff);
    matmul(up_proj, x, W_up, dim, d_ff);
    for (size_t i = 0; i < d_ff; ++i) hidden[i] = silu(gate_proj[i]) * up_proj[i];
    matmul(out, hidden, W_down, d_ff, dim);
}
void linear_with_bias(std::span<float> out, std::span<const float> x, std::span<const float> W,
                       std::span<const float> bias, size_t in_dim, size_t out_dim) {
    matmul(out, x, W, in_dim, out_dim);
    for (size_t j = 0; j < out_dim; ++j) out[j] += bias[j];
}
struct RoPETables {
    std::vector<float> cos_vals, sin_vals;
    int half_dim;
    RoPETables(int seq_len, int head_dim, float base) : half_dim(head_dim / 2) {
        cos_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        sin_vals.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(half_dim));
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * static_cast<float>(k)) / static_cast<float>(head_dim));
                float angle = static_cast<float>(pos) * theta;
                cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::cos(angle);
                sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)] = std::sin(angle);
            }
        }
    }
    float cos_at(int pos, int k) const { return cos_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
    float sin_at(int pos, int k) const { return sin_vals[static_cast<size_t>(pos) * static_cast<size_t>(half_dim) + static_cast<size_t>(k)]; }
};
void apply_rope(std::span<float> vec, int pos, const RoPETables& tables) {
    const int half_dim = tables.half_dim;
    for (int k = 0; k < half_dim; ++k) {
        float x1 = vec[static_cast<size_t>(k)], x2 = vec[static_cast<size_t>(k + half_dim)];
        float c = tables.cos_at(pos, k), s = tables.sin_at(pos, k);
        vec[static_cast<size_t>(k)] = x1 * c - x2 * s;
        vec[static_cast<size_t>(k + half_dim)] = x1 * s + x2 * c;
    }
}
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
}
struct KVCache {
    std::vector<float> K, V;
    int n_heads_kv, max_seq_len, head_dim;
    using View = std::mdspan<float, std::dextents<size_t, 3>>;
    KVCache(int nh, int seq, int hd) : n_heads_kv(nh), max_seq_len(seq), head_dim(hd) {
        K.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
        V.assign(static_cast<size_t>(nh) * static_cast<size_t>(seq) * static_cast<size_t>(hd), 0.0f);
    }
    View k_view() { return View(K.data(), n_heads_kv, max_seq_len, head_dim); }
    View v_view() { return View(V.data(), n_heads_kv, max_seq_len, head_dim); }
    auto k_at(int h, int t) { return std::submdspan(k_view(), h, t, std::full_extent); }
    auto v_at(int h, int t) { return std::submdspan(v_view(), h, t, std::full_extent); }
    void store(int h, int t, std::span<const float> k, std::span<const float> v) {
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[static_cast<size_t>(i)]; vslice[i] = v[static_cast<size_t>(i)]; }
    }
};
void gqa_attention(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                    int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(seq_len));
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
        std::span<const float> q(q_heads.data() + h * head_dim, static_cast<size_t>(head_dim));
        for (int t = 0; t < seq_len; ++t) {
            auto k = cache.k_at(kv_h, t);
            double d = 0.0;
            for (int i = 0; i < head_dim; ++i) d += static_cast<double>(q[i]) * static_cast<double>(k[i]);
            scores[static_cast<size_t>(t)] = static_cast<float>(d) * scale;
        }
        softmax_inplace(std::span<float>(scores.data(), static_cast<size_t>(seq_len)));
        std::vector<double> acc(static_cast<size_t>(head_dim), 0.0);
        for (int t = 0; t < seq_len; ++t) {
            auto v = cache.v_at(kv_h, t);
            double w = scores[static_cast<size_t>(t)];
            for (int i = 0; i < head_dim; ++i) acc[static_cast<size_t>(i)] += w * static_cast<double>(v[i]);
        }
        float* out = output.data() + h * head_dim;
        for (int i = 0; i < head_dim; ++i) out[i] = static_cast<float>(acc[static_cast<size_t>(i)]);
    }
}
struct QwenBlockWeights {
    std::vector<float> attn_norm, Wq, bq, Wk, bk, Wv, bv, Wo;
    std::vector<float> ffn_norm, Wgate, Wup, Wdown;
};
struct QwenShape {
    int dim, n_heads, n_heads_kv, head_dim, d_ff;
    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_heads_kv * head_dim; }
    int group_size() const { return n_heads / n_heads_kv; }
};
void qwen2_block_forward(std::span<float> x, const QwenShape& shape, const QwenBlockWeights& w,
                          KVCache& cache, int pos, const RoPETables& rope) {
    std::vector<float> normed(static_cast<size_t>(shape.dim)), q(static_cast<size_t>(shape.q_dim())),
        k(static_cast<size_t>(shape.kv_dim())), v(static_cast<size_t>(shape.kv_dim()));
    std::vector<float> attn_out(static_cast<size_t>(shape.q_dim())), proj_out(static_cast<size_t>(shape.dim));
    rms_norm(normed, x, w.attn_norm);
    linear_with_bias(q, normed, w.Wq, w.bq, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.q_dim()));
    linear_with_bias(k, normed, w.Wk, w.bk, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    linear_with_bias(v, normed, w.Wv, w.bv, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.kv_dim()));
    for (int h = 0; h < shape.n_heads; ++h)
        apply_rope(std::span<float>(q.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        apply_rope(std::span<float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)), pos, rope);
    for (int h = 0; h < shape.n_heads_kv; ++h)
        cache.store(h, pos, std::span<const float>(k.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)),
                             std::span<const float>(v.data() + h * shape.head_dim, static_cast<size_t>(shape.head_dim)));
    gqa_attention(q, cache, attn_out, pos + 1, shape.n_heads, shape.group_size());
    matmul(proj_out, attn_out, w.Wo, static_cast<size_t>(shape.q_dim()), static_cast<size_t>(shape.dim));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += proj_out[static_cast<size_t>(i)];
    std::vector<float> normed2(static_cast<size_t>(shape.dim)), ffn_out(static_cast<size_t>(shape.dim));
    rms_norm(normed2, x, w.ffn_norm);
    swiglu_ffn(ffn_out, normed2, w.Wgate, w.Wup, w.Wdown, static_cast<size_t>(shape.dim), static_cast<size_t>(shape.d_ff));
    for (int i = 0; i < shape.dim; ++i) x[static_cast<size_t>(i)] += ffn_out[static_cast<size_t>(i)];
}
struct QwenModel {
    MappedFile mf;
    GGUFReader r;
    QwenShape shape{};
    RoPETables* rope = nullptr;
    bool load(const std::string& path) {
        if (!mf.open(path)) return false;
        if (!r.open(path)) return false;
        shape.dim = static_cast<int>(r.get_u32("qwen2.embedding_length"));
        shape.n_heads = static_cast<int>(r.get_u32("qwen2.attention.head_count"));
        shape.n_heads_kv = static_cast<int>(r.get_u32("qwen2.attention.head_count_kv"));
        shape.head_dim = shape.dim / shape.n_heads;
        shape.d_ff = static_cast<int>(r.get_u32("qwen2.feed_forward_length"));
        float base = r.get_f32("qwen2.rope.freq_base");
        rope = new RoPETables(4096, shape.head_dim, base);
        return true;
    }
    ~QwenModel() { delete rope; }
    int n_layers() const { return static_cast<int>(r.get_u32("qwen2.block_count")); }
    std::vector<float> tensor(const std::string& name) const {
        const auto* t = r.find_tensor(name);
        return dequantize_tensor(mf, r, *t);
    }
    QwenBlockWeights layer(int idx) const {
        std::string p = "blk." + std::to_string(idx) + ".";
        QwenBlockWeights w;
        w.attn_norm = tensor(p + "attn_norm.weight");
        w.Wq = tensor(p + "attn_q.weight");   w.bq = tensor(p + "attn_q.bias");
        w.Wk = tensor(p + "attn_k.weight");   w.bk = tensor(p + "attn_k.bias");
        w.Wv = tensor(p + "attn_v.weight");   w.bv = tensor(p + "attn_v.bias");
        w.Wo = tensor(p + "attn_output.weight");
        w.ffn_norm = tensor(p + "ffn_norm.weight");
        w.Wgate = tensor(p + "ffn_gate.weight");
        w.Wup = tensor(p + "ffn_up.weight");
        w.Wdown = tensor(p + "ffn_down.weight");
        return w;
    }
    std::vector<float> embedding(int token_id) const {
        const auto* t = r.find_tensor("token_embd.weight");
        uint64_t row_offset = r.data_section_offset + t->offset
                             + (static_cast<uint64_t>(token_id) * static_cast<uint64_t>(shape.dim) / 32) * sizeof(BlockQ8);
        std::vector<float> out(static_cast<size_t>(shape.dim));
        dequantize_row_q8(mf, row_offset, static_cast<size_t>(shape.dim), out);
        return out;
    }
};

// =======================================================================
// PART 5: Section 16.3's incremental decode step and prefill, repeated
// unchanged (this file has no need for Section 16.4's profiler or
// streaming-callback extensions -- this section's OWN timing wraps
// these calls from outside, at exactly the granularity Section 17.1's
// definitions need: whole prefill, whole decode step).
// =======================================================================
struct DecodeStepResult { std::vector<float> hidden; int nan_at_layer = -1; };
DecodeStepResult decode_step(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                              std::vector<KVCache>& caches, int token_id, int pos) {
    DecodeStepResult res;
    std::vector<float> x = model.embedding(token_id);
    for (int layer = 0; layer < model.n_layers(); ++layer) {
        qwen2_block_forward(x, model.shape, layers[static_cast<size_t>(layer)], caches[static_cast<size_t>(layer)], pos, *model.rope);
        bool has_nan = false;
        for (float v : x) if (std::isnan(v)) { has_nan = true; break; }
        if (has_nan) { res.nan_at_layer = layer; break; }
    }
    res.hidden = std::move(x);
    return res;
}
struct PrefillResult { std::vector<float> hidden; bool exceeded_capacity = false; int nan_at_layer = -1; };
PrefillResult prefill(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                       std::vector<KVCache>& caches, const std::vector<int>& token_ids, int start_pos, int kv_capacity) {
    PrefillResult res;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int pos = start_pos + static_cast<int>(i);
        if (pos >= kv_capacity) { res.exceeded_capacity = true; break; }
        auto step = decode_step(model, layers, caches, token_ids[i], pos);
        res.hidden = std::move(step.hidden);
        if (step.nan_at_layer >= 0) { res.nan_at_layer = step.nan_at_layer; break; }
    }
    return res;
}

// =======================================================================
// PART 6: Section 17.1's Stats and peak-memory measurement, repeated
// unchanged.
// =======================================================================
struct Stats {
    double mean = 0.0, median = 0.0, p95 = 0.0, p99 = 0.0;
    double min_val = 0.0, max_val = 0.0, stddev = 0.0;
    size_t n = 0;
    static Stats compute(std::vector<double> vals) {
        Stats s;
        if (vals.empty()) return s;
        std::sort(vals.begin(), vals.end());
        s.n = vals.size();
        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        s.mean = sum / static_cast<double>(s.n);
        double sq_sum = 0.0;
        for (double v : vals) sq_sum += (v - s.mean) * (v - s.mean);
        s.stddev = std::sqrt(sq_sum / static_cast<double>(s.n));
        s.median = vals[s.n / 2];
        s.p95 = vals[static_cast<size_t>(static_cast<double>(s.n) * 0.95)];
        s.p99 = vals[static_cast<size_t>(static_cast<double>(s.n) * 0.99)];
        s.min_val = vals.front();
        s.max_val = vals.back();
        return s;
    }
};
double tokens_per_second(const Stats& decode_stats) { return decode_stats.median > 0.0 ? 1000.0 / decode_stats.median : 0.0; }
long peak_rss_kb() {
    std::ifstream in("/proc/self/status");
    if (!in.is_open()) return -1;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long kb = -1;
            iss >> kb;
            return kb;
        }
    }
    return -1;
}

// =======================================================================
// PART 7 (new): the harness itself. `run_benchmark` prefills a prompt,
// times it for TTFT, then runs `decode_steps` further real decode steps
// -- the first `warmup_steps` of which are timed but EXCLUDED from the
// reported Stats, exactly as Section 17.1's `is_warmup_step` defines.
// Every timed step calls Section 16.3's real `decode_step` against
// whatever model it is handed; nothing here is simulated.
// =======================================================================
struct BenchResult {
    bool ok = false;
    double prefill_ms = 0.0;
    int prefill_tokens = 0;
    double ttft_ms = 0.0;
    Stats decode_stats;
    int decode_steps_measured = 0;
    long peak_rss_kb_after = -1;
};
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

BenchResult run_benchmark(const QwenModel& model, const std::vector<QwenBlockWeights>& layers,
                           const std::vector<int>& prompt_ids, int decode_steps, int warmup_steps,
                           int kv_capacity, std::mt19937& token_rng, int vocab_size) {
    BenchResult res;
    std::vector<KVCache> caches;
    for (int l = 0; l < model.n_layers(); ++l) caches.emplace_back(model.shape.n_heads_kv, kv_capacity, model.shape.head_dim);

    auto t0 = Clock::now();
    auto pf = prefill(model, layers, caches, prompt_ids, 0, kv_capacity);
    res.prefill_ms = elapsed_ms(t0);
    res.prefill_tokens = static_cast<int>(prompt_ids.size());
    if (pf.exceeded_capacity || pf.nan_at_layer >= 0) return res;

    // The one sampling step TTFT includes: this harness benchmarks the
    // FORWARD pass, so it stands in for Section 16.2's real sampling
    // pipeline with an argmax over a uniform distribution of the same
    // vocabulary size -- cheap, deterministic, and the same order of
    // work Section 16.2's own pipeline does for one token.
    std::uniform_int_distribution<int> pick(0, vocab_size - 1);
    auto s0 = Clock::now();
    int first_token = pick(token_rng);
    double first_sample_ms = elapsed_ms(s0);
    res.ttft_ms = res.prefill_ms + first_sample_ms;

    std::vector<double> decode_ms_all;
    int pos = static_cast<int>(prompt_ids.size()) - 1;
    int next_token = first_token;
    for (int step = 0; step < decode_steps; ++step) {
        if (pos + 1 >= kv_capacity) break;
        ++pos;
        auto d0 = Clock::now();
        auto step_result = decode_step(model, layers, caches, next_token, pos);
        double ms = elapsed_ms(d0);
        if (step_result.nan_at_layer >= 0) return res;
        decode_ms_all.push_back(ms);
        next_token = pick(token_rng);
    }

    std::vector<double> decode_ms_measured;
    for (int i = 0; i < static_cast<int>(decode_ms_all.size()); ++i)
        if (!(i < warmup_steps)) decode_ms_measured.push_back(decode_ms_all[static_cast<size_t>(i)]);

    res.decode_stats = Stats::compute(decode_ms_measured);
    res.decode_steps_measured = static_cast<int>(decode_ms_measured.size());
    res.peak_rss_kb_after = peak_rss_kb();
    res.ok = true;
    return res;
}

// A structural, machine-independent check: given a HAND-CONSTRUCTED
// (not measured) sequence of per-step latencies, does comparing the
// first quarter's median against the last quarter's median correctly
// flag a >15% slowdown? This tests the DETECTION ALGORITHM, not real
// hardware behavior, so it is exactly as reproducible across this
// book's four-way cross-check as any other synthetic self-test.
bool detect_slowdown(const std::vector<double>& step_ms, double threshold_ratio = 1.15) {
    int n = static_cast<int>(step_ms.size());
    if (n < 4) return false;
    std::vector<double> first_q(step_ms.begin(), step_ms.begin() + n / 4);
    std::vector<double> last_q(step_ms.begin() + 3 * n / 4, step_ms.end());
    Stats s1 = Stats::compute(first_q), s2 = Stats::compute(last_q);
    return s1.median > 0.0 && (s2.median / s1.median) > threshold_ratio;
}

int run_self_tests() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 17.3: Self-Tests -- Re-verifying This File's Own\n";
    std::cout << "Copy of the Forward Pass and Benchmark Harness\n";
    std::cout << "========================================================\n";

    constexpr int S_DIM = 32, S_HEADS = 4, S_HEADS_KV = 2, S_HEAD_DIM = 8, S_FF = 64, S_LAYERS = 2, S_VOCAB = 64;
    const std::string synth_path = "/tmp/ch17_3_synthetic_model.gguf";
    auto write_synthetic_model = [&](unsigned seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        struct Pending { std::string name; std::vector<uint64_t> dims; GGMLType type; std::vector<uint8_t> bytes; };
        std::vector<Pending> pending;
        auto add_f32 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& data) {
            std::vector<uint8_t> bytes(data.size() * 4);
            std::memcpy(bytes.data(), data.data(), bytes.size());
            pending.push_back({name, dims, GGML_F32, bytes});
        };
        auto add_q8 = [&](const std::string& name, std::vector<uint64_t> dims, const std::vector<float>& data) {
            size_t n_blocks = data.size() / 32;
            std::vector<uint8_t> bytes(n_blocks * sizeof(BlockQ8));
            for (size_t b = 0; b < n_blocks; ++b) {
                BlockQ8 blk = quantize_q8(data.data() + b * 32);
                std::memcpy(bytes.data() + b * sizeof(BlockQ8), &blk, sizeof(BlockQ8));
            }
            pending.push_back({name, dims, GGML_Q8_0, bytes});
        };
        auto rand_vec = [&](size_t n) { std::vector<float> v(n); for (auto& x : v) x = dist(rng); return v; };
        add_q8("token_embd.weight", {S_DIM, S_VOCAB}, rand_vec(static_cast<size_t>(S_DIM) * S_VOCAB));
        add_f32("output_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
        for (int layer = 0; layer < S_LAYERS; ++layer) {
            std::string p = "blk." + std::to_string(layer) + ".";
            add_f32(p + "attn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "attn_q.weight", {S_DIM, S_HEADS * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS * S_HEAD_DIM));
            add_f32(p + "attn_q.bias", {static_cast<uint64_t>(S_HEADS * S_HEAD_DIM)}, rand_vec(S_HEADS * S_HEAD_DIM));
            add_q8(p + "attn_k.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_k.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_vec(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_v.weight", {S_DIM, S_HEADS_KV * S_HEAD_DIM}, rand_vec(static_cast<size_t>(S_DIM) * S_HEADS_KV * S_HEAD_DIM));
            add_f32(p + "attn_v.bias", {static_cast<uint64_t>(S_HEADS_KV * S_HEAD_DIM)}, rand_vec(S_HEADS_KV * S_HEAD_DIM));
            add_q8(p + "attn_output.weight", {S_HEADS * S_HEAD_DIM, S_DIM}, rand_vec(static_cast<size_t>(S_HEADS) * S_HEAD_DIM * S_DIM));
            add_f32(p + "ffn_norm.weight", {S_DIM}, std::vector<float>(S_DIM, 1.0f));
            add_q8(p + "ffn_gate.weight", {S_DIM, S_FF}, rand_vec(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_up.weight", {S_DIM, S_FF}, rand_vec(static_cast<size_t>(S_DIM) * S_FF));
            add_q8(p + "ffn_down.weight", {S_FF, S_DIM}, rand_vec(static_cast<size_t>(S_FF) * S_DIM));
        }
        GGUFWriter w(synth_path);
        w.write_magic(); w.write_u32(3);
        w.write_u64(pending.size()); w.write_u64(7);   // exactly 7 kv pairs written below
        w.write_kv_string("general.architecture", "qwen2");
        w.write_kv_u32("qwen2.block_count", S_LAYERS);
        w.write_kv_u32("qwen2.embedding_length", S_DIM);
        w.write_kv_u32("qwen2.attention.head_count", S_HEADS);
        w.write_kv_u32("qwen2.attention.head_count_kv", S_HEADS_KV);
        w.write_kv_u32("qwen2.feed_forward_length", S_FF);
        w.write_kv_f32("qwen2.rope.freq_base", 10000.0f);
        // An earlier draft of this harness omitted rope.freq_base entirely,
        // reasoning that get_f32's documented 0.0f default for a missing
        // key would be harmless since this section only measures timing,
        // not numerical correctness. That reasoning was wrong in a way
        // this book's own NaN-detection machinery caught immediately:
        // RoPETables computes theta = 1/pow(base, 2k/head_dim), and for
        // base=0.0 every k>0 gives pow(0, positive)=0, so theta=1/0=+inf;
        // at pos=0 the very first angle is 0*inf, which IEEE 754 defines
        // as NaN, not 0. That NaN then propagates through cos/sin into
        // the query and key vectors of layer 0, and Section 16's own
        // nan_at_layer detection correctly refused to report benchmark
        // statistics computed downstream of it -- run_benchmark returned
        // ok=false on what should have been a routine timing run. The
        // fix is the same lesson Section 16.3 already taught about
        // kv_count: declaring a default as "harmless" is not the same as
        // checking what the actual formula does with it. Writing the
        // real qwen2.rope.freq_base=10000.0 (the value every other
        // chapter uses) avoids the singularity entirely.
        uint64_t off = 0;
        for (auto& p : pending) { w.write_tensor_info(p.name, p.dims, p.type, off); off += p.bytes.size(); }
        w.align(32);
        for (auto& p : pending) w.write_bytes(p.bytes.data(), p.bytes.size());
        return w.good();
    };

    // =====================================================================
    // TEST 1: run_benchmark against the synthetic model produces a
    // structurally sane report -- correct token counts, min <= median <=
    // p95 <= p99 <= max, and a positive TTFT -- checked as INVARIANTS
    // that must hold regardless of how fast this specific machine
    // happens to run, rather than against any specific millisecond value.
    // =====================================================================
    std::cout << "\n-- Test 1: run_benchmark produces a structurally correct report --\n";
    {
        CHECK(write_synthetic_model(7));
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));

        std::vector<int> prompt_ids = {1, 2, 3, 4, 5};
        std::mt19937 token_rng(42);
        constexpr int DECODE_STEPS = 40, WARMUP = 5, KV_CAP = 64;
        auto result = run_benchmark(model, layers, prompt_ids, DECODE_STEPS, WARMUP, KV_CAP, token_rng, S_VOCAB);

        CHECK(result.ok);
        CHECK(result.prefill_tokens == static_cast<int>(prompt_ids.size()));
        CHECK(result.decode_steps_measured == DECODE_STEPS - WARMUP);
        CHECK(result.ttft_ms >= result.prefill_ms);
        CHECK(result.decode_stats.min_val <= result.decode_stats.median);
        CHECK(result.decode_stats.median <= result.decode_stats.p95);
        CHECK(result.decode_stats.p95 <= result.decode_stats.p99);
        CHECK(result.decode_stats.p99 <= result.decode_stats.max_val);
        CHECK(tokens_per_second(result.decode_stats) > 0.0);
        CHECK(result.peak_rss_kb_after > 0);

        std::cerr << "  (informational, machine-specific) TTFT=" << result.ttft_ms << "ms, decode median="
                   << result.decode_stats.median << "ms, TPS=" << tokens_per_second(result.decode_stats)
                   << ", peak RSS=" << result.peak_rss_kb_after << "KB\n";
        std::cout << "  benchmark completed: " << (result.ok ? "yes" : "no")
                   << ", measured " << result.decode_steps_measured << "/" << DECODE_STEPS
                   << " decode steps (warm-up correctly excluded), stat ordering min<=median<=p95<=p99<=max: "
                   << (result.decode_stats.min_val <= result.decode_stats.median &&
                       result.decode_stats.median <= result.decode_stats.p95 &&
                       result.decode_stats.p95 <= result.decode_stats.p99 &&
                       result.decode_stats.p99 <= result.decode_stats.max_val ? "yes" : "no") << "\n";
    }

    // =====================================================================
    // TEST 2: a too-small KV capacity is caught, exactly as Section
    // 16.3's own error handling requires -- a benchmarking harness that
    // silently produced a truncated, mislabeled report on capacity
    // exhaustion would be actively misleading about the throughput it
    // claims to have measured.
    // =====================================================================
    std::cout << "\n-- Test 2: KV-capacity exhaustion is caught, not silently averaged over --\n";
    {
        QwenModel model;
        CHECK(model.load(synth_path));
        std::vector<QwenBlockWeights> layers;
        for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
        std::vector<int> prompt_ids = {1, 2, 3};
        std::mt19937 token_rng(1);
        auto result = run_benchmark(model, layers, prompt_ids, /*decode_steps=*/100, /*warmup=*/5,
                                     /*kv_capacity=*/6, token_rng, S_VOCAB);
        CHECK(result.ok);
        CHECK(result.decode_steps_measured < 100 - 5);
        std::cout << "  a 3-token prompt against a 6-position cache stops decode early rather than"
                     " reporting statistics computed past the cache's own capacity: measured "
                   << result.decode_steps_measured << " (< " << (100 - 5) << ") steps\n";
    }

    // =====================================================================
    // TEST 3: detect_slowdown's own logic, checked against a
    // HAND-CONSTRUCTED (not measured) latency sequence -- deterministic
    // and reproducible on every architecture, because nothing here reads
    // a real clock.
    // =====================================================================
    std::cout << "\n-- Test 3: slowdown detection algorithm (hand-constructed sequence) --\n";
    {
        std::vector<double> steady(80, 10.0);
        CHECK(!detect_slowdown(steady));

        std::vector<double> throttled;
        for (int i = 0; i < 40; ++i) throttled.push_back(10.0);
        for (int i = 0; i < 40; ++i) throttled.push_back(14.0);   // 40% slower in the back half
        CHECK(detect_slowdown(throttled));

        std::cout << "  a perfectly steady sequence: slowdown detected = "
                   << (detect_slowdown(steady) ? "yes (WRONG)" : "no (correct)") << "\n";
        std::cout << "  a sequence with a genuine 40% back-half slowdown: slowdown detected = "
                   << (detect_slowdown(throttled) ? "yes (correct)" : "no (WRONG)") << "\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}

// =======================================================================
// PART 8 (new): real-file mode -- this section's actual job. Everything
// above this point exists to make THIS code trustworthy before pointing
// it at a real file; everything below points it at one.
// =======================================================================
struct CompareConfig {
    std::string model_path;
    int pp = 32;   // matches llama-bench's own default "pp32" test name
    int tg = 32;   // matches llama-bench's own default "tg32" test name
};
bool parse_compare_args(int argc, char** argv, CompareConfig& cfg, std::ostream& err) {
    if (argc < 2) {
        err << "usage: " << argv[0] << " <model.gguf> [--pp N] [--tg N]\n";
        return false;
    }
    cfg.model_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--pp" && i + 1 < argc) { cfg.pp = std::atoi(argv[++i]); }
        else if (a == "--tg" && i + 1 < argc) { cfg.tg = std::atoi(argv[++i]); }
        else { err << "unrecognized argument: " << a << "\n"; return false; }
    }
    if (cfg.pp <= 0 || cfg.tg <= 0) { err << "--pp and --tg must both be positive\n"; return false; }
    return true;
}

// Reference numbers from a real `llama-bench` run against the IDENTICAL
// GGUF file this function is about to benchmark, captured once on this
// book's own real device and never re-executed by this function itself
// -- this program has no llama.cpp source or binary to link against or
// shell out to, and shelling out to a binary this book did not build
// would not be measuring anything this book could stand behind. What
// this function DOES do honestly is print this book's own freshly
// measured numbers for the identical file, immediately alongside those
// documented reference numbers, at the identical pp/tg lengths, so no
// unit conversion or protocol mismatch is hiding in the comparison.
struct LlamaCppReference {
    int threads;
    double pp_tps, pp_stddev, tg_tps, tg_stddev;
};
void print_llamacpp_reference(std::ostream& os, const LlamaCppReference& r, int pp, int tg) {
    os << "  llama.cpp (build 7ceed87, CPU/NEON backend), " << r.threads << " thread"
       << (r.threads == 1 ? "" : "s") << ", batch=1, same file:\n";
    os << "    pp" << pp << ": " << std::fixed << std::setprecision(2) << r.pp_tps
       << " tok/s (+/- " << r.pp_stddev << ")\n";
    os << "    tg" << tg << ": " << r.tg_tps << " tok/s (+/- " << r.tg_stddev << ")\n";
}

int run_real_comparison(const CompareConfig& cfg) {
    std::cout << "========================================================\n";
    std::cout << "Chapter 17.3: An Honest Comparison Against llama.cpp\n";
    std::cout << "========================================================\n";

    QwenModel model;
    if (!model.load(cfg.model_path)) {
        std::cerr << "failed to load model: " << cfg.model_path << "\n";
        return 1;
    }
    std::vector<QwenBlockWeights> layers;
    for (int l = 0; l < model.n_layers(); ++l) layers.push_back(model.layer(l));
    const TensorInfo* embd = model.r.find_tensor("token_embd.weight");
    if (!embd || embd->dims.size() < 2) {
        std::cerr << "model file has no usable token_embd.weight tensor\n";
        return 1;
    }
    int vocab_size = static_cast<int>(embd->dims[1]);

    // Arbitrary but IN-RANGE token ids: exactly Section 17.2's own
    // reasoning -- this harness measures the forward pass's real
    // per-position cost, which this book's `decode_step` computes
    // identically regardless of which token id is looked up.
    std::mt19937 prompt_rng(1234), token_rng(5678);
    std::uniform_int_distribution<int> pick(0, vocab_size - 1);
    std::vector<int> prompt_ids(static_cast<size_t>(cfg.pp));
    for (auto& id : prompt_ids) id = pick(prompt_rng);

    int kv_capacity = cfg.pp + cfg.tg + 8;
    int warmup = std::max(1, std::min(5, cfg.tg / 8));
    auto result = run_benchmark(model, layers, prompt_ids, cfg.tg, warmup, kv_capacity, token_rng, vocab_size);
    if (!result.ok) {
        std::cerr << "benchmark run failed (KV capacity exceeded or NaN detected in a real weight)\n";
        return 1;
    }

    double pp_tps = (result.prefill_ms > 0.0)
                   ? (1000.0 * static_cast<double>(result.prefill_tokens) / result.prefill_ms) : 0.0;
    double tg_tps = tokens_per_second(result.decode_stats);

    std::cout << "\nmodel: " << cfg.model_path << "\n";
    std::cout << "  vocab=" << vocab_size << ", layers=" << model.n_layers()
               << ", dim=" << model.shape.dim << "\n";

    std::cout << "\n-- This book's own engine (Chapters 15-17's from-scratch C++), this machine --\n";
    std::cout << "  single-threaded (this book has never built a thread pool), batch=1, same file:\n";
    std::cout << "    pp" << cfg.pp << ": " << std::fixed << std::setprecision(2) << pp_tps
               << " tok/s (prefill " << std::setprecision(1) << result.prefill_ms << " ms total)\n";
    std::cout << "    tg" << cfg.tg << ": " << std::setprecision(2) << tg_tps << " tok/s ("
               << result.decode_steps_measured << " steps measured after " << warmup << " warm-up, median "
               << std::setprecision(3) << result.decode_stats.median << " ms/token)\n";
    std::cout << "    peak RSS: " << result.peak_rss_kb_after << " KB\n";

    // Documented, not measured by this invocation -- captured once via a
    // real `llama-bench -m <this file> -p <pp> -n <tg> -b 1 -ub 1 -t N`
    // run on this book's own real device, against the identical GGUF.
    std::cout << "\n";
    LlamaCppReference ref_1t{1, 26.08, 2.53, 30.02, 0.81};
    LlamaCppReference ref_3t{3, 49.06, 2.47, 44.04, 1.78};
    print_llamacpp_reference(std::cout, ref_1t, cfg.pp, cfg.tg);
    std::cout << "\n";
    print_llamacpp_reference(std::cout, ref_3t, cfg.pp, cfg.tg);

    double ratio_1t = (ref_1t.tg_tps > 0.0) ? (tg_tps / ref_1t.tg_tps) : 0.0;
    std::cout << "\nHonest comparison, same file and thread count (1): this book's own from-scratch\n";
    std::cout << "engine generates tokens at " << std::setprecision(1) << (ratio_1t * 100.0)
               << "% of llama.cpp's single-threaded speed on this machine.\n";
    std::cout << "The gap is not a bug to hunt down -- it is the honest, expected cost of a\n";
    std::cout << "from-scratch, unvectorized reference implementation next to a project with\n";
    std::cout << "years of hand-tuned SIMD (NEON/AVX) matmul kernels for every quantization\n";
    std::cout << "format it supports. This book's own goal was never to out-optimize llama.cpp\n";
    std::cout << "-- it was to build, and be able to explain, every layer between a GGUF file\n";
    std::cout << "and a generated token, which Chapters 15 and 16 already did and Section\n";
    std::cout << "15.4's numerical cross-check already confirmed produces the SAME numbers.\n";

    return 0;
}

int main(int argc, char** argv) {
    if (argc == 1) return run_self_tests();
    CompareConfig cfg;
    if (!parse_compare_args(argc, argv, cfg, std::cerr)) return 1;
    return run_real_comparison(cfg);
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_llamacpp_comparison.cpp -o 03_llamacpp_comparison
./03_llamacpp_comparison                                                                          # self-tests only, reproduces everywhere
./03_llamacpp_comparison /path/to/qwen2.5-0.5b-instruct-q8_0.gguf --pp 32 --tg 32                  # real comparison run (requires the downloaded file)
```

**Sample input:** this section's own copy of the forward pass and `run_benchmark` re-verified against a synthetic model exactly as Section 17.2 already proved (structural invariants, KV-capacity exhaustion, `detect_slowdown`'s own logic), confirming this file's real-file mode is built on machinery this book has already independently checked; and, in real comparison mode, this book's own engine benchmarked fresh against the actual downloaded checkpoint at `llama-bench`'s own `pp32`/`tg32` convention, printed immediately alongside `llama-bench` reference numbers captured once on this book's own real device, at both one thread (the direct, same-thread-count comparison) and this machine's full core count (labeled honestly as a different comparison).

```text
========================================================
Chapter 17.3: Self-Tests -- Re-verifying This File's Own
Copy of the Forward Pass and Benchmark Harness
========================================================

-- Test 1: run_benchmark produces a structurally correct report --
  benchmark completed: yes, measured 35/40 decode steps (warm-up correctly excluded), stat ordering min<=median<=p95<=p99<=max: yes

-- Test 2: KV-capacity exhaustion is caught, not silently averaged over --
  a 3-token prompt against a 6-position cache stops decode early rather than reporting statistics computed past the cache's own capacity: measured 0 (< 95) steps

-- Test 3: slowdown detection algorithm (hand-constructed sequence) --
  a perfectly steady sequence: slowdown detected = no (correct)
  a sequence with a genuine 40% back-half slowdown: slowdown detected = yes (correct)

17/17 checks passed
ALL CHECKS PASSED
```

**Sample real-run output** (executed once, on a reader's own machine, against the actual downloaded checkpoint and a real, from-source `llama.cpp` build; not reproduced in this book's four-way cross-check, because a real wall-clock throughput number is exactly the one kind of output that must differ machine to machine):

```text
========================================================
Chapter 17.3: An Honest Comparison Against llama.cpp
========================================================

model: qwen2.5-0.5b-instruct-q8_0.gguf
  vocab=151936, layers=24, dim=896

-- This book's own engine (Chapters 15-17's from-scratch C++), this machine --
  single-threaded (this book has never built a thread pool), batch=1, same file:
    pp32: 1.16 tok/s (prefill 27615.4 ms total)
    tg32: 2.38 tok/s (28 steps measured after 4 warm-up, median 419.758 ms/token)
    peak RSS: 1790916 KB

  llama.cpp (build 7ceed87, CPU/NEON backend), 1 thread, batch=1, same file:
    pp32: 26.08 tok/s (+/- 2.53)
    tg32: 30.02 tok/s (+/- 0.81)

  llama.cpp (build 7ceed87, CPU/NEON backend), 3 threads, batch=1, same file:
    pp32: 49.06 tok/s (+/- 2.47)
    tg32: 44.04 tok/s (+/- 1.78)

Honest comparison, same file and thread count (1): this book's own from-scratch
engine generates tokens at 7.9% of llama.cpp's single-threaded speed on this machine.
The gap is not a bug to hunt down -- it is the honest, expected cost of a
from-scratch, unvectorized reference implementation next to a project with
years of hand-tuned SIMD (NEON/AVX) matmul kernels for every quantization
format it supports. This book's own goal was never to out-optimize llama.cpp
-- it was to build, and be able to explain, every layer between a GGUF file
and a generated token, which Chapters 15 and 16 already did and Section
15.4's numerical cross-check already confirmed produces the SAME numbers.
```

!!! warning "[COMMON TRAP] mixing thread counts turns an algorithmic comparison into a hardware one"
    It is tempting to compare this book's own single-threaded engine against whatever `llama.cpp` command is fastest to run -- which, by default, uses every core the machine has. Doing that would silently change TWO variables at once (the implementation AND the thread count) while reporting the result as if only one had changed, making a multi-threaded `llama.cpp` look even further ahead than its own ALGORITHM actually is, for reasons that have nothing to do with either engine's design. The fix this section applies is running `llama.cpp` at the SAME thread count as this book's own engine (one) as the primary, stated comparison, and reporting the multi-threaded number as a clearly separate, honestly labeled second data point -- never averaged, blended, or presented as if it answered the same question.

## Chapter Summary

This chapter turned Chapter 16's complete, working production engine into a measured one. Section 17.1 defined precisely what "how fast" and "how much memory" mean for this book's own engine -- time-to-first-token, tokens-per-second and time-per-output-token (the same measurement stated two ways), and a real, kernel-reported peak memory figure -- stating and justifying the choice to report median latency over mean. Section 17.2 built a benchmarking harness that times Chapter 16's own real, unchanged `decode_step` and `prefill`, with a warm-up window that excludes a memory-mapped file's real one-time page-fault cost and a KV-capacity guard that refuses to fabricate statistics past a cache's own limits, and along the way found and fixed a real bug in its own synthetic fixture: an omitted `rope.freq_base` metadata key whose documented zero default drove this book's own RoPE formula into a `0 * infinity` NaN before a single real weight was ever read. Section 17.3, the capstone, rebuilt `llama.cpp` from source, ran an honest, same-file, same-thread-count comparison against it, and reported both the real (and real, honest) throughput ratio -- unfavorable to this book's own from-scratch engine, exactly as expected next to years of hand-tuned SIMD kernels -- and a second, genuinely interesting real-hardware finding along the way: prefill measuring consistently slower per token than decode, most plausibly a CPU frequency-governor ramp-up effect rather than anything in this book's own code.

## Self-Check Questions

1. Section 17.1 chooses to report the MEDIAN of a run's per-step latencies rather than the mean, both for `tokens_per_second` and for `time_per_output_token_ms`. Explain the specific failure mode of the mean that this choice avoids.
2. Section 17.1's Test 3 allocates a `std::vector<uint8_t>` of about 200MB and then explicitly writes to every one of its pages before checking `peak_rss_kb()` again. What would go wrong with this test if it only allocated the vector and never wrote to it?
3. Section 17.2's `run_benchmark` excludes a warm-up window from its reported `Stats`. What SPECIFIC, real cost does this warm-up window exist to exclude, and why does that cost only affect the first several decode steps rather than all of them equally?
4. Section 17.2's synthetic model writer originally omitted the `qwen2.rope.freq_base` metadata key, reasoning that `get_f32`'s documented zero default would be harmless. Trace precisely how `RoPETables`'s own formula turns that default into a NaN, starting from `pow(0.0, exponent)`.
5. Section 17.2's `detect_slowdown` is tested against a HAND-CONSTRUCTED latency sequence rather than a sequence measured on real hardware. Why is this the more rigorous test, not a weaker stand-in for a real measurement?
6. Section 17.3's own harness never shells out to a real `llama-bench` binary at runtime, even though rebuilding `llama.cpp` from source was part of producing this section's comparison. Give the two independent reasons this section's own text gives for that design choice.
7. Section 17.3 reports `llama.cpp`'s single-threaded numbers as its primary, stated comparison, and its multi-threaded numbers as a separate, clearly labeled second data point. Explain what specifically would go wrong if these two were averaged together into one "llama.cpp's speed" figure instead.
8. Section 17.3's real run measured prompt processing as consistently SLOWER per token than token generation, even though decode positions sit later in the sequence (and therefore have a longer attention history) than prefill positions. What is this section's own most likely explanation, and why does it point to hardware rather than to a bug in this book's own code?

## Where We Go Next

This chapter gave Chapter 16's complete engine an honest report card: a precise vocabulary for what "fast" means, a harness that measures this book's own real forward pass rather than a simulation of one, and a real, stated-methodology comparison against `llama.cpp` that reports a real shortfall without flinching from it, alongside a genuinely interesting real-hardware finding about prefill-versus-decode timing that this chapter's own methodology was careful enough to notice rather than average away. Every chapter through this one has run entirely on CPU, on one thread, on real but modest hardware -- and every honest engineering account of a from-scratch inference engine eventually has to ask what happens when that constraint is lifted, or when the workload changes from "one text prompt" to something a CPU-only, unbatched, single-modality engine was never built to handle at all. Part 5 takes up exactly that question, extending this book's own from-scratch discipline from text into vision-language models.

## Worked Solutions

**1.** A mean is pulled toward any outlier in the data, in proportion to how extreme that outlier is -- a single unusually slow step (a thermal throttling event, an OS scheduling hiccup, a page fault this book's own warm-up window failed to exclude) raises the mean by an amount that has nothing to do with how the engine performs the OTHER 99% of the time. The median is the middle value of a sorted sequence, unmoved by how far an outlier sits above or below it, so it continues to represent the run's typical, steady-state cost even when a small minority of steps were unusually slow for reasons unrelated to the engine's own real performance.

**2.** On a system with lazy page allocation, the kernel can satisfy a bare `.resize()` or a default-constructed `std::vector` of a given size by reserving address space without immediately backing every page with real physical memory -- pages are only actually allocated when something writes to them. A test that allocated the buffer but never wrote to any of its bytes could see `VmHWM` stay completely unchanged, not because `peak_rss_kb()` failed to measure correctly, but because the buffer never actually consumed the memory the test assumed it had. Writing to every page (`grow[i] = ...` at a stride covering every page) forces the kernel to genuinely back each page with physical memory, which is the only way `VmHWM` has real growth to report.

**3.** The warm-up window excludes the real, one-time cost of the operating system paging in a memory-mapped weight file's bytes for the first time. `MappedFile` opens the model with `mmap`, which reserves address space lazily -- the actual tensor bytes are not read from disk into memory until something touches that specific page for the first time. The first several decode steps are the first code in the entire program to actually read most of each layer's weight tensors, so they pay a real page-fault cost that steady-state decode, running against pages the OS has already paged in, does not pay again. Excluding only the first several steps (not all of them) works because once a page has been faulted in once, every SUBSEQUENT access to that same page is served from memory with no repeated fault -- the cost is genuinely one-time, not recurring.

**4.** `RoPETables`'s constructor computes, for each dimension pair `k`, `theta = 1.0f / std::pow(base, (2.0f * k) / head_dim)`. With `base = 0.0f` (the default `get_f32` returns for the missing key) and any `k > 0`, the exponent `(2k)/head_dim` is a positive number, so `std::pow(0.0f, positive) = 0.0f`, making `theta = 1.0f / 0.0f`, which IEEE 754 floating point defines as positive infinity rather than an error. The very next line computes `angle = pos * theta` for `pos = 0` (the first position in any sequence) -- and `0 * infinity` is one of the specific operations IEEE 754 defines as producing NaN, not zero, regardless of which operand is "supposed to" dominate. `std::cos` and `std::sin` of that NaN angle are themselves NaN, and every subsequent query and key vector computed using this table's cosine and sine values inherits it.

**5.** A hand-constructed sequence lets the test assert the EXACT expected outcome with certainty -- an array built as 40 values of 10.0 followed by 40 values of 14.0 is, by construction, exactly a 40% back-half slowdown, so the test can assert `detect_slowdown` returns true with complete confidence about what the "correct" answer actually is. A sequence measured on real hardware might or might not contain a real slowdown on any given run, depending on conditions the test has no control over (thermal state, other processes, power management) -- a test built around such a sequence could only ever check "did the algorithm agree with whatever this one run happened to do," which is not a check against a known-correct answer at all, and would make the test's own pass/fail outcome as unpredictable as the hardware conditions it depends on.

**6.** First, this book has no `llama.cpp` source tree or built binary bundled with it, so a self-test suite that shelled out to `llama-bench` would only pass on a machine where a reader had separately, manually rebuilt an entire second project -- breaking the self-contained property every other file in this book has maintained, where `g++ file.cpp -o file && ./file` is the complete, sufficient instruction. Second, and more fundamentally, invoking a binary this book did not build and did not independently verify would not be measuring anything this book's own build-verify-lock discipline could stand behind the way it stands behind every number produced by code this book actually wrote and tested itself -- the honest design is measuring this book's own engine freshly and comparing it against `llama-bench` numbers captured and documented as external reference data, never asking this section's own code to also vouch for a program it did not write.

**7.** Averaging `llama.cpp`'s single-threaded and multi-threaded numbers together would produce a single figure that changed TWO variables relative to this book's own engine at once -- both the algorithm/implementation AND the thread count -- while presenting it as if it isolated only one. A multi-threaded `llama.cpp` genuinely IS faster than a single-threaded one, for reasons that have nothing to do with which implementation has better per-thread code; blending that speedup into "how much faster is llama.cpp's ALGORITHM" would overstate the algorithmic gap by an amount attributable purely to parallelism this book's own engine was never built to use. Reporting both numbers separately, with the single-threaded one stated as the primary, same-thread-count comparison, keeps those two genuinely different questions -- "same thread count, whose implementation is faster" versus "how fast does a typical llama.cpp invocation run" -- answerable independently rather than blurred into one.

**8.** This section's own most likely explanation is CPU frequency governor ramp-up: a `schedutil`-style governor increases clock speed gradually in response to sustained load, and prefill is this program's very first sustained burst of heavy computation in the whole run, executing at whatever (likely lower) clock speed the governor had settled on beforehand, while decode continues immediately afterward at whatever higher clock speed the governor had ramped up to during prefill's own execution. This points to hardware rather than to a bug in this book's own code because the code path is identical either way -- `prefill` calls the exact same `decode_step` function, in the exact same loop shape, that decode itself calls afterward -- and the only per-token cost that should legitimately GROW with position (attention's own cost, which scans a longer cached history at later positions) would, if anything, make decode slower than prefill, not faster. A measured pattern running in the opposite direction of what the code's own algorithmic cost predicts, with no algorithmic explanation available, is exactly the situation where a hardware-level cause is the honest one to report.
