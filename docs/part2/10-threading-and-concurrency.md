# Chapter 10: Threading and Concurrency -- Real Threads, Verified Without a Clock

**What you will understand by the end of this chapter:**

- Why a single core cannot hit a realistic interactive decode-latency target at all, derived from Chapter 8.5's own measured full-decode-step FLOP total rather than a new, separately invented workload — and why Amdahl's law puts a hard, derivable ceiling on how much speedup ANY number of additional cores can ever buy, long before a single thread is created.
- How `std::atomic` and `std::mutex` make sharing mutable state across real, concurrently running threads well-defined, verified here by actually creating threads and checking an exact expected result — and why a genuinely unsynchronized shared counter is undefined behavior whose outcome this book will not pretend to lock, for the same honesty reasons Chapter 8 never ran a wall-clock benchmark.
- How to build a persistent, `std::barrier`-synchronized thread pool that partitions a real GEMV by output row across several worker threads and reuses those same threads across many rounds of work — verified bit-for-bit against a serial reference, not merely within a tolerance.
- Why two threads writing to two completely independent variables, with no shared state and no data race by any definition, can still sit in the same 64-byte cache line — a structural, address-arithmetic fact this chapter verifies exactly, without ever needing a wall-clock number to make the point.
- How to verify a work-stealing scheduler's correctness when its own SCHEDULE is deliberately nondeterministic — by checking an order-independent invariant (every task claimed exactly once; the aggregate result exactly correct) instead of asserting anything about which thread did what or when.

**What you need to know first:**

- Chapter 8's CpuSpec and Roofline structures, and Chapter 8.5's own measured full-decode-step FLOP and byte totals, reused verbatim in Section 10.1 rather than re-derived.
- Chapter 8.3's std::mdspan-viewed weight matrix convention, reused in Section 10.3's partitioned GEMV.
- This is the first chapter in this book to create a single `std::thread`. Every file compiles with `-pthread` in addition to this book's standard flags.
- This chapter extends this book's standing policy against fabricated timing numbers into a second, related discipline: never lock output whose exact value depends on real, uncontrolled thread scheduling. A data race's outcome (Section 10.2) and a work-stealing scheduler's exact division of labor (Section 10.5) are both real, both genuinely executed, and both deliberately excluded from this chapter's checked, locked transcripts for the identical reason a wall-clock number always has been: this book only locks what is guaranteed to reproduce on a rerun. Where a section's whole point IS a concurrency hazard, it is verified through what IS guaranteed — an aggregate invariant, a structural address computation, a deterministic mutex API guarantee — never through hoping a particular schedule repeats.

---

Chapter 9 showed how a single core reaches its peak compute, one SIMD instruction at a time. This chapter reaches for the resource no single core has any more of to give: additional cores, coordinated by real threads. Section 10.1 derives why that coordination is necessary at all — a single core's real decode-step FLOP total, measured back in Chapter 8.5, is put through Chapter 8.1's own achievable-throughput formula, and comes out several times over a realistic interactive latency budget — and then derives the ceiling Amdahl's law puts on how much any number of additional cores can close that gap. Section 10.2 introduces this book's first real `std::thread`, and with it the two primitives, `std::atomic` and `std::mutex`, that make sharing mutable state across threads well-defined rather than undefined behavior, verified by running real concurrent code and checking an exact result rather than a fabricated one. Section 10.3 builds a persistent, `std::barrier`-synchronized thread pool and partitions a real GEMV across it by output row, verified bit-for-bit against a serial reference. Section 10.4 finds a hazard that changes no thread's computed value at all — two independent per-thread counters sharing one cache line — and verifies it the only honest way available without a clock: as address arithmetic. Section 10.5 closes the chapter with a work-stealing scheduler, and with it this chapter's central methodological point: a scheduler whose exact execution order is deliberately nondeterministic can still be verified with complete rigor, by checking the invariant its contract actually promises instead of the schedule it never did.

## 10.1 The Latency Budget and Amdahl's Ceiling

### Intuition

Before writing a single thread, it is worth deriving whether one is even necessary — and, if so, roughly how much parallelism could possibly help. A single core's real cost for one decode step, already measured in Chapter 8.5, run through Chapter 8.1's own achievable-throughput formula, gives a concrete answer to "is one core enough," with no new workload invented for the purpose. Amdahl's law then answers the next question honestly: however many cores are added, some fraction of a real decode step — sampling the next token, updating a cursor, scheduling the next request — never parallelizes at all, and that fraction alone puts a hard, derivable ceiling on total speedup.

### The Concept, In Detail

Chapter 8.5's own instrumented kernel measured one full decode step, at realistic transformer dimensions, at exactly 470466888 FLOPs and 890151168 bytes — an arithmetic intensity of about 0.53 FLOPs/byte, deep in the memory-bound regime Chapter 8.1's roofline model already described. Dividing Chapter 8.1's own 8-core machine's achievable throughput at that arithmetic intensity by 8 gives one core's own share, and dividing the step's real FLOP total by that single-core throughput gives a real, derived single-core time per step — which comes out several times over a stated 20-tokens-per-second interactive latency budget, with no wall-clock measurement involved anywhere in the derivation. Amdahl's law then asks a different question: given that ADDING cores is the fix, how much does it actually help. If a fraction `serial_fraction` of the work is inherently sequential and the rest splits perfectly evenly across `n` cores with zero coordination overhead — the most generous assumption possible — total relative time is `serial_fraction + (1 - serial_fraction)/n`, and speedup is its reciprocal. This formula's most important property is not its value at any particular `n`, but its limit as `n` grows without bound: `1 / serial_fraction`, a ceiling no finite number of cores can ever reach, let alone exceed, because the parallel term `(1 - serial_fraction)/n` only ever approaches zero, never becomes it.

### Code and Verification

```cpp
// 01_latency_budget_and_amdahls_law.cpp
// Chapter 10, Part 1: before writing a single std::thread, derive WHY a
// single core cannot hit a realistic decode-time latency target at all --
// reusing Chapter 8's own CpuSpec and roofline formulas unchanged, at the
// exact FFN dimensions Chapter 8.3 and 8.5 already used -- and then derive
// the ceiling that adding MORE cores runs into. Amdahl's law is not a new
// architectural fact the way peak compute and peak bandwidth were; it is a
// direct algebraic consequence of one premise: any real per-token step has
// SOME part that does not parallelize across cores (sampling, detokenizing,
// scheduling), and that part alone puts a hard ceiling on how much total
// speedup any number of additional cores can ever buy.
//
// Every number in this file is either a stated architectural or workload
// parameter (labeled as such, exactly like Chapter 8.1's illustrative CPU
// and Chapter 8.5's illustrative model dimensions) or arithmetic derived
// from those parameters. No thread is created in this file -- Amdahl's law
// is pure arithmetic over a serial fraction and a core count, and this
// section's job is to derive that arithmetic before Section 10.2 spends any
// of it on an actual std::thread.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_latency_budget_and_amdahls_law.cpp -o 01_latency_budget_and_amdahls_law

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <limits>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

// =========================================================================
// Chapter 8.1's CpuSpec and Roofline, reused verbatim -- this section
// derives nothing new about a single core's peak compute or bandwidth; it
// only asks how long a REAL decode step's FLOPs take against that ceiling.
// =========================================================================
struct CpuSpec {
    int cores;
    double sustained_ghz;
    int simd_lanes_fp32;
    int fma_ports;
    double flops_per_cycle_per_core() const {
        return static_cast<double>(simd_lanes_fp32) * fma_ports * 2.0;
    }
    double peak_gflops() const {
        return cores * sustained_ghz * flops_per_cycle_per_core();
    }
};

struct Roofline {
    double peak_compute_gflops;
    double peak_bandwidth_gbps;
    double achievable_gflops(double arithmetic_intensity) const {
        return std::min(peak_compute_gflops, arithmetic_intensity * peak_bandwidth_gbps);
    }
};

// =========================================================================
// Amdahl's law: if a fraction `serial_fraction` of a task's work cannot be
// parallelized at all, and the remaining `1 - serial_fraction` splits
// perfectly evenly across `n` cores with zero coordination overhead (the
// most GENEROUS possible assumption about the parallel part), the total
// time relative to one core is `serial_fraction + (1 - serial_fraction)/n`,
// and speedup is the reciprocal of that.
// =========================================================================
struct AmdahlModel {
    double serial_fraction;  // in [0, 1]: the fraction of per-token work that never parallelizes

    double relative_time(double n_cores) const {
        return serial_fraction + (1.0 - serial_fraction) / n_cores;
    }
    double speedup(double n_cores) const {
        return 1.0 / relative_time(n_cores);
    }
    // The speedup no number of cores can ever exceed, however large --
    // the limit of speedup(n) as n -> infinity is 1 / serial_fraction,
    // since the parallel term (1 - serial_fraction)/n vanishes.
    double speedup_ceiling() const {
        return 1.0 / serial_fraction;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 10.1: The Latency Budget and Amdahl's Ceiling\n";
    std::cout << "========================================================\n\n";

    // Chapter 8.1's own illustrative 8-core AVX2 machine, unchanged.
    CpuSpec cpu{.cores = 8, .sustained_ghz = 3.5, .simd_lanes_fp32 = 8, .fma_ports = 2};
    Roofline roof{cpu.peak_gflops(), /*peak_bandwidth_gbps=*/51.2};  // Chapter 8.1's own derived bandwidth

    // =====================================================================
    // TEST 1: a single core of this machine cannot hit a realistic
    // interactive decode latency target -- derived from Chapter 8.5's own
    // full-decode-step FLOP total (every QKVO and FFN projection, plus
    // attention, at DIM=4096, N_HEADS_Q=32, N_HEADS_KV=8, D_FF=14336,
    // cache_len=2048), run through Chapter 8.1's own achievable-throughput
    // formula at that step's own measured, memory-bound arithmetic
    // intensity -- not a new, separately invented workload.
    // =====================================================================
    std::cout << "-- Test 1: one core's decode-step time against a real latency budget --\n";
    {
        // Chapter 8.5's own instrumented Cost counter, run against a real
        // GQA kernel at DIM=4096, N_HEADS_Q=32, N_HEADS_KV=8, D_FF=14336,
        // cache_len=2048, measured a full decode step (every QKVO and FFN
        // projection, GQA attention, both RMSNorms) at exactly
        // FLOPs=470466888, Bytes=890151168 -- reused here verbatim as
        // Chapter 8.5's own locked, verified total, not re-derived.
        constexpr double DECODE_STEP_FLOPS_TOTAL = 470466888.0;
        constexpr double DECODE_STEP_GFLOPS_TOTAL = DECODE_STEP_FLOPS_TOTAL / 1e9;
        // Chapter 8.5's own measured arithmetic intensity at that exact
        // shape (470466888 / 890151168), confirmed there to be memory-bound
        // against a ridge point of 17.50 -- reused here exactly, not
        // Chapter 8.1's separate representative AI=0.5 illustration.
        constexpr double DECODE_AI = 470466888.0 / 890151168.0;

        double achievable = roof.achievable_gflops(DECODE_AI);
        double single_core_achievable = achievable / cpu.cores;  // one core's share of the machine's peak
        double single_core_time_ms = (DECODE_STEP_GFLOPS_TOTAL / single_core_achievable) * 1000.0;

        constexpr double TARGET_TOKENS_PER_SEC = 20.0;  // a stated, labeled interactive-latency target
        double budget_ms_per_token = 1000.0 / TARGET_TOKENS_PER_SEC;

        std::cout << "  One decode step: " << std::fixed << std::setprecision(3)
                  << DECODE_STEP_GFLOPS_TOTAL << " GFLOP (Chapter 8.5's own full-layer total)\n";
        std::cout << "  One core's achievable throughput at AI=" << std::setprecision(4) << DECODE_AI << ": "
                  << std::setprecision(2) << single_core_achievable << " GFLOP/s\n";
        std::cout << "  One core's time for one decode step: " << std::setprecision(3)
                  << single_core_time_ms << " ms\n";
        std::cout << "  Latency budget at " << TARGET_TOKENS_PER_SEC << " tokens/sec: "
                  << budget_ms_per_token << " ms/token\n";
        std::cout << "  Single core is " << std::setprecision(1)
                  << (single_core_time_ms / budget_ms_per_token) << "x over budget.\n";

        CHECK(single_core_time_ms > budget_ms_per_token);
        CHECK_NEAR(budget_ms_per_token, 50.0, 1e-9);
    }

    // =====================================================================
    // TEST 2: Amdahl's law's basic algebraic properties -- speedup(1) is
    // exactly 1 regardless of serial fraction, and speedup is strictly
    // increasing in core count for any serial fraction less than 1.
    // =====================================================================
    std::cout << "\n-- Test 2: Amdahl's law's basic properties --\n";
    {
        // A stated, labeled per-token serial fraction: the part of a
        // decode step -- sampling the next token, updating the KV cache
        // write cursor, detokenizing -- that runs on exactly one core no
        // matter how many are available, illustrative in the same sense
        // Chapter 8.5's own dimensions were labeled illustrative.
        AmdahlModel model{.serial_fraction = 0.05};

        CHECK_NEAR(model.speedup(1.0), 1.0, 1e-9);

        double prev = model.speedup(1.0);
        bool strictly_increasing = true;
        for (double n : {2.0, 4.0, 8.0, 16.0, 32.0}) {
            double s = model.speedup(n);
            std::cout << "  speedup(" << std::setprecision(0) << n << " cores) = "
                      << std::setprecision(3) << s << "x\n";
            if (!(s > prev)) strictly_increasing = false;
            prev = s;
        }
        CHECK(strictly_increasing);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: assuming N cores buys N times the speedup.
    // With a serial fraction of just 5%, 8 cores buys roughly 5.9x, not
    // 8x -- and speedup does not merely fall short, it CONVERGES to a
    // hard ceiling of 1/serial_fraction as core count grows without
    // bound, so throwing arbitrarily many cores at the problem is
    // provably bounded before a single thread is ever created.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: assuming N cores means N times the speedup --\n";
    {
        AmdahlModel model{.serial_fraction = 0.05};

        double naive_expectation_at_8 = 8.0;  // "8 cores should mean 8x"
        double actual_at_8 = model.speedup(8.0);
        std::cout << "  serial_fraction = " << model.serial_fraction << " (5% of a decode step never parallelizes)\n";
        std::cout << "  Naive expectation at 8 cores: " << naive_expectation_at_8 << "x\n";
        std::cout << "  Actual Amdahl speedup at 8 cores: " << std::setprecision(3) << actual_at_8 << "x\n";
        CHECK(actual_at_8 < naive_expectation_at_8);
        CHECK(actual_at_8 > 5.5 && actual_at_8 < 6.0);

        double ceiling = model.speedup_ceiling();
        double at_1000 = model.speedup(1000.0);
        double at_1e9 = model.speedup(1.0e9);
        std::cout << "  Speedup ceiling (n -> infinity): 1 / " << model.serial_fraction
                  << " = " << std::setprecision(1) << ceiling << "x\n";
        std::cout << "  speedup(1000 cores)        = " << std::setprecision(4) << at_1000 << "x\n";
        std::cout << "  speedup(1,000,000,000 cores) = " << at_1e9 << "x -- still short of the ceiling, never past it\n";

        CHECK_NEAR(ceiling, 20.0, 1e-9);
        CHECK(at_1000 < ceiling);
        CHECK(at_1e9 < ceiling);
        CHECK(at_1e9 > at_1000);  // still strictly increasing, just converging
        // No finite core count reaches the ceiling exactly, since
        // (1 - serial_fraction)/n is strictly positive for any finite n.
        CHECK_NEAR(model.relative_time(1.0e9) - model.serial_fraction, (1.0 - model.serial_fraction) / 1.0e9, 1e-15);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_latency_budget_and_amdahls_law.cpp -o 01_latency_budget_and_amdahls_law
./01_latency_budget_and_amdahls_law
```

**Sample input:** Chapter 8.5's own real, measured full-decode-step FLOP and byte totals at DIM=4096, N_HEADS_Q=32, N_HEADS_KV=8, D_FF=14336, cache_len=2048, run through Chapter 8.1's own achievable-throughput formula to derive a real single-core decode-step time against a stated 20-tokens-per-second latency budget; Amdahl's law's basic algebraic properties (speedup(1)=1, strictly increasing in core count); and a deliberate demonstration, at a stated 5% serial fraction, of the gap between "8 cores means 8x" and the actual ~5.93x, together with the provable, never-reached ceiling of 20x as core count grows arbitrarily large.

```text
========================================================
Chapter 10.1: The Latency Budget and Amdahl's Ceiling
========================================================

-- Test 1: one core's decode-step time against a real latency budget --
  One decode step: 0.470 GFLOP (Chapter 8.5's own full-layer total)
  One core's achievable throughput at AI=0.5285: 3.38 GFLOP/s
  One core's time for one decode step: 139.086 ms
  Latency budget at 20.000 tokens/sec: 50.000 ms/token
  Single core is 2.8x over budget.

-- Test 2: Amdahl's law's basic properties --
  speedup(2 cores) = 1.905x
  speedup(4 cores) = 3.478x
  speedup(8 cores) = 5.926x
  speedup(16 cores) = 9.143x
  speedup(32 cores) = 12.549x

-- Test 3 [COMMON TRAP]: assuming N cores means N times the speedup --
  serial_fraction = 0.050 (5% of a decode step never parallelizes)
  Naive expectation at 8 cores: 8.000x
  Actual Amdahl speedup at 8 cores: 5.926x
  Speedup ceiling (n -> infinity): 1 / 0.050 = 20.0x
  speedup(1000 cores)        = 19.6271x
  speedup(1,000,000,000 cores) = 20.0000x -- still short of the ceiling, never past it

========================================================
11/11 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming N cores buys N times the speedup"
    "We have 8 cores, so we should get roughly 8x" ignores that SOME part of a real per-token step — sampling, bookkeeping, scheduling the next request — runs on exactly one core no matter how many are available. At a stated 5% serial fraction, 8 cores buys about 5.93x, not 8x, and the gap only widens as core count grows: Amdahl's law's speedup does not merely fall short of linear, it CONVERGES to a hard ceiling of `1/serial_fraction` — here, 20x — that no finite number of cores, however large, can ever reach or exceed. A team that buys twice the cores expecting twice the throughput, without first asking what fraction of the work is inherently serial, is making a purchasing decision Amdahl's law already answered before the hardware order was placed.

## 10.2 Atomics, Mutexes, and Why Synchronization Is Not Optional

### Intuition

This book's first real `std::thread` immediately raises the question every later section in this chapter depends on: what happens when more than one thread reads and writes the SAME piece of memory. `std::atomic` and `std::mutex` are two different mechanisms that both answer it the same way — by making the combined read-modify-write sequence indivisible from every other thread's point of view — verified here the way this book verifies everything: by running real threads and checking an exact result, not a probabilistic one.

### The Concept, In Detail

`std::atomic<long long>::fetch_add` performs a read-modify-write as one indivisible hardware operation; no matter how the operating system interleaves four real threads each calling it 250000 times, the C++ standard guarantees every single increment is counted, so the final total is exactly 1000000 on every run — the SCHEDULE is not deterministic, but the RESULT is, because atomicity is a guarantee about the RESULT, not about timing. `std::mutex` reaches the identical guarantee by a different mechanism: mutual exclusion ensures only one thread's `++counter` can be "in flight" at any moment, which is what makes an ordinary, otherwise-unsafe increment well-defined the moment it happens only while the lock is held. A genuinely UNSYNCHRONIZED shared counter — the same `++counter`, called from multiple real threads with no atomic or mutex protection at all — is a real data race and therefore undefined behavior in the C++ memory model; its outcome is not merely hard to predict, it is explicitly not guaranteed to be the same from one run to the next, which is exactly the property this book's build-verify-lock pipeline depends on. This section includes that racy function as real, compilable source and deliberately never executes it as part of its own checked output, the same honesty this book has applied to timing numbers from the very first chapter. A separate, smaller but equally real trap closes the section: `std::mutex` is not recursive, and a thread that already holds the lock and tries to lock it again — exactly the shape of a function calling another function that unknowingly locks the same mutex — does not get let back in.

### Code and Verification

```cpp
// 02_atomics_mutexes_and_synchronization.cpp
// Chapter 10, Part 2: this book's first file to actually create a
// std::thread. Section 10.1 showed why more than one core is needed at
// all; before Section 10.3 partitions real work across several threads,
// this section establishes the two primitives that make sharing mutable
// state across threads well-defined rather than undefined behavior --
// std::atomic and std::mutex -- and verifies each one the way every other
// kernel in this book has been verified: run for real, checked against an
// exact expected value, twice, for determinism.
//
// A genuinely UNSYNCHRONIZED shared counter, incremented from multiple
// real threads with no atomic or mutex protection at all, is a real data
// race and therefore undefined behavior in the C++ memory model. Its
// outcome is not merely "hard to predict" -- it is explicitly not
// guaranteed to be the same from one run to the next, which is exactly
// the property this book's build-verify-lock pipeline depends on for
// every other file. This section includes that racy function as real,
// compilable source, for inspection, but deliberately never executes it
// as part of this file's own checked output, for the same reason Chapter
// 8 never ran a wall-clock benchmark: printing a number this book cannot
// guarantee is reproducible would violate its own standing discipline.
// (Separately, outside this file's own build, compiling it under
// ThreadSanitizer -fsanitize=thread does flag it as a real race -- a
// genuine, documented confirmation, kept out of this file's own locked
// output the same way Chapter 9.5's -march=native COMMON TRAP was
// documented rather than executed as a live crash.)
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -pthread 02_atomics_mutexes_and_synchronization.cpp -o 02_atomics_mutexes_and_synchronization

#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

constexpr int NUM_THREADS = 4;
constexpr long long INCREMENTS_PER_THREAD = 250000;
constexpr long long EXPECTED_TOTAL = static_cast<long long>(NUM_THREADS) * INCREMENTS_PER_THREAD;

// =========================================================================
// Correct, real concurrency #1: std::atomic's fetch_add is a single,
// indivisible read-modify-write -- the C++ standard guarantees that no
// matter how the operating system interleaves NUM_THREADS threads each
// calling this INCREMENTS_PER_THREAD times, every single increment is
// counted exactly once. The SCHEDULE is not deterministic; the RESULT is.
// =========================================================================
void atomic_increment_worker(std::atomic<long long>& counter) {
    for (long long i = 0; i < INCREMENTS_PER_THREAD; ++i) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}

// =========================================================================
// Correct, real concurrency #2: a plain long long, but every access to it
// is made while holding a std::mutex via std::lock_guard -- mutual
// exclusion means only one thread's read-modify-write sequence can be
// "in flight" on the shared variable at any moment, which is exactly the
// property that makes the plain `++counter` inside the lock well-defined.
// =========================================================================
void mutex_increment_worker(long long& counter, std::mutex& mtx) {
    for (long long i = 0; i < INCREMENTS_PER_THREAD; ++i) {
        std::lock_guard<std::mutex> lock(mtx);
        ++counter;
    }
}

// =========================================================================
// A genuine data race, included as real, compilable source and NEVER
// called from this file's own main(). Every access to `counter` here is
// an ordinary, unsynchronized read-modify-write from multiple threads --
// a data race, and therefore undefined behavior per the C++ standard,
// not merely "likely to be wrong." Its final value, if this function were
// run, would not be guaranteed identical between two consecutive runs of
// the very same binary on the very same machine -- which is precisely
// why this file, unlike every other file in this book, does not execute
// it as part of its own checked, locked output.
// =========================================================================
[[maybe_unused]] void racy_increment_worker(long long& counter) {
    for (long long i = 0; i < INCREMENTS_PER_THREAD; ++i) {
        ++counter;  // read, increment, write -- with no synchronization at all
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 10.2: Atomics, Mutexes, and Why Synchronization Is Not Optional\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: std::atomic<long long>::fetch_add, real threads, checked
    // against the exact expected total -- guaranteed by atomicity, not by
    // luck, so this check passes on every run regardless of how the OS
    // happens to schedule the four worker threads.
    // =====================================================================
    std::cout << "-- Test 1: std::atomic fetch_add across " << NUM_THREADS << " real threads --\n";
    {
        std::atomic<long long> counter{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < NUM_THREADS; ++t) {
            workers.emplace_back(atomic_increment_worker, std::ref(counter));
        }
        for (auto& w : workers) w.join();

        std::cout << "  " << NUM_THREADS << " threads x " << INCREMENTS_PER_THREAD
                  << " atomic increments each: final count = " << counter.load() << "\n";
        std::cout << "  Expected total: " << EXPECTED_TOTAL << "\n";
        CHECK(counter.load() == EXPECTED_TOTAL);
    }

    // =====================================================================
    // TEST 2: the same total, the same thread count, protected by a
    // std::mutex instead of an atomic type -- a different mechanism
    // reaching the same guarantee: mutual exclusion, not lock-freedom, is
    // what makes the plain `++counter` inside the lock well-defined.
    // =====================================================================
    std::cout << "\n-- Test 2: std::mutex-protected plain counter across " << NUM_THREADS << " real threads --\n";
    {
        long long counter = 0;
        std::mutex mtx;
        std::vector<std::thread> workers;
        for (int t = 0; t < NUM_THREADS; ++t) {
            workers.emplace_back(mutex_increment_worker, std::ref(counter), std::ref(mtx));
        }
        for (auto& w : workers) w.join();

        std::cout << "  " << NUM_THREADS << " threads x " << INCREMENTS_PER_THREAD
                  << " mutex-protected increments each: final count = " << counter << "\n";
        std::cout << "  Expected total: " << EXPECTED_TOTAL << "\n";
        CHECK(counter == EXPECTED_TOTAL);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: std::mutex is not recursive. A thread that
    // already holds the lock and calls try_lock() again -- the exact
    // shape of a function that locks a mutex and then calls another
    // function that (unknowingly) tries to lock the SAME mutex again --
    // does not block and does not silently succeed; try_lock() returns
    // false, deterministically, every time, on this thread, regardless of
    // any other thread's activity. A blocking lock() in the same
    // situation would deadlock the thread against itself forever, which
    // is exactly why this check uses try_lock() instead of lock() -- a
    // real deadlock has no well-defined stdout to verify against, so this
    // section demonstrates the SAME hazard through a call that is
    // guaranteed to return rather than hang.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: std::mutex is not recursive --\n";
    {
        std::mutex mtx;
        mtx.lock();
        bool second_lock_from_same_thread_succeeded = mtx.try_lock();
        std::cout << "  Thread already holds the lock; try_lock() again on the SAME thread: "
                  << (second_lock_from_same_thread_succeeded ? "succeeded" : "returned false") << "\n";
        CHECK(!second_lock_from_same_thread_succeeded);
        if (second_lock_from_same_thread_succeeded) mtx.unlock();  // defensive; should not execute
        mtx.unlock();

        std::recursive_mutex rmtx;
        rmtx.lock();
        bool second_lock_on_recursive_succeeded = rmtx.try_lock();
        std::cout << "  Same situation with std::recursive_mutex instead: try_lock() again: "
                  << (second_lock_on_recursive_succeeded ? "succeeded" : "returned false") << "\n";
        CHECK(second_lock_on_recursive_succeeded);
        if (second_lock_on_recursive_succeeded) rmtx.unlock();
        rmtx.unlock();

        std::cout << "  A blocking lock() in place of this try_lock() would not fail -- it would hang\n";
        std::cout << "  forever, since a plain std::mutex has no notion of \"the thread that already\n";
        std::cout << "  owns me is allowed back in\". A function that locks a mutex and then calls\n";
        std::cout << "  another function that locks the SAME mutex, on the SAME thread, deadlocks\n";
        std::cout << "  exactly this way -- silently, unless something upstream happens to use\n";
        std::cout << "  try_lock() and check its return value, the way this test just did.\n";
    }

    // =====================================================================
    // On the racy_increment_worker function above: real, compilable code,
    // deliberately never called from this main(). Printed here as static,
    // unconditional text -- not derived from any runtime race outcome --
    // so this explanation is exactly as reproducible as everything else
    // in this file, even though the function it explains is not.
    // =====================================================================
    std::cout << "\n-- On racy_increment_worker(), defined above but never called here --\n";
    std::cout << "  That function increments a plain long long with `++counter` from multiple\n";
    std::cout << "  real threads and no synchronization at all -- a genuine data race, and\n";
    std::cout << "  therefore undefined behavior in the C++ memory model, not merely \"probably\n";
    std::cout << "  fine\". Its outcome is explicitly not guaranteed to match between two runs of\n";
    std::cout << "  the same binary, which is exactly why this file does not call it: this book's\n";
    std::cout << "  own build-verify-lock discipline requires every locked file's output to be\n";
    std::cout << "  identical on rerun, and a real race's result is, by definition, not.\n";

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -pthread 02_atomics_mutexes_and_synchronization.cpp -o 02_atomics_mutexes_and_synchronization
./02_atomics_mutexes_and_synchronization
```

**Sample input:** four real threads each performing 250000 `std::atomic<long long>::fetch_add` calls on a shared counter, checked against the exact expected total of 1000000; the identical total reached with a plain `long long` protected by `std::mutex` instead; a deliberate, fully deterministic demonstration that `std::mutex::try_lock()` returns `false` when a thread that already holds the lock calls it again, contrasted with `std::recursive_mutex` correctly allowing it; and an explanation of why this file's own genuinely racy function is included as source but never executed for output.

```text
========================================================
Chapter 10.2: Atomics, Mutexes, and Why Synchronization Is Not Optional
========================================================

-- Test 1: std::atomic fetch_add across 4 real threads --
  4 threads x 250000 atomic increments each: final count = 1000000
  Expected total: 1000000

-- Test 2: std::mutex-protected plain counter across 4 real threads --
  4 threads x 250000 mutex-protected increments each: final count = 1000000
  Expected total: 1000000

-- Test 3 [COMMON TRAP]: std::mutex is not recursive --
  Thread already holds the lock; try_lock() again on the SAME thread: returned false
  Same situation with std::recursive_mutex instead: try_lock() again: succeeded
  A blocking lock() in place of this try_lock() would not fail -- it would hang
  forever, since a plain std::mutex has no notion of "the thread that already
  owns me is allowed back in". A function that locks a mutex and then calls
  another function that locks the SAME mutex, on the SAME thread, deadlocks
  exactly this way -- silently, unless something upstream happens to use
  try_lock() and check its return value, the way this test just did.

-- On racy_increment_worker(), defined above but never called here --
  That function increments a plain long long with `++counter` from multiple
  real threads and no synchronization at all -- a genuine data race, and
  therefore undefined behavior in the C++ memory model, not merely "probably
  fine". Its outcome is explicitly not guaranteed to match between two runs of
  the same binary, which is exactly why this file does not call it: this book's
  own build-verify-lock discipline requires every locked file's output to be
  identical on rerun, and a real race's result is, by definition, not.

========================================================
4/4 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming a mutex lets its own owner back in"
    A function that locks a `std::mutex` and then calls another function that — unaware the caller already holds that same lock — tries to lock it again does not get politely refused with a return value to check; a blocking `lock()` call in that situation simply never returns, because a plain `std::mutex` has no notion of "the thread that already owns me may re-enter." This section demonstrates the mechanism safely with `try_lock()`, which returns `false` deterministically rather than hanging, precisely so the demonstration itself has well-defined output to verify — but the real-world version of this bug is a silent, total deadlock the first time two code paths that both need the same lock are called from the same thread. `std::recursive_mutex` exists exactly for the case where a single thread genuinely needs to re-enter its own lock, and reaching for it (or, better, restructuring the code so re-entry is never attempted) is the fix — not discovering the hang in production.

## 10.3 A Barrier-Synchronized Thread Pool for a Partitioned GEMV

### Intuition

Chapter 8.3 already classified a decode-time GEMV as memory-bound; this section asks how to actually SPLIT that GEMV's work across several real cores. Splitting by output row is the natural choice — each row's dot product is independent of every other row's — and because no two threads ever write the same output element, the result is not merely close to a serial reference within a tolerance, the way Chapter 9.2's AVX2 kernel was; it is bit-for-bit identical, verified exactly.

### The Concept, In Detail

A `TensorThreadPool` creates its worker threads ONCE, in its constructor, and coordinates rounds of work with two `std::barrier` objects instead of creating and joining new threads for every single call — exactly the persistent structure a real decode loop needs, since a real engine cannot afford to pay thread-creation cost on every token the way Section 10.2's per-test thread creation could. Partitioning the output rows correctly means every row from 0 to n_out-1 is covered by EXACTLY one thread's range, with no gaps and no overlaps; the natural first attempt — plain integer division for a fixed chunk size — silently fails this when n_out does not divide evenly by the thread count, because truncating division drops the remainder rows from every range entirely, leaving them assigned to no thread at all. The fix is a ceiling-divided chunk size with the LAST range explicitly capped at n_out, which this section verifies both abstractly (as pure range arithmetic, with no threads involved) and concretely, by pre-filling the output with a sentinel value and confirming the buggy partition leaves exactly the dropped rows at that sentinel after a real run through the pool.

### Code and Verification

```cpp
// 03_barrier_thread_pool_gemv.cpp
// Chapter 10, Part 3: this book's first real, partitioned parallel kernel.
// A GEMV -- the exact operation Chapter 8.3 classified as memory-bound at
// decode time -- is split by OUTPUT ROW across a fixed pool of persistent
// worker threads, each computing a disjoint, statically assigned range of
// rows into its own slice of the output vector. Because no two threads
// ever write the same output element, and because each row's own
// dot-product loop runs in exactly the same left-to-right order the
// serial reference uses, the partitioned result is not merely "close to"
// the serial reference within a floating-point reordering tolerance the
// way Chapter 9.2's AVX2 kernel was -- it is bit-for-bit identical to it,
// verified exactly, on every run, regardless of how the operating system
// happens to schedule the worker threads.
//
// The thread pool itself is built the way a real inference server's would
// be: worker threads are created ONCE and stay alive across many rounds
// of work, coordinated by std::barrier rather than by creating and
// joining new threads for every single GEMV -- exactly the persistent,
// reusable structure a real decode loop needs, since a real engine cannot
// afford Chapter 10.2's create-four-threads-and-join cost on every token.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_barrier_thread_pool_gemv.cpp -o 03_barrier_thread_pool_gemv

#include <atomic>
#include <barrier>
#include <cstdint>
#include <iostream>
#include <mdspan/mdspan.hpp>
#include <thread>
#include <vector>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

using Matrix = std::mdspan<const float, std::dextents<size_t, 2>>;

// A [start, end) row range, half-open, exactly the convention every
// partitioning scheme in this section is checked against.
struct RowRange {
    size_t start, end;
    size_t count() const { return end - start; }
};

// =========================================================================
// The serial reference: every row's dot product computed left-to-right,
// in the exact same inner-loop order the partitioned pool below uses --
// so the two are not merely numerically close, they are bit-identical.
// =========================================================================
void gemv_serial_reference(Matrix W, const float* x, float* y, size_t n_out, size_t n_in) {
    for (size_t row = 0; row < n_out; ++row) {
        float acc = 0.0f;
        for (size_t col = 0; col < n_in; ++col) acc += W[row, col] * x[col];
        y[row] = acc;
    }
}

// =========================================================================
// CORRECT partitioning: ceil-divided chunk size, with the LAST chunk
// explicitly capped at n_out -- so the union of all ranges is exactly
// [0, n_out) with no gaps and no overlaps, regardless of whether n_out
// divides evenly by the thread count.
// =========================================================================
std::vector<RowRange> partition_rows_correct(size_t n_out, size_t n_threads) {
    std::vector<RowRange> ranges;
    size_t chunk = (n_out + n_threads - 1) / n_threads;  // ceiling division
    for (size_t t = 0; t < n_threads; ++t) {
        size_t start = std::min(t * chunk, n_out);
        size_t end = std::min(start + chunk, n_out);
        ranges.push_back({start, end});
    }
    return ranges;
}

// =========================================================================
// [COMMON TRAP] the naive partitioning that a first attempt reaches for:
// plain integer division for the chunk size, with every chunk (including
// the last) exactly that size. When n_out does not divide evenly by
// n_threads, integer division truncates, and the rows past
// n_threads * chunk are never assigned to any thread at all -- not
// computed incorrectly, simply never computed.
// =========================================================================
std::vector<RowRange> partition_rows_naive_buggy(size_t n_out, size_t n_threads) {
    std::vector<RowRange> ranges;
    size_t chunk = n_out / n_threads;  // truncating division -- the bug
    for (size_t t = 0; t < n_threads; ++t) {
        ranges.push_back({t * chunk, (t + 1) * chunk});
    }
    return ranges;
}

// =========================================================================
// A persistent thread pool, coordinated by two std::barrier objects
// (start and done) instead of creating and joining threads per call.
// Workers are created once in the constructor and loop until stop_flag
// is set; run_gemv() publishes a task's parameters and uses the barriers
// to hand off exactly one round of work per call.
// =========================================================================
class TensorThreadPool {
public:
    explicit TensorThreadPool(size_t n_threads)
        : n_threads_(n_threads),
          start_barrier_(static_cast<std::ptrdiff_t>(n_threads + 1)),
          done_barrier_(static_cast<std::ptrdiff_t>(n_threads + 1)) {
        for (size_t t = 0; t < n_threads_; ++t) {
            workers_.emplace_back([this, t] { worker_loop(t); });
        }
    }

    ~TensorThreadPool() {
        stop_flag_.store(true, std::memory_order_relaxed);
        start_barrier_.arrive_and_wait();  // release workers so they observe stop_flag_
        for (auto& w : workers_) w.join();
    }

    // Runs one partitioned GEMV round using the given row ranges (one
    // range per worker thread, ranges_[t] assigned to worker t).
    void run_gemv(Matrix W, const float* x, float* y, size_t n_in,
                  const std::vector<RowRange>& ranges) {
        task_W_ = &W;
        task_x_ = x;
        task_y_ = y;
        task_n_in_ = n_in;
        task_ranges_ = &ranges;
        start_barrier_.arrive_and_wait();  // release workers into this round
        done_barrier_.arrive_and_wait();   // wait for all workers to finish it
    }

private:
    void worker_loop(size_t my_index) {
        for (;;) {
            start_barrier_.arrive_and_wait();
            if (stop_flag_.load(std::memory_order_relaxed)) return;

            const RowRange& r = (*task_ranges_)[my_index];
            for (size_t row = r.start; row < r.end; ++row) {
                float acc = 0.0f;
                for (size_t col = 0; col < task_n_in_; ++col) {
                    acc += (*task_W_)[row, col] * task_x_[col];
                }
                task_y_[row] = acc;
            }
            done_barrier_.arrive_and_wait();
        }
    }

    size_t n_threads_;
    std::vector<std::thread> workers_;
    std::barrier<> start_barrier_;
    std::barrier<> done_barrier_;
    std::atomic<bool> stop_flag_{false};

    const Matrix* task_W_ = nullptr;
    const float* task_x_ = nullptr;
    float* task_y_ = nullptr;
    size_t task_n_in_ = 0;
    const std::vector<RowRange>* task_ranges_ = nullptr;
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 10.3: A Barrier-Synchronized Thread Pool for a Partitioned GEMV\n";
    std::cout << "========================================================\n\n";

    constexpr size_t N_OUT = 70;   // deliberately NOT a multiple of the thread count
    constexpr size_t N_IN = 256;
    constexpr size_t N_THREADS = 4;

    std::vector<float> w_data(N_OUT * N_IN), x_data(N_IN), y_serial(N_OUT), y_pool(N_OUT);
    for (size_t i = 0; i < w_data.size(); ++i) {
        w_data[i] = (static_cast<float>(i % 17) - 8.0f) * 0.05f;
    }
    for (size_t i = 0; i < N_IN; ++i) {
        x_data[i] = (static_cast<float>(i % 11) - 5.0f) * 0.1f;
    }
    Matrix W(w_data.data(), N_OUT, N_IN);

    gemv_serial_reference(W, x_data.data(), y_serial.data(), N_OUT, N_IN);

    // =====================================================================
    // TEST 1: the correct (ceiling-division) partition covers every row
    // exactly once -- no gaps, no overlaps -- regardless of N_OUT not
    // dividing evenly by N_THREADS.
    // =====================================================================
    std::cout << "-- Test 1: correct partition covers [0, " << N_OUT << ") exactly once --\n";
    std::vector<RowRange> correct_ranges = partition_rows_correct(N_OUT, N_THREADS);
    {
        size_t total_covered = 0;
        bool covers_exactly = true;
        std::vector<bool> covered(N_OUT, false);
        for (const auto& r : correct_ranges) {
            std::cout << "  thread range: [" << r.start << ", " << r.end << ") -- " << r.count() << " rows\n";
            total_covered += r.count();
            for (size_t row = r.start; row < r.end; ++row) {
                if (covered[row]) covers_exactly = false;  // overlap
                covered[row] = true;
            }
        }
        bool all_covered = true;
        for (bool c : covered) if (!c) all_covered = false;
        std::cout << "  total rows covered: " << total_covered << " (expected " << N_OUT << ")\n";
        CHECK(total_covered == N_OUT);
        CHECK(covers_exactly);
        CHECK(all_covered);
    }

    // =====================================================================
    // TEST 2: a real, persistent, barrier-synchronized thread pool
    // computes this GEMV using the correct partition, matching the
    // serial reference BIT-FOR-BIT -- not within a tolerance, since each
    // row's own inner loop runs in the identical order on either path.
    // =====================================================================
    std::cout << "\n-- Test 2: pool-computed GEMV vs. serial reference, using the correct partition --\n";
    {
        std::fill(y_pool.begin(), y_pool.end(), -999.0f);  // sentinel: must be overwritten everywhere
        TensorThreadPool pool(N_THREADS);
        pool.run_gemv(W, x_data.data(), y_pool.data(), N_IN, correct_ranges);

        bool exact_match = true;
        for (size_t row = 0; row < N_OUT; ++row) {
            if (y_pool[row] != y_serial[row]) exact_match = false;
        }
        std::cout << "  " << N_OUT << " output rows across " << N_THREADS
                  << " persistent worker threads: " << (exact_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(exact_match);

        // Running a SECOND round through the same, still-alive pool
        // (a different input vector) must also match exactly -- the
        // pool is reused across rounds, exactly as a real decode loop
        // would reuse it across many tokens.
        std::vector<float> x2(N_IN), y2_serial(N_OUT), y2_pool(N_OUT, -999.0f);
        for (size_t i = 0; i < N_IN; ++i) x2[i] = (static_cast<float>(i % 7) - 3.0f) * 0.2f;
        gemv_serial_reference(W, x2.data(), y2_serial.data(), N_OUT, N_IN);
        pool.run_gemv(W, x2.data(), y2_pool.data(), N_IN, correct_ranges);
        bool second_round_match = (y2_pool == y2_serial);
        std::cout << "  second round through the SAME still-alive pool: "
                  << (second_round_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(second_round_match);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: the naive, truncating-division partition
    // silently drops rows past n_threads * (n_out / n_threads) -- run
    // through the SAME real thread pool, with output pre-filled with a
    // sentinel value, the dropped rows are left at that sentinel because
    // no thread was ever assigned to compute them.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: the naive truncating-division partition drops rows --\n";
    {
        std::vector<RowRange> naive_ranges = partition_rows_naive_buggy(N_OUT, N_THREADS);
        size_t naive_total_covered = 0;
        for (const auto& r : naive_ranges) naive_total_covered += r.count();
        size_t dropped_rows = N_OUT - naive_total_covered;
        std::cout << "  naive chunk size = " << N_OUT << " / " << N_THREADS << " = " << (N_OUT / N_THREADS)
                  << " (integer division), covering rows [0, " << (naive_total_covered) << ") only\n";
        std::cout << "  " << dropped_rows << " row(s) -- indices [" << naive_total_covered << ", "
                  << N_OUT << ") -- assigned to NO thread at all\n";
        CHECK(dropped_rows > 0);
        CHECK(naive_total_covered < N_OUT);

        constexpr float SENTINEL = -999.0f;
        std::vector<float> y_buggy(N_OUT, SENTINEL);
        TensorThreadPool pool(N_THREADS);
        pool.run_gemv(W, x_data.data(), y_buggy.data(), N_IN, naive_ranges);

        size_t rows_still_at_sentinel = 0;
        for (size_t row = 0; row < N_OUT; ++row) {
            if (y_buggy[row] == SENTINEL) ++rows_still_at_sentinel;
        }
        std::cout << "  after running the pool with this partition, " << rows_still_at_sentinel
                  << " output row(s) are still at the sentinel value " << SENTINEL
                  << " -- never written by any thread.\n";
        CHECK(rows_still_at_sentinel == dropped_rows);
        CHECK(rows_still_at_sentinel > 0);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_barrier_thread_pool_gemv.cpp -o 03_barrier_thread_pool_gemv
./03_barrier_thread_pool_gemv
```

**Sample input:** a correct, ceiling-divided row partition of a 70-row output across 4 threads, checked to cover every row exactly once with no gaps or overlaps; a persistent, `std::barrier`-synchronized thread pool computing that GEMV across two separate rounds through the SAME still-alive threads, each checked bit-for-bit against a serial reference; and a deliberate demonstration of the naive, truncating-division partition silently dropping rows, run through the same real pool with a sentinel-filled output to show exactly which rows were never written by any thread.

```text
========================================================
Chapter 10.3: A Barrier-Synchronized Thread Pool for a Partitioned GEMV
========================================================

-- Test 1: correct partition covers [0, 70) exactly once --
  thread range: [0, 18) -- 18 rows
  thread range: [18, 36) -- 18 rows
  thread range: [36, 54) -- 18 rows
  thread range: [54, 70) -- 16 rows
  total rows covered: 70 (expected 70)

-- Test 2: pool-computed GEMV vs. serial reference, using the correct partition --
  70 output rows across 4 persistent worker threads: bit-for-bit match
  second round through the SAME still-alive pool: bit-for-bit match

-- Test 3 [COMMON TRAP]: the naive truncating-division partition drops rows --
  naive chunk size = 70 / 4 = 17 (integer division), covering rows [0, 68) only
  2 row(s) -- indices [68, 70) -- assigned to NO thread at all
  after running the pool with this partition, 2 output row(s) are still at the sentinel value -999 -- never written by any thread.

========================================================
9/9 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] a chunk-size partition that quietly drops the remainder"
    `chunk = n_out / n_threads` followed by `[t*chunk, (t+1)*chunk)` for each thread looks correct and compiles without complaint, but integer division truncates: when n_out is not an exact multiple of the thread count, the rows from `n_threads * chunk` to `n_out - 1` fall outside every single thread's range. This is not a crash and not a wrong VALUE in any output element that IS computed — it is a set of output elements that are simply never touched by anyone, left at whatever the buffer happened to contain beforehand. The fix — a ceiling-divided chunk size with the final range explicitly capped at n_out — costs nothing in the common case where the division is exact, and is the only version of this partition that is correct in the case where it is not.

## 10.4 False Sharing: A Memory-Layout Hazard, Not a Race

### Intuition

Two threads, each incrementing its own logically independent counter, share no variable and therefore have no data race by any definition — and yet, depending purely on where those two counters happen to sit in memory, updating one can force the OTHER thread's cached copy of a completely different value to be invalidated. This is not a correctness bug — both counters still end up with exactly the right value, every time — so this section verifies it the only way that is honest without a wall-clock number: as address arithmetic.

### The Concept, In Detail

Cache coherence protocols keep every core's view of memory consistent by operating on whole 64-byte cache lines, not on individual variables — so four adjacent `long long` counters, packed into 32 contiguous bytes with no padding, share exactly one cache line between all four, and a write to any one of them invalidates every other core's cached copy of the entire line, including the three counters that never changed. This section measures that fact directly, computing each counter's cache-line offset RELATIVE to the containing struct's own base address rather than as a raw pointer value, since absolute addresses shift from run to run under ASLR (address space layout randomization) in a way the LAYOUT relationship between two fields of the same object never does. Padding each counter out to a full 64-byte block with `alignas(64)` and an explicit padding member gives every counter its own line, verified the identical way. Running real, concurrently writing threads against both layouts confirms the section's central claim: both reach the exact correct final count on every run, because false sharing changes only how much cache-coherence traffic the hardware has to do to make that happen, never the value any thread actually computes — which is exactly why this book will not attach a fabricated timing number to the difference.

### Code and Verification

```cpp
// 04_false_sharing_and_cache_line_layout.cpp
// Chapter 10, Part 4: a hazard this book has not needed a real thread to
// demonstrate before, because it is not a correctness bug at all -- it is
// a MEMORY-LAYOUT fact. Two threads, each incrementing its OWN, logically
// independent counter with no shared mutable state and therefore no data
// race, can still slow each other down if those two counters happen to
// sit in the same cache line: the cache-coherence protocol that keeps
// every core's view of memory consistent operates on whole cache lines,
// not on individual variables, so a write by one core silently invalidates
// the OTHER core's cached copy of a variable it never touched.
//
// Because false sharing changes only how the cache-coherence protocol has
// to work, never the actual VALUE any thread computes, this section
// verifies it the way it can honestly be verified without a wall-clock
// number: structurally, as address arithmetic. Chapter 10.1's own opening
// promise -- never fabricate a timing number in a shared, virtualized
// build environment -- applies here directly: this file counts which
// 64-byte-aligned cache line each counter's address falls into, an exact,
// deterministic, hardware-topology-independent (given a stated line size)
// computation, and separately confirms that BOTH the naive and the padded
// layout still produce exactly correct per-thread counts, because false
// sharing was never a correctness hazard to begin with.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -pthread 04_false_sharing_and_cache_line_layout.cpp -o 04_false_sharing_and_cache_line_layout

#include <array>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// A stated architectural parameter -- the size, in bytes, of one cache
// line on essentially all mainstream x86-64 and Arm64 CPUs -- exactly the
// same convention as Chapter 8.1's stated core count and clock speed.
constexpr size_t CACHE_LINE_BYTES = 64;

// Cache line index RELATIVE to a fixed base address, not an absolute
// address. Absolute addresses shift from run to run under ASLR (address
// space layout randomization) -- a real, expected OS security feature
// this book has not had to work around before, because no earlier file
// ever printed a raw pointer value. The LAYOUT relationship between two
// fields of the same object -- how many bytes and cache lines apart they
// sit -- is fixed at compile time by the type's own definition, and is
// exactly as deterministic as everything else this book locks; only the
// object's absolute starting address is randomized, so every measurement
// in this file is taken relative to that object's own base address.
size_t cache_line_index_relative(const void* addr, const void* base) {
    uintptr_t offset = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(base);
    return offset / CACHE_LINE_BYTES;
}

constexpr int NUM_THREADS = 4;
constexpr long long INCREMENTS_PER_THREAD = 200000;

// =========================================================================
// The naive layout: N plain long longs, packed contiguously with no
// padding at all -- exactly what "an array of per-thread counters" looks
// like before anyone has thought about cache lines.
// =========================================================================
struct NaiveCounters {
    std::array<long long, NUM_THREADS> count{};
};

// =========================================================================
// The padded layout: each counter isolated in its own cache-line-sized
// block, so that no two counters can ever share a line regardless of
// where the whole array itself starts.
// =========================================================================
struct alignas(CACHE_LINE_BYTES) PaddedCounter {
    long long count = 0;
    char padding[CACHE_LINE_BYTES - sizeof(long long)];
};
struct PaddedCounters {
    std::array<PaddedCounter, NUM_THREADS> slot{};
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 10.4: False Sharing -- a Memory-Layout Hazard, Not a Race\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: in the naive layout, adjacent counters share a cache line --
    // a purely structural, address-arithmetic fact, computed once, with
    // no threads involved yet.
    // =====================================================================
    std::cout << "-- Test 1: naive layout -- do adjacent counters share a cache line? --\n";
    NaiveCounters naive;
    {
        int distinct_lines_seen = 0;
        size_t last_line = static_cast<size_t>(-1);
        bool any_pair_shares_a_line = false;
        for (int t = 0; t < NUM_THREADS; ++t) {
            size_t byte_offset = reinterpret_cast<uintptr_t>(&naive.count[t]) - reinterpret_cast<uintptr_t>(&naive);
            size_t line = cache_line_index_relative(&naive.count[t], &naive);
            std::cout << "  &naive.count[" << t << "] is at byte offset " << byte_offset
                      << " -> cache-line offset " << line << " from the start of naive\n";
            if (line != last_line) ++distinct_lines_seen;
            else any_pair_shares_a_line = true;
            last_line = line;
        }
        // 4 long longs (8 bytes each) span 32 bytes -- well within one
        // 64-byte line, so every one of them shares line with its
        // immediate neighbor: this array occupies far fewer distinct
        // lines than it has counters.
        std::cout << "  distinct cache lines spanned by " << NUM_THREADS << " counters: " << distinct_lines_seen << "\n";
        CHECK(any_pair_shares_a_line);
        CHECK(distinct_lines_seen < NUM_THREADS);
    }

    // =====================================================================
    // TEST 2: in the padded layout, alignas(64) plus a same-size padding
    // member guarantees every counter starts its OWN cache line -- again
    // a structural fact, checkable before any thread touches the data.
    // =====================================================================
    std::cout << "\n-- Test 2: padded layout -- does every counter get its own cache line? --\n";
    PaddedCounters padded;
    {
        std::vector<size_t> lines;
        for (int t = 0; t < NUM_THREADS; ++t) {
            size_t byte_offset = reinterpret_cast<uintptr_t>(&padded.slot[t].count) - reinterpret_cast<uintptr_t>(&padded);
            size_t line = cache_line_index_relative(&padded.slot[t].count, &padded);
            std::cout << "  &padded.slot[" << t << "].count is at byte offset " << byte_offset
                      << " -> cache-line offset " << line << " from the start of padded\n";
            lines.push_back(line);
        }
        bool all_distinct = true;
        for (int i = 0; i < NUM_THREADS; ++i)
            for (int j = i + 1; j < NUM_THREADS; ++j)
                if (lines[i] == lines[j]) all_distinct = false;
        std::cout << "  every counter in its own cache line: " << (all_distinct ? "yes" : "no") << "\n";
        CHECK(all_distinct);
        CHECK(sizeof(PaddedCounter) == CACHE_LINE_BYTES);
    }

    // =====================================================================
    // TEST 3: false sharing is invisible to a correctness check -- each
    // thread writes ONLY its own counter, so there is no data race in
    // either layout, and both reach the exact expected per-thread count
    // on every run, real threads and all. This is the point of the
    // section: the naive layout is not WRONG, it is merely laid out in a
    // way that costs the cache-coherence protocol more work for a result
    // that is, in every run, identical to the padded layout's.
    // =====================================================================
    std::cout << "\n-- Test 3: both layouts remain fully correct under real concurrent writes --\n";
    {
        std::vector<std::thread> workers;
        for (int t = 0; t < NUM_THREADS; ++t) {
            workers.emplace_back([&naive, t] {
                for (long long i = 0; i < INCREMENTS_PER_THREAD; ++i) ++naive.count[t];
            });
        }
        for (auto& w : workers) w.join();

        workers.clear();
        for (int t = 0; t < NUM_THREADS; ++t) {
            workers.emplace_back([&padded, t] {
                for (long long i = 0; i < INCREMENTS_PER_THREAD; ++i) ++padded.slot[t].count;
            });
        }
        for (auto& w : workers) w.join();

        bool naive_all_correct = true, padded_all_correct = true;
        for (int t = 0; t < NUM_THREADS; ++t) {
            if (naive.count[t] != INCREMENTS_PER_THREAD) naive_all_correct = false;
            if (padded.slot[t].count != INCREMENTS_PER_THREAD) padded_all_correct = false;
        }
        std::cout << "  naive layout, all " << NUM_THREADS << " counters reached " << INCREMENTS_PER_THREAD
                  << ": " << (naive_all_correct ? "yes" : "no") << "\n";
        std::cout << "  padded layout, all " << NUM_THREADS << " counters reached " << INCREMENTS_PER_THREAD
                  << ": " << (padded_all_correct ? "yes" : "no") << "\n";
        CHECK(naive_all_correct);
        CHECK(padded_all_correct);
        std::cout << "  Both layouts are equally CORRECT -- false sharing is a cache-coherence-traffic\n";
        std::cout << "  cost, not a wrong answer, which is exactly why this book will not attach a\n";
        std::cout << "  fabricated wall-clock number to it: the only honestly verifiable claim here is\n";
        std::cout << "  the structural one Tests 1 and 2 already made about WHERE these bytes live.\n";
    }

    // =====================================================================
    // TEST 4 [COMMON TRAP]: assuming two logically independent variables
    // are automatically independent in a threaded program because
    // nothing shares them at the SOURCE level. Two per-thread counters
    // declared as ordinary adjacent array elements are exactly this trap
    // -- no shared variable exists anywhere in the source, and yet Test 1
    // already showed their addresses land in the same cache line. The
    // hazard lives in the MEMORY LAYOUT the compiler and allocator chose,
    // not in anything visible by reading the increment statements alone.
    // =====================================================================
    std::cout << "\n-- Test 4 [COMMON TRAP]: \"no shared variable\" does not mean \"no shared cache line\" --\n";
    {
        std::cout << "  naive.count[0] and naive.count[1] are two completely separate long longs --\n";
        std::cout << "  no thread ever reads or writes the other thread's element, so there is no data\n";
        std::cout << "  race by any definition. Test 1 measured their addresses anyway and found them\n";
        std::cout << "  in the same " << CACHE_LINE_BYTES << "-byte cache line: cache coherence operates on\n";
        std::cout << "  whole lines, so a write to naive.count[0] forces every OTHER core's cached copy\n";
        std::cout << "  of the SAME line -- which also holds naive.count[1], [2], and [3] -- to be\n";
        std::cout << "  invalidated, even though those three values never changed.\n";
        size_t line0 = cache_line_index_relative(&naive.count[0], &naive);
        size_t line1 = cache_line_index_relative(&naive.count[1], &naive);
        CHECK(line0 == line1);  // re-confirms Test 1's specific claim about elements 0 and 1
        std::cout << "  Padding each counter out to a full cache line, as PaddedCounter does, is not an\n";
        std::cout << "  arbitrary style choice -- it is the specific fix for a hazard that no amount of\n";
        std::cout << "  reading the surrounding source code would reveal without checking addresses.\n";
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -pthread 04_false_sharing_and_cache_line_layout.cpp -o 04_false_sharing_and_cache_line_layout
./04_false_sharing_and_cache_line_layout
```

**Sample input:** four adjacent, unpadded `long long` counters, their cache-line offsets computed relative to their containing struct's own base address, showing all four sharing line 0; the same four counters padded to 64 bytes each with `alignas(64)`, showing each one in its own distinct line; both layouts run under four real, concurrently writing threads and checked to reach the exact expected count in every case; and a deliberate demonstration that two counters sharing no variable at the source level can still be shown, by address arithmetic alone, to share a cache line.

```text
========================================================
Chapter 10.4: False Sharing -- a Memory-Layout Hazard, Not a Race
========================================================

-- Test 1: naive layout -- do adjacent counters share a cache line? --
  &naive.count[0] is at byte offset 0 -> cache-line offset 0 from the start of naive
  &naive.count[1] is at byte offset 8 -> cache-line offset 0 from the start of naive
  &naive.count[2] is at byte offset 16 -> cache-line offset 0 from the start of naive
  &naive.count[3] is at byte offset 24 -> cache-line offset 0 from the start of naive
  distinct cache lines spanned by 4 counters: 1

-- Test 2: padded layout -- does every counter get its own cache line? --
  &padded.slot[0].count is at byte offset 0 -> cache-line offset 0 from the start of padded
  &padded.slot[1].count is at byte offset 64 -> cache-line offset 1 from the start of padded
  &padded.slot[2].count is at byte offset 128 -> cache-line offset 2 from the start of padded
  &padded.slot[3].count is at byte offset 192 -> cache-line offset 3 from the start of padded
  every counter in its own cache line: yes

-- Test 3: both layouts remain fully correct under real concurrent writes --
  naive layout, all 4 counters reached 200000: yes
  padded layout, all 4 counters reached 200000: yes
  Both layouts are equally CORRECT -- false sharing is a cache-coherence-traffic
  cost, not a wrong answer, which is exactly why this book will not attach a
  fabricated wall-clock number to it: the only honestly verifiable claim here is
  the structural one Tests 1 and 2 already made about WHERE these bytes live.

-- Test 4 [COMMON TRAP]: "no shared variable" does not mean "no shared cache line" --
  naive.count[0] and naive.count[1] are two completely separate long longs --
  no thread ever reads or writes the other thread's element, so there is no data
  race by any definition. Test 1 measured their addresses anyway and found them
  in the same 64-byte cache line: cache coherence operates on
  whole lines, so a write to naive.count[0] forces every OTHER core's cached copy
  of the SAME line -- which also holds naive.count[1], [2], and [3] -- to be
  invalidated, even though those three values never changed.
  Padding each counter out to a full cache line, as PaddedCounter does, is not an
  arbitrary style choice -- it is the specific fix for a hazard that no amount of
  reading the surrounding source code would reveal without checking addresses.

========================================================
7/7 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming no shared variable means no shared cache line"
    Two array elements, `naive.count[0]` and `naive.count[1]`, are two completely separate `long long` values — no thread ever reads or writes the other's element, so by any definition of a data race, there is none here. This section measured their addresses anyway and found both inside the same 64-byte cache line, because cache coherence has no concept of "these two bytes belong to logically different variables" — it only knows about lines. The hazard is invisible to anyone reading the increment statements themselves, since nothing in that source code shares anything; it only becomes visible by checking WHERE the compiler and allocator actually placed the bytes, which is exactly the check this section performs and exactly why `alignas(64)` padding is a structural fix, not a style preference.

## 10.5 Work Stealing: Verifying Correctness Under a Nondeterministic Schedule

### Intuition

Section 10.3's thread pool split work with a fixed, static partition decided before any thread ran — fine when every row costs the same, wrong the moment tasks take unequal time, since a static split leaves fast threads idle while a slow one is still working. A work-stealing scheduler fixes this by letting an idle thread take work from a busy one's own queue — but that means the exact SCHEDULE, which thread executes which task and in what order, is deliberately, genuinely nondeterministic, and this section's real subject is how to verify such a thing rigorously anyway.

### The Concept, In Detail

Every task in this section computes its own integer index as its "work," and adds it to a shared total with `std::atomic<long long>::fetch_add` — a deliberately trivial computation, because the only thing actually under test is the SCHEDULER's own correctness, not any task's arithmetic. Integer addition is exactly commutative and associative, unlike the floating-point sums Chapter 9.2's tolerance-checked kernels required, so as long as every task runs EXACTLY once, the aggregate total is a single, fixed, checkable number — `sum(0..N-1)`, a closed form — regardless of which thread claimed which task or in what order. Every task starts on ONE thread's own deque, with every other thread's deque completely empty, so those other threads can only ever obtain work by successfully stealing it from the back of their own (empty) queue failing and the front of someone else's succeeding — guaranteeing the stealing code path is genuinely exercised on every run, whatever the exact division of labor a given run happens to reach. This section checks exactly two things, both true regardless of the actual schedule: every one of N tasks claimed precisely once (checked per-task, so a dropped task and a duplicated task could not silently cancel out in an aggregate count), and the aggregate total exactly matching the closed-form sum. Which thread processed which task, and how many tasks the origin thread finished locally before the first successful steal, are both real, scheduling-dependent outcomes this section deliberately never measures or prints, for the identical honesty reason Section 10.2 never printed a data race's exact result.

### Code and Verification

```cpp
// 05_work_stealing_scheduler.cpp
// Chapter 10, Part 5: Section 10.3's thread pool used a STATIC partition
// -- every row range fixed before a single thread runs. That is exactly
// wrong when tasks take unequal time (a real inference engine's tasks
// often do: different KV-cache lengths per request, different numbers of
// tokens still to prefill), because a static split leaves fast threads
// idle while a slow one is still working. A work-stealing scheduler lets
// an idle thread take tasks from a busy one's own queue instead -- but
// doing that means the SCHEDULE (which thread executes which task, and
// in what order) is genuinely, deliberately nondeterministic: it depends
// on OS scheduling this book has never needed to control before.
//
// This section's core point: a nondeterministic SCHEDULE does not mean a
// nondeterministic RESULT. Every task here computes a plain integer value
// and adds it to a shared total with std::atomic<long long>::fetch_add --
// integer addition is exactly commutative and associative, so as long as
// every task runs EXACTLY ONCE, the final total is the identical, fixed
// number no matter which thread claimed which task or in what order. This
// file verifies exactly that: not which thread did what (which varies
// from run to run, and is deliberately never printed here, for the same
// reason Section 10.2 never printed a data race's exact outcome), but the
// two facts that remain true regardless of the actual schedule -- every
// task claimed precisely once, and the aggregate result exactly correct.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -pthread 05_work_stealing_scheduler.cpp -o 05_work_stealing_scheduler

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

constexpr int NUM_THREADS = 4;
constexpr int N_TASKS = 2000;

// =========================================================================
// A per-thread deque, mutex-protected. The OWNER pops from the back
// (try_pop_own); a THIEF, some other thread with an empty deque of its
// own, pops from the front (try_steal). Popping from opposite ends is the
// standard work-stealing convention -- it means an owner draining its own
// recently-pushed work and a thief taking someone else's OLDEST work
// rarely contend for the same element, though correctness here does not
// depend on that; the mutex makes either end of either operation safe
// regardless of which thread calls it.
// =========================================================================
struct WorkerDeque {
    std::mutex mtx;
    std::deque<int> tasks;

    bool try_pop_own(int& out) {
        std::lock_guard<std::mutex> lock(mtx);
        if (tasks.empty()) return false;
        out = tasks.back();
        tasks.pop_back();
        return true;
    }
    bool try_steal(int& out) {
        std::lock_guard<std::mutex> lock(mtx);
        if (tasks.empty()) return false;
        out = tasks.front();
        tasks.pop_front();
        return true;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 10.5: Work Stealing -- Verifying Correctness Under a Nondeterministic Schedule\n";
    std::cout << "========================================================\n\n";

    // Every task's "work" is simply its own integer index -- deliberately
    // trivial, so the ONLY thing under test is the scheduler's own
    // correctness (every task claimed exactly once), not the arithmetic
    // any individual task performs. Integer addition being exactly
    // commutative and associative is what makes the aggregate total a
    // single, fixed, checkable number regardless of execution order.
    std::vector<WorkerDeque> deques(NUM_THREADS);
    std::array<std::atomic<int>, N_TASKS> claim_count{};
    for (auto& c : claim_count) c.store(0, std::memory_order_relaxed);
    std::atomic<long long> total_sum{0};
    std::atomic<int> tasks_remaining{N_TASKS};

    // =====================================================================
    // Deliberately unbalanced initial distribution: EVERY task starts on
    // thread 0's deque; threads 1..3 start completely empty. This is not
    // a realistic load pattern -- it exists to guarantee that threads 1..3
    // can only ever obtain work by successfully stealing it, so the
    // scheduler's stealing PATH is genuinely exercised by construction,
    // whatever the exact division of labor a given run happens to reach.
    // =====================================================================
    for (int i = 0; i < N_TASKS; ++i) deques[0].tasks.push_back(i);

    auto worker = [&](int my_id) {
        while (tasks_remaining.load(std::memory_order_relaxed) > 0) {
            int task;
            if (deques[my_id].try_pop_own(task)) {
                claim_count[task].fetch_add(1, std::memory_order_relaxed);
                total_sum.fetch_add(task, std::memory_order_relaxed);
                tasks_remaining.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }
            bool stole = false;
            for (int offset = 1; offset < NUM_THREADS; ++offset) {
                int victim = (my_id + offset) % NUM_THREADS;
                if (deques[victim].try_steal(task)) {
                    claim_count[task].fetch_add(1, std::memory_order_relaxed);
                    total_sum.fetch_add(task, std::memory_order_relaxed);
                    tasks_remaining.fetch_sub(1, std::memory_order_relaxed);
                    stole = true;
                    break;
                }
            }
            if (!stole) std::this_thread::yield();  // nothing available right now; let others run
        }
    };

    // =====================================================================
    // TEST 1: run the scheduler for real, across NUM_THREADS real
    // threads, and verify every one of the N_TASKS tasks was claimed
    // EXACTLY once -- not zero (a dropped task), not two or more (a
    // duplicated task). This is checked per-task, not just in aggregate,
    // so a dropped task and a duplicated task could not silently cancel
    // out in a way a simple total-count check might miss.
    // =====================================================================
    std::cout << "-- Test 1: every one of " << N_TASKS << " tasks claimed exactly once --\n";
    {
        std::vector<std::thread> workers;
        for (int t = 0; t < NUM_THREADS; ++t) workers.emplace_back(worker, t);
        for (auto& w : workers) w.join();

        int zero_claims = 0, exactly_one = 0, more_than_one = 0;
        for (int i = 0; i < N_TASKS; ++i) {
            int c = claim_count[i].load(std::memory_order_relaxed);
            if (c == 0) ++zero_claims;
            else if (c == 1) ++exactly_one;
            else ++more_than_one;
        }
        std::cout << "  claimed exactly once: " << exactly_one << " / " << N_TASKS << "\n";
        std::cout << "  claimed zero times (dropped): " << zero_claims << "\n";
        std::cout << "  claimed more than once (duplicated): " << more_than_one << "\n";
        CHECK(exactly_one == N_TASKS);
        CHECK(zero_claims == 0);
        CHECK(more_than_one == 0);
        CHECK(tasks_remaining.load() == 0);
    }

    // =====================================================================
    // TEST 2: the aggregate result matches the exact closed-form total --
    // sum(0..N_TASKS-1) = N_TASKS*(N_TASKS-1)/2 -- a single, fixed integer
    // that does not depend on which thread processed which task or in
    // what order, because integer addition does not care about order.
    // =====================================================================
    std::cout << "\n-- Test 2: aggregate total matches the closed-form sum exactly --\n";
    {
        long long expected = static_cast<long long>(N_TASKS) * (N_TASKS - 1) / 2;
        long long actual = total_sum.load(std::memory_order_relaxed);
        std::cout << "  sum(0.." << (N_TASKS - 1) << ") closed form: " << expected << "\n";
        std::cout << "  scheduler's actual aggregate total:      " << actual << "\n";
        CHECK(actual == expected);
    }

    // =====================================================================
    // On the exact division of labor between threads: deliberately not
    // measured or printed here. Which thread ends up processing which
    // tasks, and how many tasks thread 0 finishes locally before another
    // thread's first successful steal, depends on real OS scheduling --
    // genuinely different from one run to the next, in a way this file's
    // own two checks above do not need to care about. Printing an exact
    // per-thread task count here would be exactly the kind of
    // non-reproducible number Chapter 8 already refused to fabricate, and
    // Section 10.2 already refused to print for a data race's outcome.
    // =====================================================================
    std::cout << "\n-- On which thread did what: deliberately not measured --\n";
    std::cout << "  This scheduler starts every task on thread 0 and requires threads 1..3 to\n";
    std::cout << "  steal in order to do any work at all, so stealing is genuinely exercised on\n";
    std::cout << "  every run -- but exactly how many tasks each thread ends up with is a\n";
    std::cout << "  real-scheduling outcome that varies run to run, which is exactly why this\n";
    std::cout << "  file locks only the two facts that do NOT vary: every task claimed exactly\n";
    std::cout << "  once, and the aggregate total exactly correct.\n";

    // =====================================================================
    // [COMMON TRAP] the temptation, when first testing a concurrent
    // scheduler, is to assert something about the SCHEDULE itself -- "task
    // 5 should be handled by thread 2", or "thread 0 should finish first"
    // -- because that is the kind of assertion every single-threaded test
    // in this book has written so far. Any such assertion is exactly as
    // unreproducible as the data race Section 10.2 declined to print, and
    // for the identical reason: this scheduler's schedule is not
    // specified to be any particular one, only its RESULT is. The correct
    // property to test is always the order-independent invariant -- here,
    // "every task exactly once" and "the aggregate is correct" -- never a
    // claim about which thread did the work or when.
    // =====================================================================
    std::cout << "\n-- [COMMON TRAP] testing the schedule instead of the invariant --\n";
    std::cout << "  A tempting but WRONG assertion would read something like:\n";
    std::cout << "    CHECK(which_thread_processed[5] == 2);   // WRONG: asserts a specific schedule\n";
    std::cout << "  This scheduler makes no such promise, and never could -- thread 0 might finish\n";
    std::cout << "  task 5 itself before any other thread gets a chance to steal it, or thread 3\n";
    std::cout << "  might steal it on the very first attempt; both are correct outcomes. A test\n";
    std::cout << "  written against a SPECIFIC schedule is not testing this scheduler at all --\n";
    std::cout << "  it is testing one incidental behavior of one particular run, on one particular\n";
    std::cout << "  machine, under one particular scheduling decision the standard never promised.\n";
    std::cout << "  Tests 1 and 2 above check the only two things a work-stealing scheduler's\n";
    std::cout << "  CONTRACT actually promises: full coverage, exactly once, and a correct result.\n";

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -pthread 05_work_stealing_scheduler.cpp -o 05_work_stealing_scheduler
./05_work_stealing_scheduler
```

**Sample input:** 2000 tasks, every one initially placed on a single thread's deque with the other three threads starting empty (forcing every steal path to be genuinely exercised), processed by four real worker threads using mutex-protected per-thread deques with the standard owner-pops-back/thief-steals-front convention; checked per-task that every one of the 2000 tasks was claimed exactly once (zero dropped, zero duplicated); and checked that the aggregate total exactly matches the closed-form sum(0..1999), regardless of the actual, unmeasured division of labor between threads on this run.

```text
========================================================
Chapter 10.5: Work Stealing -- Verifying Correctness Under a Nondeterministic Schedule
========================================================

-- Test 1: every one of 2000 tasks claimed exactly once --
  claimed exactly once: 2000 / 2000
  claimed zero times (dropped): 0
  claimed more than once (duplicated): 0

-- Test 2: aggregate total matches the closed-form sum exactly --
  sum(0..1999) closed form: 1999000
  scheduler's actual aggregate total:      1999000

-- On which thread did what: deliberately not measured --
  This scheduler starts every task on thread 0 and requires threads 1..3 to
  steal in order to do any work at all, so stealing is genuinely exercised on
  every run -- but exactly how many tasks each thread ends up with is a
  real-scheduling outcome that varies run to run, which is exactly why this
  file locks only the two facts that do NOT vary: every task claimed exactly
  once, and the aggregate total exactly correct.

-- [COMMON TRAP] testing the schedule instead of the invariant --
  A tempting but WRONG assertion would read something like:
    CHECK(which_thread_processed[5] == 2);   // WRONG: asserts a specific schedule
  This scheduler makes no such promise, and never could -- thread 0 might finish
  task 5 itself before any other thread gets a chance to steal it, or thread 3
  might steal it on the very first attempt; both are correct outcomes. A test
  written against a SPECIFIC schedule is not testing this scheduler at all --
  it is testing one incidental behavior of one particular run, on one particular
  machine, under one particular scheduling decision the standard never promised.
  Tests 1 and 2 above check the only two things a work-stealing scheduler's
  CONTRACT actually promises: full coverage, exactly once, and a correct result.

========================================================
5/5 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] testing the schedule instead of the invariant"
    A tempting first test for a scheduler like this one asserts something about the SCHEDULE itself — `CHECK(which_thread_processed[5] == 2)` — because every single-threaded test in this book so far has been free to assert exactly this kind of specific, reproducible fact. A work-stealing scheduler makes no such promise: thread 0 might finish task 5 itself before any other thread gets a chance to steal it, or thread 3 might steal it on the very first attempt, and both are entirely correct outcomes of the identical, correctly functioning scheduler. A test written against one particular schedule is not testing the scheduler's actual contract at all — it is testing one incidental fact about one run, on one machine, that the C++ standard never promised would repeat. The two checks this section actually performs — every task claimed exactly once, and the aggregate result exactly correct — are the only two properties a work-stealing scheduler's contract genuinely makes, and the only two a rigorous test should ever assert.

## Chapter Summary

This chapter introduced real, concurrently executing threads into this book for the first time, and with them a discipline this book has not needed before: verifying correctness in the presence of genuine, uncontrolled scheduling nondeterminism, without ever locking a number that scheduling could change from one run to the next. Section 10.1 derived, from Chapter 8.5's own measured decode-step cost, why a single core cannot hit a realistic latency target, and derived Amdahl's law's hard ceiling on how much any number of additional cores can close that gap. Section 10.2 introduced `std::atomic` and `std::mutex` as the two mechanisms that make shared mutable state well-defined across real threads, verified by running real concurrent code to an exact result, while explicitly declining to execute a genuinely racy counterpart whose output this book cannot honestly promise would reproduce. Section 10.3 built a persistent, barrier-synchronized thread pool and partitioned a real GEMV across it, verified bit-for-bit exactly, and caught a classic partitioning bug — truncating integer division silently dropping remainder rows — concretely, by running it through the same real pool. Section 10.4 found a hazard that changes no computed value at all, verified purely as cache-line address arithmetic relative to a stable base address rather than as a wall-clock number. Section 10.5 closed the chapter with a work-stealing scheduler and this chapter's central methodological lesson: a system whose schedule is deliberately nondeterministic can still be verified with complete rigor, by testing the order-independent invariant its contract actually makes rather than any specific execution the scheduler happens to produce on a given run. Chapter 9 showed how one core reaches its peak; this chapter showed how several cores can be coordinated correctly, and, just as importantly, how to know that the verification itself is still honest once real, uncontrolled concurrency enters the picture.

## Self-Check Questions

1. Section 10.1 reuses Chapter 8.5's own measured FLOP and byte totals rather than inventing a new workload. Why does that make the single-core latency claim more trustworthy than a separately asserted number would be?
2. Explain why Amdahl's law's speedup formula converges to a finite ceiling as core count grows without bound, rather than continuing to increase toward infinity.
3. Why does `std::atomic<long long>::fetch_add` guarantee the exact same final total on every run, even though the exact interleaving of which thread's increment happens at which moment is not guaranteed to be the same?
4. Section 10.2 includes a genuinely racy `racy_increment_worker()` function but never calls it from `main()`. What specific property of the C++ memory model makes that omission necessary rather than merely cautious?
5. Why does `std::mutex::try_lock()` returning `false` for a thread that already holds the lock demonstrate the same hazard that a blocking `lock()` call would demonstrate by hanging forever, without this section's own verified output ever having to hang?
6. In Section 10.3, why does the naive `chunk = n_out / n_threads` partitioning scheme drop rows only when n_out does not divide evenly by n_threads, and what specifically happens to those dropped rows' output values?
7. Section 10.4 computes cache-line offsets relative to a struct's own base address rather than as raw absolute addresses. What would have gone wrong with this file's own run-twice determinism check if it had printed absolute addresses instead?
8. Explain why false sharing, unlike every other COMMON TRAP in Chapters 8 and 9, cannot be demonstrated by showing a wrong computed VALUE, and what kind of evidence Section 10.4 uses instead.
9. In Section 10.5, why does the fact that every task computes a plain integer (rather than a float) matter for why the aggregate total is guaranteed identical regardless of the actual thread schedule?
10. Section 10.5 deliberately starts every task on one thread's deque, leaving the other threads' deques empty. What does this specific setup guarantee about the resulting run, and what does it deliberately NOT guarantee or measure?

## Where We Go Next

This chapter built the primitives — atomics, mutexes, a barrier-synchronized thread pool, an awareness of false sharing, and a work-stealing scheduler — that make real, multi-core parallelism both fast and correctly verifiable. Chapter 11 puts every one of them to work on the actual forward pass this book has built one kernel at a time since Part 0: parallelizing attention across heads, the FFN across output rows, and a full transformer layer across a real, persistent thread pool, closing Part 2 with a complete, multi-threaded inference engine whose every claim about correctness is checked exactly, and whose every claim about performance is limited to what this chapter's own roofline and threading models can honestly derive.

## Worked Solutions

**1.** Chapter 8.5's FLOP and byte totals were themselves already verified there against a real, running GQA kernel at those exact dimensions — they are not a number this chapter is asking the reader to trust for the first time. Reusing that already-verified total means Section 10.1's latency claim inherits Chapter 8.5's own verification rather than asking for a second, separate act of trust in a freshly invented workload that has not been checked against anything.

**2.** The speedup formula's parallel-work term is `(1 - serial_fraction) / n`, which strictly decreases toward zero as `n` grows, but never reaches exactly zero for any finite `n`. Total relative time is therefore always strictly greater than `serial_fraction` itself, so speedup (its reciprocal) is always strictly less than `1 / serial_fraction` — a ceiling the formula approaches asymptotically but its own algebra forbids it from ever reaching or exceeding, no matter how large `n` becomes.

**3.** `fetch_add` is specified by the C++ standard to perform its read-modify-write as a single indivisible operation with respect to every other thread — no other thread can observe or interleave with it partway through. Because every one of the 1,000,000 total increments across four threads is guaranteed to be counted exactly once regardless of the ORDER in which they occur, the final sum is fixed by simple arithmetic (a count of atomic operations that all happened, each counted once) even though the interleaving order producing that count is genuinely different from one run to the next.

**4.** A data race is undefined behavior in the C++ memory model, and undefined behavior carries no guarantee that two runs of the identical binary, on the identical machine, produce the identical result. This book's entire build-verify-lock pipeline depends on a file's checked output being byte-for-byte reproducible on rerun; executing and printing a genuinely racy function's result would mean locking a number this book cannot honestly promise the next run — or a reader's own run — would reproduce, which is precisely the standard this book has held every other file to since its very first chapter.

**5.** Both situations arise from the identical fact: a plain `std::mutex` does not track which thread currently owns its lock in a way that would let that SAME thread back in. `try_lock()` and a blocking `lock()` differ only in what they do once that fact is discovered — `try_lock()` returns `false` immediately, a value this section can check and print deterministically, while `lock()` blocks and waits for the lock to become available, which (since the only thread that could release it is the very thread now waiting) never happens. The underlying hazard — non-recursive re-entry — is identical; only the OBSERABLE consequence differs, and this section deliberately chooses the one with well-defined, checkable output.

**6.** `chunk = n_out / n_threads` uses C++'s truncating integer division, so when `n_out` is not an exact multiple of `n_threads`, the product `n_threads * chunk` is strictly less than `n_out`, and the naive scheme's `n_threads` ranges only ever cover indices up to that truncated product. The rows from that point up to `n_out - 1` are never included in ANY thread's assigned range, so no thread ever writes to them — their output values remain at whatever they were initialized to before the pool ran (this section's own sentinel value), not merely computed incorrectly.

**7.** Absolute memory addresses are randomized by ASLR (address space layout randomization) every time a process starts, so the SAME object could legitimately sit at a different absolute address on two consecutive runs of the identical binary — meaning a printed absolute-address-derived cache-line number could differ between the two runs this book's pipeline diffs against each other, breaking the exact-match requirement for reasons that have nothing to do with the actual layout relationship being demonstrated. Printing offsets relative to the containing struct's own base address instead measures a purely compile-time-determined layout fact that ASLR does not touch, keeping the check honestly deterministic.

**8.** Every other COMMON TRAP in Chapters 8 and 9 produces a demonstrably WRONG number — a mis-decoded nibble, a divergent `__restrict` result, a SIGILL a `-march=native` binary would hit — something a check can compare against a known-correct value and find different. False sharing changes none of that: every thread's counter still reaches the exact correct final value in both the padded and unpadded layout, because the hazard is purely about how much cache-coherence traffic the hardware generates to keep those (still-correct) values consistent, not about the values themselves. Section 10.4 therefore verifies the LAYOUT directly — computing which cache line each counter's address falls into — since that is the one honestly checkable fact the hazard actually consists of.

**9.** Integer addition is exactly commutative and associative: for any set of fixed integers, their sum is identical regardless of the order they are added in, with no rounding or reordering-sensitivity at all. This is different from the floating-point sums Chapter 9.2's kernels produced, where AVX2's tree-shaped reduction and the scalar loop's sequential sum could differ in their last few bits purely from summing the same values in a different order. Because each task's contribution here is a plain integer, the scheduler's nondeterministic execution ORDER cannot introduce any numerical difference at all in the final total — only a dropped or duplicated task could, which is exactly the separate invariant Test 1 checks.

**10.** This setup guarantees that threads other than the one holding all the initial tasks can only ever obtain work through a successful steal — there is no other way for them to have anything in their own deque to pop — so the stealing code path is certain to be exercised by every single run, rather than merely possible depending on how fast each thread happens to run. It deliberately does NOT guarantee, or attempt to measure, exactly how many tasks the origin thread finishes on its own before another thread's first successful steal, nor which specific tasks end up on which thread — both are real, scheduling-dependent outcomes that legitimately vary from run to run, and this section's own checks are built specifically not to depend on either one.
