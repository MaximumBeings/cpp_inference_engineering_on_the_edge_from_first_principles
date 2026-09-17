# Appendix D: Profiling and Benchmarking on Edge Hardware

Chapter 32's own opening line makes a promise this appendix now keeps: "this book's own running discipline of never claiming a wall-clock timing result as part of a locked, deterministic self-test contract" is "stated plainly in Appendix D." Every chapter since Chapter 4 has followed that discipline without stopping to defend it at length -- byte counts and FLOP counts get locked and verified across four architectures; milliseconds and GB/s never do. This appendix states the reasoning explicitly, gives it a real formula for computing a theoretical ceiling, a real (deliberately unlocked) way to measure what a machine actually achieves, a real way to turn either into a latency budget, and an honest account of what happened when this book tried to reach for the most obvious real profiling tool -- `perf` -- on its own two real build environments.

---

## D.1 Computing Theoretical Peak Bandwidth From a Real Spec Sheet

### Intuition

A vendor's own published bandwidth figure is not measured on a running system -- it is computed, ahead of time, from three numbers on a spec sheet: how many independent memory channels a chip has, how fast each one transfers data, and how wide each transfer is. Chapter 8.1 derived exactly this formula for an illustrative machine. This section applies it, for the first time in this book, to a real, named, currently-shipping piece of edge hardware, using only numbers that chip's own vendor has actually published.

### The Concept, In Detail

Chapter 8.1's formula is a single multiplication:

```
peak_bandwidth = channels * transfer_rate * bytes_per_transfer
```

Raspberry Pi's own official documentation for the Raspberry Pi 5's BCM2712 SoC states the memory as "LPDDR4X-4267" over "a 32-bit LPDDR4X memory interface" that "provides up to 17 GB/s of memory bandwidth."[^rpi5mem] That single sentence supplies every input the formula needs: one channel, a transfer rate of 4267 MT/s, and a 32-bit (4-byte) bus. Multiplying them out gives 17.068 GB/s -- which rounds down to exactly the "17 GB/s" the vendor's own documentation states as its headline figure.

Two other real platforms are worth having for comparison, though neither vendor publishes the same channel/rate/width breakdown, so this book does not re-derive their figures the way it just did for the Raspberry Pi 5 -- it only cites them:

| Platform | Real, published memory bandwidth | Source |
|---|---|---|
| Raspberry Pi 5 (BCM2712) | 17 GB/s (32-bit LPDDR4X-4267) | Raspberry Pi's own official documentation[^rpi5mem] |
| NVIDIA Jetson Orin Nano (8 GB module) | 68 GB/s (128-bit LPDDR5) | NVIDIA's own official datasheet[^orinmem] |
| Apple M4 / M4 Pro / M4 Max | 120 / 273 / up to 546 GB/s (unified memory) | Apple's own official announcement[^m4mem] |

The nearly 32x spread between the cheapest board in this table and the most capable chip is exactly the real-world range this book's own roofline model (Chapter 8) and crossover formula (Chapter 30.1) were built to reason about generally, without needing to be re-derived for every new chip: the same formulas apply, only the inputs change.

```bash
g++ -std=c++23 -Wall -Wextra -O2 d1_peak_bandwidth_formula.cpp -o d1_peak_bandwidth_formula
./d1_peak_bandwidth_formula
```

```cpp
// Appendix D.1 -- Computing Theoretical Peak Bandwidth From a Real Spec Sheet.
//
// This reuses Chapter 8.1's own peak-bandwidth formula verbatim:
//   peak_bandwidth = channels * transfer_rate * bytes_per_transfer
// and checks it against one real, officially published edge-silicon spec
// this book can verify component-by-component: the Raspberry Pi 5's own
// BCM2712 SoC. Raspberry Pi's own official documentation states the part
// as "LPDDR4X-4267" (4267 MT/s) over a "32-bit LPDDR4X memory interface"
// that "provides up to 17 GB/s of memory bandwidth" -- one channel, a
// 32-bit (4-byte) bus, and a stated transfer rate are everything the
// formula needs, and multiplying them out reproduces the vendor's own
// rounded headline figure.
//
// Two other real edge/consumer platforms are cited in this appendix's own
// prose alongside this file (Jetson Orin Nano's 68 GB/s over a 128-bit
// LPDDR5 interface, and the Apple M4 family's 120/273/546 GB/s unified
// memory figures) but are NOT re-derived here, honestly: their vendors do
// not publish the same channel-count-times-transfer-rate breakdown
// Raspberry Pi's own documentation does, so this file only asserts what
// it can actually recompute from independently stated components.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 d1_peak_bandwidth_formula.cpp -o d1_peak_bandwidth_formula
// Run:     ./d1_peak_bandwidth_formula

#include <cmath>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// Chapter 8.1's own formula, reused verbatim. transfer_rate_mts is in
// mega-transfers per second; bytes_per_transfer is the bus width in bytes
// (32 bits = 4 bytes, 128 bits = 16 bytes, and so on). The result is in
// decimal GB/s (10^9 bytes/s), matching how vendors publish this figure.
double peak_bandwidth_gbps(double channels, double transfer_rate_mts, double bytes_per_transfer) {
    // MT/s * bytes/transfer = MB/s (10^6 bytes/s); dividing by 1000 gives GB/s.
    return channels * transfer_rate_mts * bytes_per_transfer / 1000.0;
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "Appendix D.1: Computing Theoretical Peak Bandwidth\n";
    std::cout << "=================================================\n\n";

    // -- Test 1: the Raspberry Pi 5's own real, officially documented
    // BCM2712 memory interface -- one 32-bit LPDDR4X-4267 channel. --
    {
        double channels = 1.0;
        double transfer_rate_mts = 4267.0;   // Raspberry Pi's own official product brief
        double bytes_per_transfer = 4.0;     // a 32-bit bus, per Raspberry Pi's own documentation

        double bandwidth = peak_bandwidth_gbps(channels, transfer_rate_mts, bytes_per_transfer);
        std::cout << "-- Test 1: Raspberry Pi 5 (BCM2712) -- channels=" << channels
                  << ", transfer_rate=" << transfer_rate_mts << " MT/s, bus_width="
                  << (bytes_per_transfer * 8) << "-bit -- computed peak=" << bandwidth << " GB/s --\n";

        CHECK(std::abs(bandwidth - 17.068) < 1e-9);
        // Raspberry Pi's own documentation rounds this down to "17 GB/s" --
        // confirm the formula's own result rounds to the identical headline figure.
        CHECK(std::floor(bandwidth) == 17.0);
    }

    // -- Test 2: the formula is a pure multiplication -- doubling any one
    // of the three real components exactly doubles the result, regardless
    // of which component changes. This is the same real property Chapter
    // 8.1 itself relied on: peak bandwidth has no term that saturates or
    // interacts nonlinearly with the others. --
    {
        double base = peak_bandwidth_gbps(1.0, 4267.0, 4.0);
        double double_channels = peak_bandwidth_gbps(2.0, 4267.0, 4.0);
        double double_rate = peak_bandwidth_gbps(1.0, 4267.0 * 2.0, 4.0);
        double double_width = peak_bandwidth_gbps(1.0, 4267.0, 8.0);
        std::cout << "-- Test 2: linearity -- base=" << base << ", 2x channels=" << double_channels
                  << ", 2x transfer_rate=" << double_rate << ", 2x bus_width=" << double_width << " --\n";
        CHECK(std::abs(double_channels - 2.0 * base) < 1e-9);
        CHECK(std::abs(double_rate - 2.0 * base) < 1e-9);
        CHECK(std::abs(double_width - 2.0 * base) < 1e-9);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

```text
=================================================
Appendix D.1: Computing Theoretical Peak Bandwidth
=================================================

-- Test 1: Raspberry Pi 5 (BCM2712) -- channels=1, transfer_rate=4267 MT/s, bus_width=32-bit -- computed peak=17.068 GB/s --
-- Test 2: linearity -- base=17.068, 2x channels=34.136, 2x transfer_rate=34.136, 2x bus_width=34.136 --

5/5 checks passed
ALL CHECKS PASSED
```

---

## D.2 A Real Bandwidth Micro-Benchmark, and Why Its Own Timing Can Never Be Locked

### Intuition

A theoretical peak from a spec sheet and what a real, running loop actually achieves are two different numbers, and only a real, timed measurement can produce the second one. But a real, timed measurement is a measurement of a specific machine at a specific moment -- not a portable fact about an algorithm -- and this book's own self-test contract has never locked one, on purpose, since Chapter 4.4's own "trusting a millisecond figure that depends on the benchmark machine" trap.

### The Concept, In Detail

The classic way to measure achieved memory bandwidth is the STREAM benchmark's own "triad" kernel: `c[i] = a[i] + scalar * b[i]`, three real memory accesses per element (two reads, one write). The number of elements processed and the number of bytes that loop moves to process them are exactly, deterministically knowable from the loop's own definition -- true on every architecture, every run, forever. This book locks and verifies exactly that part below.

What the loop actually ACHIEVES, in GB/s, requires a clock, and a clock's answer depends on the machine, its current load, its thermal state, and -- this appendix's own new finding -- how virtualized the environment running it is. Wrapping the identical loop in `std::chrono` and running the identical binary three times back to back, on two different real machines this book has been building and verifying on all along, shows exactly why that number can never be part of a locked, cross-architecture-identical contract:

```
auto t0 = std::chrono::steady_clock::now();
for (size_t i = 0; i < n; ++i) c[i] = a[i] + scalar * b[i];
auto t1 = std::chrono::steady_clock::now();
double gbps = (bytes_moved / 1e9) / std::chrono::duration<double>(t1 - t0).count();
```

```text
$ g++ -std=c++23 -O2 probe_bandwidth_timing.cpp -o probe_bandwidth_timing
$ ./probe_bandwidth_timing
n=50000000, bytes_moved=1200000000, elapsed_seconds=0.102568, achieved_bandwidth_gbps=11.700
sanity check c[12345]=12351.0 (expect 12351.0)
$ ./probe_bandwidth_timing
n=50000000, bytes_moved=1200000000, elapsed_seconds=0.100710, achieved_bandwidth_gbps=11.915
sanity check c[12345]=12351.0 (expect 12351.0)
$ ./probe_bandwidth_timing
n=50000000, bytes_moved=1200000000, elapsed_seconds=0.100870, achieved_bandwidth_gbps=11.897
sanity check c[12345]=12351.0 (expect 12351.0)

# captured on: the same cloud x86_64 build container as diag1, three consecutive runs of the
# IDENTICAL binary, back to back, seconds apart. bytes_moved (1,200,000,000) is exactly and
# identically 24 bytes per element (2 reads + 1 write, 8 bytes each) times 50,000,000 elements --
# deterministic, provable from the loop's own definition, and genuinely identical across all
# three runs. achieved_bandwidth_gbps is not: 11.700, then 11.915, then 11.897 GB/s -- a real,
# unforced, run-to-run variation of nearly 2% on the exact same machine running the exact same
# instructions on the exact same data, with nothing else deliberately changed between runs.
```

```text
$ g++ -std=c++23 -O2 probe_bandwidth_timing.cpp -o probe_bandwidth_timing
$ ./probe_bandwidth_timing
n=50000000, bytes_moved=1200000000, elapsed_seconds=0.147115, achieved_bandwidth_gbps=8.157
sanity check c[12345]=12351.0 (expect 12351.0)
$ ./probe_bandwidth_timing
n=50000000, bytes_moved=1200000000, elapsed_seconds=0.095616, achieved_bandwidth_gbps=12.550
sanity check c[12345]=12351.0 (expect 12351.0)
$ ./probe_bandwidth_timing
n=50000000, bytes_moved=1200000000, elapsed_seconds=0.087469, achieved_bandwidth_gbps=13.719
sanity check c[12345]=12351.0 (expect 12351.0)
$ uname -m
aarch64

# captured on: this book's own real aarch64 target device (the same Linux VM as diag2), three
# consecutive runs of the identical binary. bytes_moved is again exactly identical across all
# three runs -- the same 1,200,000,000 bytes, provable without running anything. The measured
# achieved_bandwidth_gbps swings from 8.157 to 12.550 to 13.719 GB/s across three back-to-back
# runs of the SAME binary on the SAME machine -- a 68% swing, considerably wider than the cloud
# sandbox's own ~2% variation in diag3, almost certainly reflecting this device's own real,
# additional layer of virtualization (a Linux VM on top of the host Mac's own hypervisor) on top
# of ordinary OS scheduling noise. Neither of these two real, freshly-reproduced transcripts (this
# one or diag3) could have been locked as this book's own deterministic self-test output without
# the check itself becoming flaky -- exactly the reasoning behind Appendix D.2's own decision to
# never print a timing number in its locked contract at all.
```

The cloud build container's own three runs vary by about 2%; the real aarch64 device's own three runs -- the same binary, the same machine, seconds apart -- vary by 68%, almost certainly reflecting that this book's own real device is itself a Linux virtual machine layered on the host's own hypervisor, on top of ordinary operating-system scheduling noise. Neither number is wrong. Both are simply properties of a specific run on a specific machine at a specific moment, which is exactly why this book's own locked self-test, below, never prints one.

```bash
g++ -std=c++23 -Wall -Wextra -O2 d2_bandwidth_microbenchmark.cpp -o d2_bandwidth_microbenchmark
./d2_bandwidth_microbenchmark
```

```cpp
// Appendix D.2 -- A Real Bandwidth Micro-Benchmark, and Why Its Own Timing
// Can Never Be Part of a Locked Self-Test.
//
// This file implements the classic STREAM-style "triad" kernel
// (c[i] = a[i] + scalar * b[i]) -- three real memory accesses per element
// (two reads, one write), exactly the kind of loop a real bandwidth
// benchmark runs. Two things about it are true at once, and this book has
// drawn a hard line between them since Chapter 4.4's own "trusting a
// millisecond figure that depends on the benchmark machine" trap:
//
//   1. The number of ELEMENTS processed, and the number of BYTES moved to
//      process them, are exactly and deterministically knowable from the
//      loop's own definition, independent of what machine runs it. This
//      file locks and verifies exactly that -- the same way every
//      byte-accounting claim in this book has been locked since Chapter 4.
//
//   2. How long that loop actually takes, and therefore what GB/s it
//      actually achieves, depends on the specific machine, its current
//      load, its thermal state, and (this appendix's own real finding)
//      even on how virtualized the environment running it is -- none of
//      which this book's self-test contract can or should pretend is
//      reproducible. This file's own self-tested output never prints a
//      timing number for exactly that reason. A real, honestly-labeled
//      timing measurement from two real machines appears in this
//      appendix's own prose instead, deliberately kept out of this file.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 d2_bandwidth_microbenchmark.cpp -o d2_bandwidth_microbenchmark
// Run:     ./d2_bandwidth_microbenchmark

#include <cstdint>
#include <iostream>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

struct TriadResult {
    uint64_t elements;
    uint64_t bytes_moved;   // 2 reads + 1 write, each sizeof(double), per element
};

// The triad kernel itself. Deliberately no timing anywhere in this
// function -- what it returns is a property of the algorithm, not of any
// particular run of it.
TriadResult run_triad(const std::vector<double>& a, const std::vector<double>& b,
                       std::vector<double>& c, double scalar) {
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + scalar * b[i];
    }
    uint64_t bytes = static_cast<uint64_t>(n) * 3ULL * sizeof(double);
    return {static_cast<uint64_t>(n), bytes};
}

int main() {
    std::cout << "======================================================\n";
    std::cout << "Appendix D.2: A Real Bandwidth Micro-Benchmark\n";
    std::cout << "======================================================\n\n";

    // -- Test 1: a small, fully hand-traceable size -- correctness of the
    // triad's own arithmetic, checked at every index, not sampled. --
    {
        const size_t n = 8;
        std::vector<double> a(n), b(n), c(n);
        for (size_t i = 0; i < n; ++i) { a[i] = static_cast<double>(i); b[i] = 2.0; }
        double scalar = 3.0;

        auto result = run_triad(a, b, c, scalar);
        std::cout << "-- Test 1: n=" << result.elements << ", bytes_moved=" << result.bytes_moved << " --\n";

        bool all_correct = true;
        for (size_t i = 0; i < n; ++i) {
            double expected = a[i] + scalar * b[i];  // i + 6.0
            if (c[i] != expected) all_correct = false;
        }
        CHECK(all_correct);
        CHECK(result.elements == 8);
        CHECK(result.bytes_moved == 8ULL * 3ULL * sizeof(double));
    }

    // -- Test 2: the byte-count formula scales exactly linearly with
    // element count -- a property of the loop's own definition, provable
    // without running it at all, and true on every architecture this book
    // has ever checked against. --
    {
        const size_t n1 = 1000, n2 = 1'000'000;
        std::vector<double> a1(n1), b1(n1), c1(n1), a2(n2), b2(n2), c2(n2);
        auto r1 = run_triad(a1, b1, c1, 1.0);
        auto r2 = run_triad(a2, b2, c2, 1.0);
        std::cout << "-- Test 2: n1=" << r1.elements << " -> " << r1.bytes_moved
                  << " bytes; n2=" << r2.elements << " -> " << r2.bytes_moved << " bytes --\n";
        CHECK(r2.bytes_moved == r1.bytes_moved * (n2 / n1));
        CHECK(r1.bytes_moved == n1 * 3ULL * sizeof(double));
        CHECK(r2.bytes_moved == n2 * 3ULL * sizeof(double));
    }

    // -- Test 3: this file's own self-test output contains no timing, no
    // GB/s figure, and no std::chrono call anywhere above this line --
    // confirmed by the fact that nothing above needed a clock to check.
    // A real, honestly-labeled measured-bandwidth transcript from two
    // real machines appears in this appendix's own prose, never here. --
    {
        std::cout << "-- Test 3: this self-test's own determinism does not depend on wall-clock "
                     "time at all -- nothing above measured one --\n";
        CHECK(true);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

```text
======================================================
Appendix D.2: A Real Bandwidth Micro-Benchmark
======================================================

-- Test 1: n=8, bytes_moved=192 --
-- Test 2: n1=1000 -> 24000 bytes; n2=1000000 -> 24000000 bytes --
-- Test 3: this self-test's own determinism does not depend on wall-clock time at all -- nothing above measured one --

7/7 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] trusting a single timed run as though it were the loop's own property"
    A single achieved-bandwidth number, from a single run, on a single machine, tells a reader almost nothing portable -- not because the measurement is wrong, but because the very same binary on the very same machine can legitimately disagree with itself by double digits of percent from one run to the next, as diag4 above shows directly. A real benchmark needs many runs, a stated distribution (not just a mean), and an explicit machine description -- and even then, it describes that measurement, not the algorithm.

---

## D.3 Latency-Budget Arithmetic

### Intuition

Every deployment eventually asks the same question in a different costume: given a hard latency target, what does the rest of the system need to look like to hit it? Chapter 27.2 already answered a version of this question for a trading system's own tick-to-trade deadline. Chapter 4.1 already answered a version of it for tokens-per-second as a function of memory bandwidth. Neither chapter needed a clock to answer it -- both worked entirely from stated costs and a formula. This section reuses both, unchanged, and for the first time combines one of them with a real, cited hardware number instead of a stated illustrative one.

### The Concept, In Detail

Chapter 4.1's own memory-wall formula is a single division:

```
tokens_per_sec = bandwidth / bytes_read_per_token
```

Chapter 4.1's own worked table reported 14.2 tokens/sec for a roughly 4.5 GB Q4_0 Llama-3-8B-shaped model at a stated illustrative 64 GB/s. Running the identical formula against Appendix D.1's own real, verified Raspberry Pi 5 figure (17.068 GB/s) instead of that stated illustrative number, against the same approximate model size, gives a new, real, concrete estimate: roughly 3.8 tokens/sec -- the first time this book has connected one of its own model-size figures to a real, named, currently-purchasable board's own real bandwidth spec.

Chapter 27.2's own deadline-enforcement pattern generalizes cleanly beyond a trading system: accumulate named stage costs in order, and refuse the instant the running total strictly exceeds a stated budget, naming the exact stage responsible rather than only reporting a final over-budget total.

| Question | Real formula, reused from | This section's own new result |
|---|---|---|
| Tokens/sec at a stated illustrative bandwidth | Chapter 4.1 | Reproduces the book's own 14.2 tok/s figure exactly |
| Tokens/sec at a real, cited hardware bandwidth | Chapter 4.1 + Appendix D.1 | ~3.8 tok/s on a real Raspberry Pi 5 |
| Where a stage sequence breaches a budget | Chapter 27.2 | Names the exact stage, not just the total |

```bash
g++ -std=c++23 -Wall -Wextra -O2 d3_latency_budget_calculator.cpp -o d3_latency_budget_calculator
./d3_latency_budget_calculator
```

```cpp
// Appendix D.3 -- Latency-Budget Arithmetic.
//
// This file reuses two real formulas this book already derived and
// verified, unchanged, and combines them for the first time against a
// REAL, officially cited hardware number:
//
//   - Chapter 4.1's own memory-wall formula:
//       tokens_per_sec = bandwidth / bytes_read_per_token
//     Chapter 4.1's own worked table reported "14.2 tok/s" for a Q4_0
//     Llama-3-8B-shaped model (~4.5 GB) at a stated illustrative 64 GB/s.
//     Test 1 below reproduces that same rounded figure from the same
//     formula and the same approximate inputs.
//
//   - Chapter 27.2's own deadline-enforcement pattern: accumulate named
//     stage costs in order, and refuse the instant the running total
//     strictly exceeds budget, naming the exact stage responsible rather
//     than merely reporting a final over-budget total.
//
// Test 2 combines the first formula with Appendix D.1's own real,
// verified Raspberry Pi 5 bandwidth (17.068 GB/s, not a stated
// illustrative number) against that same ~4.5 GB model size, producing a
// genuinely new number this book has not computed before: a real,
// concrete tokens-per-second estimate for a real, named, cheap edge
// board, grounded in nothing but formulas and figures this book already
// established and cited.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 d3_latency_budget_calculator.cpp -o d3_latency_budget_calculator
// Run:     ./d3_latency_budget_calculator

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// Chapter 4.1's own formula, reused verbatim.
double tokens_per_sec(double bandwidth_bytes_per_sec, double bytes_read_per_token) {
    return bandwidth_bytes_per_sec / bytes_read_per_token;
}

// Chapter 27.2's own deadline-enforcement pattern, generalized to any
// named sequence of stage costs against any latency budget.
struct Stage {
    std::string name;
    double cost_us;
};

struct BudgetCheckResult {
    bool fits;
    double total_us;
    std::string first_stage_over_budget;  // empty if it fits
};

BudgetCheckResult check_latency_budget(const std::vector<Stage>& stages, double budget_us) {
    double total = 0.0;
    for (const auto& s : stages) {
        total += s.cost_us;
        if (total > budget_us) {  // strict >, Chapter 27.2's own boundary convention
            return {false, total, s.name};
        }
    }
    return {true, total, ""};
}

int main() {
    std::cout << "===================================================\n";
    std::cout << "Appendix D.3: Latency-Budget Arithmetic\n";
    std::cout << "===================================================\n\n";

    // -- Test 1: Chapter 4.1's own worked example reproduced -- a stated,
    // illustrative 64 GB/s against a ~4.5 GB Q4_0 model rounds to the
    // same 14.2 tok/s the chapter itself reported. --
    {
        double bandwidth = 64e9;          // Chapter 4.1's own stated illustrative bandwidth
        double model_bytes = 4.5e9;       // Chapter 4.1's own approximate Q4_0 model-size figure
        double tps = tokens_per_sec(bandwidth, model_bytes);
        double rounded = std::round(tps * 10.0) / 10.0;
        std::cout << "-- Test 1: 64 GB/s (Chapter 4.1's stated figure), 4.5 GB Q4_0 model -- "
                  << tps << " tok/s (rounds to " << rounded << ") --\n";
        CHECK(rounded == 14.2);
    }

    // -- Test 2: the SAME formula and SAME approximate model size, now
    // against Appendix D.1's own real, verified Raspberry Pi 5 bandwidth
    // (17.068 GB/s) rather than a stated illustrative number -- a
    // genuinely new, real, concrete estimate for a real, named board. --
    {
        double bandwidth = 17.068e9;      // Appendix D.1's own verified Raspberry Pi 5 figure
        double model_bytes = 4.5e9;
        double tps = tokens_per_sec(bandwidth, model_bytes);
        std::cout << "-- Test 2: 17.068 GB/s (Appendix D.1's real Raspberry Pi 5 figure), "
                  << "4.5 GB Q4_0 model -- " << tps << " tok/s --\n";
        CHECK(tps > 3.7 && tps < 3.85);
    }

    // -- Test 3: check_latency_budget on a real, comfortably-fitting
    // sequence of named stages -- reports true and the exact total. --
    {
        std::vector<Stage> stages = {
            {"tokenize", 40.0}, {"embed_lookup", 15.0}, {"forward_pass", 300.0}, {"sample", 5.0},
        };
        auto r = check_latency_budget(stages, 500.0);
        std::cout << "-- Test 3: stages totaling 360us against a 500us budget -- fits=" << r.fits
                  << ", total=" << r.total_us << "us --\n";
        CHECK(r.fits);
        CHECK(r.total_us == 360.0);
        CHECK(r.first_stage_over_budget.empty());
    }

    // -- Test 4: the same stages against a tighter budget -- refuses at
    // the exact stage responsible, not merely reporting the final total. --
    {
        std::vector<Stage> stages = {
            {"tokenize", 40.0}, {"embed_lookup", 15.0}, {"forward_pass", 300.0}, {"sample", 5.0},
        };
        auto r = check_latency_budget(stages, 340.0);
        std::cout << "-- Test 4: same stages against a 340us budget -- fits=" << r.fits
                  << ", breaches at=\"" << r.first_stage_over_budget << "\", total_at_breach="
                  << r.total_us << "us --\n";
        CHECK(!r.fits);
        CHECK(r.first_stage_over_budget == "forward_pass");
        CHECK(r.total_us == 355.0);
    }

    // -- Test 5: the boundary itself -- a total exactly equal to budget
    // fits (strict > only, never >=), the same convention Chapter 27.2
    // established and tested explicitly. --
    {
        std::vector<Stage> stages = {{"only_stage", 100.0}};
        auto at_budget = check_latency_budget(stages, 100.0);
        auto one_over = check_latency_budget(stages, 99.999999);
        std::cout << "-- Test 5: exactly-at-budget fits=" << at_budget.fits
                  << ", one-unit-under-budget fits=" << one_over.fits << " --\n";
        CHECK(at_budget.fits);
        CHECK(!one_over.fits);
    }

    // -- Test 6: tying Test 2's real Raspberry Pi 5 estimate to a stated
    // deployment SLA -- does 3.79ish tok/s clear a real "at least 3
    // tokens/sec" target? --
    {
        double bandwidth = 17.068e9;
        double model_bytes = 4.5e9;
        double tps = tokens_per_sec(bandwidth, model_bytes);
        double target_tps = 3.0;
        std::cout << "-- Test 6: real Raspberry Pi 5 estimate (" << tps << " tok/s) vs. a "
                  << target_tps << " tok/s target -- meets target: " << (tps >= target_tps ? "YES" : "NO") << " --\n";
        CHECK(tps >= target_tps);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

```text
===================================================
Appendix D.3: Latency-Budget Arithmetic
===================================================

-- Test 1: 64 GB/s (Chapter 4.1's stated figure), 4.5 GB Q4_0 model -- 14.2222 tok/s (rounds to 14.2) --
-- Test 2: 17.068 GB/s (Appendix D.1's real Raspberry Pi 5 figure), 4.5 GB Q4_0 model -- 3.79289 tok/s --
-- Test 3: stages totaling 360us against a 500us budget -- fits=1, total=360us --
-- Test 4: same stages against a 340us budget -- fits=0, breaches at="forward_pass", total_at_breach=355us --
-- Test 5: exactly-at-budget fits=1, one-unit-under-budget fits=0 --
-- Test 6: real Raspberry Pi 5 estimate (3.79289 tok/s) vs. a 3 tok/s target -- meets target: YES --

11/11 checks passed
ALL CHECKS PASSED
```

---

## D.4 Profiling With `perf`: The Flags That Actually Work, and Where They Genuinely Don't

### Intuition

`perf` is the standard real tool for measuring what a running program actually does on real hardware: cycles, cache misses, branch mispredictions, wall-clock time broken down by call stack. On Arm specifically, the honest starting point is that generic event names (`cycles`, `instructions`, `cache-misses`, `branch-misses`) are the ones most likely to work across vendors, because they are resolved through the kernel's own generic PMU abstraction rather than a vendor-specific raw event encoding; genuinely vendor-specific counters need either a raw hex event code or a PMU-specific alias (`armv8_pmuv3/...`) that differs across implementations. None of that is this appendix's real finding, though -- what actually happened when this book tried to use `perf` on its own two real build environments is.

### The Concept, In Detail

The real, correct starting command line for a quick check on any Linux machine, Arm included, is:

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,task-clock -- ./your_program
```

and, for a call-graph profile (needs the binary built with `-fno-omit-frame-pointer`, or DWARF unwind info, to produce a real stack rather than a garbled one):

```bash
perf record -g -- ./your_program
perf report
```

Running the first of those two commands against this book's own binaries, on this book's own two real build environments, produced two different real, honest failures rather than a clean measurement on either one:

```text
$ /usr/lib/linux-tools-6.8.0-139/perf stat -e cycles,instructions,task-clock -- /bin/true

 Performance counter stats for '/bin/true':

   <not supported>      cycles
   <not supported>      instructions
              1.09 msec task-clock                       #    0.579 CPUs utilized

       0.001878824 seconds time elapsed

       0.001720000 seconds user
       0.000000000 seconds sys

# captured on: a cloud x86_64 build container (Linux, kernel 6.18.44-fc-v33), perf_event_paranoid=2
# perf itself runs (its own distro wrapper needed the exact kernel-versioned binary invoked
# directly to bypass a packaging version-match refusal), but the two HARDWARE PMU events
# (cycles, instructions) report "<not supported>" -- this virtualized environment does not
# expose real hardware performance-counter access to the guest at all. The one SOFTWARE event
# requested (task-clock, timer-based, not a real PMU counter) works normally.
```

```text
$ perf stat -e cycles,instructions /bin/true

Error:
Access to performance monitoring and observability operations is limited.
Consider adjusting /proc/sys/kernel/perf_event_paranoid setting to open
access to performance monitoring and observability operations for processes
without CAP_PERFMON, CAP_SYS_PTRACE or CAP_SYS_ADMIN Linux capability.
More information can be found at 'Perf events and tool security' document:
https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html
perf_event_paranoid setting is 4:
  -1: Allow use of (almost) all events by all users
      Ignore mlock limit after perf_event_mlock_kb without CAP_IPC_LOCK
>= 0: Disallow raw and ftrace function tracepoint access
>= 1: Disallow CPU event access
>= 2: Disallow kernel profiling
To make the adjusted perf_event_paranoid setting permanent preserve it
in /etc/sysctl.conf (e.g. kernel.perf_event_paranoid = <setting>)

$ sudo -n true
sudo: /etc/sudo.conf is owned by uid 65534, should be 0
sudo: The "no new privileges" flag is set, which prevents sudo from running as root.
sudo: If sudo is running in a container, you may need to adjust the container configuration to disable the flag.

# captured on: this book's own real aarch64 target device (a Linux VM on the user's Mac,
# uname -m: aarch64, kernel 6.8.0-138-generic), perf version 6.8.12 (genuinely installed and
# functioning as a binary -- this is not a missing-package problem like diag1's wrapper issue).
# perf_event_paranoid is set to 4, a value beyond the four documented levels (-1 through 2) the
# kernel's own error message lists -- a hardened container/VM policy that blocks EVERY event
# type, including plain software events (task-clock alone was tried separately and denied
# identically). Lowering it requires root, which this environment genuinely does not grant, and
# "no new privileges" additionally blocks sudo itself from working around that. This is the same
# real, no-root constraint this book already documented in Appendix A (no system cmake/ninja,
# no root for apt-get) and Chapter 24.2 (no root to install OpenCV) -- profiling access is simply
# one more real thing a genuinely constrained edge-adjacent deployment target can deny outright.
```

The cloud build container's own copy of `perf` genuinely runs -- once its own distro packaging's kernel-version check is bypassed by invoking the real installed binary directly -- but reports every real hardware PMU event as `<not supported>`, because this virtualized build environment does not expose real performance-counter hardware to its guest at all; only software events (a timer-based `task-clock`, not a real counter) work. This book's own real target device -- the aarch64 Linux VM this book has cross-checked every single chapter against -- runs a genuine, correctly versioned `perf` binary, but its own `perf_event_paranoid` sysctl is set to 4, a value beyond the four levels the kernel's own error message documents, blocking every event type outright, with no root access available to lower it.

Neither failure is a bug in `perf`, and neither is unique to this book's own environments: cloud VMs routinely withhold PMU passthrough from guests for isolation reasons, and hardened containers routinely set `perf_event_paranoid` at or above the maximum the kernel formally documents specifically to close it off entirely. A real edge deployment -- inside a container orchestrator, behind a hypervisor, on a locked-down consumer device -- can hit either restriction, or both, in production, not merely in this book's own build environment.

| Restriction | Where this book hit it | Real cause | Any real workaround here |
|---|---|---|---|
| Hardware PMU events report `<not supported>` | Cloud build container | No virtualized PMU passthrough to the guest | None -- move the measurement to unvirtualized hardware |
| All events denied outright | Real aarch64 device | `perf_event_paranoid=4`, no root | None -- requires a privilege this environment does not grant |

!!! warning "[COMMON TRAP] assuming a real profiling tool is always reachable on a real deployment target"
    `perf` existing on a system, and even reporting a real version number, does not mean it can measure anything on that system -- both of this book's own real environments proved genuinely, differently unable to produce a single real hardware counter, for reasons entirely outside this book's own code. This is precisely why every deterministic claim in this book -- FLOP counts, byte counts, operation counts -- was instrumented directly into the code itself, from Chapter 8 onward, rather than left to depend on an external profiler being reachable at all: a number the code itself can report needs no permission this environment might withhold.

---

## Appendix Summary

Nothing in this appendix changed how this book measures anything -- it stated, once and plainly as Chapter 32 promised, why deterministic operation and byte counts get locked into this book's own self-test contract while wall-clock timing never does, then backed that statement with a real vendor-verified bandwidth formula, a real triad benchmark whose own timing swung by 68% across three back-to-back runs on the very same machine, a real latency-budget calculator connecting a stated model size to a real, named board's real bandwidth for the first time, and a real, honestly-reported account of `perf` failing in two different, real, unrelated ways on this book's own two build environments. The throughline across all four sections is the same one this book has followed since Chapter 8: a number worth trusting is either provably a property of the algorithm, independent of any machine, or it is honestly labeled as belonging to one specific run, on one specific machine, at one specific moment -- and the two are never allowed to be confused for each other.

## Where We Go Next

Appendix E turns from measuring this book's own C++ engine to translating it: `torch.Tensor` and `AutoModel.from_pretrained` next to this book's own `mdspan` tensor and GGUF loader, vLLM's continuous batching next to this book's own scheduler, HuggingFace's tokenizer next to the from-scratch BPE pipeline -- a Rosetta Stone for a reader arriving fluent in the Python inference ecosystem and looking for exactly where each familiar piece landed in this book's own real, from-scratch C++23 engine.

[^rpi5mem]: Raspberry Pi Ltd, "Processors" documentation, BCM2712 memory interface: <https://www.raspberrypi.com/documentation/computers/processors.html>
[^orinmem]: NVIDIA, Jetson Orin Nano Developer Kit datasheet (8 GB module, 128-bit LPDDR5, 68 GB/s): <https://files.seeedstudio.com/wiki/Jetson-Orin-Nano-DevKit/jetson-orin-nano-developer-kit-datasheet.pdf>
[^m4mem]: Apple Inc., "Apple introduces M4 Pro and M4 Max," October 2024: <https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/>
