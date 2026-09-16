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
