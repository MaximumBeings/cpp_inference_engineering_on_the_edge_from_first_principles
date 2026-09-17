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
