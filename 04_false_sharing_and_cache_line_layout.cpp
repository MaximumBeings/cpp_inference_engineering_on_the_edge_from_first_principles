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
