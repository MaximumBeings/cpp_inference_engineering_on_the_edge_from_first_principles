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
