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
