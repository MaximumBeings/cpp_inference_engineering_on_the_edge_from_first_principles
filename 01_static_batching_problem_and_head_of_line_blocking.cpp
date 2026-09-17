// Chapter 31.1 -- The static-batching problem: a real, from-scratch discrete-
// step simulator for the naive batching scheme real early inference servers
// actually used -- fixed-size batches that cannot start until the PREVIOUS
// batch has fully retired, and cannot retire early even if some of their own
// members finish long before the others. This section quantifies, with real
// hand-traceable numbers, exactly how much real GPU-slot time this wastes on
// padding, and exactly how badly it delays a short request unlucky enough to
// arrive just after a long batch has already started (head-of-line blocking).
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_static_batching_problem_and_head_of_line_blocking.cpp -o 01_static_batching_problem_and_head_of_line_blocking
// Run:     ./01_static_batching_problem_and_head_of_line_blocking

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: a real request's own shape, in this section's own simplified but
// real discrete-step model: the number of real steps it takes to produce
// the final token of its own response is stated directly as needed_steps
// (abstracting prefill and decode into one real total for this section's
// own purposes -- Section 31.2 separates them explicitly).
// =======================================================================
struct Request {
    int id = 0;
    int arrival_step = 0;
    int needed_steps = 0;
};

struct StaticBatchResult {
    int id = 0;
    int start_step = 0;
    int completion_step = 0;
    int wasted_padding_steps = 0;
    int queue_wait_steps = 0;
};

// =======================================================================
// PART 2: the real static-batching scheduler. Requests are grouped, in real
// arrival order, into fixed-size chunks of exactly batch_size (the LAST
// chunk may be smaller if the request count does not divide evenly). A
// chunk's own batch cannot start until every one of its own members has
// arrived AND the previous batch has fully retired -- and once started, the
// WHOLE batch occupies its own slots, doing nothing useful on any slot whose
// own request already finished, until every member's own needed_steps has
// been satisfied (its own real makespan).
// =======================================================================
std::vector<StaticBatchResult> simulate_static_batching(std::vector<Request> requests, int batch_size) {
    std::stable_sort(requests.begin(), requests.end(),
                      [](const Request& a, const Request& b) { return a.arrival_step < b.arrival_step; });

    std::vector<StaticBatchResult> results;
    int t = 0;
    for (std::size_t i = 0; i < requests.size(); i += static_cast<std::size_t>(batch_size)) {
        std::size_t end = std::min(requests.size(), i + static_cast<std::size_t>(batch_size));

        int latest_arrival = 0;
        int makespan = 0;
        for (std::size_t j = i; j < end; ++j) {
            latest_arrival = std::max(latest_arrival, requests[j].arrival_step);
            makespan = std::max(makespan, requests[j].needed_steps);
        }
        int start = std::max(t, latest_arrival);
        int completion = start + makespan;

        for (std::size_t j = i; j < end; ++j) {
            StaticBatchResult r;
            r.id = requests[j].id;
            r.start_step = start;
            r.completion_step = completion;
            r.wasted_padding_steps = makespan - requests[j].needed_steps;
            r.queue_wait_steps = start - requests[j].arrival_step;
            results.push_back(r);
        }
        t = completion;
    }
    return results;
}

// Real slot-utilization accounting: how much of the real GPU-slot time this
// scheduler actually spent computing a real token, versus how much of it
// was spent holding an already-finished (or not-yet-relevant) slot idle
// through padding, purely because the batch itself had not yet retired.
double static_batching_utilization(const std::vector<Request>& requests, int batch_size) {
    std::vector<Request> sorted_requests = requests;
    std::stable_sort(sorted_requests.begin(), sorted_requests.end(),
                      [](const Request& a, const Request& b) { return a.arrival_step < b.arrival_step; });

    double useful_steps = 0.0;
    for (const Request& r : sorted_requests) useful_steps += static_cast<double>(r.needed_steps);

    double total_slot_steps = 0.0;
    int t = 0;
    for (std::size_t i = 0; i < sorted_requests.size(); i += static_cast<std::size_t>(batch_size)) {
        std::size_t end = std::min(sorted_requests.size(), i + static_cast<std::size_t>(batch_size));
        int chunk_size = static_cast<int>(end - i);
        int latest_arrival = 0;
        int makespan = 0;
        for (std::size_t j = i; j < end; ++j) {
            latest_arrival = std::max(latest_arrival, sorted_requests[j].arrival_step);
            makespan = std::max(makespan, sorted_requests[j].needed_steps);
        }
        int start = std::max(t, latest_arrival);
        total_slot_steps += static_cast<double>(chunk_size) * static_cast<double>(makespan);
        t = start + makespan;
    }
    return useful_steps / total_slot_steps;
}

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

const StaticBatchResult& find_result(const std::vector<StaticBatchResult>& results, int id) {
    for (const auto& r : results) if (r.id == id) return r;
    static StaticBatchResult empty;
    return empty;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 31.1: The Static-Batching Problem and Head-of-Line Blocking\n";
    std::cout << "========================================================\n";

    // A tiny, fully hand-traceable real workload: two requests arrive
    // together at step 0 (one long, one short), and two more arrive
    // together at step 1 (one very short, one medium), with a real
    // batch_size of 2.
    std::vector<Request> requests = {
        {0, 0, 10},
        {1, 0, 2},
        {2, 1, 1},
        {3, 1, 3},
    };
    const int batch_size = 2;

    std::cout << "\n-- Test 1: the real static-batching schedule for this tiny 4-request workload matches an "
                 "exact hand trace: batch A = {0, 1} starts at step 0 with a real makespan of 10 (bounded by "
                 "request 0's own 10 needed steps), retiring at step 10; batch B = {2, 3} cannot start until "
                 "batch A retires, at step 10, despite both of its own members having arrived at step 1 --\n";
    {
        auto results = simulate_static_batching(requests, batch_size);
        CHECK(results.size() == 4);

        const auto& r0 = find_result(results, 0);
        CHECK(r0.start_step == 0);
        CHECK(r0.completion_step == 10);
        CHECK(r0.wasted_padding_steps == 0);
        CHECK(r0.queue_wait_steps == 0);

        const auto& r1 = find_result(results, 1);
        CHECK(r1.start_step == 0);
        CHECK(r1.completion_step == 10);
        CHECK(r1.wasted_padding_steps == 8);
        CHECK(r1.queue_wait_steps == 0);

        const auto& r2 = find_result(results, 2);
        CHECK(r2.start_step == 10);
        CHECK(r2.completion_step == 13);
        CHECK(r2.wasted_padding_steps == 2);
        CHECK(r2.queue_wait_steps == 9);

        const auto& r3 = find_result(results, 3);
        CHECK(r3.start_step == 10);
        CHECK(r3.completion_step == 13);
        CHECK(r3.wasted_padding_steps == 0);
        CHECK(r3.queue_wait_steps == 9);

        std::cout << "  request 0 (10 steps needed): starts 0, completes 10, 0 wasted padding steps -- it "
                     "is the one that sets batch A's own makespan\n";
        std::cout << "  request 1 (2 steps needed): starts 0, completes 10, 8 wasted padding steps -- its "
                     "own slot sits idle for 8 real steps after it finishes, because batch A cannot retire "
                     "until request 0 also finishes\n";
        std::cout << "  request 2 (1 step needed, arrived at step 1): starts 10, completes 13, queued for 9 "
                     "real steps before batch B even begins, despite needing only 1 step of real work\n";
        std::cout << "  request 3 (3 steps needed, arrived at step 1): starts 10, completes 13, 0 wasted "
                     "padding within batch B, but also queued for the identical 9 real steps as request 2\n";
    }

    std::cout << "\n-- Test 2: this workload's own real slot utilization -- useful steps actually computed "
                 "divided by total real slot-steps spent, including padding -- matches an exact hand "
                 "computation: 16 useful steps out of 26 total slot-steps --\n";
    {
        double util = static_batching_utilization(requests, batch_size);
        CHECK(near(util, 16.0 / 26.0));
        std::cout << "  useful steps = 10 + 2 + 1 + 3 = 16; total slot-steps = (2 slots * 10-step makespan) "
                     "+ (2 slots * 3-step makespan) = 20 + 6 = 26; real utilization = 16 / 26 = "
                  << util << "\n";
    }

    std::cout << "\n-- Test 3: request 2's own real head-of-line blocking delay, computed as the difference "
                 "between its own actual completion step and the completion step it WOULD have gotten had "
                 "it been served alone, immediately upon arrival, is a real, substantial 11 steps -- despite "
                 "needing only 1 step of actual work --\n";
    {
        auto results = simulate_static_batching(requests, batch_size);
        const auto& r2 = find_result(results, 2);
        int ideal_completion = 2 /* arrival_step 1 + needed_steps 1 */;
        int blocking_delay = r2.completion_step - ideal_completion;
        CHECK(ideal_completion == 2);
        CHECK(blocking_delay == 11);
        std::cout << "  request 2 arrives at step 1 needing only 1 step -- served alone it would complete "
                     "at step 2; static batching's own head-of-line blocking instead delays it all the way "
                     "to step 13, a real, substantial 11-step penalty caused entirely by having arrived "
                     "just after a long-running batch had already started\n";
    }

    std::cout << "\n-- Test 4: at the degenerate real batch_size of 1, static batching's own scheduling "
                 "collapses to strict real FIFO with zero padding waste -- utilization is exactly 1.0, "
                 "confirming this section's own general formulas reduce correctly to the trivial real case "
                 "rather than being tuned only to the batch_size-2 example above --\n";
    {
        double util1 = static_batching_utilization(requests, 1);
        CHECK(near(util1, 1.0));

        auto results1 = simulate_static_batching(requests, 1);
        // Strict FIFO by arrival order: 0 (steps 10) -> completes 10;
        // 1 (steps 2) -> starts 10, completes 12;
        // 2 (steps 1, arrived 1) -> starts 12, completes 13;
        // 3 (steps 3, arrived 1) -> starts 13, completes 16.
        CHECK(find_result(results1, 0).completion_step == 10);
        CHECK(find_result(results1, 1).completion_step == 12);
        CHECK(find_result(results1, 2).completion_step == 13);
        CHECK(find_result(results1, 3).completion_step == 16);
        for (const auto& r : results1) CHECK(r.wasted_padding_steps == 0);
        std::cout << "  with batch_size 1, every real \"batch\" contains exactly one request, so its own "
                     "makespan always equals that request's own needed_steps exactly -- zero padding waste "
                     "by construction, and utilization of exactly " << util1 << ", confirming the general "
                     "formula behaves correctly at this trivial real boundary case\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
