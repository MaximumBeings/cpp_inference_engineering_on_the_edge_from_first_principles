// Chapter 31.3 -- A real, from-scratch continuous-batching scheduler. Unlike
// Section 31.1's own static batching, which cannot free a slot until its
// ENTIRE batch has retired, continuous batching frees a slot the instant its
// own sequence finishes -- immediately admitting the next waiting request
// into that same slot, on the very next scheduling step. This section
// restates Section 31.1's own static simulator to run it, unmodified, on the
// identical real workload, so this section's own continuous scheduler can be
// checked directly against it rather than against a hardcoded number.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_continuous_batching_scheduler_from_scratch.cpp -o 03_continuous_batching_scheduler_from_scratch
// Run:     ./03_continuous_batching_scheduler_from_scratch

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: Section 31.1's own request shape and static-batching simulator,
// restated here in full so this section's own continuous scheduler can be
// checked directly against it, on the identical real workload, within this
// one self-contained file.
// =======================================================================
struct Request {
    int id = 0;
    int arrival_step = 0;
    int needed_steps = 0;
};

struct StaticBatchResult {
    int id = 0;
    int completion_step = 0;
    int wasted_padding_steps = 0;
};

std::vector<StaticBatchResult> simulate_static_batching(std::vector<Request> requests, int batch_size) {
    std::stable_sort(requests.begin(), requests.end(),
                      [](const Request& a, const Request& b) { return a.arrival_step < b.arrival_step; });
    std::vector<StaticBatchResult> results;
    int t = 0;
    for (std::size_t i = 0; i < requests.size(); i += static_cast<std::size_t>(batch_size)) {
        std::size_t end = std::min(requests.size(), i + static_cast<std::size_t>(batch_size));
        int latest_arrival = 0, makespan = 0;
        for (std::size_t j = i; j < end; ++j) {
            latest_arrival = std::max(latest_arrival, requests[j].arrival_step);
            makespan = std::max(makespan, requests[j].needed_steps);
        }
        int start = std::max(t, latest_arrival);
        int completion = start + makespan;
        for (std::size_t j = i; j < end; ++j) {
            results.push_back({requests[j].id, completion, makespan - requests[j].needed_steps});
        }
        t = completion;
    }
    return results;
}

// =======================================================================
// PART 2: the real continuous-batching scheduler. At every discrete tick:
// (1) admit the earliest-arrived still-waiting request into any real free
// slot, as long as it has already arrived; (2) decode every currently
// active sequence by exactly one real step; (3) retire -- and immediately
// free the slot of -- any sequence whose own needed_steps has just been
// fully satisfied, making that slot available for admission on the very
// NEXT tick, without waiting for any other sequence in that same batch.
// =======================================================================
struct ContinuousResult {
    int id = 0;
    int start_tick = 0;
    int completion_tick = 0;
    int wait_ticks = 0;
};

struct ActiveSeq {
    int id = 0;
    int remaining = 0;
};

std::vector<ContinuousResult> simulate_continuous_batching(std::vector<Request> requests, int max_slots) {
    std::stable_sort(requests.begin(), requests.end(),
                      [](const Request& a, const Request& b) { return a.arrival_step < b.arrival_step; });

    std::vector<ContinuousResult> results(requests.size());
    std::vector<bool> admitted(requests.size(), false);
    std::vector<std::optional<ActiveSeq>> slots(static_cast<std::size_t>(max_slots));

    std::size_t remaining_count = requests.size();
    int tick = 0;
    while (remaining_count > 0) {
        // (1) Admit: fill every free slot with the earliest-arrived waiting request.
        for (auto& slot : slots) {
            if (slot.has_value()) continue;
            int best_idx = -1;
            for (std::size_t i = 0; i < requests.size(); ++i) {
                if (admitted[i] || requests[i].arrival_step > tick) continue;
                if (best_idx == -1 || requests[i].arrival_step < requests[static_cast<std::size_t>(best_idx)].arrival_step) {
                    best_idx = static_cast<int>(i);
                }
            }
            if (best_idx == -1) continue;
            admitted[static_cast<std::size_t>(best_idx)] = true;
            slot = ActiveSeq{requests[static_cast<std::size_t>(best_idx)].id, requests[static_cast<std::size_t>(best_idx)].needed_steps};
            results[static_cast<std::size_t>(best_idx)].id = requests[static_cast<std::size_t>(best_idx)].id;
            results[static_cast<std::size_t>(best_idx)].start_tick = tick;
            results[static_cast<std::size_t>(best_idx)].wait_ticks = tick - requests[static_cast<std::size_t>(best_idx)].arrival_step;
        }

        // (2) Decode every active sequence by one real step, (3) retire any that finish.
        for (auto& slot : slots) {
            if (!slot.has_value()) continue;
            slot->remaining -= 1;
            if (slot->remaining == 0) {
                for (auto& r : results) {
                    if (r.id == slot->id) { r.completion_tick = tick; break; }
                }
                --remaining_count;
                slot.reset();
            }
        }
        tick += 1;
    }
    return results;
}

int total_active_slot_ticks(const std::vector<Request>& requests, const std::vector<ContinuousResult>& results) {
    int total = 0;
    for (const auto& r : results) total += (r.completion_tick - r.start_tick + 1);
    (void)requests;
    return total;
}

const ContinuousResult& find_continuous(const std::vector<ContinuousResult>& results, int id) {
    for (const auto& r : results) if (r.id == id) return r;
    static ContinuousResult empty;
    return empty;
}
const StaticBatchResult& find_static(const std::vector<StaticBatchResult>& results, int id) {
    for (const auto& r : results) if (r.id == id) return r;
    static StaticBatchResult empty;
    return empty;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 31.3: A Continuous-Batching Scheduler Built From Scratch\n";
    std::cout << "========================================================\n";

    // The identical 4-request, 2-slot workload from Section 31.1.
    std::vector<Request> requests = {
        {0, 0, 10},
        {1, 0, 2},
        {2, 1, 1},
        {3, 1, 3},
    };
    const int max_slots = 2;

    std::cout << "\n-- Test 1: the real continuous-batching schedule for this identical 4-request workload "
                 "matches an exact hand trace, with every completion tick strictly earlier than that "
                 "identical request's own static-batching completion step from Section 31.1 --\n";
    {
        auto results = simulate_continuous_batching(requests, max_slots);

        CHECK(find_continuous(results, 0).start_tick == 0);
        CHECK(find_continuous(results, 0).completion_tick == 9);
        CHECK(find_continuous(results, 1).start_tick == 0);
        CHECK(find_continuous(results, 1).completion_tick == 1);
        CHECK(find_continuous(results, 2).start_tick == 2);
        CHECK(find_continuous(results, 2).completion_tick == 2);
        CHECK(find_continuous(results, 2).wait_ticks == 1);
        CHECK(find_continuous(results, 3).start_tick == 3);
        CHECK(find_continuous(results, 3).completion_tick == 5);
        CHECK(find_continuous(results, 3).wait_ticks == 2);

        std::cout << "  request 1 (2 steps) retires at tick 1, freeing its own slot immediately; request 2 "
                     "(1 step, arrived at tick 1) is admitted into that freed slot at tick 2 -- a real wait "
                     "of only 1 tick, against Section 31.1's own 9-tick wait for the identical request -- "
                     "and completes that same tick; request 3 (3 steps, arrived at tick 1) is admitted next "
                     "at tick 3 and completes at tick 5; request 0 (10 steps) occupies the other slot "
                     "throughout and completes last, at tick 9\n";
    }

    std::cout << "\n-- Test 2: this identical workload's own real static-batching schedule, computed by the "
                 "SAME simulator restated from Section 31.1 within this file, completes every request "
                 "strictly later than continuous batching did -- confirming continuous batching's own real "
                 "improvement is checked directly against a live computation, not a hardcoded number --\n";
    {
        auto continuous_results = simulate_continuous_batching(requests, max_slots);
        auto static_results = simulate_static_batching(requests, max_slots);

        CHECK(find_static(static_results, 0).completion_step == 10);
        CHECK(find_static(static_results, 1).completion_step == 10);
        CHECK(find_static(static_results, 2).completion_step == 13);
        CHECK(find_static(static_results, 3).completion_step == 13);

        for (int id : {0, 1, 2, 3}) {
            CHECK(find_continuous(continuous_results, id).completion_tick < find_static(static_results, id).completion_step);
        }
        std::cout << "  every one of the 4 requests completes strictly earlier under continuous batching "
                     "than under static batching's own identical-workload schedule: request 0 at tick 9 "
                     "versus step 10, request 1 at tick 1 versus step 10, request 2 at tick 2 versus step "
                     "13, and request 3 at tick 5 versus step 13\n";
    }

    std::cout << "\n-- Test 3: continuous batching's own real slot-tick accounting has ZERO padding waste by "
                 "construction -- the total number of active-slot ticks spent across the whole real "
                 "schedule equals exactly the sum of every request's own needed_steps, with nothing wasted "
                 "on an already-finished sequence's slot, unlike Section 31.1's own static schedule, which "
                 "wasted a real, nonzero 10 slot-steps on the identical workload --\n";
    {
        auto continuous_results = simulate_continuous_batching(requests, max_slots);
        int total_ticks = total_active_slot_ticks(requests, continuous_results);
        int sum_needed = 0;
        for (const auto& r : requests) sum_needed += r.needed_steps;
        CHECK(total_ticks == sum_needed);
        CHECK(total_ticks == 16);

        auto static_results = simulate_static_batching(requests, max_slots);
        int static_wasted = 0;
        for (const auto& r : static_results) static_wasted += r.wasted_padding_steps;
        CHECK(static_wasted == 10);
        CHECK(static_wasted > 0);

        std::cout << "  continuous batching's own total active-slot ticks, " << total_ticks << ", exactly "
                     "equals the sum of every request's own needed_steps (10 + 2 + 1 + 3 = 16) -- zero "
                     "padding waste, ever, by construction -- while static batching's own identical "
                     "workload wasted a real " << static_wasted << " slot-steps on padding\n";
    }

    std::cout << "\n-- Test 4: on a second, genuinely different real workload -- 3 requests and only 1 real "
                 "slot -- continuous batching correctly degenerates to strict FIFO, admitting each request "
                 "the instant both its own arrival has happened and the single slot is free, with zero "
                 "queue wait whenever a request has already arrived by the time the slot frees up --\n";
    {
        std::vector<Request> requests2 = {
            {0, 0, 3},
            {1, 0, 2},
            {2, 5, 1},
        };
        auto results2 = simulate_continuous_batching(requests2, 1);

        CHECK(find_continuous(results2, 0).start_tick == 0);
        CHECK(find_continuous(results2, 0).completion_tick == 2);
        CHECK(find_continuous(results2, 1).start_tick == 3);
        CHECK(find_continuous(results2, 1).completion_tick == 4);
        CHECK(find_continuous(results2, 1).wait_ticks == 3);
        CHECK(find_continuous(results2, 2).start_tick == 5);
        CHECK(find_continuous(results2, 2).completion_tick == 5);
        CHECK(find_continuous(results2, 2).wait_ticks == 0);

        int total_ticks2 = total_active_slot_ticks(requests2, results2);
        CHECK(total_ticks2 == 6);
        std::cout << "  with a single real slot, request 0 (3 steps) runs ticks 0-2; request 1 (2 steps, "
                     "already waiting since tick 0) is admitted the instant the slot frees, at tick 3, and "
                     "runs to tick 4; request 2 (1 step) does not arrive until tick 5, exactly when the "
                     "slot is free again, so its own real queue wait is 0; total active-slot ticks, "
                  << total_ticks2 << ", again exactly equals the sum of needed_steps (3 + 2 + 1 = 6)\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
