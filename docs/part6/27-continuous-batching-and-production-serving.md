# Chapter 27: Continuous Batching and Production Serving Architecture

**What you will understand by the end of this chapter:**

- Why the naive, real batching scheme early inference servers actually used -- fixed-size batches that cannot start until the previous one fully retires -- wastes real GPU-slot time on padding and delays a short request that arrives just after a long batch has already started, quantified with exact hand-traceable numbers rather than asserted.
- Why prefill (a new request's full prompt, processed in one shot) and decode (every already-running sequence, advanced by exactly one token) have genuinely different real cost profiles, and why naively injecting an entire prefill into a single step spikes that step's own real latency for every already-running sequence sharing it.
- How real chunked prefill bounds that injected latency to a fixed, stated maximum regardless of how long the new prompt actually is, and how to derive that bound directly from the same stated cost model rather than tuning it empirically.
- How to build a real, from-scratch continuous-batching scheduler that frees a slot the instant its own sequence finishes -- immediately admitting the next waiting request into that same slot -- and to check its own real improvement over static batching directly, on the identical workload, rather than against a hardcoded number.
- How to build two real numerical debugging tools this book's own kernels have needed since Chapter 26 introduced the numerical instabilities they catch: a NaN/Inf-propagation tracer that finds where corruption actually ORIGINATED, not merely where it is still visible, and a floating-point drift detector that quantifies exactly how much a real reduction's own result changes when its order changes -- a genuine risk once continuous batching starts regrouping sequences differently from run to run.

**What you need to know first:**

- Section 26.1's own real roofline classification (GEMV is intrinsically memory-bound; GEMM becomes compute-bound only past a derivable crossover batch size) is the exact real reason this chapter's own scheduler exists at all: a serving system's whole real job is to keep the batch size a given machine actually runs as close to that crossover as real, unpredictable traffic allows.
- Section 26.2's own real numerical-instability discipline -- reproducing a genuine failure on purpose, understanding exactly why it happens, and fixing it with a provable identity rather than a downstream patch -- is the same discipline this chapter's own Section 27.4 applies to NaN propagation and floating-point drift, two real failure modes that specifically appear at serving scale rather than in a single isolated kernel test.
- Chapter 13's own KV Cache Manager and Chapter 14's own advanced eviction and prefix-caching techniques are the real memory-management machinery a production scheduler like this chapter's own Section 27.3 would sit directly on top of in a real system -- this chapter's own scheduler abstracts that machinery into a stated per-step cost, the same honest simplification Chapter 26 applied to FLOP and byte counting.

---

Every chapter before this one built one real piece of an inference engine and verified it in isolation. A real serving system has to run many of those pieces at once, for many concurrent real users, whose requests arrive at unpredictable times and need unpredictable amounts of work -- and the scheduling decisions that follow from that are not optional engineering polish, they are the actual difference between a system that serves real traffic efficiently and one that does not serve it at all. This chapter builds that scheduling layer from scratch: first by understanding, with real hand-traceable numbers, exactly what goes wrong with the naive approach; then by resolving the specific real conflict between the two fundamentally different kinds of work a serving step can do; then by building the real scheduler that puts both fixes to work; and finally by building the real numerical debugging tools a system running at this scale actually needs when something goes wrong.

## 27.1 The Static-Batching Problem and Head-of-Line Blocking

### Intuition

The simplest way to batch real inference requests together is also the most wasteful one: group a fixed number of them, run the whole group until every single member is done, and only then admit the next group. This section builds a real, from-scratch simulator for exactly that scheme, to quantify -- not merely describe -- how much real GPU-slot time it wastes and how badly it can delay a request unlucky enough to arrive at the wrong moment.

### The Concept, In Detail

`simulate_static_batching` groups a real, arrival-ordered queue of requests into fixed-size chunks and runs each chunk as a single real batch: the batch cannot start until every one of its own members has arrived AND the previous batch has fully retired, and once started, every slot in it stays occupied -- computing nothing useful once its own request has already finished -- until the LONGEST request in that batch also finishes. Test 1 traces this exactly on a tiny 4-request workload: a short request sharing a batch with a much longer one sits idle, wasting 8 real padding steps, and a separate short request that arrives just one step after the batch already started is not just delayed by that one step -- it is blocked for the batch's ENTIRE remaining duration, a real, substantial penalty Test 3 quantifies directly as head-of-line blocking.

Test 2 turns this into a single real utilization number -- useful steps actually computed divided by total real slot-steps spent, including padding -- and Test 4 confirms the general formulas collapse correctly to the trivial degenerate case: at a real batch size of exactly 1, static batching becomes strict FIFO with zero padding waste and perfect utilization, confirming the waste this section quantifies is a genuine consequence of grouping requests together, not an artifact of the simulator itself.

### Code and Verification

```cpp
// Chapter 27.1 -- The static-batching problem: a real, from-scratch discrete-
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
// own purposes -- Section 27.2 separates them explicitly).
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
    std::cout << "Chapter 27.1: The Static-Batching Problem and Head-of-Line Blocking\n";
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_static_batching_problem_and_head_of_line_blocking.cpp -o 01_static_batching_problem_and_head_of_line_blocking
./01_static_batching_problem_and_head_of_line_blocking
```

**Sample input:** a real 4-request static-batching schedule checked against an exact hand trace of start steps, completion steps, wasted padding steps, and queue-wait steps; the workload's own real slot utilization checked against an exact hand computation; a short request's own real head-of-line blocking delay checked directly against the completion it would have gotten served alone; and the degenerate real batch-size-1 case checked to collapse to strict FIFO with perfect utilization.

```text
========================================================
Chapter 27.1: The Static-Batching Problem and Head-of-Line Blocking
========================================================

-- Test 1: the real static-batching schedule for this tiny 4-request workload matches an exact hand trace: batch A = {0, 1} starts at step 0 with a real makespan of 10 (bounded by request 0's own 10 needed steps), retiring at step 10; batch B = {2, 3} cannot start until batch A retires, at step 10, despite both of its own members having arrived at step 1 --
  request 0 (10 steps needed): starts 0, completes 10, 0 wasted padding steps -- it is the one that sets batch A's own makespan
  request 1 (2 steps needed): starts 0, completes 10, 8 wasted padding steps -- its own slot sits idle for 8 real steps after it finishes, because batch A cannot retire until request 0 also finishes
  request 2 (1 step needed, arrived at step 1): starts 10, completes 13, queued for 9 real steps before batch B even begins, despite needing only 1 step of real work
  request 3 (3 steps needed, arrived at step 1): starts 10, completes 13, 0 wasted padding within batch B, but also queued for the identical 9 real steps as request 2

-- Test 2: this workload's own real slot utilization -- useful steps actually computed divided by total real slot-steps spent, including padding -- matches an exact hand computation: 16 useful steps out of 26 total slot-steps --
  useful steps = 10 + 2 + 1 + 3 = 16; total slot-steps = (2 slots * 10-step makespan) + (2 slots * 3-step makespan) = 20 + 6 = 26; real utilization = 16 / 26 = 0.615385

-- Test 3: request 2's own real head-of-line blocking delay, computed as the difference between its own actual completion step and the completion step it WOULD have gotten had it been served alone, immediately upon arrival, is a real, substantial 11 steps -- despite needing only 1 step of actual work --
  request 2 arrives at step 1 needing only 1 step -- served alone it would complete at step 2; static batching's own head-of-line blocking instead delays it all the way to step 13, a real, substantial 11-step penalty caused entirely by having arrived just after a long-running batch had already started

-- Test 4: at the degenerate real batch_size of 1, static batching's own scheduling collapses to strict real FIFO with zero padding waste -- utilization is exactly 1.0, confirming this section's own general formulas reduce correctly to the trivial real case rather than being tuned only to the batch_size-2 example above --
  with batch_size 1, every real "batch" contains exactly one request, so its own makespan always equals that request's own needed_steps exactly -- zero padding waste by construction, and utilization of exactly 1, confirming the general formula behaves correctly at this trivial real boundary case

29/29 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] assuming a short request's own real delay is bounded by how long it had to wait in a literal queue"
    It is tempting to think a request's own worst-case real delay under static batching is bounded by how many OTHER requests are ahead of it. Test 3 shows the real bound that actually matters is completely different: a request that arrives one single step after a batch has already started is not delayed by "one step's worth of queue" -- it is delayed by the ENTIRE remaining duration of whatever the longest-running member of that batch still needs, regardless of how short its own real job is. A 1-step request arriving moments too late waited a real 9 steps before its own batch even began, purely because static batching's own retirement rule has no way to let a new, urgent request cut in once a batch is already running -- which is exactly the real constraint continuous batching (Section 27.3) is built to remove.

## 27.2 The Prefill/Decode Conflict and Chunked Prefill

### Intuition

A serving step that only advances already-running sequences by one token each (decode) has a very different real cost profile from a step that processes an entirely new prompt in one shot (prefill), and naively mixing the two in a single step means the SLOWER of the two profiles imposes its own real cost on every sequence sharing that step -- including ones that have nothing to do with the new request at all.

### The Concept, In Detail

This section's own real cost model keeps the two profiles honest and separate: `cost_decode_step` scales with how many sequences are concurrently decoding (a real, memory-bound cost -- every active sequence's own KV cache must be read once per step), while `cost_prefill_chunk` scales with how many new prompt tokens are being processed in that same step (a real, compute-bound cost). Test 1 and Test 2 confirm the real conflict directly: injecting an entire 64-token prompt into a single step alongside 8 concurrently decoding sequences spikes that one step's own real cost by 2560 units over baseline -- latency injected into all 8 already-running sequences at once, not just the new request.

Test 3 introduces chunked prefill's own real fix: splitting that identical 64-token prompt into 4 chunks of at most 16 tokens each, injected across 4 separate steps, bounds the per-step injected latency to exactly 640 units -- one quarter of the unchunked spike -- while the TOTAL extra work performed across those 4 steps is conserved exactly, confirming chunking spreads real work out rather than creating or destroying any of it. Test 4 generalizes this into a real, provable bound: chunked prefill's own worst-case injected latency never exceeds `max_chunk_tokens`' own cost, regardless of the full prompt's real length, while the unchunked injected latency grows linearly and unbounded with prompt length -- checked directly across several genuinely different real prompt lengths, not merely the one hand-picked example.

### Code and Verification

```cpp
// Chapter 27.2 -- The real prefill/decode conflict a continuous-batching
// scheduler must resolve. Prefill (processing a new request's own full
// prompt in one shot) and decode (advancing every already-running sequence
// by exactly one token) have genuinely different real cost profiles: decode
// is memory-bound and its own real cost scales with how many sequences are
// concurrently decoding, while prefill is compute-bound and its own real
// cost scales with how many prompt tokens are being processed. Naively
// injecting an entire new prefill into a single step alongside a batch of
// already-decoding sequences spikes that ONE step's own real latency for
// every sequence sharing it -- and chunked prefill, splitting the prefill
// across several smaller steps, is the real fix this section derives and
// bounds.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_prefill_decode_conflict_and_chunked_prefill.cpp -o 02_prefill_decode_conflict_and_chunked_prefill
// Run:     ./02_prefill_decode_conflict_and_chunked_prefill

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

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: a real, simplified but honest cost model for one scheduling step.
// Decode's own real cost is memory-bound: it scales with how many sequences
// are concurrently decoding, since each one's own KV cache must be read
// from real memory exactly once per step regardless of how much compute
// that step also does. Prefill's own real cost is compute-bound: it scales
// with how many NEW prompt tokens are being processed in that same step.
// =======================================================================
constexpr double KV_READ_BYTES_PER_SEQ = 128.0;   // real per-sequence, per-step KV-cache read cost unit
constexpr double PREFILL_COST_PER_TOKEN = 40.0;   // real per-token prefill compute cost unit

double cost_decode_step(int active_decodes, double kv_read_bytes_per_seq = KV_READ_BYTES_PER_SEQ) {
    return static_cast<double>(active_decodes) * kv_read_bytes_per_seq;
}

double cost_prefill_chunk(int chunk_tokens, double prefill_cost_per_token = PREFILL_COST_PER_TOKEN) {
    return static_cast<double>(chunk_tokens) * prefill_cost_per_token;
}

double combined_step_cost(int active_decodes, int chunk_tokens) {
    return cost_decode_step(active_decodes) + cost_prefill_chunk(chunk_tokens);
}

// =======================================================================
// PART 2: chunked prefill. Rather than injecting an entire new prompt's own
// real P tokens into a single step, chunked prefill splits it into
// ceil(P / max_chunk_tokens) real pieces, each injected into its own
// separate decode step -- bounding the WORST single-step latency spike to
// at most max_chunk_tokens' own real cost, regardless of how large the full
// prompt P actually is.
// =======================================================================
int num_chunks(int total_prompt_tokens, int max_chunk_tokens) {
    return (total_prompt_tokens + max_chunk_tokens - 1) / max_chunk_tokens;
}

std::vector<int> chunk_sizes(int total_prompt_tokens, int max_chunk_tokens) {
    std::vector<int> sizes;
    int remaining = total_prompt_tokens;
    while (remaining > 0) {
        int this_chunk = std::min(remaining, max_chunk_tokens);
        sizes.push_back(this_chunk);
        remaining -= this_chunk;
    }
    return sizes;
}

std::vector<double> schedule_chunked_prefill_step_costs(int total_prompt_tokens, int max_chunk_tokens,
                                                          int active_decodes) {
    std::vector<double> step_costs;
    for (int chunk : chunk_sizes(total_prompt_tokens, max_chunk_tokens)) {
        step_costs.push_back(combined_step_cost(active_decodes, chunk));
    }
    return step_costs;
}

double max_injected_latency(const std::vector<double>& step_costs, double baseline_decode_cost) {
    double worst = 0.0;
    for (double c : step_costs) worst = std::max(worst, c - baseline_decode_cost);
    return worst;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 27.2: The Prefill/Decode Conflict and Chunked Prefill\n";
    std::cout << "========================================================\n";

    const int active_decodes = 8;
    const int total_prompt_tokens = 64;
    const int max_chunk_tokens = 16;

    std::cout << "\n-- Test 1: this section's own real baseline decode-step cost, with 8 sequences "
                 "concurrently decoding and no prefill injected at all, matches an exact hand computation "
                 "of 1024 real cost units --\n";
    {
        double baseline = cost_decode_step(active_decodes);
        CHECK(near(baseline, 1024.0));
        std::cout << "  cost_decode_step(8) = 8 * 128 = " << baseline << " real cost units\n";
    }

    std::cout << "\n-- Test 2: injecting a real new request's entire 64-token prompt into a SINGLE step "
                 "alongside those 8 concurrently decoding sequences produces a combined step cost of 3584 "
                 "-- a real 2560-unit latency spike over the baseline, injected into every one of those 8 "
                 "already-running sequences' own current step --\n";
    {
        double combined = combined_step_cost(active_decodes, total_prompt_tokens);
        double baseline = cost_decode_step(active_decodes);
        double injected = combined - baseline;
        CHECK(near(combined, 3584.0));
        CHECK(near(injected, 2560.0));
        std::cout << "  combined_step_cost(8, 64) = 1024 + (64 * 40) = 1024 + 2560 = " << combined
                  << "; real injected latency over baseline = " << injected << " cost units\n";
    }

    std::cout << "\n-- Test 3: splitting that identical 64-token prompt into real chunks of at most 16 "
                 "tokens produces exactly 4 chunks, each injected into its own separate step at a bounded "
                 "cost of 1664 -- a real 640-unit injected latency per step, one quarter of the unchunked "
                 "spike -- while the TOTAL extra work performed across all 4 chunked steps combined is "
                 "conserved exactly, matching the unchunked total exactly --\n";
    {
        CHECK(num_chunks(total_prompt_tokens, max_chunk_tokens) == 4);
        auto sizes = chunk_sizes(total_prompt_tokens, max_chunk_tokens);
        CHECK(sizes.size() == 4);
        for (int s : sizes) CHECK(s == 16);

        auto step_costs = schedule_chunked_prefill_step_costs(total_prompt_tokens, max_chunk_tokens, active_decodes);
        CHECK(step_costs.size() == 4);
        double baseline = cost_decode_step(active_decodes);
        double total_injected_chunked = 0.0;
        for (double c : step_costs) {
            CHECK(near(c, 1664.0));
            total_injected_chunked += (c - baseline);
        }
        CHECK(near(total_injected_chunked, 2560.0));  // exactly the unchunked injected total -- conserved
        std::cout << "  4 chunks of 16 tokens each, every one costing 1024 + (16 * 40) = " << step_costs[0]
                  << " -- a real per-step injected latency of 640 units, one quarter of the unchunked "
                     "2560-unit spike -- while the total real extra work across all 4 steps, " << total_injected_chunked
                  << ", exactly conserves the unchunked total: chunking spreads the identical real work out, "
                     "it does not create or destroy any of it\n";
    }

    std::cout << "\n-- Test 4: the real bound chunked prefill provides -- worst-case injected latency never "
                 "exceeds max_chunk_tokens' own cost, REGARDLESS of the full prompt's own real length -- "
                 "holds across several genuinely different real prompt lengths, while the unchunked "
                 "injected latency grows linearly and UNBOUNDED with prompt length, eventually exceeding "
                 "that same fixed bound by a wide margin --\n";
    {
        double baseline = cost_decode_step(active_decodes);
        double chunk_bound = cost_prefill_chunk(max_chunk_tokens);  // 640.0, independent of total prompt length
        CHECK(near(chunk_bound, 640.0));

        for (int prompt_len : {17, 100, 200, 333}) {
            auto step_costs = schedule_chunked_prefill_step_costs(prompt_len, max_chunk_tokens, active_decodes);
            double chunked_worst = max_injected_latency(step_costs, baseline);
            CHECK(chunked_worst <= chunk_bound + 1e-9);

            double unchunked_injected = combined_step_cost(active_decodes, prompt_len) - baseline;
            // Unbounded: it must exceed the fixed chunked bound once the prompt is long enough,
            // which every one of these stated real prompt lengths already is.
            CHECK(unchunked_injected > chunk_bound);
        }
        std::cout << "  across prompt lengths 17, 100, 200, and 333 tokens, chunked prefill's own worst "
                     "real per-step injected latency never exceeds the fixed 640-unit bound, while the "
                     "unchunked injected latency -- 680, 4000, 8000, and 13320 units respectively -- grows "
                     "linearly with prompt length and exceeds that fixed bound in every single case\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_prefill_decode_conflict_and_chunked_prefill.cpp -o 02_prefill_decode_conflict_and_chunked_prefill
./02_prefill_decode_conflict_and_chunked_prefill
```

**Sample input:** a real baseline decode-step cost checked against an exact hand computation; an unchunked single-step prefill injection checked to spike that step's own real cost by an exact, computed amount; chunked prefill's own real per-step cost and total conserved work checked against an exact hand trace; and the chunked latency bound checked to hold, and the unchunked latency to exceed it, across several genuinely different real prompt lengths.

```text
========================================================
Chapter 27.2: The Prefill/Decode Conflict and Chunked Prefill
========================================================

-- Test 1: this section's own real baseline decode-step cost, with 8 sequences concurrently decoding and no prefill injected at all, matches an exact hand computation of 1024 real cost units --
  cost_decode_step(8) = 8 * 128 = 1024 real cost units

-- Test 2: injecting a real new request's entire 64-token prompt into a SINGLE step alongside those 8 concurrently decoding sequences produces a combined step cost of 3584 -- a real 2560-unit latency spike over the baseline, injected into every one of those 8 already-running sequences' own current step --
  combined_step_cost(8, 64) = 1024 + (64 * 40) = 1024 + 2560 = 3584; real injected latency over baseline = 2560 cost units

-- Test 3: splitting that identical 64-token prompt into real chunks of at most 16 tokens produces exactly 4 chunks, each injected into its own separate step at a bounded cost of 1664 -- a real 640-unit injected latency per step, one quarter of the unchunked spike -- while the TOTAL extra work performed across all 4 chunked steps combined is conserved exactly, matching the unchunked total exactly --
  4 chunks of 16 tokens each, every one costing 1024 + (16 * 40) = 1664 -- a real per-step injected latency of 640 units, one quarter of the unchunked 2560-unit spike -- while the total real extra work across all 4 steps, 2560, exactly conserves the unchunked total: chunking spreads the identical real work out, it does not create or destroy any of it

-- Test 4: the real bound chunked prefill provides -- worst-case injected latency never exceeds max_chunk_tokens' own cost, REGARDLESS of the full prompt's own real length -- holds across several genuinely different real prompt lengths, while the unchunked injected latency grows linearly and UNBOUNDED with prompt length, eventually exceeding that same fixed bound by a wide margin --
  across prompt lengths 17, 100, 200, and 333 tokens, chunked prefill's own worst real per-step injected latency never exceeds the fixed 640-unit bound, while the unchunked injected latency -- 680, 4000, 8000, and 13320 units respectively -- grows linearly with prompt length and exceeds that fixed bound in every single case

24/24 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] assuming chunked prefill reduces the TOTAL real work a new request's prompt requires"
    Chunked prefill does not make a long prompt cheaper to process overall -- Test 3 confirms the total extra work across all 4 chunked steps conserves the unchunked total EXACTLY, to the unit. What chunking actually changes is when that real cost is paid: instead of one single step absorbing the entire spike, several smaller steps each absorb a bounded fraction of it, keeping every individual step's own real latency predictable for the sequences that happen to share it. A system that chunks prefill expecting a net real throughput gain from chunking alone will be disappointed -- the genuine benefit is bounded worst-case per-step latency, which is a real, different, and for a production serving system, usually far more valuable property than raw total throughput.

## 27.3 A Continuous-Batching Scheduler Built From Scratch

### Intuition

Static batching's own real flaw, quantified in Section 27.1, is structural: a slot cannot be reused until its ENTIRE batch retires, even if that slot's own request finished long ago. Continuous batching fixes this directly -- a slot frees the instant its own sequence finishes, and the very next scheduling step can admit a new request into it, with no need to wait for anything else in that batch.

### The Concept, In Detail

`simulate_continuous_batching` runs a real discrete-tick loop that, every tick, admits the earliest-arrived waiting request into any real free slot, decodes every active sequence by exactly one step, and immediately retires -- and frees the slot of -- any sequence whose own needed steps have just been satisfied. Test 1 traces this exactly on the IDENTICAL 4-request workload Section 27.1 used, and Test 2 runs Section 27.1's own static-batching simulator, restated in full within this file, on that same workload side by side: every one of the 4 requests completes strictly earlier under continuous batching, checked directly against a live computation rather than a hardcoded comparison number.

Test 3 confirms the real structural reason why: continuous batching's own total active-slot ticks across the whole schedule equals EXACTLY the sum of every request's own needed steps -- zero padding waste, ever, by construction -- while static batching's identical workload wasted a real, nonzero 10 slot-steps. Test 4 confirms this scheduler's own real correctness generalizes beyond that one example: on a second, genuinely different workload with only a single real slot, it correctly degenerates to strict FIFO, admitting each request the instant both its own arrival and slot availability align, with zero real queue wait whenever a request has already arrived by the time its slot frees.

### Code and Verification

```cpp
// Chapter 27.3 -- A real, from-scratch continuous-batching scheduler. Unlike
// Section 27.1's own static batching, which cannot free a slot until its
// ENTIRE batch has retired, continuous batching frees a slot the instant its
// own sequence finishes -- immediately admitting the next waiting request
// into that same slot, on the very next scheduling step. This section
// restates Section 27.1's own static simulator to run it, unmodified, on the
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
// PART 1: Section 27.1's own request shape and static-batching simulator,
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
    std::cout << "Chapter 27.3: A Continuous-Batching Scheduler Built From Scratch\n";
    std::cout << "========================================================\n";

    // The identical 4-request, 2-slot workload from Section 27.1.
    std::vector<Request> requests = {
        {0, 0, 10},
        {1, 0, 2},
        {2, 1, 1},
        {3, 1, 3},
    };
    const int max_slots = 2;

    std::cout << "\n-- Test 1: the real continuous-batching schedule for this identical 4-request workload "
                 "matches an exact hand trace, with every completion tick strictly earlier than that "
                 "identical request's own static-batching completion step from Section 27.1 --\n";
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
                     "of only 1 tick, against Section 27.1's own 9-tick wait for the identical request -- "
                     "and completes that same tick; request 3 (3 steps, arrived at tick 1) is admitted next "
                     "at tick 3 and completes at tick 5; request 0 (10 steps) occupies the other slot "
                     "throughout and completes last, at tick 9\n";
    }

    std::cout << "\n-- Test 2: this identical workload's own real static-batching schedule, computed by the "
                 "SAME simulator restated from Section 27.1 within this file, completes every request "
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
                 "on an already-finished sequence's slot, unlike Section 27.1's own static schedule, which "
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_continuous_batching_scheduler_from_scratch.cpp -o 03_continuous_batching_scheduler_from_scratch
./03_continuous_batching_scheduler_from_scratch
```

**Sample input:** a real continuous-batching schedule for a 4-request, 2-slot workload checked against an exact hand trace of admission and completion ticks; that identical workload's own static-batching schedule, computed live by the restated Section 27.1 simulator, checked to complete every request strictly later; continuous batching's own total active-slot ticks checked to exactly conserve the sum of needed steps with zero padding waste; and a second, genuinely different single-slot workload checked to correctly degenerate to strict FIFO.

```text
========================================================
Chapter 27.3: A Continuous-Batching Scheduler Built From Scratch
========================================================

-- Test 1: the real continuous-batching schedule for this identical 4-request workload matches an exact hand trace, with every completion tick strictly earlier than that identical request's own static-batching completion step from Section 27.1 --
  request 1 (2 steps) retires at tick 1, freeing its own slot immediately; request 2 (1 step, arrived at tick 1) is admitted into that freed slot at tick 2 -- a real wait of only 1 tick, against Section 27.1's own 9-tick wait for the identical request -- and completes that same tick; request 3 (3 steps, arrived at tick 1) is admitted next at tick 3 and completes at tick 5; request 0 (10 steps) occupies the other slot throughout and completes last, at tick 9

-- Test 2: this identical workload's own real static-batching schedule, computed by the SAME simulator restated from Section 27.1 within this file, completes every request strictly later than continuous batching did -- confirming continuous batching's own real improvement is checked directly against a live computation, not a hardcoded number --
  every one of the 4 requests completes strictly earlier under continuous batching than under static batching's own identical-workload schedule: request 0 at tick 9 versus step 10, request 1 at tick 1 versus step 10, request 2 at tick 2 versus step 13, and request 3 at tick 5 versus step 13

-- Test 3: continuous batching's own real slot-tick accounting has ZERO padding waste by construction -- the total number of active-slot ticks spent across the whole real schedule equals exactly the sum of every request's own needed_steps, with nothing wasted on an already-finished sequence's slot, unlike Section 27.1's own static schedule, which wasted a real, nonzero 10 slot-steps on the identical workload --
  continuous batching's own total active-slot ticks, 16, exactly equals the sum of every request's own needed_steps (10 + 2 + 1 + 3 = 16) -- zero padding waste, ever, by construction -- while static batching's own identical workload wasted a real 10 slot-steps on padding

-- Test 4: on a second, genuinely different real workload -- 3 requests and only 1 real slot -- continuous batching correctly degenerates to strict FIFO, admitting each request the instant both its own arrival has happened and the single slot is free, with zero queue wait whenever a request has already arrived by the time the slot frees up --
  with a single real slot, request 0 (3 steps) runs ticks 0-2; request 1 (2 steps, already waiting since tick 0) is admitted the instant the slot frees, at tick 3, and runs to tick 4; request 2 (1 step) does not arrive until tick 5, exactly when the slot is free again, so its own real queue wait is 0; total active-slot ticks, 6, again exactly equals the sum of needed_steps (3 + 2 + 1 = 6)

31/31 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] assuming continuous batching's own real benefit requires more available slots than static batching gets"
    It is tempting to think continuous batching wins simply because it is somehow more generous with GPU capacity. Test 4 shows this is not the mechanism at all: with only a single real slot -- the least generous possible configuration -- continuous batching still behaves correctly and wastes zero real capacity, because it degenerates cleanly to strict FIFO. The real advantage continuous batching provides over static batching, demonstrated directly in Test 2 on an IDENTICAL 2-slot workload, comes entirely from WHEN a finished sequence's own slot becomes available for reuse -- immediately, rather than only once an entire batch retires -- not from having access to any more real hardware than static batching already had.

## 27.4 Numerical Debugging Tools: NaN-Propagation Tracing and Floating-Point Drift Detection

### Intuition

A serving system running Section 27.3's own scheduler continuously, across unpredictable real traffic, eventually hits the numerical instabilities Section 26.2 warned about -- but at serving scale, the real question is no longer just "did a NaN appear," it is "where did it actually START," and a second, quieter real risk appears alongside it: continuous batching's own traffic-dependent regrouping of sequences can change the order a real reduction is computed in from run to run, and floating-point addition is not associative.

### The Concept, In Detail

`first_nan_layer` walks a real sequence of per-layer activation snapshots and reports the FIRST layer at which any element is non-finite -- not merely one of the layers where it happens to still be visible. Test 1 builds a real, honest 5-layer trace in which corruption genuinely originates at layer 2 (a real division-by-zero in a broken normalization step) and then propagates forward automatically through ordinary arithmetic at layers 3 and 4; `first_nan_layer` correctly reports layer 2, the real point of origin. Test 2 makes the practical stakes concrete: a naive system that only checks the FINAL layer for non-finite values gives an IDENTICAL "yes, something is broken" verdict whether corruption started at layer 2 or at layer 3 in a genuinely different trace -- only the real tracer actually distinguishes them.

Test 3 and Test 4 build this section's own real floating-point drift detector: summing a huge float32 value first, then ten small ones, silently loses all 10 of them to rounding, while summing the identical 10 small values together FIRST and only then adding the huge value preserves most of their real contribution -- a genuine, measurable 8-unit drift between two equally valid reduction orders over the identical input, confirmed directly in real, compiled `float` arithmetic rather than assumed from theory. Test 4 compares both real orders against the true double-precision value and confirms a real, general, actionable finding: summing small values before combining them with a much larger one is measurably more accurate, a concrete engineering choice relevant to any real serving system whose own batching schedule changes which order a batched reduction actually runs in.

### Code and Verification

```cpp
// Chapter 27.4 -- Real numerical debugging tools for catching the bugs that
// only appear at serving scale. Part 1 builds a real NaN/Inf-propagation
// tracer: given a real sequence of per-layer activation snapshots, it finds
// the layer where corruption FIRST appeared, not merely the layer where it
// happens to still be visible -- because once a real NaN exists, ordinary
// arithmetic propagates it forward through every downstream layer
// automatically, making "where is a NaN visible" a much weaker question
// than "where did it start." Part 2 builds a real floating-point
// drift-detection tool: real float32 addition is not associative, so
// continuous batching's own real, traffic-dependent regrouping of sequences
// into different batches across different runs can produce genuinely
// different (though both individually valid) reduction results -- this
// section quantifies that real drift directly rather than assuming it away.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_nan_propagation_tracing_and_fp_drift_detection.cpp -o 04_nan_propagation_tracing_and_fp_drift_detection
// Run:     ./04_nan_propagation_tracing_and_fp_drift_detection

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }
bool nearf(float a, float b, float eps = 1e-6f) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: real NaN/Inf-propagation tracing. Given a real sequence of
// per-layer activation snapshots (one vector<double> per layer, in real
// forward-pass order), find the FIRST layer at which any element is
// non-finite -- the real point of origin, which is what an engineer
// actually needs to know to fix the bug, as opposed to merely knowing that
// a NaN exists somewhere downstream.
// =======================================================================
bool has_non_finite(const std::vector<double>& layer) {
    for (double v : layer) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    return false;
}

std::optional<int> first_nan_layer(const std::vector<std::vector<double>>& layers) {
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (has_non_finite(layers[i])) return static_cast<int>(i);
    }
    return std::nullopt;
}

std::optional<std::size_t> first_non_finite_index(const std::vector<double>& layer) {
    for (std::size_t i = 0; i < layer.size(); ++i) {
        if (std::isnan(layer[i]) || std::isinf(layer[i])) return i;
    }
    return std::nullopt;
}

// A real, minimal simulation of how a NaN actually originates and then
// propagates: layer 2's own per-example normalization divides by
// (max_abs - max_abs), a genuine 0/0 whenever every element in that example
// shares the identical max_abs -- producing one real NaN at that layer.
// Layers 3 and 4 are then computed by ordinary elementwise arithmetic on
// the PREVIOUS layer's own output, exactly how a real forward pass would --
// so the NaN propagates forward automatically, without being reintroduced.
std::vector<std::vector<double>> build_realistic_propagation_trace() {
    std::vector<std::vector<double>> layers;
    layers.push_back({1.0, 2.0, 3.0});                 // layer 0: clean
    layers.push_back({2.0, 3.0, 4.0});                 // layer 1: clean

    // layer 2: a broken per-example normalization -- divide each element by
    // (max_abs - max_abs), which is genuinely 0.0 here since every element
    // in this toy example equals the same max_abs of 4.0.
    double max_abs = 4.0;
    double denom = max_abs - max_abs;  // == 0.0, a real division-by-zero setup
    std::vector<double> layer2;
    for (double v : layers[1]) layer2.push_back(v / denom);  // 2/0=inf, 3/0=inf, 4/0=inf -- but element 0 is special below
    layer2[0] = 0.0 / denom;  // 0.0 / 0.0 is a genuine NaN, not +-inf
    layers.push_back(layer2);

    // layer 3: ordinary elementwise arithmetic on layer 2's own output --
    // arithmetic with a NaN operand produces a NaN automatically, and
    // arithmetic with a real +-inf operand stays non-finite too.
    std::vector<double> layer3;
    for (double v : layers[2]) layer3.push_back(v * 0.5 + 1.0);
    layers.push_back(layer3);

    // layer 4: the same real propagation continues one layer further.
    std::vector<double> layer4;
    for (double v : layers[3]) layer4.push_back(v - 1.0);
    layers.push_back(layer4);

    return layers;
}

// =======================================================================
// PART 2: real floating-point drift detection via reduction order. Real
// float32 addition is not associative -- summing the identical real values
// in a different order can produce a genuinely different result, because
// each individual addition rounds to the nearest representable float32.
// This matters directly for continuous batching: which sequences a given
// real step groups together (Section 27.3) can change which order a
// reduction (a sum over a batch dimension, say) is actually computed in
// across different runs of the identical logical computation.
// =======================================================================
float sum_huge_first(float huge, const std::vector<float>& small_values) {
    float s = huge;
    for (float v : small_values) s = s + v;
    return s;
}

float sum_small_first_then_huge(float huge, const std::vector<float>& small_values) {
    float s = 0.0f;
    for (float v : small_values) s = s + v;
    return s + huge;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 27.4: Numerical Debugging Tools -- NaN-Propagation Tracing and FP Drift Detection\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a real 5-layer forward-pass trace in which corruption genuinely ORIGINATES at "
                 "layer 2 (a real division-by-zero in a broken normalization step) and then propagates "
                 "forward, unmodified in cause, through layers 3 and 4 purely via ordinary arithmetic -- "
                 "first_nan_layer correctly reports the real ORIGIN, layer 2, not merely one of the 3 "
                 "layers where corruption happens to still be visible --\n";
    {
        auto layers = build_realistic_propagation_trace();
        CHECK(layers.size() == 5);
        CHECK(!has_non_finite(layers[0]));
        CHECK(!has_non_finite(layers[1]));
        CHECK(has_non_finite(layers[2]));
        CHECK(has_non_finite(layers[3]));
        CHECK(has_non_finite(layers[4]));

        auto origin = first_nan_layer(layers);
        CHECK(origin.has_value());
        CHECK(*origin == 2);

        auto bad_index = first_non_finite_index(layers[2]);
        CHECK(bad_index.has_value());
        CHECK(*bad_index == 0);
        CHECK(std::isnan(layers[2][0]));
        CHECK(std::isinf(layers[2][1]));
        CHECK(std::isinf(layers[2][2]));

        std::cout << "  layers 0 and 1 are fully finite; layer 2's own broken normalization produces a "
                     "genuine NaN at index 0 (0.0 / 0.0) and genuine +inf at indices 1 and 2 (nonzero / "
                     "0.0); layers 3 and 4 are both non-finite too, purely because ordinary elementwise "
                     "arithmetic on a non-finite input stays non-finite -- first_nan_layer correctly "
                     "reports layer 2 as the real point of origin\n";
    }

    std::cout << "\n-- Test 2: checking ONLY the final layer for non-finite values -- the naive approach a "
                 "system with no real tracing tool might fall back on -- correctly detects THAT a real "
                 "problem exists, but provides none of first_nan_layer's own actionable information about "
                 "WHERE it actually started, which is the entire real point of building a tracer at all --\n";
    {
        auto layers = build_realistic_propagation_trace();
        bool final_layer_broken = has_non_finite(layers.back());
        CHECK(final_layer_broken);

        // The naive "check the last layer" approach cannot distinguish this real trace, where
        // corruption started at layer 2, from one where it started at layer 3 or layer 4 instead --
        // both would show an identical "yes, the final layer is broken" verdict.
        std::vector<std::vector<double>> alternate_trace = {
            {1.0, 2.0, 3.0},
            {2.0, 3.0, 4.0},
            {1.0, 1.0, 1.0},                       // layer 2 clean this time
            {std::nan(""), 1.0, 1.0},               // corruption instead originates at layer 3
            {std::nan(""), 1.0, 1.0},               // and simply propagates to layer 4
        };
        CHECK(has_non_finite(alternate_trace.back()) == final_layer_broken);  // identical naive verdict
        CHECK(*first_nan_layer(alternate_trace) == 3);                        // but a genuinely different real origin
        std::cout << "  both this section's own real trace (origin at layer 2) and a genuinely different "
                     "alternate trace (origin at layer 3) show an IDENTICAL \"yes\" verdict under a naive "
                     "final-layer-only check -- first_nan_layer is what actually distinguishes them, "
                     "correctly reporting layer 2 for the first trace and layer 3 for the second\n";
    }

    std::cout << "\n-- Test 3: real float32 reduction order genuinely changes the result -- summing a huge "
                 "value first, then ten real small values, loses all 10 of them to rounding, while summing "
                 "the identical 10 small values together FIRST, then adding the huge value, preserves most "
                 "of their real contribution -- a real, measurable, nonzero 8.0-unit drift between two "
                 "equally valid reduction orders over the identical real input values --\n";
    {
        float huge = 1e8f;
        std::vector<float> small_values(10, 1.0f);

        float order_a = sum_huge_first(huge, small_values);
        float order_b = sum_small_first_then_huge(huge, small_values);

        CHECK(nearf(order_a, 100000000.0f));
        CHECK(nearf(order_b, 100000008.0f));

        float drift = order_a - order_b;
        CHECK(nearf(drift, -8.0f));
        CHECK(!nearf(order_a, order_b));  // genuinely, measurably different -- not merely a rounding artifact

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  order A (huge first, then each 1.0f added sequentially) rounds to exactly "
                  << order_a << ", silently losing all 10 real additions -- each individual 1.0f is too "
                     "small to move a float32 near 1e8 to the next representable value; order B (the 10 "
                     "small values summed together first, THEN added to the huge value) rounds to "
                  << order_b << " instead, preserving most of their real combined contribution; the real "
                     "drift between these two equally valid reduction orders is exactly " << drift << "\n";
        std::cout << std::defaultfloat << std::setprecision(6);
    }

    std::cout << "\n-- Test 4: comparing both real reduction orders against the true double-precision value "
                 "confirms neither float32 order is exactly correct, but summing small values first is "
                 "genuinely, measurably closer to the truth than summing the huge value first -- a real, "
                 "actionable finding for a serving system whose own continuous-batching schedule changes "
                 "which order a batched reduction is actually computed in from run to run --\n";
    {
        float huge = 1e8f;
        std::vector<float> small_values(10, 1.0f);
        double true_value = 1e8 + 10.0;

        float order_a = sum_huge_first(huge, small_values);
        float order_b = sum_small_first_then_huge(huge, small_values);

        double error_a = std::fabs(static_cast<double>(order_a) - true_value);
        double error_b = std::fabs(static_cast<double>(order_b) - true_value);

        CHECK(near(error_a, 10.0));
        CHECK(near(error_b, 2.0));
        CHECK(error_b < error_a);

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  against the true double-precision value of " << true_value << ", order A's own "
                     "real error is " << error_a << " while order B's own real error is only " << error_b
                  << " -- summing small values together before combining them with a much larger value is "
                     "a real, general strategy for reducing accumulated floating-point error, not merely "
                     "an artifact of this section's own specific chosen numbers\n";
        std::cout << std::defaultfloat << std::setprecision(6);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_nan_propagation_tracing_and_fp_drift_detection.cpp -o 04_nan_propagation_tracing_and_fp_drift_detection
./04_nan_propagation_tracing_and_fp_drift_detection
```

**Sample input:** a real 5-layer forward-pass trace, in which corruption genuinely originates at layer 2 and propagates forward through ordinary arithmetic, checked to have its real point of origin correctly identified rather than merely detected downstream; a naive final-layer-only check confirmed to give an identical verdict on two traces with genuinely different real origins; real float32 reduction-order drift confirmed directly in compiled arithmetic between two equally valid summation orders; and both real orders compared against the true double-precision value to confirm which is genuinely more accurate.

```text
========================================================
Chapter 27.4: Numerical Debugging Tools -- NaN-Propagation Tracing and FP Drift Detection
========================================================

-- Test 1: a real 5-layer forward-pass trace in which corruption genuinely ORIGINATES at layer 2 (a real division-by-zero in a broken normalization step) and then propagates forward, unmodified in cause, through layers 3 and 4 purely via ordinary arithmetic -- first_nan_layer correctly reports the real ORIGIN, layer 2, not merely one of the 3 layers where corruption happens to still be visible --
  layers 0 and 1 are fully finite; layer 2's own broken normalization produces a genuine NaN at index 0 (0.0 / 0.0) and genuine +inf at indices 1 and 2 (nonzero / 0.0); layers 3 and 4 are both non-finite too, purely because ordinary elementwise arithmetic on a non-finite input stays non-finite -- first_nan_layer correctly reports layer 2 as the real point of origin

-- Test 2: checking ONLY the final layer for non-finite values -- the naive approach a system with no real tracing tool might fall back on -- correctly detects THAT a real problem exists, but provides none of first_nan_layer's own actionable information about WHERE it actually started, which is the entire real point of building a tracer at all --
  both this section's own real trace (origin at layer 2) and a genuinely different alternate trace (origin at layer 3) show an IDENTICAL "yes" verdict under a naive final-layer-only check -- first_nan_layer is what actually distinguishes them, correctly reporting layer 2 for the first trace and layer 3 for the second

-- Test 3: real float32 reduction order genuinely changes the result -- summing a huge value first, then ten real small values, loses all 10 of them to rounding, while summing the identical 10 small values together FIRST, then adding the huge value, preserves most of their real contribution -- a real, measurable, nonzero 8.0-unit drift between two equally valid reduction orders over the identical real input values --
  order A (huge first, then each 1.0f added sequentially) rounds to exactly 100000000.0, silently losing all 10 real additions -- each individual 1.0f is too small to move a float32 near 1e8 to the next representable value; order B (the 10 small values summed together first, THEN added to the huge value) rounds to 100000008.0 instead, preserving most of their real combined contribution; the real drift between these two equally valid reduction orders is exactly -8.0

-- Test 4: comparing both real reduction orders against the true double-precision value confirms neither float32 order is exactly correct, but summing small values first is genuinely, measurably closer to the truth than summing the huge value first -- a real, actionable finding for a serving system whose own continuous-batching schedule changes which order a batched reduction is actually computed in from run to run --
  against the true double-precision value of 100000010.0, order A's own real error is 10.0 while order B's own real error is only 2.0 -- summing small values together before combining them with a much larger value is a real, general strategy for reducing accumulated floating-point error, not merely an artifact of this section's own specific chosen numbers

23/23 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating floating-point drift between two valid reduction orders as a genuine correctness bug"
    Test 3's own two real reduction orders produce genuinely different results -- 100000000.0 versus 100000008.0 -- and it is tempting to treat this as a bug that needs fixing. Test 4 shows the more accurate real framing: NEITHER order is exactly correct against the true double-precision value, and the difference between them is an expected, unavoidable consequence of float32's own limited real precision combined with a specific real choice of reduction order, not a sign either individual result is wrong in isolation. The genuinely actionable real finding is not "eliminate the drift" -- that is not possible while still using float32 -- it is "prefer the reduction order that is measurably closer to the truth," and a serving system whose own continuous batching changes reduction order across runs should at minimum know how large that drift can get, which is exactly what this section's own drift detector quantifies directly.

## Chapter Summary

This chapter built the real scheduling layer a production inference engine actually needs to serve unpredictable, concurrent traffic efficiently. Section 27.1 quantified static batching's own real waste with an exact hand-traceable simulator: real padding waste from grouping requests of different lengths together, and a real, substantial head-of-line blocking delay for any short request unlucky enough to arrive just after a long batch has already started. Section 27.2 resolved the real conflict between prefill's compute-bound cost and decode's memory-bound cost with chunked prefill, deriving a real, provable bound on injected per-step latency that holds regardless of prompt length. Section 27.3 combined both lessons into a real, from-scratch continuous-batching scheduler, checked directly against Section 27.1's own restated static-batching simulator on the identical workload rather than against a hardcoded comparison number, and confirmed its own real zero-padding-waste guarantee generalizes to a second, different workload. Section 27.4 closed the chapter with two real numerical debugging tools this book's own increasingly complex serving pipeline actually needs: a NaN-propagation tracer that finds where corruption genuinely originated, and a floating-point drift detector that quantifies exactly how much a real reduction's result can change when continuous batching changes its own reduction order.

## Self-Check Questions

1. Section 27.1's Test 3 shows a 1-step request delayed by a real 11-step head-of-line blocking penalty. Explain concretely why this delay is NOT simply equal to however many steps the request had to wait in a literal first-come-first-served queue.
2. Section 27.1's Test 4 shows utilization is exactly 1.0 at batch_size 1. Explain what specifically about a batch containing only one request makes padding waste structurally impossible, regardless of how long or short that one request happens to be.
3. Section 27.2 models decode's own real cost as scaling with the number of concurrently active sequences, and prefill's own real cost as scaling with the number of new prompt tokens. Explain, in your own words, why these are genuinely different real bottlenecks (memory-bound versus compute-bound) rather than the same cost measured two different ways.
4. Section 27.2's Test 3 shows chunked prefill conserves the TOTAL real extra work exactly, while Test 4 shows it bounds the WORST per-step injected latency. Explain why a system could care about the second property even though the first shows chunking provides no reduction in the first.
5. Section 27.3's Test 2 checks continuous batching against static batching using a LIVE restated simulator rather than a hardcoded comparison number. Explain one concrete real risk a hardcoded comparison number would have introduced that the live simulator avoids.
6. Section 27.3's Test 4 uses a workload with only a single real slot to test the scheduler. Explain what specific real scheduling behavior this single-slot case is actually able to verify that a multi-slot workload alone would not clearly isolate.
7. Section 27.4's Test 2 shows a naive final-layer-only NaN check gives an identical verdict on two traces with genuinely different real points of origin. Explain concretely what real debugging information is lost by only checking the final layer.
8. Section 27.4's Test 1 constructs a NaN via `0.0 / 0.0` and separate `+inf` values via `nonzero / 0.0` in the SAME broken layer. Explain why `first_non_finite_index` still correctly reports the same layer as `first_nan_layer` despite these being two different kinds of non-finite values.
9. Section 27.4's Test 3 and Test 4 show summing small values before a much larger one is measurably more accurate than the reverse order. Connect this concretely to a real risk introduced by Section 27.3's own continuous-batching scheduler, which regroups sequences into different real batches depending on real, unpredictable traffic.
10. This chapter's title pairs "continuous batching" with "production serving architecture." Choose any ONE of this chapter's 4 sections and explain concretely why the real property it derives or builds would matter LESS, or not apply at all, to a system that only ever serves a single request at a time with no concurrent traffic.

## Where We Go Next

This chapter built the real scheduling and debugging layer a serving system needs on the CPU. The final chapter takes this book's own inference engine to the GPU: Chapter 28 builds a real Flash Attention implementation around the same online-softmax idea Section 26.2's own numerically stable softmax introduced, fixing the O(N^2) memory wall standard attention hits, and closes the book with a real CUDA production engine -- complete with its own kernel-validation suite -- for the edge devices, Jetson-class boards among them, that do carry a small GPU.

## Worked Solutions

**1.** A literal first-come-first-served queue delay would only count the time a request spends waiting BEHIND other requests that are also merely waiting. Section 27.1's own real penalty is different: once a batch has started, a newly arrived request cannot be admitted at all, no matter how few other requests are ahead of it in real arrival order, until every member of the ALREADY-RUNNING batch finishes -- so the delay is driven by how much longer the running batch's own longest member still needs, not by how many requests are queued.

**2.** A batch's own real makespan is defined as the maximum needed_steps among its members; with exactly one member, that maximum IS that member's own needed_steps, so the batch's own total slot-steps (batch_size times makespan, here 1 times needed_steps) always equals that same member's own needed_steps exactly -- there is no other, shorter member left in the batch whose slot could ever sit idle waiting for a longer one, since there is only ever one member to begin with.

**3.** Decode's own real cost is dominated by reading each active sequence's own KV cache from memory once per step -- more concurrent sequences means more real memory traffic, but the actual compute per token is comparatively small (a GEMV). Prefill's own real cost is dominated by the actual compute needed to process every token of a new prompt against the model's own weights at once (closer to a GEMM) -- more prompt tokens means more real FLOPs, while the memory traffic for reading those (shared) weights barely changes. These are genuinely different physical bottlenecks -- one is bound by how fast memory can be read, the other by how fast the processor can compute -- not the same underlying cost expressed in two different units.

**4.** Chunking conserving the total real work means a system gains nothing in raw aggregate throughput from chunking alone -- the same total amount of prefill compute still has to happen somewhere. What a system gains instead is PREDICTABILITY: bounding the worst single-step latency spike protects every already-running decode sequence sharing that step from an unbounded, traffic-dependent latency spike, which matters directly for real-time serving guarantees (a maximum acceptable per-token latency) even when it does not improve the system's own total steady-state throughput number.

**5.** A hardcoded comparison number would only ever be correct for the ONE specific workload it was computed for by hand; if that workload's own numbers were ever changed even slightly (a different arrival time, a different needed_steps value), the hardcoded number would silently become stale and wrong, while a passing test would keep reporting success. Running Section 27.1's own restated simulator live means the comparison is recomputed correctly every single time the workload changes, catching a real discrepancy immediately rather than only when someone remembers to update a hardcoded expectation by hand.

**6.** A multi-slot workload could pass its own tests purely because ONE of its several slots happened to schedule correctly, while a bug in the admission logic that only manifests when there is exactly one slot to reason about (an off-by-one in how "the" free slot, rather than "a" free slot among several, gets selected) could go completely undetected. The single-slot workload isolates admission-and-retirement correctness with no other slots' own scheduling behavior to potentially mask a real bug in that specific code path.

**7.** Checking only the final layer answers a single yes-or-no question -- "is anything broken by the time the forward pass finishes" -- and provides no information about which of potentially many upstream layers actually introduced the problem. An engineer debugging a real production failure needs to know WHERE to look in the model's own code (which layer's own computation is actually broken), and a final-layer-only check gives them nothing beyond confirming that a problem exists somewhere in the entire pipeline, which they very likely already knew from a garbled model output.

**8.** `first_non_finite_index` simply scans a single layer's own vector for the FIRST position where either `std::isnan` or `std::isinf` returns true, treating both as equally "non-finite" -- it does not need to distinguish which of the two kinds of non-finite value it finds, since both are equally real evidence that something already went wrong in that layer's own computation. Because index 0 in this section's own broken layer holds a genuine NaN and index 0 is scanned first, `first_non_finite_index` reports index 0 regardless of the fact that indices 1 and 2 in that same layer hold a different, real kind of non-finite value (+inf) rather than NaN.

**9.** Section 27.3's own scheduler admits and retires sequences based on real, unpredictable arrival times, which means the SET of sequences batched together at a given step -- and therefore the order in which a batched reduction (say, summing values across that batch dimension) actually gets computed -- can differ from one real run to the next, even for logically identical requests, purely because traffic happened to arrive in a different order or at different times. Section 27.4's own Test 3 and Test 4 show this kind of reduction-order change is not merely a cosmetic difference: it can produce a real, measurably different numerical result, so a system relying on continuous batching should not assume repeated runs of the same logical workload will produce bit-identical numerical output.

**10.** Section 27.1's own entire real contribution -- quantifying wasted padding and head-of-line blocking -- assumes MULTIPLE real requests are competing for the same limited batch slots at overlapping times. A system serving exactly one request at a time, with no concurrent traffic ever, has no other request to be blocked behind and no padding to waste, since there is only ever the one real occupant of the one slot it needs; static batching's own entire failure mode, and therefore the reason continuous batching improves on it, simply does not arise when there is no real concurrency for a scheduler to have to arbitrate in the first place.
