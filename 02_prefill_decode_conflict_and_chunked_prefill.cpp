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
