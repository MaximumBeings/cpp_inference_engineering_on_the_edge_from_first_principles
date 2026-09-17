// Chapter 27.2 -- A real electronic exchange's own "tick-to-trade" budget
// is a hard, real constraint: a venue-imposed (or self-imposed, to stay
// competitive) ceiling on how many microseconds may elapse between a
// market event arriving and this system's own resulting order reaching
// the exchange. Missing that budget does not produce a slightly-late
// answer the way a slow batch job would -- it produces an answer that is
// no longer safe to act on at all, since the book state it was computed
// from may have already moved.
//
// Exactly like Chapter 10.1's own real derivation of a single decode
// step's latency against a stated interactive budget, EVERY microsecond
// value in this file is a STATED, labeled parameter -- a per-stage
// processing cost, an event's own stated arrival timestamp, a stated
// staleness limit -- never a value measured from `std::chrono` or any
// other wall clock. This is a deliberate, permanent choice, not a
// simplification: a real wall-clock measurement is nondeterministic
// across runs and across architectures (exactly the reason Chapter 10's
// own threading chapter title promises verification "without a clock"),
// and this book's own locked, byte-for-byte self-test output would
// silently break the moment any timing value in it depended on how fast
// the specific machine running the test happened to be.
//
// A note on this section's own honest scope: this file's own stage costs
// are STATED, illustrative architectural parameters -- exactly like
// Chapter 8.1's own illustrative CPU peak-bandwidth figures -- not a
// measurement of any specific real exchange's own real infrastructure.
// This section's own COMMON TRAP box returns to exactly this limitation.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_tick_to_trade_deadline_enforcement.cpp -o 02_tick_to_trade_deadline_enforcement
// Run:     ./02_tick_to_trade_deadline_enforcement

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// A single named pipeline stage and its own STATED microsecond cost --
// never measured, always a labeled architectural parameter.
struct PipelineStage {
    std::string name;
    std::int64_t cost_us;
};

// A real market event's own stated arrival timestamp (microseconds since
// some fixed epoch this file never needs to name).
struct MarketEvent {
    std::string id;
    std::int64_t arrival_time_us;
};

struct PipelineResult {
    bool rejected_stale = false;      // event was already too old before the pipeline even started
    bool missed_deadline = false;     // the pipeline itself exceeded the budget mid-run
    std::string stage_that_missed;    // empty unless missed_deadline is true
    std::int64_t total_elapsed_us = 0;
    std::vector<std::string> completed_stages;
};

// If a market event's own age (relative to a stated current time) already
// exceeds a stated staleness limit, this system does not even attempt the
// pipeline -- spending real processing time on a book state that is
// already known to be too old to act on safely would be strictly worse
// than honestly rejecting it up front. Otherwise, each stage's own STATED
// cost accumulates in order; the moment the running total exceeds the
// stated budget, the pipeline stops immediately -- it does not run
// remaining stages just to see how late the final answer would have been.
PipelineResult run_pipeline(const MarketEvent& event,
                             std::int64_t current_time_us,
                             std::int64_t staleness_limit_us,
                             const std::vector<PipelineStage>& stages,
                             std::int64_t budget_us) {
    PipelineResult r;
    const std::int64_t age_us = current_time_us - event.arrival_time_us;
    if (age_us > staleness_limit_us) {
        r.rejected_stale = true;
        r.missed_deadline = true;
        return r;
    }
    std::int64_t elapsed = 0;
    for (const auto& stage : stages) {
        elapsed += stage.cost_us;
        if (elapsed > budget_us) {
            r.missed_deadline = true;
            r.stage_that_missed = stage.name;
            r.total_elapsed_us = elapsed;
            return r;
        }
        r.completed_stages.push_back(stage.name);
    }
    r.total_elapsed_us = elapsed;
    return r;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 27.2: The Latency Budget -- Tick-to-Trade Deadline Enforcement\n";
    std::cout << "========================================================\n\n";

    const std::vector<PipelineStage> stages = {
        {"parse_market_data", 80},
        {"compute_features", 120},
        {"decide_signal", 150},
        {"send_order", 90},
    };
    const MarketEvent event{"EVT-1001", 1'000'000};

    // -- Test 1: every stage completes comfortably within a generous
    // budget -- the pipeline finishes normally, naming all four stages in
    // the order they completed, with the exact summed elapsed time. --
    {
        PipelineResult r = run_pipeline(event, 1'000'000, 500, stages, 500);
        std::cout << "-- Test 1: 4 stages (80+120+150+90=440us), budget=500us -- missed="
                  << (r.missed_deadline ? "YES" : "NO") << ", elapsed=" << r.total_elapsed_us
                  << "us, completed=" << r.completed_stages.size() << " stages --\n";
        CHECK(!r.missed_deadline);
        CHECK(!r.rejected_stale);
        CHECK(r.total_elapsed_us == 440);
        CHECK(r.completed_stages.size() == 4);
    }

    // -- Test 2: the budget set to EXACTLY the total stage cost (440us).
    // A pipeline that lands exactly ON the budget has not exceeded it --
    // this section's own stated convention is a STRICT inequality
    // (elapsed > budget), confirmed here at the exact boundary itself,
    // not merely a value comfortably inside it. --
    {
        PipelineResult r = run_pipeline(event, 1'000'000, 500, stages, 440);
        std::cout << "-- Test 2: same 4 stages, budget=440us (exactly the total cost) -- missed="
                  << (r.missed_deadline ? "YES" : "NO") << ", elapsed=" << r.total_elapsed_us << "us --\n";
        CHECK(!r.missed_deadline);
        CHECK(r.total_elapsed_us == 440);
        CHECK(r.completed_stages.size() == 4);
    }

    // -- Test 3: one microsecond under the total stage cost. The FINAL
    // stage is the one whose own cost pushes the running total past the
    // budget -- it is named exactly, and its own cost IS counted into the
    // elapsed total (the time was genuinely spent before the breach was
    // noticed), but it is NOT added to the completed-stages list, since
    // it never actually finished within budget. --
    {
        PipelineResult r = run_pipeline(event, 1'000'000, 500, stages, 439);
        std::cout << "-- Test 3: same 4 stages, budget=439us (1us short) -- missed="
                  << (r.missed_deadline ? "YES" : "NO") << ", stage_that_missed=\"" << r.stage_that_missed
                  << "\", elapsed=" << r.total_elapsed_us << "us, completed=" << r.completed_stages.size()
                  << " --\n";
        CHECK(r.missed_deadline);
        CHECK(r.stage_that_missed == "send_order");
        CHECK(r.total_elapsed_us == 440);
        CHECK(r.completed_stages.size() == 3);
    }

    // -- Test 4: an event that arrived too long ago -- its own age already
    // exceeds a stated staleness limit before the pipeline is even
    // attempted. No stage runs at all; this is a deliberate, honest
    // up-front rejection, not a wasted attempt. --
    {
        PipelineResult r = run_pipeline(event, /*current_time_us=*/1'000'900, /*staleness_limit_us=*/500,
                                         stages, 500);
        std::cout << "-- Test 4: event age = 900us against a staleness limit of 500us -- rejected_stale="
                  << (r.rejected_stale ? "YES" : "NO") << ", completed=" << r.completed_stages.size()
                  << " stages, elapsed=" << r.total_elapsed_us << "us --\n";
        CHECK(r.rejected_stale);
        CHECK(r.missed_deadline);
        CHECK(r.completed_stages.empty());
        CHECK(r.total_elapsed_us == 0);
    }

    // -- Test 5: the staleness check's own boundary -- an event whose age
    // is EXACTLY equal to the staleness limit is not yet stale (the same
    // strict-inequality convention as the deadline check itself), so the
    // pipeline proceeds normally. --
    {
        PipelineResult r = run_pipeline(event, /*current_time_us=*/1'000'500, /*staleness_limit_us=*/500,
                                         stages, 500);
        std::cout << "-- Test 5: event age = 500us, exactly at the staleness limit of 500us -- rejected_stale="
                  << (r.rejected_stale ? "YES" : "NO") << " --\n";
        CHECK(!r.rejected_stale);
        CHECK(!r.missed_deadline);
    }

    // -- Test 6: a budget of zero -- even the very first stage's own
    // nonzero cost breaches it immediately. Confirms the breach is
    // detected at the FIRST stage, not merely "some" stage, when the
    // very first one is already too expensive on its own. --
    {
        PipelineResult r = run_pipeline(event, 1'000'000, 500, stages, 0);
        std::cout << "-- Test 6: budget=0us -- missed=" << (r.missed_deadline ? "YES" : "NO")
                  << ", stage_that_missed=\"" << r.stage_that_missed << "\", completed="
                  << r.completed_stages.size() << " --\n";
        CHECK(r.missed_deadline);
        CHECK(r.stage_that_missed == "parse_market_data");
        CHECK(r.completed_stages.empty());
        CHECK(r.total_elapsed_us == 80);
    }

    // -- Test 7: the breach must be reported at the FIRST stage where the
    // running total crosses the budget, not the last -- constructed here
    // so an EARLY, expensive stage alone already exceeds the budget, even
    // though later stages are individually far cheaper and would have
    // fit comfortably in isolation. --
    {
        const std::vector<PipelineStage> front_loaded = {
            {"expensive_first_stage", 300},
            {"cheap_second_stage", 50},
            {"cheap_third_stage", 50},
        };
        PipelineResult r = run_pipeline(event, 1'000'000, 500, front_loaded, 250);
        std::cout << "-- Test 7: front-loaded stages (300, 50, 50), budget=250us -- missed="
                  << (r.missed_deadline ? "YES" : "NO") << ", stage_that_missed=\"" << r.stage_that_missed
                  << "\", elapsed=" << r.total_elapsed_us << "us --\n";
        CHECK(r.missed_deadline);
        CHECK(r.stage_that_missed == "expensive_first_stage");
        CHECK(r.total_elapsed_us == 300);
        CHECK(r.completed_stages.empty());
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
