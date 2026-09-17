# Chapter 27: Low-Latency Market-Signal Inference Near the Exchange

**What you will understand by the end of this chapter:**

- How to compute a real order book's mid-price, size-weighted "microprice," and order-book imbalance from a Level 1 top-of-book snapshot -- the same real, published market-microstructure signals a real market-making or execution system computes on every book update.
- How to derive a hard, real-time tick-to-trade latency budget entirely from stated architectural parameters -- never a wall clock -- reapplying Chapter 10's own "verified without a clock" discipline to a genuinely different constraint: a deadline, rather than a throughput target.
- How to build a real, finance-specific sentiment lexicon modeled on the real, published Loughran-McDonald methodology, and why a generic English sentiment dictionary badly misclassifies routine financial vocabulary.
- How to fuse multiple independent real signals into one auditable trading decision, reapplying Chapter 26.4's own never-suppress-a-named-flag structure -- but inverted, so a missed deadline forces the SAFEST disposition rather than the strictest one.

**What you need to know first:**

- Chapter 10.1's own discipline of deriving timing behavior entirely from stated, labeled parameters rather than a wall clock is reapplied in Section 27.2, this time to a hard deadline rather than a throughput ceiling.
- Chapter 26.4's own weighted-signal-plus-forced-override structure (this book's recurring never-suppress-a-flag discipline) is reapplied in Section 27.4 -- this time forcing HOLD, the safe disposition, rather than forcing escalation, whenever a signal missed its own stated deadline.
- Section 27.1's own order-book formulas are real, published market-microstructure quantities, not something invented for this book; Section 27.3's own sentiment lexicon is modeled on a real, cited academic methodology (Loughran and McDonald, 2011), not the full real published word list.

---

Chapter 26 turned to a bank branch or back office processing checks and invoices, where a fraud pattern -- once caught -- has no real time pressure attached to catching it a few seconds later. This chapter turns to a related but genuinely different real financial setting: inference on live market-signal data near the exchange, where the central constraint is not pattern detection at all, but a hard, real-time deadline measured in microseconds. Each of this chapter's four sections builds one real, independent piece of that pipeline -- order-book features, a deadline enforcement mechanism, a finance-specific sentiment score -- and closes with a capstone that fuses all three into one auditable trading decision, where a signal that arrived too late to act on safely is treated as no signal at all.

## 27.1 Order Book Microstructure Features

### Intuition

A real electronic exchange's own top-of-book -- the single best bid, best ask, and their own displayed sizes -- already carries real, computable signal before any history or model is involved. The mid-price is the obvious first cut, but the "microprice" is a more informative real quantity: it weights each side's price by the OPPOSITE side's own displayed size, so that heavy buying pressure (a large bid size resting against a thin ask) pulls the microprice up toward the ask, reflecting the real expectation that the thin side is more likely to be consumed next. Order-book imbalance captures the same asymmetry as a single bounded number.

### The Concept, In Detail

`compute_book_features` first checks for two real, honest failure states before computing anything: a CROSSED book (best bid at or above best ask, a real if rare market-data anomaly) and NO_LIQUIDITY (zero displayed size on both sides at once, which would otherwise divide by zero). Tests 4 and 5 confirm both are reported as an explicit `BookState`, never a silently wrong number. When the book is valid, `imbalance` is `(bid_size - ask_size) / (bid_size + ask_size)`, and `microprice` is `(ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)` -- Tests 2 and 3 confirm the real economic direction concretely: heavy bid-side pressure pulls the microprice strictly above the mid-price, toward the ask, and heavy ask-side pressure pulls it strictly below, toward the bid.

Test 6 confirms a real, legitimate one-sided edge case -- zero displayed size on exactly one side, not both -- correctly saturates imbalance at exactly +1.0 and reduces the microprice to exactly that side's own opposite price, without any special-casing beyond the single zero-liquidity guard already in place. Test 7 confirms the same formulas generalize cleanly to a general asymmetric book at entirely different price and size levels.

### Code and Verification

```cpp
// Chapter 27.1 -- Before any latency budget or trading decision matters at
// all, a real electronic exchange's own top-of-book (best bid, best ask,
// and their displayed sizes) already carries three real, published market-
// microstructure signals that require no history and no model to compute:
// the mid-price, the size-weighted "microprice," and the order-book
// imbalance. None of these three formulas are invented for this book --
// they are the same real quantities a real market-making or execution
// system computes on every single book update, exactly as real as the
// American Bankers Association's own routing-number checksum Chapter
// 26.1 implemented for a completely different real domain.
//
// The microprice in particular has a real, well-documented economic
// intuition worth stating precisely: it weights EACH side's own price by
// the OPPOSITE side's displayed size, so that a book with much more size
// resting on the bid than the ask is pulled toward the ASK price -- real
// buying pressure, reflected in a thin ask queue, is read as more likely
// to push the traded price UP toward that thin ask, not down toward the
// heavy bid. This section's own tests confirm that direction concretely,
// not just its formula.
//
// A note on this section's own honest scope: every function in this file
// operates on TOP-OF-BOOK ONLY (the single best bid and best ask level).
// A real limit order book has many price levels beneath the top, and a
// large hidden ("iceberg") order can sit at the top level without ever
// appearing in its own displayed size -- this section's own COMMON TRAP
// box returns to exactly this limitation.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_order_book_microstructure_features.cpp -o 01_order_book_microstructure_features
// Run:     ./01_order_book_microstructure_features

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
static bool near_eq(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }
#define CHECK_NEAR(a, b) CHECK(near_eq((a), (b)))

// =======================================================================
// A single price level (one side's best quote) and a top-of-book snapshot.
// Prices are real currency values (dollars); sizes are real displayed
// share counts. Both are exactly the fields a real Level 1 market-data
// feed publishes on every top-of-book update.
// =======================================================================
struct PriceLevel {
    double price;
    std::int64_t size;
};

struct BookSnapshot {
    PriceLevel bid;
    PriceLevel ask;
};

enum class BookState { OK, CROSSED, NO_LIQUIDITY };

struct BookFeatures {
    BookState state = BookState::OK;
    double mid_price = 0.0;
    double microprice = 0.0;
    double spread = 0.0;
    double imbalance = 0.0;  // in [-1, +1]; +1 = all displayed size on the bid
};

// A "crossed" book (best bid at or above best ask) is a real, if rare,
// market-data anomaly -- never a valid state to compute a signal from.
// A book with zero displayed size on BOTH sides carries no real
// microstructure information at all. Both states are reported honestly
// as an explicit BookState rather than silently returning 0.0 or NaN.
BookFeatures compute_book_features(const BookSnapshot& book) {
    BookFeatures f;
    if (book.bid.price >= book.ask.price) {
        f.state = BookState::CROSSED;
        return f;
    }
    const std::int64_t total_size = book.bid.size + book.ask.size;
    if (total_size == 0) {
        f.state = BookState::NO_LIQUIDITY;
        return f;
    }
    f.state = BookState::OK;
    f.mid_price = (book.bid.price + book.ask.price) / 2.0;
    f.spread = book.ask.price - book.bid.price;
    f.imbalance = static_cast<double>(book.bid.size - book.ask.size) / static_cast<double>(total_size);
    // Microprice: each side's price weighted by the OPPOSITE side's size.
    f.microprice = (book.ask.price * static_cast<double>(book.bid.size) +
                     book.bid.price * static_cast<double>(book.ask.size)) /
                    static_cast<double>(total_size);
    return f;
}

static const char* state_name(BookState s) {
    switch (s) {
        case BookState::OK: return "OK";
        case BookState::CROSSED: return "CROSSED";
        case BookState::NO_LIQUIDITY: return "NO_LIQUIDITY";
    }
    return "UNKNOWN";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 27.1: Order Book Microstructure Features\n";
    std::cout << "========================================================\n\n";
    std::cout << std::fixed << std::setprecision(4);

    // -- Test 1: a perfectly balanced book (equal size on both sides).
    // The microprice must reduce EXACTLY to the mid-price, and imbalance
    // must be exactly 0.0, when there is no size asymmetry at all. --
    {
        BookSnapshot book{{100.00, 500}, {100.02, 500}};
        BookFeatures f = compute_book_features(book);
        std::cout << "-- Test 1: balanced book (bid 100.00x500, ask 100.02x500) -- state="
                  << state_name(f.state) << ", mid=" << f.mid_price << ", microprice=" << f.microprice
                  << ", spread=" << f.spread << ", imbalance=" << f.imbalance << " --\n";
        CHECK(f.state == BookState::OK);
        CHECK_NEAR(f.mid_price, 100.01);
        CHECK_NEAR(f.spread, 0.02);
        CHECK_NEAR(f.imbalance, 0.0);
        CHECK_NEAR(f.microprice, f.mid_price);
    }

    // -- Test 2: heavy BID-side size pressure (900 vs. 100). The real
    // economic prediction is that the microprice is pulled UP, toward the
    // thin ask, away from the mid-price -- confirmed as a strict
    // inequality against Test 1's own balanced mid-price logic, not just
    // a formula match. --
    {
        BookSnapshot book{{100.00, 900}, {100.02, 100}};
        BookFeatures f = compute_book_features(book);
        std::cout << "-- Test 2: heavy bid pressure (bid 100.00x900, ask 100.02x100) -- imbalance="
                  << f.imbalance << ", microprice=" << f.microprice << " (pulled toward the ask) --\n";
        CHECK_NEAR(f.imbalance, 0.8);
        CHECK_NEAR(f.microprice, 100.018);
        CHECK(f.microprice > f.mid_price);
    }

    // -- Test 3: heavy ASK-side size pressure -- the mirror image of Test
    // 2, pulling the microprice DOWN toward the thin bid instead. --
    {
        BookSnapshot book{{100.00, 100}, {100.02, 900}};
        BookFeatures f = compute_book_features(book);
        std::cout << "-- Test 3: heavy ask pressure (bid 100.00x100, ask 100.02x900) -- imbalance="
                  << f.imbalance << ", microprice=" << f.microprice << " (pulled toward the bid) --\n";
        CHECK_NEAR(f.imbalance, -0.8);
        CHECK_NEAR(f.microprice, 100.002);
        CHECK(f.microprice < f.mid_price);
    }

    // -- Test 4: a crossed book (best bid at or above best ask) is a real
    // market-data anomaly, never a valid input to compute a signal from --
    // reported as an explicit CROSSED state, not a silently wrong number. --
    {
        BookSnapshot book{{100.05, 500}, {100.00, 500}};
        BookFeatures f = compute_book_features(book);
        std::cout << "-- Test 4: crossed book (bid 100.05 >= ask 100.00) -- state=" << state_name(f.state)
                  << " --\n";
        CHECK(f.state == BookState::CROSSED);
    }

    // -- Test 5: both sides showing zero displayed size at once carries no
    // real microstructure information and must never divide by zero --
    // reported as an explicit NO_LIQUIDITY state. --
    {
        BookSnapshot book{{100.00, 0}, {100.02, 0}};
        BookFeatures f = compute_book_features(book);
        std::cout << "-- Test 5: zero displayed size on both sides -- state=" << state_name(f.state)
                  << " --\n";
        CHECK(f.state == BookState::NO_LIQUIDITY);
    }

    // -- Test 6: a real, legitimate one-sided edge case -- the ask side
    // shows genuinely zero displayed size while the bid does not (a real,
    // if extreme, book state, not an error). Imbalance saturates at
    // exactly +1.0, and the microprice reduces to exactly the ask price
    // itself, since the entire weighted average is placed on it. --
    {
        BookSnapshot book{{100.00, 600}, {100.02, 0}};
        BookFeatures f = compute_book_features(book);
        std::cout << "-- Test 6: zero size on the ask only (bid 100.00x600, ask 100.02x0) -- state="
                  << state_name(f.state) << ", imbalance=" << f.imbalance << ", microprice=" << f.microprice
                  << " --\n";
        CHECK(f.state == BookState::OK);
        CHECK_NEAR(f.imbalance, 1.0);
        CHECK_NEAR(f.microprice, 100.02);
    }

    // -- Test 7: a general asymmetric book at different price and size
    // levels entirely, confirming the formulas generalize beyond the
    // round numbers of Tests 1-3. --
    {
        BookSnapshot book{{50.10, 250}, {50.14, 750}};
        BookFeatures f = compute_book_features(book);
        std::cout << "-- Test 7: general case (bid 50.10x250, ask 50.14x750) -- mid=" << f.mid_price
                  << ", spread=" << f.spread << ", imbalance=" << f.imbalance << ", microprice="
                  << f.microprice << " --\n";
        CHECK_NEAR(f.mid_price, 50.12);
        CHECK_NEAR(f.spread, 0.04);
        CHECK_NEAR(f.imbalance, -0.5);
        CHECK_NEAR(f.microprice, 50.11);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_order_book_microstructure_features.cpp -o 01_order_book_microstructure_features
./01_order_book_microstructure_features
```

**Sample input:** a perfectly balanced top-of-book checked to reduce the microprice exactly to the mid-price; a heavy bid-side imbalance and a heavy ask-side imbalance each checked to pull the microprice strictly toward the thin opposite side; a crossed book and a zero-liquidity-on-both-sides book each checked to report an explicit failure state rather than a silently wrong number; a one-sided zero-liquidity book checked as a real, legitimate edge case; and a general asymmetric book at different price and size levels entirely.

```text
========================================================
Chapter 27.1: Order Book Microstructure Features
========================================================

-- Test 1: balanced book (bid 100.00x500, ask 100.02x500) -- state=OK, mid=100.0100, microprice=100.0100, spread=0.0200, imbalance=0.0000 --
-- Test 2: heavy bid pressure (bid 100.00x900, ask 100.02x100) -- imbalance=0.8000, microprice=100.0180 (pulled toward the ask) --
-- Test 3: heavy ask pressure (bid 100.00x100, ask 100.02x900) -- imbalance=-0.8000, microprice=100.0020 (pulled toward the bid) --
-- Test 4: crossed book (bid 100.05 >= ask 100.00) -- state=CROSSED --
-- Test 5: zero displayed size on both sides -- state=NO_LIQUIDITY --
-- Test 6: zero size on the ask only (bid 100.00x600, ask 100.02x0) -- state=OK, imbalance=1.0000, microprice=100.0200 --
-- Test 7: general case (bid 50.10x250, ask 50.14x750) -- mid=50.1200, spread=0.0400, imbalance=-0.5000, microprice=50.1100 --

20/20 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a top-of-book signal as the whole order book"
    Every function in this section operates on TOP-OF-BOOK ONLY -- the single best bid and best ask level. A real limit order book has many price levels beneath the top, and a large real order can be deliberately split into small "iceberg" slices so that only a fraction of its own true size is ever displayed at the top level at once -- exactly the kind of hidden liquidity this section's own `imbalance` and `microprice` formulas have no way to see, since they only ever read the displayed size at the best level. A market participant relying on this section's own signals alone, without also tracking how the top-of-book levels are refreshed over time, would be blind to a large resting order deliberately kept off the visible top of the book -- a real, well-documented limitation of any level-1-only signal, not a bug in this section's own arithmetic.

## 27.2 The Latency Budget: Tick-to-Trade Deadline Enforcement

### Intuition

A hard real-time deadline changes what "correct" means: a trading decision computed correctly but too late is not a weaker answer, it is an answer that may already be wrong, since the book it was computed from has likely moved on. Exactly like Chapter 10.1's own derivation of a single decode step's latency, every microsecond value in this section is a stated, labeled parameter -- never a value read from a wall clock -- so this section's own locked self-test output stays identical no matter how fast or slow the specific machine running it happens to be.

### The Concept, In Detail

`run_pipeline` checks a market event's own age against a stated staleness limit BEFORE attempting any processing at all -- Test 4 confirms an event that arrived too long ago is rejected outright, with zero stages run, rather than wasting real processing time computing a decision from a book state already known to be stale. Otherwise, each named `PipelineStage`'s own stated cost accumulates in order, and the moment the running total exceeds the stated budget, the pipeline stops immediately at that exact stage -- Test 3 confirms the stage that caused the breach is named precisely, with its own cost still counted into the elapsed total (the time was genuinely spent before the breach was noticed) but NOT added to the list of stages that actually completed within budget.

Tests 2 and 5 confirm both of this section's own boundary conventions exactly: a total cost landing exactly ON the stated budget, and an event age landing exactly ON the staleness limit, both count as within bounds, using a consistent strict-inequality rule throughout. Test 7 confirms the breach is always reported at the FIRST stage where the running total crosses the budget, not merely the last, by deliberately front-loading an expensive first stage that alone already exceeds a small budget.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_tick_to_trade_deadline_enforcement.cpp -o 02_tick_to_trade_deadline_enforcement
./02_tick_to_trade_deadline_enforcement
```

**Sample input:** four named pipeline stages checked to complete comfortably within a generous budget, and again checked at the exact boundary of the total stage cost; the same stages checked one microsecond short of that total, confirming the exact stage that breaches the budget is named and its own cost is still counted; a market event checked to be rejected outright for being too old before the pipeline even starts, and again checked at the exact boundary of the staleness limit; a zero-microsecond budget checked to breach immediately at the very first stage; and a deliberately front-loaded set of stages checked to report the breach at the first stage that causes it, not a later one.

```text
========================================================
Chapter 27.2: The Latency Budget -- Tick-to-Trade Deadline Enforcement
========================================================

-- Test 1: 4 stages (80+120+150+90=440us), budget=500us -- missed=NO, elapsed=440us, completed=4 stages --
-- Test 2: same 4 stages, budget=440us (exactly the total cost) -- missed=NO, elapsed=440us --
-- Test 3: same 4 stages, budget=439us (1us short) -- missed=YES, stage_that_missed="send_order", elapsed=440us, completed=3 --
-- Test 4: event age = 900us against a staleness limit of 500us -- rejected_stale=YES, completed=0 stages, elapsed=0us --
-- Test 5: event age = 500us, exactly at the staleness limit of 500us -- rejected_stale=NO --
-- Test 6: budget=0us -- missed=YES, stage_that_missed="parse_market_data", completed=0 --
-- Test 7: front-loaded stages (300, 50, 50), budget=250us -- missed=YES, stage_that_missed="expensive_first_stage", elapsed=300us --

25/25 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating this section's own stated stage costs as a real system's actual measured latency"
    Every microsecond value in this file -- each `PipelineStage`'s own cost, every event's own arrival timestamp -- is a STATED, illustrative architectural parameter, exactly like Chapter 8.1's own illustrative CPU peak-bandwidth figures, not a measurement of any specific real exchange's own real infrastructure. A real production system's actual tick-to-trade latency depends on real network jitter, real operating-system scheduling, real cache and TLB behavior, and real garbage-collection or memory-allocation pauses in whatever language it is written in -- none of which this section's own pure, deterministic arithmetic can predict. Treating this section's own stated-parameter pipeline as a substitute for actually instrumenting and measuring a real deployed system's own real latency would repeat the identical mistake Chapter 8's own roofline model warned against: an idealized derivation is a real, useful ceiling to reason about, never a stand-in for a real measurement.

## 27.3 Financial News Sentiment Scoring

### Intuition

A real, well-documented finding in financial text analysis is that a generic English sentiment dictionary badly misclassifies financial text: ordinary business vocabulary such as "tax," "liability," and "cost" describes routine, expected line items, not bad news, yet a generic lexicon flags all three as negative. The real, published fix -- Loughran and McDonald's own finance-specific sentiment methodology -- deliberately excludes this routine vocabulary from its own negative word list. This section builds a small, explicitly illustrative lexicon modeled on that real methodology, and demonstrates the exact contrast it exists to fix.

### The Concept, In Detail

`score_sentiment` counts finance-lexicon matches into `positive_hits` and `negative_hits`, with a real, standard negation rule from lexicon-based sentiment analysis: a negation word (such as "not") flips the polarity of the single token immediately following it -- Test 4 confirms this concretely, scoring "not profitable" as negative and the identical word without the negation as positive. Test 5 is this section's own central demonstration: a headline containing only "tax" and "cost" scores an honest NEUTRAL under this section's own finance-specific lexicon, which deliberately excludes both words from its negative list, but scores NEGATIVE under a stated GENERIC comparison lexicon that (like a real general-purpose English sentiment dictionary) does treat them as negative -- the real Loughran-McDonald finding, reproduced directly rather than merely cited.

Test 6 confirms `tokenize_headline`'s own case-insensitivity and punctuation-stripping, and Test 7 confirms a headline with no lexicon matches at all reports an honest zero score and an explicit NEUTRAL label, never a crash or a fabricated nonzero result.

### Code and Verification

```cpp
// Chapter 27.3 -- A real, well-documented finding in financial text
// analysis (Loughran and McDonald, "When Is a Liability Not a
// Liability? Textual Analysis, Dictionaries, and 10-Ks," Journal of
// Finance, 2011) is that a GENERIC English sentiment word list badly
// misclassifies financial text: ordinary business vocabulary such as
// "tax," "liability," "cost," and "debt" is treated as negative by a
// generic dictionary, even though these words describe completely
// routine, expected line items in real financial disclosures and
// headlines, not bad news. The real fix Loughran-McDonald published is a
// FINANCE-SPECIFIC sentiment lexicon, deliberately built to exclude that
// routine vocabulary from its own negative word list.
//
// This section builds a small, explicitly illustrative lexicon modeled on
// that real published methodology -- not the real published word list
// itself, which spans several thousand words across seven real
// categories. This section's own COMMON TRAP box returns to exactly that
// scope limitation.
//
// A second real, standard technique from lexicon-based sentiment analysis
// is included as well: simple negation handling. A negation word (such as
// "not") flips the polarity of the very next lexicon word it precedes --
// "not profitable" should score as negative, not positive, even though
// "profitable" alone is a real positive-lexicon word.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_financial_news_sentiment_scoring.cpp -o 03_financial_news_sentiment_scoring
// Run:     ./03_financial_news_sentiment_scoring

#include <cctype>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// Lowercases a word and strips any leading/trailing punctuation, so that
// "Profit!", "profit,", and "profit" all tokenize identically.
static std::string clean_word(const std::string& raw) {
    std::string w;
    for (char c : raw) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            w += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return w;
}

static std::vector<std::string> tokenize_headline(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string raw;
    while (iss >> raw) {
        std::string cleaned = clean_word(raw);
        if (!cleaned.empty()) tokens.push_back(cleaned);
    }
    return tokens;
}

struct SentimentResult {
    int positive_hits = 0;
    int negative_hits = 0;
    int score = 0;
    std::string label;
    std::vector<std::string> matched_positive;
    std::vector<std::string> matched_negative;
};

// A finance-specific negative lexicon deliberately EXCLUDES routine
// business vocabulary (tax, liability, cost, debt) that a generic
// English lexicon would flag as negative -- this is the real
// Loughran-McDonald finding this section's own Test 5 demonstrates.
static const std::set<std::string> POS_LEXICON = {
    "profit", "profitable", "growth", "surge", "beat", "exceeded", "record", "strong", "upgrade",
};
static const std::set<std::string> NEG_LEXICON = {
    "loss", "losses", "decline", "miss", "missed", "downgrade", "weak", "lawsuit", "bankruptcy", "fraud",
};
// Words a GENERIC (non-finance-specific) sentiment lexicon would mark
// negative, that the real finance-specific methodology deliberately
// excludes from NEG_LEXICON above -- used only by Test 5 to demonstrate
// the contrast, never merged into NEG_LEXICON itself.
static const std::set<std::string> GENERIC_NEGATIVE_EXTRA = {
    "tax", "liability", "liabilities", "cost", "costs", "debt",
};
static const std::set<std::string> NEGATION_WORDS = {"not", "no", "never"};

// Negation applies only to the SINGLE token immediately following a
// negation word -- a deliberately narrow, stated scope, not an attempt
// to track negation across an entire clause.
static SentimentResult score_sentiment(const std::vector<std::string>& tokens,
                                        const std::set<std::string>& pos_lexicon,
                                        const std::set<std::string>& neg_lexicon) {
    SentimentResult r;
    bool negate_next = false;
    for (const auto& tok : tokens) {
        if (NEGATION_WORDS.contains(tok)) {
            negate_next = true;
            continue;
        }
        const bool is_pos = pos_lexicon.contains(tok);
        const bool is_neg = neg_lexicon.contains(tok);
        if (is_pos) {
            if (negate_next) {
                r.negative_hits++;
                r.matched_negative.push_back(tok + " (negated)");
            } else {
                r.positive_hits++;
                r.matched_positive.push_back(tok);
            }
        } else if (is_neg) {
            if (negate_next) {
                r.positive_hits++;
                r.matched_positive.push_back(tok + " (negated)");
            } else {
                r.negative_hits++;
                r.matched_negative.push_back(tok);
            }
        }
        negate_next = false;
    }
    r.score = r.positive_hits - r.negative_hits;
    r.label = (r.score > 0) ? "POSITIVE" : (r.score < 0) ? "NEGATIVE" : "NEUTRAL";
    return r;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 27.3: Financial News Sentiment Scoring\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: a clearly positive headline -- every matched word is a
    // real finance-positive term, none negative. --
    {
        auto tokens = tokenize_headline("Company reports record profit and strong growth");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 1: \"record profit ... strong growth\" -- positive_hits=" << r.positive_hits
                  << ", negative_hits=" << r.negative_hits << ", score=" << r.score << ", label=" << r.label
                  << " --\n";
        CHECK(r.positive_hits == 4);
        CHECK(r.negative_hits == 0);
        CHECK(r.score == 4);
        CHECK(r.label == "POSITIVE");
    }

    // -- Test 2: a clearly negative headline. --
    {
        auto tokens = tokenize_headline("Company reports steep decline and missed earnings amid lawsuit");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 2: \"decline ... missed earnings ... lawsuit\" -- positive_hits="
                  << r.positive_hits << ", negative_hits=" << r.negative_hits << ", score=" << r.score
                  << ", label=" << r.label << " --\n";
        CHECK(r.positive_hits == 0);
        CHECK(r.negative_hits == 3);
        CHECK(r.score == -3);
        CHECK(r.label == "NEGATIVE");
    }

    // -- Test 3: an evenly mixed headline -- equal positive and negative
    // hits must net to an honest NEUTRAL, not a tie-break in either
    // direction. --
    {
        auto tokens = tokenize_headline("profit and growth offset by loss and decline");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 3: \"profit and growth offset by loss and decline\" -- positive_hits="
                  << r.positive_hits << ", negative_hits=" << r.negative_hits << ", score=" << r.score
                  << ", label=" << r.label << " --\n";
        CHECK(r.positive_hits == 2);
        CHECK(r.negative_hits == 2);
        CHECK(r.score == 0);
        CHECK(r.label == "NEUTRAL");
    }

    // -- Test 4: negation flips a positive word's own contribution to
    // negative -- confirmed against the SAME word scored without a
    // preceding negation, to isolate the negation logic's own effect. --
    {
        auto negated_tokens = tokenize_headline("not profitable this quarter");
        auto negated = score_sentiment(negated_tokens, POS_LEXICON, NEG_LEXICON);
        auto plain_tokens = tokenize_headline("profitable this quarter");
        auto plain = score_sentiment(plain_tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 4: \"not profitable this quarter\" -> score=" << negated.score << " ("
                  << negated.label << "), vs. \"profitable this quarter\" -> score=" << plain.score << " ("
                  << plain.label << ") --\n";
        CHECK(negated.score == -1);
        CHECK(negated.label == "NEGATIVE");
        CHECK(plain.score == 1);
        CHECK(plain.label == "POSITIVE");
    }

    // -- Test 5: the real Loughran-McDonald finding, demonstrated
    // directly. "tax" and "cost" are routine finance vocabulary, excluded
    // from this section's own finance-specific NEG_LEXICON -- scoring
    // NEUTRAL here -- but scoring NEGATIVE under a stated GENERIC lexicon
    // that (like a real general-purpose English sentiment dictionary)
    // does treat them as negative words. --
    {
        auto tokens = tokenize_headline("The company reported higher tax and cost this quarter");
        auto finance = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::set<std::string> generic_neg = NEG_LEXICON;
        generic_neg.insert(GENERIC_NEGATIVE_EXTRA.begin(), GENERIC_NEGATIVE_EXTRA.end());
        auto generic = score_sentiment(tokens, POS_LEXICON, generic_neg);
        std::cout << "-- Test 5: \"higher tax and cost\" -- finance-lexicon score=" << finance.score << " ("
                  << finance.label << "), generic-lexicon score=" << generic.score << " (" << generic.label
                  << ") --\n";
        CHECK(finance.score == 0);
        CHECK(finance.label == "NEUTRAL");
        CHECK(generic.score == -2);
        CHECK(generic.label == "NEGATIVE");
    }

    // -- Test 6: tokenizer robustness -- mixed case and trailing
    // punctuation attached directly to each word must still match the
    // lowercase, punctuation-free lexicon entries. --
    {
        auto tokens = tokenize_headline("Profit! Growth... Exceeded expectations.");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 6: \"Profit! Growth... Exceeded expectations.\" -- positive_hits="
                  << r.positive_hits << ", score=" << r.score << ", label=" << r.label << " --\n";
        CHECK(r.positive_hits == 3);
        CHECK(r.score == 3);
        CHECK(r.label == "POSITIVE");
    }

    // -- Test 7: a headline containing no lexicon words at all must
    // report an honest, explicit zero -- NEUTRAL with empty matched
    // lists -- never a crash or a fabricated nonzero score. --
    {
        auto tokens = tokenize_headline("The quarterly meeting was held Tuesday");
        auto r = score_sentiment(tokens, POS_LEXICON, NEG_LEXICON);
        std::cout << "-- Test 7: \"The quarterly meeting was held Tuesday\" -- positive_hits="
                  << r.positive_hits << ", negative_hits=" << r.negative_hits << ", score=" << r.score
                  << ", label=" << r.label << " --\n";
        CHECK(r.positive_hits == 0);
        CHECK(r.negative_hits == 0);
        CHECK(r.score == 0);
        CHECK(r.label == "NEUTRAL");
        CHECK(r.matched_positive.empty());
        CHECK(r.matched_negative.empty());
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_financial_news_sentiment_scoring.cpp -o 03_financial_news_sentiment_scoring
./03_financial_news_sentiment_scoring
```

**Sample input:** a clearly positive headline and a clearly negative headline, each checked against the exact expected hit counts and label; an evenly mixed headline checked to net to an honest NEUTRAL; a negated positive word checked against the identical word without negation, to isolate the negation logic's own effect; a headline containing only finance-neutral vocabulary ("tax," "cost") checked against this section's own finance-specific lexicon and a stated generic comparison lexicon, reproducing the real Loughran-McDonald finding directly; a headline with mixed case and trailing punctuation checked for correct tokenization; and a headline containing no lexicon words at all checked to report an honest zero.

```text
========================================================
Chapter 27.3: Financial News Sentiment Scoring
========================================================

-- Test 1: "record profit ... strong growth" -- positive_hits=4, negative_hits=0, score=4, label=POSITIVE --
-- Test 2: "decline ... missed earnings ... lawsuit" -- positive_hits=0, negative_hits=3, score=-3, label=NEGATIVE --
-- Test 3: "profit and growth offset by loss and decline" -- positive_hits=2, negative_hits=2, score=0, label=NEUTRAL --
-- Test 4: "not profitable this quarter" -> score=-1 (NEGATIVE), vs. "profitable this quarter" -> score=1 (POSITIVE) --
-- Test 5: "higher tax and cost" -- finance-lexicon score=0 (NEUTRAL), generic-lexicon score=-2 (NEGATIVE) --
-- Test 6: "Profit! Growth... Exceeded expectations." -- positive_hits=3, score=3, label=POSITIVE --
-- Test 7: "The quarterly meeting was held Tuesday" -- positive_hits=0, negative_hits=0, score=0, label=NEUTRAL --

29/29 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating this section's own small illustrative lexicon as the real, complete Loughran-McDonald word list"
    `POS_LEXICON` and `NEG_LEXICON` in this file are small, illustrative lists built to demonstrate the real methodology, not the actual published Loughran-McDonald dictionaries, which span several thousand words across seven real categories (negative, positive, uncertainty, litigious, strong modal, weak modal, and constraining language) developed and validated against real 10-K filings. Deploying this section's own narrow illustrative vocabulary against real production news flow would miss the large majority of real finance-relevant sentiment words entirely and silently score most real headlines as a hollow NEUTRAL -- the identical narrow-closed-vocabulary caution Chapter 12's own tokenizer, Chapter 24's own closed intent vocabulary, and Chapter 26.3's own amount-in-words parser each stated for their own real, fixed, and deliberately limited scope.

## 27.4 Signal Fusion and the Trading Decision Engine

### Intuition

A trading decision is only as trustworthy as its own audit trail, and a signal that arrived too late to act on safely is not a weaker signal -- it is not a usable signal at all. This section's capstone fuses Section 27.1's order-book imbalance and price pressure with Section 27.3's sentiment score into one weighted score and a three-way action, reapplying Chapter 26.4's own never-suppress-a-named-flag structure -- but inverted for this real-time domain, so Section 27.2's own deadline signal forces the SAFEST disposition rather than the strictest one.

### The Concept, In Detail

`fuse_signal` checks the upstream book's own validity first -- Test 5 confirms an invalid book state (a crossed or no-liquidity book) forces HOLD outright, before any weighted score is even computed, regardless of how strong sentiment looks on its own. When the book is valid, the weighted score is always computed and recorded in the audit trail, but Test 4 is this section's own central check: the IDENTICAL bullish inputs that produced a clear BUY in Test 1 are forced to HOLD instead the moment `within_deadline` is false, with the would-have-been score of 20 still named in the audit trail rather than silently discarded -- a missed deadline overrides the trading action, but never erases the record of what the signal actually was.

Test 3 confirms the distinction this section's own COMMON TRAP box returns to: a genuinely weak, mixed combination of signals also lands on HOLD, but with `forced_hold_stale` explicitly false, since this HOLD was computed honestly from balanced signals, not forced by a missed deadline. Tests 6 and 7 confirm both real score thresholds exactly at their own stated boundaries, and Test 8 confirms a zeroed-out weight genuinely removes a signal's own contribution completely, even against an extreme underlying value.

### Code and Verification

```cpp
// Chapter 27.4 -- This chapter's own capstone: a real, fully auditable
// signal-fusion engine combining Section 27.1's order-book imbalance and
// microprice-vs-mid "price pressure," and Section 27.3's finance-lexicon
// sentiment score, into a single weighted score and a three-way trading
// action -- BUY, SELL, or HOLD. The central discipline reapplies Chapter
// 26.4's own never-suppress-a-named-flag structure, but inverted for this
// real-time domain: instead of one high-severity signal FORCING a
// stricter disposition, Section 27.2's own deadline signal FORCES the
// SAFEST disposition (HOLD) the instant an event is stale or missed its
// stated processing budget -- regardless of how strong the underlying
// market signal looked. A trading signal that arrives too late to act on
// safely is not a weaker signal; it is not a usable signal at all.
//
// A note on this section's own honest scope: a FORCED HOLD (triggered by
// a missed deadline) and a COMPUTED HOLD (triggered by a genuinely weak
// or balanced signal) are never conflated in this file -- each is
// reported with its own distinct, named reason, exactly because they mean
// completely different things operationally. This section's own COMMON
// TRAP box returns to exactly that distinction.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_signal_fusion_and_trading_decision_engine.cpp -o 04_signal_fusion_and_trading_decision_engine
// Run:     ./04_signal_fusion_and_trading_decision_engine

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// A bundle of ALREADY-COMPUTED signals from the prior three sections --
// Section 27.1's order-book imbalance and price pressure (microprice
// minus mid-price), Section 27.3's sentiment score, and Section 27.2's
// own deadline verdict -- exactly the same "named, pre-computed signal"
// convention Chapter 26.4's own RiskSignal struct used for its four
// fraud signals.
struct SignalInputs {
    bool book_valid = true;      // false if the upstream book state was CROSSED or NO_LIQUIDITY
    double imbalance = 0.0;      // Section 27.1's order-book imbalance, in [-1, +1]
    double price_pressure = 0.0; // Section 27.1's (microprice - mid_price), a signed dollar pressure
    int sentiment_score = 0;     // Section 27.3's net lexicon sentiment score
    bool within_deadline = true; // false if Section 27.2's pipeline was stale or missed its budget
};

enum class TradeAction { HOLD, BUY, SELL };

static std::string action_name(TradeAction a) {
    switch (a) {
        case TradeAction::HOLD: return "HOLD";
        case TradeAction::BUY: return "BUY";
        case TradeAction::SELL: return "SELL";
    }
    return "UNKNOWN";
}

struct FusedSignal {
    TradeAction action = TradeAction::HOLD;
    double weighted_score = 0.0;
    bool forced_hold_stale = false;  // true ONLY when the deadline signal itself forced HOLD
    std::vector<std::string> reasons;
};

static std::string fmt(double v, int prec = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

// The book's own state is checked FIRST -- a CROSSED or NO_LIQUIDITY book
// carries no real signal to fuse at all, regardless of sentiment or
// deadline status. Next, an event that missed Section 27.2's own deadline
// FORCES HOLD outright: the weighted score is still computed for the
// audit trail (so a human can see what the signal WOULD have been), but
// the returned action is never anything other than HOLD. Only when the
// book is valid AND the event arrived within its stated deadline does the
// weighted score actually drive the trading action.
FusedSignal fuse_signal(const SignalInputs& in, double w_imbalance, double w_pressure, double w_sentiment,
                         double buy_threshold, double sell_threshold) {
    FusedSignal f;
    if (!in.book_valid) {
        f.action = TradeAction::HOLD;
        f.reasons.push_back("book state invalid -- cannot compute a real signal, forced HOLD");
        return f;
    }

    const double imbalance_contrib = w_imbalance * in.imbalance;
    const double pressure_contrib = w_pressure * in.price_pressure;
    const double sentiment_contrib = w_sentiment * static_cast<double>(in.sentiment_score);
    f.weighted_score = imbalance_contrib + pressure_contrib + sentiment_contrib;
    f.reasons.push_back("order-book imbalance: " + fmt(in.imbalance) + " * weight " + fmt(w_imbalance, 0) +
                         " = " + fmt(imbalance_contrib));
    f.reasons.push_back("price pressure: " + fmt(in.price_pressure) + " * weight " + fmt(w_pressure, 0) +
                         " = " + fmt(pressure_contrib));
    f.reasons.push_back("sentiment score: " + fmt(static_cast<double>(in.sentiment_score), 0) +
                         " * weight " + fmt(w_sentiment, 0) + " = " + fmt(sentiment_contrib));

    if (!in.within_deadline) {
        f.forced_hold_stale = true;
        f.action = TradeAction::HOLD;
        f.reasons.push_back("event missed its processing deadline -- forced HOLD regardless of computed "
                             "score of " + fmt(f.weighted_score));
        return f;
    }

    if (f.weighted_score >= buy_threshold) {
        f.action = TradeAction::BUY;
    } else if (f.weighted_score <= sell_threshold) {
        f.action = TradeAction::SELL;
    } else {
        f.action = TradeAction::HOLD;
    }
    return f;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 27.4: Signal Fusion and the Trading Decision Engine\n";
    std::cout << "========================================================\n\n";

    const double W_IMBALANCE = 10.0;
    const double W_PRESSURE = 500.0;
    const double W_SENTIMENT = 2.0;
    const double BUY_THRESHOLD = 15.0;
    const double SELL_THRESHOLD = -15.0;

    // -- Test 1: a clearly bullish combination -- positive imbalance,
    // positive price pressure, positive sentiment, arrived within
    // deadline -- fuses to a real BUY with all three signals named. --
    {
        SignalInputs in{true, 0.6, 0.02, 2, true};
        FusedSignal f = fuse_signal(in, W_IMBALANCE, W_PRESSURE, W_SENTIMENT, BUY_THRESHOLD, SELL_THRESHOLD);
        std::cout << "-- Test 1: bullish (imbalance=0.6, pressure=0.02, sentiment=+2) -- score="
                  << fmt(f.weighted_score) << ", action=" << action_name(f.action) << " --\n";
        CHECK(f.action == TradeAction::BUY);
        CHECK(fmt(f.weighted_score) == "20.000");
        CHECK(f.reasons.size() == 3);
        CHECK(!f.forced_hold_stale);
    }

    // -- Test 2: the mirror-image bearish combination -- fuses to SELL. --
    {
        SignalInputs in{true, -0.6, -0.02, -2, true};
        FusedSignal f = fuse_signal(in, W_IMBALANCE, W_PRESSURE, W_SENTIMENT, BUY_THRESHOLD, SELL_THRESHOLD);
        std::cout << "-- Test 2: bearish (imbalance=-0.6, pressure=-0.02, sentiment=-2) -- score="
                  << fmt(f.weighted_score) << ", action=" << action_name(f.action) << " --\n";
        CHECK(f.action == TradeAction::SELL);
        CHECK(fmt(f.weighted_score) == "-20.000");
    }

    // -- Test 3: a genuinely weak, mixed combination lands inside the
    // HOLD band on its own real merits -- confirmed as a COMPUTED HOLD
    // (forced_hold_stale is false), distinct from Test 4's FORCED HOLD. --
    {
        SignalInputs in{true, 0.3, 0.01, -1, true};
        FusedSignal f = fuse_signal(in, W_IMBALANCE, W_PRESSURE, W_SENTIMENT, BUY_THRESHOLD, SELL_THRESHOLD);
        std::cout << "-- Test 3: mixed (imbalance=0.3, pressure=0.01, sentiment=-1) -- score="
                  << fmt(f.weighted_score) << ", action=" << action_name(f.action) << ", forced_hold_stale="
                  << (f.forced_hold_stale ? "YES" : "NO") << " --\n";
        CHECK(f.action == TradeAction::HOLD);
        CHECK(fmt(f.weighted_score) == "6.000");
        CHECK(!f.forced_hold_stale);
    }

    // -- Test 4 (central): the IDENTICAL bullish inputs from Test 1 --
    // score would again be 20, well past the BUY threshold -- but this
    // event missed its own processing deadline. The action is forced to
    // HOLD regardless, with the would-have-been score still recorded in
    // the audit trail rather than discarded. --
    {
        SignalInputs in{true, 0.6, 0.02, 2, /*within_deadline=*/false};
        FusedSignal f = fuse_signal(in, W_IMBALANCE, W_PRESSURE, W_SENTIMENT, BUY_THRESHOLD, SELL_THRESHOLD);
        std::cout << "-- Test 4: identical bullish inputs to Test 1, but within_deadline=false -- score="
                  << fmt(f.weighted_score) << " (would have been BUY), action=" << action_name(f.action)
                  << ", forced_hold_stale=" << (f.forced_hold_stale ? "YES" : "NO") << " --\n";
        CHECK(f.action == TradeAction::HOLD);
        CHECK(f.forced_hold_stale);
        CHECK(fmt(f.weighted_score) == "20.000");
    }

    // -- Test 5: an invalid upstream book state (a crossed or
    // no-liquidity book) forces HOLD outright, before any weighted score
    // is even computed -- regardless of how strong sentiment looks. --
    {
        SignalInputs in{/*book_valid=*/false, 0.0, 0.0, 5, true};
        FusedSignal f = fuse_signal(in, W_IMBALANCE, W_PRESSURE, W_SENTIMENT, BUY_THRESHOLD, SELL_THRESHOLD);
        std::cout << "-- Test 5: invalid book state, sentiment=+5 -- action=" << action_name(f.action)
                  << ", reason=\"" << f.reasons.front() << "\" --\n";
        CHECK(f.action == TradeAction::HOLD);
        CHECK(f.weighted_score == 0.0);
        CHECK(f.reasons.size() == 1);
        CHECK(!f.forced_hold_stale);
    }

    // -- Test 6: the BUY threshold's own exact boundary -- a score of
    // exactly 15.0 (the stated buy_threshold itself) must trigger BUY,
    // confirming the boundary is inclusive (>=), not merely a value
    // comfortably above it. --
    {
        SignalInputs in{true, 0.3, 0.02, 1, true};
        FusedSignal f = fuse_signal(in, W_IMBALANCE, W_PRESSURE, W_SENTIMENT, BUY_THRESHOLD, SELL_THRESHOLD);
        std::cout << "-- Test 6: exact BUY boundary (imbalance=0.3, pressure=0.02, sentiment=+1) -- score="
                  << fmt(f.weighted_score) << ", action=" << action_name(f.action) << " --\n";
        CHECK(fmt(f.weighted_score) == "15.000");
        CHECK(f.action == TradeAction::BUY);
    }

    // -- Test 7: the SELL threshold's own exact boundary, the mirror
    // image of Test 6 -- confirming the same inclusive convention. --
    {
        SignalInputs in{true, -0.3, -0.02, -1, true};
        FusedSignal f = fuse_signal(in, W_IMBALANCE, W_PRESSURE, W_SENTIMENT, BUY_THRESHOLD, SELL_THRESHOLD);
        std::cout << "-- Test 7: exact SELL boundary (imbalance=-0.3, pressure=-0.02, sentiment=-1) -- score="
                  << fmt(f.weighted_score) << ", action=" << action_name(f.action) << " --\n";
        CHECK(fmt(f.weighted_score) == "-15.000");
        CHECK(f.action == TradeAction::SELL);
    }

    // -- Test 8: a zero sentiment weight must zero out sentiment's own
    // contribution COMPLETELY, even against an extremely negative
    // sentiment score -- confirming the weights are genuinely
    // multiplicative, not merely an informal "tiebreaker" bonus. --
    {
        SignalInputs in{true, 0.6, 0.02, -100, true};
        FusedSignal f = fuse_signal(in, W_IMBALANCE, W_PRESSURE, /*w_sentiment=*/0.0, BUY_THRESHOLD,
                                     SELL_THRESHOLD);
        std::cout << "-- Test 8: sentiment=-100 but w_sentiment=0 -- score=" << fmt(f.weighted_score)
                  << ", action=" << action_name(f.action) << " --\n";
        CHECK(fmt(f.weighted_score) == "16.000");
        CHECK(f.action == TradeAction::BUY);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_signal_fusion_and_trading_decision_engine.cpp -o 04_signal_fusion_and_trading_decision_engine
./04_signal_fusion_and_trading_decision_engine
```

**Sample input:** a clearly bullish combination of signals checked to fuse to BUY, and its exact bearish mirror image checked to fuse to SELL; a genuinely weak, mixed combination checked to land on a COMPUTED HOLD; the identical bullish inputs from the first test checked again with a missed deadline, forcing HOLD despite an unchanged, strongly bullish underlying score; an invalid upstream book state checked to force HOLD outright regardless of sentiment; both real score thresholds checked exactly at their own stated boundaries; and a zeroed-out sentiment weight checked to remove sentiment's own contribution completely even against an extreme value.

```text
========================================================
Chapter 27.4: Signal Fusion and the Trading Decision Engine
========================================================

-- Test 1: bullish (imbalance=0.6, pressure=0.02, sentiment=+2) -- score=20.000, action=BUY --
-- Test 2: bearish (imbalance=-0.6, pressure=-0.02, sentiment=-2) -- score=-20.000, action=SELL --
-- Test 3: mixed (imbalance=0.3, pressure=0.01, sentiment=-1) -- score=6.000, action=HOLD, forced_hold_stale=NO --
-- Test 4: identical bullish inputs to Test 1, but within_deadline=false -- score=20.000 (would have been BUY), action=HOLD, forced_hold_stale=YES --
-- Test 5: invalid book state, sentiment=+5 -- action=HOLD, reason="book state invalid -- cannot compute a real signal, forced HOLD" --
-- Test 6: exact BUY boundary (imbalance=0.3, pressure=0.02, sentiment=+1) -- score=15.000, action=BUY --
-- Test 7: exact SELL boundary (imbalance=-0.3, pressure=-0.02, sentiment=-1) -- score=-15.000, action=SELL --
-- Test 8: sentiment=-100 but w_sentiment=0 -- score=16.000, action=BUY --

22/22 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] conflating a FORCED HOLD with a COMPUTED HOLD"
    A forced HOLD (`forced_hold_stale == true`) and a computed HOLD (a genuinely weak or balanced signal, with `forced_hold_stale == false`) both result in the identical `TradeAction::HOLD`, but they mean completely different things operationally. A computed HOLD is an honest statement that no real trading edge was detected in this specific signal. A forced HOLD carries no such information at all -- the underlying signal may well have been a strong, real BUY or SELL that simply arrived too late to act on safely, and Test 4 exists specifically to show the would-have-been score is still recorded rather than discarded. Logging both cases identically, without preserving the `forced_hold_stale` distinction, would hide a real, recurring latency problem -- a system missing its own deadline often enough to matter -- behind what looks like a healthy stream of ordinary, low-conviction HOLD decisions, exactly the same distinct-audit-trail discipline Chapter 26.4's own `AUTO_CLEAR` trap already established for a clean score covering only its own stated scope.

## Chapter Summary

This chapter turned from Chapter 26's fraud-pattern detection to a genuinely different real financial constraint: a hard, real-time deadline. Section 27.1 built real, published market-microstructure formulas -- mid-price, microprice, and order-book imbalance -- from a top-of-book snapshot, with explicit, honest failure states for a crossed or zero-liquidity book. Section 27.2 reapplied Chapter 10's own "verified without a clock" discipline to a hard tick-to-trade deadline, rejecting stale events outright and naming the exact pipeline stage that breaches a stated budget. Section 27.3 built a small, illustrative finance-specific sentiment lexicon modeled on the real, published Loughran-McDonald methodology, and reproduced its central finding directly: routine business vocabulary that a generic lexicon misreads as negative. Section 27.4 closed the chapter by fusing all three signals into one auditable trading decision, reapplying Chapter 26.4's own never-suppress-a-flag discipline in its own inverted form -- a missed deadline forces the safest disposition, never the riskiest one, while always preserving an honest record of what the underlying signal actually was.

## Self-Check Questions

1. Section 27.1's `compute_book_features` checks for a CROSSED book and a NO_LIQUIDITY book as two separate, explicit failure states rather than a single generic "invalid" flag. Explain one concrete reason a human reviewing this system's own output would benefit from knowing WHICH of the two failure states actually occurred.
2. Section 27.1's Test 6 constructs a book with zero displayed size on the ask side only, not both sides. Using the section's own microprice formula, explain algebraically why the microprice reduces to EXACTLY the ask price in this specific case, not merely approximately.
3. Section 27.2's own introduction explains that every timing value in the file is a stated parameter, never a value read from a wall clock. Explain concretely what would go wrong with this book's own cross-architecture verification process if `run_pipeline` instead measured its own stage costs using `std::chrono::high_resolution_clock`.
4. Section 27.2's Test 7 deliberately front-loads an expensive first stage rather than placing the expensive stage last. Explain what specific property of `run_pipeline`'s own breach-detection logic this test would fail to distinguish if the expensive stage were placed last instead, as in Test 3.
5. Section 27.3's Test 5 reproduces the real Loughran-McDonald finding using two different lexicons on the IDENTICAL headline. Explain why comparing the SAME headline under two lexicons is a stronger demonstration than simply asserting that "tax" and "cost" are absent from `NEG_LEXICON`.
6. Section 27.3's own negation rule flips the polarity of only the single token immediately following a negation word. Construct one realistic financial headline where this narrow scope would cause `score_sentiment` to miss an intended negation, and explain exactly why.
7. Section 27.4's `fuse_signal` checks `book_valid` before checking `within_deadline`. Explain what would happen differently if these two checks were reordered, using Section 27.1's own CROSSED-book failure state combined with a stale event as the concrete scenario.
8. Section 27.4's Test 4 reuses the IDENTICAL numeric inputs from Test 1, changing only `within_deadline`. Explain why holding every other input fixed makes this test a stronger check of the deadline-forcing logic than constructing a fresh, differently-valued example would be.
9. Section 27.4's own COMMON TRAP box warns against conflating a forced HOLD with a computed HOLD. Describe one concrete, real operational decision a trading desk might make differently if it could see the `forced_hold_stale` flag, compared to a system that only ever logged the final `TradeAction`.
10. Across this chapter's four sections, identify the ONE section whose own central discipline is an INVERSION of a discipline this book already established in an earlier chapter, name the earlier chapter and discipline being inverted, and explain precisely what is inverted about it.

## Where We Go Next

This chapter showed that a hard real-time deadline changes what "correct" means for a trading signal -- a strong signal computed too late must be treated as no signal at all, reapplying this book's own recurring never-suppress-a-flag discipline in its own inverted, safety-first form. Chapter 28 turns to a different real financial setting once more: insurance claims photo assessment with fraud flagging, extending Chapter 21's own real `CostRange` damage-assessment pattern with a genuine fraud-likelihood signal layered on top of it.

## Worked Solutions

**1.** A CROSSED book and a NO_LIQUIDITY book point a human reviewer toward two completely different real causes and two completely different real responses. A CROSSED book (best bid at or above best ask) usually indicates a real market-data feed problem, a stale or out-of-order update, or a genuine, rare crossed-market event at the exchange itself -- something a reviewer would investigate on the DATA FEED side. A NO_LIQUIDITY book (zero displayed size on both sides) instead indicates a real, if unusual, moment where no resting orders exist at the top of book at all -- something a reviewer would investigate on the MARKET CONDITIONS side, such as a trading halt or an illiquid instrument. Collapsing both into one generic "invalid" flag would erase exactly the distinction a reviewer needs to know which system to check first.

**2.** With `ask.size = 0`, the microprice formula `(ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)` has its second term, `bid_price * ask_size`, multiply by exactly zero, collapsing it to `bid_price * 0 = 0`. The denominator `bid_size + ask_size` becomes simply `bid_size` (since `ask_size` is 0). The whole expression reduces algebraically to `(ask_price * bid_size + 0) / bid_size`, and since `bid_size` is nonzero (checked separately by the NO_LIQUIDITY guard), this is exactly `ask_price * bid_size / bid_size = ask_price` -- an exact algebraic identity, not an approximation that merely happens to be close.

**3.** This book's own cross-architecture verification process runs the identical compiled binary (or an architecturally distinct but source-identical build) across four genuinely different execution environments -- native x86_64, a newer GCC version, an aarch64 target under emulation, and the user's own real device -- and requires the locked, byte-for-byte self-test output to match EXACTLY across all four. A real wall-clock measurement varies with the actual speed of the specific CPU running it, whether it is a native machine, an emulated aarch64 process under QEMU (which runs meaningfully slower than native execution), or a different real device entirely -- so any measured microsecond value embedded in the printed output would differ across these four legs even though the PROGRAM's own logic is identical, causing the verification's own diff-based comparison to fail for a reason that has nothing to do with a real bug in the code.

**4.** If the expensive stage were placed last, as in Test 3, a breach-detection implementation that (incorrectly) waited until the END of the loop and then scanned backward for the first stage whose CUMULATIVE total exceeded the budget would still report the correct final stage, since the last stage is trivially both the one that "ends" the loop and the one whose cumulative total first crosses the budget. Placing the expensive stage FIRST specifically distinguishes a correct implementation (which stops and reports the breach the INSTANT the running total crosses the budget, potentially after only the very first stage) from a subtly incorrect one that only checks the breach condition after summing every stage's cost first and then searching for where it happened -- the two implementations would agree on Test 3's own placement but disagree on Test 7's.

**5.** Simply asserting that "tax" and "cost" are absent from `NEG_LEXICON` only shows a property of the LEXICON's own construction -- it says nothing about whether that absence actually changes a real headline's own computed score in practice, and a reader would have to trust the assertion rather than see the effect. Running the IDENTICAL headline through both a finance-specific lexicon and a stated generic comparison lexicon and observing the SAME two words produce a NEUTRAL score in one case and a NEGATIVE score in the other demonstrates the real, practical CONSEQUENCE of the lexicon choice directly, on a concrete example, exactly the same "show the actual effect, not just the definition" standard Section 26.1's own dHash tests applied by comparing Hamming distances on real synthetic images rather than merely asserting the algorithm's own formula.

**6.** A headline such as "Earnings were not disappointing or weak this quarter" intends to negate BOTH "disappointing" and "weak" with the single word "not," but Section 27.3's own stated narrow-scope negation rule only flips the polarity of the SINGLE token immediately following "not" -- which, after tokenization, would be "disappointing" (assuming it were in the lexicon) or whatever word directly follows "not," while "weak" (appearing later in the same clause, after "or") would be scored with its own normal, un-negated polarity as an ordinary negative hit, incorrectly canceling out only part of the intended double negation.

**7.** If `within_deadline` were checked before `book_valid`, a stale event carrying an invalid (crossed) book would still be correctly forced to HOLD, since a missed deadline forces HOLD outright regardless of the book state -- so the FINAL action would not actually change in this specific scenario. What WOULD change is the audit trail's own reasons: reordering the checks would report "event missed its deadline" as the operative reason even when the book was ALSO independently invalid, hiding the fact that this specific event had two separate, independently disqualifying problems rather than just one -- exactly the same kind of lost diagnostic information Question 1 already identified for collapsing CROSSED and NO_LIQUIDITY into a single flag.

**8.** Reusing the identical numeric inputs isolates the deadline-forcing logic as the ONLY variable that changed between the two tests, which means any difference in the two tests' own results can be attributed to the deadline check alone, with complete confidence that no other input coincidentally caused the difference. A fresh, differently-valued example would leave open the possibility that the different result was caused by the DIFFERENT underlying signal values rather than the deadline logic itself, requiring a reader to trust that the two examples were constructed to isolate the same variable rather than seeing it demonstrated directly -- the identical "hold everything else fixed" principle already used by Section 26.4's own Test 5, which combined its MICR-forcing rule with signals that ALREADY independently rejected, to confirm the rule's own directionality specifically.

**9.** A trading desk that could see `forced_hold_stale` set to true on a real, recurring basis would have a concrete, actionable reason to investigate and fix a real latency problem in its own infrastructure -- network jitter, an overloaded feature-computation stage, a slow order-routing path -- since each occurrence represents a real trading opportunity the system detected but could not safely act on in time. A desk that only ever saw the final `TradeAction::HOLD`, with no way to distinguish a forced HOLD from a genuinely balanced computed HOLD, would have no way to tell whether its own system was missing real opportunities due to a fixable latency problem, or simply operating correctly in a quiet, low-conviction market -- two situations that call for completely different responses, one an engineering fix and the other no action at all.

**10.** Section 27.4's own capstone discipline is a direct inversion of Chapter 26.4's never-suppress-a-named-flag discipline. Chapter 26.4's own MICR-checksum-failure rule forces the STRICTER disposition (at least `ESCALATE_TO_REVIEW`) onto an otherwise-clean score, and explicitly never downgrades an already-stricter `AUTO_REJECT` back down. Section 27.4's own deadline-failure rule instead forces the SAFER, LEAST-risky disposition (`HOLD`) onto an otherwise-strong BUY or SELL score -- the exact opposite direction of override, appropriate to this chapter's own different real constraint: in fraud detection, missing a real red flag is the costly mistake a system must never make, while in low-latency trading, ACTING on a stale signal is the costly mistake, so the forcing rule pushes toward inaction rather than toward heightened scrutiny.
