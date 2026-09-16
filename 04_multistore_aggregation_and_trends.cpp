// Chapter 19.4 -- Section 19.3 taught one store's own inventory record
// how to update itself safely against concurrent writers. A retailer
// with hundreds of stores needs a second, genuinely different question
// answered: not "what is store 42's own count of SKU-1001 right now,"
// but "across every store, is SKU-1001 trending toward a stockout, and
// by when." This section builds that aggregation and trend-detection
// layer entirely from scratch -- real summation, a real least-squares
// linear regression, and a real, stated discipline against ever
// fabricating a number for a store that simply did not report.
//
// Every timestamp in this section is a caller-supplied discrete PERIOD
// INDEX, never real wall-clock time -- the same standing discipline this
// book has applied to every timing-adjacent value it has ever needed to
// keep deterministic and cross-architecture-comparable, from Chapter
// 10's thread scheduling to Chapter 18.1's trigger controller.
//
// The one real correctness discipline this section centers on: a store
// that did not report a period's snapshot is NOT the same fact as a
// store reporting zero on-hand inventory, and treating the two as
// interchangeable would fabricate a stockout signal for a store that may
// simply have missed a scheduled upload. `aggregate_by_sku` reports
// exactly which expected stores are missing from a given period,
// excludes them from that period's total rather than assuming zero, and
// reports how many stores' worth of real data the total actually
// reflects -- so a caller reading the aggregate can tell "42 units,
// fully reported" apart from "42 units, only 2 of 5 stores checked in."
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_multistore_aggregation_and_trends.cpp -o 04_multistore_aggregation_and_trends
// Run:     ./04_multistore_aggregation_and_trends

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: snapshots and honest, gap-aware aggregation.
// =======================================================================
struct StoreSnapshot {
    std::string store_id;
    std::string sku_id;
    int period = 0;                  // a discrete, caller-defined period index -- never real time
    int64_t on_hand_quantity = 0;
};

struct AggregateResult {
    int64_t total_quantity = 0;
    int stores_reporting = 0;
    std::vector<std::string> stores_missing;   // sorted, for deterministic output
};

// Sums `on_hand_quantity` across every snapshot matching `sku_id` and
// `period`, but ONLY for stores in `expected_stores` -- any expected
// store with no matching snapshot for this exact period is reported in
// `stores_missing` and excluded entirely from `total_quantity`, never
// silently treated as reporting zero.
AggregateResult aggregate_by_sku(const std::vector<StoreSnapshot>& snapshots, const std::string& sku_id,
                                  int period, const std::vector<std::string>& expected_stores) {
    AggregateResult res;
    std::set<std::string> reported;
    for (const auto& s : snapshots) {
        if (s.sku_id != sku_id || s.period != period) continue;
        if (!reported.insert(s.store_id).second) continue;   // a duplicate snapshot for the same store+period is not double-counted
        res.total_quantity += s.on_hand_quantity;
        ++res.stores_reporting;
    }
    for (const auto& store : expected_stores) {
        if (!reported.count(store)) res.stores_missing.push_back(store);
    }
    std::sort(res.stores_missing.begin(), res.stores_missing.end());
    return res;
}

// =======================================================================
// PART 2: real least-squares linear regression over a (period, quantity)
// time series, and the two real trend signals it powers.
// =======================================================================
enum class TrendSignal { Stable, StockoutRisk, Overstock };

std::string to_string(TrendSignal s) {
    switch (s) {
        case TrendSignal::Stable: return "STABLE";
        case TrendSignal::StockoutRisk: return "STOCKOUT_RISK";
        case TrendSignal::Overstock: return "OVERSTOCK";
    }
    return "UNKNOWN";
}

struct TrendThresholds {
    double stockout_slope_per_period = -2.0;    // flag only if declining at least this fast
    double overstock_slope_per_period = 5.0;    // flag only if growing at least this fast
    int projection_horizon_periods = 20;        // a stockout must be projected within this many periods to be actionable
};

struct TrendResult {
    TrendSignal signal = TrendSignal::Stable;
    double slope = 0.0;
    double intercept = 0.0;
    std::optional<int> projected_zero_period;
};

// A real ordinary-least-squares fit of quantity as a linear function of
// period: slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx), intercept from the
// mean point. The series is sorted by period internally so a caller does
// not have to pre-sort it, and needs at least two distinct periods --
// a single data point has no trend to fit at all.
TrendResult detect_trend(std::vector<std::pair<int, int64_t>> series, const TrendThresholds& t) {
    TrendResult res;
    std::sort(series.begin(), series.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const size_t n = series.size();
    if (n < 2) return res;   // Stable by default -- nothing to fit

    double sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (const auto& [period, qty] : series) {
        double x = static_cast<double>(period);
        double y = static_cast<double>(qty);
        sx += x; sy += y; sxy += x * y; sxx += x * x;
    }
    double n_d = static_cast<double>(n);
    double denom = n_d * sxx - sx * sx;
    if (denom == 0.0) return res;   // every period identical -- degenerate, treat as Stable rather than dividing by zero

    double slope = (n_d * sxy - sx * sy) / denom;
    double mean_x = sx / n_d, mean_y = sy / n_d;
    double intercept = mean_y - slope * mean_x;
    res.slope = slope;
    res.intercept = intercept;

    int last_period = series.back().first;

    if (slope <= t.stockout_slope_per_period) {
        // The regression line's own zero-crossing: intercept + slope*period = 0.
        double zero_period_d = -intercept / slope;
        int zero_period = static_cast<int>(std::ceil(zero_period_d));
        if (zero_period >= last_period && zero_period <= last_period + t.projection_horizon_periods) {
            res.signal = TrendSignal::StockoutRisk;
            res.projected_zero_period = zero_period;
            return res;
        }
        // Declining fast enough by rate, but the projected zero-crossing
        // falls outside the actionable horizon -- not flagged. A decline
        // that will not matter for years is not the same alert as one
        // that will matter next week, even at an identical per-period
        // rate on a much larger current quantity.
        return res;
    }
    if (slope >= t.overstock_slope_per_period) {
        res.signal = TrendSignal::Overstock;
        return res;
    }
    return res;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 19.4: Multi-Store Aggregation and Trend Detection\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: aggregation across fully-reporting stores matches a hand-computed total --\n";
    {
        std::vector<StoreSnapshot> snaps = {
            {"store-A", "SKU-1", 3, 100},
            {"store-B", "SKU-1", 3, 150},
            {"store-C", "SKU-1", 3, 75},
            {"store-A", "SKU-1", 2, 999},   // a different period -- must not be included
        };
        auto r = aggregate_by_sku(snaps, "SKU-1", 3, {"store-A", "store-B", "store-C"});
        CHECK(r.total_quantity == 325);
        CHECK(r.stores_reporting == 3);
        CHECK(r.stores_missing.empty());
        std::cout << "  3 stores at period 3: 100 + 150 + 75 = " << r.total_quantity
                   << ", all " << r.stores_reporting << " expected stores reporting, none missing\n";
    }

    std::cout << "\n-- Test 2: a store that did not report is excluded, never assumed to be zero --\n";
    {
        std::vector<StoreSnapshot> snaps = {
            {"store-A", "SKU-1", 5, 100},
            {"store-B", "SKU-1", 5, 150},
            // store-C has no snapshot at all for period 5
        };
        auto r = aggregate_by_sku(snaps, "SKU-1", 5, {"store-A", "store-B", "store-C"});
        CHECK(r.total_quantity == 250);          // NOT 250 + 0 == still 250, but for the RIGHT reason: store-C excluded, not counted as zero
        CHECK(r.stores_reporting == 2);
        CHECK(r.stores_missing.size() == 1 && r.stores_missing[0] == "store-C");
        std::cout << "  store-C never reported period 5: total is " << r.total_quantity
                   << " from " << r.stores_reporting << " reporting stores, with store-C correctly "
                     "listed as missing rather than silently folded into the total as a zero\n";
    }

    std::cout << "\n-- Test 3: a perfectly linear decline is flagged StockoutRisk with a hand-verifiable projection --\n";
    {
        std::vector<std::pair<int, int64_t>> series = {{0, 100}, {1, 90}, {2, 80}, {3, 70}, {4, 60}, {5, 50}};
        TrendThresholds t;   // stockout_slope_per_period = -2.0, horizon = 20
        auto r = detect_trend(series, t);
        CHECK(r.signal == TrendSignal::StockoutRisk);
        CHECK(std::abs(r.slope - (-10.0)) < 1e-9);       // exact for this perfectly linear series
        CHECK(std::abs(r.intercept - 100.0) < 1e-9);
        CHECK(r.projected_zero_period.has_value());
        CHECK(*r.projected_zero_period == 10);           // 100 - 10*period == 0 at period 10, hand-computed
        std::cout << "  a perfectly linear decline of 10 units/period (100 down to 50 over periods 0-5) "
                     "fits a regression slope of " << r.slope << " and intercept " << r.intercept
                   << ", correctly flagged StockoutRisk with a projected zero-inventory period of "
                   << *r.projected_zero_period << " -- matching the hand-computed zero-crossing exactly\n";
    }

    std::cout << "\n-- Test 4: a flat, noisy series is correctly left Stable --\n";
    {
        std::vector<std::pair<int, int64_t>> series = {{0, 100}, {1, 101}, {2, 99}, {3, 100}, {4, 100}, {5, 101}};
        auto r = detect_trend(series, TrendThresholds{});
        CHECK(r.signal == TrendSignal::Stable);
        std::cout << "  a series hovering around 100 units with small noise fits a near-zero slope ("
                   << r.slope << ") and is correctly left STABLE\n";
    }

    std::cout << "\n-- Test 5: a growing series is flagged Overstock --\n";
    {
        std::vector<std::pair<int, int64_t>> series = {{0, 50}, {1, 60}, {2, 70}, {3, 80}, {4, 90}, {5, 100}};
        auto r = detect_trend(series, TrendThresholds{});
        CHECK(r.signal == TrendSignal::Overstock);
        CHECK(std::abs(r.slope - 10.0) < 1e-9);
        std::cout << "  a perfectly linear growth of 10 units/period fits slope " << r.slope
                   << " and is correctly flagged OVERSTOCK\n";
    }

    std::cout << "\n-- Test 6: a decline fast enough by RATE but projected far beyond the horizon is not flagged --\n";
    {
        // Slope here is a steep -50/period (comfortably past the -2.0
        // rate threshold), but the starting quantity is enormous, so the
        // real zero-crossing sits thousands of periods away -- far
        // beyond any actionable horizon.
        std::vector<std::pair<int, int64_t>> series;
        int64_t start = 1'000'000;
        for (int p = 0; p <= 5; ++p) series.push_back({p, start - 50 * p});
        TrendThresholds t;   // projection_horizon_periods = 20
        auto r = detect_trend(series, t);
        CHECK(r.signal != TrendSignal::StockoutRisk);
        CHECK(r.signal == TrendSignal::Stable);
        double implied_zero = -r.intercept / r.slope;
        CHECK(implied_zero > 5 + t.projection_horizon_periods);   // confirm the real zero-crossing IS far beyond the horizon
        std::cout << "  a steep -50/period decline (slope " << r.slope << ", comfortably past the rate "
                     "threshold) starting from " << start << " units projects to zero at period "
                   << implied_zero << ", far beyond the " << t.projection_horizon_periods
                   << "-period horizon -- correctly left STABLE rather than raising an alert about a "
                     "shortage that is not actually imminent\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
