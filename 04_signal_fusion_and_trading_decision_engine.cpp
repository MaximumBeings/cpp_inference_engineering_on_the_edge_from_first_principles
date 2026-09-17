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
