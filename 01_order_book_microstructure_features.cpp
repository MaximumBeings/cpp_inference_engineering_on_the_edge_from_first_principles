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
