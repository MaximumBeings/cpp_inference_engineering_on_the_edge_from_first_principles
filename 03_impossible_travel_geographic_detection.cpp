// Chapter 29.3 -- A third, independent real AML and card-fraud signal
// asks a different question again: could the SAME physical person
// plausibly have been at both of two transaction locations, given how
// little time passed between them? "Impossible travel" is a real, well-
// documented technique used across both AML transaction monitoring and
// account-security fraud detection: compute the real great-circle
// distance between two consecutive transaction locations, divide by the
// real elapsed time, and compare the required speed against a
// deliberately GENEROUS real bound -- fast enough to cover an actual
// commercial flight, so a genuine cross-country traveler is never
// flagged. Only a required speed beyond even that generous bound is
// truly IMPOSSIBLE, not merely unusual, which is exactly what this
// section's own name promises and nothing more.
//
// The real distance calculation is the Haversine formula -- a real,
// published great-circle-distance formula that has computed distances
// between two points on a sphere since well before any digital computer
// existed, exactly the same "real, published formula, not invented for
// this book" standard Chapter 27.1's own microprice formula and Chapter
// 26.1's own ABA checksum already met.
//
// A note on this section's own honest scope: MAX_PLAUSIBLE_SPEED_KMH is
// stated at 900 km/h -- the real cruising speed of a typical commercial
// jet -- specifically chosen to be generous enough that no genuine
// traveler is ever flagged. This section's own COMMON TRAP box returns
// to exactly what this generosity trades away.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_impossible_travel_geographic_detection.cpp -o 03_impossible_travel_geographic_detection
// Run:     ./03_impossible_travel_geographic_detection

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numbers>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
static bool near_eq(double a, double b, double eps) { return std::fabs(a - b) < eps; }
#define CHECK_NEAR(a, b, eps) CHECK(near_eq((a), (b), (eps)))

// A transaction location, at a stated capture minute (a double, since
// this section's own real speed calculation genuinely needs continuous
// time, not a whole-minute reduction the way Section 29.2's frequency
// count did).
struct GeoTransaction {
    std::string account_id;
    double minute;
    double lat_deg;
    double lon_deg;
};

// The real, published Haversine great-circle-distance formula. R is
// Earth's own real mean radius in kilometers.
double haversine_distance_km(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
    constexpr double R = 6371.0;
    double phi1 = lat1_deg * std::numbers::pi / 180.0;
    double phi2 = lat2_deg * std::numbers::pi / 180.0;
    double dphi = (lat2_deg - lat1_deg) * std::numbers::pi / 180.0;
    double dlambda = (lon2_deg - lon1_deg) * std::numbers::pi / 180.0;
    double a = std::sin(dphi / 2.0) * std::sin(dphi / 2.0) +
               std::cos(phi1) * std::cos(phi2) * std::sin(dlambda / 2.0) * std::sin(dlambda / 2.0);
    double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
    return R * c;
}

struct TravelResult {
    bool flagged = false;
    double distance_km = 0.0;
    double required_speed_kmh = 0.0;
};

// Checks ONE consecutive pair of transactions from the same account.
// Flagged only when the required speed to cover the real distance in
// the real elapsed time is STRICTLY GREATER than the stated maximum
// plausible speed -- exactly at that speed is still, if barely, real.
TravelResult check_travel_pair(const GeoTransaction& earlier, const GeoTransaction& later,
                                double max_speed_kmh) {
    TravelResult r;
    r.distance_km = haversine_distance_km(earlier.lat_deg, earlier.lon_deg, later.lat_deg, later.lon_deg);
    double elapsed_hours = (later.minute - earlier.minute) / 60.0;
    r.required_speed_kmh = (elapsed_hours > 0.0) ? r.distance_km / elapsed_hours : 0.0;
    r.flagged = r.required_speed_kmh > max_speed_kmh;
    return r;
}

// Scans every consecutive pair in one account's own sorted transaction
// history, returning the first pair found to require an impossible
// speed.
TravelResult detect_impossible_travel(const std::vector<GeoTransaction>& account_history, double max_speed_kmh) {
    for (std::size_t i = 0; i + 1 < account_history.size(); ++i) {
        auto r = check_travel_pair(account_history[i], account_history[i + 1], max_speed_kmh);
        if (r.flagged) return r;
    }
    return {};
}

std::vector<std::string> flag_impossible_travel_accounts(std::vector<GeoTransaction> all_transactions,
                                                           double max_speed_kmh) {
    std::map<std::string, std::vector<GeoTransaction>> by_account;
    for (const auto& t : all_transactions) by_account[t.account_id].push_back(t);
    std::vector<std::string> flagged;
    for (auto& [account_id, history] : by_account) {
        std::sort(history.begin(), history.end(), [](const auto& a, const auto& b) { return a.minute < b.minute; });
        if (detect_impossible_travel(history, max_speed_kmh).flagged) flagged.push_back(account_id);
    }
    return flagged;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 29.3: Impossible Travel Geographic Detection\n";
    std::cout << "========================================================\n\n";

    const double MAX_PLAUSIBLE_SPEED_KMH = 900.0;
    // Two real, publicly known city coordinates.
    const double NYC_LAT = 40.7128, NYC_LON = -74.0060;
    const double LA_LAT = 34.0522, LA_LON = -118.2437;
    const double LONDON_LAT = 51.5074, LONDON_LON = -0.1278;
    const double PARIS_LAT = 48.8566, PARIS_LON = 2.3522;

    double dist_nyc_la = haversine_distance_km(NYC_LAT, NYC_LON, LA_LAT, LA_LON);
    double dist_london_paris = haversine_distance_km(LONDON_LAT, LONDON_LON, PARIS_LAT, PARIS_LON);
    std::cout << "(reference) real great-circle distance New York <-> Los Angeles: " << dist_nyc_la
              << " km; London <-> Paris: " << dist_london_paris << " km\n\n";

    // -- Test 1: the identical location, one minute apart -- zero
    // distance means zero required speed, never flagged regardless of
    // how little time passed. --
    {
        GeoTransaction a{"ACC-1", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-1", 1.0, NYC_LAT, NYC_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 1: identical location, 1 minute apart -- distance=" << r.distance_km
                  << " km, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(!r.flagged);
        CHECK_NEAR(r.distance_km, 0.0, 1e-9);
    }

    // -- Test 2: New York to Los Angeles in 6 real hours -- a genuinely
    // plausible cross-country flight -- not flagged. --
    {
        GeoTransaction a{"ACC-2", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-2", 360.0, LA_LAT, LA_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 2: NYC to LA in 6 hours -- required_speed=" << r.required_speed_kmh
                  << " km/h, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(!r.flagged);
        CHECK_NEAR(r.required_speed_kmh, dist_nyc_la / 6.0, 1e-6);
    }

    // -- Test 3: the identical NYC-to-LA distance, but in 30 minutes --
    // a real, genuine impossibility. --
    {
        GeoTransaction a{"ACC-3", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-3", 30.0, LA_LAT, LA_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 3: NYC to LA in 30 minutes -- required_speed=" << r.required_speed_kmh
                  << " km/h, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(r.flagged);
        CHECK(r.required_speed_kmh > MAX_PLAUSIBLE_SPEED_KMH);
    }

    // -- Test 4: the exact boundary -- elapsed time constructed so the
    // required speed lands EXACTLY on the stated maximum, confirmed not
    // flagged (an inclusive boundary, the same convention this book has
    // used for every real threshold since Chapter 26.1). --
    {
        double boundary_hours = dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH;
        GeoTransaction a{"ACC-4", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-4", boundary_hours * 60.0, LA_LAT, LA_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 4: NYC to LA in exactly " << (boundary_hours * 60.0)
                  << " minutes -- required_speed=" << r.required_speed_kmh << " km/h (max="
                  << MAX_PLAUSIBLE_SPEED_KMH << "), flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(!r.flagged);
        CHECK_NEAR(r.required_speed_kmh, MAX_PLAUSIBLE_SPEED_KMH, 1e-6);
    }

    // -- Test 5: one minute less than that exact boundary -- now
    // genuinely over the stated maximum, correctly flagged. --
    {
        double boundary_hours = dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH;
        GeoTransaction a{"ACC-5", 0.0, NYC_LAT, NYC_LON};
        GeoTransaction b{"ACC-5", boundary_hours * 60.0 - 1.0, LA_LAT, LA_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 5: one minute less than that boundary -- required_speed=" << r.required_speed_kmh
                  << " km/h, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(r.flagged);
        CHECK(r.required_speed_kmh > MAX_PLAUSIBLE_SPEED_KMH);
    }

    // -- Test 6: a much SHORTER real distance (London to Paris) but a
    // very short elapsed time -- flagged too, confirming this check is
    // genuinely about SPEED, not raw distance. --
    {
        GeoTransaction a{"ACC-6", 0.0, LONDON_LAT, LONDON_LON};
        GeoTransaction b{"ACC-6", 10.0, PARIS_LAT, PARIS_LON};
        auto r = check_travel_pair(a, b, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 6: London to Paris (" << dist_london_paris << " km) in 10 minutes -- required_speed="
                  << r.required_speed_kmh << " km/h, flagged=" << (r.flagged ? "YES" : "NO") << " --\n";
        CHECK(r.flagged);
    }

    // -- Test 7: a mixed batch across two accounts -- only the account
    // exhibiting impossible travel is named. --
    {
        std::vector<GeoTransaction> mixed = {
            {"ACC-7A", 0.0, NYC_LAT, NYC_LON}, {"ACC-7A", 30.0, LA_LAT, LA_LON},
            {"ACC-7B", 0.0, LONDON_LAT, LONDON_LON}, {"ACC-7B", 360.0, PARIS_LAT, PARIS_LON},
        };
        auto flagged = flag_impossible_travel_accounts(mixed, MAX_PLAUSIBLE_SPEED_KMH);
        std::cout << "-- Test 7: two accounts, one impossible-travel pattern, one normal -- flagged=[";
        for (std::size_t i = 0; i < flagged.size(); i++) std::cout << (i ? ", " : "") << flagged[i];
        std::cout << "] --\n";
        CHECK(flagged.size() == 1);
        CHECK(flagged[0] == "ACC-7A");
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
