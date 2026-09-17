// Chapter 28.2 -- A real, well-documented technique real insurance Special
// Investigations Units (SIUs) use is metadata timestamp analysis: every
// digital photo carries a real capture timestamp, and comparing it
// against a claim's own stated incident date catches a specific, honest
// structural impossibility -- a photo of damage cannot have been taken
// BEFORE the damage occurred. A photo timestamped earlier than a claim's
// own stated incident date is not a matter of interpretation; it is a
// real, checkable contradiction in the claim's own submitted evidence.
//
// This section builds that check from scratch using a simple, stated
// integer day-count in place of full calendar-date parsing -- exactly
// the same "reduce a real-world unit to the smallest integer that still
// carries the real property under test" discipline Section 27.2 already
// applied to microsecond timestamps rather than a full wall-clock value.
// A second, softer real signal is checked alongside it: a photo
// timestamped long after a claim was already reported may be entirely
// legitimate (follow-up documentation taken during a real repair), but
// it is still worth naming for a human reviewer rather than silently
// accepted without comment.
//
// A note on this section's own honest scope: `check_photo_timing` names
// TWO different real timing anomalies with two different real
// implications -- a pre-dated photo is a genuine structural
// impossibility, while a late photo is merely unusual -- and this
// section's own COMMON TRAP box returns to exactly that distinction.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_photo_timestamp_incident_consistency.cpp -o 02_photo_timestamp_incident_consistency
// Run:     ./02_photo_timestamp_incident_consistency

#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// A claim's own stated timeline, in whole days since an arbitrary,
// unnamed epoch -- this section never needs a real calendar, only the
// real ORDERING and DISTANCE between three real dates.
struct ClaimWindow {
    int incident_day;
    int report_day;
};

enum class TimingVerdict { CONSISTENT, PRE_DATED, STALE_OR_LATE };

struct TimingResult {
    TimingVerdict verdict;
    std::string note;
};

// A photo timestamped strictly BEFORE the incident day is a real,
// structural impossibility -- the damage had not yet happened. A photo
// timestamped after the claim's own report day, beyond a stated grace
// period (real follow-up photographs taken during an active repair are
// routine), is merely unusual, not impossible, and is named as such: a
// softer, STALE_OR_LATE signal rather than a hard contradiction.
TimingResult check_photo_timing(const ClaimWindow& window, int captured_day, int grace_period_days) {
    if (captured_day < window.incident_day) {
        int gap = window.incident_day - captured_day;
        return {TimingVerdict::PRE_DATED,
                "photo captured " + std::to_string(gap) +
                    " day(s) BEFORE the claim's own stated incident day -- a structural impossibility"};
    }
    const int latest_allowed = window.report_day + grace_period_days;
    if (captured_day > latest_allowed) {
        int gap = captured_day - latest_allowed;
        return {TimingVerdict::STALE_OR_LATE,
                "photo captured " + std::to_string(gap) +
                    " day(s) beyond the stated " + std::to_string(grace_period_days) +
                    "-day grace period after the claim's own report day"};
    }
    return {TimingVerdict::CONSISTENT, "photo timestamp falls within the claim's own expected window"};
}

struct ClaimPhotoTimestamp {
    std::string zone_label;
    int captured_day;
};

// Aggregates every photo attached to one claim -- never drops or
// silently averages an anomalous photo's own flag away.
std::vector<std::string> assess_claim_timing(const ClaimWindow& window,
                                              const std::vector<ClaimPhotoTimestamp>& photos,
                                              int grace_period_days) {
    std::vector<std::string> flagged;
    for (const auto& p : photos) {
        auto result = check_photo_timing(window, p.captured_day, grace_period_days);
        if (result.verdict != TimingVerdict::CONSISTENT) {
            flagged.push_back(p.zone_label + ": " + result.note);
        }
    }
    return flagged;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 28.2: Photo Timestamp vs. Incident Date Consistency\n";
    std::cout << "========================================================\n\n";

    const ClaimWindow window{/*incident_day=*/100, /*report_day=*/105};
    const int GRACE_PERIOD_DAYS = 14;

    // -- Test 1: a photo captured between the incident day and the
    // report day -- the ordinary, expected case. --
    {
        auto r = check_photo_timing(window, 102, GRACE_PERIOD_DAYS);
        std::cout << "-- Test 1: incident=day 100, report=day 105, photo captured day 102 -- verdict="
                  << (r.verdict == TimingVerdict::CONSISTENT ? "CONSISTENT" : "OTHER") << " --\n";
        CHECK(r.verdict == TimingVerdict::CONSISTENT);
    }

    // -- Test 2: a photo captured before the incident day -- a real
    // structural impossibility, named with the exact day gap. --
    {
        auto r = check_photo_timing(window, 99, GRACE_PERIOD_DAYS);
        std::cout << "-- Test 2: photo captured day 99, before incident day 100 -- verdict="
                  << (r.verdict == TimingVerdict::PRE_DATED ? "PRE_DATED" : "OTHER") << ", note=\"" << r.note
                  << "\" --\n";
        CHECK(r.verdict == TimingVerdict::PRE_DATED);
        CHECK(r.note.find("1 day(s) BEFORE") != std::string::npos);
    }

    // -- Test 3: the incident-day boundary itself -- a photo captured ON
    // the exact incident day (for instance, taken immediately after a
    // real accident) is legitimate, confirmed as an inclusive boundary. --
    {
        auto r = check_photo_timing(window, 100, GRACE_PERIOD_DAYS);
        std::cout << "-- Test 3: photo captured exactly on incident day 100 -- verdict="
                  << (r.verdict == TimingVerdict::CONSISTENT ? "CONSISTENT" : "OTHER") << " --\n";
        CHECK(r.verdict == TimingVerdict::CONSISTENT);
    }

    // -- Test 4: the grace-period boundary itself -- a photo captured
    // EXACTLY at report_day + grace_period_days is still within bounds,
    // the same inclusive convention as Test 3. --
    {
        auto r = check_photo_timing(window, 105 + GRACE_PERIOD_DAYS, GRACE_PERIOD_DAYS);
        std::cout << "-- Test 4: photo captured exactly at the grace-period boundary (day "
                  << (105 + GRACE_PERIOD_DAYS) << ") -- verdict="
                  << (r.verdict == TimingVerdict::CONSISTENT ? "CONSISTENT" : "OTHER") << " --\n";
        CHECK(r.verdict == TimingVerdict::CONSISTENT);
    }

    // -- Test 5: one day past that same grace-period boundary -- flagged
    // STALE_OR_LATE, not treated as a hard impossibility. --
    {
        auto r = check_photo_timing(window, 105 + GRACE_PERIOD_DAYS + 1, GRACE_PERIOD_DAYS);
        std::cout << "-- Test 5: photo captured one day past the grace period (day "
                  << (105 + GRACE_PERIOD_DAYS + 1) << ") -- verdict="
                  << (r.verdict == TimingVerdict::STALE_OR_LATE ? "STALE_OR_LATE" : "OTHER") << ", note=\""
                  << r.note << "\" --\n";
        CHECK(r.verdict == TimingVerdict::STALE_OR_LATE);
        CHECK(r.note.find("1 day(s) beyond") != std::string::npos);
    }

    // -- Test 6: a photo captured well before the incident, confirming
    // the exact day-gap arithmetic generalizes beyond a 1-day difference. --
    {
        auto r = check_photo_timing(window, 30, GRACE_PERIOD_DAYS);
        std::cout << "-- Test 6: photo captured day 30, 70 days before incident day 100 -- verdict="
                  << (r.verdict == TimingVerdict::PRE_DATED ? "PRE_DATED" : "OTHER") << ", note=\"" << r.note
                  << "\" --\n";
        CHECK(r.verdict == TimingVerdict::PRE_DATED);
        CHECK(r.note.find("70 day(s) BEFORE") != std::string::npos);
    }

    // -- Test 7: a full multi-photo claim -- exactly one pre-dated photo
    // among two otherwise consistent ones is named by its own zone
    // label, never dropped or averaged away by the honest majority. --
    {
        std::vector<ClaimPhotoTimestamp> photos = {
            {"front_bumper", 102},
            {"hood", 96},
            {"rear_bumper", 103},
        };
        auto flagged = assess_claim_timing(window, photos, GRACE_PERIOD_DAYS);
        std::cout << "-- Test 7: 3 photos (front_bumper day 102, hood day 96, rear_bumper day 103) -- "
                     "flagged=" << flagged.size() << " --\n";
        for (const auto& f : flagged) std::cout << "     " << f << "\n";
        CHECK(flagged.size() == 1);
        CHECK(flagged[0].find("hood") != std::string::npos);
        CHECK(flagged[0].find("BEFORE") != std::string::npos);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
