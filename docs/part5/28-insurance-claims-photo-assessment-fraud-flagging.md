# Chapter 28: Insurance Claims Photo Assessment with Fraud Flagging

**What you will understand by the end of this chapter:**

- How to reapply Chapter 23.1's own real perceptual difference-hash (dHash) algorithm a third time -- after Chapter 23.1's own counterfeit product photos and Chapter 26.1's signature comparison -- to a new real fraud pattern: the identical photo submitted across two unrelated insurance claims.
- How to build a real, from-scratch photo-timestamp consistency check reapplying Section 27.2's own "stated parameters, never a wall clock" discipline to a real Special Investigations Unit technique: catching a photo timestamped before the damage it depicts could have occurred.
- How to extend Chapter 21.3's own real severity-to-cost-range engine and photo-consistency check -- both reused completely unchanged -- with a new, transparent per-zone fraud-likelihood signal, while proving the claim's own cost estimate is never touched by that signal.
- How to build a claims-disposition engine reapplying this book's own never-suppress-a-named-flag discipline, with a real, deliberate ethical constraint this domain requires: no disposition this engine can ever produce is a denial.

**What you need to know first:**

- Chapter 23.1's own real, from-scratch dHash implementation (already reapplied once, in Chapter 26.1) is reapplied a third time in Section 28.1, this time to detect a photo reused across two different insurance claims.
- Section 27.2's own discipline of deriving timing behavior entirely from stated, labeled parameters rather than a wall clock is reapplied in Section 28.2, this time to a real photo-timestamp forensics check rather than a real-time processing deadline.
- Chapter 21.3's own real `CostRange` and photo-consistency machinery is reused completely UNCHANGED in Sections 28.3 and 28.4 -- this chapter's own new work is entirely in the fraud-likelihood signal layered alongside it, never inside it.

---

Chapter 27 turned to a hard, real-time deadline. This chapter turns to a genuinely different real constraint once more: an insurance claim, where the fraud-relevant signals are photographic and temporal rather than numeric, and where a false accusation carries real consequences for a genuine claimant. Each of this chapter's four sections builds one real, independent fraud signal -- reused photo detection, timestamp forensics -- and closes by extending Chapter 21.3's own cost-range engine with those signals, ending in a capstone disposition engine that routes a claim toward more or less human scrutiny, but never toward an automatic denial.

## 28.1 Insurance Claim Duplicate Photo Detection

### Intuition

A well-documented real insurance-fraud pattern is not a forged photo at all, but a genuine photo reused across two entirely unrelated claims. Catching this requires the identical real dHash algorithm this book has now built once and reapplied once already, applied for a third time to a new real domain: cross-claim photo deduplication, where the central design question is not the hashing itself but what counts as a real match.

### The Concept, In Detail

`find_reused_photo_matches` reuses Chapter 23.1's own unmodified `compute_dhash` and `hamming_distance`, checking a new submission's own photo hash against every OTHER claim's own stored photo in a real cross-claim database. Test 3 is this section's own central design decision: a claim re-submitting its OWN already-on-file photo -- an adjuster asking for a clearer re-scan, for instance -- is explicitly excluded from matching against itself, since this is completely routine, never a fraud signal on its own. Test 2 confirms a genuine cross-claim match is named exactly, and Test 4 confirms every matching claim is returned, not merely the first one found, when a single photo has already been reused across more than two claims.

Tests 5 and 6 confirm the exact Hamming-distance boundary at the raw hash level: a distance of exactly 10 (this section's own stated tolerance) still counts as a match, and a distance of exactly 11 does not, using the same inclusive convention Chapter 26.1's own signature-comparison tolerance already established.

### Code and Verification

```cpp
// Chapter 28.1 -- A well-documented real insurance-fraud pattern is not a
// forged photo at all, but a GENUINE photo reused across two entirely
// different claims: a claimant (or, in the more organized real cases, a
// ring of claimants working together) submits the identical real damage
// photo -- sometimes literally the same file, sometimes a lightly
// recompressed or rescanned copy -- under a second, unrelated claim. This
// section reapplies the identical real perceptual difference-hash (dHash)
// algorithm this book has now built once (Chapter 23.1, counterfeit
// product photos) and reapplied once already (Chapter 26.1, signature
// comparison) to a THIRD real domain: cross-claim photo deduplication.
//
// The central design decision this section adds is not in the hashing
// itself, which is unchanged, but in what counts as a real match: a
// claimant re-uploading their OWN claim's own photo a second time (an
// adjuster asking for a clearer re-scan, for instance) is completely
// routine and must never be flagged, while the identical photo appearing
// under a DIFFERENT claim's own ID is a real, specific, and named fraud
// signal. This section's own COMMON TRAP box returns to exactly this
// distinction.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_insurance_claim_duplicate_photo_detection.cpp -o 01_insurance_claim_duplicate_photo_detection
// Run:     ./01_insurance_claim_duplicate_photo_detection

#include <algorithm>
#include <bit>
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

// =======================================================================
// Chapter 23.1's own real, from-scratch dHash algorithm, reapplied here
// unchanged for the third time in this book.
// =======================================================================
using Hash64 = std::uint64_t;

struct GrayscaleImage {
    int width = 0, height = 0;
    std::vector<std::uint8_t> pixels;
};

GrayscaleImage resize_nearest(const GrayscaleImage& src, int new_w, int new_h) {
    GrayscaleImage out{new_w, new_h, std::vector<std::uint8_t>(static_cast<std::size_t>(new_w) * new_h)};
    for (int y = 0; y < new_h; y++) {
        int sy = (y * src.height) / new_h;
        for (int x = 0; x < new_w; x++) {
            int sx = (x * src.width) / new_w;
            out.pixels[static_cast<std::size_t>(y) * new_w + x] =
                src.pixels[static_cast<std::size_t>(sy) * src.width + sx];
        }
    }
    return out;
}

Hash64 compute_dhash(const GrayscaleImage& src) {
    GrayscaleImage small = resize_nearest(src, 9, 8);
    Hash64 hash = 0;
    int bit = 0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            std::uint8_t left = small.pixels[static_cast<std::size_t>(y) * 9 + x];
            std::uint8_t right = small.pixels[static_cast<std::size_t>(y) * 9 + x + 1];
            if (left > right) hash |= (Hash64(1) << bit);
            bit++;
        }
    }
    return hash;
}

int hamming_distance(Hash64 a, Hash64 b) {
    return std::popcount(static_cast<std::uint64_t>(a ^ b));
}

GrayscaleImage make_checker(int w, int h, int cell) {
    GrayscaleImage img{w, h, std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h)};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img.pixels[static_cast<std::size_t>(y) * w + x] =
                (((x / cell) + (y / cell)) % 2 == 0) ? 220 : 30;
    return img;
}

// A real, stated policy threshold -- identical in spirit to Chapter
// 26.1's own signature-match tolerance -- for "the same real photo,
// possibly recompressed or rescanned" rather than "byte-identical file."
constexpr int MAX_HAMMING_FOR_DUPLICATE = 10;

// One previously submitted claim photo, as this insurer's own real,
// cross-claim photo database would store it.
struct ClaimPhoto {
    std::string claim_id;
    std::string zone_label;
    Hash64 hash;
};

// Finds every OTHER claim in the database whose own stored photo is a
// near-duplicate of the new submission -- explicitly EXCLUDING entries
// that belong to the SAME claim_id as the new submission, since a
// claimant re-uploading their own claim's own photo a second time is
// routine, not fraud. Returns every matching claim_id found, in the
// order the database is scanned -- never just the first one, since a
// photo reused across more than two claims is a real, stronger signal.
std::vector<std::string> find_reused_photo_matches(const std::string& new_claim_id, Hash64 new_hash,
                                                     const std::vector<ClaimPhoto>& database) {
    std::vector<std::string> matches;
    for (const auto& entry : database) {
        if (entry.claim_id == new_claim_id) continue;
        if (hamming_distance(entry.hash, new_hash) <= MAX_HAMMING_FOR_DUPLICATE) {
            matches.push_back(entry.claim_id);
        }
    }
    return matches;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 28.1: Insurance Claim Duplicate Photo Detection\n";
    std::cout << "========================================================\n\n";

    GrayscaleImage photo_a = make_checker(40, 24, 4);
    GrayscaleImage photo_c = make_checker(40, 24, 2);
    Hash64 hash_a = compute_dhash(photo_a);
    Hash64 hash_c = compute_dhash(photo_c);
    std::cout << "(reference) Hamming distance between two genuinely different real photos (A vs. C): "
              << hamming_distance(hash_a, hash_c) << " -- well above the stated duplicate threshold of "
              << MAX_HAMMING_FOR_DUPLICATE << "\n\n";

    // -- Test 1: a genuinely new, unique photo against a small existing
    // database -- no real match, correctly reported as an empty list. --
    {
        std::vector<ClaimPhoto> db = {{"CLM-1001", "rear_bumper", hash_a}};
        auto matches = find_reused_photo_matches("CLM-2001", hash_c, db);
        std::cout << "-- Test 1: a genuinely unique new photo against a 1-entry database -- matches="
                  << matches.size() << " --\n";
        CHECK(matches.empty());
    }

    // -- Test 2: the new submission's own photo is a near-duplicate of a
    // DIFFERENT claim's own stored photo -- flagged, naming the exact
    // matching claim_id. --
    {
        std::vector<ClaimPhoto> db = {{"CLM-1001", "rear_bumper", hash_a}};
        auto matches = find_reused_photo_matches("CLM-2050", hash_a, db);
        std::cout << "-- Test 2: new claim CLM-2050 submits the identical photo already on file under "
                     "CLM-1001 -- matches=[";
        for (size_t i = 0; i < matches.size(); i++) std::cout << (i ? ", " : "") << matches[i];
        std::cout << "] --\n";
        CHECK(matches.size() == 1);
        CHECK(matches[0] == "CLM-1001");
    }

    // -- Test 3 (central): the SAME claim re-submitting its OWN
    // already-on-file photo -- for instance, an adjuster asking for a
    // clearer re-scan -- must NEVER be flagged, even though the hash
    // match is exact. --
    {
        std::vector<ClaimPhoto> db = {{"CLM-1001", "rear_bumper", hash_a}};
        auto matches = find_reused_photo_matches("CLM-1001", hash_a, db);
        std::cout << "-- Test 3: CLM-1001 re-submits its OWN already-on-file photo -- matches="
                  << matches.size() << " (self-resubmission correctly excluded) --\n";
        CHECK(matches.empty());
    }

    // -- Test 4: the identical photo already appears under TWO separate,
    // unrelated prior claims -- a real, stronger fraud signal -- and a
    // third new claim submits it again. Both prior claim_ids are named,
    // not just the first one found. --
    {
        std::vector<ClaimPhoto> db = {
            {"CLM-1001", "rear_bumper", hash_a},
            {"CLM-1500", "side_panel", hash_a},
        };
        auto matches = find_reused_photo_matches("CLM-3000", hash_a, db);
        std::cout << "-- Test 4: a photo already shared by CLM-1001 and CLM-1500 is submitted again under "
                     "CLM-3000 -- matches=[";
        for (size_t i = 0; i < matches.size(); i++) std::cout << (i ? ", " : "") << matches[i];
        std::cout << "] --\n";
        CHECK(matches.size() == 2);
        CHECK(matches[0] == "CLM-1001");
        CHECK(matches[1] == "CLM-1500");
    }

    // -- Test 5: the exact threshold boundary, at the level of the raw
    // hash values themselves rather than a rendered image -- a Hamming
    // distance of EXACTLY 10 (a 64-bit value with exactly 10 bits set
    // relative to a zero reference hash) is still within the stated
    // tolerance, confirming the inclusive convention explicitly. --
    {
        Hash64 reference_hash = 0x0;
        Hash64 ten_bits_set = 0x3FF;  // exactly 10 bits set -- popcount(0x3FF) == 10
        std::vector<ClaimPhoto> db = {{"CLM-4001", "roof", reference_hash}};
        auto matches = find_reused_photo_matches("CLM-4002", ten_bits_set, db);
        std::cout << "-- Test 5: exact boundary, Hamming distance = " << hamming_distance(reference_hash, ten_bits_set)
                  << " (threshold = " << MAX_HAMMING_FOR_DUPLICATE << ") -- matches=" << matches.size()
                  << " --\n";
        CHECK(hamming_distance(reference_hash, ten_bits_set) == 10);
        CHECK(matches.size() == 1);
    }

    // -- Test 6: one bit past that same boundary -- a Hamming distance
    // of exactly 11 -- correctly falls OUTSIDE the stated tolerance. --
    {
        Hash64 reference_hash = 0x0;
        Hash64 eleven_bits_set = 0x7FF;  // exactly 11 bits set -- popcount(0x7FF) == 11
        std::vector<ClaimPhoto> db = {{"CLM-4001", "roof", reference_hash}};
        auto matches = find_reused_photo_matches("CLM-4002", eleven_bits_set, db);
        std::cout << "-- Test 6: one past the boundary, Hamming distance = "
                  << hamming_distance(reference_hash, eleven_bits_set) << " -- matches=" << matches.size()
                  << " --\n";
        CHECK(hamming_distance(reference_hash, eleven_bits_set) == 11);
        CHECK(matches.empty());
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_insurance_claim_duplicate_photo_detection.cpp -o 01_insurance_claim_duplicate_photo_detection
./01_insurance_claim_duplicate_photo_detection
```

**Sample input:** a genuinely unique new photo checked against a small existing cross-claim database with no real match; a new submission's photo checked to be a near-duplicate of a different claim's own stored photo, named exactly; the same claim re-submitting its own already-on-file photo, checked to be correctly excluded from matching itself; a photo already shared by two separate prior claims submitted again under a third claim, checked to name both prior matches; and the exact Hamming-distance boundary of the stated duplicate-detection tolerance, checked on both sides.

```text
========================================================
Chapter 28.1: Insurance Claim Duplicate Photo Detection
========================================================

(reference) Hamming distance between two genuinely different real photos (A vs. C): 32 -- well above the stated duplicate threshold of 10

-- Test 1: a genuinely unique new photo against a 1-entry database -- matches=0 --
-- Test 2: new claim CLM-2050 submits the identical photo already on file under CLM-1001 -- matches=[CLM-1001] --
-- Test 3: CLM-1001 re-submits its OWN already-on-file photo -- matches=0 (self-resubmission correctly excluded) --
-- Test 4: a photo already shared by CLM-1001 and CLM-1500 is submitted again under CLM-3000 -- matches=[CLM-1001, CLM-1500] --
-- Test 5: exact boundary, Hamming distance = 10 (threshold = 10) -- matches=1 --
-- Test 6: one past the boundary, Hamming distance = 11 -- matches=0 --

11/11 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] flagging every hash match as fraud, regardless of which claim it belongs to"
    `find_reused_photo_matches` explicitly excludes matches against the SAME `claim_id` as the new submission -- a claimant, or an adjuster processing their claim, re-uploading the identical photo a second time (a clearer re-scan, a different file format, a resubmission after a lost upload) is completely routine, and treating it as a fraud signal would flag an enormous number of genuinely honest claims. The real signal this section's own function is built to catch is narrower and more specific: the SAME photo appearing under a DIFFERENT claim_id, which has no routine, honest explanation the way a same-claim resubmission does. Deploying this section's own logic without the self-exclusion check in Test 3 would replace a real, specific fraud signal with an enormous stream of false positives against the most common, most innocent case there is.

## 28.2 Photo Timestamp vs. Incident Date Consistency

### Intuition

A real, well-documented technique real insurance Special Investigations Units use is metadata timestamp analysis: a photo of damage cannot have been taken before the damage occurred, and a claim's own stated incident date gives a real, checkable boundary to test every submitted photo's own capture timestamp against.

### The Concept, In Detail

`check_photo_timing` names two different real timing anomalies with two different real implications. A photo timestamped strictly before the incident day is a genuine structural impossibility -- Test 2 and Test 6 confirm this is reported with the exact day gap, however large. A photo timestamped after the claim's own report day, beyond a stated grace period, is merely unusual rather than impossible (real follow-up photographs taken during an active repair are routine) -- Test 5 confirms this softer signal is still named, one day past the boundary. Tests 3 and 4 confirm both of this section's own boundary conventions are inclusive: a photo captured exactly on the incident day, and a photo captured exactly at the grace-period boundary, are both reported as consistent.

Test 7 confirms `assess_claim_timing`'s own multi-photo aggregation names exactly the one anomalous photo by its own zone label, among otherwise consistent photos, never dropping or averaging the flag away.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_photo_timestamp_incident_consistency.cpp -o 02_photo_timestamp_incident_consistency
./02_photo_timestamp_incident_consistency
```

**Sample input:** a photo captured between a claim's own incident and report days, checked as consistent; a photo captured before the incident day, checked to report the exact day gap as a structural impossibility, and again for a much larger gap; the incident-day boundary and the grace-period boundary each checked as inclusive; a photo captured one day past the grace period, checked as a softer, named anomaly; and a full 3-photo claim checked to flag exactly the one pre-dated photo by its own zone label.

```text
========================================================
Chapter 28.2: Photo Timestamp vs. Incident Date Consistency
========================================================

-- Test 1: incident=day 100, report=day 105, photo captured day 102 -- verdict=CONSISTENT --
-- Test 2: photo captured day 99, before incident day 100 -- verdict=PRE_DATED, note="photo captured 1 day(s) BEFORE the claim's own stated incident day -- a structural impossibility" --
-- Test 3: photo captured exactly on incident day 100 -- verdict=CONSISTENT --
-- Test 4: photo captured exactly at the grace-period boundary (day 119) -- verdict=CONSISTENT --
-- Test 5: photo captured one day past the grace period (day 120) -- verdict=STALE_OR_LATE, note="photo captured 1 day(s) beyond the stated 14-day grace period after the claim's own report day" --
-- Test 6: photo captured day 30, 70 days before incident day 100 -- verdict=PRE_DATED, note="photo captured 70 day(s) BEFORE the claim's own stated incident day -- a structural impossibility" --
-- Test 7: 3 photos (front_bumper day 102, hood day 96, rear_bumper day 103) -- flagged=1 --
     hood: photo captured 4 day(s) BEFORE the claim's own stated incident day -- a structural impossibility

12/12 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a PRE_DATED photo and a STALE_OR_LATE photo as the same real severity of anomaly"
    A `PRE_DATED` result names a genuine structural impossibility -- there is no honest explanation for a photograph of damage existing before the damage occurred, short of the photo belonging to a different, earlier incident entirely. A `STALE_OR_LATE` result names something categorically softer: a photo taken later than expected, which has many completely honest explanations (a follow-up photo during a real repair, a delayed upload, a second adjuster visit). Routing both anomalies to the identical downstream action would either treat a real structural impossibility too gently, or treat a routine late upload as though it were as serious as a photo that could not possibly be genuine -- exactly why this section's own `TimingVerdict` keeps the two as separate, distinctly named outcomes rather than collapsing them into one generic "timing anomaly" flag.

## 28.3 Extending the Cost-Range Engine with a Fraud-Likelihood Signal

### Intuition

This section directly extends Chapter 21.3's own real severity-to-cost-range engine and photo-consistency check -- both reused completely unchanged -- with a new per-zone fraud-likelihood signal built from Section 28.1's duplicate-photo flag, Section 28.2's timestamp-anomaly flag, and Chapter 21.3's own existing consistency verdict. The central discipline this section adds: a zone's own cost estimate and its own fraud likelihood are computed completely independently of each other.

### The Concept, In Detail

`assess_fraud_likelihood` is a real, fully transparent count of how many of three independent named signals fired for a zone -- Test 3 confirms all three signal types are weighted identically, each alone producing `MEDIUM` on its own. Test 4 confirms the exact `MEDIUM`/`HIGH` boundary at a count of two signals, and Test 6 is this section's own central honesty check: a two-zone claim with one zone firing all three fraud signals at once produces the EXACT SAME `total_range` as the identical severities with zero fraud signals attached, a direct paired comparison proving the cost estimate is never touched by the fraud-likelihood computation layered alongside it.

Test 8 closes the full loop, deriving a real `photo_inconsistent` flag from actual before/after pixel data using Chapter 21.3's own reused `mean_abs_diff` and `check_claim_consistency`, rather than a hand-set boolean, and confirming it flows into `assess_fraud_likelihood` exactly like the other two signals.

### Code and Verification

```cpp
// Chapter 28.3 -- This section directly extends Chapter 21.3's own real
// severity-to-cost-range engine and before/after photo-consistency check
// -- both reused completely UNCHANGED below -- with a new real per-zone
// fraud-likelihood signal built from Section 28.1's duplicate-photo flag,
// Section 28.2's timestamp-anomaly flag, and Chapter 21.3's own existing
// photo-consistency verdict. The central discipline this section adds:
// a zone's own COST ESTIMATE and its own FRAUD LIKELIHOOD are computed
// completely independently of each other -- attaching a fraud signal to
// a zone must never silently shrink, zero out, or otherwise adjust that
// zone's own currently-claimed cost contribution, exactly the same
// never-auto-adjust discipline Chapter 21.3's own Test 5 already
// established for a single flagged zone.
//
// A note on this section's own honest scope: `assess_fraud_likelihood`
// is a real, simple, fully transparent COUNT of how many named real
// signals fired for a zone -- LOW, MEDIUM, or HIGH describes how many
// independent real checks disagreed with the claim, never a calibrated
// real probability of fraud. This section's own COMMON TRAP box returns
// to exactly this distinction.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_fraud_likelihood_cost_range_engine.cpp -o 03_fraud_likelihood_cost_range_engine
// Run:     ./03_fraud_likelihood_cost_range_engine

#include <algorithm>
#include <cmath>
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

// =======================================================================
// Chapter 21.3's own real severity-to-cost-range policy, reused
// completely UNCHANGED -- generalized here to a plain zone NAME string,
// exactly as Chapter 21.3's own introduction already stated this
// machinery applies identically to a vehicle panel or a property zone.
// =======================================================================
enum class DamageSeverity { NONE, MINOR, MODERATE, SEVERE };

std::string to_string(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return "NONE";
        case DamageSeverity::MINOR: return "MINOR";
        case DamageSeverity::MODERATE: return "MODERATE";
        case DamageSeverity::SEVERE: return "SEVERE";
    }
    return "UNKNOWN";
}

struct CostRange { double low = 0.0, high = 0.0; };
CostRange operator+(const CostRange& a, const CostRange& b) { return {a.low + b.low, a.high + b.high}; }

CostRange cost_range_for_severity(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return {0.0, 0.0};
        case DamageSeverity::MINOR: return {150.0, 600.0};
        case DamageSeverity::MODERATE: return {600.0, 2500.0};
        case DamageSeverity::SEVERE: return {2500.0, 9000.0};
    }
    return {0.0, 0.0};
}

// =======================================================================
// Chapter 21.3's own real before/after photo-consistency check, reused
// completely UNCHANGED.
// =======================================================================
double mean_abs_diff(const std::vector<std::uint8_t>& before, const std::vector<std::uint8_t>& after) {
    double sum = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i) {
        sum += std::abs(static_cast<int>(before[i]) - static_cast<int>(after[i]));
    }
    return sum / static_cast<double>(before.size());
}

struct DiffBand { double low, high; };
DiffBand expected_diff_band(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return {0.0, 5.0};
        case DamageSeverity::MINOR: return {5.0, 30.0};
        case DamageSeverity::MODERATE: return {30.0, 80.0};
        case DamageSeverity::SEVERE: return {80.0, 255.0};
    }
    return {0.0, 0.0};
}

enum class ConsistencyVerdict { CONSISTENT, OVER_CLAIMED, UNDER_CLAIMED };

ConsistencyVerdict check_claim_consistency(DamageSeverity claimed, double measured_diff) {
    DiffBand band = expected_diff_band(claimed);
    if (measured_diff < band.low) return ConsistencyVerdict::OVER_CLAIMED;
    if (measured_diff > band.high) return ConsistencyVerdict::UNDER_CLAIMED;
    return ConsistencyVerdict::CONSISTENT;
}

// =======================================================================
// NEW in this section: a per-zone bundle of three independent real
// fraud signals -- Section 28.1's duplicate-photo flag, Section 28.2's
// timestamp-anomaly flag, and Chapter 21.3's own photo-consistency
// verdict above -- combined into a transparent, named fraud-likelihood
// LEVEL, never a single opaque probability.
// =======================================================================
struct FraudSignals {
    bool duplicate_photo = false;
    bool timestamp_anomaly = false;
    bool photo_inconsistent = false;
};

enum class FraudLikelihood { LOW, MEDIUM, HIGH };

std::string to_string(FraudLikelihood f) {
    switch (f) {
        case FraudLikelihood::LOW: return "LOW";
        case FraudLikelihood::MEDIUM: return "MEDIUM";
        case FraudLikelihood::HIGH: return "HIGH";
    }
    return "UNKNOWN";
}

// A real, fully transparent count: zero named signals is LOW, exactly
// one is MEDIUM, two or more is HIGH. Every one of the three signals is
// weighted identically -- this function counts WHICH real checks
// disagreed with the claim, not how "serious" any one of them is.
FraudLikelihood assess_fraud_likelihood(const FraudSignals& s) {
    int count = (s.duplicate_photo ? 1 : 0) + (s.timestamp_anomaly ? 1 : 0) + (s.photo_inconsistent ? 1 : 0);
    if (count >= 2) return FraudLikelihood::HIGH;
    if (count == 1) return FraudLikelihood::MEDIUM;
    return FraudLikelihood::LOW;
}

struct ZoneClaim {
    std::string zone_name;
    DamageSeverity claimed_severity;
    FraudSignals fraud_signals;
};

struct ZoneFraudReport {
    std::string zone_name;
    FraudLikelihood level;
};

struct ClaimAssessment {
    CostRange total_range;
    std::vector<ZoneFraudReport> per_zone_fraud;
};

// The claim's own total cost range is summed EXACTLY as Chapter 21.3's
// own `assess_claim` already did -- this loop never reads a zone's own
// FraudSignals when computing total_range, by construction, not merely
// by convention.
ClaimAssessment assess_claim(const std::vector<ZoneClaim>& zones) {
    ClaimAssessment result;
    for (const auto& z : zones) {
        result.total_range = result.total_range + cost_range_for_severity(z.claimed_severity);
        result.per_zone_fraud.push_back({z.zone_name, assess_fraud_likelihood(z.fraud_signals)});
    }
    return result;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 28.3: Extending the Cost-Range Engine with a Fraud-Likelihood Signal\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: Chapter 21.3's own severity-to-cost-range policy,
    // reused unchanged, still holds exactly. --
    {
        std::cout << "-- Test 1: Chapter 21.3's own cost-range policy, reused unchanged --\n";
        CHECK(cost_range_for_severity(DamageSeverity::MINOR).low == 150.0);
        CHECK(cost_range_for_severity(DamageSeverity::MINOR).high == 600.0);
        CHECK(cost_range_for_severity(DamageSeverity::MODERATE).low == 600.0);
        CHECK(cost_range_for_severity(DamageSeverity::SEVERE).high == 9000.0);
    }

    // -- Test 2: zero fraud signals fired -- LOW. --
    {
        auto level = assess_fraud_likelihood({false, false, false});
        std::cout << "-- Test 2: zero signals fired -- level=" << to_string(level) << " --\n";
        CHECK(level == FraudLikelihood::LOW);
    }

    // -- Test 3: exactly one signal fired, tried in turn for each of the
    // three independent signal types -- each alone is MEDIUM, confirming
    // all three are weighted identically. --
    {
        auto only_duplicate = assess_fraud_likelihood({true, false, false});
        auto only_timestamp = assess_fraud_likelihood({false, true, false});
        auto only_inconsistent = assess_fraud_likelihood({false, false, true});
        std::cout << "-- Test 3: exactly one signal fired -- duplicate-only=" << to_string(only_duplicate)
                  << ", timestamp-only=" << to_string(only_timestamp) << ", inconsistent-only="
                  << to_string(only_inconsistent) << " --\n";
        CHECK(only_duplicate == FraudLikelihood::MEDIUM);
        CHECK(only_timestamp == FraudLikelihood::MEDIUM);
        CHECK(only_inconsistent == FraudLikelihood::MEDIUM);
    }

    // -- Test 4: exactly two signals fired -- the MEDIUM/HIGH boundary,
    // confirmed exactly at the count of 2. --
    {
        auto level = assess_fraud_likelihood({true, true, false});
        std::cout << "-- Test 4: exactly two signals fired (duplicate + timestamp) -- level="
                  << to_string(level) << " --\n";
        CHECK(level == FraudLikelihood::HIGH);
    }

    // -- Test 5: all three signals fired at once. --
    {
        auto level = assess_fraud_likelihood({true, true, true});
        std::cout << "-- Test 5: all three signals fired -- level=" << to_string(level) << " --\n";
        CHECK(level == FraudLikelihood::HIGH);
    }

    // -- Test 6 (central): a two-zone claim, one clean and one with all
    // three fraud signals firing, must produce EXACTLY the same
    // total_range as the identical severities with zero fraud signals
    // attached -- a direct, paired comparison proving the cost estimate
    // is never touched by the fraud-likelihood computation. --
    {
        std::vector<ZoneClaim> clean_zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"hood", DamageSeverity::SEVERE, {}},
        };
        std::vector<ZoneClaim> flagged_zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"hood", DamageSeverity::SEVERE, {true, true, true}},
        };
        auto clean_result = assess_claim(clean_zones);
        auto flagged_result = assess_claim(flagged_zones);
        std::cout << "-- Test 6: clean total=[" << clean_result.total_range.low << ", "
                  << clean_result.total_range.high << "], all-signals-firing total=["
                  << flagged_result.total_range.low << ", " << flagged_result.total_range.high
                  << "] -- identical --\n";
        CHECK(std::abs(clean_result.total_range.low - flagged_result.total_range.low) < 1e-9);
        CHECK(std::abs(clean_result.total_range.high - flagged_result.total_range.high) < 1e-9);
        CHECK(std::abs(flagged_result.total_range.low - 3100.0) < 1e-9);   // 600 + 2500
        CHECK(std::abs(flagged_result.total_range.high - 11500.0) < 1e-9);  // 2500 + 9000
        CHECK(flagged_result.per_zone_fraud[1].level == FraudLikelihood::HIGH);
    }

    // -- Test 7: a full multi-zone claim's own per-zone fraud report
    // names EACH zone by its own name and its own individual level, not
    // merely an aggregate count. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MINOR, {}},
            {"driver_door", DamageSeverity::SEVERE, {true, false, true}},
            {"rear_bumper", DamageSeverity::MODERATE, {false, true, false}},
        };
        auto result = assess_claim(zones);
        std::cout << "-- Test 7: 3-zone claim -- ";
        for (const auto& r : result.per_zone_fraud) std::cout << r.zone_name << "=" << to_string(r.level) << " ";
        std::cout << "--\n";
        CHECK(result.per_zone_fraud.size() == 3);
        CHECK(result.per_zone_fraud[0].zone_name == "front_bumper");
        CHECK(result.per_zone_fraud[0].level == FraudLikelihood::LOW);
        CHECK(result.per_zone_fraud[1].zone_name == "driver_door");
        CHECK(result.per_zone_fraud[1].level == FraudLikelihood::HIGH);
        CHECK(result.per_zone_fraud[2].zone_name == "rear_bumper");
        CHECK(result.per_zone_fraud[2].level == FraudLikelihood::MEDIUM);
    }

    // -- Test 8: closing the full loop -- Chapter 21.3's own reused
    // `mean_abs_diff` and `check_claim_consistency` compute a REAL
    // photo_inconsistent flag from actual before/after pixel data
    // (rather than a hand-set boolean), which then flows into
    // `assess_fraud_likelihood` exactly like any other signal, all
    // while the zone's own total cost contribution stays untouched. --
    {
        std::vector<std::uint8_t> before(16, 100), after(16, 102);  // nearly identical photos
        double measured_diff = mean_abs_diff(before, after);
        auto verdict = check_claim_consistency(DamageSeverity::SEVERE, measured_diff);
        FraudSignals signals{false, false, verdict != ConsistencyVerdict::CONSISTENT};
        std::vector<ZoneClaim> zones = {{"quarter_panel", DamageSeverity::SEVERE, signals}};
        auto result = assess_claim(zones);
        std::cout << "-- Test 8: claimed SEVERE against a measured photo difference of " << measured_diff
                  << " -- photo_inconsistent=" << (signals.photo_inconsistent ? "true" : "false")
                  << ", fraud level=" << to_string(result.per_zone_fraud[0].level) << ", total=["
                  << result.total_range.low << ", " << result.total_range.high << "] --\n";
        CHECK(verdict == ConsistencyVerdict::OVER_CLAIMED);
        CHECK(signals.photo_inconsistent);
        CHECK(result.per_zone_fraud[0].level == FraudLikelihood::MEDIUM);
        CHECK(std::abs(result.total_range.low - 2500.0) < 1e-9);
        CHECK(std::abs(result.total_range.high - 9000.0) < 1e-9);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_fraud_likelihood_cost_range_engine.cpp -o 03_fraud_likelihood_cost_range_engine
./03_fraud_likelihood_cost_range_engine
```

**Sample input:** Chapter 21.3's own severity-to-cost-range policy confirmed unchanged; zero, one (each of three signal types in turn), two, and three fraud signals checked against the exact resulting likelihood level; a two-zone claim with all three signals firing on one zone checked to produce an identical total cost range to the same claim with zero signals attached; a full 3-zone claim's own per-zone report checked to name each zone individually; and a real photo-inconsistency flag derived from actual pixel data via Chapter 21.3's own reused functions, checked to flow correctly into the fraud-likelihood computation.

```text
========================================================
Chapter 28.3: Extending the Cost-Range Engine with a Fraud-Likelihood Signal
========================================================

-- Test 1: Chapter 21.3's own cost-range policy, reused unchanged --
-- Test 2: zero signals fired -- level=LOW --
-- Test 3: exactly one signal fired -- duplicate-only=MEDIUM, timestamp-only=MEDIUM, inconsistent-only=MEDIUM --
-- Test 4: exactly two signals fired (duplicate + timestamp) -- level=HIGH --
-- Test 5: all three signals fired -- level=HIGH --
-- Test 6: clean total=[3100, 11500], all-signals-firing total=[3100, 11500] -- identical --
-- Test 7: 3-zone claim -- front_bumper=LOW driver_door=HIGH rear_bumper=MEDIUM --
-- Test 8: claimed SEVERE against a measured photo difference of 2 -- photo_inconsistent=true, fraud level=MEDIUM, total=[2500, 9000] --

27/27 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a fraud-likelihood LEVEL as a calibrated probability of fraud"
    `assess_fraud_likelihood` is a real, transparent COUNT of how many named, independent real checks disagreed with a claim -- LOW, MEDIUM, or HIGH describes exactly that count, never a real, calibrated statistical probability that a specific claim is fraudulent. A zone with a `HIGH` fraud likelihood has two or three independently suspicious signals attached to it, which is real, actionable information worth a human's attention -- it is not, and was never built to be, a statement that this specific zone is "67% likely to be fraudulent" or any other precise number. Treating this section's own three-level, fully-named count as though it carried that kind of statistical precision would overstate exactly what this section's own honest, simple counting logic can support.

## 28.4 The Claims Disposition Engine

### Intuition

This chapter's capstone reuses Section 28.3's own cost-range and fraud-likelihood machinery unchanged, and routes an entire claim based on the single worst zone across it -- reapplying this book's own never-suppress-a-named-flag discipline once more, but with a real, deliberate ethical boundary this domain requires: this engine has no denial disposition at all, since a false machine denial has real consequences for a genuine claimant.

### The Concept, In Detail

`assess_claim_disposition` is driven entirely by the SINGLE WORST per-zone fraud level found anywhere in the claim -- Test 2 and Test 3 confirm one suspicious zone routes the entire claim to greater scrutiny regardless of how many other zones are completely clean, never diluted by averaging. Test 4 is this section's own central honesty check, extending Section 28.3's own Test 6 to the full disposition level: the identical zone severities produce the exact same total cost range whether or not any fraud signal fires, with only the ROUTING differing between the two cases.

Test 5 is this section's own structural guarantee: all three real dispositions this engine can ever produce are checked, by name, to contain neither "DENY" nor "REJECT" -- a real, deliberate, and testable property of the enum's own three values, not an incidental fact about any one test's own data. Test 6 confirms the disposition's own audit trail names the exact zone and fraud level responsible, never merely an aggregate count.

### Code and Verification

```cpp
// Chapter 28.4 -- This chapter's own capstone: a real, fully auditable
// claims-disposition engine that reuses Section 28.3's own cost-range
// and per-zone fraud-likelihood machinery unchanged, and routes an
// entire claim to one of exactly three real dispositions based on the
// SINGLE WORST zone across the whole claim -- reapplying this book's own
// recurring never-suppress-a-named-flag discipline one more time, in
// the same direction Chapter 26.4 applied it (toward more scrutiny, not
// less), but with a real, deliberate ethical boundary this domain
// requires that neither Chapter 26.4 nor Chapter 27.4 needed: THIS
// engine has no "deny" or "reject" disposition at all. A suspected
// insurance fraud signal is a reason to route a claim to a real human
// Special Investigations Unit reviewer, never a reason for a machine to
// deny a real claimant's own payment automatically -- a false machine
// denial has real, serious consequences for a genuine claimant, exactly
// the reason real insurance regulation in most real jurisdictions
// requires a human decision before any fraud-suspected claim is denied.
//
// A note on this section's own honest scope: `ClaimDisposition` decides
// only WHERE a claim is ROUTED, never whether it is paid, and the
// claim's own total cost range is reported identically regardless of
// disposition -- routing a claim to a human reviewer does not, and must
// not, silently withhold or alter the claimant's own currently-stated
// estimate. This section's own COMMON TRAP box returns to exactly this
// distinction.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_claims_disposition_engine.cpp -o 04_claims_disposition_engine
// Run:     ./04_claims_disposition_engine

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// Section 28.3's own cost-range and fraud-likelihood machinery, reused
// completely UNCHANGED.
// =======================================================================
enum class DamageSeverity { NONE, MINOR, MODERATE, SEVERE };

struct CostRange { double low = 0.0, high = 0.0; };
CostRange operator+(const CostRange& a, const CostRange& b) { return {a.low + b.low, a.high + b.high}; }

CostRange cost_range_for_severity(DamageSeverity s) {
    switch (s) {
        case DamageSeverity::NONE: return {0.0, 0.0};
        case DamageSeverity::MINOR: return {150.0, 600.0};
        case DamageSeverity::MODERATE: return {600.0, 2500.0};
        case DamageSeverity::SEVERE: return {2500.0, 9000.0};
    }
    return {0.0, 0.0};
}

struct FraudSignals {
    bool duplicate_photo = false;
    bool timestamp_anomaly = false;
    bool photo_inconsistent = false;
};

enum class FraudLikelihood { LOW, MEDIUM, HIGH };

std::string to_string(FraudLikelihood f) {
    switch (f) {
        case FraudLikelihood::LOW: return "LOW";
        case FraudLikelihood::MEDIUM: return "MEDIUM";
        case FraudLikelihood::HIGH: return "HIGH";
    }
    return "UNKNOWN";
}

FraudLikelihood assess_fraud_likelihood(const FraudSignals& s) {
    int count = (s.duplicate_photo ? 1 : 0) + (s.timestamp_anomaly ? 1 : 0) + (s.photo_inconsistent ? 1 : 0);
    if (count >= 2) return FraudLikelihood::HIGH;
    if (count == 1) return FraudLikelihood::MEDIUM;
    return FraudLikelihood::LOW;
}

struct ZoneClaim {
    std::string zone_name;
    DamageSeverity claimed_severity;
    FraudSignals fraud_signals;
};

// =======================================================================
// NEW in this section: a claim-level disposition, driven by the SINGLE
// WORST zone's own fraud-likelihood level across the entire claim --
// never diluted by averaging against otherwise-clean zones -- with
// exactly three possible outcomes, NONE of which is a denial.
// =======================================================================
enum class ClaimDisposition { PROCESS_NORMALLY, FLAG_FOR_SIU_REVIEW, HOLD_PENDING_VERIFICATION };

std::string to_string(ClaimDisposition d) {
    switch (d) {
        case ClaimDisposition::PROCESS_NORMALLY: return "PROCESS_NORMALLY";
        case ClaimDisposition::FLAG_FOR_SIU_REVIEW: return "FLAG_FOR_SIU_REVIEW";
        case ClaimDisposition::HOLD_PENDING_VERIFICATION: return "HOLD_PENDING_VERIFICATION";
    }
    return "UNKNOWN";
}

struct ClaimResult {
    CostRange total_range;
    ClaimDisposition disposition;
    std::vector<std::string> reasons;  // the full, named audit trail
};

// The claim's own total cost range is summed exactly as Section 28.3's
// own `assess_claim` did, completely independent of the disposition
// logic below it. The disposition itself is driven by the single WORST
// per-zone fraud level found anywhere in the claim -- a HIGH level in
// even one zone routes the ENTIRE claim to HOLD_PENDING_VERIFICATION,
// regardless of how many other zones are completely clean.
ClaimResult assess_claim_disposition(const std::vector<ZoneClaim>& zones) {
    ClaimResult result;
    FraudLikelihood worst = FraudLikelihood::LOW;
    for (const auto& z : zones) {
        result.total_range = result.total_range + cost_range_for_severity(z.claimed_severity);
        FraudLikelihood level = assess_fraud_likelihood(z.fraud_signals);
        if (level != FraudLikelihood::LOW) {
            result.reasons.push_back(z.zone_name + ": fraud likelihood " + to_string(level));
        }
        if (level == FraudLikelihood::HIGH) worst = FraudLikelihood::HIGH;
        else if (level == FraudLikelihood::MEDIUM && worst != FraudLikelihood::HIGH) worst = FraudLikelihood::MEDIUM;
    }
    if (worst == FraudLikelihood::HIGH) {
        result.disposition = ClaimDisposition::HOLD_PENDING_VERIFICATION;
    } else if (worst == FraudLikelihood::MEDIUM) {
        result.disposition = ClaimDisposition::FLAG_FOR_SIU_REVIEW;
    } else {
        result.disposition = ClaimDisposition::PROCESS_NORMALLY;
    }
    return result;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 28.4: The Claims Disposition Engine\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: an entirely clean, multi-zone claim -- processes
    // normally, with the exact hand-computed total cost range. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"hood", DamageSeverity::MINOR, {}},
        };
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 1: clean 2-zone claim -- disposition=" << to_string(r.disposition)
                  << ", total=[" << r.total_range.low << ", " << r.total_range.high << "] --\n";
        CHECK(r.disposition == ClaimDisposition::PROCESS_NORMALLY);
        CHECK(std::abs(r.total_range.low - 750.0) < 1e-9);
        CHECK(std::abs(r.total_range.high - 3100.0) < 1e-9);
        CHECK(r.reasons.empty());
    }

    // -- Test 2: one MEDIUM zone among otherwise-clean zones -- the
    // whole claim is flagged for review, never diluted by the clean
    // zones around it. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"driver_door", DamageSeverity::MINOR, {true, false, false}},
            {"rear_bumper", DamageSeverity::MINOR, {}},
        };
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 2: one MEDIUM zone among two clean zones -- disposition="
                  << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == ClaimDisposition::FLAG_FOR_SIU_REVIEW);
        CHECK(r.reasons.size() == 1);
    }

    // -- Test 3: one HIGH zone among otherwise clean/medium zones -- the
    // whole claim is held pending verification, the worst zone wins. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"driver_door", DamageSeverity::MINOR, {true, false, false}},
            {"windshield", DamageSeverity::SEVERE, {true, true, false}},
        };
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 3: one HIGH zone among a clean and a MEDIUM zone -- disposition="
                  << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == ClaimDisposition::HOLD_PENDING_VERIFICATION);
        CHECK(r.reasons.size() == 2);
    }

    // -- Test 4 (central): the IDENTICAL zone severities as Test 3
    // produce the EXACT SAME total cost range whether or not any fraud
    // signal is attached -- a direct paired comparison proving the
    // disposition logic never touches the claim's own cost estimate. --
    {
        std::vector<ZoneClaim> flagged_zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"driver_door", DamageSeverity::MINOR, {true, false, false}},
            {"windshield", DamageSeverity::SEVERE, {true, true, false}},
        };
        std::vector<ZoneClaim> clean_zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"driver_door", DamageSeverity::MINOR, {}},
            {"windshield", DamageSeverity::SEVERE, {}},
        };
        auto flagged = assess_claim_disposition(flagged_zones);
        auto clean = assess_claim_disposition(clean_zones);
        std::cout << "-- Test 4: HOLD-triggering total=[" << flagged.total_range.low << ", "
                  << flagged.total_range.high << "], fully-clean total=[" << clean.total_range.low << ", "
                  << clean.total_range.high << "] -- identical, disposition differs (" << to_string(flagged.disposition)
                  << " vs. " << to_string(clean.disposition) << ") --\n";
        CHECK(std::abs(flagged.total_range.low - clean.total_range.low) < 1e-9);
        CHECK(std::abs(flagged.total_range.high - clean.total_range.high) < 1e-9);
        CHECK(flagged.disposition != clean.disposition);
    }

    // -- Test 5 (structural never-auto-deny guarantee): every one of the
    // three real dispositions this engine can ever produce is checked by
    // name to contain neither "DENY" nor "REJECT" -- a structural
    // property of the enum's own three values, not an incidental fact
    // about this section's own test data. --
    {
        std::vector<ClaimDisposition> all = {ClaimDisposition::PROCESS_NORMALLY,
                                              ClaimDisposition::FLAG_FOR_SIU_REVIEW,
                                              ClaimDisposition::HOLD_PENDING_VERIFICATION};
        bool any_denial_wording = false;
        for (auto d : all) {
            std::string name = to_string(d);
            if (name.find("DENY") != std::string::npos || name.find("REJECT") != std::string::npos) {
                any_denial_wording = true;
            }
        }
        std::cout << "-- Test 5: all 3 real dispositions checked for denial wording -- found: "
                  << (any_denial_wording ? "YES (WRONG)" : "NONE (correct)") << " --\n";
        CHECK(all.size() == 3);
        CHECK(!any_denial_wording);
    }

    // -- Test 6: the disposition's own audit trail names the EXACT
    // zone(s) responsible for a HOLD_PENDING_VERIFICATION disposition
    // and their own fraud level, not merely an aggregate flag count. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"windshield", DamageSeverity::SEVERE, {true, true, true}},
        };
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 6: audit trail -- ";
        for (const auto& reason : r.reasons) std::cout << "[" << reason << "] ";
        std::cout << "--\n";
        CHECK(r.reasons.size() == 1);
        CHECK(r.reasons[0].find("windshield") != std::string::npos);
        CHECK(r.reasons[0].find("HIGH") != std::string::npos);
        CHECK(r.reasons[0].find("front_bumper") == std::string::npos);
    }

    // -- Test 7: the exact MEDIUM/HIGH signal-count boundary carried
    // through to the claim level -- exactly 2 signals on one zone drives
    // HOLD_PENDING_VERIFICATION, confirmed precisely at that boundary. --
    {
        std::vector<ZoneClaim> zones = {{"roof", DamageSeverity::MODERATE, {false, true, true}}};
        auto r = assess_claim_disposition(zones);
        std::cout << "-- Test 7: single zone, exactly 2 fraud signals -- disposition="
                  << to_string(r.disposition) << " --\n";
        CHECK(r.disposition == ClaimDisposition::HOLD_PENDING_VERIFICATION);
    }

    // -- Test 8: a full multi-zone, multi-signal claim's own total cost
    // range checked against an exact hand computation, tying this
    // capstone back to Chapter 21.3's own Test 1 and Test 5 numerically. --
    {
        std::vector<ZoneClaim> zones = {
            {"front_bumper", DamageSeverity::MODERATE, {}},
            {"hood", DamageSeverity::MINOR, {true, false, false}},
            {"windshield", DamageSeverity::SEVERE, {}},
        };
        auto r = assess_claim_disposition(zones);
        // hand-computed: (600+150+2500, 2500+600+9000) = (3250, 12100)
        std::cout << "-- Test 8: 3-zone claim total=[" << r.total_range.low << ", " << r.total_range.high
                  << "], disposition=" << to_string(r.disposition) << " --\n";
        CHECK(std::abs(r.total_range.low - 3250.0) < 1e-9);
        CHECK(std::abs(r.total_range.high - 12100.0) < 1e-9);
        CHECK(r.disposition == ClaimDisposition::FLAG_FOR_SIU_REVIEW);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_claims_disposition_engine.cpp -o 04_claims_disposition_engine
./04_claims_disposition_engine
```

**Sample input:** an entirely clean multi-zone claim checked to process normally with an exact hand-computed total; one MEDIUM-likelihood zone and, separately, one HIGH-likelihood zone, each checked among otherwise-clean zones to route the entire claim to the correct disposition without dilution; identical zone severities checked with and without fraud signals attached, confirming an identical total cost range with only the disposition differing; all three real dispositions checked by name for the complete absence of denial wording; the disposition's own audit trail checked to name the exact responsible zone; the exact signal-count boundary driving a disposition; and a full multi-zone, multi-signal claim's own total checked against an exact hand computation.

```text
========================================================
Chapter 28.4: The Claims Disposition Engine
========================================================

-- Test 1: clean 2-zone claim -- disposition=PROCESS_NORMALLY, total=[750, 3100] --
-- Test 2: one MEDIUM zone among two clean zones -- disposition=FLAG_FOR_SIU_REVIEW --
-- Test 3: one HIGH zone among a clean and a MEDIUM zone -- disposition=HOLD_PENDING_VERIFICATION --
-- Test 4: HOLD-triggering total=[3250, 12100], fully-clean total=[3250, 12100] -- identical, disposition differs (HOLD_PENDING_VERIFICATION vs. PROCESS_NORMALLY) --
-- Test 5: all 3 real dispositions checked for denial wording -- found: NONE (correct) --
-- Test 6: audit trail -- [windshield: fraud likelihood HIGH] --
-- Test 7: single zone, exactly 2 fraud signals -- disposition=HOLD_PENDING_VERIFICATION --
-- Test 8: 3-zone claim total=[3250, 12100], disposition=FLAG_FOR_SIU_REVIEW --

21/21 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a HOLD_PENDING_VERIFICATION disposition as equivalent to withholding the claimant's own estimate"
    `ClaimResult.total_range` is computed identically regardless of `disposition` -- Test 4 confirms this directly. `HOLD_PENDING_VERIFICATION` means only that a human Special Investigations Unit reviewer sees this claim before it proceeds further; it does not mean the claimant's own currently-stated cost estimate is hidden, reduced, or withheld pending that review. Conflating "routed to a human for review" with "payment withheld or denied" would misrepresent what this engine actually does, and -- far more seriously, in a real deployment -- could lead a downstream system to silently treat every flagged claim as though it had already been denied, exactly the kind of unauthorized escalation this chapter's own introduction named as a real, serious harm to a genuine claimant.

## Chapter Summary

This chapter turned to insurance claims, where the fraud-relevant signals are photographic and temporal, and where a false machine accusation carries real consequences for a genuine claimant. Section 28.1 reapplied Chapter 23.1's own dHash algorithm a third time, this time to detect a photo reused across two different claims, explicitly excluding a claim's own routine resubmission of its own photo. Section 28.2 built a real photo-timestamp consistency check, reapplying Section 27.2's own stated-parameter timing discipline to catch a photo that could not possibly have been taken when it claims to have been. Section 28.3 extended Chapter 21.3's own cost-range and photo-consistency engine -- reused completely unchanged -- with a new, transparent per-zone fraud-likelihood signal, proving the cost estimate itself is never touched by that signal. Section 28.4 closed the chapter with a claims-disposition engine that routes a claim toward more or less human scrutiny based on its single worst zone, while guaranteeing, structurally, that no disposition it can ever produce is a denial.

## Self-Check Questions

1. Section 28.1's `find_reused_photo_matches` excludes matches against the same `claim_id` as the new submission. Describe one concrete real scenario in which this exclusion would incorrectly hide a genuine fraud signal, if a single real claimant were allowed to submit more than one claim under different claim IDs.
2. Section 28.1's Test 4 constructs a photo already shared by two separate prior claims. Explain why returning ALL matching claim IDs, rather than just the closest or the first one found, is the correct design choice for a real fraud investigation.
3. Section 28.2's `check_photo_timing` treats a `PRE_DATED` result and a `STALE_OR_LATE` result as two separate, distinctly named outcomes rather than a single generic anomaly flag. Using the section's own COMMON TRAP box, explain concretely why a real SIU reviewer would prioritize their own investigation differently between the two.
4. Section 28.2's own introduction compares its own "stated day-count, never a wall clock" discipline to Section 27.2's identical discipline for a different constraint. Explain what specifically would go wrong with this book's own cross-architecture verification if `check_photo_timing` instead computed a photo's own age using a real calendar library's current-date function.
5. Section 28.3's Test 6 constructs a two-zone claim where one zone fires all three fraud signals at once. Explain precisely HOW `assess_claim`'s own implementation guarantees this cannot affect `total_range`, referring to the specific code structure responsible.
6. Section 28.3's own COMMON TRAP box warns against treating a fraud-likelihood LEVEL as a calibrated probability. Explain what a real insurer would need to add to this section's own simple counting logic before it could honestly be described as a calibrated probability model.
7. Section 28.4's `assess_claim_disposition` is driven by the single WORST zone's own fraud level, never an average across all zones. Construct a concrete scenario with at least four zones where averaging, instead of taking the worst, would cause a real fraud signal to be missed entirely.
8. Section 28.4's own introduction states a real, deliberate ethical reason this engine has no denial disposition, unlike Chapter 26.4's `AUTO_REJECT`. Explain why an insurance claim's own real-world consequences of a false positive differ from a real bank's own consequences of falsely auto-rejecting a check, in a way that justifies this specific structural difference between the two engines.
9. Section 28.4's Test 5 checks all three real `ClaimDisposition` values for denial wording at runtime, using string matching. Explain one real limitation of checking for the ABSENCE of a specific fraud-adjacent word as a way of proving a system can never take a specific, serious action.
10. Across this chapter's four sections, identify the ONE section that reuses an ALGORITHM this book already built rather than a DOMAIN PATTERN this book already established, and explain the difference between those two kinds of reuse using this chapter's own two clearest examples of each.

## Where We Go Next

This chapter showed that a fraud signal and a cost estimate can, and must, be computed on completely separate, independent tracks -- reapplying this book's own recurring never-suppress-a-flag discipline with a real, deliberate ethical boundary an insurance claim's own real-world stakes require. Chapter 29 turns to the last of this book's four financial-industry application chapters: on-device anti-money-laundering (AML) transaction anomaly monitoring at ATMs and point-of-sale terminals, where the constraint shifts once more, this time to detecting a pattern across a SEQUENCE of transactions rather than evaluating any single one in isolation.

## Worked Solutions

**1.** If a real claimant submitted two genuinely separate, unrelated claims (for instance, two different vehicles they own, damaged in two unrelated real incidents) and, dishonestly, used the SAME real photo of damage for both -- perhaps because they never actually damaged the second vehicle at all -- Section 28.1's own `claim_id` exclusion would not help here, since the two claim IDs are genuinely different; this scenario would still be caught correctly. The exclusion would incorrectly hide a real signal only if the SAME claim were somehow re-registered under a second, different `claim_id` for the identical incident (a real, if less common, fraud pattern in its own right: filing what is actually one incident as two separate claims to double an insurer's exposure) -- in that specific case, this section's own cross-claim check would flag the resubmission as a normal, apparently-cross-claim match rather than recognizing it as the SAME underlying incident, since `claim_id` alone cannot distinguish "the same claim, re-filed" from "a genuinely new, unrelated claim."

**2.** A single closest or first match would tell an SIU investigator only that ONE other claim shares this photo, when the real, more serious pattern -- the same photo circulating across THREE OR MORE claims -- is a far stronger, more organized fraud signal (potentially indicating a fraud ring reusing a shared pool of staged damage photos across many claimants) that a "just the first match" design would completely hide. Returning every match lets a human reviewer see the real SCALE of the reuse pattern directly, rather than having to separately re-run the same check against every other claim in the database to discover what this section's own function could have reported the first time.

**3.** A `PRE_DATED` photo describes a real structural impossibility with no honest explanation short of the photo belonging to an entirely different, earlier incident -- a reviewer would prioritize investigating this claim's own basic legitimacy immediately, since the claim's own submitted evidence is self-contradictory. A `STALE_OR_LATE` photo, by contrast, has many completely routine explanations (a delayed upload, a follow-up photo taken during an active real repair), so a reviewer would reasonably treat it as a lower-priority item worth a quick confirmation rather than an urgent investigation -- collapsing both into one generic flag would force every reviewer to manually re-derive which of the two situations they were actually looking at before they could even begin prioritizing their own queue.

**4.** This book's own cross-architecture verification requires the locked, byte-for-byte self-test output to match EXACTLY across four different execution legs, run at different real times (potentially days apart, across different verification passes). A calendar library's own "current date" function returns a genuinely different real value depending on WHEN the test actually runs, which would make `check_photo_timing`'s own computed day-gaps differ between a verification run today and a verification run next week -- the exact same category of problem Section 27.2's own COMMON TRAP box already named for a real wall-clock latency measurement, just manifesting as a difference across TIME instead of across ARCHITECTURE.

**5.** `assess_claim`'s own loop computes `result.total_range = result.total_range + cost_range_for_severity(z.claimed_severity)` using ONLY `z.claimed_severity` as its input -- `z.fraud_signals` is never read, referenced, or passed into `cost_range_for_severity` or the `total_range` accumulation anywhere in that function. The fraud-likelihood computation, `assess_fraud_likelihood(z.fraud_signals)`, is a completely separate call whose own return value is appended only to `result.per_zone_fraud`, a different field entirely -- the two computations share the same input zone but write to two different output fields through two functions that never call each other, which is what makes the independence a structural guarantee rather than a testing coincidence.

**6.** A real calibrated probability model would need real, historical ground-truth data -- a large number of PAST claims where each of these three signals either fired or did not, cross-referenced against which of those claims were later CONFIRMED, through a real investigation, to actually be fraudulent -- and a real statistical model (such as real logistic regression) fit against that historical data to learn how much each signal, and each COMBINATION of signals, actually predicts confirmed fraud in practice. Section 28.3's own simple count has none of this: it treats all three signals as equally weighted by stated policy choice, with no historical validation behind that weighting at all, which is exactly why it reports a plain descriptive LEVEL rather than a number that could be mistaken for a validated statistical probability.

**7.** Consider four zones with fraud-signal counts of 2 (HIGH), 0, 0, and 0 -- averaged naively across all four zones (treating LOW=0, MEDIUM=1, HIGH=2 as numeric scores, for instance), the average signal count would be `(2+0+0+0)/4 = 0.5`, which would likely round down to a LOW or borderline-MEDIUM overall level despite one zone carrying a real, serious two-signal HIGH-likelihood flag on its own. Taking the single WORST zone instead, as Section 28.4 actually does, correctly reports HIGH for the whole claim regardless of how many additional clean zones happen to be attached to it -- exactly the same "one flag should never be diluted by everything else looking clean" principle Chapter 26.4's own MICR-forcing rule already established for check fraud.

**8.** A real bank's own check-fraud engine (Chapter 26.4) operates on a payment INSTRUMENT before funds move -- an `AUTO_REJECT` there stops a specific transaction from clearing, and a legitimate payee whose check is wrongly rejected can typically resubmit or contact their bank promptly, with the underlying funds and account relationship otherwise unaffected. A real insurance claim, by contrast, often represents a real person's own urgent, immediate need (a home that is currently damaged, a vehicle they cannot currently drive) where a false denial can mean real, immediate financial hardship with no quick resubmission path, and many real jurisdictions' own insurance regulations specifically require a human decision-maker before a fraud-suspected claim can be denied at all -- a real regulatory and human-consequence difference this section's own structural choice (no denial disposition exists in this engine at all) reflects directly.

**9.** Checking that a string does not CONTAIN the substrings "DENY" or "REJECT" only proves that THOSE TWO SPECIFIC WORDS are absent from the three disposition names as currently written -- it says nothing about the actual BEHAVIOR of any code that might later consume a `ClaimDisposition` value. A future engineer could, in principle, add a fourth enum value named something that passes this exact string check (such as `ClaimDisposition::CLOSE_WITHOUT_PAYMENT`) while still functioning as a real denial in every practical sense; the test only guards against a very literal category of naming mistake, and a genuine structural guarantee against ever ADDING a real denial path would require reviewing every place a `ClaimDisposition` value is consumed downstream, not merely inspecting the enum's own three names.

**10.** Section 28.1 reuses an ALGORITHM this book already built -- Chapter 23.1's own real dHash perceptual-hashing implementation, copied and reapplied completely unchanged, exactly the same kind of reuse Chapter 26.1 already performed once. Section 28.3 (and Section 28.4, built directly on top of it) instead reuses a DOMAIN PATTERN this book already established -- Chapter 21.3's own real severity-to-cost-range-plus-consistency-check STRUCTURE, extended with an entirely new fraud-likelihood signal layered alongside it, rather than any single function being copied verbatim without modification. The difference is that an algorithm reuse (Section 28.1) applies IDENTICAL code to new data, while a pattern reuse (Sections 28.3-28.4) applies an established STRUCTURAL APPROACH -- separate concerns, honest ranges, named audit trails -- to build genuinely new code that follows the same real design discipline.
