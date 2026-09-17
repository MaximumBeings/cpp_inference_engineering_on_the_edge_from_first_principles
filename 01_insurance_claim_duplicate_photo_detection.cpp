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
