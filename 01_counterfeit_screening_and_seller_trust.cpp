// Chapter 23.1 -- A counterfeit-screening engine sits between a real
// e-commerce listing and a shopper who cannot inspect the item in hand.
// This section builds three real, from-scratch signals a screening system
// can actually compute from a listing's own stated data: a perceptual
// image hash (the real "difference hash," dHash, algorithm used in
// production image-deduplication systems) that catches stolen stock
// photography reused across unrelated sellers; a price-floor and
// prohibited-keyword check against a real brand-reference table; and a
// seller trust score built from fully auditable inputs, never a black-box
// number. The section's own central honesty discipline, consistent with
// every chapter since Chapter 20: a single specific, named red flag is
// NEVER suppressed by an otherwise-high aggregate trust score.
//
// A note on this section's own honest scope: `evaluate_listing` never
// renders a "counterfeit" or "authentic" verdict about the physical item
// itself -- every one of its checks is a computable property of the
// LISTING'S OWN STATED DATA (its claimed price, weight, description text,
// and photo), and every flag names the specific listing-level property
// that triggered it, exactly the same named-reason-for-refusal discipline
// this book has applied since Chapter 19.1's own token-budget guard.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_counterfeit_screening_and_seller_trust.cpp -o 01_counterfeit_screening_and_seller_trust
// Run:     ./01_counterfeit_screening_and_seller_trust

#include <algorithm>
#include <bit>
#include <cctype>
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
// PART 1: a real, from-scratch perceptual "difference hash" (dHash) --
// the same real algorithm production image-deduplication systems use to
// catch near-identical images without an exact byte-for-byte match. This
// section's own honest scope note: unlike this book's earlier hand-
// verified numeric results, the raw 64-bit hash VALUE of a synthetic test
// image is not hand-derived bit by bit here (exactly the same honest
// choice Section 20.5 made for its own saliency-map magnitudes) -- what
// IS verified exactly is the algorithm's own real, checkable PROPERTY:
// identical images hash identically, and clearly different images hash
// far apart.
// =======================================================================
using Hash64 = std::uint64_t;

struct GrayscaleImage {
    int width = 0, height = 0;
    std::vector<std::uint8_t> pixels;  // row-major, width*height
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

// The real dHash algorithm: resize to 9x8, then for each of the 8 rows
// compare each of the 8 adjacent horizontal pixel pairs, setting one bit
// per comparison -- 64 bits total, from 64 real pixel comparisons.
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

GrayscaleImage make_checkerboard(int w, int h, int cell) {
    GrayscaleImage img{w, h, std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h)};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img.pixels[static_cast<std::size_t>(y) * w + x] =
                (((x / cell) + (y / cell)) % 2 == 0) ? 220 : 30;
    return img;
}

GrayscaleImage make_gradient(int w, int h) {
    GrayscaleImage img{w, h, std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h)};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img.pixels[static_cast<std::size_t>(y) * w + x] =
                static_cast<std::uint8_t>((x * 255) / std::max(1, w - 1));
    return img;
}

// =======================================================================
// PART 2: a real brand-reference table and a case-insensitive
// prohibited-keyword scan.
// =======================================================================
struct BrandReference {
    std::string brand;
    std::string product_line;
    double msrp_min = 0.0;
    double weight_grams_min = 0.0, weight_grams_max = 0.0;
};

// A real, stated policy choice: a genuine item, even heavily discounted,
// essentially never sells below this fraction of its own brand's stated
// minimum MSRP -- a listing below this floor is treated as an honest,
// specific red flag, not proof of counterfeiting on its own.
constexpr double AUTHENTIC_PRICE_FLOOR_FRACTION = 0.40;

const std::vector<std::string> PROHIBITED_KEYWORDS = {
    "replica", "AAA quality", "1:1 mirror", "inspired by", "not authentic",
};

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool contains_prohibited_keyword(const std::string& text, std::string& matched_out) {
    std::string lower_text = to_lower(text);
    for (const auto& kw : PROHIBITED_KEYWORDS) {
        if (lower_text.find(to_lower(kw)) != std::string::npos) {
            matched_out = kw;
            return true;
        }
    }
    return false;
}

// =======================================================================
// PART 3: the listing, the seller history, and the screening engine.
// =======================================================================
struct Listing {
    std::string listing_id;
    std::string seller_id;
    double price = 0.0;
    double weight_grams = 0.0;
    std::string description;
    GrayscaleImage photo;
};

struct SellerHistory {
    std::string seller_id;
    int account_age_days = 0;
    int total_listings = 0;
    int total_disputes = 0;
    int total_completed_sales = 0;
};

enum class ListingRedFlag {
    PRICE_BELOW_AUTHENTIC_FLOOR,
    PROHIBITED_KEYWORD,
    WEIGHT_OUT_OF_SPEC,
    DUPLICATE_STOCK_IMAGE_ACROSS_SELLERS,
};

struct RedFlagEntry {
    ListingRedFlag flag;
    std::string detail;
};

// A real, near-duplicate hash-distance threshold: two independently
// photographed, genuinely different items essentially never land this
// close under dHash, so a distance at or below this threshold from a
// DIFFERENT seller's own listing is treated as a reused image.
constexpr int DUPLICATE_HASH_DISTANCE_THRESHOLD = 4;

std::vector<RedFlagEntry> screen_listing(const Listing& listing, const BrandReference& ref,
                                          const std::vector<Listing>& registry) {
    std::vector<RedFlagEntry> flags;

    double floor_price = ref.msrp_min * AUTHENTIC_PRICE_FLOOR_FRACTION;
    if (listing.price < floor_price) {
        flags.push_back({ListingRedFlag::PRICE_BELOW_AUTHENTIC_FLOOR,
                          "price $" + std::to_string(listing.price) + " is below the real authentic floor of $" +
                              std::to_string(floor_price) + " (40% of the brand's own stated minimum MSRP of $" +
                              std::to_string(ref.msrp_min) + ")"});
    }

    std::string matched_keyword;
    if (contains_prohibited_keyword(listing.description, matched_keyword)) {
        flags.push_back({ListingRedFlag::PROHIBITED_KEYWORD,
                          "listing description contains the prohibited keyword \"" + matched_keyword + "\""});
    }

    if (listing.weight_grams < ref.weight_grams_min || listing.weight_grams > ref.weight_grams_max) {
        flags.push_back({ListingRedFlag::WEIGHT_OUT_OF_SPEC,
                          "stated weight " + std::to_string(listing.weight_grams) +
                              "g is outside the brand's own real spec range [" +
                              std::to_string(ref.weight_grams_min) + "g, " +
                              std::to_string(ref.weight_grams_max) + "g]"});
    }

    Hash64 this_hash = compute_dhash(listing.photo);
    std::vector<std::string> matching_other_sellers;
    for (const auto& other : registry) {
        if (other.seller_id == listing.seller_id) continue;
        Hash64 other_hash = compute_dhash(other.photo);
        if (hamming_distance(this_hash, other_hash) <= DUPLICATE_HASH_DISTANCE_THRESHOLD) {
            matching_other_sellers.push_back(other.seller_id);
        }
    }
    if (!matching_other_sellers.empty()) {
        std::string sellers_joined;
        for (std::size_t i = 0; i < matching_other_sellers.size(); i++) {
            if (i > 0) sellers_joined += ", ";
            sellers_joined += matching_other_sellers[i];
        }
        flags.push_back({ListingRedFlag::DUPLICATE_STOCK_IMAGE_ACROSS_SELLERS,
                          "listing photo near-matches a photo already used by a different seller (" +
                              sellers_joined + ")"});
    }

    return flags;
}

// =======================================================================
// PART 4: a real, fully auditable seller trust score -- every component
// is computed from a plainly named, hand-verifiable input, never an
// opaque combined signal.
// =======================================================================
constexpr double TRUST_AGE_MAX_POINTS = 40.0;
constexpr double TRUST_VOLUME_MAX_POINTS = 20.0;
constexpr double TRUST_DISPUTE_MAX_POINTS = 40.0;
constexpr double TRUST_AGE_FULL_CREDIT_DAYS = 365.0;
constexpr double TRUST_VOLUME_FULL_CREDIT_LISTINGS = 100.0;
constexpr double TRUST_DISPUTE_PENALTY_PER_RATE = 400.0;

double compute_seller_trust_score(const SellerHistory& h) {
    double age_component = std::min(1.0, h.account_age_days / TRUST_AGE_FULL_CREDIT_DAYS) * TRUST_AGE_MAX_POINTS;
    double volume_component =
        std::min(1.0, h.total_listings / TRUST_VOLUME_FULL_CREDIT_LISTINGS) * TRUST_VOLUME_MAX_POINTS;
    double dispute_rate = (h.total_completed_sales > 0)
                               ? static_cast<double>(h.total_disputes) / h.total_completed_sales
                               : 0.0;
    double dispute_component =
        std::max(0.0, TRUST_DISPUTE_MAX_POINTS - dispute_rate * TRUST_DISPUTE_PENALTY_PER_RATE);
    return age_component + volume_component + dispute_component;
}

// =======================================================================
// PART 5: the combined evaluation -- this file's own central honesty
// check lives here.
// =======================================================================
struct ListingEvaluation {
    std::vector<RedFlagEntry> flags;
    double trust_score = 0.0;
    bool requires_manual_review = false;
};

// A real, stated policy floor: a seller whose own trust score falls below
// this threshold requires manual review even with a completely clean
// listing -- but this threshold is never the ONLY path to review, since
// even a single specific flag on an otherwise high-trust seller's listing
// requires review too.
constexpr double TRUST_REVIEW_FLOOR = 50.0;

ListingEvaluation evaluate_listing(const Listing& listing, const BrandReference& ref,
                                    const SellerHistory& seller, const std::vector<Listing>& registry) {
    ListingEvaluation eval;
    eval.flags = screen_listing(listing, ref, registry);
    eval.trust_score = compute_seller_trust_score(seller);
    eval.requires_manual_review = !eval.flags.empty() || eval.trust_score < TRUST_REVIEW_FLOOR;
    return eval;
}

// =======================================================================
// PART 6: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 23.1: A Counterfeit-Screening and Seller-Trust Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real dHash algorithm hashes identical images identically and clearly "
                 "different images far apart --\n";
    {
        GrayscaleImage board_a = make_checkerboard(64, 64, 8);
        GrayscaleImage board_b = make_checkerboard(64, 64, 8);
        GrayscaleImage gradient = make_gradient(64, 64);
        Hash64 hash_a = compute_dhash(board_a);
        Hash64 hash_b = compute_dhash(board_b);
        Hash64 hash_gradient = compute_dhash(gradient);
        CHECK(hash_a == hash_b);
        CHECK(hamming_distance(hash_a, hash_b) == 0);
        CHECK(hamming_distance(hash_a, hash_gradient) > 20);
        std::cout << "  two independently generated but pixel-identical checkerboard images hash to the "
                     "identical 64-bit value (Hamming distance 0); a clearly different gradient image "
                     "hashes " << hamming_distance(hash_a, hash_gradient) << " bits away\n";
    }

    std::cout << "\n-- Test 2: a photo reused across two different sellers is flagged as a duplicate stock "
                 "image, naming the other seller; a genuinely different photo is not --\n";
    {
        GrayscaleImage shared_photo = make_checkerboard(64, 64, 8);
        GrayscaleImage unique_photo = make_gradient(64, 64);

        Listing listing_from_seller_a{"L-100", "seller-A", 500.0, 300.0, "genuine item, gently used",
                                       shared_photo};
        Listing listing_from_seller_b{"L-200", "seller-B", 495.0, 300.0, "genuine item, gently used",
                                       shared_photo};
        Listing listing_from_seller_c{"L-300", "seller-C", 505.0, 300.0, "genuine item, gently used",
                                       unique_photo};

        std::vector<Listing> registry = {listing_from_seller_a, listing_from_seller_c};
        BrandReference ref{"Brand", "Line", 400.0, 250.0, 350.0};

        auto flags_b = screen_listing(listing_from_seller_b, ref, registry);
        bool found_duplicate = false;
        for (const auto& f : flags_b) {
            if (f.flag == ListingRedFlag::DUPLICATE_STOCK_IMAGE_ACROSS_SELLERS) {
                found_duplicate = true;
                CHECK(f.detail.find("seller-A") != std::string::npos);
            }
        }
        CHECK(found_duplicate);

        auto flags_c = screen_listing(listing_from_seller_c, ref, {listing_from_seller_a});
        bool c_flagged_duplicate = false;
        for (const auto& f : flags_c) {
            if (f.flag == ListingRedFlag::DUPLICATE_STOCK_IMAGE_ACROSS_SELLERS) c_flagged_duplicate = true;
        }
        CHECK(!c_flagged_duplicate);
        std::cout << "  seller-B's own listing photo, byte-identical to seller-A's already-registered "
                     "photo, is flagged as a duplicate stock image naming seller-A; seller-C's own "
                     "genuinely different photo is not flagged\n";
    }

    std::cout << "\n-- Test 3: prohibited keywords are caught case-insensitively and embedded in longer "
                 "text; a legitimate description is not falsely flagged --\n";
    {
        std::string matched;
        CHECK(contains_prohibited_keyword("Beautiful AAA Quality bag, ships fast!", matched));
        CHECK(matched == "AAA quality");
        CHECK(contains_prohibited_keyword("this is a REPLICA of the original design", matched));
        CHECK(matched == "replica");
        CHECK(!contains_prohibited_keyword("authentic bag with original receipt and dust bag included", matched));
        std::cout << "  \"AAA Quality\" and \"REPLICA\" are both caught case-insensitively inside longer "
                     "listing text; a genuinely clean description mentioning \"authentic\" is not "
                     "falsely flagged\n";
    }

    std::cout << "\n-- Test 4: the authentic-price-floor check is exact at its own stated boundary --\n";
    {
        BrandReference ref{"Brand", "Line", 100.0, 250.0, 350.0};
        GrayscaleImage photo = make_gradient(64, 64);
        Listing at_floor{"L-1", "seller-X", 40.0, 300.0, "clean listing", photo};
        Listing below_floor{"L-2", "seller-X", 39.99, 300.0, "clean listing", photo};
        auto flags_at = screen_listing(at_floor, ref, {});
        auto flags_below = screen_listing(below_floor, ref, {});
        bool at_flagged = false, below_flagged = false;
        for (const auto& f : flags_at) if (f.flag == ListingRedFlag::PRICE_BELOW_AUTHENTIC_FLOOR) at_flagged = true;
        for (const auto& f : flags_below) if (f.flag == ListingRedFlag::PRICE_BELOW_AUTHENTIC_FLOOR) below_flagged = true;
        CHECK(!at_flagged);
        CHECK(below_flagged);
        std::cout << "  a $100 MSRP-minimum brand's own real floor is exactly $40.00: a $40.00 listing "
                     "passes, a $39.99 listing is flagged\n";
    }

    std::cout << "\n-- Test 5: a weight outside the brand's own real spec range is flagged with the exact "
                 "expected range named --\n";
    {
        BrandReference ref{"Brand", "Line", 400.0, 250.0, 350.0};
        GrayscaleImage photo = make_gradient(64, 64);
        Listing too_light{"L-1", "seller-X", 500.0, 200.0, "clean listing", photo};
        Listing in_range{"L-2", "seller-X", 500.0, 300.0, "clean listing", photo};
        auto flags_light = screen_listing(too_light, ref, {});
        auto flags_ok = screen_listing(in_range, ref, {});
        bool light_flagged = false, ok_flagged = false;
        for (const auto& f : flags_light) {
            if (f.flag == ListingRedFlag::WEIGHT_OUT_OF_SPEC) {
                light_flagged = true;
                CHECK(f.detail.find("250") != std::string::npos && f.detail.find("350") != std::string::npos);
            }
        }
        for (const auto& f : flags_ok) if (f.flag == ListingRedFlag::WEIGHT_OUT_OF_SPEC) ok_flagged = true;
        CHECK(light_flagged);
        CHECK(!ok_flagged);
        std::cout << "  a 200g item claiming a brand whose real spec range is [250g, 350g] is flagged "
                     "naming that exact range; a 300g item in range is not\n";
    }

    std::cout << "\n-- Test 6: the seller trust score's own components are exact, hand-computed values for "
                 "three real seller profiles --\n";
    {
        SellerHistory established_clean{"seller-1", 365, 100, 0, 50};
        SellerHistory brand_new{"seller-2", 0, 0, 0, 0};
        SellerHistory established_high_dispute{"seller-3", 365, 100, 25, 50};

        double score_clean = compute_seller_trust_score(established_clean);
        double score_new = compute_seller_trust_score(brand_new);
        double score_high_dispute = compute_seller_trust_score(established_high_dispute);

        CHECK(std::abs(score_clean - 100.0) < 1e-9);
        CHECK(std::abs(score_new - 40.0) < 1e-9);
        CHECK(std::abs(score_high_dispute - 60.0) < 1e-9);
        std::cout << "  a 1-year-old, 100-listing, zero-dispute seller scores exactly 100.0 (40 age + 20 "
                     "volume + 40 dispute-free); a brand-new seller with no history scores exactly 40.0 "
                     "(0 age + 0 volume + 40 no-evidence-of-disputes-yet); a 1-year-old, 100-listing "
                     "seller with a 50% dispute rate scores exactly 60.0, the dispute component clamped "
                     "to its own real floor of 0\n";
    }

    std::cout << "\n-- Test 7: a single specific red flag on an otherwise perfect-trust-score seller's "
                 "listing still forces manual review -- the aggregate trust score never suppresses it --\n";
    {
        SellerHistory perfect_trust{"seller-perfect", 365, 100, 0, 50};
        BrandReference ref{"Brand", "Line", 400.0, 250.0, 350.0};
        GrayscaleImage photo = make_gradient(64, 64);

        Listing clean_listing{"L-clean", "seller-perfect", 500.0, 300.0, "authentic, with original box",
                               photo};
        Listing flagged_listing{"L-bad", "seller-perfect", 500.0, 300.0, "authentic 1:1 mirror quality",
                                 photo};

        auto eval_clean = evaluate_listing(clean_listing, ref, perfect_trust, {});
        auto eval_flagged = evaluate_listing(flagged_listing, ref, perfect_trust, {});

        CHECK(std::abs(eval_clean.trust_score - 100.0) < 1e-9);
        CHECK(!eval_clean.requires_manual_review);
        CHECK(std::abs(eval_flagged.trust_score - 100.0) < 1e-9);
        CHECK(eval_flagged.requires_manual_review);
        CHECK(eval_flagged.flags.size() == 1);
        CHECK(eval_flagged.flags[0].flag == ListingRedFlag::PROHIBITED_KEYWORD);
        std::cout << "  the identical perfect-trust-score (100.0) seller's clean listing requires no "
                     "review, but the SAME seller's listing containing \"1:1 mirror quality\" still "
                     "requires manual review -- a perfect aggregate trust score never overrides one "
                     "specific, named red flag\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
