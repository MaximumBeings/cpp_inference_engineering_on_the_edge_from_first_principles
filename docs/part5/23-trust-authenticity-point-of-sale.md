# Chapter 23: Trust and Authenticity at the Point of Sale: Counterfeits, Reconciliation, Expense Audits, and Luxury Goods

**What you will understand by the end of this chapter:**

- How to build a real perceptual image hash (the "difference hash," dHash, algorithm used in production image-deduplication systems) that catches stolen stock photography reused across unrelated e-commerce sellers, and a fully auditable seller trust score whose own aggregate value can never suppress a single specific, named red flag.
- How to implement the classic Levenshtein edit-distance algorithm from scratch and use it to reconcile an OCR-noisy receipt against a bank or card network's own clean transaction record, honestly reporting AMBIGUOUS when more than one transaction equally fits.
- How to build a real, checkable expense-policy rules engine that reports a claimed amount and a policy-approved amount side by side for every line item, never silently substituting one for the other without naming the exact rule behind the difference.
- How to implement the real, well-known Luhn check-digit algorithm (the same algorithm that validates credit-card numbers) as a luxury brand's own stated serial-number checksum, and why a single hard-failing structural check must never be outvoted by two other checks that happen to pass.

**What you need to know first:**

- Section 22.1's own honest AMBIGUOUS-refusal discipline (naming every fitting candidate rather than silently picking one) is the exact pattern Section 23.2's own receipt-to-transaction reconciliation reapplies when two transactions both fit every real matching rule.
- Section 21.3's own `CostRange` discipline -- reporting a claimed figure and a computed figure side by side, with a named reason for any gap, never silently substituting one for the other -- is the exact pattern Section 23.3's own expense-policy engine reapplies to a claimed amount versus a policy-approved amount.
- This book's own recurring structural-override discipline, first built as Chapter 20.1's gated sign-off and reapplied at Chapter 21.2's per-field handwriting review and Chapter 22.2's AA-only enforcement: a single specific, named red flag (Section 23.1) or a single hard-failing check (Section 23.4) is never outvoted by an otherwise-favorable aggregate score or a majority of passing checks.

---

Chapter 22 turned a continuous stream into a structured, auditable narrative. This chapter turns to a related but distinct real pattern: trust and authenticity at the point of a real financial transaction, where the stakes are a fraudulent purchase, an unreconciled charge, a policy-violating expense claim, or a counterfeit luxury item sold as genuine. Each of this chapter's four sections builds a real, from-scratch algorithm with its own well-established provenance outside this book -- a perceptual image hash, the classic Levenshtein edit distance, and the Luhn checksum are all real, independently famous algorithms this chapter implements and verifies against their own known reference values, not techniques this book invented for the occasion. And each section reapplies this book's own recurring discipline to a financial context: a named red flag is never silently smoothed over by a favorable aggregate score, an honest range or an honest ambiguity is reported rather than a false single answer, and a structural screening pass is never overstated into the final human judgment it is not.

## 23.1 A Brand-Reference Database and Counterfeit-Screening Engine for E-Commerce Listings

### Intuition

A real counterfeit-screening system cannot inspect an item in a shopper's hand, but it can compute several real, checkable properties of a LISTING'S OWN STATED DATA: whether its own photo has already been used by an unrelated seller, whether its price is implausibly below what a genuine item ever sells for, and whether its own description contains language sellers of counterfeit goods commonly use. This section builds all three from scratch, and adds a real, fully auditable seller trust score that never gets to overrule a single specific red flag.

### The Concept, In Detail

`compute_dhash` is a real, from-scratch implementation of the difference hash (dHash) algorithm production image-deduplication systems actually use: resize to 9x8, then encode 64 real horizontal pixel comparisons into a 64-bit value -- Test 1 confirms two independently generated but pixel-identical images hash identically (Hamming distance 0), while a clearly different image hashes far apart, and Test 2 confirms a photo reused across two different sellers is flagged as a duplicate stock image, naming the other seller, while a genuinely unique photo is not. `screen_listing` also checks a real, stated authentic-price floor and a case-insensitive prohibited-keyword scan -- Test 3 through Test 5 confirm each check's own exact boundary and named detail.

`compute_seller_trust_score` combines three fully auditable, individually named components -- account age, listing volume, and dispute rate -- into a single real, hand-verifiable score, confirmed exactly in Test 6 for three real seller profiles. Test 7 is this section's own central honesty check: the identical seller, with a perfect trust score of 100.0, requires no review for a clean listing but STILL requires manual review the moment their own listing description contains a single prohibited keyword -- proving a perfect aggregate score can never suppress one specific, named red flag.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_counterfeit_screening_and_seller_trust.cpp -o 01_counterfeit_screening_and_seller_trust
./01_counterfeit_screening_and_seller_trust
```

**Sample input:** two pixel-identical checkerboard images and one clearly different gradient image checked under the real dHash algorithm; a photo reused across two sellers checked to flag a duplicate, naming the other seller, while a genuinely unique photo does not; prohibited keywords caught case-insensitively inside longer listing text while a legitimate description is not falsely flagged; an authentic-price floor and a weight-spec range each checked exactly at their own stated boundaries; a seller trust score checked against exact hand-computed values for three real seller profiles; and a perfect-trust-score seller's listing checked to still require manual review the moment it contains one prohibited keyword.

```text
========================================================
Chapter 23.1: A Counterfeit-Screening and Seller-Trust Engine
========================================================

-- Test 1: the real dHash algorithm hashes identical images identically and clearly different images far apart --
  two independently generated but pixel-identical checkerboard images hash to the identical 64-bit value (Hamming distance 0); a clearly different gradient image hashes 28 bits away

-- Test 2: a photo reused across two different sellers is flagged as a duplicate stock image, naming the other seller; a genuinely different photo is not --
  seller-B's own listing photo, byte-identical to seller-A's already-registered photo, is flagged as a duplicate stock image naming seller-A; seller-C's own genuinely different photo is not flagged

-- Test 3: prohibited keywords are caught case-insensitively and embedded in longer text; a legitimate description is not falsely flagged --
  "AAA Quality" and "REPLICA" are both caught case-insensitively inside longer listing text; a genuinely clean description mentioning "authentic" is not falsely flagged

-- Test 4: the authentic-price-floor check is exact at its own stated boundary --
  a $100 MSRP-minimum brand's own real floor is exactly $40.00: a $40.00 listing passes, a $39.99 listing is flagged

-- Test 5: a weight outside the brand's own real spec range is flagged with the exact expected range named --
  a 200g item claiming a brand whose real spec range is [250g, 350g] is flagged naming that exact range; a 300g item in range is not

-- Test 6: the seller trust score's own components are exact, hand-computed values for three real seller profiles --
  a 1-year-old, 100-listing, zero-dispute seller scores exactly 100.0 (40 age + 20 volume + 40 dispute-free); a brand-new seller with no history scores exactly 40.0 (0 age + 0 volume + 40 no-evidence-of-disputes-yet); a 1-year-old, 100-listing seller with a 50% dispute rate scores exactly 60.0, the dispute component clamped to its own real floor of 0

-- Test 7: a single specific red flag on an otherwise perfect-trust-score seller's listing still forces manual review -- the aggregate trust score never suppresses it --
  the identical perfect-trust-score (100.0) seller's clean listing requires no review, but the SAME seller's listing containing "1:1 mirror quality" still requires manual review -- a perfect aggregate trust score never overrides one specific, named red flag

25/25 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a low aggregate trust score as the only real signal worth acting on"
    It would be a real, meaningful screening system to flag any seller whose own trust score falls below a stated floor for manual review and stop there -- and this section's own `evaluate_listing` does include exactly that floor. But Test 7 exists specifically to show that floor is not the whole property this section's own risk profile requires: a seller with a PERFECT trust score of 100.0 -- a full year of history, a hundred listings, zero disputes -- can still list an item whose own description contains "1:1 mirror quality," and no amount of accumulated seller history should make that specific, named red flag disappear into a favorable average. `evaluate_listing`'s own rule -- ANY flag OR a low trust score triggers review -- is what keeps a single bad listing from hiding behind an otherwise-excellent seller record.

## 23.2 Receipt-to-Transaction Reconciliation

### Intuition

A receipt extracted by an OCR or vision-language pipeline, the same real kind of pipeline this book built in Chapter 21.1, can carry small per-character extraction errors; a bank or card network's own transaction record carries the true merchant name as the payment network itself recorded it. Reconciling the two needs a real fuzzy-matching algorithm for the merchant name, a real tolerance for an added tip in the settled amount, and a real settlement-date window -- and an honest refusal to guess when more than one transaction equally fits.

### The Concept, In Detail

`levenshtein_distance` is the real, classic edit-distance algorithm, confirmed in Test 1 against its own famous exact reference value (`KITTEN` to `SITTING` is distance 3). `merchant_names_fuzzy_match` normalizes and applies a real, stated edit-distance ratio threshold built specifically for OCR-introduced noise -- Test 2 confirms a single-character OCR misread ("WALGREEN5" for "WALGREENS") fuzzy-matches correctly while a genuinely different merchant does not. `amounts_match` and `within_settlement_window` each encode a real, stated policy: a settled amount may exceed a receipt's own printed subtotal by an added tip, but only up to a real, stated cap, confirmed exactly at that boundary in Test 3; and a settlement can only occur ON OR AFTER its own purchase date, confirmed exactly at its own real window boundary in Test 4.

`reconcile_receipt` is this section's own central honesty check, reapplying Section 22.1's own AMBIGUOUS-refusal discipline to a financial-matching context: Test 6 constructs two transactions that both genuinely fit every real matching rule -- same merchant, same plausible tip-adjusted amount, both within the settlement window -- and confirms the function reports `AMBIGUOUS`, naming both candidates, rather than silently picking the nearer one. `find_transactions_missing_receipts` closes the loop in the other direction: Test 8 confirms a transaction with no receipt on file at all is flagged separately, a real, useful signal for an expense audit.

### Code and Verification

```cpp
// Chapter 23.2 -- A receipt, extracted by the same kind of OCR/VLM
// pipeline this book built in Chapter 21.1, can carry small per-character
// extraction errors; a bank or card network's own transaction record, by
// contrast, carries the true merchant name as recorded by the payment
// network itself. Reconciling the two real records requires a real fuzzy
// string-matching algorithm -- this section builds the classic Levenshtein
// edit-distance algorithm from scratch -- plus two further real matching
// rules: a settlement date can only fall ON OR AFTER a purchase date,
// never before, and a settled amount may legitimately exceed a receipt's
// own printed subtotal by an added tip, but only up to a real, stated cap.
//
// A note on this section's own honest scope: this section's own fuzzy
// merchant-name match is built specifically for OCR-INTRODUCED per-
// character noise (a misread digit or letter), not for the different
// real problem of payment-processor descriptor mangling (a transaction
// descriptor like "SQ *STARBUCKS COFFEE" for a receipt's own "Starbucks")
// -- a different real matching technique a production reconciliation
// system would need in addition to, not instead of, this one.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_receipt_transaction_reconciliation.cpp -o 02_receipt_transaction_reconciliation
// Run:     ./02_receipt_transaction_reconciliation

#include <algorithm>
#include <cctype>
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
// PART 1: the real, classic Levenshtein edit-distance algorithm, built
// from scratch with a plain O(n*m) dynamic-programming table.
// =======================================================================
int levenshtein_distance(const std::string& a, const std::string& b) {
    std::size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 0; i <= n; i++) dp[i][0] = static_cast<int>(i);
    for (std::size_t j = 0; j <= m; j++) dp[0][j] = static_cast<int>(j);
    for (std::size_t i = 1; i <= n; i++) {
        for (std::size_t j = 1; j <= m; j++) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost});
        }
    }
    return dp[n][m];
}

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

// A real, stated policy choice: an OCR'd merchant name may differ from
// its own true merchant name by up to this fraction of its own length --
// enough to absorb a single misread character on a short name, without
// absorbing a genuinely different merchant.
constexpr double MERCHANT_MATCH_MAX_EDIT_RATIO = 0.34;

bool merchant_names_fuzzy_match(const std::string& ocr_name, const std::string& true_name) {
    std::string a = to_upper(ocr_name), b = to_upper(true_name);
    int dist = levenshtein_distance(a, b);
    std::size_t max_len = std::max(a.size(), b.size());
    if (max_len == 0) return true;
    double ratio = static_cast<double>(dist) / static_cast<double>(max_len);
    return ratio <= MERCHANT_MATCH_MAX_EDIT_RATIO;
}

// =======================================================================
// PART 2: real amount matching -- exact, or tip-adjusted up to a real,
// stated cap -- and real settlement-date-window matching.
// =======================================================================
constexpr double AMOUNT_EXACT_EPSILON = 0.005;
constexpr double MAX_TIP_FRACTION = 0.25;
constexpr int SETTLEMENT_WINDOW_MAX_DAYS = 3;

bool amounts_match(double receipt_amount, double txn_amount) {
    if (std::abs(txn_amount - receipt_amount) < AMOUNT_EXACT_EPSILON) return true;
    if (txn_amount > receipt_amount) {
        double tip = txn_amount - receipt_amount;
        return tip <= receipt_amount * MAX_TIP_FRACTION + AMOUNT_EXACT_EPSILON;
    }
    return false;
}

// Settlement can only occur ON OR AFTER the purchase's own real period --
// a transaction dated before its own receipt's purchase period can never
// be the same real event, regardless of how small that gap is.
bool within_settlement_window(int receipt_period, int txn_period) {
    int delta = txn_period - receipt_period;
    return delta >= 0 && delta <= SETTLEMENT_WINDOW_MAX_DAYS;
}

// =======================================================================
// PART 3: the reconciliation engine -- honestly reports AMBIGUOUS, naming
// every fitting candidate, exactly the same discipline Section 22.1's own
// multi-camera correlation applied to a temporally ambiguous re-
// identification.
// =======================================================================
struct Receipt {
    std::string receipt_id;
    std::string merchant_name_ocr;
    double amount = 0.0;
    int purchase_period = 0;
};

struct Transaction {
    std::string transaction_id;
    std::string merchant_name;
    double amount = 0.0;
    int settlement_period = 0;
};

enum class ReconciliationOutcome { MATCHED, AMBIGUOUS, MISSING_TRANSACTION };

struct ReconciliationResult {
    ReconciliationOutcome outcome;
    std::vector<std::string> candidate_transaction_ids;
};

ReconciliationResult reconcile_receipt(const Receipt& r, const std::vector<Transaction>& transactions) {
    std::vector<std::string> candidates;
    for (const auto& t : transactions) {
        if (within_settlement_window(r.purchase_period, t.settlement_period) &&
            amounts_match(r.amount, t.amount) &&
            merchant_names_fuzzy_match(r.merchant_name_ocr, t.merchant_name)) {
            candidates.push_back(t.transaction_id);
        }
    }
    if (candidates.empty()) return {ReconciliationOutcome::MISSING_TRANSACTION, {}};
    if (candidates.size() == 1) return {ReconciliationOutcome::MATCHED, candidates};
    return {ReconciliationOutcome::AMBIGUOUS, candidates};
}

// A transaction never claimed by any receipt's own unique match is a real,
// useful signal for an expense audit: a charge with no corresponding
// receipt on file at all.
std::vector<std::string> find_transactions_missing_receipts(const std::vector<Transaction>& transactions,
                                                              const std::vector<ReconciliationResult>& results) {
    std::vector<std::string> claimed;
    for (const auto& result : results) {
        if (result.outcome == ReconciliationOutcome::MATCHED) {
            claimed.push_back(result.candidate_transaction_ids[0]);
        }
    }
    std::vector<std::string> missing;
    for (const auto& t : transactions) {
        if (std::find(claimed.begin(), claimed.end(), t.transaction_id) == claimed.end()) {
            missing.push_back(t.transaction_id);
        }
    }
    return missing;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 23.2: A Receipt-to-Transaction Reconciliation Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real Levenshtein algorithm matches its own famous exact reference "
                 "value (KITTEN -> SITTING is distance 3), alongside identical and fully-disjoint strings --\n";
    {
        CHECK(levenshtein_distance("KITTEN", "SITTING") == 3);
        CHECK(levenshtein_distance("STARBUCKS", "STARBUCKS") == 0);
        CHECK(levenshtein_distance("", "ABC") == 3);
        std::cout << "  KITTEN -> SITTING computes to exactly 3, matching this algorithm's own famous "
                     "reference value; an identical pair computes to 0; an empty string against a "
                     "3-character string computes to exactly 3 (three real insertions)\n";
    }

    std::cout << "\n-- Test 2: a single-character OCR misread is fuzzy-matched to its own true merchant "
                 "name, while a genuinely different merchant is not --\n";
    {
        CHECK(merchant_names_fuzzy_match("WALGREEN5", "WALGREENS"));
        CHECK(!merchant_names_fuzzy_match("WALGREENS", "CVS PHARMACY"));
        std::cout << "  \"WALGREEN5\" (a real OCR digit/letter confusion of the trailing S) fuzzy-matches "
                     "its own true name \"WALGREENS\" at an edit ratio of 1/9; \"WALGREENS\" does not "
                     "match the genuinely different merchant \"CVS PHARMACY\"\n";
    }

    std::cout << "\n-- Test 3: amount matching accepts an added tip up to its own real, stated cap, and "
                 "rejects one beyond it, exactly at the boundary --\n";
    {
        CHECK(amounts_match(40.00, 40.00));
        CHECK(amounts_match(40.00, 46.00));
        CHECK(amounts_match(40.00, 50.00));
        CHECK(!amounts_match(40.00, 50.01));
        CHECK(!amounts_match(40.00, 39.00));
        std::cout << "  a $40.00 receipt matches an exact $40.00 settlement, a 15% tip ($46.00), and "
                     "exactly its own real 25% tip cap ($50.00); it does NOT match $50.01 (one cent over "
                     "the cap) or a $39.00 settlement below its own printed amount\n";
    }

    std::cout << "\n-- Test 4: the settlement-date window is exact at its own real boundary, and a "
                 "settlement dated before its own purchase never matches regardless of gap size --\n";
    {
        CHECK(within_settlement_window(100, 100));
        CHECK(within_settlement_window(100, 103));
        CHECK(!within_settlement_window(100, 104));
        CHECK(!within_settlement_window(100, 99));
        CHECK(!within_settlement_window(100, 50));
        std::cout << "  a same-period settlement and a settlement exactly 3 periods later both match; "
                     "one 4 periods later does not, and one dated even 1 period before its own purchase "
                     "(let alone 50) never matches\n";
    }

    std::cout << "\n-- Test 5: a receipt with exactly one fitting transaction is reported MATCHED, naming "
                 "that transaction --\n";
    {
        Receipt r{"R-1", "TARGEI", 52.30, 200};
        std::vector<Transaction> pool = {
            {"T-1", "TARGET", 52.30, 201},
            {"T-2", "WALGREENS", 12.00, 201},
        };
        auto result = reconcile_receipt(r, pool);
        CHECK(result.outcome == ReconciliationOutcome::MATCHED);
        CHECK(result.candidate_transaction_ids.size() == 1);
        CHECK(result.candidate_transaction_ids[0] == "T-1");
        std::cout << "  an OCR'd receipt reading \"TARGEI\" for $52.30 matches exactly one real "
                     "transaction (T-1, \"TARGET\", $52.30) out of a 2-transaction pool\n";
    }

    std::cout << "\n-- Test 6: two transactions that BOTH fit every real matching rule are honestly "
                 "reported AMBIGUOUS, naming both candidates, rather than silently picking one --\n";
    {
        Receipt r{"R-2", "STARBUCKS", 12.00, 300};
        std::vector<Transaction> pool = {
            {"T-10", "STARBUCKS", 13.50, 300},
            {"T-11", "STARBUCKS", 13.50, 301},
        };
        auto result = reconcile_receipt(r, pool);
        CHECK(result.outcome == ReconciliationOutcome::AMBIGUOUS);
        CHECK(result.candidate_transaction_ids.size() == 2);
        std::cout << "  a $12.00 Starbucks receipt with a real, plausible tip finds TWO transactions "
                     "(T-10 same-day, T-11 one day later) both matching on amount, merchant, and window "
                     "-- reported AMBIGUOUS, naming both, rather than guessing which one is correct\n";
    }

    std::cout << "\n-- Test 7: a receipt with no fitting transaction anywhere in the pool is reported "
                 "MISSING_TRANSACTION --\n";
    {
        Receipt r{"R-3", "OFFICE SUPPLY CO", 200.00, 400};
        std::vector<Transaction> pool = {
            {"T-20", "OFFICE SUPPLY CO", 45.00, 400},
        };
        auto result = reconcile_receipt(r, pool);
        CHECK(result.outcome == ReconciliationOutcome::MISSING_TRANSACTION);
        CHECK(result.candidate_transaction_ids.empty());
        std::cout << "  a $200.00 receipt against a pool containing only a $45.00 transaction from the "
                     "same real merchant finds no amount-matching candidate and is reported "
                     "MISSING_TRANSACTION\n";
    }

    std::cout << "\n-- Test 8: a transaction never claimed by any receipt's own unique match is flagged "
                 "MISSING a receipt, a real, useful signal for an expense audit --\n";
    {
        std::vector<Receipt> receipts = {
            {"R-4", "TARGET", 52.30, 500},
        };
        std::vector<Transaction> pool = {
            {"T-30", "TARGET", 52.30, 500},
            {"T-31", "UNKNOWN VENDOR LLC", 890.00, 500},
        };
        std::vector<ReconciliationResult> results;
        for (const auto& r : receipts) results.push_back(reconcile_receipt(r, pool));
        auto missing = find_transactions_missing_receipts(pool, results);
        CHECK(missing.size() == 1);
        CHECK(missing[0] == "T-31");
        std::cout << "  of a 2-transaction pool, T-30 is claimed by its own matching receipt while T-31, "
                     "an $890.00 charge with no receipt on file at all, is flagged as missing a receipt\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_receipt_transaction_reconciliation.cpp -o 02_receipt_transaction_reconciliation
./02_receipt_transaction_reconciliation
```

**Sample input:** the real Levenshtein algorithm checked against its own famous KITTEN-to-SITTING reference value and two further hand-computable cases; an OCR-noisy merchant name checked to fuzzy-match its own true name while a genuinely different merchant does not; amount matching checked against an exact tip cap boundary; a settlement-date window checked against its own exact boundary and against a settlement dated before its own purchase; a receipt with exactly one fitting transaction checked to report MATCHED; two equally-fitting transactions checked to report AMBIGUOUS, naming both; a receipt with no fitting transaction checked to report MISSING_TRANSACTION; and a transaction with no matching receipt checked to be flagged as missing one.

```text
========================================================
Chapter 23.2: A Receipt-to-Transaction Reconciliation Engine
========================================================

-- Test 1: the real Levenshtein algorithm matches its own famous exact reference value (KITTEN -> SITTING is distance 3), alongside identical and fully-disjoint strings --
  KITTEN -> SITTING computes to exactly 3, matching this algorithm's own famous reference value; an identical pair computes to 0; an empty string against a 3-character string computes to exactly 3 (three real insertions)

-- Test 2: a single-character OCR misread is fuzzy-matched to its own true merchant name, while a genuinely different merchant is not --
  "WALGREEN5" (a real OCR digit/letter confusion of the trailing S) fuzzy-matches its own true name "WALGREENS" at an edit ratio of 1/9; "WALGREENS" does not match the genuinely different merchant "CVS PHARMACY"

-- Test 3: amount matching accepts an added tip up to its own real, stated cap, and rejects one beyond it, exactly at the boundary --
  a $40.00 receipt matches an exact $40.00 settlement, a 15% tip ($46.00), and exactly its own real 25% tip cap ($50.00); it does NOT match $50.01 (one cent over the cap) or a $39.00 settlement below its own printed amount

-- Test 4: the settlement-date window is exact at its own real boundary, and a settlement dated before its own purchase never matches regardless of gap size --
  a same-period settlement and a settlement exactly 3 periods later both match; one 4 periods later does not, and one dated even 1 period before its own purchase (let alone 50) never matches

-- Test 5: a receipt with exactly one fitting transaction is reported MATCHED, naming that transaction --
  an OCR'd receipt reading "TARGEI" for $52.30 matches exactly one real transaction (T-1, "TARGET", $52.30) out of a 2-transaction pool

-- Test 6: two transactions that BOTH fit every real matching rule are honestly reported AMBIGUOUS, naming both candidates, rather than silently picking one --
  a $12.00 Starbucks receipt with a real, plausible tip finds TWO transactions (T-10 same-day, T-11 one day later) both matching on amount, merchant, and window -- reported AMBIGUOUS, naming both, rather than guessing which one is correct

-- Test 7: a receipt with no fitting transaction anywhere in the pool is reported MISSING_TRANSACTION --
  a $200.00 receipt against a pool containing only a $45.00 transaction from the same real merchant finds no amount-matching candidate and is reported MISSING_TRANSACTION

-- Test 8: a transaction never claimed by any receipt's own unique match is flagged MISSING a receipt, a real, useful signal for an expense audit --
  of a 2-transaction pool, T-30 is claimed by its own matching receipt while T-31, an $890.00 charge with no receipt on file at all, is flagged as missing a receipt

24/24 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating this section's own OCR-noise fuzzy match as a general merchant-name matcher"
    `merchant_names_fuzzy_match` is built and tuned specifically for the kind of noise a receipt's own OCR extraction introduces -- a single misread character on an otherwise-correct name. It is tempting to assume the same function would also handle the different real problem of payment-processor descriptor mangling, where a transaction descriptor reads "SQ *STARBUCKS COFFEE" for a receipt's own "Starbucks" -- a difference not of a few misread characters but of an entirely different real string, prefixed and suffixed by processor-specific codes this section's own edit-distance ratio was never tuned to absorb. A production reconciliation system needs a SEPARATE real technique -- token-based matching against a known table of processor prefixes, for instance -- for that different problem; treating one section's own narrow, real fix as a general solution to every real merchant-name mismatch would produce exactly the false confidence this book's own honesty discipline exists to prevent.

## 23.3 Automated Expense-Report Policy Enforcement

### Intuition

A real corporate travel policy is a real, checkable rules engine: a per-category spending cap, a per-mile reimbursement rate, a category that is never reimbursable regardless of amount, and a receipt-required threshold. The one discipline this section adds on top of a straightforward rules engine: never let the policy-computed APPROVED amount silently replace the employee's own CLAIMED amount in the report's own total without naming, for every discrepancy, the specific rule behind it.

### The Concept, In Detail

`evaluate_line_item` applies four real, independently checkable rules by category: a real per-city-tier lodging cap (Test 2 confirms an over-cap claim in the cheapest tier is flagged against ITS OWN tier's cap, not a different one), a real daily meals cap (Test 1), a real per-mile mileage rate the claim is checked against rather than merely capped (Test 3 confirms a mismatched mileage claim is approved at the CORRECTLY COMPUTED figure while naming both the claimed and expected values), and a blanket alcohol exclusion that approves $0.00 regardless of the claimed amount and never additionally triggers the receipt-required check on top of its own exclusion (Test 4). The receipt-required threshold is confirmed exact at its own real boundary in Test 5.

`evaluate_expense_report`'s own Test 6 is this section's central honesty check, reapplying Section 21.3's own `CostRange` discipline to a policy-enforcement context: a 4-item report's own CLAIMED total still includes a $90 meal claim's full $90 even though only $75 is approved, and still includes an excluded $30 alcohol item's full $30 even though $0 is approved -- the report's own claimed figure is never silently reduced anywhere in this file; only the separate approved figure and each item's own named violation tell the real story.

### Code and Verification

```cpp
// Chapter 23.3 -- An automated expense-report policy engine is a real,
// checkable rules engine over a stated, real corporate travel policy: a
// per-category spending cap, a per-mile reimbursement rate, a blanket
// non-reimbursable category, and a receipt-required threshold. This
// section's own central honesty discipline, consistent with Section
// 21.3's own `CostRange` and this book's own recurring refusal to let a
// flagged discrepancy quietly resolve itself: the engine reports both the
// employee's own CLAIMED amount and the policy's own computed APPROVED
// amount side by side for every line item, and names the exact, specific
// rule behind every discrepancy between them -- it never silently
// substitutes one number for the other without saying so.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_expense_report_policy_enforcement.cpp -o 03_expense_report_policy_enforcement
// Run:     ./03_expense_report_policy_enforcement

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
// PART 1: the stated real policy constants and the per-city-tier lodging
// cap table.
// =======================================================================
enum class CityTier { TIER_1, TIER_2, TIER_3 };

// A real, stated per-diem-tier lookup: a major-metro tier-1 city's own
// real cost of lodging is not the same real number as a tier-3 city's,
// and this policy's own cap reflects that rather than applying one flat
// number everywhere.
double lodging_cap_for_tier(CityTier tier) {
    switch (tier) {
        case CityTier::TIER_1: return 350.0;
        case CityTier::TIER_2: return 200.0;
        case CityTier::TIER_3: return 120.0;
    }
    return 0.0;  // unreachable: CityTier's own 3 enumerators are fully handled above.
}

constexpr double MEALS_DAILY_CAP = 75.0;
constexpr double MILEAGE_RATE_PER_MILE = 0.67;
constexpr double RECEIPT_REQUIRED_THRESHOLD = 25.0;
constexpr double AMOUNT_EPSILON = 0.005;

// =======================================================================
// PART 2: the line item, the named violations, and the per-item
// evaluation.
// =======================================================================
enum class ExpenseCategory { MEALS, LODGING, MILEAGE, ALCOHOL, OFFICE_SUPPLIES };

enum class PolicyViolation {
    OVER_CATEGORY_CAP,
    ALCOHOL_NOT_REIMBURSABLE,
    MISSING_REQUIRED_RECEIPT,
    MILEAGE_CALCULATION_MISMATCH,
};

struct ExpenseLineItem {
    std::string item_id;
    ExpenseCategory category;
    double claimed_amount = 0.0;
    double miles = 0.0;
    CityTier city_tier = CityTier::TIER_3;
    bool has_receipt = false;
};

struct ViolationEntry {
    PolicyViolation violation;
    std::string detail;
};

struct LineItemEvaluation {
    std::string item_id;
    double claimed_amount = 0.0;
    double approved_amount = 0.0;
    std::vector<ViolationEntry> violations;
};

LineItemEvaluation evaluate_line_item(const ExpenseLineItem& item) {
    LineItemEvaluation eval;
    eval.item_id = item.item_id;
    eval.claimed_amount = item.claimed_amount;

    switch (item.category) {
        case ExpenseCategory::ALCOHOL: {
            eval.approved_amount = 0.0;
            eval.violations.push_back(
                {PolicyViolation::ALCOHOL_NOT_REIMBURSABLE,
                 "claimed $" + std::to_string(item.claimed_amount) +
                     " for an alcohol line item, which this policy never reimburses regardless of amount"});
            return eval;  // alcohol is fully excluded; the receipt-required check never applies to it.
        }
        case ExpenseCategory::MEALS: {
            double cap = MEALS_DAILY_CAP;
            eval.approved_amount = std::min(item.claimed_amount, cap);
            if (item.claimed_amount > cap + AMOUNT_EPSILON) {
                eval.violations.push_back(
                    {PolicyViolation::OVER_CATEGORY_CAP,
                     "claimed $" + std::to_string(item.claimed_amount) + " exceeds the real $" +
                         std::to_string(cap) + " daily meals cap by exactly $" +
                         std::to_string(item.claimed_amount - cap)});
            }
            break;
        }
        case ExpenseCategory::LODGING: {
            double cap = lodging_cap_for_tier(item.city_tier);
            eval.approved_amount = std::min(item.claimed_amount, cap);
            if (item.claimed_amount > cap + AMOUNT_EPSILON) {
                eval.violations.push_back(
                    {PolicyViolation::OVER_CATEGORY_CAP,
                     "claimed $" + std::to_string(item.claimed_amount) + " exceeds the real $" +
                         std::to_string(cap) +
                         " per-night lodging cap for this city's own stated tier by exactly $" +
                         std::to_string(item.claimed_amount - cap)});
            }
            break;
        }
        case ExpenseCategory::MILEAGE: {
            double expected = item.miles * MILEAGE_RATE_PER_MILE;
            eval.approved_amount = expected;
            if (std::abs(item.claimed_amount - expected) > AMOUNT_EPSILON) {
                eval.violations.push_back(
                    {PolicyViolation::MILEAGE_CALCULATION_MISMATCH,
                     "claimed $" + std::to_string(item.claimed_amount) + " does not match " +
                         std::to_string(item.miles) + " miles at the real $" +
                         std::to_string(MILEAGE_RATE_PER_MILE) + "/mile rate ($" + std::to_string(expected) +
                         " expected)"});
            }
            return eval;  // mileage is verified against a trip log, not a receipt, per this policy's own
                          // stated scope; the receipt-required check below never applies to it.
        }
        case ExpenseCategory::OFFICE_SUPPLIES: {
            eval.approved_amount = item.claimed_amount;
            break;
        }
    }

    if (item.claimed_amount > RECEIPT_REQUIRED_THRESHOLD + AMOUNT_EPSILON && !item.has_receipt) {
        eval.violations.push_back(
            {PolicyViolation::MISSING_REQUIRED_RECEIPT,
             "claimed $" + std::to_string(item.claimed_amount) +
                 " has no receipt on file, above the real $" + std::to_string(RECEIPT_REQUIRED_THRESHOLD) +
                 " requirement threshold"});
    }

    return eval;
}

// =======================================================================
// PART 3: the full-report aggregation -- the claimed and approved totals
// are each a real, exact sum of the per-item figures, and every
// violation stays attributed to its own specific item.
// =======================================================================
struct ExpenseReportSummary {
    double total_claimed = 0.0;
    double total_approved = 0.0;
    std::vector<LineItemEvaluation> line_items;
};

ExpenseReportSummary evaluate_expense_report(const std::vector<ExpenseLineItem>& items) {
    ExpenseReportSummary summary;
    for (const auto& item : items) {
        auto eval = evaluate_line_item(item);
        summary.total_claimed += eval.claimed_amount;
        summary.total_approved += eval.approved_amount;
        summary.line_items.push_back(eval);
    }
    return summary;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 23.3: An Automated Expense-Report Policy Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a meal claim under and exactly at the real daily cap is fully approved with "
                 "zero violations; one over the cap is flagged with the exact computed overage --\n";
    {
        ExpenseLineItem under{"E-1", ExpenseCategory::MEALS, 50.0, 0, CityTier::TIER_3, true};
        ExpenseLineItem at_cap{"E-2", ExpenseCategory::MEALS, 75.0, 0, CityTier::TIER_3, true};
        ExpenseLineItem over{"E-3", ExpenseCategory::MEALS, 90.0, 0, CityTier::TIER_3, true};
        auto eval_under = evaluate_line_item(under);
        auto eval_at_cap = evaluate_line_item(at_cap);
        auto eval_over = evaluate_line_item(over);
        CHECK(std::abs(eval_under.approved_amount - 50.0) < 1e-9);
        CHECK(eval_under.violations.empty());
        CHECK(std::abs(eval_at_cap.approved_amount - 75.0) < 1e-9);
        CHECK(eval_at_cap.violations.empty());
        CHECK(std::abs(eval_over.approved_amount - 75.0) < 1e-9);
        CHECK(eval_over.violations.size() == 1);
        CHECK(eval_over.violations[0].violation == PolicyViolation::OVER_CATEGORY_CAP);
        CHECK(eval_over.violations[0].detail.find("15") != std::string::npos);
        std::cout << "  a $50.00 and an exactly-$75.00 meal claim are both fully approved with zero "
                     "violations; a $90.00 claim is capped to $75.00 approved and flagged with the "
                     "exact $15.00 overage\n";
    }

    std::cout << "\n-- Test 2: lodging's own real per-city-tier cap is applied correctly for all 3 tiers, "
                 "and an over-cap claim in the cheapest tier is still flagged against ITS OWN tier's cap, "
                 "not a different tier's --\n";
    {
        ExpenseLineItem tier1{"E-4", ExpenseCategory::LODGING, 340.0, 0, CityTier::TIER_1, true};
        ExpenseLineItem tier2{"E-5", ExpenseCategory::LODGING, 190.0, 0, CityTier::TIER_2, true};
        ExpenseLineItem tier3_over{"E-6", ExpenseCategory::LODGING, 150.0, 0, CityTier::TIER_3, true};
        auto eval1 = evaluate_line_item(tier1);
        auto eval2 = evaluate_line_item(tier2);
        auto eval3 = evaluate_line_item(tier3_over);
        CHECK(eval1.violations.empty());
        CHECK(std::abs(eval1.approved_amount - 340.0) < 1e-9);
        CHECK(eval2.violations.empty());
        CHECK(std::abs(eval2.approved_amount - 190.0) < 1e-9);
        CHECK(eval3.violations.size() == 1);
        CHECK(std::abs(eval3.approved_amount - 120.0) < 1e-9);
        CHECK(eval3.violations[0].detail.find("30") != std::string::npos);
        std::cout << "  a $340 tier-1 stay and a $190 tier-2 stay both pass under their own real caps "
                     "($350 and $200); a $150 tier-3 stay is capped to $120.00 approved and flagged with "
                     "the exact $30.00 overage against ITS OWN tier-3 cap\n";
    }

    std::cout << "\n-- Test 3: a mileage claim matching the real rate exactly is approved with zero "
                 "violations; a claim that does not match its own computed value is flagged naming both "
                 "figures --\n";
    {
        ExpenseLineItem correct{"E-7", ExpenseCategory::MILEAGE, 67.0, 100.0, CityTier::TIER_3, false};
        ExpenseLineItem wrong{"E-8", ExpenseCategory::MILEAGE, 70.0, 100.0, CityTier::TIER_3, false};
        auto eval_correct = evaluate_line_item(correct);
        auto eval_wrong = evaluate_line_item(wrong);
        CHECK(std::abs(eval_correct.approved_amount - 67.0) < 1e-9);
        CHECK(eval_correct.violations.empty());
        CHECK(std::abs(eval_wrong.approved_amount - 67.0) < 1e-9);
        CHECK(eval_wrong.violations.size() == 1);
        CHECK(eval_wrong.violations[0].violation == PolicyViolation::MILEAGE_CALCULATION_MISMATCH);
        std::cout << "  100 miles at the real $0.67/mile rate computes to exactly $67.00: a claim of "
                     "$67.00 passes with zero violations, while a claim of $70.00 for the same 100 miles "
                     "is approved at the correctly computed $67.00 and flagged, naming both the claimed "
                     "and the expected figure\n";
    }

    std::cout << "\n-- Test 4: an alcohol line item is approved at $0.00 regardless of its claimed amount, "
                 "and never triggers the receipt-required check --\n";
    {
        ExpenseLineItem wine{"E-9", ExpenseCategory::ALCOHOL, 45.0, 0, CityTier::TIER_3, false};
        auto eval = evaluate_line_item(wine);
        CHECK(std::abs(eval.approved_amount - 0.0) < 1e-9);
        CHECK(eval.violations.size() == 1);
        CHECK(eval.violations[0].violation == PolicyViolation::ALCOHOL_NOT_REIMBURSABLE);
        std::cout << "  a $45.00 alcohol claim with no receipt on file is approved at exactly $0.00, "
                     "flagged only for ALCOHOL_NOT_REIMBURSABLE -- never additionally flagged for a "
                     "missing receipt, since a fully excluded category has nothing left to require one "
                     "for\n";
    }

    std::cout << "\n-- Test 5: the receipt-required threshold is exact at its own real boundary -- claims "
                 "at or below it never require a receipt, and claims above it do --\n";
    {
        ExpenseLineItem at_threshold{"E-10", ExpenseCategory::OFFICE_SUPPLIES, 25.0, 0, CityTier::TIER_3, false};
        ExpenseLineItem above_threshold{"E-11", ExpenseCategory::OFFICE_SUPPLIES, 25.01, 0, CityTier::TIER_3, false};
        auto eval_at = evaluate_line_item(at_threshold);
        auto eval_above = evaluate_line_item(above_threshold);
        CHECK(eval_at.violations.empty());
        CHECK(eval_above.violations.size() == 1);
        CHECK(eval_above.violations[0].violation == PolicyViolation::MISSING_REQUIRED_RECEIPT);
        std::cout << "  a $25.00 office-supply claim with no receipt requires none and is approved with "
                     "zero violations; a $25.01 claim with no receipt is flagged MISSING_REQUIRED_RECEIPT\n";
    }

    std::cout << "\n-- Test 6: a full multi-item report computes its own claimed and approved totals as "
                 "exact sums, and attributes every violation to its own specific line item, never "
                 "silently dropping a flagged item's own claimed contribution to the total --\n";
    {
        std::vector<ExpenseLineItem> items = {
            {"E-20", ExpenseCategory::MEALS, 90.0, 0, CityTier::TIER_3, true},        // over cap by 15
            {"E-21", ExpenseCategory::LODGING, 340.0, 0, CityTier::TIER_1, true},     // fine
            {"E-22", ExpenseCategory::MILEAGE, 67.0, 100.0, CityTier::TIER_3, false}, // fine
            {"E-23", ExpenseCategory::ALCOHOL, 30.0, 0, CityTier::TIER_3, false},     // excluded
        };
        auto summary = evaluate_expense_report(items);
        double expected_claimed = 90.0 + 340.0 + 67.0 + 30.0;
        double expected_approved = 75.0 + 340.0 + 67.0 + 0.0;
        CHECK(std::abs(summary.total_claimed - expected_claimed) < 1e-9);
        CHECK(std::abs(summary.total_approved - expected_approved) < 1e-9);
        CHECK(summary.line_items.size() == 4);
        int total_violations = 0;
        for (const auto& li : summary.line_items) total_violations += static_cast<int>(li.violations.size());
        CHECK(total_violations == 2);  // E-20's over-cap, E-23's alcohol exclusion
        CHECK(summary.line_items[0].item_id == "E-20" && summary.line_items[0].violations.size() == 1);
        CHECK(summary.line_items[3].item_id == "E-23" && summary.line_items[3].violations.size() == 1);
        std::cout << "  a 4-item report totals $" << expected_claimed << " claimed against $"
                   << expected_approved << " approved -- the $90 meal claim's full $90 still counts "
                     "toward the claimed total even though only $75 is approved, and the excluded $30 "
                     "alcohol item still counts toward the claimed total even though $0 is approved -- "
                     "the report's own claimed figure is never silently reduced, only the approved "
                     "figure and the named violation tell the real story\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_expense_report_policy_enforcement.cpp -o 03_expense_report_policy_enforcement
./03_expense_report_policy_enforcement
```

**Sample input:** a meal claim under, exactly at, and over its own real daily cap; lodging claims checked against all 3 real per-city-tier caps, with an over-cap claim in the cheapest tier flagged against its own tier's cap; a mileage claim matching and mismatching its own computed rate-based figure; an alcohol claim checked to approve $0.00 regardless of amount and never additionally trigger a receipt-required flag; the receipt-required threshold checked exactly at its own real boundary; and a full 4-item report checked to compute exact claimed and approved totals while attributing every violation to its own specific line item.

```text
========================================================
Chapter 23.3: An Automated Expense-Report Policy Engine
========================================================

-- Test 1: a meal claim under and exactly at the real daily cap is fully approved with zero violations; one over the cap is flagged with the exact computed overage --
  a $50.00 and an exactly-$75.00 meal claim are both fully approved with zero violations; a $90.00 claim is capped to $75.00 approved and flagged with the exact $15.00 overage

-- Test 2: lodging's own real per-city-tier cap is applied correctly for all 3 tiers, and an over-cap claim in the cheapest tier is still flagged against ITS OWN tier's cap, not a different tier's --
  a $340 tier-1 stay and a $190 tier-2 stay both pass under their own real caps ($350 and $200); a $150 tier-3 stay is capped to $120.00 approved and flagged with the exact $30.00 overage against ITS OWN tier-3 cap

-- Test 3: a mileage claim matching the real rate exactly is approved with zero violations; a claim that does not match its own computed value is flagged naming both figures --
  100 miles at the real $0.67/mile rate computes to exactly $67.00: a claim of $67.00 passes with zero violations, while a claim of $70.00 for the same 100 miles is approved at the correctly computed $67.00 and flagged, naming both the claimed and the expected figure

-- Test 4: an alcohol line item is approved at $0.00 regardless of its claimed amount, and never triggers the receipt-required check --
  a $45.00 alcohol claim with no receipt on file is approved at exactly $0.00, flagged only for ALCOHOL_NOT_REIMBURSABLE -- never additionally flagged for a missing receipt, since a fully excluded category has nothing left to require one for

-- Test 5: the receipt-required threshold is exact at its own real boundary -- claims at or below it never require a receipt, and claims above it do --
  a $25.00 office-supply claim with no receipt requires none and is approved with zero violations; a $25.01 claim with no receipt is flagged MISSING_REQUIRED_RECEIPT

-- Test 6: a full multi-item report computes its own claimed and approved totals as exact sums, and attributes every violation to its own specific line item, never silently dropping a flagged item's own claimed contribution to the total --
  a 4-item report totals $527 claimed against $482 approved -- the $90 meal claim's full $90 still counts toward the claimed total even though only $75 is approved, and the excluded $30 alcohol item still counts toward the claimed total even though $0 is approved -- the report's own claimed figure is never silently reduced, only the approved figure and the named violation tell the real story

32/32 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a policy-approved total as a corrected, replacement version of the claimed total"
    Once an expense engine has computed a real, policy-compliant APPROVED figure for a line item, it is tempting to treat that figure as simply the CORRECTED version of what the employee claimed, and report only it going forward -- after all, the engine has already done the real work of figuring out what should actually be reimbursed. Test 6 is built specifically to rule that shortcut out: the report's own claimed total of $527 and its own approved total of $482 are both reported, side by side, for the same 4 items, with the $45 gap attributed by name to two specific line items' own specific violations. Collapsing to just the $482 approved figure would discard exactly the information an expense auditor, or the employee themselves, needs to understand WHY their own reimbursement came in lower than what they submitted -- the same discipline this book has applied to every flagged discrepancy since Section 21.3's own claim-consistency check.

## 23.4 A Real-Time Luxury-Goods Authentication Engine for Consignment and Resale Counters

### Intuition

A consignment counter's real-time screen cannot replace a trained human authenticator, but it can check three real, computable structural properties against a brand's own stated specification before that human ever looks at the item: whether a serial number's own digit portion satisfies a real, independently famous check-digit algorithm, whether a measured weight falls within a real stated spec range, and whether a hardware finish is among a real stated valid list.

### The Concept, In Detail

`luhn_valid` is the real, well-known Luhn algorithm -- the same algorithm that validates credit-card numbers -- confirmed in Test 1 against its own famous exact reference value (`79927398713`) and a second independent real 16-digit reference number, with a single altered digit breaking the checksum in both cases. `validate_serial` checks this section's own stated brand serial format -- a 2-letter factory code plus an 11-digit Luhn-valid number -- structurally, and Test 2 confirms a wrong length, a lowercase factory code, and a non-digit character are each flagged by their own specific, named reason, distinct from Test 3's confirmation that a structurally valid serial can still fail specifically on its own checksum.

`authenticate_item`'s own Test 5 is this section's central honesty check, reapplying this book's own structural-override discipline one more time: an item with a perfectly in-spec weight and a perfectly recognized hardware finish is still `REJECTED_INVALID_SERIAL` outright the moment its own serial checksum fails -- two passing checks never outvote one hard-failing one. Test 7 confirms the same discipline in a softer form: an unrecognized hardware finish is `FLAGGED_FOR_EXPERT_REVIEW`, never silently passed, even when the serial and weight both look fine. And Test 4's own passing case is named, deliberately, `PASSES_STRUCTURAL_SCREENING` rather than "authentic" -- an honest, narrow claim whose own reason text states plainly that final authentication remains a human expert's own call.

### Code and Verification

```cpp
// Chapter 23.4 -- A consignment counter's own real-time authentication
// screen cannot inspect an item the way a trained human expert eventually
// will, but it CAN check three real, computable structural properties
// against a brand's own stated specification: whether a serial number's
// own digit portion satisfies a real, well-known check-digit algorithm
// (the Luhn algorithm, the same real algorithm that validates credit-card
// numbers), whether a measured weight falls within a real stated spec
// range, and whether a hardware finish is among a real stated valid list.
// This section's own central honesty discipline, consistent with every
// structural-override check since Chapter 20: a single hard-failing
// check is NEVER outvoted by two other checks that happen to pass.
//
// A note on this section's own honest scope: `authenticate_item` never
// returns a verdict named "AUTHENTIC" anywhere in this file. Its best
// possible outcome is named `PASSES_STRUCTURAL_SCREENING` -- an honest,
// narrow claim that three specific, real, checkable properties held, not
// a claim that a trained human expert's own final authentication is no
// longer needed, exactly the same honest-scope discipline Section 22.3's
// provenance validator applied to its own chronological consistency
// checks.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_luxury_goods_authentication_engine.cpp -o 04_luxury_goods_authentication_engine
// Run:     ./04_luxury_goods_authentication_engine

#include <algorithm>
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
// PART 1: the real, well-known Luhn check-digit algorithm, the same
// algorithm that validates credit-card numbers -- this section's own
// stated brand uses it as its own serial-number check-digit scheme.
// =======================================================================
bool luhn_valid(const std::string& digits) {
    int sum = 0;
    bool double_it = false;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        int d = *it - '0';
        if (double_it) {
            d *= 2;
            if (d > 9) d -= 9;
        }
        sum += d;
        double_it = !double_it;
    }
    return sum % 10 == 0;
}

// =======================================================================
// PART 2: this section's own stated serial-number format -- 2 uppercase
// factory-code letters followed by an 11-digit Luhn-valid number -- and
// its structural validator.
// =======================================================================
constexpr int SERIAL_PREFIX_LEN = 2;
constexpr int SERIAL_DIGITS_LEN = 11;
constexpr int SERIAL_TOTAL_LEN = SERIAL_PREFIX_LEN + SERIAL_DIGITS_LEN;

bool is_upper_alpha(char c) { return c >= 'A' && c <= 'Z'; }
bool is_ascii_digit(char c) { return c >= '0' && c <= '9'; }

struct SerialValidationResult {
    bool valid_format = false;
    bool valid_checksum = false;
    std::string reason;
};

SerialValidationResult validate_serial(const std::string& serial) {
    SerialValidationResult r;
    if (static_cast<int>(serial.size()) != SERIAL_TOTAL_LEN) {
        r.reason = "serial length " + std::to_string(serial.size()) + " does not match the required " +
                   std::to_string(SERIAL_TOTAL_LEN) + " characters";
        return r;
    }
    for (int i = 0; i < SERIAL_PREFIX_LEN; i++) {
        if (!is_upper_alpha(serial[static_cast<std::size_t>(i)])) {
            r.reason = "serial's own first " + std::to_string(SERIAL_PREFIX_LEN) +
                       " characters must be uppercase factory-code letters";
            return r;
        }
    }
    std::string digits = serial.substr(SERIAL_PREFIX_LEN);
    for (char c : digits) {
        if (!is_ascii_digit(c)) {
            r.reason = "serial's own digit portion contains a non-digit character";
            return r;
        }
    }
    r.valid_format = true;
    r.valid_checksum = luhn_valid(digits);
    if (!r.valid_checksum) {
        r.reason = "serial's own digit portion fails the brand's stated Luhn check-digit scheme";
    }
    return r;
}

// =======================================================================
// PART 3: the hardware spec, the item submission, and the combined
// authentication engine.
// =======================================================================
struct HardwareSpec {
    double weight_grams_min = 0.0, weight_grams_max = 0.0;
    std::vector<std::string> valid_finishes;
};

struct ItemSubmission {
    std::string item_id;
    std::string serial_number;
    double measured_weight_grams = 0.0;
    std::string hardware_finish;
};

enum class AuthenticationVerdict {
    REJECTED_INVALID_SERIAL,
    REJECTED_WEIGHT_OUT_OF_RANGE,
    FLAGGED_FOR_EXPERT_REVIEW,
    PASSES_STRUCTURAL_SCREENING,
};

struct AuthenticationResult {
    AuthenticationVerdict verdict;
    std::string reason;
};

AuthenticationResult authenticate_item(const ItemSubmission& item, const HardwareSpec& spec) {
    auto serial_result = validate_serial(item.serial_number);
    if (!serial_result.valid_format || !serial_result.valid_checksum) {
        return {AuthenticationVerdict::REJECTED_INVALID_SERIAL, serial_result.reason};
    }

    if (item.measured_weight_grams < spec.weight_grams_min || item.measured_weight_grams > spec.weight_grams_max) {
        return {AuthenticationVerdict::REJECTED_WEIGHT_OUT_OF_RANGE,
                "measured weight " + std::to_string(item.measured_weight_grams) +
                    "g is outside the real spec range [" + std::to_string(spec.weight_grams_min) + "g, " +
                    std::to_string(spec.weight_grams_max) + "g]"};
    }

    bool finish_valid = std::find(spec.valid_finishes.begin(), spec.valid_finishes.end(), item.hardware_finish) !=
                         spec.valid_finishes.end();
    if (!finish_valid) {
        return {AuthenticationVerdict::FLAGGED_FOR_EXPERT_REVIEW,
                "hardware finish \"" + item.hardware_finish +
                    "\" is not among this item's own real, stated valid finishes"};
    }

    return {AuthenticationVerdict::PASSES_STRUCTURAL_SCREENING,
            "serial checksum valid, weight in spec, hardware finish recognized -- passes this counter's "
            "own structural screen; final authentication remains a human expert's own call"};
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 23.4: A Real-Time Luxury-Goods Authentication Engine\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real Luhn algorithm matches its own famous exact reference value "
                 "(79927398713 is Luhn-valid), and a single altered digit breaks it --\n";
    {
        CHECK(luhn_valid("79927398713"));
        CHECK(!luhn_valid("79927398714"));
        CHECK(luhn_valid("4532015112830366"));
        CHECK(!luhn_valid("4532015112830367"));
        std::cout << "  \"79927398713,\" the Luhn algorithm's own famous reference test number, validates "
                     "exactly as expected; changing its own last digit by 1 breaks the checksum; the "
                     "same holds for a second, independent real 16-digit Luhn-valid reference number\n";
    }

    std::cout << "\n-- Test 2: this section's own stated serial format is checked structurally -- wrong "
                 "length, a lowercase factory code, and a non-digit in the numeric portion are each "
                 "flagged by their own specific, named reason --\n";
    {
        auto wrong_length = validate_serial("FR7992739871");
        auto lowercase_prefix = validate_serial("fr79927398713");
        auto non_digit = validate_serial("FR7992739871A");
        CHECK(!wrong_length.valid_format);
        CHECK(wrong_length.reason.find("length") != std::string::npos);
        CHECK(!lowercase_prefix.valid_format);
        CHECK(lowercase_prefix.reason.find("uppercase") != std::string::npos);
        CHECK(!non_digit.valid_format);
        CHECK(non_digit.reason.find("non-digit") != std::string::npos);
        std::cout << "  a 12-character serial (one short of the required 13) is flagged for its own "
                     "exact length; a lowercase factory-code prefix is flagged by name; a numeric "
                     "portion containing a letter is flagged by name -- each a specific, different "
                     "structural reason\n";
    }

    std::cout << "\n-- Test 3: a serial with valid structural format but a failing checksum is flagged "
                 "distinctly from a structurally invalid one --\n";
    {
        auto valid = validate_serial("FR79927398713");
        auto bad_checksum = validate_serial("FR79927398714");
        CHECK(valid.valid_format && valid.valid_checksum);
        CHECK(bad_checksum.valid_format && !bad_checksum.valid_checksum);
        CHECK(bad_checksum.reason.find("Luhn") != std::string::npos);
        std::cout << "  \"FR79927398713\" passes both structural format and checksum; "
                     "\"FR79927398714\" passes the structural format (right length, right character "
                     "classes) but fails the Luhn checksum specifically, named by its own reason\n";
    }

    std::cout << "\n-- Test 4: an item with a fully valid serial, weight, and hardware finish passes "
                 "structural screening -- an honest, narrow claim, never named \"authentic\" --\n";
    {
        HardwareSpec spec{180.0, 220.0, {"Gold-Tone", "Silver-Tone", "Gunmetal"}};
        ItemSubmission item{"ITEM-1", "FR79927398713", 200.0, "Gold-Tone"};
        auto result = authenticate_item(item, spec);
        CHECK(result.verdict == AuthenticationVerdict::PASSES_STRUCTURAL_SCREENING);
        CHECK(result.reason.find("human expert") != std::string::npos);
        std::cout << "  a valid serial, a 200g weight inside the real [180g, 220g] spec, and a "
                     "recognized \"Gold-Tone\" finish together produce PASSES_STRUCTURAL_SCREENING -- "
                     "whose own reason text explicitly states final authentication remains a human "
                     "expert's own call\n";
    }

    std::cout << "\n-- Test 5: an invalid serial checksum alone forces REJECTED_INVALID_SERIAL even when "
                 "weight and hardware finish both pass -- one hard-failing check is never outvoted by "
                 "two passing ones --\n";
    {
        HardwareSpec spec{180.0, 220.0, {"Gold-Tone", "Silver-Tone", "Gunmetal"}};
        ItemSubmission item{"ITEM-2", "FR79927398714", 200.0, "Gold-Tone"};
        auto result = authenticate_item(item, spec);
        CHECK(result.verdict == AuthenticationVerdict::REJECTED_INVALID_SERIAL);
        std::cout << "  the identical 200g \"Gold-Tone\" item, with only its own serial's checksum "
                     "digit altered, is REJECTED_INVALID_SERIAL outright -- a perfectly in-spec weight "
                     "and a perfectly recognized finish never override one specific failing checksum\n";
    }

    std::cout << "\n-- Test 6: a weight outside the real spec range is rejected with the exact range "
                 "named, even with a fully valid serial --\n";
    {
        HardwareSpec spec{180.0, 220.0, {"Gold-Tone", "Silver-Tone", "Gunmetal"}};
        ItemSubmission item{"ITEM-3", "FR79927398713", 260.0, "Gold-Tone"};
        auto result = authenticate_item(item, spec);
        CHECK(result.verdict == AuthenticationVerdict::REJECTED_WEIGHT_OUT_OF_RANGE);
        CHECK(result.reason.find("180") != std::string::npos && result.reason.find("220") != std::string::npos);
        std::cout << "  a 260g item with an otherwise perfectly valid serial is rejected, naming the "
                     "real [180g, 220g] spec range it falls outside of\n";
    }

    std::cout << "\n-- Test 7: an unrecognized hardware finish is flagged for expert review, not silently "
                 "passed, even when the serial and weight both look fine --\n";
    {
        HardwareSpec spec{180.0, 220.0, {"Gold-Tone", "Silver-Tone", "Gunmetal"}};
        ItemSubmission item{"ITEM-4", "FR79927398713", 200.0, "Rose-Gold-Tone"};
        auto result = authenticate_item(item, spec);
        CHECK(result.verdict == AuthenticationVerdict::FLAGGED_FOR_EXPERT_REVIEW);
        CHECK(result.reason.find("Rose-Gold-Tone") != std::string::npos);
        std::cout << "  a valid serial and an in-spec 200g weight both look fine, but \"Rose-Gold-Tone\" "
                     "is not among this item's own 3 stated valid finishes, so the item is flagged for "
                     "expert review rather than silently passed on the strength of its other 2 checks\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_luxury_goods_authentication_engine.cpp -o 04_luxury_goods_authentication_engine
./04_luxury_goods_authentication_engine
```

**Sample input:** the real Luhn algorithm checked against its own famous reference number and a second independent reference number, each with a single altered digit breaking the checksum; this section's own stated serial format checked structurally against a wrong length, a lowercase factory code, and a non-digit numeric portion, each flagged by its own specific reason; a structurally valid serial with a failing checksum checked distinctly from a fully invalid one; a fully valid item checked to pass structural screening with an explicit human-expert caveat in its own reason text; an invalid checksum checked to force rejection even with a perfectly in-spec weight and finish; an out-of-range weight checked to be rejected with the exact real spec range named; and an unrecognized hardware finish checked to be flagged for expert review rather than silently passed.

```text
========================================================
Chapter 23.4: A Real-Time Luxury-Goods Authentication Engine
========================================================

-- Test 1: the real Luhn algorithm matches its own famous exact reference value (79927398713 is Luhn-valid), and a single altered digit breaks it --
  "79927398713," the Luhn algorithm's own famous reference test number, validates exactly as expected; changing its own last digit by 1 breaks the checksum; the same holds for a second, independent real 16-digit Luhn-valid reference number

-- Test 2: this section's own stated serial format is checked structurally -- wrong length, a lowercase factory code, and a non-digit in the numeric portion are each flagged by their own specific, named reason --
  a 12-character serial (one short of the required 13) is flagged for its own exact length; a lowercase factory-code prefix is flagged by name; a numeric portion containing a letter is flagged by name -- each a specific, different structural reason

-- Test 3: a serial with valid structural format but a failing checksum is flagged distinctly from a structurally invalid one --
  "FR79927398713" passes both structural format and checksum; "FR79927398714" passes the structural format (right length, right character classes) but fails the Luhn checksum specifically, named by its own reason

-- Test 4: an item with a fully valid serial, weight, and hardware finish passes structural screening -- an honest, narrow claim, never named "authentic" --
  a valid serial, a 200g weight inside the real [180g, 220g] spec, and a recognized "Gold-Tone" finish together produce PASSES_STRUCTURAL_SCREENING -- whose own reason text explicitly states final authentication remains a human expert's own call

-- Test 5: an invalid serial checksum alone forces REJECTED_INVALID_SERIAL even when weight and hardware finish both pass -- one hard-failing check is never outvoted by two passing ones --
  the identical 200g "Gold-Tone" item, with only its own serial's checksum digit altered, is REJECTED_INVALID_SERIAL outright -- a perfectly in-spec weight and a perfectly recognized finish never override one specific failing checksum

-- Test 6: a weight outside the real spec range is rejected with the exact range named, even with a fully valid serial --
  a 260g item with an otherwise perfectly valid serial is rejected, naming the real [180g, 220g] spec range it falls outside of

-- Test 7: an unrecognized hardware finish is flagged for expert review, not silently passed, even when the serial and weight both look fine --
  a valid serial and an in-spec 200g weight both look fine, but "Rose-Gold-Tone" is not among this item's own 3 stated valid finishes, so the item is flagged for expert review rather than silently passed on the strength of its other 2 checks

20/20 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating PASSES_STRUCTURAL_SCREENING as equivalent to a finished authentication"
    `PASSES_STRUCTURAL_SCREENING` is the best possible outcome `authenticate_item` can ever return, and it is tempting, at a busy consignment counter, to treat that outcome as the END of the authentication process rather than its real, useful FIRST step. This section's own file never returns a verdict named "AUTHENTIC" anywhere, and Test 4's own passing reason text says so explicitly: final authentication remains a human expert's own call. The three checks this engine actually runs -- a checksum algorithm, a weight range, and a finish name lookup -- are real and genuinely useful for catching an obviously wrong item quickly, but they say nothing about the countless other real properties (stitching, hardware plating thickness, material grain, a serial number that is itself a cloned, valid-looking fake) a trained human authenticator would also check. Treating a structural pass as a finished verdict would grant this engine an authority over a real, consequential resale decision that its own three checks were never built to support.

## Chapter Summary

This chapter built four real, independently well-known algorithms -- a perceptual difference hash, the classic Levenshtein edit distance, a real corporate expense-policy rules engine, and the Luhn check-digit algorithm -- into four domains united by trust and authenticity at the point of a real transaction. Section 23.1 built a counterfeit-screening engine and a fully auditable seller trust score, proving by direct construction that a perfect aggregate trust score can never suppress a single specific, named red flag. Section 23.2 built a receipt-to-transaction reconciliation engine that honestly reports AMBIGUOUS, naming every fitting transaction, rather than silently guessing when the underlying evidence genuinely supports more than one answer. Section 23.3 built an expense-policy engine that reports a claimed figure and a policy-approved figure side by side for every line item, never silently substituting one for the other without naming the exact rule behind the gap. Section 23.4 built a real-time luxury-goods screening engine whose own best possible outcome is honestly named a structural pass, not an authentication verdict, and proved that one hard-failing check is never outvoted by two others that happen to pass.

## Self-Check Questions

1. Section 23.1's Test 7 uses a seller with a PERFECT trust score of 100.0, not merely a high one. Explain why using the maximum possible score, rather than a merely above-average one, makes this test's own point more convincingly than a smaller number would.
2. Section 23.1's `screen_listing` compares a listing's own photo hash against a registry of OTHER sellers' photos, explicitly excluding the same seller's own prior listings. Explain why comparing against the SAME seller's own past listings would not be a useful counterfeit signal, even though it would still find hash matches.
3. Section 23.2's `merchant_names_fuzzy_match` is described as tuned for OCR noise specifically, not payment-processor descriptor mangling. Using the "SQ *STARBUCKS COFFEE" example from this section's own COMMON TRAP box, explain concretely why an edit-distance-ratio threshold tuned for a single misread character would fail on that different kind of mismatch.
4. Section 23.2's Test 6 constructs an AMBIGUOUS case using two transactions with the SAME claimed merchant name and a plausible tip-adjusted amount. Construct a different, equally realistic pair of transactions that would ALSO produce an AMBIGUOUS result under this section's own real matching rules.
5. Section 23.3's mileage check computes an EXPECTED reimbursement from miles times a real per-mile rate, rather than simply capping whatever amount was claimed the way the meals and lodging checks do. Explain why a cap alone would be insufficient for mileage specifically.
6. Section 23.3's alcohol exclusion never additionally triggers the receipt-required check. Explain why applying the receipt-required check on top of the alcohol exclusion would not provide any additional real information to a reviewer.
7. Section 23.3's Test 6 reports a $527 claimed total against a $482 approved total for the same 4-item report. What decision does this design deliberately leave to a human reviewer rather than making automatically, and how does reporting both totals support that decision better than reporting only the approved total would?
8. Section 23.4's `validate_serial` returns `valid_format` and `valid_checksum` as two SEPARATE boolean fields rather than a single combined `is_valid` boolean. What real diagnostic information would collapsing them into one field lose, and construct a serial number that demonstrates the difference.
9. Section 23.4's Test 5 rejects an item purely because of its own serial checksum, despite a perfectly in-spec weight and a perfectly recognized finish. Explain concretely why a real counterfeiter is more likely to get a serial checksum wrong than to get a hardware finish or weight wrong, and what that implies about which checks deserve override authority over the others.
10. Across Sections 23.1 and 23.4, a single named red flag or a single hard-failing check always overrides an otherwise-favorable aggregate signal. Identify the one place in Section 23.3 where the SAME override discipline appears in a financial-policy context rather than a fraud-detection one, and explain why it serves the identical real purpose there.

## Where We Go Next

This chapter's four sections showed the same recurring discipline -- an honest ambiguity, a named red flag that survives a favorable aggregate score, and a structural pass that never overstates its own real scope -- generalizing once more, this time into commerce and financial trust. Chapter 24 turns to a different real problem entirely: natural language photo editing, building an edit-interpretation prompt that turns a vague instruction like "make it look better" into a structured edit plan, a complete OpenCV processing engine that executes that plan as real `cv::Mat` operations, a request-to-operation mapping table, and iterative refinement through conversation.

## Worked Solutions

**1.** A merely above-average trust score could plausibly be explained away as "not quite perfect, so some risk was already priced in" -- a reviewer might reasonably wonder whether the review requirement was really about the specific keyword flag or just about the score not being high enough on its own. Using the literal maximum possible score of 100.0 eliminates that ambiguity entirely: there is no higher score this seller could have achieved, no further accumulated history that could have helped their case, and review is STILL required -- proving conclusively that the review trigger came from the specific named flag itself, not from any residual doubt the aggregate score left unaddressed.

**2.** A seller's own past listings for the identical product line would naturally, legitimately share the same or very similar product photography -- a seller who genuinely owns 5 identical items for sale would reasonably photograph them with the same setup, lighting, and even the same physical unit across several listings, none of which indicates counterfeiting. The real, useful signal this section's own check targets is a photo appearing under a DIFFERENT seller's own account -- since two independent, unrelated sellers each independently and legitimately owning and photographing the exact same item down to the pixel is essentially never how genuine product photography actually happens, while stolen stock imagery reused across accounts is a real, documented pattern.

**3.** A single misread character changes only one or two positions in an otherwise-identical string, keeping the edit-distance ratio very small (Section 23.2's own "WALGREEN5" example scores 1/9). "SQ *STARBUCKS COFFEE" against "Starbucks," by contrast, differs by an entirely prepended processor code ("SQ *"), a different capitalization convention, and an appended descriptor word ("COFFEE") not present in the receipt's own name at all -- these are real STRUCTURAL insertions spanning several characters each, not isolated single-character substitutions, which would push the edit distance (and therefore the ratio) well past any threshold tuned for a single misread digit or letter, causing a genuinely correct match to be wrongly rejected as a mismatch.

**4.** A $12.00 receipt for a merchant with two locations that settle their own card transactions under the identical merchant name and both process transactions dated the same real day (for instance, two branches of the same chain both settling on day 300 for the customer's own two separate real purchases that day) would produce the identical situation Test 6 constructs: two transactions, same merchant name, same real amount, both within the settlement window -- and `reconcile_receipt` would correctly report AMBIGUOUS rather than guessing which real branch's charge belongs to which real receipt.

**5.** A cap alone can only catch a claim that is TOO HIGH relative to some maximum -- it has no way to catch a claim that is simply WRONG in either direction relative to what the underlying miles actually compute to, including a claim that is too LOW (understating miles driven, whether by simple arithmetic error or by other real cause) which a cap would never flag at all since it falls under any reasonable maximum. Computing the expected reimbursement directly from the real inputs (miles times rate) and comparing the claim against that specific expected value, rather than against an unrelated upper bound, is what allows this section's own mileage check to catch a mismatch in either direction, not merely an excessive one.

**6.** Once an item is excluded from reimbursement entirely at $0.00 approved regardless of its own claimed amount, telling a reviewer that it ALSO lacks a required receipt would be reporting a compliance requirement for a category of spending that is never eligible for reimbursement under any circumstances in the first place -- there is nothing further a receipt could unlock or justify for an item that is already fully excluded by category, so flagging its own missing receipt on top of the exclusion would only add noise without adding any real, actionable information a reviewer could use.

**7.** This design deliberately leaves to a human reviewer the decision of what to actually communicate back to the employee and whether any of the flagged discrepancies warrant further discussion (a genuine miscalculation the employee should be informed of, a policy the employee may not have been aware of, or a pattern worth raising with them directly) rather than having the system silently reimburse a lower amount with no visible explanation. Reporting both the $527 claimed and $482 approved totals together, with each of the $45 gap's own contributing violations named by item, lets that reviewer see exactly what changed and why, rather than presenting a single final number that would leave the employee (or the reviewer themselves, weeks later) unable to reconstruct where the difference came from.

**8.** A single combined `is_valid` boolean would tell a reviewer only that SOMETHING about the serial is wrong, with no way to distinguish a serial that is structurally malformed (wrong length, wrong character classes -- likely a data-entry error, a different product line entirely, or a very unsophisticated fake) from a serial that is structurally perfect but fails its own checksum (a much more specific and concerning signal, since it means someone constructed a plausible-LOOKING serial that does not actually satisfy the brand's own real check-digit scheme). "FR79927398714" demonstrates this exactly: it has the correct length, the correct letter prefix, and an all-digit numeric portion (so `valid_format` is true), yet its own checksum fails (`valid_checksum` is false) -- a single combined boolean would report this identically to a serial that was simply the wrong length, discarding the real distinction between "malformed" and "well-formed but fraudulent."

**9.** A hardware finish name and a target weight range are both properties a counterfeiter can directly observe from a genuine item (by weighing it, or by reading the finish name off an authentic tag) and then simply replicate or match closely, since neither number is secret or derived from anything hidden. A valid Luhn checksum, by contrast, requires knowing the SPECIFIC mathematical relationship between a serial's own digits and its own final check digit -- a counterfeiter fabricating a plausible-looking serial number by guessing or copying a DIFFERENT genuine item's own format would need to either reuse an already-issued real serial (a separate, detectable problem) or correctly compute a valid checksum for a new one, which requires knowing the checksum algorithm itself, not just observing a physical property. This is exactly why a checksum failure deserves override authority over the other two checks: it is evidence of a structural relationship being violated, not merely a measurement falling outside a range that could plausibly result from ordinary manufacturing variance.

**10.** The identical override discipline appears in Section 23.3's alcohol-exclusion rule: no matter how reasonable, well-documented, or within-cap an alcohol line item's own claimed amount might otherwise look, `evaluate_line_item` approves it at exactly $0.00 every single time, with no code path anywhere in the function that lets a low claimed amount, a valid receipt, or any other favorable property override that exclusion. It serves the identical real purpose as Section 23.1's trust-score override and Section 23.4's checksum override: a single, specific, categorical policy fact (this category is never reimbursable; this specific keyword appeared; this specific checksum failed) is treated as authoritative and non-negotiable, regardless of how favorable every OTHER signal about the same claim, listing, or item happens to be.
