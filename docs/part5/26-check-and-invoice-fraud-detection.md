# Chapter 26: Check and Invoice Fraud Detection at Bank Branches and Back Offices

**What you will understand by the end of this chapter:**

- How to implement the real American Bankers Association routing-number checksum -- the identical published formula every US bank's own check-clearing system runs on every check -- and reapply Chapter 23.1's own real perceptual difference-hash (dHash) algorithm to signature comparison instead of product photography.
- How to reapply the classic Levenshtein edit-distance algorithm, built from scratch in Chapter 23.2, to a different real fraud pattern: an invoice number altered by a single character specifically to dodge a naive exact-match duplicate-payment check.
- How to build a real, from-scratch English number-word parser that reconstructs the numeric value a check's own "amount in words" field actually claims, and cross-validates it against the numerals field -- the real redundancy a US check's own printed form exists to exploit against alteration.
- How to combine multiple independent real fraud signals into a single, fully auditable risk score and a three-way disposition, reapplying this book's own recurring discipline that one specific, high-severity red flag can force escalation on its own, regardless of how clean every other signal looks.

**What you need to know first:**

- Section 23.1's own real, from-scratch dHash implementation (resize to 9x8, encode 64 horizontal pixel comparisons into a 64-bit value, compare by Hamming distance) is reapplied unchanged in Section 26.1, this time to a signature image rather than a product photo.
- Section 23.2's own real, from-scratch Levenshtein edit-distance algorithm is reapplied unchanged in Section 26.2, this time to detect a near-identical invoice number rather than fuzzy-match an OCR-noisy merchant name.
- This book's own recurring structural-override discipline, most recently Chapter 23's own never-suppress-a-named-flag rule: one specific, high-severity signal (Section 26.4's own MICR checksum failure) forces at least a human review, regardless of how low every other signal's own score is.

---

Chapter 25 closed Part 5's run of consumer- and field-facing vision-language deployments. This chapter turns to a different real financial setting: a bank branch or back office processing checks and invoices, where the fraud patterns are well-documented, real, and -- unlike a counterfeit product photo -- often checkable by a real published formula or a real structural redundancy the payment instrument's own printed form was designed to provide. Each of this chapter's four sections reapplies an algorithm this book already built from scratch (dHash, Levenshtein) to a new real domain, or builds a new real algorithm (a routing-number checksum, a words-to-number parser) with its own well-established provenance outside this book, and closes with a capstone risk-scoring engine that combines all of them into one fully auditable disposition.

## 26.1 MICR Checksum Validation and Signature Comparison

### Intuition

A check's own MICR line is not just an identifier -- its routing number carries a real, published checksum, the same formula every US bank's own check-clearing system runs on every check that passes through it. A forged or badly transcribed routing number will very often fail this checksum outright, catching a structural problem before any human ever reads the check. Separately, a check's own payee signature can be compared against a signature on file using the identical real perceptual hash this book already built in Chapter 23.1 for a completely different purpose.

### The Concept, In Detail

`micr_checksum_valid` implements the real, published ABA routing-number formula -- `3*(d1+d4+d7) + 7*(d2+d5+d8) + 1*(d3+d6+d9)` must be a multiple of 10 -- confirmed in Test 1 against a real, publicly known routing number (JPMorgan Chase's own "021000021") and in Test 2 against the identical number with its own last digit altered, which correctly breaks the checksum. Test 3 confirms a malformed routing number (wrong length, or containing a non-digit) is rejected outright rather than silently truncated or padded.

`compute_dhash` and `hamming_distance` are Chapter 23.1's own unmodified implementations, applied here to a signature image. Test 4 confirms a pixel-identical re-scan hashes identically (Hamming distance 0); Test 5 confirms the same physical signature, re-scanned with small, realistic sensor noise, still matches under a real, stated tolerance; and Test 6 confirms a visibly different image -- standing in for a forged or substituted signature, the identical honest use of a starkly different synthetic test image Chapter 23.1's own dHash test used -- is correctly flagged as NOT matching, proving the noise tolerance does not silently absorb a genuinely different signature too.

### Code and Verification

```cpp
// Chapter 26.1 -- A check image arriving at a bank's back office carries two
// independently checkable real signals before any human ever looks at it:
// the MICR line's own routing number, which is not just an identifier but
// a real, published checksum -- the American Bankers Association's own
// weighted-digit formula, the same one every US bank's own check-clearing
// system runs on every single check -- and the payee's signature, which
// this section compares against a reference signature on file using the
// identical real perceptual "difference hash" (dHash) algorithm Chapter
// 23.1 built for counterfeit product photos, applied here to a different
// real domain. Both checks are real, independently verifiable properties
// of the check image itself, not a verdict about whether the check is
// genuine -- exactly the same honest-scope discipline Chapter 23.1's own
// listing screening applied to counterfeit detection.
//
// A note on this section's own honest scope: `micr_checksum_valid` proves
// a routing number is INTERNALLY CONSISTENT (its own digits satisfy the
// ABA's published formula) -- it does NOT prove the routing number
// belongs to a real, currently-operating bank, which would require a
// live lookup against the Federal Reserve's own routing-number registry,
// a real network dependency this book's own offline, deterministic
// discipline deliberately does not take on.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_micr_checksum_and_signature_comparison.cpp -o 01_micr_checksum_and_signature_comparison
// Run:     ./01_micr_checksum_and_signature_comparison

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
// PART 1: the real ABA routing-number checksum. A 9-digit routing number
// d1 d2 ... d9 is internally consistent exactly when:
//     3*(d1+d4+d7) + 7*(d2+d5+d8) + 1*(d3+d6+d9) is a multiple of 10.
// This is the actual, published formula every US bank's own check-
// clearing system runs -- not a simplification invented for this book.
// =======================================================================
bool micr_checksum_valid(const std::string& routing9) {
    if (routing9.size() != 9) return false;
    for (char c : routing9) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    auto d = [&](int i) { return routing9[static_cast<std::size_t>(i)] - '0'; };
    int weighted = 3 * (d(0) + d(3) + d(6)) + 7 * (d(1) + d(4) + d(7)) + 1 * (d(2) + d(5) + d(8));
    return weighted % 10 == 0;
}

// =======================================================================
// PART 2: the real, from-scratch perceptual "difference hash" (dHash)
// algorithm, identical to Chapter 23.1's own implementation, applied here
// to a signature image instead of a product photo.
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

// A real, stated policy threshold: two signature scans of the same real
// pen stroke, captured on different scanners with different lighting and
// paper skew, still land within this many differing bits out of 64 --
// the identical real tolerance dHash-based deduplication systems use for
// "the same image, re-encoded," not "a pixel-identical file."
constexpr int SIGNATURE_MATCH_MAX_HAMMING = 10;

bool signatures_match(const GrayscaleImage& reference, const GrayscaleImage& candidate) {
    return hamming_distance(compute_dhash(reference), compute_dhash(candidate)) <= SIGNATURE_MATCH_MAX_HAMMING;
}

// Three small, hand-constructed 40x24 "signature" images: a genuine
// signature stroke, the identical stroke re-scanned with mild sensor
// noise added to a handful of pixels (simulating a second real scan of
// the same physical signature), and a visibly different stroke pattern
// standing in for a forged or substituted signature.
GrayscaleImage make_stroke(int w, int h, int stroke_row_start, int stroke_row_end, bool diagonal) {
    GrayscaleImage img{w, h, std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h, 235)};
    for (int y = stroke_row_start; y < stroke_row_end; y++) {
        for (int x = 0; x < w; x++) {
            int center = diagonal ? (y - stroke_row_start) * w / std::max(1, stroke_row_end - stroke_row_start)
                                   : w / 2;
            int dist = std::abs(x - center);
            if (dist < 3) {
                img.pixels[static_cast<std::size_t>(y) * w + x] = static_cast<std::uint8_t>(20 + dist * 5);
            }
        }
    }
    return img;
}

GrayscaleImage make_checker(int w, int h, int cell) {
    GrayscaleImage img{w, h, std::vector<std::uint8_t>(static_cast<std::size_t>(w) * h)};
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img.pixels[static_cast<std::size_t>(y) * w + x] =
                (((x / cell) + (y / cell)) % 2 == 0) ? 220 : 30;
    return img;
}

GrayscaleImage add_scan_noise(GrayscaleImage img, int seed) {
    // A small, deterministic pseudo-random perturbation of a handful of
    // pixels -- standing in for real scanner sensor noise, not an actual
    // randomized test (this book's own determinism discipline: the same
    // seed always produces the identical perturbed image).
    for (std::size_t i = 0; i < img.pixels.size(); i += 37) {
        int idx = static_cast<int>((i * 2654435761u + static_cast<unsigned>(seed)) % img.pixels.size());
        int delta = static_cast<int>((i + static_cast<std::size_t>(seed)) % 11) - 5;
        int v = static_cast<int>(img.pixels[static_cast<std::size_t>(idx)]) + delta;
        img.pixels[static_cast<std::size_t>(idx)] = static_cast<std::uint8_t>(std::clamp(v, 0, 255));
    }
    return img;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.1: MICR Checksum Validation and Signature Comparison\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: a real, publicly published routing number (JPMorgan
    // Chase's own "021000021") satisfies the real ABA checksum formula. --
    std::string chase_routing = "021000021";
    bool chase_valid = micr_checksum_valid(chase_routing);
    std::cout << "-- Test 1: routing number " << chase_routing
              << " (a real, published ABA routing number) checksum-valid: "
              << (chase_valid ? "YES" : "NO") << " --\n";
    CHECK(chase_valid);

    // -- Test 2: altering a single digit of that same routing number
    // breaks the checksum -- exactly the class of single-digit
    // transcription error this formula exists to catch. --
    std::string altered_routing = "021000029";
    bool altered_valid = micr_checksum_valid(altered_routing);
    std::cout << "-- Test 2: routing number " << altered_routing
              << " (last digit altered from a valid routing number) checksum-valid: "
              << (altered_valid ? "YES" : "NO") << " --\n";
    CHECK(!altered_valid);

    // -- Test 3: a malformed routing number (wrong length, or containing
    // a non-digit) is rejected outright rather than silently truncated
    // or padded. --
    bool short_valid = micr_checksum_valid("0210000");
    bool nondigit_valid = micr_checksum_valid("02100002X");
    std::cout << "-- Test 3: a 7-digit routing number and a routing number containing a letter both checksum-valid: "
              << (short_valid || nondigit_valid ? "YES (WRONG)" : "NO (correctly rejected)") << " --\n";
    CHECK(!short_valid);
    CHECK(!nondigit_valid);

    // -- Test 4: two real dHash computations of a pixel-identical
    // signature image produce Hamming distance 0, and are correctly
    // reported as matching. --
    GrayscaleImage reference_sig = make_stroke(40, 24, 6, 18, false);
    GrayscaleImage identical_rescan = reference_sig;
    int dist_identical = hamming_distance(compute_dhash(reference_sig), compute_dhash(identical_rescan));
    std::cout << "-- Test 4: reference signature vs. a pixel-identical re-scan, Hamming distance = "
              << dist_identical << " --\n";
    CHECK(dist_identical == 0);
    CHECK(signatures_match(reference_sig, identical_rescan));

    // -- Test 5: the same physical signature, re-scanned with small,
    // realistic sensor noise, still matches under the stated tolerance. --
    GrayscaleImage noisy_rescan = add_scan_noise(reference_sig, 7);
    int dist_noisy = hamming_distance(compute_dhash(reference_sig), compute_dhash(noisy_rescan));
    bool noisy_match = signatures_match(reference_sig, noisy_rescan);
    std::cout << "-- Test 5: reference signature vs. the same signature re-scanned with sensor noise, Hamming distance = "
              << dist_noisy << ", matches under the stated tolerance of " << SIGNATURE_MATCH_MAX_HAMMING
              << ": " << (noisy_match ? "YES" : "NO") << " --\n";
    CHECK(noisy_match);

    // -- Test 6: a visibly different image -- standing in for a forged or
    // substituted signature, exactly the same honest use of a starkly
    // different synthetic image Chapter 23.1's own dHash test used -- is
    // correctly flagged as NOT matching, rather than the noise tolerance
    // silently absorbing a genuinely different signature too. --
    GrayscaleImage different_sig = make_checker(40, 24, 4);
    int dist_different = hamming_distance(compute_dhash(reference_sig), compute_dhash(different_sig));
    bool different_match = signatures_match(reference_sig, different_sig);
    std::cout << "-- Test 6: reference signature vs. a visibly different image standing in for a forged signature, Hamming distance = "
              << dist_different << ", matches under the stated tolerance: "
              << (different_match ? "YES (WRONG)" : "NO (correctly rejected)") << " --\n";
    CHECK(dist_different > SIGNATURE_MATCH_MAX_HAMMING);
    CHECK(!different_match);

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_micr_checksum_and_signature_comparison.cpp -o 01_micr_checksum_and_signature_comparison
./01_micr_checksum_and_signature_comparison
```

**Sample input:** a real, publicly known routing number checked against the real ABA checksum formula, then checked again with a single digit altered; a malformed routing number (wrong length, and separately containing a non-digit) checked to be rejected outright; a reference signature image checked against a pixel-identical re-scan (Hamming distance 0), the same signature re-scanned with realistic sensor noise (matching under a stated tolerance), and a visibly different image standing in for a forged signature (correctly rejected).

```text
========================================================
Chapter 26.1: MICR Checksum Validation and Signature Comparison
========================================================

-- Test 1: routing number 021000021 (a real, published ABA routing number) checksum-valid: YES --
-- Test 2: routing number 021000029 (last digit altered from a valid routing number) checksum-valid: NO --
-- Test 3: a 7-digit routing number and a routing number containing a letter both checksum-valid: NO (correctly rejected) --
-- Test 4: reference signature vs. a pixel-identical re-scan, Hamming distance = 0 --
-- Test 5: reference signature vs. the same signature re-scanned with sensor noise, Hamming distance = 4, matches under the stated tolerance of 10: YES --
-- Test 6: reference signature vs. a visibly different image standing in for a forged signature, Hamming distance = 34, matches under the stated tolerance: NO (correctly rejected) --

9/9 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a checksum-valid routing number as proof the check is genuine"
    A checksum-valid routing number proves only that the 9 digits printed on the MICR line are INTERNALLY CONSISTENT with the ABA's own published formula -- it says nothing about whether that routing number belongs to a real, currently-operating bank (which would require a live lookup against the Federal Reserve's own routing-number registry, a real network dependency this section's own offline, deterministic discipline deliberately does not take on), and nothing at all about whether the account number, the signature, or the amount fields are genuine. A sophisticated forger who simply copies a real, valid routing number from a genuine check onto a fraudulent one would pass this specific check perfectly -- exactly why Section 26.4's own risk engine treats a checksum failure as one input among several, forcing escalation rather than either auto-clearing OR auto-rejecting on this signal alone.

## 26.2 Invoice Duplicate Detection

### Intuition

A well-documented real invoice-fraud pattern is not a forged invoice at all, but the SAME real invoice submitted twice for payment, with its own invoice number altered by a single character specifically to dodge a naive exact-match duplicate check. Catching this requires the identical real fuzzy-matching algorithm Chapter 23.2 already built from scratch, applied to a different real signal.

### The Concept, In Detail

`levenshtein_distance` is Chapter 23.2's own unmodified real, classic edit-distance algorithm, confirmed in Test 1 against its own famous exact reference value (`KITTEN` to `SITTING` is distance 3). `flag_duplicate` combines three independent real signals -- an exact vendor match, an exact amount match, and an invoice number that is NEAR but not identical to a previously paid invoice's own number -- and Test 2 confirms an invoice number one character off from a real, already-paid invoice is flagged, naming the exact matching invoice and the exact edit distance.

Tests 3 through 6 each isolate one of the three required signals: Test 3 confirms a genuinely different invoice number (far beyond the stated edit-distance threshold) is not flagged; Test 4 confirms an EXACT resubmission -- the case a plain exact-match dedup pass already catches on its own -- falls outside this detector's own near-match scope by design; Test 5 confirms the identical near-match pattern from a different vendor is not flagged; and Test 6 confirms the identical near-match pattern at a genuinely different amount is not flagged, proving all three signals are required together, not any one alone. Test 7 confirms the exact stated edit-distance boundary itself, not merely a value comfortably inside it.

### Code and Verification

```cpp
// Chapter 26.2 -- A well-known real invoice-fraud pattern is not a forged
// invoice at all, but the SAME real invoice submitted twice for payment,
// with the invoice number altered by a single character specifically to
// dodge a naive exact-match duplicate check -- "INV-10422" resubmitted as
// "INV-10423" a week later, same vendor, same amount, hoping an
// overworked accounts-payable clerk (or a naive exact-string dedup pass)
// never notices. This section builds a real, from-scratch duplicate-
// invoice detector using the identical real Levenshtein edit-distance
// algorithm Chapter 23.2 built for OCR-noisy receipt reconciliation,
// applied here to a different real signal: an invoice number that is
// SUSPICIOUSLY CLOSE to, but not identical to, a previously paid
// invoice's own number, from the same vendor, for the same amount.
//
// A note on this section's own honest scope: `flag_duplicate` identifies
// a STRUCTURAL pattern -- a near-identical invoice number combined with
// an identical vendor and amount -- that is highly correlated with real
// double-billing fraud. It does not, and cannot, prove intent; a vendor
// that genuinely issues two separate real invoices for the same amount
// in the same week (a recurring service charge, for instance) would also
// match this pattern and should still be reviewed by a person, not
// auto-rejected -- exactly the same honest structural-pass discipline
// Chapter 23.4's own luxury-goods screening applied.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_invoice_duplicate_detection.cpp -o 02_invoice_duplicate_detection
// Run:     ./02_invoice_duplicate_detection

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
// PART 1: the real, classic Levenshtein edit-distance algorithm -- the
// identical implementation Chapter 23.2 built for receipt reconciliation.
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

// =======================================================================
// PART 2: a real, from-scratch duplicate-invoice detector combining
// three independent real signals -- exact vendor match, exact amount
// match, and a NEAR (but not identical) invoice number.
// =======================================================================
struct Invoice {
    std::string vendor;
    std::string invoice_number;
    double amount = 0.0;
};

constexpr double AMOUNT_EXACT_EPSILON = 0.005;
// A real, stated policy: an invoice number within this many single-
// character edits of a previously paid invoice number -- but NOT
// identical, which a plain exact-match dedup pass would already catch --
// is close enough to be the same real digits with one deliberately
// altered character.
constexpr int SUSPICIOUS_MAX_EDIT_DISTANCE = 2;

struct DuplicateFlag {
    bool flagged = false;
    std::string matched_invoice_number;
    int edit_distance = -1;
};

DuplicateFlag flag_duplicate(const Invoice& candidate, const std::vector<Invoice>& paid_history) {
    for (const Invoice& paid : paid_history) {
        if (to_upper(paid.vendor) != to_upper(candidate.vendor)) continue;
        if (std::abs(paid.amount - candidate.amount) >= AMOUNT_EXACT_EPSILON) continue;
        if (paid.invoice_number == candidate.invoice_number) continue;  // exact match: a plain dedup pass already catches this
        int dist = levenshtein_distance(to_upper(paid.invoice_number), to_upper(candidate.invoice_number));
        if (dist > 0 && dist <= SUSPICIOUS_MAX_EDIT_DISTANCE) {
            return DuplicateFlag{true, paid.invoice_number, dist};
        }
    }
    return DuplicateFlag{};
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.2: Invoice Duplicate Detection\n";
    std::cout << "========================================================\n\n";

    std::vector<Invoice> paid_history = {
        {"Acme Office Supply", "INV-10422", 1284.50},
        {"Northbridge Logistics", "PO-88213", 6120.00},
        {"Acme Office Supply", "INV-10517", 340.00},
    };

    // -- Test 1: the classic KITTEN-to-SITTING reference value, confirming
    // this is the real, unmodified Levenshtein algorithm. --
    int kitten_dist = levenshtein_distance("KITTEN", "SITTING");
    std::cout << "-- Test 1: Levenshtein(\"KITTEN\", \"SITTING\") = " << kitten_dist
              << " (the algorithm's own famous reference value) --\n";
    CHECK(kitten_dist == 3);

    // -- Test 2: an invoice number one character off from a real, already-
    // paid invoice, same vendor, same amount -- flagged as a likely
    // duplicate, naming the exact invoice number it matches and the exact
    // edit distance. --
    Invoice resubmit{"Acme Office Supply", "INV-10423", 1284.50};
    DuplicateFlag flag1 = flag_duplicate(resubmit, paid_history);
    std::cout << "-- Test 2: candidate " << resubmit.invoice_number << " ($" << resubmit.amount
              << ", " << resubmit.vendor << ") vs. paid history -- flagged: "
              << (flag1.flagged ? "YES" : "NO");
    if (flag1.flagged) {
        std::cout << ", matches paid invoice " << flag1.matched_invoice_number
                   << " at edit distance " << flag1.edit_distance;
    }
    std::cout << " --\n";
    CHECK(flag1.flagged);
    CHECK(flag1.matched_invoice_number == "INV-10422");
    CHECK(flag1.edit_distance == 1);

    // -- Test 3: a genuinely different invoice number for the same vendor
    // and amount -- far more than the stated edit-distance threshold from
    // any paid invoice -- is correctly NOT flagged. --
    Invoice different_number{"Acme Office Supply", "INV-99871", 1284.50};
    DuplicateFlag flag2 = flag_duplicate(different_number, paid_history);
    std::cout << "-- Test 3: candidate " << different_number.invoice_number
              << " ($" << different_number.amount << ", " << different_number.vendor
              << ") vs. paid history -- flagged: " << (flag2.flagged ? "YES (WRONG)" : "NO (correctly not flagged)") << " --\n";
    CHECK(!flag2.flagged);

    // -- Test 4: an exact re-submission of the identical invoice number --
    // the case a plain exact-match dedup pass already catches on its own
    // -- is correctly excluded from THIS detector's own near-match logic,
    // since its own job is the gap a naive exact-match check misses. --
    Invoice exact_resubmit{"Acme Office Supply", "INV-10422", 1284.50};
    DuplicateFlag flag3 = flag_duplicate(exact_resubmit, paid_history);
    std::cout << "-- Test 4: candidate " << exact_resubmit.invoice_number
              << " (an EXACT resubmission, already caught by a plain dedup pass) -- flagged by THIS near-match detector: "
              << (flag3.flagged ? "YES (WRONG)" : "NO (correctly out of this detector's own scope)") << " --\n";
    CHECK(!flag3.flagged);

    // -- Test 5: the same near-match invoice number pattern from a
    // DIFFERENT vendor is correctly not flagged -- a coincidental invoice-
    // number collision across unrelated vendors is not evidence of
    // double-billing. --
    Invoice different_vendor{"Northbridge Logistics", "INV-10423", 1284.50};
    DuplicateFlag flag4 = flag_duplicate(different_vendor, paid_history);
    std::cout << "-- Test 5: candidate " << different_vendor.invoice_number
              << " from a DIFFERENT vendor (" << different_vendor.vendor
              << ") at the identical amount -- flagged: "
              << (flag4.flagged ? "YES (WRONG)" : "NO (correctly not flagged)") << " --\n";
    CHECK(!flag4.flagged);

    // -- Test 6: the identical near-match invoice-number pattern but at a
    // genuinely different amount is correctly not flagged -- this
    // detector's own three signals (vendor, amount, near-match number)
    // are all required together, not any one alone. --
    Invoice different_amount{"Acme Office Supply", "INV-10423", 55.00};
    DuplicateFlag flag5 = flag_duplicate(different_amount, paid_history);
    std::cout << "-- Test 6: candidate " << different_amount.invoice_number
              << " at a genuinely different amount ($" << different_amount.amount
              << " vs. the paid $1284.50) -- flagged: "
              << (flag5.flagged ? "YES (WRONG)" : "NO (correctly not flagged)") << " --\n";
    CHECK(!flag5.flagged);

    // -- Test 6: an invoice number exactly AT the stated edit-distance
    // boundary (distance 2) is still flagged; the exact boundary itself
    // is checked, not just a value comfortably inside it. --
    Invoice boundary{"Acme Office Supply", "INV-10499", 1284.50};
    int boundary_dist = levenshtein_distance("INV-10422", "INV-10499");
    DuplicateFlag flag6 = flag_duplicate(boundary, paid_history);
    std::cout << "-- Test 7: candidate " << boundary.invoice_number
              << " at hand-computed edit distance " << boundary_dist
              << " from the paid invoice (the stated boundary of " << SUSPICIOUS_MAX_EDIT_DISTANCE
              << ") -- flagged: " << (flag6.flagged ? "YES" : "NO") << " --\n";
    CHECK(boundary_dist == 2);
    CHECK(flag6.flagged);

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_invoice_duplicate_detection.cpp -o 02_invoice_duplicate_detection
./02_invoice_duplicate_detection
```

**Sample input:** the real Levenshtein algorithm checked against its own famous KITTEN-to-SITTING reference value; an invoice number one character off from an already-paid invoice, same vendor and amount, checked to be flagged with the exact matching invoice number and edit distance named; a genuinely different invoice number, an exact resubmission, a different vendor, and a different amount each checked in turn to confirm all three required signals must hold together; and an invoice number checked exactly at the stated edit-distance boundary.

```text
========================================================
Chapter 26.2: Invoice Duplicate Detection
========================================================

-- Test 1: Levenshtein("KITTEN", "SITTING") = 3 (the algorithm's own famous reference value) --
-- Test 2: candidate INV-10423 ($1284.5, Acme Office Supply) vs. paid history -- flagged: YES, matches paid invoice INV-10422 at edit distance 1 --
-- Test 3: candidate INV-99871 ($1284.5, Acme Office Supply) vs. paid history -- flagged: NO (correctly not flagged) --
-- Test 4: candidate INV-10422 (an EXACT resubmission, already caught by a plain dedup pass) -- flagged by THIS near-match detector: NO (correctly out of this detector's own scope) --
-- Test 5: candidate INV-10423 from a DIFFERENT vendor (Northbridge Logistics) at the identical amount -- flagged: NO (correctly not flagged) --
-- Test 6: candidate INV-10423 at a genuinely different amount ($55 vs. the paid $1284.50) -- flagged: NO (correctly not flagged) --
-- Test 7: candidate INV-10499 at hand-computed edit distance 2 from the paid invoice (the stated boundary of 2) -- flagged: YES --

10/10 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating this detector's own structural flag as proof of fraudulent intent"
    `flag_duplicate` identifies a real, checkable STRUCTURAL pattern -- a near-identical invoice number combined with an identical vendor and amount -- that correlates strongly with real double-billing fraud. It does not, and cannot, prove intent. A vendor that genuinely issues two separate, legitimate invoices for the same recurring service charge in the same week, using an invoice-numbering scheme that happens to increment by one, would also match this exact pattern, and a flagged invoice should still go to a human reviewer for a real decision rather than being auto-rejected on this signal alone -- exactly the honest structural-pass discipline Chapter 23.4's own luxury-goods screening engine applied to its own best possible outcome.

## 26.3 Amount-in-Words Cross-Validation

### Intuition

A US check states its own amount twice -- once in numerals, once spelled out in words -- specifically so that altering the numerals alone (the classic "check washing" fraud, chemically removing and rewriting a printed amount) does not silently succeed, provided something actually cross-checks the two fields against each other. This section builds that cross-check from scratch: a real English number-word parser reconstructing the exact cent value the words field claims.

### The Concept, In Detail

`parse_amount_words` tokenizes a check-convention words phrase (splitting on hyphens as well as spaces, so "thirty-four" parses as two number words) and reconstructs its own dollar value using a real running-group accumulator: each unit or tens word adds into the current group, "hundred" multiplies the current group by 100, and "thousand" flushes the current group into the running total and resets it -- Test 7 confirms this grouping logic correctly handles a nonzero hundreds group nested inside a thousands group ("Forty-two thousand three hundred five"). An "and NN/100" cents suffix, when present, is parsed separately and added as exact integer cents, avoiding the floating-point drift this book's own quantization work in Chapter 4 already showed is unsafe for a value that must compare exactly.

`cross_check` compares the words-derived cent value against the numerals field, converted to cents via the identical round-to-nearest-cent discipline. Test 2 reproduces the classic check-washing scenario directly: the numerals altered while the words field is left untouched, caught exactly; Test 3 confirms the reverse alteration is caught by the identical logic. Test 5 confirms an honestly unparseable words field (containing a token this narrow, stated parser does not recognize) is reported as an explicit parse failure -- never silently treated as either a match or a silent zero.

### Code and Verification

```cpp
// Chapter 26.3 -- A US check states its own amount TWICE: once as digits
// ("$1,234.56") and once spelled out in words ("One thousand two hundred
// thirty-four and 56/100 dollars"). This redundancy exists specifically
// so that altering the numerals alone -- the classic "check washing"
// fraud, chemically removing and rewriting the amount -- does not
// silently succeed, provided something actually cross-checks the two
// fields against each other. This section builds a real, from-scratch
// English number-word parser that reconstructs the numeric value the
// words field actually claims, and cross-validates it against the
// numerals field.
//
// A note on this section's own honest scope: `parse_amount_words`
// implements the specific real US-check convention (a whole-dollar
// amount in words, followed by "and NN/100" for cents) -- it does not
// attempt to parse arbitrary English number phrases (currencies with
// their own different word-order conventions, or numbers phrased with
// "and" placed differently), the same deliberately narrow, honestly
// stated scope Chapter 12's own tokenizer applied to its own real
// vocabulary rather than claiming to handle every language.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_amount_in_words_cross_validation.cpp -o 03_amount_in_words_cross_validation
// Run:     ./03_amount_in_words_cross_validation

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: a real, from-scratch parser turning US check-convention words
// ("one thousand two hundred thirty-four and 56/100") into an exact cent
// count -- the same integer-cents discipline Chapter 4's own quantization
// work used to avoid floating-point drift on a monetary value.
// =======================================================================
const std::map<std::string, int> UNITS = {
    {"zero", 0}, {"one", 1}, {"two", 2}, {"three", 3}, {"four", 4},
    {"five", 5}, {"six", 6}, {"seven", 7}, {"eight", 8}, {"nine", 9},
    {"ten", 10}, {"eleven", 11}, {"twelve", 12}, {"thirteen", 13},
    {"fourteen", 14}, {"fifteen", 15}, {"sixteen", 16}, {"seventeen", 17},
    {"eighteen", 18}, {"nineteen", 19},
};
const std::map<std::string, int> TENS = {
    {"twenty", 20}, {"thirty", 30}, {"forty", 40}, {"fifty", 50},
    {"sixty", 60}, {"seventy", 70}, {"eighty", 80}, {"ninety", 90},
};

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::vector<std::string> tokenize_words(const std::string& phrase) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : phrase) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '/') {
            current += c;
        } else if (c == '-') {
            // "thirty-four" splits into two number words joined by a hyphen
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
        } else {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

// Parses the WHOLE-DOLLAR portion of a words phrase (everything up to,
// but not including, "and NN/100" or "dollars"). Returns std::nullopt on
// any token this parser does not recognize -- an honest failure, never a
// silent zero.
std::optional<long long> parse_dollar_words(const std::vector<std::string>& tokens) {
    long long total = 0;
    long long current_group = 0;
    bool saw_any = false;
    for (const std::string& raw : tokens) {
        std::string tok = to_lower(raw);
        if (tok == "and" || tok == "dollars" || tok == "dollar") continue;
        if (tok == "hundred") {
            if (current_group == 0) return std::nullopt;
            current_group *= 100;
            saw_any = true;
        } else if (tok == "thousand") {
            if (current_group == 0) return std::nullopt;
            total += current_group * 1000;
            current_group = 0;
            saw_any = true;
        } else if (auto it = UNITS.find(tok); it != UNITS.end()) {
            current_group += it->second;
            saw_any = true;
        } else if (auto it2 = TENS.find(tok); it2 != TENS.end()) {
            current_group += it2->second;
            saw_any = true;
        } else {
            return std::nullopt;  // an unrecognized token: honest failure, not a silent guess
        }
    }
    total += current_group;
    if (!saw_any) return std::nullopt;
    return total;
}

struct ParsedAmount {
    bool valid = false;
    long long total_cents = 0;
};

// Parses the full check-convention phrase, including the "and NN/100"
// cents suffix if present (cents default to 0 if the phrase states none).
ParsedAmount parse_amount_words(const std::string& phrase) {
    std::string lower = to_lower(phrase);
    std::size_t slash_pos = lower.find('/');
    long long cents = 0;
    std::string dollar_part = phrase;
    if (slash_pos != std::string::npos) {
        // Walk backward from the slash to find the start of the "NN" cents
        // numerator, then forward past "/100" to split the phrase.
        std::size_t num_start = slash_pos;
        while (num_start > 0 && std::isdigit(static_cast<unsigned char>(lower[num_start - 1]))) num_start--;
        std::string cents_str = lower.substr(num_start, slash_pos - num_start);
        if (cents_str.empty()) return ParsedAmount{false, 0};
        cents = std::stoll(cents_str);
        dollar_part = phrase.substr(0, num_start);
    }
    std::vector<std::string> tokens = tokenize_words(dollar_part);
    std::optional<long long> dollars = parse_dollar_words(tokens);
    if (!dollars.has_value() || cents < 0 || cents > 99) return ParsedAmount{false, 0};
    return ParsedAmount{true, *dollars * 100 + cents};
}

long long numerals_to_cents(double amount) {
    // Round to the nearest cent using integer arithmetic on a scaled
    // value -- the same avoid-floating-point-drift discipline used
    // wherever this book compares a monetary or quantized value exactly.
    return static_cast<long long>(amount * 100.0 + (amount >= 0 ? 0.5 : -0.5));
}

struct CrossCheckResult {
    bool matches = false;
    bool words_parse_failed = false;
    long long numerals_cents = 0;
    long long words_cents = 0;
};

CrossCheckResult cross_check(double numerals_amount, const std::string& words_phrase) {
    long long numerals_cents = numerals_to_cents(numerals_amount);
    ParsedAmount parsed = parse_amount_words(words_phrase);
    if (!parsed.valid) return CrossCheckResult{false, true, numerals_cents, 0};
    return CrossCheckResult{numerals_cents == parsed.total_cents, false, numerals_cents, parsed.total_cents};
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.3: Amount-in-Words Cross-Validation\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: a correctly matching check -- numerals and words agree
    // exactly, down to the cent. --
    CrossCheckResult r1 = cross_check(1234.56, "One thousand two hundred thirty-four and 56/100 dollars");
    std::cout << "-- Test 1: $1234.56 vs. \"One thousand two hundred thirty-four and 56/100 dollars\" -- numerals = "
              << r1.numerals_cents << " cents, words = " << r1.words_cents << " cents, match: "
              << (r1.matches ? "YES" : "NO") << " --\n";
    CHECK(r1.matches);
    CHECK(r1.numerals_cents == 123456);
    CHECK(r1.words_cents == 123456);

    // -- Test 2: the classic check-washing fraud -- the numerals field
    // altered (say, from $1,234.56 to $9,234.56) while the words field is
    // left untouched. The mismatch is caught exactly, naming both values. --
    CrossCheckResult r2 = cross_check(9234.56, "One thousand two hundred thirty-four and 56/100 dollars");
    std::cout << "-- Test 2: $9234.56 (numerals altered) vs. the SAME unaltered words field -- numerals = "
              << r2.numerals_cents << " cents, words = " << r2.words_cents << " cents, match: "
              << (r2.matches ? "YES (WRONG)" : "NO (correctly caught)") << " --\n";
    CHECK(!r2.matches);

    // -- Test 3: the reverse alteration -- the words field altered while
    // numerals are left untouched -- is caught by the identical logic,
    // since the check compares the two fields symmetrically. --
    CrossCheckResult r3 = cross_check(1234.56, "Nine thousand two hundred thirty-four and 56/100 dollars");
    std::cout << "-- Test 3: $1234.56 vs. \"Nine thousand two hundred thirty-four and 56/100 dollars\" (words altered) -- match: "
              << (r3.matches ? "YES (WRONG)" : "NO (correctly caught)") << " --\n";
    CHECK(!r3.matches);

    // -- Test 4: a whole-dollar amount with no stated cents defaults to
    // zero cents, matching a numerals amount of exactly $500.00. --
    CrossCheckResult r4 = cross_check(500.00, "Five hundred dollars");
    std::cout << "-- Test 4: $500.00 vs. \"Five hundred dollars\" (no cents stated) -- words = "
              << r4.words_cents << " cents, match: " << (r4.matches ? "YES" : "NO") << " --\n";
    CHECK(r4.matches);
    CHECK(r4.words_cents == 50000);

    // -- Test 5: a genuinely unparseable words field (containing a token
    // this narrow, stated parser does not recognize) is reported as an
    // honest parse failure, never silently treated as a match or a
    // silent zero. --
    CrossCheckResult r5 = cross_check(1234.56, "A gazillion dollars");
    std::cout << "-- Test 5: \"A gazillion dollars\" (an unrecognized token) -- words_parse_failed: "
              << (r5.words_parse_failed ? "YES (correctly reported)" : "NO (WRONG)") << " --\n";
    CHECK(r5.words_parse_failed);
    CHECK(!r5.matches);

    // -- Test 6: an amount just below the "hundred" boundary (99 dollars,
    // no hundreds/thousands grouping at all) parses correctly, confirming
    // the parser handles the plain-units case, not only multi-group
    // amounts. --
    CrossCheckResult r6 = cross_check(99.00, "Ninety-nine dollars");
    std::cout << "-- Test 6: $99.00 vs. \"Ninety-nine dollars\" -- words = " << r6.words_cents
              << " cents, match: " << (r6.matches ? "YES" : "NO") << " --\n";
    CHECK(r6.matches);
    CHECK(r6.words_cents == 9900);

    // -- Test 7: a real, larger multi-thousand amount with a nonzero
    // hundreds group inside the thousands group -- confirming the parser
    // correctly resets its own running group after each "thousand". --
    CrossCheckResult r7 = cross_check(42305.75, "Forty-two thousand three hundred five and 75/100 dollars");
    std::cout << "-- Test 7: $42305.75 vs. \"Forty-two thousand three hundred five and 75/100 dollars\" -- words = "
              << r7.words_cents << " cents, match: " << (r7.matches ? "YES" : "NO") << " --\n";
    CHECK(r7.matches);
    CHECK(r7.words_cents == 4230575);

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_amount_in_words_cross_validation.cpp -o 03_amount_in_words_cross_validation
./03_amount_in_words_cross_validation
```

**Sample input:** a correctly matching check amount checked in both numerals and words; the classic check-washing pattern reproduced in both directions (numerals altered with words untouched, and words altered with numerals untouched), each caught exactly; a whole-dollar amount with no stated cents; a genuinely unparseable words phrase checked to report an honest parse failure; a plain amount under one hundred dollars with no hundreds or thousands grouping; and a larger multi-thousand amount with a nonzero hundreds group nested inside it.

```text
========================================================
Chapter 26.3: Amount-in-Words Cross-Validation
========================================================

-- Test 1: $1234.56 vs. "One thousand two hundred thirty-four and 56/100 dollars" -- numerals = 123456 cents, words = 123456 cents, match: YES --
-- Test 2: $9234.56 (numerals altered) vs. the SAME unaltered words field -- numerals = 923456 cents, words = 123456 cents, match: NO (correctly caught) --
-- Test 3: $1234.56 vs. "Nine thousand two hundred thirty-four and 56/100 dollars" (words altered) -- match: NO (correctly caught) --
-- Test 4: $500.00 vs. "Five hundred dollars" (no cents stated) -- words = 50000 cents, match: YES --
-- Test 5: "A gazillion dollars" (an unrecognized token) -- words_parse_failed: YES (correctly reported) --
-- Test 6: $99.00 vs. "Ninety-nine dollars" -- words = 9900 cents, match: YES --
-- Test 7: $42305.75 vs. "Forty-two thousand three hundred five and 75/100 dollars" -- words = 4230575 cents, match: YES --

13/13 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] assuming this section's own narrow parser handles any English number phrase"
    `parse_amount_words` implements one specific, real, stated convention: a US check's own whole-dollar amount in words, followed by an "and NN/100" cents suffix. It does not attempt to parse arbitrary English number phrasing -- a phrase using "and" in a different position, a non-US convention for grouping large numbers, or a currency whose own printed-check convention differs -- and Test 5 exists specifically to prove an unrecognized token produces an honest, explicit parse failure rather than a silently wrong guess. Deploying this exact parser against checks drawn on a different real convention without first confirming its own vocabulary covers that convention's own words would risk exactly the silent-wrong-answer failure this book's own honesty discipline exists to prevent -- the identical narrow-scope caveat Chapter 12's own tokenizer stated for its own real, fixed vocabulary.

## 26.4 Risk Scoring and Escalation

### Intuition

A single fraud signal rarely justifies an automatic rejection on its own, and a clean aggregate score should never be allowed to silently absorb one specific, high-severity red flag. This section's capstone combines Section 26.1's MICR and signature signals, Section 26.2's duplicate-invoice signal, and Section 26.3's amount-mismatch signal into one real, fully auditable weighted score and a three-way disposition, reapplying this book's own recurring never-suppress-a-named-flag discipline one more time.

### The Concept, In Detail

`assess_risk` sums the point weight of every signal that fired into a single total score, and separately records the FULL, NAMED list of which signals fired -- Test 1 confirms a clean transaction produces an honestly empty audit trail, not a suppressed one. Test 2 and Test 3 confirm the two real score thresholds route a single moderate flag to `ESCALATE_TO_REVIEW` and two flags together to `AUTO_REJECT`, with the audit trail naming every contributing signal. Test 6 and Test 7 confirm both threshold boundaries exactly, not merely values comfortably on either side.

Test 4 is this section's own central honesty check: a failed MICR checksum, with every OTHER signal clean and a total score of exactly 0, still forces at least `ESCALATE_TO_REVIEW` -- the identical never-suppress-a-named-flag discipline Chapter 23.4's own hard-failing checksum check applied to two otherwise-passing signals. Test 5 confirms the forcing rule is directional: combined with signals that already independently cross the reject threshold, the MICR failure does not (and must not) downgrade an `AUTO_REJECT` back down to a milder `ESCALATE_TO_REVIEW`.

### Code and Verification

```cpp
// Chapter 26.4 -- This chapter's own capstone: a real, from-scratch,
// fully auditable risk-scoring engine that combines Section 26.1's MICR
// checksum and signature-comparison signals, Section 26.2's duplicate-
// invoice signal, and Section 26.3's amount-in-words cross-check into a
// single weighted score and a three-way disposition -- AUTO_CLEAR,
// ESCALATE_TO_REVIEW, or AUTO_REJECT. The central discipline, identical
// to Chapter 20's own TriageCase state machine and Chapter 23's own
// never-suppress-a-named-flag rule: the aggregate score decides the
// DISPOSITION, but every single contributing signal is named individually
// in an audit trail no aggregate number can hide, and one specific,
// high-severity red flag (a failed MICR checksum) can force escalation
// on its own, regardless of how low every other signal's score is.
//
// A note on this section's own honest scope: `assess_risk` produces a
// real, deterministic, fully-explained DISPOSITION from a fixed set of
// stated signals and weights -- it is a real screening step, not a fraud
// verdict. AUTO_CLEAR means "no stated signal in this engine's own scope
// fired," not "this transaction is definitely legitimate" -- the same
// honest structural-pass distinction Chapter 23.4's own luxury-goods
// screening drew for its own best possible outcome.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_risk_scoring_and_escalation.cpp -o 04_risk_scoring_and_escalation
// Run:     ./04_risk_scoring_and_escalation

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
// A real, auditable risk signal: a name, a point weight, and whether it
// fired -- never a bare number with no way to trace back which real
// check produced it.
// =======================================================================
struct RiskSignal {
    std::string name;
    int weight = 0;
    bool fired = false;
};

enum class Disposition { AUTO_CLEAR, ESCALATE_TO_REVIEW, AUTO_REJECT };

std::string disposition_name(Disposition d) {
    switch (d) {
        case Disposition::AUTO_CLEAR: return "AUTO_CLEAR";
        case Disposition::ESCALATE_TO_REVIEW: return "ESCALATE_TO_REVIEW";
        case Disposition::AUTO_REJECT: return "AUTO_REJECT";
    }
    return "UNKNOWN";
}

struct RiskAssessment {
    int total_score = 0;
    Disposition disposition = Disposition::AUTO_CLEAR;
    std::vector<std::string> fired_signal_names;  // the full, named audit trail
    bool forced_by_micr_failure = false;
};

// A real, stated scoring policy: below ESCALATE_THRESHOLD, auto-clear;
// at or above ESCALATE_THRESHOLD but below REJECT_THRESHOLD, escalate to
// a human reviewer; at or above REJECT_THRESHOLD, auto-reject. A failed
// MICR checksum -- structurally inconsistent with any real bank's own
// routing number -- forces at least ESCALATE_TO_REVIEW regardless of
// every other signal's own score, since it alone indicates the check
// image itself may not be genuine.
constexpr int ESCALATE_THRESHOLD = 30;
constexpr int REJECT_THRESHOLD = 70;

RiskAssessment assess_risk(const std::vector<RiskSignal>& signals, bool micr_checksum_failed) {
    RiskAssessment result;
    for (const RiskSignal& s : signals) {
        if (s.fired) {
            result.total_score += s.weight;
            result.fired_signal_names.push_back(s.name);
        }
    }
    if (result.total_score >= REJECT_THRESHOLD) {
        result.disposition = Disposition::AUTO_REJECT;
    } else if (result.total_score >= ESCALATE_THRESHOLD) {
        result.disposition = Disposition::ESCALATE_TO_REVIEW;
    } else {
        result.disposition = Disposition::AUTO_CLEAR;
    }
    if (micr_checksum_failed && result.disposition == Disposition::AUTO_CLEAR) {
        result.disposition = Disposition::ESCALATE_TO_REVIEW;
        result.forced_by_micr_failure = true;
    }
    return result;
}

void print_assessment(const std::string& label, const RiskAssessment& a) {
    std::cout << label << ": score = " << a.total_score << ", disposition = "
              << disposition_name(a.disposition);
    if (a.forced_by_micr_failure) std::cout << " (forced by MICR checksum failure)";
    std::cout << ", fired signals: [";
    for (std::size_t i = 0; i < a.fired_signal_names.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << a.fired_signal_names[i];
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.4: Risk Scoring and Escalation\n";
    std::cout << "========================================================\n\n";

    // -- Test 1: a clean transaction -- every signal from every prior
    // section passes -- is auto-cleared with an empty, honestly-empty
    // audit trail, not a suppressed one. --
    std::vector<RiskSignal> clean_signals = {
        {"duplicate_invoice", 40, false},
        {"amount_words_mismatch", 50, false},
        {"signature_mismatch", 35, false},
    };
    RiskAssessment a1 = assess_risk(clean_signals, false);
    print_assessment("-- Test 1 (clean transaction)", a1);
    CHECK(a1.total_score == 0);
    CHECK(a1.disposition == Disposition::AUTO_CLEAR);
    CHECK(a1.fired_signal_names.empty());

    // -- Test 2: a single, moderate-severity red flag (Section 26.2's
    // duplicate-invoice signal alone) crosses the escalate threshold but
    // not the reject threshold -- routed to a human reviewer, not
    // auto-rejected outright on one signal alone. --
    std::vector<RiskSignal> single_flag = {
        {"duplicate_invoice", 40, true},
        {"amount_words_mismatch", 50, false},
        {"signature_mismatch", 35, false},
    };
    RiskAssessment a2 = assess_risk(single_flag, false);
    print_assessment("-- Test 2 (one moderate flag)", a2);
    CHECK(a2.total_score == 40);
    CHECK(a2.disposition == Disposition::ESCALATE_TO_REVIEW);
    CHECK(a2.fired_signal_names.size() == 1);
    CHECK(a2.fired_signal_names[0] == "duplicate_invoice");

    // -- Test 3: two red flags together cross the reject threshold, and
    // the audit trail names BOTH contributing signals -- the aggregate
    // score never hides which specific checks actually fired. --
    std::vector<RiskSignal> two_flags = {
        {"duplicate_invoice", 40, true},
        {"amount_words_mismatch", 50, true},
        {"signature_mismatch", 35, false},
    };
    RiskAssessment a3 = assess_risk(two_flags, false);
    print_assessment("-- Test 3 (two flags, crosses reject threshold)", a3);
    CHECK(a3.total_score == 90);
    CHECK(a3.disposition == Disposition::AUTO_REJECT);
    CHECK(a3.fired_signal_names.size() == 2);

    // -- Test 4: a failed MICR checksum, on its own, with every OTHER
    // signal clean and a total score of 0 -- still forces at least
    // ESCALATE_TO_REVIEW, confirming one specific high-severity signal
    // is never outvoted by an otherwise-clean aggregate score, the
    // identical discipline Chapter 23.4's own hard-failing weight check
    // applied. --
    RiskAssessment a4 = assess_risk(clean_signals, true);
    print_assessment("-- Test 4 (MICR checksum failure alone, all other signals clean)", a4);
    CHECK(a4.total_score == 0);
    CHECK(a4.disposition == Disposition::ESCALATE_TO_REVIEW);
    CHECK(a4.forced_by_micr_failure);

    // -- Test 5: a failed MICR checksum combined with signals that
    // ALREADY cross the reject threshold on their own does not downgrade
    // the disposition -- the forcing rule only ever escalates an
    // AUTO_CLEAR upward, it never overrides an already-stricter
    // AUTO_REJECT with a milder ESCALATE_TO_REVIEW. --
    RiskAssessment a5 = assess_risk(two_flags, true);
    print_assessment("-- Test 5 (MICR checksum failure PLUS signals that already reject)", a5);
    CHECK(a5.disposition == Disposition::AUTO_REJECT);
    CHECK(!a5.forced_by_micr_failure);

    // -- Test 6: a score exactly AT the escalate threshold (30) is
    // escalated, not auto-cleared -- the boundary itself is checked, not
    // just a value comfortably below it. --
    std::vector<RiskSignal> boundary_signals = {
        {"minor_signal", 30, true},
    };
    RiskAssessment a6 = assess_risk(boundary_signals, false);
    print_assessment("-- Test 6 (score exactly at the escalate boundary of 30)", a6);
    CHECK(a6.total_score == 30);
    CHECK(a6.disposition == Disposition::ESCALATE_TO_REVIEW);

    // -- Test 7: a score exactly AT the reject threshold (70) is
    // rejected, not merely escalated -- the reject boundary is likewise
    // checked exactly. --
    std::vector<RiskSignal> reject_boundary_signals = {
        {"duplicate_invoice", 40, true},
        {"signature_mismatch", 30, true},
    };
    RiskAssessment a7 = assess_risk(reject_boundary_signals, false);
    print_assessment("-- Test 7 (score exactly at the reject boundary of 70)", a7);
    CHECK(a7.total_score == 70);
    CHECK(a7.disposition == Disposition::AUTO_REJECT);

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_risk_scoring_and_escalation.cpp -o 04_risk_scoring_and_escalation
./04_risk_scoring_and_escalation
```

**Sample input:** a clean transaction with every signal from every prior section passing, checked to auto-clear with an empty audit trail; a single moderate-severity flag checked to escalate; two flags together checked to cross the reject threshold, naming both; a failed MICR checksum alone, with every other signal clean, checked to still force escalation; the same MICR failure combined with signals that already independently reject, checked to leave the rejection undowngraded; and both real score thresholds checked exactly at their own stated boundaries.

```text
========================================================
Chapter 26.4: Risk Scoring and Escalation
========================================================

-- Test 1 (clean transaction): score = 0, disposition = AUTO_CLEAR, fired signals: []
-- Test 2 (one moderate flag): score = 40, disposition = ESCALATE_TO_REVIEW, fired signals: [duplicate_invoice]
-- Test 3 (two flags, crosses reject threshold): score = 90, disposition = AUTO_REJECT, fired signals: [duplicate_invoice, amount_words_mismatch]
-- Test 4 (MICR checksum failure alone, all other signals clean): score = 0, disposition = ESCALATE_TO_REVIEW (forced by MICR checksum failure), fired signals: []
-- Test 5 (MICR checksum failure PLUS signals that already reject): score = 90, disposition = AUTO_REJECT, fired signals: [duplicate_invoice, amount_words_mismatch]
-- Test 6 (score exactly at the escalate boundary of 30): score = 30, disposition = ESCALATE_TO_REVIEW, fired signals: [minor_signal]
-- Test 7 (score exactly at the reject boundary of 70): score = 70, disposition = AUTO_REJECT, fired signals: [duplicate_invoice, signature_mismatch]

19/19 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating AUTO_CLEAR as proof of a legitimate transaction"
    `assess_risk` produces a real, deterministic, fully-explained disposition from a fixed, stated set of signals and weights -- it is a real screening step, not a fraud verdict. `AUTO_CLEAR` means, precisely and only, that no signal within THIS engine's own stated scope fired for this specific transaction; it is not, and was never built to be, an affirmative statement that the transaction is legitimate. A fraud pattern this chapter's own four sections do not check for at all -- a stolen but structurally perfect blank check stock, for instance -- would sail through every signal in this file and land at `AUTO_CLEAR`, exactly as a well-forged item with a perfectly valid Luhn-checksummed serial number would have passed Chapter 23.4's own structural screening. An aggregate disposition of AUTO_CLEAR is only ever as strong as the specific, named list of signals that fed it -- never a substitute for knowing exactly what that list does and does not cover.

## Chapter Summary

This chapter reapplied two of this book's own already-built real algorithms -- Chapter 23.1's perceptual difference hash and Chapter 23.2's classic Levenshtein edit distance -- to a new financial domain, and added two new real algorithms of its own: the published ABA routing-number checksum and a from-scratch amount-in-words parser. Section 26.1 validated a check's own MICR routing number against its real published checksum and compared a signature image using the identical dHash algorithm Chapter 23.1 built for product photography. Section 26.2 reapplied the classic Levenshtein algorithm to catch an invoice resubmitted with a single altered character, specifically excluding the exact-resubmission case a plain dedup pass already catches. Section 26.3 built a real English number-word parser and used it to reproduce and catch the classic check-washing fraud pattern in both directions. Section 26.4 closed the chapter by combining all of the above into one fully auditable risk score, proving once more that a single high-severity signal can force escalation regardless of how clean the aggregate score otherwise looks, and that the forcing rule never downgrades an already-stricter disposition.

## Self-Check Questions

1. Section 26.1's `micr_checksum_valid` and Chapter 23.4's `luhn_valid` are both real, published check-digit algorithms applied to different real domains. Explain one concrete structural similarity between how the two algorithms combine a number's own digits into a single pass/fail result.
2. Section 26.1's own COMMON TRAP box explains that a checksum-valid routing number does not prove a check is genuine. Name one specific real fraud scenario in which a check would pass Section 26.1's own MICR checksum check perfectly while still being fraudulent.
3. Section 26.2's `flag_duplicate` explicitly excludes an EXACT invoice-number resubmission from its own near-match logic. Explain why including the exact-match case inside this same function, rather than relying on a separate plain dedup pass, would not actually change this detector's own real coverage of the fraud pattern it targets.
4. Section 26.2's Test 6 constructs a case where the same near-match invoice-number pattern occurs at a genuinely different amount and is correctly not flagged. Explain concretely why requiring an exact amount match, rather than an amount within some tolerance, is the correct design choice for THIS specific fraud pattern.
5. Section 26.3's `parse_dollar_words` resets its own running `current_group` to 0 immediately after each "thousand" token. Using the phrase "Forty-two thousand three hundred five" from Section 26.3's own Test 7, trace through what the function's own final total would incorrectly become if this reset were removed.
6. Section 26.3's own COMMON TRAP box warns against assuming the parser handles any English number phrase. Construct one specific, realistic check-amount phrase that a real US check might contain, using standard English number words, that this section's own stated parser would fail to parse correctly, and explain exactly which token or structure it does not handle.
7. Section 26.4's `assess_risk` records a fired signal's own NAME in the audit trail, not merely its point weight. Explain what specific real information a human reviewer would lose if the audit trail recorded only the total numeric score and the count of fired signals, without their names.
8. Section 26.4's Test 5 confirms the MICR-forcing rule does not downgrade an already-stricter `AUTO_REJECT` disposition. Explain why allowing the forcing rule to work in BOTH directions (also capable of downgrading a stricter disposition to a milder one) would create a real, exploitable weakness in this risk engine.
9. Section 26.1's signature-comparison Hamming-distance tolerance and Section 26.2's invoice-number edit-distance tolerance both define a real "close enough" boundary using a different real metric (Hamming distance on a fixed 64-bit hash vs. Levenshtein distance on a variable-length string). Explain why a single shared numeric threshold value would not make sense to reuse across both checks.
10. Across all four sections of this chapter, identify the ONE section whose own central test is built specifically to catch a fraud pattern that involves ALTERING a value already on file, as opposed to introducing an entirely new fraudulent item, invoice, or transaction. Explain what specifically about that fraud pattern makes it detectable by a cross-check between two fields, in a way the other three sections' own fraud patterns are not.

## Where We Go Next

This chapter showed that reapplying algorithms this book already built from scratch -- a perceptual hash, an edit-distance algorithm -- to a new financial domain can be exactly as effective as building something new, provided each reapplication is grounded in a real, well-understood fraud pattern rather than borrowed for its own sake. Chapter 27 turns to a related but distinct real financial setting: low-latency inference on live market-signal data near the exchange, where the constraint shifts from fraud-pattern detection to a hard real-time deadline measured in microseconds.

## Worked Solutions

**1.** Both algorithms combine a number's own digits using a fixed, published set of per-position WEIGHTS, sum the weighted digits together, and check the result against a fixed real modulus (the ABA formula checks divisibility by 10 directly on the weighted sum; the Luhn algorithm doubles alternating digits, sums the results after reducing any two-digit product to a single digit, and likewise checks the total's own divisibility by 10). In both cases, a single altered digit changes the weighted sum by a nonzero amount in the vast majority of cases, which is exactly why both algorithms reliably catch a single-digit transcription or forgery error without needing to compare against any external reference value at all -- the entire check is self-contained within the number's own digits.

**2.** A forger who obtains a genuine, valid routing number -- for instance, by photographing or purchasing a real blank check, or simply looking up any real bank's own publicly listed routing number -- and prints that identical, genuinely valid number onto a fraudulent check (with a fabricated account number, a forged signature, and an altered amount) would pass Section 26.1's own MICR checksum check perfectly, since the routing number itself is completely real and internally consistent; the fraud lies entirely in the OTHER fields the checksum check was never built to examine.

**3.** A plain, separate dedup pass already catches every EXACT invoice-number resubmission on its own, using a simple exact-string comparison that costs far less to compute than a full Levenschtein distance calculation across an entire paid-invoice history. Including the exact-match case inside `flag_duplicate` as well would not extend the fraud pattern actually caught -- the exact-match case was never THIS detector's own gap in the first place -- it would only duplicate work an existing, cheaper check already does completely, which is exactly why the section explicitly `continue`s past an exact match rather than flagging it redundantly.

**4.** This specific fraud pattern -- the same underlying invoice resubmitted with a cosmetically altered invoice number -- depends on the amount staying IDENTICAL, since the whole point of the fraud is billing for the identical real goods or services a second time; a fraudster attempting this pattern has no real reason to also alter the amount, and doing so would actually reduce the fraud's own plausibility to a reviewer expecting round-number consistency. Allowing an amount tolerance would instead risk flagging two entirely unrelated, legitimate invoices from the same vendor that happen to have similar invoice numbers and roughly similar (but not identical) amounts purely by coincidence -- a real false-positive risk an exact-match requirement avoids entirely.

**5.** Without the reset, `current_group` would still hold `42` (from "forty-two") at the moment "thousand" is processed, so the `total += current_group * 1000` step would still correctly add `42000` to `total` -- but critically, `current_group` would NOT be reset to 0 afterward, so it would still hold `42` going into "three hundred five." The subsequent "hundred" token would then multiply the STALE `42` by 100 to get `4200`, and adding "five" would produce a final `current_group` of `4205` instead of the intended `305`, making the function's own final total `1000*42 + 4205 = 46205` instead of the correct `42305` -- silently overcounting by exactly the un-reset group's own leftover value.

**6.** A phrase such as "One thousand and one dollars" -- a construction some real speakers and even some real printed forms use, placing "and" directly between a thousands group and a following units word rather than reserving "and" exclusively for the cents suffix -- would parse incorrectly under this section's own stated parser: `parse_dollar_words` treats every occurrence of the token "and" identically, by skipping it unconditionally regardless of where it appears, so "one thousand and one" would correctly produce `1001`. But a phrase using British-style grouping, such as "one thousand two hundred AND thirty-four" where "and" appears between the hundreds and tens groups (a genuinely common British convention this section's own US-check-convention parser was never built to expect), would still parse correctly too, since "and" is simply skipped everywhere -- the actual failure case is a phrase using a word this parser's own `UNITS` and `TENS` tables do not contain at all, such as "a hundred" (using the indefinite article "a" instead of the number word "one"), which `parse_dollar_words` would reject outright as an unrecognized token, exactly as Test 5's own "gazillion" case demonstrates.

**7.** A reviewer given only a total score of, say, 70 and a count of "2 signals fired" would know a rejection-level score was reached but would have no way to know WHICH real checks actually triggered it without re-running every check by hand against the same transaction -- precisely the information needed to decide whether to investigate a suspected duplicate invoice, chase down a signature discrepancy, or follow up on a specific amount mismatch. Recording each fired signal's own name preserves exactly the actionable detail a reviewer needs to act on the flag efficiently, rather than starting their own investigation from zero.

**8.** A bidirectional forcing rule would create a real path for a genuinely high-risk transaction (one that already independently crosses the reject threshold on its own real signals) to have its own disposition SOFTENED merely because ONE additional, unrelated check happened to pass cleanly -- for instance, a transaction with two serious fraud flags that would otherwise be auto-rejected could have its own MICR checksum happen to be perfectly valid, and if that success were allowed to downgrade the disposition, the two serious, independently-detected flags would be effectively cancelled out by one unrelated passing check. Restricting the forcing rule to only ever escalate (never downgrade) closes off that exploitable path entirely: a passing check can never make an otherwise-risky transaction look safer than its own worst individual signal already indicated.

**9.** A fixed 64-bit dHash has a fixed maximum possible Hamming distance of 64 regardless of the signature's own real-world size or complexity, so a tolerance value like 10 has a fixed, well-understood meaning as a FRACTION of that fixed 64-bit space. Levenshtein distance on an invoice number, by contrast, has no fixed maximum -- its own natural scale depends entirely on the invoice number's own length (a 5-character invoice number and a 20-character one need genuinely different absolute edit-distance tolerances to represent the same real "one or two characters changed" intent), which is exactly why Section 23.2's own merchant-name fuzzy match used an edit-distance RATIO relative to the string's own length rather than a fixed absolute value, while Section 26.2's own invoice-number check uses a small fixed absolute threshold specifically because real invoice numbers in this section's own stated scope vary little enough in length for a fixed small threshold to remain meaningful.

**10.** Section 26.3's amount-in-words cross-validation is built specifically to catch a value ALREADY PRINTED ON THE CHECK being altered after the fact (the classic check-washing pattern), detected by cross-checking two fields that should already agree on the same document. Section 26.1's MICR and signature checks validate properties of the document as originally presented, Section 26.2's duplicate detection catches an entirely new, separately-submitted fraudulent invoice rather than an alteration to an existing one, and Section 26.4's risk scoring combines signals rather than detecting a pattern of its own -- only Section 26.3's own redundant-field design gives it a structural way to catch a single field being changed in isolation, since the OTHER, unaltered field on the same document remains available as an independent real check against exactly that alteration.
