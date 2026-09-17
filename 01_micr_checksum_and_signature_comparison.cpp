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
