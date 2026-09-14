// 02_streaming_requantization.cpp
// Chapter 14, Part 2: Eviction (Chapter 13.4) is a binary decision -- a
// token is either in the cache or it is gone. Streaming re-quantization
// offers a gentler alternative: instead of evicting an aging token
// outright, compress it to a coarser bit-width. A token cached at 4-bit
// precision can be re-quantized to 3-bit, then 2-bit, as it ages past
// configurable thresholds, before eventual eviction -- graceful
// degradation instead of a hard cliff.
//
// The insight is that a token's influence on the current generation step
// is proportional to the attention weight it receives, and attention
// weight falls off sharply with distance. A token from 15,000 steps ago
// typically receives a vanishingly small attention weight, so a coarser
// (lossier) quantization of its key/value vectors introduces an error
// that gets multiplied by that tiny weight and has negligible effect on
// the output. A token from 10 steps ago receives a much larger weight,
// so it stays at high fidelity.
//
// This section uses a simplified symmetric uniform quantizer at multiple
// bit-widths rather than the full rotation-plus-codebook machinery of
// this book's own Chapter 7 (TurboQuant). A production system would
// re-quantize by dequantizing the current representation back to float
// and re-encoding with a coarser Lloyd-Max codebook from that chapter;
// the tiering policy below -- age thresholds, progressive compression,
// eventual eviction -- is identical either way, so a uniform quantizer
// keeps this file focused on the cache-management logic rather than
// re-deriving Chapter 7's codebook construction.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_streaming_requantization.cpp -o 02_streaming_requantization

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// A symmetric uniform quantizer at a chosen bit-width. Only the fields a
// consumer actually reads are kept: bits (for tier bookkeeping), scale
// (to dequantize), and the codes themselves. An earlier draft also
// stored the vector's norm, computed at quantize time but never read by
// anything -- that field is left out here.
struct MultiQuant {
    int dim;

    struct Quantized {
        int bits;
        float scale;
        std::vector<int8_t> codes;
    };

    explicit MultiQuant(int d) : dim(d) {}

    Quantized quantize(const float* x, int bits) const {
        Quantized q;
        q.bits = bits;
        q.codes.resize(dim);
        int max_int = (1 << (bits - 1)) - 1;
        float alpha = 0.0f;
        for (int i = 0; i < dim; ++i) alpha = std::max(alpha, std::fabs(x[i]));
        if (alpha < 1e-10f) {
            q.scale = 0.0f;
            std::fill(q.codes.begin(), q.codes.end(), int8_t{0});
            return q;
        }
        q.scale = alpha / max_int;
        float inv = max_int / alpha;
        for (int i = 0; i < dim; ++i) {
            q.codes[i] = static_cast<int8_t>(std::clamp(
                std::round(x[i] * inv),
                static_cast<float>(-max_int), static_cast<float>(max_int)));
        }
        return q;
    }

    void dequantize(const Quantized& q, float* out) const {
        for (int i = 0; i < dim; ++i)
            out[i] = static_cast<float>(q.codes[i]) * q.scale;
    }

    // Re-quantize from the current bit-width down to a coarser one: fully
    // dequantize, then re-encode at new_bits. The rotation-free simplified
    // quantizer used here makes this a plain float round-trip; the real
    // TurboQuant version follows the same dequantize/re-encode shape but
    // shares one rotation matrix across every bit-width and only swaps
    // the codebook.
    Quantized requantize(const Quantized& q, int new_bits) const {
        assert(new_bits <= q.bits);
        std::vector<float> tmp(dim);
        dequantize(q, tmp.data());
        return quantize(tmp.data(), new_bits);
    }

    size_t storage_bytes(const Quantized& q) const {
        return (static_cast<size_t>(q.bits) * dim + 7) / 8 + 4;  // codes + scale
    }
};

struct TieredEntry {
    int position;
    int insert_step;
    MultiQuant::Quantized key_q;
    MultiQuant::Quantized val_q;
};

struct TieredCache {
    int dim;
    int current_step;
    MultiQuant quantizer;
    int tier_3bit_age;
    int tier_2bit_age;
    int tier_evict_age;
    std::vector<TieredEntry> entries;

    TieredCache(int d, int t3, int t2, int te)
        : dim(d), current_step(0), quantizer(d),
          tier_3bit_age(t3), tier_2bit_age(t2), tier_evict_age(te) {}

    void insert(int pos, const float* key, const float* val) {
        TieredEntry e;
        e.position = pos;
        e.insert_step = current_step;
        e.key_q = quantizer.quantize(key, 4);
        e.val_q = quantizer.quantize(val, 4);
        entries.push_back(std::move(e));
    }

    struct MaintenanceStats {
        int requantized_to_3 = 0;
        int requantized_to_2 = 0;
        int evicted = 0;
    };

    MaintenanceStats maintain() {
        MaintenanceStats stats;
        for (auto& e : entries) {
            int age = current_step - e.insert_step;
            if (age >= tier_2bit_age && e.key_q.bits > 2) {
                e.key_q = quantizer.requantize(e.key_q, 2);
                e.val_q = quantizer.requantize(e.val_q, 2);
                ++stats.requantized_to_2;
            } else if (age >= tier_3bit_age && e.key_q.bits > 3) {
                e.key_q = quantizer.requantize(e.key_q, 3);
                e.val_q = quantizer.requantize(e.val_q, 3);
                ++stats.requantized_to_3;
            }
        }
        size_t before = entries.size();
        entries.erase(
            std::remove_if(entries.begin(), entries.end(),
                [&](const TieredEntry& e) {
                    return (current_step - e.insert_step) >= tier_evict_age;
                }),
            entries.end());
        stats.evicted = static_cast<int>(before - entries.size());
        return stats;
    }

    void advance_step() { ++current_step; }

    size_t total_bytes() const {
        size_t total = 0;
        for (const auto& e : entries) {
            total += quantizer.storage_bytes(e.key_q);
            total += quantizer.storage_bytes(e.val_q);
        }
        return total;
    }

    struct TierCounts { int at_4 = 0, at_3 = 0, at_2 = 0; };
    TierCounts tier_counts() const {
        TierCounts c;
        for (const auto& e : entries) {
            if (e.key_q.bits == 4) ++c.at_4;
            else if (e.key_q.bits == 3) ++c.at_3;
            else if (e.key_q.bits == 2) ++c.at_2;
        }
        return c;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 14.2: Streaming Re-Quantization -- Tiered Precision as Entries Age\n";
    std::cout << "========================================================\n";

    constexpr int DIM = 32;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    auto rv = [&] { std::vector<float> v(DIM); for (auto& x : v) x = dist(rng); return v; };

    std::cout << "\n-- Test 1: fresh tokens start at 4-bit --\n";
    {
        TieredCache cache(DIM, 100, 400, 1600);
        auto k = rv(), v = rv();
        cache.insert(0, k.data(), v.data());
        CHECK(cache.entries[0].key_q.bits == 4);
        std::cout << "  new token quantized at " << cache.entries[0].key_q.bits << "-bit\n";
    }

    std::cout << "\n-- Test 2: tokens age through the 4-bit -> 3-bit -> 2-bit -> evicted tiers --\n";
    {
        TieredCache cache(DIM, 10, 40, 160);
        for (int t = 0; t < 5; ++t) {
            auto k = rv(), v = rv();
            cache.insert(t, k.data(), v.data());
        }
        for (int s = 0; s < 15; ++s) cache.advance_step();
        auto stats1 = cache.maintain();
        auto tc1 = cache.tier_counts();
        CHECK(tc1.at_3 == 5);
        CHECK(stats1.requantized_to_3 == 5);
        std::cout << "  after 15 steps: " << tc1.at_4 << " at 4-bit, "
                   << tc1.at_3 << " at 3-bit, " << tc1.at_2 << " at 2-bit\n";

        for (int s = 0; s < 35; ++s) cache.advance_step();
        auto stats2 = cache.maintain();
        auto tc2 = cache.tier_counts();
        CHECK(tc2.at_2 == 5);
        CHECK(stats2.requantized_to_2 == 5);
        std::cout << "  after 50 steps: " << tc2.at_4 << " at 4-bit, "
                   << tc2.at_3 << " at 3-bit, " << tc2.at_2 << " at 2-bit\n";

        for (int s = 0; s < 120; ++s) cache.advance_step();
        auto stats3 = cache.maintain();
        CHECK(cache.entries.empty());
        CHECK(stats3.evicted == 5);
        std::cout << "  after 170 steps: " << cache.entries.size()
                   << " entries remain (evicted " << stats3.evicted << ")\n";
    }

    std::cout << "\n-- Test 3: tiering uses less memory than uniform 4-bit --\n";
    {
        TieredCache cache(DIM, 50, 200, 800);
        for (int s = 0; s < 500; ++s) {
            auto k = rv(), v = rv();
            cache.insert(s, k.data(), v.data());
            cache.advance_step();
            if (s % 50 == 49) cache.maintain();
        }
        cache.maintain();
        auto tc = cache.tier_counts();
        size_t actual_bytes = cache.total_bytes();
        auto sample = rv();
        size_t per_side_4bit = cache.quantizer.storage_bytes(cache.quantizer.quantize(sample.data(), 4));
        size_t uniform_4bit = cache.entries.size() * 2 * per_side_4bit;
        double savings_pct = (uniform_4bit > 0)
            ? (1.0 - static_cast<double>(actual_bytes) / static_cast<double>(uniform_4bit)) * 100.0
            : 0.0;
        std::cout << "  live entries: " << cache.entries.size() << " (4-bit=" << tc.at_4
                   << ", 3-bit=" << tc.at_3 << ", 2-bit=" << tc.at_2 << ")\n";
        std::cout << "  tiered memory: " << actual_bytes << " bytes\n";
        std::cout << "  uniform 4-bit equivalent: " << uniform_4bit << " bytes\n";
        std::cout << "  computed savings: " << std::fixed << std::setprecision(1) << savings_pct << "%\n";
        CHECK(actual_bytes < uniform_4bit);
        CHECK(tc.at_3 + tc.at_2 > 0);
    }

    std::cout << "\n-- Test 4: re-quantization error grows monotonically as bit-width drops --\n";
    {
        MultiQuant mq(DIM);
        auto original = rv();
        auto q4 = mq.quantize(original.data(), 4);
        auto q3 = mq.requantize(q4, 3);
        auto q2 = mq.requantize(q3, 2);
        std::vector<float> r4(DIM), r3(DIM), r2(DIM);
        mq.dequantize(q4, r4.data());
        mq.dequantize(q3, r3.data());
        mq.dequantize(q2, r2.data());
        auto mse = [&](const std::vector<float>& a, const std::vector<float>& b) {
            float s = 0.0f;
            for (int i = 0; i < DIM; ++i) { float e = a[i] - b[i]; s += e * e; }
            return s / DIM;
        };
        float mse4 = mse(original, r4);
        float mse3 = mse(original, r3);
        float mse2 = mse(original, r2);
        std::cout << "  4-bit MSE = " << std::scientific << std::setprecision(3) << mse4 << "\n";
        std::cout << "  3-bit MSE = " << mse3 << "\n";
        std::cout << "  2-bit MSE = " << mse2 << "\n";
        CHECK(mse4 < mse3);
        CHECK(mse3 < mse2);
        std::cout << std::fixed;
        std::cout << "  error grows as bit-width drops, as expected; old tokens receive\n";
        std::cout << "  small attention weights, so this error's effect on the output stays small\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
