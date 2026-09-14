# Chapter 14: Advanced KV Cache Management -- Sliding Windows, Streaming Re-Quantization, and Prefix Caching

**What you will understand by the end of this chapter:**

- Sliding-window attention as the same two-counter ring buffer mechanism Chapter 13.3 introduced, but with the sink-token protected region removed entirely — the window boundary itself becomes the eviction policy, with no case-by-case judgment about which tokens matter.
- Streaming re-quantization: compressing aging cache entries to progressively coarser bit-widths instead of evicting them outright, and why a token's shrinking attention weight as it ages is what makes the resulting quantization error harmless to the output.
- Prefix caching for multi-turn conversations: reusing a previous turn's KV cache entries instead of recomputing an unchanged shared history, and why a KVBlock's fixed granularity means a block straddling the point where two turns diverge must be recomputed in full even though part of it was genuinely shared.
- Why combining independently-correct techniques can surface a NEW form of waste that none of them showed alone — specifically, that prefix caching can only reuse what the runtime cache still holds, and aggressive eviction from earlier sections can silently shrink prefix caching's own benefit.

**What you need to know first:**

- Chapter 13.3's Ring Buffer: the two-counter design (a wrapping physical write slot and a monotonically increasing logical position) that Section 14.1 reuses directly, minus its sink-token protection.
- Chapter 13.4's H2O eviction policy — attention-probability-driven importance scoring with a recent-window immunity — which Section 14.4's capstone reuses as one of several combined eviction mechanisms.
- This book's own Chapter 7 (TurboQuant), which Section 14.2 references by name for the production version of the multi-bit-width re-quantization it demonstrates with a simplified stand-in quantizer.
- This is a purely single-threaded, data-structure-focused chapter, continuing Chapters 12 and 13's simplification: no `-pthread`, no `std::mdspan`, no `-ffp-contract=off` for any file here — every file compiles with just `-std=c++23 -Wall -Wextra -O2`.

---

Chapter 13 gave every future chapter a KV cache that pages efficiently, wraps without corrupting positional information, and evicts by measured usefulness rather than blind age. This chapter extends that foundation in exactly the three directions Chapter 13 did not cover, plus a capstone that ties all four techniques together. Section 14.1 applies Chapter 13.3's ring-buffer mechanism at the scale a real long-context model actually needs: a hard window of thousands of tokens, with the window boundary itself standing in for any case-by-case eviction judgment. Section 14.2 offers a gentler alternative to hard eviction: instead of a token either being in the cache or gone, it is compressed to a coarser bit-width as it ages, trading precision for memory in a way that costs the output almost nothing because old tokens receive vanishingly small attention weight anyway. Section 14.3 tackles a problem earlier chapters never addressed at all — a multi-turn conversation recomputing its own unchanged history on every single turn — with an entirely original prefix-caching design, since this chapter's own source material describes the idea only in prose with an unverified speedup figure and no accompanying tested code. Section 14.4 closes the chapter by combining sliding windows, re-quantization, prefix caching, and Chapter 13.4's H2O eviction into one complete cache manager, and — following Chapter 13.5's precedent that combining verified pieces is a new claim requiring its own verification — surfaces a genuinely new interaction: prefix caching's benefit depends on what the runtime cache still retains, and the tokens most likely to have been evicted are exactly the ones deepest in a long shared prefix.

## 14.1 Sliding-Window Attention as a Ring Buffer at Scale

### Intuition

Chapter 13.3's Ring Buffer capped memory by wrapping a fixed-size buffer and protected a handful of sink tokens from that wrap. Sliding-window attention is the same wrapping mechanism with that protection removed: nothing is exempt, and the window's edge is the entire eviction policy.

### The Concept, In Detail

A sliding window keeps exactly the most recent `W` tokens and never attends to anything older, full stop — unlike H2O, which decides case by case whether a given token still matters, sliding window makes no such judgment at all. The implementation is Chapter 13.3's two-counter ring buffer verbatim, minus the protected region: `current_pos` counts every token ever inserted and never wraps, while the physical storage slot for a given position is `position % window_size`, so the valid attention range is always `[max(0, current_pos - window_size), current_pos)`. This is not merely a memory-saving approximation retrofitted onto models that were never designed for it — Mistral 7B was explicitly TRAINED with a 4096-token sliding window, meaning the model learned during training to compress anything it might need beyond that window into its hidden state rather than relying on attention to reach arbitrarily far back. For models trained with full, unrestricted attention instead, sliding window is only an approximation, but one that tends to work well in practice regardless, because attention weight empirically falls off sharply with distance: tokens thousands of positions back typically receive negligible attention whether or not a hard window forces them out.

### Code and Verification

```cpp
// 01_sliding_window_cache.cpp
// Chapter 14, Part 1: Chapter 13 solved fragmentation (PagedAttention)
// and gave the ring buffer's wrapping-slot-vs-logical-position lesson in
// the specific context of protecting a few sink tokens. This section
// applies that same ring-buffer idea at the scale a real long-context
// model actually needs: a hard window of W tokens (typically thousands),
// beyond which the model simply never attends at all. Unlike Chapter
// 13.4's H2O, which decides case by case which tokens are important
// enough to keep, sliding window attention makes no such judgment --
// the window boundary itself IS the eviction policy, and nothing older
// than W positions back is ever read again, whether or not it mattered.
//
// This is not merely a memory-saving approximation bolted onto models
// that were never designed for it. Mistral 7B was explicitly TRAINED
// with a 4096-token sliding window, meaning the model learned to
// compress everything it might need beyond that window into its hidden
// state rather than relying on attention to reach back arbitrarily far.
// For models trained with full attention instead, sliding window is an
// approximation that tends to work well in practice anyway, because
// attention weight empirically falls off sharply with distance -- tokens
// thousands of positions back typically receive negligible attention
// regardless of whether the window forces them out.
//
// The implementation is exactly Chapter 13.3's ring buffer with the
// protected-sink-region logic removed: slot = position % W, insertion is
// O(1), attention iterates only over the currently valid W-token range,
// and memory never grows past W entries no matter how long generation
// continues.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_sliding_window_cache.cpp -o 01_sliding_window_cache

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

// A fixed-size ring buffer of exactly window_size entries. current_pos
// counts every token ever inserted (monotonically increasing, exactly
// like Chapter 13.3's rope_position); the valid attention range is
// always [max(0, current_pos - window_size), current_pos).
struct SlidingWindowCache {
    int window_size;
    int head_dim;
    int current_pos;
    std::vector<float> keys;
    std::vector<float> values;

    SlidingWindowCache(int ws, int hd)
        : window_size(ws), head_dim(hd), current_pos(0),
          keys(static_cast<size_t>(ws) * hd, 0.0f), values(static_cast<size_t>(ws) * hd, 0.0f) {}

    void insert(const float* k, const float* v) {
        int slot = current_pos % window_size;
        std::copy_n(k, head_dim, keys.data() + slot * head_dim);
        std::copy_n(v, head_dim, values.data() + slot * head_dim);
        ++current_pos;
    }

    int start_pos() const { return std::max(0, current_pos - window_size); }
    int end_pos() const { return current_pos; }
    int active_tokens() const { return end_pos() - start_pos(); }

    const float* key_at(int pos) const {
        assert(pos >= start_pos() && pos < end_pos());
        return keys.data() + (pos % window_size) * head_dim;
    }
    const float* value_at(int pos) const {
        assert(pos >= start_pos() && pos < end_pos());
        return values.data() + (pos % window_size) * head_dim;
    }

    size_t memory_bytes() const { return 2 * static_cast<size_t>(window_size) * head_dim * sizeof(float); }
};

float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 14.1: Sliding-Window Attention as a Ring Buffer at Scale\n";
    std::cout << "========================================================\n";

    constexpr int HEAD_DIM = 16;
    constexpr int WINDOW = 8;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    auto rv = [&] { std::vector<float> v(HEAD_DIM); for (auto& x : v) x = dist(rng); return v; };

    std::cout << "\n-- Test 1: within the window, every token remains fully accessible --\n";
    {
        SlidingWindowCache c(WINDOW, HEAD_DIM);
        std::vector<std::vector<float>> stored_keys;
        for (int t = 0; t < 5; ++t) {
            auto k = rv(), v = rv();
            stored_keys.push_back(k);
            c.insert(k.data(), v.data());
        }
        CHECK(c.active_tokens() == 5);
        CHECK(c.start_pos() == 0);
        CHECK(c.end_pos() == 5);
        bool match = true;
        for (int t = 0; t < 5; ++t) {
            const float* k = c.key_at(t);
            for (int i = 0; i < HEAD_DIM; ++i)
                if (std::fabs(k[i] - stored_keys[t][i]) > 1e-6f) match = false;
        }
        CHECK(match);
        std::cout << "  5 tokens inserted into an 8-slot window: all 5 accessible and byte-exact\n";
    }

    std::cout << "\n-- Test 2: beyond the window, the oldest tokens are simply gone --\n";
    {
        SlidingWindowCache c(WINDOW, HEAD_DIM);
        for (int t = 0; t < 12; ++t) {
            auto k = rv(), v = rv();
            c.insert(k.data(), v.data());
        }
        CHECK(c.current_pos == 12);
        CHECK(c.active_tokens() == 8);
        CHECK(c.start_pos() == 4);
        CHECK(c.end_pos() == 12);
        std::cout << "  12 tokens into an 8-slot window: visible range is [" << c.start_pos()
                   << ", " << c.end_pos() << "), tokens 0-3 are unrecoverable\n";
    }

    std::cout << "\n-- Test 3: memory is exactly O(window_size), independent of sequence length --\n";
    {
        SlidingWindowCache c(WINDOW, HEAD_DIM);
        size_t mem_before = c.memory_bytes();
        for (int t = 0; t < 1000; ++t) {
            auto k = rv(), v = rv();
            c.insert(k.data(), v.data());
        }
        size_t mem_after = c.memory_bytes();
        CHECK(mem_before == mem_after);
        std::cout << "  after 1000 tokens: memory = " << mem_after << " bytes, unchanged since token 0\n";
    }

    std::cout << "\n-- Test 4: attention over the window computes a valid, well-formed output --\n";
    {
        SlidingWindowCache c(WINDOW, HEAD_DIM);
        for (int t = 0; t < 20; ++t) {
            auto k = rv(), v = rv();
            c.insert(k.data(), v.data());
        }
        auto query = rv();
        float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        int n = c.active_tokens();
        std::vector<float> scores(n);
        for (int i = 0; i < n; ++i) {
            int pos = c.start_pos() + i;
            scores[i] = dot(query.data(), c.key_at(pos), HEAD_DIM) * scale;
        }
        float mx = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (float& s : scores) { s = std::exp(s - mx); sum += s; }
        for (float& s : scores) s /= sum;
        std::vector<float> out(HEAD_DIM, 0.0f);
        for (int i = 0; i < n; ++i) {
            int pos = c.start_pos() + i;
            const float* v = c.value_at(pos);
            for (int j = 0; j < HEAD_DIM; ++j) out[j] += scores[i] * v[j];
        }
        float norm_sq = 0.0f;
        for (float x : out) norm_sq += x * x;
        std::cout << "  attention over window [" << c.start_pos() << ", " << c.end_pos() << ") = " << n << " tokens\n";
        std::cout << "  output norm = " << std::sqrt(norm_sq) << "\n";
        CHECK(std::sqrt(norm_sq) > 0.0f);
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_sliding_window_cache.cpp -o 01_sliding_window_cache
./01_sliding_window_cache
```

**Sample input:** 5 tokens inserted into an 8-slot window, checked byte-exact against every stored original; 12 tokens into the same 8-slot window, checked to confirm exactly the correct 8 remain visible and the oldest 4 are gone; memory measured before and after 1000 tokens to confirm it stays exactly constant; and a full attention computation over a 20-token stream's current window checked to produce a valid, non-zero-norm output.

```text
========================================================
Chapter 14.1: Sliding-Window Attention as a Ring Buffer at Scale
========================================================

-- Test 1: within the window, every token remains fully accessible --
  5 tokens inserted into an 8-slot window: all 5 accessible and byte-exact

-- Test 2: beyond the window, the oldest tokens are simply gone --
  12 tokens into an 8-slot window: visible range is [4, 12), tokens 0-3 are unrecoverable

-- Test 3: memory is exactly O(window_size), independent of sequence length --
  after 1000 tokens: memory = 1024 bytes, unchanged since token 0

-- Test 4: attention over the window computes a valid, well-formed output --
  attention over window [12, 20) = 8 tokens
  output norm = 0.390779

========================================================
10/10 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] treating sliding window as free correctness for a full-attention-trained model"
    Mistral 7B's sliding window is not an approximation at all for that specific model — the model was trained to expect exactly this restriction and learned to compress older context into its hidden state accordingly, so applying the window changes nothing about the model's own expectations. Applying the identical mechanism to a model trained with full, unrestricted attention is a genuinely different situation: that model never learned to compress anything into its hidden state, because during training it could always attend arbitrarily far back. The fact that this approximation tends to work well in practice — because attention weight empirically decays with distance regardless of training regime — is an empirical property of typical attention distributions, not a guarantee, and a task that genuinely depends on precise recall of something far outside the window (a specific number stated thousands of tokens earlier, for instance) can fail silently: no crash, no warning, just an answer computed as though that information were never there at all.

## 14.2 Streaming Re-Quantization: Tiered Precision as Entries Age

### Intuition

Eviction, whether FIFO or H2O, is a binary decision: a token is either fully present in the cache or entirely gone. Streaming re-quantization offers a middle path — instead of discarding an aging token outright, compress it to a coarser bit-width, so the cache degrades gracefully rather than falling off a cliff.

### The Concept, In Detail

A freshly inserted token starts at high-fidelity 4-bit quantization; as it ages past configurable thresholds it is progressively re-quantized to 3-bit, then 2-bit, before eventual eviction — a tiered cache where recent tokens keep high-fidelity Key/Value vectors and old tokens hold increasingly coarse ones. The insight that makes this safe rather than merely convenient is that a token's actual influence on the CURRENT generation step is proportional to the attention weight it receives, and that weight shrinks the older a token gets (the same empirical decay Section 14.1 leaned on). A quantization error introduced into a token's Key or Value vector gets multiplied by that token's attention weight before it ever reaches the output, so a coarse quantization of a barely-attended-to old token produces an error that is itself barely-attended-to, while the SAME coarse quantization applied to a heavily-weighted recent token would visibly corrupt the output — which is exactly why the tiering is age-based rather than uniform. This section implements the mechanism with a simplified symmetric uniform quantizer at each bit-width rather than this book's own Chapter 7 (TurboQuant) machinery: a production system would re-quantize by dequantizing the current representation back to float and re-encoding with a coarser Lloyd-Max codebook from that chapter, sharing one rotation matrix across every bit-width and swapping only the codebook, but the cache-management policy — age thresholds, progressive compression, eventual eviction — is identical either way, so a uniform quantizer keeps this file's focus on that policy rather than re-deriving Chapter 7's codebook construction.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_streaming_requantization.cpp -o 02_streaming_requantization
./02_streaming_requantization
```

**Sample input:** a freshly inserted token checked to start at 4-bit; five tokens aged through 15, 50, and 170 simulated steps against thresholds of 10/40/160, checked to land at 3-bit, then 2-bit, then fully evicted at each stage; a 500-step simulation with periodic maintenance checked to use less total memory than a uniform 4-bit cache of the same live entries; and a single vector's reconstruction error checked to increase monotonically as it is re-quantized from 4-bit down through 3-bit to 2-bit.

```text
========================================================
Chapter 14.2: Streaming Re-Quantization -- Tiered Precision as Entries Age
========================================================

-- Test 1: fresh tokens start at 4-bit --
  new token quantized at 4-bit

-- Test 2: tokens age through the 4-bit -> 3-bit -> 2-bit -> evicted tiers --
  after 15 steps: 0 at 4-bit, 5 at 3-bit, 0 at 2-bit
  after 50 steps: 0 at 4-bit, 0 at 3-bit, 5 at 2-bit
  after 170 steps: 0 entries remain (evicted 5)

-- Test 3: tiering uses less memory than uniform 4-bit --
  live entries: 500 (4-bit=49, 3-bit=150, 2-bit=301)
  tiered memory: 13984 bytes
  uniform 4-bit equivalent: 20000 bytes
  computed savings: 30.1%

-- Test 4: re-quantization error grows monotonically as bit-width drops --
  4-bit MSE = 5.237e-03
  3-bit MSE = 2.583e-02
  2-bit MSE = 1.887e-01
  error grows as bit-width drops, as expected; old tokens receive
  small attention weights, so this error's effect on the output stays small

========================================================
11/11 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming a tiering scheme's claimed memory savings without running it"
    A tiering scheme's savings depend on exactly how many live entries land in each tier at the moment memory is measured, which in turn depends on the age-threshold parameters, how often maintenance actually runs, and how long the cache has been running when the measurement is taken — none of which can be derived from the tier table alone without actually simulating the policy. A back-of-envelope estimate assuming a specific, round distribution across tiers (for instance, assuming exactly evenly-spaced buckets of tokens at each bit-width) will not generally match what a real, running cache produces, because real insertion and maintenance timing rarely aligns with round numbers — this chapter's own honestly-computed savings figure, obtained by actually running the maintenance loop over 500 simulated steps, differs from a simpler static-bucket estimate for exactly this reason, and the book's standing rule is to report the number the code actually produced rather than the more convenient one a table suggests.

## 14.3 Prefix Caching for Multi-Turn Conversations

### Intuition

Every new turn in a multi-turn conversation shares a long prefix with the previous one: the system prompt, the conversation history, and the formatting tokens the chat template inserts. Recomputing the KV cache for that entire shared prefix on every single turn is pure waste — the point of a KV cache is that it is a checkpoint of the model's state, and resuming from a checkpoint should be free.

### The Concept, In Detail

This section's source material describes prefix caching only in prose, with a single narrative "speedup: 501x" figure and no accompanying tested code anywhere — this book's standing rule is to never carry forward a performance claim that was not itself computed and checked, so what follows is an original design built on Chapter 13.2's `BlockManager` and `Sequence` rather than a transcription of that figure. The mechanism compares the OLD turn's token-ID sequence against the NEW one element by element to find their longest common prefix (LCP); everything before the first mismatch is genuinely shared history, and everything from the mismatch onward is new or changed and must be recomputed. The subtlety is that a `KVBlock` is the smallest unit PagedAttention can reuse, so only WHOLE blocks that lie entirely within the matched prefix — `floor(lcp / BLOCK_SIZE)` of them — can actually be reused; a block that straddles the divergence point, partly inside the shared prefix and partly past it, must be recomputed in full even though part of its contents were technically shared, because there is no way to reuse "part of a block." This block-granularity waste is a genuine, permanent property of the design rather than an oversight to be optimized away: a conversation edit that diverges at token 25 inside a 16-token-block scheme reuses only the FIRST fully-contained block (tokens 0-15), discarding tokens 16-24 even though they matched, purely because they shared a block with tokens that did not.

### Code and Verification

```cpp
// 03_prefix_caching.cpp
// Chapter 14, Part 3: In a multi-turn conversation, each new user message
// is appended to a growing history that already includes the system
// prompt and every prior turn. Re-running the model over that entire
// shared history on every single turn is pure waste -- the KV cache
// computed for turn N is a checkpoint of the model's state after
// processing that history, and turn N+1 can simply resume from it rather
// than recomputing it from scratch.
//
// This section is original: the source material for this chapter
// describes prefix caching only in prose, with a single narrative
// "speedup: 501x" figure and no accompanying tested code anywhere. This
// book's standing rule is to never carry forward a performance number
// that was not itself computed and checked, so rather than repeat that
// figure, this file builds a real prefix cache on top of Chapter 13.2's
// BlockManager/Sequence and honestly measures its own savings.
//
// The mechanism: keep the previous turn's block table intact instead of
// releasing it, compute the longest common prefix (LCP) between the old
// and new token-ID sequences, and reuse every block that lies ENTIRELY
// within that shared prefix. A block is the smallest unit PagedAttention
// can reuse, so a block that straddles the divergence point -- partly
// inside the shared prefix, partly past it -- must be recomputed in
// full even though part of its contents were technically shared. That
// block-granularity waste is a genuine, permanent property of the
// design, not a bug to be optimized away, and Test 2 below demonstrates
// it directly.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_prefix_caching.cpp -o 03_prefix_caching

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <stdexcept>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// ---- Reused verbatim from Chapter 13.2 (BlockManager / KVBlock / Sequence) ----

constexpr int BLOCK_SIZE = 16;
constexpr int HEAD_DIM = 32;
constexpr int NUM_HEADS = 1;

struct KVBlock {
    float keys[NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
    float values[NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
    void clear() { std::memset(this, 0, sizeof(*this)); }
};

class BlockManager {
public:
    explicit BlockManager(int num_blocks) {
        m_pool.resize(num_blocks);
        for (int i = 0; i < num_blocks; ++i) m_free.push_back(i);
    }

    int alloc() {
        if (m_free.empty()) throw std::runtime_error("KV cache OOM");
        int id = m_free.front();
        m_free.pop_front();
        m_pool[id].clear();
        return id;
    }

    void free(int id) {
        assert(id >= 0 && id < static_cast<int>(m_pool.size()));
        m_free.push_back(id);
    }

    KVBlock* get(int id) {
        assert(id >= 0 && id < static_cast<int>(m_pool.size()));
        return &m_pool[id];
    }
    const KVBlock* get(int id) const { return &m_pool[id]; }

    int free_count() const { return static_cast<int>(m_free.size()); }

private:
    std::vector<KVBlock> m_pool;
    std::list<int> m_free;
};

struct BlockAddr { int block_table_idx; int slot; };
BlockAddr translate(int token_idx) { return {token_idx >> 4, token_idx & 0x0F}; }

struct Sequence {
    std::vector<int> block_table;
    int context_len = 0;

    void store_kv(BlockManager& mgr, const float* key, const float* value, int head = 0) {
        int slot = context_len % BLOCK_SIZE;
        if (slot == 0) block_table.push_back(mgr.alloc());
        KVBlock* blk = mgr.get(block_table.back());
        for (int d = 0; d < HEAD_DIM; ++d) {
            blk->keys[head][slot][d] = key[d];
            blk->values[head][slot][d] = value[d];
        }
        ++context_len;
    }

    const float* read_key(const BlockManager& mgr, int token_idx, int head = 0) const {
        auto addr = translate(token_idx);
        return mgr.get(block_table[addr.block_table_idx])->keys[head][addr.slot];
    }
    const float* read_val(const BlockManager& mgr, int token_idx, int head = 0) const {
        auto addr = translate(token_idx);
        return mgr.get(block_table[addr.block_table_idx])->values[head][addr.slot];
    }

    int num_blocks() const { return static_cast<int>(block_table.size()); }
};

// ---- New for this section: prefix matching and block-granularity reuse ----

// Compares two token-ID sequences element by element and returns how many
// leading tokens match. This is the recomputation boundary: everything
// before it is shared history, everything from it onward is new or
// changed and must be (re)computed.
int longest_common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
    int n = static_cast<int>(std::min(a.size(), b.size()));
    int i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

// A deterministic, distinguishable stand-in for "the model computed a
// real K/V vector for this token." Different token IDs always produce
// different vectors, so any test comparing vectors is really comparing
// which token's KV data ended up where.
void gen_kv(int token_id, float* k, float* v) {
    for (int d = 0; d < HEAD_DIM; ++d) {
        k[d] = static_cast<float>(token_id) * 1000.0f + static_cast<float>(d);
        v[d] = static_cast<float>(token_id) * 2000.0f + static_cast<float>(d);
    }
}

// One conversation's cache: the token IDs processed so far, and the
// paged Sequence holding their KV entries. apply_new_turn is where
// prefix caching actually happens.
struct ConversationCache {
    std::vector<int> token_ids;
    Sequence seq;

    struct TurnStats {
        int lcp = 0;                 // tokens that literally matched, element by element
        int reusable_blocks = 0;     // whole blocks entirely inside the LCP
        int reused_tokens = 0;       // reusable_blocks * BLOCK_SIZE
        int recomputed_tokens = 0;   // new_ids.size() - reused_tokens
        int blocks_freed = 0;        // stale blocks released back to the pool
    };

    // Given the FULL new token-ID sequence for this turn (old history plus
    // whatever changed or was appended), reuse every block that lies
    // entirely within the shared prefix and recompute the rest.
    TurnStats apply_new_turn(BlockManager& mgr, const std::vector<int>& new_ids) {
        TurnStats stats;
        stats.lcp = longest_common_prefix(token_ids, new_ids);
        stats.reusable_blocks = stats.lcp / BLOCK_SIZE;  // floor: partial blocks can't be reused
        stats.reused_tokens = stats.reusable_blocks * BLOCK_SIZE;

        // Free every block from the reuse boundary onward -- including a
        // block that straddled the divergence point, since a KVBlock is
        // the smallest unit of reuse and cannot be reused partially.
        for (int i = stats.reusable_blocks; i < seq.num_blocks(); ++i) {
            mgr.free(seq.block_table[i]);
            ++stats.blocks_freed;
        }
        seq.block_table.resize(stats.reusable_blocks);
        seq.context_len = stats.reused_tokens;
        token_ids.resize(stats.reused_tokens);

        // Recompute everything from the reuse boundary to the end of the
        // new sequence. In a real engine this is a forward pass through
        // the model; here gen_kv() stands in for "the model computed it."
        for (int t = stats.reused_tokens; t < static_cast<int>(new_ids.size()); ++t) {
            float k[HEAD_DIM], v[HEAD_DIM];
            gen_kv(new_ids[t], k, v);
            seq.store_kv(mgr, k, v);
            token_ids.push_back(new_ids[t]);
        }
        stats.recomputed_tokens = static_cast<int>(new_ids.size()) - stats.reused_tokens;
        return stats;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 14.3: Prefix Caching for Multi-Turn Conversations\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: block-aligned full-prefix reuse, zero waste --\n";
    {
        BlockManager mgr(64);
        ConversationCache conv;
        std::vector<int> turn1;
        for (int i = 0; i < 48; ++i) turn1.push_back(100 + i);  // 48 = exactly 3 blocks

        // Bootstrap turn 1 as if it were itself a "new turn" against an
        // empty history: everything is recomputed, nothing reused yet.
        auto boot = conv.apply_new_turn(mgr, turn1);
        CHECK(boot.reused_tokens == 0);
        CHECK(boot.recomputed_tokens == 48);
        std::vector<int> old_block_table = conv.seq.block_table;

        std::vector<int> turn2 = turn1;
        for (int i = 0; i < 20; ++i) turn2.push_back(500 + i);  // append 20 new tokens

        auto stats = conv.apply_new_turn(mgr, turn2);
        CHECK(stats.lcp == 48);
        CHECK(stats.reusable_blocks == 3);
        CHECK(stats.reused_tokens == 48);
        CHECK(stats.recomputed_tokens == 20);
        CHECK(stats.blocks_freed == 0);  // nothing straddled the boundary -- nothing to discard
        bool same_blocks = std::equal(old_block_table.begin(), old_block_table.end(),
                                       conv.seq.block_table.begin());
        CHECK(same_blocks);
        double naive_cost = static_cast<double>(turn2.size());
        double actual_cost = static_cast<double>(stats.recomputed_tokens);
        double speedup = naive_cost / actual_cost;
        std::cout << "  turn1=48 tokens (3 blocks), turn2 appends 20 tokens\n";
        std::cout << "  reused=" << stats.reused_tokens << " recomputed=" << stats.recomputed_tokens
                   << " blocks_freed=" << stats.blocks_freed << "\n";
        std::cout << "  physical block IDs for the reused prefix are unchanged (zero-copy)\n";
        std::cout << "  naive full recompute: " << static_cast<int>(naive_cost)
                   << " tokens; with prefix caching: " << static_cast<int>(actual_cost)
                   << " tokens (" << std::fixed << std::setprecision(2) << speedup << "x)\n";
    }

    std::cout << "\n-- Test 2: [COMMON TRAP] block-granularity waste on a misaligned edit --\n";
    {
        BlockManager mgr(64);
        ConversationCache conv2;
        std::vector<int> turn1;
        for (int i = 0; i < 40; ++i) turn1.push_back(i);  // ids 0..39, spans blocks 0,1,2
        conv2.apply_new_turn(mgr, turn1);

        // The user edits their message: everything up to token 25 is
        // identical, but token 25 onward is different content, and the
        // new turn is 45 tokens long overall.
        std::vector<int> turn2(turn1.begin(), turn1.begin() + 25);
        for (int i = 0; i < 20; ++i) turn2.push_back(900 + i);

        auto stats2 = conv2.apply_new_turn(mgr, turn2);
        CHECK(stats2.lcp == 25);               // 25 tokens genuinely matched
        CHECK(stats2.reusable_blocks == 1);    // but block 1 (tokens 16-31) is NOT entirely inside [0,25)
        CHECK(stats2.reused_tokens == 16);     // so only block 0 (tokens 0-15) is reused
        int wasted_shared_tokens = stats2.lcp - stats2.reused_tokens;
        CHECK(wasted_shared_tokens == 9);      // tokens 16-24 matched but got recomputed anyway
        CHECK(stats2.recomputed_tokens == 29); // 45 - 16
        CHECK(stats2.blocks_freed == 2);       // the straddling block and the fully-stale block 2

        std::cout << "  turn1=40 tokens, turn2 diverges at token 25, new length=45\n";
        std::cout << "  matched prefix (LCP) = " << stats2.lcp << " tokens\n";
        std::cout << "  actually reusable = " << stats2.reused_tokens << " tokens ("
                   << stats2.reusable_blocks << " whole block)\n";
        std::cout << "  wasted shared tokens = " << wasted_shared_tokens
                   << " (matched, but recomputed anyway because their block wasn't fully shared)\n";
    }

    std::cout << "\n-- Test 3: reused blocks' KV data matches the original, byte for byte --\n";
    {
        // Re-run test 2's exact scenario and directly compare the raw
        // bytes of the surviving block against what turn 1 wrote there.
        BlockManager mgr(64);
        ConversationCache conv;
        std::vector<int> turn1;
        for (int i = 0; i < 40; ++i) turn1.push_back(i);
        conv.apply_new_turn(mgr, turn1);
        int surviving_block_id = conv.seq.block_table[0];
        KVBlock snapshot = *mgr.get(surviving_block_id);

        std::vector<int> turn2(turn1.begin(), turn1.begin() + 25);
        for (int i = 0; i < 20; ++i) turn2.push_back(900 + i);
        conv.apply_new_turn(mgr, turn2);

        CHECK(conv.seq.block_table[0] == surviving_block_id);  // same physical block, never touched
        KVBlock after = *mgr.get(surviving_block_id);
        bool byte_identical = std::memcmp(&snapshot, &after, sizeof(KVBlock)) == 0;
        CHECK(byte_identical);
        std::cout << "  block " << surviving_block_id
                   << " untouched by apply_new_turn(): memcmp of all "
                   << sizeof(KVBlock) << " bytes is identical\n";
    }

    std::cout << "\n-- Test 4: the recomputed suffix is correct, not leftover stale data --\n";
    {
        // Rebuild Test 2's exact scenario (its own BlockManager already
        // went out of scope) and check the recomputed region directly.
        BlockManager mgr(64);
        ConversationCache conv;
        std::vector<int> turn1;
        for (int i = 0; i < 40; ++i) turn1.push_back(i);
        conv.apply_new_turn(mgr, turn1);
        std::vector<int> turn2(turn1.begin(), turn1.begin() + 25);
        for (int i = 0; i < 20; ++i) turn2.push_back(900 + i);
        conv.apply_new_turn(mgr, turn2);

        bool suffix_ok = true;
        for (int t = conv.seq.context_len - static_cast<int>(20); t < conv.seq.context_len; ++t) {
            float expect_k[HEAD_DIM], expect_v[HEAD_DIM];
            gen_kv(turn2[t], expect_k, expect_v);
            const float* got_k = conv.seq.read_key(mgr, t);
            const float* got_v = conv.seq.read_val(mgr, t);
            for (int d = 0; d < HEAD_DIM; ++d) {
                if (got_k[d] != expect_k[d] || got_v[d] != expect_v[d]) suffix_ok = false;
            }
        }
        CHECK(suffix_ok);
        // And the reused prefix (tokens 0-15) still reads back correctly too.
        bool prefix_ok = true;
        for (int t = 0; t < 16; ++t) {
            float expect_k[HEAD_DIM], expect_v[HEAD_DIM];
            gen_kv(turn1[t], expect_k, expect_v);
            const float* got_k = conv.seq.read_key(mgr, t);
            for (int d = 0; d < HEAD_DIM; ++d) if (got_k[d] != expect_k[d]) prefix_ok = false;
        }
        CHECK(prefix_ok);
        std::cout << "  recomputed suffix (last 20 tokens) matches the NEW token IDs' KV data\n";
        std::cout << "  reused prefix (first 16 tokens) still matches the ORIGINAL token IDs' KV data\n";
    }

    std::cout << "\n-- Test 5: realistic long-history conversation, honestly-computed savings --\n";
    {
        BlockManager mgr(512);
        ConversationCache conv;
        std::vector<int> history;
        for (int i = 0; i < 2048; ++i) history.push_back(i);  // system prompt + prior turns
        conv.apply_new_turn(mgr, history);

        // The next turn keeps the entire history unchanged and appends 48
        // new tokens -- the common case of a straightforward reply.
        std::vector<int> next_turn = history;
        for (int i = 0; i < 48; ++i) next_turn.push_back(90000 + i);

        auto stats = conv.apply_new_turn(mgr, next_turn);
        CHECK(stats.reused_tokens == 2048);   // 2048 is block-aligned (2048 / 16 = 128 exactly)
        CHECK(stats.recomputed_tokens == 48);
        double total_len = static_cast<double>(next_turn.size());
        double saved_tokens = total_len - stats.recomputed_tokens;
        double savings_pct = 100.0 * saved_tokens / total_len;
        std::cout << "  history=2048 tokens, next turn appends 48 tokens (total " << next_turn.size() << ")\n";
        std::cout << "  tokens recomputed: " << stats.recomputed_tokens << " / " << next_turn.size() << "\n";
        std::cout << "  tokens saved by prefix caching: " << std::fixed << std::setprecision(1)
                   << saved_tokens << " (" << savings_pct << "% of this turn's total work avoided)\n";
        CHECK(savings_pct > 90.0);
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_prefix_caching.cpp -o 03_prefix_caching
./03_prefix_caching
```

**Sample input:** a block-aligned 48-token history extended by 20 new tokens, checked for zero-waste full-prefix reuse and an honestly-computed speedup ratio; a 40-token history diverging at token 25 with a 45-token new turn, checked to demonstrate exactly 9 tokens of block-granularity waste; a byte-for-byte `memcmp` of a surviving block's raw storage before and after a turn transition; a check that the recomputed suffix matches the new token IDs while the reused prefix still matches the original ones; and a realistic 2048-token history extended by 48 tokens, checked for a token-count savings figure computed directly from the code's own counters.

```text
========================================================
Chapter 14.3: Prefix Caching for Multi-Turn Conversations
========================================================

-- Test 1: block-aligned full-prefix reuse, zero waste --
  turn1=48 tokens (3 blocks), turn2 appends 20 tokens
  reused=48 recomputed=20 blocks_freed=0
  physical block IDs for the reused prefix are unchanged (zero-copy)
  naive full recompute: 68 tokens; with prefix caching: 20 tokens (3.40x)

-- Test 2: [COMMON TRAP] block-granularity waste on a misaligned edit --
  turn1=40 tokens, turn2 diverges at token 25, new length=45
  matched prefix (LCP) = 25 tokens
  actually reusable = 16 tokens (1 whole block)
  wasted shared tokens = 9 (matched, but recomputed anyway because their block wasn't fully shared)

-- Test 3: reused blocks' KV data matches the original, byte for byte --
  block 0 untouched by apply_new_turn(): memcmp of all 4096 bytes is identical

-- Test 4: the recomputed suffix is correct, not leftover stale data --
  recomputed suffix (last 20 tokens) matches the NEW token IDs' KV data
  reused prefix (first 16 tokens) still matches the ORIGINAL token IDs' KV data

-- Test 5: realistic long-history conversation, honestly-computed savings --
  history=2048 tokens, next turn appends 48 tokens (total 2096)
  tokens recomputed: 48 / 2096
  tokens saved by prefix caching: 2048.0 (97.7% of this turn's total work avoided)

========================================================
21/21 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming a matched token means a reusable token"
    The longest common prefix between two token-ID sequences answers a purely textual question — how many leading tokens are identical — and says nothing on its own about whether those tokens' KV data can actually be reused, because reuse additionally requires that the match extends across an entire block boundary. A 9-token stretch that genuinely matches but sits inside a block that also contains 7 tokens past the divergence point is, in every sense that matters for reuse, indistinguishable from a stretch that never matched at all: it gets recomputed either way. Treating "tokens matched" and "tokens reused" as the same quantity overstates a prefix cache's real benefit by exactly the amount of block-granularity waste any given divergence point happens to produce, which is why this section reports both figures separately rather than only the more flattering one.

## 14.4 A Complete Cache Manager Combining Every Technique

### Intuition

Sections 14.1 through 14.3, plus Chapter 13.4's H2O policy, each solve a different piece of the same problem in isolation. Chapter 13.5 already established that combining independently-verified pieces is a new claim requiring its own verification, not a free consequence of the old ones — and combining these four specific techniques surfaces a genuinely new interaction none of them showed alone.

### The Concept, In Detail

This section's `CacheManager` runs sliding window trimming, age-based re-quantization, sink-token protection, and H2O-style importance eviction (with Chapter 13.4's own recent-window immunity preserved) together during a single turn's generation, then uses Section 14.3's prefix-caching mechanism to hand surviving entries off to the next turn. Running all four together re-confirms, inside the combined scenario, that each technique's own guarantee still holds: the sliding window's hard cutoff, H2O's recent-window immunity, re-quantization's progressive tiering, and sink-token survival all get checked again here rather than merely assumed to carry over from their own chapters. But the combination also does something none of the four techniques showed in isolation: Section 14.3 measured prefix-caching waste purely from block alignment, implicitly assuming that a token inside the matched prefix was still sitting in the cache waiting to be reused. Once a real eviction policy is running alongside prefix caching, that assumption can simply be false — a token can match the new turn's prefix perfectly and STILL require recomputation, because the sliding window or H2O policy already evicted it during the PREVIOUS turn, long before the new turn's prefix match was ever computed. Prefix caching can only reuse what the runtime cache still holds; matching token IDs is necessary but never sufficient, and the tokens most likely to have been evicted — the oldest ones — are exactly the tokens most likely to lie deep inside a long shared prefix, which is precisely where a naive accounting of prefix-caching savings would expect the largest payoff.

### Code and Verification

```cpp
// 04_complete_cache_manager.cpp
// Chapter 14, Part 4 (capstone): every technique in this chapter attacks
// the same problem -- an unbounded KV cache -- from a different angle.
// Sliding window (14.1) imposes a hard lookback limit. Streaming
// re-quantization (14.2) compresses entries as they age instead of
// discarding them outright. Prefix caching (14.3) avoids recomputing
// entries a previous turn already produced. And Chapter 13.4's H2O policy
// evicts by measured importance rather than by age. A real serving engine
// runs all four together, and Chapter 13.5 already established the
// reason that matters: guarantees verified for each technique in
// isolation are necessary but not sufficient, because combining
// independently-correct pieces can surface behavior none of them showed
// alone. This file builds one CacheManager that runs sliding window,
// re-quantization, sink protection, and H2O-style importance eviction
// together during a single turn's generation, then uses prefix caching
// to hand the survivors off to the next turn -- and demonstrates a
// genuinely new interaction that only appears once all four are combined.
//
// The new interaction: Section 14.3's prefix caching measured its own
// waste purely from block alignment -- a block straddling the
// divergence point had to be recomputed even though part of it matched.
// Here, a second and independent source of waste appears: a token can
// fall entirely within the matched prefix and STILL have to be
// recomputed, because the window/H2O maintenance policy already evicted
// it before the next turn even began. Prefix caching can only reuse what
// the runtime cache still has -- matching token IDs is necessary but not
// sufficient.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_complete_cache_manager.cpp -o 04_complete_cache_manager

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

// ---- Reused verbatim from Section 14.2 (MultiQuant) ----
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

    Quantized requantize(const Quantized& q, int new_bits) const {
        assert(new_bits <= q.bits);
        std::vector<float> tmp(dim);
        dequantize(q, tmp.data());
        return quantize(tmp.data(), new_bits);
    }
};

// ---- Reused verbatim from Section 14.3 (prefix matching, token->KV stand-in) ----
int longest_common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
    int n = static_cast<int>(std::min(a.size(), b.size()));
    int i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

void gen_kv(int token_id, int dim, float* k, float* v) {
    for (int d = 0; d < dim; ++d) {
        k[d] = static_cast<float>(token_id % 997) * 0.01f + static_cast<float>(d) * 0.001f;
        v[d] = static_cast<float>(token_id % 997) * 0.02f + static_cast<float>(d) * 0.001f;
    }
}

// ---- New for this section: the combined cache entry and manager ----

struct CacheEntry {
    int position;
    int insert_step;
    float importance;
    bool is_sink;
    MultiQuant::Quantized key_q, val_q;
};

// Combines every eviction/compression policy from this chapter and from
// Chapter 13.4: a sliding window sets the outer bound, sink tokens are
// always exempt (Chapter 13.3's insight, still relevant even without a
// ring buffer underneath), age-based re-quantization compresses
// survivors, and H2O importance eviction trims to a hard budget with
// Chapter 13.4's own recent-window immunity preserved.
struct CacheManager {
    int head_dim;
    int max_entries;
    int window_size;
    int n_sinks;
    int recent_window;
    int age_to_3bit, age_to_2bit;
    int current_step = 0;
    MultiQuant quantizer;
    std::vector<CacheEntry> entries;

    CacheManager(int hd, int max_ent, int window, int sinks, int recent,
                 int a3, int a2)
        : head_dim(hd), max_entries(max_ent), window_size(window), n_sinks(sinks),
          recent_window(recent), age_to_3bit(a3), age_to_2bit(a2), quantizer(hd) {}

    void insert(int pos, const float* k, const float* v) {
        CacheEntry e;
        e.position = pos;
        e.insert_step = current_step;
        e.importance = 0.0f;
        e.is_sink = pos < n_sinks;
        e.key_q = quantizer.quantize(k, 4);
        e.val_q = quantizer.quantize(v, 4);
        entries.push_back(std::move(e));
    }

    std::vector<float> attend(const float* query) {
        int n = static_cast<int>(entries.size());
        if (n == 0) return std::vector<float>(head_dim, 0.0f);
        std::vector<float> scores(n);
        std::vector<float> kbuf(head_dim), vbuf(head_dim);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int i = 0; i < n; ++i) {
            quantizer.dequantize(entries[i].key_q, kbuf.data());
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += query[d] * kbuf[d];
            scores[i] = dot * scale;
        }
        float mx = *std::max_element(scores.begin(), scores.end());
        float sum = 0.0f;
        for (float& s : scores) { s = std::exp(s - mx); sum += s; }
        for (float& s : scores) s /= sum;
        for (int i = 0; i < n; ++i) entries[i].importance += scores[i];
        std::vector<float> out(head_dim, 0.0f);
        for (int i = 0; i < n; ++i) {
            quantizer.dequantize(entries[i].val_q, vbuf.data());
            for (int d = 0; d < head_dim; ++d) out[d] += scores[i] * vbuf[d];
        }
        return out;
    }

    struct Stats { int window_trimmed = 0, requant_3 = 0, requant_2 = 0, evicted = 0; };

    Stats maintain() {
        Stats s;
        if (window_size > 0) {
            int cutoff = current_step - window_size;
            size_t before = entries.size();
            entries.erase(std::remove_if(entries.begin(), entries.end(),
                [&](const CacheEntry& e) { return !e.is_sink && e.insert_step < cutoff; }),
                entries.end());
            s.window_trimmed = static_cast<int>(before - entries.size());
        }
        for (auto& e : entries) {
            int age = current_step - e.insert_step;
            if (age >= age_to_2bit && e.key_q.bits > 2) {
                e.key_q = quantizer.requantize(e.key_q, 2);
                e.val_q = quantizer.requantize(e.val_q, 2);
                ++s.requant_2;
            } else if (age >= age_to_3bit && e.key_q.bits > 3) {
                e.key_q = quantizer.requantize(e.key_q, 3);
                e.val_q = quantizer.requantize(e.val_q, 3);
                ++s.requant_3;
            }
        }
        if (static_cast<int>(entries.size()) > max_entries) {
            int to_evict = static_cast<int>(entries.size()) - max_entries;
            std::vector<int> evictable;
            for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
                int age = current_step - entries[i].insert_step;
                if (!entries[i].is_sink && age >= recent_window) evictable.push_back(i);
            }
            std::sort(evictable.begin(), evictable.end(), [&](int a, int b) {
                return entries[a].importance < entries[b].importance;
            });
            int n_evict = std::min(to_evict, static_cast<int>(evictable.size()));
            std::vector<bool> remove(entries.size(), false);
            for (int i = 0; i < n_evict; ++i) remove[evictable[i]] = true;
            std::vector<CacheEntry> remaining;
            for (size_t i = 0; i < entries.size(); ++i) if (!remove[i]) remaining.push_back(std::move(entries[i]));
            entries = std::move(remaining);
            s.evicted = n_evict;
        }
        return s;
    }

    void advance() { ++current_step; }
    int size() const { return static_cast<int>(entries.size()); }
    bool has_position(int pos) const {
        for (const auto& e : entries) if (e.position == pos) return true;
        return false;
    }
};

// Prefix-caching handoff between turns: reuse whatever entries the OLD
// manager still holds whose position lies inside the matched prefix, and
// recompute everything else. Returns how much of the theoretical match
// was actually free.
struct HandoffStats {
    int lcp = 0;
    int theoretically_reusable = 0;  // == lcp, if the runtime cache held everything
    int actually_reused = 0;         // entries the old cache still had, inside the LCP
    int recomputed = 0;
};

HandoffStats prefix_handoff(CacheManager& new_mgr, const CacheManager& old_mgr,
                             const std::vector<int>& old_ids, const std::vector<int>& new_ids) {
    HandoffStats stats;
    stats.lcp = longest_common_prefix(old_ids, new_ids);
    stats.theoretically_reusable = stats.lcp;

    std::vector<bool> covered(new_ids.size(), false);
    for (const auto& e : old_mgr.entries) {
        if (e.position < stats.lcp) {
            new_mgr.entries.push_back(e);  // carry the quantized entry over verbatim, no recompute
            covered[e.position] = true;
            ++stats.actually_reused;
        }
    }
    for (int pos = 0; pos < static_cast<int>(new_ids.size()); ++pos) {
        if (covered[pos]) continue;
        float k[64], v[64];
        gen_kv(new_ids[pos], new_mgr.head_dim, k, v);
        new_mgr.insert(pos, k, v);
        ++stats.recomputed;
    }
    return stats;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 14.4: A Complete Cache Manager Combining Every Technique\n";
    std::cout << "========================================================\n";

    constexpr int HD = 16;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    auto rv = [&] { std::vector<float> v(HD); for (auto& x : v) x = dist(rng); return v; };

    std::cout << "\n-- Simulating 300 generation steps (window=200, budget=50, sinks=4) --\n";
    CacheManager mgr(HD, /*max_ent=*/50, /*window=*/200, /*sinks=*/4,
                      /*recent_window=*/10, /*age_to_3bit=*/50, /*age_to_2bit=*/150);
    int total_trimmed = 0, total_r3 = 0, total_r2 = 0, total_evicted = 0;
    bool window_cutoff_ever_enforced = false;
    bool attention_always_valid = true;

    for (int step = 0; step < 300; ++step) {
        auto k = rv(), v = rv();
        mgr.insert(step, k.data(), v.data());
        auto query = rv();
        auto out = mgr.attend(query.data());
        float norm_sq = 0.0f;
        for (float x : out) norm_sq += x * x;
        if (!(norm_sq > 0.0f)) attention_always_valid = false;

        if (step % 10 == 9) {
            auto stats = mgr.maintain();
            total_trimmed += stats.window_trimmed;
            total_r3 += stats.requant_3;
            total_r2 += stats.requant_2;
            total_evicted += stats.evicted;
            if (stats.window_trimmed > 0) window_cutoff_ever_enforced = true;
        }
        mgr.advance();
    }

    std::cout << "  window_trimmed=" << total_trimmed << " requant_3=" << total_r3
               << " requant_2=" << total_r2 << " evicted=" << total_evicted
               << " final_size=" << mgr.size() << "/" << mgr.max_entries << "\n";

    std::cout << "\n-- Test 1: sliding window's hard cutoff still holds inside the combined manager --\n";
    {
        CHECK(window_cutoff_ever_enforced);
        for (const auto& e : mgr.entries) {
            int age = mgr.current_step - e.insert_step;
            CHECK(e.is_sink || age <= mgr.window_size);
        }
        std::cout << "  every surviving non-sink entry is within the " << mgr.window_size << "-step window\n";
    }

    std::cout << "\n-- Test 2: H2O recent-window immunity holds inside the combined manager --\n";
    {
        // maintain() only ever treats entries with age >= recent_window as
        // evictable (see the `evictable` filter in maintain() above), so
        // any entry younger than that is immune no matter how low its
        // importance is. Confirm with an isolated probe: 4 same-age
        // entries with a tight budget of 2 should evict nothing while
        // they are all still within the recent window, then evict exactly
        // enough once they age past it.
        CacheManager probe(HD, /*max_ent=*/2, /*window=*/1000, /*sinks=*/0, /*recent=*/5, 1000, 2000);
        for (int t = 0; t < 4; ++t) { auto k = rv(), v = rv(); probe.insert(t, k.data(), v.data()); probe.advance(); }
        auto pstats = probe.maintain();
        CHECK(pstats.evicted == 0);  // all 4 entries are younger than recent_window=5: nothing is evictable yet
        for (int t = 0; t < 10; ++t) probe.advance();
        auto pstats2 = probe.maintain();
        CHECK(pstats2.evicted == 2);  // now old enough; trims down to max_entries=2
        std::cout << "  isolated probe: 0 evicted while all entries are within the recent window, "
                   << "then " << pstats2.evicted << " evicted once they age past it\n";
    }

    std::cout << "\n-- Test 3: re-quantization tiers progress with age inside the combined manager --\n";
    {
        int n4 = 0, n3 = 0, n2 = 0;
        for (const auto& e : mgr.entries) {
            if (e.key_q.bits == 4) ++n4;
            else if (e.key_q.bits == 3) ++n3;
            else if (e.key_q.bits == 2) ++n2;
        }
        CHECK(total_r3 > 0);
        CHECK(total_r2 > 0);
        CHECK(n2 > 0);  // some entries reached the coarsest tier and are still alive
        std::cout << "  final tier distribution: 4-bit=" << n4 << " 3-bit=" << n3 << " 2-bit=" << n2 << "\n";
    }

    std::cout << "\n-- Test 4: attention sinks survive the entire run; attention stays valid throughout --\n";
    {
        int sinks_alive = 0;
        for (const auto& e : mgr.entries) if (e.is_sink) ++sinks_alive;
        CHECK(sinks_alive == mgr.n_sinks);
        CHECK(attention_always_valid);
        std::cout << "  sinks_alive=" << sinks_alive << "/" << mgr.n_sinks
                   << "; attention produced a valid, non-zero-norm output at every one of 300 steps\n";
    }

    std::cout << "\n-- Test 5: prefix handoff before any eviction has happened: theoretical == actual --\n";
    {
        CacheManager old_mgr(HD, /*max_ent=*/1000, /*window=*/1000, /*sinks=*/4, /*recent=*/50, 1000, 2000);
        std::vector<int> old_ids;
        for (int t = 0; t < 60; ++t) {
            float k[HD], v[HD];
            gen_kv(t, HD, k, v);
            old_mgr.insert(t, k, v);
            old_mgr.advance();
            old_ids.push_back(t);
        }
        // no maintain() call: nothing has been evicted or trimmed yet
        std::vector<int> new_ids = old_ids;
        for (int t = 0; t < 15; ++t) new_ids.push_back(9000 + t);

        CacheManager new_mgr(HD, /*max_ent=*/1000, /*window=*/1000, /*sinks=*/4, /*recent=*/50, 1000, 2000);
        auto hs = prefix_handoff(new_mgr, old_mgr, old_ids, new_ids);
        CHECK(hs.lcp == 60);
        CHECK(hs.actually_reused == hs.theoretically_reusable);
        CHECK(hs.recomputed == 15);
        std::cout << "  lcp=" << hs.lcp << " theoretically_reusable=" << hs.theoretically_reusable
                   << " actually_reused=" << hs.actually_reused << " recomputed=" << hs.recomputed
                   << " (nothing was ever evicted, so the two figures match exactly)\n";
    }

    std::cout << "\n-- Test 6: [COMMON TRAP] prefix handoff after real eviction: matched != retained --\n";
    {
        // A tight budget and a short window guarantee the maintenance
        // policy will have evicted most early, non-sink tokens by the
        // time this turn ends -- even though those same tokens will
        // still appear, unchanged, at the front of the next turn's ids.
        CacheManager old_mgr(HD, /*max_ent=*/20, /*window=*/40, /*sinks=*/4, /*recent=*/5, 15, 30);
        std::vector<int> old_ids;
        for (int t = 0; t < 100; ++t) {
            float k[HD], v[HD];
            gen_kv(t, HD, k, v);
            old_mgr.insert(t, k, v);
            auto q = rv();
            old_mgr.attend(q.data());
            if (t % 10 == 9) old_mgr.maintain();
            old_mgr.advance();
            old_ids.push_back(t);
        }
        std::vector<int> new_ids = old_ids;  // the entire history repeats unchanged
        for (int t = 0; t < 10; ++t) new_ids.push_back(8000 + t);

        CacheManager new_mgr(HD, /*max_ent=*/20, /*window=*/40, /*sinks=*/4, /*recent=*/5, 15, 30);
        auto hs = prefix_handoff(new_mgr, old_mgr, old_ids, new_ids);
        CHECK(hs.lcp == 100);                      // the full old history matched, token for token
        CHECK(hs.actually_reused < hs.theoretically_reusable);  // but most of it is gone from the cache
        CHECK(hs.actually_reused == old_mgr.size());            // exactly what survived maintenance, no more
        int gap = hs.theoretically_reusable - hs.actually_reused;
        CHECK(gap > 0);
        std::cout << "  old cache after 100 steps of eviction: " << old_mgr.size() << " entries alive\n";
        std::cout << "  lcp=" << hs.lcp << " theoretically_reusable=" << hs.theoretically_reusable
                   << " actually_reused=" << hs.actually_reused << " recomputed=" << hs.recomputed << "\n";
        std::cout << "  gap=" << gap << " tokens matched the prefix but had already been evicted;\n";
        std::cout << "  they must be recomputed exactly as if they had never matched at all\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_complete_cache_manager.cpp -o 04_complete_cache_manager
./04_complete_cache_manager
```

**Sample input:** a 300-step combined simulation (200-token window, 50-entry budget, 4 sink tokens) checked for the sliding window's hard cutoff, H2O's recent-window immunity via an isolated probe, progressive re-quantization tiering, and sink-token survival with continuously valid attention output; a prefix handoff between two turns with no prior eviction, checked to show the theoretical and actual reuse counts matching exactly; and a prefix handoff after 100 steps of real eviction under a tight budget, checked to show a substantial, honestly-computed gap between tokens that matched the prefix and tokens the cache had actually retained.

```text
========================================================
Chapter 14.4: A Complete Cache Manager Combining Every Technique
========================================================

-- Simulating 300 generation steps (window=200, budget=50, sinks=4) --
  window_trimmed=36 requant_3=76 requant_2=40 evicted=214 final_size=50/50

-- Test 1: sliding window's hard cutoff still holds inside the combined manager --
  every surviving non-sink entry is within the 200-step window

-- Test 2: H2O recent-window immunity holds inside the combined manager --
  isolated probe: 0 evicted while all entries are within the recent window, then 2 evicted once they age past it

-- Test 3: re-quantization tiers progress with age inside the combined manager --
  final tier distribution: 4-bit=10 3-bit=36 2-bit=4

-- Test 4: attention sinks survive the entire run; attention stays valid throughout --
  sinks_alive=4/4; attention produced a valid, non-zero-norm output at every one of 300 steps

-- Test 5: prefix handoff before any eviction has happened: theoretical == actual --
  lcp=60 theoretically_reusable=60 actually_reused=60 recomputed=15 (nothing was ever evicted, so the two figures match exactly)

-- Test 6: [COMMON TRAP] prefix handoff after real eviction: matched != retained --
  old cache after 100 steps of eviction: 20 entries alive
  lcp=100 theoretically_reusable=100 actually_reused=20 recomputed=90
  gap=80 tokens matched the prefix but had already been evicted;
  they must be recomputed exactly as if they had never matched at all

========================================================
65/65 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] crediting prefix caching for tokens the cache no longer has"
    A prefix-caching implementation that only checks "does this token's ID match the previous turn's sequence at this position" and calls that a cache hit is answering the wrong question the moment any eviction policy is also running, because a matching token ID guarantees nothing about whether that token's KV data still physically exists anywhere in memory. This section's own combined scenario makes the gap concrete: a full 100-token history matches the new turn's prefix token for token, yet a tight window-and-budget policy had already evicted all but 20 of those 100 entries by the time the new turn begins, so 80 tokens that would appear "reusable" under a purely textual prefix check must be recomputed exactly as if they had never matched at all. The fix is that prefix-caching logic must always check actual cache membership, never token-ID equality alone, before counting anything as reused — and any reported prefix-caching speedup figure that was computed from matched-token counts rather than actually-reused-token counts should be treated as an upper bound, not a measurement.

## Chapter Summary

This chapter extended Chapter 13's KV cache management foundation in the three directions its own techniques had not yet covered. Section 14.1 applied Chapter 13.3's ring-buffer mechanism at real long-context scale, removing the sink-token protection so the window boundary itself becomes the entire eviction policy — exactly what Mistral 7B was trained to expect, and a reasonable approximation for models that were not. Section 14.2 replaced binary eviction with graceful degradation: aging tokens are compressed to coarser bit-widths rather than discarded outright, safely, because their shrinking attention weight makes the resulting error harmless to the output. Section 14.3 built an entirely original prefix-caching design, since this chapter's own source material offered only an unverified narrative claim, and showed that a KVBlock's fixed granularity means matched tokens are not always reusable tokens. Section 14.4 closed the chapter by combining all four techniques — sliding window, re-quantization, prefix caching, and Chapter 13.4's H2O eviction — into one manager, re-verifying each technique's own guarantee inside the combination and surfacing a genuinely new interaction: prefix caching can only reuse what the runtime cache still retains, and the tokens most likely to have been evicted are exactly the ones most likely to sit deep inside a long shared prefix. Chapters 13 and 14 together give every later chapter a KV cache that pages efficiently, bounds its own growth at real scale, degrades gracefully as entries age, evicts by measured usefulness, and avoids recomputing work a previous turn already did.

## Self-Check Questions

1. Section 14.1's sliding window removes Chapter 13.3's sink-token protection entirely. Why does that removal make sense for sliding window specifically, when Chapter 13.3 treated sink-token eviction as a real coherence problem worth solving?
2. Why is applying a fixed-size sliding window to a model trained with full, unrestricted attention described as an approximation, while applying the identical window to Mistral 7B is not?
3. Section 14.2's re-quantization tiering compresses OLD tokens to coarser bit-widths rather than NEW ones. Explain why this age-based direction, rather than the reverse, is what makes the resulting quantization error safe to ignore.
4. Section 14.2 keeps a simplified uniform quantizer rather than building this book's Chapter 7 TurboQuant machinery inline. What part of the cache-management POLICY would be identical either way, and what part would actually differ if Chapter 7's codebooks were used instead?
5. In Section 14.3, a 40-token conversation diverges at token 25 under a 16-token block size. Walk through why exactly 9 tokens (16 through 24) are recomputed despite having matched the new turn's token IDs.
6. Why does Section 14.3 insist that only WHOLE blocks entirely inside the longest common prefix can be reused, rather than reusing individual matched tokens directly regardless of block boundaries?
7. Section 14.3's source material claimed an unverified "501x" prefix-caching speedup with no accompanying code. What did this chapter do instead, and why does the book's standing rule require that choice?
8. Section 14.4 re-checks the sliding window's cutoff, H2O's recent-window immunity, and re-quantization's tiering all over again inside the combined manager, even though each was already verified in its own chapter section. Why isn't verifying each technique once, in isolation, sufficient?
9. In Section 14.4's Test 6, a 100-token history matches a new turn's prefix perfectly, yet only 20 tokens are actually reused. Explain concretely why "the tokens matched" and "the tokens are reusable" diverge so sharply in this specific scenario.
10. Suppose a prefix-caching implementation reports its speedup using only the longest-common-prefix length, without checking whether the matched tokens are still physically present in the cache. Under what conditions would that reported figure most overstate the real speedup, based on Section 14.4's own findings?

## Where We Go Next

Chapters 13 and 14 together gave every future chapter a KV cache that pages efficiently, bounds its own growth at real scale, degrades gracefully rather than falling off a cliff as entries age, evicts by measured usefulness rather than blind age or a hard window alone, and avoids recomputing a conversation's own unchanged history on every turn. With state management settled, Part 4 turns from the engineering of the inference engine itself to running it against a real, published model: Chapter 15 begins with HuggingFace's model format and produces a first generated token from Qwen2.5, the point at which everything built so far — quantization, kernels, threading, and now KV cache management — comes together into a working system.

## Worked Solutions

**1.** Chapter 13.3's Ring Buffer needed sink-token protection because it could not otherwise distinguish "this token is being evicted because the buffer is full" from "this token happens to be one the model has learned to treat as a stable attention anchor" — the wrap logic would otherwise blindly overwrite whichever slot came next, sink token or not. Sliding window's entire design point is that NO token gets that kind of individual protection: the window boundary itself is the complete eviction policy, with no case-by-case judgment at all, so adding sink protection back in would contradict the very simplicity that makes sliding window sliding window rather than a re-implementation of Chapter 13.3's ring buffer with extra steps.

**2.** Mistral 7B was explicitly trained with a 4096-token sliding window in place, meaning every gradient update during training already reflected the constraint that attention could never reach further back than that — the model learned to compress anything it might need beyond the window into its hidden state as a normal part of training, so applying the window at inference time changes nothing relative to what the model has always experienced. A model trained with full, unrestricted attention never faced that constraint during training and never learned to compress anything into its hidden state for this reason, so imposing a window at inference time asks it to operate under a restriction its own training never prepared it for — usually tolerable in practice because attention weight decays with distance anyway, but not guaranteed the way it is for a model actually trained under the window.

**3.** A token's actual contribution to the current generation step's output is its attention weight multiplied by its Key/Value vector's value; that attention weight empirically shrinks as a token ages, so a quantization error introduced into an OLD token's vector gets multiplied by a small weight before it ever reaches the output, making the error's effect on the result correspondingly small. Applying the same coarse quantization to a NEW, heavily-weighted token would multiply that same-sized error by a large weight instead, producing a visible corruption of the output — the direction of the tiering (old gets coarse, new stays precise) is exactly what keeps the growing quantization error harmless rather than merely hidden.

**4.** The cache-management POLICY — inserting fresh entries at high fidelity, checking age against configurable thresholds, re-quantizing to progressively coarser tiers, and eventually evicting — would be completely identical either way, because that policy only cares about WHEN to re-quantize, not HOW the re-quantization itself is implemented. What would actually differ is the re-quantize operation's internals: this section's simplified quantizer dequantizes to float and re-encodes with a fresh symmetric-uniform scale, while Chapter 7's TurboQuant would dequantize using its trained Lloyd-Max codebook and re-encode with a coarser codebook sharing the same rotation matrix, producing a different (generally better, since Lloyd-Max codebooks are optimized for the actual data distribution) reconstruction error at each bit-width for the identical policy timeline.

**5.** With a 16-token block size, tokens 0-15 form block 0 and tokens 16-31 form block 1; the divergence at token 25 falls inside block 1, meaning block 1 contains both genuinely-matched tokens (16 through 24) and genuinely-diverged tokens (25 through 31) at once. Because a `KVBlock` can only be reused or discarded as a whole unit — there is no mechanism to keep part of a block's stored tokens while discarding the rest — the entire block gets discarded and recomputed once ANY of its tokens are past the divergence point, which is why tokens 16 through 24, despite matching perfectly, get swept into the recomputation along with the tokens that actually changed.

**6.** Reusing an individual matched token independent of its block would require pulling that one token's Key/Value data out of a block that otherwise needs to be discarded — but a block's storage has no mechanism for partial retention; the block manager's entire interface operates on whole blocks (allocate one, free one, reference one via a block table entry), because that whole-block granularity is precisely what makes block-table indirection O(1) and allocation-free in Chapter 13.2. Building token-level partial reuse would mean reintroducing the fine-grained, per-token bookkeeping that PagedAttention's block design was built specifically to avoid, trading away the very property (cheap, block-granularity indirection) that makes the KV cache manageable at scale in exchange for a reuse optimization that only pays off on the relatively rare tokens sitting in a straddling block.

**7.** This chapter's own source material described prefix caching only in prose, with a bare "speedup: 501x" arithmetic claim and no test code, model, or measurement methodology anywhere behind it. This book's standing rule is to never carry forward a performance number that was not itself computed and checked by code in the chapter, so rather than reproduce that unverified figure, this section built its own original prefix-caching implementation from Chapter 13.2's `BlockManager` and measured its own, real speedup and savings figures directly from working code — numbers that are smaller and more specific than "501x," but that are honestly the code's own.

**8.** Verifying a technique once, in isolation, proves it behaves correctly under the specific conditions its own test constructed — Section 14.1's sliding-window tests never ran alongside an H2O eviction policy or a re-quantization schedule, for instance. Re-checking the same guarantee inside the fully combined manager is a different claim: it confirms that none of the OTHER three techniques' bookkeeping perturbs this one's behavior when all four are genuinely running together, which unit tests run in isolation cannot show by construction — exactly the same lesson Chapter 13.5 drew from Chapter 11's discovery that individually thread-count-independent phases still produced a thread-count-dependent whole once combined.

**9.** The 100-token history matches the new turn's token IDs at every single position, so the longest common prefix is the full 100 tokens — a purely textual fact about the two ID sequences. But the OLD cache was simultaneously running a tight window-and-budget eviction policy throughout those same 100 steps, and by the time the new turn begins, that policy has already discarded all but 20 of the original entries to stay within budget; those 20 survivors are the only entries that ANY handoff mechanism can actually copy over without recomputing them, regardless of how much of the token sequence textually matches. The 100-token match describes what theoretically could have been reused if nothing had ever been evicted; the 20-entry reuse describes what the runtime cache genuinely still had on hand.

**10.** That reported figure most overstates reality precisely when a lot of eviction has happened to old, deep-prefix tokens by the time the new turn begins — exactly Section 14.4's own combined scenario, where a full prefix match coincided with 80 of 100 matched tokens already being gone from the cache. A conversation with little or no eviction pressure (a generous budget, a wide window, or a short history that never triggered maintenance) would show the theoretical and actual reuse counts nearly matching, as Section 14.4's own first handoff test demonstrated; the overstatement grows specifically with how aggressively the OTHER techniques in the combined system have been evicting the very tokens a long shared prefix depends on.
