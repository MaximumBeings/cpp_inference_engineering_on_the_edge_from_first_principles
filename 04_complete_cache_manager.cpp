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
