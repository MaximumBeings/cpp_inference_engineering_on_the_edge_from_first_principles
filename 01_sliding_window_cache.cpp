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
