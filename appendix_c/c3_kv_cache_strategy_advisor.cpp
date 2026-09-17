// Appendix C.3 -- A KV Cache Strategy Advisor.
//
// Chapters 13 and 14 built five real, separately-verified KV cache
// techniques: a plain ring buffer with position/slot decoupling (13.3),
// a paged, block-table-indirected allocator (13.2), H2O attention-weighted
// eviction (13.4), sliding-window attention (14.1), streaming
// re-quantization of aging entries (14.2), and prefix caching across
// conversation turns (14.3/14.4). None of them is a universal replacement
// for the others -- each solves a real, different constraint. This file
// restates the real decision criteria Chapters 13-14 actually established
// for choosing among them, as one callable advisor, plus Chapter 13.1's
// own exact KV-cache-bytes-per-token formula reused verbatim.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 c3_kv_cache_strategy_advisor.cpp -o c3_kv_cache_strategy_advisor
// Run:     ./c3_kv_cache_strategy_advisor

#include <cstdint>
#include <iostream>
#include <string>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// How the context length itself is expected to behave -- known and small
// enough to reserve up front, genuinely unpredictable and growing, or so
// long that even a paged allocator cannot keep every token resident.
enum class ContextRegime { SHORT_BOUNDED, LONG_GROWING, VERY_LONG_MEMORY_CONSTRAINED };

enum class CacheStrategy {
    PLAIN_RING_BUFFER,
    PAGED_BLOCK_TABLE,
    SLIDING_WINDOW,
    PAGED_WITH_H2O_EVICTION,
    PREFIX_CACHING_ON_PAGED,
};

std::string to_string(CacheStrategy s) {
    switch (s) {
        case CacheStrategy::PLAIN_RING_BUFFER: return "PLAIN_RING_BUFFER";
        case CacheStrategy::PAGED_BLOCK_TABLE: return "PAGED_BLOCK_TABLE";
        case CacheStrategy::SLIDING_WINDOW: return "SLIDING_WINDOW";
        case CacheStrategy::PAGED_WITH_H2O_EVICTION: return "PAGED_WITH_H2O_EVICTION";
        case CacheStrategy::PREFIX_CACHING_ON_PAGED: return "PREFIX_CACHING_ON_PAGED";
    }
    return "UNKNOWN";
}

struct CacheRecommendation {
    CacheStrategy strategy;
    std::string reason;
};

// model_trained_with_fixed_window: Chapter 14.1's own real distinction --
// a model explicitly trained with a fixed attention window (its own
// example: Mistral 7B's 4096-token window) loses nothing from a
// same-sized window at inference time, because the window is exactly what
// it learned to expect, not an approximation of full attention.
//
// multi_turn_conversation: Chapter 14.3/14.4's own scope -- prefix
// caching is a cross-TURN optimization layered on top of a paged cache,
// not a replacement for how any single turn's own entries are stored or
// evicted.
CacheRecommendation recommend_kv_cache_strategy(ContextRegime regime,
                                                 bool model_trained_with_fixed_window,
                                                 bool multi_turn_conversation) {
    if (multi_turn_conversation) {
        return {CacheStrategy::PREFIX_CACHING_ON_PAGED,
                "Chapter 14.3/14.4: reuses shared history across turns via the block table, but only "
                "credits tokens the runtime cache still actually holds -- never token-ID equality alone "
                "-- so it must sit on top of a paged, block-addressed cache to begin with"};
    }
    if (model_trained_with_fixed_window) {
        return {CacheStrategy::SLIDING_WINDOW,
                "Chapter 14.1: a model explicitly trained with a fixed attention window loses nothing "
                "from a same-sized ring-buffer window -- it is exactly what the model learned to expect, "
                "not an approximation of full attention"};
    }
    switch (regime) {
        case ContextRegime::SHORT_BOUNDED:
            return {CacheStrategy::PLAIN_RING_BUFFER,
                    "Chapter 13.3: once the maximum context length is known and small enough to reserve "
                    "up front, a single fixed-capacity ring buffer with the write-slot and RoPE-position "
                    "counters kept separate is sufficient"};
        case ContextRegime::LONG_GROWING:
            return {CacheStrategy::PAGED_BLOCK_TABLE,
                    "Chapter 13.2: fixed-size power-of-two blocks plus a free list avoid both "
                    "worst-case-reservation waste and reallocate-and-copy stalls as a genuinely "
                    "unpredictable context grows"};
        case ContextRegime::VERY_LONG_MEMORY_CONSTRAINED:
            return {CacheStrategy::PAGED_WITH_H2O_EVICTION,
                    "Chapter 13.4: once paging alone cannot keep every token resident, H2O's attention-"
                    "probability importance score evicts the least-useful token instead of merely the "
                    "oldest, with a recent-token exemption so brand-new context is never evicted before "
                    "it has had a chance to prove itself important"};
    }
    return {CacheStrategy::PLAIN_RING_BUFFER, "unreachable"};
}

// Chapter 13.1's own exact formula, reused verbatim: for each layer, each
// GQA KV head (not query head), per cached token, 2 vectors (K, V) of
// head_dim elements each.
uint64_t kv_cache_bytes_per_token(uint64_t n_layers, uint64_t n_kv_heads, uint64_t head_dim,
                                   uint64_t bytes_per_element) {
    return n_layers * n_kv_heads * head_dim * 2 * bytes_per_element;
}

int main() {
    std::cout << "===================================================\n";
    std::cout << "Appendix C.3: The KV Cache Strategy Advisor\n";
    std::cout << "===================================================\n\n";

    // -- Test 1: a multi-turn conversation always gets prefix caching,
    // regardless of context regime or window training -- it is a layer
    // ON TOP of the underlying cache, not a competitor to it. --
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::SHORT_BOUNDED, true, true);
        std::cout << "-- Test 1: multi-turn conversation -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::PREFIX_CACHING_ON_PAGED);
    }

    // -- Test 2: a model trained with a fixed attention window gets
    // sliding window, as long as it is not also a multi-turn scenario. --
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::LONG_GROWING, true, false);
        std::cout << "-- Test 2: trained-with-fixed-window model -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::SLIDING_WINDOW);
    }

    // -- Tests 3-5: the three real context-regime defaults, with neither
    // of the two overriding conditions set. --
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::SHORT_BOUNDED, false, false);
        std::cout << "-- Test 3: short, bounded context -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::PLAIN_RING_BUFFER);
    }
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::LONG_GROWING, false, false);
        std::cout << "-- Test 4: long, unpredictably growing context -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::PAGED_BLOCK_TABLE);
    }
    {
        auto r = recommend_kv_cache_strategy(ContextRegime::VERY_LONG_MEMORY_CONSTRAINED, false, false);
        std::cout << "-- Test 5: very long, memory-constrained context -- strategy=" << to_string(r.strategy) << " --\n";
        CHECK(r.strategy == CacheStrategy::PAGED_WITH_H2O_EVICTION);
        CHECK(r.reason.find("H2O") != std::string::npos);
    }

    // -- Test 6: Chapter 13.1's own worked example reproduced exactly --
    // a Llama-3-8B-shaped config (32 layers, 8 GQA KV heads, head_dim 128,
    // FP16) costs exactly 128 KiB (131072 bytes) per cached token. --
    {
        uint64_t bytes = kv_cache_bytes_per_token(32, 8, 128, 2);
        std::cout << "-- Test 6: Llama-3-8B-shaped config -- bytes_per_token=" << bytes << " --\n";
        CHECK(bytes == 131072ULL);
        CHECK(bytes == 128ULL * 1024ULL);  // exactly 128 KiB, not merely "about" 128 KB
    }

    // -- Test 7: Chapter 13.1's own "16 GB at 128K tokens" claim
    // reproduced exactly -- at a context length of exactly 131072 tokens
    // (2^17, the real meaning of "128K" tokens), the same per-token cost
    // from Test 6 totals exactly 16 GiB, not merely "about" 16 GB. --
    {
        uint64_t bytes_per_token = kv_cache_bytes_per_token(32, 8, 128, 2);
        uint64_t context_length = 131072ULL;  // 128K tokens = 2^17
        uint64_t total_bytes = bytes_per_token * context_length;
        double total_gib = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "-- Test 7: total KV cache at 128K-token context -- total_bytes=" << total_bytes
                  << ", total_gib=" << total_gib << " --\n";
        CHECK(total_bytes == 17179869184ULL);
        CHECK(total_gib == 16.0);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
