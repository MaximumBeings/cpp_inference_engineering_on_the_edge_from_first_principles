// Appendix C.1 -- A Quantization Format Advisor.
//
// This file does not introduce a single new quantization technique. It
// turns the real, already-derived decision criteria from Chapter 4
// (Quantization Strategies), Chapter 6 (The Hybrid Quantized Engine), and
// Chapter 7 (TurboQuant) into one callable function that reproduces those
// chapters' own conclusions -- restated as executable code, not just prose,
// exactly this book's own standing discipline for every other real claim.
//
// The four real facts this advisor encodes:
//   1. Chapter 6.3: norm weights are genuinely negligible in element count
//      regardless of scale, so there is no real memory reason to compress
//      them -- keep them F32.
//   2. Chapter 6.1: attention feeds a discrete softmax-argmax decision that
//      a given error size flips at a measurable, non-negligible rate, while
//      an embedding table's own element count (2 * VOCAB * DIM) can exceed
//      every other tensor category combined at real vocabulary sizes
//      (Chapter 6.3) -- both get the more careful Q8_0 format.
//   3. Chapter 6.1: an FFN matrix's same-order-of-magnitude rounding error
//      stays a continuous, bounded output error (no discrete decision it
//      can flip), and FFN matrices outnumber attention matrices roughly
//      3:1 per layer -- Q4_0 buys the largest absolute memory savings
//      exactly where precision is least fragile.
//   4. Chapter 4.5: activations are quantized fresh every token on the
//      critical path, never persisted to a file, so there is nothing to
//      shrink a stored scale for -- dynamic INT8 keeps a plain float scale.
//      Chapter 7.4/7.5: a KV cache entry is online, uncalibrated data with
//      no offline calibration pass available, exactly the case TurboQuant's
//      data-oblivious codebooks were built for; its MSE mode (not the
//      unbiased QJL "Prod" mode) is the right choice at the KV cache's
//      typical 3-4 bit range, where the inner-product bias QJL corrects
//      for is too small to be worth the extra residual stage.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 c1_quantization_format_advisor.cpp -o c1_quantization_format_advisor
// Run:     ./c1_quantization_format_advisor

#include <iostream>
#include <string>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// Reused verbatim from Chapter 6.2's own WeightRole enum.
enum class WeightRole { NORM, EMBEDDING, ATTENTION, FFN };

// What kind of tensor this is, at the level Chapter 4/7 distinguish it:
// a static weight (quantized once, offline, unlimited calibration time),
// an online activation (quantized fresh every token), or a KV cache entry
// (online, uncalibrated, no offline pass possible).
enum class StorageKind { STATIC_WEIGHT, ONLINE_ACTIVATION, KV_CACHE_ENTRY };

enum class RecommendedFormat { F32, Q8_0, Q4_0, DYNAMIC_INT8, TURBOQUANT_MSE };

std::string to_string(RecommendedFormat f) {
    switch (f) {
        case RecommendedFormat::F32: return "F32";
        case RecommendedFormat::Q8_0: return "Q8_0";
        case RecommendedFormat::Q4_0: return "Q4_0";
        case RecommendedFormat::DYNAMIC_INT8: return "DYNAMIC_INT8";
        case RecommendedFormat::TURBOQUANT_MSE: return "TURBOQUANT_MSE";
    }
    return "UNKNOWN";
}

struct FormatRecommendation {
    RecommendedFormat format;
    std::string reason;
};

FormatRecommendation recommend_quantization_format(StorageKind kind, WeightRole role) {
    if (kind == StorageKind::KV_CACHE_ENTRY) {
        return {RecommendedFormat::TURBOQUANT_MSE,
                "Chapter 7.4: online, uncalibrated cache data with no offline pass available -- "
                "MSE mode wins at the KV cache's typical 3-4 bit range without QJL's extra residual stage"};
    }
    if (kind == StorageKind::ONLINE_ACTIVATION) {
        return {RecommendedFormat::DYNAMIC_INT8,
                "Chapter 4.5: requantized fresh every token on the critical path -- nothing persisted "
                "to a file, so there is no stored scale worth shrinking to fp16"};
    }
    // StorageKind::STATIC_WEIGHT
    switch (role) {
        case WeightRole::NORM:
            return {RecommendedFormat::F32,
                    "Chapter 6.3: norm element count is negligible regardless of scale -- nothing real to save"};
        case WeightRole::EMBEDDING:
            return {RecommendedFormat::Q8_0,
                    "Chapter 6.3: embedding element count (2*VOCAB*DIM) can dwarf every other tensor "
                    "category at real vocabulary sizes -- the careful format is worth affording here"};
        case WeightRole::ATTENTION:
            return {RecommendedFormat::Q8_0,
                    "Chapter 6.1: attention feeds a discrete softmax-argmax decision a given error size "
                    "flips at a measurable, non-negligible rate"};
        case WeightRole::FFN:
            return {RecommendedFormat::Q4_0,
                    "Chapter 6.1: an FFN gate's rounding error stays a continuous, bounded output error, "
                    "and FFN matrices outnumber attention matrices roughly 3:1 per layer -- Q4_0's larger "
                    "absolute savings land exactly where precision is least fragile"};
    }
    return {RecommendedFormat::F32, "unreachable"};
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "Appendix C.1: The Quantization Format Advisor\n";
    std::cout << "====================================================\n\n";

    // -- Tests 1-4: the four real static-weight roles reproduce exactly
    // Chapter 6.2's own per-tensor policy (NORM->F32, EMBEDDING->Q8_0,
    // ATTENTION->Q8_0, FFN->Q4_0). --
    {
        auto norm = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::NORM);
        std::cout << "-- Test 1: static NORM weight -- format=" << to_string(norm.format) << " --\n";
        CHECK(norm.format == RecommendedFormat::F32);
    }
    {
        auto embed = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::EMBEDDING);
        std::cout << "-- Test 2: static EMBEDDING weight -- format=" << to_string(embed.format) << " --\n";
        CHECK(embed.format == RecommendedFormat::Q8_0);
    }
    {
        auto attn = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::ATTENTION);
        std::cout << "-- Test 3: static ATTENTION weight -- format=" << to_string(attn.format) << " --\n";
        CHECK(attn.format == RecommendedFormat::Q8_0);
    }
    {
        auto ffn = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::FFN);
        std::cout << "-- Test 4: static FFN weight -- format=" << to_string(ffn.format) << " --\n";
        CHECK(ffn.format == RecommendedFormat::Q4_0);
    }

    // -- Test 5: an online activation, regardless of which role it feeds,
    // always gets DYNAMIC_INT8 -- the STORAGE KIND decides here, not the
    // role. --
    {
        auto act = recommend_quantization_format(StorageKind::ONLINE_ACTIVATION, WeightRole::FFN);
        std::cout << "-- Test 5: online activation -- format=" << to_string(act.format) << " --\n";
        CHECK(act.format == RecommendedFormat::DYNAMIC_INT8);
        CHECK(act.reason.find("fresh every token") != std::string::npos);
    }

    // -- Test 6: a KV cache entry always gets TURBOQUANT_MSE -- again the
    // STORAGE KIND decides, not the role (a KV cache entry is always
    // attention-side data, but that is not why it gets this answer). --
    {
        auto kv = recommend_quantization_format(StorageKind::KV_CACHE_ENTRY, WeightRole::ATTENTION);
        std::cout << "-- Test 6: KV cache entry -- format=" << to_string(kv.format) << " --\n";
        CHECK(kv.format == RecommendedFormat::TURBOQUANT_MSE);
        CHECK(kv.reason.find("3-4 bit") != std::string::npos);
    }

    // -- Test 7: a small, hand-traceable memory-policy comparison,
    // reproducing Chapter 6.3's own real structural finding on a FRESH,
    // smaller illustrative model (not a reproduction of Chapter 6.3's own
    // specific multi-billion-parameter figures): a "naive" hybrid policy
    // that protects norm AND embedding at F32 together differs from the
    // "corrected" policy this advisor actually recommends (F32 norm only,
    // Q8_0 embedding) in EXACTLY one place -- the embedding format -- so
    // embedding accounts for the ENTIRE excess between them, by
    // construction, exactly as Chapter 6.3 found. Q8_0 costs 34 bytes per
    // 32-element block (Chapter 4.2) and Q4_0 costs 18 bytes per 32-element
    // block (Chapter 4.3); this advisor uses those same real per-block
    // byte counts as a per-element average, ignoring block-boundary
    // rounding, which is accurate enough for a comparison at this scale. --
    {
        constexpr double DIM = 8, N_LAYERS = 2, D_FF = 32, VOCAB = 50;
        constexpr double BYTES_PER_ELEM_F32 = 4.0;
        constexpr double BYTES_PER_ELEM_Q8_0 = 34.0 / 32.0;   // Chapter 4.2's own block size
        constexpr double BYTES_PER_ELEM_Q4_0 = 18.0 / 32.0;   // Chapter 4.3's own block size

        const double norm_elems = N_LAYERS * 2 * DIM + DIM;          // Chapter 6.3's own formula
        const double embed_elems = 2 * VOCAB * DIM;                  // Chapter 6.3's own formula
        const double attn_elems = N_LAYERS * 4 * DIM * DIM;          // Q, K, V, O per layer
        const double ffn_elems = N_LAYERS * 3 * DIM * D_FF;          // gate, up, down per layer

        auto norm_pick = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::NORM);
        auto embed_pick = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::EMBEDDING);
        auto attn_pick = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::ATTENTION);
        auto ffn_pick = recommend_quantization_format(StorageKind::STATIC_WEIGHT, WeightRole::FFN);
        CHECK(norm_pick.format == RecommendedFormat::F32);
        CHECK(embed_pick.format == RecommendedFormat::Q8_0);
        CHECK(attn_pick.format == RecommendedFormat::Q8_0);
        CHECK(ffn_pick.format == RecommendedFormat::Q4_0);

        const double norm_bytes = norm_elems * BYTES_PER_ELEM_F32;
        const double attn_bytes = attn_elems * BYTES_PER_ELEM_Q8_0;
        const double ffn_bytes = ffn_elems * BYTES_PER_ELEM_Q4_0;

        const double naive_hybrid_embed_bytes = embed_elems * BYTES_PER_ELEM_F32;  // the mistake: F32
        const double corrected_hybrid_embed_bytes = embed_elems * BYTES_PER_ELEM_Q8_0;  // this advisor's pick

        const double naive_hybrid_total = norm_bytes + naive_hybrid_embed_bytes + attn_bytes + ffn_bytes;
        const double corrected_hybrid_total = norm_bytes + corrected_hybrid_embed_bytes + attn_bytes + ffn_bytes;
        const double uniform_q4_0_total =
            (norm_elems + embed_elems + attn_elems + ffn_elems) * BYTES_PER_ELEM_Q4_0;
        const double uniform_q8_0_total =
            (norm_elems + embed_elems + attn_elems + ffn_elems) * BYTES_PER_ELEM_Q8_0;

        const double excess = naive_hybrid_total - corrected_hybrid_total;
        const double embedding_only_diff = naive_hybrid_embed_bytes - corrected_hybrid_embed_bytes;

        std::cout << "-- Test 7: small illustrative model (DIM=" << DIM << ", N_LAYERS=" << N_LAYERS
                  << ", D_FF=" << D_FF << ", VOCAB=" << VOCAB << ") --\n";
        std::cout << "   uniform Q4_0 total    = " << uniform_q4_0_total << " bytes\n";
        std::cout << "   uniform Q8_0 total    = " << uniform_q8_0_total << " bytes\n";
        std::cout << "   naive hybrid total    = " << naive_hybrid_total << " bytes (embedding kept F32)\n";
        std::cout << "   corrected hybrid total= " << corrected_hybrid_total << " bytes (embedding at Q8_0)\n";
        std::cout << "   excess (naive - corrected) = " << excess
                  << ", embedding-only difference = " << embedding_only_diff << "\n";

        // Corrected hybrid must sit strictly between the two uniform
        // policies, exactly as Chapter 6.3 found.
        CHECK(corrected_hybrid_total > uniform_q4_0_total);
        CHECK(corrected_hybrid_total < uniform_q8_0_total);
        // The naive-vs-corrected policies differ in exactly one place --
        // the embedding format -- so the embedding-only difference must
        // account for the WHOLE excess, not merely most of it.
        CHECK(excess == embedding_only_diff);
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return g_passed == g_tests ? 0 : 1;
}
