// Chapter 26.6 -- This chapter's own capstone: every real formula built
// in Sections 26.1 through 26.5 -- FLOP counting, weight-dominated byte
// accounting, arithmetic intensity, and the real roofline crossover --
// applied together to a complete real transformer decoder layer (QKV
// projection, attention, output projection, and the FFN), at a stated
// real shape, to classify the WHOLE layer as memory-bound or
// compute-bound at a given real batch size, and to derive the real
// batch size at which that classification flips.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 06_full_transformer_layer_roofline_analysis.cpp -o 06_full_transformer_layer_roofline_analysis
// Run:     ./06_full_transformer_layer_roofline_analysis

#include <cmath>
#include <cstdint>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) < eps; }

constexpr double BYTES_PER_FLOAT32 = 4.0;

// =======================================================================
// PART 1: a real transformer decoder layer's own shape.
// =======================================================================
struct LayerShape {
    int64_t hidden = 0, heads = 0, head_dim = 0, ffn_dim = 0, seq_len = 0;
};

struct FlopBreakdown {
    double qkv = 0.0, attn = 0.0, out_proj = 0.0, ffn = 0.0, total = 0.0;
};

struct ByteBreakdown {
    double qkv = 0.0, out_proj = 0.0, ffn = 0.0, total = 0.0;
};

// =======================================================================
// PART 2: real FLOP counting for each real sub-block of the layer, at a
// stated real batch size. M = batch * seq_len is the real number of
// token-rows the QKV and output projections, and the FFN, each run as a
// GEMM over.
// =======================================================================
FlopBreakdown layer_flops(const LayerShape& s, int64_t batch) {
    double m = static_cast<double>(batch) * static_cast<double>(s.seq_len);
    double hidden = static_cast<double>(s.hidden);
    double ffn_dim = static_cast<double>(s.ffn_dim);
    double seq_len = static_cast<double>(s.seq_len);
    double b = static_cast<double>(batch);

    FlopBreakdown f;
    f.qkv = 3.0 * 2.0 * m * hidden * hidden;               // 3 separate hidden -> hidden projections
    f.attn = 4.0 * b * seq_len * seq_len * hidden;          // real QK^T plus attn*V, summed over all heads
    f.out_proj = 2.0 * m * hidden * hidden;                 // one hidden -> hidden projection
    f.ffn = 4.0 * m * hidden * ffn_dim;                     // hidden->ffn_dim, then ffn_dim->hidden
    f.total = f.qkv + f.attn + f.out_proj + f.ffn;
    return f;
}

// =======================================================================
// PART 3: real weight-byte accounting -- the same weight-dominated
// approximation Section 26.1 derived and honestly bounded, applied here
// to the layer's own 4 real weight tensors. Batch-independent by
// construction: the weights themselves do not grow with batch size.
// =======================================================================
ByteBreakdown layer_weight_bytes(const LayerShape& s) {
    double hidden = static_cast<double>(s.hidden);
    double ffn_dim = static_cast<double>(s.ffn_dim);

    ByteBreakdown b;
    b.qkv = 3.0 * hidden * hidden * BYTES_PER_FLOAT32;
    b.out_proj = hidden * hidden * BYTES_PER_FLOAT32;
    b.ffn = 2.0 * hidden * ffn_dim * BYTES_PER_FLOAT32;
    b.total = b.qkv + b.out_proj + b.ffn;
    return b;
}

double arithmetic_intensity(double flops, double bytes) { return flops / bytes; }

enum class BoundClass { MEMORY_BOUND, COMPUTE_BOUND };

BoundClass classify_bound(double ai, double ridge_point) {
    return (ai < ridge_point) ? BoundClass::MEMORY_BOUND : BoundClass::COMPUTE_BOUND;
}

// Because every term in layer_flops scales linearly with batch (M and
// the explicit batch factor in the attention term both do), and
// layer_weight_bytes does not depend on batch at all, this layer's own
// real arithmetic intensity is EXACTLY linear in batch: AI(batch) =
// AI(1) * batch. Solving AI(1) * batch = ridge_point gives the real
// crossover batch size directly.
double crossover_batch_size(const LayerShape& s, double ridge_point) {
    FlopBreakdown f1 = layer_flops(s, 1);
    ByteBreakdown bytes = layer_weight_bytes(s);
    double ai_at_batch_1 = arithmetic_intensity(f1.total, bytes.total);
    return ridge_point / ai_at_batch_1;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.6: A Full Roofline Analysis of a Transformer Layer\n";
    std::cout << "========================================================\n";

    // A tiny, fully hand-traceable real layer shape.
    LayerShape shape{4, 2, 2, 8, 2};  // hidden=4, heads=2, head_dim=2, ffn_dim=8, seq_len=2

    std::cout << "\n-- Test 1: at batch 1, every real FLOP sub-total, the real total, the real weight-"
                 "byte total, and the resulting real arithmetic intensity all match an exact hand "
                 "computation for this tiny, fully traceable layer shape --\n";
    {
        FlopBreakdown f = layer_flops(shape, 1);
        CHECK(near(f.qkv, 192.0));
        CHECK(near(f.attn, 64.0));
        CHECK(near(f.out_proj, 64.0));
        CHECK(near(f.ffn, 256.0));
        CHECK(near(f.total, 576.0));

        ByteBreakdown b = layer_weight_bytes(shape);
        CHECK(near(b.qkv, 192.0));
        CHECK(near(b.out_proj, 64.0));
        CHECK(near(b.ffn, 256.0));
        CHECK(near(b.total, 512.0));

        double ai = arithmetic_intensity(f.total, b.total);
        CHECK(near(ai, 1.125));
        std::cout << "  at batch 1, this tiny layer's own real FLOPs are exactly 192 (QKV) + 64 "
                     "(attention) + 64 (output projection) + 256 (FFN) = 576 total; its own real weight "
                     "bytes are exactly 192 + 64 + 256 = 512; its own real arithmetic intensity is "
                     "exactly 576 / 512 = 1.125 FLOPs/byte -- every sub-total matching a direct hand "
                     "computation from this layer's own stated shape\n";
    }

    std::cout << "\n-- Test 2: doubling the real batch size exactly doubles the real total FLOPs and "
                 "the real arithmetic intensity, while the real weight bytes stay exactly unchanged -- "
                 "confirming the algebraic linear-in-batch property this section's own crossover formula "
                 "depends on, checked directly rather than merely asserted --\n";
    {
        FlopBreakdown f1 = layer_flops(shape, 1);
        FlopBreakdown f2 = layer_flops(shape, 2);
        CHECK(near(f2.total, 2.0 * f1.total));
        CHECK(near(f2.total, 1152.0));

        ByteBreakdown b1 = layer_weight_bytes(shape);
        ByteBreakdown b2 = layer_weight_bytes(shape);  // batch-independent -- recomputed, still identical
        CHECK(near(b1.total, b2.total));

        double ai1 = arithmetic_intensity(f1.total, b1.total);
        double ai2 = arithmetic_intensity(f2.total, b2.total);
        CHECK(near(ai2, 2.0 * ai1));
        CHECK(near(ai2, 2.25));
        std::cout << "  at batch 2, real total FLOPs are exactly 1152 -- precisely double batch 1's 576 "
                     "-- while real weight bytes remain exactly 512, unchanged; the resulting real "
                     "arithmetic intensity, 2.25, is precisely double batch 1's 1.125, confirming this "
                     "layer's own real arithmetic intensity scales exactly linearly with batch size\n";
    }

    std::cout << "\n-- Test 3: against a stated real machine ridge point sitting between this layer's own "
                 "batch-1 and batch-2 arithmetic intensities, the SAME layer classifies as memory-bound "
                 "at batch 1 and compute-bound at batch 2 -- the real roofline model's own central "
                 "prediction, applied to a whole real transformer layer rather than a single operation "
                 "--\n";
    {
        double ridge = 2.0;
        FlopBreakdown f1 = layer_flops(shape, 1);
        FlopBreakdown f2 = layer_flops(shape, 2);
        ByteBreakdown bytes = layer_weight_bytes(shape);

        double ai1 = arithmetic_intensity(f1.total, bytes.total);
        double ai2 = arithmetic_intensity(f2.total, bytes.total);

        CHECK(classify_bound(ai1, ridge) == BoundClass::MEMORY_BOUND);
        CHECK(classify_bound(ai2, ridge) == BoundClass::COMPUTE_BOUND);
        std::cout << "  against a real stated machine ridge point of 2.0 FLOPs/byte, this layer's own "
                     "batch-1 arithmetic intensity of 1.125 classifies MEMORY_BOUND, and the identical "
                     "layer's own batch-2 arithmetic intensity of 2.25 classifies COMPUTE_BOUND -- the "
                     "same real layer shape, genuinely different real classifications, purely as a "
                     "function of real batch size\n";
    }

    std::cout << "\n-- Test 4: the real crossover batch size computed directly from this layer's own "
                 "shape and a stated ridge point falls exactly between the two batch sizes Test 3 "
                 "observed flipping classification, confirming the closed-form crossover formula "
                 "predicts the identical real transition rather than merely rationalizing it after the "
                 "fact --\n";
    {
        double ridge = 2.0;
        double crossover = crossover_batch_size(shape, ridge);
        CHECK(near(crossover, 2.0 / 1.125));
        CHECK(crossover > 1.0);
        CHECK(crossover < 2.0);
        std::cout << "  the real crossover batch size for this layer's own shape against a ridge point "
                     "of 2.0 is exactly " << crossover << " -- strictly between batch 1 (observed "
                     "MEMORY_BOUND in Test 3) and batch 2 (observed COMPUTE_BOUND) -- confirming the "
                     "closed-form crossover formula predicts precisely the transition this section "
                     "already observed directly\n";
    }

    std::cout << "\n-- Test 5: the same crossover formula behaves consistently across several genuinely "
                 "different real ridge points, always landing exactly where AI(1) * crossover = ridge, "
                 "checked algebraically rather than against a single hand-picked example --\n";
    {
        for (double ridge : {0.5, 1.125, 5.0, 40.0}) {
            double crossover = crossover_batch_size(shape, ridge);
            FlopBreakdown f1 = layer_flops(shape, 1);
            ByteBreakdown bytes = layer_weight_bytes(shape);
            double ai1 = arithmetic_intensity(f1.total, bytes.total);
            CHECK(near(ai1 * crossover, ridge, 1e-6));
        }
        std::cout << "  across 4 genuinely different real ridge points -- 0.5, 1.125 (the layer's own "
                     "exact batch-1 arithmetic intensity), 5.0, and 40.0 -- the computed real crossover "
                     "batch size always satisfies AI(1) * crossover = ridge_point exactly, confirming "
                     "the formula's own real algebraic correctness rather than a single coincidental "
                     "match\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
