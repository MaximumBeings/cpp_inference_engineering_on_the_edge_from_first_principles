// 01_kv_cache_bandwidth_estimator.cpp
// Chapter 13, Part 1: every chapter through Part 2 optimized how fast the
// CPU could compute. This chapter starts from a different bottleneck: an
// autoregressive model is stateful, and generating the Nth token requires
// attending over all N-1 tokens that came before. Storing their Key and
// Value vectors instead of recomputing them turns an O(N^2) total-FLOP
// generation into O(N) per step -- the KV cache is what makes that trade,
// swapping recomputation for memory. But memory has its own speed limit,
// and once that trade is made, the KV cache itself becomes the dominant
// cost at long context.
//
// This section makes the decode-time "memory wall" concrete with one
// equation: tokens/second is bounded above by (memory bandwidth) /
// (bytes that must be read per token step), where those bytes are the
// model's weights PLUS its KV cache. More cores, a higher clock, or a
// wider FPU do not move this ceiling at all -- decode is memory-bound,
// not compute-bound, which is exactly the opposite of the prefill phase
// Chapter 8's roofline model analyzed. Every number this file prints is
// derived from that one equation and real, published model shapes
// (Llama 3 8B/70B, Mistral 7B) -- nothing here is a fabricated benchmark.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_kv_cache_bandwidth_estimator.cpp -o 01_kv_cache_bandwidth_estimator

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) < (tol))

// Every parameter needed to compute a model's weight size and KV cache
// size. All values below are real published model shapes, not invented.
struct ModelCfg {
    std::string name;
    double params_billions;
    int n_layers;
    int n_heads_kv;   // GQA: fewer KV heads than query heads
    int head_dim;
};

struct QuantScheme {
    std::string name;
    double bytes_per_weight;  // FP32=4.0, FP16=2.0; Q8/Q4 include their per-block scale overhead
};

struct Analysis {
    double weight_gb;
    double kv_cache_gb;
    double total_gb;
    double max_tokens_per_sec;
    double kv_pct_of_total;
};

// The one equation this whole section derives from: decode-step data
// volume (weights + KV cache) divided into available bandwidth is the
// theoretical ceiling on tokens/second. KV cache size is 2 tensors
// (K and V) x n_layers x n_heads_kv x context_len x head_dim x bytes.
Analysis compute(const ModelCfg& m, const QuantScheme& q, int ctx_len, double bandwidth_gbs) {
    double weight_gb = m.params_billions * q.bytes_per_weight;  // params_billions already in units of 1e9
    double kv_elements = 2.0 * m.n_layers * m.n_heads_kv * static_cast<double>(ctx_len) * m.head_dim;
    double kv_dtype_bytes = 2.0;  // KV cache stored as FP16 in every configuration this section checks
    double kv_gb = kv_elements * kv_dtype_bytes / 1e9;
    double total_gb = weight_gb + kv_gb;
    double max_tps = bandwidth_gbs / total_gb;
    double kv_pct = kv_gb / total_gb * 100.0;
    return {weight_gb, kv_gb, total_gb, max_tps, kv_pct};
}

void print_context_table(const ModelCfg& m, const QuantScheme& q, double bw) {
    std::cout << "\n  " << m.name << " (" << q.name << ") @ " << bw << " GB/s:\n";
    std::cout << "    context  weights   kv_cache    total   tok/s   kv%\n";
    for (int ctx : {512, 1024, 2048, 4096, 8192, 16384, 32768, 131072}) {
        auto a = compute(m, q, ctx, bw);
        if (a.total_gb > 256.0) break;
        std::cout << "    " << std::setw(6) << ctx << "  "
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << a.weight_gb << "  "
                  << std::setw(9) << a.kv_cache_gb << "  "
                  << std::setw(7) << a.total_gb << "  "
                  << std::setprecision(1) << std::setw(6) << a.max_tokens_per_sec << "  "
                  << std::setprecision(0) << std::setw(4) << a.kv_pct_of_total << "%\n";
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 13.1: The Memory Wall -- Bandwidth-Limited Decode\n";
    std::cout << "========================================================\n";

    ModelCfg llama3_8b  = {"Llama 3 8B",  8.0,  32, 8, 128};
    QuantScheme fp16 = {"FP16", 2.0};
    QuantScheme fp32 = {"FP32", 4.0};
    QuantScheme q4   = {"Q4",   0.5625};  // 4 bits + one FP16 scale per 32 weights: (4 + 16/32)/8 bytes

    std::cout << "\n-- Test 1: KV cache size formula matches a hand calculation --\n";
    {
        // Llama 3 8B, FP16 KV, 4096-token context:
        // 2 (K,V) x 32 layers x 8 KV heads x 4096 tokens x 128 dims x 2 bytes
        auto a = compute(llama3_8b, fp16, 4096, 90.0);
        double expected_bytes = 2.0 * 32 * 8 * 4096 * 128 * 2;
        double expected_gb = expected_bytes / 1e9;
        std::cout << "  hand calc:  " << expected_bytes << " bytes = " << expected_gb << " GB\n";
        std::cout << "  computed:   " << a.kv_cache_gb << " GB\n";
        CHECK_NEAR(a.kv_cache_gb, expected_gb, 0.001);
        CHECK_NEAR(expected_gb, 0.536870912, 1e-6);
    }

    std::cout << "\n-- Test 2: speed limit for a realistic Q4 configuration --\n";
    {
        auto a = compute(llama3_8b, q4, 4096, 90.0);
        std::cout << "  Llama 3 8B, Q4 weights, ctx=4096, DDR5 @ 90 GB/s:\n";
        std::cout << "    weights:  " << std::fixed << std::setprecision(3) << a.weight_gb << " GB\n";
        std::cout << "    kv cache: " << a.kv_cache_gb << " GB\n";
        std::cout << "    total:    " << a.total_gb << " GB per decode step\n";
        std::cout << "    speed:    " << std::setprecision(1) << a.max_tokens_per_sec << " tok/s (theoretical ceiling)\n";
        CHECK(a.max_tokens_per_sec > 15.0 && a.max_tokens_per_sec < 25.0);
    }

    std::cout << "\n-- Test 3: speed degrades monotonically as context grows --\n";
    {
        double prev = 1e9;
        for (int ctx : {512, 1024, 2048, 4096, 8192}) {
            auto a = compute(llama3_8b, q4, ctx, 90.0);
            CHECK(a.max_tokens_per_sec < prev);
            std::cout << "  ctx=" << std::setw(6) << ctx << " -> "
                      << std::fixed << std::setprecision(1) << a.max_tokens_per_sec << " tok/s\n";
            prev = a.max_tokens_per_sec;
        }
    }

    std::cout << "\n-- Test 4: KV cache dominates bandwidth at very long context --\n";
    {
        auto a = compute(llama3_8b, q4, 131072, 90.0);
        std::cout << "  at 128K context: weights=" << std::fixed << std::setprecision(2) << a.weight_gb
                  << " GB, kv_cache=" << a.kv_cache_gb << " GB ("
                  << std::setprecision(0) << a.kv_pct_of_total << "% of total bandwidth)\n";
        CHECK(a.kv_pct_of_total > 70.0);
    }

    std::cout << "\n-- Test 5: quantization speedup is largest when the KV cache is small --\n";
    {
        auto a_fp32 = compute(llama3_8b, fp32, 512, 90.0);
        auto a_q4 = compute(llama3_8b, q4, 512, 90.0);
        double speedup = a_q4.max_tokens_per_sec / a_fp32.max_tokens_per_sec;
        std::cout << "  ctx=512 (weight-dominated): FP32=" << std::fixed << std::setprecision(1)
                  << a_fp32.max_tokens_per_sec << " tok/s, Q4=" << a_q4.max_tokens_per_sec
                  << " tok/s, speedup=" << std::setprecision(2) << speedup << "x\n";
        // Weight bytes/param drops from 4.0 to 0.5625, a 7.11x compression;
        // at a context short enough that KV cache is nearly negligible,
        // speed scales close to that same ratio.
        CHECK(speedup > 4.0 && speedup < 8.0);
    }

    std::cout << "\n-- reference table: Llama 3 8B, Q4 weights, DDR5 @ 90 GB/s --\n";
    print_context_table(llama3_8b, q4, 90.0);

    std::cout << "\n-- reference table: hardware bandwidth comparison (ctx=4096, Q4) --\n";
    struct HW { const char* name; double bw; };
    std::cout << "    hardware               bandwidth(GB/s)   tok/s\n";
    for (const auto& hw : (HW[]){
             {"Laptop DDR4", 40.0}, {"Desktop DDR5", 90.0}, {"MacBook M2", 100.0},
             {"MacBook M2 Ultra", 400.0}, {"A100 HBM2e", 2000.0}, {"H100 HBM3e", 3350.0}}) {
        auto a = compute(llama3_8b, q4, 4096, hw.bw);
        std::cout << "    " << std::left << std::setw(20) << hw.name << std::right
                  << std::setw(12) << std::fixed << std::setprecision(0) << hw.bw
                  << std::setw(10) << std::setprecision(1) << a.max_tokens_per_sec << "\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
