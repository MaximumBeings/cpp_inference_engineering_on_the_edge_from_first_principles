// 05_full_decode_step_multithreaded.cpp
// Chapter 11, Part 5 (capstone): every technique this chapter built,
// combined into one real, complete decode step -- and checked against a
// stricter standard than any single section needed on its own: the exact
// same result, bit-for-bit, no matter how many worker threads compute it.
//
// The pieces, each reused verbatim from its own section:
//   - Section 11.2's generalized ParallelPool and row-parallel matmul
//     phases (Q/K/V batched, output projection, FFN gate/up batched, FFN
//     down) -- already thread-count-independent, because row-parallel
//     assigns disjoint OUTPUT rows and never reorders any row's own
//     accumulation, regardless of how many threads share the work.
//   - Section 11.3's group-aligned head-parallel attention -- likewise
//     thread-count-independent for the identical reason, one head per
//     "row."
//   - Section 11.4's fixed-block reduction -- the one phase that is NOT
//     automatically thread-count-independent by construction: RMSNorm's
//     sum-of-squares is a genuine reduction over the whole vector, and
//     this section uses the fixed-block technique specifically so it
//     does not undo the guarantee every other phase already provides.
//
// Test 3 demonstrates why that last piece is not optional: swapping
// RMSNorm's reduction for the (individually reproducible, but not
// cross-thread-count-stable) per-thread-indexed version from Section
// 11.4's own contrast breaks bit-exactness for the ENTIRE decode step,
// even though every matmul and every attention head is still computed by
// row/head-parallel phases that never reorder anything on their own. One
// non-thread-count-independent reduction anywhere in a pipeline is enough
// to break the guarantee for the whole pipeline.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 05_full_decode_step_multithreaded.cpp -o 05_full_decode_step_multithreaded

#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mdspan/mdspan.hpp>
#include <set>
#include <span>
#include <thread>
#include <vector>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// =========================================================================
// SHARED INFRASTRUCTURE (Sections 11.2/11.3, unchanged): RowRange,
// ceiling partition, group-aligned partition, and the generalized
// persistent pool with explicit-ranges dispatch.
// =========================================================================
struct RowRange { size_t start, end; };
std::vector<RowRange> partition_ceil(size_t total, size_t n_threads) {
    std::vector<RowRange> ranges;
    size_t chunk = (total + n_threads - 1) / n_threads;
    for (size_t t = 0; t < n_threads; ++t) {
        size_t start = std::min(t * chunk, total);
        size_t end = std::min(start + chunk, total);
        ranges.push_back({start, end});
    }
    return ranges;
}
std::vector<RowRange> partition_group_aligned(size_t n_heads_q, size_t n_heads_kv, size_t group_size, size_t n_threads) {
    std::vector<RowRange> kv_ranges = partition_ceil(n_heads_kv, n_threads);
    std::vector<RowRange> q_ranges;
    for (const auto& kvr : kv_ranges) {
        size_t q_start = kvr.start * group_size;
        size_t q_end = std::min(kvr.end * group_size, n_heads_q);
        q_ranges.push_back({q_start, q_end});
    }
    return q_ranges;
}

class ParallelPool {
public:
    explicit ParallelPool(size_t n_threads)
        : n_threads_(n_threads),
          start_barrier_(static_cast<std::ptrdiff_t>(n_threads + 1)),
          done_barrier_(static_cast<std::ptrdiff_t>(n_threads + 1)) {
        for (size_t t = 0; t < n_threads_; ++t) workers_.emplace_back([this, t] { worker_loop(t); });
    }
    ~ParallelPool() {
        stop_flag_.store(true, std::memory_order_relaxed);
        start_barrier_.arrive_and_wait();
        for (auto& w : workers_) w.join();
    }
    void run_ranges(std::vector<RowRange> ranges, std::function<void(size_t, size_t)> fn) {
        task_fn_ = std::move(fn);
        task_ranges_ = std::move(ranges);
        start_barrier_.arrive_and_wait();
        done_barrier_.arrive_and_wait();
    }

private:
    void worker_loop(size_t my_index) {
        for (;;) {
            start_barrier_.arrive_and_wait();
            if (stop_flag_.load(std::memory_order_relaxed)) return;
            const RowRange& r = task_ranges_[my_index];
            if (r.start < r.end) task_fn_(r.start, r.end);
            done_barrier_.arrive_and_wait();
        }
    }
    size_t n_threads_;
    std::vector<std::thread> workers_;
    std::barrier<> start_barrier_;
    std::barrier<> done_barrier_;
    std::atomic<bool> stop_flag_{false};
    std::function<void(size_t, size_t)> task_fn_;
    std::vector<RowRange> task_ranges_;
};

// =========================================================================
// REUSED KERNELS (Chapters 3.1/3.2/3.4), each called from exactly ONE
// shared function regardless of code path -- Section 11.2's fix for the
// aarch64 FMA-contraction hazard, applied consistently here too.
// =========================================================================
void matmul_rows(std::span<float> out, std::span<const float> x, std::span<const float> W,
                  size_t in_dim, size_t row_start, size_t row_end) {
    for (size_t j = row_start; j < row_end; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < in_dim; ++i) sum += x[i] * W[j * in_dim + i];
        out[j] = sum;
    }
}
inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
void swiglu(std::span<const float> gate_proj, std::span<const float> up_proj, std::span<float> out) {
    for (size_t i = 0; i < out.size(); ++i) out[i] = silu(gate_proj[i]) * up_proj[i];
}
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
}

struct KVCache {
    std::vector<float> K, V;
    int n_heads_kv, max_seq_len, head_dim;
    using View = std::mdspan<float, std::dextents<size_t, 3>>;
    KVCache(int nh, int seq, int hd) : n_heads_kv(nh), max_seq_len(seq), head_dim(hd) {
        K.assign(static_cast<size_t>(nh) * seq * hd, 0.0f);
        V.assign(static_cast<size_t>(nh) * seq * hd, 0.0f);
    }
    View k_view() { return View(K.data(), n_heads_kv, max_seq_len, head_dim); }
    View v_view() { return View(V.data(), n_heads_kv, max_seq_len, head_dim); }
    auto k_at(int h, int t) { return std::submdspan(k_view(), h, t, std::full_extent); }
    auto v_at(int h, int t) { return std::submdspan(v_view(), h, t, std::full_extent); }
    void store(int h, int t, std::span<const float> k, std::span<const float> v) {
        auto kslice = k_at(h, t); auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[i]; vslice[i] = v[i]; }
    }
};
void attend_one_head(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                      int seq_len, int h, int kv_h) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len);
    std::span<const float> q(q_heads.data() + h * head_dim, head_dim);
    for (int t = 0; t < seq_len; ++t) {
        auto k = cache.k_at(kv_h, t);
        float d = 0.0f;
        for (int i = 0; i < head_dim; ++i) d += q[i] * k[i];
        scores[t] = d * scale;
    }
    softmax_inplace(std::span<float>(scores.data(), seq_len));
    float* out = output.data() + h * head_dim;
    std::fill(out, out + head_dim, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        auto v = cache.v_at(kv_h, t);
        float w = scores[t];
        for (int i = 0; i < head_dim; ++i) out[i] += w * v[i];
    }
}
inline int kv_head_of(int query_head, int group_size) { return query_head / group_size; }

// =========================================================================
// SECTION 11.4's FIXED-BLOCK REDUCTION, specialized to a sum-of-squares
// (RMSNorm's own reduction) instead of a two-array dot product. One
// shared per-block function, called identically whether there is one
// thread or several.
// =========================================================================
constexpr size_t RMS_BLOCK_SIZE = 16;
float block_sum_sq(std::span<const float> x, size_t blk) {
    size_t s = blk * RMS_BLOCK_SIZE, e = std::min(s + RMS_BLOCK_SIZE, x.size());
    float sum = 0.0f;
    for (size_t i = s; i < e; ++i) sum += x[i] * x[i];
    return sum;
}
// Serial (n_threads=1 in spirit): walk every block in fixed order,
// combine in fixed order. No pool required -- this IS what the parallel
// version below reduces to when everything happens to land on one
// thread, which is exactly why the two are guaranteed to agree.
float sum_sq_fixed_blocks_serial(std::span<const float> x) {
    size_t num_blocks = (x.size() + RMS_BLOCK_SIZE - 1) / RMS_BLOCK_SIZE;
    float total = 0.0f;
    for (size_t blk = 0; blk < num_blocks; ++blk) total += block_sum_sq(x, blk);
    return total;
}
float sum_sq_fixed_blocks_parallel(ParallelPool& pool, std::span<const float> x, size_t n_threads) {
    size_t num_blocks = (x.size() + RMS_BLOCK_SIZE - 1) / RMS_BLOCK_SIZE;
    std::vector<float> block_sums(num_blocks, 0.0f);
    pool.run_ranges(partition_ceil(num_blocks, n_threads), [&](size_t block_lo, size_t block_hi) {
        for (size_t blk = block_lo; blk < block_hi; ++blk) block_sums[blk] = block_sum_sq(x, blk);
    });
    float total = 0.0f;
    for (size_t blk = 0; blk < num_blocks; ++blk) total += block_sums[blk];
    return total;
}
// [COMMON TRAP, Test 3] the ALTERNATIVE, per-thread-indexed reduction
// Section 11.4 showed is reproducible for a fixed thread count but NOT
// across different ones -- used here to show what happens to a WHOLE
// decode step's bit-exactness when this one piece uses it instead.
float sum_sq_thread_indexed_parallel(ParallelPool& pool, std::span<const float> x, size_t n_threads) {
    std::vector<RowRange> ranges = partition_ceil(x.size(), n_threads);
    std::vector<float> partials(n_threads, 0.0f);
    pool.run_ranges(ranges, [&](size_t lo, size_t hi) {
        size_t t = 0;
        for (size_t k = 0; k < ranges.size(); ++k) if (ranges[k].start == lo && ranges[k].end == hi) { t = k; break; }
        float sum = 0.0f;
        for (size_t i = lo; i < hi; ++i) sum += x[i] * x[i];
        partials[t] = sum;
    });
    float total = 0.0f;
    for (size_t t = 0; t < n_threads; ++t) total += partials[t];
    return total;
}
void rms_norm_scale(std::span<float> out, std::span<const float> x, std::span<const float> weights,
                     float sum_sq, float epsilon = 1e-6f) {
    float rms_inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(x.size()) + epsilon);
    for (size_t i = 0; i < x.size(); ++i) out[i] = (x[i] * rms_inv) * weights[i];
}

// =========================================================================
// Model dimensions: identical GQA shape to Sections 11.2/11.3.
// =========================================================================
constexpr size_t DIM = 64, N_HEADS_Q = 8, N_HEADS_KV = 2, GROUP = N_HEADS_Q / N_HEADS_KV;
constexpr size_t HEAD_DIM = DIM / N_HEADS_Q, D_FF = 176;
constexpr int CACHE_LEN = 5;

struct LayerWeights {
    std::vector<float> attn_norm_w, ffn_norm_w, Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;
    LayerWeights(unsigned seed) {
        attn_norm_w.assign(DIM, 1.0f); ffn_norm_w.assign(DIM, 1.0f);
        auto fill = [&](std::vector<float>& v, size_t n, unsigned s) {
            v.resize(n);
            for (size_t i = 0; i < n; ++i) v[i] = 0.02f * (static_cast<float>((i * 37 + s * 101) % 23) - 11.0f);
        };
        fill(Wq, DIM * DIM, seed + 1);
        fill(Wk, (N_HEADS_KV * HEAD_DIM) * DIM, seed + 2);
        fill(Wv, (N_HEADS_KV * HEAD_DIM) * DIM, seed + 3);
        fill(Wo, DIM * DIM, seed + 4);
        fill(Wgate, D_FF * DIM, seed + 5);
        fill(Wup, D_FF * DIM, seed + 6);
        fill(Wdown, DIM * D_FF, seed + 7);
    }
};
KVCache make_filled_cache(unsigned seed) {
    KVCache cache(static_cast<int>(N_HEADS_KV), CACHE_LEN + 1, static_cast<int>(HEAD_DIM));
    std::vector<float> k(HEAD_DIM), v(HEAD_DIM);
    for (int h = 0; h < static_cast<int>(N_HEADS_KV); ++h) {
        for (int t = 0; t < CACHE_LEN; ++t) {
            for (size_t i = 0; i < HEAD_DIM; ++i) {
                k[i] = 0.01f * static_cast<float>((h * 13 + t * 7 + static_cast<int>(i) + static_cast<int>(seed)) % 11 - 5);
                v[i] = 0.02f * static_cast<float>((h * 5 + t * 3 + static_cast<int>(i) + static_cast<int>(seed)) % 9 - 4);
            }
            cache.store(h, t, k, v);
        }
    }
    return cache;
}

// =========================================================================
// SERIAL REFERENCE: one layer's full decode step. RMSNorm's reduction
// uses sum_sq_fixed_blocks_serial specifically so the parallel version's
// use of the SAME block structure is a fair, bit-exact comparison -- not
// a comparison against a differently-ordered plain left-to-right sum.
// =========================================================================
void layer_serial(LayerWeights& w, KVCache& cache, std::vector<float>& x) {
    std::vector<float> xn(DIM);
    rms_norm_scale(xn, x, w.attn_norm_w, sum_sq_fixed_blocks_serial(x));

    std::vector<float> q(DIM), k(N_HEADS_KV * HEAD_DIM), v(N_HEADS_KV * HEAD_DIM);
    matmul_rows(q, xn, w.Wq, DIM, 0, DIM);
    matmul_rows(k, xn, w.Wk, DIM, 0, N_HEADS_KV * HEAD_DIM);
    matmul_rows(v, xn, w.Wv, DIM, 0, N_HEADS_KV * HEAD_DIM);

    for (size_t h = 0; h < N_HEADS_KV; ++h)
        cache.store(static_cast<int>(h), CACHE_LEN,
                    std::span<const float>(k.data() + h * HEAD_DIM, HEAD_DIM),
                    std::span<const float>(v.data() + h * HEAD_DIM, HEAD_DIM));

    std::vector<float> attn_out(DIM);
    for (size_t h = 0; h < N_HEADS_Q; ++h)
        attend_one_head(q, cache, attn_out, CACHE_LEN + 1, static_cast<int>(h), kv_head_of(static_cast<int>(h), GROUP));

    std::vector<float> proj(DIM);
    matmul_rows(proj, attn_out, w.Wo, DIM, 0, DIM);
    for (size_t i = 0; i < DIM; ++i) x[i] += proj[i];

    rms_norm_scale(xn, x, w.ffn_norm_w, sum_sq_fixed_blocks_serial(x));
    std::vector<float> gate(D_FF), up(D_FF), hidden(D_FF), ffn_out(DIM);
    matmul_rows(gate, xn, w.Wgate, DIM, 0, D_FF);
    matmul_rows(up, xn, w.Wup, DIM, 0, D_FF);
    swiglu(gate, up, hidden);
    matmul_rows(ffn_out, hidden, w.Wdown, D_FF, 0, DIM);
    for (size_t i = 0; i < DIM; ++i) x[i] += ffn_out[i];
}

// =========================================================================
// PARALLEL LAYER: the whole decode step through ONE persistent pool.
// use_thread_indexed_reduction selects Test 3's deliberately-broken
// RMSNorm reduction instead of the fixed-block one -- everything else
// about the pipeline is unchanged.
// =========================================================================
void layer_parallel(ParallelPool& pool, size_t n_threads, LayerWeights& w, KVCache& cache,
                     std::vector<float>& x, bool use_thread_indexed_reduction = false) {
    constexpr size_t KV_ROWS = N_HEADS_KV * HEAD_DIM;
    std::vector<float> xn(DIM);
    float sum_sq1 = use_thread_indexed_reduction
        ? sum_sq_thread_indexed_parallel(pool, x, n_threads)
        : sum_sq_fixed_blocks_parallel(pool, x, n_threads);
    rms_norm_scale(xn, x, w.attn_norm_w, sum_sq1);

    std::vector<float> q(DIM), k(KV_ROWS), v(KV_ROWS);
    pool.run_ranges(partition_ceil(DIM, n_threads), [&](size_t lo, size_t hi) {
        matmul_rows(q, xn, w.Wq, DIM, lo, hi);
        size_t kv_lo = std::min(lo, KV_ROWS), kv_hi = std::min(hi, KV_ROWS);
        if (kv_lo < kv_hi) { matmul_rows(k, xn, w.Wk, DIM, kv_lo, kv_hi); matmul_rows(v, xn, w.Wv, DIM, kv_lo, kv_hi); }
    });

    for (size_t h = 0; h < N_HEADS_KV; ++h)
        cache.store(static_cast<int>(h), CACHE_LEN,
                    std::span<const float>(k.data() + h * HEAD_DIM, HEAD_DIM),
                    std::span<const float>(v.data() + h * HEAD_DIM, HEAD_DIM));

    std::vector<float> attn_out(DIM);
    std::vector<RowRange> attn_ranges = partition_group_aligned(N_HEADS_Q, N_HEADS_KV, GROUP, n_threads);
    pool.run_ranges(attn_ranges, [&](size_t lo, size_t hi) {
        for (size_t h = lo; h < hi; ++h)
            attend_one_head(q, cache, attn_out, CACHE_LEN + 1, static_cast<int>(h), kv_head_of(static_cast<int>(h), GROUP));
    });

    std::vector<float> proj(DIM);
    pool.run_ranges(partition_ceil(DIM, n_threads), [&](size_t lo, size_t hi) { matmul_rows(proj, attn_out, w.Wo, DIM, lo, hi); });
    for (size_t i = 0; i < DIM; ++i) x[i] += proj[i];

    float sum_sq2 = use_thread_indexed_reduction
        ? sum_sq_thread_indexed_parallel(pool, x, n_threads)
        : sum_sq_fixed_blocks_parallel(pool, x, n_threads);
    rms_norm_scale(xn, x, w.ffn_norm_w, sum_sq2);

    std::vector<float> gate(D_FF), up(D_FF), hidden(D_FF), ffn_out(DIM);
    pool.run_ranges(partition_ceil(D_FF, n_threads), [&](size_t lo, size_t hi) {
        matmul_rows(gate, xn, w.Wgate, DIM, lo, hi);
        matmul_rows(up, xn, w.Wup, DIM, lo, hi);
    });
    swiglu(gate, up, hidden);
    pool.run_ranges(partition_ceil(DIM, n_threads), [&](size_t lo, size_t hi) { matmul_rows(ffn_out, hidden, w.Wdown, D_FF, lo, hi); });
    for (size_t i = 0; i < DIM; ++i) x[i] += ffn_out[i];
}

std::vector<float> initial_x(unsigned seed) {
    std::vector<float> x(DIM);
    for (size_t i = 0; i < DIM; ++i) x[i] = 0.025f * (static_cast<float>((i * 3 + seed) % 17) - 8.0f);
    return x;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 11.5: A Fully Multi-Threaded Decode Step\n";
    std::cout << "========================================================\n\n";

    constexpr int N_LAYERS = 4;
    std::vector<unsigned> seeds; for (int l = 0; l < N_LAYERS; ++l) seeds.push_back(200 + l * 31);

    auto run_serial_reference = [&]() {
        std::vector<LayerWeights> layers; for (unsigned s : seeds) layers.emplace_back(s);
        std::vector<KVCache> caches; for (unsigned s : seeds) caches.push_back(make_filled_cache(s));
        std::vector<float> x = initial_x(11);
        for (int l = 0; l < N_LAYERS; ++l) layer_serial(layers[l], caches[l], x);
        return x;
    };
    std::vector<float> x_ref = run_serial_reference();

    // =====================================================================
    // TEST 1: the full 4-layer decode step, run through ONE persistent
    // pool with 4 threads -- every phase (parallel RMSNorm reduction,
    // batched QKV, group-aligned head-parallel attention, output
    // projection, batched FFN gate/up, FFN down) -- matches the serial
    // reference bit-for-bit.
    // =====================================================================
    std::cout << "-- Test 1: full " << N_LAYERS << "-layer decode step, 4 threads, vs. serial reference --\n";
    {
        std::vector<LayerWeights> layers; for (unsigned s : seeds) layers.emplace_back(s);
        std::vector<KVCache> caches; for (unsigned s : seeds) caches.push_back(make_filled_cache(s));
        std::vector<float> x = initial_x(11);
        constexpr size_t N_THREADS = 4;
        {
            ParallelPool pool(N_THREADS);
            for (int l = 0; l < N_LAYERS; ++l) layer_parallel(pool, N_THREADS, layers[l], caches[l], x);
        }
        bool exact_match = (x == x_ref);
        std::cout << "  DIM=" << DIM << " D_FF=" << D_FF << " " << N_LAYERS << " layers, " << N_THREADS
                  << " threads: " << (exact_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(exact_match);
    }

    // =====================================================================
    // TEST 2: the SAME pipeline, run again with a DIFFERENT thread count
    // (3 threads, through a freshly created pool) -- still matches the
    // serial reference bit-for-bit. This is the chapter's real payoff:
    // not merely "correct," and not merely "reproducible on a rerun," but
    // reproducible ACROSS a configuration change, because every phase
    // that combines partial results (the RMSNorm reduction) was built
    // with that specific property in mind.
    // =====================================================================
    std::cout << "\n-- Test 2: SAME decode step, DIFFERENT thread count (3), vs. serial reference --\n";
    {
        std::vector<LayerWeights> layers; for (unsigned s : seeds) layers.emplace_back(s);
        std::vector<KVCache> caches; for (unsigned s : seeds) caches.push_back(make_filled_cache(s));
        std::vector<float> x = initial_x(11);
        constexpr size_t N_THREADS = 3;
        {
            ParallelPool pool(N_THREADS);
            for (int l = 0; l < N_LAYERS; ++l) layer_parallel(pool, N_THREADS, layers[l], caches[l], x);
        }
        bool exact_match = (x == x_ref);
        std::cout << "  same " << N_LAYERS << " layers, " << N_THREADS << " threads instead of 4: "
                  << (exact_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(exact_match);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: swap ONLY the RMSNorm reduction for Section
    // 11.4's thread-indexed version -- reproducible for a fixed thread
    // count, but not across different ones. Every matmul phase and every
    // attention head is untouched, still perfectly row/head-parallel --
    // and the whole decode step's output still breaks across a thread-
    // count change anyway, because one non-thread-count-independent
    // reduction is enough to poison everything computed from it onward.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: one thread-indexed reduction breaks the WHOLE pipeline --\n";
    {
        auto run_with_threads = [&](size_t n_threads) {
            std::vector<LayerWeights> layers; for (unsigned s : seeds) layers.emplace_back(s);
            std::vector<KVCache> caches; for (unsigned s : seeds) caches.push_back(make_filled_cache(s));
            std::vector<float> x = initial_x(11);
            ParallelPool pool(n_threads);
            for (int l = 0; l < N_LAYERS; ++l) layer_parallel(pool, n_threads, layers[l], caches[l], x, /*use_thread_indexed_reduction=*/true);
            return x;
        };
        std::vector<float> x4 = run_with_threads(4);
        std::vector<float> x3 = run_with_threads(3);
        bool same_output_across_thread_counts = (x4 == x3);
        std::cout << "  4-thread run vs. 3-thread run, same decode step, broken reduction only: "
                  << (same_output_across_thread_counts ? "still matched (unexpected)" : "DIFFERS, as expected") << "\n";
        std::cout << "  Every matmul phase and every attention head in this run is still exactly the\n";
        std::cout << "  same row/head-parallel code Tests 1 and 2 just verified as thread-count-\n";
        std::cout << "  independent on its own. The ONLY change is which reduction RMSNorm uses --\n";
        std::cout << "  and that alone is enough to make the final output depend on thread count.\n";
        CHECK(!same_output_across_thread_counts);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
