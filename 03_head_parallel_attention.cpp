// 03_head_parallel_attention.cpp
// Chapter 11, Part 3: Section 11.2 kept attention itself sequential and
// parallelized only the surrounding matmuls. Attention is, in fact, the
// most embarrassingly parallel phase in the whole layer: GQA computes an
// entirely independent score vector, softmax, and weighted sum for every
// query head, writing to a disjoint slice of the output vector -- no
// combination step, no shared mutable state between heads, exactly the
// row-parallel shape Chapter 10.3 and Section 11.2 already established
// as bit-for-bit reproducible.
//
// The real design question this section answers is not "can attention be
// parallelized by head" (it obviously can) but "how should heads be
// assigned to threads." GQA's whole premise is that several query heads
// SHARE one KV head's cached rows -- a "group." Partitioning query heads
// evenly by INDEX, ignoring group boundaries, routinely splits a single
// KV head's group across two different threads, so that KV head's cache
// rows get fetched by more than one thread instead of once. Partitioning
// by whole KV GROUPS instead -- each thread owns one or more complete
// groups, never a fraction of one -- guarantees every KV head's rows are
// touched by exactly one thread. This section counts that redundancy
// directly (which KV head index each thread's assigned heads resolve to,
// and how many DISTINCT threads end up touching each one) rather than
// timing anything, the same structural-arithmetic verification Chapter
// 10.3's Test 1 used for its own row partition.
//
// This file also inherits Section 11.2's own hard-won lesson: the serial
// reference and every parallel code path call the IDENTICAL free function
// (attend_one_head) for a single head's full computation, and the file
// compiles with -ffp-contract=off, so "bit-for-bit" is a claim that holds
// on every architecture this book cross-checks, not an accident of one
// compiler's default FMA-fusion heuristic.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_head_parallel_attention.cpp -o 03_head_parallel_attention

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
// KVCache: Chapter 3.4's mdspan-backed cache, reused verbatim.
// =========================================================================
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
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[i]; vslice[i] = v[i]; }
    }
};

// =========================================================================
// ONE HEAD'S FULL COMPUTATION -- scores, softmax, weighted sum -- as a
// single free function called from every code path in this file (serial
// loop, naive-partition parallel phase, group-aligned parallel phase).
// This is the fix for Section 11.2's discovered aarch64 hazard: there is
// only ONE place this arithmetic is written, so there is no second,
// textually-different call site for the compiler's FMA-contraction
// heuristic to treat differently.
// =========================================================================
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
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;

    float* out = output.data() + h * head_dim;
    std::fill(out, out + head_dim, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        auto v = cache.v_at(kv_h, t);
        float w = scores[t];
        for (int i = 0; i < head_dim; ++i) out[i] += w * v[i];
    }
}

// Every query head's own KV head, for a fixed group_size.
inline int kv_head_of(int query_head, int group_size) { return query_head / group_size; }

// =========================================================================
// SERIAL REFERENCE: every head, in order, via attend_one_head.
// =========================================================================
void gqa_attention_serial(std::span<const float> q_heads, KVCache& cache, std::span<float> output,
                           int seq_len, int n_heads_q, int group_size) {
    for (int h = 0; h < n_heads_q; ++h) attend_one_head(q_heads, cache, output, seq_len, h, kv_head_of(h, group_size));
}

// =========================================================================
// Ceiling-division row partition (Chapter 10.3's, unchanged).
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

// =========================================================================
// GENERALIZED PERSISTENT POOL (Section 11.2's ParallelPool), extended
// with an EXPLICIT-ranges entry point: instead of always auto-dividing
// [0, total) by ceiling division, the caller can hand each worker its own
// arbitrary range directly -- exactly what a group-aligned attention
// partition (Test 3) needs, since group-aligned ranges are not evenly
// sized and are not what partition_ceil(n_heads_q, n_threads) would ever
// produce on its own.
// =========================================================================
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

    void parallel_for(size_t total, std::function<void(size_t, size_t)> fn) {
        run_ranges(partition_ceil(total, n_threads_), std::move(fn));
    }
    // ranges.size() must equal n_threads_; a range may be empty (start==end).
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

// Head-parallel attention using WHATEVER ranges the caller assigned --
// each thread computes attend_one_head for every h in its own [start,end)
// range. Disjoint output slices, no combination step, no shared mutable
// state beyond read-only access to the same KVCache -- exactly Chapter
// 10.3's row-parallel shape, one head per "row."
void gqa_attention_head_parallel(ParallelPool& pool, std::span<const float> q_heads, KVCache& cache,
                                  std::span<float> output, int seq_len, int group_size,
                                  const std::vector<RowRange>& ranges) {
    pool.run_ranges(ranges, [&](size_t lo, size_t hi) {
        for (size_t h = lo; h < hi; ++h) attend_one_head(q_heads, cache, output, seq_len,
                                                          static_cast<int>(h), kv_head_of(static_cast<int>(h), group_size));
    });
}

// =========================================================================
// A GROUP-ALIGNED partition: assign whole KV GROUPS (not raw query-head
// indices) to threads, by ceiling-dividing the KV HEAD count across
// threads and expanding each thread's KV-head range back into the query
// heads that belong to it. Every KV head's group is now owned by exactly
// one thread -- there is no way for one KV head's rows to be read by two
// different threads, because a KV head only ever belongs to one range.
// =========================================================================
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

// Counts, for a given set of thread ranges, how many DISTINCT threads
// touch each KV head -- the redundancy this section cares about. Purely
// index arithmetic, no threads involved, exactly Chapter 10.3's Test 1
// style of checking a partition's structure before ever running it.
struct RedundancyReport {
    long long total_touches = 0;     // sum over threads of |distinct kv heads that thread touches|
    long long distinct_kv_heads = 0; // kv heads touched by at least one thread
    long long redundant_kv_heads = 0; // kv heads touched by MORE than one thread
};
RedundancyReport measure_redundancy(const std::vector<RowRange>& ranges, size_t n_heads_kv, size_t group_size) {
    std::vector<int> touch_count(n_heads_kv, 0);
    for (const auto& r : ranges) {
        std::set<size_t> kv_touched;
        for (size_t h = r.start; h < r.end; ++h) kv_touched.insert(h / group_size);
        for (size_t kv : kv_touched) touch_count[kv]++;
    }
    RedundancyReport rep;
    for (size_t kv = 0; kv < n_heads_kv; ++kv) {
        if (touch_count[kv] > 0) { rep.total_touches += touch_count[kv]; rep.distinct_kv_heads++; }
        if (touch_count[kv] > 1) rep.redundant_kv_heads++;
    }
    return rep;
}

// =========================================================================
// Model dimensions: N_HEADS_Q=8 sharing N_HEADS_KV=2 (GROUP=4) -- the
// same GQA shape Section 11.2 used, small enough to run instantly.
// =========================================================================
constexpr int N_HEADS_Q = 8, N_HEADS_KV = 2, GROUP = N_HEADS_Q / N_HEADS_KV, HEAD_DIM = 8;
constexpr int SEQ_LEN = 10;

KVCache make_filled_cache(unsigned seed) {
    KVCache cache(N_HEADS_KV, SEQ_LEN, HEAD_DIM);
    std::vector<float> k(HEAD_DIM), v(HEAD_DIM);
    for (int h = 0; h < N_HEADS_KV; ++h) {
        for (int t = 0; t < SEQ_LEN; ++t) {
            for (int i = 0; i < HEAD_DIM; ++i) {
                k[i] = 0.01f * static_cast<float>((h * 13 + t * 7 + i + static_cast<int>(seed)) % 11 - 5);
                v[i] = 0.02f * static_cast<float>((h * 5 + t * 3 + i + static_cast<int>(seed)) % 9 - 4);
            }
            cache.store(h, t, k, v);
        }
    }
    return cache;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 11.3: Head-Parallel Attention\n";
    std::cout << "========================================================\n\n";

    std::vector<float> q(N_HEADS_Q * HEAD_DIM);
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.03f * (static_cast<float>(i % 11) - 5.0f);

    KVCache cache_ref = make_filled_cache(5);
    std::vector<float> out_ref(N_HEADS_Q * HEAD_DIM);
    gqa_attention_serial(q, cache_ref, out_ref, SEQ_LEN, N_HEADS_Q, GROUP);

    // =====================================================================
    // TEST 1: head-parallel attention, naive per-index partition across
    // 4 threads, matches the serial reference bit-for-bit -- every head
    // is fully independent, so which thread computes which head changes
    // nothing about the result.
    // =====================================================================
    std::cout << "-- Test 1: head-parallel (naive partition) matches serial, bit-for-bit --\n";
    {
        constexpr size_t N_THREADS = 4;
        std::vector<RowRange> naive_ranges = partition_ceil(N_HEADS_Q, N_THREADS);
        KVCache cache_test = make_filled_cache(5);
        std::vector<float> out_test(N_HEADS_Q * HEAD_DIM);
        ParallelPool pool(N_THREADS);
        gqa_attention_head_parallel(pool, q, cache_test, out_test, SEQ_LEN, GROUP, naive_ranges);
        bool exact_match = (out_test == out_ref);
        std::cout << "  " << N_HEADS_Q << " heads across " << N_THREADS << " threads: "
                  << (exact_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(exact_match);
    }

    // =====================================================================
    // TEST 2 [COMMON TRAP]: that same naive, per-index partition, though
    // fully CORRECT, routinely splits a KV group across two threads --
    // this section's real subject. With GROUP=4 and 4 threads, each
    // thread's 2-head range covers only HALF of some KV head's group, so
    // that KV head's cache rows are read by two different threads instead
    // of one.
    // =====================================================================
    std::cout << "\n-- Test 2 [COMMON TRAP]: naive partition reads some KV heads redundantly --\n";
    {
        constexpr size_t N_THREADS = 4;
        std::vector<RowRange> naive_ranges = partition_ceil(N_HEADS_Q, N_THREADS);
        for (size_t t = 0; t < naive_ranges.size(); ++t) {
            std::cout << "  thread " << t << ": query heads [" << naive_ranges[t].start
                      << ", " << naive_ranges[t].end << ")\n";
        }
        RedundancyReport rep = measure_redundancy(naive_ranges, N_HEADS_KV, GROUP);
        std::cout << "  distinct KV heads touched: " << rep.distinct_kv_heads << " / " << N_HEADS_KV << "\n";
        std::cout << "  KV heads touched by MORE than one thread: " << rep.redundant_kv_heads << "\n";
        std::cout << "  total (thread, kv-head) touches: " << rep.total_touches
                  << " (ideal, no redundancy: " << N_HEADS_KV << ")\n";
        CHECK(rep.distinct_kv_heads == N_HEADS_KV);
        CHECK(rep.redundant_kv_heads > 0);
        CHECK(rep.total_touches > N_HEADS_KV);
    }

    // =====================================================================
    // TEST 3: the FIX -- a group-aligned partition assigns whole KV
    // groups to threads, so every KV head is touched by exactly one
    // thread. Verified two ways: zero redundancy (structural), and the
    // real threaded computation still matches the serial reference
    // bit-for-bit (the fix costs nothing in correctness).
    // =====================================================================
    std::cout << "\n-- Test 3: group-aligned partition eliminates the redundancy --\n";
    {
        constexpr size_t N_THREADS = 2;   // one thread per KV head -- the natural fit
        std::vector<RowRange> aligned_ranges = partition_group_aligned(N_HEADS_Q, N_HEADS_KV, GROUP, N_THREADS);
        for (size_t t = 0; t < aligned_ranges.size(); ++t) {
            std::cout << "  thread " << t << ": query heads [" << aligned_ranges[t].start
                      << ", " << aligned_ranges[t].end << ")\n";
        }
        RedundancyReport rep = measure_redundancy(aligned_ranges, N_HEADS_KV, GROUP);
        std::cout << "  KV heads touched by MORE than one thread: " << rep.redundant_kv_heads
                  << " (was " << "nonzero" << " with the naive partition above)\n";
        CHECK(rep.redundant_kv_heads == 0);
        CHECK(rep.total_touches == N_HEADS_KV);

        KVCache cache_test = make_filled_cache(5);
        std::vector<float> out_test(N_HEADS_Q * HEAD_DIM);
        ParallelPool pool(N_THREADS);
        gqa_attention_head_parallel(pool, q, cache_test, out_test, SEQ_LEN, GROUP, aligned_ranges);
        bool exact_match = (out_test == out_ref);
        std::cout << "  group-aligned partition vs. serial reference: "
                  << (exact_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(exact_match);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
