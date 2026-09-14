// 02_persistent_pool_full_layer.cpp
// Chapter 11, Part 2: Section 10.3's TensorThreadPool ran exactly one task
// shape -- a GEMV, forever, every round. A real transformer layer needs
// that same persistent pool to run several DIFFERENT task shapes back to
// back within a single token: the Q/K/V projections (each n_out=DIM), the
// output projection (n_out=DIM), the FFN gate/up projections (each
// n_out=D_FF), and the FFN down projection (n_out=DIM) -- with RMSNorm,
// attention, and SwiGLU's elementwise gate running sequentially in
// between. This section generalizes the pool from "always run a GEMV"
// to "run any row-partitioned task the caller hands it," so ONE pool,
// created once, can carry an entire layer's parallel phases.
//
// The chapter's real kernels are reused verbatim rather than re-derived:
// rms_norm (Chapter 3.1), the naive matmul and swiglu elementwise gate
// (Chapter 3.2), and gqa_attention with its mdspan-backed KVCache
// (Chapter 3.4). Attention itself stays SEQUENTIAL in this section --
// Section 11.3 comes back to parallelize it across heads specifically --
// so what this section verifies is that the pool's four MATMUL phases,
// run through the SAME persistent pool across an entire layer (and then
// across several consecutive layers), match a serial reference exactly.
//
// A second, independent point this section makes concrete: independent
// matmuls can only be BATCHED into a single parallel phase (one pair of
// barrier arrivals instead of several) when they share the same output
// row count. Q, K, and V all project to DIM rows, so one phase computes
// all three; the FFN gate and up projections both produce D_FF rows, so
// they batch together too -- but D_FF != DIM here, so gate/up cannot be
// batched into the SAME phase as the output or down projections without
// silently leaving rows uncomputed, which Test 3 demonstrates directly.
//
// A THIRD hazard surfaced only while cross-checking this file on aarch64
// during authoring, and is worth stating plainly: the very first version
// of this file matched serial-vs-parallel bit-for-bit on x86_64 but NOT
// on aarch64, even though both platforms ran the identical source and
// both were internally deterministic (5+ repeated runs matched each
// other on each platform separately). The cause was GCC's default
// -ffp-contract=fast: it permits fusing a*b+c into one rounding step
// (a hardware FMA) wherever the compiler judges it profitable, and that
// judgment is made per call site, AFTER inlining -- so the exact same
// accumulation, textually identical, can get fused in one code path and
// not another, on an architecture (aarch64) where scalar FMA is always
// available, while staying un-fused everywhere on this x86_64 build
// (compiled without -mfma, where no scalar hardware FMA exists to fuse
// into at all). This is Chapter 9.2's FMA-contraction lesson resurfacing
// in a new, sharper form: there, two DIFFERENT kernels (scalar vs AVX2)
// were expected to differ and were compared within a tolerance; here,
// ONE kernel's serial and parallel code paths were claimed to be
// bit-identical, and silently were not, purely because of where the
// compiler chose to fuse. Passing -ffp-contract=off restores the literal,
// separate multiply-then-add the source asks for, on every architecture,
// which is what this file's compile line below now does -- and which
// this section's own Test 1 and Test 2 depend on to make "bit-for-bit"
// a claim that is actually true everywhere, not just true on whichever
// machine happened to be used to author it.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_persistent_pool_full_layer.cpp -o 02_persistent_pool_full_layer

#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mdspan/mdspan.hpp>
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
// REUSED VERBATIM: Chapter 3.1's RMSNorm, Chapter 3.2's naive matmul and
// SwiGLU elementwise gate, Chapter 3.4's GQA attention and its
// mdspan-backed KVCache. Only the names are qualified where this file's
// own generalized pool needs a distinct RowRange/partition helper,
// already established in Chapter 10.3 and reused unchanged here.
// =========================================================================
void rms_norm(std::span<float> out, std::span<const float> x,
              std::span<const float> weights, float epsilon = 1e-6f) {
    const size_t d = x.size();
    float sum_sq = 0.0f;
    for (size_t i = 0; i < d; ++i) sum_sq += x[i] * x[i];
    float rms_inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(d) + epsilon);
    for (size_t i = 0; i < d; ++i) out[i] = (x[i] * rms_inv) * weights[i];
}

void matmul(std::span<float> out, std::span<const float> x,
            std::span<const float> W, size_t in_dim, size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < in_dim; ++i) sum += x[i] * W[j * in_dim + i];
        out[j] = sum;
    }
}
// Same matmul, but computing only rows [row_start, row_end) -- the exact
// per-thread body a partitioned phase runs; matmul() above is just this
// called once with the full row range, so the two can never disagree.
void matmul_rows(std::span<float> out, std::span<const float> x,
                  std::span<const float> W, size_t in_dim,
                  size_t row_start, size_t row_end) {
    for (size_t j = row_start; j < row_end; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < in_dim; ++i) sum += x[i] * W[j * in_dim + i];
        out[j] = sum;
    }
}

inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
void swiglu(std::span<const float> gate_proj, std::span<const float> up_proj,
            std::span<float> out) {
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = silu(gate_proj[i]) * up_proj[i];
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
        auto kslice = k_at(h, t);
        auto vslice = v_at(h, t);
        for (int i = 0; i < head_dim; ++i) { kslice[i] = k[i]; vslice[i] = v[i]; }
    }
};

void gqa_attention(std::span<const float> q_heads, KVCache& cache,
                    std::span<float> output, int seq_len, int n_heads_q, int group_size) {
    const int head_dim = cache.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len);
    for (int h = 0; h < n_heads_q; ++h) {
        int kv_h = h / group_size;
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
}

// =========================================================================
// Chapter 10.3's ceiling-division row partition, unchanged.
// =========================================================================
struct RowRange { size_t start, end; };
std::vector<RowRange> partition_rows(size_t n_out, size_t n_threads) {
    std::vector<RowRange> ranges;
    size_t chunk = (n_out + n_threads - 1) / n_threads;
    for (size_t t = 0; t < n_threads; ++t) {
        size_t start = std::min(t * chunk, n_out);
        size_t end = std::min(start + chunk, n_out);
        ranges.push_back({start, end});
    }
    return ranges;
}

// =========================================================================
// GENERALIZED PERSISTENT POOL: where Chapter 10.3's TensorThreadPool only
// ever ran a GEMV, this pool's task descriptor is an arbitrary
// std::function<void(size_t,size_t)> -- "compute this row range, however
// the caller defines that" -- so the SAME pool instance, created once,
// can run a Q/K/V projection phase this round and an FFN down-projection
// phase the next round, with no new threads and no new synchronization
// primitives created in between.
// =========================================================================
class ParallelPool {
public:
    explicit ParallelPool(size_t n_threads)
        : n_threads_(n_threads),
          start_barrier_(static_cast<std::ptrdiff_t>(n_threads + 1)),
          done_barrier_(static_cast<std::ptrdiff_t>(n_threads + 1)) {
        for (size_t t = 0; t < n_threads_; ++t) {
            workers_.emplace_back([this, t] { worker_loop(t); });
        }
    }

    ~ParallelPool() {
        stop_flag_.store(true, std::memory_order_relaxed);
        start_barrier_.arrive_and_wait();
        for (auto& w : workers_) w.join();
    }

    // Runs fn(row_start, row_end) once per worker, partitioning [0, total)
    // by ceiling division across n_threads_ -- one round, any task shape.
    void parallel_for(size_t total, std::function<void(size_t, size_t)> fn) {
        task_fn_ = std::move(fn);
        task_ranges_ = partition_rows(total, n_threads_);
        start_barrier_.arrive_and_wait();
        done_barrier_.arrive_and_wait();
    }

private:
    void worker_loop(size_t my_index) {
        for (;;) {
            start_barrier_.arrive_and_wait();
            if (stop_flag_.load(std::memory_order_relaxed)) return;
            const RowRange& r = task_ranges_[my_index];
            task_fn_(r.start, r.end);
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
// Model dimensions: small enough to run in milliseconds, shaped enough to
// exercise every phase described above. D_FF is deliberately NOT equal to
// DIM (176 vs 64) -- the fact Test 3 depends on.
// =========================================================================
constexpr size_t DIM = 64, N_HEADS_Q = 8, N_HEADS_KV = 2, GROUP = N_HEADS_Q / N_HEADS_KV;
constexpr size_t HEAD_DIM = DIM / N_HEADS_Q;
constexpr size_t D_FF = 176;
constexpr size_t N_THREADS = 4;
constexpr int CACHE_LEN = 5;   // already-cached positions before this new token

struct LayerWeights {
    std::vector<float> attn_norm_w, ffn_norm_w;
    std::vector<float> Wq, Wk, Wv, Wo;
    std::vector<float> Wgate, Wup, Wdown;
    LayerWeights(unsigned seed) {
        attn_norm_w.assign(DIM, 1.0f);
        ffn_norm_w.assign(DIM, 1.0f);
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
// SERIAL REFERENCE: one full layer's worth of a decode step -- RMSNorm,
// QKV, attention (against a KV cache already holding CACHE_LEN positions,
// with this call's own K/V stored at position CACHE_LEN), output
// projection, residual, RMSNorm, SwiGLU FFN, residual. Every matmul here
// is the plain, single-threaded matmul() from Chapter 3.2.
// =========================================================================
void layer_serial(LayerWeights& w, KVCache& cache, std::vector<float>& x) {
    std::vector<float> xn(DIM);
    rms_norm(xn, x, w.attn_norm_w);

    std::vector<float> q(DIM), k(N_HEADS_KV * HEAD_DIM), v(N_HEADS_KV * HEAD_DIM);
    matmul(q, xn, w.Wq, DIM, DIM);
    matmul(k, xn, w.Wk, DIM, N_HEADS_KV * HEAD_DIM);
    matmul(v, xn, w.Wv, DIM, N_HEADS_KV * HEAD_DIM);

    // Store this step's new K/V at position CACHE_LEN, then attend over
    // all CACHE_LEN+1 positions (the cached ones plus this new one).
    for (size_t h = 0; h < N_HEADS_KV; ++h) {
        cache.store(static_cast<int>(h), CACHE_LEN,
                    std::span<const float>(k.data() + h * HEAD_DIM, HEAD_DIM),
                    std::span<const float>(v.data() + h * HEAD_DIM, HEAD_DIM));
    }
    std::vector<float> attn_out(DIM);
    gqa_attention(q, cache, attn_out, CACHE_LEN + 1, static_cast<int>(N_HEADS_Q), static_cast<int>(GROUP));

    std::vector<float> proj(DIM);
    matmul(proj, attn_out, w.Wo, DIM, DIM);
    for (size_t i = 0; i < DIM; ++i) x[i] += proj[i];

    rms_norm(xn, x, w.ffn_norm_w);
    std::vector<float> gate(D_FF), up(D_FF), hidden(D_FF), ffn_out(DIM);
    matmul(gate, xn, w.Wgate, DIM, D_FF);
    matmul(up, xn, w.Wup, DIM, D_FF);
    swiglu(gate, up, hidden);
    matmul(ffn_out, hidden, w.Wdown, D_FF, DIM);
    for (size_t i = 0; i < DIM; ++i) x[i] += ffn_out[i];
}

// =========================================================================
// PARALLEL LAYER: identical computation, run through ONE persistent
// ParallelPool. Q, K, and V batch into a single phase (all three project
// to a shared row space handled by one parallel_for call, since each row
// index j selects the SAME row j of Wq, Wk-or-nothing... -- see note
// below); the output projection is its own phase; gate and up batch into
// a second phase (both D_FF rows); the down projection is its own phase.
// Attention itself stays sequential, exactly as the serial reference
// computes it -- Section 11.3 parallelizes attention specifically.
//
// Note on batching Q/K/V: Wk and Wv only have N_HEADS_KV*HEAD_DIM rows,
// fewer than Wq's DIM rows (GQA's whole point -- fewer KV heads than
// query heads). A single parallel_for(DIM, fn) is still correct here
// because fn only writes into k/v when the row index is in range; the
// phase's SHARED row count is DIM (Wq's), and k/v rows beyond their own
// smaller extent are simply skipped by that same lambda, not computed
// out of bounds.
// =========================================================================
void layer_parallel(ParallelPool& pool, LayerWeights& w, KVCache& cache, std::vector<float>& x) {
    std::vector<float> xn(DIM);
    rms_norm(xn, x, w.attn_norm_w);

    std::vector<float> q(DIM), k(N_HEADS_KV * HEAD_DIM), v(N_HEADS_KV * HEAD_DIM);
    constexpr size_t KV_ROWS = N_HEADS_KV * HEAD_DIM;
    pool.parallel_for(DIM, [&](size_t lo, size_t hi) {
        for (size_t j = lo; j < hi; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < DIM; ++i) sum += xn[i] * w.Wq[j * DIM + i];
            q[j] = sum;
        }
        size_t kv_lo = std::min(lo, KV_ROWS), kv_hi = std::min(hi, KV_ROWS);
        for (size_t j = kv_lo; j < kv_hi; ++j) {
            float sk = 0.0f, sv = 0.0f;
            for (size_t i = 0; i < DIM; ++i) { sk += xn[i] * w.Wk[j * DIM + i]; sv += xn[i] * w.Wv[j * DIM + i]; }
            k[j] = sk; v[j] = sv;
        }
    });

    for (size_t h = 0; h < N_HEADS_KV; ++h) {
        cache.store(static_cast<int>(h), CACHE_LEN,
                    std::span<const float>(k.data() + h * HEAD_DIM, HEAD_DIM),
                    std::span<const float>(v.data() + h * HEAD_DIM, HEAD_DIM));
    }
    std::vector<float> attn_out(DIM);
    gqa_attention(q, cache, attn_out, CACHE_LEN + 1, static_cast<int>(N_HEADS_Q), static_cast<int>(GROUP));

    std::vector<float> proj(DIM);
    pool.parallel_for(DIM, [&](size_t lo, size_t hi) {
        matmul_rows(proj, attn_out, w.Wo, DIM, lo, hi);
    });
    for (size_t i = 0; i < DIM; ++i) x[i] += proj[i];

    rms_norm(xn, x, w.ffn_norm_w);
    std::vector<float> gate(D_FF), up(D_FF), hidden(D_FF), ffn_out(DIM);
    pool.parallel_for(D_FF, [&](size_t lo, size_t hi) {
        matmul_rows(gate, xn, w.Wgate, DIM, lo, hi);
        matmul_rows(up, xn, w.Wup, DIM, lo, hi);
    });
    swiglu(gate, up, hidden);
    pool.parallel_for(DIM, [&](size_t lo, size_t hi) {
        matmul_rows(ffn_out, hidden, w.Wdown, D_FF, lo, hi);
    });
    for (size_t i = 0; i < DIM; ++i) x[i] += ffn_out[i];
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 11.2: A Persistent Pool Across a Full Layer's Phases\n";
    std::cout << "========================================================\n\n";

    // =====================================================================
    // TEST 1: one full layer, serial reference vs. the same computation
    // run through a persistent ParallelPool -- every matmul phase is
    // row-parallel with disjoint writes, so (as Chapter 10.3 established)
    // the result is bit-for-bit identical, not merely within tolerance.
    // =====================================================================
    std::cout << "-- Test 1: one full layer, serial vs. pool-parallel, bit-for-bit --\n";
    {
        LayerWeights w(7);
        KVCache cache_serial = make_filled_cache(3);
        KVCache cache_pool = make_filled_cache(3);
        std::vector<float> x_serial(DIM), x_pool(DIM);
        for (size_t i = 0; i < DIM; ++i) {
            float val = 0.03f * (static_cast<float>(i % 13) - 6.0f);
            x_serial[i] = val; x_pool[i] = val;
        }

        layer_serial(w, cache_serial, x_serial);
        {
            ParallelPool pool(N_THREADS);
            layer_parallel(pool, w, cache_pool, x_pool);
        }

        bool exact_match = (x_serial == x_pool);
        std::cout << "  DIM=" << DIM << " D_FF=" << D_FF << " N_HEADS_Q=" << N_HEADS_Q
                  << " N_HEADS_KV=" << N_HEADS_KV << ": "
                  << (exact_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(exact_match);
    }

    // =====================================================================
    // TEST 2: the SAME persistent pool, still alive, runs SIX consecutive
    // layers -- a different task shape every phase, every layer -- and
    // matches a serial reference run the same six times. This is the
    // point Chapter 10.3 could not make with a single fixed GEMV shape:
    // one pool instance genuinely carries an entire multi-phase, multi-
    // layer forward pass, never recreated between layers or phases.
    // =====================================================================
    std::cout << "\n-- Test 2: SAME pool reused across 6 consecutive layers --\n";
    {
        constexpr int N_LAYERS = 6;
        std::vector<LayerWeights> layers;
        for (int l = 0; l < N_LAYERS; ++l) layers.emplace_back(100 + l * 17);

        std::vector<float> x_serial(DIM), x_pool(DIM);
        for (size_t i = 0; i < DIM; ++i) {
            float val = 0.02f * (static_cast<float>(i % 9) - 4.0f);
            x_serial[i] = val; x_pool[i] = val;
        }
        std::vector<KVCache> caches_serial, caches_pool;
        for (int l = 0; l < N_LAYERS; ++l) {
            caches_serial.push_back(make_filled_cache(static_cast<unsigned>(l + 1)));
            caches_pool.push_back(make_filled_cache(static_cast<unsigned>(l + 1)));
        }

        for (int l = 0; l < N_LAYERS; ++l) layer_serial(layers[l], caches_serial[l], x_serial);
        {
            ParallelPool pool(N_THREADS);
            for (int l = 0; l < N_LAYERS; ++l) layer_parallel(pool, layers[l], caches_pool[l], x_pool);
        }

        bool exact_match = (x_serial == x_pool);
        std::cout << "  " << N_LAYERS << " layers through one still-alive pool: "
                  << (exact_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(exact_match);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: batching only works when the batched matmuls
    // share the same output row count. Here, a phase is deliberately
    // (mis)written to batch the FFN down-projection (DIM=64 rows) using
    // the OUTPUT projection's row count instead of its own -- exactly the
    // mistake of copy-pasting one phase's total into a different phase.
    // Rows [DIM, D_FF) of a D_FF-shaped task never get assigned when the
    // phase is told there are only DIM of them.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: batching with the wrong shared row count drops rows --\n";
    {
        LayerWeights w(9);
        std::vector<float> xn(DIM, 0.1f);
        constexpr float SENTINEL = -999.0f;
        std::vector<float> gate(D_FF, SENTINEL), up(D_FF, SENTINEL);

        ParallelPool pool(N_THREADS);
        // BUG: this phase computes the gate/up projections (D_FF=176 rows
        // each) but was told the row count is DIM=64 -- as if whoever
        // wrote this phase copy-pasted the output-projection's total.
        pool.parallel_for(DIM, [&](size_t lo, size_t hi) {
            matmul_rows(gate, xn, w.Wgate, DIM, lo, hi);
            matmul_rows(up, xn, w.Wup, DIM, lo, hi);
        });

        size_t gate_still_sentinel = 0, up_still_sentinel = 0;
        for (size_t j = 0; j < D_FF; ++j) {
            if (gate[j] == SENTINEL) ++gate_still_sentinel;
            if (up[j] == SENTINEL) ++up_still_sentinel;
        }
        size_t expected_dropped = D_FF - DIM;
        std::cout << "  phase told total=" << DIM << " but gate/up actually have " << D_FF << " rows\n";
        std::cout << "  gate rows never computed: " << gate_still_sentinel
                   << " (expected " << expected_dropped << ")\n";
        std::cout << "  up rows never computed:   " << up_still_sentinel
                   << " (expected " << expected_dropped << ")\n";
        CHECK(gate_still_sentinel == expected_dropped);
        CHECK(up_still_sentinel == expected_dropped);
        CHECK(expected_dropped > 0);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
