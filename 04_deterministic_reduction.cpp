// 04_deterministic_reduction.cpp
// Chapter 11, Part 4: Chapter 11.1 already showed that combining partial
// sums into per-thread-ID-ordered slots, then reducing those slots in
// fixed thread order, is reproducible run-to-run FOR A GIVEN thread
// count. This section asks the harder question the earlier one never
// tested: is that SAME technique reproducible when the THREAD COUNT
// itself changes? A real engine does not always run with the same
// thread count -- a smaller request, a busy machine, a differently
// configured deployment might use 2 threads where another run used 8 --
// and floating-point addition is not associative, so a different thread
// count means different chunk boundaries, which means a different
// summation ORDER, which can mean a different final bit pattern, even
// though every run is numerically correct.
//
// The fix generalizes the "write to your own slot, combine in fixed
// order" idea from per-THREAD slots to per-BLOCK slots, where the block
// boundaries are a FIXED CONSTANT chosen independently of how many
// threads happen to be running. A block's own internal sum is always
// computed the same way (left-to-right over that block's fixed element
// range) no matter which thread is assigned to compute it, and the final
// combination always walks the same fixed sequence of block indices --
// so however many threads did the work, and however the blocks were
// divided up among them, the exact same set of intermediate sums gets
// combined in the exact same order every time.
//
// Reused from Section 11.2's discovered hazard: this file compiles with
// -ffp-contract=off, and both reduction strategies below route every
// block-level or chunk-level accumulation through ONE shared function
// per strategy, called identically from every thread count and every
// thread -- so a genuine cross-thread-count difference in Test 2 reflects
// summation order, not an accident of where a multiply-add got fused.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread 04_deterministic_reduction.cpp -o 04_deterministic_reduction

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <thread>
#include <vector>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

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

// A minimal persistent pool, created fresh per thread-count trial in this
// file (Section 11.2 already proved reuse across many DIFFERENT task
// shapes with one still-alive pool; this section's variable is the
// THREAD COUNT itself, which a real std::barrier-based pool cannot
// change after construction, so a new pool per thread count is the
// correct thing to build here, not a shortcut around anything).
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

bool bit_equal(float a, float b) {
    uint32_t ai, bi;
    std::memcpy(&ai, &a, 4);
    std::memcpy(&bi, &b, 4);
    return ai == bi;
}

// =========================================================================
// STRATEGY A: per-THREAD-ID slots, combined in fixed thread-ID order.
// Chapter 11.1's own fixed-order combine, generalized to a dot product.
// Reproducible run-to-run for a GIVEN n_threads (Test 1 confirms this
// again here); NOT guaranteed identical when n_threads itself changes,
// because chunk boundaries -- and therefore summation order -- change
// with it (Test 2 shows this concretely).
// =========================================================================
float dot_thread_indexed(ParallelPool& pool, const std::vector<float>& a, const std::vector<float>& b, size_t n_threads) {
    size_t n = a.size();
    std::vector<float> partials(n_threads, 0.0f);
    std::vector<RowRange> ranges = partition_ceil(n, n_threads);
    pool.run_ranges(ranges, [&](size_t lo, size_t hi) {
        size_t t = 0;
        for (size_t k = 0; k < ranges.size(); ++k) if (ranges[k].start == lo && ranges[k].end == hi) { t = k; break; }
        float sum = 0.0f;
        for (size_t i = lo; i < hi; ++i) sum += a[i] * b[i];
        partials[t] = sum;
    });
    float result = 0.0f;
    for (size_t t = 0; t < n_threads; ++t) result += partials[t];
    return result;
}

// =========================================================================
// STRATEGY B: fixed-size BLOCKS, independent of n_threads. Block b always
// covers elements [b*BLOCK_SIZE, min((b+1)*BLOCK_SIZE, n)) -- a fact
// determined entirely by BLOCK_SIZE and the array size, never by how many
// threads are running. Threads are assigned RANGES OF BLOCKS (ceiling-
// divided, same as always), each thread fully reduces its own blocks
// (fixed left-to-right order within each block) and writes each block's
// sum into its own globally-indexed slot -- disjoint writes, no matter
// how blocks are split across threads. The final combination walks every
// block index in fixed order, 0 .. NUM_BLOCKS-1, regardless of n_threads.
// =========================================================================
constexpr size_t BLOCK_SIZE = 16;

float dot_fixed_blocks(ParallelPool& pool, const std::vector<float>& a, const std::vector<float>& b, size_t n_threads) {
    size_t n = a.size();
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<float> block_sums(num_blocks, 0.0f);
    std::vector<RowRange> block_ranges = partition_ceil(num_blocks, n_threads);
    pool.run_ranges(block_ranges, [&](size_t block_lo, size_t block_hi) {
        for (size_t blk = block_lo; blk < block_hi; ++blk) {
            size_t elem_start = blk * BLOCK_SIZE;
            size_t elem_end = std::min(elem_start + BLOCK_SIZE, n);
            float sum = 0.0f;
            for (size_t i = elem_start; i < elem_end; ++i) sum += a[i] * b[i];
            block_sums[blk] = sum;
        }
    });
    float result = 0.0f;
    for (size_t blk = 0; blk < num_blocks; ++blk) result += block_sums[blk];
    return result;
}

// Plain serial dot product, no blocking at all -- a sanity reference for
// "is this number in the right ballpark," never used for a bit-exact
// comparison, since neither strategy above shares its summation order.
float dot_serial(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 11.4: Deterministic Floating-Point Reduction, Independent of Thread Count\n";
    std::cout << "========================================================\n\n";

    // N is deliberately awkward (not a clean multiple of any tested
    // thread count or of BLOCK_SIZE) so every partition below has to
    // handle a genuinely uneven last chunk or last block.
    constexpr size_t N = 997;
    std::vector<float> a(N), b(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = 0.1f * static_cast<float>((i * 7 + 3) % 23) - 1.0f;
        b[i] = 0.05f * static_cast<float>((i * 11 + 5) % 19) - 0.4f;
    }
    const std::vector<size_t> THREAD_COUNTS = {1, 2, 3, 4, 5, 8};
    float serial_ref = dot_serial(a, b);

    // =====================================================================
    // TEST 1: dot_fixed_blocks matches a plain serial dot product within
    // a loose tolerance (both are valid summation orders of the same
    // terms; neither is asserted bit-exact against the other), AND is
    // reproducible run-to-run for a FIXED thread count.
    // =====================================================================
    std::cout << "-- Test 1: fixed-block reduction is numerically sound and self-reproducible --\n";
    {
        ParallelPool pool(4);
        float r1 = dot_fixed_blocks(pool, a, b, 4);
        float r2 = dot_fixed_blocks(pool, a, b, 4);
        std::cout << "  plain serial dot product:        " << serial_ref << "\n";
        std::cout << "  fixed-block reduction (4 threads): " << r1 << "\n";
        CHECK(std::fabs(r1 - serial_ref) < 1e-3f);
        CHECK(bit_equal(r1, r2));
    }

    // =====================================================================
    // TEST 2: the main result. dot_fixed_blocks produces a BIT-IDENTICAL
    // result across every tested thread count -- 1, 2, 3, 4, 5, and 8 --
    // because block boundaries never depend on n_threads, only on
    // BLOCK_SIZE and N.
    // =====================================================================
    std::cout << "\n-- Test 2: fixed-block reduction is bit-identical ACROSS different thread counts --\n";
    {
        std::set<uint32_t> seen_bits;
        float first = 0.0f;
        for (size_t nt : THREAD_COUNTS) {
            ParallelPool pool(nt);
            float r = dot_fixed_blocks(pool, a, b, nt);
            if (seen_bits.empty()) first = r;
            uint32_t bits; std::memcpy(&bits, &r, 4);
            seen_bits.insert(bits);
            std::cout << "  n_threads=" << nt << ": " << std::fixed << std::setprecision(8) << r
                      << (bit_equal(r, first) ? "  (matches n_threads=1)" : "  (DIFFERS)") << "\n";
        }
        std::cout << "  unique bit patterns across " << THREAD_COUNTS.size() << " thread counts: " << seen_bits.size() << "\n";
        CHECK(seen_bits.size() == 1);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: the per-THREAD-indexed reduction (Chapter
    // 11.1's own fixed-order combine, generalized here) is reproducible
    // for a FIXED thread count -- exactly what 11.1 verified -- but is
    // NOT guaranteed bit-identical when the thread count changes, because
    // a different thread count moves the chunk boundaries, which changes
    // which terms get added together first.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: thread-indexed reduction is NOT stable across thread counts --\n";
    {
        // Self-reproducibility for a FIXED thread count, first -- this
        // part is NOT the trap; it is exactly what 11.1 already showed.
        ParallelPool pool4(4);
        float r1 = dot_thread_indexed(pool4, a, b, 4);
        float r2 = dot_thread_indexed(pool4, a, b, 4);
        std::cout << "  same thread count (4) twice: " << (bit_equal(r1, r2) ? "bit-identical" : "DIFFERS") << " (expected: bit-identical)\n";
        CHECK(bit_equal(r1, r2));

        // Now vary the thread count and look for ANY disagreement.
        std::set<uint32_t> seen_bits;
        for (size_t nt : THREAD_COUNTS) {
            ParallelPool pool(nt);
            float r = dot_thread_indexed(pool, a, b, nt);
            uint32_t bits; std::memcpy(&bits, &r, 4);
            seen_bits.insert(bits);
            std::cout << "  n_threads=" << nt << ": " << std::fixed << std::setprecision(8) << r << "\n";
        }
        std::cout << "  unique bit patterns across " << THREAD_COUNTS.size() << " thread counts: " << seen_bits.size()
                  << " (fixed-block reduction above got exactly 1)\n";
        std::cout << "  A test asserting \"thread-indexed reduction is deterministic\" and stopping\n";
        std::cout << "  after checking only ONE thread count -- exactly what Chapter 11.1 checked --\n";
        std::cout << "  would have missed this. \"Reproducible\" and \"reproducible regardless of\n";
        std::cout << "  configuration\" are different claims, and only the fixed-block scheme makes the\n";
        std::cout << "  second one.\n";
        CHECK(seen_bits.size() > 1);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
