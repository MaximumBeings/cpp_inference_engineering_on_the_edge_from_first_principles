// 03_barrier_thread_pool_gemv.cpp
// Chapter 10, Part 3: this book's first real, partitioned parallel kernel.
// A GEMV -- the exact operation Chapter 8.3 classified as memory-bound at
// decode time -- is split by OUTPUT ROW across a fixed pool of persistent
// worker threads, each computing a disjoint, statically assigned range of
// rows into its own slice of the output vector. Because no two threads
// ever write the same output element, and because each row's own
// dot-product loop runs in exactly the same left-to-right order the
// serial reference uses, the partitioned result is not merely "close to"
// the serial reference within a floating-point reordering tolerance the
// way Chapter 9.2's AVX2 kernel was -- it is bit-for-bit identical to it,
// verified exactly, on every run, regardless of how the operating system
// happens to schedule the worker threads.
//
// The thread pool itself is built the way a real inference server's would
// be: worker threads are created ONCE and stay alive across many rounds
// of work, coordinated by std::barrier rather than by creating and
// joining new threads for every single GEMV -- exactly the persistent,
// reusable structure a real decode loop needs, since a real engine cannot
// afford Chapter 10.2's create-four-threads-and-join cost on every token.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_barrier_thread_pool_gemv.cpp -o 03_barrier_thread_pool_gemv

#include <atomic>
#include <barrier>
#include <cstdint>
#include <iostream>
#include <mdspan/mdspan.hpp>
#include <thread>
#include <vector>

// -- Test harness --------------------------------------------------------
static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

using Matrix = std::mdspan<const float, std::dextents<size_t, 2>>;

// A [start, end) row range, half-open, exactly the convention every
// partitioning scheme in this section is checked against.
struct RowRange {
    size_t start, end;
    size_t count() const { return end - start; }
};

// =========================================================================
// The serial reference: every row's dot product computed left-to-right,
// in the exact same inner-loop order the partitioned pool below uses --
// so the two are not merely numerically close, they are bit-identical.
// =========================================================================
void gemv_serial_reference(Matrix W, const float* x, float* y, size_t n_out, size_t n_in) {
    for (size_t row = 0; row < n_out; ++row) {
        float acc = 0.0f;
        for (size_t col = 0; col < n_in; ++col) acc += W[row, col] * x[col];
        y[row] = acc;
    }
}

// =========================================================================
// CORRECT partitioning: ceil-divided chunk size, with the LAST chunk
// explicitly capped at n_out -- so the union of all ranges is exactly
// [0, n_out) with no gaps and no overlaps, regardless of whether n_out
// divides evenly by the thread count.
// =========================================================================
std::vector<RowRange> partition_rows_correct(size_t n_out, size_t n_threads) {
    std::vector<RowRange> ranges;
    size_t chunk = (n_out + n_threads - 1) / n_threads;  // ceiling division
    for (size_t t = 0; t < n_threads; ++t) {
        size_t start = std::min(t * chunk, n_out);
        size_t end = std::min(start + chunk, n_out);
        ranges.push_back({start, end});
    }
    return ranges;
}

// =========================================================================
// [COMMON TRAP] the naive partitioning that a first attempt reaches for:
// plain integer division for the chunk size, with every chunk (including
// the last) exactly that size. When n_out does not divide evenly by
// n_threads, integer division truncates, and the rows past
// n_threads * chunk are never assigned to any thread at all -- not
// computed incorrectly, simply never computed.
// =========================================================================
std::vector<RowRange> partition_rows_naive_buggy(size_t n_out, size_t n_threads) {
    std::vector<RowRange> ranges;
    size_t chunk = n_out / n_threads;  // truncating division -- the bug
    for (size_t t = 0; t < n_threads; ++t) {
        ranges.push_back({t * chunk, (t + 1) * chunk});
    }
    return ranges;
}

// =========================================================================
// A persistent thread pool, coordinated by two std::barrier objects
// (start and done) instead of creating and joining threads per call.
// Workers are created once in the constructor and loop until stop_flag
// is set; run_gemv() publishes a task's parameters and uses the barriers
// to hand off exactly one round of work per call.
// =========================================================================
class TensorThreadPool {
public:
    explicit TensorThreadPool(size_t n_threads)
        : n_threads_(n_threads),
          start_barrier_(static_cast<std::ptrdiff_t>(n_threads + 1)),
          done_barrier_(static_cast<std::ptrdiff_t>(n_threads + 1)) {
        for (size_t t = 0; t < n_threads_; ++t) {
            workers_.emplace_back([this, t] { worker_loop(t); });
        }
    }

    ~TensorThreadPool() {
        stop_flag_.store(true, std::memory_order_relaxed);
        start_barrier_.arrive_and_wait();  // release workers so they observe stop_flag_
        for (auto& w : workers_) w.join();
    }

    // Runs one partitioned GEMV round using the given row ranges (one
    // range per worker thread, ranges_[t] assigned to worker t).
    void run_gemv(Matrix W, const float* x, float* y, size_t n_in,
                  const std::vector<RowRange>& ranges) {
        task_W_ = &W;
        task_x_ = x;
        task_y_ = y;
        task_n_in_ = n_in;
        task_ranges_ = &ranges;
        start_barrier_.arrive_and_wait();  // release workers into this round
        done_barrier_.arrive_and_wait();   // wait for all workers to finish it
    }

private:
    void worker_loop(size_t my_index) {
        for (;;) {
            start_barrier_.arrive_and_wait();
            if (stop_flag_.load(std::memory_order_relaxed)) return;

            const RowRange& r = (*task_ranges_)[my_index];
            for (size_t row = r.start; row < r.end; ++row) {
                float acc = 0.0f;
                for (size_t col = 0; col < task_n_in_; ++col) {
                    acc += (*task_W_)[row, col] * task_x_[col];
                }
                task_y_[row] = acc;
            }
            done_barrier_.arrive_and_wait();
        }
    }

    size_t n_threads_;
    std::vector<std::thread> workers_;
    std::barrier<> start_barrier_;
    std::barrier<> done_barrier_;
    std::atomic<bool> stop_flag_{false};

    const Matrix* task_W_ = nullptr;
    const float* task_x_ = nullptr;
    float* task_y_ = nullptr;
    size_t task_n_in_ = 0;
    const std::vector<RowRange>* task_ranges_ = nullptr;
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 10.3: A Barrier-Synchronized Thread Pool for a Partitioned GEMV\n";
    std::cout << "========================================================\n\n";

    constexpr size_t N_OUT = 70;   // deliberately NOT a multiple of the thread count
    constexpr size_t N_IN = 256;
    constexpr size_t N_THREADS = 4;

    std::vector<float> w_data(N_OUT * N_IN), x_data(N_IN), y_serial(N_OUT), y_pool(N_OUT);
    for (size_t i = 0; i < w_data.size(); ++i) {
        w_data[i] = (static_cast<float>(i % 17) - 8.0f) * 0.05f;
    }
    for (size_t i = 0; i < N_IN; ++i) {
        x_data[i] = (static_cast<float>(i % 11) - 5.0f) * 0.1f;
    }
    Matrix W(w_data.data(), N_OUT, N_IN);

    gemv_serial_reference(W, x_data.data(), y_serial.data(), N_OUT, N_IN);

    // =====================================================================
    // TEST 1: the correct (ceiling-division) partition covers every row
    // exactly once -- no gaps, no overlaps -- regardless of N_OUT not
    // dividing evenly by N_THREADS.
    // =====================================================================
    std::cout << "-- Test 1: correct partition covers [0, " << N_OUT << ") exactly once --\n";
    std::vector<RowRange> correct_ranges = partition_rows_correct(N_OUT, N_THREADS);
    {
        size_t total_covered = 0;
        bool covers_exactly = true;
        std::vector<bool> covered(N_OUT, false);
        for (const auto& r : correct_ranges) {
            std::cout << "  thread range: [" << r.start << ", " << r.end << ") -- " << r.count() << " rows\n";
            total_covered += r.count();
            for (size_t row = r.start; row < r.end; ++row) {
                if (covered[row]) covers_exactly = false;  // overlap
                covered[row] = true;
            }
        }
        bool all_covered = true;
        for (bool c : covered) if (!c) all_covered = false;
        std::cout << "  total rows covered: " << total_covered << " (expected " << N_OUT << ")\n";
        CHECK(total_covered == N_OUT);
        CHECK(covers_exactly);
        CHECK(all_covered);
    }

    // =====================================================================
    // TEST 2: a real, persistent, barrier-synchronized thread pool
    // computes this GEMV using the correct partition, matching the
    // serial reference BIT-FOR-BIT -- not within a tolerance, since each
    // row's own inner loop runs in the identical order on either path.
    // =====================================================================
    std::cout << "\n-- Test 2: pool-computed GEMV vs. serial reference, using the correct partition --\n";
    {
        std::fill(y_pool.begin(), y_pool.end(), -999.0f);  // sentinel: must be overwritten everywhere
        TensorThreadPool pool(N_THREADS);
        pool.run_gemv(W, x_data.data(), y_pool.data(), N_IN, correct_ranges);

        bool exact_match = true;
        for (size_t row = 0; row < N_OUT; ++row) {
            if (y_pool[row] != y_serial[row]) exact_match = false;
        }
        std::cout << "  " << N_OUT << " output rows across " << N_THREADS
                  << " persistent worker threads: " << (exact_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(exact_match);

        // Running a SECOND round through the same, still-alive pool
        // (a different input vector) must also match exactly -- the
        // pool is reused across rounds, exactly as a real decode loop
        // would reuse it across many tokens.
        std::vector<float> x2(N_IN), y2_serial(N_OUT), y2_pool(N_OUT, -999.0f);
        for (size_t i = 0; i < N_IN; ++i) x2[i] = (static_cast<float>(i % 7) - 3.0f) * 0.2f;
        gemv_serial_reference(W, x2.data(), y2_serial.data(), N_OUT, N_IN);
        pool.run_gemv(W, x2.data(), y2_pool.data(), N_IN, correct_ranges);
        bool second_round_match = (y2_pool == y2_serial);
        std::cout << "  second round through the SAME still-alive pool: "
                  << (second_round_match ? "bit-for-bit match" : "MISMATCH") << "\n";
        CHECK(second_round_match);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: the naive, truncating-division partition
    // silently drops rows past n_threads * (n_out / n_threads) -- run
    // through the SAME real thread pool, with output pre-filled with a
    // sentinel value, the dropped rows are left at that sentinel because
    // no thread was ever assigned to compute them.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: the naive truncating-division partition drops rows --\n";
    {
        std::vector<RowRange> naive_ranges = partition_rows_naive_buggy(N_OUT, N_THREADS);
        size_t naive_total_covered = 0;
        for (const auto& r : naive_ranges) naive_total_covered += r.count();
        size_t dropped_rows = N_OUT - naive_total_covered;
        std::cout << "  naive chunk size = " << N_OUT << " / " << N_THREADS << " = " << (N_OUT / N_THREADS)
                  << " (integer division), covering rows [0, " << (naive_total_covered) << ") only\n";
        std::cout << "  " << dropped_rows << " row(s) -- indices [" << naive_total_covered << ", "
                  << N_OUT << ") -- assigned to NO thread at all\n";
        CHECK(dropped_rows > 0);
        CHECK(naive_total_covered < N_OUT);

        constexpr float SENTINEL = -999.0f;
        std::vector<float> y_buggy(N_OUT, SENTINEL);
        TensorThreadPool pool(N_THREADS);
        pool.run_gemv(W, x_data.data(), y_buggy.data(), N_IN, naive_ranges);

        size_t rows_still_at_sentinel = 0;
        for (size_t row = 0; row < N_OUT; ++row) {
            if (y_buggy[row] == SENTINEL) ++rows_still_at_sentinel;
        }
        std::cout << "  after running the pool with this partition, " << rows_still_at_sentinel
                  << " output row(s) are still at the sentinel value " << SENTINEL
                  << " -- never written by any thread.\n";
        CHECK(rows_still_at_sentinel == dropped_rows);
        CHECK(rows_still_at_sentinel > 0);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
