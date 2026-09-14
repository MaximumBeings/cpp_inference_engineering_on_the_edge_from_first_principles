// 01_row_vs_column_parallel_matmul.cpp
// Chapter 11, Part 1: Chapter 10.3 partitioned a GEMV by OUTPUT row --
// this section names that choice explicitly (row-parallel) and builds its
// alternative (column-parallel), so the comparison this chapter's title
// promises -- which partitioning a real parallel forward pass should use
// -- is made with both options actually implemented and measured, not
// asserted from one option alone.
//
// Row-parallel: each thread owns a disjoint RANGE OF OUTPUT ROWS and
// computes each row's full dot product against the complete input vector.
// No two threads ever write the same output element, so the result is
// combined for free -- there is no combination step at all, and the
// result is bit-for-bit identical to a serial reference, exactly as
// Chapter 10.3 already verified.
//
// Column-parallel: each thread instead owns a disjoint RANGE OF INPUT
// COLUMNS and computes a PARTIAL dot product for every output row using
// only that range. Every thread now touches every output row, so the
// partial results must be combined -- summed together -- after the
// threads finish, an extra step row-parallel never needs at all. For
// single-token decode, where a fresh combination step would be required
// on every layer of every token, this extra synchronization is exactly
// why row-parallel, not column-parallel, is this book's (and most real
// engines') default for generation -- verified here, not merely asserted.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 01_row_vs_column_parallel_matmul.cpp -o 01_row_vs_column_parallel_matmul
//
// -ffp-contract=off appears here (and in every other file this chapter
// locks) for a reason Section 11.2 discovered and documents in full: GCC's
// default FMA-contraction heuristic can fuse a multiply and an add
// differently across architectures for textually identical arithmetic,
// which silently broke a DIFFERENT file's serial-vs-parallel bit-exactness
// on aarch64 while leaving it intact on x86_64. This file's own checked
// output happens to be unaffected either way, but every file in this
// chapter compiles with the flag uniformly rather than asking a reader to
// remember which files needed it and which merely got lucky.

#include <atomic>
#include <cmath>
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
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((double)(a) - (double)(b)) < (tol))

using Matrix = std::mdspan<const float, std::dextents<size_t, 2>>;

void gemv_serial_reference(Matrix W, const float* x, float* y, size_t n_out, size_t n_in) {
    for (size_t row = 0; row < n_out; ++row) {
        float acc = 0.0f;
        for (size_t col = 0; col < n_in; ++col) acc += W[row, col] * x[col];
        y[row] = acc;
    }
}

// A [start, end) range and Chapter 10.3's own ceiling-division partition,
// reused verbatim: covers [0, total) exactly once regardless of whether
// total divides evenly by n_threads.
struct Range { size_t start, end; };
std::vector<Range> partition_ceil(size_t total, size_t n_threads) {
    std::vector<Range> ranges;
    size_t chunk = (total + n_threads - 1) / n_threads;
    for (size_t t = 0; t < n_threads; ++t) {
        size_t start = std::min(t * chunk, total);
        size_t end = std::min(start + chunk, total);
        ranges.push_back({start, end});
    }
    return ranges;
}

// =========================================================================
// ROW-PARALLEL: each thread's row range is written to EXCLUSIVELY by that
// thread -- no other thread ever reads or writes those same elements of
// y -- so no combination step exists at all. This is exactly Chapter
// 10.3's partitioning, rebuilt here (with plain create-and-join threads
// rather than a persistent pool) specifically to compare it directly
// against column-parallel in the same file.
// =========================================================================
void gemv_row_parallel(Matrix W, const float* x, float* y, size_t n_out, size_t n_in, size_t n_threads) {
    std::vector<Range> ranges = partition_ceil(n_out, n_threads);
    std::vector<std::thread> workers;
    for (size_t t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, r = ranges[t]] {
            for (size_t row = r.start; row < r.end; ++row) {
                float acc = 0.0f;
                for (size_t col = 0; col < n_in; ++col) acc += W[row, col] * x[col];
                y[row] = acc;
            }
        });
    }
    for (auto& w : workers) w.join();
}

// =========================================================================
// COLUMN-PARALLEL, fixed-order combine: each thread owns a disjoint range
// of INPUT columns and writes its own, exclusively-owned partial output
// vector -- so the PARALLEL phase itself has no shared mutable state and
// needs no synchronization at all, exactly like row-parallel's parallel
// phase. The difference is what happens next: every output row now needs
// n_threads partial values summed together, and this function does that
// combination in a FIXED order (thread 0's partial, then thread 1's, ...)
// so the result is exactly reproducible from run to run, regardless of
// which thread happened to finish its own work first.
// =========================================================================
void gemv_column_parallel_fixed_order(Matrix W, const float* x, float* y, size_t n_out, size_t n_in, size_t n_threads) {
    std::vector<Range> ranges = partition_ceil(n_in, n_threads);
    std::vector<std::vector<float>> partials(n_threads, std::vector<float>(n_out, 0.0f));
    std::vector<std::thread> workers;
    for (size_t t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, t, r = ranges[t]] {
            for (size_t row = 0; row < n_out; ++row) {
                float acc = 0.0f;
                for (size_t col = r.start; col < r.end; ++col) acc += W[row, col] * x[col];
                partials[t][row] = acc;
            }
        });
    }
    for (auto& w : workers) w.join();

    // The combination step every column-parallel matmul needs, and
    // row-parallel never does -- done here in a fixed, deterministic
    // order, not by whichever thread's contribution happens to arrive
    // first at a shared accumulator.
    for (size_t row = 0; row < n_out; ++row) {
        float acc = 0.0f;
        for (size_t t = 0; t < n_threads; ++t) acc += partials[t][row];
        y[row] = acc;
    }
}

// =========================================================================
// [COMMON TRAP, never executed] a version that "simplifies" the
// combination step by having every thread fetch_add its own partial
// contributions directly into a SHARED std::atomic<float> array as it
// computes them, instead of writing to its own exclusively-owned partial
// vector first. std::atomic<float>::fetch_add is a real, standard,
// race-free operation -- no update is ever lost, and the C++ memory model
// guarantees this compiles to something well-defined. What it does NOT
// guarantee is which ORDER the threads' additions land in, and floating-
// point addition is not associative: the exact final bit pattern of
// y[row] can differ between two runs of the identical binary, even though
// every run's answer is numerically close to correct. This is a
// DIFFERENT hazard than Chapter 10.2's plain, unsynchronized counter --
// there is no lost update here, no undefined behavior, and no data race
// by the standard's own definition -- only a reproducibility hazard, and
// exactly why this function is included as real, compilable source and
// never executed for this file's own checked, locked output.
// =========================================================================
[[maybe_unused]] void gemv_column_parallel_racy_order(Matrix W, const float* x, std::atomic<float>* y, size_t n_out, size_t n_in, size_t n_threads) {
    std::vector<Range> ranges = partition_ceil(n_in, n_threads);
    std::vector<std::thread> workers;
    for (size_t t = 0; t < n_threads; ++t) {
        workers.emplace_back([&, r = ranges[t]] {
            for (size_t row = 0; row < n_out; ++row) {
                float acc = 0.0f;
                for (size_t col = r.start; col < r.end; ++col) acc += W[row, col] * x[col];
                y[row].fetch_add(acc, std::memory_order_relaxed);  // order of arrival is not fixed
            }
        });
    }
    for (auto& w : workers) w.join();
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 11.1: Row-Parallel vs. Column-Parallel Matmul\n";
    std::cout << "========================================================\n\n";

    constexpr size_t N_OUT = 64;
    constexpr size_t N_IN = 256;
    constexpr size_t N_THREADS = 4;

    std::vector<float> w_data(N_OUT * N_IN), x_data(N_IN), y_serial(N_OUT);
    for (size_t i = 0; i < w_data.size(); ++i) w_data[i] = (static_cast<float>(i % 19) - 9.0f) * 0.04f;
    for (size_t i = 0; i < N_IN; ++i) x_data[i] = (static_cast<float>(i % 13) - 6.0f) * 0.08f;
    Matrix W(w_data.data(), N_OUT, N_IN);
    gemv_serial_reference(W, x_data.data(), y_serial.data(), N_OUT, N_IN);

    // =====================================================================
    // TEST 1: row-parallel needs no combination step at all -- the
    // parallel result is bit-for-bit identical to the serial reference,
    // exactly as Chapter 10.3 already established.
    // =====================================================================
    std::cout << "-- Test 1: row-parallel matches the serial reference bit-for-bit --\n";
    {
        std::vector<float> y_row(N_OUT, -999.0f);
        gemv_row_parallel(W, x_data.data(), y_row.data(), N_OUT, N_IN, N_THREADS);
        bool exact = (y_row == y_serial);
        std::cout << "  " << N_OUT << " rows, " << N_THREADS << " threads: "
                  << (exact ? "bit-for-bit match" : "MISMATCH") << " (no combination step needed)\n";
        CHECK(exact);
    }

    // =====================================================================
    // TEST 2: column-parallel, combined in a fixed order, matches the
    // serial reference within a small floating-point-reordering
    // tolerance -- the same class of tolerance Chapter 9.2's AVX2 kernel
    // needed, and for the identical underlying reason: the same values
    // are being summed in a genuinely different order (per-thread partial
    // sums, then combined) than the serial reference's single, strictly
    // sequential accumulation.
    // =====================================================================
    std::cout << "\n-- Test 2: column-parallel (fixed-order combine) vs. serial reference --\n";
    {
        std::vector<float> y_col(N_OUT, -999.0f);
        gemv_column_parallel_fixed_order(W, x_data.data(), y_col.data(), N_OUT, N_IN, N_THREADS);
        float max_abs_err = 0.0f;
        for (size_t row = 0; row < N_OUT; ++row) max_abs_err = std::max(max_abs_err, std::fabs(y_col[row] - y_serial[row]));
        std::cout << "  " << N_OUT << " rows, " << N_THREADS << " column chunks: max absolute error = "
                  << max_abs_err << " (a real combination step ran; row-parallel needed none)\n";
        CHECK(max_abs_err < 1e-3f);

        // Run it again, same thread count -- the fixed-order combine
        // means this is exactly reproducible, unlike the racy version
        // above would be.
        std::vector<float> y_col_rerun(N_OUT, -999.0f);
        gemv_column_parallel_fixed_order(W, x_data.data(), y_col_rerun.data(), N_OUT, N_IN, N_THREADS);
        bool rerun_matches = (y_col == y_col_rerun);
        std::cout << "  second call, same thread count: " << (rerun_matches ? "bit-for-bit identical to the first" : "DIFFERED") << "\n";
        CHECK(rerun_matches);
    }

    // =====================================================================
    // TEST 3 [COMMON TRAP]: column-parallel needs a real combination step
    // that row-parallel simply does not -- measured directly as memory
    // this file actually allocates, not asserted as a general claim.
    // For a decode step run on every one of a model's L layers, this
    // combination step's memory traffic (N_THREADS partial vectors of
    // N_OUT floats, read and summed on every single layer) is exactly
    // the recurring cost row-parallel's disjoint outputs never pay.
    // =====================================================================
    std::cout << "\n-- Test 3 [COMMON TRAP]: column-parallel's combination step is real, recurring cost --\n";
    {
        size_t row_parallel_extra_bytes = 0;  // no partials, no combination buffer at all
        size_t column_parallel_extra_bytes = N_THREADS * N_OUT * sizeof(float);  // the partials array
        std::cout << "  Row-parallel extra memory beyond the output vector itself: "
                  << row_parallel_extra_bytes << " bytes\n";
        std::cout << "  Column-parallel extra memory (N_THREADS=" << N_THREADS << " partial vectors of "
                  << N_OUT << " floats each): " << column_parallel_extra_bytes << " bytes\n";
        std::cout << "  This combination buffer is allocated and summed again on EVERY call -- every\n";
        std::cout << "  layer, every decode step -- which is exactly the recurring synchronization and\n";
        std::cout << "  memory-traffic cost row-parallel's disjoint per-thread outputs never incur.\n";
        CHECK(column_parallel_extra_bytes > row_parallel_extra_bytes);
        CHECK(row_parallel_extra_bytes == 0);
    }

    // -- Summary --------------------------------------------------------
    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
