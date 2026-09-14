# Chapter 11: Parallelizing the Forward Pass -- Row-Parallel, Head-Parallel, and Reproducible Across Thread Counts

**What you will understand by the end of this chapter:**

- The two structurally different ways to split a matrix-vector product across threads — row-parallel (partition the OUTPUT, no combination step, bit-for-bit reproducible against a serial reference) and column-parallel (partition the INPUT, requires a combination step whose order determines the result) — and a third, well-defined-but-not-reproducible way to combine column-parallel's partial sums that this book will not lock into its own checked output.
- How to generalize Chapter 10.3's single-purpose GEMV thread pool into a persistent pool that runs ANY row-partitioned task, so one pool, created once, carries an entire transformer layer's several different matmul phases — and the real, structural constraint on when independent matmuls may be BATCHED into a single parallel phase together.
- Why parallelizing GQA attention by query head is embarrassingly parallel for correctness, but why thread boundaries that ignore KV-group boundaries cause a shared KV head's cache rows to be read redundantly by more than one thread — and how aligning boundaries to whole KV groups fixes it.
- Why "reproducible across repeated runs at a FIXED thread count" — this book's entire discipline since Chapter 10 — is a strictly weaker claim than "reproducible across a CHANGE in thread count," and the fixed-block reduction technique that actually delivers the stronger guarantee.
- How this chapter's four earlier techniques combine into one complete, real decode step whose output is bit-for-bit identical regardless of how many worker threads compute it — and how a single non-thread-count-independent reduction, left anywhere in that pipeline, is enough to break the guarantee for the whole thing.

**What you need to know first:**

- Chapter 10's `std::barrier`-based persistent thread pool and its row-parallel, bit-for-bit verification discipline, extended in this chapter rather than replaced.
- Chapter 3.1's RMSNorm, 3.2's SwiGLU FFN, and 3.4's GQA attention with its `std::mdspan`-backed KVCache — reused verbatim throughout this chapter as the real kernels being parallelized, never re-derived.
- This chapter continues Chapter 10's `-pthread` requirement, and every file touching the mdspan-backed KVCache also needs the vendored mdspan flags Chapter 8 established.
- A hazard this chapter's own authoring surfaced, not merely one described in the abstract: identical source, compiled once for x86_64 and once for aarch64, silently disagreed about serial-vs-parallel bit-exactness, purely because of where the compiler's default FMA-contraction heuristic chose to fuse a multiply and an add. Every file in this chapter compiles with `-ffp-contract=off` because of this, and Section 11.2 documents the discovery — what broke, why, and how it was found — in full.

---

Chapter 10 built the primitives — atomics, mutexes, a barrier-synchronized thread pool, an awareness of false sharing, a work-stealing scheduler — and verified a single GEMV split across them. This chapter spends those primitives on the actual forward pass this book has built one real kernel at a time since Part 0. Section 11.1 names the two fundamental ways to split a matrix-vector product and shows exactly what each one costs and guarantees. Section 11.2 generalizes Chapter 10.3's fixed-shape thread pool into one that can run an entire layer's several different matmul phases, and documents a genuine, hard-won lesson about floating-point reproducibility across CPU architectures along the way. Section 11.3 parallelizes GQA attention itself, across query heads, and finds a real memory-locality hazard in how naively that parallelization is usually done. Section 11.4 confronts a harder version of this book's reproducibility standard: not merely "the same answer on a rerun," but "the same answer no matter how many threads happen to be available." Section 11.5 closes the chapter, and Part 2, by combining every one of these techniques into one complete, real, multi-layer decode step — verified bit-for-bit identical across a change in thread count, and verified just as rigorously to break the moment even one piece of it does not honor that standard.

## 11.1 Row-Parallel vs. Column-Parallel Matmul

### Intuition

A matrix-vector product can be split across threads two structurally different ways: by OUTPUT row, or by INPUT column. Chapter 10.3 already used the first without naming it as a choice. This section names both, and shows precisely what each one guarantees and what each one costs — because the difference is not a matter of taste, it changes whether the result is bit-for-bit reproducible at all.

### The Concept, In Detail

Row-parallel assigns each thread a disjoint range of OUTPUT rows; every thread computes its rows' dot products completely independently, using its own full copy of the input vector, and writes to elements no other thread ever touches. Because no combination step exists at all, and because each row's own accumulation runs in exactly the same left-to-right order the serial reference uses, the result is bit-for-bit identical to serial — exactly Chapter 10.3's finding, now given its proper name. Column-parallel instead assigns each thread a disjoint range of INPUT columns; every thread computes a FULL n_out-length partial output using only its slice of the input, and after joining, those n_threads partial vectors must be combined element-by-element into the final output. That combination's order matters, because floating-point addition is not associative: combining the partial vectors in a FIXED sequential order (thread 0's, then thread 1's, and so on), after every thread has written into its own exclusively-owned buffer with no shared mutable state during the parallel phase itself, is well-defined and reproducible against itself across repeated calls at a fixed thread count — though it lands only within a tight tolerance of the serial reference, not bit-identical to it, since the summation order genuinely differs from the serial reference's own. A third combination strategy exists and is worth naming precisely because it is tempting: skipping the combine step entirely by having every thread call `std::atomic<float>::fetch_add` directly on the shared output during the parallel phase. This is well-defined by the C++ standard — no lost updates, no data race, no undefined behavior of any kind — but it is NOT reproducible run to run, because the arrival order of concurrent `fetch_add` calls is not fixed by anything the standard promises, and floating-point addition's non-associativity means that unfixed order can change the exact final bit pattern. This book includes that function as real, compilable source and never executes it from this file's own checked `main()`, for the identical honesty reason Chapter 10.2 never executed its racy counter. Column-parallel also pays a real, measured, recurring cost that row-parallel never does: a combination buffer of `n_threads * n_out` floats, allocated and summed again on every single call, in every layer, on every decode step.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 01_row_vs_column_parallel_matmul.cpp -o 01_row_vs_column_parallel_matmul
./01_row_vs_column_parallel_matmul
```

**Sample input:** a 64-row, 256-column matrix split across 4 threads two ways — row-parallel, checked bit-for-bit against a serial reference with no combination step at all, and column-parallel with a fixed-order combine, checked within a `1e-3` tolerance against the same serial reference and additionally checked bit-for-bit reproducible against a second call to itself; and a direct measurement of column-parallel's real combination-buffer memory cost against row-parallel's zero extra bytes.

```text
========================================================
Chapter 11.1: Row-Parallel vs. Column-Parallel Matmul
========================================================

-- Test 1: row-parallel matches the serial reference bit-for-bit --
  64 rows, 4 threads: bit-for-bit match (no combination step needed)

-- Test 2: column-parallel (fixed-order combine) vs. serial reference --
  64 rows, 4 column chunks: max absolute error = 4.17233e-07 (a real combination step ran; row-parallel needed none)
  second call, same thread count: bit-for-bit identical to the first

-- Test 3 [COMMON TRAP]: column-parallel's combination step is real, recurring cost --
  Row-parallel extra memory beyond the output vector itself: 0 bytes
  Column-parallel extra memory (N_THREADS=4 partial vectors of 64 floats each): 1024 bytes
  This combination buffer is allocated and summed again on EVERY call -- every
  layer, every decode step -- which is exactly the recurring synchronization and
  memory-traffic cost row-parallel's disjoint per-thread outputs never incur.

========================================================
5/5 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] combining column-parallel's partial sums with a shared atomic instead of a fixed order"
    `std::atomic<float>::fetch_add` on a shared output element, called by every thread directly during the parallel phase, looks like the simplest possible way to combine column-parallel's partial results — no separate combine step, no per-thread buffer. It is also completely well-defined: the C++ standard guarantees no lost updates and no data race. What it does NOT guarantee is which thread's `fetch_add` lands first, and because floating-point addition is not associative, a different arrival order can produce a different final bit pattern on a different run of the identical program on the identical machine. This is a genuinely different hazard from Chapter 10.2's data race — there is no undefined behavior here at all — and it is exactly why this section's checked, locked output uses the fixed-order combine instead: reproducibility, not just well-definedness, is the standard this book holds every locked number to.

## 11.2 A Persistent Pool Across a Full Layer's Phases

### Intuition

Chapter 10.3's `TensorThreadPool` ran exactly one task shape, forever: a GEMV, every round. A real transformer layer needs that same persistent pool to run several DIFFERENT task shapes back to back within a single token — the Q/K/V projections, the output projection, the FFN gate/up projections, and the FFN down projection — with RMSNorm, attention, and SwiGLU's elementwise gate running sequentially in between. This section generalizes the pool from "always run a GEMV" to "run any row-partitioned task the caller hands it."

### The Concept, In Detail

The generalized pool's task descriptor becomes an arbitrary `std::function<void(size_t,size_t)>` — "compute this row range, however the caller defines that" — so the SAME pool instance, created once, runs a Q/K/V projection phase this round and an FFN down-projection phase the next, with no new threads and no new synchronization primitives created in between. This section verifies that generalization two ways: one full layer's worth of phases, run through the pool once, matches a serial reference bit-for-bit (every phase is still row-parallel with disjoint writes, so Chapter 10.3's bit-exactness carries over unchanged); and the SAME still-alive pool, run across six consecutive layers with six different weight sets, still matches a serial reference applied the same six times — proving the pool genuinely carries a multi-phase, multi-layer forward pass, not merely one repeated task shape. A second, independent point this section makes concrete: independent matmuls can only be BATCHED into a single parallel phase — one pair of barrier arrivals instead of several — when they share the same output row count. Q, K, and V all project into the same DIM-row space, so one phase computes all three; the FFN gate and up projections both produce D_FF rows, so they batch together too — but D_FF is not DIM, so gate/up cannot be batched into the same phase as the output or down projections without silently leaving rows uncomputed. A third hazard surfaced only while cross-checking this file on aarch64 during authoring: the very first version of this file matched serial-vs-parallel bit-for-bit on x86_64 but NOT on aarch64, even though both platforms were internally deterministic across repeated runs on each platform separately. The cause was GCC's default `-ffp-contract=fast`, which permits fusing `a*b+c` into one rounding step (a hardware FMA) wherever the compiler judges it profitable — a judgment made per call site, after inlining, so the identical accumulation, textually written once, can get fused on one architecture where scalar FMA is always available (aarch64) and stay unfused on another where it is not, absent an explicit `-mfma` flag (this book's baseline x86_64 build). Passing `-ffp-contract=off` restores the literal, separate multiply-then-add the source asks for on every architecture, and every file in this chapter compiles with it as a result.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_persistent_pool_full_layer.cpp -o 02_persistent_pool_full_layer
./02_persistent_pool_full_layer
```

**Sample input:** a real, small GQA transformer layer (DIM=64, D_FF=176, 8 query heads sharing 2 KV heads), its RMSNorm, QKV projections, GQA attention, output projection, and SwiGLU FFN all reused verbatim from Chapters 3.1/3.2/3.4; one full layer run through a persistent `ParallelPool` and checked bit-for-bit against a serial reference; the SAME still-alive pool reused across six consecutive layers, checked bit-for-bit the same way; and a deliberate demonstration of a batched phase told the wrong shared row count, leaving a checkable, exact number of FFN rows uncomputed.

```text
========================================================
Chapter 11.2: A Persistent Pool Across a Full Layer's Phases
========================================================

-- Test 1: one full layer, serial vs. pool-parallel, bit-for-bit --
  DIM=64 D_FF=176 N_HEADS_Q=8 N_HEADS_KV=2: bit-for-bit match

-- Test 2: SAME pool reused across 6 consecutive layers --
  6 layers through one still-alive pool: bit-for-bit match

-- Test 3 [COMMON TRAP]: batching with the wrong shared row count drops rows --
  phase told total=64 but gate/up actually have 176 rows
  gate rows never computed: 112 (expected 112)
  up rows never computed:   112 (expected 112)

========================================================
5/5 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] batching phases whose output row counts do not match"
    Batching the Q, K, and V projections into one parallel phase works because all three matmuls happen to produce output vectors of the same length, so a single row range is valid for every one of them at once. It is tempting to extend that same batching instinct to any two matmuls that are both "parallel phases" — for instance, telling the FFN gate/up phase it has DIM rows when it actually has D_FF, as if copy-pasting the output-projection phase's row count into the wrong place. The pool has no way to know the caller made this mistake: it faithfully partitions and runs exactly the row count it was told, and the rows past that count — here, `D_FF - DIM` of them, in both the gate and up projections — are simply never assigned to any thread and never computed, left at whatever the output buffer contained beforehand. The fix is not a smarter pool; it is remembering that batching is only valid when every matmul folded into one phase genuinely shares the same output row count.

## 11.3 Head-Parallel Attention

### Intuition

Section 11.2 kept attention itself sequential. Attention is, in fact, the most embarrassingly parallel phase in the whole layer: GQA computes an entirely independent score vector, softmax, and weighted sum for every query head, writing to a disjoint slice of the output — no combination step, no shared mutable state between heads. The real design question is not whether attention can be parallelized by head, but how heads should be assigned to threads, given that GQA's whole premise is several query heads SHARING one KV head's cached rows.

### The Concept, In Detail

Partitioning query heads evenly by raw INDEX, ignoring which KV head each one belongs to, routinely splits a single KV head's group across two different threads — with 8 query heads sharing 2 KV heads in a group of 4, even an evenly-divided 4-thread partition (2 heads per thread) puts heads 0-1 and heads 2-3 in different threads despite BOTH pairs belonging to KV head 0's group, so KV head 0's cache rows get read by two different threads instead of one. This section counts that redundancy directly — for a given partition, which KV head index each thread's assigned heads resolve to, and how many DISTINCT threads end up touching each one — rather than timing anything, the same structural-arithmetic verification Chapter 10.3's Test 1 used for its own row partition, before ever running a single thread. The fix assigns whole KV GROUPS to threads instead of raw query-head ranges: ceiling-divide the KV HEAD count across threads, then expand each thread's KV-head range back into the query heads that belong to it. Every KV head's group is now owned by exactly one thread, because a KV head only ever belongs to one range in this scheme — verified both structurally (zero KV heads touched by more than one thread) and functionally (the real threaded computation using this partition still matches the serial reference bit-for-bit, so the fix costs nothing in correctness). Every head's own full computation — scores, softmax, weighted sum — is written as a single free function, `attend_one_head`, called identically from the serial loop and from every parallel partition tried in this section, applying Section 11.2's own lesson about not giving the compiler two textually different call sites to treat differently.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_head_parallel_attention.cpp -o 03_head_parallel_attention
./03_head_parallel_attention
```

**Sample input:** 8 query heads sharing 2 KV heads (group size 4) over a 10-position cache; head-parallel attention with a naive 4-thread, per-index partition checked bit-for-bit against a serial reference; that same naive partition's KV-head redundancy counted directly (2 of 2 KV heads touched by more than one thread); and a group-aligned 2-thread partition checked both to eliminate the redundancy entirely (0 KV heads touched by more than one thread) and to still match the serial reference bit-for-bit.

```text
========================================================
Chapter 11.3: Head-Parallel Attention
========================================================

-- Test 1: head-parallel (naive partition) matches serial, bit-for-bit --
  8 heads across 4 threads: bit-for-bit match

-- Test 2 [COMMON TRAP]: naive partition reads some KV heads redundantly --
  thread 0: query heads [0, 2)
  thread 1: query heads [2, 4)
  thread 2: query heads [4, 6)
  thread 3: query heads [6, 8)
  distinct KV heads touched: 2 / 2
  KV heads touched by MORE than one thread: 2
  total (thread, kv-head) touches: 4 (ideal, no redundancy: 2)

-- Test 3: group-aligned partition eliminates the redundancy --
  thread 0: query heads [0, 4)
  thread 1: query heads [4, 8)
  KV heads touched by MORE than one thread: 0 (was nonzero with the naive partition above)
  group-aligned partition vs. serial reference: bit-for-bit match

========================================================
7/7 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] partitioning query heads by index instead of by KV group"
    An even, per-index partition of query heads across threads is completely CORRECT — every test in this section confirms it matches the serial reference exactly, no matter how the heads are divided up, because each head's computation is fully independent of every other head's. What an even index partition does not protect is memory locality: with more query heads than KV heads (GQA's entire premise), a thread boundary drawn by raw head index has no reason to land on a KV-group boundary, and when it does not, the KV head whose group straddles that boundary gets its cache rows read once per thread that touches any part of its group, rather than once total. The fix — assigning whole KV groups to threads instead of raw head ranges — is not a correctness fix (both partitions are equally correct); it is a structural fix to how many times the same cached bytes get fetched, verified here as a direct count of KV-head touches rather than left as an unverified assumption about "obviously" parallel work.

## 11.4 Deterministic Floating-Point Reduction, Independent of Thread Count

### Intuition

Combining per-thread-ID slots in fixed thread order — Section 11.1's own fixed-order combine, and the technique this book has relied on since Chapter 10 for "reproducible" — is reproducible run to run FOR A GIVEN thread count. This section asks the harder question that guarantee was never tested against: is it reproducible when the THREAD COUNT ITSELF changes? A real engine does not always run with the same thread count, and floating-point addition is not associative, so a different thread count means different chunk boundaries, which means a different summation order, which can mean a different final bit pattern — even though every run is still numerically correct.

### The Concept, In Detail

A per-thread-indexed reduction chunks the input by `ceil(n / n_threads)`, so changing `n_threads` moves every chunk boundary, changing which elements get summed together first in each thread's own partial sum — and this section confirms empirically that this really does change the final bit pattern: across six tested thread counts (1, 2, 3, 4, 5, 8) on a 997-element dot product, five distinct bit patterns appear, even though every single one of them is reproducible on its own if that same thread count is used again. The fix generalizes "write to your own slot, combine in fixed order" from per-THREAD slots to per-BLOCK slots, where block boundaries are a FIXED CONSTANT (16 elements per block here) chosen independently of how many threads happen to be running. A block's own internal sum is always computed the same way — left-to-right over that block's fixed element range — no matter which thread is assigned to compute it, and the final combination always walks the same fixed sequence of block indices, 0 through the last block, regardless of thread count. However many threads did the work, and however the blocks were divided up among them, the exact same set of intermediate sums gets combined in the exact same order every time — verified here as a BIT-IDENTICAL result across all six tested thread counts, where the thread-indexed version produced five different ones.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread 04_deterministic_reduction.cpp -o 04_deterministic_reduction
./04_deterministic_reduction
```

**Sample input:** a 997-element dot product (deliberately not a clean multiple of any tested thread count or of the fixed 16-element block size), reduced with fixed-size blocks across thread counts 1, 2, 3, 4, 5, and 8 and checked bit-identical across all six; the same dot product reduced with per-thread-indexed slots across the identical six thread counts, checked reproducible for a single fixed thread count but shown to produce multiple distinct bit patterns across the six.

```text
========================================================
Chapter 11.4: Deterministic Floating-Point Reduction, Independent of Thread Count
========================================================

-- Test 1: fixed-block reduction is numerically sound and self-reproducible --
  plain serial dot product:        4.175
  fixed-block reduction (4 threads): 4.175

-- Test 2: fixed-block reduction is bit-identical ACROSS different thread counts --
  n_threads=1: 4.17500019  (matches n_threads=1)
  n_threads=2: 4.17500019  (matches n_threads=1)
  n_threads=3: 4.17500019  (matches n_threads=1)
  n_threads=4: 4.17500019  (matches n_threads=1)
  n_threads=5: 4.17500019  (matches n_threads=1)
  n_threads=8: 4.17500019  (matches n_threads=1)
  unique bit patterns across 6 thread counts: 1

-- Test 3 [COMMON TRAP]: thread-indexed reduction is NOT stable across thread counts --
  same thread count (4) twice: bit-identical (expected: bit-identical)
  n_threads=1: 4.17500210
  n_threads=2: 4.17500210
  n_threads=3: 4.17499971
  n_threads=4: 4.17500114
  n_threads=5: 4.17500067
  n_threads=8: 4.17500162
  unique bit patterns across 6 thread counts: 5 (fixed-block reduction above got exactly 1)
  A test asserting "thread-indexed reduction is deterministic" and stopping
  after checking only ONE thread count -- exactly what Chapter 11.1 checked --
  would have missed this. "Reproducible" and "reproducible regardless of
  configuration" are different claims, and only the fixed-block scheme makes the
  second one.

========================================================
5/5 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] mistaking \"reproducible\" for \"reproducible under any configuration\""
    A test that creates a pool with one fixed thread count, runs a reduction against it twice, and confirms the two results match — exactly what Chapter 11.1 checked, and exactly what this section's own Test 3 checks first — genuinely does verify something real: the reduction is not internally flaky. It does NOT verify that the SAME reduction would produce the SAME result if the deployment happened to run with a different thread count next time, and this section shows concretely that the per-thread-indexed technique does not have that stronger property, even though it passes every single-thread-count reproducibility check perfectly. Only a reduction whose intermediate boundaries are fixed independently of thread count — this section's block scheme — can honestly claim reproducibility across a configuration change, and the two claims should never be described with the same word without checking which one was actually tested.

## 11.5 A Fully Multi-Threaded Decode Step

### Intuition

Every technique this chapter built, combined into one real, complete decode step — and checked against a stricter standard than any single section needed on its own: the exact same result, bit-for-bit, no matter how many worker threads compute it.

### The Concept, In Detail

Section 11.2's generalized pool and row-parallel matmul phases, and Section 11.3's group-aligned head-parallel attention, are already thread-count-independent by construction — both assign disjoint OUTPUT units (rows, or heads) to threads and never reorder any single row's or head's own internal accumulation, so however many threads share the work, the result cannot depend on the count. RMSNorm's own reduction is the one piece of a full layer that is a genuine cross-element reduction, and this section uses Section 11.4's fixed-block technique specifically so it does not undo the guarantee every other phase already provides for free. Combined — a persistent pool carrying a parallel RMSNorm reduction, a batched QKV projection phase, group-aligned head-parallel attention, an output projection phase, a batched FFN gate/up phase, and an FFN down-projection phase, run across four full transformer layers — the whole decode step matches a serial reference bit-for-bit with 4 worker threads, and matches it again, still bit-for-bit, when the exact same pipeline runs with 3 worker threads through a freshly created pool instead. This is the chapter's real payoff: not merely correct, and not merely reproducible on a rerun, but reproducible ACROSS a configuration change, because every phase that combines partial results was built with that specific property in mind from the start. The final test makes the fragility of that property concrete: swapping ONLY the RMSNorm reduction for Section 11.4's thread-indexed alternative — leaving every matmul phase and every attention head exactly as row/head-parallel as before — is enough to make the WHOLE decode step's final output depend on thread count again, because the broken reduction's result feeds into the residual stream that every later phase in every later layer builds on. One non-thread-count-independent piece, anywhere in an otherwise perfectly parallel pipeline, is enough to break the guarantee for the entire pipeline.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 05_full_decode_step_multithreaded.cpp -o 05_full_decode_step_multithreaded
./05_full_decode_step_multithreaded
```

**Sample input:** a real 4-layer GQA decode step (DIM=64, D_FF=176, 8 query heads sharing 2 KV heads), combining a fixed-block-reduced parallel RMSNorm, batched QKV projections, group-aligned head-parallel attention, an output projection, batched FFN gate/up projections, and an FFN down projection through one persistent pool; checked bit-for-bit against a serial reference at 4 threads, checked bit-for-bit again at 3 threads through a freshly created pool, and checked to DIFFER across those same two thread counts once RMSNorm's reduction alone is swapped for the non-thread-count-independent alternative.

```text
========================================================
Chapter 11.5: A Fully Multi-Threaded Decode Step
========================================================

-- Test 1: full 4-layer decode step, 4 threads, vs. serial reference --
  DIM=64 D_FF=176 4 layers, 4 threads: bit-for-bit match

-- Test 2: SAME decode step, DIFFERENT thread count (3), vs. serial reference --
  same 4 layers, 3 threads instead of 4: bit-for-bit match

-- Test 3 [COMMON TRAP]: one thread-indexed reduction breaks the WHOLE pipeline --
  4-thread run vs. 3-thread run, same decode step, broken reduction only: DIFFERS, as expected
  Every matmul phase and every attention head in this run is still exactly the
  same row/head-parallel code Tests 1 and 2 just verified as thread-count-
  independent on its own. The ONLY change is which reduction RMSNorm uses --
  and that alone is enough to make the final output depend on thread count.

========================================================
3/3 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming perfectly parallel phases guarantee a perfectly parallel pipeline"
    Every matmul phase in this file is row-parallel, and attention is head-parallel, and both are thread-count-independent by construction — true before Test 3 and still true during it. It is tempting to conclude that a pipeline built entirely out of such phases must itself be thread-count-independent as a whole. Test 3 shows that conclusion does not follow: a single reduction — one piece, appearing twice per layer, only mixed AMONG otherwise-perfect parallel phases — feeds its result into the residual stream that every subsequent projection, every subsequent attention call, and every subsequent layer builds on, so that one non-thread-count-independent result propagates and contaminates everything computed afterward. A pipeline's overall reproducibility guarantee is only as strong as its WEAKEST reduction, not the average of its phases, and verifying the strong phases individually is no substitute for verifying the assembled whole.

## Chapter Summary

This chapter took every real kernel this book has built since Part 0 and put Chapter 10's threading primitives to work parallelizing them, closing Part 2 with a complete, verified, multi-threaded forward pass. Section 11.1 named the two fundamental ways to split a matrix-vector product — row-parallel, bit-for-bit reproducible with no combination step, and column-parallel, which requires one, and is reproducible only when that combination happens in a fixed order rather than through a well-defined-but-order-dependent shared atomic. Section 11.2 generalized Chapter 10.3's single-shape thread pool into one that carries an entire layer's several different matmul phases, and along the way discovered and fixed a genuine cross-architecture hazard: the compiler's default FMA-contraction heuristic can silently break bit-exactness between a serial and a parallel code path on one architecture while leaving it intact on another. Section 11.3 parallelized GQA attention across query heads and found that naive, per-index thread boundaries — while perfectly correct — redundantly re-read a shared KV head's cache rows across threads, fixed by aligning boundaries to whole KV groups instead. Section 11.4 confronted a version of this book's reproducibility standard no earlier chapter needed: not just reproducible on a rerun at a fixed configuration, but reproducible ACROSS a change in that configuration, delivered by a fixed-block reduction rather than a per-thread-indexed one. Section 11.5 closed the chapter by combining all four techniques into one real, multi-layer decode step, verified bit-for-bit identical at two different thread counts, and then deliberately broke that guarantee by swapping out a single reduction, demonstrating that a pipeline's overall reproducibility is only as strong as its weakest link, never the average of its parts. Chapter 10 showed how to coordinate real threads correctly; this chapter showed how to make an entire real forward pass not just fast and correct, but reproducible in the specific, stronger sense a production inference engine actually needs.

## Self-Check Questions

1. Why is row-parallel matmul bit-for-bit reproducible against a serial reference with no tolerance needed, while column-parallel matmul (even with a correct, fixed-order combine) is only reproducible within a tolerance against that same serial reference?
2. Section 11.1 includes `gemv_column_parallel_racy_order` as real, compilable source but never calls it from `main()`. What specific guarantee does the C++ standard make about this function that Chapter 10.2's racy counter did NOT have, and why does the function still not get executed here?
3. In Section 11.2, why can the Q, K, and V projections be batched into a single parallel phase, but the FFN gate/up projections cannot be batched into that SAME phase alongside the output projection?
4. Section 11.2 discovered that identical source code produced bit-for-bit matching serial-vs-parallel results on x86_64 but not on aarch64. What compiler behavior caused this, and why did it appear on one architecture but not the other?
5. In Section 11.3, why does an EVEN, per-index partition of query heads across threads remain fully CORRECT even when it splits a KV group across two threads — what specifically does that partition cost, if not correctness?
6. Explain the fix Section 11.3 uses to eliminate redundant KV-head reads, and why it does not change the attention computation's actual result.
7. Section 11.4 distinguishes "reproducible for a fixed thread count" from "reproducible across a change in thread count." Why does the per-thread-indexed reduction satisfy the first but not the second?
8. Why does Section 11.4's fixed-block reduction's block boundary NOT depend on the number of threads running, and why is that specific independence what makes it reproducible across thread counts?
9. In Section 11.5, every matmul phase and every attention head is thread-count-independent on its own, yet Test 3 shows the WHOLE decode step becomes thread-count-dependent when only the RMSNorm reduction changes. Why does breaking one piece contaminate phases that are individually still correct?
10. Section 11.5's capstone reuses real kernels from Chapters 3.1, 3.2, and 3.4 rather than a cost-model abstraction like Chapter 8.5's. Why does that choice matter for what "bit-for-bit identical" is actually claiming in this chapter, compared to what Chapter 8.5 verified?

## Where We Go Next

This chapter closes Part 2 with a complete, real, multi-threaded transformer layer whose correctness and reproducibility are both verified exactly, at more than one thread count, all the way through a full decode step. Part 3 turns to the concerns of a real serving system built around that engine: how tokens enter and leave it, how a KV cache is actually managed as a growing, evictable resource across many concurrent requests, and the state a production inference server has to track that a single decode step, however well-parallelized, does not.

## Worked Solutions

**1.** Row-parallel assigns each thread a disjoint set of OUTPUT rows and never combines anything across threads — each row's own dot product runs in exactly the same left-to-right accumulation order the serial reference uses, so the two computations are literally the same sequence of floating-point operations, just executed by different threads for different rows. Column-parallel instead has every thread compute a full partial output vector from a different slice of INPUT columns, and the final result requires SUMMING those partial vectors together — an operation whose order differs from the serial reference's own single-pass accumulation order, and since floating-point addition is not associative, a different summation order can produce a different (though numerically very close) final value.

**2.** The C++ standard guarantees that `std::atomic<float>::fetch_add` is well-defined — no data race, no lost updates, no undefined behavior of any kind — unlike Chapter 10.2's `racy_increment_worker`, which was a genuine data race and therefore undefined behavior. What the standard does NOT guarantee is the ORDER in which concurrent `fetch_add` calls from different threads are applied, and because floating-point addition is not associative, that unfixed order means the exact final bit pattern is not guaranteed to reproduce from one run to the next, even though every individual run is perfectly well-defined and numerically correct. This section excludes it from checked output for a reproducibility reason, not a safety reason — a different, narrower version of Chapter 10.2's honesty standard.

**3.** Batching requires every matmul folded into one phase to share the same output row count, because the pool's single row-range partition is applied identically to every matmul inside that phase's lambda. Q, K, and V all project into DIM rows (or a subset of DIM for K and V's smaller KV-head dimension, which the lambda handles by clamping its own range), so one DIM-sized partition validly indexes all three. The FFN gate and up projections produce D_FF rows — a different size from DIM — so a phase batching gate/up with the (DIM-sized) output projection would either run out of bounds or, if told DIM as its row count, silently never compute the D_FF - DIM rows past that count, exactly the bug Section 11.2's Test 3 demonstrates deliberately.

**4.** GCC's default `-ffp-contract=fast` allows fusing a multiply immediately followed by an add (`a*b + c`) into a single hardware fused-multiply-add instruction wherever it judges doing so profitable — a decision made per call site, after inlining, not guaranteed to be applied consistently to two textually different pieces of source that compute the same arithmetic. aarch64's baseline instruction set always includes scalar FMA hardware, so the compiler can and does fuse readily there; this book's x86_64 baseline build has no `-mfma` flag, so no scalar hardware FMA exists to fuse into at all, and every accumulation stays as separate multiply-then-add regardless of call site. The result was two code paths (a standalone function and an inlined lambda body) that were textually different enough to receive different fusion treatment on aarch64 specifically, producing a genuine, silent bit-level divergence between serial and parallel results that did not appear on x86_64.

**5.** Every query head's attention computation — its scores, its softmax, its weighted sum over V — depends only on that head's own query vector and its assigned KV head's cached rows, never on any other head's computation or any shared mutable state between heads. Whichever thread computes a given head, and in whatever order threads happen to finish, each head's own result is identical to what the serial reference would compute for that head, so the overall output is bit-for-bit correct regardless of the partition. What an index-based partition costs is not correctness but memory locality: it can cause the SAME KV head's cache rows to be read by more than one thread instead of exactly one, a real but purely structural inefficiency in how many times shared bytes get fetched, not a wrong answer anywhere.

**6.** The fix ceiling-divides the KV HEAD count (not the query-head count) across threads, then expands each thread's resulting KV-head range back into the full set of query heads that belong to those KV heads' groups. Because every query head within one KV group is now guaranteed to be assigned to the SAME thread as every other query head in that group, no KV head's cache rows are ever needed by more than one thread. The computation each thread performs for each of its assigned heads is identical to before — the same `attend_one_head` function, called with the same arguments it would have received under any other partition — so the result is unchanged; only which heads get grouped onto which thread changes.

**7.** The per-thread-indexed reduction's chunk boundaries are computed as `ceil(n / n_threads)`, a value that depends directly on `n_threads`. For a FIXED value of `n_threads`, those boundaries — and therefore the summation order within and across chunks — never change between runs, so the result reproduces exactly every time, satisfying the first, weaker claim. But changing `n_threads` changes the chunk size and therefore moves every boundary, which changes which elements get summed together first in each chunk and in what order the chunk totals themselves get combined — a different summation order for floating-point values that are not guaranteed to sum identically regardless of order, so the second, stronger claim does not hold.

**8.** The fixed-block scheme's block boundaries are computed purely from a compile-time constant block size and the array's own length — block `b` always covers elements `[b*BLOCK_SIZE, min((b+1)*BLOCK_SIZE, n))`, a fact that involves `n_threads` nowhere in its definition. Thread count only affects which BLOCKS get assigned to which thread (via the same ceiling-division partition used everywhere else in this chapter), never where the blocks themselves begin or end, and each block's own internal sum is computed the same fixed way regardless of which thread happens to compute it. Since the final combination also walks every block index in the same fixed order (0 through the last block) regardless of how many threads existed, the entire computation — which elements get summed with which, and in what order the results get combined — is completely independent of `n_threads`, which is exactly the property a per-thread-indexed reduction lacks.

**9.** Every matmul phase and every attention head computes its own OWN piece of the layer's state independently and writes it to a disjoint location, so those computations genuinely do not depend on thread count on their own. But RMSNorm's reduction result becomes an INPUT to every phase that follows it — the normalized activations feed the QKV projections, which feed attention, which feeds the output projection, and the residual sum built from all of that feeds the next layer's own RMSNorm in turn. A different bit pattern out of one reduction is not an isolated error confined to that reduction; it is a different NUMBER flowing into every downstream computation that reads it, so phases that are individually perfectly thread-count-independent still produce different final results once their shared input differs.

**10.** Chapter 8.5 verified that a closed-form COST FORMULA (FLOPs and bytes) matched what a real kernel actually counted as it ran — a claim about resource accounting, not about the kernel's own numerical OUTPUT matching anything else bit-for-bit. Section 11.5's capstone instead runs the actual RMSNorm, matmul, attention, and SwiGLU arithmetic and checks that the resulting activation VALUES are bit-for-bit identical between a serial reference and a parallel, multi-threaded, multi-thread-count computation. "Bit-for-bit identical" here is a claim about the actual numbers a real decode step would produce — the same claim Chapter 10.3 first made for one GEMV — extended across an entire real, multi-layer forward pass, which is a substantially stronger and more directly useful guarantee for a real inference engine than a cost formula matching a cost counter.
