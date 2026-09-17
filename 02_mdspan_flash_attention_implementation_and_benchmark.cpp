// Chapter 28.2 -- A real, from-scratch, std::mdspan-based tiled Flash
// Attention implementation, checked directly against a naive full-matrix
// reference for exact numerical agreement, and "benchmarked" the way this
// book's own Appendix D discipline requires: with real, deterministic
// operation counts and real, deterministic peak-memory byte counts, never
// with genuinely variable wall-clock timing, which this book has kept out
// of every locked self-test contract from Chapter 1 onward. Q, K, V, and the
// output O are all real 2D matrices, viewed through std::mdspan over flat
// storage -- Chapter 2.1's own non-owning-view technique, applied here to
// this book's own final numerical kernel.
//
// This book's own real aarch64 hardware has only GCC 11.4.0, which predates
// C++23's multi-argument subscript-operator syntax (view[i, j]); exactly as
// Chapter 13's own mdspan appendix did, every mdspan access below goes
// through the idx2() helper (an ordinary function call, not new syntax) so
// this file compiles unchanged on that exact toolchain.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_mdspan_flash_attention_implementation_and_benchmark.cpp -o 02_mdspan_flash_attention_implementation_and_benchmark
// Run:     ./02_mdspan_flash_attention_implementation_and_benchmark

#include <mdspan/mdspan.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

using MatrixView = std::mdspan<double, std::dextents<size_t, 2>>;
using ConstMatrixView = std::mdspan<const double, std::dextents<size_t, 2>>;

std::array<size_t, 2> idx2(size_t a, size_t b) { return {a, b}; }

// A real, live operation counter -- incremented at the actual innermost
// multiply-accumulate step of whichever implementation is running, so this
// section's own real FLOP comparison is an empirically COUNTED fact about
// what each implementation actually executed, not merely a formula both are
// assumed to satisfy.
struct MacCounter {
    long long count = 0;
    void add(long long n) { count += n; }
};

// =======================================================================
// PART 1: naive attention -- materializes the REAL, FULL Nq x Nk score
// matrix as one real mdspan-viewed allocation before taking a single
// softmax, exactly the real behavior Section 28.1 quantified the memory
// cost of. Chapter 26.2's own shift-invariant stable softmax is applied to
// each real row.
// =======================================================================
void naive_attention_full(ConstMatrixView q, ConstMatrixView k, ConstMatrixView v, MatrixView out,
                           std::vector<double>& score_storage, MacCounter& macs) {
    size_t nq = q.extent(0), d = q.extent(1), nk = k.extent(0);
    score_storage.assign(nq * nk, 0.0);
    MatrixView scores(score_storage.data(), nq, nk);

    for (size_t i = 0; i < nq; ++i) {
        double m = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < nk; ++j) {
            double s = 0.0;
            for (size_t dd = 0; dd < d; ++dd) s += q[idx2(i, dd)] * k[idx2(j, dd)];
            macs.add(static_cast<long long>(d));
            scores[idx2(i, j)] = s;
            m = std::max(m, s);
        }
        double sum = 0.0;
        std::vector<double> e(nk);
        for (size_t j = 0; j < nk; ++j) {
            e[j] = std::exp(scores[idx2(i, j)] - m);
            sum += e[j];
        }
        for (size_t dd = 0; dd < d; ++dd) {
            double acc = 0.0;
            for (size_t j = 0; j < nk; ++j) acc += e[j] * v[idx2(j, dd)];
            macs.add(static_cast<long long>(nk));
            out[idx2(i, dd)] = acc / sum;
        }
    }
}

// =======================================================================
// PART 2: tiled Flash Attention -- Section 28.1's own online-softmax
// recurrence, applied per real Q-block over real K/V blocks, using a
// SINGLE, block-sized real score buffer reused across every block rather
// than ever materializing the full Nq x Nk matrix.
// =======================================================================
void tiled_flash_attention(ConstMatrixView q, ConstMatrixView k, ConstMatrixView v, MatrixView out,
                            size_t block_rows, size_t block_cols,
                            std::vector<double>& score_block_storage, MacCounter& macs) {
    size_t nq = q.extent(0), d = q.extent(1), nk = k.extent(0);
    score_block_storage.assign(block_rows * block_cols, 0.0);
    MatrixView score_block(score_block_storage.data(), block_rows, block_cols);

    for (size_t qi = 0; qi < nq; qi += block_rows) {
        size_t qend = std::min(nq, qi + block_rows);
        size_t rows_here = qend - qi;

        std::vector<double> m(rows_here, -std::numeric_limits<double>::infinity());
        std::vector<double> l(rows_here, 0.0);
        std::vector<std::vector<double>> o(rows_here, std::vector<double>(d, 0.0));

        for (size_t ki = 0; ki < nk; ki += block_cols) {
            size_t kend = std::min(nk, ki + block_cols);
            size_t cols_here = kend - ki;

            for (size_t ri = 0; ri < rows_here; ++ri) {
                double m_block = -std::numeric_limits<double>::infinity();
                for (size_t cj = 0; cj < cols_here; ++cj) {
                    double s = 0.0;
                    for (size_t dd = 0; dd < d; ++dd) s += q[idx2(qi + ri, dd)] * k[idx2(ki + cj, dd)];
                    macs.add(static_cast<long long>(d));
                    score_block[idx2(ri, cj)] = s;
                    m_block = std::max(m_block, s);
                }
                double m_new = std::max(m[ri], m_block);
                double correction = std::exp(m[ri] - m_new);
                double l_new = correction * l[ri];
                for (size_t dd = 0; dd < d; ++dd) o[ri][dd] *= correction;

                for (size_t cj = 0; cj < cols_here; ++cj) {
                    double e = std::exp(score_block[idx2(ri, cj)] - m_new);
                    l_new += e;
                    for (size_t dd = 0; dd < d; ++dd) o[ri][dd] += e * v[idx2(ki + cj, dd)];
                    macs.add(static_cast<long long>(d));
                }
                m[ri] = m_new;
                l[ri] = l_new;
            }
        }

        for (size_t ri = 0; ri < rows_here; ++ri)
            for (size_t dd = 0; dd < d; ++dd) out[idx2(qi + ri, dd)] = o[ri][dd] / l[ri];
    }
}

bool matrix_near(ConstMatrixView a, ConstMatrixView b, double eps = 1e-9) {
    if (a.extent(0) != b.extent(0) || a.extent(1) != b.extent(1)) return false;
    for (size_t i = 0; i < a.extent(0); ++i)
        for (size_t j = 0; j < a.extent(1); ++j)
            if (!near(a[idx2(i, j)], b[idx2(i, j)], eps)) return false;
    return true;
}

// A deterministic, hand-inspectable Q/K/V generator -- no randomness
// anywhere, so every shape this section tests is exactly reproducible.
void fill_deterministic(std::vector<double>& buf, size_t rows, size_t cols, double base) {
    buf.assign(rows * cols, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            buf[i * cols + j] = base + static_cast<double>(i) * 0.7 - static_cast<double>(j) * 0.3;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 28.2: A std::mdspan-Based Flash Attention Implementation and Benchmark\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: naive full-matrix attention, on a real, hand-verifiable single-query, "
                 "2-key case with a zero query vector, reduces to an exact unweighted average of the value "
                 "rows -- the identical degenerate case Section 28.1 verified by hand, now computed through "
                 "a real mdspan-viewed matrix rather than raw vectors --\n";
    {
        std::vector<double> qbuf = {0.0, 0.0};
        std::vector<double> kbuf = {1.0, 0.0, 0.0, 1.0};
        std::vector<double> vbuf = {2.0, 4.0, 6.0, 8.0};
        std::vector<double> obuf(2, 0.0);
        ConstMatrixView q(qbuf.data(), 1, 2), k(kbuf.data(), 2, 2), v(vbuf.data(), 2, 2);
        MatrixView out(obuf.data(), 1, 2);
        std::vector<double> scores;
        MacCounter macs;
        naive_attention_full(q, k, v, out, scores, macs);
        CHECK(near(out[idx2(0, 0)], 4.0));
        CHECK(near(out[idx2(0, 1)], 6.0));
        std::cout << "  naive_attention_full's own real mdspan-based output is {4.0, 6.0} -- exactly the "
                     "average of {2.0, 4.0} and {6.0, 8.0}, matching Section 28.1's own hand-verified "
                     "result exactly\n";
    }

    std::cout << "\n-- Test 2: tiled Flash Attention agrees with naive full-matrix attention to within 1e-9 "
                 "on every real output element, across several genuinely different real shapes and tile "
                 "sizes -- confirming the tiled real implementation is mathematically identical to the "
                 "naive reference, not merely a plausible-looking approximation --\n";
    {
        struct Shape { size_t nq, nk, d, br, bc; };
        std::vector<Shape> shapes = {
            {4, 6, 3, 2, 2}, {5, 5, 4, 1, 3}, {8, 3, 2, 3, 1}, {6, 9, 5, 4, 4}, {3, 3, 3, 3, 3},
        };
        for (const auto& sh : shapes) {
            std::vector<double> qbuf, kbuf, vbuf;
            fill_deterministic(qbuf, sh.nq, sh.d, 1.0);
            fill_deterministic(kbuf, sh.nk, sh.d, 0.5);
            fill_deterministic(vbuf, sh.nk, sh.d, 2.0);
            ConstMatrixView q(qbuf.data(), sh.nq, sh.d), k(kbuf.data(), sh.nk, sh.d), v(vbuf.data(), sh.nk, sh.d);

            std::vector<double> naive_out_buf(sh.nq * sh.d, 0.0), tiled_out_buf(sh.nq * sh.d, 0.0);
            MatrixView naive_out(naive_out_buf.data(), sh.nq, sh.d), tiled_out(tiled_out_buf.data(), sh.nq, sh.d);

            std::vector<double> naive_scores, tiled_scores;
            MacCounter naive_macs, tiled_macs;
            naive_attention_full(q, k, v, naive_out, naive_scores, naive_macs);
            tiled_flash_attention(q, k, v, tiled_out, sh.br, sh.bc, tiled_scores, tiled_macs);

            CHECK(matrix_near(ConstMatrixView(naive_out_buf.data(), sh.nq, sh.d),
                               ConstMatrixView(tiled_out_buf.data(), sh.nq, sh.d)));
        }
        std::cout << "  across 5 genuinely different real (Nq, Nk, d, block_rows, block_cols) shape "
                     "combinations, tiled Flash Attention's own real output matches naive full-matrix "
                     "attention's own output to within 1e-9 on every element, every single time\n";
    }

    std::cout << "\n-- Test 3: tiled Flash Attention performs EXACTLY the same real number of multiply-"
                 "accumulate operations as naive attention, counted directly at each implementation's own "
                 "innermost real loop -- confirming Flash Attention's own genuine benefit is reduced real "
                 "memory traffic, not reduced real compute, across the identical shapes Test 2 used --\n";
    {
        struct Shape { size_t nq, nk, d, br, bc; };
        std::vector<Shape> shapes = {
            {4, 6, 3, 2, 2}, {5, 5, 4, 1, 3}, {8, 3, 2, 3, 1}, {6, 9, 5, 4, 4}, {3, 3, 3, 3, 3},
        };
        for (const auto& sh : shapes) {
            std::vector<double> qbuf, kbuf, vbuf;
            fill_deterministic(qbuf, sh.nq, sh.d, 1.0);
            fill_deterministic(kbuf, sh.nk, sh.d, 0.5);
            fill_deterministic(vbuf, sh.nk, sh.d, 2.0);
            ConstMatrixView q(qbuf.data(), sh.nq, sh.d), k(kbuf.data(), sh.nk, sh.d), v(vbuf.data(), sh.nk, sh.d);

            std::vector<double> naive_out_buf(sh.nq * sh.d, 0.0), tiled_out_buf(sh.nq * sh.d, 0.0);
            MatrixView naive_out(naive_out_buf.data(), sh.nq, sh.d), tiled_out(tiled_out_buf.data(), sh.nq, sh.d);
            std::vector<double> naive_scores, tiled_scores;
            MacCounter naive_macs, tiled_macs;
            naive_attention_full(q, k, v, naive_out, naive_scores, naive_macs);
            tiled_flash_attention(q, k, v, tiled_out, sh.br, sh.bc, tiled_scores, tiled_macs);

            long long expected = 2LL * static_cast<long long>(sh.nq) * static_cast<long long>(sh.nk) * static_cast<long long>(sh.d);
            CHECK(naive_macs.count == expected);
            CHECK(tiled_macs.count == expected);
            CHECK(naive_macs.count == tiled_macs.count);
        }
        std::cout << "  for every one of the 5 real shapes, both implementations perform exactly "
                     "2*Nq*Nk*d real multiply-accumulate operations -- an empirically counted fact, not an "
                     "assumed formula -- confirming tiling changes only WHEN and WHERE those real "
                     "operations touch memory, never how many of them there are\n";
    }

    std::cout << "\n-- Test 4: naive attention's own real peak score-buffer size grows with Nq and Nk exactly "
                 "as Section 28.1 quantified, while tiled Flash Attention's own real peak score-buffer size "
                 "-- read directly from an actually allocated buffer's own real byte count, not merely a "
                 "formula -- stays fixed at block_rows * block_cols regardless of how large the real "
                 "underlying sequence grows --\n";
    {
        size_t d = 4, br = 2, bc = 2;
        std::vector<std::pair<size_t, size_t>> nq_nk_pairs = {{4, 4}, {8, 8}, {16, 16}};

        double tiled_peak_bytes = static_cast<double>(br * bc * sizeof(double));
        CHECK(near(tiled_peak_bytes, 32.0));

        double previous_naive_peak = 0.0;
        for (const auto& [nq, nk] : nq_nk_pairs) {
            std::vector<double> qbuf, kbuf, vbuf;
            fill_deterministic(qbuf, nq, d, 1.0);
            fill_deterministic(kbuf, nk, d, 0.5);
            fill_deterministic(vbuf, nk, d, 2.0);
            ConstMatrixView q(qbuf.data(), nq, d), k(kbuf.data(), nk, d), v(vbuf.data(), nk, d);
            std::vector<double> naive_out_buf(nq * d, 0.0);
            MatrixView naive_out(naive_out_buf.data(), nq, d);
            std::vector<double> naive_scores;
            MacCounter naive_macs;
            naive_attention_full(q, k, v, naive_out, naive_scores, naive_macs);

            double naive_peak_bytes = static_cast<double>(naive_scores.size() * sizeof(double));
            CHECK(near(naive_peak_bytes, static_cast<double>(nq * nk * sizeof(double))));
            if (previous_naive_peak > 0.0) CHECK(naive_peak_bytes > previous_naive_peak);
            previous_naive_peak = naive_peak_bytes;

            std::vector<double> tiled_out_buf(nq * d, 0.0);
            MatrixView tiled_out(tiled_out_buf.data(), nq, d);
            std::vector<double> tiled_scores;
            MacCounter tiled_macs;
            tiled_flash_attention(q, k, v, tiled_out, br, bc, tiled_scores, tiled_macs);
            double this_tiled_peak = static_cast<double>(tiled_scores.size() * sizeof(double));
            CHECK(near(this_tiled_peak, tiled_peak_bytes));
        }
        std::cout << "  naive attention's own real peak score-buffer size grows from 128 to 512 to 2048 "
                     "bytes as (Nq, Nk) grows from (4,4) to (8,8) to (16,16); tiled Flash Attention's own "
                     "real peak score-buffer size, read directly from its one reused allocation, stays "
                     "fixed at exactly " << tiled_peak_bytes << " bytes (2 x 2 doubles) at every one of "
                     "those same 3 real sequence lengths\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
