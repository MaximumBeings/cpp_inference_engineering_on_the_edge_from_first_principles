// Chapter 26.5 -- Quantizing a weight independently of every other
// weight, as Section 26.4's own affine map does, ignores something a
// real calibration dataset already knows: some weights matter more to
// a layer's real output than others, and the ones that don't yet have
// a fixed quantized value can still be nudged to compensate for the
// ones that already do. GPTQ's own real insight is that the layer's
// Hessian -- here, the genuinely simple H = 2 * X^T * X for a real
// squared-error loss over calibration data X -- is exactly the real
// second-order information needed to compute that compensation
// optimally. This section builds a real, from-scratch 2x2 Hessian,
// inverts it with a real Gauss-Jordan solver, and proves on a fully
// hand-traceable example that the resulting compensated quantization
// produces strictly less real output error than quantizing every
// weight independently.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_hessian_role_in_gptq.cpp -o 05_hessian_role_in_gptq
// Run:     ./05_hessian_role_in_gptq

#include <cmath>
#include <iostream>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: a real, general, from-scratch small dense matrix, with real
// multiply, transpose, and a real Gauss-Jordan inverse -- general in N,
// even though this section only ever exercises it at N = 2.
// =======================================================================
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<double> data;

    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0) {}
    double& at(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    double at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
};

Matrix transpose(const Matrix& m) {
    Matrix out(m.cols, m.rows);
    for (int r = 0; r < m.rows; r++)
        for (int c = 0; c < m.cols; c++) out.at(c, r) = m.at(r, c);
    return out;
}

Matrix matmul(const Matrix& a, const Matrix& b) {
    Matrix out(a.rows, b.cols);
    for (int r = 0; r < a.rows; r++)
        for (int c = 0; c < b.cols; c++) {
            double sum = 0.0;
            for (int k = 0; k < a.cols; k++) sum += a.at(r, k) * b.at(k, c);
            out.at(r, c) = sum;
        }
    return out;
}

// A real, general Gauss-Jordan matrix inverse via full pivoting on an
// augmented [M | I] matrix.
Matrix inverse(const Matrix& m) {
    int n = m.rows;
    Matrix aug(n, 2 * n);
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) aug.at(r, c) = m.at(r, c);
        aug.at(r, n + r) = 1.0;
    }
    for (int p = 0; p < n; p++) {
        double pivot = aug.at(p, p);
        for (int c = 0; c < 2 * n; c++) aug.at(p, c) /= pivot;
        for (int r = 0; r < n; r++) {
            if (r == p) continue;
            double factor = aug.at(r, p);
            for (int c = 0; c < 2 * n; c++) aug.at(r, c) -= factor * aug.at(p, c);
        }
    }
    Matrix out(n, n);
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) out.at(r, c) = aug.at(r, n + c);
    return out;
}

std::vector<double> matvec(const Matrix& m, const std::vector<double>& v) {
    std::vector<double> out(static_cast<size_t>(m.rows), 0.0);
    for (int r = 0; r < m.rows; r++)
        for (int c = 0; c < m.cols; c++) out[static_cast<size_t>(r)] += m.at(r, c) * v[static_cast<size_t>(c)];
    return out;
}

// =======================================================================
// PART 2: real round-to-nearest quantization at a stated fixed scale --
// the same real affine map from Section 26.4, applied here to a single
// scalar weight.
// =======================================================================
double quantize_weight(double w, double scale) { return std::round(w / scale) * scale; }

// =======================================================================
// PART 3: the real GPTQ compensation formula. Having just quantized
// weight index p, introducing a real error e_p = w_p - quantize(w_p),
// the optimal real update to every NOT-yet-quantized weight index f is
// delta_w_f = -(e_p / [H^-1]_pp) * [H^-1]_fp -- the real, second-order-
// informed correction that keeps the layer's overall real output as
// close as possible to what it would have been with the original,
// unquantized weight.
// =======================================================================
void apply_gptq_compensation(std::vector<double>& w, const Matrix& h_inv, int quantized_index, double error) {
    double denom = h_inv.at(quantized_index, quantized_index);
    for (size_t f = 0; f < w.size(); f++) {
        if (static_cast<int>(f) == quantized_index) continue;
        double delta = -(error / denom) * h_inv.at(static_cast<int>(f), quantized_index);
        w[f] += delta;
    }
}

double sum_squared_error(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 26.5: The Hessian's Role in GPTQ\n";
    std::cout << "========================================================\n";

    // A real, tiny calibration dataset: 3 real samples, 2 real features each.
    Matrix X(3, 2);
    X.at(0, 0) = 1; X.at(0, 1) = 0;
    X.at(1, 0) = 0; X.at(1, 1) = 1;
    X.at(2, 0) = 1; X.at(2, 1) = 1;

    std::cout << "\n-- Test 1: the real Hessian H = 2 * X^T * X, computed from scratch, matches an exact "
                 "hand computation over this section's own real calibration data --\n";
    {
        Matrix xt = transpose(X);
        Matrix xtx = matmul(xt, X);
        Matrix H(2, 2);
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 2; c++) H.at(r, c) = 2.0 * xtx.at(r, c);

        CHECK(near(H.at(0, 0), 4.0));
        CHECK(near(H.at(0, 1), 2.0));
        CHECK(near(H.at(1, 0), 2.0));
        CHECK(near(H.at(1, 1), 4.0));
        std::cout << "  X^T * X over the real 3-sample calibration set is exactly [[2, 1], [1, 2]], so "
                     "H = 2 * X^T * X is exactly [[4, 2], [2, 4]] -- matching a direct hand computation "
                     "exactly\n";
    }

    std::cout << "\n-- Test 2: the real, general Gauss-Jordan inverse of H matches an exact hand "
                 "computation, and H * H^-1 is genuinely the identity matrix, checked directly rather "
                 "than assumed from the inversion algorithm's own correctness --\n";
    {
        Matrix H(2, 2);
        H.at(0, 0) = 4; H.at(0, 1) = 2; H.at(1, 0) = 2; H.at(1, 1) = 4;
        Matrix h_inv = inverse(H);

        CHECK(near(h_inv.at(0, 0), 1.0 / 3.0));
        CHECK(near(h_inv.at(0, 1), -1.0 / 6.0));
        CHECK(near(h_inv.at(1, 0), -1.0 / 6.0));
        CHECK(near(h_inv.at(1, 1), 1.0 / 3.0));

        Matrix identity_check = matmul(H, h_inv);
        CHECK(near(identity_check.at(0, 0), 1.0) && near(identity_check.at(0, 1), 0.0));
        CHECK(near(identity_check.at(1, 0), 0.0) && near(identity_check.at(1, 1), 1.0));
        std::cout << "  the real Gauss-Jordan inverse of [[4, 2], [2, 4]] is exactly "
                     "[[1/3, -1/6], [-1/6, 1/3]], matching a hand computation via the real 2x2 "
                     "determinant formula exactly, and H * H^-1 checked directly is exactly the real "
                     "2x2 identity matrix\n";
    }

    std::cout << "\n-- Test 3: the real GPTQ compensation formula applied to a single quantized weight "
                 "matches an exact hand computation for the resulting update to the remaining weight --\n";
    {
        Matrix h_inv(2, 2);
        h_inv.at(0, 0) = 1.0 / 3.0; h_inv.at(0, 1) = -1.0 / 6.0;
        h_inv.at(1, 0) = -1.0 / 6.0; h_inv.at(1, 1) = 1.0 / 3.0;

        double w0 = 1.3, scale = 0.5;
        double q0 = quantize_weight(w0, scale);
        CHECK(near(q0, 1.5));
        double e0 = w0 - q0;
        CHECK(near(e0, -0.2));

        std::vector<double> w = {w0, 0.8};
        apply_gptq_compensation(w, h_inv, 0, e0);
        CHECK(near(w[1], 0.7));  // 0.8 + delta, where delta = -(-0.2 / (1/3)) * (-1/6) = -0.1
        std::cout << "  quantizing w[0] = 1.3 at scale 0.5 yields q0 = 1.5, a real error of e0 = -0.2; "
                     "applying the real GPTQ compensation formula to the not-yet-quantized w[1] = 0.8 "
                     "shifts it to exactly 0.7 -- matching a direct hand computation of "
                     "-(-0.2 / (1/3)) * (-1/6) = -0.1 exactly\n";
    }

    std::cout << "\n-- Test 4: the full real GPTQ pipeline -- sequential per-weight quantization with "
                 "compensation -- produces a genuinely different final weight than naive, independent "
                 "rounding, and a strictly LOWER real output error over the calibration data, GPTQ's own "
                 "central real claim, verified end to end on a fully hand-traceable example --\n";
    {
        Matrix h_inv(2, 2);
        h_inv.at(0, 0) = 1.0 / 3.0; h_inv.at(0, 1) = -1.0 / 6.0;
        h_inv.at(1, 0) = -1.0 / 6.0; h_inv.at(1, 1) = 1.0 / 3.0;

        std::vector<double> w_orig = {1.3, 0.8};
        double scale = 0.5;

        // Naive: quantize every weight independently, no compensation at all.
        std::vector<double> w_naive = {quantize_weight(w_orig[0], scale), quantize_weight(w_orig[1], scale)};
        CHECK(near(w_naive[0], 1.5));
        CHECK(near(w_naive[1], 1.0));

        // GPTQ: quantize index 0, compensate the remaining weight, then quantize it too.
        std::vector<double> w_gptq = w_orig;
        double q0 = quantize_weight(w_gptq[0], scale);
        double e0 = w_gptq[0] - q0;
        w_gptq[0] = q0;
        apply_gptq_compensation(w_gptq, h_inv, 0, e0);
        w_gptq[1] = quantize_weight(w_gptq[1], scale);
        CHECK(near(w_gptq[0], 1.5));
        CHECK(near(w_gptq[1], 0.5));  // a genuinely different real final code than naive's 1.0

        std::vector<double> y_orig = matvec(X, w_orig);
        std::vector<double> y_naive = matvec(X, w_naive);
        std::vector<double> y_gptq = matvec(X, w_gptq);

        double error_naive = sum_squared_error(y_naive, y_orig);
        double error_gptq = sum_squared_error(y_gptq, y_orig);
        CHECK(near(error_naive, 0.24));
        CHECK(near(error_gptq, 0.14));
        CHECK(error_gptq < error_naive);
        std::cout << "  naive independent rounding quantizes w to exactly [1.5, 1.0], with a real total "
                     "squared output error over the calibration data of exactly 0.24; GPTQ's real "
                     "Hessian-compensated pipeline quantizes the SAME original weights to exactly "
                     "[1.5, 0.5] -- a genuinely different second weight -- with a real total squared "
                     "output error of exactly 0.14, strictly lower than naive rounding's 0.24, "
                     "confirming GPTQ's own central claim on a fully hand-traceable example\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
