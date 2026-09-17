// Appendix B, Challenge 1 -- Chapter 30.2 warned that naive softmax
// (exp(x_i) / sum(exp(x_j))) silently overflows to NaN on large real
// inputs, and fixed it with the shift-invariant identity
// softmax(x) == softmax(x - max(x)). Before compiling and running this
// file, predict: for the input {1000.0, 1001.0, 1002.0}, does the naive
// version produce NaN, and does the shifted version still produce the
// mathematically correct probabilities?
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 challenge1_softmax_overflow.cpp -o challenge1_softmax_overflow
// Run:     ./challenge1_softmax_overflow

#include <cmath>
#include <cstdio>
#include <vector>

std::vector<double> naive_softmax(const std::vector<double>& x) {
    std::vector<double> out(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i]);
        sum += out[i];
    }
    for (auto& v : out) v /= sum;
    return out;
}

std::vector<double> shifted_softmax(const std::vector<double>& x) {
    double m = x[0];
    for (double v : x) if (v > m) m = v;
    std::vector<double> out(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::exp(x[i] - m);
        sum += out[i];
    }
    for (auto& v : out) v /= sum;
    return out;
}

void print_vec(const char* label, const std::vector<double>& v) {
    // NaN is printed uniformly as "nan" regardless of its sign bit: real glibc/libm
    // implementations disagree on whether a NaN produced by 0.0/0.0 carries a set sign
    // bit (x86_64 glibc prints "-nan", aarch64 prints "nan" for the identical computation),
    // and that sign bit carries no mathematical meaning here -- printing it directly would
    // make this file's own locked output architecture-dependent for no real reason.
    printf("%s: [", label);
    for (size_t i = 0; i < v.size(); ++i) {
        if (std::isnan(v[i])) printf("nan%s", i + 1 < v.size() ? ", " : "");
        else printf("%g%s", v[i], i + 1 < v.size() ? ", " : "");
    }
    printf("]\n");
}

int main() {
    std::vector<double> x = {1000.0, 1001.0, 1002.0};
    printf("input: [1000, 1001, 1002]\n\n");

    auto naive = naive_softmax(x);
    print_vec("naive softmax   ", naive);

    auto shifted = shifted_softmax(x);
    print_vec("shifted softmax ", shifted);

    bool naive_has_nan = false;
    for (double v : naive) if (std::isnan(v)) naive_has_nan = true;

    printf("\nnaive softmax produced NaN: %s\n", naive_has_nan ? "YES" : "NO");
    printf("shifted softmax sums to 1.0: %s (sum = %g)\n",
           std::fabs((shifted[0] + shifted[1] + shifted[2]) - 1.0) < 1e-9 ? "YES" : "NO",
           shifted[0] + shifted[1] + shifted[2]);

    return 0;
}
