// Appendix B, Challenge 2 -- Chapter 31.4 warned that float32 addition is
// NOT associative: summing the same values in a different order can
// produce a genuinely different result, not just a hypothetical one.
// Before compiling and running this file, predict: does summing one very
// large value plus eight small values give the same float32 result
// forwards (large first) as backwards (large last)?
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 challenge2_reduction_order.cpp -o challenge2_reduction_order
// Run:     ./challenge2_reduction_order

#include <cstdio>
#include <vector>

int main() {
    std::vector<float> values;
    values.push_back(100000.0f);
    for (int i = 0; i < 8; ++i) values.push_back(0.3f);

    printf("values: [100000.0, then eight copies of 0.3]\n\n");

    float forward = 0.0f;
    for (size_t i = 0; i < values.size(); ++i) forward += values[i];

    float backward = 0.0f;
    for (size_t i = values.size(); i-- > 0; ) backward += values[i];

    printf("forward sum  (100000.0 first): %.6f\n", forward);
    printf("backward sum (100000.0 last):  %.6f\n", backward);
    printf("\nforward == backward: %s\n", forward == backward ? "YES" : "NO");
    if (forward != backward) {
        printf("difference: %.6f\n", backward - forward);
    }

    return 0;
}
