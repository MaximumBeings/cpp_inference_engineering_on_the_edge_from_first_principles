// Appendix B, Challenge 3 -- Chapter 26.4 proved that affine
// re-quantization is NOT associative: quantizing at a fine scale, then
// re-quantizing that already-quantized value at a coarser scale, can give
// a different final dequantized value than quantizing the ORIGINAL value
// directly at the coarse scale. Before compiling and running this file,
// predict: for the value 0.246, fine scale 0.07, and coarse scale 0.5, do
// "fine-then-coarse" and "direct-coarse" produce the same dequantized
// result?
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 challenge3_requant_nonassoc.cpp -o challenge3_requant_nonassoc
// Run:     ./challenge3_requant_nonassoc

#include <cmath>
#include <cstdio>

// A minimal real affine quantizer: round-to-nearest, zero-point-free
// (symmetric), matching Chapter 26.4's own scale/2 error-bound analysis.
int quantize(double value, double scale) {
    return static_cast<int>(std::lround(value / scale));
}

double dequantize(int q, double scale) {
    return q * scale;
}

int main() {
    double original = 0.246;
    double fine_scale = 0.07;
    double coarse_scale = 0.5;

    printf("original value: %.4f\n", original);
    printf("fine scale: %.4f, coarse scale: %.4f\n\n", fine_scale, coarse_scale);

    // Path A: quantize directly at the coarse scale.
    int direct_q = quantize(original, coarse_scale);
    double direct_result = dequantize(direct_q, coarse_scale);
    printf("direct-coarse:    quantize(%.4f, scale=%.4f) = %d -> dequantize = %.4f\n",
           original, coarse_scale, direct_q, direct_result);

    // Path B: quantize at the fine scale first, dequantize back to a real
    // value, THEN quantize that already-lossy value at the coarse scale.
    int fine_q = quantize(original, fine_scale);
    double fine_dequant = dequantize(fine_q, fine_scale);
    int fine_then_coarse_q = quantize(fine_dequant, coarse_scale);
    double fine_then_coarse_result = dequantize(fine_then_coarse_q, coarse_scale);
    printf("fine-then-coarse: quantize(%.4f, scale=%.4f) = %d -> dequantize = %.4f -> "
           "quantize(%.4f, scale=%.4f) = %d -> dequantize = %.4f\n",
           original, fine_scale, fine_q, fine_dequant,
           fine_dequant, coarse_scale, fine_then_coarse_q, fine_then_coarse_result);

    printf("\ndirect-coarse == fine-then-coarse: %s\n",
           direct_result == fine_then_coarse_result ? "YES" : "NO");
    if (direct_result != fine_then_coarse_result) {
        printf("difference: %.4f\n", fine_then_coarse_result - direct_result);
    }

    return 0;
}
