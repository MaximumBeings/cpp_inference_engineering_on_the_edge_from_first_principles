# Appendix B: Practice Quiz

Every chapter in this book already ends with its own self-check questions and worked solutions, testing that chapter's own material in isolation. This appendix is different: it tests whether the ideas actually connect across Parts -- whether you can look at a KV cache eviction problem and recognize the same underlying shape a re-quantization bug has, or say why a diagnosis-incapable medical report schema and a compile-time no-facial-recognition guarantee are solving the same kind of problem in different domains. B.2 asks fourteen conceptual questions, two from each of this book's seven Parts, with answers in B.3. B.4 raises the stakes: three short, genuinely compiled programs, each built around a specific real mistake this book warned about somewhere -- read the code, commit to a prediction, then compile and run it yourself before reading the revealed output.

## B.1 How to Use This Quiz

Answer every question in B.2 in your own words -- out loud, or on paper -- before reading B.3's answer. For B.4, do the same with actual code: read the program, write down what you believe it will print, and only then compile it and compare. A prediction you get wrong tells you exactly which chapter to reread; a prediction you get right by guessing, without being able to explain why, is worth treating the same way. Every one of B.4's programs compiles and runs in seconds -- there is no reason to skip the "compile and run it yourself" half of the exercise, and every reason not to: this book's entire method has been "genuinely compiled, genuinely run, never assumed," and this quiz is not an exception.

## B.2 Conceptual Review Questions

**Part 0 -- Foundations (Chapters 2-3)**

1. Why does `std::mdspan` improve on a raw pointer plus manual index arithmetic for representing a tensor, and what specific GCC-version constraint did this book have to work around for its own multi-dimensional subscript syntax?
2. Why does RoPE (rotary position embedding) encode position via rotation rather than by directly adding a position vector to the embedding, per the composition identity `R(a)^T * R(b) = R(b-a)` derived in Chapter 26.3?

**Part 1 -- Shrinking the Model: Quantization (Chapters 4-7)**

3. Why does affine (scale + zero-point) quantization need a zero-point term at all, rather than always assuming values are symmetric around zero?
4. What property of a memory-mapped GGUF file makes it possible to load a multi-gigabyte model without reading the whole file into RAM up front?

**Part 2 -- Performance on the Metal (Chapters 8-11)**

5. In the roofline model, what does it mean for an operation to be "memory-bound," and why doesn't raising a chip's clock speed alone fix it?
6. Why must this book's SIMD dot-product code have a genuine scalar "tail" loop after its vectorized main loop, rather than only ever using 8-wide (AVX2) or 4-wide (NEON) operations?

**Part 3 -- Serving State: Tokenization and the KV Cache (Chapters 12-14)**

7. Why can't a simple ring buffer alone implement a KV cache eviction policy for multiple concurrent, variable-length sequences?
8. What problem does prefix caching solve for a server handling many requests that share a common system prompt?

**Part 4 -- From Model to Binary (Chapters 15-17)**

9. Chapter 15 found a real cross-layer KV-cache-sharing bug by comparing against an independently-built llama.cpp. Why is comparing against an independent second implementation a stronger check than re-reading your own code more carefully?
10. Why is benchmarking inference speed by wall-clock time alone risky across different machines, and what did this book measure instead in its own locked self-tests?

**Part 5 -- Deploying Vision-Language Intelligence at the Edge (Chapters 18-25)**

11. Chapter 20's medical imaging triage system is deliberately built to be diagnosis-incapable. What does that constraint change about the system's own output schema, compared to a system that WAS allowed to output a diagnosis?
12. Chapter 25's body-worn camera scene-tagging model enforces a no-facial-recognition boundary using a C++20 concept and `static_assert` rather than a runtime check. Why is a compile-time guarantee stronger here than a runtime one?

**Part 6 -- Going Further: Mathematical Foundations and GPU Acceleration (Chapters 26-28)**

13. Why does naive softmax overflow to NaN on large real inputs, and what single algebraic identity (used throughout Chapters 26.2 and 28.1) fixes it without changing the mathematical result?
14. Chapter 28 could not launch its CUDA kernel on a real GPU anywhere in this book's own pipeline. What could a clean `nvcc` compile still prove about that kernel, and what could it NOT prove?

## B.3 Conceptual Review Answers

**1.** `std::mdspan` attaches a tensor's shape (extents) and layout policy directly to the view type, so indexing math is computed in one consistent place rather than re-derived by hand as `row * cols + col` at every call site, where a transposed dimension or an off-by-one is easy to introduce silently. This book's own real constraint: its actual aarch64 target device runs GCC 11.4.0, which predates GCC 12's support for C++23's multi-argument `operator[]` (`view[i, j]`) and ships no native `<mdspan>` header at all -- so every `mdspan`-based file in this book uses a vendored, header-only reference implementation plus an `idx2()`/`idx3()` helper that packages indices into the `std::array` overload GCC 11 does support, a convention established in Chapter 13's own appendix note and reused unchanged through Chapter 28.

**2.** Rotating the query and key vectors by their own absolute positions makes their dot product depend ONLY on the relative distance between them, not on either position alone -- exactly the property attention needs, since two tokens three positions apart near the start of a sequence should relate the same way as two tokens three positions apart near the end. This falls directly out of the rotation-composition identity `R(a)^T * R(b) = R(b-a)`. Adding a position vector directly has no equivalent algebraic guarantee: two tokens' additive position encodings don't combine into a clean function of relative distance when dotted together, so a model using additive position encoding has to learn that structure from data instead of getting it for free from the geometry.

**3.** A purely symmetric (scale-only) quantizer maps real zero to integer zero, which wastes representable range whenever the real distribution being quantized is not centered at zero -- activations after a ReLU, for instance, are all non-negative, so a symmetric int8 range of [-128, 127] would leave the entire negative half of that range completely unused. The zero-point shifts which integer value represents real zero, letting the quantizer's full integer range map onto the ACTUAL span of the data, at the cost of one extra offset to track and add back during dequantization.

**4.** A GGUF file lays its weight tensors out at known, fixed byte offsets recorded in its own header metadata, so `mmap` can map the file's bytes directly into the process's virtual address space without the operating system reading any tensor's actual bytes off disk until that specific memory page is first touched. It is the file's own contiguous, offset-addressable layout that lets the OS defer most of the real I/O work, rather than requiring an explicit "read the whole file into a buffer" step before any single weight can be used.

**5.** An operation is memory-bound when its arithmetic intensity -- FLOPs performed per byte moved from memory -- is low enough that the time spent fetching its operands exceeds the time spent computing on them once they arrive; the roofline model's diagonal bandwidth line, not its flat compute-bound ceiling, is what caps its achievable throughput in that regime. Raising clock speed only raises the height of the compute ceiling; it does nothing to the rate at which bytes can move from memory to the compute units, so a memory-bound kernel stays capped by the same bandwidth-limited diagonal no matter how fast the arithmetic units themselves could go.

**6.** AVX2 processes 8 float32 values per instruction and NEON processes 4, but a real input vector's length is very rarely an exact multiple of either width -- this book's own dot-product files (and Appendix A's own smoke test) deliberately use a 257-element vector specifically because it is not a multiple of 4 or 8, forcing the vectorized main loop to stop early and a plain scalar loop to handle whatever elements remain. Skipping that tail loop would either silently drop the trailing elements from the result entirely or read past the end of the array trying to force one more full-width SIMD load.

**7.** A ring buffer's eviction order is purely positional: it always overwrites the oldest slot next, with no way to distinguish "this sequence is still actively being generated" from "this sequence finished ten steps ago and is safe to evict." Multiple concurrent sequences of different, changing lengths need eviction decisions based on each sequence's own actual state, which requires the richer per-sequence bookkeeping Chapter 13's own KV cache manager tracks, not a single shared circular position counter.

**8.** Without prefix caching, every request -- even ones sharing an identical, possibly-long system prompt with many other requests -- must recompute that entire shared prefix's KV cache from scratch, wasting compute recomputing values a previous request already produced. Prefix caching recognizes when a new request's opening tokens exactly match an already-cached sequence's tokens and reuses those existing KV cache entries directly, paying the shared prefix's compute cost exactly once no matter how many later requests go on to use it.

**9.** Re-reading your own code mainly re-exercises the same mental model that produced the bug in the first place -- if you believed a RoPE pairing convention was correct when you wrote it, rereading the identical code tends to confirm that same belief rather than surface the assumption that's actually wrong. An independently built second implementation encodes a different set of assumptions arrived at separately, so a genuine numerical disagreement between the two systems is real signal that at least one of them has an actual bug, in a way no amount of rereading a single implementation can substitute for -- which is exactly how Chapter 15 found four separate real bugs against an independently-built llama.cpp.

**10.** Wall-clock time is entangled with everything about the specific machine it was measured on -- clock frequency, thermal throttling, background load, cache sizes -- so a timing number from one run on one machine says little about whether an approach is genuinely more efficient in a way that would hold on different hardware, and it cannot be locked into an exactly-reproducible self-test the way this book's own build-verify-lock discipline requires. This book measured real, machine-independent quantities instead: exact multiply-accumulate counts (Chapter 28.2), exact peak-memory byte counts read from actual allocation sizes, and exact padding-waste and utilization counts (Chapter 27.1) -- numbers reproducible bit-for-bit on any machine that also explain WHY one approach is faster, not just that it measured faster once.

**11.** A system that could output a diagnosis would need some kind of diagnosis field, or a confidence-over-conditions structure, somewhere in its result type. Chapter 20 instead defines a report schema that structurally has no such field at all -- only observations, flagged regions, and a routing decision into a human radiologist's worklist -- so there is no diagnosis-shaped value anywhere in the type for a caller to misread as one. The constraint is enforced by what the schema CAN represent, not by a policy note asking a caller not to over-interpret a field that could otherwise be read as a diagnosis.

**12.** A runtime check has to actually execute on every code path to catch a violation, so a new code path added later -- one nobody thought to add the check to -- can silently reintroduce exactly the behavior the check was meant to prevent, discovered only if that path happens to run during testing. Chapter 25's `HasIdentityField` concept plus `static_assert` instead makes adding an identity field to the scene-tagging model's data type a COMPILE failure everywhere that type is used, so no code path can exist at all where the violation could occur -- the guarantee holds by construction rather than by every future author remembering to preserve it.

**13.** `exp()` of a large real input overflows a float's or double's finite range and becomes infinity well before the division by the sum happens, and infinity divided by an infinity-containing sum evaluates to NaN, silently corrupting the entire softmax output rather than just its largest entry. The shift-invariant identity `softmax(x) == softmax(x - max(x))` fixes this WITHOUT changing the mathematical result: subtracting the input's own maximum guarantees the largest shifted value is exactly 0 (so `exp(0) = 1`, never overflowing) while every other shifted value is negative (so its `exp()` is a safe fraction below 1), and the algebra guarantees the final normalized probabilities come out identical to what the numerically-impossible unshifted computation would have produced had it not overflowed.

**14.** A clean compile across three real Jetson-class architectures proves the kernel is syntactically valid CUDA C++ and that the compiler could generate real device code implementing its stated logic for each target -- a genuine, meaningful fact about the kernel's correctness at the SOURCE level. It proves nothing about the kernel's actual RUNTIME behavior: a race condition in the shared-memory tile load, a tile-boundary off-by-one, or a genuine numerical error in the online-softmax update would all compile without any error, since none of them are visible from source code alone -- only actually launching the kernel on real hardware and comparing its output against the CPU golden reference, which this book's own pipeline was honest about never having a GPU available to do, could close that gap.

## B.4 Predict-the-Output Challenges

### Challenge 1: The Softmax Overflow Trap

Chapter 26.2 warned that naive softmax (`exp(x_i) / sum(exp(x_j))`) silently overflows to NaN on large real inputs, and fixed it with the shift-invariant identity `softmax(x) == softmax(x - max(x))`. Before compiling and running the program below, predict: for the input `{1000.0, 1001.0, 1002.0}`, does the naive version produce NaN, and does the shifted version still produce the mathematically correct probabilities?

```cpp
// Appendix B, Challenge 1 -- Chapter 26.2 warned that naive softmax
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 challenge1_softmax_overflow.cpp -o challenge1_softmax_overflow
./challenge1_softmax_overflow
```

**Revealed output:**

```text
input: [1000, 1001, 1002]

naive softmax   : [nan, nan, nan]
shifted softmax : [0.0900306, 0.244728, 0.665241]

naive softmax produced NaN: YES
shifted softmax sums to 1.0: YES (sum = 1)
```

### Challenge 2: Reduction Order and Float32 Non-Associativity

Chapter 27.4 warned that float32 addition is NOT associative: summing the same values in a different order can produce a genuinely different result, not just a hypothetical one. Before compiling and running the program below, predict: does summing one large value plus eight small values give the same float32 result forwards (large value first) as backwards (large value last)?

```cpp
// Appendix B, Challenge 2 -- Chapter 27.4 warned that float32 addition is
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 challenge2_reduction_order.cpp -o challenge2_reduction_order
./challenge2_reduction_order
```

**Revealed output:**

```text
values: [100000.0, then eight copies of 0.3]

forward sum  (100000.0 first): 100002.375000
backward sum (100000.0 last):  100002.398438

forward == backward: NO
difference: 0.023438
```

### Challenge 3: The Re-Quantization Trap

Chapter 26.4 proved that affine re-quantization is NOT associative: quantizing at a fine scale, then re-quantizing that already-quantized value at a coarser scale, can give a different final dequantized value than quantizing the ORIGINAL value directly at the coarse scale. Before compiling and running the program below, predict: for the value `0.246`, fine scale `0.07`, and coarse scale `0.5`, do "fine-then-coarse" and "direct-coarse" produce the same dequantized result?

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 challenge3_requant_nonassoc.cpp -o challenge3_requant_nonassoc
./challenge3_requant_nonassoc
```

**Revealed output:**

```text
original value: 0.2460
fine scale: 0.0700, coarse scale: 0.5000

direct-coarse:    quantize(0.2460, scale=0.5000) = 0 -> dequantize = 0.0000
fine-then-coarse: quantize(0.2460, scale=0.0700) = 4 -> dequantize = 0.2800 -> quantize(0.2800, scale=0.5000) = 1 -> dequantize = 0.5000

direct-coarse == fine-then-coarse: NO
difference: 0.5000
```

## Appendix Summary

B.2 and B.3 asked whether this book's individual ideas actually connect across Parts -- whether a reader can recognize the same underlying shape (a rounding boundary crossed twice, a guarantee enforced by construction rather than by policy, a benchmark substituted with something exactly reproducible) recurring in problems that look nothing alike on the surface, from tensor foundations all the way through GPU kernels. B.4 made three of the book's most consequential real mistakes concrete and checkable: softmax silently overflowing to NaN on realistic logit magnitudes, float32 addition genuinely depending on evaluation order, and re-quantization losing information in a way that a direct quantization at the same final scale does not. All three compile and run in seconds -- if a prediction did not match the revealed output, that mismatch is worth chasing back to the chapter that first built the idea, which this appendix has named at every step along the way.

## Where We Go Next

Appendix C consolidates the decision trees scattered across individual chapters -- when to reach for blockwise vs. TurboQuant quantization, when a mutex is enough vs. when a lock-free structure is warranted, which KV cache eviction policy fits which serving pattern -- into a single reference. Appendix D restates this book's own running discipline around timing, determinism, and what a locked self-test contract actually promises. Appendix E is a Rosetta Stone for readers arriving fluent in the Python inference ecosystem. Appendix F catalogs common failure modes -- NaN propagation, false sharing, floating-point drift, and alignment bugs -- much of it drawing directly on material this book already built in Chapter 27.4 and Chapter 10.
