# Chapter 3: The Computational Graph -- RMSNorm, SwiGLU, RoPE, and Attention

**What you will understand by the end of this chapter:**

- How RMSNorm, SwiGLU, and RoPE each replace an older Transformer component with a cheaper or more capable alternative, and the exact arithmetic each one performs, worked by hand before it is ever compiled.
- How Grouped-Query Attention groups many query heads onto a smaller number of key/value heads, why the KV cache it reads from is a genuinely three-dimensional tensor, and how a numerically stable softmax keeps that computation finite regardless of how large the raw attention scores get.
- How a static compute graph turns "record once, replay per token" into a real performance and memory-planning strategy, and how mmap turns a multi-gigabyte weight file into a set of zero-copy views instead of a heap-sized copy of the whole file.

**What you need to know first:**

- Chapter 2's vocabulary: `std::mdspan`/`std::submdspan` for multi-dimensional views, the bump-pointer arena, and `std::span` as the plain, shape-less alternative for genuinely one-dimensional data.
- Ordinary C++ templates, `std::vector`, RAII, and enough linear algebra to be comfortable with a dot product and a 2D rotation matrix.

---

Chapter 2 gave this book the tools to describe and allocate tensors; this chapter gives it something to compute with them. Every operation here is a genuine piece of a modern transformer block, in the order it actually runs: RMSNorm before attention and before the feed-forward network, RoPE applied to Q and K right after their projections, Grouped-Query Attention reading a KV cache that grows by one position per generated token, and a SwiGLU feed-forward network that alone accounts for roughly two-thirds of a forward pass's total FLOPs. The chapter closes by assembling these into a static compute graph — recorded once, replayed cheaply every token, with a liveness-aware memory plan — and by loading the weights that graph reads through `mmap`, so that a multi-gigabyte checkpoint never has to be copied wholesale into RAM before the first token can generate.

## 3.1 RMSNorm: Cheap, Memory-Bound Normalization

### Intuition

Deep networks need their activations kept in a stable numeric range from layer to layer, or training and inference both become unreliable. The original Transformer used LayerNorm (subtract the mean, divide by the standard deviation); Llama, Mistral, and Gemma all use RMSNorm instead, which skips the mean entirely and normalizes only by the root mean square — one reduction pass instead of two, for empirically comparable model quality.

### The Concept, In Detail

For an input vector `x` of length `d` and a learned per-dimension weight `w`, RMSNorm computes `rms = sqrt(mean(x[i]^2) + epsilon)` and then `out[i] = (x[i] / rms) * w[i]`, where `epsilon` (typically `1e-6`) exists purely to keep the division finite when `x` is the zero vector. The kernel touches roughly `3d` FLOPs (square each element, sum them, one sqrt and one reciprocal, then two more multiplies per element) against `3d * 4` bytes of memory traffic (read `x`, read `w`, write the output) — an arithmetic intensity far below what a modern CPU can sustain per byte of memory bandwidth, which is why RMSNorm is memory-bandwidth bound rather than compute bound: SIMD widens the sum-of-squares loop's throughput, but it cannot make the kernel faster than the memory system can feed it. A useful correctness property worth checking directly rather than trusting by assumption: RMSNorm is scale-invariant. Multiplying the entire input by any positive constant leaves the output completely unchanged, because that same constant appears in, and cancels out of, the RMS computed from it.

### Code and Verification

```cpp
// Chapter 3.1 -- RMSNorm normalizes an activation vector by its root
// mean square, then rescales by a learned per-dimension weight. It
// replaced LayerNorm in Llama, Mistral, and Gemma because it drops
// LayerNorm's mean-subtraction pass entirely: one reduction (sum of
// squares) instead of two (mean, then variance), for empirically
// comparable quality at roughly two-thirds the memory traffic.
//
// This kernel is heavily memory-bandwidth bound, not compute bound: it
// touches ~3d FLOPs against 3d*4 bytes of traffic (read x, read
// weights, write output), an arithmetic intensity around 0.25
// FLOPs/byte on a machine that can typically sustain 10-20. SIMD
// speeds up the sum-of-squares loop's throughput; it cannot fix a
// kernel that is waiting on memory, not the ALU.

#include <cmath>     // std::sqrt, std::fabs, std::isfinite, std::sin
#include <span>
#include <vector>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// out, x, and weights all have the same length d. `out` may alias `x`
// for an in-place normalize; in the full engine `out` is a scratch
// buffer carved from Chapter 2's arena.
void rms_norm(std::span<float> out, std::span<const float> x,
              std::span<const float> weights, float epsilon = 1e-6f) {
    const size_t d = x.size();

    // Pass 1: the only reduction RMSNorm needs -- sum of squares.
    float sum_sq = 0.0f;
    for (size_t i = 0; i < d; ++i) sum_sq += x[i] * x[i];

    // Reciprocal RMS: computing 1/sqrt(...) once and multiplying by it
    // is cheaper than computing sqrt(...) and dividing by it in the
    // loop below, since a divide is markedly slower than a multiply on
    // essentially every mainstream CPU. epsilon keeps this finite even
    // when x is the all-zero vector.
    float rms_inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(d) + epsilon);

    // Pass 2: every output element is independent of every other one --
    // no cross-iteration dependency, so this loop vectorizes freely.
    for (size_t i = 0; i < d; ++i) out[i] = (x[i] * rms_inv) * weights[i];
}

float compute_rms(std::span<const float> v) {
    float ss = 0.0f;
    for (float x : v) ss += x * x;
    return std::sqrt(ss / static_cast<float>(v.size()));
}

int main() {
    // --- Unit weights: output RMS must land at ~1.0 by construction ---
    {
        std::vector<float> x = {2.0f, -1.0f, 0.5f, 3.0f};
        std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> out(4);
        rms_norm(out, x, w);
        // sum_sq = 4 + 1 + 0.25 + 9 = 14.25; rms_inv = 1/sqrt(14.25/4 + 1e-6) ~ 0.52981
        CHECK_NEAR(out[0], 1.0596f, 0.001f);
        CHECK_NEAR(out[1], -0.5298f, 0.001f);
        CHECK_NEAR(out[2], 0.2649f, 0.001f);
        CHECK_NEAR(out[3], 1.5894f, 0.001f);
        CHECK_NEAR(compute_rms(out), 1.0f, 0.001f);
    }

    // --- Non-unit (learned) weights: each dimension scales independently ---
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> w = {2.0f, 0.5f, 1.0f, 3.0f};
        std::vector<float> out(4);
        rms_norm(out, x, w);
        // rms_inv = 1/sqrt(30/4) ~ 0.36515; out[3] = 4 * 0.36515 * 3 ~ 4.3818
        CHECK_NEAR(out[3], 4.3818f, 0.005f);
    }

    // --- All-zero input: epsilon must keep this finite, not NaN ---
    {
        std::vector<float> x(8, 0.0f), w(8, 1.0f), out(8);
        rms_norm(out, x, w);
        bool all_finite = true;
        for (float v : out) if (!std::isfinite(v)) all_finite = false;
        CHECK(all_finite);
        CHECK(out[0] == 0.0f);
    }

    // --- Scale invariance: rms_norm(a*x) == rms_norm(x) for any a>0 ---
    // RMS itself scales by `a`, so `a` cancels out of the normalize step --
    // this is the entire point of the operation: it strips global scale
    // and keeps everything else.
    {
        std::vector<float> x = {0.5f, 1.2f, -0.3f, 0.8f};
        std::vector<float> x_scaled = {50.0f, 120.0f, -30.0f, 80.0f};   // 100x
        std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> out1(4), out2(4);
        rms_norm(out1, x, w);
        rms_norm(out2, x_scaled, w);
        for (size_t i = 0; i < 4; ++i) CHECK_NEAR(out1[i], out2[i], 0.0001f);
    }

    // --- Realistic dimension (d=64): output RMS still lands at 1.0 ---
    float rms64 = 0.0f;
    {
        constexpr size_t D = 64;
        std::vector<float> x(D), w(D, 1.0f), out(D);
        for (size_t i = 0; i < D; ++i) x[i] = std::sin(static_cast<float>(i));
        rms_norm(out, x, w);
        rms64 = compute_rms(out);
        CHECK_NEAR(rms64, 1.0f, 0.001f);
    }

    std::vector<float> x0 = {2.0f, -1.0f, 0.5f, 3.0f}, w0(4, 1.0f), out0(4);
    rms_norm(out0, x0, w0);
    std::cout << "rms_norm([2, -1, 0.5, 3], w=[1,1,1,1]) = ["
              << out0[0] << ", " << out0[1] << ", " << out0[2] << ", " << out0[3] << "]\n";
    std::cout << "RMS of that output (should be 1.0): " << compute_rms(out0) << "\n";
    std::cout << "RMS of a d=64 output (should also be 1.0): " << rms64 << "\n";
    std::cout << "rms_norm(x) == rms_norm(100*x) for the same weights: yes\n";
    std::cout << "all outputs finite on all-zero input (epsilon guard): yes\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_rmsnorm.cpp -o 01_rmsnorm
./01_rmsnorm
```

**Sample input:** the hand-worked `x = [2.0, -1.0, 0.5, 3.0]`, `w = [1,1,1,1]` example from this section, plus a second vector with non-unit learned weights, an all-zero vector (the epsilon stability case), a 100x-scaled copy of the first vector (the scale-invariance case), and a realistic `d=64` vector.

**Sample output:**

```text
rms_norm([2, -1, 0.5, 3], w=[1,1,1,1]) = [1.05963, -0.529813, 0.264906, 1.58944]
RMS of that output (should be 1.0): 1
RMS of a d=64 output (should also be 1.0): 0.999999
rms_norm(x) == rms_norm(100*x) for the same weights: yes
all outputs finite on all-zero input (epsilon guard): yes

13/13 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Forgetting epsilon, or placing it outside the square root"
    Without `epsilon`, an all-zero (or numerically tiny) input vector produces `sqrt(0) = 0`, and the very next step divides by that zero, turning the entire output into `NaN` or `Inf` — not a rare edge case in practice, since a padded token or a freshly-reset scratch buffer can genuinely be all zeros. Just as easy to get subtly wrong is where `epsilon` gets added: `sqrt(mean_sq) + epsilon` still leaves `mean_sq = 0` unprotected at the moment the square root is taken, while `sqrt(mean_sq + epsilon)` (this section's version) protects the vulnerable operation directly. The difference sounds cosmetic until the input actually is all zeros, at which point one version returns a small, well-defined number and the other returns `NaN`.

## 3.2 SwiGLU: The Gated Feed-Forward Network

### Intuition

The feed-forward network is where a transformer stores most of its factual knowledge, and its hidden dimension is typically three to four times the model's own hidden size, which is why it dominates a forward pass's FLOPs budget (roughly two-thirds of it end to end). The original Transformer used two matrices and a ReLU; Llama-style models use three matrices and a gated activation called SwiGLU, trading one extra matrix multiply for richer gradients during training and, this section argues, a real difference in which neurons stay trainable at all.

### The Concept, In Detail

`FFN(x) = (SiLU(x . W_gate) elementwise* (x . W_up)) . W_down`. `SiLU(x) = x * sigmoid(x)` is smooth everywhere, including for negative `x`: `SiLU(-1) ~ -0.269`, small but never zero, and never zero-gradient — unlike `ReLU(-1) = 0`, which hard-clamps both the value and the gradient flowing back through it. The "up" projection `W_up` is not an activation at all; it is a second, ungated linear projection whose output the gated `SiLU(x . W_gate)` term multiplies elementwise, which is what makes this a GATED unit rather than merely a different activation function — the gate learns which dimensions of the up-projected signal to let through, continuously, rather than the FFN applying the same fixed nonlinearity to every dimension independently.

### Code and Verification

```cpp
// Chapter 3.2 -- the feed-forward block is where a transformer stores
// most of its factual knowledge, and it dominates the FLOPs budget of
// a forward pass (roughly two-thirds of it, since the FFN's hidden
// dimension is typically 3-4x the model's hidden size). The original
// Transformer used two matrices and a ReLU; Llama-style models use
// three matrices and a gated activation called SwiGLU:
//
//   FFN(x) = (SiLU(x * W_gate) (elementwise*) (x * W_up)) * W_down
//
// The third matrix (the "up" projection) is a learned gate: instead of
// hard-clamping negative values to zero the way ReLU does, SiLU softly
// suppresses them while keeping a nonzero gradient everywhere -- fewer
// permanently "dead" neurons during training.

#include <cmath>
#include <span>
#include <vector>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// SiLU(x) = x * sigmoid(x). Unlike ReLU(-1) = 0 (a dead gradient),
// SiLU(-1) = -1 * sigmoid(-1) ~ -0.269 -- small, but never zero, and
// never zero-gradient.
inline float silu(float x) { return x * (1.0f / (1.0f + std::exp(-x))); }
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// gate_proj = x * W_gate, up_proj = x * W_up (both computed by matmuls
// outside this function). This kernel is the elementwise part in
// between: SiLU(gate) times up, feeding the down projection next.
void swiglu(std::span<const float> gate_proj, std::span<const float> up_proj,
            std::span<float> out) {
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = silu(gate_proj[i]) * up_proj[i];
}

// Naive O(in*out) matmul: out[j] = sum_i x[i] * W[j*in_dim + i], W
// stored row-major. Production code replaces this with a BLAS call or
// a quantized kernel (Part 1); the shape contract stays identical.
void matmul(std::span<float> out, std::span<const float> x,
            std::span<const float> W, size_t in_dim, size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < in_dim; ++i) sum += x[i] * W[j * in_dim + i];
        out[j] = sum;
    }
}

void swiglu_ffn(std::span<float> out, std::span<const float> x,
                 std::span<const float> W_gate, std::span<const float> W_up,
                 std::span<const float> W_down, size_t dim, size_t d_ff) {
    std::vector<float> gate_proj(d_ff), up_proj(d_ff), hidden(d_ff);
    matmul(gate_proj, x, W_gate, dim, d_ff);
    matmul(up_proj, x, W_up, dim, d_ff);
    swiglu(gate_proj, up_proj, hidden);
    matmul(out, hidden, W_down, d_ff, dim);
}

int main() {
    // --- SiLU properties, checked against the closed-form sigmoid ---
    CHECK_NEAR(silu(0.0f), 0.0f, 1e-6f);
    CHECK_NEAR(silu(1.0f), 0.7311f, 0.001f);
    CHECK_NEAR(silu(-1.0f), -0.2689f, 0.001f);   // negative, never hard-zeroed
    CHECK(silu(10.0f) > 9.99f);                   // saturates toward x for large x
    CHECK(silu(-1.0f) != relu(-1.0f));             // the actual point of this section

    // --- Kernel output, traced by hand ---
    {
        std::vector<float> gate = {0.8f, -1.2f, 0.3f, 1.5f};
        std::vector<float> up = {0.5f, 0.9f, -0.4f, 1.1f};
        std::vector<float> out(4);
        swiglu(gate, up, out);
        CHECK_NEAR(out[0], 0.2760f, 0.001f);
        CHECK_NEAR(out[1], -0.2500f, 0.001f);
        CHECK_NEAR(out[2], -0.0689f, 0.001f);
        CHECK_NEAR(out[3], 1.3490f, 0.001f);
    }

    // --- Full FFN round-trip with identity weight matrices ---
    // W_gate = W_up = W_down = I collapses the whole pipeline to
    // FFN(x) = SiLU(x) elementwise* x, letting the full 3-matmul
    // pipeline be checked against a closed form with no weight
    // randomness involved.
    float out3_0 = 0.0f;
    {
        constexpr size_t D = 4;
        std::vector<float> I(D * D, 0.0f);
        for (size_t i = 0; i < D; ++i) I[i * D + i] = 1.0f;
        std::vector<float> x = {1.0f, -0.5f, 2.0f, 0.0f};
        std::vector<float> out(D);
        swiglu_ffn(out, x, I, I, I, D, D);
        for (size_t i = 0; i < D; ++i)
            CHECK_NEAR(out[i], silu(x[i]) * x[i], 0.001f);
        out3_0 = out[0];
    }

    std::cout << "silu(0)=" << silu(0.0f) << " silu(1)=" << silu(1.0f)
              << " silu(-1)=" << silu(-1.0f) << " relu(-1)=" << relu(-1.0f) << "\n";
    std::cout << "swiglu([0.8,-1.2,0.3,1.5], [0.5,0.9,-0.4,1.1]) = ["
              << silu(0.8f) * 0.5f << ", " << silu(-1.2f) * 0.9f << ", "
              << silu(0.3f) * -0.4f << ", " << silu(1.5f) * 1.1f << "]\n";
    std::cout << "identity-weight FFN([1,-0.5,2,0])[0] = " << out3_0
              << " (expected silu(1)*1 = " << silu(1.0f) << ")\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_swiglu.cpp -o 02_swiglu
./02_swiglu
```

**Sample input:** `SiLU` evaluated at several points to check it against ReLU directly; a hand-traced `gate_proj`/`up_proj` pair from this section's own worked example; and a full 3-matmul FFN with every weight matrix set to the identity, collapsing the whole pipeline to a closed form (`SiLU(x) elementwise* x`) that can be checked without any weight randomness involved.

**Sample output:**

```text
silu(0)=0 silu(1)=0.731059 silu(-1)=-0.268941 relu(-1)=0
swiglu([0.8,-1.2,0.3,1.5], [0.5,0.9,-0.4,1.1]) = [0.27599, -0.249993, -0.0689331, 1.349]
identity-weight FFN([1,-0.5,2,0])[0] = 0.731059 (expected silu(1)*1 = 0.731059)

13/13 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Treating SwiGLU as 'SiLU instead of ReLU' and stopping there"
    Swapping only the activation function (`ReLU(x . W1) . W2` becomes `SiLU(x . W1) . W2`) is a different, smaller change than SwiGLU actually is — it misses the THIRD matrix entirely. SwiGLU's defining property is the elementwise gate: one projection's activated output multiplies a second, separate projection's raw output, which is why the down projection's input dimension is `d_ff` fed from a gating multiply, not directly from a single activated matmul. A "SwiGLU" implementation with only two weight matrices has quietly become a differently-activated ordinary FFN, not the gated unit this section describes.

## 3.3 RoPE: Encoding Position Through Rotation

### Intuition

The raw dot product inside self-attention is position-blind: `Q . K` gives the identical score whether the two tokens are adjacent or a thousand positions apart, so a transformer needs some separate mechanism to inject position. Rotary Positional Embeddings do this not by adding a position vector to the embedding, but by rotating the Q and K vectors themselves by an angle that grows with position — and the useful algebraic fact that falls out of choosing a ROTATION specifically is that the rotated dot product depends only on the difference between the two positions, never on either position by itself.

### The Concept, In Detail

RoPE treats a `head_dim`-length vector as `head_dim / 2` independent 2D pairs, and rotates pair `k` by angle `m * theta_k`, where `m` is the token's position and `theta_k = 1 / base^(2k / head_dim)` is a frequency that shrinks as `k` grows. Low-`k` pairs rotate quickly and distinguish nearby positions; high-`k` pairs rotate slowly and distinguish only coarse, long-range position — a fast second hand and a slow hour hand sharing one clock face. Rotating `Q` at position `m` and `K` at position `n` and then taking their dot product is algebraically equivalent to rotating `Q` by `(m - n)` and leaving `K` unrotated, which is exactly why this section's fourth test constructs two entirely different absolute positions with the same relative gap and checks that the resulting dot products land on the identical number.

### Code and Verification

```cpp
// Chapter 3.3 -- self-attention's dot product Q.K is, by itself,
// position-blind: it gives the same score whether two tokens are
// adjacent or a thousand positions apart. RoPE (Rotary Positional
// Embeddings) fixes this not by adding a position vector, but by
// ROTATING each Q and K vector by an angle proportional to its
// position, treating the vector as d/2 independent 2D pairs. The key
// algebraic fact that makes this useful: rotating Q by angle m*theta
// and K by angle n*theta leaves their dot product depending only on
// the DIFFERENCE (m-n)*theta, never on m or n individually -- the
// model sees relative distance, not absolute position, for free.

#include <cmath>
#include <span>
#include <vector>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// Precomputed cos/sin tables: cos(m*theta_k) and sin(m*theta_k) for
// every (position, pair) combination, computed once at model load.
// theta_k = 1 / base^(2k/head_dim) decreases with k, so low-k pairs
// rotate fast (encode fine, local position) and high-k pairs rotate
// slowly (encode coarse, long-range position) -- a second hand and an
// hour hand on the same clock face.
struct RoPETables {
    std::vector<float> cos_vals, sin_vals;   // both [seq_len * half_dim]
    int half_dim;

    RoPETables(int seq_len, int head_dim, float base = 10000.0f)
        : half_dim(head_dim / 2) {
        cos_vals.resize(seq_len * half_dim);
        sin_vals.resize(seq_len * half_dim);
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int k = 0; k < half_dim; ++k) {
                float theta = 1.0f / std::pow(base, (2.0f * k) / head_dim);
                float angle = static_cast<float>(pos) * theta;
                cos_vals[pos * half_dim + k] = std::cos(angle);
                sin_vals[pos * half_dim + k] = std::sin(angle);
            }
        }
    }
    float cos_at(int pos, int k) const { return cos_vals[pos * half_dim + k]; }
    float sin_at(int pos, int k) const { return sin_vals[pos * half_dim + k]; }
};

// Rotates `vec` in place, pair by pair: (vec[2k], vec[2k+1]) is a 2D
// vector, rotated by the standard rotation matrix at angle m*theta_k.
void apply_rope(std::span<float> vec, int pos, const RoPETables& tables) {
    const int half_dim = tables.half_dim;
    for (int k = 0; k < half_dim; ++k) {
        float x1 = vec[2 * k], x2 = vec[2 * k + 1];
        float c = tables.cos_at(pos, k), s = tables.sin_at(pos, k);
        vec[2 * k] = x1 * c - x2 * s;
        vec[2 * k + 1] = x1 * s + x2 * c;
    }
}

float dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}
float norm(std::span<const float> v) { return std::sqrt(dot(v, v)); }

int main() {
    RoPETables tables(16, 4, 10000.0f);   // max 16 positions, head_dim=4 (2 pairs)

    // --- Hand-traced example: d=4, position=3 ---
    std::vector<float> traced = {1.0f, 0.5f, -0.3f, 0.8f};
    {
        apply_rope(traced, 3, tables);
        CHECK_NEAR(traced[0], -1.06055f, 0.001f);
        CHECK_NEAR(traced[1], -0.35388f, 0.001f);
        CHECK_NEAR(traced[2], -0.32386f, 0.001f);
        CHECK_NEAR(traced[3], 0.79064f, 0.001f);
    }

    // --- Rotation must preserve vector length: ||Rx|| == ||x|| ---
    // A fundamental property of any rotation matrix; a wrong sign or
    // swapped sin/cos anywhere in apply_rope breaks this immediately.
    float norm_before = 0.0f, norm_after = 0.0f;
    {
        std::vector<float> v = {0.5f, 1.2f, -0.8f, 0.3f};
        norm_before = norm(v);
        apply_rope(v, 7, tables);
        norm_after = norm(v);
        CHECK_NEAR(norm_before, norm_after, 0.0001f);
    }

    // --- Position 0 is the identity rotation (angle = 0) ---
    {
        std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> original = v;
        apply_rope(v, 0, tables);
        for (size_t i = 0; i < v.size(); ++i) CHECK_NEAR(v[i], original[i], 1e-5f);
    }

    // --- The relative-position property: dot depends only on (m - n) ---
    float dot_a = 0.0f, dot_b = 0.0f;
    {
        std::vector<float> Q = {0.6f, 0.4f, -0.2f, 0.8f};
        std::vector<float> K = {0.3f, 0.7f, 0.5f, 0.1f};

        std::vector<float> Qa = Q, Ka = K;
        apply_rope(Qa, 5, tables); apply_rope(Ka, 3, tables);   // distance 2
        dot_a = dot(Qa, Ka);

        std::vector<float> Qb = Q, Kb = K;
        apply_rope(Qb, 2, tables); apply_rope(Kb, 0, tables);   // distance 2
        dot_b = dot(Qb, Kb);

        CHECK_NEAR(dot_a, dot_b, 0.0001f);
    }

    std::cout << "rope(pos=3, [1, 0.5, -0.3, 0.8]) = ["
              << traced[0] << ", " << traced[1] << ", " << traced[2] << ", " << traced[3] << "]\n";
    std::cout << "norm before rotation: " << norm_before << ", after: " << norm_after
              << " (rotation preserves length): " << (std::fabs(norm_before - norm_after) < 0.0001f ? "yes" : "no") << "\n";
    std::cout << "dot(Q@5, K@3) = " << dot_a << ", dot(Q@2, K@0) = " << dot_b
              << " (same relative distance 2, same dot): " << (std::fabs(dot_a - dot_b) < 0.0001f ? "yes" : "no") << "\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_rope.cpp -o 03_rope
./03_rope
```

**Sample input:** a precomputed cos/sin table for 16 positions and `head_dim=4`; the `position=3` worked example this section traces by hand; a norm-preservation check at position 7; the position-0 identity case; and two Q/K pairs placed at different absolute positions but the same relative distance of 2.

**Sample output:**

```text
rope(pos=3, [1, 0.5, -0.3, 0.8]) = [-1.06055, -0.353876, -0.323861, 0.790641]
norm before rotation: 1.55563, after: 1.55563 (rotation preserves length): yes
dot(Q@5, K@3) = 0.0529662, dot(Q@2, K@0) = 0.0529662 (same relative distance 2, same dot): yes

10/10 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Applying RoPE to V, or applying it after the attention dot product"
    RoPE only ever rotates Q and K, and only before the `Q . K` dot product is computed — never V, and never the attention output. V carries CONTENT, not position-dependent relevance, so rotating it would corrupt the actual values being blended together for no benefit; and rotating Q or K after the dot product is simply too late to have any effect on the score that dot product produces. The one-line summary worth keeping in mind while wiring a real attention block together: RoPE lives strictly between the QKV projections and the attention score computation, and touches exactly two of the three tensors that pass through that gap.

## 3.4 Grouped-Query Attention with a KV Cache

### Intuition

`Attention(Q, K, V) = softmax(Q . K^T / sqrt(head_dim)) . V` lets a token look back across every earlier position and blend their value vectors by relevance. Multi-Head Attention gives every query head its own key/value head (most accurate, most KV-cache memory); Multi-Query Attention collapses every query head onto a single shared key/value head (least memory, most quality loss); Grouped-Query Attention — what Llama 3 and Mistral actually ship — sits deliberately between the two, letting groups of query heads share one key/value head, for a KV cache `group_size` times smaller than MHA's cache at a fraction of MQA's quality cost.

### The Concept, In Detail

The KV cache is genuinely three-dimensional — `[kv_head, position, head_dim]` — so this section stores it behind Chapter 2's `std::mdspan` vocabulary rather than the flat `std::span` views the rest of this chapter uses for ordinary vectors, and extracts one head's one cached position with `std::submdspan` exactly as Chapter 2.2 extracted one attention head from a larger tensor. Naive softmax overflows the instant a raw score gets large (`exp(100)` alone is already far past what a `float` can represent), so this section's kernel subtracts the running maximum before exponentiating — algebraically exact (the shift cancels between numerator and denominator) and numerically safe, since every value `exp()` is asked to compute afterward is at most `exp(0) = 1`.

### Code and Verification

```cpp
// Chapter 3.4 -- attention lets a token look back at every earlier
// token's key and value vectors and blend them by relevance:
//
//   Attention(Q, K, V) = softmax(Q . K^T / sqrt(head_dim)) . V
//
// Grouped-Query Attention (GQA) is the specific shape Llama 3 and
// Mistral use: many query heads (32, say) share a much smaller number
// of key/value heads (8, say), each KV head serving a "group" of
// query heads. This is a deliberate point on a spectrum between
// ordinary Multi-Head Attention (one KV head per Q head -- most
// accurate, most KV-cache memory) and Multi-Query Attention (one KV
// head total -- least memory, most quality loss); GQA's KV cache is
// group_size times smaller than MHA's for a fraction of MQA's quality
// cost.
//
// The KV cache itself is genuinely three-dimensional --
// [kv_head, position, head_dim] -- so unlike Chapter 3's other
// kernels (which operate on flat vectors and use std::span), this
// section stores it behind the std::mdspan vocabulary Chapter 2
// established, and extracts one head's one position with
// std::submdspan exactly as Chapter 2.2 did for multi-head attention.

#include <mdspan/mdspan.hpp>
#include <cmath>
#include <span>
#include <vector>
#include <algorithm>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// Softmax, shifted by the running max before exponentiating: exp(100)
// overflows a float (2.69e43), but exp(score - max) is always in
// (0, 1], so every intermediate value this loop touches stays finite
// -- the shift is exact (it cancels between numerator and
// denominator), not an approximation.
void softmax_inplace(std::span<float> scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_val); sum += s; }
    float inv_sum = 1.0f / sum;
    for (float& s : scores) s *= inv_sum;
}

float dot_product(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

// A 3D mdspan over one flat buffer: [n_heads_kv, max_seq_len, head_dim].
// k_at(h, t) uses std::submdspan to fix the first two dimensions and
// keep only head_dim -- a 1D, zero-copy slice into this one (head,
// position) slot, exactly Chapter 2.2's "drop a dimension" pattern.
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

// Decode-phase GQA: one new query token attends to every cached
// position. `q_heads` is [n_heads_q * head_dim]; query head h reads
// KV head h/group_size, so group_size query heads always share the
// exact same K and V slices.
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

int main() {
    // --- Softmax: correctness against a hand-worked example, plus overflow safety ---
    {
        std::vector<float> scores = {0.050f, 0.295f, 0.970f};
        softmax_inplace(scores);
        CHECK_NEAR(scores[0], 0.2089f, 0.001f);
        CHECK_NEAR(scores[1], 0.2670f, 0.001f);
        CHECK_NEAR(scores[2], 0.5242f, 0.001f);

        std::vector<float> large = {100.0f, 200.0f, 300.0f};   // would overflow naive exp()
        softmax_inplace(large);
        bool all_finite = true;
        for (float p : large) if (!std::isfinite(p)) all_finite = false;
        CHECK(all_finite);
        CHECK_NEAR(large[0] + large[1] + large[2], 1.0f, 1e-5f);
    }

    // --- Full attention trace, one Q head, one KV head, 3 cached positions ---
    std::vector<float> traced_output;
    {
        constexpr int HEAD_DIM = 4, SEQ_LEN = 3, N_Q = 1, N_KV = 1, GROUP = 1;
        KVCache cache(N_KV, SEQ_LEN + 1, HEAD_DIM);
        cache.store(0, 0, std::vector<float>{0.1f, 0.4f, 0.9f, -0.2f}, std::vector<float>{1.0f, 0.0f, 0.5f, -0.3f});
        cache.store(0, 1, std::vector<float>{0.8f, -0.1f, 0.3f, 0.5f}, std::vector<float>{0.2f, 0.8f, -0.1f, 0.4f});
        cache.store(0, 2, std::vector<float>{0.3f, 0.7f, -0.5f, 1.0f}, std::vector<float>{0.6f, 0.3f, 0.9f, 0.1f});

        std::vector<float> Q = {0.5f, 1.2f, -0.3f, 0.8f};
        std::vector<float> output(N_Q * HEAD_DIM, 0.0f);
        gqa_attention(Q, cache, output, SEQ_LEN, N_Q, GROUP);

        CHECK_NEAR(output[0], 0.577f, 0.005f);
        CHECK_NEAR(output[1], 0.371f, 0.005f);
        CHECK_NEAR(output[2], 0.550f, 0.005f);
        CHECK_NEAR(output[3], 0.096f, 0.005f);
        traced_output = output;
    }

    // --- GQA sharing: two Q heads reading the SAME KV head diverge with Q ---
    std::vector<float> shared_output;
    {
        constexpr int HEAD_DIM = 4, SEQ_LEN = 2, N_Q = 2, N_KV = 1, GROUP = 2;
        KVCache cache(N_KV, SEQ_LEN + 1, HEAD_DIM);
        cache.store(0, 0, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f}, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
        cache.store(0, 1, std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f}, std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f});

        std::vector<float> Q = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};   // head 0 -> K0, head 1 -> K1
        std::vector<float> output(N_Q * HEAD_DIM, 0.0f);
        gqa_attention(Q, cache, output, SEQ_LEN, N_Q, GROUP);

        // scale = 1/sqrt(4) = 0.5. Head 0: dot(Q0,K0)=1, dot(Q0,K1)=0 -> scaled
        // [0.5, 0] -> softmax [0.62246, 0.37754] -> output = 0.62246*V0 + 0.37754*V1
        // = 1.37754 in every coordinate. Head 1 sees the mirror image: scaled
        // [0, 0.5] -> softmax [0.37754, 0.62246] -> output = 1.62246 in every
        // coordinate. Both heads read the identical K, V, and identical
        // softmax formula -- only the query differs, and that alone is enough
        // to visibly separate the two outputs.
        for (int i = 0; i < HEAD_DIM; ++i) CHECK_NEAR(output[i], 1.37754f, 0.001f);
        for (int i = 0; i < HEAD_DIM; ++i) CHECK_NEAR(output[HEAD_DIM + i], 1.62246f, 0.001f);
        CHECK(output[HEAD_DIM] > output[0]);   // head 1 leans further toward V1 than head 0 does
        shared_output = output;
    }

    std::cout << "attention output (1 Q head, 3 cached positions): ["
              << traced_output[0] << ", " << traced_output[1] << ", "
              << traced_output[2] << ", " << traced_output[3] << "]\n";
    std::cout << "GQA head 0 output (query aligned with K0, closer to V0=[1,1,1,1]): ["
              << shared_output[0] << ", " << shared_output[1] << ", "
              << shared_output[2] << ", " << shared_output[3] << "]\n";
    std::cout << "GQA head 1 output (query aligned with K1, closer to V1=[2,2,2,2]): ["
              << shared_output[4] << ", " << shared_output[5] << ", "
              << shared_output[6] << ", " << shared_output[7] << "]\n";
    std::cout << "large-score softmax [100,200,300] stayed finite: yes\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 \
    -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental \
    -I/path/to/mdspan/include 04_gqa_attention_kv_cache.cpp -o 04_gqa_attention_kv_cache
./04_gqa_attention_kv_cache
```

**Sample input:** a numerically stable softmax checked against both a hand-worked example and a deliberately overflow-prone input (`[100, 200, 300]`); a single query head attending over 3 cached positions, traced against this section's own worked arithmetic; and two query heads sharing one KV head, each aligned with a different cached key, to show that GQA's shared cache still lets different queries land on visibly different outputs.

**Sample output:**

```text
attention output (1 Q head, 3 cached positions): [0.576802, 0.370779, 0.54954, 0.0965085]
GQA head 0 output (query aligned with K0, closer to V0=[1,1,1,1]): [1.37754, 1.37754, 1.37754, 1.37754]
GQA head 1 output (query aligned with K1, closer to V1=[2,2,2,2]): [1.62246, 1.62246, 1.62246, 1.62246]
large-score softmax [100,200,300] stayed finite: yes

18/18 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Computing kv_head as h % group_size instead of h / group_size"
    GQA groups query heads into CONTIGUOUS blocks that share one KV head — heads `0..group_size-1` share KV head 0, heads `group_size..2*group_size-1` share KV head 1, and so on — which is integer division (`h / group_size`), not modulo. Using `h % group_size` by mistake interleaves the grouping instead (heads `0, group_size, 2*group_size, ...` would all incorrectly map to KV head 0), silently reading the wrong cached keys and values for every query head except the first `group_size` of them, with no crash or error to reveal the mistake — only quietly wrong attention output.

## 3.5 The Static Compute Graph: Record Once, Replay Cheaply

### Intuition

An eager execution engine inspects shapes, allocates outputs, and dispatches a kernel from scratch for every single operation, every single token — overhead that is small per call but is paid thousands of times per forward pass. A static graph separates recording (walk the model once, store a closure per operation) from execution (call each closure in order), so the hot, per-token path becomes nothing more than a list of function-pointer calls with no shape checks and no allocation.

### The Concept, In Detail

Once every tensor's lifetime is known ahead of time — which operation creates it, which operation is the last one to read it — tensors whose lifetimes never overlap can share the same backing arena memory, exactly the way Chapter 2's arena `reset()` let one token's scratch buffers become the next token's scratch buffers. This section's own policy, restated from Getting Started: a genuinely computed, deterministic quantity is preferred over a wall-clock number wherever one exists for the argument being made, and "how many bytes a liveness-aware plan needs, against how many a naive one needs" is exactly such a quantity — a real, reproducible byte count, not a millisecond figure that depends on the machine it happened to run on. The worked example below builds a small first-fit memory planner for exactly this purpose and locks the resulting byte counts, not a timing comparison, into this chapter's verified output.

### Code and Verification

```cpp
// Chapter 3.5 -- an eager execution engine (PyTorch's default mode)
// re-inspects shapes, re-allocates outputs, and re-dispatches a kernel
// from scratch for every single operation, every single token. A
// STATIC graph instead separates two phases that never need to repeat
// together: record the whole operation sequence once, as a list of
// closures, then replay that exact list every token with nothing more
// than a function-pointer call per step -- no shape checks, no
// allocation, no bookkeeping.
//
// The other half of a static graph's payoff is memory planning: once
// every tensor's lifetime (which op creates it, which op last reads
// it) is known up front, tensors whose lifetimes never overlap can
// share the same backing memory. This book's own policy (Getting
// Started) prefers a genuinely computed, deterministic quantity over
// a wall-clock number wherever one exists for the argument being
// made -- and "how many bytes of arena a liveness-aware plan needs,
// against how many a naive one needs" is exactly such a quantity: it
// never depends on which machine runs it, unlike a millisecond figure
// for a single function-pointer call.

#include <functional>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// One recorded operation: a closure capturing whatever pointers and
// sizes it needs at RECORD time. At EXECUTE time the entire hot path
// is "call it" -- no branching on what kind of op this is.
using Task = std::function<void()>;

class ComputeGraph {
public:
    void add_op(std::string name, Task fn) {
        m_names.push_back(std::move(name));
        m_tape.push_back(std::move(fn));
    }
    void execute() { for (auto& op : m_tape) op(); }
    size_t op_count() const { return m_tape.size(); }
private:
    std::vector<Task> m_tape;
    std::vector<std::string> m_names;
};

// Stand-ins for real kernels (RMSNorm, matmul, ...) that write
// recognizable, hand-checkable values instead of doing real math --
// what matters here is proving the GRAPH replays operations in the
// right order with the right data, not re-deriving Section 3.1-3.4's
// kernels a second time.
void fake_norm(float* out, const float* in, size_t n, float tag) {
    for (size_t i = 0; i < n; ++i) out[i] = in[i] + tag;
}
void fake_add(float* out, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

class Arena {
    std::vector<float> m_buf;
    size_t m_off = 0;
public:
    explicit Arena(size_t n_floats) : m_buf(n_floats, 0.0f) {}
    float* alloc(size_t n) {
        float* p = m_buf.data() + m_off;
        m_off += n;
        return p;
    }
};

// A tensor's liveness interval: created at op `birth`, last read at op
// `death` (inclusive). Two intervals that do not overlap can share the
// same backing offset -- this is a first-fit greedy planner, not the
// optimal one a real allocator would use, but it is enough to make the
// savings concrete and checkable.
struct TensorLifetime {
    std::string name;
    size_t bytes;
    int birth, death;
};

struct Placement { size_t offset; };

// Greedy first-fit: scan tensors in birth order, and for each one, try
// to reuse the lowest offset among tensors whose intervals have
// already ended (death < this tensor's birth). If none fits, extend
// the arena. Deterministic given a fixed input order -- no timing
// involved anywhere in this computation.
std::vector<Placement> plan_offsets(const std::vector<TensorLifetime>& tensors) {
    std::vector<Placement> placements(tensors.size());
    std::vector<bool> placed(tensors.size(), false);
    std::vector<size_t> block_offset, block_size;
    std::vector<int> block_owner_death;

    std::vector<size_t> order(tensors.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return tensors[a].birth < tensors[b].birth;
    });

    for (size_t idx : order) {
        const auto& t = tensors[idx];
        int best_block = -1;
        for (size_t b = 0; b < block_offset.size(); ++b) {
            if (block_owner_death[b] < t.birth && block_size[b] >= t.bytes) {
                best_block = static_cast<int>(b);
                break;
            }
        }
        if (best_block >= 0) {
            placements[idx] = {block_offset[static_cast<size_t>(best_block)]};
            block_owner_death[static_cast<size_t>(best_block)] = t.death;
        } else {
            size_t new_offset = 0;
            for (size_t b = 0; b < block_offset.size(); ++b)
                new_offset = std::max(new_offset, block_offset[b] + block_size[b]);
            block_offset.push_back(new_offset);
            block_size.push_back(t.bytes);
            block_owner_death.push_back(t.death);
            placements[idx] = {new_offset};
        }
        placed[idx] = true;
    }
    return placements;
}

int main() {
    constexpr size_t DIM = 8;

    // --- Record and execute a 3-op pipeline: Norm -> Double -> Residual Add ---
    {
        Arena arena(DIM * 10);
        float* input = arena.alloc(DIM);
        float* norm = arena.alloc(DIM);
        float* scaled = arena.alloc(DIM);
        float* output = arena.alloc(DIM);
        for (size_t i = 0; i < DIM; ++i) input[i] = 1.0f;

        ComputeGraph graph;
        graph.add_op("Norm", [input, norm]() { fake_norm(norm, input, DIM, 10.0f); });
        graph.add_op("Double", [norm, scaled]() { fake_add(scaled, norm, norm, DIM); });
        graph.add_op("ResidualAdd", [input, scaled, output]() { fake_add(output, input, scaled, DIM); });
        graph.execute();

        CHECK_NEAR(norm[0], 11.0f, 1e-4f);
        CHECK_NEAR(scaled[0], 22.0f, 1e-4f);
        CHECK_NEAR(output[0], 23.0f, 1e-4f);
    }

    // --- The graph is recorded once; only the input buffer changes per token ---
    std::vector<float> replay_results;
    {
        Arena arena(DIM * 5);
        float* x = arena.alloc(DIM);
        float* out = arena.alloc(DIM);

        ComputeGraph graph;
        graph.add_op("Scale3", [x, out]() { for (size_t i = 0; i < DIM; ++i) out[i] = x[i] * 3.0f; });
        graph.add_op("AddOne", [out]() { for (size_t i = 0; i < DIM; ++i) out[i] += 1.0f; });

        for (int tok = 1; tok <= 3; ++tok) {
            for (size_t i = 0; i < DIM; ++i) x[i] = static_cast<float>(tok);
            graph.execute();
            float expected = static_cast<float>(tok) * 3.0f + 1.0f;
            CHECK_NEAR(out[0], expected, 1e-4f);
            replay_results.push_back(out[0]);
        }
    }

    // --- Liveness-aware memory planning (Section 3.5's own worked example) ---
    // Five tensors from one transformer layer, sized in bytes (float = 4B),
    // exactly Section 6.2's own scenario: T_A (norm output, 4096 floats),
    // T_B (QKV output, 3x bigger), T_C (attention output), T_D (FFN output).
    size_t naive_bytes = 0, planned_bytes = 0;
    {
        std::vector<TensorLifetime> tensors = {
            {"T_A", 4096 * 4, 0, 1},   // created at op 0, last read at op 1
            {"T_B", 12288 * 4, 1, 3},  // created at op 1, last read at op 3
            {"T_C", 4096 * 4, 3, 4},   // created at op 3, last read at op 4
            {"T_D", 4096 * 4, 4, 4},   // created at op 4, itself the output
        };
        auto placements = plan_offsets(tensors);

        for (const auto& t : tensors) naive_bytes += t.bytes;   // no reuse at all
        for (size_t i = 0; i < tensors.size(); ++i)
            planned_bytes = std::max(planned_bytes, placements[i].offset + tensors[i].bytes);

        // T_A and T_C must NOT overlap in time (T_A dies at op 1, T_C is born
        // at op 3), so the planner should have reused T_A's slot for T_C.
        CHECK(placements[0].offset == placements[2].offset);   // T_A and T_C share a slot
        CHECK(placements[1].offset != placements[0].offset);   // T_B (still live) does not
        CHECK(planned_bytes < naive_bytes);                    // the whole point of planning
    }

    std::cout << "3-op pipeline: norm[0]=11, scaled[0]=22, output[0]=23 (all exact): yes\n";
    std::cout << "same graph replayed for tokens 1,2,3 -> out[0] = "
              << replay_results[0] << ", " << replay_results[1] << ", " << replay_results[2] << "\n";
    std::cout << "naive (no reuse) arena bytes needed: " << naive_bytes << "\n";
    std::cout << "liveness-planned arena bytes needed: " << planned_bytes << "\n";
    std::cout << "planned uses fewer bytes than naive: " << (planned_bytes < naive_bytes ? "yes" : "no") << "\n";
    std::cout << "T_A and T_C (non-overlapping lifetimes) share one arena slot: yes\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_static_compute_graph.cpp -o 05_static_compute_graph
./05_static_compute_graph
```

**Sample input:** a 3-operation graph (normalize, double, residual-add) executed once and checked against hand-computed exact values; the same 2-operation graph replayed for three different simulated tokens with only the input buffer changing; and this section's own memory-planning scenario — four tensors from one transformer layer with the exact liveness intervals Section 3.5 describes in prose.

**Sample output:**

```text
3-op pipeline: norm[0]=11, scaled[0]=22, output[0]=23 (all exact): yes
same graph replayed for tokens 1,2,3 -> out[0] = 4, 7, 10
naive (no reuse) arena bytes needed: 98304
liveness-planned arena bytes needed: 65536
planned uses fewer bytes than naive: yes
T_A and T_C (non-overlapping lifetimes) share one arena slot: yes

9/9 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Reordering operations after recording, without re-running liveness analysis"
    A memory plan is only valid for the exact operation order it was computed from — it encodes which tensor's slot is safe to reuse based on precisely when each tensor is born and when it is last read. Reordering the tape after the fact (even a change that looks harmless, like moving a debug-only operation earlier) can make two tensors that the planner assumed never overlap suddenly alive at the same time, and the shared arena slot the planner assigned them turns into silent aliasing: one write corrupting data the other tensor still needs, with nothing in the type system able to catch it, since both are just `float*` into the same arena as far as the compiler is concerned.

## 3.6 mmap Weight Loading: Zero-Copy Access to a Multi-Gigabyte File

### Intuition

Loading a large weight file the ordinary way (`fread`/`ifstream` into a heap buffer) means reading every byte off disk before a single weight is usable, and doubles memory usage in the process — the OS's own page cache and the freshly `malloc`'d heap buffer both hold a copy of the same bytes. `mmap` maps the file directly into the process's address space instead: the mapping itself is essentially instant no matter the file's size, and the OS only pulls a page off disk the first time something actually reads from it.

### The Concept, In Detail

`mmap(..., PROT_READ, MAP_PRIVATE, fd, 0)` maps a file read-only (a write through the resulting pointer is a hard `SIGSEGV`, not silent corruption of a model's own weight file) and privately (even if that guarantee were relaxed, a write would be copy-on-write and process-local). The file descriptor can be closed immediately after the `mmap()` call succeeds — the mapping remains valid without it, and closing early keeps a program that maps many files from exhausting its file-descriptor table. Once mapped, a weight tensor is built by wrapping a `std::span` (or, for a multi-dimensional weight matrix, a Chapter 2 `mdspan`) directly around a pointer into the mapped region: zero-copy in the fullest sense, since the "allocation" backing that view was never made by this program's own heap at all.

### Code and Verification

```cpp
// Chapter 3.6 -- loading a multi-gigabyte weight file with
// fread/ifstream means reading every byte off disk into a heap buffer
// before a single weight is used: tens of seconds for a large model,
// and double the memory footprint (the OS's page cache AND your own
// heap both hold a copy). mmap instead maps the file directly into
// the process's virtual address space and lets the OS satisfy actual
// reads with lazy page faults -- only the pages a chapter's tensors
// genuinely touch ever come off disk, and the mapping itself is
// essentially instant regardless of file size.
//
// This is also the payoff of Chapter 2's std::span/mdspan vocabulary
// stated as plainly as it gets: a weight tensor built from an mmap'd
// region is a view whose backing "allocation" was never allocated by
// this program's heap at all -- reading it is exactly as zero-copy as
// reading a std::vector's buffer, and the OS transparently pages the
// file behind it.

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstddef>
#include <span>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cmath>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) < (tol))

// Maps a file read-only into this process's address space. PROT_READ
// means a write through the resulting pointer is a hard SIGSEGV, not
// silent corruption -- model weights should never be writable.
// MAP_PRIVATE means even if that guarantee were relaxed, any write
// would be copy-on-write and process-local, never touching the file
// on disk. The file descriptor is closed immediately after mmap(): the
// mapping stays valid without it, and closing early keeps this
// process's fd table from filling up when many files are mapped.
class MmapLoader {
public:
    explicit MmapLoader(const std::string& path) {
        m_fd = open(path.c_str(), O_RDONLY);
        if (m_fd == -1) throw std::runtime_error("failed to open: " + path);

        struct stat sb;
        if (fstat(m_fd, &sb) == -1) { close(m_fd); throw std::runtime_error("fstat failed"); }
        m_size = static_cast<size_t>(sb.st_size);

        void* ptr = mmap(nullptr, m_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
        if (ptr == MAP_FAILED) { close(m_fd); throw std::runtime_error("mmap failed"); }
        m_data = static_cast<const std::byte*>(ptr);

        close(m_fd);
        m_fd = -1;
    }
    ~MmapLoader() {
        if (m_data != nullptr) munmap(const_cast<void*>(static_cast<const void*>(m_data)), m_size);
    }
    MmapLoader(const MmapLoader&) = delete;
    MmapLoader& operator=(const MmapLoader&) = delete;

    // A zero-copy view of `count` floats starting at `byte_offset` --
    // no data is copied; the span's pointer lands directly inside the
    // mmap'd region. This is exactly how a real engine builds its
    // weight tensors: `floats_at(header.offset, n)` feeds straight
    // into a TensorView/mdspan constructor from Chapter 2.
    std::span<const float> floats_at(size_t byte_offset, size_t count) const {
        if (byte_offset + count * sizeof(float) > m_size)
            throw std::out_of_range("read past end of mapped file");
        return {reinterpret_cast<const float*>(m_data + byte_offset), count};
    }
    size_t size_bytes() const { return m_size; }

private:
    const std::byte* m_data = nullptr;
    size_t m_size = 0;
    int m_fd = -1;
};

// Writes a tiny fake weight file: three 32-float blocks with distinct,
// hand-checkable values, standing in for a real GGUF/SafeTensors file
// (Chapter 5 builds a real one; this section only needs a file to map).
void write_test_weights(const std::string& path, int n) {
    std::vector<float> A(n), B(n), C(n);
    for (int i = 0; i < n; ++i) {
        A[static_cast<size_t>(i)] = static_cast<float>(i + 1);          // 1..n
        B[static_cast<size_t>(i)] = static_cast<float>(100 + i);        // 100..(99+n)
        C[static_cast<size_t>(i)] = static_cast<float>(-(i + 1));       // -1..-n
    }
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) throw std::runtime_error("cannot create test file: " + path);
    auto write_vec = [&](const std::vector<float>& v) {
        ssize_t written = write(fd, v.data(), v.size() * sizeof(float));
        if (written != static_cast<ssize_t>(v.size() * sizeof(float)))
            throw std::runtime_error("short write while creating test file");
    };
    write_vec(A); write_vec(B); write_vec(C);
    close(fd);
}

int main() {
    constexpr int N = 32;
    const std::string test_file = "/tmp/ch3_mmap_test_weights.bin";
    write_test_weights(test_file, N);

    size_t file_size = 0;
    float wa0 = 0, wa31 = 0, wb0 = 0, wc31 = 0;
    // --- Basic load: file size and every slice's exact values ---
    {
        MmapLoader loader(test_file);
        file_size = loader.size_bytes();
        CHECK(file_size == 3 * static_cast<size_t>(N) * sizeof(float));

        auto W_A = loader.floats_at(0 * static_cast<size_t>(N) * sizeof(float), N);
        auto W_B = loader.floats_at(1 * static_cast<size_t>(N) * sizeof(float), N);
        auto W_C = loader.floats_at(2 * static_cast<size_t>(N) * sizeof(float), N);

        CHECK_NEAR(W_A[0], 1.0f, 1e-5f);
        CHECK_NEAR(W_A[31], 32.0f, 1e-5f);
        CHECK_NEAR(W_B[0], 100.0f, 1e-5f);
        CHECK_NEAR(W_C[31], -32.0f, 1e-5f);
        wa0 = W_A[0]; wa31 = W_A[31]; wb0 = W_B[0]; wc31 = W_C[31];
    }

    // --- Zero-copy proof: reading through the view needs no memcpy step ---
    float dot = 0.0f;
    {
        MmapLoader loader(test_file);
        auto W_A = loader.floats_at(0, 32);
        CHECK(W_A.data() != nullptr);
        std::vector<float> x(32, 1.0f);
        for (size_t i = 0; i < 32; ++i) dot += x[i] * W_A[i];
        // W_A = [1..32], x = all-ones -> dot = 1+2+...+32 = 528, exactly.
        CHECK_NEAR(dot, 528.0f, 0.01f);
    }

    // --- Out-of-bounds reads are rejected, not silently satisfied ---
    bool last_float_ok = false, oob_rejected = false;
    {
        MmapLoader loader(test_file);
        try { auto s = loader.floats_at(file_size - sizeof(float), 1); (void)s; last_float_ok = true; }
        catch (...) {}
        try { auto s = loader.floats_at(file_size - sizeof(float), 2); (void)s; }
        catch (const std::out_of_range&) { oob_rejected = true; }
    }
    CHECK(last_float_ok);
    CHECK(oob_rejected);

    unlink(test_file.c_str());

    std::cout << "mapped file size: " << file_size << " bytes ("
              << 3 * N * sizeof(float) << " expected)\n";
    std::cout << "W_A[0]=" << wa0 << " W_A[31]=" << wa31
              << " W_B[0]=" << wb0 << " W_C[31]=" << wc31 << "\n";
    std::cout << "dot(all-ones, W_A) = " << dot << " (expected 528)\n";
    std::cout << "reading the last valid float succeeded: " << (last_float_ok ? "yes" : "no") << "\n";
    std::cout << "reading one float past the end was rejected: " << (oob_rejected ? "yes" : "no") << "\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 06_mmap_weight_loading.cpp -o 06_mmap_weight_loading
./06_mmap_weight_loading
```

**Sample input:** a small, hand-constructed binary file holding three 32-float blocks with distinct, checkable values, `mmap`'d and read back as three `std::span<const float>` views with no intervening copy.

**Sample output:**

```text
mapped file size: 384 bytes (384 expected)
W_A[0]=1 W_A[31]=32 W_B[0]=100 W_C[31]=-32
dot(all-ones, W_A) = 528 (expected 528)
reading the last valid float succeeded: yes
reading one float past the end was rejected: yes

9/9 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Assuming a mmap'd pointer is safe to read past munmap(), or past the file's own size"
    Unlike a `std::vector`, `mmap` performs no bounds checking of its own — reading past the mapped region's end is undefined behavior the operating system may or may not catch with a segfault, and reading through a pointer after `munmap()` has already run is a use-after-free into address space the kernel may have already reassigned to something else entirely. Both mistakes are silent right up until they are not: a `floats_at()`-style accessor that checks its own bounds before ever handing back a `std::span` (as this section's does) turns the first hazard into a clean, catchable exception; nothing but disciplined ownership — never reading through an `MmapLoader` after it has been destroyed — protects against the second.

## Chapter Summary

RMSNorm, SwiGLU, and RoPE each replace one Transformer-original component with something Llama-style models found cheaper or more capable: RMSNorm drops LayerNorm's mean-subtraction pass for one memory-bound reduction, SwiGLU adds a third, gating matrix to a two-matrix FFN so that negative activations are softly suppressed rather than hard-zeroed, and RoPE rotates Q and K by a position-dependent angle so that their dot product encodes relative distance rather than either absolute position. Grouped-Query Attention groups many query heads onto a smaller number of key/value heads, reading from a genuinely three-dimensional KV cache built on Chapter 2's `std::mdspan` and sliced with `std::submdspan`, and a numerically stable, max-subtracted softmax keeps that computation finite no matter how large the raw attention scores get. The static compute graph turns "record once, replay per token" into a concrete performance strategy and, more importantly, a memory-planning one: this chapter locked a genuinely computed byte count — not a wall-clock number — into its own verified output to show a liveness-aware plan using fewer bytes than a naive one, exactly this book's stated policy for when an argument needs a reproducible number rather than a machine-specific one. And `mmap` closes the chapter by making weight loading itself zero-copy, turning a multi-gigabyte file into a set of `std::span`/`mdspan` views the OS pages in lazily rather than a heap-sized copy paid for up front. Chapter 4 begins Part 1: shrinking every one of these tensors with quantization, starting from the same honesty discipline — genuinely compiled, genuinely run, nothing asserted that was not checked — this chapter and Chapter 2 both followed.

## Self-Check Questions

1. Section 3.1 states RMSNorm is memory-bandwidth bound, not compute bound. Using the FLOPs-versus-bytes figures this section gives, explain in your own words why adding more SIMD lanes to the sum-of-squares loop would not meaningfully speed this kernel up.
2. Section 3.2's identity-weight test collapses the full three-matmul SwiGLU pipeline to `SiLU(x) elementwise* x`. Explain why setting `W_gate = W_up = W_down = I` (the identity matrix) is what makes this collapse happen, rather than merely simplifying the test.
3. Using Section 3.3's frequency formula `theta_k = 1 / base^(2k / head_dim)`, explain why pair index `k=0` always rotates the fastest of any pair in the vector, regardless of `head_dim` or `base`.
4. Section 3.3's fourth test checks `dot(Q@5, K@3) == dot(Q@2, K@0)`. Both pairs have relative distance 2, but name one thing that is NOT held equal between the two cases, and explain why RoPE's own math guarantees the dot product is unaffected by that difference anyway.
5. Section 3.4 computes `kv_h = h / group_size`, integer division. For `n_heads_q=8` and `n_heads_kv=2` (so `group_size=4`), list which query heads map to KV head 0 and which map to KV head 1.
6. A colleague suggests skipping the max-subtraction step in Section 3.4's softmax "since it doesn't change the mathematical answer." Using the specific numbers `[100, 200, 300]` this section's own code tests, explain concretely what would happen to `exp(300)` in a 32-bit float, and why the max-subtracted version avoids it while still returning the exact same probabilities.
7. Section 3.5's memory planner places `T_A` and `T_C` at the same arena offset but gives `T_B` its own separate offset. Using the four tensors' birth/death intervals this section lists, explain what specifically makes `T_A`/`T_C` shareable but `T_B` not.
8. Section 3.5 states this book prefers a byte count over a millisecond figure "wherever one exists for the argument being made." Name one kind of argument about a static compute graph's benefit where a byte count could NOT stand in for a timing measurement, and briefly say why.
9. Section 3.6's `floats_at()` throws `std::out_of_range` rather than returning a truncated or empty span when a read would go past the mapped file's end. Explain why silently truncating the requested count instead would be a worse design for a function real tensor-construction code calls.
10. Both Section 3.4 (the KV cache) and Section 3.6 (mmap'd weights) build `std::span`/`mdspan` views over memory this program's own heap never allocated. Name the two different "owners" of that memory in each case.

## Where We Go Next

Part 1 begins with Chapter 4: quantization, starting from the affine scale/zero-point scheme that turns a 32-bit float tensor into an 8-bit or 4-bit integer one at a fraction of the memory, and the blockwise scaling that keeps a single outlier value from wrecking an entire tensor's precision. Every kernel this chapter built — RMSNorm, SwiGLU, RoPE, GQA attention — reappears in Part 1 reading quantized weights instead of full-precision ones, through the exact `std::span`/`mdspan` interfaces this chapter already established, so that quantization becomes a change in what a tensor's bytes MEAN rather than a change in how any of this chapter's call sites are written.

## Worked Solutions

**1.** The sum-of-squares loop performs `d` multiplies and `d-1` adds, but the kernel as a whole reads two `d`-length vectors and writes one, for `3d * 4` bytes of memory traffic against roughly `3d` FLOPs — an arithmetic intensity around 0.25 FLOPs per byte, far below what a modern CPU can sustain per byte of memory bandwidth (order 10-20). A kernel this far below the hardware's compute-to-bandwidth ratio spends nearly all of its time waiting for memory to arrive, not waiting for the ALU to finish arithmetic; adding more SIMD lanes makes the ALU finish its share of the work faster, but the ALU was never the bottleneck; the memory bus was, and it now sits idle even more of the time.

**2.** `swiglu_ffn` computes `matmul(gate_proj, x, W_gate)`, `matmul(up_proj, x, W_up)`, then `SiLU(gate_proj) elementwise* up_proj`, then a final `matmul` through `W_down`. Setting `W_gate = I` makes `gate_proj` exactly equal to `x` (an identity matmul reproduces its input unchanged); setting `W_up = I` makes `up_proj` exactly equal to `x` as well; and setting `W_down = I` leaves the down projection's output exactly equal to its input. With both projections collapsed to `x` itself and the down projection collapsed to a no-op, the entire pipeline reduces algebraically to `SiLU(x) elementwise* x` — not an approximation of that formula, but a genuinely exact identity, because every matrix standing between the formula and the code is literally the identity matrix.

**3.** `theta_k` shrinks as `k` grows, for any `base > 1` and any positive `head_dim`, because `2k/head_dim` is an increasing function of `k` and `base` raised to a larger power produces a larger denominator, hence a smaller `theta_k = 1/base^(2k/head_dim)`. At `k=0` specifically, the exponent `2*0/head_dim = 0`, so `theta_0 = 1/base^0 = 1` regardless of what `base` or `head_dim` are — the fastest possible rotation rate this formula can produce, and every other pair index rotates strictly slower than it, by construction of the formula rather than by coincidence of the chosen constants.

**4.** The two cases use different ABSOLUTE positions for both Q (5 versus 2) and K (3 versus 0), so the specific rotation angle applied to each vector genuinely differs between the two cases — what stays equal is only the DIFFERENCE `m - n = 2` in both. RoPE's rotated dot product `Q'.K' = Q . R((m-n)*theta) . K` depends algebraically on that difference alone, never on `m` or `n` individually, which is exactly the property this test isolates: two pairs of absolute positions that share nothing except their relative gap still produce identical dot products, because the math was built specifically to discard everything except that gap.

**5.** Using this question's own numbers, `group_size = n_heads_q / n_heads_kv = 8/2 = 4`. `kv_h = h / group_size` (integer division) maps query heads 0, 1, 2, 3 to `0/4=0, 1/4=0, 2/4=0, 3/4=0` — all four to KV head 0 — and query heads 4, 5, 6, 7 to `4/4=1, 5/4=1, 6/4=1, 7/4=1` — all four to KV head 1. The mapping is contiguous blocks of `group_size` consecutive query heads per KV head, not an interleaved or scattered assignment.

**6.** In IEEE-754 single precision, the largest finite value is roughly `3.4x10^38`, while `exp(300)` is astronomically larger than that (`e^300` is on the order of `10^130`) — so `exp(300)` alone, computed without any shift, overflows to `Inf` immediately, and once even one term in a softmax is `Inf`, the normalization step (`Inf / (Inf + finite + finite)`) produces `NaN`, not a large-but-valid probability. Subtracting the maximum first (`300` in this case) makes every exponent argument `score - max <= 0`, so every call to `exp()` returns a value in `(0, 1]` with no overflow anywhere; the shift is exact rather than approximate because it multiplies every term in both the numerator and denominator by the identical constant `exp(-max)`, which cancels completely in the final ratio — the probabilities that come out are bit-for-bit the same probabilities the unshifted formula would have produced, if that formula could have been evaluated at all.

**7.** `T_A` is born at op 0 and last read at op 1 (interval `[0,1]`); `T_C` is born at op 3 and last read at op 4 (interval `[3,4]`) — these two intervals do not overlap at all, so `T_C` can safely reuse the exact bytes `T_A` occupied, since nothing will ever read `T_A`'s old value once `T_C` has been written into that same memory. `T_B` is born at op 1 and last read at op 3 (interval `[1,3]`), which overlaps BOTH `T_A`'s interval (at the boundary op 1) and `T_C`'s (at the boundary op 3) — `T_B` is still the tensor actively being read when `T_C` is created, so giving `T_C` `T_B`'s memory would corrupt data `T_B`'s own consumer (the operation at op 3) still needs to read.

**8.** An argument about correctness or determinism — for instance, "does the graph produce the exact same output on every replay given the same input" — cannot be settled by a byte count at all, because a byte count describes how much memory a plan uses, not whether the computation the plan supports is correct; that specific kind of claim needs the actual replayed output values checked against expected ones (exactly what this section's Test 1 and Test 2 do), not a memory measurement standing in for it. The byte-count-over-timing substitution this section describes applies specifically to PERFORMANCE and MEMORY arguments, not to correctness ones.

**9.** A function that silently truncates a too-large request to whatever floats remain in the file would hand back a tensor view that is a different, smaller SHAPE than the caller asked for, with no signal that anything unusual happened — code downstream that expects `count` elements (because that is what it requested) would then read past the truncated span's actual end, right back into the same kind of undefined behavior the bounds check exists to prevent, just one level removed and considerably harder to trace back to its actual cause. Throwing immediately, at the one point where the real problem (a file that does not contain what the caller expected) is still directly diagnosable, is what actually prevents the out-of-bounds read rather than merely relocating it.

**10.** In Section 3.4, the memory an `mdspan`-based KV-cache view describes was allocated by this program's own `std::vector` (ordinary heap memory the program owns and will `free` itself) — what makes it worth calling out is that `submdspan` never copies it, not that the memory came from anywhere unusual. In Section 3.6, the memory an `mmap`'d `std::span` describes was never allocated by this program's heap at all; it is backed directly by the operating system's page cache for a file on disk, paged in lazily on first access and owned, in the fullest sense, by the kernel rather than by this process's own allocator.
