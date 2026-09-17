# Chapter 30: Mathematical Foundations for Kernel Authors: FLOPs, the Roofline, and the Hessian

**What you will understand by the end of this chapter:**

- Why a dot product's own real arithmetic intensity is a fixed constant regardless of length, why GEMV (the operation autoregressive decode actually runs) is intrinsically memory-bound at any model size, and the exact real batch size at which GEMM crosses over from memory-bound to compute-bound on a given machine.
- The real algebraic identity that makes naive softmax overflow, why the fix works, and why the identical bug reappears in an even more dangerous, silent form in log-sum-exp.
- The real trigonometric identity that explains, rather than merely implements, why RoPE's position-dependent rotation encodes relative position in a real dot product.
- Why quantization is a real affine map whose only non-linear step is rounding, the real, provably tight error bound that follows from it, and the real non-associativity that makes re-quantization a genuinely different operation from quantizing once.
- How a layer's real Hessian, computed directly from calibration data, gives GPTQ the real second-order information needed to compensate not-yet-quantized weights for the error already introduced by the ones that are -- proven, not merely asserted, to beat naive independent rounding.
- How to combine every real formula in this chapter into a complete roofline analysis of an entire real transformer decoder layer, and to derive the real batch size at which the whole layer's own bound classification flips.

**What you need to know first:**

- Chapter 8's own Roofline Model (peak FLOPs, peak bandwidth, and the ridge point separating memory-bound work from compute-bound work) is the real framework this chapter builds directly on top of -- this chapter derives the actual formulas Chapter 8 introduced conceptually, and applies them to real transformer operations.
- Chapter 3's own working RoPE implementation and Chapter 4's own working affine quantization are both real, correct code this chapter does not repeat -- Section 30.3 and Section 30.4 instead derive and prove WHY those working implementations behave the way they do.
- Section 19.4's and Section 22.3's own real least-squares reuse, and this book's general habit of implementing an independently-famous real algorithm from scratch and verifying it against known values (Sections 23.1, 23.2, 23.4), both recur here: Section 30.5's own from-scratch Gauss-Jordan matrix inverse and GPTQ compensation formula follow the identical discipline.

---

Every chapter before this one built a working system and verified it against the correct answer. This chapter asks a different question of the same material: not "does it work," but "why does it work, and what does that explain about its own real limits." A dot product's own arithmetic intensity explains why decode is slow no matter how fast the GPU is. A rotation matrix's own algebra explains why RoPE encodes relative position at all. A Hessian explains why GPTQ beats naive rounding, not just that it does. Part 6 exists to give the kernel-level intuition Parts 1 through 5 relied on without deriving, and this chapter is where that derivation happens -- entirely in real, checkable, from-scratch C++, exactly like every chapter before it.

## 30.1 The Dot Product as Inference's Atomic Unit, and the Real GEMV/GEMM Crossover

### Intuition

Every matrix multiply in a transformer decomposes into real dot products, and a real dot product's own arithmetic intensity -- FLOPs moved per byte read -- never improves no matter how long the vectors get. That single, unglamorous fact is the entire reason autoregressive decode is memory-bound, and it is what makes GEMM's own real transition to compute-bound, as batch size grows, a genuinely derivable number rather than a rule of thumb.

### The Concept, In Detail

Test 1 confirms the dot product's own constant arithmetic intensity directly: an 8-element and an 8000-element dot product both compute to exactly 0.25 FLOPs/byte. Test 2 extends this to GEMV -- an M x N matrix against an N-length vector -- and confirms its own real arithmetic intensity approaches exactly 0.5 as the matrix grows, which is the real, precise reason a batch-1 GEMV (decode) never becomes compute-bound simply because the model gets bigger: bigger M and N do not change the ratio.

GEMM is genuinely different, and Test 3 derives why: for a batch of M rows through a K x N weight matrix, when M is small relative to K and N (the real, ordinary case for LLM serving), the weight matrix's own bytes dominate real traffic, producing the clean closed-form result AI(M) = M/2 -- arithmetic intensity that actually grows with batch size. Test 3 also honestly quantifies how far this approximation drifts from an exact byte accounting as M grows, rather than presenting it as exact everywhere. Test 4 closes the section by deriving a real, concrete crossover batch size -- the exact point where a stated real machine's own ridge point (Chapter 8) is reached -- from stated real hardware numbers end to end.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover.cpp -o 01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover
./01_dot_product_arithmetic_intensity_and_gemv_gemm_crossover
```

**Sample input:** a dot product's own arithmetic intensity checked to be an identical constant at two very different lengths; GEMV's own arithmetic intensity checked to converge toward exactly 0.5 as its real shape grows; GEMM's own weight-dominated M/2 approximation checked against an exact byte accounting, with the approximation's own real drift honestly quantified; and a real, stated machine's own crossover batch size derived end to end from its stated peak compute and peak bandwidth figures.

@@OUT1@@

!!! warning "[COMMON TRAP] assuming a bigger model changes GEMV's own real memory-bound verdict"
    It is tempting to think a sufficiently large model, with enough real FLOPs per token, must eventually become compute-bound even during single-token decode. Test 2 shows precisely why that intuition is wrong: GEMV's own real arithmetic intensity converges toward a FIXED ceiling of exactly 0.5 FLOPs/byte as the matrix grows -- it does not keep climbing. A bigger model has proportionally more real FLOPs AND proportionally more real bytes to read, and those two quantities grow together, leaving the ratio essentially unchanged. The only real lever that moves arithmetic intensity for a linear layer is the BATCH size, not the model size -- which is exactly why real serving systems build continuous batching (this book's own next chapter) rather than simply hoping a bigger GPU fixes decode-time memory-boundedness on its own.

## 30.2 Numerically Stable Softmax: The Log-Sum-Exp Fix

### Intuition

Softmax's own textbook definition is exactly correct mathematically and dangerously wrong to implement literally: a single sufficiently large real logit overflows a real floating-point exponential before the actual probability computation is finished.

### The Concept, In Detail

Test 1 reproduces this real failure deliberately, in real 32-bit float -- the precision real production kernels actually run: `naive_softmax_f32({1.0, 2.0, 100.0})` produces a genuine NaN for index 2, the one entry that should hold nearly all the real probability mass, because `exp(100.0f)` overflows float32 to `+inf` and `inf/inf` is undefined. Test 2 confirms the real, shift-invariant fix -- subtracting `max(x)` before exponentiating, which is mathematically identical to the original by real algebraic cancellation -- produces a fully valid distribution on the identical input. Test 3 confirms naive and stable softmax genuinely agree on safe-magnitude input, proving they are the same real function differing only in robustness.

Test 4 and Test 5 apply the identical shift-invariant identity to log-sum-exp, and Test 5 is this section's own central, sharper point: `naive_log_sum_exp` on a sufficiently large real input does not crash and does not produce a NaN -- it silently returns `+inf`, a plainly wrong finite-valued answer masquerading as a real number, which is a more dangerous failure mode than Test 1's NaN precisely because nothing about it looks obviously broken to a caller who never separately checked.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_numerically_stable_softmax_and_log_sum_exp.cpp -o 02_numerically_stable_softmax_and_log_sum_exp
./02_numerically_stable_softmax_and_log_sum_exp
```

**Sample input:** naive softmax in real float32 checked to produce a genuine NaN on a real large-magnitude logit; stable softmax on the identical input checked to produce a fully valid distribution summing to exactly 1.0; naive and stable softmax checked to agree on safe-magnitude input; naive and stable log-sum-exp checked to agree on safe-magnitude input; and naive log-sum-exp on a real large-magnitude input checked to silently return +inf while stable log-sum-exp on the identical input returns the correct, genuinely finite value.

@@OUT2@@

!!! warning "[COMMON TRAP] treating the absence of a NaN as proof a numerical computation is correct"
    Test 1's naive softmax fails loudly enough that a NaN check would catch it immediately. Test 5's naive log-sum-exp is the more instructive real failure precisely because it does NOT fail loudly: `+inf` is a normal, valid-looking `double`, and a system that only checks `std::isnan` on its own outputs would let this real bug through completely undetected, silently corrupting every downstream computation that treats that `+inf` as a legitimate log-probability. The real lesson is not "check for NaN" -- it is that a numerically unstable formula can fail in whatever way is easiest for it to fail, and the only real fix is the shift-invariant identity itself, applied everywhere the unstable formula would otherwise be used, not a downstream check for one specific symptom.

## 30.3 RoPE's Rotation Math

### Intuition

Chapter 3 implemented RoPE as working code. This section asks why rotating a query and a key by their own real, position-dependent angles before taking their dot product actually encodes relative position at all -- and the answer turns out to be a clean, provable real trigonometric identity, not a fortunate coincidence.

### The Concept, In Detail

Test 1 confirms the rotation matrix itself against exact, hand-verifiable real angles. Test 2 is the section's own real mathematical core: `R(a)` transposed, composed with `R(b)`, equals exactly `R(b - a)` -- checked as a genuine 2x2 matrix equality, not merely asserted -- because a rotation matrix is real and orthogonal, so its own transpose is its own inverse.

Test 3 turns that identity into RoPE's own central, load-bearing property: the real dot product of a rotated query at position m and a rotated key at position n depends only on the real relative offset `(m - n)`, confirmed directly by showing three genuinely different absolute position pairs sharing the identical offset produce numerically identical dot products, while a pair with a different offset produces a measurably different one. Test 4 bridges the two: the concrete multi-position computation from Test 3 is confirmed to equal `q^T * R(theta * (n - m)) * k`, computed directly via the abstract identity from Test 2 -- proving RoPE's own real relative-position property is a direct algebraic consequence of rotation composition, not an empirical accident.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_rope_rotation_math.cpp -o 03_rope_rotation_math
./03_rope_rotation_math
```

**Sample input:** the rotation matrix checked against exact real angles (0, pi/2, pi radians); the rotation-composition identity R(a)^T * R(b) = R(b - a) checked as a real 2x2 matrix equality across several angle pairs; RoPE's own relative-position property checked directly across multiple absolute position pairs sharing an identical real offset, and against a pair with a genuinely different offset; and the concrete rotated dot product checked to match the abstract identity applied directly.

@@OUT3@@

!!! warning "[COMMON TRAP] treating RoPE's relative-position property as something to verify only empirically"
    It is possible to convince yourself RoPE "seems to" encode relative position by trying a few position pairs and noticing the dot products look related. Test 2 and Test 4 exist to do something stronger: derive the REASON algebraically (rotation composition reduces to a single angle subtraction, because a rotation matrix's transpose is its own real inverse) and then confirm the concrete numerical computation matches that abstract identity exactly. The difference matters because an empirical-only check cannot tell you whether the property holds for every possible position pair or only the ones you happened to try; the algebraic identity, once confirmed to match the concrete computation, guarantees it holds for all of them.

## 30.4 Quantization as Affine Algebra: Real Error Bounds and the Optimal Scale

### Intuition

Chapter 4 implemented affine quantization as working code. Treating it explicitly as an affine map -- `quantize(x) = round(x / scale) + zero_point` -- makes its own real error bound and its own real failure to compose across repeated quantization both directly provable rather than merely observed.

### The Concept, In Detail

Test 1 confirms the real optimal scale formula and an exact round-trip at both endpoints of a stated range. Test 2 is a real, general property check, not a hand-picked example: across 1001 real sample points spanning a stated range, not one is ever reconstructed more than `scale/2` away from its own true value -- the real error bound rounding produces. Test 3 goes further and confirms this bound is genuinely TIGHT: a real value placed exactly at a quantization bin's own midpoint reconstructs with an error close to the full `scale/2` bound, not comfortably inside it.

Test 4 is this section's own sharper, more consequential point: re-quantization does not commute. Quantizing a real value once, directly, at a coarse scale can produce a genuinely different integer code than quantizing it finely first, dequantizing, and then quantizing that result at the identical coarse scale -- a real, checkable non-associativity with direct consequences for any system, including this book's own Chapter 14 streaming re-quantization, that quantizes more than once.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_quantization_as_affine_algebra.cpp -o 04_quantization_as_affine_algebra
./04_quantization_as_affine_algebra
```

**Sample input:** the optimal scale formula and an exact endpoint round-trip checked against a stated range; the real scale/2 error bound checked across 1001 sample points spanning a stated range with zero violations; the bound's own tightness checked at a real bin midpoint; and re-quantization's own real non-associativity checked by comparing a direct coarse quantization against a fine-then-coarse round trip on the identical original value.

@@OUT4@@

!!! warning "[COMMON TRAP] assuming re-quantizing an already-quantized value is the same as quantizing the original"
    A system that dequantizes a value to re-scale it -- converting from one bit width to another, say, during a real re-quantization pass -- might assume the result is equivalent to having quantized the true original value directly at the new scale, since dequantization is "supposed to" recover the original. Test 4 shows this assumption is false in general: the intermediate fine-grained rounding step introduces its own small real error, and that error can be just enough to push the value across a coarser bin's own boundary, landing on a genuinely different final code than a direct quantization would have. Any real system that quantizes more than once -- exactly the situation Chapter 14's own streaming re-quantization is built to handle carefully -- has to treat this as a real, accumulating source of error, not something dequantization quietly undoes.

## 30.5 The Hessian's Role in GPTQ: Second-Order Error Compensation

### Intuition

Quantizing every weight independently, as Section 30.4's own affine map does in isolation, ignores real information a calibration dataset already provides: once one weight is quantized, the weights that are not yet quantized can be nudged to compensate for the error that quantization just introduced. GPTQ's real insight is that the layer's own Hessian is exactly the second-order information needed to compute that compensation optimally.

### The Concept, In Detail

Test 1 and Test 2 build this section's own real, from-scratch linear-algebra foundation: `H = 2 * X^T * X`, computed directly from a tiny real calibration dataset, matches an exact hand computation, and a real, general Gauss-Jordan inverse of that Hessian is confirmed correct not by trusting the algorithm but by checking `H * H^-1` is genuinely the identity matrix directly.

Test 3 confirms the real GPTQ compensation formula -- `delta_w_f = -(e_p / [H^-1]_pp) * [H^-1]_fp` -- against an exact hand computation for a single quantized weight's effect on the one remaining weight. Test 4 is this section's own central, real proof: running the full sequential GPTQ pipeline (quantize, compute the real error, compensate every remaining weight, repeat) on a fully hand-traceable two-weight example produces a genuinely different final quantized weight than naive independent rounding, and a strictly LOWER real total squared output error over the calibration data -- 0.14 for GPTQ's compensated result against 0.24 for naive rounding on the identical original weights. GPTQ's real benefit is not asserted here; it is computed and compared directly.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_hessian_role_in_gptq.cpp -o 05_hessian_role_in_gptq
./05_hessian_role_in_gptq
```

**Sample input:** the real Hessian H = 2 * X^T * X checked against an exact hand computation over a tiny calibration dataset; a real, general Gauss-Jordan matrix inverse checked against an exact hand computation, with H * H^-1 checked directly against the identity matrix; the real GPTQ compensation formula checked against an exact hand computation for a single weight; and the full GPTQ pipeline checked end to end against naive independent rounding on a fully hand-traceable two-weight example, confirming a strictly lower real total squared output error.

@@OUT5@@

!!! warning "[COMMON TRAP] assuming Hessian-based compensation always helps a LATER weight, never a chain of them"
    It is easy to read GPTQ's own compensation formula and assume its benefit is limited to the single, immediately adjacent weight it updates. Test 4's own real pipeline shows the mechanism is genuinely sequential and cumulative: quantizing weight 0 changes the value weight 1 is compensated toward, and in a layer with more than 2 weights, quantizing weight 1 (now itself already nudged once) would go on to compensate weight 2, and so on. Each real compensation step uses the CURRENT state of the not-yet-quantized weights, not the original ones -- which is exactly why GPTQ processes weights in a specific real sequence rather than computing every compensation independently up front from the unquantized original values.

## 30.6 A Full Roofline Analysis of a Transformer Layer

### Intuition

Every real formula this chapter derived -- arithmetic intensity, weight-dominated byte accounting, the roofline crossover -- was built on a single operation at a time. This capstone section applies all of them together to an entire real transformer decoder layer, to classify the WHOLE layer's own real bound and to derive the exact batch size where that classification flips.

### The Concept, In Detail

Test 1 confirms every real FLOP sub-total (QKV projection, attention, output projection, FFN) and the real weight-byte total for a tiny, fully hand-traceable layer shape, matching a direct hand computation exactly: 576 total FLOPs, 512 total weight bytes, an arithmetic intensity of exactly 1.125. Test 2 confirms a real, general algebraic property this section's own crossover formula depends on: doubling batch size exactly doubles both total FLOPs and arithmetic intensity, while weight bytes -- which do not depend on batch at all -- stay exactly unchanged.

Test 3 applies Section 30.1's own roofline classification to the WHOLE layer: against a stated real ridge point of 2.0, the identical layer classifies `MEMORY_BOUND` at batch 1 and `COMPUTE_BOUND` at batch 2. Test 4 confirms the real, closed-form crossover formula predicts precisely this observed transition rather than merely rationalizing it afterward, and Test 5 confirms the formula's own real algebraic correctness across several genuinely different stated ridge points at once.

### Code and Verification

@@CODE6@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 06_full_transformer_layer_roofline_analysis.cpp -o 06_full_transformer_layer_roofline_analysis
./06_full_transformer_layer_roofline_analysis
```

**Sample input:** every real FLOP and weight-byte sub-total for a tiny, fully hand-traceable transformer layer shape, checked against an exact hand computation; batch-doubling checked to exactly double FLOPs and arithmetic intensity while leaving weight bytes unchanged; the whole layer's own bound classification checked to flip from memory-bound to compute-bound between batch 1 and batch 2 against a stated ridge point; the closed-form crossover batch size checked to fall exactly between those two batch sizes; and the crossover formula's own algebraic correctness checked across 4 different stated ridge points.

@@OUT6@@

!!! warning "[COMMON TRAP] analyzing one operation's own roofline classification and assuming it tells you the whole layer's"
    A transformer layer is not a single GEMM -- it is 4 real sub-blocks (QKV projection, attention, output projection, FFN), each with its own real FLOP and byte profile, and nothing guarantees they all cross the ridge point at the identical batch size. This section's own capstone works because every one of those 4 sub-blocks happens to scale identically with batch (linearly, since each one's own real formula has an explicit or implicit batch factor) while none of their weight bytes depend on batch at all -- which is what allows a SINGLE whole-layer arithmetic intensity, and a single whole-layer crossover batch size, to exist at all. A real architecture with a sub-block that scales differently (for instance, a mixture-of-experts layer whose active weight bytes themselves depend on which real experts a given batch routes to) would need its own, separately derived crossover analysis rather than reusing this section's own single-formula shortcut.

## Chapter Summary

This chapter derived the real mathematics Parts 1 through 5 relied on without deriving. Section 30.1 showed a dot product's own arithmetic intensity is a fixed constant, explaining why GEMV-based decode is intrinsically memory-bound, and derived the real batch size at which GEMM crosses over to compute-bound. Section 30.2 reproduced softmax's real overflow failure on purpose, fixed it with a real shift-invariant identity, and showed the identical bug reappears silently, as a wrong finite value rather than a NaN, in log-sum-exp. Section 30.3 proved RoPE's own relative-position property follows directly from a real rotation-composition identity. Section 30.4 treated quantization as a genuine affine map, deriving its own provably tight error bound and its own real non-associativity under repeated quantization. Section 30.5 built a real Hessian and a real Gauss-Jordan inverse from scratch to prove, on a fully hand-traceable example, that GPTQ's second-order compensation produces strictly lower real error than naive independent rounding. Section 30.6 closed the chapter by combining every one of these real formulas into a complete roofline analysis of an entire transformer decoder layer.

## Self-Check Questions

1. Section 30.1 shows GEMV's own arithmetic intensity converges toward exactly 0.5 as the matrix grows, rather than continuing to increase. Explain, in terms of what grows in the numerator versus the denominator, why a bigger matrix alone can never push GEMV past this ceiling.
2. Section 30.1's Test 3 shows the weight-dominated M/2 approximation's own real error grows as batch size M increases. Explain concretely which bytes the approximation ignores, and why ignoring them matters more at larger M.
3. Section 30.2's Test 5 describes naive log-sum-exp's silent +inf as a MORE dangerous failure than naive softmax's NaN in Test 1. Explain concretely why a system that only checks for NaN would not catch this failure.
4. Section 30.3's Test 2 confirms R(a)^T * R(b) = R(b - a) as a real matrix identity. Explain what specific property of a rotation matrix (true of rotation matrices in general, not just this one) is what makes its own transpose equal its own inverse.
5. Section 30.4's Test 3 constructs a real value exactly at a quantization bin's own midpoint to demonstrate the scale/2 bound is tight. Explain why a value chosen randomly within the range, rather than at a midpoint, would be much less likely to demonstrate this.
6. Section 30.4's Test 4 shows re-quantization does not commute. Construct, in your own words, a concrete real scenario (outside this section's own hand-picked example) in a serving system where this specific non-associativity could silently degrade model quality over time.
7. Section 30.5's GPTQ compensation formula divides by `[H^-1]_pp`. Explain, in terms of what the Hessian represents about a weight's own real sensitivity, what a very LARGE value of `[H^-1]_pp` would suggest about that weight, and how that would affect the resulting compensation.
8. Section 30.5's Test 4 processes the 2 weights in a SPECIFIC order (index 0 before index 1). Explain why processing them in the opposite order could produce a genuinely different final result.
9. Section 30.6's whole-layer crossover analysis depends on every sub-block scaling identically with batch size. Name one real architectural change to a transformer layer (not necessarily mixture-of-experts) that could break this assumption, and explain concretely why.
10. This chapter is titled "Mathematical Foundations for Kernel Authors." Choose any ONE of this chapter's 6 sections and explain concretely how the specific mathematical property it derives would change a real decision a kernel author makes when writing or optimizing an actual transformer inference kernel.

## Where We Go Next

This chapter derived the mathematics; the next two chapters put it to work at serving scale. Chapter 31 builds a real continuous-batching scheduler from scratch, directly exploiting Section 30.1's own real batch-size-dependent arithmetic intensity to keep a serving system as close to compute-bound as real traffic allows, and adds real numerical debugging tools -- NaN-propagation tracing and floating-point drift detection -- for catching the bugs Section 30.2's own numerical instability foreshadowed, at a scale where they only appear under real production load. Chapter 32 closes the book by taking this book's own inference engine to the GPU, building a real Flash Attention implementation around the same online-softmax idea Section 30.2 introduced, and a real CUDA production engine for the edge devices that carry a small GPU.

## Worked Solutions

**1.** GEMV's real arithmetic intensity is `2*M*N / (M*N*4 + N*4 + M*4)`. As M and N both grow, the numerator and the dominant `M*N*4` term in the denominator both grow proportionally to `M*N`, so their ratio approaches a fixed constant (`2 / 4 = 0.5`) rather than continuing to increase -- the `N*4` and `M*4` terms (the vector and output bytes) shrink to a vanishing fraction of total bytes as the matrix grows, but the matrix term itself scales in lockstep with the FLOPs, so there is no way for a bigger matrix alone to change the ratio's own real limit.

**2.** The approximation counts only the weight matrix's own bytes (`K*N*4`) and ignores the batch's own input bytes (`M*K*4`) and output bytes (`M*N*4`). At M=1, those ignored bytes are a tiny fraction of the weight bytes, so the approximation is nearly exact. As M grows, the ignored input and output bytes grow linearly with M while the weight bytes stay fixed -- so the ignored bytes become a progressively larger real fraction of total traffic, and the approximation's own overestimate of arithmetic intensity (since it undercounts the true denominator) grows correspondingly larger.

**3.** A system checking only `std::isnan` would see `naive_log_sum_exp`'s own `+inf` result, note that `std::isnan(+inf)` is false, and conclude the computation succeeded -- `+inf` is a normal, valid-looking `double` value that simply happens to be wrong. Catching this failure requires either using the always-correct stable formula in the first place, or separately checking `std::isinf`, which is an easy check to omit precisely because `+inf` does not "look like" an error the way a NaN does.

**4.** A rotation matrix is orthogonal: its own columns (and rows) are unit vectors that are pairwise perpendicular. For any real orthogonal matrix, the general linear-algebra identity `A^T * A = I` holds, which is exactly the statement that `A^T` is `A`'s own real inverse. This is true of every real rotation matrix, in any number of dimensions, not merely the 2D case this section works with -- it follows from what a rotation actually IS (a transformation that preserves lengths and angles), not from any property specific to this section's own particular angles.

**5.** A randomly chosen value within the range is, with high probability, somewhere between a bin's own center and its edge, and the resulting error is typically much smaller than `scale/2` -- most real values do not happen to land exactly at the worst-case point. Only a value constructed deliberately at the exact midpoint between two adjacent quantization levels is guaranteed to sit at the real maximum possible distance from both, which is precisely why Test 3 constructs that value on purpose rather than sampling one randomly and hoping it happens to be near the boundary.

**6.** Consider a real KV cache using Chapter 14's own streaming re-quantization: cached keys and values, once written at a fine-grained scale, get re-quantized to a coarser scale as they age out of a hot window to save memory. If this re-quantization is applied repeatedly -- perhaps a value gets moved between cache tiers more than once as access patterns shift -- each individual re-quantization step is a fresh dequantize-then-quantize round trip, and Section 30.4's own Test 4 shows this is NOT equivalent to quantizing the true original value directly at the final coarse scale. Over enough repeated tier transitions, this could silently accumulate more real error than a single, correctly-designed direct re-quantization from the original cached value would have produced.

**7.** `[H^-1]_pp` being large means that, from the Hessian's own real perspective (built from calibration data `X`), weight `p`'s own contribution to the layer's output is comparatively insensitive to small perturbations in that weight relative to the other weights -- intuitively, the calibration data does not "notice" weight `p` moving very much. Since the compensation formula divides by `[H^-1]_pp`, a very large value there would make the resulting `delta_w_f` correction SMALL (dividing by a large number), meaning a weight the Hessian considers relatively unimportant produces a smaller real compensation to the remaining weights when it is quantized -- exactly the intuitively correct behavior, since a change to an insensitive weight has less real output error to compensate for in the first place.

**8.** The compensation formula updates the remaining, not-yet-quantized weights using the CURRENT values of `H^-1` restricted to those remaining indices, and each subsequent weight is quantized using its own value AFTER any prior compensation has already been applied to it. If weight 1 were processed before weight 0, weight 0 would instead be the one receiving compensation based on weight 1's own quantization error, and since the two weights' own real errors (`e_0` and `e_1`) are generally different, and the `H^-1` submatrix used to compensate differs depending on which index is being treated as "already quantized," the sequence of quantized values -- and therefore the final total output error -- would generally come out genuinely different.

**9.** A mixture-of-experts FFN, where each real token is routed to only a small subset of a much larger set of expert weight matrices, breaks this assumption directly: the ACTIVE weight bytes a given batch actually touches depend on which real experts that batch's own tokens route to, not on a single fixed weight tensor size the way this section's own dense FFN does. A batch of tokens that happens to route to many different real experts touches far more total weight bytes than a batch that concentrates on a few, so the layer's own real arithmetic intensity for a mixture-of-experts FFN is not a clean function of batch size alone the way this section's dense capstone example is -- it also depends on the real routing decisions made at that specific batch, which this section's single whole-layer formula has no way to account for.

**10.** Section 30.1's own real crossover batch size directly informs a genuinely practical decision: a kernel author building a serving system's own batching scheduler (this book's next chapter) can use the exact derived crossover point to decide how aggressively to batch requests together before dispatching a GEMM kernel -- below the crossover batch, the kernel is memory-bound and further optimizing its own compute (say, using a more arithmetically efficient but more complex kernel) buys little real speedup, since the bottleneck is bandwidth; above the crossover, the kernel is compute-bound and the opposite investment (a more compute-efficient kernel, even at some cost to memory access patterns) becomes the right real engineering trade-off. Without this section's own derived formula, a kernel author would be guessing at this trade-off rather than computing it directly from the machine's own stated peak compute and peak bandwidth figures.
