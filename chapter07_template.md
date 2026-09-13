# Chapter 7: TurboQuant -- Online Vector Quantization with Near-Optimal Distortion

**What you will understand by the end of this chapter:**

- Why a random rotation is not decoration but the load-bearing trick that makes vector quantization tractable: it converts the worst possible input (all energy in one coordinate) into an average-case input (energy spread evenly across every coordinate), verified directly by rotating a spike vector and measuring how far its energy actually spreads, and by confirming empirically that a rotated coordinate's variance matches the theoretical value the rest of the chapter depends on.
- Why that rotation must be genuinely orthogonal and not merely "random-looking" — a raw Gaussian matrix scrambles a vector just as thoroughly as a proper rotation does, but only a proper rotation can be undone by its own transpose, and this chapter measures exactly how badly reconstruction breaks when that property is missing.
- How the Lloyd-Max algorithm turns a known probability distribution into a provably optimal quantizer for that distribution, and why this beats Chapters 2-6's uniformly-spaced blockwise bins whenever the underlying values are not uniformly distributed to begin with.
- Why a quantizer built to minimize reconstruction error is not automatically safe to use for inner products — it can be systematically, not just randomly, wrong — and how a second, independent 1-bit correction (QJL) restores unbiasedness without needing a better quantizer for the first stage.
- How to apply the whole pipeline to the KV cache, the one part of a running transformer that blockwise quantization was never well suited for: it grows online, one vector at a time, with no calibration data available in advance, which is exactly the setting TurboQuant's data-oblivious codebooks were designed for.
- Where TurboQuant stops being the right tool: Chapters 2-6's blockwise Q4_0/Q8_0 remain the correct choice for static weight matrices, because TurboQuant's O(d^2) per-vector rotation cost does not belong inside a fused SIMD matmul inner loop the way a per-block integer scale does.

**What you need to know first:**

- Chapters 2-4's blockwise Q8_0/Q4_0 quantization, its fixed 32-element blocks, and its per-block scale — this chapter is defined mostly by contrast with that approach, and Section 7.5 measures the two head to head.
- Basic familiarity with probability densities and expectation is helpful for Sections 7.2 and 7.3, though every formula used is derived or numerically verified in code rather than assumed.

---

Every quantizer built so far in this book has worked one 32-element block at a time: find the block's largest magnitude, pick a scale, round each of the 32 values independently. That approach makes no use of the fact that those 32 values are part of a much longer vector living somewhere specific in space — it treats each block as an isolated signal with no relationship to its neighbors. TurboQuant, the subject of this chapter, throws that assumption away. It treats an entire vector as one geometric object on a high-dimensional sphere, applies a single principled transformation to that whole object, and only then quantizes coordinate by coordinate — but with a quantizer that has been custom-built for the exact statistical shape those coordinates are now guaranteed to have. Section 7.1 builds and verifies the rotation step that makes this possible. Section 7.2 builds the optimal per-coordinate quantizer the rotation sets up, combining the two into TurboQuant's reconstruction-optimal "MSE mode." Section 7.3 shows that MSE mode, despite being optimal for reconstruction, is quietly biased for a different and equally important task — estimating inner products — and derives the 1-bit correction that fixes it. Section 7.4 puts the resulting quantizer to work on the KV cache, the online, uncalibrated, ever-growing structure that blockwise quantization was never well matched to. Section 7.5 closes the chapter by comparing TurboQuant against Chapters 2-6's blockwise approach directly, on the same vectors, and checks TurboQuant's own distortion against the information-theoretic floor no quantizer of any kind can beat.

## 7.1 Random Rotation: Turning Worst-Case Vectors into Average-Case Ones

### Intuition

Imagine trying to quantize the vector `[1, 0, 0, ..., 0]` — all of its length concentrated in a single coordinate — with one bit per coordinate. The first coordinate gets a bit's worth of useful information; every other coordinate is exactly zero, and a 1-bit quantizer has no representable value for "exactly zero," so those bits are wasted entirely. This is not a contrived edge case to a coordinate-wise quantizer; it is the worst input such a quantizer can ever see, and any fixed set of quantization bins can be defeated by some adversarial arrangement of a vector's energy. TurboQuant's answer is to make that adversarial arrangement impossible to construct in the first place: multiply every vector, before quantizing it, by the same random orthogonal matrix. A famous fact about high-dimensional geometry does the rest — after a genuinely random rotation, a unit vector's energy is spread almost perfectly evenly across every coordinate, regardless of how concentrated it was before the rotation.

### The Concept, In Detail

A matrix is orthogonal if multiplying by it never changes a vector's length or the angle between any two vectors — formally, `Pi^T Pi = I`, which is exactly the property that makes `Pi^T` the matrix's own inverse. The standard way to build a genuinely random ("Haar-distributed") orthogonal matrix is to start with a matrix of independent Gaussian entries and orthogonalize its columns, one against the others, using Gram-Schmidt — this is the same computation QR decomposition performs, and the resulting orthogonal factor is proven to be uniformly distributed over the space of all rotations. This chapter builds that construction directly rather than calling a linear-algebra library, both because it is short enough to verify by hand and because seeing exactly where the orthogonality comes from matters for understanding why skipping it breaks everything downstream. Two properties of the resulting matrix `Pi` matter for the rest of the chapter: it preserves length (`||Pi x|| = ||x||`), and its transpose exactly undoes it (`Pi^T (Pi x) = x`), so quantizing in the rotated coordinate system and then rotating back afterward changes nothing about what a vector "means" — it only changes which coordinate system it is described in. The chapter also verifies something the rest of the theory depends on but no prior chapter needed: that a single coordinate of a rotated random unit vector really does behave the way the Beta-distribution theory (Section 7.2) predicts, with variance shrinking as `1/d`. This is not asserted from the literature — it is checked by rotating thousands of independent random unit vectors and measuring the empirical variance of one coordinate directly.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_rotation_and_beta_distribution.cpp -o 01_rotation_and_beta_distribution
./01_rotation_and_beta_distribution
```

**Sample input:** a 32x32 constructed rotation matrix, checked for orthogonality; a 64-dimensional random vector, rotated and rotated back to confirm an exact round-trip; a worst-case "spike" vector `[1, 0, ..., 0]` before and after rotation; 20,000 independent random unit vectors in 128 dimensions, used to measure the empirical variance of one rotated coordinate against the theoretical prediction; and a deliberately non-orthogonal raw Gaussian matrix used in place of a proper rotation.

@@OUT1@@

!!! warning "[COMMON TRAP] a random-looking matrix is not a rotation"
    A matrix of independent Gaussian entries scrambles a vector just as thoroughly as a genuine rotation does — multiplying by it produces a vector that looks nothing like the input. The difference only shows up when you try to undo it. A proper rotation's transpose is its exact inverse (`Pi^T Pi = I`), so rotating and then "rotating back" with the transpose reconstructs the original vector exactly, with no quantization involved at all. A raw Gaussian matrix `G` has no such guarantee — `G^T G` is not the identity — so "rotate with `G`, then rotate back with `G^T`" reconstructs a vector with a large, structural error, even before a single bit of quantization has happened. This is not a rounding error to be tolerated; it is a correctness bug, and it is the reason this chapter builds its rotation via explicit Gram-Schmidt orthogonalization rather than skipping straight to "any matrix full of random numbers will do."

## 7.2 Lloyd-Max Optimal Codebooks and the MSE Quantizer

### Intuition

Section 7.1 established that a coordinate of a rotated unit vector behaves like a draw from a known probability distribution — concentrated near zero, with a shape that depends only on the dimension `d`. Knowing the distribution in advance is a gift: instead of spacing quantization bins uniformly (as Chapters 2-6's blockwise formats do, since they have no distributional assumption to exploit), TurboQuant can place bins exactly where the probability mass actually is — densely near zero, sparsely in the rarely-visited tails — and prove that no other placement does better for a fixed number of bits. That placement problem is the Lloyd-Max algorithm: continuous, distribution-aware k-means.

### The Concept, In Detail

A coordinate of a random unit vector in `d` dimensions has density proportional to `(1 - x^2)^((d-3)/2)` on `[-1, 1]` — concentrated near zero, and more sharply so as `d` grows. The Lloyd-Max algorithm finds the `2^bits` centroids that minimize expected squared error against this density by alternating two steps until convergence: given the current centroids, the optimal boundaries are simply the midpoints between consecutive centroids (any value closer to one centroid than its neighbor should be assigned to that centroid); and given the current boundaries, the optimal centroid for each bin is the conditional mean of the density within that bin, computed here by numerical (trapezoidal) integration since the Beta density has no simple closed-form conditional mean. Because the target distribution depends only on `d` and `bits`, every codebook can be precomputed once, offline, before any real data arrives — this is what makes TurboQuant an online algorithm suitable for a KV cache that has no calibration pass available, unlike clustering-based vector quantization (Product Quantization), which needs to see representative data in advance. Combining the rotation from Section 7.1 with this codebook gives the complete "MSE mode" pipeline: normalize the input to the unit sphere and store its norm as one float (the codebook was solved for a UNIT-norm coordinate distribution, so the actual scale has to be factored out and reapplied separately), rotate, quantize each rotated coordinate to its nearest centroid, and store the indices. This normalize-then-rotate order is not optional bookkeeping — it is the step this section's own trap gets wrong.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_lloyd_max_mse_quantizer.cpp -o 02_lloyd_max_mse_quantizer
./02_lloyd_max_mse_quantizer
```

**Sample input:** Lloyd-Max codebooks solved numerically for `bits = 1..4` at `d = 64`, checked for sortedness and symmetry and, for `bits = 1`, checked against the closed-form large-`d` approximation; a complete TurboQuant MSE quantizer round-tripped on a random 64-dimensional vector; the same quantizer's reconstruction error measured across 50 vectors at each bit-width from 1 to 4; and a deliberately un-normalized input fed both through the correct pipeline and through a version that skips the normalize step.

@@OUT2@@

!!! warning "[COMMON TRAP] quantizing before normalizing"
    The codebook in this section was solved for a coordinate distribution with variance `1/d` — that is, for coordinates of a vector that is ALREADY on the unit sphere. Feeding the codebook a rotated coordinate from a vector with a large norm skips the one step that makes the codebook's calibration valid: the rotated coordinates land far outside the range the codebook was built for, and every one of them saturates at whichever outermost centroid is closest, since the codebook has no representable value out there. This is not the same failure mode as ordinary quantization rounding — rounding error stays proportional to the input, while a missing scale factor produces an error that grows without bound as the input's true norm grows, because the reconstruction is stuck near the unit sphere no matter how large the original vector actually was. "Always normalize first" is not a stylistic preference; it is the precondition the entire codebook calibration depends on.

## 7.3 The Inner-Product Bias Problem and the QJL Fix

### Intuition

Section 7.2's quantizer is provably optimal for one specific goal: minimizing the expected squared error between a vector and its reconstruction. Attention does not need a reconstructed vector to look at — it needs an accurate DOT PRODUCT between a query and a cached key. Those turn out to be different goals, and optimizing for the first does not automatically deliver the second. At low bit-widths, the MSE-optimal quantizer's inner-product estimate is not merely noisy around the true value — it is systematically, predictably too small, by a constant multiplicative factor that never fully disappears at any finite bit-width.

### The Concept, In Detail

The mechanism is best seen at the extreme: a 1-bit MSE quantizer for large `d` has just two centroids, `+/- sqrt(2/(pi*d))`, and it reconstructs every coordinate as one of those two values based only on the ROTATED coordinate's sign — all information about the original coordinate's magnitude within its half is thrown away. When that reconstruction is dotted with an arbitrary query vector, the expected value of the resulting estimate is not the true inner product but exactly `2/pi` (about 0.637) times it — a multiplicative shrinkage baked into the quantizer's own construction, not a symptom of any particular unlucky rotation. At higher bit-widths this shrinkage factor climbs toward 1.0, but it never reaches it at any finite bit budget. The fix does not require a better single-stage quantizer — it requires a second, independent stage aimed specifically at inner products. Writing `x = x_tilde_mse + r`, where `r` is the residual the MSE stage failed to capture, the true inner product decomposes as `<y, x> = <y, x_tilde_mse> + <y, r>`. The MSE stage already gives the first term; QJL (Quantized Johnson-Lindenstrauss) supplies the second by quantizing the RESIDUAL with a completely different, 1-bit-per-coordinate scheme: multiply by a fresh random Gaussian matrix `S`, keep only the sign of each projection, and dequantize with the rescaling `sqrt(pi/2)/d * S^T * signs`. That specific rescaling is what makes `<y, QJL_dequant(r)>` an unbiased estimator of `<y, r>` — proved by the symmetry of the Gaussian distribution — and adding it to the MSE term exactly cancels the MSE stage's own bias in expectation, at the cost of one additional bit per coordinate. The residual and the MSE reconstruction must be computed in the SAME coordinate system before they are combined — both in the original, un-rotated space — which is precisely the step this section's own trap gets wrong.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_qjl_unbiased_inner_product.cpp -o 03_qjl_unbiased_inner_product
./03_qjl_unbiased_inner_product
```

**Sample input:** a 1-bit MSE-only quantizer's inner-product estimate, averaged over 400 independent random rotations, compared against the theoretical `2/pi` shrinkage factor; QJL's own unbiasedness, averaged over 500 random projection matrices; the combined TurboQuant "Prod" quantizer's unbiasedness at 3 bits, averaged over 300 trials; the combined estimator's variance measured at bit-widths 2 through 4; and a deliberate demonstration of what happens when the residual is computed in rotated coordinates instead of original ones before being combined with the MSE term.

@@OUT3@@

!!! warning "[COMMON TRAP] combining the two stages in different coordinate systems"
    The MSE stage's quantization happens in ROTATED coordinates (that is where the Beta-distribution codebook applies), but its reconstruction is rotated back to ORIGINAL coordinates before anything else touches it. The residual that QJL quantizes must be computed in that same original coordinate system — `x_hat - x_hat_mse` — because QJL's own random matrix `S` and its unbiasedness proof are stated for the space the QUERY lives in, which is the original, un-rotated space. Subtracting in rotated coordinates instead (`y - y_tilde`, skipping the rotate-back step) and then adding the resulting QJL correction directly to the original-space MSE term mixes two vectors that are expressed in different bases — the correction and the query it gets dotted with are no longer describing the same coordinates. The combined estimate does not crash or produce an obviously wrong shape; it simply drifts further from the true inner product than either stage's error alone would predict, because the two pieces being added together do not actually correspond to each other.

## 7.4 Compressing the KV Cache with TurboQuant

### Intuition

The KV cache is the one structure in a running transformer that blockwise quantization was never a comfortable fit for: it does not exist until generation starts, it grows by one key and one value vector per token per head per layer, and there is no representative sample of it available in advance to calibrate a codebook against. TurboQuant's codebooks, precomputed purely from `d` and the bit budget with no data dependence at all, are exactly suited to this: every vector that will ever be cached can be quantized the moment it is produced, using a codebook that was already finished before generation even began.

### The Concept, In Detail

Applying Section 7.2's MSE-mode quantizer to a KV cache is mostly a matter of applying it once per cached vector, independently — the same rotation matrix and codebook serve every key and every value at a given head dimension, but each vector gets and stores its OWN norm, because cached key and value magnitudes genuinely vary from token to token; nothing about the codebook's universality extends to sharing that one per-vector scalar. MSE mode, not the unbiased "Prod" mode from Section 7.3, is the right choice here in the typical 3-4 bit range: the inner-product bias at those bit-widths is small enough that the simpler, one-bit-cheaper quantizer wins on cost without a meaningful quality trade-off, though Prod mode remains available whenever that trade-off does not hold — vector-database search at very low bit-widths, for instance, where biased rankings across millions of vectors compound into real errors. Simulating a full attention step — compute scores against every cached key, softmax, weight the cached values — with quantized keys and values in place of FP32 ones shows the attention weights themselves shift only slightly and the final output vector's error stays small, and running the same quantizer across a growing cache shows that per-vector error does not compound as more tokens are added, because each vector is quantized independently with no running state carried between them. Translating this into a memory budget for realistic model configurations shows several-fold compression relative to FP16 at 3 bits per coordinate — a number obtained here from straightforward byte-count arithmetic, not from re-running the retrieval-quality benchmarks (Needle-in-a-Haystack and similar) that the TurboQuant paper itself reports; those quality claims are cited from the literature rather than re-derived in this toy simulation.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_kv_cache_quantization.cpp -o 04_kv_cache_quantization
./04_kv_cache_quantization
```

**Sample input:** single-head attention over a 24-token cache (head_dim=32, 3 bits) with FP32 keys/values compared against TurboQuant-quantized ones; per-vector reconstruction error measured across four consecutive 32-token chunks of a growing cache to check for compounding drift; a memory-footprint comparison between FP16 and TurboQuant 3-bit for three representative model configurations; and a deliberate demonstration of reusing one cached vector's norm for every later vector instead of storing a norm per vector.

@@OUT4@@

!!! warning "[COMMON TRAP] one norm for the whole cache instead of one per vector"
    The rotation matrix and the codebook are genuinely shared across every vector in the cache — that sharing is what makes the codebook "universal" and the whole scheme data-oblivious. The norm is not part of that shared machinery; it is per-vector data, exactly as essential to a correct reconstruction as the quantization indices themselves, because real cached keys and values do not all have the same magnitude from token to token. Reusing the very first cached vector's norm for every subsequent vector — as if the norm were some fixed property of the layer rather than of each individual vector — leaves the INDICES correct while silently corrupting the SCALE of every reconstructed vector whose true magnitude differs from that first one, and in a real sequence, almost all of them will differ. Nothing about this failure looks like ordinary quantization noise: the reconstructed direction is fine, the reconstructed length is simply wrong.

## 7.5 TurboQuant vs. Blockwise Quantization, and the Distortion Bounds

### Intuition

The chapter has built an alternative to Chapters 2-6's blockwise Q8_0/Q4_0, not a replacement for it — the two approaches are optimized for different situations, and the honest way to see that is to run them against each other on the same vectors and look at where each one wins. It is also worth asking a harder question about TurboQuant specifically: not just "is it better than blockwise here," but "how much better could ANY quantizer possibly be" — because a technique described as "near-optimal" ought to be checked against the theoretical optimum it claims to approach, not just against one specific competitor.

### The Concept, In Detail

Running Chapters 2-4's blockwise Q8_0 and Q4_0 alongside TurboQuant's MSE quantizer at several bit-widths, on the same set of random vectors, with the same set of query vectors for measuring inner-product error, shows the expected shape of the trade-off: Q8_0's larger byte budget buys the lowest reconstruction error of the group, Q4_0 trades some of that accuracy for roughly half the bytes, and TurboQuant at a comparable bit-width matches or beats Q4_0's reconstruction quality using a comparable byte budget — while TurboQuant at very low bit-widths (2 bits) trades substantially more accuracy for a much smaller footprint than either blockwise format can reach at all, since Q4_0 has no lower-bit-width sibling. Checking TurboQuant's own measured distortion against the Shannon lower bound (`4^-bits`, the information-theoretic floor for any quantizer of a unit-sphere vector at that bit budget, independent of `d`) shows the measured-to-bound ratio staying within a small, roughly constant factor across every bit-width tested — close to (though not an exact reproduction of) the 2.7x cap the TurboQuant paper proves analytically for the idealized infinite-dimensional codebook. Comparing byte budgets fairly across the two families requires counting EVERY byte a real system would actually store, not just the part that is convenient to count — which is exactly where this section's own trap lives.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_turboquant_vs_blockwise.cpp -o 05_turboquant_vs_blockwise
./05_turboquant_vs_blockwise
```

**Sample input:** Q8_0, Q4_0, and TurboQuant MSE at 2, 3, and 4 bits, compared head to head for per-element MSE, average inner-product error, and bytes per vector across 100 random 64-dimensional vectors and 10 query vectors; TurboQuant's measured total squared error against the Shannon lower bound `4^-bits` at bit-widths 1 through 4; and a deliberate compression-ratio comparison that first omits, then includes, each format's per-vector or per-block overhead bytes.

@@OUT5@@

!!! warning "[COMMON TRAP] dropping the overhead bytes from a compression ratio"
    Every format in this comparison has some per-vector or per-block overhead beyond the raw quantized values themselves — TurboQuant's stored norm (4 bytes), or a blockwise format's stored scale (2 bytes as fp16, per 32-element block). That overhead is easy to leave out of a quick compression-ratio calculation, and it is NOT a rounding error to do so — it inflates the reported ratio in a way that gets worse, not better, at smaller dimensions, because a fixed few bytes of overhead is a much larger fraction of a small vector's total size than of a large one's. A single 32-element vector at 2-bit TurboQuant looks like 16x compression if only the index bytes are counted, and a real 10.7x once the stored norm is included — the same distortion in the other direction hits blockwise formats too. Neither number is fabricated, but only one of them describes what an actual running system would need to store.

## Chapter Summary

This chapter built TurboQuant from first principles as a genuinely different family of quantizer from every blockwise format in Chapters 2 through 6 — one that treats a whole vector as a single geometric object rather than a sequence of independent 32-element blocks. Section 7.1 built and verified the random rotation that makes this possible, confirming both that it spreads a worst-case vector's energy evenly across every coordinate and that a rotated coordinate's variance matches the theoretical prediction the rest of the chapter depends on — and showed concretely what breaks if the "rotation" is not actually orthogonal. Section 7.2 built the Lloyd-Max codebook that exploits the resulting known distribution, combined it with the rotation into a complete reconstruction-optimal quantizer, and demonstrated why skipping the normalize step before quantizing is not a rounding error but a missing scale factor with unbounded consequences. Section 7.3 showed that reconstruction-optimal is not the same as inner-product-safe: the MSE quantizer is systematically biased for dot products, and the QJL residual correction fixes that bias by adding a second, independent 1-bit stage — provided the two stages are combined in a consistent coordinate system, which this section's own trap showed is not automatic. Section 7.4 applied the resulting quantizer to the KV cache, the online, uncalibrated structure blockwise quantization struggles with, and showed that a per-vector norm is not optional bookkeeping any more than the quantization indices themselves are. Section 7.5 closed the chapter by measuring TurboQuant against Chapters 2-6's blockwise formats directly and against the Shannon lower bound that no quantizer of any kind can beat, landing within a small constant factor of that bound at every bit-width tested — while also showing how easily a compression-ratio comparison can be quietly skewed by dropping the overhead bytes real storage would require. TurboQuant does not replace blockwise quantization; the two are complementary, and Chapter 4's hybrid engine now has both tools available — Q4_0/Q8_0 for the static weights a fused SIMD matmul needs, and TurboQuant for the KV cache that arrives one vector at a time with no calibration data at all.

## Self-Check Questions

1. Why does quantizing the worst-case vector `[1, 0, ..., 0]` with a coordinate-wise 1-bit quantizer waste almost all of its bit budget, and how does a random rotation fix this without changing what the vector "means"?
2. What specifically fails if the "rotation" matrix used is a raw matrix of independent Gaussian entries rather than an orthogonalized one, and why does the failure not depend on any quantization being involved at all?
3. Why can TurboQuant's Lloyd-Max codebooks be precomputed entirely offline, with no data dependence, in a way that Product Quantization's k-means-based codebooks cannot?
4. Section 7.2 normalizes a vector to the unit sphere before rotating and quantizing it. What goes wrong, concretely, if that normalization step is skipped and the raw vector is rotated and quantized directly?
5. Explain why a quantizer that is provably optimal for minimizing reconstruction error (MSE) can still be systematically biased when used to estimate inner products. What does "systematically biased" mean here, as opposed to merely noisy?
6. Walk through how the QJL residual stage restores unbiasedness without needing a better first-stage quantizer. What specific property of the rescaling factor `sqrt(pi/2)/d` makes this work?
7. Why must the residual quantized by QJL be computed in the same coordinate system as the query vector it will later be dotted with, and what goes wrong if it is computed in rotated coordinates instead?
8. Why is a per-vector stored norm not optional in KV cache quantization, even though the rotation matrix and codebook are shared across every vector in the cache?
9. Why is TurboQuant not a good replacement for Chapters 2-6's blockwise Q4_0/Q8_0 when quantizing static weight matrices for a fused matmul kernel?
10. What does it mean for TurboQuant's measured distortion to be checked "against the Shannon lower bound," and why is landing within a small constant factor of that bound a meaningfully stronger claim than just "lower error than one specific competitor"?

## Where We Go Next

This chapter, like Chapter 6, has been a supplementary deep dive rather than a direct continuation of the hybrid engine's forward pass — TurboQuant compresses the KV cache, a structure that exists only during generation and grows one vector at a time, which is a fundamentally different problem from the static weight loading Chapters 2 through 6 focused on. With both tools now available — blockwise Q4_0/Q8_0 for weights, TurboQuant for an online KV cache — Part 1's quantization story is complete. Part 2 turns from what a model's numbers are represented as to how fast an engine can actually move those numbers through a CPU: Chapter 8 covers the memory wall and the roofline model, the framework for understanding when an inference engine's speed is limited by arithmetic and when it is limited by how fast data can be fetched from memory in the first place — a question that shapes every optimization decision Part 2 makes from that point on.

## Worked Solutions

**1.** A 1-bit coordinate-wise quantizer has exactly two representable values for each coordinate, typically something like `+c` and `-c`, with no representable value for "exactly zero." The spike vector `[1, 0, ..., 0]` has one coordinate carrying all of the vector's information and every other coordinate at exactly zero — so the quantizer spends one bit meaningfully on the first coordinate and wastes every other bit forcing a zero into a nonzero bin. A random orthogonal rotation does not add or remove any information from the vector (it preserves length and inner products exactly), but it redistributes WHERE that information sits among the coordinates: after rotation, a vector that started as a spike has its energy spread almost evenly across all `d` coordinates, each at roughly `1/sqrt(d)`, so every coordinate now carries a comparable, nonzero amount of information for the quantizer to spend its bit on.

**2.** A raw Gaussian matrix `G` is not orthogonal — `G^T G` is not the identity matrix — which means `G^T` is not `G`'s inverse. Multiplying a vector by `G` still scrambles it thoroughly, exactly as a proper rotation would, but multiplying the result by `G^T` does not undo that scrambling; it produces a different vector with a large, structural discrepancy from the original. This failure has nothing to do with quantization: it shows up even in a pure "rotate, then rotate back" round-trip with no coordinate ever touched by a codebook, because the bug is in the linear algebra itself, not in any rounding step layered on top of it.

**3.** Product Quantization's codebooks are built by running k-means on a REPRESENTATIVE SAMPLE of the actual data to be quantized — the codebook is calibrated to whatever distribution that sample happens to have, which requires seeing real data before any quantization can begin. TurboQuant's codebooks are built by solving the Lloyd-Max problem for the Beta distribution that ANY rotated unit vector's coordinates follow, a fact that depends only on the dimension `d` and holds regardless of what the original, un-rotated data actually looked like. Because the random rotation guarantees this same statistical shape for every input, one codebook computed purely from `d` and the bit-width serves every vector that will ever arrive — including a KV cache vector produced by a token that has not been generated yet.

**4.** The Lloyd-Max codebook in Section 7.2 was solved for the distribution of a coordinate on the UNIT sphere, which has variance `1/d`. If the raw, un-normalized vector is rotated and quantized directly, its rotated coordinates have a variance that scales with the SQUARE of the vector's true norm, not `1/d` — for any vector whose norm is meaningfully different from 1, most or all of the rotated coordinates land far outside the range the codebook's bins actually cover, and every one of them saturates at whichever outermost centroid happens to be closest. The resulting reconstruction error is not proportional rounding noise; it is a missing scale factor, and it grows without bound as the true norm grows further from 1, because the reconstruction stays stuck near the unit sphere regardless of how large the actual input was.

**5.** "Systematically biased" means the estimator's EXPECTED value — its average over many independent randomizations of the quantizer, such as many different random rotation matrices — differs from the true inner product by a consistent, predictable amount in one direction, not just by random noise that would average out to zero over repeated trials. The MSE-optimal quantizer's inner-product estimate is a clear example: at low bit-widths, its expected value is a fixed fraction (as small as `2/pi` at 1 bit) of the true inner product, EVERY time, for EVERY vector, not merely wrong on some unlucky draws and right on others. A quantizer can be "optimal" for one loss function (squared reconstruction error) while being provably, unavoidably wrong in a specific direction for a different one (inner-product estimation), because the two loss functions are simply measuring different things.

**6.** Writing `x = x_tilde_mse + r`, the true inner product with a query `y` splits as `<y, x> = <y, x_tilde_mse> + <y, r>`. The MSE stage already supplies the first term exactly (it is just the dequantized MSE reconstruction). QJL supplies an UNBIASED estimate of the second term, `<y, r>`, using only the sign of a random Gaussian projection of `r` — and the specific rescaling `sqrt(pi/2)/d` is exactly the constant that makes `E[<y, QJL_dequant(r)>]` equal `<y, r>`, a fact that follows from the symmetry of the Gaussian distribution used to build the projection matrix `S`. Adding an unbiased estimate of the missing term to the exact first term gives, in expectation, the true inner product — the QJL stage does not need to reconstruct `r` accurately at all (and it does not: keeping only a sign per coordinate throws away almost all of `r`'s information), it only needs its INNER PRODUCT WITH `y`, in expectation, to be correct.

**7.** QJL's unbiasedness proof is stated for a random Gaussian matrix `S` applied directly to the vector whose inner product with an arbitrary query is being estimated — and that query lives in the ORIGINAL, un-rotated coordinate space, since that is the space attention scores and vector-database queries are actually computed in. If the residual is instead computed in ROTATED coordinates (subtracting before rotating the MSE term back), the resulting QJL-quantized correction describes a residual expressed in a different basis than the query it will later be dotted with. Adding that correction directly to the original-space MSE term and dotting the sum with an original-space query does not correspond to any consistent linear-algebra operation — the pieces being summed are not describing the same coordinates — and the resulting estimate drifts further from the true inner product than either stage's individual error would suggest, without producing any obviously wrong shape or crash to signal the mistake.

**8.** The rotation matrix and the Lloyd-Max codebook are universal precisely because they do not depend on any individual vector's data — the same `Pi` and the same codebook are mathematically valid for every vector of a given dimension. A vector's NORM is exactly the piece of information that normalization deliberately strips out before quantization, specifically because real vectors do not all have the same magnitude — and that is just as true of cached keys and values across different tokens in a sequence as it is of any other vector. Storing one norm per vector is what lets the shared rotation and codebook be reused universally while still recovering each vector's true scale at dequantization time; without it, every reconstructed vector would be rescaled by whichever single norm happened to be stored, which is correct for at most one vector in the entire cache.

**9.** A fused SIMD matmul kernel needs to dequantize and multiply-accumulate a weight block using only cheap, local arithmetic — Chapters 2-6's blockwise formats fit this exactly, since dequantizing one block requires only that block's own stored scale and a handful of integer multiplies. TurboQuant's dequantization requires a full `O(d^2)` matrix-vector multiply by the rotation matrix's transpose for every single vector before its values are usable at all — a cost that is negligible when applied once per KV cache vector as it is produced, but would dominate the inner loop of a matmul that touches every weight in a matrix multiple times per forward pass. Static weights also do not benefit from TurboQuant's core advantage: they are quantized once, offline, with as much calibration effort as desired, which is exactly the setting blockwise per-block scales already handle well, whereas TurboQuant's advantage is specifically for data that arrives online with no calibration opportunity at all.

**10.** The Shannon lower bound is not "how well some other specific quantizer happens to do" — it is a proof, from information theory, of the smallest possible distortion ANY quantizer whatsoever could achieve for a given bit budget on a unit-sphere vector, regardless of how clever or specialized that hypothetical quantizer might be. Beating one specific competitor (such as Q4_0) only shows TurboQuant is better than that one alternative; it says nothing about how much further improvement might still be possible. Landing within a small, bounded factor of the Shannon lower bound at every bit-width tested is a much stronger claim: it means no future quantizer, however sophisticated, could improve on TurboQuant by more than that same small constant factor, because the lower bound itself is a hard floor that no algorithm can cross.
