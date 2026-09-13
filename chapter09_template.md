# Chapter 9: SIMD Vectorization -- AVX2 on x86, NEON on Arm

**What you will understand by the end of this chapter:**

- What a SIMD register actually is — a fixed number of bits divided into lanes of a stated element width — derived from that single fact rather than memorized as a vendor-specific rule, and why doubling a chip's peak compute this way still buys nothing for a memory-bound kernel, exactly per Chapter 8's own roofline model.
- How to write a real, hand-vectorized AVX2 kernel for the Q8_0 and Q4_0 quantized formats this book has used since Chapter 4 — including the specific integer unpacking a 4-bit nibble format demands before it can be widened into arithmetic — and verify it bit-for-bit-close against the scalar reference it must agree with.
- Why the identical logical operation — widening an 8-bit integer to 32 bits — takes one AVX2 instruction and two NEON instructions, a genuine, verifiable asymmetry between the two instruction sets rather than a naming difference, and why porting intrinsic-for-intrinsic across architectures is a real source of silent bugs.
- Why a compiler's auto-vectorizer inserts a runtime pointer-aliasing check by default, what `__restrict` actually promises when it removes that check, and why breaking that promise produces a silently WRONG answer rather than a crash or a compile error — demonstrated here as an empirically measured divergence in computed values, not asserted from a disassembly listing.
- How to make one binary run correctly, and use the fastest instruction set available, on every customer's CPU — by querying the CPU's real feature bitmap once at startup, not by assuming at compile time that every CPU looks like the machine that built the binary.

**What you need to know first:**

- Chapter 8's roofline model — peak compute, peak bandwidth, and the ridge point that separates memory-bound kernels from compute-bound ones — Section 9.1 reuses Chapter 8.1's own `CpuSpec` struct and its own memory-bound COMMON TRAP directly, changing only the SIMD lane count.
- Chapter 4's `fp16_t`, `BlockQ8`, and `BlockQ4` structs and their `quantize_q8`/`quantize_q4` functions, reused verbatim throughout this chapter's Sections 9.2 and 9.3.
- This book's standing policy against fabricated timing numbers applies here in a new way: this chapter does not claim any specific SIMD kernel is "N times faster" than its scalar counterpart on the reader's own machine, because a shared, virtualized build environment gives no wall-clock number that would reproduce there. Every quantity in this chapter is either a derived architectural fact (register width, lane count), a directly measured correctness result (maximum absolute error against a scalar reference), or a directly observed compiler behavior (which offsets diverge, which kernel a runtime query selects) — never an estimated speedup.
- This chapter is the first in the book whose sections genuinely cannot all run on the same machine: Section 9.2's AVX2 code requires an x86_64 CPU, and Section 9.3's NEON code requires an Arm CPU. Both are still held to the book's full build-and-verify discipline — compiled for real, run twice to confirm determinism — just on the architecture each one actually targets, using cross-compilation and emulation to keep both inside one reproducible pipeline, with the Arm section additionally confirmed on real Arm hardware.

---

Chapter 8 established that a kernel's speed is capped by two independent resources, compute and bandwidth, and that the ratio between them — arithmetic intensity — decides which one actually limits a given kernel. This chapter is about the compute side of that ceiling: how a CPU's peak FLOP rate is actually achieved in practice, one instruction at a time, by processing more than one data element per instruction. Section 9.1 derives what a SIMD register buys a kernel from nothing but its bit width and element width, reusing Chapter 8.1's own peak-compute formula to show precisely how much of a chip's peak throughput its register width accounts for — and reusing Chapter 8.1's own COMMON TRAP to show, again, that none of it matters for a kernel that is memory-bound. Section 9.2 writes real AVX2 intrinsics against the Q8_0 and Q4_0 formats this book has quantized weights into since Chapter 4, including the integer unpacking a packed 4-bit format demands before any arithmetic can touch it. Section 9.3 writes the same two kernels again in Arm NEON, and finds a genuine, verifiable difference in how many instructions the identical logical step costs on the two architectures. Section 9.4 steps back from hand-written intrinsics entirely to ask what a compiler's auto-vectorizer does on its own, and what specific promise `__restrict` makes to unlock it. Section 9.5 closes the chapter with the piece every one of these kernels needs before it can ship: a runtime check that decides, once, at startup, which instruction set the CPU actually sitting under the binary supports.

## 9.1 The SIMD Register Model, Derived From Register Width

### Intuition

A SIMD register is not a mysterious accelerator; it is a wider bucket that a fixed number of narrower values are poured into together. A 256-bit AVX2 register and a 128-bit NEON register both follow the identical rule — lanes equal register bits divided by element bits — and that one arithmetic fact, not a lookup table of vendor trivia, is enough to derive exactly how much peak compute a given register width is responsible for.

### The Concept, In Detail

`lanes_per_register(register_bits, element_bits) = register_bits / element_bits` gives 8 packed FP32 lanes for a 256-bit AVX2 register and 4 for a 128-bit NEON register — and, since register capacity is fixed in bits rather than in element count, exactly double as many lanes (32 and 16, respectively) for an 8-bit integer element, with no separate rule required. Chapter 8.1's `CpuSpec::peak_gflops()` formula is reused here completely unchanged, with only `simd_lanes_fp32` swapped between the two values: at identical core count and clock speed, AVX2's eight lanes against NEON's four produce exactly double the peak FP32 throughput, a ratio that falls directly out of the formula being linear in lane count. Feeding both machines' peak compute into Chapter 8.1's own achievable-throughput formula at a representative memory-bound decode arithmetic intensity reproduces that chapter's own COMMON TRAP in a new guise: the two peak-compute figures differ by a factor of two, and the two ACHIEVABLE throughputs at this arithmetic intensity differ by exactly zero, because a memory-bound kernel's achievable throughput was never a function of peak compute to begin with.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_simd_register_model.cpp -o 01_simd_register_model
./01_simd_register_model
```

**Sample input:** lane counts for AVX2's 256-bit register and NEON's 128-bit register at both FP32 and int8 element widths, derived from the single register-bits-over-element-bits formula; Chapter 8.1's own peak-compute formula evaluated at those two lane counts, holding core count and clock speed fixed; and a deliberate demonstration, at Chapter 8.1's own representative decode-time arithmetic intensity, of doubling the SIMD lane count changing achievable throughput by nothing at all.

@@OUT1@@

!!! warning "[COMMON TRAP] assuming a wider SIMD register always means a faster kernel"
    Doubling a chip's SIMD lane count really does double its peak compute — that part of the arithmetic is not in dispute, and this section verifies it directly. But peak compute is only the ceiling a COMPUTE-bound kernel can approach; a memory-bound kernel's achievable throughput, per Chapter 8.1's own formula, is `arithmetic_intensity * peak_bandwidth`, an expression that does not mention SIMD width at all. At a representative decode-time arithmetic intensity, AVX2's eight lanes and NEON's four lanes produce identical achievable throughput, down to the last decimal place, because both machines hit the same bandwidth ceiling long before either one's compute ceiling becomes relevant. "This CPU has wider SIMD registers" is evidence of nothing for a kernel's real-world speed until the kernel's own arithmetic intensity relative to that machine's ridge point has been checked first.

## 9.2 AVX2: Vectorized Quantized Dot Products on x86

### Intuition

The Q8_0 and Q4_0 dot product this book has computed scalar-element-by-scalar-element since Chapter 4 is exactly the kind of loop SIMD exists for: the same handful of operations, repeated independently across many elements, with no data dependency between one element and the next. AVX2 turns that repetition into eight-wide (for Q8_0's already-int8 elements, widened to int32) parallel arithmetic per instruction — but a packed 4-bit format cannot simply be "widened"; its two values per byte have to be pried apart first, and getting that unpacking wrong is where a hand-vectorized quantized kernel most often goes quietly wrong.

### The Concept, In Detail

`avx2_dot_q8` widens each block's 32 signed int8 weights to int32 with a single `_mm256_cvtepi8_epi32` instruction, converts to float, and accumulates against the dequantized activation values with `_mm256_fmadd_ps`, reducing the resulting 8-wide accumulator to one scalar with a tree of horizontal adds. `avx2_dot_q4` has one additional step before that same pipeline can run: each byte of a Q4_0 block packs two 4-bit nibbles with a `+8` bias, so the low nibble is isolated with `_mm_and_si128`, the high nibble with `_mm_srli_epi16` followed by the same mask, the bias is removed from both with `_mm_sub_epi8` (relying on two's-complement wraparound, since there is no signed 4-bit subtract), and the two nibble streams are interleaved back into their original element order with `_mm_unpacklo_epi8`/`_mm_unpackhi_epi8` before the identical widen-and-FMA pipeline `avx2_dot_q8` already used takes over. Both vectorized kernels are checked against the scalar reference over 200 randomly generated blocks, and the tiny nonzero error that remains — a few times ten-to-the-minus-six — is not a correctness bug: floating-point addition is not associative, and AVX2's tree-shaped horizontal reduction genuinely sums the same 32 products in a different order than the scalar loop's strictly sequential accumulation, a real, expected, and bounded source of last-bit divergence rather than a sign that either kernel disagrees about what the answer should be.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -mavx2 -mfma 02_avx2_quantized_dot_product.cpp -o 02_avx2_quantized_dot_product
./02_avx2_quantized_dot_product
```

**Sample input:** the AVX2 Q8_0 and Q4_0 dot-product kernels, each checked against the scalar reference over 200 randomly generated 32-element blocks, reporting the maximum absolute error observed; and a deliberate demonstration of what happens to a raw 4-bit nibble's value when it is sign-extended directly instead of having its bias subtracted first.

@@OUT2@@

!!! warning "[COMMON TRAP] sign-extending a quantized nibble instead of removing its bias"
    A Q4_0 nibble is stored as an unsigned 4-bit value in the range `[0, 15]`, encoding a signed value in the range `[-8, 7]` by adding a fixed bias of 8 at encode time — so decoding it correctly means SUBTRACTING that bias, not sign-extending the raw 4 bits as though they already were a signed quantity. Sign-extending a raw nibble of `1` (which the correct decode would turn into `-7`) instead leaves it as `1` — identical to the raw bits whenever they happen to already be less than 8, and never negative at all regardless of what value the block actually encoded, since a 4-bit field has no sign bit of its own to extend. This is not a rare edge case: roughly half of any real Q4_0 block's nibbles decode to a value that this bug gets wrong, and every one of those errors is silent, because the buggy output is still a plausible-looking float, just the wrong one.

## 9.3 Arm NEON: The Same Kernel, a Different Register

### Intuition

Every transformer that runs on a phone, a laptop's efficiency cores, or an Arm-based server needs the identical Q8_0 and Q4_0 dot product Section 9.2 just vectorized for AVX2 — and Arm NEON can vectorize it too, with the same lane-based reasoning Section 9.1 already established. But NEON is not AVX2 with the function names swapped: at least one operation Section 9.2 leaned on as a single instruction has no NEON equivalent at all, and finding that out empirically here is exactly the kind of asymmetry a chapter that only ever wrote AVX2 could never have surfaced.

### The Concept, In Detail

NEON's 128-bit registers hold four packed FP32 lanes or sixteen packed int8 lanes, per Section 9.1's own formula, so `neon_dot_q8` processes two 16-element chunks of a 32-element Q8_0 block rather than AVX2's one 32-wide pass, accumulating with `vfmaq_f32` and reducing with `vaddvq_f32`, an aarch64-only horizontal-sum instruction with no 32-bit-Arm equivalent. `neon_dot_q4`'s nibble unpacking mirrors Section 9.2's structure — mask, shift, remove the bias, interleave the two nibble streams back into order with `vzip1q_s8`/`vzip2q_s8` — but the widening step that AVX2 did in ONE instruction (`_mm256_cvtepi8_epi32`, int8 straight to int32) has no direct NEON counterpart: NEON's `vmovl_s8` only reaches int16, so reaching int32 requires a SECOND widening call, `vmovl_s16`, applied to that intermediate result. This is not a naming difference to paper over with a macro; it is a genuine two-instructions-versus-one asymmetry between the architectures for the identical logical operation, and a port that assumed one NEON call could replace one AVX2 call here would simply produce a value still packed at the wrong width for the FMA that follows. Both NEON kernels are checked against the same scalar reference Section 9.2 used, over the same 200 randomly generated blocks, and agree to within the same last-few-bits floating-point reordering tolerance.

### Code and Verification

@@CODE3@@

**Compile and run (cross-compiled and emulated, since this book's build environment is x86_64):**

```bash
aarch64-linux-gnu-g++ -std=c++23 -Wall -Wextra -O2 03_neon_quantized_dot_product.cpp -o 03_neon_quantized_dot_product
qemu-aarch64 -L /usr/aarch64-linux-gnu ./03_neon_quantized_dot_product
```

**Sample input:** the NEON Q8_0 and Q4_0 dot-product kernels, each checked against the same scalar reference Section 9.2 used, over 200 randomly generated 32-element blocks; and a deliberate, explicit demonstration that widening an int8 value all the way to int32 costs NEON two instructions (`vmovl_s8` then `vmovl_s16`) where AVX2 needed only one.

@@OUT3@@

!!! warning "[COMMON TRAP] porting an AVX2 intrinsic to NEON one-instruction-for-one-instruction"
    AVX2's `_mm256_cvtepi8_epi32` widens a packed int8 value directly to int32 in a single instruction, and it is natural to assume every AVX2 intrinsic has some equally direct NEON equivalent waiting to be substituted in. NEON's int8-to-int16 widen, `vmovl_s8`, stops at 16 bits — there is no `vmovl_s8`-to-int32 instruction — so reaching int32 genuinely requires a second call, `vmovl_s16`, applied to the already-widened result. A direct one-for-one port that copies AVX2's instruction COUNT rather than its semantic effect would leave NEON's intermediate values sitting at 16 bits where the following FMA step expects 32, either failing to compile against the wrong-width type or, with an unchecked cast, silently computing on truncated or misinterpreted data. Verifying this kind of platform-specific instruction-count difference in code, as this section does, catches it before it becomes a shipped, silent bug on whichever architecture was ported to second.

## 9.4 Auto-Vectorization and the Restrict Promise

### Intuition

Neither Section 9.2 nor Section 9.3 is the only way to get a vectorized loop: a modern compiler's `-O3` auto-vectorizer will widen a sufficiently simple elementwise loop on its own, with no intrinsics at all, PROVIDED it can prove doing so is safe. Two raw pointers are not provably safe to reorder in general, so the compiler's default is a runtime check; `__restrict` is the programmer's promise that removes it — and this section verifies both halves of that promise are real, including what happens the moment it is broken.

### The Concept, In Detail

An ordinary AXPY loop (`y[i] = a*x[i] + y[i]`) written with two unqualified pointers compiles, at `-O3`, into code that checks the DISTANCE between `y` and `x` at runtime and takes a vectorized path only when that distance guarantees no unsafe overlap within the vector width the compiler chose to use, falling back to an ordinary sequential loop otherwise — so this version agrees with a strictly sequential scalar reference for every possible pointer relationship, overlapping or not, by construction of that fallback. Qualifying both pointers `__restrict` removes the check entirely: the compiler takes the promise of no aliasing at face value and vectorizes unconditionally, which is correct, and never needs the fallback's overhead, exactly when the promise is true. This section measures what happens when it is not: feeding the same set of overlapping buffers to both versions shows the unqualified version still agreeing with the sequential reference at every tested overlap distance, while the `__restrict`-qualified version silently diverges from the correct answer at every distance narrower than this specific build's actual safe vectorized width — a width this section finds empirically to be wider than one AVX2 register alone, because this compiler's auto-vectorizer chose to unroll the loop to process more than one register's worth of floats per iteration. That measured width is a fact about this compiler, these flags, and this run, not a portable architectural constant, which is precisely why the section verifies it by running the code rather than by asserting "past one register width is safe" as a rule of thumb.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O3 -mavx2 -mfma 04_restrict_and_autovectorization.cpp -o 04_restrict_and_autovectorization
./04_restrict_and_autovectorization
```

**Sample input:** three functions performing the identical AXPY loop — a strictly sequential reference, an unqualified-pointer version, and a `__restrict`-qualified version — agreeing exactly on 20 trials of non-overlapping buffers; the unqualified version staying correct across a range of overlapping-buffer offsets thanks to its runtime alias check; and a deliberate demonstration of the `__restrict`-qualified version silently diverging from the correct answer across that same range of overlaps, at every offset narrower than this build's own measured safe distance.

@@OUT4@@

!!! warning "[COMMON TRAP] trusting a hand-derived rule of thumb for how far apart __restrict pointers must be"
    It is tempting to reason "the vector register is N-wide, so as long as my overlapping pointers are at least N elements apart, `__restrict` is safe" — but `__restrict` makes exactly one promise: NO aliasing, full stop, not "aliasing no closer than the register width." This section's own measurement shows the actual safe distance on this build is wider than the 8-lane AVX2 register alone, because the compiler chose to unroll the vectorized loop to handle more than one register's worth of data per iteration, a decision made by the optimizer, at this optimization level, on this compiler version — none of which the source code controls or can predict by inspection. A hand-derived "stay N elements apart" rule that happened to work on one build can silently break on a different compiler, a different flag, or a different unrolling decision on the very same compiler's next release. The only distance that is actually safe to rely on is the one `__restrict` itself promises: none of them alias, ever.

## 9.5 Runtime CPU Feature Detection and Dispatch

### Intuition

Every kernel this chapter has written so far assumes its target instruction set is simply available — but a single shipped binary has no such guarantee about the machine it will actually run on. The fix is not to compile a different binary per customer; it is to ask the CPU, once, at startup, which instruction sets it actually supports, and to keep a scalar fallback ready for the machines that answer "not this one."

### The Concept, In Detail

`__builtin_cpu_supports("avx2")`, backed by `__builtin_cpu_init()`'s real CPUID query, answers a question about the machine the binary is running on RIGHT NOW, not a question about the machine that compiled it — and a small dispatch table built around that query, choosing a function pointer once and calling through it from then on, lets one binary use AVX2 where it is available and fall back to a scalar kernel, correctly, where it is not. On Arm this section's dispatch table has exactly one entry: NEON is part of the aarch64 base instruction set, present unconditionally on every aarch64 CPU capable of running the binary at all, so there is no equivalent feature to query and nothing to dispatch between. The COMMON TRAP this section exists to prevent is not a bug in the dispatch logic itself; it is skipping dispatch altogether. A binary built with `-march=native` tells the compiler, at compile time, that the BUILD machine's entire instruction set is always available — so `__builtin_cpu_supports` becomes a check the compiler is free to assume always succeeds, and ordinary auto-vectorized loops (not just hand-written intrinsics) may freely use instructions the build machine has and a customer's older CPU does not. The result on that older CPU is not a graceful, slow fallback; it is `SIGILL`, an illegal-instruction crash, on the first such instruction the CPU actually tries to execute — a correctness failure a test suite run only on the build machine has no way to ever observe.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O3 -mavx2 -mfma 05_runtime_dispatch.cpp -o 05_runtime_dispatch
./05_runtime_dispatch
```

**Sample input:** a runtime dispatch table selecting between a scalar dot-product kernel and an AVX2+FMA kernel based on `__builtin_cpu_supports`, checked against the scalar reference across ten sizes including several not divisible by the vector width; the AVX2 kernel checked directly against the scalar reference on a machine confirmed by that same query to support it; and a documented explanation, checked against this program's own dispatch behavior, of why a `-march=native` build has no equivalent safety net.

@@OUT5@@

!!! warning "[COMMON TRAP] shipping a -march=native binary to a fleet of unknown CPUs"
    `-march=native` produces the fastest possible binary for the exact machine that compiles it, which makes it an easy default to reach for — and exactly the wrong choice for any binary that will run on hardware other than the machine that built it. This section's own dispatch table calls a real, runtime CPUID query before ever touching an AVX2 instruction, so it degrades to a correct, merely slower scalar kernel on a CPU that lacks AVX2; a `-march=native` build has no such query anywhere in its compiled code; it simply assumes, everywhere the compiler saw an opportunity to use it, that the build machine's full instruction set is present. Deployed to a fleet with even one older CPU model in it, that assumption does not produce a slow build — this section's own scalar fallback is what "slow but correct" actually looks like — it produces a crash on that CPU's first attempt to execute an instruction it was never able to run, a failure mode a build machine's own test suite is structurally incapable of catching.

## Chapter Summary

This chapter took Chapter 8's roofline model onto the compute side of the ridge: how a CPU's peak FLOP rate is actually realized, one SIMD instruction at a time. Section 9.1 derived the SIMD register model from nothing but register width divided by element width, reusing Chapter 8.1's own peak-compute formula to show precisely how much of a chip's throughput its register width accounts for, and reusing that same chapter's own COMMON TRAP to reconfirm that none of it matters for a memory-bound kernel. Section 9.2 hand-vectorized this book's Q8_0 and Q4_0 dot products in real AVX2 intrinsics, including the integer unpacking a packed 4-bit format demands, verified against the scalar reference to within floating-point reordering tolerance. Section 9.3 wrote the identical two kernels in Arm NEON and found a genuine, measured asymmetry between the architectures: widening an int8 value to int32 costs AVX2 one instruction and NEON two, a fact this book verified by running code on both architectures rather than assuming a symmetric intrinsic-for-intrinsic port would be safe. Section 9.4 stepped back from hand-written intrinsics to measure what a compiler's own auto-vectorizer does with an ordinary loop, and to measure, empirically rather than by assertion, exactly how far apart two `__restrict`-qualified pointers must genuinely never overlap before the silently wrong answers stop. Section 9.5 closed the chapter with the piece every one of its kernels needs before it can ship: a runtime CPU feature query that lets one binary use the fastest instruction set an actual customer machine supports, and fall back correctly, rather than crash, on the machines that support less. Chapter 8 asked whether a kernel was memory-bound or compute-bound; this chapter asked, for the compute-bound half of that question, how the arithmetic itself actually gets executed — and, just as importantly, how to make sure it executes correctly on hardware the code was never compiled on.

## Self-Check Questions

1. Derive, from the single formula `lanes_per_register(register_bits, element_bits)`, why an AVX2 register holds exactly four times as many int8 lanes as it holds FP32 lanes, without looking up either number separately.
2. Section 9.1 reuses Chapter 8.1's peak-compute formula unchanged, swapping only `simd_lanes_fp32`. Why does doubling that one parameter double peak compute exactly, rather than approximately?
3. Explain why AVX2's Q8_0 dot product and its scalar reference do not produce bit-for-bit identical results, and why that small disagreement is not treated as a correctness bug in this section.
4. What specific arithmetic mistake does Section 9.2's COMMON TRAP make when decoding a Q4_0 nibble, and why does the buggy result look like a plausible floating-point value instead of an obviously wrong one?
5. Why does NEON's int8-to-int32 widen require two separate instructions where AVX2 needs only one, and what would go wrong if a port from AVX2 to NEON assumed the two operations cost the same number of instructions?
6. In Section 9.4, why does the unqualified `axpy_noalias_unchecked` function agree with the sequential reference at every tested overlapping offset, while the `__restrict`-qualified `axpy_restrict` function does not?
7. Section 9.4 measures the `__restrict` version's actual safe overlap distance empirically rather than assuming it equals the AVX2 register's lane count. What did that measurement find, and why did it come out wider than one register's width?
8. What specific question does `__builtin_cpu_supports("avx2")` answer, and why is that a different question than what `-march=native` bakes into a compiled binary?
9. Why does Section 9.5's dispatch table have only one entry on Arm, while its x86_64 counterpart has two?
10. Explain why a binary built with `-march=native` and shipped to a fleet of mixed-generation CPUs can pass every test run on the build machine and still fail in production, in a way this chapter's dispatch-table approach specifically avoids.

## Where We Go Next

This chapter showed how a single core reaches its peak compute — by processing several data elements per instruction, whether by hand-written intrinsics or by trusting the compiler's own auto-vectorizer under the `__restrict` promise. Chapter 10 moves to the next resource a real inference engine has available and has not yet used: multiple cores. Threading and concurrency introduce a different set of correctness hazards than the aliasing this chapter examined — data races, memory ordering, and false sharing chief among them — and the same standing discipline applies: every claim about what a multi-threaded kernel is doing will be verified by running real, deterministic code, never asserted from how the scheduler is assumed to behave.

## Worked Solutions

**1.** `lanes_per_register` divides a fixed register bit width by the element's own bit width, so for a fixed register (AVX2's 256 bits), the lane count is inversely proportional to element width. An int8 element is `32/8 = 4` times narrower than an FP32 element, so dividing the same 256 bits by a value four times smaller produces exactly four times as many lanes — 32 int8 lanes against 8 FP32 lanes — a direct consequence of the division, not a separate fact about either element type.

**2.** `peak_gflops()` multiplies `simd_lanes_fp32` by `fma_ports`, by `2.0`, by `cores`, by `sustained_ghz` — every one of these is a plain multiplicative factor, so the formula is linear in each of them individually, `simd_lanes_fp32` included. Doubling a single linear factor in a product of factors always exactly doubles the product, which is why AVX2's 8 lanes against NEON's 4 lanes produces an exactly 2.00x peak-compute ratio rather than an approximate one — the relationship is arithmetic identity, not an empirical measurement with room for error.

**3.** The scalar reference sums 32 products in one fixed, strictly sequential order, while AVX2's horizontal reduction sums the same 32 products through a tree of pairwise adds — a genuinely different grouping of the same terms. Floating-point addition is not associative, so a different grouping of otherwise-identical values can legitimately produce a result that differs in its last few bits, a real and expected property of floating-point arithmetic rather than a sign that the vectorized kernel computed something conceptually different from the scalar one — which is why this section treats a few-times-ten-to-the-minus-six error as a pass, not a failure.

**4.** The trap sign-extends the raw, unsigned 4-bit nibble value directly, treating it as though its top bit already indicated a sign — but a Q4_0 nibble has no sign bit at all; its correct decoding is `stored_value - 8`, converting the unsigned range `[0, 15]` into the signed range `[-8, 7]` by subtracting a fixed bias. Sign-extending the raw bits instead of subtracting the bias leaves small stored values (below 8) completely unchanged and never produces a negative number regardless of what the block actually encoded — and because the buggy output is still an ordinary-looking float, nothing about its shape signals that anything went wrong.

**5.** AVX2's `_mm256_cvtepi8_epi32` widens a packed int8 lane directly to int32 in one instruction because AVX2 provides that specific conversion; NEON's `vmovl_s8` only reaches int16, with no equivalent instruction that jumps straight to int32, so reaching int32 requires a second call, `vmovl_s16`, applied to the intermediate 16-bit result. A port that assumed a one-to-one instruction correspondence would either fail to compile (a type mismatch between the still-16-bit intermediate and a 32-bit-expecting FMA) or, worse, compile against a mismatched-width value if the intermediate type were forced through an unchecked cast, corrupting every value that passed through it.

**6.** The unqualified version gives the compiler no promise that `y` and `x` do not alias, so at `-O3` it inserts a runtime check comparing their distance and only takes the vectorized path when that distance is provably safe, falling back to an ordinary sequential loop — identical, by construction, to the sequential reference — whenever it is not. The `__restrict`-qualified version removes that check entirely, on the strength of the programmer's promise that the two pointers never alias; when that promise is actually broken by the caller, the compiler has no mechanism left to detect it, and the unconditionally vectorized code reads stale, not-yet-updated values exactly where the true aliasing would require reading freshly written ones.

**7.** The measurement found the actual safe distance for this build to be 16 elements, not 8 — wider than one 256-bit AVX2 register's own 8-float lane count. It came out wider because the compiler's optimizer chose to UNROLL the vectorized loop, processing two full vector registers' worth of data (16 floats) per outer loop iteration rather than one, a decision made independently of the register's own width by this specific compiler at this specific optimization level.

**8.** `__builtin_cpu_supports("avx2")` answers "does the CPU actually executing this code, right now, support AVX2" — a question resolved at RUNTIME by a real CPUID query the first time it is asked. `-march=native` answers a completely different question at COMPILE time: "does the machine currently compiling this code support AVX2" — and once compiled, the resulting binary carries no runtime check of its own at all, having been built on the assumption that whatever the build machine supported will always be present wherever the binary later runs.

**9.** NEON is part of the aarch64 base instruction set architecture — every aarch64 CPU capable of running an aarch64 binary at all already has NEON, unconditionally, so there is no meaningful runtime question to ask and therefore nothing to dispatch between; a one-entry table with no query is the correct, complete answer on that architecture. AVX2, by contrast, is an OPTIONAL x86_64 extension that not every x86_64 CPU implements, so the x86_64 table genuinely needs two entries and a real runtime query to choose between them.

**10.** A test suite run on the build machine exercises the binary on the exact CPU whose instruction set `-march=native` baked in, so every AVX2 (or newer) instruction the compiler emitted is one the test machine can actually execute — the tests pass, correctly, because nothing about that machine ever exposes the missing runtime check. Shipped to a customer machine with an older CPU lacking one of those instructions, the binary reaches an instruction the CPU has no decoder for and raises `SIGILL`, a failure mode that categorically cannot appear on the build machine, however thoroughly it was tested there. Section 9.5's dispatch table avoids this by making the instruction-set choice at RUNTIME, on the machine that will actually execute the code, with a correct (if slower) fallback for exactly the CPUs a `-march=native` build would crash on.
