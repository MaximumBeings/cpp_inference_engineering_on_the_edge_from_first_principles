# Chapter 2: The Tensor Engine -- std::mdspan, Arena Allocators, and Mixed Precision

**What you will understand by the end of this chapter:**

- How `std::mdspan` turns a raw pointer into a multi-dimensional, indexed view with zero copies and zero allocations, and how to read its three template parameters — element type, extents, and layout — to know exactly what shape and memory order any given `mdspan` describes.
- How `std::submdspan` slices a tensor — extracting one attention head, or a window of sequence positions, from a larger buffer — without copying a single byte, and why the resulting sub-view's strides are not always what a first guess would suggest.
- Why loop order over the same data can differ by a large, measured factor depending on which dimension is stride-1, why per-token scratch allocation belongs in a bump-pointer arena rather than behind `malloc`/`free`, and why BF16 rather than FP16 is the default storage format for modern LLM weights.

**What you need to know first:**

- Ordinary C++ templates, `std::vector`, and RAII (Chapter 1 assumed no more than this, and this chapter assumes nothing beyond it).
- What a transformer's forward pass needs at a conceptual level — token embeddings, attention heads, a hidden dimension — since this chapter's examples are drawn directly from that domain rather than from abstract matrices.

---

Chapter 1 argued that edge inference is a different engineering problem from cloud inference: memory is scarce, allocation has to be predictable, and the difference between a contiguous buffer and one three cache lines away can decide whether a token generation loop hits its latency budget. This chapter builds the concrete C++23 tools this book uses everywhere from here forward to act on that argument. `std::mdspan` gives every later chapter a single, standard way to describe a tensor's shape without owning its memory. `std::submdspan` gives multi-head attention a way to slice that tensor per head without copying. The arena allocator gives the token generation loop a scratch-memory strategy with no per-token syscall and no fragmentation. And `std::variant`-based mixed precision gives every later chapter a type-safe way to store the same logical tensor as FP32 or BF16 without duplicating kernel code. Every other chapter in Part 0 through Part 5 assumes all four of these are already comfortable tools by the time they are needed again.

## 2.1 std::mdspan: A Zero-Copy View Over Flat Memory

### Intuition

A `std::vector<float>` is just a flat run of bytes; nothing about it says whether those bytes represent one long vector, a 32x128 tile, or a batch of token embeddings. `std::mdspan` adds exactly that missing layer — a way to say "read this flat buffer as if it had this shape" — without allocating anywhere or copying a single element. It is a view in the same sense `std::string_view` and `std::span` are views: a pointer plus metadata, nothing more.

### The Concept, In Detail

`std::mdspan<ElementType, Extents, Layout>` has three template parameters, and each answers a different question. `ElementType` is the scalar stored at each index — `float` for activations, `const float` for read-only weights, `std::bfloat16_t` or a hand-rolled equivalent for half-memory storage. `Extents` describes the shape: `std::dextents<size_t, N>` means all `N` dimensions are supplied at construction time (fully dynamic), `std::extents<size_t, 32, 128>` bakes every dimension into the type itself (fully static, zero runtime shape metadata), and `std::extents<size_t, std::dynamic_extent, 4096>` mixes the two — exactly the realistic case for an LLM embedding tensor, where the batch size changes per request but the hidden dimension is fixed by the checkpoint. `Layout` controls how a multi-dimensional index maps to one flat offset: `layout_right` (the default) is row-major, where the *last* index changes fastest in memory — the C and C++ convention — and `layout_left` is column-major, the Fortran and BLAS convention, where the *first* index changes fastest instead. The angle-bracket-versus-parenthesis split matters here: whatever is known at compile time goes in `<>`, and only the sizes that are *not* already in the type get passed to the constructor in `()`. For the mixed-extents case above, that means exactly one runtime argument — the batch size — even though the tensor is logically two-dimensional, because the second dimension already lives in the type.

### Code and Verification

```cpp
// Chapter 2.1 -- std::mdspan is a NON-OWNING view over flat memory: it
// never allocates or frees anything, it just adds indexed, multi-
// dimensional access on top of a pointer you already own. This file
// proves that claim directly (mutating through the view changes the
// underlying buffer, not a copy) and then walks all three ways a
// shape can be known: fully dynamic (every size supplied at
// construction), fully static (every size baked into the type), and
// mixed (the common LLM case: a batch size that changes per request,
// paired with a hidden dimension fixed by the model architecture).

#include <mdspan/mdspan.hpp>
#include <vector>
#include <iostream>
#include <cstdint>

static int g_tests = 0, g_passed = 0;
// Variadic on purpose: matrix[1, 2] contains a top-level comma, which the
// preprocessor would otherwise treat as two separate macro arguments.
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// ---- Part 1: a dynamic 2D matrix, and proof that it is a VIEW ----
//
// std::mdspan(ptr, rows, cols) with no explicit template arguments uses
// CTAD (class template argument deduction): the element type comes from
// ptr's pointee type, the extents type becomes dextents<size_t,2> because
// two runtime sizes were passed, and the layout defaults to layout_right
// (row-major, the C/C++ convention: the last index is the one that
// changes fastest in memory).
void demo_dynamic_matrix() {
    std::vector<float> data = {
        1, 2, 3, 4,     // row 0 -- flat indices 0..3
        5, 6, 7, 8,     // row 1 -- flat indices 4..7
        9, 10, 11, 12   // row 2 -- flat indices 8..11
    };

    auto matrix = std::mdspan(data.data(), 3, 4);   // 3 rows, 4 cols

    // Row-major index math: flat = row * stride(0) + col * stride(1)
    //                            = row * num_cols  + col * 1
    // matrix[1, 2] -> 1*4 + 2 = 6 -> data[6] = 7
    CHECK(matrix[1, 2] == 7.0f);
    CHECK(matrix[0, 0] == 1.0f);
    CHECK(matrix[2, 3] == 12.0f);
    CHECK(matrix.extent(0) == 3);   // rows
    CHECK(matrix.extent(1) == 4);   // cols
    CHECK(matrix.stride(0) == 4);   // skip 4 elements to reach the next row
    CHECK(matrix.stride(1) == 1);   // adjacent columns are adjacent in memory

    // The proof that this is a VIEW, not a copy: writing through the
    // mdspan must change data[] itself. matrix[0,0] -> flat index 0.
    matrix[0, 0] = 99.0f;
    CHECK(data[0] == 99.0f);
    // And the reverse direction: mutating the vector must show up
    // through the view immediately, since both just read the same bytes.
    data[11] = -1.0f;
    CHECK(matrix[2, 3] == -1.0f);
}

// ---- Part 2: fully static extents -- zero runtime shape metadata ----
//
// std::extents<size_t, 32, 128> bakes both dimensions into the TYPE.
// No sizes are passed to the constructor at all: the compiler already
// knows the full stride formula and can constant-fold or unroll freely.
void demo_static_extents() {
    using Shape = std::extents<size_t, 32, 128>;
    float tile_buf[32 * 128];
    for (size_t i = 0; i < 32 * 128; ++i) tile_buf[i] = static_cast<float>(i);

    auto tile = std::mdspan<float, Shape>(tile_buf);   // no size args -- shape is in the type

    // tile[5, 7] -> flat index 5*128 + 7 = 647
    CHECK(tile[5, 7] == 647.0f);
    CHECK(tile.extent(0) == 32);
    CHECK(tile.extent(1) == 128);
}

// ---- Part 3: mixed extents -- the realistic LLM embedding case ----
//
// batch_size is NOT known until a request arrives; EMBED_DIM is a
// property of the model checkpoint and never changes at runtime. Only
// the dynamic slot is passed in parentheses -- the compile-time 4096 is
// already part of the type and must NOT be passed again.
void demo_mixed_extents() {
    constexpr size_t EMBED_DIM = 4096;         // Llama-3-8B hidden size
    size_t batch_size = 3;                     // decided per request

    std::vector<float> raw(batch_size * EMBED_DIM);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<float>(i);

    using Shape = std::extents<size_t, std::dynamic_extent, 4096>;
    auto embeddings = std::mdspan<float, Shape>(raw.data(), batch_size);

    CHECK(embeddings.extent(0) == 3);     // dynamic: the batch we just passed in
    CHECK(embeddings.extent(1) == 4096);  // static: baked into the type

    // embeddings[2, 100] -> flat index 2*4096 + 100 = 8292
    CHECK(embeddings[2, 100] == 8292.0f);
    CHECK(embeddings[0, 0] == 0.0f);
    CHECK(embeddings[1, 0] == 4096.0f);
    CHECK(embeddings[2, 99] == 8291.0f);
}

int main() {
    demo_dynamic_matrix();
    demo_static_extents();
    demo_mixed_extents();

    std::cout << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 \
    -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental \
    -I/path/to/mdspan/include 01_mdspan_views_and_extents.cpp -o 01_mdspan_views_and_extents
./01_mdspan_views_and_extents
```

**Sample input:** a 3x4 row-major matrix of the values 1 through 12, a fully static 32x128 tile, and a `[batch=3, 4096]` mixed-extents embedding tensor, each constructed as this section describes.

**Sample output:**

```text
18/18 checks passed
ALL CHECKS PASSED
```

Every check above passes without a single heap allocation beyond the original `std::vector` — `std::mdspan` itself never calls `new`, `malloc`, or anything that could fail independently of the buffer it was handed.

!!! warning "[COMMON TRAP] Assuming std::mdspan copies its data, or that it can outlive the buffer it views"
    `std::mdspan` owns nothing. It stores a pointer and some shape/stride metadata, full stop — there is no reference count, no lifetime extension, and no copy of a single byte. If the `std::vector` (or array, or arena allocation) an `mdspan` was built over goes out of scope, is resized in a way that reallocates, or is freed, every `mdspan` still pointing at it is now a dangling view, and using it is undefined behavior exactly as if a raw pointer had outlived its target — because that is precisely what happened. This is not a defect to work around; it is the entire reason `mdspan` is cheap enough to build and destroy freely, and later sections (the arena allocator especially) depend directly on this "just a view" property to make `reset()` an O(1) operation.

## 2.2 Zero-Copy Slicing With std::submdspan

### Intuition

Multi-head attention stores Q, K, and V projections as one tensor shaped `[Batch, Seq, Heads, HeadDim]`, but every head needs to operate on its own slice independently. Copying each head out into its own buffer before processing it would mean copying most of the tensor, every single step, purely to satisfy a shape requirement the data already satisfies — it is contiguous, it is already there, it just needs a different *view* of the same bytes. `std::submdspan` is that view.

### The Concept, In Detail

`std::submdspan(view, arg0, arg1, ...)` takes one argument per dimension of `view`, and each argument is one of three things: `std::full_extent` keeps that dimension exactly as it is, an integer `k` fixes that dimension to index `k` and *drops* it from the result (the resulting rank is one less than the input's), and `std::pair{start, end}` keeps a half-open range `[start, end)` of that dimension without dropping it. The one property every reader needs to internalize before using this in real code: `submdspan` never reorganizes memory, so the strides of the *kept* dimensions do not necessarily shrink to match the new, smaller shape. Extracting a single attention head from a `[Batch, Seq, Heads, HeadDim]` tensor keeps that head's `HeadDim` elements contiguous (they always were), but moving from one sequence position to the next, *within that one head*, still has to skip over every other head's data in between — so the resulting sub-view's stride for the sequence dimension is unchanged from the original tensor's, not shrunk down to `HeadDim`. Getting this wrong — assuming a sliced view is as contiguous as its shape suggests — is exactly the trap this section's example is built to catch.

### Code and Verification

```cpp
// Chapter 2.2 -- std::submdspan builds a sub-view of an existing mdspan
// with NO copy: it returns a new mdspan over the same buffer, with a
// smaller shape and strides adjusted so indexing still lands on the
// right elements. This is exactly the operation multi-head attention
// needs: Q/K/V projections are stored as one tensor of shape
// [Batch, Seq, Heads, HeadDim], and each head must be extracted and
// operated on independently without copying its slice out first.
//
// Every dimension argument to submdspan is one of:
//   std::full_extent        -- keep the whole dimension (rank unchanged)
//   an integer k             -- fix this dimension to index k (rank drops by 1)
//   std::pair{start, end}    -- keep a half-open range [start, end) (rank unchanged)

#include <mdspan/mdspan.hpp>
#include <vector>
#include <iostream>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

int main() {
    // Small, hand-traceable shape: [Batch=1, Seq=4, Heads=3, HeadDim=2].
    // data[i] = i, so every flat index IS its own value -- makes every
    // check below verifiable by hand from the stride arithmetic alone.
    constexpr size_t BATCH = 1, SEQ = 4, HEADS = 3, HEAD_DIM = 2;
    constexpr size_t TOTAL = BATCH * SEQ * HEADS * HEAD_DIM;  // 24

    std::vector<float> data(TOTAL);
    for (size_t i = 0; i < TOTAL; ++i) data[i] = static_cast<float>(i);

    // Row-major strides for [1,4,3,2]: stride(3)=1, stride(2)=2,
    // stride(1)=HEADS*HEAD_DIM=6, stride(0)=SEQ*HEADS*HEAD_DIM=24.
    using Shape4D = std::dextents<size_t, 4>;
    auto full = std::mdspan<float, Shape4D>(data.data(), BATCH, SEQ, HEADS, HEAD_DIM);

    CHECK(full.stride(0) == 24 && full.stride(1) == 6 &&
          full.stride(2) == 2  && full.stride(3) == 1);
    CHECK(full[0, 0, 0, 0] == 0.0f);
    CHECK(full[0, 0, 1, 0] == 2.0f);   // 0*24 + 0*6 + 1*2 + 0 = 2
    CHECK(full[0, 1, 0, 0] == 6.0f);   // 0*24 + 1*6 + 0*2 + 0 = 6
    CHECK(full[0, 3, 2, 1] == 23.0f);  // 0*24 + 3*6 + 2*2 + 1 = 23

    // --- Extract head 1 across every batch and sequence position ---
    // dim 2 (Heads) collapses to the integer 1: rank drops from 4 to 3.
    // Result shape: [Batch=1, Seq=4, HeadDim=2].
    auto head1 = std::submdspan(full,
        std::full_extent,   // keep all batches
        std::full_extent,   // keep all sequence positions
        1,                  // fix head index = 1 (this dimension is dropped)
        std::full_extent);  // keep all head dims

    CHECK(head1.extent(0) == 1 && head1.extent(1) == 4 && head1.extent(2) == 2);
    // The elements within one head stay stride-1 (dim 3 was untouched),
    // but moving to the SAME head at the next sequence position still
    // has to skip over the other two heads -- so stride(1) stays 6, not 2.
    // This is the key fact submdspan must get right: it never reorganizes
    // memory, only the metadata describing how to walk it.
    CHECK(head1.stride(0) == 24 && head1.stride(1) == 6 && head1.stride(2) == 1);
    CHECK(head1[0, 0, 0] == 2.0f && head1[0, 0, 1] == 3.0f);   // seq=0 -> flat 2,3
    CHECK(head1[0, 1, 0] == 8.0f && head1[0, 3, 0] == 20.0f);  // seq=1,3 -> flat 8,20

    // Proof that submdspan is a VIEW, not a copy: writing through it
    // must land in the original buffer, at the flat index the stride
    // math predicts.
    float saved = head1[0, 0, 0];
    head1[0, 0, 0] = 999.0f;
    CHECK(data[2] == 999.0f);
    head1[0, 0, 0] = saved;  // restore for the checks that follow

    // --- Range slice: keep only sequence positions 1 and 2 ---
    // dim 1 becomes std::pair{1,3}: rank is UNCHANGED (still 4), but
    // extent(1) shrinks from 4 to 2. Strides are untouched -- only the
    // starting offset and the reported extent change.
    auto seq_window = std::submdspan(full,
        std::full_extent,
        std::pair{1UL, 3UL},   // seq positions 1 and 2 only
        std::full_extent,
        std::full_extent);

    CHECK(seq_window.extent(0) == 1 && seq_window.extent(1) == 2 &&
          seq_window.extent(2) == 3 && seq_window.extent(3) == 2);
    CHECK(seq_window[0, 0, 0, 0] == 6.0f);   // corresponds to full[0,1,0,0]
    CHECK(seq_window[0, 1, 0, 0] == 12.0f);  // corresponds to full[0,2,0,0]

    // --- Combined: one head from a sequence window, in one call ---
    // seq -> pair{1,3} (kept, shrunk), head -> 2 (dropped). Result rank 3.
    auto combined = std::submdspan(full,
        std::full_extent,
        std::pair{1UL, 3UL},
        2,
        std::full_extent);

    CHECK(combined.extent(0) == 1 && combined.extent(1) == 2 && combined.extent(2) == 2);
    CHECK(combined[0, 0, 0] == 10.0f && combined[0, 0, 1] == 11.0f);  // seq=1, head=2
    CHECK(combined[0, 1, 0] == 16.0f && combined[0, 1, 1] == 17.0f);  // seq=2, head=2

    std::cout << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 \
    -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental \
    -I/path/to/mdspan/include 02_submdspan_attention_slicing.cpp -o 02_submdspan_attention_slicing
./02_submdspan_attention_slicing
```

**Sample input:** a hand-traceable `[Batch=1, Seq=4, Heads=3, HeadDim=2]` tensor of 24 elements, filled so that `data[i] == i`, making every stride and index claim below checkable by hand.

**Sample output:**

```text
16/16 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Assuming a fixed (dropped) dimension resets the remaining strides to be contiguous"
    Fixing `Heads` to a single index removes that dimension from the *shape*, but every other dimension keeps the exact stride it had in the original tensor — because the underlying bytes never moved. In this section's own trace, extracting head 1 leaves the sequence dimension's stride at 6 (the full `Heads * HeadDim` of the original tensor), not 2 (`HeadDim` alone, which is what a "the slice is now its own small tensor" assumption would predict). Code that reads a sub-view's `.stride()` values directly is safe; code that assumes a sliced tensor is as tightly packed as a freshly-allocated one of the same shape will silently compute wrong flat offsets the moment it bypasses `mdspan`'s own indexing.

## 2.3 Cache-Optimized Layouts: Why Loop Order Is Not Free

### Intuition

Reading `data[5]` does not fetch four bytes — the CPU pulls in a full 64-byte cache line, sixteen floats, whether the program asked for them or not. A loop that reads memory in increasing address order gets the rest of that cache line for free, and the hardware prefetcher keeps the *next* line ready before the loop even asks for it. A loop that jumps across a large stride on every iteration gets none of that: each access is a fresh, uncached fetch, and the prefetcher — tuned for sequential access — cannot help at all.

### The Concept, In Detail

For an `N x N` matrix, `layout_right` (row-major) gives stride `N` to the row dimension and stride 1 to the column dimension; `layout_left` (column-major) is the exact reverse. The golden rule follows directly: the innermost loop should iterate whichever dimension has stride 1, because that is the dimension where consecutive iterations touch consecutive memory. Get this backward — iterate the stride-`N` dimension innermost — and every iteration of the inner loop jumps `N` elements forward, guaranteeing a fresh cache line (and, for large enough `N`, a fresh page) on nearly every single access. This book's own policy (stated in full on the Getting Started page) is to never fabricate a timing number, and specifically to prefer a genuinely computed, deterministic quantity over a wall-clock measurement wherever this book's argument can be made with one — because a millisecond figure captured once, on one machine, is not reproducible on a rerun, let alone on a reader's own hardware. The code below follows that policy directly: the two loops are genuinely timed, but only a qualitative, best-of-several-trials verdict (was the stride-`N` traversal slower) is locked into this section's checked output; the raw millisecond figures are still computed for real, and printed, but to `stderr`, explicitly labeled as informational and expected to differ on every rerun.

### Code and Verification

```cpp
// Chapter 2.3 -- this file measures the SAME data, walked in the SAME
// loop order, through two different std::mdspan layouts, to show that
// layout choice is a real performance decision and not a cosmetic one.
//
// This book's own getting-started page states its policy directly:
// timing numbers are never fabricated, and wherever a genuinely
// computed, deterministic quantity can stand in for a wall-clock
// number, this book uses that instead, specifically because wall-clock
// time captured once is not reproducible on a rerun or on a reader's
// own hardware. So this file's LOCKED, checked output (stdout) is a
// deterministic structural fact -- the stride each loop actually walks,
// and a checksum proving both traversals visited every element exactly
// once -- plus a qualitative finding (which layout was faster) taken as
// the best of several trials. The genuinely-measured raw millisecond
// numbers are also printed, to stderr, explicitly labeled as
// informational and expected to differ on every rerun and on every
// machine.

#include <mdspan/mdspan.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <limits>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

constexpr size_t N = 2048;          // N*N = 4,194,304 floats (16 MB)
constexpr int TRIALS = 5;

// Sums every element of `view` by walking i (outer) then j (inner), and
// returns both the sum and the elapsed time of that one trial.
template <typename View>
std::pair<float, double> timed_sum(View view) {
    auto t0 = std::chrono::steady_clock::now();
    float s = 0.0f;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            s += view[i, j];
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {s, ms};
}

int main() {
    // All-ones data: every partial sum along either traversal is an
    // integer no larger than N*N = 4,194,304, which is well under
    // float's exact-integer ceiling of 2^24 (16,777,216). That makes
    // the two sums below exactly, bit-for-bit equal regardless of
    // summation order -- a deterministic correctness check, not a
    // fuzzy "close enough" comparison.
    std::vector<float> data(N * N, 1.0f);

    // Same pointer, same bytes -- only the stride math differs.
    auto row_view = std::mdspan<float, std::dextents<size_t, 2>,
                                 std::layout_right>(data.data(), N, N);
    auto col_view = std::mdspan<float, std::dextents<size_t, 2>,
                                 std::layout_left>(data.data(), N, N);

    CHECK(row_view.stride(0) == N && row_view.stride(1) == 1);
    CHECK(col_view.stride(0) == 1 && col_view.stride(1) == N);

    double best_row_ms = std::numeric_limits<double>::infinity();
    double best_col_ms = std::numeric_limits<double>::infinity();
    float row_sum = 0.0f, col_sum = 0.0f;

    for (int t = 0; t < TRIALS; ++t) {
        auto [rs, rms] = timed_sum(row_view);
        auto [cs, cms] = timed_sum(col_view);
        row_sum = rs; col_sum = cs;
        best_row_ms = std::min(best_row_ms, rms);
        best_col_ms = std::min(best_col_ms, cms);
        // Raw, per-trial numbers: informational only. Sent to stderr so
        // they never enter this file's locked, verified stdout -- they
        // are expected to differ on every rerun and on every machine.
        std::cerr << "trial " << t << " (informational, will differ on rerun): "
                   << "row=" << rms << " ms, col=" << cms << " ms\n";
    }

    CHECK(row_sum == static_cast<float>(N * N));
    CHECK(col_sum == static_cast<float>(N * N));
    CHECK(row_sum == col_sum);

    // The deterministic structural facts this section's argument
    // actually rests on: the inner loop's stride under each layout.
    std::cout << "row-major inner-loop stride: " << row_view.stride(1)
              << " element(s) (" << row_view.stride(1) * sizeof(float) << " bytes) -- sequential\n";
    std::cout << "col-major inner-loop stride: " << col_view.stride(1)
              << " element(s) (" << col_view.stride(1) * sizeof(float) << " bytes) -- jumps one full row per step\n";
    std::cout << "row-major sum: " << static_cast<long long>(row_sum) << "\n";
    std::cout << "col-major sum: " << static_cast<long long>(col_sum) << "\n";
    std::cout << "sums match (every element visited exactly once, either order): "
              << (row_sum == col_sum ? "yes" : "NO -- BUG") << "\n";

    // Qualitative finding, taken from the BEST of several trials (per
    // this book's own benchmarking checklist: several trials, take the
    // most favorable one to each side, since a single slow trial from
    // scheduling noise should not decide the comparison). The raw
    // winning times are still not part of this locked block -- only the
    // yes/no verdict is.
    bool col_slower = best_col_ms > best_row_ms;
    std::cout << "col-major (stride-" << N << ") slower than row-major (stride-1) "
              << "on this run's best trial: " << (col_slower ? "yes" : "no") << "\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 \
    -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental \
    -I/path/to/mdspan/include 03_layout_cache_impact.cpp -o 03_layout_cache_impact
./03_layout_cache_impact
```

**Sample input:** a single `2048x2048` (4,194,304-element) buffer of all `1.0f`, viewed twice — once `layout_right`, once `layout_left` — summed with the identical `i`-outer, `j`-inner loop both times.

**Sample output:**

```text
row-major inner-loop stride: 1 element(s) (4 bytes) -- sequential
col-major inner-loop stride: 2048 element(s) (8192 bytes) -- jumps one full row per step
row-major sum: 4194304
col-major sum: 4194304
sums match (every element visited exactly once, either order): yes
col-major (stride-2048) slower than row-major (stride-1) on this run's best trial: yes

5/5 checks passed
ALL CHECKS PASSED
```

On the machine this book was built on, the stride-2048 traversal ran roughly two to two-and-a-half times slower than the stride-1 traversal, on every trial — a real, measured, and repeatable qualitative result, even though the exact millisecond figures behind it are not something this book locks in as if they were portable to different hardware.

!!! warning "[COMMON TRAP] Assuming -O3 -march=native will fix a bad loop order"
    Aggressive optimization flags can vectorize a stride-1 inner loop far more effectively than a stride-`N` one, which sometimes makes the *gap* between the two orderings even larger at higher optimization levels, not smaller — there is no compiler flag that turns a memory-bound, cache-hostile access pattern into a cache-friendly one after the fact. The fix is always in the loop order or the underlying layout choice itself, never in the optimizer being asked to work harder around it.

## 2.4 The Arena Allocator: O(1) Alloc and Reset for Per-Token Scratch

### Intuition

Every token a model generates needs a handful of scratch buffers — an activation vector, a temporary for layer norm, room for attention scores — that live for exactly one step and are then completely dead. Routing that through `malloc`/`free` on every single token means paying allocator bookkeeping and syscall overhead on the hottest loop in the program, and risking heap fragmentation as thousands of same-sized allocations are made and freed in a tight cycle. An arena sidesteps both problems by asking the operating system for memory exactly once.

### The Concept, In Detail

An `Arena` makes one `std::aligned_alloc` call at construction, aligned to 64 bytes — one CPU cache line, and the width of an AVX-512 register, since an unaligned pointer handed to an aligned SIMD load is undefined behavior rather than merely slow. From there, `alloc<T>(count)` is a bump-pointer allocation: advance an offset by `count * sizeof(T)` (plus whatever padding `std::align` needs to satisfy `T`'s own alignment), and hand back the old offset — no syscall, no bookkeeping beyond one integer addition. `reset()` is the other half of the design: it rewinds the offset to zero and touches nothing else. It does not zero memory, and it does not call any destructors — it is safe only because the caller already knows every allocation made since the last reset is logically dead, exactly the guarantee one full token-generation step provides. A pleasant side effect follows directly: the next token's allocations reuse the *exact same physical addresses* as the previous token's, so they are often still warm in L2 or L3 cache.

### Code and Verification

```cpp
// Chapter 2.4 -- token generation allocates and frees small scratch
// tensors on every single step (an activation buffer, a temporary for
// layer norm, and so on). Routing that through malloc/free invites heap
// fragmentation and per-call syscall/bookkeeping overhead on the
// hottest loop in the whole program. An Arena instead makes ONE
// allocation up front, then serves every request with a pointer bump
// (an add and a comparison -- no syscall), and "frees" everything for
// the next token with a single O(1) reset that does not zero or touch
// any memory at all.
//
// Every allocation is aligned to 64 bytes -- one CPU cache line, and
// the width of an AVX-512 register -- because an unaligned pointer
// handed to an aligned SIMD load is undefined behavior, not just slow.

#include <cstddef>      // std::byte
#include <cstdlib>      // std::aligned_alloc, std::free
#include <memory>       // std::align
#include <stdexcept>    // std::runtime_error, std::bad_alloc
#include <iostream>
#include <chrono>
#include <cstdint>
#include <mdspan/mdspan.hpp>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

class Arena {
    static constexpr size_t ALIGN = 64;   // one cache line / one AVX-512 register

    std::byte* m_mem;
    size_t m_cap;
    size_t m_off = 0;

public:
    explicit Arena(size_t bytes) : m_cap(bytes) {
        m_mem = static_cast<std::byte*>(std::aligned_alloc(ALIGN, bytes));
        if (!m_mem) throw std::runtime_error("Arena allocation failed");
    }
    ~Arena() { std::free(m_mem); }   // the only free() call in the hot path

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Returns a pointer to `count` objects of T, aligned to
    // max(alignof(T), 64). Cost: one std::align call (a handful of
    // integer ops), never a syscall. Throws std::bad_alloc if the
    // arena is exhausted.
    template <typename T>
    T* alloc(size_t count) {
        size_t need = count * sizeof(T);
        void* ptr = m_mem + m_off;
        size_t space = m_cap - m_off;
        size_t al = alignof(T) > ALIGN ? alignof(T) : ALIGN;

        if (!std::align(al, need, ptr, space))
            throw std::bad_alloc{};

        size_t pad = static_cast<std::byte*>(ptr) - (m_mem + m_off);
        m_off += pad + need;
        return reinterpret_cast<T*>(ptr);
    }

    // "Free" everything in O(1): just rewind the offset. The bytes
    // themselves are untouched -- the next round of alloc() calls will
    // simply overwrite them, and since they are the SAME physical
    // addresses as last time, they are likely still warm in L2/L3.
    void reset() { m_off = 0; }

    size_t used() const { return m_off; }
    size_t capacity() const { return m_cap; }
};

// One simulated token-generation step: allocate two scratch vectors
// from the arena, fill them through mdspan views, and return a
// checksum so the caller can confirm the arithmetic is correct without
// needing the buffers to outlive this call (the arena will reclaim them).
float inference_step(Arena& arena) {
    constexpr size_t DIM = 4096;
    float* buf_a = arena.alloc<float>(DIM);
    float* buf_b = arena.alloc<float>(DIM);

    using Shape = std::extents<size_t, DIM>;
    auto vec_a = std::mdspan(buf_a, Shape{});
    auto vec_b = std::mdspan(buf_b, Shape{});

    float checksum = 0.0f;
    for (size_t i = 0; i < DIM; ++i) {
        vec_a[i] = static_cast<float>(i);
        vec_b[i] = vec_a[i] * 0.5f;
        checksum += vec_b[i];
    }
    return checksum;
}

int main() {
    // --- Structural checks: capacity, alignment, reset semantics ---
    Arena scratch(1024 * 1024);   // 1 MiB
    CHECK(scratch.capacity() == 1024 * 1024);
    CHECK(scratch.used() == 0);

    float* p1 = scratch.alloc<float>(10);
    CHECK(p1 != nullptr);
    CHECK(scratch.used() > 0);
    // Every allocation must land on a 64-byte boundary -- checked as a
    // boolean, never by printing the address itself (a raw pointer
    // value is not reproducible across runs or machines, so it can
    // never appear in this file's locked, verified output).
    CHECK(reinterpret_cast<uintptr_t>(p1) % 64 == 0);

    scratch.reset();
    CHECK(scratch.used() == 0);
    float* p2 = scratch.alloc<float>(10);
    CHECK(p2 == p1);   // reset() rewinds the offset -- same physical bytes reused

    // A request that cannot fit must throw, not silently corrupt memory.
    bool caught = false;
    try {
        Arena tiny(256);
        tiny.alloc<float>(1000);   // 4000 bytes > 256-byte capacity
    } catch (const std::bad_alloc&) {
        caught = true;
    }
    CHECK(caught);

    // --- Correctness of the simulated token loop ---
    // vec_b[i] = i * 0.5, for i in [0, 4096); checksum = 0.5 * sum(i).
    // sum(0..4095) = 4095*4096/2 = 8,386,560; checksum = 4,193,280.0,
    // which (like Section 2.3's benchmark) stays well under float's
    // exact-integer ceiling of 2^24, so this is bit-for-bit reproducible.
    Arena token_arena(1024 * 1024);
    float checksum = inference_step(token_arena);
    CHECK(checksum == 4193280.0f);
    token_arena.reset();
    CHECK(token_arena.used() == 0);

    // --- Timed loop: 1000 simulated token steps, alloc+reset each time ---
    // The result this section actually needs is structural (the arena
    // ends every iteration back at zero, proving reset() genuinely
    // reclaims both buffers with no leak) -- so that is what gets
    // locked into stdout. The genuinely-measured wall-clock total is
    // still computed for real, but goes to stderr, informational only,
    // exactly as Section 2.3 treated its own timing.
    constexpr int TOKENS = 1000;
    auto t0 = std::chrono::steady_clock::now();
    bool all_zero_after_reset = true;
    for (int t = 0; t < TOKENS; ++t) {
        inference_step(token_arena);
        token_arena.reset();
        if (token_arena.used() != 0) all_zero_after_reset = false;
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cerr << "informational, will differ on rerun: " << TOKENS
              << " simulated token steps in " << us / 1000.0 << " ms ("
              << us / TOKENS << " us/token)\n";

    CHECK(all_zero_after_reset);

    std::cout << "arena capacity: " << scratch.capacity() << " bytes\n";
    std::cout << "arena used after alloc+reset+realloc cycle: " << scratch.used() << " bytes\n";
    std::cout << "reused same address after reset(): " << (p2 == p1 ? "yes" : "no") << "\n";
    std::cout << "all " << TOKENS << " simulated token steps left the arena at offset 0 after reset(): "
              << (all_zero_after_reset ? "yes" : "no") << "\n";
    std::cout << "single-step checksum (vec_b sum, DIM=4096): " << checksum << "\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 \
    -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental \
    -I/path/to/mdspan/include 04_arena_allocator.cpp -o 04_arena_allocator
./04_arena_allocator
```

**Sample input:** a 1 MiB arena exercised through an alloc/reset/realloc cycle, a deliberately too-small 256-byte arena used to confirm overflow throws rather than corrupts, and 1,000 simulated token-generation steps (each allocating two 4096-element scratch vectors and resetting) run back to back.

**Sample output:**

```text
arena capacity: 1048576 bytes
arena used after alloc+reset+realloc cycle: 40 bytes
reused same address after reset(): yes
all 1000 simulated token steps left the arena at offset 0 after reset(): yes
single-step checksum (vec_b sum, DIM=4096): 4.19328e+06

11/11 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Calling reset() while a pointer from before the reset is still in use"
    `reset()` does not know, and cannot know, whether every pointer it previously handed out is actually done being used — it trusts the caller completely. Holding onto a pointer from before a `reset()` call and dereferencing it afterward is a use-after-free in exactly the same sense as a dangling pointer into `free()`'d memory, except the corruption can be far more confusing to debug: the memory is very likely to still contain *plausible-looking* data (quite possibly the same tensor, briefly, until the next `alloc()` overwrites it), rather than failing loudly. The arena's contract is strict and entirely on the caller: every allocation made since the last `reset()` must be provably dead — not merely unused for now — before `reset()` is called again.

## 2.5 Mixed Precision: Type-Safe Multi-Format Storage With std::variant

### Intuition

Model weights need to be stored compactly; accumulators and intermediate activations often need more headroom than a compact format provides. A codebase that handles both ends up needing the same kernel logic to run correctly over more than one underlying element type, and the naive way to allow that — a `void*` plus a manually-tracked type tag — turns any mismatch between the tag and the actual bytes into silent, undefined-behavior garbage rather than a compiler error.

### The Concept, In Detail

`std::variant<std::vector<float>, std::vector<bf16_t>>` is a tagged union that always knows which alternative it currently holds, and `std::visit` dispatches to a templated callable, instantiated once per alternative the variant can hold — at runtime, there is exactly one branch (which instantiation to call), and everything after that is ordinary, fully-optimizable, monomorphic code, not a chain of runtime type checks repeated per element. BF16 (bfloat16) keeps FP32's full 8-bit exponent and only truncates the mantissa from 23 bits to 7, which is precisely why it is the default storage format for modern open LLM checkpoints: it has essentially the same *dynamic range* as FP32, so it almost never overflows during ordinary inference, at half of FP32's memory footprint. FP16 makes a different trade — more mantissa bits (better precision) for a much narrower 5-bit exponent, capping its representable range at 65,504 in magnitude, a bound that ordinary LLM activations exceed often enough that FP16 is a genuine overflow hazard at the same 2-byte size BF16 occupies safely.

### Code and Verification

```cpp
// Chapter 2.5 -- BF16 (bfloat16) is the de facto storage format for
// modern open LLM weights (Llama, Mistral, Gemma and most others ship
// checkpoints in it). It keeps FP32's 8-bit exponent -- so it has
// essentially the same DYNAMIC RANGE as FP32 and almost never overflows
// during inference -- while dropping to a 7-bit mantissa, for half the
// memory of FP32 and roughly 2-3 decimal digits of precision. FP16 is a
// different trade entirely: it keeps more mantissa bits (better
// precision) but only a 5-bit exponent, so its range tops out at
// +-65,504 -- a bound ordinary LLM activations cross often enough that
// FP16 is a genuine overflow hazard where BF16, at the same 2-byte
// size, is not.
//
// This file builds a minimal, portable BF16 type by hand (truncating
// the top 16 bits of an FP32's bit pattern -- exactly what real BF16
// hardware conversion does), then stores both FP32 and BF16 tensors
// behind a single std::variant so the SAME kernel code (written once,
// as a template) can run correctly over either representation, with
// the compiler choosing which at compile time from the variant's
// active type, not a runtime type check on every element.

#include <variant>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>   // std::memcpy -- safe type-punning between float and uint32_t
#include <cmath>     // std::isfinite

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// A hand-rolled BF16: 16 bits, holding exactly the upper half of an
// FP32's bit pattern (same 1 sign + 8 exponent bits, top 7 of the 23
// mantissa bits). std::memcpy is used instead of a pointer cast to
// avoid undefined behavior from type-punning through a reinterpreted
// pointer -- the compiler optimizes it down to the same instructions
// on every mainstream target.
struct bf16_t {
    uint16_t bits = 0;

    bf16_t() = default;
    bf16_t(float f) {
        uint32_t fbits;
        std::memcpy(&fbits, &f, 4);
        bits = static_cast<uint16_t>(fbits >> 16);   // keep only the top 16 bits
    }
    operator float() const {
        uint32_t fbits = static_cast<uint32_t>(bits) << 16;   // zero-fill the lower 16
        float f;
        std::memcpy(&f, &fbits, 4);
        return f;
    }
};

// A tagged union: the variant always knows which type it currently
// holds, and accessing it through std::visit is checked by the
// compiler, not left to a raw void* cast that silently reads garbage
// if the assumed type is wrong.
using Storage = std::variant<std::vector<float>, std::vector<bf16_t>>;

// One visitor, instantiated once per type the variant can hold. At
// runtime there is exactly one branch (which instantiation to call);
// everything after that is ordinary, monomorphic, fully-inlinable code.
struct ScaleKernel {
    float factor;
    template <typename Vec>
    void operator()(Vec& vec) const {
        for (auto& val : vec) {
            float f = static_cast<float>(val);
            val = static_cast<typename Vec::value_type>(f * factor);
        }
    }
};

int main() {
    Storage fp32_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    Storage bf16_data = std::vector<bf16_t>{
        bf16_t(1.0f), bf16_t(2.0f), bf16_t(3.0f), bf16_t(4.0f)
    };

    // Round-trip check: 1.0, 2.0, and 4.0 are all exact powers of two
    // (or 1), so truncating BF16's lower 16 mantissa bits loses nothing
    // for these specific values -- the round trip must be bit-exact.
    CHECK(static_cast<float>(bf16_t(1.0f)) == 1.0f);
    CHECK(static_cast<float>(bf16_t(2.0f)) == 2.0f);
    CHECK(static_cast<float>(bf16_t(4.0f)) == 4.0f);

    auto read_all = [](const Storage& s) {
        std::vector<float> out;
        std::visit([&](const auto& vec) {
            for (const auto& v : vec) out.push_back(static_cast<float>(v));
        }, s);
        return out;
    };

    auto before_fp32 = read_all(fp32_data);
    auto before_bf16 = read_all(bf16_data);
    CHECK(before_fp32 == (std::vector<float>{1, 2, 3, 4}));
    CHECK(before_bf16 == (std::vector<float>{1, 2, 3, 4}));

    // Apply the SAME kernel object to both storages. The compiler
    // generates two instantiations of ScaleKernel::operator() --
    // one for std::vector<float>, one for std::vector<bf16_t> -- and
    // std::visit picks the right one for whichever type each variant
    // currently holds.
    std::visit(ScaleKernel{0.5f}, fp32_data);
    std::visit(ScaleKernel{0.5f}, bf16_data);

    auto after_fp32 = read_all(fp32_data);
    auto after_bf16 = read_all(bf16_data);
    // 0.5, 1, 1.5, 2 are all exactly representable in both FP32 and
    // BF16 (each needs only a handful of mantissa bits), so both
    // results must match the exact expected values, not merely be
    // "close" -- this is a genuine correctness check, not a tolerance
    // check papering over precision loss.
    CHECK(after_fp32 == (std::vector<float>{0.5f, 1.0f, 1.5f, 2.0f}));
    CHECK(after_bf16 == (std::vector<float>{0.5f, 1.0f, 1.5f, 2.0f}));

    // The dynamic-range argument for BF16 over FP16, made concrete:
    // an activation value comfortably inside BF16's range but already
    // past FP16's maximum finite value.
    constexpr float LARGE_ACTIVATION = 100000.0f;    // plausible unnormalized LLM activation
    constexpr float FP16_MAX = 65504.0f;             // largest finite FP16 value
    bf16_t large_bf16(LARGE_ACTIVATION);
    CHECK(LARGE_ACTIVATION > FP16_MAX);                          // this value would overflow FP16
    CHECK(static_cast<float>(large_bf16) > FP16_MAX);            // BF16 represents it as a large finite value
    CHECK(std::isfinite(static_cast<float>(large_bf16)));        // -- specifically, not inf and not NaN

    std::cout << "bf16_t round-trip of 1,2,4 exact: yes\n";
    std::cout << "fp32 before scaling: 1 2 3 4\n";
    std::cout << "bf16 before scaling: 1 2 3 4\n";
    std::cout << "fp32 after scaling by 0.5: 0.5 1 1.5 2\n";
    std::cout << "bf16 after scaling by 0.5: 0.5 1 1.5 2\n";
    std::cout << "activation " << static_cast<long long>(LARGE_ACTIVATION)
              << " exceeds FP16 max (" << static_cast<long long>(FP16_MAX) << "): yes\n";
    std::cout << "same activation stored as finite BF16: yes\n";

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_mixed_precision_bf16.cpp -o 05_mixed_precision_bf16
./05_mixed_precision_bf16
```

**Sample input:** an FP32 and a hand-rolled-BF16 `std::variant`-backed tensor, each holding the values 1, 2, 3, 4; a single `ScaleKernel{0.5}` visitor applied identically to both; and one activation value (100,000) chosen specifically to sit past FP16's maximum finite value while remaining an ordinary finite number in both FP32 and BF16.

**Sample output:**

```text
bf16_t round-trip of 1,2,4 exact: yes
fp32 before scaling: 1 2 3 4
bf16 before scaling: 1 2 3 4
fp32 after scaling by 0.5: 0.5 1 1.5 2
bf16 after scaling by 0.5: 0.5 1 1.5 2
activation 100000 exceeds FP16 max (65504): yes
same activation stored as finite BF16: yes

10/10 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] Treating FP16 as 'just a smaller BF16'"
    Both are 16-bit floats, but they split those bits differently, and the difference is not cosmetic: BF16's 8-bit exponent matches FP32's range almost exactly, while FP16's 5-bit exponent gives it a hard ceiling of 65,504 — a value unremarkable activations inside a transformer's residual stream cross often enough in practice that naively storing activations in FP16 is a standing overflow risk, silently producing `inf` rather than a loud error. Reaching for "a 16-bit float" to save memory is not a complete decision; which 16-bit float determines whether that decision is safe for LLM inference or not.

## Chapter Summary

`std::mdspan` gives this book a single, standard vocabulary for describing a tensor's shape and memory layout without owning or copying its data — read through its three template parameters (element type, extents, layout) rather than through ad hoc pointer-and-size pairs, every later chapter's code assumes this vocabulary is already familiar. `std::submdspan` extends that vocabulary to slicing: `std::full_extent`, an integer, or a `std::pair` range per dimension, with the one sharp edge this chapter measured directly — a dropped dimension does not make the remaining ones any more contiguous than they already were. Section 2.3 showed loop order is not a stylistic choice but a measured, repeatable performance decision, and did so while keeping this book's own promise never to fabricate a timing number: a deterministic structural fact and a best-of-several-trials qualitative verdict are what got locked into this chapter's verified output, not a raw millisecond figure. The arena allocator turns per-token scratch memory into one allocation and a sequence of O(1) bumps and resets, trading `malloc`/`free`'s generality for exactly the lifetime guarantee a token generation loop already provides for free. And `std::variant` plus `std::visit` let this book's later kernels be written once and run correctly over FP32 or BF16 storage, with BF16's FP32-matching exponent range explaining directly why it, and not FP16, is the default for modern LLM weights. Chapter 3 builds the computational graph these tensors flow through, using exactly the `mdspan`-based views this chapter established as its own vocabulary for what a node's input and output actually are.

## Self-Check Questions

1. Section 2.1's mixed-extents example passes only ONE runtime argument (`batch_size`) to construct a two-dimensional `mdspan`. Explain what determines the other dimension, and why passing it a second time as a runtime argument would be a compile error rather than merely redundant.
2. A colleague claims that because `std::mdspan` "is just a view," reading through it must always be at least as fast as reading through the raw pointer directly. Using Section 2.3's own measured result, explain why this claim conflates two different things.
3. Section 2.2's `head1` sub-view has `stride(1) == 6`, not `2`. Walk through, using the flat-index arithmetic this section already gave for the full tensor, why the stride does not shrink to match `head1`'s own smaller shape.
4. Suppose `std::submdspan` were called with `std::pair{0, 4}` instead of the integer `2` for the `Heads` dimension in Section 2.2's "combined" example. Would the resulting rank be 2, 3, or 4, and why?
5. The arena's `alloc<T>(count)` function can throw `std::bad_alloc`. Name the one condition that triggers it, and explain why this is preferable to the alternative of silently returning a pointer into memory the arena does not actually have room for.
6. Section 2.4 states that `reset()` does not zero memory. Using the "Common Trap" for that section, explain concretely why a stale pointer read after `reset()` is often MORE dangerous to debug than an ordinary use-after-free, not less.
7. Both FP16 and BF16 use 16 bits. Using the exponent/mantissa split this section gave for each, explain in one sentence why an activation value of 100,000 is representable in one and not the other.
8. Section 2.3 locks a boolean ("slower: yes/no") into its verified output rather than the millisecond numbers its own code genuinely computes. Explain what this book's own stated policy is for when a genuinely-measured number belongs in verified output at all, and when it belongs only in an informational side channel.

## Where We Go Next

Chapter 3 builds the computational graph that ties these tensors together into an actual forward pass — nodes wrapping `mdspan` views of arena-allocated buffers, edges describing which node's output feeds which node's input, and a scheduling order that respects those dependencies. Every operator this book adds from here forward, from quantized matrix multiplies to KV-cache updates, is a node in exactly that graph, reading and writing tensors through the same `mdspan` vocabulary this chapter established.

## Worked Solutions

**1.** The second dimension, 4096, is already part of the `mdspan`'s TYPE (`std::extents<size_t, std::dynamic_extent, 4096>`), not something supplied at runtime — the compiler has it baked in at compile time. Passing it again as a constructor argument would be a compile error (not merely redundant) because the constructor for a mixed-extents `mdspan` only accepts arguments for the slots marked `std::dynamic_extent`; there is exactly one such slot here, so exactly one argument is the only valid arity, and the compiler enforces this through the constructor's own signature rather than at runtime.

**2.** The claim conflates a NO-COPY guarantee with a SPEED guarantee — `mdspan` never being slower than a raw pointer for the SAME access pattern is true (it compiles down to the identical stride arithmetic), but Section 2.3 measured two different access patterns through two different `mdspan` layouts over the identical underlying buffer, and one was measurably, repeatably slower. "Zero-copy" describes what happens to the DATA (nothing — it is never duplicated); it says nothing about which traversal ORDER a given loop chooses to walk that data in, which is exactly the variable Section 2.3 isolated.

**3.** The full tensor's stride(1) (the sequence dimension) is `HEADS * HEAD_DIM = 6`, because advancing one sequence position in the ORIGINAL, unsliced layout means skipping past all three heads' worth of data. Fixing the `Heads` dimension to index 1 removes that dimension from `head1`'s reported SHAPE, but it does not move a single byte in memory — the same three heads' worth of data still separates one sequence position's head-1 slice from the next one's, so the stride needed to walk from one to the other is unchanged at 6, even though `head1` itself no longer has a `Heads` dimension to report that number against.

**4.** Rank 4, unchanged from the input. `std::pair{0, 4}` is a RANGE argument, which Section 2.2 defined as keeping its dimension (with a possibly smaller extent) rather than dropping it — only a bare integer drops a dimension and reduces rank. Since `Heads` has exactly 3 valid indices (0, 1, 2), `std::pair{0, 4}` would in fact be an out-of-bounds range for this specific tensor (asking to keep indices 0 through 3), which is a separate bug from the rank question, but the RANK itself would still be 4, not 3, purely from the argument being a range rather than an integer.

**5.** `alloc<T>(count)` throws `std::bad_alloc` when the requested, alignment-padded size does not fit in the arena's remaining capacity (`m_cap - m_off`) — the exact condition Section 2.4's own code checks via `std::align`'s return value. Throwing is preferable to silently returning an out-of-bounds pointer because the alternative is not a smaller, degraded allocation — it is a pointer into memory the arena does not own at all, which the caller would then read and write as if it were valid, corrupting whatever real data happens to occupy those bytes instead of failing at the one point where the actual problem (the arena is out of room) is still directly diagnosable.

**6.** An ordinary use-after-free (from `malloc`/`free`) often crashes quickly or is caught by tools like AddressSanitizer, because the memory has been returned to the allocator and may be reused for something of a completely different shape or poisoned deliberately. A stale pointer after an arena `reset()` is far more likely to still contain data from the SAME tensor shape as before — quite possibly nearly the same numbers, briefly, until the next `alloc()` call overwrites them — so a bug reading through it can produce plausible-looking output that is subtly, silently wrong (stale values from the previous token) rather than an obvious crash, making it considerably harder to notice, let alone debug.

**7.** BF16 keeps FP32's 8-bit exponent, giving it the same enormous representable range (up to roughly 3.4x10^38) that FP32 has, so 100,000 sits nowhere near its ceiling. FP16 only has a 5-bit exponent, capping its largest finite value at 65,504 — and 100,000 exceeds that ceiling, so FP16 cannot represent it as a finite number at all (it would become `inf`), while BF16, with room to spare in its exponent, represents it exactly as an ordinary finite value, only losing some mantissa precision the way it would for any value.

**8.** This book's stated policy (Getting Started, restated in Section 2.3) is: wherever a genuinely computed, deterministic quantity can carry the actual argument being made, that quantity — not a wall-clock number — belongs in this book's locked, verified output, specifically because a millisecond figure captured once is not reproducible on a rerun or on different hardware. A genuinely-measured wall-clock number is never fabricated, and is still computed and printed for real, but it belongs only in an informational side channel (this chapter used `stderr`, explicitly labeled) whenever the argument itself does not require the exact number — only a stable, real comparison, such as "slower" or "faster" — to hold.
