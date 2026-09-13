# Chapter 10: Threading and Concurrency -- Real Threads, Verified Without a Clock

**What you will understand by the end of this chapter:**

- Why a single core cannot hit a realistic interactive decode-latency target at all, derived from Chapter 8.5's own measured full-decode-step FLOP total rather than a new, separately invented workload — and why Amdahl's law puts a hard, derivable ceiling on how much speedup ANY number of additional cores can ever buy, long before a single thread is created.
- How `std::atomic` and `std::mutex` make sharing mutable state across real, concurrently running threads well-defined, verified here by actually creating threads and checking an exact expected result — and why a genuinely unsynchronized shared counter is undefined behavior whose outcome this book will not pretend to lock, for the same honesty reasons Chapter 8 never ran a wall-clock benchmark.
- How to build a persistent, `std::barrier`-synchronized thread pool that partitions a real GEMV by output row across several worker threads and reuses those same threads across many rounds of work — verified bit-for-bit against a serial reference, not merely within a tolerance.
- Why two threads writing to two completely independent variables, with no shared state and no data race by any definition, can still sit in the same 64-byte cache line — a structural, address-arithmetic fact this chapter verifies exactly, without ever needing a wall-clock number to make the point.
- How to verify a work-stealing scheduler's correctness when its own SCHEDULE is deliberately nondeterministic — by checking an order-independent invariant (every task claimed exactly once; the aggregate result exactly correct) instead of asserting anything about which thread did what or when.

**What you need to know first:**

- Chapter 8's CpuSpec and Roofline structures, and Chapter 8.5's own measured full-decode-step FLOP and byte totals, reused verbatim in Section 10.1 rather than re-derived.
- Chapter 8.3's std::mdspan-viewed weight matrix convention, reused in Section 10.3's partitioned GEMV.
- This is the first chapter in this book to create a single `std::thread`. Every file compiles with `-pthread` in addition to this book's standard flags.
- This chapter extends this book's standing policy against fabricated timing numbers into a second, related discipline: never lock output whose exact value depends on real, uncontrolled thread scheduling. A data race's outcome (Section 10.2) and a work-stealing scheduler's exact division of labor (Section 10.5) are both real, both genuinely executed, and both deliberately excluded from this chapter's checked, locked transcripts for the identical reason a wall-clock number always has been: this book only locks what is guaranteed to reproduce on a rerun. Where a section's whole point IS a concurrency hazard, it is verified through what IS guaranteed — an aggregate invariant, a structural address computation, a deterministic mutex API guarantee — never through hoping a particular schedule repeats.

---

Chapter 9 showed how a single core reaches its peak compute, one SIMD instruction at a time. This chapter reaches for the resource no single core has any more of to give: additional cores, coordinated by real threads. Section 10.1 derives why that coordination is necessary at all — a single core's real decode-step FLOP total, measured back in Chapter 8.5, is put through Chapter 8.1's own achievable-throughput formula, and comes out several times over a realistic interactive latency budget — and then derives the ceiling Amdahl's law puts on how much any number of additional cores can close that gap. Section 10.2 introduces this book's first real `std::thread`, and with it the two primitives, `std::atomic` and `std::mutex`, that make sharing mutable state across threads well-defined rather than undefined behavior, verified by running real concurrent code and checking an exact result rather than a fabricated one. Section 10.3 builds a persistent, `std::barrier`-synchronized thread pool and partitions a real GEMV across it by output row, verified bit-for-bit against a serial reference. Section 10.4 finds a hazard that changes no thread's computed value at all — two independent per-thread counters sharing one cache line — and verifies it the only honest way available without a clock: as address arithmetic. Section 10.5 closes the chapter with a work-stealing scheduler, and with it this chapter's central methodological point: a scheduler whose exact execution order is deliberately nondeterministic can still be verified with complete rigor, by checking the invariant its contract actually promises instead of the schedule it never did.

## 10.1 The Latency Budget and Amdahl's Ceiling

### Intuition

Before writing a single thread, it is worth deriving whether one is even necessary — and, if so, roughly how much parallelism could possibly help. A single core's real cost for one decode step, already measured in Chapter 8.5, run through Chapter 8.1's own achievable-throughput formula, gives a concrete answer to "is one core enough," with no new workload invented for the purpose. Amdahl's law then answers the next question honestly: however many cores are added, some fraction of a real decode step — sampling the next token, updating a cursor, scheduling the next request — never parallelizes at all, and that fraction alone puts a hard, derivable ceiling on total speedup.

### The Concept, In Detail

Chapter 8.5's own instrumented kernel measured one full decode step, at realistic transformer dimensions, at exactly 470466888 FLOPs and 890151168 bytes — an arithmetic intensity of about 0.53 FLOPs/byte, deep in the memory-bound regime Chapter 8.1's roofline model already described. Dividing Chapter 8.1's own 8-core machine's achievable throughput at that arithmetic intensity by 8 gives one core's own share, and dividing the step's real FLOP total by that single-core throughput gives a real, derived single-core time per step — which comes out several times over a stated 20-tokens-per-second interactive latency budget, with no wall-clock measurement involved anywhere in the derivation. Amdahl's law then asks a different question: given that ADDING cores is the fix, how much does it actually help. If a fraction `serial_fraction` of the work is inherently sequential and the rest splits perfectly evenly across `n` cores with zero coordination overhead — the most generous assumption possible — total relative time is `serial_fraction + (1 - serial_fraction)/n`, and speedup is its reciprocal. This formula's most important property is not its value at any particular `n`, but its limit as `n` grows without bound: `1 / serial_fraction`, a ceiling no finite number of cores can ever reach, let alone exceed, because the parallel term `(1 - serial_fraction)/n` only ever approaches zero, never becomes it.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_latency_budget_and_amdahls_law.cpp -o 01_latency_budget_and_amdahls_law
./01_latency_budget_and_amdahls_law
```

**Sample input:** Chapter 8.5's own real, measured full-decode-step FLOP and byte totals at DIM=4096, N_HEADS_Q=32, N_HEADS_KV=8, D_FF=14336, cache_len=2048, run through Chapter 8.1's own achievable-throughput formula to derive a real single-core decode-step time against a stated 20-tokens-per-second latency budget; Amdahl's law's basic algebraic properties (speedup(1)=1, strictly increasing in core count); and a deliberate demonstration, at a stated 5% serial fraction, of the gap between "8 cores means 8x" and the actual ~5.93x, together with the provable, never-reached ceiling of 20x as core count grows arbitrarily large.

@@OUT1@@

!!! warning "[COMMON TRAP] assuming N cores buys N times the speedup"
    "We have 8 cores, so we should get roughly 8x" ignores that SOME part of a real per-token step — sampling, bookkeeping, scheduling the next request — runs on exactly one core no matter how many are available. At a stated 5% serial fraction, 8 cores buys about 5.93x, not 8x, and the gap only widens as core count grows: Amdahl's law's speedup does not merely fall short of linear, it CONVERGES to a hard ceiling of `1/serial_fraction` — here, 20x — that no finite number of cores, however large, can ever reach or exceed. A team that buys twice the cores expecting twice the throughput, without first asking what fraction of the work is inherently serial, is making a purchasing decision Amdahl's law already answered before the hardware order was placed.

## 10.2 Atomics, Mutexes, and Why Synchronization Is Not Optional

### Intuition

This book's first real `std::thread` immediately raises the question every later section in this chapter depends on: what happens when more than one thread reads and writes the SAME piece of memory. `std::atomic` and `std::mutex` are two different mechanisms that both answer it the same way — by making the combined read-modify-write sequence indivisible from every other thread's point of view — verified here the way this book verifies everything: by running real threads and checking an exact result, not a probabilistic one.

### The Concept, In Detail

`std::atomic<long long>::fetch_add` performs a read-modify-write as one indivisible hardware operation; no matter how the operating system interleaves four real threads each calling it 250000 times, the C++ standard guarantees every single increment is counted, so the final total is exactly 1000000 on every run — the SCHEDULE is not deterministic, but the RESULT is, because atomicity is a guarantee about the RESULT, not about timing. `std::mutex` reaches the identical guarantee by a different mechanism: mutual exclusion ensures only one thread's `++counter` can be "in flight" at any moment, which is what makes an ordinary, otherwise-unsafe increment well-defined the moment it happens only while the lock is held. A genuinely UNSYNCHRONIZED shared counter — the same `++counter`, called from multiple real threads with no atomic or mutex protection at all — is a real data race and therefore undefined behavior in the C++ memory model; its outcome is not merely hard to predict, it is explicitly not guaranteed to be the same from one run to the next, which is exactly the property this book's build-verify-lock pipeline depends on. This section includes that racy function as real, compilable source and deliberately never executes it as part of its own checked output, the same honesty this book has applied to timing numbers from the very first chapter. A separate, smaller but equally real trap closes the section: `std::mutex` is not recursive, and a thread that already holds the lock and tries to lock it again — exactly the shape of a function calling another function that unknowingly locks the same mutex — does not get let back in.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -pthread 02_atomics_mutexes_and_synchronization.cpp -o 02_atomics_mutexes_and_synchronization
./02_atomics_mutexes_and_synchronization
```

**Sample input:** four real threads each performing 250000 `std::atomic<long long>::fetch_add` calls on a shared counter, checked against the exact expected total of 1000000; the identical total reached with a plain `long long` protected by `std::mutex` instead; a deliberate, fully deterministic demonstration that `std::mutex::try_lock()` returns `false` when a thread that already holds the lock calls it again, contrasted with `std::recursive_mutex` correctly allowing it; and an explanation of why this file's own genuinely racy function is included as source but never executed for output.

@@OUT2@@

!!! warning "[COMMON TRAP] assuming a mutex lets its own owner back in"
    A function that locks a `std::mutex` and then calls another function that — unaware the caller already holds that same lock — tries to lock it again does not get politely refused with a return value to check; a blocking `lock()` call in that situation simply never returns, because a plain `std::mutex` has no notion of "the thread that already owns me may re-enter." This section demonstrates the mechanism safely with `try_lock()`, which returns `false` deterministically rather than hanging, precisely so the demonstration itself has well-defined output to verify — but the real-world version of this bug is a silent, total deadlock the first time two code paths that both need the same lock are called from the same thread. `std::recursive_mutex` exists exactly for the case where a single thread genuinely needs to re-enter its own lock, and reaching for it (or, better, restructuring the code so re-entry is never attempted) is the fix — not discovering the hang in production.

## 10.3 A Barrier-Synchronized Thread Pool for a Partitioned GEMV

### Intuition

Chapter 8.3 already classified a decode-time GEMV as memory-bound; this section asks how to actually SPLIT that GEMV's work across several real cores. Splitting by output row is the natural choice — each row's dot product is independent of every other row's — and because no two threads ever write the same output element, the result is not merely close to a serial reference within a tolerance, the way Chapter 9.2's AVX2 kernel was; it is bit-for-bit identical, verified exactly.

### The Concept, In Detail

A `TensorThreadPool` creates its worker threads ONCE, in its constructor, and coordinates rounds of work with two `std::barrier` objects instead of creating and joining new threads for every single call — exactly the persistent structure a real decode loop needs, since a real engine cannot afford to pay thread-creation cost on every token the way Section 10.2's per-test thread creation could. Partitioning the output rows correctly means every row from 0 to n_out-1 is covered by EXACTLY one thread's range, with no gaps and no overlaps; the natural first attempt — plain integer division for a fixed chunk size — silently fails this when n_out does not divide evenly by the thread count, because truncating division drops the remainder rows from every range entirely, leaving them assigned to no thread at all. The fix is a ceiling-divided chunk size with the LAST range explicitly capped at n_out, which this section verifies both abstractly (as pure range arithmetic, with no threads involved) and concretely, by pre-filling the output with a sentinel value and confirming the buggy partition leaves exactly the dropped rows at that sentinel after a real run through the pool.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -pthread -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 03_barrier_thread_pool_gemv.cpp -o 03_barrier_thread_pool_gemv
./03_barrier_thread_pool_gemv
```

**Sample input:** a correct, ceiling-divided row partition of a 70-row output across 4 threads, checked to cover every row exactly once with no gaps or overlaps; a persistent, `std::barrier`-synchronized thread pool computing that GEMV across two separate rounds through the SAME still-alive threads, each checked bit-for-bit against a serial reference; and a deliberate demonstration of the naive, truncating-division partition silently dropping rows, run through the same real pool with a sentinel-filled output to show exactly which rows were never written by any thread.

@@OUT3@@

!!! warning "[COMMON TRAP] a chunk-size partition that quietly drops the remainder"
    `chunk = n_out / n_threads` followed by `[t*chunk, (t+1)*chunk)` for each thread looks correct and compiles without complaint, but integer division truncates: when n_out is not an exact multiple of the thread count, the rows from `n_threads * chunk` to `n_out - 1` fall outside every single thread's range. This is not a crash and not a wrong VALUE in any output element that IS computed — it is a set of output elements that are simply never touched by anyone, left at whatever the buffer happened to contain beforehand. The fix — a ceiling-divided chunk size with the final range explicitly capped at n_out — costs nothing in the common case where the division is exact, and is the only version of this partition that is correct in the case where it is not.

## 10.4 False Sharing: A Memory-Layout Hazard, Not a Race

### Intuition

Two threads, each incrementing its own logically independent counter, share no variable and therefore have no data race by any definition — and yet, depending purely on where those two counters happen to sit in memory, updating one can force the OTHER thread's cached copy of a completely different value to be invalidated. This is not a correctness bug — both counters still end up with exactly the right value, every time — so this section verifies it the only way that is honest without a wall-clock number: as address arithmetic.

### The Concept, In Detail

Cache coherence protocols keep every core's view of memory consistent by operating on whole 64-byte cache lines, not on individual variables — so four adjacent `long long` counters, packed into 32 contiguous bytes with no padding, share exactly one cache line between all four, and a write to any one of them invalidates every other core's cached copy of the entire line, including the three counters that never changed. This section measures that fact directly, computing each counter's cache-line offset RELATIVE to the containing struct's own base address rather than as a raw pointer value, since absolute addresses shift from run to run under ASLR (address space layout randomization) in a way the LAYOUT relationship between two fields of the same object never does. Padding each counter out to a full 64-byte block with `alignas(64)` and an explicit padding member gives every counter its own line, verified the identical way. Running real, concurrently writing threads against both layouts confirms the section's central claim: both reach the exact correct final count on every run, because false sharing changes only how much cache-coherence traffic the hardware has to do to make that happen, never the value any thread actually computes — which is exactly why this book will not attach a fabricated timing number to the difference.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -pthread 04_false_sharing_and_cache_line_layout.cpp -o 04_false_sharing_and_cache_line_layout
./04_false_sharing_and_cache_line_layout
```

**Sample input:** four adjacent, unpadded `long long` counters, their cache-line offsets computed relative to their containing struct's own base address, showing all four sharing line 0; the same four counters padded to 64 bytes each with `alignas(64)`, showing each one in its own distinct line; both layouts run under four real, concurrently writing threads and checked to reach the exact expected count in every case; and a deliberate demonstration that two counters sharing no variable at the source level can still be shown, by address arithmetic alone, to share a cache line.

@@OUT4@@

!!! warning "[COMMON TRAP] assuming no shared variable means no shared cache line"
    Two array elements, `naive.count[0]` and `naive.count[1]`, are two completely separate `long long` values — no thread ever reads or writes the other's element, so by any definition of a data race, there is none here. This section measured their addresses anyway and found both inside the same 64-byte cache line, because cache coherence has no concept of "these two bytes belong to logically different variables" — it only knows about lines. The hazard is invisible to anyone reading the increment statements themselves, since nothing in that source code shares anything; it only becomes visible by checking WHERE the compiler and allocator actually placed the bytes, which is exactly the check this section performs and exactly why `alignas(64)` padding is a structural fix, not a style preference.

## 10.5 Work Stealing: Verifying Correctness Under a Nondeterministic Schedule

### Intuition

Section 10.3's thread pool split work with a fixed, static partition decided before any thread ran — fine when every row costs the same, wrong the moment tasks take unequal time, since a static split leaves fast threads idle while a slow one is still working. A work-stealing scheduler fixes this by letting an idle thread take work from a busy one's own queue — but that means the exact SCHEDULE, which thread executes which task and in what order, is deliberately, genuinely nondeterministic, and this section's real subject is how to verify such a thing rigorously anyway.

### The Concept, In Detail

Every task in this section computes its own integer index as its "work," and adds it to a shared total with `std::atomic<long long>::fetch_add` — a deliberately trivial computation, because the only thing actually under test is the SCHEDULER's own correctness, not any task's arithmetic. Integer addition is exactly commutative and associative, unlike the floating-point sums Chapter 9.2's tolerance-checked kernels required, so as long as every task runs EXACTLY once, the aggregate total is a single, fixed, checkable number — `sum(0..N-1)`, a closed form — regardless of which thread claimed which task or in what order. Every task starts on ONE thread's own deque, with every other thread's deque completely empty, so those other threads can only ever obtain work by successfully stealing it from the back of their own (empty) queue failing and the front of someone else's succeeding — guaranteeing the stealing code path is genuinely exercised on every run, whatever the exact division of labor a given run happens to reach. This section checks exactly two things, both true regardless of the actual schedule: every one of N tasks claimed precisely once (checked per-task, so a dropped task and a duplicated task could not silently cancel out in an aggregate count), and the aggregate total exactly matching the closed-form sum. Which thread processed which task, and how many tasks the origin thread finished locally before the first successful steal, are both real, scheduling-dependent outcomes this section deliberately never measures or prints, for the identical honesty reason Section 10.2 never printed a data race's exact result.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -pthread 05_work_stealing_scheduler.cpp -o 05_work_stealing_scheduler
./05_work_stealing_scheduler
```

**Sample input:** 2000 tasks, every one initially placed on a single thread's deque with the other three threads starting empty (forcing every steal path to be genuinely exercised), processed by four real worker threads using mutex-protected per-thread deques with the standard owner-pops-back/thief-steals-front convention; checked per-task that every one of the 2000 tasks was claimed exactly once (zero dropped, zero duplicated); and checked that the aggregate total exactly matches the closed-form sum(0..1999), regardless of the actual, unmeasured division of labor between threads on this run.

@@OUT5@@

!!! warning "[COMMON TRAP] testing the schedule instead of the invariant"
    A tempting first test for a scheduler like this one asserts something about the SCHEDULE itself — `CHECK(which_thread_processed[5] == 2)` — because every single-threaded test in this book so far has been free to assert exactly this kind of specific, reproducible fact. A work-stealing scheduler makes no such promise: thread 0 might finish task 5 itself before any other thread gets a chance to steal it, or thread 3 might steal it on the very first attempt, and both are entirely correct outcomes of the identical, correctly functioning scheduler. A test written against one particular schedule is not testing the scheduler's actual contract at all — it is testing one incidental fact about one run, on one machine, that the C++ standard never promised would repeat. The two checks this section actually performs — every task claimed exactly once, and the aggregate result exactly correct — are the only two properties a work-stealing scheduler's contract genuinely makes, and the only two a rigorous test should ever assert.

## Chapter Summary

This chapter introduced real, concurrently executing threads into this book for the first time, and with them a discipline this book has not needed before: verifying correctness in the presence of genuine, uncontrolled scheduling nondeterminism, without ever locking a number that scheduling could change from one run to the next. Section 10.1 derived, from Chapter 8.5's own measured decode-step cost, why a single core cannot hit a realistic latency target, and derived Amdahl's law's hard ceiling on how much any number of additional cores can close that gap. Section 10.2 introduced `std::atomic` and `std::mutex` as the two mechanisms that make shared mutable state well-defined across real threads, verified by running real concurrent code to an exact result, while explicitly declining to execute a genuinely racy counterpart whose output this book cannot honestly promise would reproduce. Section 10.3 built a persistent, barrier-synchronized thread pool and partitioned a real GEMV across it, verified bit-for-bit exactly, and caught a classic partitioning bug — truncating integer division silently dropping remainder rows — concretely, by running it through the same real pool. Section 10.4 found a hazard that changes no computed value at all, verified purely as cache-line address arithmetic relative to a stable base address rather than as a wall-clock number. Section 10.5 closed the chapter with a work-stealing scheduler and this chapter's central methodological lesson: a system whose schedule is deliberately nondeterministic can still be verified with complete rigor, by testing the order-independent invariant its contract actually makes rather than any specific execution the scheduler happens to produce on a given run. Chapter 9 showed how one core reaches its peak; this chapter showed how several cores can be coordinated correctly, and, just as importantly, how to know that the verification itself is still honest once real, uncontrolled concurrency enters the picture.

## Self-Check Questions

1. Section 10.1 reuses Chapter 8.5's own measured FLOP and byte totals rather than inventing a new workload. Why does that make the single-core latency claim more trustworthy than a separately asserted number would be?
2. Explain why Amdahl's law's speedup formula converges to a finite ceiling as core count grows without bound, rather than continuing to increase toward infinity.
3. Why does `std::atomic<long long>::fetch_add` guarantee the exact same final total on every run, even though the exact interleaving of which thread's increment happens at which moment is not guaranteed to be the same?
4. Section 10.2 includes a genuinely racy `racy_increment_worker()` function but never calls it from `main()`. What specific property of the C++ memory model makes that omission necessary rather than merely cautious?
5. Why does `std::mutex::try_lock()` returning `false` for a thread that already holds the lock demonstrate the same hazard that a blocking `lock()` call would demonstrate by hanging forever, without this section's own verified output ever having to hang?
6. In Section 10.3, why does the naive `chunk = n_out / n_threads` partitioning scheme drop rows only when n_out does not divide evenly by n_threads, and what specifically happens to those dropped rows' output values?
7. Section 10.4 computes cache-line offsets relative to a struct's own base address rather than as raw absolute addresses. What would have gone wrong with this file's own run-twice determinism check if it had printed absolute addresses instead?
8. Explain why false sharing, unlike every other COMMON TRAP in Chapters 8 and 9, cannot be demonstrated by showing a wrong computed VALUE, and what kind of evidence Section 10.4 uses instead.
9. In Section 10.5, why does the fact that every task computes a plain integer (rather than a float) matter for why the aggregate total is guaranteed identical regardless of the actual thread schedule?
10. Section 10.5 deliberately starts every task on one thread's deque, leaving the other threads' deques empty. What does this specific setup guarantee about the resulting run, and what does it deliberately NOT guarantee or measure?

## Where We Go Next

This chapter built the primitives — atomics, mutexes, a barrier-synchronized thread pool, an awareness of false sharing, and a work-stealing scheduler — that make real, multi-core parallelism both fast and correctly verifiable. Chapter 11 puts every one of them to work on the actual forward pass this book has built one kernel at a time since Part 0: parallelizing attention across heads, the FFN across output rows, and a full transformer layer across a real, persistent thread pool, closing Part 2 with a complete, multi-threaded inference engine whose every claim about correctness is checked exactly, and whose every claim about performance is limited to what this chapter's own roofline and threading models can honestly derive.

## Worked Solutions

**1.** Chapter 8.5's FLOP and byte totals were themselves already verified there against a real, running GQA kernel at those exact dimensions — they are not a number this chapter is asking the reader to trust for the first time. Reusing that already-verified total means Section 10.1's latency claim inherits Chapter 8.5's own verification rather than asking for a second, separate act of trust in a freshly invented workload that has not been checked against anything.

**2.** The speedup formula's parallel-work term is `(1 - serial_fraction) / n`, which strictly decreases toward zero as `n` grows, but never reaches exactly zero for any finite `n`. Total relative time is therefore always strictly greater than `serial_fraction` itself, so speedup (its reciprocal) is always strictly less than `1 / serial_fraction` — a ceiling the formula approaches asymptotically but its own algebra forbids it from ever reaching or exceeding, no matter how large `n` becomes.

**3.** `fetch_add` is specified by the C++ standard to perform its read-modify-write as a single indivisible operation with respect to every other thread — no other thread can observe or interleave with it partway through. Because every one of the 1,000,000 total increments across four threads is guaranteed to be counted exactly once regardless of the ORDER in which they occur, the final sum is fixed by simple arithmetic (a count of atomic operations that all happened, each counted once) even though the interleaving order producing that count is genuinely different from one run to the next.

**4.** A data race is undefined behavior in the C++ memory model, and undefined behavior carries no guarantee that two runs of the identical binary, on the identical machine, produce the identical result. This book's entire build-verify-lock pipeline depends on a file's checked output being byte-for-byte reproducible on rerun; executing and printing a genuinely racy function's result would mean locking a number this book cannot honestly promise the next run — or a reader's own run — would reproduce, which is precisely the standard this book has held every other file to since its very first chapter.

**5.** Both situations arise from the identical fact: a plain `std::mutex` does not track which thread currently owns its lock in a way that would let that SAME thread back in. `try_lock()` and a blocking `lock()` differ only in what they do once that fact is discovered — `try_lock()` returns `false` immediately, a value this section can check and print deterministically, while `lock()` blocks and waits for the lock to become available, which (since the only thread that could release it is the very thread now waiting) never happens. The underlying hazard — non-recursive re-entry — is identical; only the OBSERABLE consequence differs, and this section deliberately chooses the one with well-defined, checkable output.

**6.** `chunk = n_out / n_threads` uses C++'s truncating integer division, so when `n_out` is not an exact multiple of `n_threads`, the product `n_threads * chunk` is strictly less than `n_out`, and the naive scheme's `n_threads` ranges only ever cover indices up to that truncated product. The rows from that point up to `n_out - 1` are never included in ANY thread's assigned range, so no thread ever writes to them — their output values remain at whatever they were initialized to before the pool ran (this section's own sentinel value), not merely computed incorrectly.

**7.** Absolute memory addresses are randomized by ASLR (address space layout randomization) every time a process starts, so the SAME object could legitimately sit at a different absolute address on two consecutive runs of the identical binary — meaning a printed absolute-address-derived cache-line number could differ between the two runs this book's pipeline diffs against each other, breaking the exact-match requirement for reasons that have nothing to do with the actual layout relationship being demonstrated. Printing offsets relative to the containing struct's own base address instead measures a purely compile-time-determined layout fact that ASLR does not touch, keeping the check honestly deterministic.

**8.** Every other COMMON TRAP in Chapters 8 and 9 produces a demonstrably WRONG number — a mis-decoded nibble, a divergent `__restrict` result, a SIGILL a `-march=native` binary would hit — something a check can compare against a known-correct value and find different. False sharing changes none of that: every thread's counter still reaches the exact correct final value in both the padded and unpadded layout, because the hazard is purely about how much cache-coherence traffic the hardware generates to keep those (still-correct) values consistent, not about the values themselves. Section 10.4 therefore verifies the LAYOUT directly — computing which cache line each counter's address falls into — since that is the one honestly checkable fact the hazard actually consists of.

**9.** Integer addition is exactly commutative and associative: for any set of fixed integers, their sum is identical regardless of the order they are added in, with no rounding or reordering-sensitivity at all. This is different from the floating-point sums Chapter 9.2's kernels produced, where AVX2's tree-shaped reduction and the scalar loop's sequential sum could differ in their last few bits purely from summing the same values in a different order. Because each task's contribution here is a plain integer, the scheduler's nondeterministic execution ORDER cannot introduce any numerical difference at all in the final total — only a dropped or duplicated task could, which is exactly the separate invariant Test 1 checks.

**10.** This setup guarantees that threads other than the one holding all the initial tasks can only ever obtain work through a successful steal — there is no other way for them to have anything in their own deque to pop — so the stealing code path is certain to be exercised by every single run, rather than merely possible depending on how fast each thread happens to run. It deliberately does NOT guarantee, or attempt to measure, exactly how many tasks the origin thread finishes on its own before another thread's first successful steal, nor which specific tasks end up on which thread — both are real, scheduling-dependent outcomes that legitimately vary from run to run, and this section's own checks are built specifically not to depend on either one.
