# Chapter 27: Continuous Batching and Production Serving Architecture

**What you will understand by the end of this chapter:**

- Why the naive, real batching scheme early inference servers actually used -- fixed-size batches that cannot start until the previous one fully retires -- wastes real GPU-slot time on padding and delays a short request that arrives just after a long batch has already started, quantified with exact hand-traceable numbers rather than asserted.
- Why prefill (a new request's full prompt, processed in one shot) and decode (every already-running sequence, advanced by exactly one token) have genuinely different real cost profiles, and why naively injecting an entire prefill into a single step spikes that step's own real latency for every already-running sequence sharing it.
- How real chunked prefill bounds that injected latency to a fixed, stated maximum regardless of how long the new prompt actually is, and how to derive that bound directly from the same stated cost model rather than tuning it empirically.
- How to build a real, from-scratch continuous-batching scheduler that frees a slot the instant its own sequence finishes -- immediately admitting the next waiting request into that same slot -- and to check its own real improvement over static batching directly, on the identical workload, rather than against a hardcoded number.
- How to build two real numerical debugging tools this book's own kernels have needed since Chapter 26 introduced the numerical instabilities they catch: a NaN/Inf-propagation tracer that finds where corruption actually ORIGINATED, not merely where it is still visible, and a floating-point drift detector that quantifies exactly how much a real reduction's own result changes when its order changes -- a genuine risk once continuous batching starts regrouping sequences differently from run to run.

**What you need to know first:**

- Section 26.1's own real roofline classification (GEMV is intrinsically memory-bound; GEMM becomes compute-bound only past a derivable crossover batch size) is the exact real reason this chapter's own scheduler exists at all: a serving system's whole real job is to keep the batch size a given machine actually runs as close to that crossover as real, unpredictable traffic allows.
- Section 26.2's own real numerical-instability discipline -- reproducing a genuine failure on purpose, understanding exactly why it happens, and fixing it with a provable identity rather than a downstream patch -- is the same discipline this chapter's own Section 27.4 applies to NaN propagation and floating-point drift, two real failure modes that specifically appear at serving scale rather than in a single isolated kernel test.
- Chapter 13's own KV Cache Manager and Chapter 14's own advanced eviction and prefix-caching techniques are the real memory-management machinery a production scheduler like this chapter's own Section 27.3 would sit directly on top of in a real system -- this chapter's own scheduler abstracts that machinery into a stated per-step cost, the same honest simplification Chapter 26 applied to FLOP and byte counting.

---

Every chapter before this one built one real piece of an inference engine and verified it in isolation. A real serving system has to run many of those pieces at once, for many concurrent real users, whose requests arrive at unpredictable times and need unpredictable amounts of work -- and the scheduling decisions that follow from that are not optional engineering polish, they are the actual difference between a system that serves real traffic efficiently and one that does not serve it at all. This chapter builds that scheduling layer from scratch: first by understanding, with real hand-traceable numbers, exactly what goes wrong with the naive approach; then by resolving the specific real conflict between the two fundamentally different kinds of work a serving step can do; then by building the real scheduler that puts both fixes to work; and finally by building the real numerical debugging tools a system running at this scale actually needs when something goes wrong.

## 27.1 The Static-Batching Problem and Head-of-Line Blocking

### Intuition

The simplest way to batch real inference requests together is also the most wasteful one: group a fixed number of them, run the whole group until every single member is done, and only then admit the next group. This section builds a real, from-scratch simulator for exactly that scheme, to quantify -- not merely describe -- how much real GPU-slot time it wastes and how badly it can delay a request unlucky enough to arrive at the wrong moment.

### The Concept, In Detail

`simulate_static_batching` groups a real, arrival-ordered queue of requests into fixed-size chunks and runs each chunk as a single real batch: the batch cannot start until every one of its own members has arrived AND the previous batch has fully retired, and once started, every slot in it stays occupied -- computing nothing useful once its own request has already finished -- until the LONGEST request in that batch also finishes. Test 1 traces this exactly on a tiny 4-request workload: a short request sharing a batch with a much longer one sits idle, wasting 8 real padding steps, and a separate short request that arrives just one step after the batch already started is not just delayed by that one step -- it is blocked for the batch's ENTIRE remaining duration, a real, substantial penalty Test 3 quantifies directly as head-of-line blocking.

Test 2 turns this into a single real utilization number -- useful steps actually computed divided by total real slot-steps spent, including padding -- and Test 4 confirms the general formulas collapse correctly to the trivial degenerate case: at a real batch size of exactly 1, static batching becomes strict FIFO with zero padding waste and perfect utilization, confirming the waste this section quantifies is a genuine consequence of grouping requests together, not an artifact of the simulator itself.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_static_batching_problem_and_head_of_line_blocking.cpp -o 01_static_batching_problem_and_head_of_line_blocking
./01_static_batching_problem_and_head_of_line_blocking
```

**Sample input:** a real 4-request static-batching schedule checked against an exact hand trace of start steps, completion steps, wasted padding steps, and queue-wait steps; the workload's own real slot utilization checked against an exact hand computation; a short request's own real head-of-line blocking delay checked directly against the completion it would have gotten served alone; and the degenerate real batch-size-1 case checked to collapse to strict FIFO with perfect utilization.

@@OUT1@@

!!! warning "[COMMON TRAP] assuming a short request's own real delay is bounded by how long it had to wait in a literal queue"
    It is tempting to think a request's own worst-case real delay under static batching is bounded by how many OTHER requests are ahead of it. Test 3 shows the real bound that actually matters is completely different: a request that arrives one single step after a batch has already started is not delayed by "one step's worth of queue" -- it is delayed by the ENTIRE remaining duration of whatever the longest-running member of that batch still needs, regardless of how short its own real job is. A 1-step request arriving moments too late waited a real 9 steps before its own batch even began, purely because static batching's own retirement rule has no way to let a new, urgent request cut in once a batch is already running -- which is exactly the real constraint continuous batching (Section 27.3) is built to remove.

## 27.2 The Prefill/Decode Conflict and Chunked Prefill

### Intuition

A serving step that only advances already-running sequences by one token each (decode) has a very different real cost profile from a step that processes an entirely new prompt in one shot (prefill), and naively mixing the two in a single step means the SLOWER of the two profiles imposes its own real cost on every sequence sharing that step -- including ones that have nothing to do with the new request at all.

### The Concept, In Detail

This section's own real cost model keeps the two profiles honest and separate: `cost_decode_step` scales with how many sequences are concurrently decoding (a real, memory-bound cost -- every active sequence's own KV cache must be read once per step), while `cost_prefill_chunk` scales with how many new prompt tokens are being processed in that same step (a real, compute-bound cost). Test 1 and Test 2 confirm the real conflict directly: injecting an entire 64-token prompt into a single step alongside 8 concurrently decoding sequences spikes that one step's own real cost by 2560 units over baseline -- latency injected into all 8 already-running sequences at once, not just the new request.

Test 3 introduces chunked prefill's own real fix: splitting that identical 64-token prompt into 4 chunks of at most 16 tokens each, injected across 4 separate steps, bounds the per-step injected latency to exactly 640 units -- one quarter of the unchunked spike -- while the TOTAL extra work performed across those 4 steps is conserved exactly, confirming chunking spreads real work out rather than creating or destroying any of it. Test 4 generalizes this into a real, provable bound: chunked prefill's own worst-case injected latency never exceeds `max_chunk_tokens`' own cost, regardless of the full prompt's real length, while the unchunked injected latency grows linearly and unbounded with prompt length -- checked directly across several genuinely different real prompt lengths, not merely the one hand-picked example.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_prefill_decode_conflict_and_chunked_prefill.cpp -o 02_prefill_decode_conflict_and_chunked_prefill
./02_prefill_decode_conflict_and_chunked_prefill
```

**Sample input:** a real baseline decode-step cost checked against an exact hand computation; an unchunked single-step prefill injection checked to spike that step's own real cost by an exact, computed amount; chunked prefill's own real per-step cost and total conserved work checked against an exact hand trace; and the chunked latency bound checked to hold, and the unchunked latency to exceed it, across several genuinely different real prompt lengths.

@@OUT2@@

!!! warning "[COMMON TRAP] assuming chunked prefill reduces the TOTAL real work a new request's prompt requires"
    Chunked prefill does not make a long prompt cheaper to process overall -- Test 3 confirms the total extra work across all 4 chunked steps conserves the unchunked total EXACTLY, to the unit. What chunking actually changes is when that real cost is paid: instead of one single step absorbing the entire spike, several smaller steps each absorb a bounded fraction of it, keeping every individual step's own real latency predictable for the sequences that happen to share it. A system that chunks prefill expecting a net real throughput gain from chunking alone will be disappointed -- the genuine benefit is bounded worst-case per-step latency, which is a real, different, and for a production serving system, usually far more valuable property than raw total throughput.

## 27.3 A Continuous-Batching Scheduler Built From Scratch

### Intuition

Static batching's own real flaw, quantified in Section 27.1, is structural: a slot cannot be reused until its ENTIRE batch retires, even if that slot's own request finished long ago. Continuous batching fixes this directly -- a slot frees the instant its own sequence finishes, and the very next scheduling step can admit a new request into it, with no need to wait for anything else in that batch.

### The Concept, In Detail

`simulate_continuous_batching` runs a real discrete-tick loop that, every tick, admits the earliest-arrived waiting request into any real free slot, decodes every active sequence by exactly one step, and immediately retires -- and frees the slot of -- any sequence whose own needed steps have just been satisfied. Test 1 traces this exactly on the IDENTICAL 4-request workload Section 27.1 used, and Test 2 runs Section 27.1's own static-batching simulator, restated in full within this file, on that same workload side by side: every one of the 4 requests completes strictly earlier under continuous batching, checked directly against a live computation rather than a hardcoded comparison number.

Test 3 confirms the real structural reason why: continuous batching's own total active-slot ticks across the whole schedule equals EXACTLY the sum of every request's own needed steps -- zero padding waste, ever, by construction -- while static batching's identical workload wasted a real, nonzero 10 slot-steps. Test 4 confirms this scheduler's own real correctness generalizes beyond that one example: on a second, genuinely different workload with only a single real slot, it correctly degenerates to strict FIFO, admitting each request the instant both its own arrival and slot availability align, with zero real queue wait whenever a request has already arrived by the time its slot frees.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_continuous_batching_scheduler_from_scratch.cpp -o 03_continuous_batching_scheduler_from_scratch
./03_continuous_batching_scheduler_from_scratch
```

**Sample input:** a real continuous-batching schedule for a 4-request, 2-slot workload checked against an exact hand trace of admission and completion ticks; that identical workload's own static-batching schedule, computed live by the restated Section 27.1 simulator, checked to complete every request strictly later; continuous batching's own total active-slot ticks checked to exactly conserve the sum of needed steps with zero padding waste; and a second, genuinely different single-slot workload checked to correctly degenerate to strict FIFO.

@@OUT3@@

!!! warning "[COMMON TRAP] assuming continuous batching's own real benefit requires more available slots than static batching gets"
    It is tempting to think continuous batching wins simply because it is somehow more generous with GPU capacity. Test 4 shows this is not the mechanism at all: with only a single real slot -- the least generous possible configuration -- continuous batching still behaves correctly and wastes zero real capacity, because it degenerates cleanly to strict FIFO. The real advantage continuous batching provides over static batching, demonstrated directly in Test 2 on an IDENTICAL 2-slot workload, comes entirely from WHEN a finished sequence's own slot becomes available for reuse -- immediately, rather than only once an entire batch retires -- not from having access to any more real hardware than static batching already had.

## 27.4 Numerical Debugging Tools: NaN-Propagation Tracing and Floating-Point Drift Detection

### Intuition

A serving system running Section 27.3's own scheduler continuously, across unpredictable real traffic, eventually hits the numerical instabilities Section 26.2 warned about -- but at serving scale, the real question is no longer just "did a NaN appear," it is "where did it actually START," and a second, quieter real risk appears alongside it: continuous batching's own traffic-dependent regrouping of sequences can change the order a real reduction is computed in from run to run, and floating-point addition is not associative.

### The Concept, In Detail

`first_nan_layer` walks a real sequence of per-layer activation snapshots and reports the FIRST layer at which any element is non-finite -- not merely one of the layers where it happens to still be visible. Test 1 builds a real, honest 5-layer trace in which corruption genuinely originates at layer 2 (a real division-by-zero in a broken normalization step) and then propagates forward automatically through ordinary arithmetic at layers 3 and 4; `first_nan_layer` correctly reports layer 2, the real point of origin. Test 2 makes the practical stakes concrete: a naive system that only checks the FINAL layer for non-finite values gives an IDENTICAL "yes, something is broken" verdict whether corruption started at layer 2 or at layer 3 in a genuinely different trace -- only the real tracer actually distinguishes them.

Test 3 and Test 4 build this section's own real floating-point drift detector: summing a huge float32 value first, then ten small ones, silently loses all 10 of them to rounding, while summing the identical 10 small values together FIRST and only then adding the huge value preserves most of their real contribution -- a genuine, measurable 8-unit drift between two equally valid reduction orders over the identical input, confirmed directly in real, compiled `float` arithmetic rather than assumed from theory. Test 4 compares both real orders against the true double-precision value and confirms a real, general, actionable finding: summing small values before combining them with a much larger one is measurably more accurate, a concrete engineering choice relevant to any real serving system whose own batching schedule changes which order a batched reduction actually runs in.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_nan_propagation_tracing_and_fp_drift_detection.cpp -o 04_nan_propagation_tracing_and_fp_drift_detection
./04_nan_propagation_tracing_and_fp_drift_detection
```

**Sample input:** a real 5-layer forward-pass trace, in which corruption genuinely originates at layer 2 and propagates forward through ordinary arithmetic, checked to have its real point of origin correctly identified rather than merely detected downstream; a naive final-layer-only check confirmed to give an identical verdict on two traces with genuinely different real origins; real float32 reduction-order drift confirmed directly in compiled arithmetic between two equally valid summation orders; and both real orders compared against the true double-precision value to confirm which is genuinely more accurate.

@@OUT4@@

!!! warning "[COMMON TRAP] treating floating-point drift between two valid reduction orders as a genuine correctness bug"
    Test 3's own two real reduction orders produce genuinely different results -- 100000000.0 versus 100000008.0 -- and it is tempting to treat this as a bug that needs fixing. Test 4 shows the more accurate real framing: NEITHER order is exactly correct against the true double-precision value, and the difference between them is an expected, unavoidable consequence of float32's own limited real precision combined with a specific real choice of reduction order, not a sign either individual result is wrong in isolation. The genuinely actionable real finding is not "eliminate the drift" -- that is not possible while still using float32 -- it is "prefer the reduction order that is measurably closer to the truth," and a serving system whose own continuous batching changes reduction order across runs should at minimum know how large that drift can get, which is exactly what this section's own drift detector quantifies directly.

## Chapter Summary

This chapter built the real scheduling layer a production inference engine actually needs to serve unpredictable, concurrent traffic efficiently. Section 27.1 quantified static batching's own real waste with an exact hand-traceable simulator: real padding waste from grouping requests of different lengths together, and a real, substantial head-of-line blocking delay for any short request unlucky enough to arrive just after a long batch has already started. Section 27.2 resolved the real conflict between prefill's compute-bound cost and decode's memory-bound cost with chunked prefill, deriving a real, provable bound on injected per-step latency that holds regardless of prompt length. Section 27.3 combined both lessons into a real, from-scratch continuous-batching scheduler, checked directly against Section 27.1's own restated static-batching simulator on the identical workload rather than against a hardcoded comparison number, and confirmed its own real zero-padding-waste guarantee generalizes to a second, different workload. Section 27.4 closed the chapter with two real numerical debugging tools this book's own increasingly complex serving pipeline actually needs: a NaN-propagation tracer that finds where corruption genuinely originated, and a floating-point drift detector that quantifies exactly how much a real reduction's result can change when continuous batching changes its own reduction order.

## Self-Check Questions

1. Section 27.1's Test 3 shows a 1-step request delayed by a real 11-step head-of-line blocking penalty. Explain concretely why this delay is NOT simply equal to however many steps the request had to wait in a literal first-come-first-served queue.
2. Section 27.1's Test 4 shows utilization is exactly 1.0 at batch_size 1. Explain what specifically about a batch containing only one request makes padding waste structurally impossible, regardless of how long or short that one request happens to be.
3. Section 27.2 models decode's own real cost as scaling with the number of concurrently active sequences, and prefill's own real cost as scaling with the number of new prompt tokens. Explain, in your own words, why these are genuinely different real bottlenecks (memory-bound versus compute-bound) rather than the same cost measured two different ways.
4. Section 27.2's Test 3 shows chunked prefill conserves the TOTAL real extra work exactly, while Test 4 shows it bounds the WORST per-step injected latency. Explain why a system could care about the second property even though the first shows chunking provides no reduction in the first.
5. Section 27.3's Test 2 checks continuous batching against static batching using a LIVE restated simulator rather than a hardcoded comparison number. Explain one concrete real risk a hardcoded comparison number would have introduced that the live simulator avoids.
6. Section 27.3's Test 4 uses a workload with only a single real slot to test the scheduler. Explain what specific real scheduling behavior this single-slot case is actually able to verify that a multi-slot workload alone would not clearly isolate.
7. Section 27.4's Test 2 shows a naive final-layer-only NaN check gives an identical verdict on two traces with genuinely different real points of origin. Explain concretely what real debugging information is lost by only checking the final layer.
8. Section 27.4's Test 1 constructs a NaN via `0.0 / 0.0` and separate `+inf` values via `nonzero / 0.0` in the SAME broken layer. Explain why `first_non_finite_index` still correctly reports the same layer as `first_nan_layer` despite these being two different kinds of non-finite values.
9. Section 27.4's Test 3 and Test 4 show summing small values before a much larger one is measurably more accurate than the reverse order. Connect this concretely to a real risk introduced by Section 27.3's own continuous-batching scheduler, which regroups sequences into different real batches depending on real, unpredictable traffic.
10. This chapter's title pairs "continuous batching" with "production serving architecture." Choose any ONE of this chapter's 4 sections and explain concretely why the real property it derives or builds would matter LESS, or not apply at all, to a system that only ever serves a single request at a time with no concurrent traffic.

## Where We Go Next

This chapter built the real scheduling and debugging layer a serving system needs on the CPU. The final chapter takes this book's own inference engine to the GPU: Chapter 28 builds a real Flash Attention implementation around the same online-softmax idea Section 26.2's own numerically stable softmax introduced, fixing the O(N^2) memory wall standard attention hits, and closes the book with a real CUDA production engine -- complete with its own kernel-validation suite -- for the edge devices, Jetson-class boards among them, that do carry a small GPU.

## Worked Solutions

**1.** A literal first-come-first-served queue delay would only count the time a request spends waiting BEHIND other requests that are also merely waiting. Section 27.1's own real penalty is different: once a batch has started, a newly arrived request cannot be admitted at all, no matter how few other requests are ahead of it in real arrival order, until every member of the ALREADY-RUNNING batch finishes -- so the delay is driven by how much longer the running batch's own longest member still needs, not by how many requests are queued.

**2.** A batch's own real makespan is defined as the maximum needed_steps among its members; with exactly one member, that maximum IS that member's own needed_steps, so the batch's own total slot-steps (batch_size times makespan, here 1 times needed_steps) always equals that same member's own needed_steps exactly -- there is no other, shorter member left in the batch whose slot could ever sit idle waiting for a longer one, since there is only ever one member to begin with.

**3.** Decode's own real cost is dominated by reading each active sequence's own KV cache from memory once per step -- more concurrent sequences means more real memory traffic, but the actual compute per token is comparatively small (a GEMV). Prefill's own real cost is dominated by the actual compute needed to process every token of a new prompt against the model's own weights at once (closer to a GEMM) -- more prompt tokens means more real FLOPs, while the memory traffic for reading those (shared) weights barely changes. These are genuinely different physical bottlenecks -- one is bound by how fast memory can be read, the other by how fast the processor can compute -- not the same underlying cost expressed in two different units.

**4.** Chunking conserving the total real work means a system gains nothing in raw aggregate throughput from chunking alone -- the same total amount of prefill compute still has to happen somewhere. What a system gains instead is PREDICTABILITY: bounding the worst single-step latency spike protects every already-running decode sequence sharing that step from an unbounded, traffic-dependent latency spike, which matters directly for real-time serving guarantees (a maximum acceptable per-token latency) even when it does not improve the system's own total steady-state throughput number.

**5.** A hardcoded comparison number would only ever be correct for the ONE specific workload it was computed for by hand; if that workload's own numbers were ever changed even slightly (a different arrival time, a different needed_steps value), the hardcoded number would silently become stale and wrong, while a passing test would keep reporting success. Running Section 27.1's own restated simulator live means the comparison is recomputed correctly every single time the workload changes, catching a real discrepancy immediately rather than only when someone remembers to update a hardcoded expectation by hand.

**6.** A multi-slot workload could pass its own tests purely because ONE of its several slots happened to schedule correctly, while a bug in the admission logic that only manifests when there is exactly one slot to reason about (an off-by-one in how "the" free slot, rather than "a" free slot among several, gets selected) could go completely undetected. The single-slot workload isolates admission-and-retirement correctness with no other slots' own scheduling behavior to potentially mask a real bug in that specific code path.

**7.** Checking only the final layer answers a single yes-or-no question -- "is anything broken by the time the forward pass finishes" -- and provides no information about which of potentially many upstream layers actually introduced the problem. An engineer debugging a real production failure needs to know WHERE to look in the model's own code (which layer's own computation is actually broken), and a final-layer-only check gives them nothing beyond confirming that a problem exists somewhere in the entire pipeline, which they very likely already knew from a garbled model output.

**8.** `first_non_finite_index` simply scans a single layer's own vector for the FIRST position where either `std::isnan` or `std::isinf` returns true, treating both as equally "non-finite" -- it does not need to distinguish which of the two kinds of non-finite value it finds, since both are equally real evidence that something already went wrong in that layer's own computation. Because index 0 in this section's own broken layer holds a genuine NaN and index 0 is scanned first, `first_non_finite_index` reports index 0 regardless of the fact that indices 1 and 2 in that same layer hold a different, real kind of non-finite value (+inf) rather than NaN.

**9.** Section 27.3's own scheduler admits and retires sequences based on real, unpredictable arrival times, which means the SET of sequences batched together at a given step -- and therefore the order in which a batched reduction (say, summing values across that batch dimension) actually gets computed -- can differ from one real run to the next, even for logically identical requests, purely because traffic happened to arrive in a different order or at different times. Section 27.4's own Test 3 and Test 4 show this kind of reduction-order change is not merely a cosmetic difference: it can produce a real, measurably different numerical result, so a system relying on continuous batching should not assume repeated runs of the same logical workload will produce bit-identical numerical output.

**10.** Section 27.1's own entire real contribution -- quantifying wasted padding and head-of-line blocking -- assumes MULTIPLE real requests are competing for the same limited batch slots at overlapping times. A system serving exactly one request at a time, with no concurrent traffic ever, has no other request to be blocked behind and no padding to waste, since there is only ever the one real occupant of the one slot it needs; static batching's own entire failure mode, and therefore the reason continuous batching improves on it, simply does not arise when there is no real concurrency for a scheduler to have to arbitrate in the first place.
