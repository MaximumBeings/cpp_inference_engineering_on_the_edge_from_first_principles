# Appendix D: Profiling and Benchmarking on Edge Hardware

Chapter 32's own opening line makes a promise this appendix now keeps: "this book's own running discipline of never claiming a wall-clock timing result as part of a locked, deterministic self-test contract" is "stated plainly in Appendix D." Every chapter since Chapter 4 has followed that discipline without stopping to defend it at length -- byte counts and FLOP counts get locked and verified across four architectures; milliseconds and GB/s never do. This appendix states the reasoning explicitly, gives it a real formula for computing a theoretical ceiling, a real (deliberately unlocked) way to measure what a machine actually achieves, a real way to turn either into a latency budget, and an honest account of what happened when this book tried to reach for the most obvious real profiling tool -- `perf` -- on its own two real build environments.

---

## D.1 Computing Theoretical Peak Bandwidth From a Real Spec Sheet

### Intuition

A vendor's own published bandwidth figure is not measured on a running system -- it is computed, ahead of time, from three numbers on a spec sheet: how many independent memory channels a chip has, how fast each one transfers data, and how wide each transfer is. Chapter 8.1 derived exactly this formula for an illustrative machine. This section applies it, for the first time in this book, to a real, named, currently-shipping piece of edge hardware, using only numbers that chip's own vendor has actually published.

### The Concept, In Detail

Chapter 8.1's formula is a single multiplication:

```
peak_bandwidth = channels * transfer_rate * bytes_per_transfer
```

Raspberry Pi's own official documentation for the Raspberry Pi 5's BCM2712 SoC states the memory as "LPDDR4X-4267" over "a 32-bit LPDDR4X memory interface" that "provides up to 17 GB/s of memory bandwidth."[^rpi5mem] That single sentence supplies every input the formula needs: one channel, a transfer rate of 4267 MT/s, and a 32-bit (4-byte) bus. Multiplying them out gives 17.068 GB/s -- which rounds down to exactly the "17 GB/s" the vendor's own documentation states as its headline figure.

Two other real platforms are worth having for comparison, though neither vendor publishes the same channel/rate/width breakdown, so this book does not re-derive their figures the way it just did for the Raspberry Pi 5 -- it only cites them:

| Platform | Real, published memory bandwidth | Source |
|---|---|---|
| Raspberry Pi 5 (BCM2712) | 17 GB/s (32-bit LPDDR4X-4267) | Raspberry Pi's own official documentation[^rpi5mem] |
| NVIDIA Jetson Orin Nano (8 GB module) | 68 GB/s (128-bit LPDDR5) | NVIDIA's own official datasheet[^orinmem] |
| Apple M4 / M4 Pro / M4 Max | 120 / 273 / up to 546 GB/s (unified memory) | Apple's own official announcement[^m4mem] |

The nearly 32x spread between the cheapest board in this table and the most capable chip is exactly the real-world range this book's own roofline model (Chapter 8) and crossover formula (Chapter 30.1) were built to reason about generally, without needing to be re-derived for every new chip: the same formulas apply, only the inputs change.

```bash
g++ -std=c++23 -Wall -Wextra -O2 d1_peak_bandwidth_formula.cpp -o d1_peak_bandwidth_formula
./d1_peak_bandwidth_formula
```

@@CODE1@@

@@OUT1@@

---

## D.2 A Real Bandwidth Micro-Benchmark, and Why Its Own Timing Can Never Be Locked

### Intuition

A theoretical peak from a spec sheet and what a real, running loop actually achieves are two different numbers, and only a real, timed measurement can produce the second one. But a real, timed measurement is a measurement of a specific machine at a specific moment -- not a portable fact about an algorithm -- and this book's own self-test contract has never locked one, on purpose, since Chapter 4.4's own "trusting a millisecond figure that depends on the benchmark machine" trap.

### The Concept, In Detail

The classic way to measure achieved memory bandwidth is the STREAM benchmark's own "triad" kernel: `c[i] = a[i] + scalar * b[i]`, three real memory accesses per element (two reads, one write). The number of elements processed and the number of bytes that loop moves to process them are exactly, deterministically knowable from the loop's own definition -- true on every architecture, every run, forever. This book locks and verifies exactly that part below.

What the loop actually ACHIEVES, in GB/s, requires a clock, and a clock's answer depends on the machine, its current load, its thermal state, and -- this appendix's own new finding -- how virtualized the environment running it is. Wrapping the identical loop in `std::chrono` and running the identical binary three times back to back, on two different real machines this book has been building and verifying on all along, shows exactly why that number can never be part of a locked, cross-architecture-identical contract:

```
auto t0 = std::chrono::steady_clock::now();
for (size_t i = 0; i < n; ++i) c[i] = a[i] + scalar * b[i];
auto t1 = std::chrono::steady_clock::now();
double gbps = (bytes_moved / 1e9) / std::chrono::duration<double>(t1 - t0).count();
```

@@DIAG3@@

@@DIAG4@@

The cloud build container's own three runs vary by about 2%; the real aarch64 device's own three runs -- the same binary, the same machine, seconds apart -- vary by 68%, almost certainly reflecting that this book's own real device is itself a Linux virtual machine layered on the host's own hypervisor, on top of ordinary operating-system scheduling noise. Neither number is wrong. Both are simply properties of a specific run on a specific machine at a specific moment, which is exactly why this book's own locked self-test, below, never prints one.

```bash
g++ -std=c++23 -Wall -Wextra -O2 d2_bandwidth_microbenchmark.cpp -o d2_bandwidth_microbenchmark
./d2_bandwidth_microbenchmark
```

@@CODE2@@

@@OUT2@@

!!! warning "[COMMON TRAP] trusting a single timed run as though it were the loop's own property"
    A single achieved-bandwidth number, from a single run, on a single machine, tells a reader almost nothing portable -- not because the measurement is wrong, but because the very same binary on the very same machine can legitimately disagree with itself by double digits of percent from one run to the next, as diag4 above shows directly. A real benchmark needs many runs, a stated distribution (not just a mean), and an explicit machine description -- and even then, it describes that measurement, not the algorithm.

---

## D.3 Latency-Budget Arithmetic

### Intuition

Every deployment eventually asks the same question in a different costume: given a hard latency target, what does the rest of the system need to look like to hit it? Chapter 27.2 already answered a version of this question for a trading system's own tick-to-trade deadline. Chapter 4.1 already answered a version of it for tokens-per-second as a function of memory bandwidth. Neither chapter needed a clock to answer it -- both worked entirely from stated costs and a formula. This section reuses both, unchanged, and for the first time combines one of them with a real, cited hardware number instead of a stated illustrative one.

### The Concept, In Detail

Chapter 4.1's own memory-wall formula is a single division:

```
tokens_per_sec = bandwidth / bytes_read_per_token
```

Chapter 4.1's own worked table reported 14.2 tokens/sec for a roughly 4.5 GB Q4_0 Llama-3-8B-shaped model at a stated illustrative 64 GB/s. Running the identical formula against Appendix D.1's own real, verified Raspberry Pi 5 figure (17.068 GB/s) instead of that stated illustrative number, against the same approximate model size, gives a new, real, concrete estimate: roughly 3.8 tokens/sec -- the first time this book has connected one of its own model-size figures to a real, named, currently-purchasable board's own real bandwidth spec.

Chapter 27.2's own deadline-enforcement pattern generalizes cleanly beyond a trading system: accumulate named stage costs in order, and refuse the instant the running total strictly exceeds a stated budget, naming the exact stage responsible rather than only reporting a final over-budget total.

| Question | Real formula, reused from | This section's own new result |
|---|---|---|
| Tokens/sec at a stated illustrative bandwidth | Chapter 4.1 | Reproduces the book's own 14.2 tok/s figure exactly |
| Tokens/sec at a real, cited hardware bandwidth | Chapter 4.1 + Appendix D.1 | ~3.8 tok/s on a real Raspberry Pi 5 |
| Where a stage sequence breaches a budget | Chapter 27.2 | Names the exact stage, not just the total |

```bash
g++ -std=c++23 -Wall -Wextra -O2 d3_latency_budget_calculator.cpp -o d3_latency_budget_calculator
./d3_latency_budget_calculator
```

@@CODE3@@

@@OUT3@@

---

## D.4 Profiling With `perf`: The Flags That Actually Work, and Where They Genuinely Don't

### Intuition

`perf` is the standard real tool for measuring what a running program actually does on real hardware: cycles, cache misses, branch mispredictions, wall-clock time broken down by call stack. On Arm specifically, the honest starting point is that generic event names (`cycles`, `instructions`, `cache-misses`, `branch-misses`) are the ones most likely to work across vendors, because they are resolved through the kernel's own generic PMU abstraction rather than a vendor-specific raw event encoding; genuinely vendor-specific counters need either a raw hex event code or a PMU-specific alias (`armv8_pmuv3/...`) that differs across implementations. None of that is this appendix's real finding, though -- what actually happened when this book tried to use `perf` on its own two real build environments is.

### The Concept, In Detail

The real, correct starting command line for a quick check on any Linux machine, Arm included, is:

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,task-clock -- ./your_program
```

and, for a call-graph profile (needs the binary built with `-fno-omit-frame-pointer`, or DWARF unwind info, to produce a real stack rather than a garbled one):

```bash
perf record -g -- ./your_program
perf report
```

Running the first of those two commands against this book's own binaries, on this book's own two real build environments, produced two different real, honest failures rather than a clean measurement on either one:

@@DIAG1@@

@@DIAG2@@

The cloud build container's own copy of `perf` genuinely runs -- once its own distro packaging's kernel-version check is bypassed by invoking the real installed binary directly -- but reports every real hardware PMU event as `<not supported>`, because this virtualized build environment does not expose real performance-counter hardware to its guest at all; only software events (a timer-based `task-clock`, not a real counter) work. This book's own real target device -- the aarch64 Linux VM this book has cross-checked every single chapter against -- runs a genuine, correctly versioned `perf` binary, but its own `perf_event_paranoid` sysctl is set to 4, a value beyond the four levels the kernel's own error message documents, blocking every event type outright, with no root access available to lower it.

Neither failure is a bug in `perf`, and neither is unique to this book's own environments: cloud VMs routinely withhold PMU passthrough from guests for isolation reasons, and hardened containers routinely set `perf_event_paranoid` at or above the maximum the kernel formally documents specifically to close it off entirely. A real edge deployment -- inside a container orchestrator, behind a hypervisor, on a locked-down consumer device -- can hit either restriction, or both, in production, not merely in this book's own build environment.

| Restriction | Where this book hit it | Real cause | Any real workaround here |
|---|---|---|---|
| Hardware PMU events report `<not supported>` | Cloud build container | No virtualized PMU passthrough to the guest | None -- move the measurement to unvirtualized hardware |
| All events denied outright | Real aarch64 device | `perf_event_paranoid=4`, no root | None -- requires a privilege this environment does not grant |

!!! warning "[COMMON TRAP] assuming a real profiling tool is always reachable on a real deployment target"
    `perf` existing on a system, and even reporting a real version number, does not mean it can measure anything on that system -- both of this book's own real environments proved genuinely, differently unable to produce a single real hardware counter, for reasons entirely outside this book's own code. This is precisely why every deterministic claim in this book -- FLOP counts, byte counts, operation counts -- was instrumented directly into the code itself, from Chapter 8 onward, rather than left to depend on an external profiler being reachable at all: a number the code itself can report needs no permission this environment might withhold.

---

## Appendix Summary

Nothing in this appendix changed how this book measures anything -- it stated, once and plainly as Chapter 32 promised, why deterministic operation and byte counts get locked into this book's own self-test contract while wall-clock timing never does, then backed that statement with a real vendor-verified bandwidth formula, a real triad benchmark whose own timing swung by 68% across three back-to-back runs on the very same machine, a real latency-budget calculator connecting a stated model size to a real, named board's real bandwidth for the first time, and a real, honestly-reported account of `perf` failing in two different, real, unrelated ways on this book's own two build environments. The throughline across all four sections is the same one this book has followed since Chapter 8: a number worth trusting is either provably a property of the algorithm, independent of any machine, or it is honestly labeled as belonging to one specific run, on one specific machine, at one specific moment -- and the two are never allowed to be confused for each other.

## Where We Go Next

Appendix E turns from measuring this book's own C++ engine to translating it: `torch.Tensor` and `AutoModel.from_pretrained` next to this book's own `mdspan` tensor and GGUF loader, vLLM's continuous batching next to this book's own scheduler, HuggingFace's tokenizer next to the from-scratch BPE pipeline -- a Rosetta Stone for a reader arriving fluent in the Python inference ecosystem and looking for exactly where each familiar piece landed in this book's own real, from-scratch C++23 engine.

[^rpi5mem]: Raspberry Pi Ltd, "Processors" documentation, BCM2712 memory interface: <https://www.raspberrypi.com/documentation/computers/processors.html>
[^orinmem]: NVIDIA, Jetson Orin Nano Developer Kit datasheet (8 GB module, 128-bit LPDDR5, 68 GB/s): <https://files.seeedstudio.com/wiki/Jetson-Orin-Nano-DevKit/jetson-orin-nano-developer-kit-datasheet.pdf>
[^m4mem]: Apple Inc., "Apple introduces M4 Pro and M4 Max," October 2024: <https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/>
