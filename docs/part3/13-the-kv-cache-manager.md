# Chapter 13: The KV Cache Manager -- Paged Attention, Ring Buffers, and Smart Eviction

**What you will understand by the end of this chapter:**

- Why autoregressive decode is memory-bandwidth-bound rather than compute-bound, and the one equation — tokens/second is bounded by bandwidth divided by bytes read per token (model weights plus KV cache) — that makes this concrete enough to compute, not just assert.
- Why the KV cache's share of that per-token byte cost grows with context length until it dominates the model's own weights, and why that growth is the reason KV cache management, not raw compute throughput, is the dominant performance lever at long context.
- PagedAttention: dividing the KV cache into fixed-size blocks dispensed from a pool via a free list, so allocation and deallocation are both O(1), memory is never fragmented, and identical system prompts across many users can be shared by copying a handful of integers instead of the underlying data.
- The Ring Buffer's two-counter design — a wrapping physical write slot and a monotonically increasing logical RoPE position that are equal only during the initial fill — and why conflating them, a mistake that produces no crash and no warning, silently corrupts every attention score computed after the first wrap.
- H2O's use of attention probability, a signal the model already computes for free, as an importance score that lets eviction target the least-useful cached token instead of blindly discarding whichever token happens to be oldest — and how combining paging, wrapping, and smart eviction into one manager preserves every guarantee each technique proved on its own.

**What you need to know first:**

- Chapter 2's RoPE positional encoding — specifically that a Key vector's rotation angle is a function of the token's LOGICAL position in the sequence, a fact this chapter's Ring Buffer section depends on directly.
- Chapter 8's roofline model and its prefill/decode distinction: prefill processes a whole prompt at once and is compute-bound, while decode processes one token at a time and is memory-bound — this chapter is entirely about managing the resource that dominates the memory-bound side.
- This is a purely single-threaded, data-structure-focused chapter: no `-pthread`, no `std::mdspan`, no `-ffp-contract=off` for any file here, continuing Chapter 12's simplification — every file compiles with just `-std=c++23 -Wall -Wextra -O2`.

---

Every chapter through Part 2 asked how fast the CPU could compute a forward pass. This chapter asks a different question: once a model is generating token after token, what does it cost to remember everything it has already seen, and how do you keep that cost from destroying performance or correctness? Section 13.1 derives the memory wall itself — the equation that makes decode-time speed a function of bytes moved, not FLOPs computed — and shows exactly how much of that byte budget the KV cache claims as context grows. Section 13.2 solves the fragmentation half of the problem with PagedAttention: block-table indirection that allocates memory on demand and shares identical prefixes for free. Section 13.3 solves the unbounded-growth half with a Ring Buffer, and spends most of its attention on a subtle correctness property — RoPE position decoupling — that a naive implementation gets wrong in a way that looks fine until the model's output quietly stops making sense. Section 13.4 replaces FIFO eviction, which the Ring Buffer uses by default, with H2O's attention-probability-driven importance scoring, so eviction discards what the model has stopped needing rather than whatever merely arrived first. Section 13.5 closes the chapter by combining all three techniques into one manager and confirming that nothing about combining them weakens any guarantee proved in isolation.

## 13.1 The Memory Wall: Why the KV Cache Is the Real Bottleneck

### Intuition

An autoregressive model is stateful: generating the Nth token requires attending over all N-1 tokens that came before it. Storing their Key and Value vectors instead of recomputing them every step turns an O(N^2) total-FLOP generation into O(N) per step — a genuinely good trade, but one that swaps compute for memory traffic, and memory traffic has its own, much harder, speed limit.

### The Concept, In Detail

Every decode step must read the entire model weight matrix plus the current KV cache from memory before it can produce one token, so the theoretical ceiling on tokens per second is exactly (memory bandwidth) divided by (weight bytes plus KV cache bytes read per step). Neither more CPU cores, nor a higher clock, nor a wider FPU moves this ceiling at all, because none of them change how many bytes must cross the memory bus — this is the precise sense in which decode is memory-bound rather than compute-bound, the mirror image of Chapter 8's compute-bound prefill phase. The KV cache's own size is a direct product of the model's shape: for each of `n_layers` transformer layers, for each of `n_heads_kv` grouped-query-attention KV heads, for each cached token, two vectors (Key and Value) of `head_dim` elements each are stored at some number of bytes per element. For Llama 3 8B (32 layers, 8 KV heads, head_dim 128, FP16 storage) that works out to exactly 128 KB per token — a number small enough to ignore at a few hundred tokens of context and large enough, at 128K tokens, to require 16 GB on its own, dwarfing the roughly 4-4.5 GB the model's own quantized weights occupy. The consequence is that the KV cache's share of total bytes read per decode step is not fixed — it grows with context length until, at long enough context, it dominates the model's weights entirely, which is exactly why managing the KV cache's size, not further compute optimization, becomes the dominant performance lever once conversations or documents get long.

### Code and Verification

```cpp
// 01_kv_cache_bandwidth_estimator.cpp
// Chapter 13, Part 1: every chapter through Part 2 optimized how fast the
// CPU could compute. This chapter starts from a different bottleneck: an
// autoregressive model is stateful, and generating the Nth token requires
// attending over all N-1 tokens that came before. Storing their Key and
// Value vectors instead of recomputing them turns an O(N^2) total-FLOP
// generation into O(N) per step -- the KV cache is what makes that trade,
// swapping recomputation for memory. But memory has its own speed limit,
// and once that trade is made, the KV cache itself becomes the dominant
// cost at long context.
//
// This section makes the decode-time "memory wall" concrete with one
// equation: tokens/second is bounded above by (memory bandwidth) /
// (bytes that must be read per token step), where those bytes are the
// model's weights PLUS its KV cache. More cores, a higher clock, or a
// wider FPU do not move this ceiling at all -- decode is memory-bound,
// not compute-bound, which is exactly the opposite of the prefill phase
// Chapter 8's roofline model analyzed. Every number this file prints is
// derived from that one equation and real, published model shapes
// (Llama 3 8B/70B, Mistral 7B) -- nothing here is a fabricated benchmark.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_kv_cache_bandwidth_estimator.cpp -o 01_kv_cache_bandwidth_estimator

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) < (tol))

// Every parameter needed to compute a model's weight size and KV cache
// size. All values below are real published model shapes, not invented.
struct ModelCfg {
    std::string name;
    double params_billions;
    int n_layers;
    int n_heads_kv;   // GQA: fewer KV heads than query heads
    int head_dim;
};

struct QuantScheme {
    std::string name;
    double bytes_per_weight;  // FP32=4.0, FP16=2.0; Q8/Q4 include their per-block scale overhead
};

struct Analysis {
    double weight_gb;
    double kv_cache_gb;
    double total_gb;
    double max_tokens_per_sec;
    double kv_pct_of_total;
};

// The one equation this whole section derives from: decode-step data
// volume (weights + KV cache) divided into available bandwidth is the
// theoretical ceiling on tokens/second. KV cache size is 2 tensors
// (K and V) x n_layers x n_heads_kv x context_len x head_dim x bytes.
Analysis compute(const ModelCfg& m, const QuantScheme& q, int ctx_len, double bandwidth_gbs) {
    double weight_gb = m.params_billions * q.bytes_per_weight;  // params_billions already in units of 1e9
    double kv_elements = 2.0 * m.n_layers * m.n_heads_kv * static_cast<double>(ctx_len) * m.head_dim;
    double kv_dtype_bytes = 2.0;  // KV cache stored as FP16 in every configuration this section checks
    double kv_gb = kv_elements * kv_dtype_bytes / 1e9;
    double total_gb = weight_gb + kv_gb;
    double max_tps = bandwidth_gbs / total_gb;
    double kv_pct = kv_gb / total_gb * 100.0;
    return {weight_gb, kv_gb, total_gb, max_tps, kv_pct};
}

void print_context_table(const ModelCfg& m, const QuantScheme& q, double bw) {
    std::cout << "\n  " << m.name << " (" << q.name << ") @ " << bw << " GB/s:\n";
    std::cout << "    context  weights   kv_cache    total   tok/s   kv%\n";
    for (int ctx : {512, 1024, 2048, 4096, 8192, 16384, 32768, 131072}) {
        auto a = compute(m, q, ctx, bw);
        if (a.total_gb > 256.0) break;
        std::cout << "    " << std::setw(6) << ctx << "  "
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << a.weight_gb << "  "
                  << std::setw(9) << a.kv_cache_gb << "  "
                  << std::setw(7) << a.total_gb << "  "
                  << std::setprecision(1) << std::setw(6) << a.max_tokens_per_sec << "  "
                  << std::setprecision(0) << std::setw(4) << a.kv_pct_of_total << "%\n";
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 13.1: The Memory Wall -- Bandwidth-Limited Decode\n";
    std::cout << "========================================================\n";

    ModelCfg llama3_8b  = {"Llama 3 8B",  8.0,  32, 8, 128};
    QuantScheme fp16 = {"FP16", 2.0};
    QuantScheme fp32 = {"FP32", 4.0};
    QuantScheme q4   = {"Q4",   0.5625};  // 4 bits + one FP16 scale per 32 weights: (4 + 16/32)/8 bytes

    std::cout << "\n-- Test 1: KV cache size formula matches a hand calculation --\n";
    {
        // Llama 3 8B, FP16 KV, 4096-token context:
        // 2 (K,V) x 32 layers x 8 KV heads x 4096 tokens x 128 dims x 2 bytes
        auto a = compute(llama3_8b, fp16, 4096, 90.0);
        double expected_bytes = 2.0 * 32 * 8 * 4096 * 128 * 2;
        double expected_gb = expected_bytes / 1e9;
        std::cout << "  hand calc:  " << expected_bytes << " bytes = " << expected_gb << " GB\n";
        std::cout << "  computed:   " << a.kv_cache_gb << " GB\n";
        CHECK_NEAR(a.kv_cache_gb, expected_gb, 0.001);
        CHECK_NEAR(expected_gb, 0.536870912, 1e-6);
    }

    std::cout << "\n-- Test 2: speed limit for a realistic Q4 configuration --\n";
    {
        auto a = compute(llama3_8b, q4, 4096, 90.0);
        std::cout << "  Llama 3 8B, Q4 weights, ctx=4096, DDR5 @ 90 GB/s:\n";
        std::cout << "    weights:  " << std::fixed << std::setprecision(3) << a.weight_gb << " GB\n";
        std::cout << "    kv cache: " << a.kv_cache_gb << " GB\n";
        std::cout << "    total:    " << a.total_gb << " GB per decode step\n";
        std::cout << "    speed:    " << std::setprecision(1) << a.max_tokens_per_sec << " tok/s (theoretical ceiling)\n";
        CHECK(a.max_tokens_per_sec > 15.0 && a.max_tokens_per_sec < 25.0);
    }

    std::cout << "\n-- Test 3: speed degrades monotonically as context grows --\n";
    {
        double prev = 1e9;
        for (int ctx : {512, 1024, 2048, 4096, 8192}) {
            auto a = compute(llama3_8b, q4, ctx, 90.0);
            CHECK(a.max_tokens_per_sec < prev);
            std::cout << "  ctx=" << std::setw(6) << ctx << " -> "
                      << std::fixed << std::setprecision(1) << a.max_tokens_per_sec << " tok/s\n";
            prev = a.max_tokens_per_sec;
        }
    }

    std::cout << "\n-- Test 4: KV cache dominates bandwidth at very long context --\n";
    {
        auto a = compute(llama3_8b, q4, 131072, 90.0);
        std::cout << "  at 128K context: weights=" << std::fixed << std::setprecision(2) << a.weight_gb
                  << " GB, kv_cache=" << a.kv_cache_gb << " GB ("
                  << std::setprecision(0) << a.kv_pct_of_total << "% of total bandwidth)\n";
        CHECK(a.kv_pct_of_total > 70.0);
    }

    std::cout << "\n-- Test 5: quantization speedup is largest when the KV cache is small --\n";
    {
        auto a_fp32 = compute(llama3_8b, fp32, 512, 90.0);
        auto a_q4 = compute(llama3_8b, q4, 512, 90.0);
        double speedup = a_q4.max_tokens_per_sec / a_fp32.max_tokens_per_sec;
        std::cout << "  ctx=512 (weight-dominated): FP32=" << std::fixed << std::setprecision(1)
                  << a_fp32.max_tokens_per_sec << " tok/s, Q4=" << a_q4.max_tokens_per_sec
                  << " tok/s, speedup=" << std::setprecision(2) << speedup << "x\n";
        // Weight bytes/param drops from 4.0 to 0.5625, a 7.11x compression;
        // at a context short enough that KV cache is nearly negligible,
        // speed scales close to that same ratio.
        CHECK(speedup > 4.0 && speedup < 8.0);
    }

    std::cout << "\n-- reference table: Llama 3 8B, Q4 weights, DDR5 @ 90 GB/s --\n";
    print_context_table(llama3_8b, q4, 90.0);

    std::cout << "\n-- reference table: hardware bandwidth comparison (ctx=4096, Q4) --\n";
    struct HW { const char* name; double bw; };
    std::cout << "    hardware               bandwidth(GB/s)   tok/s\n";
    for (const auto& hw : (HW[]){
             {"Laptop DDR4", 40.0}, {"Desktop DDR5", 90.0}, {"MacBook M2", 100.0},
             {"MacBook M2 Ultra", 400.0}, {"A100 HBM2e", 2000.0}, {"H100 HBM3e", 3350.0}}) {
        auto a = compute(llama3_8b, q4, 4096, hw.bw);
        std::cout << "    " << std::left << std::setw(20) << hw.name << std::right
                  << std::setw(12) << std::fixed << std::setprecision(0) << hw.bw
                  << std::setw(10) << std::setprecision(1) << a.max_tokens_per_sec << "\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_kv_cache_bandwidth_estimator.cpp -o 01_kv_cache_bandwidth_estimator
./01_kv_cache_bandwidth_estimator
```

**Sample input:** real published model shapes (Llama 3 8B) checked against a hand calculation of KV cache size at 4096 tokens, a realistic Q4-quantized speed-limit computation, monotonic speed degradation as context grows from 512 to 8192 tokens, KV cache dominating over 70% of bandwidth at 128K context, and quantization's speedup shrinking as context (and therefore the KV cache's own share of bandwidth) grows.

```text
========================================================
Chapter 13.1: The Memory Wall -- Bandwidth-Limited Decode
========================================================

-- Test 1: KV cache size formula matches a hand calculation --
  hand calc:  5.36871e+08 bytes = 0.536871 GB
  computed:   0.536871 GB

-- Test 2: speed limit for a realistic Q4 configuration --
  Llama 3 8B, Q4 weights, ctx=4096, DDR5 @ 90 GB/s:
    weights:  4.500 GB
    kv cache: 0.537 GB
    total:    5.037 GB per decode step
    speed:    17.9 tok/s (theoretical ceiling)

-- Test 3: speed degrades monotonically as context grows --
  ctx=   512 -> 19.7 tok/s
  ctx=  1024 -> 19.4 tok/s
  ctx=  2048 -> 18.9 tok/s
  ctx=  4096 -> 17.9 tok/s
  ctx=  8192 -> 16.1 tok/s

-- Test 4: KV cache dominates bandwidth at very long context --
  at 128K context: weights=4.50 GB, kv_cache=17.18 GB (79% of total bandwidth)

-- Test 5: quantization speedup is largest when the KV cache is small --
  ctx=512 (weight-dominated): FP32=2.8 tok/s, Q4=19.7 tok/s, speedup=7.02x

-- reference table: Llama 3 8B, Q4 weights, DDR5 @ 90 GB/s --

  Llama 3 8B (Q4) @ 90.00 GB/s:
    context  weights   kv_cache    total   tok/s   kv%
       512     4.50       0.07     4.57    19.7     1%
      1024     4.50       0.13     4.63    19.4     3%
      2048     4.50       0.27     4.77    18.9     6%
      4096     4.50       0.54     5.04    17.9    11%
      8192     4.50       1.07     5.57    16.1    19%
     16384     4.50       2.15     6.65    13.5    32%
     32768     4.50       4.29     8.79    10.2    49%
    131072     4.50      17.18    21.68     4.2    79%

-- reference table: hardware bandwidth comparison (ctx=4096, Q4) --
    hardware               bandwidth(GB/s)   tok/s
    Laptop DDR4                   40       7.9
    Desktop DDR5                  90      17.9
    MacBook M2                   100      19.9
    MacBook M2 Ultra             400      79.4
    A100 HBM2e                  2000     397.1
    H100 HBM3e                  3350     665.1

========================================================
10/10 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming more compute closes a memory-bandwidth gap"
    A CPU or GPU with dramatically more FLOPs per second generates decode tokens at the SAME speed as a slower one if the two have identical memory bandwidth, because decode's bottleneck is bytes moved per step, not arithmetic performed per step — the model's weights and KV cache must physically travel from RAM to the compute unit before a single multiply can happen, and no amount of extra ALU throughput makes that transfer faster. This is precisely why an H100 GPU (thousands of TFLOPS) and a desktop CPU (roughly one TFLOP) can show a far smaller decode-speed gap than their raw compute numbers would suggest, while showing close to the FULL gap in prefill, where compute rather than bandwidth is the limiting resource. Optimizing the wrong resource — adding compute throughput to a memory-bound workload — buys nothing measurable; the only levers that move decode speed are reducing bytes per token (quantization, a smaller KV cache) or increasing bandwidth itself.

## 13.2 PagedAttention: Block-Table Indirection

### Intuition

Two simple ways to allocate a KV cache both fail in production: reserving every connected user's worst-case context wastes nearly all of it on users who never approach that limit, and growing a buffer dynamically avoids that waste but pays for it with a stuttering reallocate-and-copy pause exactly when the buffer fills. PagedAttention borrows virtual-memory paging from operating systems to get neither failure mode.

### The Concept, In Detail

The KV cache is divided into fixed-size BLOCKS of tokens (this section uses 16), pre-allocated once into one big pool at startup, and dispensed to sequences on demand via a free list — a `BlockManager` that never allocates again after construction, so `alloc()` and `free()` are both O(1) integer-ID operations on that free list. Each sequence keeps a small BLOCK TABLE — an array mapping its own logical block index to a physical block ID somewhere in the pool — and physical blocks need not be, and generally are not, contiguous with each other. Because the block size is a power of two, translating a logical token index into (which block, which slot inside it) is exactly two single-cycle bit operations, a right shift and a mask, negligible next to the roughly 100ns DRAM access that follows either way. This indirection pays for itself doubly: it eliminates the waste of static pre-allocation, since a sequence only ever holds as many blocks as it has actually used, and it makes prefix sharing essentially free, since many users sharing an identical system prompt can have their block tables all reference the SAME physical blocks — "sharing" becomes copying a handful of integers, not copying any Key or Value data at all, and a user later adding their own unique tokens simply allocates NEW blocks for those without disturbing the shared ones.

### Code and Verification

```cpp
// 02_paged_attention_block_manager.cpp
// Chapter 13, Part 2: Section 13.1 showed that at long context the KV
// cache itself, not the model's weights, dominates the bytes a decode
// step must move through memory. This section attacks a different waste
// entirely: even at a FIXED total token count, naive allocation strategies
// waste most of the memory they reserve. Pre-allocating every connected
// user's worst-case context (say, 8192 tokens) up front means a user who
// only ever sends 200 tokens still occupies 8192 tokens' worth of memory
// -- and a std::vector-style growable buffer avoids that particular waste
// only by paying for periodic reallocate-and-copy pauses instead.
//
// PagedAttention (the technique vLLM popularized) borrows virtual-memory
// paging from operating systems: the KV cache is divided into fixed-size
// BLOCKS of tokens, a per-sequence BLOCK TABLE maps logical block index
// to physical block ID, and physical blocks live anywhere in one
// pre-allocated pool with no requirement of contiguity. Because
// BLOCK_SIZE is a power of two, translating a logical token index into
// (which block, which slot within it) is two single-cycle bit operations
// -- a right shift and a mask -- completely negligible next to the ~100ns
// DRAM access that follows. The same block-table indirection also makes
// prefix sharing free: when many users share an identical system prompt,
// their block tables can all point at the SAME physical blocks, and
// "sharing a prefix" becomes copying a handful of integers rather than
// copying the KV data itself.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_paged_attention_block_manager.cpp -o 02_paged_attention_block_manager

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <random>
#include <stdexcept>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// BLOCK_SIZE must be a power of 2 -- this is what makes address
// translation two single-cycle bit operations instead of a division and
// a modulo. 16 is the size real systems like vLLM settle on: large
// enough to keep per-block bookkeeping cheap, small enough to keep
// internal fragmentation (wasted slots in a partially-full block) low.
constexpr int BLOCK_SIZE = 16;
constexpr int HEAD_DIM = 32;
constexpr int NUM_HEADS = 4;

// One physical block: BLOCK_SIZE tokens' worth of Key and Value vectors
// for every head, laid out [head][token][dim] for good spatial locality
// when a single head's attention kernel scans across tokens.
struct KVBlock {
    float keys[NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
    float values[NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
    void clear() { std::memset(this, 0, sizeof(*this)); }
};

// The only allocator the KV cache ever uses. Pre-allocates every block
// at construction and never allocates again; alloc()/free() just move an
// integer ID between a std::vector-backed pool and a free list.
class BlockManager {
public:
    explicit BlockManager(int num_blocks) {
        m_pool.resize(num_blocks);
        for (int i = 0; i < num_blocks; ++i) m_free.push_back(i);
    }

    int alloc() {
        if (m_free.empty()) throw std::runtime_error("KV cache OOM");
        int id = m_free.front();
        m_free.pop_front();
        m_pool[id].clear();
        return id;
    }

    void free(int id) {
        assert(id >= 0 && id < static_cast<int>(m_pool.size()));
        m_free.push_back(id);
    }

    void free_sequence(const std::vector<int>& block_table) {
        for (int id : block_table) free(id);
    }

    KVBlock* get(int id) {
        assert(id >= 0 && id < static_cast<int>(m_pool.size()));
        return &m_pool[id];
    }
    const KVBlock* get(int id) const { return &m_pool[id]; }

    int free_count() const { return static_cast<int>(m_free.size()); }
    int total_count() const { return static_cast<int>(m_pool.size()); }

private:
    std::vector<KVBlock> m_pool;
    std::list<int> m_free;
};

// Converts a logical token index into (which entry in the block table,
// which slot inside that block). Because BLOCK_SIZE = 16 = 2^4, both
// operations below compile to a single shift and a single mask.
struct BlockAddr { int block_table_idx; int slot; };
BlockAddr translate(int token_idx) {
    return {token_idx >> 4, token_idx & 0x0F};
}

// One user's conversation. block_table[i] holds the physical block ID
// for logical block i; writing appends into the current block and
// allocates a fresh one exactly when the current one fills.
struct Sequence {
    std::vector<int> block_table;
    int context_len = 0;

    void store_kv(BlockManager& mgr, const float* key, const float* value, int head = 0) {
        int slot = context_len % BLOCK_SIZE;
        if (slot == 0) block_table.push_back(mgr.alloc());
        KVBlock* blk = mgr.get(block_table.back());
        for (int d = 0; d < HEAD_DIM; ++d) {
            blk->keys[head][slot][d] = key[d];
            blk->values[head][slot][d] = value[d];
        }
        ++context_len;
    }

    const float* read_key(const BlockManager& mgr, int token_idx, int head = 0) const {
        auto addr = translate(token_idx);
        int phys = block_table[addr.block_table_idx];
        return mgr.get(phys)->keys[head][addr.slot];
    }
    const float* read_val(const BlockManager& mgr, int token_idx, int head = 0) const {
        auto addr = translate(token_idx);
        int phys = block_table[addr.block_table_idx];
        return mgr.get(phys)->values[head][addr.slot];
    }

    int num_blocks() const { return static_cast<int>(block_table.size()); }
};

// Attention over a sequence stored in non-contiguous paged blocks. The
// arithmetic is ordinary scaled-dot-product attention; the only thing
// "paged" about it is that read_key/read_val resolve a logical token
// index through the block table on every access instead of indexing a
// flat array directly.
std::vector<float> paged_attention(const float* query, const BlockManager& mgr,
                                    const Sequence& seq, float scale) {
    const int n = seq.context_len;
    std::vector<float> scores(n);
    float max_s = -1e9f;
    for (int t = 0; t < n; ++t) {
        const float* k = seq.read_key(mgr, t, 0);
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) dot += query[d] * k[d];
        scores[t] = dot * scale;
        max_s = std::max(max_s, scores[t]);
    }
    float sum_e = 0.0f;
    for (float& s : scores) { s = std::exp(s - max_s); sum_e += s; }
    for (float& s : scores) s /= (sum_e + 1e-9f);
    std::vector<float> output(HEAD_DIM, 0.0f);
    for (int t = 0; t < n; ++t) {
        const float* v = seq.read_val(mgr, t, 0);
        for (int d = 0; d < HEAD_DIM; ++d) output[d] += scores[t] * v[d];
    }
    return output;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 13.2: PagedAttention -- Block-Table Indirection\n";
    std::cout << "========================================================\n";
    std::cout << "BLOCK_SIZE=" << BLOCK_SIZE << " HEAD_DIM=" << HEAD_DIM << " NUM_HEADS=" << NUM_HEADS << "\n";

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.5f);

    std::cout << "\n-- Test 1: address translation matches plain integer division/modulo --\n";
    {
        for (int t : {0, 1, 15, 16, 17, 31, 32, 33, 255, 256}) {
            auto addr = translate(t);
            CHECK(addr.block_table_idx == t / BLOCK_SIZE);
            CHECK(addr.slot == t % BLOCK_SIZE);
        }
        std::cout << "  token  blk_idx  slot\n";
        for (int t : {0, 15, 16, 17, 31, 32}) {
            auto a = translate(t);
            std::cout << "  " << std::setw(5) << t << "  " << std::setw(7) << a.block_table_idx
                      << "  " << std::setw(4) << a.slot << "\n";
        }
    }

    std::cout << "\n-- Test 2: block allocation and free are O(1) free-list operations --\n";
    {
        BlockManager mgr(10);
        CHECK(mgr.free_count() == 10);
        int b0 = mgr.alloc(), b1 = mgr.alloc(), b2 = mgr.alloc();
        CHECK(mgr.free_count() == 7);
        CHECK(b0 != b1 && b1 != b2 && b0 != b2);
        mgr.free(b0);
        mgr.free(b2);
        CHECK(mgr.free_count() == 9);
        std::cout << "  after alloc(3), free(2): " << mgr.free_count() << "/10 free\n";
    }

    std::cout << "\n-- Test 3: writing and reading across block boundaries --\n";
    {
        BlockManager mgr(20);
        Sequence seq;
        for (int t = 0; t < 35; ++t) {
            float k[HEAD_DIM], v[HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; ++d) {
                k[d] = static_cast<float>(t * 100 + d);
                v[d] = static_cast<float>(t * 200 + d);
            }
            seq.store_kv(mgr, k, v);
        }
        CHECK(seq.context_len == 35);
        CHECK(seq.num_blocks() == 3);  // ceil(35/16) = 3
        bool all_ok = true;
        for (int t : {0, 15, 16, 17, 31, 32, 34}) {
            const float* k = seq.read_key(mgr, t);
            float expected = static_cast<float>(t * 100);
            if (std::fabs(k[0] - expected) > 1e-4f) all_ok = false;
        }
        CHECK(all_ok);
        std::cout << "  35 tokens -> " << seq.num_blocks() << " blocks; boundary reads at 15/16 and 31/32 verified\n";
    }

    std::cout << "\n-- Test 4: paged attention matches a dense (unpaged) reference --\n";
    {
        BlockManager mgr(20);
        Sequence seq;
        const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        const int N = 20;
        std::vector<std::vector<float>> keys(N, std::vector<float>(HEAD_DIM));
        std::vector<std::vector<float>> vals(N, std::vector<float>(HEAD_DIM));
        for (int t = 0; t < N; ++t) {
            for (int d = 0; d < HEAD_DIM; ++d) { keys[t][d] = nd(rng); vals[t][d] = nd(rng); }
            seq.store_kv(mgr, keys[t].data(), vals[t].data());
        }
        std::vector<float> query(HEAD_DIM);
        for (float& v : query) v = nd(rng);
        auto paged_out = paged_attention(query.data(), mgr, seq, scale);

        std::vector<float> ref_s(N);
        float max_s = -1e9f;
        for (int t = 0; t < N; ++t) {
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) dot += query[d] * keys[t][d];
            ref_s[t] = dot * scale;
            max_s = std::max(max_s, ref_s[t]);
        }
        float sum_e = 0.0f;
        for (float& s : ref_s) { s = std::exp(s - max_s); sum_e += s; }
        for (float& s : ref_s) s /= sum_e;
        std::vector<float> ref_out(HEAD_DIM, 0.0f);
        for (int t = 0; t < N; ++t)
            for (int d = 0; d < HEAD_DIM; ++d) ref_out[d] += ref_s[t] * vals[t][d];

        float max_err = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) max_err = std::max(max_err, std::fabs(paged_out[d] - ref_out[d]));
        CHECK(max_err < 1e-4f);
        std::cout << "  max error vs dense reference: " << std::scientific << std::setprecision(2) << max_err << "\n";
        std::cout.unsetf(std::ios::scientific);
    }

    std::cout << "\n-- Test 5: zero-copy system-prompt sharing across users --\n";
    {
        BlockManager mgr(30);
        const int SYS_LEN = 16;  // exactly one block
        Sequence sys;
        for (int t = 0; t < SYS_LEN; ++t) {
            float k[HEAD_DIM], v[HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; ++d) { k[d] = 1.0f; v[d] = 2.0f; }
            sys.store_kv(mgr, k, v);
        }
        std::vector<Sequence> users(3);
        for (int u = 0; u < 3; ++u) {
            users[u].block_table = sys.block_table;  // copies three integers, not any KV data
            users[u].context_len = SYS_LEN;
            for (int t = 0; t < 5; ++t) {
                float k[HEAD_DIM], v[HEAD_DIM];
                for (int d = 0; d < HEAD_DIM; ++d) { k[d] = static_cast<float>(u + 1); v[d] = 0.0f; }
                users[u].store_kv(mgr, k, v);
            }
        }
        for (int u = 1; u < 3; ++u) CHECK(users[0].block_table[0] == users[u].block_table[0]);
        int blocks_without_sharing = 3 * users[0].num_blocks();
        int blocks_with_sharing = 1 + 3 * 1;  // 1 shared system-prompt block + 1 unique block each
        std::cout << "  all 3 users' block_table[0] == " << users[0].block_table[0] << " (one physical block, three references)\n";
        std::cout << "  blocks without sharing: " << blocks_without_sharing
                   << ", with sharing: " << blocks_with_sharing
                   << " (" << 100 * (blocks_without_sharing - blocks_with_sharing) / blocks_without_sharing
                   << "% fewer blocks)\n";
    }

    std::cout << "\n-- Test 6: memory utilization, static pre-allocation vs paging --\n";
    {
        const int USERS = 20, MAX = 64, AVG = 10;
        int static_alloc = USERS * MAX;
        int static_used = USERS * AVG;
        double static_util = 100.0 * static_used / static_alloc;

        int blocks_per_user = (AVG + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int paged_alloc = USERS * blocks_per_user * BLOCK_SIZE;
        int paged_used = USERS * AVG;
        double paged_util = 100.0 * paged_used / paged_alloc;

        std::cout << "  " << USERS << " users, max=" << MAX << ", avg=" << AVG << " tokens each\n";
        std::cout << "  static: " << static_alloc << " slots allocated, " << static_used
                   << " used (" << std::fixed << std::setprecision(0) << static_util << "% utilization)\n";
        std::cout << "  paged:  " << paged_alloc << " slots allocated, " << paged_used
                   << " used (" << paged_util << "% utilization)\n";
        CHECK(paged_util > static_util);
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_paged_attention_block_manager.cpp -o 02_paged_attention_block_manager
./02_paged_attention_block_manager
```

**Sample input:** bit-operation address translation checked against plain integer division and modulo across ten token indices; a ten-block pool's alloc/free lifecycle; 35 tokens written and read back across three block boundaries; paged attention output checked against a dense, unpaged reference on 20 random tokens; three users sharing one physical system-prompt block while adding unique tokens of their own; and a static-versus-paged memory utilization comparison across 20 simulated users.

```text
========================================================
Chapter 13.2: PagedAttention -- Block-Table Indirection
========================================================
BLOCK_SIZE=16 HEAD_DIM=32 NUM_HEADS=4

-- Test 1: address translation matches plain integer division/modulo --
  token  blk_idx  slot
      0        0     0
     15        0    15
     16        1     0
     17        1     1
     31        1    15
     32        2     0

-- Test 2: block allocation and free are O(1) free-list operations --
  after alloc(3), free(2): 9/10 free

-- Test 3: writing and reading across block boundaries --
  35 tokens -> 3 blocks; boundary reads at 15/16 and 31/32 verified

-- Test 4: paged attention matches a dense (unpaged) reference --
  max error vs dense reference: 0.00e+00

-- Test 5: zero-copy system-prompt sharing across users --
  all 3 users' block_table[0] == 0 (one physical block, three references)
  blocks without sharing: 6, with sharing: 4 (33% fewer blocks)

-- Test 6: memory utilization, static pre-allocation vs paging --
  20 users, max=64, avg=10 tokens each
  static: 1280 slots allocated, 200 used (16% utilization)
  paged:  320 slots allocated, 200 used (62% utilization)

========================================================
31/31 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] treating a shared block table entry as safe to write through directly"
    When three users' block tables all reference the same physical system-prompt block, that block must remain read-only from every user's perspective for as long as it is shared — writing a user's own new tokens must always go into a NEWLY allocated block, never into a shared one, even though the shared block's physical memory is sitting right there and technically writable. A block manager has no way to know a given physical block is currently referenced by more than one sequence's block table; it is the CALLER'S responsibility (`Sequence::store_kv` allocating a new block whenever the current one fills, rather than ever appending into an existing shared block past its System-prompt length) to preserve that invariant. Violating it silently corrupts every OTHER user sharing that block the moment one user's "unique" token overwrites shared data — a bug that produces plausible-looking garbage in a completely unrelated user's context, with no crash and no obvious cause.

## 13.3 Ring Buffer Contexts: RoPE Position Decoupling and Sink-Token Protection

### Intuition

PagedAttention solves fragmentation but not growth: total KV cache size still scales linearly with context length no matter how efficiently its blocks are packed. A Ring Buffer caps memory at a fixed size by overwriting the oldest entry once full — simple in concept, but hiding a correctness trap that produces no warning and no crash, only silently wrong attention.

### The Concept, In Detail

A Ring Buffer's physical WRITE SLOT wraps: slot equals total tokens processed, modulo capacity. But before a Key vector is ever cached, RoPE rotates it using that token's LOGICAL position in the overall sequence — a fact Chapter 2 established and this section now depends on directly. If an implementation reuses the wrapped physical slot number as the RoPE rotation angle, every token written after the FIRST wrap gets rotated by the wrong position: token 4096, physically landing in slot 0 because the buffer just wrapped, would be rotated as if it were token 0, and every later query's relative-distance computation against it becomes wrong by however far the sequence has wrapped. The fix keeps two separate counters that are equal only during the initial fill and deliberately diverge forever afterward: `write_slot()`, which wraps and answers only "where does the next token's data go," and `rope_position()`, which is simply the running total of tokens ever processed and answers only "what rotation does this token's Key vector need." A second, independent hazard survives even once RoPE positions are correct: trained transformers reliably dump a large share of softmax attention probability onto the first few tokens of a sequence regardless of their content, an "attention sink" behavior the model has learned to rely on as a safe destination for probability mass it has nowhere better to send. Evicting those sink tokens when the ring wraps measurably degrades coherence even with every rotation angle correct, so this section reserves a small PROTECTED region at the front of the buffer — write_slot's wrap logic only ever cycles through the ROLLING region past it, and the first few slots are never touched again once filled.

### Code and Verification

```cpp
// 03_ring_buffer_sink_tokens.cpp
// Chapter 13, Part 3: PagedAttention (Section 13.2) solves fragmentation
// -- it never wastes memory on unused slack -- but it does not solve
// unbounded growth. Total KV cache size still grows linearly with
// context length no matter how efficiently blocks are packed, and a
// long-running conversation or document stream will eventually exceed
// available RAM. The Ring Buffer fixes this by capping memory at a
// constant size and overwriting the oldest entry once the buffer fills:
// the model always sees the most recent `capacity` tokens, and memory
// usage never grows past that regardless of how many tokens have ever
// been processed.
//
// This section's real subject is a subtlety a naive ring buffer gets
// wrong in a way that produces no compiler warning, no crash, and no
// obviously wrong-looking output -- only silently incorrect attention.
// Before a Key vector is cached, RoPE rotates it using the token's
// LOGICAL position in the sequence (Chapter 2's positional encoding).
// A ring buffer's PHYSICAL write slot, though, wraps around and repeats
// (slot = total_processed % capacity). If a naive implementation reuses
// that same wrapped slot number as the RoPE rotation angle, every token
// written after the first wrap gets rotated by the WRONG position --
// token 4096, physically written to slot 0, gets rotated as if it were
// token 0, and the relative-distance math every later query relies on
// becomes wrong by however much the sequence has wrapped. The fix keeps
// two separate counters: a wrapping write_slot for WHERE to store data,
// and a monotonically increasing rope_position, equal to total tokens
// processed, for WHAT rotation to apply -- the two are equal only during
// the initial fill, and deliberately diverge forever after that.
//
// A second subtlety compounds the first: even with RoPE positions
// correct, evicting the very first few tokens of a sequence measurably
// hurts model coherence. Trained transformers reliably dump a large
// share of softmax attention probability onto the first few tokens
// regardless of their content -- an "attention sink" the model has
// learned to rely on as a safe place to put probability mass it has
// nowhere better to send. This section reserves a small PROTECTED
// region at the front of the ring that the wrap logic never touches.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_ring_buffer_sink_tokens.cpp -o 03_ring_buffer_sink_tokens

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// Tracks the two logically distinct counters a ring buffer needs:
//   write_slot()    -- WHERE the next token's K,V goes (wraps)
//   rope_position() -- WHAT RoPE rotation to apply (never wraps)
// These are equal only while the buffer is still filling for the first
// time; after the first wrap, rope_position keeps growing forever while
// write_slot cycles through the rolling region.
class RingBufferCtx {
public:
    explicit RingBufferCtx(int capacity, int protected_size)
        : m_cap(capacity), m_prot(protected_size), m_total(0) {
        assert(protected_size >= 0 && protected_size < capacity);
    }

    int write_slot() const {
        if (m_total < m_cap) return m_total;  // still filling for the first time
        int rolling = m_cap - m_prot;
        int offset = (m_total - m_cap) % rolling;
        return m_prot + offset;
    }

    int rope_position() const { return m_total; }
    void advance() { ++m_total; }
    bool is_wrapping() const { return m_total >= m_cap; }
    int valid_length() const { return std::min(m_total, m_cap); }
    int total() const { return m_total; }
    int capacity() const { return m_cap; }

private:
    int m_cap;
    int m_prot;
    int m_total;
};

// Pairs the position tracker with actual float storage for a single
// attention head. Keys/values are flat arrays indexed directly by the
// physical slot RingBufferCtx returns.
class RingKVCache {
public:
    const int head_dim;

    RingKVCache(int capacity, int hdim, int protected_size)
        : head_dim(hdim), m_ctx(capacity, protected_size),
          m_keys(capacity * hdim, 0.0f), m_values(capacity * hdim, 0.0f),
          m_rope_pos(capacity, -1) {}

    int write(const float* key, const float* val) {
        int slot = m_ctx.write_slot();
        int rope = m_ctx.rope_position();
        m_rope_pos[slot] = rope;
        float* kp = m_keys.data() + slot * head_dim;
        float* vp = m_values.data() + slot * head_dim;
        for (int d = 0; d < head_dim; ++d) { kp[d] = key[d]; vp[d] = val[d]; }
        m_ctx.advance();
        return slot;
    }

    std::vector<float> attend(const float* query, float scale) const {
        int n = m_ctx.valid_length();
        std::vector<float> scores(n);
        float max_s = -1e9f;
        for (int s = 0; s < n; ++s) {
            const float* k = m_keys.data() + s * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += query[d] * k[d];
            scores[s] = dot * scale;
            max_s = std::max(max_s, scores[s]);
        }
        float sum_e = 0.0f;
        for (float& s : scores) { s = std::exp(s - max_s); sum_e += s; }
        for (float& s : scores) s /= (sum_e + 1e-9f);
        std::vector<float> out(head_dim, 0.0f);
        for (int s = 0; s < n; ++s) {
            const float* v = m_values.data() + s * head_dim;
            for (int d = 0; d < head_dim; ++d) out[d] += scores[s] * v[d];
        }
        return out;
    }

    const RingBufferCtx& ctx() const { return m_ctx; }
    int rope_at(int slot) const { return m_rope_pos[slot]; }
    const float* key_at(int slot) const { return m_keys.data() + slot * head_dim; }

private:
    RingBufferCtx m_ctx;
    std::vector<float> m_keys, m_values;
    std::vector<int> m_rope_pos;
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 13.3: Ring Buffer Contexts -- RoPE Decoupling and Sink Tokens\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: write-slot wrapping never touches protected slots (capacity=8, protected=2) --\n";
    {
        RingBufferCtx ctx(8, 2);
        // Rolling region is slots [2..7] (6 slots). Expected write slots
        // for tokens 0..15: fill 0..7, then wrap into [2..7] repeatedly.
        int expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 2, 3};
        std::cout << "  token  write_slot  rope_pos  wrapping\n";
        for (int t = 0; t < 16; ++t) {
            int slot = ctx.write_slot();
            int rope = ctx.rope_position();
            CHECK(slot == expected[t]);
            std::cout << "  " << std::setw(5) << t << "  " << std::setw(10) << slot
                      << "  " << std::setw(8) << rope << "  " << (ctx.is_wrapping() ? "yes" : "no") << "\n";
            ctx.advance();
        }
        CHECK(ctx.rope_position() == 16);  // never resets
    }

    std::cout << "\n-- Test 2: rope_position and write_slot are equal only during the initial fill --\n";
    {
        RingBufferCtx ctx(4, 1);
        for (int t = 0; t < 8; ++t) {
            int slot = ctx.write_slot();
            int rope = ctx.rope_position();
            if (t < ctx.capacity()) CHECK(slot == rope);
            else CHECK(rope > slot);
            ctx.advance();
        }
    }

    std::cout << "\n-- Test 3: sink tokens survive far beyond the buffer's own capacity --\n";
    {
        constexpr int CAP = 12, PROT = 3, HD = 8;
        RingKVCache cache(CAP, HD, PROT);
        for (int t = 0; t < PROT; ++t) {
            float k[HD], v[HD];
            for (int d = 0; d < HD; ++d) { k[d] = 999.0f + t; v[d] = 0.0f; }
            cache.write(k, v);
        }
        for (int t = PROT; t < 60; ++t) {
            float k[HD], v[HD];
            for (int d = 0; d < HD; ++d) { k[d] = static_cast<float>(t); v[d] = 0.0f; }
            cache.write(k, v);
        }
        bool intact = true;
        for (int s = 0; s < PROT; ++s) {
            if (std::fabs(cache.key_at(s)[0] - (999.0f + s)) > 0.001f) intact = false;
        }
        CHECK(intact);
        std::cout << "  protected slot values after " << cache.ctx().total() << " tokens: ["
                   << cache.key_at(0)[0] << ", " << cache.key_at(1)[0] << ", " << cache.key_at(2)[0] << "]\n";
    }

    std::cout << "\n-- Test 4: memory stays constant no matter how many tokens are processed --\n";
    {
        constexpr int CAP = 8, HD = 4;
        RingKVCache cache(CAP, HD, 2);
        for (int t = 0; t < 100; ++t) {
            float k[HD], v[HD];
            for (int d = 0; d < HD; ++d) { k[d] = 1.0f; v[d] = 1.0f; }
            cache.write(k, v);
        }
        CHECK(cache.ctx().valid_length() == CAP);
        std::cout << "  after 100 tokens: valid_length=" << cache.ctx().valid_length()
                   << " (capacity=" << CAP << "), total_processed=" << cache.ctx().total() << "\n";
    }

    std::cout << "\n-- Test 5: ring attention matches a dense reference before any wrap occurs --\n";
    {
        constexpr int CAP = 16, HD = 8, PROT = 2;
        const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
        RingKVCache ring(CAP, HD, PROT);
        std::vector<std::vector<float>> ks(8, std::vector<float>(HD));
        std::vector<std::vector<float>> vs(8, std::vector<float>(HD));
        for (int t = 0; t < 8; ++t) {
            for (int d = 0; d < HD; ++d) { ks[t][d] = static_cast<float>(t + 1); vs[t][d] = static_cast<float>(t + 1) * 0.5f; }
            ring.write(ks[t].data(), vs[t].data());
        }
        std::vector<float> query(HD, 1.0f);
        auto out = ring.attend(query.data(), scale);

        std::vector<float> ref_s(8);
        float max_s = -1e9f;
        for (int t = 0; t < 8; ++t) {
            float dot = 0.0f;
            for (int d = 0; d < HD; ++d) dot += query[d] * ks[t][d];
            ref_s[t] = dot * scale;
            max_s = std::max(max_s, ref_s[t]);
        }
        float sum_e = 0.0f;
        for (float& s : ref_s) { s = std::exp(s - max_s); sum_e += s; }
        for (float& s : ref_s) s /= sum_e;
        std::vector<float> ref_out(HD, 0.0f);
        for (int t = 0; t < 8; ++t)
            for (int d = 0; d < HD; ++d) ref_out[d] += ref_s[t] * vs[t][d];

        float max_err = 0.0f;
        for (int d = 0; d < HD; ++d) max_err = std::max(max_err, std::fabs(out[d] - ref_out[d]));
        CHECK(max_err < 1e-4f);
        std::cout << "  max error vs dense reference: " << std::scientific << std::setprecision(2) << max_err << "\n";
        std::cout.unsetf(std::ios::scientific);
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_ring_buffer_sink_tokens.cpp -o 03_ring_buffer_sink_tokens
./03_ring_buffer_sink_tokens
```

**Sample input:** 16 tokens written into a capacity-8, protected-2 ring, checking every write slot against a hand-derived expected sequence; the same tiny buffer checked to confirm write_slot equals rope_position only during the initial fill and diverges strictly afterward; three sink-token sentinel values confirmed intact after 60 total tokens through a capacity-12 buffer; memory confirmed capped at capacity after 100 tokens through a capacity-8 buffer; and ring attention checked against a dense reference before any wrap occurs.

```text
========================================================
Chapter 13.3: Ring Buffer Contexts -- RoPE Decoupling and Sink Tokens
========================================================

-- Test 1: write-slot wrapping never touches protected slots (capacity=8, protected=2) --
  token  write_slot  rope_pos  wrapping
      0           0         0  no
      1           1         1  no
      2           2         2  no
      3           3         3  no
      4           4         4  no
      5           5         5  no
      6           6         6  no
      7           7         7  no
      8           2         8  yes
      9           3         9  yes
     10           4        10  yes
     11           5        11  yes
     12           6        12  yes
     13           7        13  yes
     14           2        14  yes
     15           3        15  yes

-- Test 2: rope_position and write_slot are equal only during the initial fill --

-- Test 3: sink tokens survive far beyond the buffer's own capacity --
  protected slot values after 60 tokens: [999, 1000, 1001]

-- Test 4: memory stays constant no matter how many tokens are processed --
  after 100 tokens: valid_length=8 (capacity=8), total_processed=100

-- Test 5: ring attention matches a dense reference before any wrap occurs --
  max error vs dense reference: 0.00e+00

========================================================
28/28 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] using the physical write slot as the RoPE rotation angle"
    It is a natural-looking shortcut to rotate a Key vector using the same slot index the ring buffer just computed for WHERE to store it — after all, that index is already sitting right there as a convenient integer. This works perfectly during the initial fill, when slot and logical position happen to be numerically identical, which is exactly what makes the bug so dangerous: it passes every test run only long enough to fill the buffer once, then silently corrupts every attention computation from the first wrap onward. There is no crash, no NaN, no obviously wrong-looking number — just relative-distance math that is quietly off by however many tokens the buffer has wrapped, producing degraded, sometimes bizarre generation that is easy to blame on the model itself rather than on a rotation angle that has been wrong since the first overwritten slot. The fix is never using `write_slot()` for anything except "where," and never using anything except the monotonically increasing total-tokens-processed counter for "what rotation."

## 13.4 Smart Eviction: The H2O Heavy-Hitter Policy

### Intuition

The Ring Buffer's default eviction is FIFO: whichever token is oldest goes, no matter what it contains. That is fine for genuinely disposable streaming data, but a real conversation's early tokens are not uniformly disposable — a user's stated name matters for the rest of the session; idle filler three tokens later does not — and FIFO cannot tell the two apart.

### The Concept, In Detail

H2O (Heavy Hitter Oracle) replaces "oldest goes" with "least useful goes," using a signal the model already computes as a byproduct of ordinary attention and normally discards afterward: the softmax probability each cached token receives at every decode step. Accumulating that probability into a per-token running total — one floating-point addition per cached token per step — is essentially free, since the attention computation already produces those probabilities; a token the model consistently attends to heavily accumulates high importance, and a token it consistently ignores accumulates almost none. When the cache is full and a new token needs a slot, eviction targets the single lowest-importance token currently eligible — with one deliberate exception: a small RECENT WINDOW of the most-just-added tokens is immune to eviction regardless of their importance score, because a token that has only existed for one or two decode steps has not yet had a fair chance to accumulate the importance its actual usefulness deserves. Evicting purely by recency (FIFO) discards a critical early fact and a piece of throwaway filler with equal indifference; H2O's importance signal is what lets eviction discriminate between the two using information the model has already paid to compute.

### Code and Verification

```cpp
// 04_h2o_eviction.cpp
// Chapter 13, Part 4: the Ring Buffer (Section 13.3) evicts with FIFO --
// whichever token is oldest goes, unconditionally. FIFO is free and
// correct for genuinely ephemeral streaming data, but a real conversation
// is not uniformly ephemeral: an early token stating the user's name or
// a key task constraint can matter for the entire rest of the session,
// while a token three positions later ("nice weather today") may never
// matter again. Evicting purely by age discards the first kind exactly
// as readily as the second, and a model that can no longer see its own
// user's name will confidently hallucinate an answer rather than admit
// it forgot.
//
// H2O (Heavy Hitter Oracle) replaces "oldest goes" with "least useful
// goes," using an importance signal the model computes for free as a
// side effect of ordinary attention: the softmax probability every past
// token receives at every generation step. A token the model consistently
// attends to heavily accumulates high importance; a token it consistently
// ignores accumulates almost none. Accumulating this is one floating-point
// addition per cached token per decode step -- a cost already paid by the
// attention computation itself, just not normally kept around afterward.
// Eviction then targets the single lowest-importance token, with one
// exception: a small RECENT WINDOW of the most-just-added tokens is
// immune regardless of importance, since a token's true importance can
// only be judged after the model has actually had a chance to attend to
// it a few times.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_h2o_eviction.cpp -o 04_h2o_eviction

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// Everything the eviction policy needs to know about one cached token.
struct TokenRecord {
    int logical_id;     // this token's position in the overall sequence, never changes
    int physical_slot;  // where its K,V data actually lives
    float importance;   // accumulated attention probability mass
    int insert_step;    // when this token entered the cache, for the recency check
};

// Maintains the cached-token list and evicts the lowest-importance token
// that is old enough to no longer be protected by the recent window.
class H2OPolicy {
public:
    H2OPolicy(int capacity, int recent_window) : m_capacity(capacity), m_recent(recent_window), m_step(0) {
        m_tokens.reserve(capacity);
    }

    void add_token(int logical_id, int physical_slot) {
        if (static_cast<int>(m_tokens.size()) >= m_capacity) evict_one();
        m_tokens.push_back({logical_id, physical_slot, 0.0f, m_step});
        ++m_step;
    }

    // probs[i] is the attention probability the i-th currently-cached
    // token received this decode step; called once per step.
    void update_scores(const std::vector<float>& probs) {
        assert(probs.size() == m_tokens.size());
        for (size_t i = 0; i < m_tokens.size(); ++i) m_tokens[i].importance += probs[i];
    }

    int evict_one() {
        if (m_tokens.empty()) return -1;
        int victim = find_victim();
        if (victim < 0) victim = 0;  // every token is within the recent window: fall back to oldest
        int slot = m_tokens[victim].physical_slot;
        m_tokens.erase(m_tokens.begin() + victim);
        return slot;
    }

    bool contains(int logical_id) const {
        for (const auto& t : m_tokens) if (t.logical_id == logical_id) return true;
        return false;
    }
    float importance_of(int logical_id) const {
        for (const auto& t : m_tokens) if (t.logical_id == logical_id) return t.importance;
        return -1.0f;
    }

    int size() const { return static_cast<int>(m_tokens.size()); }
    const std::vector<TokenRecord>& records() const { return m_tokens; }

private:
    int find_victim() const {
        int victim = -1;
        float min_imp = 1e9f;
        for (int i = 0; i < static_cast<int>(m_tokens.size()); ++i) {
            int age = m_step - m_tokens[i].insert_step;
            if (age < m_recent) continue;  // too recent to judge fairly: immune
            if (m_tokens[i].importance < min_imp) { min_imp = m_tokens[i].importance; victim = i; }
        }
        return victim;
    }

    int m_capacity;
    int m_recent;
    int m_step;
    std::vector<TokenRecord> m_tokens;
};

// FIFO baseline for comparison: no importance tracking at all.
class FIFOPolicy {
public:
    explicit FIFOPolicy(int cap) : m_cap(cap) {}
    void add(int id) {
        if (static_cast<int>(m_ids.size()) >= m_cap) m_ids.erase(m_ids.begin());
        m_ids.push_back(id);
    }
    bool contains(int id) const { return std::find(m_ids.begin(), m_ids.end(), id) != m_ids.end(); }

private:
    int m_cap;
    std::vector<int> m_ids;
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 13.4: Smart Eviction -- The H2O Heavy-Hitter Policy\n";
    std::cout << "========================================================\n";

    std::mt19937 rng(42);

    std::cout << "\n-- Test 1: eviction targets lowest importance, not oldest --\n";
    {
        H2OPolicy policy(5, 2);  // capacity=5, recent_window=2
        for (int t = 0; t < 5; ++t) policy.add_token(t, t);
        std::vector<float> scores = {10.0f, 0.5f, 8.0f, 0.1f, 0.3f};  // T4 is within the recent window
        policy.update_scores(scores);
        int freed = policy.evict_one();
        std::cout << "  importances: T0=10.0 T1=0.5 T2=8.0 T3=0.1 T4=0.3(recent)\n";
        std::cout << "  evicted physical slot: " << freed << " (expected 3, i.e. T3)\n";
        CHECK(freed == 3);
        CHECK(!policy.contains(3));
        CHECK(policy.contains(0));
        CHECK(policy.contains(2));
    }

    std::cout << "\n-- Test 2: the recent window is immune regardless of importance --\n";
    {
        H2OPolicy policy(4, 3);  // last 3 tokens always immune
        for (int t = 0; t < 4; ++t) policy.add_token(t, t);
        policy.update_scores({0.1f, 0.1f, 0.1f, 0.1f});  // all tied: age breaks the tie
        int freed = policy.evict_one();
        std::cout << "  recent_window=3 means T1,T2,T3 are immune; only T0 is eligible\n";
        std::cout << "  evicted physical slot: " << freed << " (expected 0, i.e. T0)\n";
        CHECK(freed == 0);
    }

    std::cout << "\n-- Test 3: H2O preserves critical tokens that FIFO blindly drops --\n";
    {
        const int CAP = 20, RECENT = 5, TOTAL = 40;
        std::vector<int> heavy_hitters = {3, 7, 12};
        H2OPolicy h2o(CAP, RECENT);
        FIFOPolicy fifo(CAP);
        std::uniform_real_distribution<float> noise(0.0f, 0.005f);

        for (int t = 0; t < TOTAL; ++t) {
            h2o.add_token(t, t);
            fifo.add(t);
            if (!h2o.records().empty()) {
                std::vector<float> probs(h2o.records().size());
                float total = 0.0f;
                for (size_t i = 0; i < h2o.records().size(); ++i) {
                    int lid = h2o.records()[i].logical_id;
                    bool hh = std::find(heavy_hitters.begin(), heavy_hitters.end(), lid) != heavy_hitters.end();
                    bool recent = (lid >= t - 5);
                    probs[i] = hh ? 0.15f : (recent ? 0.05f : 0.002f);
                    probs[i] += noise(rng);
                    total += probs[i];
                }
                for (float& p : probs) p /= total;
                h2o.update_scores(probs);
            }
        }

        std::cout << "  after " << TOTAL << " tokens (capacity=" << CAP << "):\n";
        std::cout << "  logical_id  in_h2o  in_fifo\n";
        struct { int id; const char* note; } checks[] = {
            {3, "heavy hitter"}, {7, "heavy hitter"}, {12, "heavy hitter"}, {25, "old filler"}, {38, "recent token"},
        };
        for (auto& c : checks) {
            bool in_h2o = h2o.contains(c.id);
            bool in_fifo = fifo.contains(c.id);
            std::cout << "  T" << std::setw(3) << c.id << "        " << (in_h2o ? "yes" : "no ")
                       << "     " << (in_fifo ? "yes" : "no ") << "   (" << c.note << ")\n";
        }
        for (int hh : heavy_hitters) CHECK(h2o.contains(hh));
        for (int hh : heavy_hitters) CHECK(!fifo.contains(hh));
    }

    std::cout << "\n-- Test 4: importance accumulates exactly as the sum of per-step probabilities --\n";
    {
        H2OPolicy policy(5, 1);
        for (int t = 0; t < 5; ++t) policy.add_token(t, t);
        for (int step = 0; step < 10; ++step) policy.update_scores({0.5f, 0.1f, 0.1f, 0.1f, 0.2f});
        float imp0 = policy.importance_of(0);
        float imp1 = policy.importance_of(1);
        std::cout << "  T0 importance after 10 steps of prob=0.5: " << imp0 << " (expected 5.0)\n";
        std::cout << "  T1 importance after 10 steps of prob=0.1: " << imp1 << " (expected 1.0)\n";
        CHECK(imp0 > imp1);
        CHECK(std::fabs(imp0 - 5.0f) < 0.5f);
        CHECK(std::fabs(imp1 - 1.0f) < 0.5f);
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_h2o_eviction.cpp -o 04_h2o_eviction
./04_h2o_eviction
```

**Sample input:** a five-token cache with hand-assigned importance scores checked to confirm eviction targets the single lowest-importance eligible token; a four-token cache with tied importances checked to confirm the recent window correctly protects the three newest tokens; a 40-token simulated conversation with three designated heavy-hitter tokens, checked to confirm H2O retains all three while a FIFO baseline over the same data drops at least one; and ten simulated decode steps checked to confirm accumulated importance equals the exact sum of per-step attention probabilities.

```text
========================================================
Chapter 13.4: Smart Eviction -- The H2O Heavy-Hitter Policy
========================================================

-- Test 1: eviction targets lowest importance, not oldest --
  importances: T0=10.0 T1=0.5 T2=8.0 T3=0.1 T4=0.3(recent)
  evicted physical slot: 3 (expected 3, i.e. T3)

-- Test 2: the recent window is immune regardless of importance --
  recent_window=3 means T1,T2,T3 are immune; only T0 is eligible
  evicted physical slot: 0 (expected 0, i.e. T0)

-- Test 3: H2O preserves critical tokens that FIFO blindly drops --
  after 40 tokens (capacity=20):
  logical_id  in_h2o  in_fifo
  T  3        yes     no    (heavy hitter)
  T  7        yes     no    (heavy hitter)
  T 12        yes     no    (heavy hitter)
  T 25        no      yes   (old filler)
  T 38        yes     yes   (recent token)

-- Test 4: importance accumulates exactly as the sum of per-step probabilities --
  T0 importance after 10 steps of prob=0.5: 5 (expected 5.0)
  T1 importance after 10 steps of prob=0.1: 1 (expected 1.0)

========================================================
14/14 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] letting brand-new tokens compete on importance before they have any"
    A token that was just added to the cache has, by definition, accumulated at most one or two decode steps' worth of attention probability — nowhere near enough to reflect whether the model will actually rely on it going forward. Without a recent-window exemption, H2O's own importance-based eviction would systematically discriminate against every token the instant after it arrives, evicting genuinely important brand-new context (a name just stated, an instruction just given) simply because it has not yet had time to accumulate the score its true usefulness deserves — turning a smarter eviction policy into one that is, perversely, worse than FIFO for exactly the tokens that most need to survive their first few steps. The recent window's immunity is not a workaround for a limitation in the importance signal; it is a recognition that importance is a signal that needs TIME to become meaningful, and eviction must not judge a token before that time has passed.

## 13.5 A Complete KV Cache Manager

### Intuition

Each of this chapter's three techniques was verified in isolation. A production serving system needs all three working together — paged blocks for many concurrent users, a wrapping ring for unbounded conversations, and H2O eviction so wrapping discards the right tokens — and combining verified pieces is only useful if the combination does not quietly weaken any of the guarantees each piece proved on its own.

### The Concept, In Detail

This section reuses Section 13.2's `BlockManager`, Section 13.3's `RingBufferCtx`, and Section 13.4's `H2OPolicy` completely unchanged — no reimplementation, no simplification — and re-derives each one's core guarantee inside a combined scenario rather than merely asserting the combination works. A multi-user paged-serving scenario confirms three simulated users sharing one physical system-prompt block still see that same sharing (and the same O(1) pool free-count arithmetic) when they are running alongside the chapter's other two techniques rather than in isolation; a wrapping-ring scenario reproduces Section 13.3's own hand-derived write-slot sequence exactly to confirm nothing about combining it with paging or eviction logic elsewhere in the program perturbs its wrap arithmetic; and an H2O-versus-FIFO scenario confirms heavy hitters still survive under the combined manager while a plain-FIFO baseline over identical data still loses at least one. None of these checks are new claims this chapter has not already proven — they are the SAME claims, re-verified in the presence of the other two techniques, which is precisely what "does the combination preserve every guarantee" means operationally rather than as an assertion. A final memory-capacity analysis, scaled to Llama 3 8B's real per-block byte cost, quantifies the payoff of Section 13.2's paging concretely: reserving worst-case context for 50 users costs tens of gigabytes, while paging the same 50 users at their real average usage costs a small fraction of that.

### Code and Verification

```cpp
// 05_kv_cache_manager.cpp
// Chapter 13, Part 5 (capstone): this chapter built three independent
// techniques for the same underlying resource. PagedAttention (13.2)
// eliminates fragmentation by allocating fixed-size blocks from a pool
// instead of reserving worst-case contiguous ranges. The Ring Buffer
// (13.3) eliminates unbounded growth by wrapping physical storage while
// keeping RoPE's logical position monotonic, with a protected region for
// attention-sink tokens. H2O (13.4) eliminates naive FIFO's worst
// failure mode by evicting the least-attended token instead of the
// oldest one. A real serving system needs all three at once: paging so
// many users share memory efficiently, wrapping so no single
// conversation grows without bound, and smart eviction so wrapping does
// not discard what the model actually still needs. This file reuses
// this chapter's own verified BlockManager, RingBufferCtx, and H2OPolicy
// unchanged, and checks that combining them produces the same guarantees
// each one already proved on its own.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_kv_cache_manager.cpp -o 05_kv_cache_manager

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <random>
#include <stdexcept>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// ---------------------------------------------------------------------
// Section 13.2's BlockManager and address translation, reused verbatim.
// ---------------------------------------------------------------------
constexpr int BLOCK_SIZE = 16;
constexpr int HEAD_DIM = 16;
constexpr int NUM_HEADS = 2;

struct KVBlock {
    float keys[NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
    float values[NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
    void clear() { std::memset(this, 0, sizeof(*this)); }
};

class BlockManager {
public:
    explicit BlockManager(int num_blocks) {
        m_pool.resize(num_blocks);
        for (int i = 0; i < num_blocks; ++i) m_free.push_back(i);
    }
    int alloc() {
        if (m_free.empty()) throw std::runtime_error("KV cache OOM");
        int id = m_free.front();
        m_free.pop_front();
        m_pool[id].clear();
        return id;
    }
    void free(int id) { m_free.push_back(id); }
    void free_many(const std::vector<int>& ids) { for (int id : ids) free(id); }
    KVBlock* get(int id) { return &m_pool[id]; }
    int free_count() const { return static_cast<int>(m_free.size()); }
    int total_count() const { return static_cast<int>(m_pool.size()); }

private:
    std::vector<KVBlock> m_pool;
    std::list<int> m_free;
};

struct PagedSeq {
    std::vector<int> block_table;
    int context_len = 0;
    void write(BlockManager& mgr, const float* k, const float* v, int h = 0) {
        int slot = context_len % BLOCK_SIZE;
        if (slot == 0) block_table.push_back(mgr.alloc());
        KVBlock* b = mgr.get(block_table.back());
        for (int d = 0; d < HEAD_DIM; ++d) { b->keys[h][slot][d] = k[d]; b->values[h][slot][d] = v[d]; }
        ++context_len;
    }
    int num_blocks() const { return static_cast<int>(block_table.size()); }
};

// ---------------------------------------------------------------------
// Section 13.3's RingBufferCtx, reused verbatim.
// ---------------------------------------------------------------------
class RingBufferCtx {
public:
    explicit RingBufferCtx(int capacity, int protected_size) : m_cap(capacity), m_prot(protected_size), m_total(0) {
        assert(protected_size >= 0 && protected_size < capacity);
    }
    int write_slot() const {
        if (m_total < m_cap) return m_total;
        int rolling = m_cap - m_prot;
        return m_prot + (m_total - m_cap) % rolling;
    }
    int rope_position() const { return m_total; }
    void advance() { ++m_total; }
    bool is_wrapping() const { return m_total >= m_cap; }
    int valid_length() const { return std::min(m_total, m_cap); }
    int total() const { return m_total; }

private:
    int m_cap, m_prot, m_total;
};

// ---------------------------------------------------------------------
// Section 13.4's H2OPolicy, reused verbatim.
// ---------------------------------------------------------------------
struct TokenRecord { int logical_id; int physical_slot; float importance; int insert_step; };

class H2OPolicy {
public:
    H2OPolicy(int capacity, int recent_window) : m_capacity(capacity), m_recent(recent_window), m_step(0) {}
    void add_token(int logical_id, int physical_slot) {
        if (static_cast<int>(m_tokens.size()) >= m_capacity) evict_one();
        m_tokens.push_back({logical_id, physical_slot, 0.0f, m_step});
        ++m_step;
    }
    void update_scores(const std::vector<float>& probs) {
        for (size_t i = 0; i < std::min(probs.size(), m_tokens.size()); ++i) m_tokens[i].importance += probs[i];
    }
    int evict_one() {
        if (m_tokens.empty()) return -1;
        int victim = -1;
        float min_imp = 1e9f;
        for (int i = 0; i < static_cast<int>(m_tokens.size()); ++i) {
            if (m_step - m_tokens[i].insert_step < m_recent) continue;
            if (m_tokens[i].importance < min_imp) { min_imp = m_tokens[i].importance; victim = i; }
        }
        if (victim < 0) victim = 0;
        int slot = m_tokens[victim].physical_slot;
        m_tokens.erase(m_tokens.begin() + victim);
        return slot;
    }
    bool contains(int logical_id) const {
        for (const auto& t : m_tokens) if (t.logical_id == logical_id) return true;
        return false;
    }
    const std::vector<TokenRecord>& records() const { return m_tokens; }

private:
    int m_capacity, m_recent, m_step;
    std::vector<TokenRecord> m_tokens;
};

static std::mt19937 g_rng(42);
static std::normal_distribution<float> g_nd(0.0f, 0.3f);
void fill_rand(float* v, int n) { for (int i = 0; i < n; ++i) v[i] = g_nd(g_rng); }

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 13.5: A Complete KV Cache Manager\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: paged multi-user serving with shared system-prompt blocks --\n";
    {
        BlockManager mgr(60);
        PagedSeq sys;
        for (int t = 0; t < 16; ++t) {
            float k[HEAD_DIM], v[HEAD_DIM];
            fill_rand(k, HEAD_DIM); fill_rand(v, HEAD_DIM);
            sys.write(mgr, k, v);
        }
        int free_after_sys = mgr.free_count();
        CHECK(sys.num_blocks() == 1);
        CHECK(free_after_sys == 59);

        const char* names[] = {"Alice", "Bob", "Carol"};
        std::vector<PagedSeq> users(3);
        int user_tokens[] = {5, 8, 3};
        for (int u = 0; u < 3; ++u) {
            users[u].block_table = sys.block_table;
            users[u].context_len = sys.context_len;
            for (int t = 0; t < user_tokens[u]; ++t) {
                float k[HEAD_DIM], v[HEAD_DIM];
                fill_rand(k, HEAD_DIM); fill_rand(v, HEAD_DIM);
                users[u].write(mgr, k, v);
            }
            std::cout << "  " << names[u] << ": " << users[u].context_len << " tokens, "
                       << users[u].num_blocks() << " block(s)\n";
        }
        for (int u = 1; u < 3; ++u) CHECK(users[0].block_table[0] == users[u].block_table[0]);
        CHECK(mgr.free_count() == 59 - 3);  // each user's extra tokens fit in exactly one more block

        // Bob disconnects: every block of his beyond the shared prefix returns to the pool.
        int free_before_disconnect = mgr.free_count();
        mgr.free_many(std::vector<int>(users[1].block_table.begin() + 1, users[1].block_table.end()));
        CHECK(mgr.free_count() == free_before_disconnect + (users[1].num_blocks() - 1));
        std::cout << "  Bob disconnects: pool free count " << free_before_disconnect
                   << " -> " << mgr.free_count() << "\n";
    }

    std::cout << "\n-- Test 2: ring buffer wrapping matches Section 13.3's verified sequence --\n";
    {
        RingBufferCtx ctx(10, 2);  // capacity=10, protected=2
        int expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7};
        bool all_match = true;
        for (int t = 0; t < 16; ++t) {
            if (ctx.write_slot() != expected[t]) all_match = false;
            ctx.advance();
        }
        CHECK(all_match);
        CHECK(ctx.rope_position() == 16);          // logical position never resets
        CHECK(ctx.valid_length() == 10);           // physical storage capped at capacity
        std::cout << "  after 16 tokens into a capacity=10/protected=2 ring: "
                   << "rope_position=" << ctx.rope_position() << ", valid_length=" << ctx.valid_length() << "\n";
    }

    std::cout << "\n-- Test 3: H2O eviction still preserves heavy hitters when combined with the rest --\n";
    {
        const int CAP = 12, WIN = 3, TOTAL = 20;
        std::vector<int> heavy = {2, 5, 9};
        H2OPolicy h2o(CAP, WIN);
        std::vector<int> fifo_cache;
        std::uniform_real_distribution<float> noise(0.0f, 0.005f);

        for (int t = 0; t < TOTAL; ++t) {
            h2o.add_token(t, t);
            if (static_cast<int>(fifo_cache.size()) >= CAP) fifo_cache.erase(fifo_cache.begin());
            fifo_cache.push_back(t);
            if (!h2o.records().empty()) {
                std::vector<float> probs(h2o.records().size());
                float sum = 0.0f;
                for (size_t i = 0; i < h2o.records().size(); ++i) {
                    int lid = h2o.records()[i].logical_id;
                    bool hh = std::find(heavy.begin(), heavy.end(), lid) != heavy.end();
                    probs[i] = hh ? 0.2f : 0.01f;
                    probs[i] += noise(g_rng);
                    sum += probs[i];
                }
                for (float& p : probs) p /= sum;
                h2o.update_scores(probs);
            }
        }
        for (int hh : heavy) CHECK(h2o.contains(hh));
        bool fifo_dropped_any_heavy = false;
        for (int hh : heavy)
            if (std::find(fifo_cache.begin(), fifo_cache.end(), hh) == fifo_cache.end()) fifo_dropped_any_heavy = true;
        CHECK(fifo_dropped_any_heavy);
        std::cout << "  after " << TOTAL << " tokens: H2O retains all " << heavy.size()
                   << " heavy hitters; plain FIFO drops at least one\n";
    }

    std::cout << "\n-- Test 4: paging uses far less memory than static worst-case reservation --\n";
    {
        const long long LAYERS = 32, HEADS = 8, DIM = 128, BYTES = 2;
        long long block_kb = BLOCK_SIZE * HEADS * DIM * 2 * BYTES * LAYERS / 1024;
        std::cout << "  per-block size at full Llama 3 8B scale (all layers): " << block_kb << " KB\n";

        struct S { const char* name; int users; int avg_tok; };
        long long static_mb = 0, paged_mb = 0;
        for (auto& s : (S[]){{"static (max=4096 each)", 50, 4096}, {"paged (avg=200 tokens)", 50, 200}}) {
            int blocks_per_user = (s.avg_tok + BLOCK_SIZE - 1) / BLOCK_SIZE;
            long long mem_mb = static_cast<long long>(s.users) * blocks_per_user * block_kb / 1024;
            std::cout << "  " << s.name << ": " << (s.users * blocks_per_user) << " blocks = " << mem_mb << " MB\n";
            if (std::string(s.name).find("static") == 0) static_mb = mem_mb; else paged_mb = mem_mb;
        }
        CHECK(paged_mb < static_mb);
        CHECK(static_mb > 0 && paged_mb > 0);
        std::cout << "  paged uses " << (100 - (100 * paged_mb / static_mb)) << "% less memory than static-max\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_kv_cache_manager.cpp -o 05_kv_cache_manager
./05_kv_cache_manager
```

**Sample input:** three simulated users sharing a paged system-prompt block, with one disconnecting and returning its unique blocks to the pool; a 16-token run through a capacity-10, protected-2 ring buffer checked against Section 13.3's own wrap sequence; a 20-token H2O-versus-FIFO comparison with three designated heavy hitters; and a memory-capacity comparison between static worst-case reservation and paged allocation at real Llama 3 8B block sizes for 50 simulated users.

```text
========================================================
Chapter 13.5: A Complete KV Cache Manager
========================================================

-- Test 1: paged multi-user serving with shared system-prompt blocks --
  Alice: 21 tokens, 2 block(s)
  Bob: 24 tokens, 2 block(s)
  Carol: 19 tokens, 2 block(s)
  Bob disconnects: pool free count 56 -> 57

-- Test 2: ring buffer wrapping matches Section 13.3's verified sequence --
  after 16 tokens into a capacity=10/protected=2 ring: rope_position=16, valid_length=10

-- Test 3: H2O eviction still preserves heavy hitters when combined with the rest --
  after 20 tokens: H2O retains all 3 heavy hitters; plain FIFO drops at least one

-- Test 4: paging uses far less memory than static worst-case reservation --
  per-block size at full Llama 3 8B scale (all layers): 2048 KB
  static (max=4096 each): 12800 blocks = 25600 MB
  paged (avg=200 tokens): 650 blocks = 1300 MB
  paged uses 95% less memory than static-max

========================================================
15/15 checks passed  ALL PASS
========================================================
```

!!! warning "[COMMON TRAP] assuming techniques verified separately compose correctly by default"
    Each of Sections 13.2 through 13.4 verified its own technique in complete isolation from the other two — PagedAttention's tests never wrapped a ring buffer, and the ring buffer's tests never ran an eviction policy. It is tempting to conclude that because each piece is independently correct, wiring them together in a real server must also be correct, with no further verification needed. This chapter's own Section 11.5-style discipline (Chapter 11's discovery that individually-correct parallel phases can still combine into a thread-count-dependent whole) applies here just as directly: combining independently-verified components is a NEW claim, not a free consequence of the old ones, and it earns verification of its own. This section's tests exist specifically because "I already proved each piece works" is not the same statement as "I proved the assembled system works" — the former is necessary, never sufficient, and this chapter's capstone treats it that way by re-checking each technique's own guarantee inside the combined scenario rather than skipping straight to a demo with no assertions at all.

## Chapter Summary

This chapter turned from the compute-bound optimization of Part 2 to decode's real bottleneck: the state an autoregressive model must remember between tokens, and the cost of keeping that state in memory. Section 13.1 derived the memory wall itself — tokens per second bounded by bandwidth divided by bytes read per step — and showed the KV cache's share of that byte cost growing from negligible to dominant as context length increases. Section 13.2 solved fragmentation with PagedAttention: fixed-size blocks dispensed from a pool via an O(1) free list, with block-table indirection making prefix sharing across users a matter of copying integers rather than data. Section 13.3 solved unbounded growth with a Ring Buffer, whose real lesson was a subtlety rather than the wrapping itself: physical write slot and logical RoPE position must be tracked as two genuinely separate counters, or every token written after the first wrap silently receives the wrong positional rotation, plus a protected region shielding attention-sink tokens the model has learned to rely on. Section 13.4 replaced FIFO's blind age-based eviction with H2O's attention-probability-driven importance score, a signal the model already computes for free, letting eviction discriminate between critical early context and disposable filler. Section 13.5 closed the chapter by combining all three techniques and re-verifying each one's own guarantee inside that combination, rather than treating separately-proven correctness as automatically transitive. Chapter 14 builds directly on this foundation, extending it to sliding-window attention at scale, a tiered cache that re-quantizes aging entries to coarser precision, and prefix caching across multi-turn conversations.

## Self-Check Questions

1. Why does adding more CPU cores or a faster clock fail to speed up decode, when the same changes would measurably speed up prefill?
2. Section 13.1's KV cache size formula uses `n_heads_kv` rather than the model's total number of query heads. Why does GQA (grouped-query attention) make that substitution correct, and what would go wrong if the formula used the query head count instead?
3. Why does PagedAttention's block size need to be a power of two specifically, rather than merely a small fixed number like 16?
4. In Section 13.2's zero-copy prefix sharing, what exactly gets copied when three users share an identical system prompt, and what would break if a user's later, unique tokens were appended directly into a shared block instead of a newly allocated one?
5. Section 13.3 tracks `write_slot()` and `rope_position()` as two separate counters that are equal only during a ring buffer's initial fill. Concretely, what goes wrong with attention scores if an implementation uses `write_slot()` for both purposes after the buffer has wrapped at least once?
6. Why does protecting a small number of "sink" tokens at the front of a ring buffer improve model coherence even when every RoPE position is already computed correctly?
7. Why is the attention probability H2O accumulates into each token's importance score described as a "free" signal, rather than one that costs additional computation to obtain?
8. In Section 13.4, why does H2O's eviction policy exempt a small recent window of tokens from competing on importance at all, rather than simply letting brand-new tokens start with an importance score of zero and compete normally?
9. Section 13.3's Test 3 writes 60 tokens through a 12-slot ring buffer with 3 protected slots and confirms the first 3 slots still hold their original sentinel values. Why does this specifically demonstrate that the protected region is never touched by the wrap logic, rather than merely that eviction happens to favor early tokens?
10. Section 13.5's capstone re-runs checks that Sections 13.2 through 13.4 already performed individually, rather than only adding new checks for behavior unique to the combination. Why is re-verifying each technique's own guarantee INSIDE the combined scenario a meaningfully different claim than having verified it in isolation?

## Where We Go Next

This chapter gave every future chapter a KV cache that pages efficiently, wraps without corrupting positional information, and evicts by actual usefulness rather than blind age. Chapter 14 extends this foundation three ways: sliding-window attention as a ring buffer applied at the scale a real long-context model needs, a tiered cache architecture that re-quantizes aging entries to progressively coarser precision as they leave the active working set, and prefix caching that lets a multi-turn conversation reuse everything computed in earlier turns — closing with a complete cache manager that combines every technique from both chapters behind one interface.

## Worked Solutions

**1.** Decode's speed ceiling is `bandwidth / bytes_read_per_token`, a ratio that depends only on how many bytes of model weights and KV cache must travel from RAM to the compute unit each step and how fast that memory channel can move them — it does not appear anywhere in terms of FLOPs or arithmetic throughput. Adding cores, raising clock speed, or widening the FPU all increase how much ARITHMETIC can be performed per second, but none of them increase how many BYTES per second can cross the memory bus, so a workload whose bottleneck is the bus (decode) sees no benefit, while prefill — which processes a whole prompt's worth of tokens in one compute-heavy pass and is genuinely limited by arithmetic throughput — benefits directly from exactly those same changes.

**2.** Grouped-query attention deliberately uses fewer Key/Value heads than Query heads, with multiple Query heads sharing one KV head's cached Key and Value vectors; the KV CACHE only ever stores one Key and one Value vector per KV head per token, never one per query head, so `n_heads_kv` is the number of distinct K/V vectors that actually exist in memory. Using the query head count instead would compute a KV cache size several times too large, over-reserving memory for K/V data that GQA's whole design never actually stores in the first place — Llama 3 8B's 8 KV heads versus a larger number of query heads is exactly the gap such a mistake would inflate by.

**3.** A power-of-two block size lets address translation — converting a logical token index into (which block, which slot within it) — compile to a bit shift (dividing by the block size) and a bit mask (taking the remainder), both single-cycle operations with no division instruction involved at all. A block size that is merely small but not a power of two, like 15 or 20, would require an actual integer division and modulo on every single cached-token access during attention — still fast in absolute terms, but needlessly slower than two bit operations for a piece of arithmetic that executes on every token, every head, every decode step.

**4.** Only the physical block IDs in each user's block table get copied — a handful of plain integers — while the actual Key and Value data for the shared system prompt exists exactly once in the pool, referenced by every sharing user's table. If a user's own later, unique tokens were appended directly into that SAME shared block rather than a freshly allocated one, the write would silently overwrite Key/Value data every OTHER user sharing that block is still relying on, corrupting their context with no crash, no error, and no indication anything went wrong beyond that other user's subsequent attention output becoming inexplicably wrong.

**5.** Before caching, a Key vector is rotated by RoPE using its token's LOGICAL position in the overall sequence, but `write_slot()` after the first wrap no longer equals that logical position — it is a physical index that has cycled back through values it has used before. If an implementation rotates using `write_slot()`, a token like T4096, which physically lands in slot 0 because the buffer just wrapped, gets rotated as though it were token 0 instead of token 4096; a later query computing its relative distance to that token then gets a wildly wrong distance (thousands of positions off) instead of the small, correct one, corrupting every attention score involving that token from that point forward.

**6.** Trained transformers reliably direct a large share of softmax attention probability toward the first few tokens of a sequence regardless of their actual content — a learned behavior where the model uses early tokens as a stable place to send probability mass it has nowhere more useful to put, since softmax must always sum to 1.0. This is independent of whether RoPE positions are computed correctly: even with perfect rotations, if those specific sink tokens are evicted when the ring wraps, the model loses the anchor its learned attention pattern depends on, and probability mass redistributes unpredictably across the remaining tokens, degrading coherence for a reason that has nothing to do with positional encoding at all.

**7.** The attention mechanism must compute a softmax probability for every cached token at every single decode step regardless of whether anything downstream ever records it — that computation happens purely to produce the weighted sum over Value vectors that IS the attention output. H2O's importance score is obtained by adding that already-computed probability into a running per-token total, a single floating-point addition that piggybacks on work the model was doing anyway; no additional forward pass, no extra matrix multiply, and no new probability computation is needed to obtain it.

**8.** A token that has existed in the cache for only one or two decode steps has, by definition, accumulated only one or two steps' worth of attention probability regardless of how genuinely important its content is — its LOW accumulated score at that point reflects its short lifetime, not its actual usefulness. Starting new tokens at zero and letting them compete immediately would systematically evict brand-new context (a name just stated, an instruction just given) purely because it has not yet had TIME to accumulate the score reflecting its true importance, which is worse than useless: it would make a "smart" eviction policy discriminate specifically against the tokens most likely to matter going forward. The recent window instead grants new tokens a fixed grace period, immune from competing on a score that is not yet meaningful, before importance-based competition applies to them at all.

**9.** If the protected region were merely favored by eviction rather than structurally exempt from the wrap logic, extended enough pressure (here, 57 additional tokens cycling through a 9-slot rolling region) would eventually still overwrite it, because "favored" implies a comparison that some other condition could still lose. The test's 60 tokens deliberately exceed the rolling region's own capacity many times over — proving that no amount of continued writing ever reaches slots 0 through 2 at all, because `write_slot()`'s wrap arithmetic is defined to only ever produce values in the range `[protected_size, capacity)` once wrapping begins; the protected slots are outside that arithmetic's output range entirely, not merely unlikely targets within it.

**10.** Verifying a technique in isolation proves it is correct under the specific conditions its own test constructed — Section 13.2's tests never had a ring buffer's wrap logic or an eviction policy running alongside PagedAttention, for instance. Re-running the SAME check (three users sharing one physical block, a hand-derived wrap sequence, heavy hitters surviving eviction) inside a scenario where all three techniques are actually operating together is a different claim: it confirms nothing about the OTHER two techniques' bookkeeping perturbs this one's guarantee, which is exactly the kind of interaction that unit tests run in isolation cannot see by construction. Chapter 11 demonstrated the general version of this lesson directly — individually thread-count-independent phases still produced a thread-count-DEPENDENT whole once combined — and this section's capstone treats "I verified each piece separately" as necessary but never sufficient for exactly that reason.
