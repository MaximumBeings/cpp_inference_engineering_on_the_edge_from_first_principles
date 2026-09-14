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
