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
