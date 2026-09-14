// 03_prefix_caching.cpp
// Chapter 14, Part 3: In a multi-turn conversation, each new user message
// is appended to a growing history that already includes the system
// prompt and every prior turn. Re-running the model over that entire
// shared history on every single turn is pure waste -- the KV cache
// computed for turn N is a checkpoint of the model's state after
// processing that history, and turn N+1 can simply resume from it rather
// than recomputing it from scratch.
//
// This section is original: the source material for this chapter
// describes prefix caching only in prose, with a single narrative
// "speedup: 501x" figure and no accompanying tested code anywhere. This
// book's standing rule is to never carry forward a performance number
// that was not itself computed and checked, so rather than repeat that
// figure, this file builds a real prefix cache on top of Chapter 13.2's
// BlockManager/Sequence and honestly measures its own savings.
//
// The mechanism: keep the previous turn's block table intact instead of
// releasing it, compute the longest common prefix (LCP) between the old
// and new token-ID sequences, and reuse every block that lies ENTIRELY
// within that shared prefix. A block is the smallest unit PagedAttention
// can reuse, so a block that straddles the divergence point -- partly
// inside the shared prefix, partly past it -- must be recomputed in
// full even though part of its contents were technically shared. That
// block-granularity waste is a genuine, permanent property of the
// design, not a bug to be optimized away, and Test 2 below demonstrates
// it directly.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_prefix_caching.cpp -o 03_prefix_caching

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <stdexcept>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(expr) do { \
    g_tests++; \
    if (expr) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #expr << "\n"; } \
} while (0)

// ---- Reused verbatim from Chapter 13.2 (BlockManager / KVBlock / Sequence) ----

constexpr int BLOCK_SIZE = 16;
constexpr int HEAD_DIM = 32;
constexpr int NUM_HEADS = 1;

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

    void free(int id) {
        assert(id >= 0 && id < static_cast<int>(m_pool.size()));
        m_free.push_back(id);
    }

    KVBlock* get(int id) {
        assert(id >= 0 && id < static_cast<int>(m_pool.size()));
        return &m_pool[id];
    }
    const KVBlock* get(int id) const { return &m_pool[id]; }

    int free_count() const { return static_cast<int>(m_free.size()); }

private:
    std::vector<KVBlock> m_pool;
    std::list<int> m_free;
};

struct BlockAddr { int block_table_idx; int slot; };
BlockAddr translate(int token_idx) { return {token_idx >> 4, token_idx & 0x0F}; }

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
        return mgr.get(block_table[addr.block_table_idx])->keys[head][addr.slot];
    }
    const float* read_val(const BlockManager& mgr, int token_idx, int head = 0) const {
        auto addr = translate(token_idx);
        return mgr.get(block_table[addr.block_table_idx])->values[head][addr.slot];
    }

    int num_blocks() const { return static_cast<int>(block_table.size()); }
};

// ---- New for this section: prefix matching and block-granularity reuse ----

// Compares two token-ID sequences element by element and returns how many
// leading tokens match. This is the recomputation boundary: everything
// before it is shared history, everything from it onward is new or
// changed and must be (re)computed.
int longest_common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
    int n = static_cast<int>(std::min(a.size(), b.size()));
    int i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

// A deterministic, distinguishable stand-in for "the model computed a
// real K/V vector for this token." Different token IDs always produce
// different vectors, so any test comparing vectors is really comparing
// which token's KV data ended up where.
void gen_kv(int token_id, float* k, float* v) {
    for (int d = 0; d < HEAD_DIM; ++d) {
        k[d] = static_cast<float>(token_id) * 1000.0f + static_cast<float>(d);
        v[d] = static_cast<float>(token_id) * 2000.0f + static_cast<float>(d);
    }
}

// One conversation's cache: the token IDs processed so far, and the
// paged Sequence holding their KV entries. apply_new_turn is where
// prefix caching actually happens.
struct ConversationCache {
    std::vector<int> token_ids;
    Sequence seq;

    struct TurnStats {
        int lcp = 0;                 // tokens that literally matched, element by element
        int reusable_blocks = 0;     // whole blocks entirely inside the LCP
        int reused_tokens = 0;       // reusable_blocks * BLOCK_SIZE
        int recomputed_tokens = 0;   // new_ids.size() - reused_tokens
        int blocks_freed = 0;        // stale blocks released back to the pool
    };

    // Given the FULL new token-ID sequence for this turn (old history plus
    // whatever changed or was appended), reuse every block that lies
    // entirely within the shared prefix and recompute the rest.
    TurnStats apply_new_turn(BlockManager& mgr, const std::vector<int>& new_ids) {
        TurnStats stats;
        stats.lcp = longest_common_prefix(token_ids, new_ids);
        stats.reusable_blocks = stats.lcp / BLOCK_SIZE;  // floor: partial blocks can't be reused
        stats.reused_tokens = stats.reusable_blocks * BLOCK_SIZE;

        // Free every block from the reuse boundary onward -- including a
        // block that straddled the divergence point, since a KVBlock is
        // the smallest unit of reuse and cannot be reused partially.
        for (int i = stats.reusable_blocks; i < seq.num_blocks(); ++i) {
            mgr.free(seq.block_table[i]);
            ++stats.blocks_freed;
        }
        seq.block_table.resize(stats.reusable_blocks);
        seq.context_len = stats.reused_tokens;
        token_ids.resize(stats.reused_tokens);

        // Recompute everything from the reuse boundary to the end of the
        // new sequence. In a real engine this is a forward pass through
        // the model; here gen_kv() stands in for "the model computed it."
        for (int t = stats.reused_tokens; t < static_cast<int>(new_ids.size()); ++t) {
            float k[HEAD_DIM], v[HEAD_DIM];
            gen_kv(new_ids[t], k, v);
            seq.store_kv(mgr, k, v);
            token_ids.push_back(new_ids[t]);
        }
        stats.recomputed_tokens = static_cast<int>(new_ids.size()) - stats.reused_tokens;
        return stats;
    }
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 14.3: Prefix Caching for Multi-Turn Conversations\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: block-aligned full-prefix reuse, zero waste --\n";
    {
        BlockManager mgr(64);
        ConversationCache conv;
        std::vector<int> turn1;
        for (int i = 0; i < 48; ++i) turn1.push_back(100 + i);  // 48 = exactly 3 blocks

        // Bootstrap turn 1 as if it were itself a "new turn" against an
        // empty history: everything is recomputed, nothing reused yet.
        auto boot = conv.apply_new_turn(mgr, turn1);
        CHECK(boot.reused_tokens == 0);
        CHECK(boot.recomputed_tokens == 48);
        std::vector<int> old_block_table = conv.seq.block_table;

        std::vector<int> turn2 = turn1;
        for (int i = 0; i < 20; ++i) turn2.push_back(500 + i);  // append 20 new tokens

        auto stats = conv.apply_new_turn(mgr, turn2);
        CHECK(stats.lcp == 48);
        CHECK(stats.reusable_blocks == 3);
        CHECK(stats.reused_tokens == 48);
        CHECK(stats.recomputed_tokens == 20);
        CHECK(stats.blocks_freed == 0);  // nothing straddled the boundary -- nothing to discard
        bool same_blocks = std::equal(old_block_table.begin(), old_block_table.end(),
                                       conv.seq.block_table.begin());
        CHECK(same_blocks);
        double naive_cost = static_cast<double>(turn2.size());
        double actual_cost = static_cast<double>(stats.recomputed_tokens);
        double speedup = naive_cost / actual_cost;
        std::cout << "  turn1=48 tokens (3 blocks), turn2 appends 20 tokens\n";
        std::cout << "  reused=" << stats.reused_tokens << " recomputed=" << stats.recomputed_tokens
                   << " blocks_freed=" << stats.blocks_freed << "\n";
        std::cout << "  physical block IDs for the reused prefix are unchanged (zero-copy)\n";
        std::cout << "  naive full recompute: " << static_cast<int>(naive_cost)
                   << " tokens; with prefix caching: " << static_cast<int>(actual_cost)
                   << " tokens (" << std::fixed << std::setprecision(2) << speedup << "x)\n";
    }

    std::cout << "\n-- Test 2: [COMMON TRAP] block-granularity waste on a misaligned edit --\n";
    {
        BlockManager mgr(64);
        ConversationCache conv2;
        std::vector<int> turn1;
        for (int i = 0; i < 40; ++i) turn1.push_back(i);  // ids 0..39, spans blocks 0,1,2
        conv2.apply_new_turn(mgr, turn1);

        // The user edits their message: everything up to token 25 is
        // identical, but token 25 onward is different content, and the
        // new turn is 45 tokens long overall.
        std::vector<int> turn2(turn1.begin(), turn1.begin() + 25);
        for (int i = 0; i < 20; ++i) turn2.push_back(900 + i);

        auto stats2 = conv2.apply_new_turn(mgr, turn2);
        CHECK(stats2.lcp == 25);               // 25 tokens genuinely matched
        CHECK(stats2.reusable_blocks == 1);    // but block 1 (tokens 16-31) is NOT entirely inside [0,25)
        CHECK(stats2.reused_tokens == 16);     // so only block 0 (tokens 0-15) is reused
        int wasted_shared_tokens = stats2.lcp - stats2.reused_tokens;
        CHECK(wasted_shared_tokens == 9);      // tokens 16-24 matched but got recomputed anyway
        CHECK(stats2.recomputed_tokens == 29); // 45 - 16
        CHECK(stats2.blocks_freed == 2);       // the straddling block and the fully-stale block 2

        std::cout << "  turn1=40 tokens, turn2 diverges at token 25, new length=45\n";
        std::cout << "  matched prefix (LCP) = " << stats2.lcp << " tokens\n";
        std::cout << "  actually reusable = " << stats2.reused_tokens << " tokens ("
                   << stats2.reusable_blocks << " whole block)\n";
        std::cout << "  wasted shared tokens = " << wasted_shared_tokens
                   << " (matched, but recomputed anyway because their block wasn't fully shared)\n";
    }

    std::cout << "\n-- Test 3: reused blocks' KV data matches the original, byte for byte --\n";
    {
        // Re-run test 2's exact scenario and directly compare the raw
        // bytes of the surviving block against what turn 1 wrote there.
        BlockManager mgr(64);
        ConversationCache conv;
        std::vector<int> turn1;
        for (int i = 0; i < 40; ++i) turn1.push_back(i);
        conv.apply_new_turn(mgr, turn1);
        int surviving_block_id = conv.seq.block_table[0];
        KVBlock snapshot = *mgr.get(surviving_block_id);

        std::vector<int> turn2(turn1.begin(), turn1.begin() + 25);
        for (int i = 0; i < 20; ++i) turn2.push_back(900 + i);
        conv.apply_new_turn(mgr, turn2);

        CHECK(conv.seq.block_table[0] == surviving_block_id);  // same physical block, never touched
        KVBlock after = *mgr.get(surviving_block_id);
        bool byte_identical = std::memcmp(&snapshot, &after, sizeof(KVBlock)) == 0;
        CHECK(byte_identical);
        std::cout << "  block " << surviving_block_id
                   << " untouched by apply_new_turn(): memcmp of all "
                   << sizeof(KVBlock) << " bytes is identical\n";
    }

    std::cout << "\n-- Test 4: the recomputed suffix is correct, not leftover stale data --\n";
    {
        // Rebuild Test 2's exact scenario (its own BlockManager already
        // went out of scope) and check the recomputed region directly.
        BlockManager mgr(64);
        ConversationCache conv;
        std::vector<int> turn1;
        for (int i = 0; i < 40; ++i) turn1.push_back(i);
        conv.apply_new_turn(mgr, turn1);
        std::vector<int> turn2(turn1.begin(), turn1.begin() + 25);
        for (int i = 0; i < 20; ++i) turn2.push_back(900 + i);
        conv.apply_new_turn(mgr, turn2);

        bool suffix_ok = true;
        for (int t = conv.seq.context_len - static_cast<int>(20); t < conv.seq.context_len; ++t) {
            float expect_k[HEAD_DIM], expect_v[HEAD_DIM];
            gen_kv(turn2[t], expect_k, expect_v);
            const float* got_k = conv.seq.read_key(mgr, t);
            const float* got_v = conv.seq.read_val(mgr, t);
            for (int d = 0; d < HEAD_DIM; ++d) {
                if (got_k[d] != expect_k[d] || got_v[d] != expect_v[d]) suffix_ok = false;
            }
        }
        CHECK(suffix_ok);
        // And the reused prefix (tokens 0-15) still reads back correctly too.
        bool prefix_ok = true;
        for (int t = 0; t < 16; ++t) {
            float expect_k[HEAD_DIM], expect_v[HEAD_DIM];
            gen_kv(turn1[t], expect_k, expect_v);
            const float* got_k = conv.seq.read_key(mgr, t);
            for (int d = 0; d < HEAD_DIM; ++d) if (got_k[d] != expect_k[d]) prefix_ok = false;
        }
        CHECK(prefix_ok);
        std::cout << "  recomputed suffix (last 20 tokens) matches the NEW token IDs' KV data\n";
        std::cout << "  reused prefix (first 16 tokens) still matches the ORIGINAL token IDs' KV data\n";
    }

    std::cout << "\n-- Test 5: realistic long-history conversation, honestly-computed savings --\n";
    {
        BlockManager mgr(512);
        ConversationCache conv;
        std::vector<int> history;
        for (int i = 0; i < 2048; ++i) history.push_back(i);  // system prompt + prior turns
        conv.apply_new_turn(mgr, history);

        // The next turn keeps the entire history unchanged and appends 48
        // new tokens -- the common case of a straightforward reply.
        std::vector<int> next_turn = history;
        for (int i = 0; i < 48; ++i) next_turn.push_back(90000 + i);

        auto stats = conv.apply_new_turn(mgr, next_turn);
        CHECK(stats.reused_tokens == 2048);   // 2048 is block-aligned (2048 / 16 = 128 exactly)
        CHECK(stats.recomputed_tokens == 48);
        double total_len = static_cast<double>(next_turn.size());
        double saved_tokens = total_len - stats.recomputed_tokens;
        double savings_pct = 100.0 * saved_tokens / total_len;
        std::cout << "  history=2048 tokens, next turn appends 48 tokens (total " << next_turn.size() << ")\n";
        std::cout << "  tokens recomputed: " << stats.recomputed_tokens << " / " << next_turn.size() << "\n";
        std::cout << "  tokens saved by prefix caching: " << std::fixed << std::setprecision(1)
                   << saved_tokens << " (" << savings_pct << "% of this turn's total work avoided)\n";
        CHECK(savings_pct > 90.0);
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
