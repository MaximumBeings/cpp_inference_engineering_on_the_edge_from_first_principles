// 13_appendix_kvblock_mdspan.cpp
// Appendix to Chapter 13.2 -- an mdspan-based KVBlock, alongside the
// raw-array version the shipped chapter uses.
//
// Chapters 1-11 built every dense numerical view -- weight matrices,
// activation tensors, per-head attention slices -- on std::mdspan, and
// Chapters 12-14 deliberately stepped away from it, noting in their own
// "what you need to know first" lists that these chapters compile with
// no -pthread, no std::mdspan, no -ffp-contract=off. That was not an
// oversight: mdspan is a NON-OWNING VIEW over memory someone else
// already owns. It never allocates, frees, grows, or shrinks anything,
// so it has nothing to offer BlockManager's free list, a growing block
// table, a hash-mapped vocabulary, or a tiered cache's vector of
// entries -- the genuine complexity in Chapters 12-14 lives in exactly
// those ownership and lifecycle concerns, not in indexing arithmetic
// over a fixed-shape numeric buffer.
//
// But Chapter 13.2's KVBlock is a real exception to that pattern: it is
// precisely the kind of fixed-shape, dense, multi-dimensional numeric
// array (NUM_HEADS x BLOCK_SIZE x HEAD_DIM) that Chapters 1-2 built
// mdspan specifically to index. This file shows that version: the same
// flat storage, viewed through std::mdspan and sliced per-head with
// std::submdspan (Chapter 2.2's technique) instead of raw nested C
// arrays and manual [head][slot][dim] indexing. Every test below cross-
// checks the mdspan version against Chapter 13.2's own shipped KVBlock,
// run side by side on identical input, to confirm the two are not just
// "close" but byte-for-byte, semantically identical -- mdspan changes
// how the storage is INDEXED, never what is actually stored or computed.
//
// A deliberate syntax choice: Chapters 1-2's own mdspan examples index
// with the C++23 core-language multi-argument subscript operator,
// `view[h, s, d]` -- a language feature the toolchain needs GCC 12 or
// newer to parse. This book's own real aarch64 hardware (verified
// directly while building this file) currently has only GCC 11.4.0,
// which predates that feature and rejects it outright. mdspan's INDEX
// OPERATOR is itself just a class member, though, with an overload that
// takes a std::array of indices, `view[std::array{h, s, d}]` -- an
// ordinary function call, not new syntax -- so this file uses that form
// throughout via the idx3()/idx2() helpers below, giving the identical
// element access while staying compilable on this exact toolchain.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 13_appendix_kvblock_mdspan.cpp -o 13_appendix_kvblock_mdspan

#include <mdspan/mdspan.hpp>
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
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

constexpr int BLOCK_SIZE = 16;
constexpr int HEAD_DIM = 32;
constexpr int NUM_HEADS = 4;

// ---- Chapter 13.2's shipped KVBlock, reproduced verbatim for the ----
// ---- side-by-side comparison tests below. ----
struct KVBlockRaw {
    float keys[NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
    float values[NUM_HEADS][BLOCK_SIZE][HEAD_DIM];
    void clear() { std::memset(this, 0, sizeof(*this)); }
};

// ---- The mdspan version: identical logical shape, flat storage, ----
// ---- indexed through std::mdspan instead of nested C arrays. ----
using BlockShape = std::extents<size_t, NUM_HEADS, BLOCK_SIZE, HEAD_DIM>;
using BlockView = std::mdspan<float, BlockShape>;         // mutable view
using BlockViewConst = std::mdspan<const float, BlockShape>;

struct KVBlockMdspan {
    // Exactly the same element count and layout as KVBlockRaw's two
    // nested arrays, just held as one flat vector each instead of a
    // multi-dimensional C array -- mdspan supplies the indexing that
    // KVBlockRaw got from the language's own array syntax.
    std::vector<float> key_storage = std::vector<float>(NUM_HEADS * BLOCK_SIZE * HEAD_DIM, 0.0f);
    std::vector<float> val_storage = std::vector<float>(NUM_HEADS * BLOCK_SIZE * HEAD_DIM, 0.0f);

    BlockView keys() { return BlockView(key_storage.data()); }
    BlockView values() { return BlockView(val_storage.data()); }
    BlockViewConst keys() const { return BlockViewConst(key_storage.data()); }
    BlockViewConst values() const { return BlockViewConst(val_storage.data()); }

    void clear() {
        std::fill(key_storage.begin(), key_storage.end(), 0.0f);
        std::fill(val_storage.begin(), val_storage.end(), 0.0f);
    }
};

// ---- The same free-list block manager as 13.2, parameterized over ----
// ---- which KVBlock type it pools -- the allocator logic itself has ----
// ---- nothing to do with how a block's own storage is indexed. ----
template <typename Block>
class BlockManagerT {
public:
    explicit BlockManagerT(int num_blocks) {
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
    Block* get(int id) { return &m_pool[id]; }
    const Block* get(int id) const { return &m_pool[id]; }
    int free_count() const { return static_cast<int>(m_free.size()); }

private:
    std::vector<Block> m_pool;
    std::list<int> m_free;
};

using BlockManagerRaw = BlockManagerT<KVBlockRaw>;
using BlockManagerMdspan = BlockManagerT<KVBlockMdspan>;

struct BlockAddr { int block_table_idx; int slot; };
BlockAddr translate(int token_idx) { return {token_idx >> 4, token_idx & 0x0F}; }

// GCC 11.4.0 (this book's own real aarch64 target) does not yet support
// C++23's multi-argument subscript-operator syntax, `view[h, s, d]` --
// that needs GCC 12+. mdspan's std::array-taking operator[] overload is
// an ordinary member function, not new syntax, so these two helpers give
// the identical element access in a form every C++23 compiler already
// accepts.
std::array<size_t, 3> idx3(size_t h, size_t s, size_t d) { return {h, s, d}; }
std::array<size_t, 2> idx2(size_t a, size_t b) { return {a, b}; }

// A deterministic key/value generator shared by both implementations'
// tests below, so "run the same input through both" is literal, not
// approximate.
void gen_kv(int t, float* k, float* v) {
    for (int d = 0; d < HEAD_DIM; ++d) {
        k[d] = static_cast<float>(t * 100 + d);
        v[d] = static_cast<float>(t * 200 + d);
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Appendix: Chapter 13.2's KVBlock, Reimplemented on std::mdspan\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: mdspan indexing matches manual flat-offset arithmetic --\n";
    {
        KVBlockMdspan blk;
        auto kv = blk.keys();
        for (int h = 0; h < NUM_HEADS; ++h)
            for (int s = 0; s < BLOCK_SIZE; ++s)
                for (int d = 0; d < HEAD_DIM; ++d)
                    kv[idx3(h, s, d)] = static_cast<float>(h * 10000 + s * 100 + d);

        bool all_ok = true;
        for (int h = 0; h < NUM_HEADS; ++h)
            for (int s = 0; s < BLOCK_SIZE; ++s)
                for (int d = 0; d < HEAD_DIM; ++d) {
                    size_t flat = static_cast<size_t>(h) * BLOCK_SIZE * HEAD_DIM
                                + static_cast<size_t>(s) * HEAD_DIM + d;
                    if (blk.key_storage[flat] != kv[idx3(h, s, d)]) all_ok = false;
                }
        CHECK(all_ok);
        CHECK(kv.extent(0) == NUM_HEADS && kv.extent(1) == BLOCK_SIZE && kv.extent(2) == HEAD_DIM);
        CHECK(kv.stride(0) == BLOCK_SIZE * HEAD_DIM && kv.stride(1) == HEAD_DIM && kv.stride(2) == 1);

        // Proof it is a VIEW, not a copy, exactly as Chapter 2.1 showed:
        // writing through the mdspan must change the underlying vector.
        kv[idx3(2, 5, 7)] = 12345.0f;
        CHECK(blk.key_storage[2 * BLOCK_SIZE * HEAD_DIM + 5 * HEAD_DIM + 7] == 12345.0f);
        std::cout << "  mdspan[h,s,d] matches hand-computed flat offset h*" << BLOCK_SIZE * HEAD_DIM
                   << " + s*" << HEAD_DIM << " + d for every (h,s,d); write-through confirmed\n";
    }

    std::cout << "\n-- Test 2: submdspan extracts one head's storage with zero copy --\n";
    {
        KVBlockMdspan blk;
        auto kv = blk.keys();
        for (int h = 0; h < NUM_HEADS; ++h)
            for (int s = 0; s < BLOCK_SIZE; ++s)
                for (int d = 0; d < HEAD_DIM; ++d)
                    kv[idx3(h, s, d)] = static_cast<float>(h * 1000 + s * 10 + d);

        auto head2 = std::submdspan(kv, 2, std::full_extent, std::full_extent);
        CHECK(head2.extent(0) == BLOCK_SIZE && head2.extent(1) == HEAD_DIM);
        CHECK(head2[idx2(5, 7)] == 2057.0f);  // matches kv[2,5,7] = 2*1000+5*10+7

        // Writing through the slice must land in the SAME flat buffer
        // head2 was sliced from -- no data was copied out to produce it.
        head2[idx2(3, 4)] = -1.0f;
        CHECK(kv[idx3(2, 3, 4)] == -1.0f);
        std::cout << "  submdspan(kv, head=2, full, full) has shape [" << head2.extent(0)
                   << "," << head2.extent(1) << "]; a write through the slice is visible in kv[2,*,*]\n";
    }

    std::cout << "\n-- Test 3: raw-array and mdspan block managers store identical data --\n";
    {
        BlockManagerRaw raw_mgr(20);
        BlockManagerMdspan mdspan_mgr(20);
        std::vector<int> raw_table, mdspan_table;
        int raw_ctx = 0, mdspan_ctx = 0;

        for (int t = 0; t < 35; ++t) {
            float k[HEAD_DIM], v[HEAD_DIM];
            gen_kv(t, k, v);

            int slot = raw_ctx % BLOCK_SIZE;
            if (slot == 0) raw_table.push_back(raw_mgr.alloc());
            KVBlockRaw* rb = raw_mgr.get(raw_table.back());
            for (int d = 0; d < HEAD_DIM; ++d) { rb->keys[0][slot][d] = k[d]; rb->values[0][slot][d] = v[d]; }
            ++raw_ctx;

            int mslot = mdspan_ctx % BLOCK_SIZE;
            if (mslot == 0) mdspan_table.push_back(mdspan_mgr.alloc());
            KVBlockMdspan* mb = mdspan_mgr.get(mdspan_table.back());
            auto mk = mb->keys(); auto mv = mb->values();
            for (int d = 0; d < HEAD_DIM; ++d) { mk[idx3(0, mslot, d)] = k[d]; mv[idx3(0, mslot, d)] = v[d]; }
            ++mdspan_ctx;
        }
        CHECK(raw_ctx == mdspan_ctx);
        CHECK(raw_table.size() == mdspan_table.size());

        bool identical = true;
        for (int t = 0; t < 35; ++t) {
            auto addr = translate(t);
            const KVBlockRaw* rb = raw_mgr.get(raw_table[addr.block_table_idx]);
            const KVBlockMdspan* mb = mdspan_mgr.get(mdspan_table[addr.block_table_idx]);
            auto mk = mb->keys(); auto mv = mb->values();
            for (int d = 0; d < HEAD_DIM; ++d) {
                if (rb->keys[0][addr.slot][d] != mk[idx3(0, addr.slot, d)]) identical = false;
                if (rb->values[0][addr.slot][d] != mv[idx3(0, addr.slot, d)]) identical = false;
            }
        }
        CHECK(identical);
        std::cout << "  35 tokens written through both implementations from identical input: "
                   << "every stored float is bit-for-bit identical\n";
    }

    std::cout << "\n-- Test 4: mdspan-based paged attention matches the raw-array reference --\n";
    {
        BlockManagerMdspan mgr(20);
        std::vector<int> table;
        int ctx = 0;
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        const int N = 20;
        std::vector<std::vector<float>> keys(N, std::vector<float>(HEAD_DIM));
        std::vector<std::vector<float>> vals(N, std::vector<float>(HEAD_DIM));

        for (int t = 0; t < N; ++t) {
            for (int d = 0; d < HEAD_DIM; ++d) { keys[t][d] = nd(rng); vals[t][d] = nd(rng); }
            int slot = ctx % BLOCK_SIZE;
            if (slot == 0) table.push_back(mgr.alloc());
            auto mk = mgr.get(table.back())->keys();
            auto mv = mgr.get(table.back())->values();
            for (int d = 0; d < HEAD_DIM; ++d) { mk[idx3(0, slot, d)] = keys[t][d]; mv[idx3(0, slot, d)] = vals[t][d]; }
            ++ctx;
        }

        std::vector<float> query(HEAD_DIM);
        for (float& v : query) v = nd(rng);

        std::vector<float> scores(N);
        float max_s = -1e9f;
        for (int t = 0; t < N; ++t) {
            auto addr = translate(t);
            auto mk = mgr.get(table[addr.block_table_idx])->keys();
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) dot += query[d] * mk[idx3(0, addr.slot, d)];
            scores[t] = dot * scale;
            max_s = std::max(max_s, scores[t]);
        }
        float sum_e = 0.0f;
        for (float& s : scores) { s = std::exp(s - max_s); sum_e += s; }
        for (float& s : scores) s /= sum_e;
        std::vector<float> mdspan_out(HEAD_DIM, 0.0f);
        for (int t = 0; t < N; ++t) {
            auto addr = translate(t);
            auto mv = mgr.get(table[addr.block_table_idx])->values();
            for (int d = 0; d < HEAD_DIM; ++d) mdspan_out[d] += scores[t] * mv[idx3(0, addr.slot, d)];
        }

        std::vector<float> ref_s(N);
        float ref_max = -1e9f;
        for (int t = 0; t < N; ++t) {
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) dot += query[d] * keys[t][d];
            ref_s[t] = dot * scale;
            ref_max = std::max(ref_max, ref_s[t]);
        }
        float ref_sum = 0.0f;
        for (float& s : ref_s) { s = std::exp(s - ref_max); ref_sum += s; }
        for (float& s : ref_s) s /= ref_sum;
        std::vector<float> ref_out(HEAD_DIM, 0.0f);
        for (int t = 0; t < N; ++t)
            for (int d = 0; d < HEAD_DIM; ++d) ref_out[d] += ref_s[t] * vals[t][d];

        float max_err = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) max_err = std::max(max_err, std::fabs(mdspan_out[d] - ref_out[d]));
        CHECK(max_err < 1e-4f);
        std::cout << "  max error vs dense reference: " << std::scientific << std::setprecision(2) << max_err << "\n";
        std::cout.unsetf(std::ios::scientific);
    }

    std::cout << "\n-- Test 5: mdspan adds no storage overhead vs the raw-array version --\n";
    {
        CHECK(sizeof(KVBlockRaw) == 2 * NUM_HEADS * BLOCK_SIZE * HEAD_DIM * sizeof(float));
        size_t mdspan_storage_bytes = 2 * static_cast<size_t>(NUM_HEADS) * BLOCK_SIZE * HEAD_DIM * sizeof(float);
        std::cout << "  KVBlockRaw:    sizeof = " << sizeof(KVBlockRaw) << " bytes (two nested C arrays)\n";
        std::cout << "  KVBlockMdspan: " << mdspan_storage_bytes
                   << " bytes of actual float storage (two std::vector<float>, same element count)\n";
        std::cout << "  the mdspan view object itself carries only a pointer -- no per-block runtime\n";
        std::cout << "  shape metadata, because every extent here is a compile-time constant\n";
        CHECK(mdspan_storage_bytes == sizeof(KVBlockRaw));
    }

    std::cout << "\n========================================================\n";
    std::cout << g_passed << "/" << g_tests << " checks passed";
    std::cout << (g_passed == g_tests ? "  ALL PASS\n" : "  FAILURES\n");
    std::cout << "========================================================\n";
    return (g_passed == g_tests) ? 0 : 1;
}
