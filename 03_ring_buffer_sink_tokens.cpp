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
