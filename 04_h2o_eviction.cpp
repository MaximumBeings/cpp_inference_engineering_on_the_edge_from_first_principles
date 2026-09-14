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
