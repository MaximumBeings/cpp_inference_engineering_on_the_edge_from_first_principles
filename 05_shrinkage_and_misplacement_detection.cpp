// Chapter 19.5 -- This section closes the loop this chapter opened:
// Section 19.1's planogram states what SHOULD be on a shelf, Section
// 19.2's batch processor and its structured-output parser produce what
// a VLM actually OBSERVED there, and this section reconciles the two
// into the two real findings a store operator actually acts on --
// MISPLACEMENT and SHRINKAGE -- which are genuinely different claims
// requiring genuinely different evidence.
//
// A misplacement is immediate: if a slot's own detection names a
// DIFFERENT sku than the planogram expects, that is a fact about THIS
// ONE photo, true the moment it is observed, and needs no corroboration
// from any other photo to report. A missing item is not the same kind
// of claim. A single photo showing an empty slot could mean real
// shrinkage, or it could mean a customer is mid-reach, a restocking cart
// is blocking the shelf, or the photo caught a brief, ordinary moment of
// bare shelf between two deliveries -- and reporting a shrinkage alert
// off ONE empty-looking photo would be exactly the kind of overconfident
// claim this book has refused to make since Chapter 15's own honest
// treatment of numerical precision. `ShrinkageDetector` tracks, per
// slot, how many CONSECUTIVE reconciliations in a row have found it
// empty, resets that count the instant the item reappears, and only
// raises SHRINKAGE_SUSPECTED once a stated threshold of consecutive
// misses is crossed -- corroboration over time standing in for the
// single asymmetric-confidence-threshold reasoning Section 18.4 already
// applied to a single photo's own disposition, now applied across a
// SEQUENCE of photos instead of a single one.
//
// This section's own scope ends at producing that finding. A real
// deployment would report each SHRINKAGE_SUSPECTED slot by calling
// Section 19.3's own `create_shrinkage_ticket` against the real ERP/WMS
// backend -- this section does not repeat that REST machinery here,
// since nothing about ITS OWN real subject, reconciling a planogram
// against detections gathered over time, depends on it.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 05_shrinkage_and_misplacement_detection.cpp -o 05_shrinkage_and_misplacement_detection
// Run:     ./05_shrinkage_and_misplacement_detection

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the planogram slot and parsed-detection shapes, exactly
// Sections 19.1 and 19.2's own structures, repeated here self-contained.
// =======================================================================
struct ShelfSlot {
    int shelf_index = 0;
    int position_index = 0;
    std::string sku_id;
    std::string product_name;
    int expected_facings = 0;
};

struct Planogram {
    std::string store_layout_id;
    std::vector<ShelfSlot> slots;
};

struct ParsedDetection {
    std::string sku_id;
    int shelf_index = 0;
    int position_index = 0;
    double confidence = 0.0;
};

struct SlotKey {
    int shelf_index;
    int position_index;
    bool operator==(const SlotKey& o) const { return shelf_index == o.shelf_index && position_index == o.position_index; }
};
struct SlotKeyHash {
    size_t operator()(const SlotKey& k) const {
        return std::hash<int>{}(k.shelf_index) * 31 + std::hash<int>{}(k.position_index);
    }
};

// =======================================================================
// PART 2: reconciliation -- immediate misplacement, and shrinkage that
// requires corroboration across consecutive reconciliations.
// =======================================================================
enum class SlotStatus { OK, MISPLACED, EMPTY_OBSERVED, SHRINKAGE_SUSPECTED };

std::string to_string(SlotStatus s) {
    switch (s) {
        case SlotStatus::OK: return "OK";
        case SlotStatus::MISPLACED: return "MISPLACED";
        case SlotStatus::EMPTY_OBSERVED: return "EMPTY_OBSERVED";
        case SlotStatus::SHRINKAGE_SUSPECTED: return "SHRINKAGE_SUSPECTED";
    }
    return "UNKNOWN";
}

struct SlotFinding {
    int shelf_index = 0;
    int position_index = 0;
    std::string expected_sku;
    std::string observed_sku;   // empty when no detection landed on this slot at all
    SlotStatus status = SlotStatus::OK;
};

class ShrinkageDetector {
public:
    explicit ShrinkageDetector(int consecutive_empty_threshold) : threshold_(consecutive_empty_threshold) {}

    // Reconciles ONE period's worth of detections against the planogram.
    // Must be called once per period, in chronological order, for the
    // per-slot consecutive-empty counters to mean anything real -- this
    // is real, caller-managed state, not a pure function, because
    // shrinkage is fundamentally a claim about a SEQUENCE of
    // observations, not any single one.
    std::vector<SlotFinding> reconcile(const Planogram& pg, const std::vector<ParsedDetection>& detections) {
        std::unordered_map<SlotKey, ParsedDetection, SlotKeyHash> by_slot;
        for (const auto& d : detections) {
            by_slot[SlotKey{d.shelf_index, d.position_index}] = d;
        }

        std::vector<SlotFinding> findings;
        findings.reserve(pg.slots.size());
        for (const auto& slot : pg.slots) {
            SlotKey key{slot.shelf_index, slot.position_index};
            SlotFinding f;
            f.shelf_index = slot.shelf_index;
            f.position_index = slot.position_index;
            f.expected_sku = slot.sku_id;

            auto it = by_slot.find(key);
            if (it == by_slot.end()) {
                int& streak = empty_streak_[key];
                ++streak;
                f.observed_sku = "";
                f.status = (streak >= threshold_) ? SlotStatus::SHRINKAGE_SUSPECTED : SlotStatus::EMPTY_OBSERVED;
            } else {
                empty_streak_[key] = 0;   // the item is visibly present again -- the streak resets, it does not merely pause
                f.observed_sku = it->second.sku_id;
                f.status = (it->second.sku_id == slot.sku_id) ? SlotStatus::OK : SlotStatus::MISPLACED;
            }
            findings.push_back(f);
        }
        return findings;
    }

    int current_streak(int shelf_index, int position_index) const {
        auto it = empty_streak_.find(SlotKey{shelf_index, position_index});
        return (it == empty_streak_.end()) ? 0 : it->second;
    }

private:
    int threshold_;
    std::unordered_map<SlotKey, int, SlotKeyHash> empty_streak_;
};

// A real reconciliation report summarizing a full multi-period run: the
// distinct slots that were EVER flagged SHRINKAGE_SUSPECTED across the
// run, and the total count of individual MISPLACED events (a slot
// misplaced in three different periods counts as three real events,
// since each is an independent observation a store associate would have
// had three separate chances to correct).
struct ReconciliationReport {
    std::vector<SlotKey> shrinkage_suspected_slots;
    int misplaced_event_count = 0;
};

ReconciliationReport summarize(const std::vector<std::vector<SlotFinding>>& all_periods) {
    ReconciliationReport report;
    std::unordered_map<SlotKey, bool, SlotKeyHash> seen_shrinkage;
    for (const auto& period : all_periods) {
        for (const auto& f : period) {
            if (f.status == SlotStatus::SHRINKAGE_SUSPECTED) {
                SlotKey key{f.shelf_index, f.position_index};
                if (!seen_shrinkage[key]) {
                    seen_shrinkage[key] = true;
                    report.shrinkage_suspected_slots.push_back(key);
                }
            } else if (f.status == SlotStatus::MISPLACED) {
                ++report.misplaced_event_count;
            }
        }
    }
    return report;
}

// =======================================================================
// PART 3: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 19.5: Shrinkage and Misplacement Detection from Ordinary Shelf Photographs\n";
    std::cout << "========================================================\n";

    Planogram pg;
    pg.store_layout_id = "STORE-777";
    pg.slots = {
        {1, 1, "SKU-A", "Item A", 3},
        {1, 2, "SKU-B", "Item B", 2},
    };

    std::cout << "\n-- Test 1: a misplacement is flagged immediately, on the very first observation --\n";
    {
        ShrinkageDetector det(3);
        std::vector<ParsedDetection> obs = {
            {"SKU-A", 1, 1, 0.9},
            {"SKU-WRONG", 1, 2, 0.95},   // slot (1,2) expects SKU-B
        };
        auto findings = det.reconcile(pg, obs);
        CHECK(findings.size() == 2);
        CHECK(findings[0].status == SlotStatus::OK);
        CHECK(findings[1].status == SlotStatus::MISPLACED);
        CHECK(findings[1].observed_sku == "SKU-WRONG" && findings[1].expected_sku == "SKU-B");
        std::cout << "  slot (1,1) matches its expected SKU-A: OK; slot (1,2) shows SKU-WRONG instead "
                     "of the expected SKU-B: flagged MISPLACED on the very first photo, no corroboration needed\n";
    }

    std::cout << "\n-- Test 2: a single missing observation is NOT enough to suspect shrinkage --\n";
    {
        ShrinkageDetector det(3);
        std::vector<ParsedDetection> obs = {{"SKU-A", 1, 1, 0.9}};   // slot (1,2) has no detection at all
        auto findings = det.reconcile(pg, obs);
        CHECK(findings[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(findings[1].status != SlotStatus::SHRINKAGE_SUSPECTED);
        CHECK(det.current_streak(1, 2) == 1);
        std::cout << "  slot (1,2) missing on its first observation: correctly EMPTY_OBSERVED, not yet "
                     "SHRINKAGE_SUSPECTED (streak=" << det.current_streak(1, 2) << ", threshold=3)\n";
    }

    std::cout << "\n-- Test 3: three CONSECUTIVE missing observations cross the threshold --\n";
    {
        ShrinkageDetector det(3);
        std::vector<ParsedDetection> present = {{"SKU-A", 1, 1, 0.9}};   // slot (1,2) always missing below
        auto f1 = det.reconcile(pg, present);
        auto f2 = det.reconcile(pg, present);
        auto f3 = det.reconcile(pg, present);
        CHECK(f1[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(f2[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(f3[1].status == SlotStatus::SHRINKAGE_SUSPECTED);
        std::cout << "  slot (1,2) missing across 3 consecutive reconciliations: EMPTY_OBSERVED, "
                     "EMPTY_OBSERVED, then correctly SHRINKAGE_SUSPECTED on crossing the threshold "
                     "on the 3rd\n";
    }

    std::cout << "\n-- Test 4: the item reappearing resets the streak -- it does not merely pause it --\n";
    {
        ShrinkageDetector det(3);
        std::vector<ParsedDetection> present_both = {{"SKU-A", 1, 1, 0.9}, {"SKU-B", 1, 2, 0.9}};
        std::vector<ParsedDetection> missing_b = {{"SKU-A", 1, 1, 0.9}};

        det.reconcile(pg, missing_b);   // streak(1,2) = 1
        det.reconcile(pg, missing_b);   // streak(1,2) = 2
        auto reappear = det.reconcile(pg, present_both);   // reappears -- streak resets to 0
        CHECK(reappear[1].status == SlotStatus::OK);
        int streak_after_reset = det.current_streak(1, 2);
        CHECK(streak_after_reset == 0);

        auto after1 = det.reconcile(pg, missing_b);   // streak = 1 again, NOT 3
        auto after2 = det.reconcile(pg, missing_b);   // streak = 2, still below threshold
        CHECK(after1[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(after2[1].status == SlotStatus::EMPTY_OBSERVED);
        CHECK(after2[1].status != SlotStatus::SHRINKAGE_SUSPECTED);
        std::cout << "  after 2 misses, the item reappears (streak resets to " << streak_after_reset
                   << "); two MORE misses after that reappearance land at streak="
                   << det.current_streak(1, 2) << ", correctly still EMPTY_OBSERVED rather than "
                     "treating the two separate miss-runs as one continuous 4-miss shrinkage streak\n";
    }

    std::cout << "\n-- Test 5: a full multi-period reconciliation report is correct and complete --\n";
    {
        ShrinkageDetector det(3);
        std::vector<std::vector<SlotFinding>> all_periods;
        // Period 1: both slots fine.
        all_periods.push_back(det.reconcile(pg, {{"SKU-A", 1, 1, 0.9}, {"SKU-B", 1, 2, 0.9}}));
        // Period 2: slot (1,1) misplaced (a real, immediate event); slot (1,2) fine.
        all_periods.push_back(det.reconcile(pg, {{"SKU-WRONG", 1, 1, 0.9}, {"SKU-B", 1, 2, 0.9}}));
        // Period 3: slot (1,1) back to normal; slot (1,2) goes missing.
        all_periods.push_back(det.reconcile(pg, {{"SKU-A", 1, 1, 0.9}}));
        // Period 4: slot (1,2) still missing.
        all_periods.push_back(det.reconcile(pg, {{"SKU-A", 1, 1, 0.9}}));
        // Period 5: slot (1,2) still missing -- crosses the threshold of 3.
        all_periods.push_back(det.reconcile(pg, {{"SKU-A", 1, 1, 0.9}}));

        auto report = summarize(all_periods);
        CHECK(report.misplaced_event_count == 1);           // exactly the one period-2 event
        CHECK(report.shrinkage_suspected_slots.size() == 1);
        CHECK(report.shrinkage_suspected_slots[0].shelf_index == 1 && report.shrinkage_suspected_slots[0].position_index == 2);

        std::cout << "  across 5 periods: exactly " << report.misplaced_event_count
                   << " misplacement event (period 2's slot (1,1)) and exactly "
                   << report.shrinkage_suspected_slots.size() << " slot ever flagged shrinkage-suspected "
                     "(slot (1,2), after 3 consecutive missing periods 3 through 5) -- a complete, "
                     "accurate report a real store operator could act on directly\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
