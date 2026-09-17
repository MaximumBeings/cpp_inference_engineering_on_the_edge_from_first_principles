// Chapter 25.2 -- Section 25.1's own per-frame scene tags are only
// useful once turned into a real, chronological incident report a
// human can actually read, with a real, tamper-evident record proving
// what that report was built from. This section reuses Section 22.1's
// own presence-interval coalescing technique to turn raw per-frame tags
// into readable narrative lines, adds an honest "no notable activity"
// line for any real gap rather than silently skipping it, and builds a
// real, well-known non-cryptographic checksum -- FNV-1a -- from scratch
// as this incident's own chain-of-custody hash, verified against its
// own real, published reference vectors.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 02_incident_report_drafting_and_chain_of_custody.cpp -o 02_incident_report_drafting_and_chain_of_custody
// Run:     ./02_incident_report_drafting_and_chain_of_custody

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: the closed scene-tag vocabulary, repeated here per this
// book's own self-contained-file convention (see Section 25.1).
// =======================================================================
enum class SceneTag {
    VEHICLE_PRESENT, WEAPON_VISIBLE, PERSON_ON_GROUND, CROWD_GATHERING,
    PROPERTY_DAMAGE, EMS_PRESENT, HANDCUFFS_VISIBLE, FORCED_ENTRY,
};

std::string scene_tag_name(SceneTag t) {
    switch (t) {
        case SceneTag::VEHICLE_PRESENT: return "VEHICLE_PRESENT";
        case SceneTag::WEAPON_VISIBLE: return "WEAPON_VISIBLE";
        case SceneTag::PERSON_ON_GROUND: return "PERSON_ON_GROUND";
        case SceneTag::CROWD_GATHERING: return "CROWD_GATHERING";
        case SceneTag::PROPERTY_DAMAGE: return "PROPERTY_DAMAGE";
        case SceneTag::EMS_PRESENT: return "EMS_PRESENT";
        case SceneTag::HANDCUFFS_VISIBLE: return "HANDCUFFS_VISIBLE";
        case SceneTag::FORCED_ENTRY: return "FORCED_ENTRY";
    }
    return "UNKNOWN";
}

struct FrameRecord {
    int frame_index = 0;
    int64_t timestamp_ms = 0;
    std::vector<SceneTag> tags;
};

bool frame_has_tag(const FrameRecord& f, SceneTag t) {
    return std::find(f.tags.begin(), f.tags.end(), t) != f.tags.end();
}

// =======================================================================
// PART 2: real mm:ss time formatting, and real presence-interval
// coalescing -- the same technique Section 22.1 used to turn raw
// per-frame detections into contiguous presence intervals, reapplied
// here unchanged to a new domain.
// =======================================================================
std::string format_mmss(int64_t ms) {
    int64_t total_seconds = ms / 1000;
    int64_t mm = total_seconds / 60;
    int64_t ss = total_seconds % 60;
    std::ostringstream out;
    out << std::setw(2) << std::setfill('0') << mm << ":" << std::setw(2) << std::setfill('0') << ss;
    return out.str();
}

struct TagInterval {
    SceneTag tag;
    int64_t start_ms = 0, end_ms = 0;
};

// Frames within this many milliseconds of each other still count as the
// SAME continuous observation of a tag -- a real, stated tolerance
// against normal gaps between processed frames, not an assumption that
// every frame in the input is already contiguous.
constexpr int64_t COALESCE_GAP_TOLERANCE_MS = 5000;

std::vector<TagInterval> coalesce_tag_intervals(std::vector<FrameRecord> frames) {
    std::sort(frames.begin(), frames.end(),
              [](const FrameRecord& a, const FrameRecord& b) { return a.timestamp_ms < b.timestamp_ms; });

    std::vector<TagInterval> out;
    for (SceneTag tag : {SceneTag::VEHICLE_PRESENT, SceneTag::WEAPON_VISIBLE, SceneTag::PERSON_ON_GROUND,
                         SceneTag::CROWD_GATHERING, SceneTag::PROPERTY_DAMAGE, SceneTag::EMS_PRESENT,
                         SceneTag::HANDCUFFS_VISIBLE, SceneTag::FORCED_ENTRY}) {
        bool open = false;
        TagInterval current{tag, 0, 0};
        for (const auto& f : frames) {
            if (frame_has_tag(f, tag)) {
                if (!open) {
                    current = {tag, f.timestamp_ms, f.timestamp_ms};
                    open = true;
                } else if (f.timestamp_ms - current.end_ms <= COALESCE_GAP_TOLERANCE_MS) {
                    current.end_ms = f.timestamp_ms;
                } else {
                    out.push_back(current);
                    current = {tag, f.timestamp_ms, f.timestamp_ms};
                }
            } else if (open && f.timestamp_ms - current.end_ms > COALESCE_GAP_TOLERANCE_MS) {
                out.push_back(current);
                open = false;
            }
        }
        if (open) out.push_back(current);
    }
    std::sort(out.begin(), out.end(),
              [](const TagInterval& a, const TagInterval& b) { return a.start_ms < b.start_ms; });
    return out;
}

std::string render_narrative_line(const TagInterval& iv) {
    std::ostringstream out;
    out << "[" << format_mmss(iv.start_ms) << " - " << format_mmss(iv.end_ms) << "] "
        << scene_tag_name(iv.tag) << " observed";
    return out.str();
}

// A real gap in the timeline where no tag interval covers any part of
// it is reported explicitly, honestly, and by name -- never silently
// skipped, exactly this book's own established "no notable activity"
// (rather than no line at all) discipline.
constexpr int64_t NOTABLE_GAP_THRESHOLD_MS = 10000;

std::vector<std::string> draft_report(std::vector<FrameRecord> frames, int64_t incident_end_ms) {
    std::sort(frames.begin(), frames.end(),
              [](const FrameRecord& a, const FrameRecord& b) { return a.timestamp_ms < b.timestamp_ms; });
    auto intervals = coalesce_tag_intervals(frames);

    std::vector<std::string> lines;
    int64_t cursor = 0;
    for (const auto& iv : intervals) {
        if (iv.start_ms - cursor >= NOTABLE_GAP_THRESHOLD_MS) {
            std::ostringstream gap;
            gap << "[" << format_mmss(cursor) << " - " << format_mmss(iv.start_ms) << "] "
                << "No notable activity";
            lines.push_back(gap.str());
        }
        lines.push_back(render_narrative_line(iv));
        cursor = std::max(cursor, iv.end_ms);
    }
    if (incident_end_ms - cursor >= NOTABLE_GAP_THRESHOLD_MS) {
        std::ostringstream gap;
        gap << "[" << format_mmss(cursor) << " - " << format_mmss(incident_end_ms) << "] "
            << "No notable activity";
        lines.push_back(gap.str());
    }
    return lines;
}

// =======================================================================
// PART 3: a real, well-known 32-bit FNV-1a hash, built from scratch and
// verified against its own real, published reference vectors -- this
// book's own dHash (23.1), Levenshtein distance (23.2), and Luhn
// checksum (23.4) all followed the identical discipline: implement a
// real, independently-famous algorithm, then verify it against a real
// reference value nobody could quietly fudge.
// =======================================================================
constexpr uint32_t FNV_OFFSET_BASIS_32 = 0x811c9dc5u;
constexpr uint32_t FNV_PRIME_32 = 16777619u;

uint32_t fnv1a_32(const std::string& data) {
    uint32_t hash = FNV_OFFSET_BASIS_32;
    for (unsigned char c : data) {
        hash ^= c;
        hash *= FNV_PRIME_32;
    }
    return hash;
}

// This incident's own canonical, order-sensitive serialization -- the
// exact bytes the chain-of-custody hash is computed over. Changing a
// single tag on a single frame changes this string, and therefore
// changes the hash.
std::string canonical_serialization(const std::vector<FrameRecord>& frames) {
    std::ostringstream out;
    for (const auto& f : frames) {
        out << f.frame_index << ":" << f.timestamp_ms << ":";
        for (SceneTag t : f.tags) out << scene_tag_name(t) << ",";
        out << ";";
    }
    return out.str();
}

uint32_t chain_of_custody_hash(const std::vector<FrameRecord>& frames) {
    return fnv1a_32(canonical_serialization(frames));
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 25.2: Incident Report Drafting and Chain of Custody\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: this section's own from-scratch FNV-1a implementation matches its real, "
                 "published reference vectors exactly --\n";
    {
        CHECK(fnv1a_32("") == 0x811c9dc5u);
        CHECK(fnv1a_32("a") == 0xe40c292cu);
        CHECK(fnv1a_32("b") == 0xe70c2de5u);
        CHECK(fnv1a_32("foobar") == 0xbf9cf968u);
        std::cout << "  fnv1a_32(\"\") = 0x811c9dc5, fnv1a_32(\"a\") = 0xe40c292c, "
                     "fnv1a_32(\"b\") = 0xe70c2de5, fnv1a_32(\"foobar\") = 0xbf9cf968 -- all 4 match "
                     "FNV-1a's own real, published reference vectors exactly\n";
    }

    std::cout << "\n-- Test 2: real presence-interval coalescing merges contiguous same-tag frames into "
                 "one interval, and splits into two once a real gap exceeds this section's own stated "
                 "tolerance --\n";
    {
        std::vector<FrameRecord> frames = {
            {0, 0, {SceneTag::WEAPON_VISIBLE}},
            {30, 1000, {SceneTag::WEAPON_VISIBLE}},
            {60, 2000, {SceneTag::WEAPON_VISIBLE}},
            {90, 20000, {SceneTag::WEAPON_VISIBLE}},  // 18s after the last WEAPON_VISIBLE frame -- a new interval
        };
        auto intervals = coalesce_tag_intervals(frames);
        std::vector<TagInterval> weapon_intervals;
        for (const auto& iv : intervals) if (iv.tag == SceneTag::WEAPON_VISIBLE) weapon_intervals.push_back(iv);
        CHECK(weapon_intervals.size() == 2);
        CHECK(weapon_intervals[0].start_ms == 0 && weapon_intervals[0].end_ms == 2000);
        CHECK(weapon_intervals[1].start_ms == 20000 && weapon_intervals[1].end_ms == 20000);
        std::cout << "  WEAPON_VISIBLE at 0ms, 1000ms, and 2000ms (each within the 5000ms tolerance) "
                     "coalesces into one interval [0ms, 2000ms]; the same tag reappearing at 20000ms, "
                     "18 real seconds later, correctly starts its own separate second interval\n";
    }

    std::cout << "\n-- Test 3: a coalesced interval's own narrative line renders to an exact, "
                 "deterministic mm:ss-formatted string --\n";
    {
        TagInterval iv{SceneTag::PERSON_ON_GROUND, 4000, 12000};
        CHECK(render_narrative_line(iv) == "[00:04 - 00:12] PERSON_ON_GROUND observed");
        TagInterval iv2{SceneTag::VEHICLE_PRESENT, 65000, 130000};
        CHECK(render_narrative_line(iv2) == "[01:05 - 02:10] VEHICLE_PRESENT observed");
        std::cout << "  an interval from 4000ms to 12000ms renders to exactly "
                     "\"[00:04 - 00:12] PERSON_ON_GROUND observed\"; an interval crossing a real minute "
                     "boundary, 65000ms to 130000ms, renders to exactly "
                     "\"[01:05 - 02:10] VEHICLE_PRESENT observed\"\n";
    }

    std::cout << "\n-- Test 4: draft_report sorts genuinely out-of-order input frames before drafting, "
                 "producing the identical real narrative an already-sorted input would --\n";
    {
        std::vector<FrameRecord> sorted_frames = {
            {0, 0, {SceneTag::FORCED_ENTRY}},
            {10, 1000, {SceneTag::FORCED_ENTRY}},
        };
        std::vector<FrameRecord> shuffled_frames = {
            {10, 1000, {SceneTag::FORCED_ENTRY}},
            {0, 0, {SceneTag::FORCED_ENTRY}},
        };
        auto report_sorted = draft_report(sorted_frames, 2000);
        auto report_shuffled = draft_report(shuffled_frames, 2000);
        CHECK(report_sorted == report_shuffled);
        CHECK(!report_sorted.empty());
        CHECK(report_sorted[0] == "[00:00 - 00:01] FORCED_ENTRY observed");
        std::cout << "  the identical two frames, supplied once in real chronological order and once "
                     "deliberately shuffled, draft to the exact identical report -- draft_report never "
                     "assumes its own input already arrived sorted\n";
    }

    std::cout << "\n-- Test 5: a real, genuine gap with no tag activity at all is reported explicitly, "
                 "never silently omitted --\n";
    {
        std::vector<FrameRecord> frames = {
            {0, 0, {SceneTag::VEHICLE_PRESENT}},
            {600, 60000, {SceneTag::EMS_PRESENT}},
        };
        auto report = draft_report(frames, 90000);
        bool has_gap_line = false;
        for (const auto& line : report) if (line.find("No notable activity") != std::string::npos) has_gap_line = true;
        CHECK(has_gap_line);
        CHECK(report.back().find("No notable activity") != std::string::npos);
        std::cout << "  a real 59-second silence between VEHICLE_PRESENT ending at 0ms and EMS_PRESENT "
                     "starting at 60000ms, and a real 30-second silence after EMS_PRESENT ends and the "
                     "incident's own stated end time, both get their own explicit "
                     "\"No notable activity\" line rather than simply not appearing\n";
    }

    std::cout << "\n-- Test 6: the chain-of-custody hash is a real, tamper-evident checksum -- an "
                 "identical record hashes identically, and changing even one tag on one frame changes "
                 "the hash --\n";
    {
        std::vector<FrameRecord> original = {
            {0, 0, {SceneTag::VEHICLE_PRESENT}},
            {30, 1000, {SceneTag::WEAPON_VISIBLE, SceneTag::PERSON_ON_GROUND}},
        };
        std::vector<FrameRecord> identical_copy = original;
        std::vector<FrameRecord> tampered = original;
        tampered[1].tags = {SceneTag::WEAPON_VISIBLE};  // PERSON_ON_GROUND quietly removed

        uint32_t hash_original = chain_of_custody_hash(original);
        uint32_t hash_copy = chain_of_custody_hash(identical_copy);
        uint32_t hash_tampered = chain_of_custody_hash(tampered);

        CHECK(hash_original == hash_copy);
        CHECK(hash_original != hash_tampered);
        std::cout << "  an identical copy of the same incident record hashes to the exact same real "
                     "chain-of-custody value; silently removing just the PERSON_ON_GROUND tag from one "
                     "single frame changes that hash, proving the checksum really does depend on every "
                     "real tag on every real frame\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
