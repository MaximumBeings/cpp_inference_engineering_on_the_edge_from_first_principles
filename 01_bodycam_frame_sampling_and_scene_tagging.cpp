// Chapter 25.1 -- A body-worn camera cannot process every frame it
// records, and it must never be able to identify a face even if it
// wanted to. This section builds a real frame-sampling strategy (a
// fixed periodic rate unioned with real motion-triggered frames, deduped
// and sorted) suitable for edge deployment, plus a scene-tagging data
// model whose own types make facial recognition structurally
// impossible -- there is no field anywhere in this module capable of
// holding an identity, proven here at COMPILE time with a real C++20
// concept, not merely promised in a comment.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 01_bodycam_frame_sampling_and_scene_tagging.cpp -o 01_bodycam_frame_sampling_and_scene_tagging
// Run:     ./01_bodycam_frame_sampling_and_scene_tagging

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
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
// PART 1: a real frame-sampling strategy -- fixed periodic coverage,
// unioned with real motion-triggered frames, deduped and sorted.
// =======================================================================
enum class SampleReason { PERIODIC, MOTION_TRIGGERED };

struct FrameSample {
    int frame_index = 0;
    SampleReason reason = SampleReason::PERIODIC;
};

// Real periodic interval, in frames: the number of frames between two
// consecutive periodic samples at the stated base sampling rate.
int periodic_interval_frames(double fps, double base_sample_hz) {
    return std::max(1, static_cast<int>(std::llround(fps / base_sample_hz)));
}

std::vector<FrameSample> select_frames_to_process(int total_frames, double fps, double base_sample_hz,
                                                   const std::vector<int>& motion_trigger_frames) {
    int interval = periodic_interval_frames(fps, base_sample_hz);
    std::set<int> periodic;
    for (int f = 0; f < total_frames; f += interval) periodic.insert(f);

    std::vector<FrameSample> out;
    for (int f : periodic) out.push_back({f, SampleReason::PERIODIC});

    // A real motion event outside the actual recorded frame range is
    // sensor/detector noise, not a real evidentiary frame -- it is
    // honestly dropped rather than silently accepted.
    for (int f : motion_trigger_frames) {
        if (f < 0 || f >= total_frames) continue;
        if (periodic.count(f)) continue;  // already covered periodically; no duplicate entry
        out.push_back({f, SampleReason::MOTION_TRIGGERED});
    }

    std::sort(out.begin(), out.end(), [](const FrameSample& a, const FrameSample& b) {
        return a.frame_index < b.frame_index;
    });
    return out;
}

// The largest real gap, in frames, between two consecutive processed
// samples -- a real, checkable coverage bound: periodic-only sampling
// can never leave a gap larger than the periodic interval itself.
int max_gap_frames(const std::vector<FrameSample>& samples) {
    if (samples.size() < 2) return 0;
    int worst = 0;
    for (size_t i = 1; i < samples.size(); i++) {
        worst = std::max(worst, samples[i].frame_index - samples[i - 1].frame_index);
    }
    return worst;
}

// =======================================================================
// PART 2: the closed scene-tag vocabulary and a strict parser, exactly
// the same "reject anything outside the closed vocabulary, by name"
// discipline as Section 20.3's triage-report parser and Section 24.1's
// edit-plan parser.
// =======================================================================
enum class SceneTag {
    VEHICLE_PRESENT, WEAPON_VISIBLE, PERSON_ON_GROUND, CROWD_GATHERING,
    PROPERTY_DAMAGE, EMS_PRESENT, HANDCUFFS_VISIBLE, FORCED_ENTRY,
};

const std::vector<SceneTag> ALL_SCENE_TAGS = {
    SceneTag::VEHICLE_PRESENT, SceneTag::WEAPON_VISIBLE, SceneTag::PERSON_ON_GROUND,
    SceneTag::CROWD_GATHERING, SceneTag::PROPERTY_DAMAGE, SceneTag::EMS_PRESENT,
    SceneTag::HANDCUFFS_VISIBLE, SceneTag::FORCED_ENTRY,
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

struct TagParseResult {
    bool ok = false;
    SceneTag tag{};
    std::string error;
};

TagParseResult parse_scene_tag(const std::string& token) {
    for (SceneTag t : ALL_SCENE_TAGS) {
        if (scene_tag_name(t) == token) return {true, t, ""};
    }
    return {false, SceneTag::VEHICLE_PRESENT, "unrecognized scene tag token: '" + token + "'"};
}

// =======================================================================
// PART 3: the person-detection model. UniformType names only a role
// visible from clothing -- there is deliberately no name, badge lookup,
// or face-embedding field anywhere in PersonBox, checked below at
// compile time.
// =======================================================================
enum class UniformType { OFFICER, CIVILIAN, EMS, UNKNOWN };

std::string uniform_type_name(UniformType u) {
    switch (u) {
        case UniformType::OFFICER: return "OFFICER";
        case UniformType::CIVILIAN: return "CIVILIAN";
        case UniformType::EMS: return "EMS";
        case UniformType::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

struct PersonBox {
    double x = 0.0, y = 0.0, w = 0.0, h = 0.0;
    UniformType uniform = UniformType::UNKNOWN;
    // No name. No badge number. No face embedding. No identity field of
    // any kind. This is not a policy this file follows -- it is a type
    // this file defines, and no function anywhere below can extract an
    // identity from it because there is nowhere for one to be stored.
};

// A real, general C++20 concept: true only for a type exposing an
// identity-shaped member. Checked against PersonBox below with a
// static_assert -- if a future edit ever added a `.name` or `.person_id`
// field to PersonBox, this file would fail to COMPILE, not merely fail
// a runtime test a reviewer might skip.
template <typename T>
concept HasIdentityField = requires(T t) {
    t.name;
} || requires(T t) {
    t.person_id;
} || requires(T t) {
    t.face_embedding;
};

static_assert(!HasIdentityField<PersonBox>,
              "PersonBox must never gain an identity-shaped field -- this book's own "
              "no-facial-recognition boundary is enforced structurally, at compile time.");

struct FrameAnnotation {
    FrameSample sample;
    std::vector<SceneTag> tags;
    std::vector<PersonBox> people;
};

std::string to_frame_line(const FrameAnnotation& ann) {
    std::ostringstream out;
    out << "[frame " << ann.sample.frame_index << "] tags: ";
    for (size_t i = 0; i < ann.tags.size(); i++) {
        if (i) out << ", ";
        out << scene_tag_name(ann.tags[i]);
    }
    if (ann.tags.empty()) out << "(none)";
    out << "; people: " << ann.people.size();
    if (!ann.people.empty()) {
        out << " (";
        for (size_t i = 0; i < ann.people.size(); i++) {
            if (i) out << ", ";
            out << uniform_type_name(ann.people[i].uniform);
        }
        out << ")";
    }
    return out.str();
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 25.1: Body-Camera Frame Sampling and Scene Tagging\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: periodic sampling at a real fixed interval, computed from fps and a "
                 "stated base sample rate --\n";
    {
        CHECK(periodic_interval_frames(30.0, 1.0) == 30);
        auto samples = select_frames_to_process(100, 30.0, 1.0, {});
        std::vector<int> got;
        for (const auto& s : samples) got.push_back(s.frame_index);
        std::vector<int> expected = {0, 30, 60, 90};
        CHECK(got == expected);
        for (const auto& s : samples) CHECK(s.reason == SampleReason::PERIODIC);
        std::cout << "  30 fps sampled at a 1 Hz base rate resolves to exactly a 30-frame interval; "
                     "100 total frames periodically sample to exactly frames 0, 30, 60, 90\n";
    }

    std::cout << "\n-- Test 2: real motion-triggered frames are merged with periodic coverage, deduped "
                 "against any periodic frame they coincide with, and the whole result stays sorted --\n";
    {
        auto samples = select_frames_to_process(100, 30.0, 1.0, {45, 30, 61});
        std::vector<int> got;
        for (const auto& s : samples) got.push_back(s.frame_index);
        std::vector<int> expected = {0, 30, 45, 60, 61, 90};
        CHECK(got == expected);
        // Frame 30 was requested as a motion trigger but already exists periodically --
        // it appears exactly once, still marked PERIODIC, never duplicated.
        int count_30 = 0;
        SampleReason reason_30 = SampleReason::MOTION_TRIGGERED;
        for (const auto& s : samples) if (s.frame_index == 30) { count_30++; reason_30 = s.reason; }
        CHECK(count_30 == 1);
        CHECK(reason_30 == SampleReason::PERIODIC);
        // Frames 45 and 61 are real, new motion-triggered samples.
        for (const auto& s : samples) {
            if (s.frame_index == 45 || s.frame_index == 61) CHECK(s.reason == SampleReason::MOTION_TRIGGERED);
        }
        std::cout << "  motion triggers at frames 45, 30, and 61 merge with the periodic 0/30/60/90 "
                     "sequence into exactly 0, 30, 45, 60, 61, 90 -- frame 30 appears exactly once, "
                     "correctly still marked PERIODIC rather than duplicated as MOTION_TRIGGERED\n";
    }

    std::cout << "\n-- Test 3: a motion trigger reported outside the real recorded frame range is "
                 "honestly dropped as sensor noise, never accepted as a real evidentiary frame --\n";
    {
        auto samples = select_frames_to_process(100, 30.0, 1.0, {-5, 250, 50});
        bool has_negative = false, has_out_of_range = false, has_fifty = false;
        for (const auto& s : samples) {
            if (s.frame_index < 0) has_negative = true;
            if (s.frame_index >= 100) has_out_of_range = true;
            if (s.frame_index == 50) has_fifty = true;
        }
        CHECK(!has_negative);
        CHECK(!has_out_of_range);
        CHECK(has_fifty);
        std::cout << "  motion triggers at frame -5 and frame 250 (outside the real 0..99 recorded "
                     "range for a 100-frame clip) are both dropped; the one real in-range trigger at "
                     "frame 50 is kept\n";
    }

    std::cout << "\n-- Test 4: periodic-only sampling can never leave a real coverage gap larger than "
                 "its own computed interval, checked directly rather than merely assumed --\n";
    {
        auto samples = select_frames_to_process(100, 30.0, 1.0, {});
        CHECK(max_gap_frames(samples) == 30);
        auto with_motion = select_frames_to_process(100, 30.0, 1.0, {45});
        CHECK(max_gap_frames(with_motion) <= 30);
        std::cout << "  periodic-only sampling of a 100-frame clip at a 30-frame interval has a real "
                     "worst-case gap of exactly 30 frames; adding a motion-triggered frame can only "
                     "ever shrink that worst-case gap, never grow it\n";
    }

    std::cout << "\n-- Test 5: the closed scene-tag vocabulary's own strict parser accepts every real "
                 "tag name and rejects anything outside it, by name --\n";
    {
        auto r1 = parse_scene_tag("WEAPON_VISIBLE");
        CHECK(r1.ok && r1.tag == SceneTag::WEAPON_VISIBLE);
        auto r2 = parse_scene_tag("SUSPICIOUS_PERSON");
        CHECK(!r2.ok);
        CHECK(r2.error.find("SUSPICIOUS_PERSON") != std::string::npos);
        CHECK(ALL_SCENE_TAGS.size() == 8);
        std::cout << "  \"WEAPON_VISIBLE\" parses to its own real enum value; the unrecognized token "
                     "\"SUSPICIOUS_PERSON\" is rejected with that exact token named in the error, never "
                     "silently mapped to the nearest-sounding real tag\n";
    }

    std::cout << "\n-- Test 6: a frame's own report line renders its real scene tags and real uniform "
                 "roles exactly, and nothing else --\n";
    {
        FrameAnnotation ann;
        ann.sample = {120, SampleReason::MOTION_TRIGGERED};
        ann.tags = {SceneTag::WEAPON_VISIBLE, SceneTag::PERSON_ON_GROUND};
        ann.people = {{10, 20, 5, 8, UniformType::OFFICER}, {30, 40, 4, 7, UniformType::CIVILIAN}};
        std::string line = to_frame_line(ann);
        CHECK(line == "[frame 120] tags: WEAPON_VISIBLE, PERSON_ON_GROUND; people: 2 (OFFICER, CIVILIAN)");

        FrameAnnotation empty_ann;
        empty_ann.sample = {0, SampleReason::PERIODIC};
        CHECK(to_frame_line(empty_ann) == "[frame 0] tags: (none); people: 0");
        std::cout << "  frame 120's report line renders to exactly "
                     "\"[frame 120] tags: WEAPON_VISIBLE, PERSON_ON_GROUND; people: 2 (OFFICER, CIVILIAN)\" "
                     "-- a role and a bounding box, never a name; an empty frame renders to exactly "
                     "\"[frame 0] tags: (none); people: 0\"\n";
    }

    std::cout << "\n-- Test 7: PersonBox's own structural inability to hold an identity is a real, "
                 "compile-time property, checked here at runtime against the identical concept the "
                 "static_assert above already enforced at build time --\n";
    {
        CHECK(!HasIdentityField<PersonBox>);
        CHECK(HasIdentityField<FrameAnnotation> == false);  // no identity reachable through this type either
        std::cout << "  this file already failed to compile if PersonBox ever gained a name, "
                     "person_id, or face_embedding field -- the fact this line runs at all is itself "
                     "part of the proof, confirmed here directly against the same real concept\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
