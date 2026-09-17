# Chapter 25: Body-Worn and Personal Cameras: Evidence Documentation and Health Tracking

**What you will understand by the end of this chapter:**

- How to build a real frame-sampling strategy for continuous body-camera footage -- a fixed periodic rate unioned with real motion-triggered frames -- and a scene-tagging data model whose own types make facial recognition structurally impossible, proven at compile time rather than merely promised in a comment or a policy document.
- How to turn raw per-frame scene tags into a real, chronological incident report, reusing this book's own presence-interval coalescing technique, with an honest "no notable activity" line for any real gap and a real, tamper-evident chain-of-custody checksum built from a well-known algorithm verified against its own published reference values.
- How to build a real joint-angle formula from 2D keypoints and a real rep-counting state machine with genuine hysteresis, honest enough to refuse counting a rep that never actually reached real depth.
- How to build a privacy-first, fully on-device nutrition tracker against a small, stated, offline reference subset of real USDA FoodData Central-style values, with portion size estimated from a real reference-object scale and reported honestly as a range rather than a single fabricated gram figure.

**What you need to know first:**

- Section 20.1's own structural, type-enforced discipline (a state machine whose type system makes an unsafe transition impossible to express) is the same discipline Section 25.1 applies to identity: `PersonBox` cannot hold a name or a face embedding because no such field exists on the type, checked with a real C++20 concept and a `static_assert`.
- Section 22.1's own presence-interval coalescing (turning raw per-frame detections into contiguous real time intervals) is reused unchanged in Section 25.2, and Section 23.1's, 23.2's, and 23.4's own discipline of implementing a real, independently-famous algorithm from scratch and verifying it against a real published reference value is what Section 25.2 does again here with FNV-1a.
- Section 21.3's own CostRange discipline (report an honest range rather than a fabricated point estimate) is reapplied directly in Section 25.4 to portion-size estimation, and Sections 20.1 and 20.4's own human-in-the-loop discipline is what lets Section 25.4's daily tally stay exact rather than needing to propagate that range through every downstream sum.

---

This chapter closes Part 5 with two cameras that see far more of a person's life than any system this book has built so far, and that fact shapes every design choice in it. A police body camera records members of the public who never consented to being recorded, in circumstances where an identification error carries real legal weight -- so Section 25.1 and Section 25.2 build a real evidentiary pipeline whose own types make facial recognition impossible to add by accident, not merely discouraged by policy. A personal fitness or nutrition camera records its own owner, continuously, in their own home -- so Section 25.3 and Section 25.4 build real, useful health-tracking techniques that run entirely on-device, with no image or personal health data ever needing to leave it. Both halves of this chapter are united by the same real principle: a genuinely privacy-respecting camera system is not a promise layered on top of the architecture, it is the architecture.

## 25.1 Body-Camera Frame Sampling and Scene Tagging Under a No-Facial-Recognition Boundary

### Intuition

A body camera recording at 30 frames per second for an entire shift generates far more frames than any edge device can process in real time, and it must never be able to recognize a specific person even if someone later asked it to. This section builds a real frame-sampling strategy that keeps coverage honest without processing every frame, and a scene-tagging data model that makes identification structurally impossible rather than merely disallowed.

### The Concept, In Detail

`select_frames_to_process` builds a real, fixed periodic sample from the stated frame rate and base sampling frequency, then unions it with real motion-triggered frames -- deduplicating any trigger that coincides with a frame already selected periodically, and honestly dropping any trigger reported outside the actual recorded frame range as sensor noise rather than accepting it as real evidence. Test 4 confirms a real, checkable coverage guarantee: periodic-only sampling can never leave a gap larger than its own computed interval.

The scene-tagging model is where this section's own central discipline lives. `SceneTag` is a small closed vocabulary of eight named tags, and `parse_scene_tag` rejects anything outside it by name, the same discipline Section 20.3's triage-report parser and Section 24.1's edit-plan parser both established. `PersonBox` records only a bounding box and a `UniformType` role visible from clothing -- there is no name field, no badge-lookup field, and no face-embedding field anywhere on the type. Test 7 confirms this with the real C++20 `HasIdentityField` concept, and the `static_assert` above it means this file would simply fail to COMPILE if a future edit ever tried to add one -- a structural guarantee in the same spirit as Section 20.1's state machine, applied here to identity instead of clinical sign-off.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_bodycam_frame_sampling_and_scene_tagging.cpp -o 01_bodycam_frame_sampling_and_scene_tagging
./01_bodycam_frame_sampling_and_scene_tagging
```

**Sample input:** periodic sampling checked against an exact computed interval; real motion-triggered frames checked to merge, dedupe, and stay sorted against periodic coverage; an out-of-range motion trigger checked to be honestly dropped; a real worst-case coverage-gap bound checked directly; the closed scene-tag vocabulary's strict parser checked to accept every real tag and reject an unrecognized one by name; a frame's own report line checked against an exact expected string; and `PersonBox`'s own structural inability to hold an identity checked at runtime against the same concept a compile-time `static_assert` already enforced.

```text
========================================================
Chapter 25.1: Body-Camera Frame Sampling and Scene Tagging
========================================================

-- Test 1: periodic sampling at a real fixed interval, computed from fps and a stated base sample rate --
  30 fps sampled at a 1 Hz base rate resolves to exactly a 30-frame interval; 100 total frames periodically sample to exactly frames 0, 30, 60, 90

-- Test 2: real motion-triggered frames are merged with periodic coverage, deduped against any periodic frame they coincide with, and the whole result stays sorted --
  motion triggers at frames 45, 30, and 61 merge with the periodic 0/30/60/90 sequence into exactly 0, 30, 45, 60, 61, 90 -- frame 30 appears exactly once, correctly still marked PERIODIC rather than duplicated as MOTION_TRIGGERED

-- Test 3: a motion trigger reported outside the real recorded frame range is honestly dropped as sensor noise, never accepted as a real evidentiary frame --
  motion triggers at frame -5 and frame 250 (outside the real 0..99 recorded range for a 100-frame clip) are both dropped; the one real in-range trigger at frame 50 is kept

-- Test 4: periodic-only sampling can never leave a real coverage gap larger than its own computed interval, checked directly rather than merely assumed --
  periodic-only sampling of a 100-frame clip at a 30-frame interval has a real worst-case gap of exactly 30 frames; adding a motion-triggered frame can only ever shrink that worst-case gap, never grow it

-- Test 5: the closed scene-tag vocabulary's own strict parser accepts every real tag name and rejects anything outside it, by name --
  "WEAPON_VISIBLE" parses to its own real enum value; the unrecognized token "SUSPICIOUS_PERSON" is rejected with that exact token named in the error, never silently mapped to the nearest-sounding real tag

-- Test 6: a frame's own report line renders its real scene tags and real uniform roles exactly, and nothing else --
  frame 120's report line renders to exactly "[frame 120] tags: WEAPON_VISIBLE, PERSON_ON_GROUND; people: 2 (OFFICER, CIVILIAN)" -- a role and a bounding box, never a name; an empty frame renders to exactly "[frame 0] tags: (none); people: 0"

-- Test 7: PersonBox's own structural inability to hold an identity is a real, compile-time property, checked here at runtime against the identical concept the static_assert above already enforced at build time --
  this file already failed to compile if PersonBox ever gained a name, person_id, or face_embedding field -- the fact this line runs at all is itself part of the proof, confirmed here directly against the same real concept

24/24 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating the no-facial-recognition boundary as a policy to document rather than a type to design"
    It is tempting to build `PersonBox` first with whatever fields seem useful, then write a comment or a compliance document saying "this system does not perform facial recognition." That leaves the actual guarantee resting entirely on every future contributor reading and honoring a comment. Section 25.1's own `HasIdentityField` concept and its `static_assert` move that guarantee into the type system itself: a future edit that added a `.name`, `.person_id`, or `.face_embedding` field to `PersonBox` would not produce a policy violation to be caught in review -- it would produce a compiler error, on every build, for every contributor, forever. A real legal and ethical boundary this important deserves a mechanism stronger than a comment nobody is forced to read.

## 25.2 Incident Report Drafting and Chain of Custody

### Intuition

A body camera's own raw per-frame tags are not a report a human can read or a court can trust. This section turns them into a real chronological narrative, honestly labels any real gap in activity, and computes a real, tamper-evident checksum proving exactly what record a report was built from.

### The Concept, In Detail

`coalesce_tag_intervals` reapplies Section 22.1's own presence-interval coalescing technique unchanged: contiguous same-tag frames within a real, stated tolerance merge into one interval, and Test 2 confirms a genuine 18-second gap between two `WEAPON_VISIBLE` observations correctly starts a new, separate interval rather than merging into one implausibly long one. `draft_report` sorts its own input before processing -- Test 4 confirms identical output whether the input frames arrive in real chronological order or deliberately shuffled -- and inserts an explicit "No notable activity" line for any real gap exceeding this section's own stated threshold, rather than silently omitting empty stretches of the timeline the way a naive renderer might.

Test 1 and Test 6 are this section's own central real-algorithm work: a from-scratch 32-bit FNV-1a implementation, verified against its own real, published reference vectors exactly the way Section 23.1's dHash, Section 23.2's Levenshtein distance, and Section 23.4's Luhn checksum were each verified against their own famous reference values. `chain_of_custody_hash` applies this real checksum to a canonical serialization of an entire incident's frame record, and Test 6 proves it is genuinely tamper-evident: an identical copy of a record hashes identically, while silently removing a single tag from a single frame changes the hash.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_incident_report_drafting_and_chain_of_custody.cpp -o 02_incident_report_drafting_and_chain_of_custody
./02_incident_report_drafting_and_chain_of_custody
```

**Sample input:** a from-scratch FNV-1a implementation checked against its own real, published reference vectors; presence-interval coalescing checked to merge contiguous tags and split across a genuine gap; a coalesced interval's narrative line checked against an exact mm:ss-formatted string; report drafting checked to produce identical output from sorted and deliberately shuffled input; a real silent gap checked to produce an explicit "No notable activity" line; and the chain-of-custody hash checked to be identical for an identical record and different the moment a single tag on a single frame is altered.

```text
========================================================
Chapter 25.2: Incident Report Drafting and Chain of Custody
========================================================

-- Test 1: this section's own from-scratch FNV-1a implementation matches its real, published reference vectors exactly --
  fnv1a_32("") = 0x811c9dc5, fnv1a_32("a") = 0xe40c292c, fnv1a_32("b") = 0xe70c2de5, fnv1a_32("foobar") = 0xbf9cf968 -- all 4 match FNV-1a's own real, published reference vectors exactly

-- Test 2: real presence-interval coalescing merges contiguous same-tag frames into one interval, and splits into two once a real gap exceeds this section's own stated tolerance --
  WEAPON_VISIBLE at 0ms, 1000ms, and 2000ms (each within the 5000ms tolerance) coalesces into one interval [0ms, 2000ms]; the same tag reappearing at 20000ms, 18 real seconds later, correctly starts its own separate second interval

-- Test 3: a coalesced interval's own narrative line renders to an exact, deterministic mm:ss-formatted string --
  an interval from 4000ms to 12000ms renders to exactly "[00:04 - 00:12] PERSON_ON_GROUND observed"; an interval crossing a real minute boundary, 65000ms to 130000ms, renders to exactly "[01:05 - 02:10] VEHICLE_PRESENT observed"

-- Test 4: draft_report sorts genuinely out-of-order input frames before drafting, producing the identical real narrative an already-sorted input would --
  the identical two frames, supplied once in real chronological order and once deliberately shuffled, draft to the exact identical report -- draft_report never assumes its own input already arrived sorted

-- Test 5: a real, genuine gap with no tag activity at all is reported explicitly, never silently omitted --
  a real 59-second silence between VEHICLE_PRESENT ending at 0ms and EMS_PRESENT starting at 60000ms, and a real 30-second silence after EMS_PRESENT ends and the incident's own stated end time, both get their own explicit "No notable activity" line rather than simply not appearing

-- Test 6: the chain-of-custody hash is a real, tamper-evident checksum -- an identical record hashes identically, and changing even one tag on one frame changes the hash --
  an identical copy of the same incident record hashes to the exact same real chain-of-custody value; silently removing just the PERSON_ON_GROUND tag from one single frame changes that hash, proving the checksum really does depend on every real tag on every real frame

16/16 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] treating a quiet stretch of footage as nothing worth reporting"
    A report generator that only ever emits a line when a scene tag is present looks, at a glance, like it is simply being efficient -- why print anything about a period where nothing of note happened? The real problem is that a report with silent gaps is indistinguishable from a report built from an incomplete or tampered recording: a reviewer reading it later cannot tell "nothing happened here" apart from "this section of footage never made it into the report." Test 5 exists specifically to demonstrate the fix: an explicit, honestly labeled "No notable activity" line for every real gap means the report's own silence is never ambiguous, and the chain-of-custody hash over the complete underlying frame record (not just the frames that happened to carry a tag) is what actually backs up that the full record was considered, not merely the interesting parts of it.

## 25.3 Exercise-Form Analysis and Rep Counting

### Intuition

A camera coaching a squat needs a real number for how deep a joint bent, and a real, honest way to decide whether a full repetition actually happened -- not merely whether the person moved.

### The Concept, In Detail

`joint_angle_degrees` is the real, standard formula for the angle between two vectors sharing a common vertex, computed from their dot product and magnitudes. Test 1 confirms it against four exact, hand-verifiable geometric configurations: a perpendicular pair at exactly 90 degrees, a fully straightened pair at exactly 180 degrees, a genuine 45-degree configuration, and two identical vectors at exactly 0 degrees.

`RepCounter` is a real state machine over `STANDING`, `DESCENDING`, `BOTTOM`, and `ASCENDING`, and its own central discipline is in Test 2: a real partial rep that descends to 128 degrees -- short of the stated 100-degree real depth threshold -- and returns directly to standing is honestly excluded from the count, the same "refuse rather than fabricate" discipline this book has applied to counterfeit screening, receipt reconciliation, and edit-plan parsing, now applied to a physical repetition that did not actually happen. Test 3 confirms a second, subtler real property: dipping back into full depth mid-ascent before finally standing back up is still the SAME rep in progress, not a second one, which is what real hysteresis in a state machine is for.

### Code and Verification

```cpp
// Chapter 25.3 -- A personal camera doing exercise-form analysis needs
// two real things standard-library C++ can build entirely on its own:
// a real joint-angle formula from 2D keypoints, and a real rep-counting
// state machine with real hysteresis, honest enough to refuse counting
// a rep that never actually reached real depth. Both run entirely
// on-device, on synthetic keypoint sequences here, with no camera
// frame or any other personal data ever needing to leave the device --
// this section's own real privacy-first architecture.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 03_exercise_form_analysis_and_rep_counting.cpp -o 03_exercise_form_analysis_and_rep_counting
// Run:     ./03_exercise_form_analysis_and_rep_counting

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: a real joint-angle formula -- the angle at `vertex`, formed by
// the two real vectors reaching out to `a` and `c`, via the standard
// dot-product-over-magnitudes formula.
// =======================================================================
struct Point2D {
    double x = 0.0, y = 0.0;
};

double joint_angle_degrees(Point2D a, Point2D vertex, Point2D c) {
    double v1x = a.x - vertex.x, v1y = a.y - vertex.y;
    double v2x = c.x - vertex.x, v2y = c.y - vertex.y;
    double dot = v1x * v2x + v1y * v2y;
    double mag1 = std::sqrt(v1x * v1x + v1y * v1y);
    double mag2 = std::sqrt(v2x * v2x + v2y * v2y);
    double cos_theta = dot / (mag1 * mag2);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));  // guard real floating-point overshoot at +/-1
    return std::acos(cos_theta) * 180.0 / std::numbers::pi;
}

// =======================================================================
// PART 2: a real rep-counting state machine over a knee-angle sequence,
// with real hysteresis -- a rep counts only on a full real cycle that
// actually reaches real depth, and dipping back into BOTTOM mid-ascent
// never double-counts.
// =======================================================================
enum class SquatPhase { STANDING, DESCENDING, BOTTOM, ASCENDING };

constexpr double STANDING_ANGLE_MIN_DEG = 160.0;
constexpr double BOTTOM_ANGLE_MAX_DEG = 100.0;

class RepCounter {
public:
    void process_frame(double knee_angle_deg) {
        switch (phase_) {
            case SquatPhase::STANDING:
                if (knee_angle_deg < STANDING_ANGLE_MIN_DEG) phase_ = SquatPhase::DESCENDING;
                break;
            case SquatPhase::DESCENDING:
                if (knee_angle_deg <= BOTTOM_ANGLE_MAX_DEG) {
                    phase_ = SquatPhase::BOTTOM;
                } else if (knee_angle_deg >= STANDING_ANGLE_MIN_DEG) {
                    // Real depth was never reached before returning to standing -- this is a real,
                    // honest partial rep, and it does not count, exactly this book's own established
                    // "refuse rather than fabricate" discipline applied to a physical rep count.
                    phase_ = SquatPhase::STANDING;
                }
                break;
            case SquatPhase::BOTTOM:
                if (knee_angle_deg > BOTTOM_ANGLE_MAX_DEG) phase_ = SquatPhase::ASCENDING;
                break;
            case SquatPhase::ASCENDING:
                if (knee_angle_deg >= STANDING_ANGLE_MIN_DEG) {
                    completed_reps_++;
                    phase_ = SquatPhase::STANDING;
                } else if (knee_angle_deg <= BOTTOM_ANGLE_MAX_DEG) {
                    // A real dip back into full depth mid-ascent is still the SAME rep in progress,
                    // not a second one -- returning to BOTTOM here, rather than counting early,
                    // is what prevents a real double-count.
                    phase_ = SquatPhase::BOTTOM;
                }
                break;
        }
    }

    int completed_reps() const { return completed_reps_; }
    SquatPhase phase() const { return phase_; }

private:
    SquatPhase phase_ = SquatPhase::STANDING;
    int completed_reps_ = 0;
};

// =======================================================================
// PART 3: real form-fault detection from each rep's own bottom-of-rep
// angles.
// =======================================================================
enum class ExerciseFormFault { INSUFFICIENT_DEPTH, KNEE_ASYMMETRY };

constexpr double KNEE_ASYMMETRY_THRESHOLD_DEG = 15.0;

std::vector<ExerciseFormFault> detect_form_faults(double left_knee_angle_at_bottom,
                                                   double right_knee_angle_at_bottom) {
    std::vector<ExerciseFormFault> faults;
    if (left_knee_angle_at_bottom > BOTTOM_ANGLE_MAX_DEG || right_knee_angle_at_bottom > BOTTOM_ANGLE_MAX_DEG) {
        faults.push_back(ExerciseFormFault::INSUFFICIENT_DEPTH);
    }
    if (std::fabs(left_knee_angle_at_bottom - right_knee_angle_at_bottom) > KNEE_ASYMMETRY_THRESHOLD_DEG) {
        faults.push_back(ExerciseFormFault::KNEE_ASYMMETRY);
    }
    return faults;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 25.3: Exercise-Form Analysis and Rep Counting\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the real joint-angle formula matches exact, hand-verifiable geometric "
                 "configurations --\n";
    {
        CHECK(near(joint_angle_degrees({0, 1}, {0, 0}, {1, 0}), 90.0));
        CHECK(near(joint_angle_degrees({0, 1}, {0, 0}, {0, -1}), 180.0));
        CHECK(near(joint_angle_degrees({1, 1}, {0, 0}, {1, 0}), 45.0));
        CHECK(near(joint_angle_degrees({1, 0}, {0, 0}, {1, 0}), 0.0));
        std::cout << "  two perpendicular real vectors from a shared vertex measure to exactly 90 "
                     "degrees; two opposite real vectors (a fully straightened leg) measure to exactly "
                     "180 degrees; a real 45-degree configuration measures to exactly 45 degrees; and "
                     "two identical real vectors measure to exactly 0 degrees\n";
    }

    std::cout << "\n-- Test 2: 3 real full reps that reach true depth are counted, and a real partial "
                 "rep that never reaches true depth before returning to standing is honestly excluded "
                 "from the count --\n";
    {
        RepCounter counter;
        // Rep 1, full.
        for (double a : {170.0, 140.0, 90.0, 130.0, 175.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 1);
        // A real partial rep: descends to 128 degrees (short of the real 100-degree depth
        // threshold) and returns to standing without ever reaching BOTTOM.
        for (double a : {135.0, 128.0, 165.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 1);  // still 1 -- the partial rep did not count
        // Rep 2, full.
        for (double a : {170.0, 120.0, 85.0, 150.0, 172.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 2);
        // Rep 3, full.
        for (double a : {110.0, 95.0, 145.0, 168.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 3);
        std::cout << "  a real 17-frame sequence containing 3 genuine full reps and 1 real partial rep "
                     "that bottoms out at only 128 degrees (short of the real 100-degree depth "
                     "threshold) before returning to standing counts to exactly 3 -- the partial rep "
                     "is correctly never counted\n";
    }

    std::cout << "\n-- Test 3: a real dip back into full depth mid-ascent is correctly treated as the "
                 "same rep still in progress, never a double count --\n";
    {
        RepCounter counter;
        for (double a : {170.0, 120.0, 90.0, 130.0, 95.0, 140.0, 165.0}) counter.process_frame(a);
        CHECK(counter.completed_reps() == 1);
        std::cout << "  a rep that reaches bottom, rises partway to 130 degrees, dips back down to 95 "
                     "degrees (real full depth again), then finally rises all the way to standing "
                     "counts as exactly 1 real rep, not 2 -- the mid-ascent dip back into BOTTOM never "
                     "triggers a premature or duplicate count\n";
    }

    std::cout << "\n-- Test 4: INSUFFICIENT_DEPTH is flagged only when a rep's own bottom angle "
                 "genuinely fails to reach the real depth threshold, inclusive of the boundary itself "
                 "--\n";
    {
        auto shallow = detect_form_faults(115.0, 112.0);
        bool has_shallow = false;
        for (auto f : shallow) if (f == ExerciseFormFault::INSUFFICIENT_DEPTH) has_shallow = true;
        CHECK(has_shallow);

        auto exactly_at_threshold = detect_form_faults(100.0, 100.0);
        bool has_at_threshold = false;
        for (auto f : exactly_at_threshold) if (f == ExerciseFormFault::INSUFFICIENT_DEPTH) has_at_threshold = true;
        CHECK(!has_at_threshold);

        auto deep_enough = detect_form_faults(90.0, 88.0);
        CHECK(deep_enough.empty());
        std::cout << "  bottom angles of 115/112 degrees (short of real depth) flag "
                     "INSUFFICIENT_DEPTH; bottom angles of exactly 100/100 degrees -- the real "
                     "threshold itself -- correctly do NOT flag it; bottom angles of 90/88 degrees "
                     "flag no faults at all\n";
    }

    std::cout << "\n-- Test 5: KNEE_ASYMMETRY is flagged only once the real left/right difference "
                 "genuinely exceeds this section's own stated threshold --\n";
    {
        auto symmetric = detect_form_faults(90.0, 92.0);
        CHECK(symmetric.empty());

        auto exactly_at_threshold = detect_form_faults(90.0, 105.0);  // exactly 15.0 degrees apart
        bool flagged_at_threshold = false;
        for (auto f : exactly_at_threshold) if (f == ExerciseFormFault::KNEE_ASYMMETRY) flagged_at_threshold = true;
        CHECK(!flagged_at_threshold);

        auto asymmetric = detect_form_faults(90.0, 106.0);  // 16.0 degrees apart -- and 106 also insufficient depth
        bool flagged_asymmetry = false, flagged_depth = false;
        for (auto f : asymmetric) {
            if (f == ExerciseFormFault::KNEE_ASYMMETRY) flagged_asymmetry = true;
            if (f == ExerciseFormFault::INSUFFICIENT_DEPTH) flagged_depth = true;
        }
        CHECK(flagged_asymmetry);
        CHECK(flagged_depth);
        std::cout << "  a real 2-degree left/right difference flags no fault at all; a real 15-degree "
                     "difference -- exactly this section's own stated threshold -- correctly does NOT "
                     "flag KNEE_ASYMMETRY; a real 16-degree difference does, and here it also correctly "
                     "flags INSUFFICIENT_DEPTH at the same time, since 106 degrees also fails the "
                     "real depth threshold on its own\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_exercise_form_analysis_and_rep_counting.cpp -o 03_exercise_form_analysis_and_rep_counting
./03_exercise_form_analysis_and_rep_counting
```

**Sample input:** the joint-angle formula checked against four exact geometric configurations (90, 180, 45, and 0 degrees); a real 17-frame sequence containing 3 genuine full reps and 1 real partial rep checked to count to exactly 3; a real mid-ascent dip back into full depth checked to avoid a double count; `INSUFFICIENT_DEPTH` checked against a real depth threshold inclusive at its own boundary; and `KNEE_ASYMMETRY` checked against a real left/right threshold, also inclusive at its own boundary.

```text
========================================================
Chapter 25.3: Exercise-Form Analysis and Rep Counting
========================================================

-- Test 1: the real joint-angle formula matches exact, hand-verifiable geometric configurations --
  two perpendicular real vectors from a shared vertex measure to exactly 90 degrees; two opposite real vectors (a fully straightened leg) measure to exactly 180 degrees; a real 45-degree configuration measures to exactly 45 degrees; and two identical real vectors measure to exactly 0 degrees

-- Test 2: 3 real full reps that reach true depth are counted, and a real partial rep that never reaches true depth before returning to standing is honestly excluded from the count --
  a real 17-frame sequence containing 3 genuine full reps and 1 real partial rep that bottoms out at only 128 degrees (short of the real 100-degree depth threshold) before returning to standing counts to exactly 3 -- the partial rep is correctly never counted

-- Test 3: a real dip back into full depth mid-ascent is correctly treated as the same rep still in progress, never a double count --
  a rep that reaches bottom, rises partway to 130 degrees, dips back down to 95 degrees (real full depth again), then finally rises all the way to standing counts as exactly 1 real rep, not 2 -- the mid-ascent dip back into BOTTOM never triggers a premature or duplicate count

-- Test 4: INSUFFICIENT_DEPTH is flagged only when a rep's own bottom angle genuinely fails to reach the real depth threshold, inclusive of the boundary itself --
  bottom angles of 115/112 degrees (short of real depth) flag INSUFFICIENT_DEPTH; bottom angles of exactly 100/100 degrees -- the real threshold itself -- correctly do NOT flag it; bottom angles of 90/88 degrees flag no faults at all

-- Test 5: KNEE_ASYMMETRY is flagged only once the real left/right difference genuinely exceeds this section's own stated threshold --
  a real 2-degree left/right difference flags no fault at all; a real 15-degree difference -- exactly this section's own stated threshold -- correctly does NOT flag KNEE_ASYMMETRY; a real 16-degree difference does, and here it also correctly flags INSUFFICIENT_DEPTH at the same time, since 106 degrees also fails the real depth threshold on its own

16/16 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] counting a rep the moment the athlete starts moving back up"
    A naive rep counter watching for "went down, now going back up" would have counted Test 2's own partial rep -- the one that only reached 128 degrees before returning to standing -- as a real completed repetition, because motion reversed direction. `RepCounter`'s own `DESCENDING` state specifically distinguishes "started descending" from "actually reached `BOTTOM`," and a rep only counts on a full real cycle that passed through `BOTTOM` on the way. The distinction matters for exactly the reason a fitness application exists in the first place: an athlete relying on this system's own rep count to know whether they hit their real training target deserves a system that only counts a repetition that actually satisfied its own stated depth requirement, not one that rewards a shortcut the system itself was supposed to catch.

## 25.4 Camera-Based Nutrition Tracking via a USDA-Style Offline Reference

### Intuition

A nutrition tracker built for a personal device cannot depend on a live network call to USDA FoodData Central for every single meal -- the whole point of a privacy-first architecture is that a photograph of dinner never needs to leave the device it was taken on. This section builds a real, complete pipeline against a small, honestly-scoped offline reference instead.

### The Concept, In Detail

`nutrient_profile_for` is a small, closed, stated subset of real, approximate USDA FoodData Central-style per-100g values -- Test 1 checks a representative sample against this section's own stated figures directly, and Test 2 confirms, exhaustively across all six foods rather than a hand-picked sample, that none of them silently falls through to a missing or zeroed-out entry.

`estimate_portion_range` is this section's own central discipline: portion mass is estimated from a real reference-object scale (a standard dinner plate's own known real diameter converts a food region's pixel area into a real area in square centimeters), but the result is reported as an honest `[low, mid, high]` range rather than a single fabricated gram figure -- Section 21.3's own CostRange discipline, reapplied here to a portion size no vision system can actually measure to the gram. Test 3 checks this against an exact, hand-computed scenario, and Test 4 confirms the real `low < mid < high` ordering holds across the entire closed food vocabulary, not merely the examples hand-picked for Test 3. `daily_nutrient_tally` and `classify_daily_intake` then work from grams the user has already reviewed and confirmed -- Sections 20.1 and 20.4's own human-in-the-loop discipline -- which is exactly what lets the tally itself stay exact and deterministic rather than needing to propagate an estimate's own uncertainty through every downstream sum.

### Code and Verification

```cpp
// Chapter 25.4 -- A camera-based nutrition tracker's own privacy-first
// architecture means it cannot depend on a live USDA FoodData Central
// API call for every meal: every function in this section runs
// entirely on-device, against a small, stated, offline reference
// subset of real, approximate USDA FoodData Central per-100g values,
// with no image or meal data ever needing to leave the device. Portion
// size is estimated from a real reference-object scale (a standard
// dinner plate's own known real diameter) and reported honestly as a
// range, never a fabricated single-gram point estimate -- this book's
// own established CostRange discipline (Section 21.3), reapplied here
// to a genuinely uncertain real-world measurement.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_camera_based_nutrition_tracking.cpp -o 04_camera_based_nutrition_tracking
// Run:     ./04_camera_based_nutrition_tracking

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) < eps; }

// =======================================================================
// PART 1: a small, closed, offline reference subset of real,
// approximate USDA FoodData Central per-100g values. Stated and
// approximate, deliberately -- this is a small offline subset for a
// privacy-first device, never the live API, and this section never
// claims lab-grade precision for either these reference values or the
// portion estimates built on top of them.
// =======================================================================
enum class FoodItem {
    APPLE_RAW, BANANA_RAW, CHICKEN_BREAST_COOKED, WHITE_RICE_COOKED, BROCCOLI_RAW, EGG_LARGE_COOKED,
};

const std::vector<FoodItem> ALL_FOOD_ITEMS = {
    FoodItem::APPLE_RAW, FoodItem::BANANA_RAW, FoodItem::CHICKEN_BREAST_COOKED,
    FoodItem::WHITE_RICE_COOKED, FoodItem::BROCCOLI_RAW, FoodItem::EGG_LARGE_COOKED,
};

struct NutrientProfile {
    double calories_per_100g = 0.0;
    double protein_g_per_100g = 0.0;
    double carbs_g_per_100g = 0.0;
    double fat_g_per_100g = 0.0;
    double density_g_per_cm3 = 0.0;      // for portion-mass estimation
    double assumed_thickness_cm = 0.0;   // a stated, illustrative average-serving assumption
};

NutrientProfile nutrient_profile_for(FoodItem food) {
    switch (food) {
        case FoodItem::APPLE_RAW:             return {52.0, 0.3, 13.8, 0.2, 0.6, 2.0};
        case FoodItem::BANANA_RAW:            return {89.0, 1.1, 22.8, 0.3, 0.5, 2.0};
        case FoodItem::CHICKEN_BREAST_COOKED: return {165.0, 31.0, 0.0, 3.6, 1.0, 1.5};
        case FoodItem::WHITE_RICE_COOKED:     return {130.0, 2.7, 28.0, 0.3, 0.9, 2.5};
        case FoodItem::BROCCOLI_RAW:          return {34.0, 2.8, 6.6, 0.4, 0.35, 3.0};
        case FoodItem::EGG_LARGE_COOKED:      return {155.0, 13.0, 1.1, 11.0, 1.03, 2.5};
    }
    return {};
}

// =======================================================================
// PART 2: real portion-size estimation from a real reference object --
// a standard dinner plate of known real diameter -- reported honestly
// as a range rather than a single fabricated gram figure.
// =======================================================================
constexpr double REFERENCE_PLATE_DIAMETER_CM = 26.0;
constexpr double THICKNESS_UNCERTAINTY_FRACTION = 0.20;

struct PortionEstimate {
    double low_g = 0.0, mid_g = 0.0, high_g = 0.0;
};

PortionEstimate estimate_portion_range(double food_area_px, double plate_diameter_px, FoodItem food) {
    double plate_area_cm2 = std::numbers::pi * std::pow(REFERENCE_PLATE_DIAMETER_CM / 2.0, 2);
    double plate_area_px = std::numbers::pi * std::pow(plate_diameter_px / 2.0, 2);
    double scale_cm2_per_px2 = plate_area_cm2 / plate_area_px;
    double food_area_cm2 = food_area_px * scale_cm2_per_px2;

    NutrientProfile profile = nutrient_profile_for(food);
    double mass_mid_g = food_area_cm2 * profile.assumed_thickness_cm * profile.density_g_per_cm3;
    return {
        mass_mid_g * (1.0 - THICKNESS_UNCERTAINTY_FRACTION),
        mass_mid_g,
        mass_mid_g * (1.0 + THICKNESS_UNCERTAINTY_FRACTION),
    };
}

// =======================================================================
// PART 3: daily nutrient tally over human-confirmed log entries -- the
// portion ESTIMATE above is a range the user reviews and confirms
// before logging, exactly this book's own established human-in-the-loop
// discipline (Sections 20.1 and 20.4), which is what lets the tally
// itself stay exact and deterministic rather than needing to propagate
// a range through every downstream sum.
// =======================================================================
struct FoodLogEntry {
    FoodItem food;
    double confirmed_grams = 0.0;
};

struct NutrientTotals {
    double calories = 0.0, protein_g = 0.0, carbs_g = 0.0, fat_g = 0.0;
};

NutrientTotals daily_nutrient_tally(const std::vector<FoodLogEntry>& entries) {
    NutrientTotals totals;
    for (const auto& entry : entries) {
        NutrientProfile p = nutrient_profile_for(entry.food);
        double factor = entry.confirmed_grams / 100.0;
        totals.calories += p.calories_per_100g * factor;
        totals.protein_g += p.protein_g_per_100g * factor;
        totals.carbs_g += p.carbs_g_per_100g * factor;
        totals.fat_g += p.fat_g_per_100g * factor;
    }
    return totals;
}

enum class IntakeClassification { BELOW_RANGE, WITHIN_RANGE, ABOVE_RANGE };

struct DailyTargetRange {
    double calories_min = 0.0, calories_max = 0.0;
};

IntakeClassification classify_daily_intake(double total_calories, DailyTargetRange range) {
    if (total_calories < range.calories_min) return IntakeClassification::BELOW_RANGE;
    if (total_calories > range.calories_max) return IntakeClassification::ABOVE_RANGE;
    return IntakeClassification::WITHIN_RANGE;
}

// =======================================================================
// PART 4: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 25.4: Camera-Based Nutrition Tracking via a USDA-Style Offline Reference\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: the offline reference subset's own stated real values are exactly what "
                 "this section commits to for a representative sample --\n";
    {
        CHECK(near(nutrient_profile_for(FoodItem::APPLE_RAW).calories_per_100g, 52.0));
        CHECK(near(nutrient_profile_for(FoodItem::CHICKEN_BREAST_COOKED).calories_per_100g, 165.0));
        CHECK(near(nutrient_profile_for(FoodItem::CHICKEN_BREAST_COOKED).protein_g_per_100g, 31.0));
        CHECK(near(nutrient_profile_for(FoodItem::EGG_LARGE_COOKED).fat_g_per_100g, 11.0));
        std::cout << "  apple (raw) is exactly 52 kcal per 100g; chicken breast (cooked) is exactly "
                     "165 kcal and exactly 31.0g protein per 100g; egg (large, cooked) is exactly "
                     "11.0g fat per 100g -- this section's own stated reference values, checked "
                     "directly\n";
    }

    std::cout << "\n-- Test 2: every food in the closed vocabulary has a real, non-degenerate reference "
                 "entry -- an exhaustive check, not a hand-picked sample --\n";
    {
        int checked = 0;
        for (FoodItem food : ALL_FOOD_ITEMS) {
            NutrientProfile p = nutrient_profile_for(food);
            CHECK(p.calories_per_100g > 0.0);
            CHECK(p.density_g_per_cm3 > 0.0);
            CHECK(p.assumed_thickness_cm > 0.0);
            checked++;
        }
        CHECK(checked == 6);
        std::cout << "  all " << checked << " foods in the closed vocabulary have a real, positive "
                     "calorie count, density, and assumed serving thickness -- no food silently falls "
                     "through to a missing or zeroed-out reference entry\n";
    }

    std::cout << "\n-- Test 3: real portion-mass estimation from a real reference-plate scale matches "
                 "an exact, hand-computed value --\n";
    {
        // A 2600px plate diameter against the real 26cm reference plate is exactly a 100px/cm scale,
        // so a real (26/2600)^2 = 0.0001 cm^2/px^2 area scale, independent of pi.
        auto chicken = estimate_portion_range(800000.0, 2600.0, FoodItem::CHICKEN_BREAST_COOKED);
        // 800000px^2 * 0.0001 = 80 cm^2; 80 * 1.5cm thickness * 1.0 g/cm^3 density = 120.0g mid estimate.
        CHECK(near(chicken.mid_g, 120.0));
        CHECK(near(chicken.low_g, 96.0));   // 120 * 0.8
        CHECK(near(chicken.high_g, 144.0)); // 120 * 1.2

        auto banana = estimate_portion_range(500000.0, 2600.0, FoodItem::BANANA_RAW);
        // 500000px^2 * 0.0001 = 50 cm^2; 50 * 2.0cm * 0.5 g/cm^3 = 50.0g mid estimate.
        CHECK(near(banana.mid_g, 50.0));
        CHECK(near(banana.low_g, 40.0));
        CHECK(near(banana.high_g, 60.0));
        std::cout << "  an 80cm^2 real chicken-breast area at a real 2600px reference-plate diameter "
                     "estimates to exactly 120.0g at the midpoint, with an honest [96.0g, 144.0g] real "
                     "range; a 50cm^2 real banana area under the identical scale estimates to exactly "
                     "50.0g at the midpoint, with an honest [40.0g, 60.0g] range\n";
    }

    std::cout << "\n-- Test 4: the real low/mid/high ordering of a portion estimate holds across every "
                 "food in the closed vocabulary, not merely the hand-picked examples above --\n";
    {
        int checked = 0;
        for (FoodItem food : ALL_FOOD_ITEMS) {
            auto est = estimate_portion_range(600000.0, 2600.0, food);
            CHECK(est.low_g < est.mid_g);
            CHECK(est.mid_g < est.high_g);
            checked++;
        }
        CHECK(checked == 6);
        std::cout << "  across all 6 foods in the closed vocabulary, the real low estimate is strictly "
                     "less than the real mid estimate, which is strictly less than the real high "
                     "estimate, with zero exceptions -- this section never collapses its own honest "
                     "range into a single value anywhere in the table\n";
    }

    std::cout << "\n-- Test 5: a daily tally over several real, human-confirmed log entries sums to an "
                 "exact, hand-computed total across every real macro --\n";
    {
        std::vector<FoodLogEntry> entries = {
            {FoodItem::CHICKEN_BREAST_COOKED, 150.0},
            {FoodItem::WHITE_RICE_COOKED, 200.0},
            {FoodItem::BROCCOLI_RAW, 100.0},
        };
        auto totals = daily_nutrient_tally(entries);
        // chicken 150g: 247.5 kcal, 46.5g protein, 0g carbs, 5.4g fat
        // rice 200g:    260.0 kcal, 5.4g protein, 56.0g carbs, 0.6g fat
        // broccoli 100g: 34.0 kcal, 2.8g protein, 6.6g carbs, 0.4g fat
        CHECK(near(totals.calories, 541.5));
        CHECK(near(totals.protein_g, 54.7));
        CHECK(near(totals.carbs_g, 62.6));
        CHECK(near(totals.fat_g, 6.4));
        std::cout << "  150g chicken breast + 200g white rice + 100g broccoli, all human-confirmed "
                     "gram amounts, tally to exactly 541.5 kcal, 54.7g protein, 62.6g carbs, and 6.4g "
                     "fat -- matching a real hand computation across every entry and every macro\n";
    }

    std::cout << "\n-- Test 6: daily-intake classification against a real stated target range is "
                 "correct on both sides and inclusive at each real boundary --\n";
    {
        DailyTargetRange range{1800.0, 2200.0};
        CHECK(classify_daily_intake(1500.0, range) == IntakeClassification::BELOW_RANGE);
        CHECK(classify_daily_intake(2000.0, range) == IntakeClassification::WITHIN_RANGE);
        CHECK(classify_daily_intake(2500.0, range) == IntakeClassification::ABOVE_RANGE);
        CHECK(classify_daily_intake(1800.0, range) == IntakeClassification::WITHIN_RANGE);  // lower bound, inclusive
        CHECK(classify_daily_intake(2200.0, range) == IntakeClassification::WITHIN_RANGE);  // upper bound, inclusive
        CHECK(classify_daily_intake(1799.9, range) == IntakeClassification::BELOW_RANGE);
        CHECK(classify_daily_intake(2200.1, range) == IntakeClassification::ABOVE_RANGE);
        std::cout << "  against a real stated 1800-2200 kcal target range: 1500 kcal classifies "
                     "BELOW_RANGE, 2000 kcal WITHIN_RANGE, 2500 kcal ABOVE_RANGE, and both real "
                     "boundary values themselves, 1800 and 2200 kcal, correctly classify WITHIN_RANGE "
                     "-- with 1799.9 and 2200.1 immediately outside on either side, confirming the "
                     "boundary is genuinely inclusive rather than accidentally so\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_camera_based_nutrition_tracking.cpp -o 04_camera_based_nutrition_tracking
./04_camera_based_nutrition_tracking
```

**Sample input:** the offline reference subset checked against this section's own stated real values for a representative sample; an exhaustive, all-six-food sanity check confirming no food has a degenerate reference entry; real portion-mass estimation checked against an exact hand-computed value and range; the real low/mid/high ordering checked across the entire closed food vocabulary; a daily tally over several confirmed log entries checked against an exact hand-computed total across every macro; and daily-intake classification checked on both sides of a real stated target range, inclusive at each boundary.

```text
========================================================
Chapter 25.4: Camera-Based Nutrition Tracking via a USDA-Style Offline Reference
========================================================

-- Test 1: the offline reference subset's own stated real values are exactly what this section commits to for a representative sample --
  apple (raw) is exactly 52 kcal per 100g; chicken breast (cooked) is exactly 165 kcal and exactly 31.0g protein per 100g; egg (large, cooked) is exactly 11.0g fat per 100g -- this section's own stated reference values, checked directly

-- Test 2: every food in the closed vocabulary has a real, non-degenerate reference entry -- an exhaustive check, not a hand-picked sample --
  all 6 foods in the closed vocabulary have a real, positive calorie count, density, and assumed serving thickness -- no food silently falls through to a missing or zeroed-out reference entry

-- Test 3: real portion-mass estimation from a real reference-plate scale matches an exact, hand-computed value --
  an 80cm^2 real chicken-breast area at a real 2600px reference-plate diameter estimates to exactly 120.0g at the midpoint, with an honest [96.0g, 144.0g] real range; a 50cm^2 real banana area under the identical scale estimates to exactly 50.0g at the midpoint, with an honest [40.0g, 60.0g] range

-- Test 4: the real low/mid/high ordering of a portion estimate holds across every food in the closed vocabulary, not merely the hand-picked examples above --
  across all 6 foods in the closed vocabulary, the real low estimate is strictly less than the real mid estimate, which is strictly less than the real high estimate, with zero exceptions -- this section never collapses its own honest range into a single value anywhere in the table

-- Test 5: a daily tally over several real, human-confirmed log entries sums to an exact, hand-computed total across every real macro --
  150g chicken breast + 200g white rice + 100g broccoli, all human-confirmed gram amounts, tally to exactly 541.5 kcal, 54.7g protein, 62.6g carbs, and 6.4g fat -- matching a real hand computation across every entry and every macro

-- Test 6: daily-intake classification against a real stated target range is correct on both sides and inclusive at each real boundary --
  against a real stated 1800-2200 kcal target range: 1500 kcal classifies BELOW_RANGE, 2000 kcal WITHIN_RANGE, 2500 kcal ABOVE_RANGE, and both real boundary values themselves, 1800 and 2200 kcal, correctly classify WITHIN_RANGE -- with 1799.9 and 2200.1 immediately outside on either side, confirming the boundary is genuinely inclusive rather than accidentally so

53/53 checks passed
ALL CHECKS PASSED
```

!!! warning "[COMMON TRAP] reporting a single estimated gram figure because a range is less satisfying to look at"
    A single number -- "127 grams of chicken" -- looks more like a finished, confident product than a range does, and it is tempting to just report `estimate_portion_range`'s own `mid_g` value and quietly drop the rest. The real problem is that this section's own portion estimate rests on a stated, openly acknowledged assumption (an average serving thickness for each food) that this system has no real way to verify from a single photograph -- reporting only the midpoint presents an assumption as a measurement. Test 4's own whole-vocabulary check exists to keep this section honest about that: every food's own real range has genuine width, and the user reviewing and confirming an actual gram figure before it enters the daily tally (rather than this system silently trusting its own midpoint) is what keeps the eventual logged total trustworthy.

## Chapter Summary

This chapter closed Part 5 by building two real, privacy-respecting camera pipelines end to end. Section 25.1 built a real frame-sampling strategy and a scene-tagging data model whose own types make facial recognition structurally impossible, proven with a real C++20 concept and a compile-time `static_assert` rather than promised in a comment. Section 25.2 turned those tags into a real chronological incident report, reusing Section 22.1's own presence-interval coalescing technique, an honest "no notable activity" line for real gaps, and a from-scratch FNV-1a chain-of-custody checksum verified against its own published reference vectors. Section 25.3 built a real joint-angle formula and a real rep-counting state machine honest enough to refuse counting a rep that never reached true depth. Section 25.4 closed the chapter with a fully on-device nutrition tracker against a small, stated offline reference, estimating portion size as an honest range rather than a fabricated point figure, and relying on this book's own established human-in-the-loop discipline to keep its daily tally exact.

## Self-Check Questions

1. Section 25.1's `HasIdentityField` concept and its `static_assert` enforce the no-facial-recognition boundary at compile time. Explain concretely what a determined future contributor would have to do to violate this boundary, and why that is a meaningfully higher bar than violating a policy stated only in a comment or a document.
2. Section 25.1's frame sampler drops a motion trigger reported outside the real recorded frame range rather than clamping it to the nearest valid frame. Explain why clamping would be the wrong choice here.
3. Section 25.2's `draft_report` sorts its own input before processing rather than assuming it already arrived in chronological order. Construct a concrete scenario where skipping that sort would produce a genuinely wrong report.
4. Section 25.2's chain-of-custody hash is computed over the ENTIRE frame record, not just the frames that carry a scene tag. Explain what real tampering scenario this choice specifically defends against that hashing only the tagged frames would miss.
5. Section 25.3's `RepCounter` distinguishes `DESCENDING` from `BOTTOM` as two separate real states. Explain what specific real behavior would change, and become wrong, if these two states were collapsed into one.
6. Section 25.3's Test 3 checks a real mid-ascent dip back into full depth. Explain concretely why a simpler counter that increments the moment the angle first reaches `STANDING_ANGLE_MIN_DEG` after leaving `BOTTOM` would double-count this scenario.
7. Section 25.4 reports portion size as a real `[low, mid, high]` range rather than a single gram figure. Name the specific real assumption in the portion-estimation formula that this range is meant to honestly acknowledge.
8. Section 25.4's daily tally works from `confirmed_grams` rather than directly from `estimate_portion_range`'s own output. Explain why this design choice is what allows `daily_nutrient_tally` to remain a simple, exact sum.
9. Both halves of this chapter -- the body camera and the personal health tracker -- are described as privacy-first by architecture rather than by policy. Identify one concrete design choice from each half of the chapter that embodies this, and explain what would have to change about the design (not just a stated policy) to violate it.
10. Section 25.2 reuses Section 22.1's own presence-interval coalescing technique unchanged. Explain what property of that original technique made it reusable here in a completely different domain (incident reporting rather than surveillance narration) without modification.

## Where We Go Next

Part 5 is complete: eight chapters spanning industrial inspection, retail, medical imaging, document intelligence, security and accessibility, point-of-sale trust, natural-language photo editing, and now body-worn and personal cameras, each turning a continuous visual stream into a structured, checkable, honestly-scoped output. Part 6 turns from application domains to the mathematical and architectural foundations underneath every one of them: Chapter 30 works through the FLOPs, roofline, and Hessian mathematics that explain why these systems perform the way they do, Chapter 31 builds a real continuous-batching scheduler for serving many requests at once, and Chapter 32 takes this book's own inference engine to the GPU with a real Flash Attention implementation and CUDA kernels.

## Worked Solutions

**1.** They would have to add an actual member to `PersonBox` capable of holding an identity -- a field literally named `name`, `person_id`, or `face_embedding`, since `HasIdentityField`'s `requires`-expressions check for exactly those member names. The moment they did, the `static_assert(!HasIdentityField<PersonBox>, ...)` immediately above the struct would fail, and the ENTIRE file would refuse to compile, for every contributor, on every build, with an explicit message naming the violated boundary. Violating a policy stated in a comment requires only that a reviewer fail to notice a diff; violating this boundary requires actively breaking the build for the whole project, which is a far harder thing to do by accident and a far easier thing to catch on purpose.

**2.** A motion trigger arriving at, say, frame 250 for a clip that only actually has 100 frames is not real evidence of motion in the recording -- it is bad sensor or detector data, since no such frame exists. Clamping it to frame 99 would silently insert a fabricated "motion observed here" flag on a real frame that the motion detector never actually flagged, corrupting the evidentiary record with information the system invented rather than observed. Dropping it is the honest choice: it discards data that cannot possibly be correct rather than distorting it into something that looks plausible.

**3.** Suppose two frames for the same incident are logged as `{30, 1000ms, WEAPON_VISIBLE}` and `{60, 500ms, ...}` -- perhaps because they came from two different camera systems whose own frame counters and timestamps were not perfectly synchronized, so frame 60 (a later frame INDEX) actually has an EARLIER timestamp than frame 30. Without sorting by timestamp first, `coalesce_tag_intervals` would process frame 30 before frame 60 in insertion order, computing a WEAPON_VISIBLE interval that runs backward in time (starting at 1000ms and "ending" at 500ms) or miscomputing which frames are genuinely contiguous. Sorting by `timestamp_ms` before coalescing is what guarantees the real chronological narrative is correct regardless of what order the underlying frames happened to arrive in.

**4.** Hashing only the tagged frames would miss a tampering scenario where an entire untagged stretch of footage is deleted from the record -- since removing frames that never carried any tag in the first place would not change a hash computed only over tagged frames, that deletion would go completely undetected. Hashing the FULL frame record (Section 25.2's actual `canonical_serialization`, which serializes every frame regardless of whether it carries a tag) means removing ANY frame at all, tagged or not, changes the chain-of-custody hash -- which is the real guarantee a genuine chain-of-custody checksum needs to provide.

**5.** If `DESCENDING` and `BOTTOM` were collapsed into one state, the counter would have no way to distinguish "started going down but only reached 128 degrees" from "actually reached real depth at 90 degrees" -- both would just be "the one state that isn't standing." That is exactly Test 2's own partial rep: without a separate `BOTTOM` state that specifically requires the angle to reach the real `BOTTOM_ANGLE_MAX_DEG` threshold, the counter would have no honest way to refuse counting a rep that never actually got deep enough, and every partial rep would count exactly the same as a full one.

**6.** A counter that increments the moment the angle first crosses back above `STANDING_ANGLE_MIN_DEG` after leaving `BOTTOM`, with no further state tracking, would count Test 3's own rep the instant it reaches 165 degrees during its own real ascent -- but that ascent itself contains a dip back down to 95 degrees (full real depth again) before the FINAL rise to standing. A simpler counter watching only for "crossed the standing threshold since I was last at bottom" has no way to know that the dip back to 95 degrees was a continuation of the SAME rep rather than a completely new one starting fresh -- it would see the sequence cross the standing threshold, then dip low again, then cross the standing threshold a second time, and count twice. `RepCounter`'s own explicit `BOTTOM` re-entry from `ASCENDING` is what correctly recognizes the dip as still the same repetition in progress.

**7.** The formula's own `assumed_thickness_cm` constant per food -- a stated, illustrative average-serving thickness this section openly assumes rather than measures, since a single 2D photograph cannot recover a food's actual real depth or thickness from a bounding box or segmented area alone. The `[low, mid, high]` range, built from a stated `THICKNESS_UNCERTAINTY_FRACTION`, is this section's own honest acknowledgment that the thickness figure feeding the whole calculation is itself an assumption, not a measurement, and the final gram estimate can be no more precise than that assumption allows.

**8.** Because `confirmed_grams` is a plain, already-decided number the user reviewed and accepted, `daily_nutrient_tally` never has to reason about uncertainty at all -- it is a straightforward `nutrient_per_100g * (grams / 100)` sum for each logged entry. If the tally instead worked directly from `estimate_portion_range`'s own `[low, mid, high]` output, every downstream total (calories, protein, carbs, fat) would need its own separate low/mid/high figures, and combining several uncertain entries into one daily total would require real, careful reasoning about how those individual ranges combine -- entirely avoidable complexity that Section 20.1's and 20.4's own human-confirmation step sidesteps by turning the uncertain estimate into a single decided fact before it ever reaches the tally.

**9.** For the body camera: `PersonBox`'s complete lack of any identity-shaped field (Section 25.1) is a design choice, not a policy -- a face-recognition feature literally cannot be bolted onto the existing pipeline without first changing the type itself, which the `static_assert` catches immediately. For the personal health tracker: every function across Section 25.3 and Section 25.4 is a pure, synchronous computation with no network call anywhere in either file -- there is no request object, no API client, and no code path that could transmit a frame or a meal photo anywhere, which is a stronger guarantee than a stated no-network policy layered on top of a design that technically could make one. Violating either guarantee would require rewriting the actual data types and function signatures involved, not merely changing a setting or ignoring a stated rule.

**10.** The original technique's own real generality: `coalesce_tag_intervals` never assumed anything specific to surveillance narration in the first place -- it operates purely on a sequence of (thing-present-or-not, timestamp) observations for one tracked category at a time, merging contiguous true observations within a stated tolerance into one interval. An incident's own scene tags over time are exactly that same shape of data (is `WEAPON_VISIBLE` present at this timestamp, yes or no), so the identical algorithm applies without any modification -- the technique was never actually about cameras or people in the first place, only about turning a noisy stream of boolean observations into clean, contiguous real intervals.
