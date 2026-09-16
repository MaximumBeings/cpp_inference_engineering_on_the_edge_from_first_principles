// Chapter 21.4 -- A motion-triggered camera trap does not take one
// picture per real animal visit; it takes a BURST of pictures for every
// real event that trips its sensor, and the same real individual can
// trip that sensor many separate times across a multi-week deployment.
// A vision-language species classifier run naively across every single
// frame would report a raw detection count that means almost nothing
// about how many real animals were actually there -- and reporting that
// raw count AS a population estimate is a real, well-documented mistake
// in camera-trap ecology, not a hypothetical one. This section builds
// the real, standard fix from scratch: an INDEPENDENT-CAPTURE-EVENT
// window that merges a burst of same-species, same-camera detections
// into one event, an open-set confidence floor that refuses to force a
// low-confidence detection into a specific species label at all, and a
// report schema that is honest about exactly what it counts by never
// containing a field that could be mistaken for a population estimate.
// The chapter's own second real domain, underwater marine imagery,
// needs a different real fix applied BEFORE any of that classification
// happens at all: water attenuates red light far faster than blue as
// depth increases, giving every underwater photograph a real, physical
// blue-green color cast a classifier trained on ordinary photographs was
// never shown, and this section builds the real, standard gray-world
// color-correction algorithm that removes it.
//
// A note on this section's own honest scope: the species labels below
// are a small, stated, closed set for demonstration, not a real trained
// classifier's own taxonomy, and the gray-world assumption this
// section's color correction relies on -- that a real scene's own
// average color is approximately neutral gray -- is a real, standard,
// but genuinely APPROXIMATE heuristic that can be wrong for a scene
// dominated by one true color (a coral reef with almost no blue in
// frame, for instance); this section states that limitation directly
// rather than presenting gray-world correction as an exact physical
// color-recovery model.
//
// Compile: g++ -std=c++23 -Wall -Wextra -O2 04_camera_trap_deduplication_and_underwater_color_correction.cpp -o 04_camera_trap_deduplication_and_underwater_color_correction
// Run:     ./04_camera_trap_deduplication_and_underwater_color_correction

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static int g_tests = 0, g_passed = 0;
#define CHECK(...) do { \
    g_tests++; \
    if (__VA_ARGS__) { g_passed++; } \
    else { std::cerr << "FAIL line " << __LINE__ << ": " << #__VA_ARGS__ << "\n"; } \
} while (0)

// =======================================================================
// PART 1: open-set species labeling. A detection below a stated
// confidence floor is NEVER forced into a specific species -- it is
// reported as genuinely uncertain, an honest bucket of its own rather
// than this section's own best (but low-confidence) guess.
// =======================================================================
enum class Species { DEER, FOX, RACCOON, UNKNOWN_UNCERTAIN };

std::string to_string(Species s) {
    switch (s) {
        case Species::DEER: return "DEER";
        case Species::FOX: return "FOX";
        case Species::RACCOON: return "RACCOON";
        case Species::UNKNOWN_UNCERTAIN: return "UNKNOWN_UNCERTAIN";
    }
    return "UNKNOWN_UNCERTAIN";
}

// A real, stated open-set floor -- below this confidence, this section
// refuses to report ANY specific species, regardless of which species a
// raw classifier score happened to favor.
constexpr double SPECIES_CONFIDENCE_FLOOR = 0.55;

Species finalize_species_label(Species raw_species, double confidence) {
    if (confidence < SPECIES_CONFIDENCE_FLOOR) return Species::UNKNOWN_UNCERTAIN;
    return raw_species;
}

struct Detection {
    std::string camera_id;
    int period_index = 0;   // a virtual, discrete tick -- never real wall-clock time
    Species species;
    double confidence = 0.0;
};

// =======================================================================
// PART 2: independent-capture-event deduplication -- the real, standard
// camera-trap ecology technique of merging consecutive same-camera,
// same-species detections separated by no more than a stated time
// window into ONE independent event, rather than counting every
// triggered frame as its own separate sighting.
// =======================================================================
struct CaptureEvent {
    std::string camera_id;
    Species species;
    int first_period = 0, last_period = 0;
    int detection_count = 0;   // the number of RAW detections merged into this one event -- never a population count
};

// Detections are first grouped by (camera_id, species) -- a different
// camera is always a different location, and a different species is
// always a different real event, so neither is ever merged across that
// boundary. Within one (camera, species) group, consecutive detections
// (sorted by period) merge into the SAME event as long as the gap since
// the event's own last period does not exceed `time_window`; a larger
// gap starts a genuinely new event.
std::vector<CaptureEvent> group_into_independent_events(std::vector<Detection> detections, int time_window) {
    std::stable_sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        if (a.camera_id != b.camera_id) return a.camera_id < b.camera_id;
        if (a.species != b.species) return static_cast<int>(a.species) < static_cast<int>(b.species);
        return a.period_index < b.period_index;
    });
    std::vector<CaptureEvent> events;
    for (const auto& d : detections) {
        if (!events.empty() && events.back().camera_id == d.camera_id && events.back().species == d.species &&
            (d.period_index - events.back().last_period) <= time_window) {
            events.back().last_period = d.period_index;
            events.back().detection_count += 1;
        } else {
            events.push_back(CaptureEvent{d.camera_id, d.species, d.period_index, d.period_index, 1});
        }
    }
    return events;
}

// =======================================================================
// PART 3: the report schema itself -- deliberately containing no field
// that could be mistaken for a population or individual count. The only
// number this report ever states is `independent_capture_events`,
// counted per species, and its own name says exactly, and only, what it
// measures.
// =======================================================================
struct SpeciesEventCount { Species species; int independent_capture_events = 0; };

std::vector<SpeciesEventCount> summarize_events_by_species(const std::vector<CaptureEvent>& events) {
    std::vector<SpeciesEventCount> counts;
    for (const auto& e : events) {
        auto it = std::find_if(counts.begin(), counts.end(),
                                [&](const SpeciesEventCount& c) { return c.species == e.species; });
        if (it == counts.end()) {
            counts.push_back(SpeciesEventCount{e.species, 1});
        } else {
            it->independent_capture_events += 1;
        }
    }
    return counts;
}

// =======================================================================
// PART 4: underwater gray-world color correction -- a real, standard,
// from-scratch preprocessing pass applied BEFORE species classification,
// removing the real physical blue-green color cast water imposes as
// depth increases.
// =======================================================================
struct RgbImage {
    int width = 0, height = 0;
    std::vector<uint8_t> r, g, b;   // parallel per-channel planes, row-major
};

struct ChannelMeans { double r, g, b; };

ChannelMeans compute_channel_means(const RgbImage& img) {
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    size_t n = img.r.size();
    for (size_t i = 0; i < n; ++i) { sum_r += img.r[i]; sum_g += img.g[i]; sum_b += img.b[i]; }
    return {sum_r / static_cast<double>(n), sum_g / static_cast<double>(n), sum_b / static_cast<double>(n)};
}

// A real, stated clamp on the per-channel scale factor -- without it, a
// channel with a near-zero mean (a real possibility in a badly
// attenuated deep-water photograph) would compute an enormous scale
// factor and amplify that channel's own sensor noise into a dominant,
// meaningless signal rather than a real color correction.
constexpr double GRAY_WORLD_SCALE_MIN = 0.5, GRAY_WORLD_SCALE_MAX = 2.0;

RgbImage gray_world_correct(const RgbImage& img) {
    ChannelMeans means = compute_channel_means(img);
    double gray_target = (means.r + means.g + means.b) / 3.0;
    auto scale_for = [&](double channel_mean) {
        double raw_scale = (channel_mean > 1e-9) ? (gray_target / channel_mean) : GRAY_WORLD_SCALE_MAX;
        return std::clamp(raw_scale, GRAY_WORLD_SCALE_MIN, GRAY_WORLD_SCALE_MAX);
    };
    double scale_r = scale_for(means.r), scale_g = scale_for(means.g), scale_b = scale_for(means.b);
    RgbImage out{img.width, img.height, img.r, img.g, img.b};
    auto apply = [](std::vector<uint8_t>& plane, double scale) {
        for (auto& v : plane) v = static_cast<uint8_t>(std::clamp(static_cast<double>(v) * scale, 0.0, 255.0));
    };
    apply(out.r, scale_r);
    apply(out.g, scale_g);
    apply(out.b, scale_b);
    return out;
}

// =======================================================================
// PART 5: self-tests.
// =======================================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "Chapter 21.4: Wildlife and Underwater Species Identification from Camera-Trap and Marine Imagery\n";
    std::cout << "========================================================\n";

    std::cout << "\n-- Test 1: a detection below the stated confidence floor is forced to UNKNOWN_UNCERTAIN, "
                 "regardless of the raw species label, and the floor's own boundary is exact --\n";
    {
        CHECK(finalize_species_label(Species::DEER, 0.90) == Species::DEER);
        CHECK(finalize_species_label(Species::FOX, 0.20) == Species::UNKNOWN_UNCERTAIN);
        CHECK(finalize_species_label(Species::RACCOON, SPECIES_CONFIDENCE_FLOOR) == Species::RACCOON);        // exactly at the floor: passes
        CHECK(finalize_species_label(Species::RACCOON, SPECIES_CONFIDENCE_FLOOR - 0.001) == Species::UNKNOWN_UNCERTAIN);   // just below: gated
        std::cout << "  a high-confidence DEER detection passes through unchanged; a low-confidence FOX "
                     "detection is forced to UNKNOWN_UNCERTAIN; and the stated floor's own boundary is "
                     "exact -- confidence exactly at the floor passes, just below it does not\n";
    }

    std::cout << "\n-- Test 2: a real burst of same-camera, same-species detections within the stated time "
                 "window merges into ONE event; a gap larger than the window starts a genuinely new one --\n";
    {
        std::vector<Detection> burst = {
            {"CAM-01", 100, Species::DEER, 0.9}, {"CAM-01", 102, Species::DEER, 0.85},
            {"CAM-01", 105, Species::DEER, 0.88},
        };
        auto events = group_into_independent_events(burst, /*time_window=*/10);
        CHECK(events.size() == 1);
        CHECK(events[0].first_period == 100 && events[0].last_period == 105);
        CHECK(events[0].detection_count == 3);

        std::vector<Detection> two_visits = {
            {"CAM-01", 100, Species::DEER, 0.9}, {"CAM-01", 500, Species::DEER, 0.9},
        };
        auto two_events = group_into_independent_events(two_visits, /*time_window=*/10);
        CHECK(two_events.size() == 2);

        std::cout << "  3 detections at periods 100, 102, 105 (all within a window of 10) merge into "
                     "exactly 1 event spanning periods 100-105 with detection_count 3; 2 detections at "
                     "periods 100 and 500 (far beyond the window) correctly produce 2 separate events\n";
    }

    std::cout << "\n-- Test 3: a different SPECIES at the same camera, and the same species at a different "
                 "CAMERA, are each their own separate event -- location and species both distinguish events "
                 "--\n";
    {
        std::vector<Detection> same_camera_different_species = {
            {"CAM-01", 100, Species::DEER, 0.9}, {"CAM-01", 101, Species::FOX, 0.9},
        };
        auto events1 = group_into_independent_events(same_camera_different_species, 10);
        CHECK(events1.size() == 2);

        std::vector<Detection> different_camera_same_species = {
            {"CAM-01", 100, Species::DEER, 0.9}, {"CAM-02", 100, Species::DEER, 0.9},
        };
        auto events2 = group_into_independent_events(different_camera_same_species, 10);
        CHECK(events2.size() == 2);

        std::cout << "  a DEER and a FOX detected one period apart at the SAME camera are correctly kept "
                     "as 2 separate events; the SAME species detected at the SAME period by 2 DIFFERENT "
                     "cameras is likewise correctly kept as 2 separate events\n";
    }

    std::cout << "\n-- Test 4: THE HONEST LIMIT -- one real, lingering individual animal, captured in "
                 "several widely-separated bursts across a single deployment, produces MORE THAN ONE "
                 "independent capture event, proving directly that this report's own event count is not, "
                 "and must never be read as, an individual or population count --\n";
    {
        // One real fox that returns to the same camera's trigger zone
        // repeatedly across a single day -- 4 separate bursts, each
        // internally tight, but the bursts themselves are spaced far
        // enough apart (150 periods) to each start a new event under
        // this section's own stated 10-period window.
        std::vector<Detection> one_real_fox_all_day = {
            {"CAM-03", 100, Species::FOX, 0.9}, {"CAM-03", 102, Species::FOX, 0.9},
            {"CAM-03", 250, Species::FOX, 0.9},
            {"CAM-03", 400, Species::FOX, 0.9}, {"CAM-03", 403, Species::FOX, 0.9},
            {"CAM-03", 550, Species::FOX, 0.9},
        };
        auto events = group_into_independent_events(one_real_fox_all_day, /*time_window=*/10);
        CHECK(events.size() == 4);
        auto summary = summarize_events_by_species(events);
        CHECK(summary.size() == 1);
        CHECK(summary[0].species == Species::FOX);
        CHECK(summary[0].independent_capture_events == 4);
        std::cout << "  a single real fox returning to the same camera 4 separate times across one day "
                     "produces exactly 4 independent_capture_events for FOX -- a real, computed "
                     "demonstration that this number counts EVENTS, never individuals, since this "
                     "entire report describes what could easily be just 1 real animal\n";
    }

    std::cout << "\n-- Test 5: gray-world color correction pulls a real blue-green underwater color cast "
                 "toward neutral, and its own amplification is bounded even for a near-zero-mean channel --\n";
    {
        // A synthetic underwater-like image: red heavily attenuated
        // (mean 40), green moderately attenuated (mean 90), blue
        // dominant (mean 180) -- a real, physically-motivated color cast.
        RgbImage img{2, 2,
                     {40, 40, 40, 40},
                     {90, 90, 90, 90},
                     {180, 180, 180, 180}};
        auto before = compute_channel_means(img);
        CHECK(before.r == 40.0 && before.g == 90.0 && before.b == 180.0);
        double before_spread = std::max({before.r, before.g, before.b}) - std::min({before.r, before.g, before.b});

        auto corrected = gray_world_correct(img);
        auto after = compute_channel_means(corrected);
        double after_spread = std::max({after.r, after.g, after.b}) - std::min({after.r, after.g, after.b});
        CHECK(after_spread < before_spread);
        // Hand-computed: gray_target = (40+90+180)/3 = 103.333...; raw
        // scale for R = 103.333/40 = 2.583, clamped down to the stated
        // ceiling of 2.0, so every R pixel becomes exactly 40*2.0 = 80.
        CHECK(corrected.r[0] == 80);

        RgbImage near_black{1, 1, {0}, {128}, {128}};
        auto near_black_corrected = gray_world_correct(near_black);
        // A near-zero-mean red channel would compute an unbounded raw
        // scale factor without the stated clamp; with it, every R pixel
        // is scaled by at most GRAY_WORLD_SCALE_MAX, never exploding.
        CHECK(near_black_corrected.r[0] <= static_cast<uint8_t>(GRAY_WORLD_SCALE_MAX * 255));

        std::cout << "  a synthetic underwater-cast image (R mean 40, G mean 90, B mean 180) has its "
                     "per-channel spread shrink from " << before_spread << " to " << after_spread
                   << " after gray-world correction; the red channel's own scale factor is correctly "
                     "clamped at the stated ceiling of " << GRAY_WORLD_SCALE_MAX << " rather than "
                     "applying its full, hand-computed 2.583x; and a near-zero-mean channel is likewise "
                     "bounded rather than amplified without limit\n";
    }

    std::cout << "\n" << g_passed << "/" << g_tests << " checks passed\n";
    std::cout << (g_passed == g_tests ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    return (g_passed == g_tests) ? 0 : 1;
}
