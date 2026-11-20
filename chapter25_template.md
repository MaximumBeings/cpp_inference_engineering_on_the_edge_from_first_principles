# Chapter 25: Body-Worn and Personal Cameras: Evidence Documentation and Health Tracking

<!-- edge:opener:start -->
<figure class="chapter-figure" markdown="1">

[![Chapter 25: Sample camera streams into reviewable evidence](../images/chapter-openers/25.svg?v=6620ce66d042)](../images/chapter-openers/25.svg?v=6620ce66d042)

<figcaption>Chapter roadmap. Open the image for a full-size view.</figcaption>
</figure>
<!-- edge:opener:end -->

**What you will understand by the end of this chapter:**

- How to combine periodic and motion-triggered frame sampling with a scene-tag schema and a narrowly scoped compile-time field-name guard.
- How to render chronological annotation reports without confusing missing observations with inactivity, and demonstrate a non-security annotation checksum.
- How to build a real joint-angle formula from 2D keypoints and a real rep-counting state machine with genuine hysteresis, honest enough to refuse counting a rep that never actually reached real depth.
- How to build a privacy-first, fully on-device nutrition tracker against a small, stated, offline reference subset of real USDA FoodData Central-style values, with portion size estimated from a real reference-object scale and reported honestly as a range rather than a single fabricated gram figure.

**What you need to know first:**

`PersonBox` currently contains only geometry and a clothing-role enum. `HasIdentityField` rejects exactly three member names (`name`, `person_id`, `face_embedding`) at compile time. This is a narrow schema regression guard: a differently named identity field or recognition elsewhere in the pipeline is not excluded. Privacy requires separate data-flow, retention and access-control review.
- Section 22.1's own presence-interval coalescing (turning raw per-frame detections into contiguous real time intervals) is reused unchanged in Section 25.2, and Section 23.1's, 23.2's, and 23.4's own discipline of implementing a real, independently-famous algorithm from scratch and verifying it against a real published reference value is what Section 25.2 does again here with FNV-1a.
- Section 21.3's own CostRange discipline (report an honest range rather than a fabricated point estimate) is reapplied directly in Section 25.4 to portion-size estimation, and Sections 20.1 and 20.4's own human-in-the-loop discipline is what lets Section 25.4's daily tally stay exact rather than needing to propagate that range through every downstream sum.

---

This chapter demonstrates annotation and personal-camera computations using supplied observations. The body-camera examples do not implement facial recognition or an evidentiary custody system; the personal-camera examples contain no network transmission. These educational properties do not establish a complete deployed privacy architecture.

<!-- chapter-figure -->
<figure class="chapter-figure" markdown="1">

![Sample camera streams into reviewable evidence. Frame sampling and timestamps connect model findings to the source recording.](../images/25-overview.svg?v=9395e36b4767){: loading="lazy" width="760" }

  <figcaption>Keep the original recording and its provenance alongside generated descriptions.</figcaption>
</figure>
<!-- /chapter-figure -->

## 25.1 Body-Camera Frame Sampling and Scene Tagging Under a No-Facial-Recognition Boundary

<!-- edge:concept-25.1:start -->
<figure class="chapter-figure" markdown="1">

[![Sample frames with a limited schema guard](../images/technical/ch25/25-01.svg?v=3a2515cbb296)](../images/technical/ch25/25-01.svg?v=3a2515cbb296)

<figcaption>Three forbidden field names are checked. This does not establish pipeline-wide privacy.</figcaption>
</figure>
<!-- edge:concept-25.1:end -->


### Intuition

`PersonBox` currently contains only geometry and a clothing-role enum. `HasIdentityField` rejects exactly three member names (`name`, `person_id`, `face_embedding`) at compile time. This is a narrow schema regression guard: a differently named identity field or recognition elsewhere in the pipeline is not excluded. Privacy requires separate data-flow, retention and access-control review.

### The Concept, In Detail

`select_frames_to_process` builds a real, fixed periodic sample from the stated frame rate and base sampling frequency, then unions it with real motion-triggered frames -- deduplicating any trigger that coincides with a frame already selected periodically, and honestly dropping any trigger reported outside the actual recorded frame range as sensor noise rather than accepting it as real evidence. Test 4 confirms a real, checkable coverage guarantee: periodic-only sampling can never leave a gap larger than its own computed interval.

`PersonBox` currently contains only geometry and a clothing-role enum. `HasIdentityField` rejects exactly three member names (`name`, `person_id`, `face_embedding`) at compile time. This is a narrow schema regression guard: a differently named identity field or recognition elsewhere in the pipeline is not excluded. Privacy requires separate data-flow, retention and access-control review.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_bodycam_frame_sampling_and_scene_tagging.cpp -o 01_bodycam_frame_sampling_and_scene_tagging
./01_bodycam_frame_sampling_and_scene_tagging
```

**Sample input:** periodic and motion-triggered sampling; strict scene-tag parsing; deterministic report formatting; and a schema check for three specifically forbidden member names.

@@OUT1@@

!!! warning "[COMMON TRAP] mistaking a field-name guard for pipeline privacy"
    `PersonBox` currently contains only geometry and a clothing-role enum. `HasIdentityField` rejects exactly three member names (`name`, `person_id`, `face_embedding`) at compile time. This is a narrow schema regression guard: a differently named identity field or recognition elsewhere in the pipeline is not excluded. Privacy requires separate data-flow, retention and access-control review.

## 25.2 Incident Report Drafting and Chain of Custody

<!-- edge:concept-25.2:start -->
<figure class="chapter-figure" markdown="1">

[![An annotation checksum is not evidence integrity](../images/technical/ch25/25-02.svg?v=71bb169c622d)](../images/technical/ch25/25-02.svg?v=71bb169c622d)

<figcaption>Production evidence needs source hashes, protected digests and custody procedures beyond this example.</figcaption>
</figure>
<!-- edge:concept-25.2:end -->


### Intuition

`draft_report` sorts observations, coalesces nearby tags and labels long gaps "No tagged observations; coverage unknown". A missing tag is not proof of inactivity: sampled, unavailable or unprocessed footage can all leave gaps. This example does not establish continuous negative-observation coverage.

### The Concept, In Detail

`draft_report` sorts observations, coalesces nearby tags and labels long gaps "No tagged observations; coverage unknown". A missing tag is not proof of inactivity: sampled, unavailable or unprocessed footage can all leave gaps. This example does not establish continuous negative-observation coverage.

The example computes a noncryptographic 32-bit FNV-1a checksum of canonical annotation records. Published reference vectors and one changed-record test validate this demonstration, not collision resistance or tamper protection. It hashes no original video bytes and protects no reference digest. Evidence integrity requires secure hashes of source files and annotations, independently protected reference digests and documented custody procedures. See [RFC 9923](https://datatracker.ietf.org/doc/html/rfc9923) and [NIST IR 8387](https://nvlpubs.nist.gov/nistpubs/ir/2022/NIST.IR.8387.pdf).

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_incident_report_drafting_and_chain_of_custody.cpp -o 02_incident_report_drafting_and_chain_of_custody
./02_incident_report_drafting_and_chain_of_custody
```

**Sample input:** FNV reference vectors; interval coalescing; chronological report formatting; explicit unknown-coverage gaps; deterministic identical-record checksums and one changed-tag comparison.

@@OUT2@@

!!! warning "[COMMON TRAP] treating a quiet stretch of footage as nothing worth reporting"
    `draft_report` sorts observations, coalesces nearby tags and labels long gaps "No tagged observations; coverage unknown". A missing tag is not proof of inactivity: sampled, unavailable or unprocessed footage can all leave gaps. This example does not establish continuous negative-observation coverage. The example computes a noncryptographic 32-bit FNV-1a checksum of canonical annotation records. Published reference vectors and one changed-record test validate this demonstration, not collision resistance or tamper protection. It hashes no original video bytes and protects no reference digest. Evidence integrity requires secure hashes of source files and annotations, independently protected reference digests and documented custody procedures. See [RFC 9923](https://datatracker.ietf.org/doc/html/rfc9923) and [NIST IR 8387](https://nvlpubs.nist.gov/nistpubs/ir/2022/NIST.IR.8387.pdf).

## 25.3 Exercise-Form Analysis and Rep Counting

<!-- edge:concept-25.3:start -->
<figure class="chapter-figure" markdown="1">

[![Rep counting follows a defined state transition](../images/technical/ch25/25-03.svg?v=400fea3be040)](../images/technical/ch25/25-03.svg?v=400fea3be040)

<figcaption>Occlusion and camera angle can invalidate pose estimates; provide an uncertain state.</figcaption>
</figure>
<!-- edge:concept-25.3:end -->


### Intuition

A camera coaching a squat needs a real number for how deep a joint bent, and a real, honest way to decide whether a full repetition actually happened -- not merely whether the person moved.

### The Concept, In Detail

`joint_angle_degrees` is the real, standard formula for the angle between two vectors sharing a common vertex, computed from their dot product and magnitudes. Test 1 confirms it against four exact, hand-verifiable geometric configurations: a perpendicular pair at exactly 90 degrees, a fully straightened pair at exactly 180 degrees, a genuine 45-degree configuration, and two identical vectors at exactly 0 degrees.

`RepCounter` is a real state machine over `STANDING`, `DESCENDING`, `BOTTOM`, and `ASCENDING`, and its own central discipline is in Test 2: a real partial rep that descends to 128 degrees -- short of the stated 100-degree real depth threshold -- and returns directly to standing is honestly excluded from the count, the same "refuse rather than fabricate" discipline this book has applied to counterfeit screening, receipt reconciliation, and edit-plan parsing, now applied to a physical repetition that did not actually happen. Test 3 confirms a second, subtler real property: dipping back into full depth mid-ascent before finally standing back up is still the SAME rep in progress, not a second one, which is what real hysteresis in a state machine is for.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_exercise_form_analysis_and_rep_counting.cpp -o 03_exercise_form_analysis_and_rep_counting
./03_exercise_form_analysis_and_rep_counting
```

**Sample input:** the joint-angle formula checked against four exact geometric configurations (90, 180, 45, and 0 degrees); a real 17-frame sequence containing 3 genuine full reps and 1 real partial rep checked to count to exactly 3; a real mid-ascent dip back into full depth checked to avoid a double count; `INSUFFICIENT_DEPTH` checked against a real depth threshold inclusive at its own boundary; and `KNEE_ASYMMETRY` checked against a real left/right threshold, also inclusive at its own boundary.

@@OUT3@@

!!! warning "[COMMON TRAP] counting a rep the moment the athlete starts moving back up"
    A naive rep counter watching for "went down, now going back up" would have counted Test 2's own partial rep -- the one that only reached 128 degrees before returning to standing -- as a real completed repetition, because motion reversed direction. `RepCounter`'s own `DESCENDING` state specifically distinguishes "started descending" from "actually reached `BOTTOM`," and a rep only counts on a full real cycle that passed through `BOTTOM` on the way. The distinction matters for exactly the reason a fitness application exists in the first place: an athlete relying on this system's own rep count to know whether they hit their real training target deserves a system that only counts a repetition that actually satisfied its own stated depth requirement, not one that rewards a shortcut the system itself was supposed to catch.

## 25.4 Camera-Based Nutrition Tracking via a USDA-Style Offline Reference

<!-- edge:concept-25.4:start -->
<figure class="chapter-figure" markdown="1">

[![Nutrition estimates depend on several uncertain links](../images/technical/ch25/25-04.svg?v=82d39537a343)](../images/technical/ch25/25-04.svg?v=82d39537a343)

<figcaption>An image cannot reliably determine ingredients or exact portion mass on its own.</figcaption>
</figure>
<!-- edge:concept-25.4:end -->


### Intuition

A nutrition tracker built for a personal device cannot depend on a live network call to USDA FoodData Central for every single meal -- the whole point of a privacy-first architecture is that a photograph of dinner never needs to leave the device it was taken on. This section builds a real, complete pipeline against a small, honestly-scoped offline reference instead.

### The Concept, In Detail

`nutrient_profile_for` is a small, closed, stated subset of real, approximate USDA FoodData Central-style per-100g values -- Test 1 checks a representative sample against this section's own stated figures directly, and Test 2 confirms, exhaustively across all six foods rather than a hand-picked sample, that none of them silently falls through to a missing or zeroed-out entry.

`estimate_portion_range` is this section's own central discipline: portion mass is estimated from a real reference-object scale (a standard dinner plate's own known real diameter converts a food region's pixel area into a real area in square centimeters), but the result is reported as an honest `[low, mid, high]` range rather than a single fabricated gram figure -- Section 21.3's own CostRange discipline, reapplied here to a portion size no vision system can actually measure to the gram. Test 3 checks this against an exact, hand-computed scenario, and Test 4 confirms the real `low < mid < high` ordering holds across the entire closed food vocabulary, not merely the examples hand-picked for Test 3. `daily_nutrient_tally` and `classify_daily_intake` then work from grams the user has already reviewed and confirmed -- Sections 20.1 and 20.4's own human-in-the-loop discipline -- which is exactly what lets the tally itself stay exact and deterministic rather than needing to propagate an estimate's own uncertainty through every downstream sum.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_camera_based_nutrition_tracking.cpp -o 04_camera_based_nutrition_tracking
./04_camera_based_nutrition_tracking
```

**Sample input:** the offline reference subset checked against this section's own stated real values for a representative sample; an exhaustive, all-six-food sanity check confirming no food has a degenerate reference entry; real portion-mass estimation checked against an exact hand-computed value and range; the real low/mid/high ordering checked across the entire closed food vocabulary; a daily tally over several confirmed log entries checked against an exact hand-computed total across every macro; and daily-intake classification checked on both sides of a real stated target range, inclusive at each boundary.

@@OUT4@@

!!! warning "[COMMON TRAP] reporting a single estimated gram figure because a range is less satisfying to look at"
    A single number -- "127 grams of chicken" -- looks more like a finished, confident product than a range does, and it is tempting to just report `estimate_portion_range`'s own `mid_g` value and quietly drop the rest. The real problem is that this section's own portion estimate rests on a stated, openly acknowledged assumption (an average serving thickness for each food) that this system has no real way to verify from a single photograph -- reporting only the midpoint presents an assumption as a measurement. Test 4's own whole-vocabulary check exists to keep this section honest about that: every food's own real range has genuine width, and the user reviewing and confirming an actual gram figure before it enters the daily tally (rather than this system silently trusting its own midpoint) is what keeps the eventual logged total trustworthy.

## Chapter Summary

This chapter demonstrates frame sampling, a limited annotation schema guard, interval reports and non-security checksums, followed by joint-angle, repetition-counting and nutrition examples. Unknown coverage, identity privacy and evidentiary integrity require additional architecture beyond these demonstrations.

## Self-Check Questions

1. Which three member names does `HasIdentityField` check, and why does this not rule out facial recognition elsewhere?
2. Section 25.1's frame sampler drops a motion trigger reported outside the real recorded frame range rather than clamping it to the nearest valid frame. Explain why clamping would be the wrong choice here.
3. Section 25.2's `draft_report` sorts its own input before processing rather than assuming it already arrived in chronological order. Construct a concrete scenario where skipping that sort would produce a genuinely wrong report.
4. What does hashing all supplied annotation records cover, and what video-integrity and custody properties remain untested?
5. Section 25.3's `RepCounter` distinguishes `DESCENDING` from `BOTTOM` as two separate real states. Explain what specific real behavior would change, and become wrong, if these two states were collapsed into one.
6. Section 25.3's Test 3 checks a real mid-ascent dip back into full depth. Explain concretely why a simpler counter that increments the moment the angle first reaches `STANDING_ANGLE_MIN_DEG` after leaving `BOTTOM` would double-count this scenario.
7. Section 25.4 reports portion size as a real `[low, mid, high]` range rather than a single gram figure. Name the specific real assumption in the portion-estimation formula that this range is meant to honestly acknowledge.
8. Section 25.4's daily tally works from `confirmed_grams` rather than directly from `estimate_portion_range`'s own output. Explain why this design choice is what allows `daily_nutrient_tally` to remain a simple, exact sum.
9. Both halves of this chapter -- the body camera and the personal health tracker -- are described as privacy-first by architecture rather than by policy. Identify one concrete design choice from each half of the chapter that embodies this, and explain what would have to change about the design (not just a stated policy) to violate it.
10. Section 25.2 reuses Section 22.1's own presence-interval coalescing technique unchanged. Explain what property of that original technique made it reusable here in a completely different domain (incident reporting rather than surveillance narration) without modification.

## Where We Go Next

This chapter closes out the book's original run of vision-language deployment domains -- eight chapters spanning industrial inspection, retail, medical imaging, document intelligence, security and accessibility, point-of-sale trust, natural-language photo editing, and now body-worn and personal cameras, each turning a continuous visual stream into a structured, checkable, honestly-scoped output. Part 5 continues, though: Chapter 26 turns to a different real financial setting entirely -- check and invoice fraud detection at a bank branch or back office, where the fraud patterns are well-documented and often checkable by a real published formula or a real structural redundancy the payment instrument's own printed form was designed to provide.

## Worked Solutions

**1.** `PersonBox` currently contains only geometry and a clothing-role enum. `HasIdentityField` rejects exactly three member names (`name`, `person_id`, `face_embedding`) at compile time. This is a narrow schema regression guard: a differently named identity field or recognition elsewhere in the pipeline is not excluded. Privacy requires separate data-flow, retention and access-control review.

**2.** A motion trigger arriving at, say, frame 250 for a clip that only actually has 100 frames is not real evidence of motion in the recording -- it is bad sensor or detector data, since no such frame exists. Clamping it to frame 99 would silently insert a fabricated "motion observed here" flag on a real frame that the motion detector never actually flagged, corrupting the evidentiary record with information the system invented rather than observed. Dropping it is the honest choice: it discards data that cannot possibly be correct rather than distorting it into something that looks plausible.

**3.** Suppose two frames for the same incident are logged as `{30, 1000ms, WEAPON_VISIBLE}` and `{60, 500ms, ...}` -- perhaps because they came from two different camera systems whose own frame counters and timestamps were not perfectly synchronized, so frame 60 (a later frame INDEX) actually has an EARLIER timestamp than frame 30. Without sorting by timestamp first, `coalesce_tag_intervals` would process frame 30 before frame 60 in insertion order, computing a WEAPON_VISIBLE interval that runs backward in time (starting at 1000ms and "ending" at 500ms) or miscomputing which frames are genuinely contiguous. Sorting by `timestamp_ms` before coalescing is what guarantees the real chronological narrative is correct regardless of what order the underlying frames happened to arrive in.

**4.** The example computes a noncryptographic 32-bit FNV-1a checksum of canonical annotation records. Published reference vectors and one changed-record test validate this demonstration, not collision resistance or tamper protection. It hashes no original video bytes and protects no reference digest. Evidence integrity requires secure hashes of source files and annotations, independently protected reference digests and documented custody procedures. See [RFC 9923](https://datatracker.ietf.org/doc/html/rfc9923) and [NIST IR 8387](https://nvlpubs.nist.gov/nistpubs/ir/2022/NIST.IR.8387.pdf).

**5.** If `DESCENDING` and `BOTTOM` were collapsed into one state, the counter would have no way to distinguish "started going down but only reached 128 degrees" from "actually reached real depth at 90 degrees" -- both would just be "the one state that isn't standing." That is exactly Test 2's own partial rep: without a separate `BOTTOM` state that specifically requires the angle to reach the real `BOTTOM_ANGLE_MAX_DEG` threshold, the counter would have no honest way to refuse counting a rep that never actually got deep enough, and every partial rep would count exactly the same as a full one.

**6.** A counter that increments the moment the angle first crosses back above `STANDING_ANGLE_MIN_DEG` after leaving `BOTTOM`, with no further state tracking, would count Test 3's own rep the instant it reaches 165 degrees during its own real ascent -- but that ascent itself contains a dip back down to 95 degrees (full real depth again) before the FINAL rise to standing. A simpler counter watching only for "crossed the standing threshold since I was last at bottom" has no way to know that the dip back to 95 degrees was a continuation of the SAME rep rather than a completely new one starting fresh -- it would see the sequence cross the standing threshold, then dip low again, then cross the standing threshold a second time, and count twice. `RepCounter`'s own explicit `BOTTOM` re-entry from `ASCENDING` is what correctly recognizes the dip as still the same repetition in progress.

**7.** The formula's own `assumed_thickness_cm` constant per food -- a stated, illustrative average-serving thickness this section openly assumes rather than measures, since a single 2D photograph cannot recover a food's actual real depth or thickness from a bounding box or segmented area alone. The `[low, mid, high]` range, built from a stated `THICKNESS_UNCERTAINTY_FRACTION`, is this section's own honest acknowledgment that the thickness figure feeding the whole calculation is itself an assumption, not a measurement, and the final gram estimate can be no more precise than that assumption allows.

**8.** Because `confirmed_grams` is a plain, already-decided number the user reviewed and accepted, `daily_nutrient_tally` never has to reason about uncertainty at all -- it is a straightforward `nutrient_per_100g * (grams / 100)` sum for each logged entry. If the tally instead worked directly from `estimate_portion_range`'s own `[low, mid, high]` output, every downstream total (calories, protein, carbs, fat) would need its own separate low/mid/high figures, and combining several uncertain entries into one daily total would require real, careful reasoning about how those individual ranges combine -- entirely avoidable complexity that Section 20.1's and 20.4's own human-confirmation step sidesteps by turning the uncertain estimate into a single decided fact before it ever reaches the tally.

**9.** `PersonBox` currently contains only geometry and a clothing-role enum. `HasIdentityField` rejects exactly three member names (`name`, `person_id`, `face_embedding`) at compile time. This is a narrow schema regression guard: a differently named identity field or recognition elsewhere in the pipeline is not excluded. Privacy requires separate data-flow, retention and access-control review. The personal-camera examples make no network calls, but deployment dependencies and later changes still require review.

**10.** The original technique's own real generality: `coalesce_tag_intervals` never assumed anything specific to surveillance narration in the first place -- it operates purely on a sequence of (thing-present-or-not, timestamp) observations for one tracked category at a time, merging contiguous true observations within a stated tolerance into one interval. An incident's own scene tags over time are exactly that same shape of data (is `WEAPON_VISIBLE` present at this timestamp, yes or no), so the identical algorithm applies without any modification -- the technique was never actually about cameras or people in the first place, only about turning a noisy stream of boolean observations into clean, contiguous real intervals.
