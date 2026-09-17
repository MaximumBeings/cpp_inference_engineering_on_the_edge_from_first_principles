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

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_bodycam_frame_sampling_and_scene_tagging.cpp -o 01_bodycam_frame_sampling_and_scene_tagging
./01_bodycam_frame_sampling_and_scene_tagging
```

**Sample input:** periodic sampling checked against an exact computed interval; real motion-triggered frames checked to merge, dedupe, and stay sorted against periodic coverage; an out-of-range motion trigger checked to be honestly dropped; a real worst-case coverage-gap bound checked directly; the closed scene-tag vocabulary's strict parser checked to accept every real tag and reject an unrecognized one by name; a frame's own report line checked against an exact expected string; and `PersonBox`'s own structural inability to hold an identity checked at runtime against the same concept a compile-time `static_assert` already enforced.

@@OUT1@@

!!! warning "[COMMON TRAP] treating the no-facial-recognition boundary as a policy to document rather than a type to design"
    It is tempting to build `PersonBox` first with whatever fields seem useful, then write a comment or a compliance document saying "this system does not perform facial recognition." That leaves the actual guarantee resting entirely on every future contributor reading and honoring a comment. Section 25.1's own `HasIdentityField` concept and its `static_assert` move that guarantee into the type system itself: a future edit that added a `.name`, `.person_id`, or `.face_embedding` field to `PersonBox` would not produce a policy violation to be caught in review -- it would produce a compiler error, on every build, for every contributor, forever. A real legal and ethical boundary this important deserves a mechanism stronger than a comment nobody is forced to read.

## 25.2 Incident Report Drafting and Chain of Custody

### Intuition

A body camera's own raw per-frame tags are not a report a human can read or a court can trust. This section turns them into a real chronological narrative, honestly labels any real gap in activity, and computes a real, tamper-evident checksum proving exactly what record a report was built from.

### The Concept, In Detail

`coalesce_tag_intervals` reapplies Section 22.1's own presence-interval coalescing technique unchanged: contiguous same-tag frames within a real, stated tolerance merge into one interval, and Test 2 confirms a genuine 18-second gap between two `WEAPON_VISIBLE` observations correctly starts a new, separate interval rather than merging into one implausibly long one. `draft_report` sorts its own input before processing -- Test 4 confirms identical output whether the input frames arrive in real chronological order or deliberately shuffled -- and inserts an explicit "No notable activity" line for any real gap exceeding this section's own stated threshold, rather than silently omitting empty stretches of the timeline the way a naive renderer might.

Test 1 and Test 6 are this section's own central real-algorithm work: a from-scratch 32-bit FNV-1a implementation, verified against its own real, published reference vectors exactly the way Section 23.1's dHash, Section 23.2's Levenshtein distance, and Section 23.4's Luhn checksum were each verified against their own famous reference values. `chain_of_custody_hash` applies this real checksum to a canonical serialization of an entire incident's frame record, and Test 6 proves it is genuinely tamper-evident: an identical copy of a record hashes identically, while silently removing a single tag from a single frame changes the hash.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_incident_report_drafting_and_chain_of_custody.cpp -o 02_incident_report_drafting_and_chain_of_custody
./02_incident_report_drafting_and_chain_of_custody
```

**Sample input:** a from-scratch FNV-1a implementation checked against its own real, published reference vectors; presence-interval coalescing checked to merge contiguous tags and split across a genuine gap; a coalesced interval's narrative line checked against an exact mm:ss-formatted string; report drafting checked to produce identical output from sorted and deliberately shuffled input; a real silent gap checked to produce an explicit "No notable activity" line; and the chain-of-custody hash checked to be identical for an identical record and different the moment a single tag on a single frame is altered.

@@OUT2@@

!!! warning "[COMMON TRAP] treating a quiet stretch of footage as nothing worth reporting"
    A report generator that only ever emits a line when a scene tag is present looks, at a glance, like it is simply being efficient -- why print anything about a period where nothing of note happened? The real problem is that a report with silent gaps is indistinguishable from a report built from an incomplete or tampered recording: a reviewer reading it later cannot tell "nothing happened here" apart from "this section of footage never made it into the report." Test 5 exists specifically to demonstrate the fix: an explicit, honestly labeled "No notable activity" line for every real gap means the report's own silence is never ambiguous, and the chain-of-custody hash over the complete underlying frame record (not just the frames that happened to carry a tag) is what actually backs up that the full record was considered, not merely the interesting parts of it.

## 25.3 Exercise-Form Analysis and Rep Counting

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

This chapter closes out the book's original run of vision-language deployment domains -- eight chapters spanning industrial inspection, retail, medical imaging, document intelligence, security and accessibility, point-of-sale trust, natural-language photo editing, and now body-worn and personal cameras, each turning a continuous visual stream into a structured, checkable, honestly-scoped output. Part 5 continues, though: Chapter 26 turns to a different real financial setting entirely -- check and invoice fraud detection at a bank branch or back office, where the fraud patterns are well-documented and often checkable by a real published formula or a real structural redundancy the payment instrument's own printed form was designed to provide.

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
