# Chapter 24: Natural Language Photo Editing: From Prompt to `cv::Mat` on Edge Hardware

**What you will understand by the end of this chapter:**

- How to build a real, stated system prompt and a strict structural parser that translate any natural-language photo-edit request, however vague, into a closed, small vocabulary of named edit intents and strengths -- never a free-form description of "enhancements."
- How to build a complete, real OpenCV processing engine that executes a structured edit plan as genuine `cv::Mat` operations -- brightness, contrast, warmth, saturation, sharpening, denoising, and cropping -- verified against hand-computed pixel values and real statistical properties.
- How to build a real, tabulated resolution table mapping every closed-vocabulary intent and strength to a concrete operation parameter, checked for a real, general table-integrity property, plus a real conflict detector that catches a directly self-contradictory edit request before it ever reaches the processing engine.
- How to build a real, structural conversation-state engine that lets a user refine an edit plan turn by turn -- nudging a strength up or down, undoing a prior refinement, or replacing an intent with its own direct opposite when a follow-up request means the user changed their mind.

**What you need to know first:**

- Section 20.3's own triage-report prompt-and-parser pattern (a real, stated system prompt paired with a strict structural parser that rejects anything outside a closed vocabulary, naming the exact reason) is the exact pattern Section 24.1 reapplies to photo-edit requests.
- Section 21.3's own honest-range discipline and Section 22.1's own AMBIGUOUS-refusal discipline both recur here: Section 24.3's conflict detector refuses to silently resolve a self-contradictory plan, and Section 24.4's own refinement engine refuses to silently wrap a strength past its own real ceiling or floor.
- Chapter 19.4's own real least-squares trend-fitting reuse in Section 22.3 already established this book's own pattern of reusing one chapter's real technique unchanged in a later, unrelated domain; Section 24.3's resolution table and Section 24.2's own engine constants are built to match exactly for the identical reason.

---

Every chapter since Chapter 18 has built a real technique that turns an unstructured visual input into a structured, checkable output. This chapter turns that direction around: it takes a structured request and turns it into unstructured pixels, via a real, verifiable pipeline the whole way through. Section 24.1 builds the closed vocabulary and the strict parser that keep an ambiguous natural-language request from ever reaching an image-processing engine as anything other than a small, named, checkable set of operations. Section 24.2 builds the real engine, linking against actual OpenCV rather than a hand-rolled equivalent -- a deliberate, honestly-documented departure from every other file in this book, which depends on nothing but the C++ standard library. Section 24.3 builds the deterministic table connecting the two, plus a real conflict check. Section 24.4 closes the loop with the real conversational state a genuine photo-editing session needs: the ability to ask for "a bit more," to undo, and to change one's mind without the system holding two contradictory instructions at once.

## 24.1 An Edit-Interpretation Prompt and Structured Plan Parser

### Intuition

"Make it look better" names no operation, no direction, and no strength -- a real image-processing engine cannot execute it. This section builds the same real discipline Section 20.3 applied to a triage report: a stated system prompt that forces any natural-language request, however vague, into a closed, small vocabulary of named edit intents and strengths, plus a strict parser that rejects anything outside that vocabulary, naming the exact reason.

### The Concept, In Detail

The stated `EDIT_INTERPRETATION_SYSTEM_PROMPT` commits to exactly 12 named edit intents and exactly 3 named strength levels, and -- critically -- commits in advance to a specific, conservative default mapping for any vague request naming no specific edit, rather than leaving that case to the model's own unconstrained judgment. `parse_edit_plan` is the strict, structural downstream parser: Test 2 through Test 5 confirm an unrecognized intent token, an unrecognized strength token, a duplicate intent, and a structurally malformed line are each rejected with their own specific, named reason, exactly the same discipline as Section 20.3's own triage-report parser and Section 21.1's own structured extraction parser.

Test 6 is this section's own central honesty check: the EXACT text the system prompt itself commits to for a vague request parses to exactly the stated conservative 3-intent default plan (a subtle contrast increase, a subtle saturation increase, and a subtle sharpen) -- never a more aggressive, unrequested combination the model might otherwise be tempted to produce for an ambiguous instruction.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_edit_interpretation_prompt_and_plan_parser.cpp -o 01_edit_interpretation_prompt_and_plan_parser
./01_edit_interpretation_prompt_and_plan_parser
```

**Sample input:** a valid 2-intent completion checked to parse to exactly its own entries in order, and an empty request checked to parse to a valid, empty plan; an intent token and a strength token outside the closed vocabulary each rejected by name; the same intent appearing twice in one plan rejected by name; a structurally malformed line rejected by name; and the system prompt's own exact stated default-plan text for a vague request checked to parse to exactly the conservative 3-entry default plan.

@@OUT1@@

!!! warning "[COMMON TRAP] treating a vague request as a case the parser, not the prompt, should handle"
    It is tempting to let a vague request like "make it look better" pass through unconstrained -- have the parser accept any reasonable-looking completion the model produces, on the theory that a vague REQUEST justifies a flexible RESPONSE. Test 6 exists specifically to show why that gets the division of responsibility backwards: this section's own system prompt commits, in its own stated text, to one specific, conservative, named default plan for exactly this case, and the parser's own job is to verify that the model actually produced THAT plan (or another fully valid one), never to relax its own closed vocabulary just because the originating request happened to be unspecific. A vague request is a prompt-engineering problem to solve with a stated default, not a parsing problem to solve by accepting more.

## 24.2 A Complete OpenCV Processing Engine

### Intuition

A structured edit plan is only useful once something executes it as real pixel operations. This section builds that engine on genuine OpenCV `cv::Mat` calls, verified against hand-computed expected values wherever the underlying arithmetic is exact, and against real, checkable statistical properties -- mean, standard deviation -- wherever it is not.

### The Concept, In Detail

`apply_brightness` and `apply_contrast` use `cv::Mat::convertTo` directly: Test 1 confirms brightness is a pure, exact integer offset with zero rounding ambiguity at all 3 real strength levels, and Test 2 confirms contrast's own real `(old-128)*alpha+128` formula produces exact hand-computed values, including a real clamp at the 255 ceiling when a STRONG increase would otherwise overflow it. `apply_warmth` splits real BGR channels and shifts red and blue in exactly opposite directions by an exact integer amount, confirmed in Test 3, and `apply_saturation` performs a real BGR-to-HSV-and-back round trip, with Test 4 confirming full desaturation produces an exactly neutral gray (all 3 channels equal) while a partial decrease still leaves real, measurably reduced color.

`apply_sharpen` uses a real, from-scratch plus-shaped kernel whose own weights are constructed to sum to exactly 1 -- Test 5 confirms this real mean-preserving property holds (before and after means differ by less than 2.0 on a real checkerboard) while standard deviation rises, a real, checkable signature of increased local contrast at edges. `apply_denoise` and `apply_crop_center_square` are confirmed in Test 6 and Test 7 against real statistical and geometric expectations, and Test 8 confirms the full "make it look better" default plan, applied end to end, produces a real, measurably different image at the identical resolution.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_opencv_edit_processing_engine.cpp -isystem /usr/include/opencv4 -o 02_opencv_edit_processing_engine -lopencv_imgproc -lopencv_core
./02_opencv_edit_processing_engine
```

**Sample input:** brightness and contrast checked against exact hand-computed values including a real clamp at the 255 ceiling; warmth checked to shift red and blue in exactly opposite real directions while leaving green untouched; full desaturation checked to produce an exactly neutral gray while a partial decrease leaves real, measurable color; sharpening checked to approximately preserve mean brightness while increasing standard deviation on a real checkerboard; denoising checked to reduce standard deviation, more so at a larger real kernel size; center-square cropping checked against exact output dimensions and a marker-pixel offset check; and the full default plan applied end to end checked to produce a real, non-identical result at the same resolution.

@@OUT2@@

!!! warning "[COMMON TRAP] expecting this section's own verification to match this book's usual 4-way architecture check"
    Every other file in this book depends on nothing but the C++ standard library, which is what lets this book's own usual pipeline cross-compile and run each one, completely unmodified, across 4 separate CPU architectures and toolchains. OpenCV is a real, large, dynamically-linked SYSTEM library that must be built or installed natively per real target platform -- there is no portable static build this book can simply cross-compile the way it cross-compiles its own self-contained code, and this section's own real target device has no root access to install one there. This section is verified instead on 2 real compilers (this system's own default GCC and GCC 14), both linking the identical real, installed OpenCV 4.6.0 -- confirming this section's own real pixel arithmetic is deterministic across compiler versions -- rather than this book's usual 4-way check. This is not a gap quietly papered over: it is a real, honest constraint of depending on a real external system library, and a genuine edge deployment would resolve it the same way -- installing or building OpenCV natively for each specific target device, exactly as this section's own limitation implies.

## 24.3 A Request-to-Operation Mapping Table

### Intuition

Section 24.1's closed vocabulary and Section 24.2's real engine need a deterministic bridge: an explicit table mapping every (intent, strength) pair to one concrete numeric parameter, plus a real check that a plan asking for two directly opposing edits at once is caught before it ever reaches the engine.

### The Concept, In Detail

`resolve_operation_params` is the real, tabulated resolution table, confirmed in Test 1 against hand-computed values matching Section 24.2's own engine constants exactly, and Test 2 confirms `DESATURATE_FULL` and `CROP_CENTER_SQUARE` resolve to the identical real value regardless of which strength token accompanies them -- their own stated real absoluteness, verified directly rather than merely asserted in a comment. Test 3 is this section's own general table-integrity check: across all 10 strength-sensitive intents in the entire table, a STRONG parameter's own real distance from its op family's neutral value is strictly greater than MODERATE's, which is strictly greater than SUBTLE's -- checked programmatically across the WHOLE table at once, not merely for a hand-picked example.

`detect_conflicting_intents` checks a real, stated table of directly opposing intent pairs -- Test 4 confirms a plan combining `BRIGHTNESS_INCREASE` and `BRIGHTNESS_DECREASE`, or `SATURATION_INCREASE` and `DESATURATE_FULL`, is flagged, while a plan combining 3 genuinely unrelated intents is not. Test 5 confirms a conflict-free plan resolves cleanly through `resolve_plan_to_operations` to its own exact, real sequence of concrete parameters.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_request_to_operation_mapping_table.cpp -o 03_request_to_operation_mapping_table
./03_request_to_operation_mapping_table
```

**Sample input:** the resolution table checked against hand-computed constants across every op family; `DESATURATE_FULL` and `CROP_CENTER_SQUARE` checked to resolve identically at all 3 strength levels; a general table-integrity property (STRONG's own magnitude of change strictly exceeds MODERATE's, which strictly exceeds SUBTLE's) checked programmatically across all 10 strength-sensitive intents at once; directly opposing intent pairs checked to be flagged while unrelated intents are not; and a conflict-free plan checked to resolve to its own exact, real sequence of concrete operation parameters.

@@OUT3@@

!!! warning "[COMMON TRAP] checking a table's own integrity with a handful of hand-picked examples instead of the whole table"
    It would be a real, meaningful test to confirm that STRONG produces a bigger effect than SUBTLE for a couple of intents chosen by hand -- brightness, say, and saturation -- and stop there. Test 3 is built specifically to do more: it iterates over EVERY strength-sensitive intent in the entire table and checks the identical real property for each one, catching a mistake in an intent nobody thought to hand-check individually (a copy-paste error in the `WARMTH_DECREASE` row, for instance, that a spot check of `BRIGHTNESS_INCREASE` and `SATURATION_INCREASE` would never have caught). A resolution table this size earns a real, general, whole-table property check precisely because a table is exactly the kind of structure where one silently wrong row hides behind nine correct ones.

## 24.4 Iterative Refinement Through Conversation

### Intuition

A real photo-editing conversation is rarely one request. It is a request followed by "a bit more," "actually undo that," or "no, make it cooler instead" -- and each of those needs a real, structural answer, not a fresh, independent reparse that forgets everything that came before.

### The Concept, In Detail

`apply_refinement`'s `INCREASE_STRENGTH` and `DECREASE_STRENGTH` actions step a strength up or down one real level at a time -- Test 1 confirms an increase caps honestly at `CAPPED_AT_MAX` rather than wrapping back to SUBTLE once STRONG is reached, and Test 2 confirms a decrease past SUBTLE removes the intent entirely (`REMOVED_AT_MIN`) rather than leaving it unchanged, a real, sensible interpretation of "less than the smallest amount." `UNDO_LAST` reverts exactly one refinement at a time through a real history stack, confirmed in Test 3 to revert one step per call and to honestly report `NOTHING_TO_UNDO`, leaving the plan unchanged, once that stack is empty.

Test 5 is this section's own central discipline, and its own point of genuine contrast with Section 24.3: a follow-up request for `WARMTH_DECREASE` against a plan already containing `WARMTH_INCREASE` does not get flagged as a conflict the way Section 24.3's own fresh-plan detector correctly would -- it REPLACES the existing intent, because a follow-up in an ongoing conversation is the user changing their mind about a specific prior instruction, not a second, independent, simultaneously-held contradiction. Test 6 confirms this whole engine composes correctly across a full multi-turn conversation, including a replace and an undo in the same session, with undo reverting exactly the one turn it should and no more.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_iterative_refinement_conversation.cpp -o 04_iterative_refinement_conversation
./04_iterative_refinement_conversation
```

**Sample input:** repeated "more" requests checked to step strength up one real level at a time, capping honestly at STRONG; a "less" request on an already-SUBTLE intent checked to remove it entirely; undo checked to revert exactly one refinement per call through a real history stack, and to honestly refuse once that stack is empty; a genuinely new, non-conflicting intent checked to simply apply; a follow-up request directly opposing an intent already in the plan checked to replace it rather than coexist with it; and a full multi-turn conversation, including a replace and an undo, checked against its own exact final plan state.

@@OUT4@@

!!! warning "[COMMON TRAP] applying Section 24.3's own flag-a-conflict rule inside a live conversation"
    Section 24.3's own `detect_conflicting_intents` is the right behavior for a single, freshly-parsed plan with no history behind it: if a request produces `BRIGHTNESS_INCREASE` and `BRIGHTNESS_DECREASE` in the SAME breath, that really is a contradiction worth flagging for clarification, since nothing establishes which one the user actually meant. Applying that identical rule inside an ONGOING conversation would be a real regression: Test 5's own "make it cooler" follow-up, checked against a plan already containing "make it warmer" from a previous turn, is not two contradictory instructions issued at once -- it is the SAME real property being revised, later, by the same user. Section 24.4's own REPLACE behavior exists because a conversation's own history changes what a new, opposing request MEANS; treating every later utterance as though it arrived with no context at all would make ordinary, real conversational refinement -- "actually, cooler" -- impossible to express without the system flagging the user's own change of mind as an error.

## Chapter Summary

This chapter built a complete, real pipeline from a vague natural-language photo-edit request to real, executed pixel operations and back through a live conversation. Section 24.1 built a stated system prompt and a strict structural parser that force any request into a closed, checkable vocabulary, committing in advance to a specific conservative default for vague requests rather than leaving that case unconstrained. Section 24.2 built a real, genuine OpenCV processing engine -- this book's first section to depend on an external system library rather than the C++ standard library alone -- verified across 2 real compilers against the identical real OpenCV installation, with an honestly documented, real reason why this book's usual 4-way architecture check does not apply here. Section 24.3 built the deterministic resolution table connecting the two, checked with a real, general whole-table integrity property, plus a real conflict detector for a single, freshly-parsed plan. Section 24.4 closed the loop with a real conversational refinement engine whose own central discipline -- replacing rather than flagging an intent a later turn directly opposes -- is a deliberate, well-justified departure from Section 24.3's own rule, because a conversation's own history changes what a new request means.

## Self-Check Questions

1. Section 24.1's system prompt commits, in its own stated text, to a specific default plan for a vague request. Explain why placing that decision in the PROMPT rather than the PARSER is the correct division of responsibility.
2. Section 24.1's parser rejects a duplicate intent rather than silently keeping the last occurrence. Construct a concrete scenario where silently keeping the last occurrence would produce a genuinely different, and wrong, result from rejecting it outright.
3. Section 24.2's sharpening kernel is specifically constructed so its own weights sum to exactly 1. Explain what real, visible artifact would appear in a repeatedly sharpened image if the kernel's weights instead summed to, say, 1.1.
4. Section 24.2's own COMMON TRAP box explains why this section does not receive this book's usual 4-way architecture check. Name the specific real constraint that makes OpenCV different from every other file in this book, and explain why a statically-linked cross-compiled OpenCV binary would not actually solve it for a real edge deployment either.
5. Section 24.3's Test 3 checks its own table-integrity property across all 10 strength-sensitive intents at once rather than a hand-picked sample. Explain concretely what kind of real mistake this whole-table check would catch that a 2-intent spot check would miss.
6. Section 24.3 marks `DESATURATE_FULL` and `CROP_CENTER_SQUARE` as `strength_invariant`. Explain why "STRONGLY desaturate to full" is not a coherent instruction in the way "STRONGLY increase brightness" is.
7. Section 24.4's `DECREASE_STRENGTH` action removes an already-SUBTLE intent entirely rather than leaving it unchanged. Explain why leaving it unchanged would misrepresent what the user actually asked for.
8. Section 24.4's Test 5 REPLACES an opposing intent rather than flagging it the way Section 24.3's detector would. Using the "make it warmer" then "make it cooler" example, explain the one piece of real information available to Section 24.4 that is NOT available to Section 24.3's own fresh-plan detector, and why that piece of information is what justifies the different behavior.
9. Section 24.4's `UNDO_LAST` reverts exactly one refinement per call through a real history stack. Construct a concrete 3-refinement conversation where undoing twice produces a DIFFERENT result than simply re-parsing the original request from scratch would.
10. Across all 4 sections in this chapter, identify the one recurring real discipline that also appeared in Chapter 21.3's `CostRange` and Chapter 22.1's `AMBIGUOUS` outcome, and explain how it shows up differently in Section 24.1's default-plan commitment versus Section 24.4's replace-on-conflict behavior.

## Where We Go Next

This chapter built a complete, real pipeline from an ambiguous natural-language request to genuine, verified pixel operations and back through a live, stateful conversation -- and along the way, took on this book's first real external system-library dependency, with an honestly documented account of what that costs in cross-architecture verification. Chapter 25 turns to a closely related real domain with a different, sharper constraint: body-worn and personal cameras, building police body-camera scene tagging and report drafting under an explicit no-facial-recognition legal boundary with a real frame-sampling strategy for edge deployment, alongside exercise-form analysis and camera-based nutrition tracking under a privacy-first architecture for personal health devices.

## Worked Solutions

**1.** The prompt is the component that actually GENERATES a response to an ambiguous request, so it is the only place a specific, considered default can be chosen deliberately and reviewed in advance -- the parser's own job is strictly to verify that whatever came back is well-formed and within the closed vocabulary, a mechanical check that has no way to know whether a particular set of 3 intents is a REASONABLE response to "make it look better" versus an arbitrary but structurally valid one. Moving the default into the parser would mean the parser silently substitutes its own judgment for the model's whenever a request seems vague, which conflates two genuinely separate concerns: deciding what a reasonable response looks like, and checking that a given response is well-formed.

**2.** A user submits a request that a model translates into two edits for the same real intent because it genuinely could not tell which one the user meant -- for instance, an ambiguous phrase that could plausibly map to either `WARMTH_INCREASE` at MODERATE or at STRONG, and the model (incorrectly, but plausibly) emits both due to its own uncertainty. Silently keeping the last occurrence would apply STRONG without any indication that the model's own output was internally inconsistent about what the user wanted; rejecting the duplicate surfaces that real inconsistency immediately, giving the system a chance to ask for clarification rather than guessing which of the two genuinely different real edits to apply.

**3.** A kernel whose own weights sum to more than 1 amplifies overall brightness every time it is applied, not just local contrast at edges -- repeatedly sharpening the same image (a real, common workflow when a user asks to "sharpen it more" several times in a row) would make the image progressively BRIGHTER and more washed out with each pass, a real, visible artifact having nothing to do with the actual edge-enhancement the user asked for, purely because the kernel's own arithmetic does not preserve the image's real overall light level the way a properly normalized (sum-to-1) kernel does.

**4.** The specific real constraint is that OpenCV is a large, dynamically-linked SYSTEM library that must be built or installed natively per target platform, unlike every other file in this book, which depends on nothing but the portable C++ standard library. A statically-linked cross-compiled OpenCV binary would not actually solve this for a real edge deployment because a real deployment needs OpenCV's own platform-specific optimizations (NEON intrinsics on ARM, hardware-accelerated color conversion, a vendor's own tuned build for their specific SoC) to run acceptably fast on real, constrained edge hardware -- a generic, statically cross-compiled build sacrifices exactly the platform-specific tuning that makes running OpenCV on real edge hardware worthwhile in the first place, so even if this book COULD produce one, it would not represent what a real deployment should actually do.

**5.** A whole-table check would catch a mistake in any ONE of the other 8 intents nobody happened to hand-pick for a spot check -- for instance, a copy-paste error where `WARMTH_DECREASE`'s own MODERATE and STRONG values were accidentally swapped (16 and 28 reversed), which would make MODERATE's own distance from neutral LARGER than STRONG's for that one specific intent. A 2-intent spot check testing only, say, `BRIGHTNESS_INCREASE` and `SATURATION_INCREASE` would report success while this real, specific error sat undetected in a third intent nobody thought to verify individually -- exactly the kind of mistake a table of a dozen near-identical-looking rows is prone to, and exactly what iterating over the WHOLE table closes off.

**6.** "Strongly increase brightness" describes a MATTER OF DEGREE -- brightness can be increased a little or a lot, and STRONG names a real, specific point along that continuum. "Fully desaturate" already names an absolute endpoint -- zero saturation, a real, complete removal of color -- and there is no possible state MORE desaturated than completely gray; asking to do it "strongly" versus "subtly" describes no real difference in the resulting image, because the destination is already the same regardless of how emphatically it is requested, which is exactly why the operation is coherent only as strength-invariant.

**7.** A user who says "actually, less bright" about an edit that is already at the smallest real amount this system offers (SUBTLE) is expressing that even that smallest amount was too much -- leaving it unchanged at SUBTLE would apply an edit the user has now explicitly said they do not want any part of, misrepresenting their own most recent, most specific statement of intent. Removing the intent entirely is the only response that actually reflects what "less than the least" means in a system with a real, finite lower bound: there is no smaller positive amount to fall back to, so the honest next step is none at all.

**8.** The one piece of real information available to Section 24.4 that Section 24.3's own detector cannot see is CONVERSATION HISTORY -- specifically, that `WARMTH_INCREASE` was not merely present in a static snapshot but was ADDED by this same user in an earlier real turn of this same ongoing conversation. That history is what licenses the inference that a later, opposing request is a REVISION of that specific earlier choice rather than an independent, simultaneous instruction; Section 24.3's detector, operating on a single freshly-parsed plan with no memory of how it came to contain what it contains, has no basis to distinguish "the user wants both directions at once" (worth flagging) from "the user changed their mind" (worth replacing), so it can only, correctly, flag the co-occurrence and let a higher-level system with access to history make the distinction Section 24.4 is built to make.

**9.** Start with a plan containing `SHARPEN` at SUBTLE. Refinement 1: increase `SHARPEN` to MODERATE. Refinement 2: add `WARMTH_INCREASE` at MODERATE. Refinement 3: increase `SHARPEN` to STRONG. Undoing twice from this point reverts refinement 3 (back to `SHARPEN` at MODERATE) and then refinement 2 (removing `WARMTH_INCREASE` entirely) -- leaving a plan with only `SHARPEN` at MODERATE. Re-parsing the ORIGINAL request from scratch, by contrast, would reproduce the very first plan: `SHARPEN` at SUBTLE, with no `WARMTH_INCREASE` ever having existed and no memory that `SHARPEN` was ever bumped to MODERATE along the way. The two results share the absence of `WARMTH_INCREASE`, but disagree on `SHARPEN`'s own strength (MODERATE after 2 undos, versus SUBTLE from a fresh reparse) -- undo reverts to an intermediate REAL state the conversation actually passed through, while a fresh reparse has no such memory at all.

**10.** The recurring discipline is committing, in advance and explicitly, to a specific real behavior for a case that could otherwise be resolved arbitrarily or silently -- Chapter 21.3's `CostRange` commits to reporting an honest range rather than collapsing to a false-precision point, and Chapter 22.1's `AMBIGUOUS` outcome commits to naming every fitting candidate rather than silently picking one. In Section 24.1, this shows up as committing, in the SYSTEM PROMPT's own stated text, to one specific conservative default plan for a vague request, rather than leaving that case for the model to resolve however it sees fit at request time. In Section 24.4, the identical discipline shows up in the OPPOSITE form: rather than committing to always flagging a contradiction (the safer-looking default), it commits to REPLACING an opposing intent specifically because conversational history changes what the contradiction actually means -- the same underlying commitment to a deliberate, considered, explicitly justified real behavior, chosen in advance rather than resolved arbitrarily in the moment, applied to two situations whose own real context calls for opposite concrete responses.
