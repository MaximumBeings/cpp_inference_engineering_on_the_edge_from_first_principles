# Chapter 20: Medical Imaging Triage: Preliminary Radiology Reads and Human-in-the-Loop Workflow

**What you will understand by the end of this chapter:**

- Why a clinical triage system needs its human sign-off requirement built as a STRUCTURAL, type-system-enforced property of its own state machine, rather than a documented policy step a caller could forget or route around -- and how to build exactly one function capable of finalizing a case, gated on both the correct state and a real, non-empty clinician identifier.
- How to parse a real, standard subset of the DICOM file format (Explicit VR Little Endian tag/VR/length encoding) from scratch, and how to apply the exact real DICOM linear VOI LUT windowing formula a radiology workstation itself uses to turn raw pixel samples into a viewable grayscale image.
- How to design a structured triage-report schema and a strict parser that is deliberately, structurally INCAPABLE of representing a diagnosis -- restricting a queue-priority field to exactly three literal tokens and refusing any injected extra field outright -- and how to attach one unconditional, unomittable disclaimer to every report this chapter's own code can produce.
- How to extend a single case's own human-in-the-loop discipline across an entire worklist QUEUE using real optimistic concurrency, so that two radiologists racing to claim the same study cannot both succeed, and so that only the radiologist accountable for a case can ever advance or finalize it.
- How to build a real, honest explainability technique -- occlusion-based saliency, reusing Chapter 18's own frozen vision encoder unmodified -- and, just as importantly, how to prove by direct computation the specific, real limit of what that technique can and cannot tell you.

**What you need to know first:**

- Chapter 18.2's complete vision encoder (patchification, 2D RoPE, bidirectional ViT blocks, the 2x2 spatial merger), which Section 20.5 reuses byte-for-byte frozen, with no changes to any of its own five parts.
- Chapter 18.4's asymmetric threshold-based auto-disposition and Chapter 19.3's optimistic-concurrency `expected_version` PUT, both of which this chapter deliberately contrasts against or reuses in a domain where the stakes are considerably higher: Section 20.1 builds a STRUCTURAL alternative to threshold-based auto-disposition, and Section 20.4 reapplies Section 19.3's own concurrency discipline to a claim on a worklist item rather than a warehouse quantity.
- This chapter is handled with the extra care every section states directly: nothing in this chapter is legal, regulatory, or medical advice, no section claims compliance with any jurisdiction's software-as-a-medical-device framework, and every fixture, image, and report anywhere in this chapter is synthetic -- no real patient data and no real medical image appears anywhere in this book.

---

Chapter 18 taught a vision-language model to inspect a manufactured part and decide, on its own, whether the part passed. Chapter 19 taught the same core to reconcile a shelf against a planogram and raise its own findings. Medical imaging triage cannot work either way. A system that reads a scan and suggests how urgently a human should look at it is assisting a clinical decision, never making one, and this chapter states why that difference is not a matter of policy wording but of what a piece of software is structurally CAPABLE of doing on its own. This chapter builds every real piece that discipline requires, in the order a real triage pipeline would need them -- a state machine that cannot reach a final disposition without a real clinician's own sign-off, a real parser for the DICOM files a scan actually arrives in, a report format structurally incapable of smuggling a diagnosis into a queue-priority field, a worklist queue that extends the same sign-off discipline across many radiologists and many cases at once, and, finally, an honest account of how far a real explainability technique can see into the model's own reasoning, and exactly where it stops.

## 20.1 A Structural Human-in-the-Loop Triage State Machine

### Intuition

Chapter 18.4's `classify` decided, entirely on its own, whether a manufactured part passed or failed, gated only by a confidence threshold. A system suggesting how urgently a radiologist should look at a scan cannot be allowed to finalize anything on its own, at any confidence level -- and this section builds that guarantee as a property of the TYPE itself, not a rule a caller has to remember to follow.

### The Concept, In Detail

`TriageCase` moves through a real, ordered state machine -- `RECEIVED`, `AI_TRIAGE_COMPLETE`, `PENDING_HUMAN_REVIEW`, and finally `FINALIZED` or `REJECTED` -- and its own `final_disposition_` field is PRIVATE, reachable from exactly one function in the entire class: `record_human_signoff`. That function refuses to run unless the case already sits in `PENDING_HUMAN_REVIEW`, reachable only via `submit_ai_triage` followed by `route_to_review` in that order, AND unless it is given a real, non-empty clinician identifier -- Test 3 confirms an otherwise-valid sign-off attempt with an empty `clinician_id` is refused just as loudly as a wrong state, and leaves the case completely unaffected, because an unattributed sign-off is not a real audit-trail entry at all. Test 2 confirms the ordering itself is enforced, not merely the final step: attempting to finalize a case that skipped `route_to_review` entirely, and attempting to route a case that skipped `submit_ai_triage` entirely, are each refused with a specific, distinguishing error.

`submit_ai_triage` records the AI's own suggested priority into a field kept completely separate from `final_disposition_`, so a caller can never confuse what the AI suggested with what was actually decided -- and the audit log itself marks that suggestion "advisory" explicitly, in the text of the log entry, not only in a comment. A human reviewer disagreeing with the AI's own routing entirely is a real, expected outcome rather than an error: `reject_ai_triage`, tested in Test 4, moves a case to `REJECTED` with `final_disposition_` never touched, and the rejecting clinician's identity recorded by name in the audit trail. Test 5 confirms the audit trail itself is complete and correctly ordered across a full happy-path run: all four real transitions appear in sequence, with the AI's own suggestion explicitly marked advisory and the final entry naming the actual clinician who finalized the case.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_human_in_the_loop_triage_state_machine.cpp -o 01_human_in_the_loop_triage_state_machine
./01_human_in_the_loop_triage_state_machine
```

**Sample input:** a full, correctly-ordered happy path checked to reach `FINALIZED` with a 4-entry audit trail attributing the sign-off to a named clinician; finalizing a case that skipped review, and routing a case that skipped AI triage, each checked to be refused; an empty `clinician_id` checked to be refused even when the state is otherwise correct, leaving the case completely unaffected; a clinician rejecting the AI's own suggested routing outright, checked to leave `final_disposition_` unset while recording the rejecting clinician by name; and the full audit trail of a complete happy-path run checked against its exact expected 4 entries, in order.

@@OUT1@@

!!! warning "[COMMON TRAP] a threshold-based auto-disposition rule generalizing from Chapter 18.4 into a domain where it does not belong"
    Chapter 18.4's `classify` auto-accepted or auto-rejected a manufactured part once its own confidence score crossed a stated threshold, and that design was the right one for a defect-detection line where a wrong call costs a discarded part. Reapplying that same pattern here -- auto-finalizing a triage case whenever the AI's own suggested priority carries a high enough confidence score -- would be a serious, substantive error, not a matter of degree: a manufactured part can be re-inspected, but a clinical prioritization decision this section's own docstrings state plainly is not this book's to make cannot be un-made by adding more confidence-threshold precision. `TriageCase`'s own private `final_disposition_` field, reachable from exactly one gated function, is what makes the difference structural rather than a matter of choosing the right threshold value -- there is no threshold high enough that this section's own code would ever let a case reach `FINALIZED` without a real clinician's own attributed sign-off.

## 20.2 DICOM Ingestion and Windowing for Display

### Intuition

A shelf photograph in Chapter 19 arrived as an ordinary raw pixel buffer. A medical scan does not: it arrives as a real, standardized DICOM file carrying its own pixel data alongside real metadata -- dimensions, bit depth, and a WINDOW CENTER and WINDOW WIDTH the scanner or radiologist intends the image to be viewed through -- and that container has to be parsed for real before a single pixel reaches Chapter 18's own vision encoder.

### The Concept, In Detail

This section implements Explicit VR Little Endian, DICOM's single most common transfer syntax, and reads only the handful of real, standard data elements a windowed-extraction pipeline actually needs: Rows, Columns, BitsAllocated, WindowCenter, WindowWidth, and PixelData, each addressed by its own real standard tag. It states its own scope as plainly as this book has stated the scope of every other external format it has implemented a real subset of since GGUF in Chapter 5: no Implicit VR, no big-endian transfer syntax, no compressed pixel data, and 16-bit signed grayscale samples only -- correctly and verifiably implementing the real, specific slice of the standard this chapter's own windowing math depends on, not a claim to the whole of it.

`DicomReader::get_ds` correctly reverses two real encoding rules `DicomWriter` itself has to apply when producing a valid file: a DS (decimal string) value of odd length must be padded to an even length with a trailing space, which Test 5 confirms round-trips a padded `"500"` back to exactly `500.0`; and a DS element MAY carry multiple backslash-separated values, of which this section's own stated scope reads only the first, which Test 5 also confirms directly against a real three-valued `"1024\2048\4096"` string. Test 4 confirms the reader refuses a file with corrupted magic bytes outright, by name, rather than parsing corrupted input as if it were valid.

`apply_window` is the exact real DICOM linear VOI LUT formula (PS3.3, C.11.2.1.2): a sample at or below the lower clip boundary maps to the display minimum, a sample above the upper clip boundary maps to the display maximum, and everything between maps linearly with no discontinuity at either edge. Test 2 checks this formula against five hand-computed values at once -- both clip boundaries exactly, both clipped regions, and the formula's own true midpoint -- and Test 3 confirms the same formula applied through the real file-parsing path, against a synthetic file whose five raw samples were chosen specifically to span clipped-low, the lower boundary, a near-midpoint value, the upper boundary, and clipped-high, produces exactly the expected windowed byte at every one of the five positions.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_dicom_ingestion_and_windowing.cpp -o 02_dicom_ingestion_and_windowing
./02_dicom_ingestion_and_windowing
```

**Sample input:** a synthetic 4x4, 16-bit DICOM file checked to round-trip Rows, Columns, BitsAllocated, WindowCenter, and WindowWidth exactly as written; the linear VOI LUT formula checked against five hand-computed values spanning both clip boundaries, both clipped regions, and its own true midpoint; a 1x5 synthetic image with five raw samples chosen to span every one of those same cases, checked to extract to exactly the expected windowed bytes; a file with corrupted magic bytes checked to be refused by name rather than parsed; and a space-padded odd-length DS value together with a real multi-valued DS, each checked to parse correctly.

@@OUT2@@

!!! warning "[COMMON TRAP] constructing a DicomReader in the same scope as an unflushed DicomWriter"
    This section's own first working draft of Test 1 failed immediately with `std::bad_optional_access`, thrown from inside `DicomReader::open` on what should have been a valid, freshly-written file. The real cause was not a parsing bug at all: `DicomWriter`'s member `std::ofstream` had never been flushed or closed before a `DicomReader` was constructed and asked to read the exact same file path back within the same scope, so the writer's own destructor -- which would eventually close the file -- had not yet run, and the file `DicomReader::open` actually opened was zero bytes long. The fix was a real, explicit `close()` method on `DicomWriter`, called before every test constructs a reader against the same path. Two file handles pointed at the same path do not automatically observe each other's writes in the order the source code suggests; only a closed (or explicitly flushed) writer guarantees a reader opened afterward sees the complete file.

## 20.3 A Structured Triage-Reporting Prompt and Parser

### Intuition

Section 20.2 can now hand a real, windowed image to Chapter 18's vision-language pipeline. What that pipeline says back has to be constrained just as carefully as the state machine that gates its own finalization -- not merely by asking it nicely to avoid diagnosing, but by giving it a report SCHEMA that cannot represent a diagnosis even if a model's own text tried to write one into it.

### The Concept, In Detail

`render_triage_prompt` states its own constraint directly in the rendered prompt text itself -- "You are NOT diagnosing this study" and "URGENCY is a QUEUE PRIORITY for human review, not a diagnosis" -- but this section does not stop at asking politely. `parse_finding_line`'s own strict field-count check refuses ANY line carrying more than the exact four expected fields (FINDING, LOCATION, URGENCY, CONFIDENCE), which Test 2 confirms directly against a deliberately injected `"DIAGNOSIS=acute myocardial infarction"` fifth field -- refused outright, with no path for that field to survive into a `ParsedFinding` at all. The `Urgency` enum itself is restricted to exactly three literal tokens (`ROUTINE`, `EXPEDITED`, `STAT`), and Test 2 confirms both a generic invalid value (`"URGENT"`) and, more pointedly, a diagnosis-shaped string (`"PNEUMONIA"`) written directly into the URGENCY field are each refused by the identical check -- there is no way to smuggle a diagnosis into the one field the schema does allow free-form-adjacent content near, because that field's own valid values are a closed, exhaustively-checked set, not free text at all.

`build_triage_report` is the ONLY function in this file capable of constructing a `TriageReport`, and it unconditionally sets `disclaimer` to the exact fixed `MANDATORY_DISCLAIMER` string, with no parameter capable of changing or omitting it -- Test 5 confirms two reports built from completely different findings still carry byte-identical disclaimers. `overall_priority` is computed as the single MOST URGENT finding across the whole study via `urgency_rank`'s own max, never an average -- Test 4 confirms a study with two `ROUTINE` findings and one `STAT` finding correctly reports `STAT` as its overall priority, and that an empty findings list correctly defaults to `ROUTINE` rather than leaving the field in an undefined state.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_triage_report_prompt_and_parser.cpp -o 03_triage_report_prompt_and_parser
./03_triage_report_prompt_and_parser
```

**Sample input:** a valid CT chest study checked to validate and render a deterministic prompt explicitly stating it is not a diagnosis request, alongside an unknown modality checked to be refused; all three real queue-priority tiers checked to parse correctly, alongside a free-text `"URGENT"`, a diagnosis-shaped `"PNEUMONIA"` written into URGENCY, and an explicit injected `DIAGNOSIS=...` field, each checked to be refused; an out-of-range confidence and a missing field each checked to be refused; a study with mixed ROUTINE and STAT findings checked to report STAT as its overall priority, and an empty findings list checked to default to ROUTINE; and two reports with completely different findings checked to carry byte-identical, unconditional disclaimers.

@@OUT3@@

!!! warning "[COMMON TRAP] treating 'ask the model not to diagnose' as sufficient on its own"
    A system prompt that tells a vision-language model not to diagnose is a real, worthwhile instruction, and `render_triage_prompt` states exactly that instruction directly in its own rendered text. But an instruction inside a prompt is advisory in exactly the same sense Section 20.1's AI-suggested priority is advisory -- nothing about a model FOLLOWING that instruction is structurally guaranteed the way `TriageCase`'s own sign-off gate is. This section's own real answer is not a better-worded instruction; it is a report schema `parse_finding_line` enforces at the PARSING layer, entirely independent of whether the underlying model happened to follow the prompt's own request. A diagnosis-shaped string written into the URGENCY field, or appended as an extra field entirely, is refused by a strict structural check, not by hoping the model read the instruction carefully -- exactly the same shift from "ask nicely" to "structurally cannot" that Section 20.1 already made for human sign-off.

## 20.4 PACS Worklist Integration with a Human-in-the-Loop Queue

### Intuition

Section 20.1 made one case's own human sign-off structural. A real PACS worklist has to enforce a second, equally real property across an entire QUEUE of cases at once: when two radiologists are both looking at the same worklist, at most one of them may actually be working a given case, and once someone signs a case off, nothing may silently overwrite that decision.

### The Concept, In Detail

This section states its own scope the same way Chapter 18.5 stated OPC UA's: it does not model any real PACS vendor's actual wire protocol, and instead builds the one piece worth building from scratch -- a worklist whose claim-and-sign-off discipline is enforced by the type itself. `WorklistEntry::claim` is a real compare-and-set on `state_ == UNASSIGNED`, reusing exactly the optimistic-concurrency idea Section 19.3's `put_stock_level` already applied to an inventory record, now protecting which human is accountable for a study instead of a warehouse quantity. Test 2 confirms this directly: a second radiologist's claim on an already-claimed accession number is refused, and the refusal names the FIRST radiologist by identity rather than silently overwriting the claim or failing generically.

Every transition after the initial claim requires not only the correct state but the SAME radiologist who is already accountable for the case -- Test 3 confirms a second radiologist can neither begin review nor finalize a case claimed by someone else, and that `finalize` separately refuses an empty report summary even from the correctly-accountable radiologist, for the same reason Section 20.1 refused an empty `clinician_id`. Test 4 confirms a `FINALIZED` case accepts no further transition of any kind -- not a reclaim, not a re-review, not a re-finalize, and not a rejection -- leaving its original report completely untouched, while a separately-rejected case correctly records its own rejection reason. `pending_by_priority` orders the still-unclaimed queue with `std::stable_sort`, specifically so that two `STAT` studies added minutes apart are never silently reordered on a later view of the same queue -- Test 1 confirms both the priority ordering and the FIFO tie-break within a single priority tier directly.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_pacs_worklist_integration.cpp -o 04_pacs_worklist_integration
./04_pacs_worklist_integration
```

**Sample input:** four cases at three different priorities checked to surface in STAT-then-EXPEDITED-then-ROUTINE order with FIFO tie-breaking within a tier, alongside a duplicate accession number checked to be refused; a first claim checked to succeed and a second claim on the same accession checked to be refused by naming the current holder, alongside an empty radiologist id checked to be refused; a case claimed by one radiologist checked to refuse review and finalization attempts from a different radiologist, and to refuse an empty report summary even from the correct one; a finalized case checked to refuse every further transition while a separate case's rejection is correctly recorded; and a full claim-review-finalize sequence checked against its exact expected 4-entry audit trail.

@@OUT4@@

!!! warning "[COMMON TRAP] letting 'claimed' alone stand in for 'accountable'"
    It is tempting to treat `claim` as the whole of this section's own safety property -- once a radiologist has claimed a case, the thinking goes, the hard part (preventing a double-claim) is done. Test 3 exists specifically because that is not the whole property: a SECOND radiologist could, in a system that checked only the current state and not WHO is claiming it, still begin reviewing or even finalize a case someone else claimed, simply by knowing its accession number and finding it sitting in `ASSIGNED` or `IN_REVIEW`. Every one of `begin_review`, `finalize`, and `reject` in this section separately checks `radiologist_id == *claimed_by_`, not merely the state -- accountability, once established by a successful claim, has to be checked again at every single subsequent transition, not assumed to still hold just because the state looks right.

## 20.5 Occlusion-Based Saliency and the Honest Limits of Explainability

### Intuition

Every earlier section in this chapter built a real structural guarantee about WHO may act and WHEN. None of that answers a different, harder question a real deployment will be asked: why did the model suggest this study was urgent at all? This section builds one real, honest answer, and then builds the proof of exactly what that answer cannot tell you.

### The Concept, In Detail

The technique itself reuses Chapter 18.2's own vision encoder completely unmodified -- Parts 1 through 5 of this section's own file are byte-for-byte the same patchification, 2D RoPE, bidirectional ViT blocks, and 2x2 merger Chapter 18.2 already verified, because a saliency score computed against a DIFFERENT network would not actually mean anything about the one this chapter cares about. `occlusion_saliency_map` masks one patch at a time -- replacing it with the dataset's own per-channel MEAN color, which Test 5 confirms is exactly the zero vector after Part 1's own normalization, deliberately never black, since a black patch would introduce an artificial dark edge that is itself a new signal the model could react to -- reruns the frozen encoder, and measures how far the resulting whole-image representation moved via a plain L2 distance. Test 3 confirms this genuinely localizes WHERE the model is sensitive to input content: a single deliberately anomalous patch in an otherwise-uniform synthetic image scores strictly higher than every one of the fifteen ordinary background patches.

Test 4 is this section's own most important result, built specifically to prove the technique's real limit by direct computation rather than by assertion: two DIFFERENT anomalous patches, occluded separately from the same baseline image, move the whole-image representation by a comparable magnitude (a ratio of roughly 2x between the two scores, checked directly to stay under a stated factor of 5) -- but their own displacement VECTORS, compared by cosine similarity, point in substantially different directions in the model's own hidden space (measured well below a stated 0.9 similarity threshold). A scalar saliency score answers "how much did the representation move," and this section's own numbers prove directly that two very different underlying causes can produce a similar-looking answer to that question while meaning something completely different about WHAT changed and WHY -- which is exactly the honest limit this section states in its own opening comment and then proves computationally rather than merely asserting.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off 05_occlusion_saliency_and_explainability_limits.cpp -o 05_occlusion_saliency_and_explainability_limits
./05_occlusion_saliency_and_explainability_limits
```

**Sample input:** an unmodified synthetic image run twice through the frozen encoder, checked to produce a byte-identical pooled representation; a 16-patch saliency map checked for correct length, non-negativity, and genuine non-triviality; a single deliberately anomalous patch checked to score strictly higher than the highest-scoring ordinary background patch; two differently-located anomalies checked to move the representation by a comparable magnitude (ratio under 5x) while their displacement vectors point in substantially different directions (cosine similarity under 0.9); and the occlusion baseline itself checked to be exactly the zero vector, matching a real patch filled with the rounded per-channel mean pixel value to within floating-point tolerance.

@@OUT5@@

!!! warning "[COMMON TRAP] mistaking a saliency score's MAGNITUDE for an explanation of WHY"
    A saliency map that highlights "this region mattered most" is genuinely useful for deciding WHERE a radiologist's own attention might be best spent first, and this section's Test 3 proves that use is real. It is a substantively different, and considerably more dangerous, claim to treat two similarly-scored regions as having mattered for the SAME REASON, or to treat a high score as evidence the model's own suggestion was clinically well-founded rather than driven by an artifact the model happened to be sensitive to. Test 4's own two anomalies were constructed to be genuinely different causes -- different locations, different pixel values -- and still landed within a factor of 2 of each other in raw magnitude, while their real displacement vectors shared a cosine similarity of roughly 0.2, nowhere near the same direction. A displacement score is a real, computable WHERE. It was never a WHY, and this section builds no code, anywhere, that would let a caller mistake one for the other.

## Chapter Summary

This chapter took the human-in-the-loop discipline this book has treated carefully since Chapter 18.4's own threshold-based auto-disposition and rebuilt it, deliberately, as something considerably stronger: a STRUCTURAL guarantee a system is incapable of routing around, rather than a policy a caller has to remember to follow. Section 20.1 built `TriageCase`, whose only path to a final disposition runs through one gated function requiring both the correct state and a real, non-empty clinician identifier. Section 20.2 implemented a real, standard subset of the DICOM file format and the exact real DICOM linear VOI LUT windowing formula, verified against hand-computed values at every clip boundary and at the formula's own true midpoint, and found a genuine resource-lifetime bug -- an unflushed writer stream -- during its own construction. Section 20.3 built a triage-report schema and parser structurally incapable of representing a diagnosis, restricting a queue-priority field to three literal tokens and refusing any injected extra field outright, with one unconditional disclaimer attached by the only function capable of constructing a report at all. Section 20.4 extended that same sign-off discipline across an entire worklist queue using real optimistic concurrency, so that a claim on a case behaves exactly as safely under concurrent radiologists as Section 19.3's own inventory PUT behaves under concurrent writers. Section 20.5 closed the chapter with a real, honest explainability technique built on Chapter 18's own frozen vision encoder, and proved by direct computation -- not by assertion -- the specific, real limit of what a saliency score can and cannot tell a radiologist about why a model said what it said.

## Self-Check Questions

1. Section 20.1's `final_disposition_` field is private, reachable from exactly one function. Explain specifically why a documented POLICY requiring human sign-off before finalization would be a meaningfully weaker guarantee than what this section actually built, even if every caller in practice always followed that policy correctly.
2. Section 20.1's Test 2 checks that skipping EITHER `route_to_review` or the AI-triage step is refused, not only that skipping the final sign-off step is refused. Why does enforcing the full ordering matter, beyond simply gating the last step?
3. Section 20.2 needed an explicit `close()` method added to `DicomWriter` mid-construction. Explain the real bug this fixed, and why a `DicomReader` opened immediately after writing the same path could observe a zero-byte file despite the writer code appearing to have already written real data to it.
4. Section 20.2's windowing formula clips a raw sample to a display minimum or maximum outside a stated range, and maps linearly between. What would go visually wrong with a displayed image if a real implementation used a discontinuous (step) function at the clip boundaries instead of the exact linear VOI LUT formula this section implements?
5. Section 20.3's `parse_finding_line` refuses an injected `DIAGNOSIS=...` field with the same check that refuses a merely malformed line. Why is a strict field-COUNT check, rather than a check for specific forbidden field names, the more robust way to make that refusal general?
6. Section 20.3's `overall_priority` is computed as the single most urgent finding via a max operation, never an average. Construct a concrete two-finding example where averaging urgency ranks would produce a meaningfully worse (less safe) outcome than taking the max.
7. Section 20.4's `claim` alone is not sufficient to prevent an unaccountable radiologist from advancing a case, which is why `begin_review`, `finalize`, and `reject` each separately check `radiologist_id == *claimed_by_`. Why isn't checking the state alone (e.g., `state_ == ASSIGNED`) sufficient once a claim has already succeeded?
8. Section 20.4's `pending_by_priority` uses `std::stable_sort` rather than `std::sort`. Explain the specific, observable bug an ordinary unstable sort could introduce across two calls to this same function, using two STAT-priority cases as your example.
9. Section 20.5 occludes a patch with the dataset's own per-channel mean color rather than black. Explain specifically what a black-patch occlusion baseline would risk introducing that a mean-color occlusion does not.
10. Section 20.5's Test 4 checks BOTH that two anomalies' saliency scores are comparable in magnitude AND that their displacement vectors have low cosine similarity. Explain why checking only the magnitude comparison, without the cosine-similarity check, would fail to prove this section's own central claim about the limits of explainability.

## Where We Go Next

This chapter pushed this book's own human-in-the-loop discipline from a single gated function (Section 20.1) to an entire worklist queue (Section 20.4), and closed with an honest, computationally-proven account of what a real explainability technique can and cannot promise a human reviewer. The next several chapters return this book's own vision-language core to lower-stakes domains -- document intelligence, security and accessibility, trust and counterfeit detection, natural-language photo editing, and personal cameras -- where the same discipline this chapter built still applies, but where the cost of a wrong call is measured in inconvenience or expense rather than in the kind of stakes this chapter took the time to handle carefully. Chapter 21 takes this book's own vision-language pipeline into document intelligence: insurance claims, environmental compliance filings, and the real, structured extraction problem a page of dense, real-world paperwork poses to a model that has, until now, only ever looked at a shelf or a manufacturing line.

## Worked Solutions

**1.** A documented policy is only ever as reliable as the humans and processes that remember to follow it every single time, under every kind of pressure (a busy shift, an unfamiliar caller, a rushed integration) -- it can be forgotten, misread, or bypassed by a caller who simply calls a lower-level function directly. `TriageCase`'s own structural guarantee does not depend on anyone remembering anything: there is no second code path anywhere in the class, accidental or otherwise, that can ever set `final_disposition_` without passing through `record_human_signoff`'s own two checks. The difference is not about whether people are careful; it is about whether an uncareful moment -- a bug, a rushed integration, a caller who never read the policy at all -- is even CAPABLE of producing an unauthorized finalization, and with this section's own design, it categorically is not.

**2.** If only the final sign-off step were gated, a caller could still construct a case that reached `PENDING_HUMAN_REVIEW` through some other, unintended path -- skipping the AI-triage step entirely, for instance -- and a reviewer signing off on such a case would be attesting to having reviewed an AI suggestion that was never actually produced, corrupting the audit trail's own claim about what actually happened. Enforcing the FULL ordering means every field this book's own audit trail records (that an AI triage occurred, that it was explicitly routed for review, that a named clinician signed off) is guaranteed to correspond to something that genuinely happened in that order, not merely that the LAST step happened to be performed by a human.

**3.** The real bug was a resource-lifetime issue, not a logic error in the parsing code itself: `DicomWriter`'s member `std::ofstream` buffers its writes and only guarantees they reach the actual file on disk when the stream is flushed or closed, which normally happens automatically when the `ofstream`'s own destructor runs at the end of its scope. Constructing a `DicomReader` and calling `open()` on the SAME path while the `DicomWriter` object was still alive (and therefore its destructor had not yet run) meant the reader could observe the file exactly as the filesystem currently saw it -- often still zero bytes, with all of the writer's own buffered data not yet flushed to disk. The fix, an explicit `close()` method called before constructing the reader, makes the flush happen at a known point in the code rather than relying on destructor timing that the source code's own visual order does not guarantee.

**4.** A discontinuous step function at either clip boundary would create a visible, artificial hard edge in the displayed image at exactly the raw sample value where the step occurs -- two adjacent pixels differing in raw value by only 1 unit, straddling that boundary, would display as two dramatically different brightness levels instead of two nearly-identical ones, even though the underlying anatomy they represent is continuous. A radiologist viewing such an image could easily mistake that artificial discontinuity for a real anatomical edge or boundary that does not actually exist in the tissue being imaged -- exactly the kind of fabricated visual feature the real linear VOI LUT formula's own smooth, continuous mapping is specifically designed to avoid.

**5.** A check for specific forbidden field names (refusing anything literally named `DIAGNOSIS`, for instance) only catches the exact names someone thought to forbid in advance, and a differently-named field carrying the same kind of content (`IMPRESSION=...`, `ASSESSMENT=...`, or any other name not on that specific list) would sail through unrefused. A strict field-COUNT check refuses ANY line that does not have EXACTLY the four expected fields, regardless of what any extra field happens to be named -- it does not need to anticipate every name someone might try, because it refuses the sheer presence of an unexpected fifth field on structural grounds alone, which is a strictly more general and more robust defense than an ever-growing denylist of specific names.

**6.** Consider a study with one `STAT` finding (rank 2) and one `ROUTINE` finding (rank 0). The max-based approach correctly reports the study's own overall priority as `STAT` -- exactly the outcome a real triage queue needs, since the study genuinely does contain a finding requiring urgent review, however calm the other finding in the same study may be. Averaging the two ranks would produce something between `ROUTINE` and `EXPEDITED` (rank 1, "EXPEDITED," if rounded, or a genuinely intermediate non-integer value if not), which would route a study containing a real STAT-level finding into a LESS urgent review queue than it deserves -- exactly the kind of averaged-away urgent signal a real clinical triage system cannot afford to produce, since the calm finding does nothing to make the urgent one less urgent.

**7.** Checking only `state_ == ASSIGNED` would permit ANY caller who happens to know a case's own accession number -- not only the radiologist who actually claimed it -- to call `begin_review` or `finalize` on that case, since the state alone carries no information about WHO is accountable for it. Two different radiologists working from the same shared worklist could each attempt to advance the identical case, and a state-only check would let either of them succeed, silently overwriting or duplicating the other's own work with no record of which radiologist's actions should actually count. Checking `radiologist_id == *claimed_by_` at every subsequent transition is what actually enforces "the SAME accountable person throughout," rather than merely "someone, anyone, at the right moment in the state machine."

**8.** With an ordinary unstable sort, two STAT-priority cases added at different times could be returned in one order on one call to `pending_by_priority` and in the OPPOSITE order on a later call, purely because an unstable sort makes no guarantee about the relative order of elements it considers equal (both are priority `STAT`) -- even though neither case's own data changed between the two calls. A radiologist's worklist view could then show case A above case B one moment and B above A the next, with no real event having occurred to justify the reordering, which is exactly the kind of confusing, unexplainable queue behavior `std::stable_sort`'s own guarantee (elements considered equal keep their original relative order) rules out entirely.

**9.** A black occlusion patch has a raw pixel value of 0 in every channel, which is very unlikely to be anywhere near the image's own actual per-channel mean -- so replacing a patch with black does not merely "remove" that patch's own content, it also introduces a NEW, artificially large edge at the boundary between the black patch and its real neighboring patches, an edge the model may react to simply because sharp edges are salient to a vision encoder, regardless of whether the ORIGINAL patch's content was actually important. A mean-color occlusion, by contrast, is (in the normalized space Part 1 uses) exactly the zero vector -- the smallest, most neutral edit available, since it introduces no new edge of its own, meaning any resulting change in the model's output can be attributed more confidently to the removal of that patch's own real content rather than to an artifact the occlusion method itself introduced.

**10.** Checking only that the two scores are of comparable magnitude would be equally consistent with a much less interesting (and less honest) possible outcome: that the two anomalies, despite being different in location and pixel value, happened to move the representation in ROUGHLY THE SAME DIRECTION as well as by a similar amount -- in which case a defender of the technique could reasonably argue the saliency score, while scalar, was still capturing something like "the same kind of importance" both times. The cosine-similarity check is what actually rules that alternative explanation out: a similarity of roughly 0.2, far from 1.0, demonstrates directly that the two comparable-magnitude scores correspond to the representation moving in substantially DIFFERENT directions in the model's own hidden space -- proving the magnitude number alone genuinely cannot distinguish "the same kind of change happened twice" from "two different things happened to look similarly sized," which is the specific, real limit this section set out to prove rather than merely assert.
