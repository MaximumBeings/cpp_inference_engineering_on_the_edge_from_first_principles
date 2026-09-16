# Chapter 22: Security Surveillance, Accessibility Compliance, and Art Authentication

**What you will understand by the end of this chapter:**

- How to build a real presence-interval coalescing pipeline over raw per-frame detections, render a deterministic natural-language narration sentence from that structured state, and honestly report AMBIGUOUS rather than silently guessing when a multi-camera correlation genuinely has more than one equally plausible answer.
- How to implement WCAG 2.x's own real, publicly standardized relative-luminance and contrast-ratio formulas from scratch, verified against a real, famous exact reference value, and how to build an audit report that names the SPECIFIC success criterion every violation fails rather than a vague "accessibility issue."
- How to build a structural, chronological provenance-chain validator that flags gaps, overlaps, and impossible or missing current-owner records by name and exact computed size, and a real ordinary-least-squares condition-trend fit that classifies a conservator's periodic scores without ever rendering a fraud or authenticity verdict itself.

**What you need to know first:**

- Chapter 21's own recurring honesty discipline -- honest ranges instead of false-precision point estimates (Section 21.3's `CostRange`), and named, specific-criterion refusals rather than vague errors (Section 21.1's strict parser) -- both recur in this chapter's own domains, applied to security correlation and accessibility auditing respectively.
- Section 21.1's own honest AMBIGUOUS-style refusal discipline (rather than forcing a confident-looking answer out of data that does not support one) is the exact pattern Section 22.1's own multi-camera correlation reapplies when two candidate re-identifications both fit a transition window equally well.
- Chapter 19.4's own real ordinary-least-squares trend-fitting technique, applied there to retail sell-through trends, is reused verbatim in Section 22.3 to fit a conservator's periodic condition scores.

---

Chapter 21 turned a single photograph into a structured extraction, claim assessment, or species label. This chapter turns to three domains united by a different real pattern: turning a CONTINUOUS stream -- a camera feed sampled over time, a rendered user interface, an ownership history spanning decades -- into a structured, auditable narrative, without ever collapsing that narrative's own real uncertainty into a false confident answer. A security system correlating detections across cameras must say AMBIGUOUS when two candidates both fit; an accessibility audit must cite the exact, real, standardized criterion a violation fails; and a provenance and condition-trend engine must flag a structural inconsistency by its exact computed size and classify a real trend, without ever pronouncing an artwork itself authentic, forged, or accurately dated. None of these three sections needs Chapter 20's own clinical-grade guarantees, but every one of them needs this book's own recurring discipline: report the honest range or the honest ambiguity, name the specific failure, and never let a computed signal be mistaken for a verdict it was never built to render.

## 22.1 A Multi-Camera Surveillance Narration Engine with Temporal Correlation

### Intuition

A raw stream of per-frame detections is not, by itself, a narrative a human reviewer can act on -- it is a flood of individually meaningless coordinates. This section builds the real structural step between the two: coalescing raw detections into presence intervals, rendering those intervals as a deterministic narration sentence, and correlating a subject's departure from one camera with their arrival at another, honestly refusing to guess when more than one arrival genuinely fits.

### The Concept, In Detail

`build_presence_intervals` is a real, from-scratch coalescing pass over raw per-frame detections, grouped by camera and track, merging consecutive detections within a stated gap tolerance into a single presence interval and starting a new interval once a real gap exceeds that tolerance -- Test 1 confirms both the merging behavior across a tolerable gap and the splitting behavior across an intolerable one, checked against exact hand-computed interval boundaries. `render_narration_sentence` is a real, deterministic string-formatting function producing the same exact sentence for the same input every time, a property Test 2 confirms directly by rendering the identical interval twice and checking byte-for-byte equality -- a property this book's own determinism discipline treats as a real correctness requirement, not an afterthought, for any narration a human reviewer or an audit log will read back later.

`find_correlated_entry` is this section's own central honesty check: given a subject's departure from one camera and a real, stated adjacency table of plausible transition times between camera pairs, it searches candidate arrivals at adjacent cameras and returns `FOUND` when exactly one candidate fits the transition window, `NOT_FOUND` when none do, and -- critically -- `AMBIGUOUS`, naming every fitting candidate, when two or more candidates both fit equally well. Test 4 constructs exactly this scenario: two separate subjects arriving at an adjacent camera within the same real transition window after the tracked subject's own departure, and confirms the function reports `AMBIGUOUS` with both candidates named, rather than silently picking the nearer or the first one.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_surveillance_narration_and_multi_camera_correlation.cpp -o 01_surveillance_narration_and_multi_camera_correlation
./01_surveillance_narration_and_multi_camera_correlation
```

**Sample input:** a burst of same-camera, same-track detections across a tolerable gap checked to coalesce into one presence interval, and a wider gap checked to split into two; the same interval rendered twice checked to produce byte-identical narration sentences; a real camera-adjacency table checked against its own stated minimum and maximum transition times; a departure with exactly one fitting candidate arrival checked to report FOUND; a departure with zero fitting candidates checked to report NOT_FOUND; and a departure with two equally-fitting candidate arrivals checked to report AMBIGUOUS, naming both candidates rather than picking one.

@@OUT1@@

!!! warning "[COMMON TRAP] treating an AMBIGUOUS correlation as a system failure to be silently resolved"
    It is tempting to treat `find_correlated_entry`'s own `AMBIGUOUS` result as an unfinished computation -- surely the system could pick the CLOSER of the two candidates in time, or the one with the higher raw detection confidence, and just report a single answer. Test 4 is built specifically to show why that temptation is exactly backwards: when two real candidates both genuinely fit the stated transition window, picking one and reporting it as `FOUND` would manufacture a specific, confident-looking claim -- "the subject who left Camera 3 is the same person who entered Camera 5" -- that the underlying data does not actually support. `AMBIGUOUS`, naming both real candidates, is the honestly complete answer; a human reviewer with access to a face-matching system, a badge log, or simply the original footage is positioned to resolve what this section's own temporal-correlation-only evidence genuinely cannot.

## 22.2 An Accessibility-Compliance Auditing Engine

### Intuition

An accessibility audit is one of the few domains this book has reached where the structured extraction a vision-language model produces from a screenshot can be checked against a real, EXACT, publicly standardized formula rather than a stated policy choice of this book's own invention. This section builds that real math -- relative luminance, contrast ratio, and the AA/AAA pass thresholds -- from scratch, verified against a real, famous reference value.

### The Concept, In Detail

`relative_luminance` and `contrast_ratio` implement the exact WCAG 2.x formulas precisely as the specification defines them -- Test 1 confirms pure black text on a pure white background computes to a contrast ratio of exactly 21:1, WCAG's own famous exact reference value, and white on white computes to exactly 1:1. `contrast_ratio` computes its own lighter/darker ordering internally rather than trusting a caller to pass colors in the correct order, and Test 2 confirms this directly: `contrast_ratio(light, dark)` and `contrast_ratio(dark, light)` both compute to the identical value, guarding against a real, common implementation bug that silently produces a ratio below 1.0 for exactly half of all real color pairs.

`audit_element` runs three real, independently named checks -- 1.4.3 Contrast Minimum, 2.5.5 Target Size, and 1.1.1 Non-text Content -- against a single UI element, and Test 4 confirms a fully compliant element produces zero violations while a non-compliant element is flagged with all three real, specific criteria named at once, never a single vague "accessibility issue" covering all three failures. Test 5's own full multi-element audit report confirms every violation is attributed to its own specific element by id, with zero false positives contributed by the genuinely compliant elements sharing the same report.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_accessibility_compliance_audit.cpp -o 02_accessibility_compliance_audit
./02_accessibility_compliance_audit
```

**Sample input:** pure black on pure white checked against WCAG's own famous exact 21:1 reference value, and white on white checked against exactly 1:1; forward and reversed color-pair arguments checked to produce identical contrast ratios; AA and AAA classification checked exactly at their own stated threshold boundaries for both normal and large text; a fully compliant 48x48 black-on-white button checked to produce zero violations while a 30x30 low-contrast icon with no alt text is flagged with all 3 real, specific criteria at once; and a 3-element audit report checked to attribute exactly 3 violations, all to the one non-compliant element, with zero false positives on the two compliant elements sharing the report.

@@OUT2@@

!!! warning "[COMMON TRAP] treating an AAA-level failure as an equally urgent violation as an AA-level one"
    WCAG defines two real, separately useful conformance levels -- AA, the level most real accessibility regulations (including the ADA and Section 508) actually cite as their own required minimum, and the considerably stricter, optional AAA. It is tempting to have an audit engine flag any element failing EITHER level as a violation, on the theory that more strictness can only help -- but this section's own `audit_element` deliberately enforces only the AA bar, exactly as regulatory practice actually requires, and an element that fails AAA while still passing AA is not reported as a violation at all. Flagging every AAA shortfall as an urgent violation would flood a real report with a volume of noise that would make the genuinely required AA failures -- the ones with real regulatory and human-usability consequences -- harder, not easier, for a reviewer to find and act on.

## 22.3 An Art-Condition Assessment and Provenance-Verification Engine

### Intuition

A stated ownership history is either chronologically consistent or it is not, and that consistency can be checked by real, structural date arithmetic alone, with no judgment about authenticity required at all. Separately, a conservator's periodic condition scores describe a real, fittable trend -- the same ordinary-least-squares regression this book already built for a retail sell-through trend in Chapter 19.4, reapplied here to a very different real domain.

### The Concept, In Detail

`validate_provenance_chain` checks three real, independent structural rules over a stated ownership chain, sorted by acquisition period: Rule 1 flags any record whose disposal period is not strictly after its own acquisition period; Rule 2 requires exactly one open-ended (currently-owned) record, and that record must be the chronologically LAST one, flagging a chain with zero open-ended records, more than one, or one that is open-ended but not last; Rule 3 checks every consecutive pair of records for exact contiguity, flagging a real, exactly-computed gap or overlap in periods rather than a vague "inconsistent dates" message. Test 2 confirms both a real 20-period gap and a real 50-period overlap are each flagged with their own exact computed size, and Test 3 and Test 4 together confirm all of Rule 1 and Rule 2's own failure cases -- dispose-before-acquire, no current owner, two simultaneous current owners, and a stale current-owner claim contradicted by a later record -- are each flagged by name.

`fit_condition_trend` is the exact same real ordinary-least-squares formula this book built in Chapter 19.4, applied here to a conservator's periodic condition scores rather than retail sell-through data -- Test 5 confirms the fit matches hand-computed slope and intercept values exactly for a perfectly linear decline and a perfectly linear improvement, and `classify_condition_trend`'s own stated rate thresholds correctly classify a declining trend as `ACCELERATING_DETERIORATION`, a flat noisy trend as `STABLE`, and an improving trend -- consistent with a real conservation treatment -- as `IMPROVING`. Neither function anywhere in this file renders a verdict on an artwork's own authenticity, valuation, or attribution; both report only what their own real, checkable inputs structurally support.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_art_provenance_and_condition_trend.cpp -o 03_art_provenance_and_condition_trend
./03_art_provenance_and_condition_trend
```

**Sample input:** a fully contiguous, chronologically consistent 3-owner chain checked to validate with zero issues; a real 20-period gap and a real 50-period overlap each checked against their own exact hand-computed size; a dispose-before-acquire record and a chain with no current owner at all each checked to be flagged; two simultaneously open-ended owners and an open-ended owner who is not chronologically last each checked to be flagged; and real least-squares condition-trend fitting checked against hand-computed slope and intercept values for a decline, a flat noisy series, and an improvement, each correctly classified.

@@OUT3@@

!!! warning "[COMMON TRAP] treating a validated provenance chain or a classified condition trend as an authenticity verdict"
    A chain that passes `validate_provenance_chain` with zero issues has been checked for exactly one real, narrow property: that its OWN STATED dates are internally, chronologically consistent with each other. It has not been checked against any external record, and a chronologically consistent chain of entirely fabricated ownership claims would pass this same validator just as cleanly as a genuine one -- consistency is a real, necessary property of an honest provenance record, but it is not, on its own, sufficient evidence of one. The same discipline applies to `classify_condition_trend`: a real, correctly computed `ACCELERATING_DETERIORATION` classification describes a real statistical trend in stated numeric scores, not a diagnosis of WHY the work is deteriorating or a judgment about the work's own authenticity -- both functions in this file report exactly what their own narrow, real inputs support, and nothing more.

## Chapter Summary

This chapter moved this book's own recurring honesty discipline into three domains united by a continuous-stream, rather than single-photograph, structure. Section 22.1 built a real presence-interval coalescing pipeline, a deterministic narration renderer, and a multi-camera correlation engine that honestly reports AMBIGUOUS, naming every fitting candidate, rather than silently picking one when the underlying temporal evidence genuinely supports more than one answer. Section 22.2 implemented WCAG 2.x's own real, publicly standardized contrast-ratio formula from scratch, verified against its own famous exact 21:1 reference value, and built an audit engine that names the specific success criterion -- never a vague label -- behind every reported violation. Section 22.3 built a structural, chronological provenance-chain validator that flags a gap, overlap, or an impossible or missing current-owner record by exact computed size, and reused Chapter 19.4's own real ordinary-least-squares regression to classify a conservator's condition-score trend, with neither function ever rendering the authenticity verdict that remains, correctly, outside either one's own real scope.

## Self-Check Questions

1. Section 22.1's `find_correlated_entry` can return `FOUND`, `NOT_FOUND`, or `AMBIGUOUS`. Explain why collapsing `AMBIGUOUS` into `NOT_FOUND` (treating "too many candidates" the same as "no candidates") would lose real, useful information a human reviewer could act on.
2. Section 22.1's `render_narration_sentence` is checked for byte-identical output across repeated calls with the same input. Why does this determinism property matter specifically for a narration a human reviewer or an audit log will read, beyond this book's own general determinism discipline?
3. Section 22.2's `contrast_ratio` computes its own lighter/darker ordering internally rather than requiring the caller to pass arguments in a specific order. Describe the specific, real bug this design choice prevents.
4. Section 22.2 enforces only the AA conformance level, not the stricter AAA level, in its own violation reporting. Explain why this is described as a deliberate, real policy choice rather than a limitation of the underlying contrast-ratio math.
5. Section 22.2's Test 4 confirms a non-compliant element is flagged with all 3 applicable criteria at once, named individually. What real problem would a single combined "multiple issues found" message cause that 3 separately named violations do not?
6. Section 22.3's Rule 2 requires exactly one open-ended record, and that record must be chronologically last. Construct a concrete 3-owner chain that would satisfy "exactly one open-ended record" but still be correctly flagged by the "must be last" condition.
7. Section 22.3's Rule 3 reports a gap or overlap by its own exact computed size (for instance, "gap of 20") rather than simply "inconsistent dates." What real, practical use does the exact size provide to whoever investigates the flagged chain that a generic message would not?
8. Section 22.3's `fit_condition_trend` reuses Chapter 19.4's own retail-trend regression formula unchanged. What does the reuse of the identical formula across two very different real domains demonstrate about what ordinary least squares actually depends on?
9. Section 22.3 explicitly states that neither `validate_provenance_chain` nor `classify_condition_trend` renders an authenticity verdict. Construct a concrete scenario where a chain passes provenance validation with zero issues despite describing an entirely fabricated ownership history.
10. Across all three sections in this chapter, identify the one recurring structural choice -- present in `AMBIGUOUS`, in the named WCAG criteria, and in the named provenance-chain issues -- that ties this chapter's own honesty discipline together, and explain why a single boolean "flagged: true/false" output in any of the three would have been a real regression.

## Where We Go Next

This chapter's three domains showed the same recurring discipline -- honest ambiguity, specifically named failures, and a computed signal that never overstates its own real scope -- generalizing across security correlation, accessibility auditing, and provenance and condition assessment. Chapter 23 turns to a fourth domain sharing a closely related real pattern: trust and authenticity at the point of sale, building a brand-reference database and counterfeit-screening engine for e-commerce listings, receipt-to-transaction reconciliation, automated expense-report policy enforcement, and a real-time luxury-goods authentication engine for consignment and resale counters.

## Worked Solutions

**1.** Collapsing `AMBIGUOUS` into `NOT_FOUND` would tell a human reviewer "no plausible arrival was found for this departure," which is a real, actionable claim that the subject likely left the monitored area entirely -- when the true situation is the opposite: TWO real, named candidates both plausibly account for the subject's arrival, and the correct next step is for a reviewer to examine those two specific candidates (checking face-matching, badge logs, or the original footage) rather than searching more broadly for a subject who may, in fact, already be accounted for by one of the two named candidates.

**2.** A narration a human reviewer or an audit log reads back later is often read hours, days, or in a legal or compliance context, months after the original event -- if the identical underlying interval data could render as two different sentences depending on when or how it was rendered, a reviewer comparing an audit log entry against a live re-render of the same data could see an apparent discrepancy that has nothing to do with the underlying events at all, undermining exactly the kind of reliable, checkable record this book's own determinism discipline is built to guarantee everywhere else in this system.

**3.** The specific bug this design prevents is a contrast-ratio function that assumes its own caller will always pass the lighter color first (or the darker color first) and computes `(a + 0.05) / (b + 0.05)` directly on whichever order it receives -- for any color pair where the caller happens to pass the darker color first, this produces a ratio BELOW 1.0, which is not merely an inverted number but a value WCAG's own formula never produces for any real color pair, silently corrupting exactly half of all real contrast computations depending entirely on argument order rather than the colors' own real relationship.

**4.** AA is described as a deliberate policy choice because it is the level real accessibility regulations -- the ADA and Section 508 are both named directly -- actually cite as their own required legal minimum, meaning an audit enforcing AA reports exactly the violations that carry real regulatory and legal consequences. AAA is a real, valid, stricter bar that WCAG itself defines, but treating an AAA shortfall as an equally urgent violation would report a large volume of real but non-mandatory issues alongside the mandatory ones, with no way for a report reader to distinguish "you are out of legal compliance" from "you have room to be more accessible than the law strictly requires" -- the underlying contrast math is identical either way; only the threshold applied to it differs.

**5.** A single combined message like "3 issues found" (or worse, "accessibility issues found") gives a report reader no way to know, without opening the underlying code or re-running the audit with more verbose logging, WHICH of potentially many possible criteria actually failed -- a developer fixing the element would have to independently re-derive whether the problem is contrast, target size, or a missing text alternative. Three separately named violations let that same developer go directly to the specific, real fix each one requires -- adjusting a color, enlarging a touch target, or adding alt text -- without first having to rediagnose which of several real, unrelated problems is actually present.

**6.** Owner A acquires at period 0 and never disposes (open-ended); Owner B acquires at period 100 and also never disposes (open-ended) -- wait, this satisfies "exactly one open-ended record" only if exactly one of the two lacks a disposal period. Construct instead: Owner A acquires at period 0, disposes at period 100; Owner B acquires at period 100, never disposes (open-ended, and chronologically last) -- this passes Rule 2 cleanly. To construct a FAILING case with exactly one open-ended record that still fails the "must be last" condition: Owner A acquires at period 0, and is recorded with NO disposal period (open-ended); Owner B acquires at period 100 and disposes at period 200. Exactly one record (Owner A) is open-ended, but Owner A is NOT the chronologically last record by acquisition order -- Owner B's own later transaction record, despite Owner A's own claim of still owning the work, is exactly the stale, inconsistent current-owner claim Test 4 is built to catch.

**7.** The exact computed size tells whoever investigates the flagged chain WHERE to look and how large a discrepancy to expect to find -- a "gap of 20" periods between two owners is a concrete, bounded window a researcher can search real auction records, shipping manifests, or estate documents for, while a "gap of 400" periods between the same two owners would suggest a considerably more significant, and more suspicious, missing chapter in the object's own history. A generic "inconsistent dates" message would require the investigator to first re-derive the size and location of the discrepancy from the raw records themselves before any real investigation could even begin.

**8.** The identical formula working correctly across a retail sell-through curve (Chapter 19.4) and a conservator's periodic condition scores (this chapter) demonstrates that ordinary least squares depends only on having a real, ordered independent variable (a time period) and a real, numeric dependent variable (a sales figure, a condition score) with an assumed-linear relationship between them -- it has no domain-specific knowledge of retail or of art conservation built into it anywhere, which is exactly why the same real mathematical technique, unchanged, generalizes cleanly to any domain that produces the same basic shape of data.

**9.** A forger constructs an entirely fictional chain of three "owners" -- invented names, with acquisition and disposal periods that are all mutually contiguous and chronologically sound, ending with a fabricated open-ended "current owner" record -- and submits this fictional chain as the object's provenance. `validate_provenance_chain` checks ONLY whether the chain's own STATED dates are self-consistent with each other, which this fabricated chain, having been constructed specifically to be self-consistent, satisfies perfectly, producing zero issues despite every single record in the chain being entirely invented; catching that would require independently verifying each record against real external evidence (auction house records, insurance documents, exhibition catalogs), which is explicitly outside what this section's own structural date-arithmetic validator was ever built to do.

**10.** The recurring structural choice is that every one of this chapter's three outputs -- `CorrelationResult`'s own `AMBIGUOUS` outcome naming its candidates, `audit_element`'s own list of specifically named WCAG criteria, and `validate_provenance_chain`'s own list of specifically named, exactly-sized issues -- reports MULTIPLE, INDEPENDENTLY NAMED findings (or the honest absence of a single findable answer) rather than a single collapsed signal. A boolean "flagged: true/false" in any of the three would have been a real regression because it would discard exactly the information a human reviewer needs to act: WHICH candidates are ambiguous, WHICH specific criterion failed, or WHERE and how large a provenance inconsistency is -- collapsing any of these to a single bit preserves the alarm but destroys the actionable detail this chapter's entire discipline exists to keep visible.
