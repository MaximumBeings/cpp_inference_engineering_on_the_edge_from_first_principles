# Chapter 28: Insurance Claims Photo Assessment with Fraud Flagging

**What you will understand by the end of this chapter:**

- How to reapply Chapter 23.1's own real perceptual difference-hash (dHash) algorithm a third time -- after Chapter 23.1's own counterfeit product photos and Chapter 26.1's signature comparison -- to a new real fraud pattern: the identical photo submitted across two unrelated insurance claims.
- How to build a real, from-scratch photo-timestamp consistency check reapplying Section 27.2's own "stated parameters, never a wall clock" discipline to a real Special Investigations Unit technique: catching a photo timestamped before the damage it depicts could have occurred.
- How to extend Chapter 21.3's own real severity-to-cost-range engine and photo-consistency check -- both reused completely unchanged -- with a new, transparent per-zone fraud-likelihood signal, while proving the claim's own cost estimate is never touched by that signal.
- How to build a claims-disposition engine reapplying this book's own never-suppress-a-named-flag discipline, with a real, deliberate ethical constraint this domain requires: no disposition this engine can ever produce is a denial.

**What you need to know first:**

- Chapter 23.1's own real, from-scratch dHash implementation (already reapplied once, in Chapter 26.1) is reapplied a third time in Section 28.1, this time to detect a photo reused across two different insurance claims.
- Section 27.2's own discipline of deriving timing behavior entirely from stated, labeled parameters rather than a wall clock is reapplied in Section 28.2, this time to a real photo-timestamp forensics check rather than a real-time processing deadline.
- Chapter 21.3's own real `CostRange` and photo-consistency machinery is reused completely UNCHANGED in Sections 28.3 and 28.4 -- this chapter's own new work is entirely in the fraud-likelihood signal layered alongside it, never inside it.

---

Chapter 27 turned to a hard, real-time deadline. This chapter turns to a genuinely different real constraint once more: an insurance claim, where the fraud-relevant signals are photographic and temporal rather than numeric, and where a false accusation carries real consequences for a genuine claimant. Each of this chapter's four sections builds one real, independent fraud signal -- reused photo detection, timestamp forensics -- and closes by extending Chapter 21.3's own cost-range engine with those signals, ending in a capstone disposition engine that routes a claim toward more or less human scrutiny, but never toward an automatic denial.

## 28.1 Insurance Claim Duplicate Photo Detection

### Intuition

A well-documented real insurance-fraud pattern is not a forged photo at all, but a genuine photo reused across two entirely unrelated claims. Catching this requires the identical real dHash algorithm this book has now built once and reapplied once already, applied for a third time to a new real domain: cross-claim photo deduplication, where the central design question is not the hashing itself but what counts as a real match.

### The Concept, In Detail

`find_reused_photo_matches` reuses Chapter 23.1's own unmodified `compute_dhash` and `hamming_distance`, checking a new submission's own photo hash against every OTHER claim's own stored photo in a real cross-claim database. Test 3 is this section's own central design decision: a claim re-submitting its OWN already-on-file photo -- an adjuster asking for a clearer re-scan, for instance -- is explicitly excluded from matching against itself, since this is completely routine, never a fraud signal on its own. Test 2 confirms a genuine cross-claim match is named exactly, and Test 4 confirms every matching claim is returned, not merely the first one found, when a single photo has already been reused across more than two claims.

Tests 5 and 6 confirm the exact Hamming-distance boundary at the raw hash level: a distance of exactly 10 (this section's own stated tolerance) still counts as a match, and a distance of exactly 11 does not, using the same inclusive convention Chapter 26.1's own signature-comparison tolerance already established.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_insurance_claim_duplicate_photo_detection.cpp -o 01_insurance_claim_duplicate_photo_detection
./01_insurance_claim_duplicate_photo_detection
```

**Sample input:** a genuinely unique new photo checked against a small existing cross-claim database with no real match; a new submission's photo checked to be a near-duplicate of a different claim's own stored photo, named exactly; the same claim re-submitting its own already-on-file photo, checked to be correctly excluded from matching itself; a photo already shared by two separate prior claims submitted again under a third claim, checked to name both prior matches; and the exact Hamming-distance boundary of the stated duplicate-detection tolerance, checked on both sides.

@@OUT1@@

!!! warning "[COMMON TRAP] flagging every hash match as fraud, regardless of which claim it belongs to"
    `find_reused_photo_matches` explicitly excludes matches against the SAME `claim_id` as the new submission -- a claimant, or an adjuster processing their claim, re-uploading the identical photo a second time (a clearer re-scan, a different file format, a resubmission after a lost upload) is completely routine, and treating it as a fraud signal would flag an enormous number of genuinely honest claims. The real signal this section's own function is built to catch is narrower and more specific: the SAME photo appearing under a DIFFERENT claim_id, which has no routine, honest explanation the way a same-claim resubmission does. Deploying this section's own logic without the self-exclusion check in Test 3 would replace a real, specific fraud signal with an enormous stream of false positives against the most common, most innocent case there is.

## 28.2 Photo Timestamp vs. Incident Date Consistency

### Intuition

A real, well-documented technique real insurance Special Investigations Units use is metadata timestamp analysis: a photo of damage cannot have been taken before the damage occurred, and a claim's own stated incident date gives a real, checkable boundary to test every submitted photo's own capture timestamp against.

### The Concept, In Detail

`check_photo_timing` names two different real timing anomalies with two different real implications. A photo timestamped strictly before the incident day is a genuine structural impossibility -- Test 2 and Test 6 confirm this is reported with the exact day gap, however large. A photo timestamped after the claim's own report day, beyond a stated grace period, is merely unusual rather than impossible (real follow-up photographs taken during an active repair are routine) -- Test 5 confirms this softer signal is still named, one day past the boundary. Tests 3 and 4 confirm both of this section's own boundary conventions are inclusive: a photo captured exactly on the incident day, and a photo captured exactly at the grace-period boundary, are both reported as consistent.

Test 7 confirms `assess_claim_timing`'s own multi-photo aggregation names exactly the one anomalous photo by its own zone label, among otherwise consistent photos, never dropping or averaging the flag away.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_photo_timestamp_incident_consistency.cpp -o 02_photo_timestamp_incident_consistency
./02_photo_timestamp_incident_consistency
```

**Sample input:** a photo captured between a claim's own incident and report days, checked as consistent; a photo captured before the incident day, checked to report the exact day gap as a structural impossibility, and again for a much larger gap; the incident-day boundary and the grace-period boundary each checked as inclusive; a photo captured one day past the grace period, checked as a softer, named anomaly; and a full 3-photo claim checked to flag exactly the one pre-dated photo by its own zone label.

@@OUT2@@

!!! warning "[COMMON TRAP] treating a PRE_DATED photo and a STALE_OR_LATE photo as the same real severity of anomaly"
    A `PRE_DATED` result names a genuine structural impossibility -- there is no honest explanation for a photograph of damage existing before the damage occurred, short of the photo belonging to a different, earlier incident entirely. A `STALE_OR_LATE` result names something categorically softer: a photo taken later than expected, which has many completely honest explanations (a follow-up photo during a real repair, a delayed upload, a second adjuster visit). Routing both anomalies to the identical downstream action would either treat a real structural impossibility too gently, or treat a routine late upload as though it were as serious as a photo that could not possibly be genuine -- exactly why this section's own `TimingVerdict` keeps the two as separate, distinctly named outcomes rather than collapsing them into one generic "timing anomaly" flag.

## 28.3 Extending the Cost-Range Engine with a Fraud-Likelihood Signal

### Intuition

This section directly extends Chapter 21.3's own real severity-to-cost-range engine and photo-consistency check -- both reused completely unchanged -- with a new per-zone fraud-likelihood signal built from Section 28.1's duplicate-photo flag, Section 28.2's timestamp-anomaly flag, and Chapter 21.3's own existing consistency verdict. The central discipline this section adds: a zone's own cost estimate and its own fraud likelihood are computed completely independently of each other.

### The Concept, In Detail

`assess_fraud_likelihood` is a real, fully transparent count of how many of three independent named signals fired for a zone -- Test 3 confirms all three signal types are weighted identically, each alone producing `MEDIUM` on its own. Test 4 confirms the exact `MEDIUM`/`HIGH` boundary at a count of two signals, and Test 6 is this section's own central honesty check: a two-zone claim with one zone firing all three fraud signals at once produces the EXACT SAME `total_range` as the identical severities with zero fraud signals attached, a direct paired comparison proving the cost estimate is never touched by the fraud-likelihood computation layered alongside it.

Test 8 closes the full loop, deriving a real `photo_inconsistent` flag from actual before/after pixel data using Chapter 21.3's own reused `mean_abs_diff` and `check_claim_consistency`, rather than a hand-set boolean, and confirming it flows into `assess_fraud_likelihood` exactly like the other two signals.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_fraud_likelihood_cost_range_engine.cpp -o 03_fraud_likelihood_cost_range_engine
./03_fraud_likelihood_cost_range_engine
```

**Sample input:** Chapter 21.3's own severity-to-cost-range policy confirmed unchanged; zero, one (each of three signal types in turn), two, and three fraud signals checked against the exact resulting likelihood level; a two-zone claim with all three signals firing on one zone checked to produce an identical total cost range to the same claim with zero signals attached; a full 3-zone claim's own per-zone report checked to name each zone individually; and a real photo-inconsistency flag derived from actual pixel data via Chapter 21.3's own reused functions, checked to flow correctly into the fraud-likelihood computation.

@@OUT3@@

!!! warning "[COMMON TRAP] treating a fraud-likelihood LEVEL as a calibrated probability of fraud"
    `assess_fraud_likelihood` is a real, transparent COUNT of how many named, independent real checks disagreed with a claim -- LOW, MEDIUM, or HIGH describes exactly that count, never a real, calibrated statistical probability that a specific claim is fraudulent. A zone with a `HIGH` fraud likelihood has two or three independently suspicious signals attached to it, which is real, actionable information worth a human's attention -- it is not, and was never built to be, a statement that this specific zone is "67% likely to be fraudulent" or any other precise number. Treating this section's own three-level, fully-named count as though it carried that kind of statistical precision would overstate exactly what this section's own honest, simple counting logic can support.

## 28.4 The Claims Disposition Engine

### Intuition

This chapter's capstone reuses Section 28.3's own cost-range and fraud-likelihood machinery unchanged, and routes an entire claim based on the single worst zone across it -- reapplying this book's own never-suppress-a-named-flag discipline once more, but with a real, deliberate ethical boundary this domain requires: this engine has no denial disposition at all, since a false machine denial has real consequences for a genuine claimant.

### The Concept, In Detail

`assess_claim_disposition` is driven entirely by the SINGLE WORST per-zone fraud level found anywhere in the claim -- Test 2 and Test 3 confirm one suspicious zone routes the entire claim to greater scrutiny regardless of how many other zones are completely clean, never diluted by averaging. Test 4 is this section's own central honesty check, extending Section 28.3's own Test 6 to the full disposition level: the identical zone severities produce the exact same total cost range whether or not any fraud signal fires, with only the ROUTING differing between the two cases.

Test 5 is this section's own structural guarantee: all three real dispositions this engine can ever produce are checked, by name, to contain neither "DENY" nor "REJECT" -- a real, deliberate, and testable property of the enum's own three values, not an incidental fact about any one test's own data. Test 6 confirms the disposition's own audit trail names the exact zone and fraud level responsible, never merely an aggregate count.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_claims_disposition_engine.cpp -o 04_claims_disposition_engine
./04_claims_disposition_engine
```

**Sample input:** an entirely clean multi-zone claim checked to process normally with an exact hand-computed total; one MEDIUM-likelihood zone and, separately, one HIGH-likelihood zone, each checked among otherwise-clean zones to route the entire claim to the correct disposition without dilution; identical zone severities checked with and without fraud signals attached, confirming an identical total cost range with only the disposition differing; all three real dispositions checked by name for the complete absence of denial wording; the disposition's own audit trail checked to name the exact responsible zone; the exact signal-count boundary driving a disposition; and a full multi-zone, multi-signal claim's own total checked against an exact hand computation.

@@OUT4@@

!!! warning "[COMMON TRAP] treating a HOLD_PENDING_VERIFICATION disposition as equivalent to withholding the claimant's own estimate"
    `ClaimResult.total_range` is computed identically regardless of `disposition` -- Test 4 confirms this directly. `HOLD_PENDING_VERIFICATION` means only that a human Special Investigations Unit reviewer sees this claim before it proceeds further; it does not mean the claimant's own currently-stated cost estimate is hidden, reduced, or withheld pending that review. Conflating "routed to a human for review" with "payment withheld or denied" would misrepresent what this engine actually does, and -- far more seriously, in a real deployment -- could lead a downstream system to silently treat every flagged claim as though it had already been denied, exactly the kind of unauthorized escalation this chapter's own introduction named as a real, serious harm to a genuine claimant.

## Chapter Summary

This chapter turned to insurance claims, where the fraud-relevant signals are photographic and temporal, and where a false machine accusation carries real consequences for a genuine claimant. Section 28.1 reapplied Chapter 23.1's own dHash algorithm a third time, this time to detect a photo reused across two different claims, explicitly excluding a claim's own routine resubmission of its own photo. Section 28.2 built a real photo-timestamp consistency check, reapplying Section 27.2's own stated-parameter timing discipline to catch a photo that could not possibly have been taken when it claims to have been. Section 28.3 extended Chapter 21.3's own cost-range and photo-consistency engine -- reused completely unchanged -- with a new, transparent per-zone fraud-likelihood signal, proving the cost estimate itself is never touched by that signal. Section 28.4 closed the chapter with a claims-disposition engine that routes a claim toward more or less human scrutiny based on its single worst zone, while guaranteeing, structurally, that no disposition it can ever produce is a denial.

## Self-Check Questions

1. Section 28.1's `find_reused_photo_matches` excludes matches against the same `claim_id` as the new submission. Describe one concrete real scenario in which this exclusion would incorrectly hide a genuine fraud signal, if a single real claimant were allowed to submit more than one claim under different claim IDs.
2. Section 28.1's Test 4 constructs a photo already shared by two separate prior claims. Explain why returning ALL matching claim IDs, rather than just the closest or the first one found, is the correct design choice for a real fraud investigation.
3. Section 28.2's `check_photo_timing` treats a `PRE_DATED` result and a `STALE_OR_LATE` result as two separate, distinctly named outcomes rather than a single generic anomaly flag. Using the section's own COMMON TRAP box, explain concretely why a real SIU reviewer would prioritize their own investigation differently between the two.
4. Section 28.2's own introduction compares its own "stated day-count, never a wall clock" discipline to Section 27.2's identical discipline for a different constraint. Explain what specifically would go wrong with this book's own cross-architecture verification if `check_photo_timing` instead computed a photo's own age using a real calendar library's current-date function.
5. Section 28.3's Test 6 constructs a two-zone claim where one zone fires all three fraud signals at once. Explain precisely HOW `assess_claim`'s own implementation guarantees this cannot affect `total_range`, referring to the specific code structure responsible.
6. Section 28.3's own COMMON TRAP box warns against treating a fraud-likelihood LEVEL as a calibrated probability. Explain what a real insurer would need to add to this section's own simple counting logic before it could honestly be described as a calibrated probability model.
7. Section 28.4's `assess_claim_disposition` is driven by the single WORST zone's own fraud level, never an average across all zones. Construct a concrete scenario with at least four zones where averaging, instead of taking the worst, would cause a real fraud signal to be missed entirely.
8. Section 28.4's own introduction states a real, deliberate ethical reason this engine has no denial disposition, unlike Chapter 26.4's `AUTO_REJECT`. Explain why an insurance claim's own real-world consequences of a false positive differ from a real bank's own consequences of falsely auto-rejecting a check, in a way that justifies this specific structural difference between the two engines.
9. Section 28.4's Test 5 checks all three real `ClaimDisposition` values for denial wording at runtime, using string matching. Explain one real limitation of checking for the ABSENCE of a specific fraud-adjacent word as a way of proving a system can never take a specific, serious action.
10. Across this chapter's four sections, identify the ONE section that reuses an ALGORITHM this book already built rather than a DOMAIN PATTERN this book already established, and explain the difference between those two kinds of reuse using this chapter's own two clearest examples of each.

## Where We Go Next

This chapter showed that a fraud signal and a cost estimate can, and must, be computed on completely separate, independent tracks -- reapplying this book's own recurring never-suppress-a-flag discipline with a real, deliberate ethical boundary an insurance claim's own real-world stakes require. Chapter 29 turns to the last of this book's four financial-industry application chapters: on-device anti-money-laundering (AML) transaction anomaly monitoring at ATMs and point-of-sale terminals, where the constraint shifts once more, this time to detecting a pattern across a SEQUENCE of transactions rather than evaluating any single one in isolation.

## Worked Solutions

**1.** If a real claimant submitted two genuinely separate, unrelated claims (for instance, two different vehicles they own, damaged in two unrelated real incidents) and, dishonestly, used the SAME real photo of damage for both -- perhaps because they never actually damaged the second vehicle at all -- Section 28.1's own `claim_id` exclusion would not help here, since the two claim IDs are genuinely different; this scenario would still be caught correctly. The exclusion would incorrectly hide a real signal only if the SAME claim were somehow re-registered under a second, different `claim_id` for the identical incident (a real, if less common, fraud pattern in its own right: filing what is actually one incident as two separate claims to double an insurer's exposure) -- in that specific case, this section's own cross-claim check would flag the resubmission as a normal, apparently-cross-claim match rather than recognizing it as the SAME underlying incident, since `claim_id` alone cannot distinguish "the same claim, re-filed" from "a genuinely new, unrelated claim."

**2.** A single closest or first match would tell an SIU investigator only that ONE other claim shares this photo, when the real, more serious pattern -- the same photo circulating across THREE OR MORE claims -- is a far stronger, more organized fraud signal (potentially indicating a fraud ring reusing a shared pool of staged damage photos across many claimants) that a "just the first match" design would completely hide. Returning every match lets a human reviewer see the real SCALE of the reuse pattern directly, rather than having to separately re-run the same check against every other claim in the database to discover what this section's own function could have reported the first time.

**3.** A `PRE_DATED` photo describes a real structural impossibility with no honest explanation short of the photo belonging to an entirely different, earlier incident -- a reviewer would prioritize investigating this claim's own basic legitimacy immediately, since the claim's own submitted evidence is self-contradictory. A `STALE_OR_LATE` photo, by contrast, has many completely routine explanations (a delayed upload, a follow-up photo taken during an active real repair), so a reviewer would reasonably treat it as a lower-priority item worth a quick confirmation rather than an urgent investigation -- collapsing both into one generic flag would force every reviewer to manually re-derive which of the two situations they were actually looking at before they could even begin prioritizing their own queue.

**4.** This book's own cross-architecture verification requires the locked, byte-for-byte self-test output to match EXACTLY across four different execution legs, run at different real times (potentially days apart, across different verification passes). A calendar library's own "current date" function returns a genuinely different real value depending on WHEN the test actually runs, which would make `check_photo_timing`'s own computed day-gaps differ between a verification run today and a verification run next week -- the exact same category of problem Section 27.2's own COMMON TRAP box already named for a real wall-clock latency measurement, just manifesting as a difference across TIME instead of across ARCHITECTURE.

**5.** `assess_claim`'s own loop computes `result.total_range = result.total_range + cost_range_for_severity(z.claimed_severity)` using ONLY `z.claimed_severity` as its input -- `z.fraud_signals` is never read, referenced, or passed into `cost_range_for_severity` or the `total_range` accumulation anywhere in that function. The fraud-likelihood computation, `assess_fraud_likelihood(z.fraud_signals)`, is a completely separate call whose own return value is appended only to `result.per_zone_fraud`, a different field entirely -- the two computations share the same input zone but write to two different output fields through two functions that never call each other, which is what makes the independence a structural guarantee rather than a testing coincidence.

**6.** A real calibrated probability model would need real, historical ground-truth data -- a large number of PAST claims where each of these three signals either fired or did not, cross-referenced against which of those claims were later CONFIRMED, through a real investigation, to actually be fraudulent -- and a real statistical model (such as real logistic regression) fit against that historical data to learn how much each signal, and each COMBINATION of signals, actually predicts confirmed fraud in practice. Section 28.3's own simple count has none of this: it treats all three signals as equally weighted by stated policy choice, with no historical validation behind that weighting at all, which is exactly why it reports a plain descriptive LEVEL rather than a number that could be mistaken for a validated statistical probability.

**7.** Consider four zones with fraud-signal counts of 2 (HIGH), 0, 0, and 0 -- averaged naively across all four zones (treating LOW=0, MEDIUM=1, HIGH=2 as numeric scores, for instance), the average signal count would be `(2+0+0+0)/4 = 0.5`, which would likely round down to a LOW or borderline-MEDIUM overall level despite one zone carrying a real, serious two-signal HIGH-likelihood flag on its own. Taking the single WORST zone instead, as Section 28.4 actually does, correctly reports HIGH for the whole claim regardless of how many additional clean zones happen to be attached to it -- exactly the same "one flag should never be diluted by everything else looking clean" principle Chapter 26.4's own MICR-forcing rule already established for check fraud.

**8.** A real bank's own check-fraud engine (Chapter 26.4) operates on a payment INSTRUMENT before funds move -- an `AUTO_REJECT` there stops a specific transaction from clearing, and a legitimate payee whose check is wrongly rejected can typically resubmit or contact their bank promptly, with the underlying funds and account relationship otherwise unaffected. A real insurance claim, by contrast, often represents a real person's own urgent, immediate need (a home that is currently damaged, a vehicle they cannot currently drive) where a false denial can mean real, immediate financial hardship with no quick resubmission path, and many real jurisdictions' own insurance regulations specifically require a human decision-maker before a fraud-suspected claim can be denied at all -- a real regulatory and human-consequence difference this section's own structural choice (no denial disposition exists in this engine at all) reflects directly.

**9.** Checking that a string does not CONTAIN the substrings "DENY" or "REJECT" only proves that THOSE TWO SPECIFIC WORDS are absent from the three disposition names as currently written -- it says nothing about the actual BEHAVIOR of any code that might later consume a `ClaimDisposition` value. A future engineer could, in principle, add a fourth enum value named something that passes this exact string check (such as `ClaimDisposition::CLOSE_WITHOUT_PAYMENT`) while still functioning as a real denial in every practical sense; the test only guards against a very literal category of naming mistake, and a genuine structural guarantee against ever ADDING a real denial path would require reviewing every place a `ClaimDisposition` value is consumed downstream, not merely inspecting the enum's own three names.

**10.** Section 28.1 reuses an ALGORITHM this book already built -- Chapter 23.1's own real dHash perceptual-hashing implementation, copied and reapplied completely unchanged, exactly the same kind of reuse Chapter 26.1 already performed once. Section 28.3 (and Section 28.4, built directly on top of it) instead reuses a DOMAIN PATTERN this book already established -- Chapter 21.3's own real severity-to-cost-range-plus-consistency-check STRUCTURE, extended with an entirely new fraud-likelihood signal layered alongside it, rather than any single function being copied verbatim without modification. The difference is that an algorithm reuse (Section 28.1) applies IDENTICAL code to new data, while a pattern reuse (Sections 28.3-28.4) applies an established STRUCTURAL APPROACH -- separate concerns, honest ranges, named audit trails -- to build genuinely new code that follows the same real design discipline.
