# Chapter 29: On-Device AML Transaction Anomaly Monitoring at ATMs and POS Terminals

<!-- edge:opener:start -->
<figure class="chapter-figure" markdown="1">

[![Chapter 29: Monitor transactions with explicit windows](../images/chapter-openers/29.svg?v=db0c2532043c)](../images/chapter-openers/29.svg?v=db0c2532043c)

<figcaption>Chapter roadmap. Open the image for a full-size view.</figcaption>
</figure>
<!-- edge:opener:end -->

**What you will understand by the end of this chapter:**

- How to detect "structuring" (smurfing) -- a real, specifically named federal offense under the US Bank Secrecy Act -- by scanning a rolling window across a SEQUENCE of transactions, since no individual transaction in the pattern is suspicious on its own.
- How to build a real transaction-velocity anomaly check, reapplying Section 27.2's own "stated parameters, never a wall clock" discipline to a frequency-based signal instead of a deadline.
- How to compute a geographic speed anomaly with explicit zero-time and chronological-input handling, without treating it as proof of fraud.
- How to combine three independent, sequence-based signals into one per-account disposition, closing this book's four-chapter run through real financial-industry deployments with a final, real, legally precise never-suppress-a-flag discipline: a Suspicious Activity Report can only ever be RECOMMENDED by this engine, never filed by it.

**What you need to know first:**

- Section 27.2's own discipline of deriving timing behavior entirely from stated, labeled parameters rather than a wall clock is reapplied in Section 29.2, this time to a transaction-frequency window instead of a processing deadline.
- Section 28.2's own stated-integer-day-count convention for a claim's own timeline is reapplied in Section 29.1, this time to a rolling structuring-detection window.
- Chapter 26.1's own real, published ABA checksum and Chapter 27.1's own real, published microprice formula are joined in this chapter by a third real, published formula: the Haversine great-circle-distance formula, used in Section 29.3.
- This chapter's own central shift from every prior finance chapter: every check in this chapter evaluates a SEQUENCE of transactions from the same account together, since no single transaction in any of this chapter's three real patterns is suspicious in isolation.

---

Chapter 28 turned to a claim's own photographic and temporal evidence. This chapter closes this book's four-chapter run through real financial-industry deployments with a genuinely different real constraint once more: catching a pattern that only becomes visible across a SEQUENCE of transactions from the same account, where no single transaction in the sequence is suspicious by itself. Each of this chapter's four sections builds one real, independent, sequence-based AML signal, and closes with a capstone that combines all three into one auditable, per-account disposition -- using a demonstration review policy that does not implement filing.

<!-- chapter-figure -->
<figure class="chapter-figure" markdown="1">

![Monitor transactions with explicit windows. Aggregate event history before combining anomaly signals.](../images/29-overview.svg?v=c4c9072dab62){: loading="lazy" width="760" }

  <figcaption>An alert needs contextual review; thresholds alone cannot establish money laundering.</figcaption>
</figure>
<!-- /chapter-figure -->

## 29.1 AML Structuring (Smurfing) Detection

<!-- edge:concept-29.1:start -->
<figure class="chapter-figure" markdown="1">

[![Windowed aggregation detects a pattern candidate](../images/technical/ch29/29-01.svg?v=9dec6f445cae)](../images/technical/ch29/29-01.svg?v=9dec6f445cae)

<figcaption>Structuring rules depend on jurisdiction and context. A pattern flag is not proof of criminal conduct.</figcaption>
</figure>
<!-- edge:concept-29.1:end -->


### Intuition

This example uses a simplified screening heuristic: within a rolling per-account window, flag an aggregate strictly above $10,000 only when every individual amount is below $10,000. Its boundaries are implementation choices, not the definition of structuring. Structuring depends on a purpose to evade reporting and can involve any amount, multiple days or institutions. A negative result does not establish legitimacy, and a positive result does not establish criminal intent. The CTR reporting threshold and a structuring determination are different questions. See [FFIEC guidance](https://bsaaml.ffiec.gov/manual/Appendices/08).

### The Concept, In Detail

This example uses a simplified screening heuristic: within a rolling per-account window, flag an aggregate strictly above $10,000 only when every individual amount is below $10,000. Its boundaries are implementation choices, not the definition of structuring. Structuring depends on a purpose to evade reporting and can involve any amount, multiple days or institutions. A negative result does not establish legitimacy, and a positive result does not establish criminal intent. The CTR reporting threshold and a structuring determination are different questions. See [FFIEC guidance](https://bsaaml.ffiec.gov/manual/Appendices/08).

Test 7 confirms `flag_structuring_accounts`'s own per-account independence: a mixed batch spanning two accounts flags only the one actually exhibiting the pattern, by name.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_aml_structuring_detection.cpp -o 01_aml_structuring_detection
./01_aml_structuring_detection
```

**Sample input:** single payments, rolling aggregates below/at/above the illustrative threshold, dispersed payments and per-account isolation. These tests validate rule behavior, not legal structuring determinations.

@@OUT1@@

!!! warning "[COMMON TRAP] treating this section's own function as a complete real AML structuring detector"
    A real, deployed structuring detector considers many additional real signals this section's own narrow, illustrative function does not: transactions across MULTIPLE accounts controlled by the same real person (a well-documented refinement real structuring schemes use to spread transactions even further), transactions at multiple different branches or ATMs specifically chosen to avoid a single teller's own suspicion, and a real employee's own trained judgment about a customer's stated purpose for a transaction. `detect_structuring` catches exactly one real, narrow, and well-documented pattern -- multiple transactions from the SAME account, each individually under the threshold, summing past it within a short window -- and should be understood as one real signal among many a real compliance program would need, not a complete real detection system on its own.

## 29.2 Transaction Velocity Anomaly Detection

<!-- edge:concept-29.2:start -->
<figure class="chapter-figure" markdown="1">

[![Velocity counts need a defined time window](../images/technical/ch29/29-02.svg?v=ad73c0ba61b8)](../images/technical/ch29/29-02.svg?v=ad73c0ba61b8)

<figcaption>Illustrative transaction counts. State the window, entity key and threshold before interpreting an alert.</figcaption>
</figure>
<!-- edge:concept-29.2:end -->


### Intuition

A second, independent real signal has nothing to do with WHICH amounts moved, only how often. A real account suddenly transacting far more frequently than any ordinary customer would is a well-documented real signal for a compromised card or account being drained quickly, independent of whether any single amount looks unusual.

### The Concept, In Detail

`detect_velocity_anomaly` reapplies Section 27.2's own "stated parameters, never a wall clock" discipline to a frequency count instead of a deadline: every timing value in this file is a stated integer minute, and the function scans every possible rolling window in one account's own sorted history, flagging only when a window's own transaction count is strictly greater than a stated maximum. Tests 3 and 4 confirm the exact boundary of that maximum: exactly the stated count within a window is still within bounds, one more is not. Test 5 confirms a genuinely high total transaction count, properly spread out over a long enough span, is correctly NOT a velocity anomaly on its own -- the check is about a real BURST, not a real total.

Test 6 confirms `flag_velocity_accounts`'s own per-account independence, the same discipline Section 29.1 already established for a different signal entirely.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_transaction_velocity_anomaly_detection.cpp -o 02_transaction_velocity_anomaly_detection
./02_transaction_velocity_anomaly_detection
```

**Sample input:** two ordinary transactions well spaced apart, checked as normal; a real burst of four transactions within a short span, checked to be flagged with the exact maximum count found; the exact stated maximum count within the window, and one transaction past it, each checked at the precise boundary; four transactions properly spread across a longer span at a steady pace, checked as NOT a velocity anomaly despite the same total count as the flagged burst; and a mixed two-account batch checked to flag only the actually bursting account.

@@OUT2@@

!!! warning "[COMMON TRAP] treating a velocity anomaly as interchangeable with Section 29.1's own structuring signal"
    Section 29.1's structuring check and this section's own velocity check are deliberately independent: a burst of several small, ordinary purchases at different terminals is a real velocity anomaly with no structuring signal attached to it at all (no amount ever approaches the real CTR threshold), while a slow, patient structuring scheme spread across exactly the stated window with only two or three transactions total triggers Section 29.1's own check without ever coming close to this section's own frequency threshold. Treating either signal as a stand-in for the other -- assuming a clean velocity check means an account cannot be structuring, or the reverse -- would miss real, well-documented patterns that only this chapter's OTHER independent signal is actually built to catch.

## 29.3 Impossible Travel Geographic Detection

<!-- edge:concept-29.3:start -->
<figure class="chapter-figure" markdown="1">

[![Travel consistency compares distance with elapsed time](../images/technical/ch29/29-03.svg?v=e395141c4d4f)](../images/technical/ch29/29-03.svg?v=e395141c4d4f)

<figcaption>Check timestamp quality, location confidence and shared accounts. A rule hit can have benign causes.</figcaption>
</figure>
<!-- edge:concept-29.3:end -->


### Intuition

A third, independent real signal asks a different question again: could the same physical person plausibly have been at both of two transaction locations, given how little time passed between them? "Impossible travel" is a real, well-documented technique used across both AML and account-security fraud detection, built on a deliberately generous real bound -- fast enough to cover an actual commercial flight -- as an illustrative anomaly heuristic, with possible false positives and false negatives.

### The Concept, In Detail

`check_travel_pair` computes the real Haversine great-circle distance between two consecutive transaction locations -- a real, published formula, exactly the same "real, not invented for this book" standard Chapter 26.1's ABA checksum and Chapter 27.1's microprice formula already met -- divides by the real elapsed time, and flags only when the required speed strictly exceeds a stated, deliberately generous maximum of 900 km/h, the real cruising speed of a typical commercial jet. Test 2 confirms a genuinely plausible cross-country flight in 6 real hours is correctly NOT flagged, while Test 3 confirms the identical distance in 30 minutes is a real, genuine impossibility. Tests 4 and 5 confirm the exact speed boundary, constructed algorithmically from the real distance itself rather than a fragile hardcoded value, so the boundary check remains exact regardless of which two real cities it uses.

Test 6 confirms this check is genuinely about SPEED, not raw distance: a much shorter real distance (London to Paris) still triggers a flag when the elapsed time is short enough, and Test 7 confirms the same per-account independence this chapter's other two sections already established.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_impossible_travel_geographic_detection.cpp -o 03_impossible_travel_geographic_detection
./03_impossible_travel_geographic_detection
```

**Sample input:** the identical location one minute apart, checked to require zero speed; a real cross-country distance covered in a genuinely plausible 6 hours, checked as not flagged; the identical distance covered in 30 minutes, checked as a real impossibility; the exact speed boundary constructed algorithmically from the real distance, and one minute less than it, each checked precisely; a much shorter real distance covered in a very short time, checked to still be flagged on speed alone; and a mixed two-account batch checked to flag only the account exhibiting impossible travel.

@@OUT3@@

!!! warning "[COMMON TRAP] treating MAX_PLAUSIBLE_SPEED_KMH's own generosity as a weakness rather than a deliberate design choice"
    The 900 km/h threshold is an illustrative policy, not a universal physical boundary or false-positive guarantee. Location provenance, shared cards, card-not-present transactions and flight ground speeds can undermine the assumption that one person traveled between two recorded locations. Equal-time distant observations are flagged, same-location ties are accepted and reversed timestamps are rejected.

## 29.4 The Transaction Monitoring Disposition Engine

<!-- edge:concept-29.4:start -->
<figure class="chapter-figure" markdown="1">

[![Monitoring produces an explainable review record](../images/technical/ch29/29-04.svg?v=20acfab7ae0d)](../images/technical/ch29/29-04.svg?v=20acfab7ae0d)

<figcaption>Signals support a compliance workflow. The model does not determine guilt.</figcaption>
</figure>
<!-- edge:concept-29.4:end -->


### Intuition

This example routes accounts to review and does not implement SAR filing. Its two-signal threshold is an illustrative policy, not a legal rule or proof that filing is warranted. The recommendation name distinguishes routing from a completed filing; institutional responsibility and review governance are separate from electronic submission mechanics. FinCEN supports automated/batch transmission ([transmission methods](https://bsaefiling.fincen.gov/transmission-methods)); this example does not claim that US law universally prohibits automated submission.

### The Concept, In Detail

This example routes accounts to review and does not implement SAR filing. Its two-signal threshold is an illustrative policy, not a legal rule or proof that filing is warranted. The recommendation name distinguishes routing from a completed filing; institutional responsibility and review governance are separate from electronic submission mechanics. FinCEN supports automated/batch transmission ([transmission methods](https://bsaefiling.fincen.gov/transmission-methods)); this example does not claim that US law universally prohibits automated submission.

Test 6 confirms the disposition's own audit trail names the exact signals responsible, and Test 7 confirms every account in a mixed batch is assessed completely independently, closing this chapter's own recurring per-account-isolation discipline one final time.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_transaction_monitoring_disposition_engine.cpp -o 04_transaction_monitoring_disposition_engine
./04_transaction_monitoring_disposition_engine
```

**Sample input:** a completely clean account, checked to require no action; each of the three independent signal types fired alone, checked to produce the identical intermediate disposition; exactly two signals fired together, checked at the exact boundary of this engine's own strongest disposition; all three signals fired at once, checked to remain at that same strongest tier; the strongest disposition's own name checked, precisely, for the real legal distinction between a recommendation and an actual filing; a disposition's own audit trail checked to name the exact contributing signals; and a full multi-account batch checked for complete per-account independence.

@@OUT4@@

!!! warning "[COMMON TRAP] treating FILE_SAR_RECOMMENDED as equivalent to a SAR having actually been filed"
    This example routes accounts to review and does not implement SAR filing. Its two-signal threshold is an illustrative policy, not a legal rule or proof that filing is warranted. The recommendation name distinguishes routing from a completed filing; institutional responsibility and review governance are separate from electronic submission mechanics. FinCEN supports automated/batch transmission ([transmission methods](https://bsaefiling.fincen.gov/transmission-methods)); this example does not claim that US law universally prohibits automated submission.

## Chapter Summary

This chapter demonstrates three configurable screening signals and an auditable review-routing policy. None establishes criminal conduct or regulatory compliance. Geographic anomalies require reliable timestamps and location provenance; a speed threshold can produce both false positives and false negatives.

## Self-Check Questions

1. Section 29.1's `detect_structuring` requires every individual transaction in a flagged window to stay strictly under the CTR threshold. Explain what real, specific pattern this function would fail to catch if this individual-transaction check were removed, using only the window-sum condition.
2. Section 29.1's own COMMON TRAP box names a real refinement this section's own function does not implement: transactions spread across multiple accounts controlled by the same person. Explain concretely what additional real data this book's own `Transaction` struct would need to gain before that refinement could be implemented at all.
3. Section 29.2's Test 5 constructs 4 transactions spread across 45 minutes at a steady 15-minute pace, confirming this is NOT a velocity anomaly. Explain why this specific test is a stronger check of `detect_velocity_anomaly`'s own correctness than a test with only 2 well-spaced transactions would be.
4. Section 29.2's own COMMON TRAP box explains that a structuring pattern and a velocity anomaly are independent signals. Construct one concrete, realistic transaction sequence that would trigger Section 29.1's own structuring check while triggering NEITHER Section 29.2's velocity check nor Section 29.3's impossible-travel check.
5. Section 29.3's Test 4 constructs its own boundary using `dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH` directly, rather than a hardcoded number of minutes. Explain why this algorithmic construction makes the test more robust than picking a specific round number of minutes and checking it against the stated 900 km/h constant.
6. Section 29.3's own COMMON TRAP box explains the real tradeoff behind choosing a generous 900 km/h threshold. Describe one concrete real scenario in which a MORE generous threshold (say, 1200 km/h) would cause this section's own check to miss a real, genuinely impossible pattern that the stated 900 km/h threshold would have caught.
7. Why is a two-signal review-routing threshold a policy choice rather than a legal SAR filing rule?
8. Why must a recommendation be distinguished from a completed filing, regardless of the submission mechanism?
9. Across Sections 29.1 through 29.3, each section's own detection function operates on ONE account's own transaction history at a time, with a separate aggregator function handling multiple accounts. Explain one concrete real advantage of keeping these two responsibilities in separate functions, rather than combining per-account detection and multi-account aggregation into a single function.
10. This chapter's own introduction states its central shift from every prior finance chapter: evaluating a SEQUENCE of transactions rather than any single one in isolation. Identify the ONE specific design choice, present in all three of Sections 29.1, 29.2, and 29.3, that makes this sequence-based evaluation possible, and explain why Chapters 26 and 28's own per-document and per-claim checks did not need it.


## Where We Go Next

This chapter demonstrates three configurable screening signals and an auditable review-routing policy. None establishes criminal conduct or regulatory compliance. Geographic anomalies require reliable timestamps and location provenance; a speed threshold can produce both false positives and false negatives.

## Worked Solutions

**1.** Without the individual-transaction check, `detect_structuring` would also flag a SINGLE large, ordinary transaction whose own amount happens to exceed the threshold together with a few small, unrelated transactions in the same window -- for instance, a genuine $12,000 transaction (which already, correctly, triggers a real CTR directly and reports itself) alongside a completely unrelated $50 purchase in the same window would sum to $12,050, past the threshold, but this is not structuring at all, since the $12,000 transaction was never hidden below the threshold in the first place. The individual-transaction check exists specifically to isolate the real pattern this function targets: amounts DELIBERATELY kept under the threshold, not merely a window whose sum happens to cross it for any reason at all.

**2.** The `Transaction` struct would need a field identifying the REAL PERSON behind an account, independent of the account_id itself -- for instance, a real customer identifier (a legal name, a real government-issued ID number, or an internal customer-relationship identifier) shared across every account that same real person controls. Without this field, `detect_structuring`'s own per-account grouping in `flag_structuring_accounts` has no way to recognize that two different `account_id` values might belong to the same real person spreading a structuring pattern across both of them -- the function would need to group by this new customer-identity field instead of (or in addition to) `account_id` to catch that specific real refinement.

**3.** A test with only 2 well-spaced transactions could pass `detect_velocity_anomaly` correctly even under a subtly WRONG implementation -- for instance, one that only checks the count of the very FIRST window found, rather than scanning every possible rolling window for the true maximum -- since with only 2 transactions there is little room for such a bug to produce a visibly wrong answer. Test 5's own 4 transactions at a steady pace force a CORRECT implementation to check MULTIPLE overlapping windows (starting at minute 0, 15, 30, and 45) and confirm the maximum count across every one of them is still within bounds, which would catch a bug that only checked one window, or that failed to correctly recompute the count for each new rolling window start.

**4.** Three transactions of $4,000 each, all from account ACC-X, all made at the SAME single ATM location (so no geographic movement occurs at all, keeping Section 29.3's own impossible-travel check clean), spread exactly 1 day apart across days 1, 2, and 3 -- comfortably within Section 29.1's own 3-day structuring window (summing to $12,000, correctly flagged) but far too infrequent (only 3 transactions across 3 full days) to approach Section 29.2's own velocity threshold of more than 3 transactions within a single 30-MINUTE window, since the transactions here are separated by entire days, not minutes.

**5.** A hardcoded round number of minutes checked against the stated 900 km/h constant would only remain a valid boundary test for THIS SPECIFIC real distance (New York to Los Angeles) -- if a future edit to this section changed which two cities Test 4 used, or adjusted the stated `MAX_PLAUSIBLE_SPEED_KMH` value for any reason, a hardcoded minute value would silently stop testing the actual boundary at all, landing at some arbitrary point either safely inside or past it without anyone noticing until the test's own assertions started failing for the wrong reason. Constructing `boundary_hours` directly from `dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH` guarantees the test always lands EXACTLY on the true boundary implied by whatever real distance and real speed constant the file currently uses, remaining a true boundary test even if either value changes later.

**6.** A pair requiring 1,050 km/h triggers a 900 km/h heuristic but not a 1,200 km/h heuristic. Neither threshold proves fraud: aircraft ground speeds, shared accounts, card-not-present activity and location errors can invalidate the physical-travel assumption.

**7.** Both functions share the identical structural approach: count how many independent, boolean, pre-named signals fired, and map that count onto an ordered set of outcomes via simple integer comparisons (`count >= 2`, `count == 1`), never weighting any one signal more heavily than another. The two functions differ in what they map that count ONTO: Section 28.3's `assess_fraud_likelihood` returns a `FraudLikelihood` LEVEL (LOW/MEDIUM/HIGH) describing a single zone's own suspicion, feeding into a SEPARATE disposition function (Section 28.4's own `assess_claim_disposition`) one level higher, while Section 29.4's `assess_account` maps its count DIRECTLY onto the final, actionable `MonitoringDisposition` in one single function -- there is no separate zone-level/claim-level split in this chapter, since Section 29.4's own three signals are already computed at the account level Section 29.4 itself operates on.

**8.** This example routes accounts to review and does not implement SAR filing. Its two-signal threshold is an illustrative policy, not a legal rule or proof that filing is warranted. The recommendation name distinguishes routing from a completed filing; institutional responsibility and review governance are separate from electronic submission mechanics. FinCEN supports automated/batch transmission ([transmission methods](https://bsaefiling.fincen.gov/transmission-methods)); this example does not claim that US law universally prohibits automated submission.

**9.** Keeping per-account detection (`detect_structuring`, `detect_velocity_anomaly`, `detect_impossible_travel`) and multi-account aggregation (`flag_structuring_accounts`, `flag_velocity_accounts`, `flag_impossible_travel_accounts`) as separate functions lets each per-account function be tested, and reasoned about, in complete isolation -- exactly as this section's own Tests 1 through 6 in each of the first three sections do, using a single account's own history directly, with no grouping or sorting logic involved at all. A combined function would force every single-account test to also exercise the grouping and sorting logic every time, making it harder to tell whether a test failure came from the real detection LOGIC itself or from the unrelated bookkeeping of splitting a mixed batch by account -- the same general "test the core logic separately from the batch-orchestration wrapper around it" principle this book's own capstone engines have followed since Chapter 26.4.

**10.** The one specific design choice present in all three of Sections 29.1, 29.2, and 29.3 is that each section's own core detection function (`detect_structuring`, `detect_velocity_anomaly`, `detect_impossible_travel`) takes a WHOLE VECTOR of an account's own transaction history as its input, rather than a single transaction -- and internally scans across multiple entries in that vector (a rolling window, or consecutive pairs) to compute its own result. Chapter 26's own MICR/invoice/amount checks and Chapter 28's own duplicate-photo/timestamp checks each operate on ONE document, ONE photo, or ONE claim's own data at a time, since every real pattern those two chapters target (a forged checksum, a reused photo, a mismatched amount) is genuinely visible within a SINGLE item's own data -- this chapter's own three real AML patterns, by contrast, are only real patterns AT ALL when considered across multiple transactions together, which is exactly why accepting a whole sequence as input, rather than one transaction, is the one design choice this chapter's sequence-based evaluation could not do without.
