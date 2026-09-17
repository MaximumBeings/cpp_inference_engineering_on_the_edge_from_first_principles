# Chapter 29: On-Device AML Transaction Anomaly Monitoring at ATMs and POS Terminals

**What you will understand by the end of this chapter:**

- How to detect "structuring" (smurfing) -- a real, specifically named federal offense under the US Bank Secrecy Act -- by scanning a rolling window across a SEQUENCE of transactions, since no individual transaction in the pattern is suspicious on its own.
- How to build a real transaction-velocity anomaly check, reapplying Section 27.2's own "stated parameters, never a wall clock" discipline to a frequency-based signal instead of a deadline.
- How to build a real "impossible travel" check from the real, published Haversine great-circle-distance formula, catching a required travel speed that could not possibly be genuine, while deliberately staying generous enough never to flag a real traveler.
- How to combine three independent, sequence-based signals into one per-account disposition, closing this book's four-chapter run through real financial-industry deployments with a final, real, legally precise never-suppress-a-flag discipline: a Suspicious Activity Report can only ever be RECOMMENDED by this engine, never filed by it.

**What you need to know first:**

- Section 27.2's own discipline of deriving timing behavior entirely from stated, labeled parameters rather than a wall clock is reapplied in Section 29.2, this time to a transaction-frequency window instead of a processing deadline.
- Section 28.2's own stated-integer-day-count convention for a claim's own timeline is reapplied in Section 29.1, this time to a rolling structuring-detection window.
- Chapter 26.1's own real, published ABA checksum and Chapter 27.1's own real, published microprice formula are joined in this chapter by a third real, published formula: the Haversine great-circle-distance formula, used in Section 29.3.
- This chapter's own central shift from every prior finance chapter: every check in this chapter evaluates a SEQUENCE of transactions from the same account together, since no single transaction in any of this chapter's three real patterns is suspicious in isolation.

---

Chapter 28 turned to a claim's own photographic and temporal evidence. This chapter closes this book's four-chapter run through real financial-industry deployments with a genuinely different real constraint once more: catching a pattern that only becomes visible across a SEQUENCE of transactions from the same account, where no single transaction in the sequence is suspicious by itself. Each of this chapter's four sections builds one real, independent, sequence-based AML signal, and closes with a capstone that combines all three into one auditable, per-account disposition -- never an automated filing, since real law reserves that decision for a human.

## 29.1 AML Structuring (Smurfing) Detection

### Intuition

"Structuring" is a real, specifically named offense under the US Bank Secrecy Act: deliberately breaking a large cash transaction into several smaller ones, each kept under the real $10,000 Currency Transaction Report threshold, specifically to avoid the reporting the full amount would trigger. No individual transaction in the pattern looks suspicious -- the pattern is only visible across several transactions from the same account within a real, short window of time.

### The Concept, In Detail

`detect_structuring` scans every possible rolling window within one account's own sorted transaction history, flagging a window only when BOTH real conditions hold: every individual transaction inside it stays strictly under the real CTR threshold, and the window's own sum is strictly greater than it. Tests 5 and 6 confirm this section's own precise real legal boundary: the actual US regulation applies to a transaction "in excess of" $10,000 -- strictly more, not $10,000 exactly -- and this section's own code matches that real distinction precisely rather than rounding it away. Test 4 confirms the check is a genuine ROLLING window, not a whole-history sum: the identical three amounts that trigger a flag when clustered within the window do not trigger one when spread across nine days instead.

Test 7 confirms `flag_structuring_accounts`'s own per-account independence: a mixed batch spanning two accounts flags only the one actually exhibiting the pattern, by name.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_aml_structuring_detection.cpp -o 01_aml_structuring_detection
./01_aml_structuring_detection
```

**Sample input:** a single transaction well over the real CTR threshold, checked to be a direct CTR case rather than structuring; three transactions each under the threshold but summing well past it within the stated window, checked to be flagged; the identical pattern of small transactions with a sum safely under the threshold, checked as legitimate; the identical flagged amounts spread across a much longer span, checked as NOT flagged since no single window contains enough of them; the exact real regulatory boundary of $10,000.00 exactly, and one cent past it; and a mixed two-account batch checked to flag only the account actually exhibiting the pattern.

@@OUT1@@

!!! warning "[COMMON TRAP] treating this section's own function as a complete real AML structuring detector"
    A real, deployed structuring detector considers many additional real signals this section's own narrow, illustrative function does not: transactions across MULTIPLE accounts controlled by the same real person (a well-documented refinement real structuring schemes use to spread transactions even further), transactions at multiple different branches or ATMs specifically chosen to avoid a single teller's own suspicion, and a real employee's own trained judgment about a customer's stated purpose for a transaction. `detect_structuring` catches exactly one real, narrow, and well-documented pattern -- multiple transactions from the SAME account, each individually under the threshold, summing past it within a short window -- and should be understood as one real signal among many a real compliance program would need, not a complete real detection system on its own.

## 29.2 Transaction Velocity Anomaly Detection

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

### Intuition

A third, independent real signal asks a different question again: could the same physical person plausibly have been at both of two transaction locations, given how little time passed between them? "Impossible travel" is a real, well-documented technique used across both AML and account-security fraud detection, built on a deliberately generous real bound -- fast enough to cover an actual commercial flight -- so a genuine cross-country traveler is never flagged.

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
    A tighter, less generous speed threshold would catch MORE real anomalies, but at a real, serious cost: a genuine customer who legitimately flew from one real city to another between two ordinary transactions would be flagged as fraudulent, purely for traveling normally. This section's own stated 900 km/h bound is deliberately set close to the real physical limit of ordinary human travel specifically so that ONLY a genuinely impossible required speed -- one no real commercial transportation could achieve -- is ever flagged, exactly the same "real, useful signal without an unacceptable false-positive cost" design tradeoff Section 26.1's own signature-comparison tolerance and Section 28.1's own duplicate-photo threshold already made in their own respective domains.

## 29.4 The Transaction Monitoring Disposition Engine

### Intuition

This chapter's capstone, and the fourth and final financial-industry chapter this book set out to build, combines Section 29.1's structuring signal, Section 29.2's velocity signal, and Section 29.3's impossible-travel signal into one per-account disposition -- with a real, legally precise boundary specific to this domain: under real US law, a Suspicious Activity Report can only ever be filed by a human compliance officer, never by an automated system.

### The Concept, In Detail

`assess_account` counts how many of the three independent signals fired for one account -- Test 2 confirms all three are weighted identically, each alone producing `INTERNAL_REVIEW`. Test 3 confirms the exact boundary at a count of two signals, driving this engine's own strongest disposition, `FILE_SAR_RECOMMENDED`. Test 5 is this section's own central, real legal precision check: the strongest disposition's own name is checked, precisely, to end in "RECOMMENDED" rather than stating a filing has actually occurred -- a real, legally meaningful distinction this engine's own naming enforces structurally, not merely by convention.

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
    `MonitoringDisposition::FILE_SAR_RECOMMENDED` means only that TWO OR MORE independent real signals fired for this account, which real compliance practice treats as strong enough evidence to warrant a human compliance officer's own real review and, potentially, a real SAR filing -- it does not mean a SAR has been filed, and this engine has no code path that could ever file one automatically, since real US law reserves that decision for a human at a regulated institution. A downstream system that treated this disposition as equivalent to an actual completed filing would misrepresent this engine's own real legal role, exactly the same category of harm Chapter 28.4's own COMMON TRAP box already named for conflating a claim routed to human review with a claim that had already been denied.

## Chapter Summary

This chapter closed this book's four-chapter run through real financial-industry deployments with a genuinely different real constraint: catching a pattern visible only across a sequence of transactions, never in any single one alone. Section 29.1 built a real structuring detector matching the exact legal boundary of the US Bank Secrecy Act's own CTR threshold. Section 29.2 reapplied Section 27.2's own stated-parameter timing discipline to a transaction-frequency burst instead of a deadline. Section 29.3 built a real "impossible travel" check from the real, published Haversine formula, deliberately generous enough to never flag a genuine traveler. Section 29.4 closed the chapter, and this book's four-chapter finance arc, by combining all three signals into one auditable disposition with a final, real legal precision: this engine can only ever recommend a Suspicious Activity Report, never file one, since real law reserves that decision for a human.

## Self-Check Questions

1. Section 29.1's `detect_structuring` requires every individual transaction in a flagged window to stay strictly under the CTR threshold. Explain what real, specific pattern this function would fail to catch if this individual-transaction check were removed, using only the window-sum condition.
2. Section 29.1's own COMMON TRAP box names a real refinement this section's own function does not implement: transactions spread across multiple accounts controlled by the same person. Explain concretely what additional real data this book's own `Transaction` struct would need to gain before that refinement could be implemented at all.
3. Section 29.2's Test 5 constructs 4 transactions spread across 45 minutes at a steady 15-minute pace, confirming this is NOT a velocity anomaly. Explain why this specific test is a stronger check of `detect_velocity_anomaly`'s own correctness than a test with only 2 well-spaced transactions would be.
4. Section 29.2's own COMMON TRAP box explains that a structuring pattern and a velocity anomaly are independent signals. Construct one concrete, realistic transaction sequence that would trigger Section 29.1's own structuring check while triggering NEITHER Section 29.2's velocity check nor Section 29.3's impossible-travel check.
5. Section 29.3's Test 4 constructs its own boundary using `dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH` directly, rather than a hardcoded number of minutes. Explain why this algorithmic construction makes the test more robust than picking a specific round number of minutes and checking it against the stated 900 km/h constant.
6. Section 29.3's own COMMON TRAP box explains the real tradeoff behind choosing a generous 900 km/h threshold. Describe one concrete real scenario in which a MORE generous threshold (say, 1200 km/h) would cause this section's own check to miss a real, genuinely impossible pattern that the stated 900 km/h threshold would have caught.
7. Section 29.4's `assess_account` counts fired signals using a simple integer count, exactly like Section 28.3's own `assess_fraud_likelihood`. Identify one structural similarity and one structural difference between these two functions' own real designs, referring to their actual code.
8. Section 29.4's own introduction explains a real, legally grounded reason its strongest disposition is named `FILE_SAR_RECOMMENDED` rather than `FILE_SAR`. Explain why Chapter 26.4's own `AUTO_REJECT` disposition did not need this same naming caution for its own domain.
9. Across Sections 29.1 through 29.3, each section's own detection function operates on ONE account's own transaction history at a time, with a separate aggregator function handling multiple accounts. Explain one concrete real advantage of keeping these two responsibilities in separate functions, rather than combining per-account detection and multi-account aggregation into a single function.
10. This chapter's own introduction states its central shift from every prior finance chapter: evaluating a SEQUENCE of transactions rather than any single one in isolation. Identify the ONE specific design choice, present in all three of Sections 29.1, 29.2, and 29.3, that makes this sequence-based evaluation possible, and explain why Chapters 26 and 28's own per-document and per-claim checks did not need it.

## Where We Go Next

This chapter closed this book's four-chapter run through real financial-industry deployments -- check and invoice fraud, low-latency market signals, insurance claims, and now sequence-based AML monitoring -- each grounded in a real, well-documented pattern from its own domain, and each ending in an auditable disposition that names its own reasoning rather than hiding behind a single opaque score. With all four now complete, Part 5 itself is complete: twelve real, deployed case-study systems spanning industrial inspection, retail, medical imaging, document intelligence, security and authenticity, natural language photo editing, personal cameras, and now four real financial-industry deployments. The next step revisits Chapter 32's own closing count of real edge deployments across this entire book, now that Part 5's own final real total is settled.

## Worked Solutions

**1.** Without the individual-transaction check, `detect_structuring` would also flag a SINGLE large, ordinary transaction whose own amount happens to exceed the threshold together with a few small, unrelated transactions in the same window -- for instance, a genuine $12,000 transaction (which already, correctly, triggers a real CTR directly and reports itself) alongside a completely unrelated $50 purchase in the same window would sum to $12,050, past the threshold, but this is not structuring at all, since the $12,000 transaction was never hidden below the threshold in the first place. The individual-transaction check exists specifically to isolate the real pattern this function targets: amounts DELIBERATELY kept under the threshold, not merely a window whose sum happens to cross it for any reason at all.

**2.** The `Transaction` struct would need a field identifying the REAL PERSON behind an account, independent of the account_id itself -- for instance, a real customer identifier (a legal name, a real government-issued ID number, or an internal customer-relationship identifier) shared across every account that same real person controls. Without this field, `detect_structuring`'s own per-account grouping in `flag_structuring_accounts` has no way to recognize that two different `account_id` values might belong to the same real person spreading a structuring pattern across both of them -- the function would need to group by this new customer-identity field instead of (or in addition to) `account_id` to catch that specific real refinement.

**3.** A test with only 2 well-spaced transactions could pass `detect_velocity_anomaly` correctly even under a subtly WRONG implementation -- for instance, one that only checks the count of the very FIRST window found, rather than scanning every possible rolling window for the true maximum -- since with only 2 transactions there is little room for such a bug to produce a visibly wrong answer. Test 5's own 4 transactions at a steady pace force a CORRECT implementation to check MULTIPLE overlapping windows (starting at minute 0, 15, 30, and 45) and confirm the maximum count across every one of them is still within bounds, which would catch a bug that only checked one window, or that failed to correctly recompute the count for each new rolling window start.

**4.** Three transactions of $4,000 each, all from account ACC-X, all made at the SAME single ATM location (so no geographic movement occurs at all, keeping Section 29.3's own impossible-travel check clean), spread exactly 1 day apart across days 1, 2, and 3 -- comfortably within Section 29.1's own 3-day structuring window (summing to $12,000, correctly flagged) but far too infrequent (only 3 transactions across 3 full days) to approach Section 29.2's own velocity threshold of more than 3 transactions within a single 30-MINUTE window, since the transactions here are separated by entire days, not minutes.

**5.** A hardcoded round number of minutes checked against the stated 900 km/h constant would only remain a valid boundary test for THIS SPECIFIC real distance (New York to Los Angeles) -- if a future edit to this section changed which two cities Test 4 used, or adjusted the stated `MAX_PLAUSIBLE_SPEED_KMH` value for any reason, a hardcoded minute value would silently stop testing the actual boundary at all, landing at some arbitrary point either safely inside or past it without anyone noticing until the test's own assertions started failing for the wrong reason. Constructing `boundary_hours` directly from `dist_nyc_la / MAX_PLAUSIBLE_SPEED_KMH` guarantees the test always lands EXACTLY on the true boundary implied by whatever real distance and real speed constant the file currently uses, remaining a true boundary test even if either value changes later.

**6.** Consider a genuinely fraudulent pattern where a compromised card is used in New York and, exactly 3 hours later, used again in London -- a real distance of roughly 5,570 km, requiring a real speed of about 1,857 km/h, which is well beyond what any real commercial passenger transportation could achieve (commercial supersonic passenger flight ended with Concorde's retirement in 2003) and would be correctly flagged as impossible under the stated 900 km/h threshold. Raising the threshold to 1,200 km/h would still catch this specific example, but a slightly less extreme version of the identical fraud pattern -- say, a required speed of exactly 1,050 km/h, still an entirely real impossibility for any genuine traveler -- would slip through undetected under the more generous 1,200 km/h bound, while the stated 900 km/h threshold would have caught it correctly.

**7.** Both functions share the identical structural approach: count how many independent, boolean, pre-named signals fired, and map that count onto an ordered set of outcomes via simple integer comparisons (`count >= 2`, `count == 1`), never weighting any one signal more heavily than another. The two functions differ in what they map that count ONTO: Section 28.3's `assess_fraud_likelihood` returns a `FraudLikelihood` LEVEL (LOW/MEDIUM/HIGH) describing a single zone's own suspicion, feeding into a SEPARATE disposition function (Section 28.4's own `assess_claim_disposition`) one level higher, while Section 29.4's `assess_account` maps its count DIRECTLY onto the final, actionable `MonitoringDisposition` in one single function -- there is no separate zone-level/claim-level split in this chapter, since Section 29.4's own three signals are already computed at the account level Section 29.4 itself operates on.

**8.** Chapter 26.4's own `AUTO_REJECT` disposition governs a bank's own internal decision about whether to CLEAR a specific check for processing -- a real, ordinary business decision a bank's own automated system is legally and operationally permitted to make about its own transaction processing. Section 29.4's own `FILE_SAR_RECOMMENDED`, by contrast, concerns a specific REGULATORY FILING (a Suspicious Activity Report) that real US law under the Bank Secrecy Act explicitly reserves for a human decision-maker at a regulated institution -- no amount of business judgment delegated to an automated check-clearing system changes who is legally permitted to file a SAR, which is exactly why this specific disposition name needed the extra legal precision that an internal processing decision like `AUTO_REJECT` never required.

**9.** Keeping per-account detection (`detect_structuring`, `detect_velocity_anomaly`, `detect_impossible_travel`) and multi-account aggregation (`flag_structuring_accounts`, `flag_velocity_accounts`, `flag_impossible_travel_accounts`) as separate functions lets each per-account function be tested, and reasoned about, in complete isolation -- exactly as this section's own Tests 1 through 6 in each of the first three sections do, using a single account's own history directly, with no grouping or sorting logic involved at all. A combined function would force every single-account test to also exercise the grouping and sorting logic every time, making it harder to tell whether a test failure came from the real detection LOGIC itself or from the unrelated bookkeeping of splitting a mixed batch by account -- the same general "test the core logic separately from the batch-orchestration wrapper around it" principle this book's own capstone engines have followed since Chapter 26.4.

**10.** The one specific design choice present in all three of Sections 29.1, 29.2, and 29.3 is that each section's own core detection function (`detect_structuring`, `detect_velocity_anomaly`, `detect_impossible_travel`) takes a WHOLE VECTOR of an account's own transaction history as its input, rather than a single transaction -- and internally scans across multiple entries in that vector (a rolling window, or consecutive pairs) to compute its own result. Chapter 26's own MICR/invoice/amount checks and Chapter 28's own duplicate-photo/timestamp checks each operate on ONE document, ONE photo, or ONE claim's own data at a time, since every real pattern those two chapters target (a forged checksum, a reused photo, a mismatched amount) is genuinely visible within a SINGLE item's own data -- this chapter's own three real AML patterns, by contrast, are only real patterns AT ALL when considered across multiple transactions together, which is exactly why accepting a whole sequence as input, rather than one transaction, is the one design choice this chapter's sequence-based evaluation could not do without.
