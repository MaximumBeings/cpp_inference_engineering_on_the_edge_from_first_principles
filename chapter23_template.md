# Chapter 23: Trust and Authenticity at the Point of Sale: Counterfeits, Reconciliation, Expense Audits, and Luxury Goods

**What you will understand by the end of this chapter:**

- How to build a real perceptual image hash (the "difference hash," dHash, algorithm used in production image-deduplication systems) that catches stolen stock photography reused across unrelated e-commerce sellers, and a fully auditable seller trust score whose own aggregate value can never suppress a single specific, named red flag.
- How to implement the classic Levenshtein edit-distance algorithm from scratch and use it to reconcile an OCR-noisy receipt against a bank or card network's own clean transaction record, honestly reporting AMBIGUOUS when more than one transaction equally fits.
- How to build a real, checkable expense-policy rules engine that reports a claimed amount and a policy-approved amount side by side for every line item, never silently substituting one for the other without naming the exact rule behind the difference.
- How to implement the real, well-known Luhn check-digit algorithm (the same algorithm that validates credit-card numbers) as a luxury brand's own stated serial-number checksum, and why a single hard-failing structural check must never be outvoted by two other checks that happen to pass.

**What you need to know first:**

- Section 22.1's own honest AMBIGUOUS-refusal discipline (naming every fitting candidate rather than silently picking one) is the exact pattern Section 23.2's own receipt-to-transaction reconciliation reapplies when two transactions both fit every real matching rule.
- Section 21.3's own `CostRange` discipline -- reporting a claimed figure and a computed figure side by side, with a named reason for any gap, never silently substituting one for the other -- is the exact pattern Section 23.3's own expense-policy engine reapplies to a claimed amount versus a policy-approved amount.
- This book's own recurring structural-override discipline, first built as Chapter 20.1's gated sign-off and reapplied at Chapter 21.2's per-field handwriting review and Chapter 22.2's AA-only enforcement: a single specific, named red flag (Section 23.1) or a single hard-failing check (Section 23.4) is never outvoted by an otherwise-favorable aggregate score or a majority of passing checks.

---

Chapter 22 turned a continuous stream into a structured, auditable narrative. This chapter turns to a related but distinct real pattern: trust and authenticity at the point of a real financial transaction, where the stakes are a fraudulent purchase, an unreconciled charge, a policy-violating expense claim, or a counterfeit luxury item sold as genuine. Each of this chapter's four sections builds a real, from-scratch algorithm with its own well-established provenance outside this book -- a perceptual image hash, the classic Levenshtein edit distance, and the Luhn checksum are all real, independently famous algorithms this chapter implements and verifies against their own known reference values, not techniques this book invented for the occasion. And each section reapplies this book's own recurring discipline to a financial context: a named red flag is never silently smoothed over by a favorable aggregate score, an honest range or an honest ambiguity is reported rather than a false single answer, and a structural screening pass is never overstated into the final human judgment it is not.

## 23.1 A Brand-Reference Database and Counterfeit-Screening Engine for E-Commerce Listings

### Intuition

A real counterfeit-screening system cannot inspect an item in a shopper's hand, but it can compute several real, checkable properties of a LISTING'S OWN STATED DATA: whether its own photo has already been used by an unrelated seller, whether its price is implausibly below what a genuine item ever sells for, and whether its own description contains language sellers of counterfeit goods commonly use. This section builds all three from scratch, and adds a real, fully auditable seller trust score that never gets to overrule a single specific red flag.

### The Concept, In Detail

`compute_dhash` is a real, from-scratch implementation of the difference hash (dHash) algorithm production image-deduplication systems actually use: resize to 9x8, then encode 64 real horizontal pixel comparisons into a 64-bit value -- Test 1 confirms two independently generated but pixel-identical images hash identically (Hamming distance 0), while a clearly different image hashes far apart, and Test 2 confirms a photo reused across two different sellers is flagged as a duplicate stock image, naming the other seller, while a genuinely unique photo is not. `screen_listing` also checks a real, stated authentic-price floor and a case-insensitive prohibited-keyword scan -- Test 3 through Test 5 confirm each check's own exact boundary and named detail.

`compute_seller_trust_score` combines three fully auditable, individually named components -- account age, listing volume, and dispute rate -- into a single real, hand-verifiable score, confirmed exactly in Test 6 for three real seller profiles. Test 7 is this section's own central honesty check: the identical seller, with a perfect trust score of 100.0, requires no review for a clean listing but STILL requires manual review the moment their own listing description contains a single prohibited keyword -- proving a perfect aggregate score can never suppress one specific, named red flag.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_counterfeit_screening_and_seller_trust.cpp -o 01_counterfeit_screening_and_seller_trust
./01_counterfeit_screening_and_seller_trust
```

**Sample input:** two pixel-identical checkerboard images and one clearly different gradient image checked under the real dHash algorithm; a photo reused across two sellers checked to flag a duplicate, naming the other seller, while a genuinely unique photo does not; prohibited keywords caught case-insensitively inside longer listing text while a legitimate description is not falsely flagged; an authentic-price floor and a weight-spec range each checked exactly at their own stated boundaries; a seller trust score checked against exact hand-computed values for three real seller profiles; and a perfect-trust-score seller's listing checked to still require manual review the moment it contains one prohibited keyword.

@@OUT1@@

!!! warning "[COMMON TRAP] treating a low aggregate trust score as the only real signal worth acting on"
    It would be a real, meaningful screening system to flag any seller whose own trust score falls below a stated floor for manual review and stop there -- and this section's own `evaluate_listing` does include exactly that floor. But Test 7 exists specifically to show that floor is not the whole property this section's own risk profile requires: a seller with a PERFECT trust score of 100.0 -- a full year of history, a hundred listings, zero disputes -- can still list an item whose own description contains "1:1 mirror quality," and no amount of accumulated seller history should make that specific, named red flag disappear into a favorable average. `evaluate_listing`'s own rule -- ANY flag OR a low trust score triggers review -- is what keeps a single bad listing from hiding behind an otherwise-excellent seller record.

## 23.2 Receipt-to-Transaction Reconciliation

### Intuition

A receipt extracted by an OCR or vision-language pipeline, the same real kind of pipeline this book built in Chapter 21.1, can carry small per-character extraction errors; a bank or card network's own transaction record carries the true merchant name as the payment network itself recorded it. Reconciling the two needs a real fuzzy-matching algorithm for the merchant name, a real tolerance for an added tip in the settled amount, and a real settlement-date window -- and an honest refusal to guess when more than one transaction equally fits.

### The Concept, In Detail

`levenshtein_distance` is the real, classic edit-distance algorithm, confirmed in Test 1 against its own famous exact reference value (`KITTEN` to `SITTING` is distance 3). `merchant_names_fuzzy_match` normalizes and applies a real, stated edit-distance ratio threshold built specifically for OCR-introduced noise -- Test 2 confirms a single-character OCR misread ("WALGREEN5" for "WALGREENS") fuzzy-matches correctly while a genuinely different merchant does not. `amounts_match` and `within_settlement_window` each encode a real, stated policy: a settled amount may exceed a receipt's own printed subtotal by an added tip, but only up to a real, stated cap, confirmed exactly at that boundary in Test 3; and a settlement can only occur ON OR AFTER its own purchase date, confirmed exactly at its own real window boundary in Test 4.

`reconcile_receipt` is this section's own central honesty check, reapplying Section 22.1's own AMBIGUOUS-refusal discipline to a financial-matching context: Test 6 constructs two transactions that both genuinely fit every real matching rule -- same merchant, same plausible tip-adjusted amount, both within the settlement window -- and confirms the function reports `AMBIGUOUS`, naming both candidates, rather than silently picking the nearer one. `find_transactions_missing_receipts` closes the loop in the other direction: Test 8 confirms a transaction with no receipt on file at all is flagged separately, a real, useful signal for an expense audit.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_receipt_transaction_reconciliation.cpp -o 02_receipt_transaction_reconciliation
./02_receipt_transaction_reconciliation
```

**Sample input:** the real Levenshtein algorithm checked against its own famous KITTEN-to-SITTING reference value and two further hand-computable cases; an OCR-noisy merchant name checked to fuzzy-match its own true name while a genuinely different merchant does not; amount matching checked against an exact tip cap boundary; a settlement-date window checked against its own exact boundary and against a settlement dated before its own purchase; a receipt with exactly one fitting transaction checked to report MATCHED; two equally-fitting transactions checked to report AMBIGUOUS, naming both; a receipt with no fitting transaction checked to report MISSING_TRANSACTION; and a transaction with no matching receipt checked to be flagged as missing one.

@@OUT2@@

!!! warning "[COMMON TRAP] treating this section's own OCR-noise fuzzy match as a general merchant-name matcher"
    `merchant_names_fuzzy_match` is built and tuned specifically for the kind of noise a receipt's own OCR extraction introduces -- a single misread character on an otherwise-correct name. It is tempting to assume the same function would also handle the different real problem of payment-processor descriptor mangling, where a transaction descriptor reads "SQ *STARBUCKS COFFEE" for a receipt's own "Starbucks" -- a difference not of a few misread characters but of an entirely different real string, prefixed and suffixed by processor-specific codes this section's own edit-distance ratio was never tuned to absorb. A production reconciliation system needs a SEPARATE real technique -- token-based matching against a known table of processor prefixes, for instance -- for that different problem; treating one section's own narrow, real fix as a general solution to every real merchant-name mismatch would produce exactly the false confidence this book's own honesty discipline exists to prevent.

## 23.3 Automated Expense-Report Policy Enforcement

### Intuition

A real corporate travel policy is a real, checkable rules engine: a per-category spending cap, a per-mile reimbursement rate, a category that is never reimbursable regardless of amount, and a receipt-required threshold. The one discipline this section adds on top of a straightforward rules engine: never let the policy-computed APPROVED amount silently replace the employee's own CLAIMED amount in the report's own total without naming, for every discrepancy, the specific rule behind it.

### The Concept, In Detail

`evaluate_line_item` applies four real, independently checkable rules by category: a real per-city-tier lodging cap (Test 2 confirms an over-cap claim in the cheapest tier is flagged against ITS OWN tier's cap, not a different one), a real daily meals cap (Test 1), a real per-mile mileage rate the claim is checked against rather than merely capped (Test 3 confirms a mismatched mileage claim is approved at the CORRECTLY COMPUTED figure while naming both the claimed and expected values), and a blanket alcohol exclusion that approves $0.00 regardless of the claimed amount and never additionally triggers the receipt-required check on top of its own exclusion (Test 4). The receipt-required threshold is confirmed exact at its own real boundary in Test 5.

`evaluate_expense_report`'s own Test 6 is this section's central honesty check, reapplying Section 21.3's own `CostRange` discipline to a policy-enforcement context: a 4-item report's own CLAIMED total still includes a $90 meal claim's full $90 even though only $75 is approved, and still includes an excluded $30 alcohol item's full $30 even though $0 is approved -- the report's own claimed figure is never silently reduced anywhere in this file; only the separate approved figure and each item's own named violation tell the real story.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_expense_report_policy_enforcement.cpp -o 03_expense_report_policy_enforcement
./03_expense_report_policy_enforcement
```

**Sample input:** a meal claim under, exactly at, and over its own real daily cap; lodging claims checked against all 3 real per-city-tier caps, with an over-cap claim in the cheapest tier flagged against its own tier's cap; a mileage claim matching and mismatching its own computed rate-based figure; an alcohol claim checked to approve $0.00 regardless of amount and never additionally trigger a receipt-required flag; the receipt-required threshold checked exactly at its own real boundary; and a full 4-item report checked to compute exact claimed and approved totals while attributing every violation to its own specific line item.

@@OUT3@@

!!! warning "[COMMON TRAP] treating a policy-approved total as a corrected, replacement version of the claimed total"
    Once an expense engine has computed a real, policy-compliant APPROVED figure for a line item, it is tempting to treat that figure as simply the CORRECTED version of what the employee claimed, and report only it going forward -- after all, the engine has already done the real work of figuring out what should actually be reimbursed. Test 6 is built specifically to rule that shortcut out: the report's own claimed total of $527 and its own approved total of $482 are both reported, side by side, for the same 4 items, with the $45 gap attributed by name to two specific line items' own specific violations. Collapsing to just the $482 approved figure would discard exactly the information an expense auditor, or the employee themselves, needs to understand WHY their own reimbursement came in lower than what they submitted -- the same discipline this book has applied to every flagged discrepancy since Section 21.3's own claim-consistency check.

## 23.4 A Real-Time Luxury-Goods Authentication Engine for Consignment and Resale Counters

### Intuition

A consignment counter's real-time screen cannot replace a trained human authenticator, but it can check three real, computable structural properties against a brand's own stated specification before that human ever looks at the item: whether a serial number's own digit portion satisfies a real, independently famous check-digit algorithm, whether a measured weight falls within a real stated spec range, and whether a hardware finish is among a real stated valid list.

### The Concept, In Detail

`luhn_valid` is the real, well-known Luhn algorithm -- the same algorithm that validates credit-card numbers -- confirmed in Test 1 against its own famous exact reference value (`79927398713`) and a second independent real 16-digit reference number, with a single altered digit breaking the checksum in both cases. `validate_serial` checks this section's own stated brand serial format -- a 2-letter factory code plus an 11-digit Luhn-valid number -- structurally, and Test 2 confirms a wrong length, a lowercase factory code, and a non-digit character are each flagged by their own specific, named reason, distinct from Test 3's confirmation that a structurally valid serial can still fail specifically on its own checksum.

`authenticate_item`'s own Test 5 is this section's central honesty check, reapplying this book's own structural-override discipline one more time: an item with a perfectly in-spec weight and a perfectly recognized hardware finish is still `REJECTED_INVALID_SERIAL` outright the moment its own serial checksum fails -- two passing checks never outvote one hard-failing one. Test 7 confirms the same discipline in a softer form: an unrecognized hardware finish is `FLAGGED_FOR_EXPERT_REVIEW`, never silently passed, even when the serial and weight both look fine. And Test 4's own passing case is named, deliberately, `PASSES_STRUCTURAL_SCREENING` rather than "authentic" -- an honest, narrow claim whose own reason text states plainly that final authentication remains a human expert's own call.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_luxury_goods_authentication_engine.cpp -o 04_luxury_goods_authentication_engine
./04_luxury_goods_authentication_engine
```

**Sample input:** the real Luhn algorithm checked against its own famous reference number and a second independent reference number, each with a single altered digit breaking the checksum; this section's own stated serial format checked structurally against a wrong length, a lowercase factory code, and a non-digit numeric portion, each flagged by its own specific reason; a structurally valid serial with a failing checksum checked distinctly from a fully invalid one; a fully valid item checked to pass structural screening with an explicit human-expert caveat in its own reason text; an invalid checksum checked to force rejection even with a perfectly in-spec weight and finish; an out-of-range weight checked to be rejected with the exact real spec range named; and an unrecognized hardware finish checked to be flagged for expert review rather than silently passed.

@@OUT4@@

!!! warning "[COMMON TRAP] treating PASSES_STRUCTURAL_SCREENING as equivalent to a finished authentication"
    `PASSES_STRUCTURAL_SCREENING` is the best possible outcome `authenticate_item` can ever return, and it is tempting, at a busy consignment counter, to treat that outcome as the END of the authentication process rather than its real, useful FIRST step. This section's own file never returns a verdict named "AUTHENTIC" anywhere, and Test 4's own passing reason text says so explicitly: final authentication remains a human expert's own call. The three checks this engine actually runs -- a checksum algorithm, a weight range, and a finish name lookup -- are real and genuinely useful for catching an obviously wrong item quickly, but they say nothing about the countless other real properties (stitching, hardware plating thickness, material grain, a serial number that is itself a cloned, valid-looking fake) a trained human authenticator would also check. Treating a structural pass as a finished verdict would grant this engine an authority over a real, consequential resale decision that its own three checks were never built to support.

## Chapter Summary

This chapter built four real, independently well-known algorithms -- a perceptual difference hash, the classic Levenshtein edit distance, a real corporate expense-policy rules engine, and the Luhn check-digit algorithm -- into four domains united by trust and authenticity at the point of a real transaction. Section 23.1 built a counterfeit-screening engine and a fully auditable seller trust score, proving by direct construction that a perfect aggregate trust score can never suppress a single specific, named red flag. Section 23.2 built a receipt-to-transaction reconciliation engine that honestly reports AMBIGUOUS, naming every fitting transaction, rather than silently guessing when the underlying evidence genuinely supports more than one answer. Section 23.3 built an expense-policy engine that reports a claimed figure and a policy-approved figure side by side for every line item, never silently substituting one for the other without naming the exact rule behind the gap. Section 23.4 built a real-time luxury-goods screening engine whose own best possible outcome is honestly named a structural pass, not an authentication verdict, and proved that one hard-failing check is never outvoted by two others that happen to pass.

## Self-Check Questions

1. Section 23.1's Test 7 uses a seller with a PERFECT trust score of 100.0, not merely a high one. Explain why using the maximum possible score, rather than a merely above-average one, makes this test's own point more convincingly than a smaller number would.
2. Section 23.1's `screen_listing` compares a listing's own photo hash against a registry of OTHER sellers' photos, explicitly excluding the same seller's own prior listings. Explain why comparing against the SAME seller's own past listings would not be a useful counterfeit signal, even though it would still find hash matches.
3. Section 23.2's `merchant_names_fuzzy_match` is described as tuned for OCR noise specifically, not payment-processor descriptor mangling. Using the "SQ *STARBUCKS COFFEE" example from this section's own COMMON TRAP box, explain concretely why an edit-distance-ratio threshold tuned for a single misread character would fail on that different kind of mismatch.
4. Section 23.2's Test 6 constructs an AMBIGUOUS case using two transactions with the SAME claimed merchant name and a plausible tip-adjusted amount. Construct a different, equally realistic pair of transactions that would ALSO produce an AMBIGUOUS result under this section's own real matching rules.
5. Section 23.3's mileage check computes an EXPECTED reimbursement from miles times a real per-mile rate, rather than simply capping whatever amount was claimed the way the meals and lodging checks do. Explain why a cap alone would be insufficient for mileage specifically.
6. Section 23.3's alcohol exclusion never additionally triggers the receipt-required check. Explain why applying the receipt-required check on top of the alcohol exclusion would not provide any additional real information to a reviewer.
7. Section 23.3's Test 6 reports a $527 claimed total against a $482 approved total for the same 4-item report. What decision does this design deliberately leave to a human reviewer rather than making automatically, and how does reporting both totals support that decision better than reporting only the approved total would?
8. Section 23.4's `validate_serial` returns `valid_format` and `valid_checksum` as two SEPARATE boolean fields rather than a single combined `is_valid` boolean. What real diagnostic information would collapsing them into one field lose, and construct a serial number that demonstrates the difference.
9. Section 23.4's Test 5 rejects an item purely because of its own serial checksum, despite a perfectly in-spec weight and a perfectly recognized finish. Explain concretely why a real counterfeiter is more likely to get a serial checksum wrong than to get a hardware finish or weight wrong, and what that implies about which checks deserve override authority over the others.
10. Across Sections 23.1 and 23.4, a single named red flag or a single hard-failing check always overrides an otherwise-favorable aggregate signal. Identify the one place in Section 23.3 where the SAME override discipline appears in a financial-policy context rather than a fraud-detection one, and explain why it serves the identical real purpose there.

## Where We Go Next

This chapter's four sections showed the same recurring discipline -- an honest ambiguity, a named red flag that survives a favorable aggregate score, and a structural pass that never overstates its own real scope -- generalizing once more, this time into commerce and financial trust. Chapter 24 turns to a different real problem entirely: natural language photo editing, building an edit-interpretation prompt that turns a vague instruction like "make it look better" into a structured edit plan, a complete OpenCV processing engine that executes that plan as real `cv::Mat` operations, a request-to-operation mapping table, and iterative refinement through conversation.

## Worked Solutions

**1.** A merely above-average trust score could plausibly be explained away as "not quite perfect, so some risk was already priced in" -- a reviewer might reasonably wonder whether the review requirement was really about the specific keyword flag or just about the score not being high enough on its own. Using the literal maximum possible score of 100.0 eliminates that ambiguity entirely: there is no higher score this seller could have achieved, no further accumulated history that could have helped their case, and review is STILL required -- proving conclusively that the review trigger came from the specific named flag itself, not from any residual doubt the aggregate score left unaddressed.

**2.** A seller's own past listings for the identical product line would naturally, legitimately share the same or very similar product photography -- a seller who genuinely owns 5 identical items for sale would reasonably photograph them with the same setup, lighting, and even the same physical unit across several listings, none of which indicates counterfeiting. The real, useful signal this section's own check targets is a photo appearing under a DIFFERENT seller's own account -- since two independent, unrelated sellers each independently and legitimately owning and photographing the exact same item down to the pixel is essentially never how genuine product photography actually happens, while stolen stock imagery reused across accounts is a real, documented pattern.

**3.** A single misread character changes only one or two positions in an otherwise-identical string, keeping the edit-distance ratio very small (Section 23.2's own "WALGREEN5" example scores 1/9). "SQ *STARBUCKS COFFEE" against "Starbucks," by contrast, differs by an entirely prepended processor code ("SQ *"), a different capitalization convention, and an appended descriptor word ("COFFEE") not present in the receipt's own name at all -- these are real STRUCTURAL insertions spanning several characters each, not isolated single-character substitutions, which would push the edit distance (and therefore the ratio) well past any threshold tuned for a single misread digit or letter, causing a genuinely correct match to be wrongly rejected as a mismatch.

**4.** A $12.00 receipt for a merchant with two locations that settle their own card transactions under the identical merchant name and both process transactions dated the same real day (for instance, two branches of the same chain both settling on day 300 for the customer's own two separate real purchases that day) would produce the identical situation Test 6 constructs: two transactions, same merchant name, same real amount, both within the settlement window -- and `reconcile_receipt` would correctly report AMBIGUOUS rather than guessing which real branch's charge belongs to which real receipt.

**5.** A cap alone can only catch a claim that is TOO HIGH relative to some maximum -- it has no way to catch a claim that is simply WRONG in either direction relative to what the underlying miles actually compute to, including a claim that is too LOW (understating miles driven, whether by simple arithmetic error or by other real cause) which a cap would never flag at all since it falls under any reasonable maximum. Computing the expected reimbursement directly from the real inputs (miles times rate) and comparing the claim against that specific expected value, rather than against an unrelated upper bound, is what allows this section's own mileage check to catch a mismatch in either direction, not merely an excessive one.

**6.** Once an item is excluded from reimbursement entirely at $0.00 approved regardless of its own claimed amount, telling a reviewer that it ALSO lacks a required receipt would be reporting a compliance requirement for a category of spending that is never eligible for reimbursement under any circumstances in the first place -- there is nothing further a receipt could unlock or justify for an item that is already fully excluded by category, so flagging its own missing receipt on top of the exclusion would only add noise without adding any real, actionable information a reviewer could use.

**7.** This design deliberately leaves to a human reviewer the decision of what to actually communicate back to the employee and whether any of the flagged discrepancies warrant further discussion (a genuine miscalculation the employee should be informed of, a policy the employee may not have been aware of, or a pattern worth raising with them directly) rather than having the system silently reimburse a lower amount with no visible explanation. Reporting both the $527 claimed and $482 approved totals together, with each of the $45 gap's own contributing violations named by item, lets that reviewer see exactly what changed and why, rather than presenting a single final number that would leave the employee (or the reviewer themselves, weeks later) unable to reconstruct where the difference came from.

**8.** A single combined `is_valid` boolean would tell a reviewer only that SOMETHING about the serial is wrong, with no way to distinguish a serial that is structurally malformed (wrong length, wrong character classes -- likely a data-entry error, a different product line entirely, or a very unsophisticated fake) from a serial that is structurally perfect but fails its own checksum (a much more specific and concerning signal, since it means someone constructed a plausible-LOOKING serial that does not actually satisfy the brand's own real check-digit scheme). "FR79927398714" demonstrates this exactly: it has the correct length, the correct letter prefix, and an all-digit numeric portion (so `valid_format` is true), yet its own checksum fails (`valid_checksum` is false) -- a single combined boolean would report this identically to a serial that was simply the wrong length, discarding the real distinction between "malformed" and "well-formed but fraudulent."

**9.** A hardware finish name and a target weight range are both properties a counterfeiter can directly observe from a genuine item (by weighing it, or by reading the finish name off an authentic tag) and then simply replicate or match closely, since neither number is secret or derived from anything hidden. A valid Luhn checksum, by contrast, requires knowing the SPECIFIC mathematical relationship between a serial's own digits and its own final check digit -- a counterfeiter fabricating a plausible-looking serial number by guessing or copying a DIFFERENT genuine item's own format would need to either reuse an already-issued real serial (a separate, detectable problem) or correctly compute a valid checksum for a new one, which requires knowing the checksum algorithm itself, not just observing a physical property. This is exactly why a checksum failure deserves override authority over the other two checks: it is evidence of a structural relationship being violated, not merely a measurement falling outside a range that could plausibly result from ordinary manufacturing variance.

**10.** The identical override discipline appears in Section 23.3's alcohol-exclusion rule: no matter how reasonable, well-documented, or within-cap an alcohol line item's own claimed amount might otherwise look, `evaluate_line_item` approves it at exactly $0.00 every single time, with no code path anywhere in the function that lets a low claimed amount, a valid receipt, or any other favorable property override that exclusion. It serves the identical real purpose as Section 23.1's trust-score override and Section 23.4's checksum override: a single, specific, categorical policy fact (this category is never reimbursable; this specific keyword appeared; this specific checksum failed) is treated as authoritative and non-negotiable, regardless of how favorable every OTHER signal about the same claim, listing, or item happens to be.
