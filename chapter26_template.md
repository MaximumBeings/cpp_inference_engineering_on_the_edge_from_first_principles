# Chapter 26: Check and Invoice Fraud Detection at Bank Branches and Back Offices

**What you will understand by the end of this chapter:**

- How to implement the real American Bankers Association routing-number checksum -- the identical published formula every US bank's own check-clearing system runs on every check -- and reapply Chapter 23.1's own real perceptual difference-hash (dHash) algorithm to signature comparison instead of product photography.
- How to reapply the classic Levenshtein edit-distance algorithm, built from scratch in Chapter 23.2, to a different real fraud pattern: an invoice number altered by a single character specifically to dodge a naive exact-match duplicate-payment check.
- How to build a real, from-scratch English number-word parser that reconstructs the numeric value a check's own "amount in words" field actually claims, and cross-validates it against the numerals field -- the real redundancy a US check's own printed form exists to exploit against alteration.
- How to combine multiple independent real fraud signals into a single, fully auditable risk score and a three-way disposition, reapplying this book's own recurring discipline that one specific, high-severity red flag can force escalation on its own, regardless of how clean every other signal looks.

**What you need to know first:**

- Section 23.1's own real, from-scratch dHash implementation (resize to 9x8, encode 64 horizontal pixel comparisons into a 64-bit value, compare by Hamming distance) is reapplied unchanged in Section 26.1, this time to a signature image rather than a product photo.
- Section 23.2's own real, from-scratch Levenshtein edit-distance algorithm is reapplied unchanged in Section 26.2, this time to detect a near-identical invoice number rather than fuzzy-match an OCR-noisy merchant name.
- This book's own recurring structural-override discipline, most recently Chapter 23's own never-suppress-a-named-flag rule: one specific, high-severity signal (Section 26.4's own MICR checksum failure) forces at least a human review, regardless of how low every other signal's own score is.

---

Chapter 25 closed Part 5's run of consumer- and field-facing vision-language deployments. This chapter turns to a different real financial setting: a bank branch or back office processing checks and invoices, where the fraud patterns are well-documented, real, and -- unlike a counterfeit product photo -- often checkable by a real published formula or a real structural redundancy the payment instrument's own printed form was designed to provide. Each of this chapter's four sections reapplies an algorithm this book already built from scratch (dHash, Levenshtein) to a new real domain, or builds a new real algorithm (a routing-number checksum, a words-to-number parser) with its own well-established provenance outside this book, and closes with a capstone risk-scoring engine that combines all of them into one fully auditable disposition.

## 26.1 MICR Checksum Validation and Signature Comparison

### Intuition

A check's own MICR line is not just an identifier -- its routing number carries a real, published checksum, the same formula every US bank's own check-clearing system runs on every check that passes through it. A forged or badly transcribed routing number will very often fail this checksum outright, catching a structural problem before any human ever reads the check. Separately, a check's own payee signature can be compared against a signature on file using the identical real perceptual hash this book already built in Chapter 23.1 for a completely different purpose.

### The Concept, In Detail

`micr_checksum_valid` implements the real, published ABA routing-number formula -- `3*(d1+d4+d7) + 7*(d2+d5+d8) + 1*(d3+d6+d9)` must be a multiple of 10 -- confirmed in Test 1 against a real, publicly known routing number (JPMorgan Chase's own "021000021") and in Test 2 against the identical number with its own last digit altered, which correctly breaks the checksum. Test 3 confirms a malformed routing number (wrong length, or containing a non-digit) is rejected outright rather than silently truncated or padded.

`compute_dhash` and `hamming_distance` are Chapter 23.1's own unmodified implementations, applied here to a signature image. Test 4 confirms a pixel-identical re-scan hashes identically (Hamming distance 0); Test 5 confirms the same physical signature, re-scanned with small, realistic sensor noise, still matches under a real, stated tolerance; and Test 6 confirms a visibly different image -- standing in for a forged or substituted signature, the identical honest use of a starkly different synthetic test image Chapter 23.1's own dHash test used -- is correctly flagged as NOT matching, proving the noise tolerance does not silently absorb a genuinely different signature too.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_micr_checksum_and_signature_comparison.cpp -o 01_micr_checksum_and_signature_comparison
./01_micr_checksum_and_signature_comparison
```

**Sample input:** a real, publicly known routing number checked against the real ABA checksum formula, then checked again with a single digit altered; a malformed routing number (wrong length, and separately containing a non-digit) checked to be rejected outright; a reference signature image checked against a pixel-identical re-scan (Hamming distance 0), the same signature re-scanned with realistic sensor noise (matching under a stated tolerance), and a visibly different image standing in for a forged signature (correctly rejected).

@@OUT1@@

!!! warning "[COMMON TRAP] treating a checksum-valid routing number as proof the check is genuine"
    A checksum-valid routing number proves only that the 9 digits printed on the MICR line are INTERNALLY CONSISTENT with the ABA's own published formula -- it says nothing about whether that routing number belongs to a real, currently-operating bank (which would require a live lookup against the Federal Reserve's own routing-number registry, a real network dependency this section's own offline, deterministic discipline deliberately does not take on), and nothing at all about whether the account number, the signature, or the amount fields are genuine. A sophisticated forger who simply copies a real, valid routing number from a genuine check onto a fraudulent one would pass this specific check perfectly -- exactly why Section 26.4's own risk engine treats a checksum failure as one input among several, forcing escalation rather than either auto-clearing OR auto-rejecting on this signal alone.

## 26.2 Invoice Duplicate Detection

### Intuition

A well-documented real invoice-fraud pattern is not a forged invoice at all, but the SAME real invoice submitted twice for payment, with its own invoice number altered by a single character specifically to dodge a naive exact-match duplicate check. Catching this requires the identical real fuzzy-matching algorithm Chapter 23.2 already built from scratch, applied to a different real signal.

### The Concept, In Detail

`levenshtein_distance` is Chapter 23.2's own unmodified real, classic edit-distance algorithm, confirmed in Test 1 against its own famous exact reference value (`KITTEN` to `SITTING` is distance 3). `flag_duplicate` combines three independent real signals -- an exact vendor match, an exact amount match, and an invoice number that is NEAR but not identical to a previously paid invoice's own number -- and Test 2 confirms an invoice number one character off from a real, already-paid invoice is flagged, naming the exact matching invoice and the exact edit distance.

Tests 3 through 6 each isolate one of the three required signals: Test 3 confirms a genuinely different invoice number (far beyond the stated edit-distance threshold) is not flagged; Test 4 confirms an EXACT resubmission -- the case a plain exact-match dedup pass already catches on its own -- falls outside this detector's own near-match scope by design; Test 5 confirms the identical near-match pattern from a different vendor is not flagged; and Test 6 confirms the identical near-match pattern at a genuinely different amount is not flagged, proving all three signals are required together, not any one alone. Test 7 confirms the exact stated edit-distance boundary itself, not merely a value comfortably inside it.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_invoice_duplicate_detection.cpp -o 02_invoice_duplicate_detection
./02_invoice_duplicate_detection
```

**Sample input:** the real Levenshtein algorithm checked against its own famous KITTEN-to-SITTING reference value; an invoice number one character off from an already-paid invoice, same vendor and amount, checked to be flagged with the exact matching invoice number and edit distance named; a genuinely different invoice number, an exact resubmission, a different vendor, and a different amount each checked in turn to confirm all three required signals must hold together; and an invoice number checked exactly at the stated edit-distance boundary.

@@OUT2@@

!!! warning "[COMMON TRAP] treating this detector's own structural flag as proof of fraudulent intent"
    `flag_duplicate` identifies a real, checkable STRUCTURAL pattern -- a near-identical invoice number combined with an identical vendor and amount -- that correlates strongly with real double-billing fraud. It does not, and cannot, prove intent. A vendor that genuinely issues two separate, legitimate invoices for the same recurring service charge in the same week, using an invoice-numbering scheme that happens to increment by one, would also match this exact pattern, and a flagged invoice should still go to a human reviewer for a real decision rather than being auto-rejected on this signal alone -- exactly the honest structural-pass discipline Chapter 23.4's own luxury-goods screening engine applied to its own best possible outcome.

## 26.3 Amount-in-Words Cross-Validation

### Intuition

A US check states its own amount twice -- once in numerals, once spelled out in words -- specifically so that altering the numerals alone (the classic "check washing" fraud, chemically removing and rewriting a printed amount) does not silently succeed, provided something actually cross-checks the two fields against each other. This section builds that cross-check from scratch: a real English number-word parser reconstructing the exact cent value the words field claims.

### The Concept, In Detail

`parse_amount_words` tokenizes a check-convention words phrase (splitting on hyphens as well as spaces, so "thirty-four" parses as two number words) and reconstructs its own dollar value using a real running-group accumulator: each unit or tens word adds into the current group, "hundred" multiplies the current group by 100, and "thousand" flushes the current group into the running total and resets it -- Test 7 confirms this grouping logic correctly handles a nonzero hundreds group nested inside a thousands group ("Forty-two thousand three hundred five"). An "and NN/100" cents suffix, when present, is parsed separately and added as exact integer cents, avoiding the floating-point drift this book's own quantization work in Chapter 4 already showed is unsafe for a value that must compare exactly.

`cross_check` compares the words-derived cent value against the numerals field, converted to cents via the identical round-to-nearest-cent discipline. Test 2 reproduces the classic check-washing scenario directly: the numerals altered while the words field is left untouched, caught exactly; Test 3 confirms the reverse alteration is caught by the identical logic. Test 5 confirms an honestly unparseable words field (containing a token this narrow, stated parser does not recognize) is reported as an explicit parse failure -- never silently treated as either a match or a silent zero.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_amount_in_words_cross_validation.cpp -o 03_amount_in_words_cross_validation
./03_amount_in_words_cross_validation
```

**Sample input:** a correctly matching check amount checked in both numerals and words; the classic check-washing pattern reproduced in both directions (numerals altered with words untouched, and words altered with numerals untouched), each caught exactly; a whole-dollar amount with no stated cents; a genuinely unparseable words phrase checked to report an honest parse failure; a plain amount under one hundred dollars with no hundreds or thousands grouping; and a larger multi-thousand amount with a nonzero hundreds group nested inside it.

@@OUT3@@

!!! warning "[COMMON TRAP] assuming this section's own narrow parser handles any English number phrase"
    `parse_amount_words` implements one specific, real, stated convention: a US check's own whole-dollar amount in words, followed by an "and NN/100" cents suffix. It does not attempt to parse arbitrary English number phrasing -- a phrase using "and" in a different position, a non-US convention for grouping large numbers, or a currency whose own printed-check convention differs -- and Test 5 exists specifically to prove an unrecognized token produces an honest, explicit parse failure rather than a silently wrong guess. Deploying this exact parser against checks drawn on a different real convention without first confirming its own vocabulary covers that convention's own words would risk exactly the silent-wrong-answer failure this book's own honesty discipline exists to prevent -- the identical narrow-scope caveat Chapter 12's own tokenizer stated for its own real, fixed vocabulary.

## 26.4 Risk Scoring and Escalation

### Intuition

A single fraud signal rarely justifies an automatic rejection on its own, and a clean aggregate score should never be allowed to silently absorb one specific, high-severity red flag. This section's capstone combines Section 26.1's MICR and signature signals, Section 26.2's duplicate-invoice signal, and Section 26.3's amount-mismatch signal into one real, fully auditable weighted score and a three-way disposition, reapplying this book's own recurring never-suppress-a-named-flag discipline one more time.

### The Concept, In Detail

`assess_risk` sums the point weight of every signal that fired into a single total score, and separately records the FULL, NAMED list of which signals fired -- Test 1 confirms a clean transaction produces an honestly empty audit trail, not a suppressed one. Test 2 and Test 3 confirm the two real score thresholds route a single moderate flag to `ESCALATE_TO_REVIEW` and two flags together to `AUTO_REJECT`, with the audit trail naming every contributing signal. Test 6 and Test 7 confirm both threshold boundaries exactly, not merely values comfortably on either side.

Test 4 is this section's own central honesty check: a failed MICR checksum, with every OTHER signal clean and a total score of exactly 0, still forces at least `ESCALATE_TO_REVIEW` -- the identical never-suppress-a-named-flag discipline Chapter 23.4's own hard-failing checksum check applied to two otherwise-passing signals. Test 5 confirms the forcing rule is directional: combined with signals that already independently cross the reject threshold, the MICR failure does not (and must not) downgrade an `AUTO_REJECT` back down to a milder `ESCALATE_TO_REVIEW`.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_risk_scoring_and_escalation.cpp -o 04_risk_scoring_and_escalation
./04_risk_scoring_and_escalation
```

**Sample input:** a clean transaction with every signal from every prior section passing, checked to auto-clear with an empty audit trail; a single moderate-severity flag checked to escalate; two flags together checked to cross the reject threshold, naming both; a failed MICR checksum alone, with every other signal clean, checked to still force escalation; the same MICR failure combined with signals that already independently reject, checked to leave the rejection undowngraded; and both real score thresholds checked exactly at their own stated boundaries.

@@OUT4@@

!!! warning "[COMMON TRAP] treating AUTO_CLEAR as proof of a legitimate transaction"
    `assess_risk` produces a real, deterministic, fully-explained disposition from a fixed, stated set of signals and weights -- it is a real screening step, not a fraud verdict. `AUTO_CLEAR` means, precisely and only, that no signal within THIS engine's own stated scope fired for this specific transaction; it is not, and was never built to be, an affirmative statement that the transaction is legitimate. A fraud pattern this chapter's own four sections do not check for at all -- a stolen but structurally perfect blank check stock, for instance -- would sail through every signal in this file and land at `AUTO_CLEAR`, exactly as a well-forged item with a perfectly valid Luhn-checksummed serial number would have passed Chapter 23.4's own structural screening. An aggregate disposition of AUTO_CLEAR is only ever as strong as the specific, named list of signals that fed it -- never a substitute for knowing exactly what that list does and does not cover.

## Chapter Summary

This chapter reapplied two of this book's own already-built real algorithms -- Chapter 23.1's perceptual difference hash and Chapter 23.2's classic Levenshtein edit distance -- to a new financial domain, and added two new real algorithms of its own: the published ABA routing-number checksum and a from-scratch amount-in-words parser. Section 26.1 validated a check's own MICR routing number against its real published checksum and compared a signature image using the identical dHash algorithm Chapter 23.1 built for product photography. Section 26.2 reapplied the classic Levenshtein algorithm to catch an invoice resubmitted with a single altered character, specifically excluding the exact-resubmission case a plain dedup pass already catches. Section 26.3 built a real English number-word parser and used it to reproduce and catch the classic check-washing fraud pattern in both directions. Section 26.4 closed the chapter by combining all of the above into one fully auditable risk score, proving once more that a single high-severity signal can force escalation regardless of how clean the aggregate score otherwise looks, and that the forcing rule never downgrades an already-stricter disposition.

## Self-Check Questions

1. Section 26.1's `micr_checksum_valid` and Chapter 23.4's `luhn_valid` are both real, published check-digit algorithms applied to different real domains. Explain one concrete structural similarity between how the two algorithms combine a number's own digits into a single pass/fail result.
2. Section 26.1's own COMMON TRAP box explains that a checksum-valid routing number does not prove a check is genuine. Name one specific real fraud scenario in which a check would pass Section 26.1's own MICR checksum check perfectly while still being fraudulent.
3. Section 26.2's `flag_duplicate` explicitly excludes an EXACT invoice-number resubmission from its own near-match logic. Explain why including the exact-match case inside this same function, rather than relying on a separate plain dedup pass, would not actually change this detector's own real coverage of the fraud pattern it targets.
4. Section 26.2's Test 6 constructs a case where the same near-match invoice-number pattern occurs at a genuinely different amount and is correctly not flagged. Explain concretely why requiring an exact amount match, rather than an amount within some tolerance, is the correct design choice for THIS specific fraud pattern.
5. Section 26.3's `parse_dollar_words` resets its own running `current_group` to 0 immediately after each "thousand" token. Using the phrase "Forty-two thousand three hundred five" from Section 26.3's own Test 7, trace through what the function's own final total would incorrectly become if this reset were removed.
6. Section 26.3's own COMMON TRAP box warns against assuming the parser handles any English number phrase. Construct one specific, realistic check-amount phrase that a real US check might contain, using standard English number words, that this section's own stated parser would fail to parse correctly, and explain exactly which token or structure it does not handle.
7. Section 26.4's `assess_risk` records a fired signal's own NAME in the audit trail, not merely its point weight. Explain what specific real information a human reviewer would lose if the audit trail recorded only the total numeric score and the count of fired signals, without their names.
8. Section 26.4's Test 5 confirms the MICR-forcing rule does not downgrade an already-stricter `AUTO_REJECT` disposition. Explain why allowing the forcing rule to work in BOTH directions (also capable of downgrading a stricter disposition to a milder one) would create a real, exploitable weakness in this risk engine.
9. Section 26.1's signature-comparison Hamming-distance tolerance and Section 26.2's invoice-number edit-distance tolerance both define a real "close enough" boundary using a different real metric (Hamming distance on a fixed 64-bit hash vs. Levenshtein distance on a variable-length string). Explain why a single shared numeric threshold value would not make sense to reuse across both checks.
10. Across all four sections of this chapter, identify the ONE section whose own central test is built specifically to catch a fraud pattern that involves ALTERING a value already on file, as opposed to introducing an entirely new fraudulent item, invoice, or transaction. Explain what specifically about that fraud pattern makes it detectable by a cross-check between two fields, in a way the other three sections' own fraud patterns are not.

## Where We Go Next

This chapter showed that reapplying algorithms this book already built from scratch -- a perceptual hash, an edit-distance algorithm -- to a new financial domain can be exactly as effective as building something new, provided each reapplication is grounded in a real, well-understood fraud pattern rather than borrowed for its own sake. Chapter 27 turns to a related but distinct real financial setting: low-latency inference on live market-signal data near the exchange, where the constraint shifts from fraud-pattern detection to a hard real-time deadline measured in microseconds.

## Worked Solutions

**1.** Both algorithms combine a number's own digits using a fixed, published set of per-position WEIGHTS, sum the weighted digits together, and check the result against a fixed real modulus (the ABA formula checks divisibility by 10 directly on the weighted sum; the Luhn algorithm doubles alternating digits, sums the results after reducing any two-digit product to a single digit, and likewise checks the total's own divisibility by 10). In both cases, a single altered digit changes the weighted sum by a nonzero amount in the vast majority of cases, which is exactly why both algorithms reliably catch a single-digit transcription or forgery error without needing to compare against any external reference value at all -- the entire check is self-contained within the number's own digits.

**2.** A forger who obtains a genuine, valid routing number -- for instance, by photographing or purchasing a real blank check, or simply looking up any real bank's own publicly listed routing number -- and prints that identical, genuinely valid number onto a fraudulent check (with a fabricated account number, a forged signature, and an altered amount) would pass Section 26.1's own MICR checksum check perfectly, since the routing number itself is completely real and internally consistent; the fraud lies entirely in the OTHER fields the checksum check was never built to examine.

**3.** A plain, separate dedup pass already catches every EXACT invoice-number resubmission on its own, using a simple exact-string comparison that costs far less to compute than a full Levenschtein distance calculation across an entire paid-invoice history. Including the exact-match case inside `flag_duplicate` as well would not extend the fraud pattern actually caught -- the exact-match case was never THIS detector's own gap in the first place -- it would only duplicate work an existing, cheaper check already does completely, which is exactly why the section explicitly `continue`s past an exact match rather than flagging it redundantly.

**4.** This specific fraud pattern -- the same underlying invoice resubmitted with a cosmetically altered invoice number -- depends on the amount staying IDENTICAL, since the whole point of the fraud is billing for the identical real goods or services a second time; a fraudster attempting this pattern has no real reason to also alter the amount, and doing so would actually reduce the fraud's own plausibility to a reviewer expecting round-number consistency. Allowing an amount tolerance would instead risk flagging two entirely unrelated, legitimate invoices from the same vendor that happen to have similar invoice numbers and roughly similar (but not identical) amounts purely by coincidence -- a real false-positive risk an exact-match requirement avoids entirely.

**5.** Without the reset, `current_group` would still hold `42` (from "forty-two") at the moment "thousand" is processed, so the `total += current_group * 1000` step would still correctly add `42000` to `total` -- but critically, `current_group` would NOT be reset to 0 afterward, so it would still hold `42` going into "three hundred five." The subsequent "hundred" token would then multiply the STALE `42` by 100 to get `4200`, and adding "five" would produce a final `current_group` of `4205` instead of the intended `305`, making the function's own final total `1000*42 + 4205 = 46205` instead of the correct `42305` -- silently overcounting by exactly the un-reset group's own leftover value.

**6.** A phrase such as "One thousand and one dollars" -- a construction some real speakers and even some real printed forms use, placing "and" directly between a thousands group and a following units word rather than reserving "and" exclusively for the cents suffix -- would parse incorrectly under this section's own stated parser: `parse_dollar_words` treats every occurrence of the token "and" identically, by skipping it unconditionally regardless of where it appears, so "one thousand and one" would correctly produce `1001`. But a phrase using British-style grouping, such as "one thousand two hundred AND thirty-four" where "and" appears between the hundreds and tens groups (a genuinely common British convention this section's own US-check-convention parser was never built to expect), would still parse correctly too, since "and" is simply skipped everywhere -- the actual failure case is a phrase using a word this parser's own `UNITS` and `TENS` tables do not contain at all, such as "a hundred" (using the indefinite article "a" instead of the number word "one"), which `parse_dollar_words` would reject outright as an unrecognized token, exactly as Test 5's own "gazillion" case demonstrates.

**7.** A reviewer given only a total score of, say, 70 and a count of "2 signals fired" would know a rejection-level score was reached but would have no way to know WHICH real checks actually triggered it without re-running every check by hand against the same transaction -- precisely the information needed to decide whether to investigate a suspected duplicate invoice, chase down a signature discrepancy, or follow up on a specific amount mismatch. Recording each fired signal's own name preserves exactly the actionable detail a reviewer needs to act on the flag efficiently, rather than starting their own investigation from zero.

**8.** A bidirectional forcing rule would create a real path for a genuinely high-risk transaction (one that already independently crosses the reject threshold on its own real signals) to have its own disposition SOFTENED merely because ONE additional, unrelated check happened to pass cleanly -- for instance, a transaction with two serious fraud flags that would otherwise be auto-rejected could have its own MICR checksum happen to be perfectly valid, and if that success were allowed to downgrade the disposition, the two serious, independently-detected flags would be effectively cancelled out by one unrelated passing check. Restricting the forcing rule to only ever escalate (never downgrade) closes off that exploitable path entirely: a passing check can never make an otherwise-risky transaction look safer than its own worst individual signal already indicated.

**9.** A fixed 64-bit dHash has a fixed maximum possible Hamming distance of 64 regardless of the signature's own real-world size or complexity, so a tolerance value like 10 has a fixed, well-understood meaning as a FRACTION of that fixed 64-bit space. Levenshtein distance on an invoice number, by contrast, has no fixed maximum -- its own natural scale depends entirely on the invoice number's own length (a 5-character invoice number and a 20-character one need genuinely different absolute edit-distance tolerances to represent the same real "one or two characters changed" intent), which is exactly why Section 23.2's own merchant-name fuzzy match used an edit-distance RATIO relative to the string's own length rather than a fixed absolute value, while Section 26.2's own invoice-number check uses a small fixed absolute threshold specifically because real invoice numbers in this section's own stated scope vary little enough in length for a fixed small threshold to remain meaningful.

**10.** Section 26.3's amount-in-words cross-validation is built specifically to catch a value ALREADY PRINTED ON THE CHECK being altered after the fact (the classic check-washing pattern), detected by cross-checking two fields that should already agree on the same document. Section 26.1's MICR and signature checks validate properties of the document as originally presented, Section 26.2's duplicate detection catches an entirely new, separately-submitted fraudulent invoice rather than an alteration to an existing one, and Section 26.4's risk scoring combines signals rather than detecting a pattern of its own -- only Section 26.3's own redundant-field design gives it a structural way to catch a single field being changed in isolation, since the OTHER, unaltered field on the same document remains available as an independent real check against exactly that alteration.
