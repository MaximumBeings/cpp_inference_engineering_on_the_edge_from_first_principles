# Chapter 27: Low-Latency Market-Signal Inference Near the Exchange

**What you will understand by the end of this chapter:**

- How to compute a real order book's mid-price, size-weighted "microprice," and order-book imbalance from a Level 1 top-of-book snapshot -- the same real, published market-microstructure signals a real market-making or execution system computes on every book update.
- How to derive a hard, real-time tick-to-trade latency budget entirely from stated architectural parameters -- never a wall clock -- reapplying Chapter 10's own "verified without a clock" discipline to a genuinely different constraint: a deadline, rather than a throughput target.
- How to build a real, finance-specific sentiment lexicon modeled on the real, published Loughran-McDonald methodology, and why a generic English sentiment dictionary badly misclassifies routine financial vocabulary.
- How to fuse multiple independent real signals into one auditable trading decision, reapplying Chapter 26.4's own never-suppress-a-named-flag structure -- but inverted, so a missed deadline forces the SAFEST disposition rather than the strictest one.

**What you need to know first:**

- Chapter 10.1's own discipline of deriving timing behavior entirely from stated, labeled parameters rather than a wall clock is reapplied in Section 27.2, this time to a hard deadline rather than a throughput ceiling.
- Chapter 26.4's own weighted-signal-plus-forced-override structure (this book's recurring never-suppress-a-flag discipline) is reapplied in Section 27.4 -- this time forcing HOLD, the safe disposition, rather than forcing escalation, whenever a signal missed its own stated deadline.
- Section 27.1's own order-book formulas are real, published market-microstructure quantities, not something invented for this book; Section 27.3's own sentiment lexicon is modeled on a real, cited academic methodology (Loughran and McDonald, 2011), not the full real published word list.

---

Chapter 26 turned to a bank branch or back office processing checks and invoices, where a fraud pattern -- once caught -- has no real time pressure attached to catching it a few seconds later. This chapter turns to a related but genuinely different real financial setting: inference on live market-signal data near the exchange, where the central constraint is not pattern detection at all, but a hard, real-time deadline measured in microseconds. Each of this chapter's four sections builds one real, independent piece of that pipeline -- order-book features, a deadline enforcement mechanism, a finance-specific sentiment score -- and closes with a capstone that fuses all three into one auditable trading decision, where a signal that arrived too late to act on safely is treated as no signal at all.

## 27.1 Order Book Microstructure Features

### Intuition

A real electronic exchange's own top-of-book -- the single best bid, best ask, and their own displayed sizes -- already carries real, computable signal before any history or model is involved. The mid-price is the obvious first cut, but the "microprice" is a more informative real quantity: it weights each side's price by the OPPOSITE side's own displayed size, so that heavy buying pressure (a large bid size resting against a thin ask) pulls the microprice up toward the ask, reflecting the real expectation that the thin side is more likely to be consumed next. Order-book imbalance captures the same asymmetry as a single bounded number.

### The Concept, In Detail

`compute_book_features` first checks for two real, honest failure states before computing anything: a CROSSED book (best bid at or above best ask, a real if rare market-data anomaly) and NO_LIQUIDITY (zero displayed size on both sides at once, which would otherwise divide by zero). Tests 4 and 5 confirm both are reported as an explicit `BookState`, never a silently wrong number. When the book is valid, `imbalance` is `(bid_size - ask_size) / (bid_size + ask_size)`, and `microprice` is `(ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)` -- Tests 2 and 3 confirm the real economic direction concretely: heavy bid-side pressure pulls the microprice strictly above the mid-price, toward the ask, and heavy ask-side pressure pulls it strictly below, toward the bid.

Test 6 confirms a real, legitimate one-sided edge case -- zero displayed size on exactly one side, not both -- correctly saturates imbalance at exactly +1.0 and reduces the microprice to exactly that side's own opposite price, without any special-casing beyond the single zero-liquidity guard already in place. Test 7 confirms the same formulas generalize cleanly to a general asymmetric book at entirely different price and size levels.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_order_book_microstructure_features.cpp -o 01_order_book_microstructure_features
./01_order_book_microstructure_features
```

**Sample input:** a perfectly balanced top-of-book checked to reduce the microprice exactly to the mid-price; a heavy bid-side imbalance and a heavy ask-side imbalance each checked to pull the microprice strictly toward the thin opposite side; a crossed book and a zero-liquidity-on-both-sides book each checked to report an explicit failure state rather than a silently wrong number; a one-sided zero-liquidity book checked as a real, legitimate edge case; and a general asymmetric book at different price and size levels entirely.

@@OUT1@@

!!! warning "[COMMON TRAP] treating a top-of-book signal as the whole order book"
    Every function in this section operates on TOP-OF-BOOK ONLY -- the single best bid and best ask level. A real limit order book has many price levels beneath the top, and a large real order can be deliberately split into small "iceberg" slices so that only a fraction of its own true size is ever displayed at the top level at once -- exactly the kind of hidden liquidity this section's own `imbalance` and `microprice` formulas have no way to see, since they only ever read the displayed size at the best level. A market participant relying on this section's own signals alone, without also tracking how the top-of-book levels are refreshed over time, would be blind to a large resting order deliberately kept off the visible top of the book -- a real, well-documented limitation of any level-1-only signal, not a bug in this section's own arithmetic.

## 27.2 The Latency Budget: Tick-to-Trade Deadline Enforcement

### Intuition

A hard real-time deadline changes what "correct" means: a trading decision computed correctly but too late is not a weaker answer, it is an answer that may already be wrong, since the book it was computed from has likely moved on. Exactly like Chapter 10.1's own derivation of a single decode step's latency, every microsecond value in this section is a stated, labeled parameter -- never a value read from a wall clock -- so this section's own locked self-test output stays identical no matter how fast or slow the specific machine running it happens to be.

### The Concept, In Detail

`run_pipeline` checks a market event's own age against a stated staleness limit BEFORE attempting any processing at all -- Test 4 confirms an event that arrived too long ago is rejected outright, with zero stages run, rather than wasting real processing time computing a decision from a book state already known to be stale. Otherwise, each named `PipelineStage`'s own stated cost accumulates in order, and the moment the running total exceeds the stated budget, the pipeline stops immediately at that exact stage -- Test 3 confirms the stage that caused the breach is named precisely, with its own cost still counted into the elapsed total (the time was genuinely spent before the breach was noticed) but NOT added to the list of stages that actually completed within budget.

Tests 2 and 5 confirm both of this section's own boundary conventions exactly: a total cost landing exactly ON the stated budget, and an event age landing exactly ON the staleness limit, both count as within bounds, using a consistent strict-inequality rule throughout. Test 7 confirms the breach is always reported at the FIRST stage where the running total crosses the budget, not merely the last, by deliberately front-loading an expensive first stage that alone already exceeds a small budget.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 02_tick_to_trade_deadline_enforcement.cpp -o 02_tick_to_trade_deadline_enforcement
./02_tick_to_trade_deadline_enforcement
```

**Sample input:** four named pipeline stages checked to complete comfortably within a generous budget, and again checked at the exact boundary of the total stage cost; the same stages checked one microsecond short of that total, confirming the exact stage that breaches the budget is named and its own cost is still counted; a market event checked to be rejected outright for being too old before the pipeline even starts, and again checked at the exact boundary of the staleness limit; a zero-microsecond budget checked to breach immediately at the very first stage; and a deliberately front-loaded set of stages checked to report the breach at the first stage that causes it, not a later one.

@@OUT2@@

!!! warning "[COMMON TRAP] treating this section's own stated stage costs as a real system's actual measured latency"
    Every microsecond value in this file -- each `PipelineStage`'s own cost, every event's own arrival timestamp -- is a STATED, illustrative architectural parameter, exactly like Chapter 8.1's own illustrative CPU peak-bandwidth figures, not a measurement of any specific real exchange's own real infrastructure. A real production system's actual tick-to-trade latency depends on real network jitter, real operating-system scheduling, real cache and TLB behavior, and real garbage-collection or memory-allocation pauses in whatever language it is written in -- none of which this section's own pure, deterministic arithmetic can predict. Treating this section's own stated-parameter pipeline as a substitute for actually instrumenting and measuring a real deployed system's own real latency would repeat the identical mistake Chapter 8's own roofline model warned against: an idealized derivation is a real, useful ceiling to reason about, never a stand-in for a real measurement.

## 27.3 Financial News Sentiment Scoring

### Intuition

A real, well-documented finding in financial text analysis is that a generic English sentiment dictionary badly misclassifies financial text: ordinary business vocabulary such as "tax," "liability," and "cost" describes routine, expected line items, not bad news, yet a generic lexicon flags all three as negative. The real, published fix -- Loughran and McDonald's own finance-specific sentiment methodology -- deliberately excludes this routine vocabulary from its own negative word list. This section builds a small, explicitly illustrative lexicon modeled on that real methodology, and demonstrates the exact contrast it exists to fix.

### The Concept, In Detail

`score_sentiment` counts finance-lexicon matches into `positive_hits` and `negative_hits`, with a real, standard negation rule from lexicon-based sentiment analysis: a negation word (such as "not") flips the polarity of the single token immediately following it -- Test 4 confirms this concretely, scoring "not profitable" as negative and the identical word without the negation as positive. Test 5 is this section's own central demonstration: a headline containing only "tax" and "cost" scores an honest NEUTRAL under this section's own finance-specific lexicon, which deliberately excludes both words from its negative list, but scores NEGATIVE under a stated GENERIC comparison lexicon that (like a real general-purpose English sentiment dictionary) does treat them as negative -- the real Loughran-McDonald finding, reproduced directly rather than merely cited.

Test 6 confirms `tokenize_headline`'s own case-insensitivity and punctuation-stripping, and Test 7 confirms a headline with no lexicon matches at all reports an honest zero score and an explicit NEUTRAL label, never a crash or a fabricated nonzero result.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_financial_news_sentiment_scoring.cpp -o 03_financial_news_sentiment_scoring
./03_financial_news_sentiment_scoring
```

**Sample input:** a clearly positive headline and a clearly negative headline, each checked against the exact expected hit counts and label; an evenly mixed headline checked to net to an honest NEUTRAL; a negated positive word checked against the identical word without negation, to isolate the negation logic's own effect; a headline containing only finance-neutral vocabulary ("tax," "cost") checked against this section's own finance-specific lexicon and a stated generic comparison lexicon, reproducing the real Loughran-McDonald finding directly; a headline with mixed case and trailing punctuation checked for correct tokenization; and a headline containing no lexicon words at all checked to report an honest zero.

@@OUT3@@

!!! warning "[COMMON TRAP] treating this section's own small illustrative lexicon as the real, complete Loughran-McDonald word list"
    `POS_LEXICON` and `NEG_LEXICON` in this file are small, illustrative lists built to demonstrate the real methodology, not the actual published Loughran-McDonald dictionaries, which span several thousand words across seven real categories (negative, positive, uncertainty, litigious, strong modal, weak modal, and constraining language) developed and validated against real 10-K filings. Deploying this section's own narrow illustrative vocabulary against real production news flow would miss the large majority of real finance-relevant sentiment words entirely and silently score most real headlines as a hollow NEUTRAL -- the identical narrow-closed-vocabulary caution Chapter 12's own tokenizer, Chapter 24's own closed intent vocabulary, and Chapter 26.3's own amount-in-words parser each stated for their own real, fixed, and deliberately limited scope.

## 27.4 Signal Fusion and the Trading Decision Engine

### Intuition

A trading decision is only as trustworthy as its own audit trail, and a signal that arrived too late to act on safely is not a weaker signal -- it is not a usable signal at all. This section's capstone fuses Section 27.1's order-book imbalance and price pressure with Section 27.3's sentiment score into one weighted score and a three-way action, reapplying Chapter 26.4's own never-suppress-a-named-flag structure -- but inverted for this real-time domain, so Section 27.2's own deadline signal forces the SAFEST disposition rather than the strictest one.

### The Concept, In Detail

`fuse_signal` checks the upstream book's own validity first -- Test 5 confirms an invalid book state (a crossed or no-liquidity book) forces HOLD outright, before any weighted score is even computed, regardless of how strong sentiment looks on its own. When the book is valid, the weighted score is always computed and recorded in the audit trail, but Test 4 is this section's own central check: the IDENTICAL bullish inputs that produced a clear BUY in Test 1 are forced to HOLD instead the moment `within_deadline` is false, with the would-have-been score of 20 still named in the audit trail rather than silently discarded -- a missed deadline overrides the trading action, but never erases the record of what the signal actually was.

Test 3 confirms the distinction this section's own COMMON TRAP box returns to: a genuinely weak, mixed combination of signals also lands on HOLD, but with `forced_hold_stale` explicitly false, since this HOLD was computed honestly from balanced signals, not forced by a missed deadline. Tests 6 and 7 confirm both real score thresholds exactly at their own stated boundaries, and Test 8 confirms a zeroed-out weight genuinely removes a signal's own contribution completely, even against an extreme underlying value.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_signal_fusion_and_trading_decision_engine.cpp -o 04_signal_fusion_and_trading_decision_engine
./04_signal_fusion_and_trading_decision_engine
```

**Sample input:** a clearly bullish combination of signals checked to fuse to BUY, and its exact bearish mirror image checked to fuse to SELL; a genuinely weak, mixed combination checked to land on a COMPUTED HOLD; the identical bullish inputs from the first test checked again with a missed deadline, forcing HOLD despite an unchanged, strongly bullish underlying score; an invalid upstream book state checked to force HOLD outright regardless of sentiment; both real score thresholds checked exactly at their own stated boundaries; and a zeroed-out sentiment weight checked to remove sentiment's own contribution completely even against an extreme value.

@@OUT4@@

!!! warning "[COMMON TRAP] conflating a FORCED HOLD with a COMPUTED HOLD"
    A forced HOLD (`forced_hold_stale == true`) and a computed HOLD (a genuinely weak or balanced signal, with `forced_hold_stale == false`) both result in the identical `TradeAction::HOLD`, but they mean completely different things operationally. A computed HOLD is an honest statement that no real trading edge was detected in this specific signal. A forced HOLD carries no such information at all -- the underlying signal may well have been a strong, real BUY or SELL that simply arrived too late to act on safely, and Test 4 exists specifically to show the would-have-been score is still recorded rather than discarded. Logging both cases identically, without preserving the `forced_hold_stale` distinction, would hide a real, recurring latency problem -- a system missing its own deadline often enough to matter -- behind what looks like a healthy stream of ordinary, low-conviction HOLD decisions, exactly the same distinct-audit-trail discipline Chapter 26.4's own `AUTO_CLEAR` trap already established for a clean score covering only its own stated scope.

## Chapter Summary

This chapter turned from Chapter 26's fraud-pattern detection to a genuinely different real financial constraint: a hard, real-time deadline. Section 27.1 built real, published market-microstructure formulas -- mid-price, microprice, and order-book imbalance -- from a top-of-book snapshot, with explicit, honest failure states for a crossed or zero-liquidity book. Section 27.2 reapplied Chapter 10's own "verified without a clock" discipline to a hard tick-to-trade deadline, rejecting stale events outright and naming the exact pipeline stage that breaches a stated budget. Section 27.3 built a small, illustrative finance-specific sentiment lexicon modeled on the real, published Loughran-McDonald methodology, and reproduced its central finding directly: routine business vocabulary that a generic lexicon misreads as negative. Section 27.4 closed the chapter by fusing all three signals into one auditable trading decision, reapplying Chapter 26.4's own never-suppress-a-flag discipline in its own inverted form -- a missed deadline forces the safest disposition, never the riskiest one, while always preserving an honest record of what the underlying signal actually was.

## Self-Check Questions

1. Section 27.1's `compute_book_features` checks for a CROSSED book and a NO_LIQUIDITY book as two separate, explicit failure states rather than a single generic "invalid" flag. Explain one concrete reason a human reviewing this system's own output would benefit from knowing WHICH of the two failure states actually occurred.
2. Section 27.1's Test 6 constructs a book with zero displayed size on the ask side only, not both sides. Using the section's own microprice formula, explain algebraically why the microprice reduces to EXACTLY the ask price in this specific case, not merely approximately.
3. Section 27.2's own introduction explains that every timing value in the file is a stated parameter, never a value read from a wall clock. Explain concretely what would go wrong with this book's own cross-architecture verification process if `run_pipeline` instead measured its own stage costs using `std::chrono::high_resolution_clock`.
4. Section 27.2's Test 7 deliberately front-loads an expensive first stage rather than placing the expensive stage last. Explain what specific property of `run_pipeline`'s own breach-detection logic this test would fail to distinguish if the expensive stage were placed last instead, as in Test 3.
5. Section 27.3's Test 5 reproduces the real Loughran-McDonald finding using two different lexicons on the IDENTICAL headline. Explain why comparing the SAME headline under two lexicons is a stronger demonstration than simply asserting that "tax" and "cost" are absent from `NEG_LEXICON`.
6. Section 27.3's own negation rule flips the polarity of only the single token immediately following a negation word. Construct one realistic financial headline where this narrow scope would cause `score_sentiment` to miss an intended negation, and explain exactly why.
7. Section 27.4's `fuse_signal` checks `book_valid` before checking `within_deadline`. Explain what would happen differently if these two checks were reordered, using Section 27.1's own CROSSED-book failure state combined with a stale event as the concrete scenario.
8. Section 27.4's Test 4 reuses the IDENTICAL numeric inputs from Test 1, changing only `within_deadline`. Explain why holding every other input fixed makes this test a stronger check of the deadline-forcing logic than constructing a fresh, differently-valued example would be.
9. Section 27.4's own COMMON TRAP box warns against conflating a forced HOLD with a computed HOLD. Describe one concrete, real operational decision a trading desk might make differently if it could see the `forced_hold_stale` flag, compared to a system that only ever logged the final `TradeAction`.
10. Across this chapter's four sections, identify the ONE section whose own central discipline is an INVERSION of a discipline this book already established in an earlier chapter, name the earlier chapter and discipline being inverted, and explain precisely what is inverted about it.

## Where We Go Next

This chapter showed that a hard real-time deadline changes what "correct" means for a trading signal -- a strong signal computed too late must be treated as no signal at all, reapplying this book's own recurring never-suppress-a-flag discipline in its own inverted, safety-first form. Chapter 28 turns to a different real financial setting once more: insurance claims photo assessment with fraud flagging, extending Chapter 21's own real `CostRange` damage-assessment pattern with a genuine fraud-likelihood signal layered on top of it.

## Worked Solutions

**1.** A CROSSED book and a NO_LIQUIDITY book point a human reviewer toward two completely different real causes and two completely different real responses. A CROSSED book (best bid at or above best ask) usually indicates a real market-data feed problem, a stale or out-of-order update, or a genuine, rare crossed-market event at the exchange itself -- something a reviewer would investigate on the DATA FEED side. A NO_LIQUIDITY book (zero displayed size on both sides) instead indicates a real, if unusual, moment where no resting orders exist at the top of book at all -- something a reviewer would investigate on the MARKET CONDITIONS side, such as a trading halt or an illiquid instrument. Collapsing both into one generic "invalid" flag would erase exactly the distinction a reviewer needs to know which system to check first.

**2.** With `ask.size = 0`, the microprice formula `(ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)` has its second term, `bid_price * ask_size`, multiply by exactly zero, collapsing it to `bid_price * 0 = 0`. The denominator `bid_size + ask_size` becomes simply `bid_size` (since `ask_size` is 0). The whole expression reduces algebraically to `(ask_price * bid_size + 0) / bid_size`, and since `bid_size` is nonzero (checked separately by the NO_LIQUIDITY guard), this is exactly `ask_price * bid_size / bid_size = ask_price` -- an exact algebraic identity, not an approximation that merely happens to be close.

**3.** This book's own cross-architecture verification process runs the identical compiled binary (or an architecturally distinct but source-identical build) across four genuinely different execution environments -- native x86_64, a newer GCC version, an aarch64 target under emulation, and the user's own real device -- and requires the locked, byte-for-byte self-test output to match EXACTLY across all four. A real wall-clock measurement varies with the actual speed of the specific CPU running it, whether it is a native machine, an emulated aarch64 process under QEMU (which runs meaningfully slower than native execution), or a different real device entirely -- so any measured microsecond value embedded in the printed output would differ across these four legs even though the PROGRAM's own logic is identical, causing the verification's own diff-based comparison to fail for a reason that has nothing to do with a real bug in the code.

**4.** If the expensive stage were placed last, as in Test 3, a breach-detection implementation that (incorrectly) waited until the END of the loop and then scanned backward for the first stage whose CUMULATIVE total exceeded the budget would still report the correct final stage, since the last stage is trivially both the one that "ends" the loop and the one whose cumulative total first crosses the budget. Placing the expensive stage FIRST specifically distinguishes a correct implementation (which stops and reports the breach the INSTANT the running total crosses the budget, potentially after only the very first stage) from a subtly incorrect one that only checks the breach condition after summing every stage's cost first and then searching for where it happened -- the two implementations would agree on Test 3's own placement but disagree on Test 7's.

**5.** Simply asserting that "tax" and "cost" are absent from `NEG_LEXICON` only shows a property of the LEXICON's own construction -- it says nothing about whether that absence actually changes a real headline's own computed score in practice, and a reader would have to trust the assertion rather than see the effect. Running the IDENTICAL headline through both a finance-specific lexicon and a stated generic comparison lexicon and observing the SAME two words produce a NEUTRAL score in one case and a NEGATIVE score in the other demonstrates the real, practical CONSEQUENCE of the lexicon choice directly, on a concrete example, exactly the same "show the actual effect, not just the definition" standard Section 26.1's own dHash tests applied by comparing Hamming distances on real synthetic images rather than merely asserting the algorithm's own formula.

**6.** A headline such as "Earnings were not disappointing or weak this quarter" intends to negate BOTH "disappointing" and "weak" with the single word "not," but Section 27.3's own stated narrow-scope negation rule only flips the polarity of the SINGLE token immediately following "not" -- which, after tokenization, would be "disappointing" (assuming it were in the lexicon) or whatever word directly follows "not," while "weak" (appearing later in the same clause, after "or") would be scored with its own normal, un-negated polarity as an ordinary negative hit, incorrectly canceling out only part of the intended double negation.

**7.** If `within_deadline` were checked before `book_valid`, a stale event carrying an invalid (crossed) book would still be correctly forced to HOLD, since a missed deadline forces HOLD outright regardless of the book state -- so the FINAL action would not actually change in this specific scenario. What WOULD change is the audit trail's own reasons: reordering the checks would report "event missed its deadline" as the operative reason even when the book was ALSO independently invalid, hiding the fact that this specific event had two separate, independently disqualifying problems rather than just one -- exactly the same kind of lost diagnostic information Question 1 already identified for collapsing CROSSED and NO_LIQUIDITY into a single flag.

**8.** Reusing the identical numeric inputs isolates the deadline-forcing logic as the ONLY variable that changed between the two tests, which means any difference in the two tests' own results can be attributed to the deadline check alone, with complete confidence that no other input coincidentally caused the difference. A fresh, differently-valued example would leave open the possibility that the different result was caused by the DIFFERENT underlying signal values rather than the deadline logic itself, requiring a reader to trust that the two examples were constructed to isolate the same variable rather than seeing it demonstrated directly -- the identical "hold everything else fixed" principle already used by Section 26.4's own Test 5, which combined its MICR-forcing rule with signals that ALREADY independently rejected, to confirm the rule's own directionality specifically.

**9.** A trading desk that could see `forced_hold_stale` set to true on a real, recurring basis would have a concrete, actionable reason to investigate and fix a real latency problem in its own infrastructure -- network jitter, an overloaded feature-computation stage, a slow order-routing path -- since each occurrence represents a real trading opportunity the system detected but could not safely act on in time. A desk that only ever saw the final `TradeAction::HOLD`, with no way to distinguish a forced HOLD from a genuinely balanced computed HOLD, would have no way to tell whether its own system was missing real opportunities due to a fixable latency problem, or simply operating correctly in a quiet, low-conviction market -- two situations that call for completely different responses, one an engineering fix and the other no action at all.

**10.** Section 27.4's own capstone discipline is a direct inversion of Chapter 26.4's never-suppress-a-named-flag discipline. Chapter 26.4's own MICR-checksum-failure rule forces the STRICTER disposition (at least `ESCALATE_TO_REVIEW`) onto an otherwise-clean score, and explicitly never downgrades an already-stricter `AUTO_REJECT` back down. Section 27.4's own deadline-failure rule instead forces the SAFER, LEAST-risky disposition (`HOLD`) onto an otherwise-strong BUY or SELL score -- the exact opposite direction of override, appropriate to this chapter's own different real constraint: in fraud detection, missing a real red flag is the costly mistake a system must never make, while in low-latency trading, ACTING on a stale signal is the costly mistake, so the forcing rule pushes toward inaction rather than toward heightened scrutiny.
