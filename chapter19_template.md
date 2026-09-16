# Chapter 19: Retail Inventory and Supply Chain Intelligence: Shelf Photography to Action

**What you will understand by the end of this chapter:**

- How to encode a real retail planogram -- the diagram stating which SKU belongs in which shelf slot and how many facings it should have -- into a deterministic, VLM-ready system prompt, with real validation, a real run-length compression pass for large stores, and a conservative token-budget guard that refuses to silently truncate a planogram it cannot fit.
- How to run Chapter 18's own real vision-language pipeline across a BATCH of shelf photographs, isolating one photo's own failure from the rest of the batch, and how to build a strict, from-scratch parser for the structured text a shelf-inspection prompt asks a VLM to answer with.
- How to model a REST integration into a retailer's own ERP/WMS systems as a from-scratch JSON serializer and a synthetic backend enforcing real HTTP status codes and real optimistic-concurrency conflict detection -- the discipline that keeps a live inventory record from suffering a silent lost update under concurrent writers.
- How to aggregate inventory snapshots honestly across many stores, excluding a store that did not report from a period's total rather than ever fabricating it as zero, and how to fit a real least-squares trend line to detect a stockout or overstock signal, gated by both a rate threshold and a projection horizon.
- How to reconcile a planogram against detections gathered over time into two genuinely different findings: an immediate misplacement, true the moment a single photo shows it, and a shrinkage suspicion, which this chapter's own detector refuses to raise until a stated number of CONSECUTIVE missing observations corroborates it.

**What you need to know first:**

- Chapter 18's complete vision-language pipeline -- Section 18.2's vision encoder and Section 18.3's fusion into the unchanged Qwen2 decoder -- which Section 19.2 runs, self-contained, across a real batch of photos rather than the single photo Chapter 18 ever tested it on.
- Chapter 18.4's asymmetric confidence-threshold reasoning and Chapter 18.5's interface-level protocol abstraction, both of which this chapter's own Sections 19.3 and 19.5 apply again in a genuinely different domain -- REST/ERP integration and evidence-based shrinkage detection, respectively.
- This chapter is more tractable end to end than Chapter 18: every section here is a real, fully verifiable, from-scratch implementation with no external spec this book cannot check byte-for-byte, and no section needed a reduced cross-architecture check the way Chapter 18.4's SQLite dependency once did.

---

Chapter 18 taught a vision-language model to inspect a single manufactured part in isolation and decide, from what it saw, whether the part was good. A retail shelf asks a related but genuinely different question: not "what is this," but "is this where it is supposed to be, and if it is missing, has it been missing long enough to matter." This chapter builds every piece that question requires, in the order a real retail deployment would need them -- encoding what SHOULD be on a shelf, observing what actually is there across a batch of real photographs, reporting both into the ERP and WMS systems a retailer already runs its business on, aggregating that reporting honestly across every store in a chain, and finally reconciling expectation against observation into the two real findings -- misplacement and shrinkage -- a store operator can actually act on.

## 19.1 Encoding a Planogram Directly into the System Prompt

### Intuition

A vision-language model looking at a shelf photograph has no way to know what SHOULD be there unless it is told, in the very same prompt as the photograph, exactly what a real retail planogram specifies: which SKU belongs in which shelf slot, and how many adjacent facings it should have.

### The Concept, In Detail

`Planogram` and `ShelfSlot` are a direct, real encoding of that retail concept -- a store layout id and a flat list of slots, each stating a shelf index, a position index, a SKU id and product name, and an expected facing count. `validate_planogram` refuses, rather than silently accepting, the two ways such a structure can fail to describe a physically meaningful shelf: two different slots claiming the identical (shelf, position) coordinate, which cannot both be true of one real shelf at once, and a slot claiming zero or negative facings, which is not a real stocking instruction at all.

`render_full_prompt` renders a planogram deterministically by first sorting every slot by (shelf, position) regardless of the order it was added in -- Test 2 confirms directly that the same three slots, added in forward and reversed order, render to byte-identical text -- because a system prompt's own content should depend on what a planogram SAYS, never on the incidental order a caller happened to populate it in.

A large store's own planogram can run long enough to threaten a real context-window budget, and this section takes that threat seriously rather than assuming it away. `render_compressed_prompt` applies a real run-length-style compression, merging a genuine RUN of consecutive positions on the same shelf sharing an identical SKU, product name, and facing count into a single range line -- Test 5 confirms this merges exactly the real compressible run in a synthetic planogram and, just as importantly, does NOT swallow an adjacent position whose SKU genuinely differs into that same range. `estimate_token_count` deliberately OVER-counts rather than under-counts, applying a stated conservative multiplier to a plain whitespace word count rather than re-running Chapter 12's own real BPE tokenizer -- because English text typically encodes to MORE sub-word tokens than words, and a budget guard whose whole job is refusing what does not actually fit must never be the more optimistic estimate of the two. `render_within_budget` tries the full rendering first, falls back to the compressed rendering only if the full one does not fit, and refuses outright -- returning no prompt at all -- if even the compressed rendering exceeds the stated budget, rather than fabricating a silently truncated planogram a VLM would have no way to know was incomplete.

### Code and Verification

@@CODE1@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 01_planogram_and_system_prompt.cpp -o 01_planogram_and_system_prompt
./01_planogram_and_system_prompt
```

**Sample input:** a well-formed planogram checked to validate and render deterministically and independently of insertion order; a duplicate slot and a non-positive facing count each checked to be refused with a specific, distinguishing error; two different planograms checked to render to genuinely different prompts; a real 4-position compressible run checked to merge into one range line without swallowing an adjacent differing position; and the token-budget guard checked to fit a small planogram's full rendering directly, fall back to compression for a large but compressible one under a tight budget, and refuse outright under an impossibly tight one.

@@OUT1@@

!!! warning "[COMMON TRAP] a token-budget estimate that under-counts is worse than no estimate at all"
    It is tempting to estimate a prompt's token cost by counting whitespace-delimited words directly, since that is the cheapest computation available without re-running a real tokenizer. Real BPE tokenizers typically split many English words into MULTIPLE sub-word tokens, so a plain word count systematically UNDER-estimates the real cost -- exactly backwards for a guard whose entire purpose is refusing a prompt that will not actually fit. This section's own `estimate_token_count` applies a stated, deliberately generous multiplier for exactly this reason: a budget guard that is sometimes too cautious wastes nothing but a little context-window headroom, while a budget guard that ever under-counts can let through a prompt a real model would truncate mid-instruction, with no warning to either the caller or the model that anything was cut off.

## 19.2 A Batch Shelf-Photo Processor

### Intuition

Section 19.1 built the prompt; this section runs the actual inspection. A retail batch job processes many shelf photographs in one run, and this section's own real, stated discipline is that a single bad photo must never take the rest of the batch down with it.

### The Concept, In Detail

`process_batch` runs Chapter 18's own real, unchanged vision encoder and multimodal fusion across every photo in a batch, against the SAME planogram-derived token template -- a real retail batch job inspects many photos against one shared prompt, not a bespoke one per photo. Each photo's own outcome, success or a specific kind of failure (a fusion count mismatch, KV-capacity exhaustion, or a NaN), is recorded into its own `BatchPhotoResult` rather than raised as an exception that would abort the whole run. Test 2 proves this isolation is real: a deliberately mis-sized photo among two well-formed ones trips `build_fused_embeddings`'s own count-mismatch refusal from Chapter 18.3, and the batch correctly reports 2 successes and exactly 1 isolated failure, naming the specific photo that failed and why -- the other two photos' own results are completely unaffected.

This section's own second, separate concern is a real, strict parser for the structured text a shelf-inspection prompt asks a VLM to answer with: one detection per line, in a stated `SKU=...;SHELF=...;POSITION=...;CONFIDENCE=...` format. `parse_detection_line` is tested against REPRESENTATIVE EXAMPLE completions, not against genuine output from the real forward pass Test 1 and Test 2 already verified -- and that gap is stated honestly rather than glossed over. This book's own from-scratch model weights are random and untrained, exactly as they have been since Chapter 15's own synthetic self-tests; only a real, fine-tuned checkpoint would ever cause the real forward pass to emit genuinely structured text. What this section CAN and does verify for real, without depending on trained weights at all, is that the real multimodal pipeline itself runs correctly across a batch, and that the parser downstream of whatever a real deployed model eventually says is strict and correct: `std::from_chars` is used throughout, matching this book's own standing preference for checked return values over exceptions, and Test 4 confirms all eight real ways a detection line can be malformed -- a missing field, a duplicate field, an unexpected extra field, a non-integer SHELF, a non-numeric CONFIDENCE, an out-of-range CONFIDENCE, an empty SKU, and a field with no `=` at all -- are each refused with a specific, distinguishing error.

### Code and Verification

@@CODE2@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 -ffp-contract=off -DMDSPAN_IMPL_STANDARD_NAMESPACE=std -DMDSPAN_IMPL_PROPOSED_NAMESPACE=experimental -I_vendor_mdspan/include 02_batch_shelf_photo_processor.cpp -o 02_batch_shelf_photo_processor
./02_batch_shelf_photo_processor
```

**Sample input:** a batch of three well-formed synthetic photos run end-to-end through the real vision-language pipeline, checked for finiteness and determinism across two independent batch runs; a batch with one deliberately mis-sized photo among two good ones, checked to isolate exactly that photo's failure while the other two still succeed; the structured-output parser checked against three well-formed lines (varying field order and boundary confidence values); and the parser checked to reject all eight real kinds of malformed input with specific, distinguishing errors.

@@OUT2@@

!!! warning "[COMMON TRAP] one bad photo aborting an entire batch turns a small, isolated problem into a large one"
    A batch loop that lets one photo's exception propagate out of the whole function trades a small, specific, easily-diagnosed problem (photo `photo-bad` failed with a fusion count mismatch) for a large, vague one (the ENTIRE batch of photos this run was supposed to inspect produced no results at all). On a real production line running this batch job on a fixed schedule, that difference is the difference between one shelf going un-reconciled today and the ENTIRE STORE going un-reconciled today, from a single photo that was probably just corrupted or mis-captured. `process_batch`'s own per-photo isolation, tested directly in Test 2, is what keeps a real, specific failure from ever escalating into an unrelated, unnecessary one.

## 19.3 REST Integration into ERP/WMS Systems

### Intuition

Sections 19.1 and 19.2 can now tell what a shelf should hold and what a photograph shows it actually holds. None of that reaches a retailer's own business systems until it crosses a real interface nearly every modern retail backend actually exposes: a REST API exchanging JSON over HTTP's own real methods and status codes.

### The Concept, In Detail

This section is stated as an INTERFACE-LEVEL abstraction, the same honest voice Chapter 18.5 applied to OPC UA: there is no real ERP or WMS endpoint reachable from this environment, so this section builds its own small, from-scratch JSON serializer -- with real string escaping, verified in Test 1 against a string containing a newline, a quote, and a backslash all at once -- and a `SyntheticErpBackend` implementing real REST semantics standing in for a genuine system. What this section does not claim is byte-for-byte compliance with any one real vendor's own actual API shape; what it teaches faithfully is the general discipline every one of those real systems' own integrations still has to get right.

That discipline centers on optimistic concurrency. A real inventory record is being updated by this chapter's own reconciliation job, by cashiers ringing up sales, and by other stores' own systems, all at once -- a PUT that blindly overwrites whatever is currently stored risks a genuine LOST UPDATE, where two concurrent writers each believe they are applying the authoritative value and whichever lands second silently erases the other. `put_stock_level`'s own `expected_version` parameter is the same real idea as an HTTP `If-Match` header carrying an ETag: Test 4 confirms a PUT whose caller does not currently hold the record's own latest version is refused with a real `409 Conflict`, and -- just as important as the refusal itself -- that the record's stored value is left completely untouched by the rejected write, never partially applied. Test 3 draws the finer distinction this discipline depends on: PUTting the identical quantity a second time, using the version the FIRST PUT actually returned, succeeds and leaves the business-observable quantity unchanged, even though the bookkeeping version counter itself still advances -- version numbers are not business state, only the mechanism that protects it. Test 6 draws the matching contrast on the POST side: `create_shrinkage_ticket` is a real, intentionally NON-idempotent operation, and two calls with byte-identical inputs correctly produce two different ticket ids, because each POST genuinely creates a new resource rather than updating an existing one.

### Code and Verification

@@CODE3@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 03_erp_wms_rest_integration.cpp -o 03_erp_wms_rest_integration
./03_erp_wms_rest_integration
```

**Sample input:** the JSON serializer checked against a string with a newline, a quote, and a backslash, and against a real 3-field object's exact expected serialization with field order preserved; a GET on an unknown record checked to return 404 and a first PUT checked to create it at version 1; repeating the same PUT with the current version checked to leave the observable quantity unchanged; a stale-version PUT checked to be refused with 409 while leaving the stored record completely untouched; an empty SKU and a negative quantity each checked to be refused with 400 without side effects; and two identical-input POSTs to `create_shrinkage_ticket` checked to produce two different ticket ids.

@@OUT3@@

!!! warning "[COMMON TRAP] treating a PUT's version counter as though it were part of the business data"
    It is tempting to read `put_stock_level`'s returned version advancing from 1 to 2 as a sign that "something changed," when Test 3 deliberately constructs the case where the underlying, business-relevant quantity did NOT change at all -- the same value (50) was written twice, each time with the correct current version. The version counter's own real job is protecting against a LOST UPDATE, not describing the business state itself; a caller (or a dashboard built on top of this backend) that conflates "version incremented" with "inventory actually changed" would report a false change event on every single idempotent re-confirmation a real reconciliation job performs, drowning genuine change events in noise generated by nothing more than the concurrency-control mechanism doing its job correctly.

## 19.4 Multi-Store Aggregation and Trend Detection

### Intuition

A single store's own inventory record answers "how much of this SKU does store 42 have right now." A retail chain needs a genuinely different question answered across every store at once: is this SKU trending toward a stockout, and by when -- and answering that honestly requires knowing which stores' numbers are even reflected in the total.

### The Concept, In Detail

Every timestamp in this section is a caller-supplied discrete PERIOD INDEX, never real wall-clock time -- this book's own standing discipline against locking real, machine-specific timing into output this book's cross-architecture check compares byte-for-byte, applied here to a business reporting period instead of a hardware clock tick. `aggregate_by_sku` sums on-hand quantity across every store that actually reported a given period, and this section's own real discipline is what it does with a store that did NOT report: Test 2 confirms that store is excluded entirely from both the total and the reporting count, and named explicitly in `stores_missing`, rather than ever being folded into the total as an assumed zero. A store's silence is a fact about DATA COMPLETENESS, and treating it as a fact about INVENTORY would fabricate a stockout signal for a store that may simply have missed a scheduled upload.

`detect_trend` fits a real ordinary-least-squares regression line to a (period, quantity) time series -- Test 3 confirms the fitted slope and intercept against a perfectly linear synthetic decline are exact, hand-computable values, and that the resulting projected zero-inventory period matches a hand-computed zero-crossing precisely. A trend is only flagged StockoutRisk when it clears BOTH a rate threshold (declining at least a stated amount per period) AND a projection-horizon check (the regression line's own zero-crossing must fall within a stated number of future periods) -- Test 6 proves both gates are doing real, independent work by constructing a series that clears the rate threshold by a wide margin (a steep decline of 50 units per period) but starts from such a large quantity that its real zero-crossing sits thousands of periods away, far beyond any actionable horizon, and is correctly left Stable rather than raising an alert about a shortage that is not actually imminent.

### Code and Verification

@@CODE4@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 04_multistore_aggregation_and_trends.cpp -o 04_multistore_aggregation_and_trends
./04_multistore_aggregation_and_trends
```

**Sample input:** aggregation across three fully-reporting stores checked against a hand-computed total; a store that did not report a period checked to be excluded from that period's total and named in `stores_missing`, never assumed to be zero; a perfectly linear decline checked against an exact, hand-computable regression slope, intercept, and zero-crossing projection; a flat noisy series checked to remain Stable; a perfectly linear growth checked to be flagged Overstock; and a steep-by-rate decline from an enormous starting quantity checked to remain Stable because its real projected zero-crossing falls far beyond the stated horizon.

@@OUT4@@

!!! warning "[COMMON TRAP] folding a non-reporting store into an aggregate as an implicit zero"
    Summing on-hand quantity across "every store in my list" and simply not adding anything for a store with no snapshot LOOKS identical, in the resulting total, to correctly excluding that store -- until the very next question asked of that total is "how many units of this SKU exist across the chain," at which point a missing store's real, unknown, possibly-substantial inventory has silently become zero in every downstream calculation. `aggregate_by_sku`'s own `stores_missing` field exists specifically so a caller can distinguish "3 units total, all 3 stores reporting" from "3 units total, only 1 of 3 stores reporting" -- two totals that happen to share a number but mean completely different things about how much of the chain's real inventory that number actually reflects.

## 19.5 Shrinkage and Misplacement Detection from Ordinary Shelf Photographs

### Intuition

This section closes the loop the chapter opened: Section 19.1's planogram states what SHOULD be on a shelf, Section 19.2's batch processor and parser produce what a VLM actually observed there, and this section reconciles the two into the findings a store operator genuinely acts on.

### The Concept, In Detail

A misplacement is immediate: if a slot's own detection names a DIFFERENT SKU than the planogram expects, that is a fact about ONE photo, true the moment it is observed, and Test 1 confirms `ShrinkageDetector::reconcile` flags it on the very first call, with no corroboration required. A missing item is a genuinely different kind of claim. A single empty-looking photo could mean real shrinkage, or it could mean a customer mid-reach, a restocking cart blocking the shelf, or an ordinary brief gap between deliveries -- and reporting shrinkage off one such photo would be exactly the kind of overconfident claim this book has refused to make since its own earliest treatment of numerical precision. `ShrinkageDetector` instead tracks, per slot, how many CONSECUTIVE reconciliations in a row have found it empty, and only raises `SHRINKAGE_SUSPECTED` once a stated threshold is crossed -- Test 2 confirms a single miss is correctly left `EMPTY_OBSERVED`, and Test 3 confirms three consecutive misses correctly cross a threshold of three.

The counter's own reset behavior is this section's own sharpest correctness point, and Test 4 is built specifically to prove it: two misses, followed by the item genuinely reappearing, followed by two MORE misses, must land at a streak of 2 (still below threshold) rather than 4 -- a detector that merely PAUSED the count across the reappearance, rather than genuinely RESETTING it, would treat two unrelated two-miss runs as one continuous shrinkage signal the real data never supported. Test 5's own multi-period reconciliation report ties every piece of this chapter together into the artifact a store operator would actually receive: across five simulated periods, exactly one real misplacement event and exactly one slot ever crossing the shrinkage threshold, each attributable to a specific shelf position and a specific run of evidence. This section's own scope ends at producing that finding -- a real deployment would report each `SHRINKAGE_SUSPECTED` slot by calling Section 19.3's own `create_shrinkage_ticket` against the real ERP/WMS backend, closing the chapter's own loop from a shelf photograph all the way to an actionable ticket in the systems a retailer already runs.

### Code and Verification

@@CODE5@@

**Compile and run:**

```bash
g++ -std=c++23 -Wall -Wextra -O2 05_shrinkage_and_misplacement_detection.cpp -o 05_shrinkage_and_misplacement_detection
./05_shrinkage_and_misplacement_detection
```

**Sample input:** a misplaced slot checked to be flagged on the very first observation; a single missing observation checked to remain `EMPTY_OBSERVED` below threshold; three consecutive missing observations checked to cross a threshold of three; an item reappearing after two misses checked to reset the streak to zero, with two further misses afterward correctly still below threshold rather than continuing a stale count; and a full five-period reconciliation checked against a hand-verified report of exactly one misplacement event and exactly one shrinkage-suspected slot.

@@OUT5@@

!!! warning "[COMMON TRAP] pausing a streak counter across a reappearance instead of genuinely resetting it"
    A detector that only decrements or freezes its consecutive-miss counter when an item reappears, rather than resetting it to zero, would treat an item that goes missing for two periods, comes back for one photo, and then goes missing for two MORE periods as continuous evidence of the SAME four-period shrinkage event -- when the real, honest interpretation is two separate two-period gaps, each individually below whatever threshold a retailer considers actionable. Test 4's own construction proves `ShrinkageDetector` gets this right by checking the counter's exact value immediately after the reappearance (zero) and again after two further misses (two, not four) -- the single number that would silently reveal a "pause instead of reset" bug if the reset logic were wrong.

## Chapter Summary

This chapter took Chapter 18's real vision-language pipeline into a genuinely different deployment target: a retail shelf, where what should be there has to be stated as explicitly as what a camera actually shows. Section 19.1 built a real, validated planogram structure, a deterministic rendering with run-length compression for large stores, and a conservative token-budget guard that refuses to fabricate a silently truncated prompt. Section 19.2 ran Chapter 18's own real pipeline across a batch of photographs with per-photo failure isolation, and built a strict structured-output parser tested honestly against representative example completions rather than genuine output this book's own untrained synthetic weights could never produce. Section 19.3 modeled a REST integration into a retailer's ERP/WMS systems as a from-scratch JSON serializer and a synthetic backend enforcing real optimistic-concurrency conflict detection, distinguishing an idempotent PUT's bookkeeping version from its actual business state and contrasting it against a genuinely non-idempotent POST. Section 19.4 aggregated inventory honestly across many stores -- never fabricating a non-reporting store's contribution as zero -- and fit a real least-squares regression to detect a stockout or overstock trend, gated by both a rate threshold and a projection horizon so a slow decline on a large quantity is not mistaken for an imminent shortage. Section 19.5 closed the chapter's own loop, reconciling the planogram against detections gathered over time into an immediate misplacement finding and a genuinely more cautious, evidence-corroborated shrinkage finding, with a reset-on-reappearance discipline this section proved correct by hand.

## Self-Check Questions

1. Section 19.1's `estimate_token_count` applies a stated multiplier to a plain whitespace word count rather than counting words directly. Explain specifically why an UNDER-estimate would be the more dangerous error for a token-budget guard to make, compared to an OVER-estimate.
2. Section 19.1's `render_within_budget` tries the full rendering, then the compressed rendering, and only then refuses outright. Why does it never attempt a THIRD, more aggressive form of truncation instead of refusing?
3. Section 19.2's `process_batch` isolates each photo's own failure into its own `BatchPhotoResult`. Describe the specific, concrete difference in outcome between a batch job built this way and one where a single photo's fusion-count mismatch is allowed to raise an exception out of the whole batch function.
4. Section 19.2's own structured-output parser is tested against representative example completions rather than genuine output from the real forward pass its own Test 1 and Test 2 already verified. Explain exactly why this gap exists, and what Section 19.2 DOES verify for real despite it.
5. Section 19.3's Test 3 shows a PUT's own version counter advancing from 1 to 2 even though the business-observable quantity did not change. Explain why treating that version increment as evidence of a real inventory change would be a mistake.
6. Section 19.3's `create_shrinkage_ticket` is deliberately NOT idempotent, unlike `put_stock_level`. Explain the real difference between these two operations that justifies this design choice.
7. Section 19.4's `aggregate_by_sku` reports a `stores_missing` list rather than simply omitting a non-reporting store's contribution silently. What real downstream question becomes impossible to answer correctly without that list?
8. Section 19.4's Test 6 constructs a series with a steep, threshold-clearing decline rate that is nonetheless left `Stable`. Explain why BOTH a rate threshold and a projection-horizon check are necessary, using this test's own numbers.
9. Section 19.5 flags a misplacement immediately but requires multiple consecutive observations before suspecting shrinkage. Explain the real, substantive difference between these two claims that justifies treating them so differently.
10. Section 19.5's Test 4 checks the detector's exact streak value immediately after an item reappears and again after two further misses. What SPECIFIC bug would this test catch that a test only checking the FINAL status after all five reconciliations might miss?

## Where We Go Next

This chapter showed that the same real vision-language core built in Chapter 18 can move from an isolated manufacturing part to a full retail reconciliation loop -- planogram, batch inspection, ERP/WMS integration, honest multi-store aggregation, and evidence-based shrinkage detection -- by changing what surrounds that core, not the core itself. Retail's own real stakes are largely economic: a missed stockout costs a sale, a false shrinkage alert costs an afternoon of a store associate's time. The next domain this book turns to raises the stakes considerably. Chapter 20 takes this same real vision-language discipline into medical imaging triage, where a VLM's own output feeds a human radiologist's own workflow rather than an inventory system, and where this book's own honest-scope discipline has to extend into genuinely new territory: the regulatory boundaries such a system must respect, and a stated, honest account of exactly where a vision-language model's own interpretation stops being reliable enough to act on without a human in the loop.

## Worked Solutions

**1.** An under-estimate is the more dangerous error because it can cause the guard to report a prompt as fitting within budget when a real tokenizer would actually produce MORE tokens than the stated limit -- meaning a real deployment would send a prompt a real model then truncates mid-instruction, silently and without warning, potentially cutting off the planogram itself partway through a shelf's own listing. An over-estimate's only cost is refusing a prompt that might have technically fit, wasting a small amount of available context-window headroom -- a far cheaper mistake than silently shipping a broken prompt to a real deployed model.

**2.** A third, more aggressive truncation (dropping slots entirely, or abbreviating product names) would mean the rendered prompt no longer describes the ACTUAL planogram -- a VLM told an incomplete, silently-edited version of the shelf's own real expected contents could then reasonably compare a photograph against slots that were never actually communicated to it, generating a plausible-looking but factually ungrounded misplacement or shrinkage report. Refusing outright when even the compressed rendering does not fit is the honest alternative: it tells the CALLER clearly that this specific planogram cannot be safely summarized at this budget, rather than quietly handing the model a fabricated, incomplete picture of the shelf and letting a wrong answer look exactly like a right one.

**3.** With per-photo isolation, a batch job processing ten photos where one is corrupted or mis-captured produces nine real, usable inspection results and one specific, named, diagnosable failure -- a store operator or an automated reconciliation job downstream still gets nine-tenths of the day's real work done, and knows exactly which photo needs a re-take. Without isolation, that same single bad photo would cause the entire batch function to raise an exception and produce NO results at all for any of the ten photos, turning one small, specific, easily-explained problem (one photo failed) into a much larger, vaguer one (the whole store's reconciliation run failed today, for a reason that requires digging through logs to even identify).

**4.** The gap exists because this book's own from-scratch model weights, exactly as they have been since Chapter 15's own self-tests, are randomly initialized and never trained -- so a real forward pass through them produces mathematically real, finite, deterministic numbers, but has no reason to produce genuinely structured, meaningful text, since nothing about random weights was ever taught what that structure should look like. Only a real, fine-tuned checkpoint would cause the real forward pass to emit text a strict parser could meaningfully be tested against as GENUINE model output. What Section 19.2 verifies for real despite this: that the real vision-language pipeline itself (encoding, fusion, decoding) runs correctly and deterministically across a whole batch of photos without crashing, without NaNs, and with correct per-photo failure isolation -- and, completely separately, that the parser downstream of whatever text a real deployed model eventually produces is strict and handles every real malformed case correctly, using representative example strings rather than this book's own meaningless synthetic output.

**5.** The version counter's real job is protecting against a lost update under concurrent writers, not describing the business state itself -- Test 3 deliberately writes the SAME quantity (50) twice, each time with the correct current version, specifically to demonstrate that the version can advance (from 1 to 2) while the actual inventory count a store's own dashboard would display never changes at all. A caller or a monitoring system that treated every version increment as a real inventory change event would report a false "something changed" alert on every single idempotent re-confirmation a routine reconciliation job performs, which would very quickly drown any GENUINE change events in noise generated purely by the concurrency-control mechanism doing exactly what it is supposed to do.

**6.** `put_stock_level` sets an ABSOLUTE, addressable value (a specific SKU's stock level at a specific store) that a caller can meaningfully re-apply with the intent "make sure this ends up at this value" -- a PUT is naturally idempotent because "set X to 50" run twice ends in the identical state as running it once. `create_shrinkage_ticket` instead records a NEW EVENT (a distinct instance of suspected shrinkage, worth its own audit trail) each time it is called, and there is no meaningful sense in which two calls with identical inputs should collapse into "the same ticket" -- a real shrinkage investigation genuinely benefits from a separate ticket per detected event, even if two events happen to share the same store, SKU, and description text.

**7.** Without `stores_missing`, a caller cannot distinguish "the total reflects every store I expected to hear from" from "the total reflects only some of the stores I expected to hear from, and the rest simply have not reported yet" -- two situations that can produce the exact same total_quantity number while meaning completely different things about how much of the chain's real inventory that number actually accounts for. A trend-detection or restocking decision made on an incomplete aggregate, without knowing it was incomplete, could badly misjudge how urgent a real shortage is, or manufacture an apparent shortage from what is really just a reporting gap.

**8.** Test 6's series has a real, hand-computable slope of -50 per period -- comfortably past the stated rate threshold of -2.0, so a rate-threshold check ALONE would flag it as declining fast enough to worry about. But the series starts from 1,000,000 units, so its real, hand-computable zero-crossing sits roughly 20,000 periods away -- vastly beyond the stated 20-period projection horizon. A rate check alone would raise an alert about a shortage that will not actually happen for tens of thousands of periods, which is not an actionable warning; requiring BOTH the rate threshold AND the horizon check ensures a StockoutRisk signal means something genuinely urgent -- fast enough decline AND close enough in time -- rather than merely "declining somewhat quickly, whenever that eventually catches up."

**9.** A misplacement is a claim about what a single photograph directly shows RIGHT NOW -- if the detected SKU at a slot differs from the expected one, that mismatch is a complete, self-contained fact requiring no further evidence, exactly the way Chapter 18.1's own dropped-packet detection needed no corroboration beyond the one frame it was reported in. A missing item is a claim about the ABSENCE of something, which a single photograph cannot distinguish from several ordinary, non-shrinkage explanations (a customer mid-reach, a restocking cart temporarily blocking the view, a brief gap between deliveries) -- only a PATTERN of the same slot being empty across multiple, separately-captured observations makes a genuine shrinkage explanation more likely than these ordinary alternatives, which is exactly why the shrinkage claim requires corroboration that the misplacement claim does not.

**10.** A test that only checks the FINAL status after all five reconciliations in Test 4 could still pass even if the detector merely PAUSED its counter across the reappearance instead of genuinely resetting it to zero -- for instance, a buggy implementation that resumed counting from 2 (rather than 0) after the reappearance would reach a streak of 4 after the two further misses, which happens to still be `EMPTY_OBSERVED` at a threshold of 3 only by coincidence of this specific test's exact numbers, and a slightly different threshold or miss count could let that same bug slip through a final-status-only check undetected. Checking the exact intermediate streak VALUE immediately after the reset (confirming it is precisely 0, not merely "still below threshold") is the one assertion that would catch a "pause instead of reset" bug directly, regardless of how many further misses follow or what threshold is configured.
