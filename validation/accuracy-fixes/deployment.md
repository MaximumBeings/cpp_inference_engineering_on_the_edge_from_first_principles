# Deployment and CUDA accuracy corrections

- **TA-07:** Narrowed security claims: FNV is a non-security annotation checksum; no source-video or protected-digest guarantee. Corrected template, source comments and sample stdout; cited RFC 9923 and NIST IR 8387.

- **TA-08:** Narrowed legal claims: rolling aggregate rule is an illustrative screening heuristic, not the definition of structuring. Removed legitimacy and exact-legal-boundary conclusions from explanation, source and output.

- **TA-09:** Narrowed standards scope: retained 44 CSS pixels as optional WCAG 2.5.5 AAA advisory for assumed non-exempt targets. Clearly distinguished AA 2.5.8 and exceptions; no complete accessibility or legal conformance claim.

- **TA-19:** Functionality fix: distant same-time transactions receive infinite required speed and are flagged; coincident same-place events retain zero speed; reversed timestamps throw invalid_argument. Four regression checks cover direct and batch paths.

- **TA-20:** Narrowed legal claims: review routing and two-signal threshold are example policy. Distinguished institutional governance from automated/batch filing transmission; no universal human-only filing law asserted.

- **TA-21:** Scope correction only: all three attention implementations expressly compute unscaled noncausal single-head softmax(Q K^T)V, without production decoder integration. CUDA math unchanged.

- **TA-22:** Scope correction: fixed score-tile storage is not fixed total attention memory. Numerical comparisons are within tolerance, not bitwise equality or universal proof. Updated template, source comments and sample stdout.

- **TA-23:** Narrowed privacy claims: schema concept checks exactly three field names. Added a regression demonstrating that a differently named identity member is outside the guard; no pipeline-wide privacy guarantee.

- **TA-24:** Functionality/output correction: gaps now say No tagged observations; coverage unknown. Added empty-observation checks so missing coverage cannot be narrated as inactivity.

## Verification

Changed host examples compiled with clang++ -std=c++23 -Wall -Wextra -O2 and passed: schema 25/25; incident report 19/19; accessibility 19/19; AML heuristic 10/10; travel 17/17; routing 19/19; online softmax 11/11. Their captured stdout was refreshed. Chapter builders 22, 25, 29 and 32 regenerated published Markdown. All four full chapter verifiers passed, including locked-code/output matching and deterministic host reruns. Their transcripts are saved alongside this report.

No CUDA kernel arithmetic changed and no new GPU execution is claimed. Existing A100 evidence remains historical validation of its explicitly stated contract; local Chapter 32 verification skips CUDA compilation because nvcc is unavailable. New legal/security/standards text narrows educational scope rather than implementing evidentiary custody, a filing service, or full WCAG conformance. Illustration specs 25-01 and 25-02 were updated; the integrating agent runs the global illustration builder.
