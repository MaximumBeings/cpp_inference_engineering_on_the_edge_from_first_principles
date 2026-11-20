# References and KV complexity corrections

TA-25: corrected the Chapter 13 template and source comments to compare per-step and total dense-attention costs separately. KV caching avoids repeated prefix projections; attention remains O(N) per decode step and O(N^2) across N generated tokens at fixed model dimensions. Added fixed-window qualification.

TA-26: replaced absent Chapter 1/Appendix F references with existing foundations, concurrency, serving and diagnostics sections. Updated source comments as well as templates and generated pages.

Validation: Chapter 13 verifier compiled and reran all five examples with matching output; sample-output and inference regression suites passed. See adjacent logs. The reference regression checks all published Markdown pages for absent chapter references.
