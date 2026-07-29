# Validation report

This report records the CPU validation performed on the anonymous review
package. It distinguishes formal checks from empirical diagnostics.

## Software checks

- Package validator: passed
- Python source parsing: 57 files passed
- YAML parsing: 23 files passed
- Full-scale no-data smoke test: 8/8 passed
- ModelNet no-data forward-pass smoke test: passed for SPM and ASP
- Rigor-suite implementation tests: 17/17 passed

## Formal mechanism checks

`experiments/rigor_suite/verification/run_all.py` completed with all formal
checks passing:

- positive measured drift, stopping-time bound, and pathwise threshold
  monotonicity;
- selective-risk bound on the evaluated threshold sweep;
- diminishing greedy gains and sampled-chain submodularity audit;
- rank cap, exact low-rank score reconstruction, and parameter accounting;
- unmasked revisits, reduced coverage, and late-increment stall;
- capacity-matched membrane sufficiency probe;
- Gumbel-max agreement identity and finite straight-through gradients.

## Empirical diagnostics

The following results are reported descriptively and are not relabeled as
theorem checks:

- High-threshold censoring did not match the fraction of examples with
  non-positive fitted drift in the 12-epoch tiny run.
- The tiny-run SSP did not track the label-aware oracle ordering more closely
  than a random ordering.
- The tiny-run membrane-driven policy did not beat the random ordering in
  early anytime accuracy.
- Final-step ECE was 0.1577 in the tiny run.
- The unmasked policy was less accurate and no faster in the tiny run.
- Trained Gumbel selection agreement exceeded untrained agreement.

These diagnostics depend on optimization scale and are intentionally visible.
They are not required premises or logical consequences of the corrected formal
statements in the technical supplement.

