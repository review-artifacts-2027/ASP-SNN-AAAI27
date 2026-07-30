# ModelNet Source Diagnostics

> **Not manuscript checkpoints or headline results.**

This directory preserves all in-scope weights and generated outputs that were
present in the approved source snapshots. The files are short, seed-0
diagnostic runs. They are provided for inspection and traceability, not as a
claim of paper-result reproduction.

## Inventory

| Dataset | Checkpoints | JSON files | CSV files | A1 baseline full-horizon accuracy |
| --- | ---: | ---: | ---: | ---: |
| ModelNet10 | 25 | 50 | 5 | 0.4812775 |
| ModelNet40 | 25 | 50 | 5 | 0.3991086 |

Each dataset contains A1 threshold, A2 masking, A3 geometry, A4 SSP-dimension,
and A5 policy diagnostics. Every variant uses seed 0.

## Diagnostic recipe

The exact base configurations are in `configs/`. Both use 30 epochs,
`d_model=128`, `enc_hidden=128`, `d_ssp=64`, `k_slices=16`, and batch size 32.
Variant overrides are encoded by the A1-A5 configurations in
`../../experiments/modelnet/configs/ablations/`.

The retained default ModelNet configs are not substituted for these diagnostics
because they now encode a different schedule. Use the configs in this directory
when loading these checkpoints.

## Layout

```text
results/<dataset>/<experiment>/rows.csv
results/<dataset>/<experiment>/seed0/<variant>/{history,summary}.json
weights/<dataset>/<experiment>/seed0/<variant>/model.pt
```

Validate all file counts and formats:

```bash
python tools/validate_repository.py
```

Validate that every state dictionary loads strictly into the retained model:

```bash
python tools/validate_diagnostic_weights.py
```
