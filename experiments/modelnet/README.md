# ModelNet10 and ModelNet40

This folder contains the point-cloud classification workflow used for the
ModelNet experiments, along with source-level ablations and verification
utilities.

## Layout

- `asp/` — model, spiking selective policy, geometry, training, and metrics.
- `configs/base/` — base configurations for ModelNet10, ModelNet40, and a
  small synthetic sanity-check dataset.
- `configs/ablations/` — A1–A5 ablation overrides.
- `configs/strengthening/` — additional point-cloud stress-test overrides.
- `experiments/run.py` — combined training/evaluation runner.

## Data layout

Set `data_root` in the base configuration (or edit the YAML file).  The loader
expects the standard `modelnet40_normal_resampled` extraction:

```text
data_root/
  modelnet40_normal_resampled/
    modelnet10_shape_names.txt
    modelnet40_shape_names.txt
    modelnet10_train.txt
    modelnet10_test.txt
    modelnet40_train.txt
    modelnet40_test.txt
    class_name/*.txt
```

Each shape file must contain comma-separated point rows.  The loader reads the
first three coordinates and resamples them to `n_points`.

## Run order

From this directory:

```bash
python tests/test_suite.py
```

Run a short synthetic smoke experiment before a dataset-backed run:

```bash
python -m experiments.run \
  --base configs/base/synthetic.yaml \
  --exp configs/ablations/A1_theta.yaml \
  --seeds 0 --epochs 1
```

Train a ModelNet40 configuration:

```bash
python -m experiments.run \
  --base configs/base/modelnet40.yaml \
  --exp configs/ablations/A1_theta.yaml \
  --seeds 0
```

Use `configs/base/modelnet10.yaml` for ModelNet10.  Output files are written
under `results/`, which is ignored by Git.

## Evaluation

`experiments/run.py` evaluates after the configured interval and writes a JSON
summary per seed.  `experiments/aggregate.py` can combine those summaries once
all intended seeds have completed.

See `../../docs/PAPER_CODE_CROSSCHECK.md` before treating a default
configuration as an exact reproduction of manuscript results.

## Bundled source diagnostics

The available seed-0 diagnostic outputs are stored outside this experiment
folder:

- Results: `../../artifacts/source_diagnostics/results/`
- Weights: `../../artifacts/source_diagnostics/weights/`
- Exact diagnostic base configs:
  `../../artifacts/source_diagnostics/configs/`

These are 30-epoch diagnostic runs and are not manuscript checkpoints. Validate
all 50 state dictionaries against this implementation from the repository root:

```bash
python tools/validate_diagnostic_weights.py
```
