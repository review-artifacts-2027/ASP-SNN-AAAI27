# ASP/SSP Ablation and Verification Suite

This folder contains a compact implementation for controlled experiments on
Active Spiking Perception (ASP) and its Slice Selection Policy (SSP). It is
separate from `../modelnet/train_modelnet.py`, which is the four-chunk
full-scale ModelNet paper pipeline.

## Run order

Run commands from this directory:

```bash
pip install -r requirements.txt
python tests/test_suite.py
python verification/run_all.py
```

Then run an experiment by combining one base config with one experiment config:

```bash
python -m experiments.run \
  --base configs/base/modelnet40.yaml \
  --exp configs/ablations/A1_theta.yaml

python -m experiments.run \
  --base configs/base/cifar100.yaml \
  --exp configs/ablations/A5_policy.yaml

python -m experiments.aggregate \
  --dir results/A5_policy/cifar100 --baseline ssp
```

The default five-seed protocol is `--seeds 0 1 2 3 4`.

## File map

```text
asp/
  geometry.py       point-cloud slicing and six-component descriptors
  ssp.py            selection policies, masking, and Gumbel sampling
  lif.py            learnable LIF dynamics and surrogate gradients
  backbone.py       modality-neutral slice encoder
  model.py          ASP model and early-exit trajectory
  train.py          losses and temperature schedule
  eval.py           anytime evaluation
  metrics.py        calibration, selective risk, and exit metrics
  teacher.py        distillation teacher
  datasets/         synthetic, ModelNet, ScanObjectNN, and CIFAR adapters
configs/
  base/             dataset/model/training defaults
  ablations/        A1-A5 controlled ablations
  strengthening/    S1-S6 stress and transfer tests
experiments/
  run.py            unified training/evaluation runner
  aggregate.py      confidence intervals and comparison summaries
verification/
  V1_stopping_time.py
  V2_selective_risk.py
  V3_submodularity.py
  V4_rank_equivalence.py
  V5_masking_dynamics.py
  V6_sufficiency.py
  V7_gumbel.py
tests/test_suite.py implementation checks
train_kd.py         optional distillation entrypoint
```

## Dataset behavior

- `synthetic`: generated locally; use this first.
- `modelnet10` and `modelnet40`: expect the normal-resampled directory under
  the configured `data_root`.
- `scanobjectnn`: expects PB-T50-RS HDF5 files under the configured root.
- `cifar10` and `cifar100`: downloaded through `torchvision` when absent.

The synthetic checks test mechanisms and software contracts. They are not
substitutes for full benchmark runs.

## Theory-to-code map

| Mechanism | Executable check |
|---|---|
| Stopping-time drift and threshold behavior | `verification/V1_stopping_time.py` |
| Selective-risk bound and calibration | `verification/V2_selective_risk.py` |
| Conditional submodularity audit | `verification/V3_submodularity.py` |
| Low-rank SSP parameterization | `verification/V4_rank_equivalence.py` |
| Coverage with and without masking | `verification/V5_masking_dynamics.py` |
| Membrane-state sufficiency probe | `verification/V6_sufficiency.py` |
| Gumbel-max selection consistency | `verification/V7_gumbel.py` |

Formal checks return a nonzero exit code when their stated mathematical or
software condition fails. Empirical comparisons that are not logical
consequences of a theorem are printed as `DIAGNOSTIC: OBSERVED` or
`DIAGNOSTIC: NOT OBSERVED` and never relabeled as formal passes.
