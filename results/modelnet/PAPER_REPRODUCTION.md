# ModelNet10/40 reproduction record

Entrypoint: `experiments/modelnet/train_modelnet.py`

Recorded configuration:

- 300 epochs, batch size 64, single seed
- 1,024 points, dimension 384, depth 12
- 128 groups of 32 points, four ASP chunks
- knowledge distillation, BF16, and `torch.compile`
- NVIDIA H100 NVL, PyTorch 2.11.0+cu128
- canonical ModelNet10/40 train/test splits from
  `modelnet40_normal_resampled` text point clouds

The source run and the paper run used the same split sizes and point count but
different point sampling sources. Exact percentages should therefore not be
expected from this single-seed reproduction.

## Results

### ModelNet10

| Metric | Reproduction | Reported | Difference |
|---|---:|---:|---:|
| Teacher | 90.75% | 91.19% | -0.44 pp |
| SPM | 91.63% | 92.51% | -0.88 pp |
| ASP | 92.07% | 93.28% | -1.21 pp |
| ASP - SPM | +0.44 pp | +0.77 pp | - |
| Average chunks | 3.96 / 4 | 3.89 / 4 | - |
| Firing rate | 27.11% | 26.27% | +0.84 pp |

### ModelNet40

| Metric | Reproduction | Reported | Difference |
|---|---:|---:|---:|
| Teacher | 88.82% | 89.14% | -0.32 pp |
| SPM | 87.68% | 89.51% | -1.83 pp |
| ASP | 86.95% | 89.10% | -2.15 pp |
| ASP - SPM | -0.73 pp | -0.41 pp | - |
| Average chunks | 3.84 / 4 | 3.84 / 4 | - |
| Firing rate | 24.22% | 24.45% | -0.23 pp |

Both qualitative comparisons reproduce: ASP is above SPM on ModelNet10 and
below SPM on ModelNet40. The late-exit behavior and firing rates are also close
to the reported values. The remaining 1-2 percentage-point student gap is not
separable from seed and point-sampling variation with one run.

## Included artifacts

```text
ModelNet10/final_results.json
ModelNet10/histories.json
ModelNet40/final_results.json
ModelNet40/histories.json
run_logs/a100_mn10.log
run_logs/a100_mn40.log
```

Model weights and optimizer states are excluded from the review archive. The
JSON histories contain the full 300-epoch SPM and ASP curves.

