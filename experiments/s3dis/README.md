# S3DIS Semantic Segmentation

This folder contains the indoor point-cloud semantic-segmentation workflow.

## Layout

- `datasets/s3dis.py` — Area-based room loader and block sampler.
- `datasets/slicing.py` — point-slice construction.
- `models/` — encoder, spiking policy, LIF units, room prior, and
  segmentation head.
- `configs/s3dis_seg.yaml` — retained Area-5 test configuration.
- `train_s3dis.py` — training entry point.
- `eval_s3dis.py` — aggregated room-level evaluation entry point.

## Data layout

Set `data_dir` in `configs/s3dis_seg.yaml`.  The loader accepts either of the
following preprocessed layouts:

```text
data/s3dis/
  Area_1/*.npy
  ...
  Area_6/*.npy
```

or:

```text
data/s3dis/raw/
  Area_1_*.npy
  ...
```

Each room array must have columns `[x, y, z, r, g, b, semantic_label]`.
With the default split, Area 5 is held out for evaluation.

## Run order

From this directory:

```bash
python train_s3dis.py --config configs/s3dis_seg.yaml
```

After training, evaluate a checkpoint with the configuration's test-area
settings:

```bash
python eval_s3dis.py \
  --config configs/s3dis_seg.yaml \
  --checkpoint checkpoints/s3dis_best.pt
```

`scripts/run_s3dis.sh` is a SLURM wrapper for the training command.  Generated
checkpoints, logs, room-summary caches, and result files are ignored by Git.

The retained source configuration differs from parts of the manuscript's
detailed recipe; see `../../docs/PAPER_CODE_CROSSCHECK.md` before using it as
an exact reproduction command.
