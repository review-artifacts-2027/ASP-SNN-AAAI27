# ShapeNetPart Segmentation

This folder contains the point-cloud part-segmentation workflow.

## Layout

- `datasets/shapenetpart.py` — HDF5 dataset loader and preprocessing.
- `datasets/slicing.py` — coarse and fine point-slice construction.
- `models/` — encoder, spiking policy, LIF units, energy helper, and
  segmentation head.
- `configs/shapenet_seg.yaml` — default retained configuration.
- `train_shapenet.py` — training entry point.
- `eval_shapenet.py` — checkpoint evaluation entry point.

## Data layout

Set `data_dir` in `configs/shapenet_seg.yaml` to an extracted ShapeNetPart HDF5
directory.  The retained default is:

```text
data/shapenet_part_seg_hdf5_data/
  ply_data_train*.h5
  ply_data_val*.h5
  ply_data_test*.h5
```

The training loader validates the expected HDF5 fields when it starts.

## Run order

From this directory:

```bash
python train_shapenet.py --config configs/shapenet_seg.yaml
```

The program writes checkpoints and CSV logs to the configured `ckpt_dir` and
`log_dir`; both are ignored by Git.  To evaluate a trained checkpoint, use:

```bash
python eval_shapenet.py \
  --config configs/shapenet_seg.yaml \
  --ckpt checkpoints/shapenet_best.pt
```

For a cluster environment, `scripts/run_shapenet.sh` wraps the same command in
a SLURM job script.  Adjust the resource directives and environment activation
to suit the local scheduler.

The retained configuration is documented as a source snapshot.  Review
`../../docs/PAPER_CODE_CROSSCHECK.md` for known differences from the detailed
manuscript recipe.
