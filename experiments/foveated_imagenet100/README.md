# Foveated ImageNet-100

This folder contains the retained foveated image-classification model and
training entry point.

## Layout

- `datasets/imagenet.py` — ImageFolder-style loader and transforms.
- `models/foveater_asp.py` — foveated ASP model.
- `configs/imagenet_foveater.yaml` — retained configuration.
- `train_imagenet_foveater.py` — training and smoke-test entry point.

## Data layout

The loader uses an ImageFolder-style tree.  For ImageNet-100, point
`data_dir` to a directory with exactly the intended 100 class folders:

```text
data/imagenet100/
  train/
    class_a/*.JPEG
    ...
  val/
    class_a/*.JPEG
    ...
```

## Run order

Run a dataset-free smoke test first:

```bash
python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set smoke=true epochs=1 batch_size=2 num_workers=0
```

Then point the configuration at the ImageNet-100 directory and set its class
count before training:

```bash
python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set data_dir=data/imagenet100 num_classes=100
```

The `scripts/run_imagenet_foveater.sh` wrapper reads `DATA_DIR`,
`NUM_CLASSES`, and `EPOCHS` environment variables.  Logs and checkpoints are
Git-ignored.

This is a partial retained implementation.  See
`../../docs/PAPER_CODE_CROSSCHECK.md` for the missing pieces of the detailed
paper recipe.
