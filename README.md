# ASP-SNN: Anonymous Reproducibility Archive

This repository is an anonymous code archive for the experiments evaluated in
the accompanying submission.  It contains source code and configuration files
only: datasets, trained weights, generated logs, and manuscript files are not
included.

## Included experiment folders

| Folder | Task | Primary entry point |
| --- | --- | --- |
| `experiments/modelnet` | ModelNet10 / ModelNet40 point-cloud classification | `experiments/run.py` |
| `experiments/shapenetpart` | ShapeNetPart point-cloud part segmentation | `train_shapenet.py` |
| `experiments/s3dis` | S3DIS indoor semantic segmentation | `train_s3dis.py` |
| `experiments/foveated_imagenet100` | Foveated ImageNet-100 classification | `train_imagenet_foveater.py` |

Each experiment folder has a focused README that documents its data layout,
configuration file, training command, and evaluation command.

## Setup

Python 3.10+ and a CUDA-compatible PyTorch installation are recommended.

```bash
conda env create -f environment.yml
conda activate asp-snn
```

For an existing Python environment, install PyTorch appropriate for the local
CUDA runtime first, then run:

```bash
pip install -r requirements.txt
```

## Recommended order

1. Run the repository check: `python tools/validate_repository.py`.
2. Read the README in the experiment folder of interest.
3. Prepare that experiment's dataset in the layout described there.
4. Train using its provided configuration.
5. Evaluate with the corresponding evaluation entry point.

## Scope and paper cross-check

The supported public scope is restricted to the four experiment families listed
above.  The repository deliberately excludes datasets and code paths outside
that scope.  See [Repository Scope](docs/REPOSITORY_SCOPE.md) and
[Paper/Code Cross-check](docs/PAPER_CODE_CROSSCHECK.md) for the inclusions,
exclusions, and known configuration limitations of the retained snapshots.

## Reproducibility note

The source snapshots are preserved faithfully enough to inspect and execute the
available workflows, but some published training recipes are only partially
encoded in the retained configurations.  This repository does not claim that a
default command reproduces every numerical result in the submission; the
cross-check documents those gaps explicitly.
