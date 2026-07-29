# Active Spiking Perception: Anonymous Code Supplement

This archive accompanies the anonymous submission "Active Spiking Perception
for Anytime 3D Point Cloud Recognition." It is a review snapshot: source code,
configs, executable verification checks, and compact result records are
included; datasets, checkpoints, caches, and version-control history are not.

## Start here

Run these commands from the archive root.

```bash
conda env create -f environment.yml
conda activate asp-snn
python tools/validate_package.py
```

For a pip environment, install a PyTorch build appropriate for the machine,
then run `pip install -r requirements.txt`.

The fastest no-data review sequence is:

```bash
# 1. Full-scale implementation: eight CPU smoke checks.
cd experiments/full_scale
python smoke_test.py
cd ../..

# 2. Paper ModelNet implementation: small CPU forward pass.
python experiments/modelnet/train_modelnet.py \
  --smoke --no_compile --points 32 --groups 8 --group_size 4 \
  --dim 32 --depth 1 --timestep 1 --asp_steps 4

# 3. Ablation/verification implementation: unit checks.
cd experiments/rigor_suite
python tests/test_suite.py
python verification/run_all.py
cd ../..
```

The first two checks take no command-line dataset or network access.
`verification/run_all.py` is CPU-runnable but takes longer because it executes
all seven mechanism checks. Formal failures produce a nonzero exit code;
empirical comparisons outside a theorem's assumptions are labeled explicitly
as observed or not observed diagnostics.

## What each folder contains

```text
experiments/
  modelnet/
    train_modelnet.py       paper ModelNet10/40 SPM + ASP + KD pipeline
  full_scale/
    models/                 classifier, segmentor, LIF, SSP, FoveaTer
    datasets/               ShapeNetPart, ScanObjectNN, S3DIS, ImageNet
    configs/                one YAML file per full-scale task
    train_*.py              task-specific training entrypoints
    eval_*.py               task-specific evaluation entrypoints
    smoke_test.py           eight no-data implementation checks
    scripts/                local and Slurm launch helpers
  rigor_suite/
    asp/                    compact ASP/SSP implementation
    configs/base/           ModelNet10/40, CIFAR-10/100, ScanObjectNN
    configs/ablations/      A1-A5 ablations
    configs/strengthening/  S1-S6 stress tests
    experiments/            unified runner and statistics aggregator
    verification/           V1-V7 theorem/mechanism checks
    tests/                  implementation tests
results/modelnet/           compact JSON histories, metrics, and run logs
docs/VALIDATION_REPORT.md   formal-check and diagnostic outcomes
tools/validate_package.py   syntax, YAML, anonymity, and archive checks
```

There are two intentionally separate implementations. `modelnet/` is the
self-contained pipeline used for the paper's four-chunk ModelNet experiment.
`rigor_suite/` is the smaller 16-slice implementation used for controlled
ablations and mechanism verification. Keeping them separate prevents config
or import collisions and makes the claim-to-code mapping explicit.

## Dataset and entrypoint map

| Dataset or check | Entrypoint | Data behavior |
|---|---|---|
| ModelNet10/40 paper run | `experiments/modelnet/train_modelnet.py` | Uses an existing `--data_dir` or downloads through `kagglehub` |
| ShapeNetPart | `experiments/full_scale/train_shapenet.py` | Expects the path in `configs/shapenet_seg.yaml` |
| ScanObjectNN | `experiments/full_scale/train_scanobj.py` | Expects PB-T50-RS under the configured path |
| S3DIS | `experiments/full_scale/train_s3dis.py` | Uses the Area-5 protocol in the config |
| ImageNet/ImageNet-100 | `experiments/full_scale/train_imagenet_foveater.py` | Expects an ImageFolder-style directory |
| ModelNet/CIFAR ablations | `experiments/rigor_suite/experiments/run.py` | CIFAR can download through `torchvision`; point-cloud data is local |
| Theory mechanisms | `experiments/rigor_suite/verification/run_all.py` | Synthetic, no external dataset |

N-MNIST is not included because no N-MNIST implementation was present in the
source snapshot used to build this archive. This archive therefore makes no
N-MNIST reproducibility claim.

## Reproduce the ModelNet runs

Run the two datasets separately so checkpoints and logs remain unambiguous:

```bash
python experiments/modelnet/train_modelnet.py \
  --datasets ModelNet10 --epochs 300 --seed 42 \
  --data_dir data --ckpt_dir outputs/modelnet10

python experiments/modelnet/train_modelnet.py \
  --datasets ModelNet40 --epochs 300 --seed 42 \
  --data_dir data --ckpt_dir outputs/modelnet40
```

The defaults match the recorded reproduction runs: 1,024 points, dimension
384, depth 12, 128 groups, four ASP chunks, batch size 64, BF16 when available,
and knowledge distillation. Full training requires a CUDA GPU. Existing
metrics and the exact logged configuration are in `results/modelnet/`.

## Run the ablations and strengthening experiments

Run these commands from `experiments/rigor_suite`:

```bash
python -m experiments.run \
  --base configs/base/modelnet40.yaml \
  --exp configs/ablations/A1_theta.yaml

python -m experiments.run \
  --base configs/base/cifar100.yaml \
  --exp configs/ablations/A5_policy.yaml

python -m experiments.run \
  --base configs/base/scanobjectnn.yaml \
  --exp configs/strengthening/S1_occlusion.yaml

python -m experiments.aggregate \
  --dir results/A5_policy/modelnet40 --baseline ssp
```

Use `--seeds 0 1 2 3 4` for the configured five-seed protocol. The synthetic
base config is useful for quick end-to-end checks; synthetic results are
mechanism checks, not benchmark claims.

## Run the full-scale tasks

Run these commands from `experiments/full_scale`. Every training script accepts
`--config`, `--resume`, and YAML overrides through `--set`.

```bash
python train_shapenet.py --config configs/shapenet_seg.yaml
python train_scanobj.py --config configs/scanobj_cls.yaml
python train_s3dis.py --config configs/s3dis_seg.yaml
python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml
```

Evaluation:

```bash
python eval_shapenet.py --ckpt checkpoints/shapenet_best.pt --per_cat
python eval_scanobj.py --ckpt checkpoints/scanobj_best.pt --n_votes 10
python eval_s3dis.py --ckpt checkpoints/s3dis_best.pt --per_class
```

The shell files under `experiments/full_scale/scripts/` provide matching smoke,
full-run, parallel-GPU, and Slurm examples.

## Outputs and reproducibility notes

- Training writes checkpoints and logs to the paths in the selected config.
- ModelNet writes `final_results.json` and `histories.json` beside checkpoints.
- The included JSON files are small enough for review; model weights are
  excluded from the submission archive.
- The recorded ModelNet reproduction uses a single seed and a different point
  sampling source from the reported run. Read
  `results/modelnet/PAPER_REPRODUCTION.md` before comparing exact percentages.
- Analytical firing-rate and chunk-count measures are reported separately;
  they are not hardware energy measurements.

## Anonymous-review safeguards

The package contains no Git history, contributor metadata, author names,
affiliations, personal email addresses, public repository links, or AI
co-author trailers. Do not add identifying metadata or external mutable links
until double-blind review is complete.
