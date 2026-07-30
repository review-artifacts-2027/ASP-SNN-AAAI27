# Paper/Code Cross-check

This document records the relationship between the retained code snapshots and
the anonymous manuscript.  It is deliberately conservative: an implementation
is marked as incomplete whenever the available configuration does not encode
the complete training recipe described in the paper or supplement.

| Experiment | Manuscript claim | Retained implementation | Review status |
| --- | --- | --- | --- |
| ModelNet10 / ModelNet40 | Point-cloud classification results with the reported full training recipe. | The retained classifier, ModelNet adapter, configs, and verification tools are runnable.  The default snapshot configuration uses a smaller setting and shorter schedule than the manuscript's full recipe. | Not an exact default reproduction. |
| ShapeNetPart | Part-segmentation results. | The model, data pipeline, train/evaluation programs, and segmentation configuration are retained.  The available configuration uses a different simulation horizon from the detailed supplement. | Not an exact default reproduction. |
| S3DIS | Indoor semantic-segmentation results. | The model, data pipeline, train/evaluation programs, and S3DIS configurations are retained.  The available training file uses settings that differ from the teacher and crop/horizon recipe described in the supplement. | Not an exact default reproduction. |
| Foveated ImageNet-100 | Foveated classification and compute/accuracy comparisons. | The foveated model, ImageNet-style loader, configuration, and training entry point are retained.  The snapshot does not encode every augmentation, control, or optimization component listed in the supplement. | Partial implementation only. |

## What this means for reviewers

The archive is suitable for reading the retained implementations, inspecting
the available configurations, and running smoke tests or source-level
experiments after datasets are supplied.  It should not be interpreted as a
claim that a single default command reproduces every number, table, or energy
comparison in the manuscript.

No trained checkpoints or result tables are bundled.  Where a manuscript
training recipe has a component absent from the retained snapshot, that absence
is documented here rather than silently filled with a substitute
implementation.

## Exclusions required by manuscript scope

The manuscript explicitly identifies one benchmark as not evaluated.  Its code
path, configuration, data adapter, and result artifacts are therefore not part
of this archive.  Other non-manuscript benchmark families are similarly
excluded.

## Configuration references

- ModelNet: `experiments/modelnet/configs/`
- ShapeNetPart: `experiments/shapenetpart/configs/shapenet_seg.yaml`
- S3DIS: `experiments/s3dis/configs/s3dis_seg.yaml`
- Foveated ImageNet-100:
  `experiments/foveated_imagenet100/configs/imagenet_foveater.yaml`

The configuration files are preserved as available source artifacts, not
retroactively rewritten to imply unsupported result reproduction.
