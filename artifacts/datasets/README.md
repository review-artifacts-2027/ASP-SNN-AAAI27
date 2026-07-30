# Dataset Preparation

No raw or preprocessed dataset payload is bundled. Prepare each dataset in the
layout documented by its experiment:

| Dataset | Setup and expected layout |
| --- | --- |
| ModelNet10 / ModelNet40 | `../../experiments/modelnet/README.md` |
| ShapeNetPart | `../../experiments/shapenetpart/README.md` |
| S3DIS | `../../experiments/s3dis/README.md` |
| ImageNet-100 subset | `../../experiments/foveated_imagenet100/README.md` |

Keep downloaded data outside version control. The top-level `.gitignore`
excludes common dataset roots and binary formats.
