# Repository Scope

This is an anonymous, code-only reproducibility archive.  Its public scope is
intentionally limited to the following experiment families:

| Experiment | Included code |
| --- | --- |
| ModelNet10 / ModelNet40 | Point-cloud classification model, ModelNet loader, synthetic sanity checks, configurations, and verification utilities. |
| ShapeNetPart | Part-segmentation model, loader, slicing utilities, configurations, training, and evaluation. |
| S3DIS | Indoor semantic-segmentation model, loader, slicing utilities, configurations, training, and evaluation. |
| Foveated ImageNet-100 | Foveated image model, ImageNet-style loader, configuration, and training entry point. |

The archive does not contain source code, configurations, results, or datasets
for unrelated benchmarks.  In particular, the benchmark that the manuscript
states was not evaluated is excluded entirely.

## Intentionally excluded artifacts

- Manuscript PDFs and local review material.
- Dataset downloads, extracted dataset files, and preprocessing products.
- Model weights, checkpoints, logs, tensorboard runs, cached Python files, and
  generated result files.
- Historical repository metadata, source URLs, author information, and commit
  history.

The top-level `.gitignore` enforces these exclusions for normal development.
Run `python tools/validate_repository.py` before packaging or publishing a new
revision.
