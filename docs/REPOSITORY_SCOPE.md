# Repository Scope

This is an anonymous reproducibility archive. Its public scope is intentionally
limited to the following experiment families and approved-source artifacts:

| Experiment | Included code |
| --- | --- |
| ModelNet10 / ModelNet40 | Point-cloud classification model, ModelNet loader, synthetic sanity checks, configurations, verification utilities, and clearly labeled source-diagnostic weights/results. |
| ShapeNetPart | Part-segmentation model, loader, slicing utilities, configurations, training, and evaluation. |
| S3DIS | Indoor semantic-segmentation model, loader, slicing utilities, configurations, training, and evaluation. |
| Foveated ImageNet-100 | Foveated image model, ImageNet-style loader, configuration, and training entry point. |

The archive does not contain source code, configurations, results, or datasets
for unrelated benchmarks.  In particular, the benchmark that the manuscript
states was not evaluated is excluded entirely.

## Included artifact boundary

The artifact bundle contains 50 seed-0 ModelNet diagnostic checkpoints and
their 100 JSON and 10 CSV output files. They are isolated under
`artifacts/source_diagnostics/` because their 30-epoch configuration and
recorded accuracy do not match the manuscript recipe or headline results.

No dataset payload was present in the approved source snapshots. The central
dataset manifest therefore records expected layouts without redistributing
third-party data.

## Intentionally excluded artifacts

- Manuscript PDFs and local review material.
- Dataset downloads, extracted dataset files, and preprocessing products.
- ShapeNetPart, S3DIS, and ImageNet-100 weights/results, because none were
  present in the approved snapshots.
- Unrelated benchmark weights/results, training logs, tensorboard runs, cached
  Python files, and uncurated generated files.
- Historical repository metadata, identity-bearing project URLs, author
  information, and commit history.

The top-level `.gitignore` enforces these exclusions for normal development.
Run `python tools/validate_repository.py` before packaging or publishing a new
revision.
