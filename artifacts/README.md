# Artifact Inventory

This directory separates dataset requirements, generated results, and trained
weights from the implementation under `experiments/`.

| Experiment family | Raw dataset included | Weights/results included |
| --- | --- | --- |
| ModelNet10 / ModelNet40 | No | Yes, source diagnostics only |
| ShapeNetPart | No | No approved-source artifact was available |
| S3DIS | No | No approved-source artifact was available |
| Foveated ImageNet-100 | No | No approved-source artifact was available |

Raw datasets are not redistributed because no dataset payload or redistribution
grant was present in the approved source snapshots. See `datasets/` for the
machine-readable manifest and setup-document links.

The ModelNet bundle under `source_diagnostics/` contains every available
in-scope checkpoint and result file from the approved snapshots. Read its
warning before interpreting the files.
