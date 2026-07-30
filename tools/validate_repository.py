"""Repository-level pre-publication checks for this anonymous archive.

This check is intentionally lightweight: it validates the repository layout,
parses source/configuration files, and rejects artifacts that should never be
committed.  It does not download datasets or execute long training jobs.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "README.md",
    "artifacts/README.md",
    "artifacts/datasets/manifest.yaml",
    "artifacts/source_diagnostics/README.md",
    "artifacts/source_diagnostics/configs/modelnet10.yaml",
    "artifacts/source_diagnostics/configs/modelnet40.yaml",
    "docs/REPOSITORY_SCOPE.md",
    "docs/PAPER_CODE_CROSSCHECK.md",
    "experiments/modelnet/README.md",
    "experiments/shapenetpart/README.md",
    "experiments/s3dis/README.md",
    "experiments/foveated_imagenet100/README.md",
    "tools/validate_diagnostic_weights.py",
)
IGNORED_DIR_NAMES = {
    ".git", ".pytest_cache", ".mypy_cache", "__pycache__", "AAAI",
}
FORBIDDEN_PATH_TOKENS = {
    "scan" + "object",
    "ci" + "far",
    "n" + "mnist",
    "n-" + "mnist",
}
FORBIDDEN_TEXT_TOKENS = FORBIDDEN_PATH_TOKENS | {
    "clau" + "de",
    "anthro" + "pic",
}
PUBLIC_TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".pdf", ".pt", ".pth", ".ckpt", ".pyc", ".pyo", ".h5", ".hdf5",
    ".npy", ".npz", ".zip", ".tar", ".gz",
}
DIAGNOSTIC_WEIGHT_ROOT = Path("artifacts/source_diagnostics/weights")
DIAGNOSTIC_RESULT_ROOT = Path("artifacts/source_diagnostics/results")
EXPECTED_DIAGNOSTIC_COUNTS = {
    "modelnet10": {"weights": 25, "json": 50, "csv": 5},
    "modelnet40": {"weights": 25, "json": 50, "csv": 5},
}


def iter_public_files(root: Path):
    """Yield public files while excluding Git internals and transient caches."""
    for path in root.rglob("*"):
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.is_file():
            yield path


def check_layout(errors: list[str]) -> None:
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def check_artifacts(errors: list[str]) -> None:
    for path in iter_public_files(ROOT):
        relative = path.relative_to(ROOT)
        lowered = str(relative).replace("\\", "/").lower()
        is_curated_weight = (
            path.suffix.lower() == ".pt"
            and relative.is_relative_to(DIAGNOSTIC_WEIGHT_ROOT)
            and path.name == "model.pt"
        )
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not is_curated_weight:
            errors.append(f"forbidden generated/data artifact: {relative}")
        if is_curated_weight and path.stat().st_size > 1_000_000:
            errors.append(f"unexpectedly large diagnostic checkpoint: {relative}")
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            errors.append(f"out-of-scope benchmark path: {relative}")
        if ".git" in relative.parts:
            errors.append(f"nested Git metadata: {relative}")


def check_public_text(errors: list[str]) -> None:
    for path in iter_public_files(ROOT):
        if path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        try:
            content = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"cannot inspect public text {relative}: {exc}")
            continue
        if any(token in content for token in FORBIDDEN_TEXT_TOKENS):
            errors.append(f"excluded benchmark or assistant marker in {relative}")


def check_diagnostics(errors: list[str]) -> None:
    for dataset, expected in EXPECTED_DIAGNOSTIC_COUNTS.items():
        weight_root = ROOT / DIAGNOSTIC_WEIGHT_ROOT / dataset
        result_root = ROOT / DIAGNOSTIC_RESULT_ROOT / dataset
        weights = list(weight_root.rglob("model.pt")) if weight_root.is_dir() else []
        json_files = list(result_root.rglob("*.json")) if result_root.is_dir() else []
        csv_files = list(result_root.rglob("*.csv")) if result_root.is_dir() else []
        observed = {
            "weights": len(weights),
            "json": len(json_files),
            "csv": len(csv_files),
        }
        for kind, count in expected.items():
            if observed[kind] != count:
                errors.append(
                    f"{dataset} diagnostic {kind} count is "
                    f"{observed[kind]}; expected {count}"
                )

        summaries = list(result_root.rglob("summary.json"))
        histories = list(result_root.rglob("history.json"))
        if len(summaries) != expected["weights"]:
            errors.append(
                f"{dataset} summary count is {len(summaries)}; "
                f"expected {expected['weights']}"
            )
        if len(histories) != expected["weights"]:
            errors.append(
                f"{dataset} history count is {len(histories)}; "
                f"expected {expected['weights']}"
            )
        for path in summaries:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("dataset") != dataset:
                errors.append(
                    f"diagnostic dataset mismatch in {path.relative_to(ROOT)}"
                )
            if payload.get("seed") != 0:
                errors.append(
                    f"unexpected diagnostic seed in {path.relative_to(ROOT)}"
                )


def check_syntax(errors: list[str]) -> None:
    yaml = None
    try:
        import yaml as yaml_module
        yaml = yaml_module
    except ImportError:
        errors.append("PyYAML is not installed; cannot validate YAML files")

    for path in iter_public_files(ROOT):
        relative = path.relative_to(ROOT)
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
            elif yaml is not None and path.suffix in {".yaml", ".yml"}:
                with path.open("r", encoding="utf-8") as handle:
                    yaml.safe_load(handle)
            elif path.suffix == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    json.load(handle)
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        raise ValueError("CSV has no header")
                    list(reader)
        except (
            OSError, SyntaxError, UnicodeDecodeError, ValueError,
            json.JSONDecodeError, csv.Error,
        ) as exc:
            errors.append(f"invalid {relative}: {exc}")


def main() -> int:
    errors: list[str] = []
    check_layout(errors)
    check_artifacts(errors)
    check_public_text(errors)
    check_diagnostics(errors)
    check_syntax(errors)

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
