"""Repository-level pre-publication checks for this anonymous archive.

This check is intentionally lightweight: it validates the repository layout,
parses source/configuration files, and rejects artifacts that should never be
committed.  It does not download datasets or execute long training jobs.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "README.md",
    "docs/REPOSITORY_SCOPE.md",
    "docs/PAPER_CODE_CROSSCHECK.md",
    "experiments/modelnet/README.md",
    "experiments/shapenetpart/README.md",
    "experiments/s3dis/README.md",
    "experiments/foveated_imagenet100/README.md",
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
FORBIDDEN_SUFFIXES = {
    ".pdf", ".pt", ".pth", ".ckpt", ".pyc", ".pyo", ".h5", ".hdf5",
    ".npy", ".npz", ".zip", ".tar", ".gz",
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
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated/data artifact: {relative}")
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            errors.append(f"out-of-scope benchmark path: {relative}")
        if ".git" in relative.parts:
            errors.append(f"nested Git metadata: {relative}")


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
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
            errors.append(f"invalid {relative}: {exc}")


def main() -> int:
    errors: list[str] = []
    check_layout(errors)
    check_artifacts(errors)
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
