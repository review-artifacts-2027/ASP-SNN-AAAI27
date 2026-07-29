"""Validate the anonymous code supplement without importing project modules."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    "requirements.txt",
    "environment.yml",
    "experiments/modelnet/train_modelnet.py",
    "experiments/full_scale/smoke_test.py",
    "experiments/rigor_suite/tests/test_suite.py",
    "experiments/rigor_suite/verification/run_all.py",
    "docs/VALIDATION_REPORT.md",
    "results/modelnet/PAPER_REPRODUCTION.md",
)

FORBIDDEN_DIRS = {".git", "__pycache__", ".pytest_cache", ".idea", ".vscode"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".ckpt"}
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".log",
    ".tex",
    ".txt",
}

# These generic patterns detect common double-blind leaks without embedding
# any private names in the distributed checker.
_AI_TERMS = ("clau" + "de", "anth" + "ropic", "co-" + "authored-by")

IDENTITY_PATTERNS = {
    "personal email": re.compile(
        r"\b[\w.+-]+@(?:gmail|outlook|hotmail|icloud|protonmail)\.[a-z]{2,}\b",
        re.IGNORECASE,
    ),
    "GitHub account URL": re.compile(r"https?://(?:www\.)?github\.com/[^/\s]+/", re.IGNORECASE),
    "AI attribution": re.compile(
        r"\b(?:" + "|".join(re.escape(term) for term in _AI_TERMS) + r")\b",
        re.IGNORECASE,
    ),
    "user-specific path": re.compile(
        r"(?:[A-Za-z]:\\Users\\[^\\\s]+|/home/[^/\s]+|/Users/[^/\s]+)",
        re.IGNORECASE,
    ),
}


def iter_files():
    for path in ROOT.rglob("*"):
        if path.is_file():
            yield path


def validate_layout(errors: list[str]) -> None:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            errors.append(f"forbidden directory: {relative}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/cache file: {relative}")
        if path.is_file() and path.stat().st_size > 25 * 1024 * 1024:
            errors.append(f"file exceeds 25 MiB: {relative}")


def validate_python(errors: list[str]) -> int:
    checked = 0
    for path in ROOT.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            checked += 1
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"Python validation failed for {path.relative_to(ROOT)}: {exc}")
    return checked


def validate_yaml(errors: list[str]) -> int:
    try:
        import yaml
    except ImportError:
        errors.append("PyYAML is unavailable; install requirements.txt before validation")
        return 0

    checked = 0
    for path in (*ROOT.rglob("*.yaml"), *ROOT.rglob("*.yml")):
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            checked += 1
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            errors.append(f"YAML validation failed for {path.relative_to(ROOT)}: {exc}")
    return checked


def validate_anonymity(errors: list[str]) -> int:
    checked = 0
    for path in iter_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"cannot read text file {path.relative_to(ROOT)}: {exc}")
            continue
        checked += 1
        for label, pattern in IDENTITY_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} found in {path.relative_to(ROOT)}")
    return checked


def main() -> int:
    errors: list[str] = []
    validate_layout(errors)
    python_count = validate_python(errors)
    yaml_count = validate_yaml(errors)
    text_count = validate_anonymity(errors)

    if errors:
        print("PACKAGE VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PACKAGE VALIDATION PASSED")
    print(f"- Python files parsed: {python_count}")
    print(f"- YAML files parsed:   {yaml_count}")
    print(f"- Text files scanned:  {text_count}")
    print("- No Git metadata, checkpoints, caches, identity leaks, or oversized files found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
