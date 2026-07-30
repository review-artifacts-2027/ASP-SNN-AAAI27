from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELNET_ROOT = ROOT / "experiments" / "modelnet"
CONFIG_ROOT = ROOT / "artifacts" / "source_diagnostics" / "configs"
WEIGHT_ROOT = ROOT / "artifacts" / "source_diagnostics" / "weights"
EXPECTED_COUNTS = {"modelnet10": 25, "modelnet40": 25}

sys.path.insert(0, str(MODELNET_ROOT))

from asp.model import ASPConfig, ASPModel

VARIANT_OVERRIDES = {
    "A1_theta": {
        "baseline": {},
    },
    "A2_masking": {
        "mask_on": {"use_mask": True},
        "mask_off": {"use_mask": False},
    },
    "A3_geometry": {
        "full": {"drop_desc": []},
        "no_centroid": {"drop_desc": ["cx", "cy", "cz"]},
        "no_cx": {"drop_desc": ["cx"]},
        "no_cy": {"drop_desc": ["cy"]},
        "no_cz": {"drop_desc": ["cz"]},
        "no_dist": {"drop_desc": ["dist"]},
        "no_spread": {"drop_desc": ["spread"]},
        "no_count": {"drop_desc": ["count"]},
        "only_spread": {"drop_desc": ["cx", "cy", "cz", "dist", "count"]},
    },
    "A4_dssp": {
        "d2": {"d_ssp": 2},
        "d4": {"d_ssp": 4},
        "d6": {"d_ssp": 6},
        "d8": {"d_ssp": 8},
        "d16": {"d_ssp": 16},
        "d32": {"d_ssp": 32},
        "d64": {"d_ssp": 64},
        "d64_rank8": {"d_ssp": 64, "ssp_rank": 8},
        "d128": {"d_ssp": 128},
    },
    "A5_policy": {
        "ssp": {"policy": "ssp"},
        "random": {"policy": "random"},
        "fixed": {"policy": "fixed"},
        "geometry_only": {"policy": "geometry_only"},
    },
}


def load_base_config(dataset: str) -> dict:
    path = CONFIG_ROOT / f"{dataset}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_checkpoint(path: Path) -> None:
    relative = path.relative_to(WEIGHT_ROOT)
    if len(relative.parts) != 5:
        raise ValueError(f"unexpected checkpoint layout: {relative}")
    dataset, experiment, seed, variant, filename = relative.parts
    if seed != "seed0" or filename != "model.pt":
        raise ValueError(f"unexpected checkpoint path: {relative}")
    try:
        override = VARIANT_OVERRIDES[experiment][variant]
    except KeyError as exc:
        raise ValueError(f"unknown diagnostic variant: {relative}") from exc

    config = load_base_config(dataset)
    config.update(override)
    model = ASPModel(ASPConfig.from_dict(config))
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dictionary: {relative}")
    model.load_state_dict(state, strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly load every bundled ModelNet diagnostic checkpoint."
    )
    parser.parse_args()

    errors: list[str] = []
    loaded = 0
    for dataset, expected in EXPECTED_COUNTS.items():
        paths = sorted((WEIGHT_ROOT / dataset).rglob("model.pt"))
        if len(paths) != expected:
            errors.append(
                f"{dataset}: found {len(paths)} checkpoints; expected {expected}"
            )
        for path in paths:
            try:
                validate_checkpoint(path)
                loaded += 1
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                errors.append(str(exc))

    if errors:
        print("Diagnostic checkpoint validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Validated {loaded} diagnostic checkpoints.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
