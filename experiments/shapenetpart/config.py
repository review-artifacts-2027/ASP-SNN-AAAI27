"""
config.py — Unified configuration loader.

Loads YAML config files and provides attribute-style access. Supports
command-line overrides via --set key=value syntax. All values are
auto-cast to int/float/bool where possible.

IMPORTANT: PyYAML on some platforms/versions reads scientific notation
like '4e-4' as a string instead of float 0.0004. The _auto_cast()
function handles this by attempting float() conversion on all string
values after YAML loading.
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml


def _auto_cast(v):
    """
    Auto-cast a value to its most specific Python type.

    Handles:
        '42'     -> int 42
        '4e-4'   -> float 0.0004
        '0.001'  -> float 0.001
        'true'   -> bool True
        'false'  -> bool False
        'hello'  -> str 'hello'  (unchanged)

    Also handles lists by recursively casting elements.
    """
    if isinstance(v, str):
        # Bool
        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        if v.lower() == 'none':
            return None
        # Int (only if no decimal point or scientific notation)
        if 'e' not in v.lower() and '.' not in v:
            try:
                return int(v)
            except ValueError:
                pass
        # Float (handles '4e-4', '1e-3', '0.001', '5e-5', etc.)
        try:
            return float(v)
        except ValueError:
            pass
    elif isinstance(v, list):
        return [_auto_cast(item) for item in v]
    return v


class Config:
    """Attribute-access wrapper around a dictionary."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, _auto_cast(v))

    def __repr__(self):
        items = ", ".join(f"{k}={v}" for k, v in self.__dict__.items()
                         if not k.startswith("_"))
        return f"Config({items})"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_")}


def load_config(yaml_path: str, extra_overrides: dict = None) -> Config:
    """Load a YAML config file and apply optional overrides."""
    with open(yaml_path) as f:
        cfg_dict = yaml.safe_load(f) or {}

    if extra_overrides:
        for k, v in extra_overrides.items():
            if v is not None:
                cfg_dict[k] = v

    # Ensure output directories exist
    os.makedirs(cfg_dict.get("ckpt_dir", "checkpoints"), exist_ok=True)
    os.makedirs(cfg_dict.get("log_dir", "logs"), exist_ok=True)

    # Device — respect LOCAL_RANK for SLURM multi-GPU nodes
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        cfg_dict["device"] = torch.device(f"cuda:{local_rank}")
    else:
        cfg_dict["device"] = torch.device("cpu")

    return Config(cfg_dict)


def set_seed(seed: int = 42):
    """Reproducibility seed for all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # P2 OPTIMIZE: Enable TF32 on H100/A100 for ~10% free speedup with
    # negligible accuracy impact (matmul TF32 has ~1e-3 relative error,
    # well below the noise floor of our SNN gradients).
    # Reference: NVIDIA H100 Tensor Core documentation.
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def base_argparser(description: str = "ASP-SNN") -> argparse.ArgumentParser:
    """Shared argument parser used by all training/eval scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, required=False,
                   help="Path to YAML config file")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint for resuming training")
    p.add_argument("--set", nargs="*", default=[],
                   help="Override config values: --set lr=1e-3 epochs=100")
    return p


def parse_overrides(args) -> dict:
    """Parse --set key=value pairs into a dict with auto type casting."""
    overrides = {}
    for item in getattr(args, "set", []):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        overrides[k] = _auto_cast(v)
    return overrides