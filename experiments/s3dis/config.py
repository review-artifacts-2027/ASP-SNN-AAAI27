import argparse
import os
import random

import numpy as np
import torch
import yaml


def _auto_cast(v):
    if isinstance(v, str):
        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        if v.lower() == 'none':
            return None

        if 'e' not in v.lower() and '.' not in v:
            try:
                return int(v)
            except ValueError:
                pass

        try:
            return float(v)
        except ValueError:
            pass
    elif isinstance(v, list):
        return [_auto_cast(item) for item in v]
    return v


class Config:
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
    with open(yaml_path) as f:
        cfg_dict = yaml.safe_load(f) or {}

    if extra_overrides:
        for k, v in extra_overrides.items():
            if v is not None:
                cfg_dict[k] = v

    os.makedirs(cfg_dict.get("ckpt_dir", "checkpoints"), exist_ok=True)
    os.makedirs(cfg_dict.get("log_dir", "logs"), exist_ok=True)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        cfg_dict["device"] = torch.device(f"cuda:{local_rank}")
    else:
        cfg_dict["device"] = torch.device("cpu")

    return Config(cfg_dict)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def base_argparser(description: str = "ASP-SNN") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, required=False,
                   help="Path to YAML config file")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint for resuming training")
    p.add_argument("--set", nargs="*", default=[],
                   help="Override config values: --set lr=1e-3 epochs=100")
    return p


def parse_overrides(args) -> dict:
    overrides = {}
    for item in getattr(args, "set", []):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        overrides[k] = _auto_cast(v)
    return overrides
