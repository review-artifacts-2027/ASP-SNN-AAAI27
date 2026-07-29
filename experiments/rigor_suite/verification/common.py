"""Shared harness for verification scripts V1-V7.

Trains (once, cached) a tiny ASP model on the synthetic dataset so every
theoretical claim is tested against a real trained policy on CPU in minutes.
Scale-up: point --base at any benchmark config and rerun the same scripts.
"""
from __future__ import annotations

import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asp.eval import collect, evaluate            # noqa: E402
from asp.model import ASPConfig, ASPModel         # noqa: E402
from asp.train import train_model                 # noqa: E402
from asp.datasets import SyntheticPointDataset    # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "verification")
os.makedirs(RESULTS, exist_ok=True)

TINY = dict(dataset="synthetic", num_classes=8, modality="points", d_model=64,
            enc_hidden=48, k_slices=16, points_per_slice=24, n_points=192,
            d_ssp=64, ssp_rank=0, policy="ssp", use_mask=True, theta=0.7,
            epochs=int(os.environ.get("ASP_VERIF_EPOCHS", 12)), lr=2e-3,
            batch_size=32, lambda_exit=0.1, lambda_sparse=0.01,
            n_per_class_train=60, n_per_class_test=24, eval_every=100)


def loaders(cfg):
    from torch.utils.data import DataLoader
    tr = SyntheticPointDataset(cfg["n_per_class_train"], cfg["n_points"],
                               cfg["k_slices"], cfg["points_per_slice"], seed=0)
    te = SyntheticPointDataset(cfg["n_per_class_test"], cfg["n_points"],
                               cfg["k_slices"], cfg["points_per_slice"], seed=1)
    return (DataLoader(tr, cfg["batch_size"], shuffle=True, drop_last=True),
            DataLoader(te, cfg["batch_size"]))


def get_model(tag="tiny", retrain=False, **over):
    cfg = {**TINY, **over}
    ck = os.path.join(RESULTS, f"ckpt_{tag}.pt")
    tr, te = loaders(cfg)
    torch.manual_seed(0)
    model = ASPModel(ASPConfig.from_dict(cfg))
    if os.path.exists(ck) and not retrain:
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        print(f"[common] loaded cached {ck}")
    else:
        print(f"[common] training {tag} model ({cfg['epochs']} epochs, CPU)...")
        model, _ = train_model(cfg, tr, te)
        torch.save(model.state_dict(), ck)
    return model, tr, te, cfg


def save_csv(name, rows):
    if not rows:
        return
    path = os.path.join(RESULTS, name)
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"[common] wrote {path}")


def maybe_plot(fn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fn(plt)
    except Exception as e:  # plotting is best-effort
        print(f"[common] plot skipped: {e}")


def verdict(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def diagnostic(name, observed, detail=""):
    """Report an empirical observation without turning it into a theorem test."""
    status = "OBSERVED" if observed else "NOT OBSERVED"
    print(f"[DIAGNOSTIC: {status}] {name} {detail}")
    return observed
