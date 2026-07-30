from __future__ import annotations

import torch

from . import metrics as M


def anytime_curve(logits: torch.Tensor, labels: torch.Tensor,
                  ks=None) -> dict:

    B, K, C = logits.shape
    ks = list(ks) if ks is not None else list(range(1, K + 1))
    out = {}
    for k in ks:
        preds = logits[:, k - 1].argmax(-1)
        out[k] = (preds == labels).float().mean().item()
    return out


def budget_exit_stats(logits: torch.Tensor, margins: torch.Tensor,
                      labels: torch.Tensor, theta: float) -> dict:

    es = M.exits_from_margins(margins, theta)
    preds = M.exit_predictions(logits, es)
    acc = (preds == labels).float().mean().item()
    return {"theta": theta, "tau_bar": es.float().mean().item(),
            "acc_theta": acc, "exit_rate": (es < margins.shape[1]).float().mean().item()}


def summarize_rule(raw: dict, thetas, ks=None) -> dict:
    logits, margins, labels = raw["logits"], raw["margins"], raw["labels"]
    return {
        "acc_full_K": (logits[:, -1].argmax(-1) == labels).float().mean().item(),
        "anytime": anytime_curve(logits, labels, ks),
        "theta_rows": [budget_exit_stats(logits, margins, labels, t) for t in thetas],
    }
