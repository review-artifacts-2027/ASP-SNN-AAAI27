"""
asp/anytime.py — accuracy@k anytime curve and tau-bar/acc@theta at matched
budget. These are the confound-free metrics for the M=16 selection ablation:

  * accuracy@k : accuracy of argmax(logits after EXACTLY k observed slices).
                 Removes the tau-bar confound entirely — every rule is compared
                 at the SAME observation budget k. This is the primary evidence
                 for W1/W2/W3.
  * tau_bar(theta), acc@theta : the deployment view under margin early exit,
                 derived post-hoc from the stored margin trajectory (no retrain).

All operate on the trajectory tensors produced by asp.eval.collect /
asp.oracle.collect_oracle (logits:(B,K,C), margins:(B,K), labels:(B,)).
Uses asp.metrics for exit logic so behaviour matches the rest of the suite.
"""
from __future__ import annotations

import torch

from . import metrics as M


def anytime_curve(logits: torch.Tensor, labels: torch.Tensor,
                  ks=None) -> dict:
    """accuracy@k for k in ks (default 1..K). Returns {k: acc_float}."""
    B, K, C = logits.shape
    ks = list(ks) if ks is not None else list(range(1, K + 1))
    out = {}
    for k in ks:
        preds = logits[:, k - 1].argmax(-1)
        out[k] = (preds == labels).float().mean().item()
    return out


def budget_exit_stats(logits: torch.Tensor, margins: torch.Tensor,
                      labels: torch.Tensor, theta: float) -> dict:
    """tau_bar and acc under margin-exit at threshold theta (Theorem-2 object)."""
    es = M.exits_from_margins(margins, theta)          # (B,) in 1..K
    preds = M.exit_predictions(logits, es)
    acc = (preds == labels).float().mean().item()
    return {"theta": theta, "tau_bar": es.float().mean().item(),
            "acc_theta": acc, "exit_rate": (es < margins.shape[1]).float().mean().item()}


def summarize_rule(raw: dict, thetas, ks=None) -> dict:
    """Full per-rule summary from a collect()/collect_oracle() dict.
    `raw` must have keys logits, margins, labels."""
    logits, margins, labels = raw["logits"], raw["margins"], raw["labels"]
    return {
        "acc_full_K": (logits[:, -1].argmax(-1) == labels).float().mean().item(),
        "anytime": anytime_curve(logits, labels, ks),
        "theta_rows": [budget_exit_stats(logits, margins, labels, t) for t in thetas],
    }