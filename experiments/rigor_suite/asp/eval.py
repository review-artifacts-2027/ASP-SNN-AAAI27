"""Single-pass evaluation: stores full trajectories so every theta-dependent
metric (A1 sweep, drift, exits, selective risk) comes from ONE inference run."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import metrics as M


@torch.no_grad()
def collect(model, loader, device="cpu", keep_membrane=False) -> dict:
    model.eval()
    outs = {"logits": [], "margins": [], "selections": [], "labels": [],
            "membranes": []}
    for batch in loader:
        regions, desc, anchors, labels = [b.to(device) if b is not None else None
                                          for b in batch]
        o = model.forward_infer(regions, desc, anchors, theta=2.0,  # never exit early;
                                keep_membrane=keep_membrane)        # exits re-derived
        outs["logits"].append(o["logits"])
        outs["margins"].append(o["margins"])
        outs["selections"].append(o["selections"])
        outs["labels"].append(labels)
        if keep_membrane:
            outs["membranes"].append(o["membranes"])
    res = {k: torch.cat(v) for k, v in outs.items() if v}
    return res


def evaluate(model, loader, device="cpu", thetas=(0.5, 0.6, 0.7, 0.8, 0.9),
             keep_membrane=False) -> dict:
    raw = collect(model, loader, device, keep_membrane)
    logits, margins, labels = raw["logits"], raw["margins"], raw["labels"]
    sel = raw["selections"]
    C = logits.shape[-1]
    preds_T = logits[:, -1].argmax(-1)
    p_final = F.softmax(logits[:, -1], -1)
    correct_T = preds_T == labels
    res = {
        "acc_full_T": M.overall_accuracy(preds_T, labels),
        "per_class_full": M.per_class_accuracy(preds_T, labels, C),
        "theta_rows": M.theta_sweep(logits, margins, labels, list(thetas)),
        "revisit": M.revisit_stats(sel),
        "drift": M.margin_drift(margins),
        "ece": M.ece(p_final.amax(-1), correct_T),
        "risk_coverage": M.risk_coverage(margins[:, -1], correct_T),
        "raw": raw,
    }
    return res
