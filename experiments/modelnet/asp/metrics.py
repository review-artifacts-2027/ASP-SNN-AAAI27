"""Logging metrics for ablations, strengthening experiments, and verification.

All exit-dependent metrics are recomputed from stored margin/logit trajectories,
so a single inference pass per model yields the entire theta sweep (A1).
"""
from __future__ import annotations

import math

import torch

# Loihi-2 energy constants (proposal Sec. 10.3): AC ~2.3e-3 pJ, MAC ~8.4e-3 pJ.
E_AC_PJ, E_MAC_PJ = 2.3e-3, 8.4e-3


# ---------------------------------------------------------------- exit logic
def exits_from_margins(margins: torch.Tensor, theta: float) -> torch.Tensor:
    """margins: (B,T) -> exit step in {1..T} (T if never exceeded). Theorem 2 object."""
    B, T = margins.shape
    hit = margins > theta                                  # (B,T)
    first = torch.where(hit.any(1), hit.float().argmax(1) + 1,
                        torch.full((B,), T, device=margins.device))
    return first.long()


def exit_predictions(logits: torch.Tensor, exit_step: torch.Tensor) -> torch.Tensor:
    """logits: (B,T,C), exit_step: (B,) in 1..T -> predicted class at exit."""
    idx = (exit_step - 1).clamp(min=0)
    sel = torch.gather(logits, 1, idx.view(-1, 1, 1).expand(-1, 1, logits.shape[-1]))
    return sel.squeeze(1).argmax(-1)


def theta_sweep(logits: torch.Tensor, margins: torch.Tensor, labels: torch.Tensor,
                thetas: list[float]) -> list[dict]:
    """A1: (accuracy, avg slices, exit rate, exited-sample risk) per theta."""
    rows = []
    T = margins.shape[1]
    for th in thetas:
        es = exits_from_margins(margins, th)
        preds = exit_predictions(logits, es)
        acc = (preds == labels).float().mean().item()
        exited = es < T
        risk_exited = ((preds != labels) & exited).float().sum().item() / max(exited.sum().item(), 1)
        rows.append({"theta": th, "accuracy": acc,
                     "avg_slices": es.float().mean().item(),
                     "slice_utilization": es.float().mean().item() / T,
                     "exit_rate": exited.float().mean().item(),
                     "risk_exited": risk_exited})
    return rows


# ------------------------------------------------------------------ accuracy
def overall_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def per_class_accuracy(preds: torch.Tensor, labels: torch.Tensor, c: int) -> list[float]:
    return [((preds == labels) & (labels == i)).float().sum().item() /
            max((labels == i).sum().item(), 1) for i in range(c)]


# ------------------------------------------------------- selection behaviour
def revisit_stats(selections: torch.Tensor) -> dict:
    """A2/T4: selections (B,T) -> revisit rate and distinct-slice coverage curve."""
    B, T = selections.shape
    distinct_curve, revisits = [], torch.zeros(B)
    for t in range(1, T + 1):
        s = selections[:, :t]
        d = torch.tensor([len(set(s[b].tolist())) for b in range(B)], dtype=torch.float)
        distinct_curve.append(d.mean().item())
    revisits = T - torch.tensor([len(set(selections[b].tolist())) for b in range(B)],
                                dtype=torch.float)
    return {"revisit_rate": (revisits / T).mean().item(),
            "distinct_curve": distinct_curve,
            "final_distinct": distinct_curve[-1]}


def margin_drift(margins: torch.Tensor) -> dict:
    """T2: per-step increments of the margin process. delta_hat = mean increment."""
    inc = margins[:, 1:] - margins[:, :-1]                 # (B,T-1)
    per_step = inc.mean(0)                                 # E[dM_t]
    return {"delta_hat": inc.mean().item(),
            "delta_min_step": per_step.min().item(),
            "per_step_drift": per_step.tolist(),
            "m1_mean": margins[:, 0].mean().item(),
            "frac_negative_drift_samples": (inc.mean(1) <= 0).float().mean().item()}


# ---------------------------------------------------------------- calibration
def ece(probs_top: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (probs_top > edges[i]) & (probs_top <= edges[i + 1])
        if m.any():
            e += m.float().mean().item() * abs(probs_top[m].mean().item()
                                               - correct[m].float().mean().item())
    return e


def risk_coverage(margins_final: torch.Tensor, correct: torch.Tensor) -> dict:
    """S5: selective risk-coverage curve ordered by confidence margin; AURC."""
    order = margins_final.argsort(descending=True)
    err = (~correct[order]).float()
    cum_risk = err.cumsum(0) / torch.arange(1, len(err) + 1)
    coverage = torch.arange(1, len(err) + 1).float() / len(err)
    return {"aurc": cum_risk.mean().item(),
            "coverage": coverage.tolist(), "risk": cum_risk.tolist()}


# -------------------------------------------------------------------- energy
def energy_proxy_pj(avg_steps: float, points_per_slice: int, enc_hidden: int,
                    d_model: int, k_slices: int, d_ssp: int,
                    firing_rate: float = 0.15) -> dict:
    """Documented Loihi-2 estimate. Backbone ops are AC on spikes; SSP adds
    O(K*d_ssp) AC per step (paper Sec. 5). Reported per inference."""
    enc_ops = points_per_slice * (3 * enc_hidden + enc_hidden * d_model)
    head_ops = d_model * d_model
    per_step_ac = (enc_ops + head_ops) * firing_rate
    ssp_ac = k_slices * d_ssp
    e = avg_steps * (per_step_ac + ssp_ac) * E_AC_PJ
    e_fixed = k_slices * per_step_ac * E_AC_PJ
    return {"energy_pj": e, "energy_fixed_order_pj": e_fixed,
            "energy_ratio": e / max(e_fixed, 1e-9),
            "ssp_overhead_frac": (avg_steps * ssp_ac) / max(avg_steps * (per_step_ac + ssp_ac), 1e-9)}


# ---------------------------------------------------------------- statistics
def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            conc += s > 0
            disc += s < 0
    denom = n * (n - 1) / 2
    return (conc - disc) / denom if denom else 0.0


def welch_ttest(x: list[float], y: list[float]) -> dict:
    """Welch's t-test with normal-approx p-value (fine for reporting; use scipy
    at full scale for exact df)."""
    import statistics as st
    nx, ny = len(x), len(y)
    mx, my = st.mean(x), st.mean(y)
    vx = st.variance(x) if nx > 1 else 0.0
    vy = st.variance(y) if ny > 1 else 0.0
    se = math.sqrt(vx / nx + vy / ny) if (vx or vy) else 1e-12
    t = (mx - my) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"t": t, "p_normal_approx": p, "mean_diff": mx - my}


def ci95(vals: list[float]) -> dict:
    import statistics as st
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": m, "ci95": 1.96 * sd / math.sqrt(max(len(vals), 1)), "std": sd}
