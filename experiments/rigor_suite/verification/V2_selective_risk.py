"""V2 — Theorem 1 (margin exit selective-risk bound).

Claim: under margin calibration, the error rate of early-exited samples obeys
    P(err | exit at margin > theta) <= (C-1)(1-theta)/C,
The script also reports, as diagnostics, whether empirical risk is monotone in
theta and the final-step ECE. Neither diagnostic is implied by the bound alone.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import diagnostic, get_model, save_csv, maybe_plot, verdict, RESULTS
from asp.eval import collect
from asp import metrics as M
import torch.nn.functional as F

model, tr, te, cfg = get_model()
raw = collect(model, te)
logits, margins, labels = raw["logits"], raw["margins"], raw["labels"]
C = cfg["num_classes"]
rows = M.theta_sweep(logits, margins, labels, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ok_bound = True
for r in rows:
    r["risk_bound_thm1"] = (C - 1) * (1 - r["theta"]) / C
    if r["exit_rate"] > 0.02 and r["risk_exited"] > r["risk_bound_thm1"] + 1e-9:
        ok_bound = False
save_csv("V2_selective_risk.csv", rows)

p_final = F.softmax(logits[:, -1], -1)
correct = logits[:, -1].argmax(-1) == labels
e = M.ece(p_final.amax(-1), correct)
risks = [r["risk_exited"] for r in rows if r["exit_rate"] > 0.02]
mono = all(risks[i] >= risks[i + 1] - 0.02 for i in range(len(risks) - 1))
ok_a = verdict("V2a risk_exited <= (C-1)(1-theta)/C for all theta", ok_bound)
diagnostic("V2b risk_exited ~monotone decreasing in theta", mono)
diagnostic("V2c final-step calibration audit", e <= 0.10, f"(ECE={e:.4f})")

def plot(plt):
    th = [r["theta"] for r in rows]
    plt.figure(figsize=(5, 3.5))
    plt.plot(th, [r["risk_exited"] for r in rows], "o-", label="empirical risk (exited)")
    plt.plot(th, [r["risk_bound_thm1"] for r in rows], "k--", label="Thm 1 bound")
    plt.xlabel(r"$\theta$"); plt.ylabel("selective risk"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "V2_selective_risk.png"), dpi=150)
maybe_plot(plot)
sys.exit(0 if ok_a else 1)
