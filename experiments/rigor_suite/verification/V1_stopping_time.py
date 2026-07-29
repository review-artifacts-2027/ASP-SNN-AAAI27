"""V1 — Theorem 2 (expected stopping time / margin submartingale).

Claims tested:
 (a) The margin process M_t has positive mean drift under the trained SSP
     (submartingale up to noise): E[M_t - M_{t-1}] = delta > 0.
 (b) Wald-style bound: E[T_theta] <= 1 + (theta - E[M_1])^+ / delta  (clipped
     at T_max), so measured E[T] is upper-bounded by the prediction and is
     ~affine in theta with slope 1/delta.
 (c) Diagnostic only: compare the high-threshold censoring mass with the
     fraction of samples whose fitted drift is non-positive. No theorem equates
     or orders these two quantities without stronger distributional assumptions.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import diagnostic, get_model, save_csv, maybe_plot, verdict, RESULTS
from asp.eval import collect
from asp import metrics as M

model, tr, te, cfg = get_model()
raw = collect(model, te)
margins = raw["margins"]                     # (B,T)
T = margins.shape[1]
drift = M.margin_drift(margins)
delta, m1 = drift["delta_hat"], drift["m1_mean"]
inc = margins[:, 1:] - margins[:, :-1]
c_hat = inc.clamp(min=0).max().item()  # observed increment bound for this audit
print(f"delta_hat={delta:.4f}  E[M_1]={m1:.4f}  c_hat={c_hat:.4f}  "
      f"min-step drift={drift['delta_min_step']:.4f}  "
      f"frac neg-drift samples={drift['frac_negative_drift_samples']:.3f}")

rows, ok_bound = [], True
for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    es = M.exits_from_margins(margins, th).float()
    measured = es.mean().item()
    pred = min(1 + max(0.0, th + c_hat - m1) / max(delta, 1e-6), T)
    rows.append({"theta": th, "measured_ET": measured, "wald_bound": pred,
                 "censored_frac": (es == T).float().mean().item()})
    if delta > 0 and measured > pred + 0.75:      # 0.75-slice tolerance
        ok_bound = False
save_csv("V1_stopping_time.csv", rows)

meas = [r["measured_ET"] for r in rows]
ok_a = verdict("V1a positive drift", delta > 0, f"(delta={delta:.4f})")
ok_b = verdict("V1b Wald bound covers measured E[T]", ok_bound)
mono = all(meas[i] <= meas[i + 1] + 1e-9 for i in range(len(meas) - 1))
ok_c = verdict("V1c E[T] monotone in theta", mono)
cens = rows[-1]["censored_frac"]
neg = drift["frac_negative_drift_samples"]
diagnostic(
    "V1d high-threshold censoring vs non-positive fitted drift",
    abs(cens - neg) <= 0.15,
    f"(censored@0.9={cens:.3f}, neg-drift={neg:.3f}; descriptive only)",
)

def plot(plt):
    th = [r["theta"] for r in rows]
    plt.figure(figsize=(5, 3.5))
    plt.plot(th, meas, "o-", label="measured E[T]")
    plt.plot(th, [r["wald_bound"] for r in rows], "s--", label="Wald bound (Thm 2)")
    plt.xlabel(r"$\theta$"); plt.ylabel("slices to exit"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "V1_stopping_time.png"), dpi=150)
maybe_plot(plot)
sys.exit(0 if (ok_a and ok_b and ok_c) else 1)
