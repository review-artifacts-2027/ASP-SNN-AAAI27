"""V5 — Masking coverage theorem and unmasked-revisit diagnostics.

Same trained weights, mask flipped OFF at inference. Predictions:
 (a) revisit rate >> 0 and distinct-slice curve plateaus below K;
 (b) margin drift delta collapses toward/below the masked drift;
 (c) diagnostic only: compare accuracy and average slices to exit. The coverage
     theorem does not establish a universal accuracy or latency ordering.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import diagnostic, get_model, save_csv, maybe_plot, verdict, RESULTS
from asp.eval import collect
from asp import metrics as M

model, tr, te, cfg = get_model()
res = {}
for mode in ["mask_on", "mask_off"]:
    model.ssp.use_mask = mode == "mask_on"
    raw = collect(model, te)
    rev = M.revisit_stats(raw["selections"])
    drift = M.margin_drift(raw["margins"])
    inc = (raw["margins"][:, 1:] - raw["margins"][:, :-1]).abs()
    early_abs = inc[:, :5].mean().item()          # steps 2-6
    late_abs = inc[:, -5:].mean().item()          # last 5 steps
    sweep = M.theta_sweep(raw["logits"], raw["margins"], raw["labels"], [0.7])[0]
    acc_T = M.overall_accuracy(raw["logits"][:, -1].argmax(-1), raw["labels"])
    res[mode] = {"revisit_rate": rev["revisit_rate"],
                 "final_distinct": rev["final_distinct"],
                 "distinct_curve": rev["distinct_curve"],
                 "delta_hat": drift["delta_hat"], "acc_full_T": acc_T,
                 "early_abs_inc": early_abs, "late_abs_inc": late_abs,
                 "avg_slices@0.7": sweep["avg_slices"],
                 "acc@0.7": sweep["accuracy"]}
model.ssp.use_mask = True
save_csv("V5_masking.csv", [{"mode": k, **{kk: vv for kk, vv in v.items()
                                           if not isinstance(vv, list)}}
                            for k, v in res.items()])
on, off = res["mask_on"], res["mask_off"]
K = cfg["k_slices"]
ok_a = verdict("V5a no-mask revisits & coverage plateau",
               off["revisit_rate"] > 0.1 and off["final_distinct"] < 0.75 * K,
               f"(revisit={off['revisit_rate']:.2f}, distinct={off['final_distinct']:.1f}/{K})")
# Thm 6 predicts increments VANISH at the no-mask fixed point (late steps),
# while the masked policy keeps injecting new evidence (nonzero |dM|).
stall_ratio_off = off["late_abs_inc"] / max(off["early_abs_inc"], 1e-9)
stall_ratio_on = on["late_abs_inc"] / max(on["early_abs_inc"], 1e-9)
ok_b = verdict("V5b fixed-point stall: late |dM| collapses w/o mask",
               off["late_abs_inc"] < on["late_abs_inc"]
               and stall_ratio_off < stall_ratio_on,
               f"(late |dM| on={on['late_abs_inc']:.4f} off={off['late_abs_inc']:.4f}; "
               f"stall ratio on={stall_ratio_on:.2f} off={stall_ratio_off:.2f})")
diagnostic(
    "V5c unmasked policy is no more accurate and no faster",
    off["acc_full_T"] <= on["acc_full_T"] + 0.005
    and off["avg_slices@0.7"] >= on["avg_slices@0.7"] - 0.25,
    f"(acc {on['acc_full_T']:.3f}->{off['acc_full_T']:.3f}, "
    f"slices {on['avg_slices@0.7']:.2f}->{off['avg_slices@0.7']:.2f})",
)

def plot(plt):
    plt.figure(figsize=(5, 3.5))
    for k, v in res.items():
        plt.plot(range(1, K + 1), v["distinct_curve"], "o-", label=k)
    plt.plot(range(1, K + 1), range(1, K + 1), "k:", label="perfect coverage")
    plt.xlabel("step t"); plt.ylabel("distinct slices visited"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "V5_coverage.png"), dpi=150)
maybe_plot(plot)
sys.exit(0 if (ok_a and ok_b) else 1)
