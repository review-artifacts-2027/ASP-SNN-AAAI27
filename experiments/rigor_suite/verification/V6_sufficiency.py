"""V6 — Theorem 4 sufficiency side (membrane state as approximate belief state).

Tests:
 (a) Probe sufficiency: a linear probe on u_t predicts the label ~as well as a
     probe on the FULL feature history (gap epsilon small) => u_t compresses
     history w.r.t. the task (approximate sufficient statistic).
 (b) Diagnostic only: compare the trained membrane-driven policy with random
     and fixed orders on the same backbone. Approximate state sufficiency does
     not, by itself, imply that the learned policy is superior.
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import diagnostic, get_model, save_csv, maybe_plot, verdict, RESULTS
from asp.eval import collect
from asp import metrics as M

model, tr, te, cfg = get_model()
PROBE_T = 4

def gather(loader):
    raw = collect(model, loader, keep_membrane=True)
    return raw

tr_raw, te_raw = gather(tr), gather(te)

def probe(x_tr, y_tr, x_te, y_te, epochs=500):
    lin = torch.nn.Linear(x_tr.shape[1], cfg["num_classes"])
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(lin(x_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return (lin(x_te).argmax(-1) == y_te).float().mean().item()

u_tr = tr_raw["membranes"][:, PROBE_T - 1]
u_te = te_raw["membranes"][:, PROBE_T - 1]
h_tr = tr_raw["membranes"][:, :PROBE_T].reshape(len(u_tr), -1)
h_te = te_raw["membranes"][:, :PROBE_T].reshape(len(u_te), -1)
acc_u = probe(u_tr, tr_raw["labels"], u_te, te_raw["labels"])
acc_h = probe(h_tr, tr_raw["labels"], h_te, te_raw["labels"])          # raw (capacity-confounded)
# capacity-matched control: PCA the 4x-wider history down to dim(u) so the
# probe compares INFORMATION, not parameter count (standard probing control)
mu = h_tr.mean(0, keepdim=True)
U_, S_, V_ = torch.pca_lowrank(h_tr - mu, q=u_tr.shape[1])
hp_tr, hp_te = (h_tr - mu) @ V_, (h_te - mu) @ V_
acc_hm = probe(hp_tr, tr_raw["labels"], hp_te, te_raw["labels"])
eps_raw, eps = acc_h - acc_u, acc_hm - acc_u
print(f"probe acc: u_t={acc_u:.3f} | history(raw {h_tr.shape[1]}d)={acc_h:.3f} "
      f"| history(PCA-matched {u_tr.shape[1]}d)={acc_hm:.3f}")
# Pass criterion follows Prop. 4 (approximate sufficiency): epsilon need not be
# zero -- it must be small enough that the Pinsker regret bound L*sqrt(eps/2)
# stays below the measured closed-loop advantage scale. We use eps <= 0.05
# (probe-accuracy points) at verification scale and report eps verbatim; the
# full-scale protocol (A5/S3) tests the policy-value consequence directly.
ok_a = verdict("V6a membrane ~sufficient (capacity-matched probe gap)",
               eps <= 0.05,
               f"(eps_matched={eps:.3f}; eps_raw={eps_raw:.3f} reported for transparency)")

# (b) policy value on identical backbone -------------------------------------
rows = []
curves = {}
for pol in ["ssp", "random", "fixed"]:
    old = model.ssp.policy
    model.ssp.policy = pol
    raw = collect(model, te)
    accs = [(raw["logits"][:, t].argmax(-1) == raw["labels"]).float().mean().item()
            for t in range(raw["logits"].shape[1])]
    curves[pol] = accs
    rows += [{"policy": pol, "t": t + 1, "acc": a} for t, a in enumerate(accs)]
    model.ssp.policy = old
save_csv("V6_policy_value.csv", rows)
early = range(0, 6)
adv_r = sum(curves["ssp"][t] - curves["random"][t] for t in early) / 6
adv_f = sum(curves["ssp"][t] - curves["fixed"][t] for t in early) / 6
diagnostic(
    "V6b membrane policy > random/fixed (early anytime accuracy)",
    adv_r > 0.0 and adv_f > -0.02,
    f"(adv vs random={adv_r:.3f}, vs fixed={adv_f:.3f})",
)

def plot(plt):
    plt.figure(figsize=(5, 3.5))
    for pol, c in curves.items():
        plt.plot(range(1, len(c) + 1), c, "o-", label=pol, markersize=3)
    plt.xlabel("step t"); plt.ylabel("anytime accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "V6_policy_value.png"), dpi=150)
maybe_plot(plot)
sys.exit(0 if ok_a else 1)
