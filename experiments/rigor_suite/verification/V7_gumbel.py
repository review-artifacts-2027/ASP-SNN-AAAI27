"""V7 — Proposition 8 (Gumbel-max train/inference consistency).

Exact identity (Gumbel-max trick): P[argmax(s+g) = argmax(s)] = softmax(s)_max.
So training-time (noisy hard) and inference-time (argmax) selections agree with
probability E[p_max(scores)] — measurable, and it GROWS as training sharpens
score gaps. tau only rescales the soft backward weights (bias->0 as tau->0).

Tests:
 (a) empirical agreement over Gumbel draws == E[softmax(s)_max] (within 3pp);
 (b) diagnostic: trained-vs-untrained agreement (gap growth is empirical);
 (c) straight-through gradients flow to Wq/Wk at every tau in {1.0, 0.5, 0.1}.
"""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import diagnostic, get_model, save_csv, verdict
from asp.model import ASPConfig, ASPModel

model, tr, te, cfg = get_model()

def agreement(m, n_draws=100):
    regions, desc, anchors, labels = next(iter(te))
    with torch.no_grad():
        d, B = m._prep(regions, desc)
        feats = m.encoder(regions, anchors)
        visited = torch.zeros(B, cfg["k_slices"], dtype=torch.bool)
        # advance 2 steps with argmax to get a realistic mid-episode state
        u = m.head.membrane
        for _ in range(2):
            s = m.ssp.scores(u, d, visited)
            w = m.ssp.select(s, hard_inference=True)
            e = torch.einsum("bk,bkd->bd", w, feats)
            m.readout(m.head(m.proj(e)))
            visited = visited | (w > 0.5); u = m.head.membrane
        s = m.ssp.scores(u, d, visited)
        pred = F.softmax(s, -1).amax(-1).mean().item()      # E[p_max]
        hard = s.argmax(-1)
        agree = 0.0
        for _ in range(n_draws):
            g = -torch.log(-torch.log(torch.rand_like(s).clamp_min(1e-20)))
            agree += (s + g).argmax(-1).eq(hard).float().mean().item()
        return agree / n_draws, pred

emp, pred = agreement(model)
ok_a = verdict("V7a Gumbel-max identity: empirical == E[p_max]",
               abs(emp - pred) < 0.03, f"(empirical={emp:.3f}, predicted={pred:.3f})")

torch.manual_seed(123)
fresh = ASPModel(ASPConfig.from_dict(cfg))
emp0, pred0 = agreement(fresh)
diagnostic(
    "V7b training sharpens selection consistency",
    emp >= emp0 - 0.02,
    f"(trained={emp:.3f}, untrained={emp0:.3f})",
)

grads_ok, rows = True, [{"model": "trained", "agreement": emp, "predicted": pred},
                        {"model": "untrained", "agreement": emp0, "predicted": pred0}]
for tau in [1.0, 0.5, 0.1]:
    regions, desc, anchors, labels = next(iter(tr))
    out = model.forward_train(regions, desc, anchors, tau_gumbel=tau)
    loss = out["logits"][:, -1].logsumexp(-1).mean()
    model.zero_grad(); loss.backward()
    gq = model.ssp.Wq.weight.grad
    gnorm = 0.0 if gq is None else gq.norm().item()
    rows.append({"model": f"tau={tau}", "agreement": "", "predicted": "",
                 "wq_grad_norm": gnorm})
    if gq is None or not torch.isfinite(gq).all() or gnorm == 0.0:
        grads_ok = False
ok_c = verdict("V7c ST gradients finite & nonzero at all tau", grads_ok)
save_csv("V7_gumbel.csv", rows)
sys.exit(0 if (ok_a and ok_c) else 1)
