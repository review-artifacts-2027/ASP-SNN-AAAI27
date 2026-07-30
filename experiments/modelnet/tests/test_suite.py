"""Correctness rechecks for the ASP/SSP implementation. Run: python tests/test_suite.py"""
import os, sys, math
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asp.model import ASPConfig, ASPModel
from asp.geometry import slice_point_cloud, DESC_COMPONENTS
from asp.datasets import SyntheticPointDataset
from asp import metrics as M

torch.manual_seed(0)
FAILED = []

def check(name, cond, detail=""):
    print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILED.append(name)

cfg = ASPConfig(num_classes=8, d_model=64, k_slices=16, points_per_slice=16,
                enc_hidden=32, d_ssp=64)
model = ASPModel(cfg)
pts = torch.randn(4, 128, 3)
slices, desc, anchors = slice_point_cloud(pts, 16, 16)

# 1. shapes and descriptor semantics
check("slice shapes", slices.shape == (4, 16, 16, 3) and desc.shape == (4, 16, 6))
check("descriptor count sums to 1", torch.allclose(desc[..., 5].sum(-1),
                                                   torch.ones(4), atol=1e-5))
check("desc has 6 named components", len(DESC_COMPONENTS) == 6)

# 2. training path: gradients reach the SSP through Gumbel-ST
out = model.forward_train(slices, desc, anchors, tau_gumbel=1.0)
loss = out["logits"].sum()
loss.backward()
check("train logits shape", out["logits"].shape == (4, 16, 8))
check("SSP Wq gets gradient", model.ssp.Wq.weight.grad is not None
      and model.ssp.Wq.weight.grad.abs().sum() > 0)
check("SSP Wk gets gradient", model.ssp.Wk.weight.grad is not None
      and model.ssp.Wk.weight.grad.abs().sum() > 0)

# 3. masking: with mask, all K selections distinct; never revisit
inf = model.forward_infer(slices, desc, anchors, theta=2.0)
sel = inf["selections"]
distinct = [len(set(sel[b].tolist())) for b in range(4)]
check("mask => sampling w/o replacement", all(d == 16 for d in distinct), str(distinct))

# 4. no-mask flag actually allows revisits (T4 precondition)
model.ssp.use_mask = False
inf2 = model.forward_infer(slices, desc, anchors, theta=2.0)
d2 = [len(set(inf2["selections"][b].tolist())) for b in range(4)]
check("no-mask allows revisits", any(d < 16 for d in d2), str(d2))
model.ssp.use_mask = True

# 5. exit threshold monotonicity: higher theta => more slices
es_lo = M.exits_from_margins(inf["margins"], 0.2).float().mean()
es_hi = M.exits_from_margins(inf["margins"], 0.9).float().mean()
check("E[T] monotone in theta", es_lo <= es_hi + 1e-6, f"{es_lo:.2f} vs {es_hi:.2f}")

# 6. parameter accounting (Thm 5 arithmetic)
d, D = 64, 64
check("ssp params = d*(D+6) full rank",
      model.ssp.param_count() == d * (D + 6), str(model.ssp.param_count()))
from asp.ssp import SSP
budget = SSP(d_model=128, d_ssp=64, rank=8)
check("rank-8 budget ~2K params at D=128", budget.param_count() == 8*128 + 8*64 + 64*6,
      str(budget.param_count()))
b6 = SSP(d_model=128, d_ssp=6)
check("minimal exact d=6 params", b6.param_count() == 6 * (128 + 6), str(b6.param_count()))

# 7. bilinear rank cap holds numerically on random init
r = torch.linalg.matrix_rank(budget.bilinear_form()).item()
check("rank(B) <= 6", r <= 6, f"rank={r}")

# 8. random policy uniform over unvisited; fixed policy is ordinal
model.ssp.policy = "random"
i1 = model.forward_infer(slices, desc, anchors, theta=2.0)["selections"]
check("random policy covers all slices", all(len(set(i1[b].tolist())) == 16
                                             for b in range(4)))
model.ssp.policy = "fixed"
i2 = model.forward_infer(slices, desc, anchors, theta=2.0)["selections"]
check("fixed policy = ordinal order",
      torch.equal(i2[0], torch.arange(16)), str(i2[0].tolist()))
model.ssp.policy = "ssp"

# 9. tiny overfit sanity: loss decreases on one batch
ds = SyntheticPointDataset(6, 128, 16, 16, seed=0)
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=16, shuffle=True)
m2 = ASPModel(ASPConfig(num_classes=8, d_model=48, enc_hidden=32,
                        k_slices=16, points_per_slice=16))
opt = torch.optim.Adam(m2.parameters(), lr=3e-3)
losses = []
batch = next(iter(dl))
for step in range(120):
    reg, de, an, lab = batch
    o = m2.forward_train(reg, de, an, 1.0)
    l = torch.nn.functional.cross_entropy(
        o["logits"].reshape(-1, 8), lab.unsqueeze(1).expand(-1, 16).reshape(-1))
    opt.zero_grad(); l.backward(); opt.step()
    losses.append(l.item())
check("overfit sanity: loss drops >30%", losses[-1] < 0.7 * losses[0],
      f"{losses[0]:.3f} -> {losses[-1]:.3f}")

# 10. margin/exit bookkeeping consistent with stored logits
th = 0.5
es = M.exits_from_margins(inf["margins"], th)
pr = M.exit_predictions(inf["logits"], es)
check("exit preds derivable from trajectories", pr.shape == (4,))

print("\n" + ("ALL TESTS PASS" if not FAILED else f"FAILURES: {FAILED}"))
sys.exit(1 if FAILED else 0)
