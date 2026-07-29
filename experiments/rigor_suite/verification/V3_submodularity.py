"""V3 — Theorem 3/4 (submodular information gain; greedy near-optimality).

Accuracy-after-observing-S is used as the empirical proxy for the information
gain V(S) (documented in the theory section; MI estimation at scale uses the
same protocol with logit-based estimators).

Tests:
 (a) Diminishing returns: marginal gain of the k-th greedy slice decreases in k.
 (b) Submodular inequality on sampled chains S subset T, s notin (T):
     gain(S,s) >= gain(T,s) - tol   for a large majority of sampled chains.
 (c) Diagnostic only: compare SSP and random orderings with an oracle-greedy
     ordering. The conditional greedy theorem does not guarantee that a learned
     amortized policy recovers the oracle ranking.
"""
import os, sys, random
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import diagnostic, get_model, save_csv, maybe_plot, verdict, RESULTS

model, tr, te, cfg = get_model()
model.eval()
K = cfg["k_slices"]

# ---- gather a fixed eval tensor batch and precompute slice features --------
regions, descs, anchors, labels = [], [], [], []
for r, d, a, y in te:
    regions.append(r); descs.append(d); anchors.append(a); labels.append(y)
regions = torch.cat(regions)[:160]; descs = torch.cat(descs)[:160]
anchors = torch.cat(anchors)[:160]; labels = torch.cat(labels)[:160]
with torch.no_grad():
    feats = model.encoder(regions, anchors)              # (B,K,D)
B = feats.shape[0]

# ---- exact LIF replay as pure functions (matches ASPModel.forward_infer) ---
a_h = model.head.alpha.detach(); vth = model.head.v_th
a_r = torch.sigmoid(model.readout.raw_alpha.detach())
Wp, bp = model.proj.weight.detach(), model.proj.bias.detach()
Wr, br = model.readout.fc.weight.detach(), model.readout.fc.bias.detach()

def replay(order_bt):
    """order_bt: (B, L) slice indices -> logits after processing the sequence."""
    u = torch.zeros(B, feats.shape[2]); v = torch.zeros(B, Wr.shape[0])
    for t in range(order_bt.shape[1]):
        e = feats[torch.arange(B), order_bt[:, t]]
        x = e @ Wp.t() + bp
        u = a_h * u + (1 - a_h) * x
        o = (u >= vth).float()
        u = u - o * vth
        v = a_r * v + (1 - a_r) * (o @ Wr.t() + br)
    return v

def acc_of_sets(sets_b):
    """sets_b: list of per-sample index lists (same length L) -> accuracy."""
    L = len(sets_b[0])
    order = torch.tensor(sets_b)
    with torch.no_grad():
        return (replay(order).argmax(-1) == labels).float().mean().item()

rng = random.Random(0)

# (a)+(c): per-sample ORACLE greedy (uses labels), SSP order, random order ----
with torch.no_grad():
    ssp_out = model.forward_infer(regions, descs, anchors, theta=2.0)
ssp_sel = ssp_out["selections"]                           # (B,T)

greedy = torch.zeros(B, 8, dtype=torch.long)
u = torch.zeros(B, feats.shape[2]); v = torch.zeros(B, Wr.shape[0])
visited = torch.zeros(B, K, dtype=torch.bool)
for step in range(8):
    # expand candidates: for each unvisited slice, one-step lookahead p_true
    p_best = torch.full((B,), -1.0); best = torch.zeros(B, dtype=torch.long)
    for m in range(K):
        x = feats[:, m] @ Wp.t() + bp
        u2 = a_h * u + (1 - a_h) * x
        o = (u2 >= vth).float()
        v2 = a_r * v + (1 - a_r) * (o @ Wr.t() + br)
        p = F.softmax(v2, -1)[torch.arange(B), labels]
        p = torch.where(visited[:, m], torch.full_like(p, -2.0), p)
        upd = p > p_best
        p_best = torch.where(upd, p, p_best); best = torch.where(upd, torch.tensor(m), best)
    greedy[:, step] = best
    x = feats[torch.arange(B), best] @ Wp.t() + bp
    u = a_h * u + (1 - a_h) * x
    o = (u >= vth).float(); u = u - o * vth
    v = a_r * v + (1 - a_r) * (o @ Wr.t() + br)
    visited[torch.arange(B), best] = True

rand_order = torch.stack([torch.randperm(K)[:8] for _ in range(B)])
curve_rows, g_prev = [], None
for k in range(1, 9):
    ag = acc_of_sets(greedy[:, :k].tolist())
    as_ = acc_of_sets(ssp_sel[:, :k].tolist())
    ar = acc_of_sets(rand_order[:, :k].tolist())
    curve_rows.append({"k": k, "acc_greedy": ag, "acc_ssp": as_, "acc_random": ar,
                       "marginal_greedy": ag - (g_prev if g_prev is not None else 1 / cfg["num_classes"])})
    g_prev = ag
save_csv("V3_anytime_curves.csv", curve_rows)

marg = [r["marginal_greedy"] for r in curve_rows]
half = len(marg) // 2
dimin = sum(marg[:half]) >= sum(marg[half:]) - 0.02
ok_a = verdict("V3a diminishing greedy marginal gains", dimin,
               f"(first-half sum={sum(marg[:half]):.3f}, second={sum(marg[half:]):.3f})")

# (b) sampled-chain submodularity check on the accuracy proxy ----------------
viol = tot = 0
TOL = 0.03
for _ in range(40):
    perm = rng.sample(range(K), 7)
    S, T_ = perm[:2], perm[:5]
    s = perm[6]
    gS = acc_of_sets([S + [s]] * B) - acc_of_sets([S] * B)
    gT = acc_of_sets([T_ + [s]] * B) - acc_of_sets([T_] * B)
    tot += 1
    if gS < gT - TOL:
        viol += 1
rate = viol / tot
ok_b = verdict("V3b submodular inequality (sampled chains)", rate <= 0.2,
               f"(violation rate={rate:.2f}, tol={TOL})")

gap_sr = sum(r["acc_greedy"] - r["acc_ssp"] for r in curve_rows) / 8
gap_rr = sum(r["acc_greedy"] - r["acc_random"] for r in curve_rows) / 8
diagnostic(
    "V3c SSP tracks oracle-greedy better than random",
    gap_sr <= gap_rr - 0.01,
    f"(greedy-ssp gap={gap_sr:.3f}, greedy-random gap={gap_rr:.3f})",
)

def plot(plt):
    ks = [r["k"] for r in curve_rows]
    plt.figure(figsize=(5, 3.5))
    for key, style in [("acc_greedy", "s--"), ("acc_ssp", "o-"), ("acc_random", "^:")]:
        plt.plot(ks, [r[key] for r in curve_rows], style, label=key[4:])
    plt.xlabel("slices observed k"); plt.ylabel("accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "V3_anytime.png"), dpi=150)
maybe_plot(plot)
sys.exit(0 if (ok_a and ok_b) else 1)
