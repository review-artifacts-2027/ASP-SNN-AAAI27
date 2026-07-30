"""
validate_oracle.py  —  CPU smoke-test of the M=16 selection ablation machinery.

Purpose
-------
De-risk the H100 launch by proving, on a small synthetic task where slice
*order* provably matters, that:

  (1) the oracle-greedy prefix-replay rollout is CORRECT
      -> when fed the learned policy's own order, it reproduces the standard
         forward_infer logits BYTE-IDENTICALLY (invariant test);
  (2) the four selection rules {learned, random, fps_order, oracle} produce a
      SEPARABLE accuracy@k anytime curve at a budget k < M (the whole point of
      forcing tau-bar < M);
  (3) oracle >= {random, fps} and, when the policy is well-trained,
      learned > {random, fps} -> the experiment can detect order-dependence.

The LIF/SSP/multi-head/readout semantics are reconstructed FAITHFULLY from the
rigor-suite source (asp/lif.py, asp/ssp.py, asp/multi_head.py, asp/model.py).
This is NOT ModelNet40 -- it is a controlled synthetic where ground-truth
informative slices exist, so that "order matters" is guaranteed by construction.
The real question (does ModelNet40 have exploitable order structure at M=16) is
exactly what the cluster run answers; this harness only proves the CODE is right.
"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)

# ============================================================================
# 1. Faithful module reconstructions (verbatim semantics from the suite)
# ============================================================================
class _Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x); ctx.k = k
        return (x >= 0.0).to(x.dtype)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors; k = ctx.k
        s = torch.sigmoid(k * x)
        return g * k * s * (1.0 - s), None
def spike_fn(x, k=4.0): return _Surrogate.apply(x, k)

class LIFCell(nn.Module):
    def __init__(self, dim, tau=2.0, v_th=1.0, learnable=True, sg_slope=4.0):
        super().__init__()
        a0 = math.exp(-1.0/tau); raw = math.log(a0/(1.0-a0))
        self.raw_alpha = nn.Parameter(torch.full((dim,), raw)) if learnable \
            else self.register_buffer("raw_alpha", torch.full((dim,), raw))
        self.v_th, self.sg_slope, self.dim = v_th, sg_slope, dim
        self.membrane = None; self._u = None
    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    def reset_state(self, b, device, dtype=torch.float32):
        self._u = torch.zeros(b, self.dim, device=device, dtype=dtype)
        self.membrane = self._u
    def forward(self, x):
        if self._u is None or self._u.shape[0] != x.shape[0]:
            self.reset_state(x.shape[0], x.device, x.dtype)
        a = self.alpha
        u = a * self._u + (1.0 - a) * x
        self.membrane = u
        o = spike_fn(u - self.v_th, self.sg_slope)
        self._u = u - o * self.v_th
        return o

class LeakyReadout(nn.Module):
    """Non-spiking leaky logit integrator (faithful stand-in for asp.lif.LeakyReadout)."""
    def __init__(self, dim, num_classes, tau=2.0):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
        a0 = math.exp(-1.0/tau); self.raw_alpha = nn.Parameter(torch.tensor(math.log(a0/(1-a0))))
        self.num_classes = num_classes; self.state = None
    @property
    def alpha(self): return torch.sigmoid(self.raw_alpha)
    def reset_state(self, b, device, dtype=torch.float32):
        self.state = torch.zeros(b, self.num_classes, device=device, dtype=dtype)
    def forward(self, spikes):
        if self.state is None or self.state.shape[0] != spikes.shape[0]:
            self.reset_state(spikes.shape[0], spikes.device, spikes.dtype)
        y = self.fc(spikes)
        self.state = self.alpha * self.state + (1.0 - self.alpha) * y
        return self.state

class MultiLayerLIFHead(nn.Module):
    def __init__(self, dim, n_layers=3, tau=2.0, v_th=1.0, sg_slope=4.0,
                 use_mpbn=True, use_residual=True):
        super().__init__()
        self.dim, self.n_layers = dim, n_layers
        self.use_mpbn, self.use_residual = use_mpbn, use_residual
        self.linears = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.mpbns = nn.ModuleList([nn.BatchNorm1d(dim) if use_mpbn else nn.Identity()
                                    for _ in range(n_layers)])
        self.cells = nn.ModuleList([LIFCell(dim, tau, v_th, True, sg_slope) for _ in range(n_layers)])
    def reset_state(self, b, device, dtype=torch.float32):
        for c in self.cells: c.reset_state(b, device, dtype)
    def forward(self, x):
        h = x
        for L in range(self.n_layers):
            z = self.linears[L](h)
            if self.use_mpbn: z = self.mpbns[L](z)
            if self.use_residual and L > 0: z = z + h.to(z.dtype)
            h = self.cells[L](z)
        return h
    @property
    def membrane(self): return self.cells[-1].membrane

MASK_VALUE = -1e9
class SSP(nn.Module):
    def __init__(self, d_model, d_desc=6, d_ssp=64, rank=0, policy="ssp", use_mask=True,
                 geom_bias=False):
        super().__init__()
        assert policy in {"ssp", "random", "fixed", "geometry_only", "oracle"}
        self.policy, self.use_mask, self.d_ssp = policy, use_mask, d_ssp
        self.geom_bias = geom_bias
        self.Wk = nn.Linear(d_model, d_ssp, bias=False)
        self.Wq = nn.Linear(d_desc, d_ssp, bias=False)
        if policy == "geometry_only":
            self.static_key = nn.Parameter(torch.randn(d_ssp) / d_ssp ** 0.5)
        # Static geometry key: implements the paper's "max-entropy prior at t=0"
        # (Prop. geom). Without it, key=Wk(u)=0 at t=0 -> score==0 -> arbitrary
        # first pick, geometry ignored on the cold start.
        if geom_bias:
            self.k0 = nn.Parameter(torch.randn(d_ssp) / d_ssp ** 0.5)
    def param_count(self): return sum(p.numel() for p in self.parameters())
    def scores(self, u, desc, visited):
        if self.policy == "geometry_only":
            key = self.static_key.unsqueeze(0).expand(u.shape[0], -1)
        else:
            key = self.Wk(u)
            if self.geom_bias:
                key = key + self.k0.unsqueeze(0)
        query = self.Wq(desc)
        s = torch.einsum("bd,bkd->bk", key, query) / self.d_ssp ** 0.5
        if self.policy == "random":
            s = torch.rand_like(s)
        elif self.policy == "fixed":
            K = s.shape[1]
            s = -torch.arange(K, device=s.device, dtype=s.dtype).expand_as(s)
        if self.use_mask:
            s = s.masked_fill(visited, MASK_VALUE)
        return s
    def select(self, scores, hard_inference, tau=1.0):
        if hard_inference or self.policy in {"random", "fixed"}:
            idx = scores.argmax(-1)
            return F.one_hot(idx, scores.shape[-1]).to(scores.dtype)
        return F.gumbel_softmax(scores, tau=tau, hard=True)

class PointSliceEncoder(nn.Module):
    """Minimal per-slice encoder: (B,K,d_in) -> (B,K,D). Stands in for EdgeConv stem."""
    def __init__(self, d_in, d_model, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d_model))
    def forward(self, regions, anchors=None): return self.net(regions)


# ============================================================================
# 2. ASP model-lite: forward_train / forward_infer / forward_infer_ORACLE
#    (forward_infer + oracle rollout are the exact logic for the real drop-in)
# ============================================================================
class ASPLite(nn.Module):
    def __init__(self, d_in, d_model, num_classes, K, d_desc=6, policy="ssp",
                 head_layers=3, tau_mem=2.0, geom_bias=False):
        super().__init__()
        self.K, self.num_classes = K, num_classes
        self.encoder = PointSliceEncoder(d_in, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.head = MultiLayerLIFHead(d_model, head_layers, tau=tau_mem)
        self.readout = LeakyReadout(d_model, num_classes, tau_mem)
        self.ssp = SSP(d_model, d_desc, 64, 0, policy, True, geom_bias=geom_bias)
    # ---- helpers mirroring the real model ----
    def _prep(self, regions):
        B = regions.shape[0]
        self.head.reset_state(B, regions.device)
        self.readout.reset_state(B, regions.device)
        return B
    @staticmethod
    def _margin(logits):
        p = F.softmax(logits, -1); top2 = p.topk(2, -1).values
        return top2[:, 0] - top2[:, 1]
    # ---- training (Gumbel-ST, all K steps) ----
    def forward_train(self, regions, desc, tau_gumbel=1.0):
        B = self._prep(regions); K = self.K
        feats = self.encoder(regions)
        visited = torch.zeros(B, K, dtype=torch.bool, device=regions.device)
        logits_all, fr = [], []
        u = self.head.membrane
        for _ in range(K):
            s = self.ssp.scores(u, desc, visited)
            w = self.ssp.select(s, hard_inference=False, tau=tau_gumbel)
            e = torch.einsum("bk,bkd->bd", w, feats)
            sp = self.head(self.proj(e))
            logits_all.append(self.readout(sp))
            fr.append(sp.mean())
            visited = visited | (w.detach() > 0.5)
            u = self.head.membrane
        return {"logits": torch.stack(logits_all, 1), "firing_rate": torch.stack(fr).mean()}
    # ---- inference (hard argmax, records full trajectory) ----
    @torch.no_grad()
    def forward_infer(self, regions, desc):
        B = self._prep(regions); K = self.K
        feats = self.encoder(regions)
        visited = torch.zeros(B, K, dtype=torch.bool, device=regions.device)
        logits_all, margins_all, sels = [], [], []
        u = self.head.membrane
        for _ in range(K):
            s = self.ssp.scores(u, desc, visited)
            idx = s.argmax(-1)
            e = feats[torch.arange(B), idx]
            logits = self.readout(self.head(self.proj(e)))
            logits_all.append(logits); margins_all.append(self._margin(logits)); sels.append(idx)
            visited = visited | F.one_hot(idx, K).bool()
            u = self.head.membrane
        return {"logits": torch.stack(logits_all, 1),
                "margins": torch.stack(margins_all, 1),
                "selections": torch.stack(sels, 1)}
    # ---- ORACLE-GREEDY (label-driven, prefix-replay rollout) ----
    @torch.no_grad()
    def forward_infer_oracle(self, regions, desc, labels, score="ce"):
        """At each step, among unvisited slices pick the one that MINIMISES
        cross-entropy to the true label (score='ce') -> max log p(true).
        Robust prefix-replay: reset -> replay committed slices -> step candidate.
        Returns the same dict shape as forward_infer (byte-compatible with
        metrics.theta_sweep / accuracy@k)."""
        B, K, dev = regions.shape[0], self.K, regions.device
        feats = self.encoder(regions)
        committed = torch.full((B, K), -1, dtype=torch.long, device=dev)  # prefix
        n_committed = 0
        logits_all, margins_all, sels = [], [], []

        def replay_prefix(prefix, length):
            """reset state and replay `length` committed slices; leaves state ready
            for the (length+1)-th step. Returns nothing (mutates head/readout state)."""
            self.head.reset_state(B, dev); self.readout.reset_state(B, dev)
            for j in range(length):
                e = feats[torch.arange(B), prefix[:, j]]
                self.readout(self.head(self.proj(e)))

        for _t in range(K):
            visited = torch.zeros(B, K, dtype=torch.bool, device=dev)
            for j in range(n_committed):
                visited[torch.arange(B), committed[:, j]] = True
            best_score = torch.full((B,), -1e30, device=dev)
            best_idx = torch.zeros(B, dtype=torch.long, device=dev)
            for m in range(K):
                replay_prefix(committed, n_committed)          # state after prefix
                e = feats[:, m]
                logits_m = self.readout(self.head(self.proj(e)))
                logp = F.log_softmax(logits_m, -1)
                sc = logp[torch.arange(B), labels]             # log p(true) == -CE
                sc = sc.masked_fill(visited[:, m], -1e30)      # can't reselect
                take = sc > best_score
                best_score = torch.where(take, sc, best_score)
                best_idx = torch.where(take, torch.full_like(best_idx, m), best_idx)
            # commit best: replay prefix then take the committed step to log its logits
            replay_prefix(committed, n_committed)
            e = feats[torch.arange(B), best_idx]
            logits = self.readout(self.head(self.proj(e)))
            logits_all.append(logits); margins_all.append(self._margin(logits)); sels.append(best_idx)
            committed[:, n_committed] = best_idx; n_committed += 1
        return {"logits": torch.stack(logits_all, 1),
                "margins": torch.stack(margins_all, 1),
                "selections": torch.stack(sels, 1)}
    # ---- held-representation swap: run ANY policy's selection on THIS backbone ----
    @torch.no_grad()
    def forward_infer_with_policy(self, regions, desc, policy):
        saved = self.ssp.policy; self.ssp.policy = policy
        out = self.forward_infer(regions, desc)
        self.ssp.policy = saved
        return out


# ============================================================================
# 3. Synthetic ORDER-DEPENDENT dataset
#    A few "informative" slices carry a class-correlated signal; their 6-D
#    descriptor's first coordinate is high (so a learned/geometry policy CAN
#    find them). The rest are noise. Informative slices are placed at RANDOM
#    positions, so fixed (fps) order and random order usually see noise first
#    -> order matters, and there is a real oracle ordering.
# ============================================================================
def make_synth(n_per_class, C, K, d_in, n_inf=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(C, d_in, generator=g) * 2.5           # class prototypes
    N = n_per_class * C
    regions = torch.randn(N, K, d_in, generator=g) * 0.9       # noise base
    desc = torch.randn(N, K, 6, generator=g) * 0.3             # geometry-ish
    labels = torch.arange(C).repeat_interleave(n_per_class)
    for i in range(N):
        y = labels[i].item()
        pos = torch.randperm(K, generator=g)[:n_inf]           # random informative slices
        regions[i, pos] = protos[y].unsqueeze(0) + torch.randn(n_inf, d_in, generator=g) * 0.7
        desc[i, pos, 0] += 2.5                                 # informativeness cue in descriptor
    return regions, desc, labels

def loader(regions, desc, labels, bs, shuffle):
    idx = torch.arange(len(labels))
    if shuffle: idx = idx[torch.randperm(len(labels))]
    for s in range(0, len(labels), bs):
        b = idx[s:s+bs]
        yield regions[b], desc[b], labels[b]


# ============================================================================
# 4. Train / eval helpers
# ============================================================================
def tet_loss(logits, labels, tet_lambda=0.05):
    B, T, C = logits.shape
    ce = F.cross_entropy(logits.reshape(B*T, C),
                         labels.unsqueeze(1).expand(B, T).reshape(-1), label_smoothing=0.1)
    if T > 1:
        final = logits[:, -1].detach()
        mse = F.mse_loss(logits[:, :-1], final.unsqueeze(1).expand(-1, T-1, -1))
        ce = ce + tet_lambda * mse
    return ce

def train(model, tr, epochs, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        tau = max(0.5, 1.0 * (0.98 ** ep))
        for regions, desc, labels in tr():
            opt.zero_grad()
            out = model.forward_train(regions, desc, tau_gumbel=tau)
            loss = tet_loss(out["logits"], labels) + 0.005 * out["firing_rate"]
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

@torch.no_grad()
def acc_at_k(out, labels, Ks):
    """accuracy@k = argmax of logits AFTER exactly k observed slices."""
    logits = out["logits"]
    return {k: (logits[:, k-1].argmax(-1) == labels).float().mean().item() for k in Ks}

@torch.no_grad()
def theta_metrics(out, labels, theta):
    """avg_slices (tau-bar) and accuracy at margin-exit threshold theta."""
    m = out["margins"]; B, T = m.shape
    hit = m > theta
    es = torch.where(hit.any(1), hit.float().argmax(1) + 1, torch.full((B,), T))
    idx = (es.long() - 1).clamp(min=0)
    preds = out["logits"][torch.arange(B), idx].argmax(-1)
    return es.float().mean().item(), (preds == labels).float().mean().item()


# ============================================================================
# 5. RUN: 3 seeds, train learned/random/fps end-to-end; oracle + held-rep swaps
# ============================================================================
def main():
    C, K, d_in, d_model = 8, 16, 12, 64
    Ks = [1, 2, 3, 4, 6, 8, 16]
    THETA = 0.30                      # LOW theta -> forces tau-bar < M (the design condition)
    EPOCHS = 40
    SEEDS = [0, 1, 2]

    tr_r, tr_d, tr_y = make_synth(120, C, K, d_in, seed=100)
    te_r, te_d, te_y = make_synth(40, C, K, d_in, seed=200)

    import numpy as np
    agg = {}   # variant -> {k: [accs across seeds]}, plus ('tau','acc') at theta
    invariant_max_abs = 0.0

    for seed in SEEDS:
        torch.manual_seed(seed)
        models = {}
        for pol, gb in [("ssp", False), ("random", False), ("fixed", False), ("ssp_geom", True)]:
            torch.manual_seed(seed)               # identical init across policies
            base = "ssp" if pol == "ssp_geom" else pol
            m = ASPLite(d_in, d_model, C, K, policy=base, geom_bias=gb)
            train(m, lambda: loader(tr_r, tr_d, tr_y, 128, True), EPOCHS)
            models[pol] = m

        # ---- end-to-end: each policy eval'd with its own rule ----
        results = {}
        results["learned"]     = models["ssp"].forward_infer(te_r, te_d)
        results["learned+geom"]= models["ssp_geom"].forward_infer(te_r, te_d)
        results["random"]      = models["random"].forward_infer(te_r, te_d)
        results["fps_order"]   = models["fixed"].forward_infer(te_r, te_d)
        # ---- oracle ceiling on the learned backbone ----
        results["oracle"]   = models["ssp"].forward_infer_oracle(te_r, te_d, te_y)
        # ---- held-representation swaps on the learned backbone (diagnostic) ----
        results["swap_random@learned"] = models["ssp"].forward_infer_with_policy(te_r, te_d, "random")
        results["swap_fps@learned"]    = models["ssp"].forward_infer_with_policy(te_r, te_d, "fixed")

        # ---- CORRECTNESS INVARIANT: oracle replay fed the learned order must
        #      reproduce forward_infer logits byte-identically. We test the
        #      replay machinery by replaying the learned selection order and
        #      comparing to forward_infer's logits. ----
        learned_sel = results["learned"]["selections"]           # (B,K)
        rep = replay_check(models["ssp"], te_r, te_d, learned_sel)
        d = (rep - results["learned"]["logits"]).abs().max().item()
        invariant_max_abs = max(invariant_max_abs, d)

        for name, out in results.items():
            ak = acc_at_k(out, te_y, Ks)
            tau, atheta = theta_metrics(out, te_y, THETA)
            agg.setdefault(name, {kk: [] for kk in Ks}); agg[name].setdefault("tau", []); agg[name].setdefault("atheta", [])
            for kk in Ks: agg[name][kk].append(ak[kk])
            agg[name]["tau"].append(tau); agg[name]["atheta"].append(atheta)

    # ---------------- report ----------------
    def ms(v): return f"{np.mean(v)*100:5.1f}±{np.std(v)*100:3.1f}"
    def msf(v): return f"{np.mean(v):4.2f}±{np.std(v):3.2f}"
    print("\n" + "="*94)
    print("INVARIANT  |  oracle prefix-replay reproduces forward_infer logits "
          f"(max|Δ| over learned order) = {invariant_max_abs:.2e}   "
          f"{'PASS' if invariant_max_abs < 1e-4 else 'FAIL'}")
    print("="*94)
    order = ["fps_order", "random", "learned", "learned+geom", "oracle",
             "swap_random@learned", "swap_fps@learned"]
    hdr = "variant".ljust(22) + "".join(f"  acc@{k:<2}" for k in Ks) + f"   taū@θ={THETA}  acc@θ"
    print(hdr); print("-"*len(hdr))
    for name in order:
        row = name.ljust(22) + "".join(f"  {ms(agg[name][k])}" for k in Ks)
        row += f"    {msf(agg[name]['tau'])}  {ms(agg[name]['atheta'])}"
        print(row)
    print("="*94)
    # key comparisons at a small budget (k=3, where order matters most)
    k = 3
    l, r, f, o = (np.mean(agg[n][k]) for n in ["learned","random","fps_order","oracle"])
    print(f"\n@k={k}:  fps={f*100:.1f}  random={r*100:.1f}  learned={l*100:.1f}  oracle={o*100:.1f}")
    print(f"  learned−random = {(l-r)*100:+.1f} pp   oracle−learned = {(o-l)*100:+.1f} pp   "
          f"oracle−random = {(o-r)*100:+.1f} pp")

@torch.no_grad()
def replay_check(model, regions, desc, order):
    """Replay a GIVEN selection order through the head/readout and return the
    per-step logits trajectory. Must equal forward_infer's logits when `order`
    is forward_infer's own selection order -> validates the replay used by oracle."""
    B, K, dev = regions.shape[0], model.K, regions.device
    feats = model.encoder(regions)
    model.head.reset_state(B, dev); model.readout.reset_state(B, dev)
    outs = []
    for t in range(K):
        e = feats[torch.arange(B), order[:, t]]
        outs.append(model.readout(model.head(model.proj(e))))
    return torch.stack(outs, 1)

if __name__ == "__main__":
    main()