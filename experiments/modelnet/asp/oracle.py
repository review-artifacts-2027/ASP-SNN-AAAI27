"""
asp/oracle.py — label-driven oracle-greedy selection (upper bound) for the
M=16 selection ablation (paper Supplementary D.3: the "oracle-greedy upper
bound" selection rule).

Design
------
At each acquisition step t, among UNVISITED slices, pick the slice whose
observation MINIMISES cross-entropy to the true label — i.e. maximises
log p(true | observed prefix + candidate). This is the greedy oracle that the
paper's "near-optimality of greedy selection" theorem references. It uses the
label, so it is an EVALUATION-time ceiling, not a trainable policy.

The rollout is done by PREFIX-REPLAY through the model's PUBLIC API only
(reset_state / forward), so it is robust to the exact internal state layout of
LIFCell / MultiLayerLIFHead / LeakyReadout and needs NO edits to model.py:

    for each step t:
        for each candidate slice m (unvisited):
            reset head+readout ; replay the t committed slices ; step slice m
            score(m) = log p_softmax(true)                # -CE
        commit argmax_m score(m)  (replay prefix once more, take the step,
                                   record its logits/margin as the trajectory)

Correctness invariant (validated on CPU in validate_oracle.py):
    replaying forward_infer's OWN selection order through this machinery
    reproduces forward_infer's logits byte-identically (max|Δ| = 0).

Cost: O(K * tau_bar^2) head-forwards over the episode. With K=16 and
tau_bar<=16 this is ~2k tiny head-forwards per batch — negligible.

The returned dict matches forward_infer / asp.eval.collect exactly
(keys: logits, margins, selections), so asp.metrics.theta_sweep,
exits_from_margins, exit_predictions, and asp.anytime.anytime_curve all work
on it unchanged.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def forward_infer_oracle(model, regions, desc, anchors_xyz=None, labels=None):
    """Oracle-greedy hard loop. `labels` (B,) is REQUIRED (it is the oracle
    signal). Returns {logits:(B,K,C), margins:(B,K), selections:(B,K)}."""
    assert labels is not None, "oracle-greedy needs ground-truth labels"
    model.eval()
    K = model.cfg.k_slices
    # _prep applies mask_descriptor(drop_desc) and resets head+readout; reuse it
    # for the masked descriptor + batch size, then manage our own resets below.
    desc, B = model._prep(regions, desc)
    device = regions.device
    labels = labels.to(device)

    # Encoder runs on ALL K slices once (oracle is inherently non-streaming).
    feats = model.encoder(regions, anchors_xyz)                     # (B, K, D)
    ctx = model._global_ctx(feats)                                  # None unless Phase-6

    def step_feature(e_bd):
        """One head+readout step from the CURRENT (already-primed) state."""
        h = model.proj(e_bd)
        if ctx is not None:
            h = h + model.ctx_gate * ctx
        return model.readout(model.head(h))

    def prime_to_prefix(prefix, length):
        """Reset, then replay `length` committed slices so state is ready for the
        (length+1)-th step. prefix: (B, K) long; only [:, :length] is used."""
        model.head.reset_state(B, device)
        model.readout.reset_state(B, device)
        ar = torch.arange(B, device=device)
        for j in range(length):
            step_feature(feats[ar, prefix[:, j]])

    ar = torch.arange(B, device=device)
    committed = torch.zeros(B, K, dtype=torch.long, device=device)
    n_committed = 0
    logits_all, margins_all, sels = [], [], []

    for _t in range(K):
        # visited mask from committed prefix
        visited = torch.zeros(B, K, dtype=torch.bool, device=device)
        for j in range(n_committed):
            visited[ar, committed[:, j]] = True

        best_score = torch.full((B,), -1e30, device=device)
        best_idx = torch.zeros(B, dtype=torch.long, device=device)
        for m in range(K):
            prime_to_prefix(committed, n_committed)
            logits_m = step_feature(feats[:, m])
            logp_true = torch.log_softmax(logits_m, -1)[ar, labels]  # -CE
            logp_true = logp_true.masked_fill(visited[:, m], -1e30)
            take = logp_true > best_score
            best_score = torch.where(take, logp_true, best_score)
            best_idx = torch.where(take, torch.full_like(best_idx, m), best_idx)

        # Commit: re-prime prefix, take the winning step, record its trajectory.
        prime_to_prefix(committed, n_committed)
        logits_t = step_feature(feats[ar, best_idx])
        margin, _ = model._margin_entropy(logits_t)
        logits_all.append(logits_t)
        margins_all.append(margin)
        sels.append(best_idx)
        committed[:, n_committed] = best_idx
        n_committed += 1

    return {
        "logits": torch.stack(logits_all, 1),
        "margins": torch.stack(margins_all, 1),
        "selections": torch.stack(sels, 1),
    }


@torch.no_grad()
def collect_oracle(model, loader, device="cpu"):
    """Mirror of asp.eval.collect but for the oracle rule (passes labels in)."""
    model.eval()
    outs = {"logits": [], "margins": [], "selections": [], "labels": []}
    for batch in loader:
        regions, desc, anchors, labels = [b.to(device) if b is not None else None
                                          for b in batch]
        o = forward_infer_oracle(model, regions, desc, anchors, labels)
        outs["logits"].append(o["logits"])
        outs["margins"].append(o["margins"])
        outs["selections"].append(o["selections"])
        outs["labels"].append(labels)
    return {k: torch.cat(v) for k, v in outs.items() if v}


@torch.no_grad()
def collect_with_policy(model, loader, device="cpu", policy=None):
    """Held-representation swap: run ANY SSP policy ('ssp'|'random'|'fixed'|
    'geometry_only') on the model's CURRENT (frozen) backbone. Reuses the
    standard hard-argmax loop via forward_infer. Restores the original policy.
    Returns the same dict shape as asp.eval.collect (logits/margins/selections/labels)."""
    saved = model.ssp.policy
    if policy is not None:
        model.ssp.policy = policy
    try:
        outs = {"logits": [], "margins": [], "selections": [], "labels": []}
        for batch in loader:
            regions, desc, anchors, labels = [b.to(device) if b is not None else None
                                              for b in batch]
            o = model.forward_infer(regions, desc, anchors, theta=2.0)  # never early-exit
            outs["logits"].append(o["logits"])
            outs["margins"].append(o["margins"])
            outs["selections"].append(o["selections"])
            outs["labels"].append(labels)
        return {k: torch.cat(v) for k, v in outs.items() if v}
    finally:
        model.ssp.policy = saved