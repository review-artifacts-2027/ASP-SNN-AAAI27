from __future__ import annotations

import torch


@torch.no_grad()
def forward_infer_oracle(model, regions, desc, anchors_xyz=None, labels=None):
    assert labels is not None, "oracle-greedy needs ground-truth labels"
    model.eval()
    K = model.cfg.k_slices

    desc, B = model._prep(regions, desc)
    device = regions.device
    labels = labels.to(device)

    feats = model.encoder(regions, anchors_xyz)
    ctx = model._global_ctx(feats)

    def step_feature(e_bd):
        h = model.proj(e_bd)
        if ctx is not None:
            h = h + model.ctx_gate * ctx
        return model.readout(model.head(h))

    def prime_to_prefix(prefix, length):
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
        visited = torch.zeros(B, K, dtype=torch.bool, device=device)
        for j in range(n_committed):
            visited[ar, committed[:, j]] = True

        best_score = torch.full((B,), -1e30, device=device)
        best_idx = torch.zeros(B, dtype=torch.long, device=device)
        for m in range(K):
            prime_to_prefix(committed, n_committed)
            logits_m = step_feature(feats[:, m])
            logp_true = torch.log_softmax(logits_m, -1)[ar, labels]
            logp_true = logp_true.masked_fill(visited[:, m], -1e30)
            take = logp_true > best_score
            best_score = torch.where(take, logp_true, best_score)
            best_idx = torch.where(take, torch.full_like(best_idx, m), best_idx)

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
    saved = model.ssp.policy
    if policy is not None:
        model.ssp.policy = policy
    try:
        outs = {"logits": [], "margins": [], "selections": [], "labels": []}
        for batch in loader:
            regions, desc, anchors, labels = [b.to(device) if b is not None else None
                                              for b in batch]
            o = model.forward_infer(regions, desc, anchors, theta=2.0)
            outs["logits"].append(o["logits"])
            outs["margins"].append(o["margins"])
            outs["selections"].append(o["selections"])
            outs["labels"].append(labels)
        return {k: torch.cat(v) for k, v in outs.items() if v}
    finally:
        model.ssp.policy = saved
