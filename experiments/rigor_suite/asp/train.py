"""Training loop: composite loss L = L_CE + lambda_exit*L_exit + lambda_sparse*L_sparse
(+ optional logit-KD hook), Gumbel temperature annealing 1.0 -> 0.5 (proposal 11.3).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .model import ASPConfig, ASPModel


def composite_loss(out: dict, labels: torch.Tensor, lambda_exit: float = 0.1,
                   lambda_sparse: float = 0.01, entropy_beta: float = 0.1,
                   teacher_logits: torch.Tensor | None = None,
                   lambda_kd: float = 0.5, kd_temp: float = 4.0,
                   kd_schedule: str = "ramp") -> torch.Tensor:
    logits = out["logits"]                                  # (B,T,C)
    B, T, C = logits.shape
    ce = F.cross_entropy(logits.reshape(B * T, C),
                         labels.repeat_interleave(T) if False else
                         labels.unsqueeze(1).expand(B, T).reshape(-1))
    p = F.softmax(logits, dim=-1)
    pmax = p.amax(-1).clamp_min(1e-8)
    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
    l_exit = (-pmax.log() + entropy_beta * ent).mean()
    loss = ce + lambda_exit * l_exit + lambda_sparse * out["firing_rate"]
    if teacher_logits is not None:                          # logit KD (optional)
        # Which steps to distil. The teacher sees all K slices; at step t the
        # student has seen t of them, so demanding an exact match early is not
        # achievable and the residual just fights the CE term.
        #   final -> only t=T (teacher and student see the same evidence)
        #   ramp  -> weight t/T, so supervision grows with available evidence
        #   all   -> uniform over t (strongest, most mismatched)
        s = F.log_softmax(logits / kd_temp, -1)                       # (B,T,C)
        tp = F.softmax(teacher_logits / kd_temp, -1).unsqueeze(1)     # (B,1,C)
        kl = F.kl_div(s, tp.expand(-1, T, -1), reduction="none").sum(-1)   # (B,T)
        if kd_schedule == "final":
            w = torch.zeros(T, device=logits.device); w[-1] = 1.0
        elif kd_schedule == "ramp":
            w = torch.arange(1, T + 1, device=logits.device,
                             dtype=logits.dtype) / T
        else:                                                  # "all"
            w = torch.ones(T, device=logits.device, dtype=logits.dtype)
        w = w / w.sum()
        kd = (kl * w.unsqueeze(0)).sum(-1).mean()
        loss = loss + lambda_kd * kd * kd_temp ** 2
    return loss


def anneal_tau(epoch: int, total: int, start: float = 1.0, end: float = 0.5) -> float:
    return start + (end - start) * min(epoch / max(total - 1, 1), 1.0)


def train_model(cfg: dict, train_loader, test_loader, device: str = "cpu",
                log_fn=print) -> tuple[ASPModel, list[dict]]:
    mcfg = ASPConfig.from_dict(cfg)
    model = ASPModel(mcfg).to(device)
    epochs = cfg.get("epochs", 30)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                           weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for ep in range(epochs):
        model.train()
        tau = anneal_tau(ep, epochs, cfg.get("tau_start", 1.0), cfg.get("tau_end", 0.5))
        t0, tot, n = time.time(), 0.0, 0
        for batch in train_loader:
            regions, desc, anchors, labels = [b.to(device) if b is not None else None
                                              for b in batch]
            out = model.forward_train(regions, desc, anchors, tau_gumbel=tau)
            loss = composite_loss(out, labels,
                                  cfg.get("lambda_exit", 0.1),
                                  cfg.get("lambda_sparse", 0.01))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * labels.shape[0]; n += labels.shape[0]
        sched.step()
        row = {"epoch": ep, "tau_gumbel": tau, "train_loss": tot / n,
               "sec": time.time() - t0}
        if test_loader is not None and (ep % cfg.get("eval_every", 5) == 0
                                        or ep == epochs - 1):
            from .eval import evaluate
            ev = evaluate(model, test_loader, device,
                          thetas=[mcfg.theta])
            row.update({"test_acc_full": ev["acc_full_T"],
                        "test_acc_theta": ev["theta_rows"][0]["accuracy"],
                        "avg_slices": ev["theta_rows"][0]["avg_slices"]})
        history.append(row)
        log_fn(f"[ep {ep:03d}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                            else f"{k}={v}" for k, v in row.items()))
    return model, history
