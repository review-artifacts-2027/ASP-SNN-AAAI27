from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .model import ASPConfig, ASPModel


def composite_loss(out: dict, labels: torch.Tensor,
                   lambda_exit: float = 0.0,
                   lambda_sparse: float = 0.005,
                   entropy_beta: float = 0.1,
                   teacher_logits: torch.Tensor | None = None,
                   lambda_kd: float = 0.5, kd_temp: float = 4.0,
                   label_smoothing: float = 0.1,
                   *,
                   loss_mode: str = "tet",
                   tet_lambda: float = 0.05,
                   ) -> torch.Tensor:

    logits = out["logits"]
    B, T, C = logits.shape

    if labels.ndim == 1:
        tgt = labels.unsqueeze(1).expand(B, T).reshape(-1)
        ce = F.cross_entropy(
            logits.reshape(B * T, C), tgt,
            label_smoothing=label_smoothing,
        )
    elif labels.ndim == 2:
        if labels.shape != (B, C):
            raise ValueError(
                f"soft-target labels have shape {tuple(labels.shape)}, "
                f"expected (B={B}, C={C})"
            )
        tgt = labels.unsqueeze(1).expand(B, T, C).reshape(B * T, C)
        ce = F.cross_entropy(logits.reshape(B * T, C), tgt)
    else:
        raise ValueError(
            f"labels must be (B,) or (B, C); got shape {tuple(labels.shape)}"
        )
    loss = ce

    if loss_mode == "tet":
        if tet_lambda > 0.0 and T > 1:
            final_sg = logits[:, -1, :].detach()
            earlier  = logits[:, :-1, :]
            mse = F.mse_loss(
                earlier,
                final_sg.unsqueeze(1).expand_as(earlier),
                reduction="mean",
            )
            loss = loss + tet_lambda * mse
    elif loss_mode == "legacy":
        if lambda_exit > 0.0:
            p = F.softmax(logits, dim=-1)
            pmax = p.amax(-1).clamp_min(1e-8)
            ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
            l_exit = (-pmax.log() + entropy_beta * ent).mean()
            loss = loss + lambda_exit * l_exit
    else:
        raise ValueError(
            f"composite_loss: unknown loss_mode={loss_mode!r} "
            f"(expected 'tet' or 'legacy')"
        )

    if lambda_sparse > 0.0 and "firing_rate" in out:
        loss = loss + lambda_sparse * out["firing_rate"]

    if teacher_logits is not None:
        if teacher_logits.shape != (B, C):
            raise ValueError(
                f"teacher_logits shape {tuple(teacher_logits.shape)} "
                f"!= (B={B}, C={C})"
            )
        s = F.log_softmax(logits[:, -1] / kd_temp, dim=-1)
        t = F.softmax(teacher_logits / kd_temp, dim=-1)
        loss = loss + lambda_kd * F.kl_div(
            s, t, reduction="batchmean") * (kd_temp ** 2)

    return loss


def anneal_tau(epoch: int, total: int,
               start: float = 1.0, end: float = 0.5) -> float:

    return start + (end - start) * min(epoch / max(total - 1, 1), 1.0)


def build_scheduler(optimizer: torch.optim.Optimizer,
                    epochs: int,
                    warmup_epochs: int,
                    min_lr_ratio: float = 0.01) -> torch.optim.lr_scheduler.LambdaLR:

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _amp_supported(device: torch.device | str) -> bool:
    if isinstance(device, str):
        if device == "cpu":
            return False
        dev = torch.device(device)
    else:
        dev = device
    if dev.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _autocast_ctx(device: torch.device | str, dtype: torch.dtype):
    dev_str = device if isinstance(device, str) else device.type
    dev_type = "cuda" if dev_str != "cpu" else "cpu"
    try:
        return torch.amp.autocast(device_type=dev_type, dtype=dtype)
    except (AttributeError, TypeError):
        if dev_type == "cuda":
            return torch.cuda.amp.autocast(dtype=dtype)
        return nullcontext()


def train_model(cfg: dict, train_loader, test_loader,
                device: str = "cpu", log_fn=print) -> tuple[ASPModel, list[dict]]:

    mcfg = ASPConfig.from_dict(cfg)
    model = ASPModel(mcfg).to(device)

    epochs         = int(cfg.get("epochs", 300))
    warmup_epochs  = int(cfg.get("warmup_epochs", 20))
    lr             = float(cfg.get("lr", 1e-3))
    weight_decay   = float(cfg.get("weight_decay", 0.05))
    min_lr_ratio   = float(cfg.get("min_lr_ratio", 0.01))
    label_smooth   = float(cfg.get("label_smoothing", 0.1))
    grad_clip      = float(cfg.get("grad_clip", 1.0))

    loss_mode      = str(cfg.get("loss_mode", "tet")).lower()
    tet_lambda     = float(cfg.get("tet_lambda", 0.05))
    lambda_sparse  = float(cfg.get("lambda_sparse", 0.005))
    lambda_exit    = float(cfg.get("lambda_exit", 0.0))
    entropy_beta   = float(cfg.get("entropy_beta", 0.1))

    mixup_alpha    = float(cfg.get("mixup_alpha", 0.0))
    cutmix_alpha   = float(cfg.get("cutmix_alpha", 0.0))
    cutmix_prob    = float(cfg.get("cutmix_prob", 0.5))
    mix_prob       = float(cfg.get("mix_prob", 1.0))
    mix_off_epoch  = cfg.get("mix_off_epoch", None)
    mix_off_epoch  = 2 * epochs // 3 if mix_off_epoch is None else int(mix_off_epoch)
    mix_grid_hw    = tuple(cfg.get("mix_grid_hw", (4, 4)))
    mix_patch_hw   = tuple(cfg.get("mix_patch_hw", (8, 8)))
    mix_in_ch      = int(cfg.get("mix_in_channels", 3))

    lambda_kd      = float(cfg.get("lambda_kd", 0.5))
    kd_temp        = float(cfg.get("kd_temp", 4.0))

    tau_start      = float(cfg.get("tau_start", 1.0))
    tau_end        = float(cfg.get("tau_end", 0.5))
    eval_every     = int(cfg.get("eval_every", 5))

    if loss_mode not in ("tet", "legacy"):
        raise ValueError(f"loss_mode must be 'tet' or 'legacy', got {loss_mode!r}")

    use_amp_req = bool(cfg.get("use_amp", True))
    use_amp     = use_amp_req and _amp_supported(device)
    if use_amp_req and not use_amp:
        log_fn("[amp] bf16 requested but not supported on this device; "
               "training in fp32.")

    mixer = None

    teacher = None
    if cfg.get("teacher_ckpt"):
        raise ValueError(
            "teacher_ckpt is not supported by the public point-cloud archive"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer, epochs=epochs, warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
    )

    log_fn(
        f"[train] optim=AdamW lr={lr:.2e} wd={weight_decay} "
        f"warmup={warmup_epochs} epochs={epochs} "
        f"label_smooth={label_smooth} grad_clip={grad_clip} "
        f"amp={'bf16' if use_amp else 'off'}"
    )
    log_fn(
        f"[loss] mode={loss_mode} "
        + (f"tet_lambda={tet_lambda} " if loss_mode == "tet" else "")
        + f"lambda_sparse={lambda_sparse}"
    )
    if mixer is not None:
        log_fn(
            f"[mix]  mixup_alpha={mixup_alpha} cutmix_alpha={cutmix_alpha} "
            f"cutmix_prob={cutmix_prob} mix_prob={mix_prob} "
            f"mix_off_epoch={mix_off_epoch}"
        )
    else:
        log_fn("[mix]  disabled")
    if teacher is not None:
        log_fn(f"[kd]   lambda_kd={lambda_kd} kd_temp={kd_temp}")

    history: list[dict] = []

    for ep in range(epochs):
        model.train()
        tau = anneal_tau(ep, epochs, tau_start, tau_end)
        mixer_active = (mixer is not None) and (ep < mix_off_epoch)

        t0, tot, n = time.time(), 0.0, 0
        for batch in train_loader:
            regions, desc, anchors, labels = [
                b.to(device) if b is not None else None for b in batch
            ]

            if mixer_active:
                regions, desc, anchors, labels = mixer(
                    regions, desc, anchors, labels
                )

            teacher_logits = None

            optimizer.zero_grad(set_to_none=True)

            ctx = _autocast_ctx(device, torch.bfloat16) if use_amp else nullcontext()
            with ctx:
                out = model.forward_train(regions, desc, anchors,
                                          tau_gumbel=tau)
                loss = composite_loss(
                    out, labels,
                    lambda_exit=lambda_exit,
                    lambda_sparse=lambda_sparse,
                    entropy_beta=entropy_beta,
                    teacher_logits=teacher_logits,
                    lambda_kd=lambda_kd, kd_temp=kd_temp,
                    label_smoothing=label_smooth,
                    loss_mode=loss_mode,
                    tet_lambda=tet_lambda,
                )

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bsz = regions.shape[0]
            tot += loss.item() * bsz
            n += bsz

        scheduler.step()

        row = {
            "epoch": ep,
            "tau_gumbel": tau,
            "mix_on": mixer_active,
            "kd_on": teacher is not None,
            "train_loss": tot / max(n, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }

        if test_loader is not None and (ep % eval_every == 0 or ep == epochs - 1):
            from .eval import evaluate
            ev = evaluate(model, test_loader, device, thetas=[mcfg.theta])
            row.update({
                "test_acc_full":  ev["acc_full_T"],
                "test_acc_theta": ev["theta_rows"][0]["accuracy"],
                "avg_slices":     ev["theta_rows"][0]["avg_slices"],
            })

        history.append(row)
        log_fn(
            f"[ep {ep:03d}] "
            + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
            )
        )

    return model, history
