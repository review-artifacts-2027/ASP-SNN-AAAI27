"""
train_imagenet_foveater.py - Train FoveaTer-style ASP on ImageNet.

Usage:
    python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml
    python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml --set smoke=true epochs=1 batch_size=2
"""

import math
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from config import base_argparser, load_config, parse_overrides, set_seed
from datasets.imagenet import build_imagenet_loaders
from models.foveater_asp import FoveaTerASP


def _top1(logits, labels):
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def _make_smoke_loaders(cfg):
    n_train = max(int(cfg.batch_size) * 2, 8)
    n_val = max(int(cfg.batch_size), 4)
    train_x = torch.randn(n_train, 3, cfg.image_size, cfg.image_size)
    train_y = torch.randint(0, cfg.num_classes, (n_train,))
    val_x = torch.randn(n_val, 3, cfg.image_size, cfg.image_size)
    val_y = torch.randint(0, cfg.num_classes, (n_val,))
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, cfg.num_classes


def _active_image_loss(logits_final, logits_all, labels, cfg):
    ce = F.cross_entropy(
        logits_final,
        labels,
        label_smoothing=getattr(cfg, "label_smooth", 0.0),
    )

    aux = logits_final.new_tensor(0.0)
    if len(logits_all) > 1:
        for logits_t in logits_all[:-1]:
            aux = aux + F.cross_entropy(
                logits_t,
                labels,
                label_smoothing=getattr(cfg, "label_smooth", 0.0),
            )
        aux = aux / (len(logits_all) - 1)

    exit_loss = logits_final.new_tensor(0.0)
    if getattr(cfg, "lambda_exit", 0.0) > 0:
        total = logits_final.new_tensor(0.0)
        T = len(logits_all)
        for t, logits_t in enumerate(logits_all):
            max_prob = logits_t.softmax(dim=-1).max(dim=-1).values
            total = total + ((T - t) / T) * (1.0 - max_prob).mean()
        exit_loss = total / T

    loss = (
        ce
        + getattr(cfg, "lambda_aux", 1.0) * aux
        + getattr(cfg, "lambda_exit", 0.0) * exit_loss
    )
    return loss, {
        "ce": ce.item(),
        "aux": aux.item(),
        "exit": exit_loss.item(),
        "total": loss.item(),
    }


def _build_model(cfg):
    return FoveaTerASP(
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        feature_grid=cfg.feature_grid,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        max_fixations=cfg.max_fixations,
        max_tokens=cfg.max_tokens,
        dropout=getattr(cfg, "dropout", 0.0),
        accumulator_decay=getattr(cfg, "accumulator_decay", 0.5),
        ior_strength=getattr(cfg, "ior_strength", 1.0),
    )


def _build_loaders(cfg, device):
    if getattr(cfg, "smoke", False):
        return _make_smoke_loaders(cfg)

    train_loader, val_loader, discovered_classes, _ = build_imagenet_loaders(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        workers=cfg.num_workers,
        image_size=cfg.image_size,
        resize_size=cfg.resize_size,
        pin_memory=(device.type == "cuda"),
    )
    if discovered_classes != cfg.num_classes:
        print(
            f"[info] ImageFolder discovered {discovered_classes} classes; "
            f"model is configured for {cfg.num_classes}."
        )
    return train_loader, val_loader, discovered_classes


def _make_scheduler(optimizer, cfg):
    def lr_lambda(epoch):
        warmup = getattr(cfg, "warmup_epochs", 5)
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        min_ratio = cfg.min_lr / max(cfg.lr, 1e-12)
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _train_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    total_loss = total_acc = total_first = 0.0
    n_batches = 0
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    t0 = time.time()

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits_final, logits_all = model.forward_active_train(
                images,
                max_fixations=cfg.max_fixations,
                random_initial=True,
            )
            loss, parts = _active_image_loss(logits_final, logits_all, labels, cfg)

        if not torch.isfinite(loss):
            print(f"[skip] batch {i}: non-finite loss={loss.item():.4f}")
            continue

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        total_loss += parts["total"]
        total_acc += _top1(logits_final, labels)
        total_first += _top1(logits_all[0], labels)
        n_batches += 1

        if (i + 1) % getattr(cfg, "verbose_every", 50) == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(loader)}] "
                f"loss={total_loss/max(n_batches,1):.4f} "
                f"acc={total_acc/max(n_batches,1):.3f} "
                f"acc1={total_first/max(n_batches,1):.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"{elapsed:.0f}s"
            )

        debug_steps = getattr(cfg, "debug_steps", 0)
        if debug_steps and (i + 1) >= debug_steps:
            break

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "acc": total_acc / denom,
        "acc_first": total_first / denom,
    }


@torch.no_grad()
def _evaluate(model, loader, device, cfg):
    model.eval()
    total_loss = total_acc = total_exit = 0.0
    n_batches = n_images = 0
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits, exit_step, _ = model.forward_active_infer(
                images,
                threshold=cfg.exit_threshold,
                max_fixations=cfg.max_fixations,
                initial_fixation="center",
            )
            loss = F.cross_entropy(logits, labels)

        B = labels.size(0)
        total_loss += loss.item()
        total_acc += _top1(logits, labels)
        total_exit += exit_step * B
        n_batches += 1
        n_images += B

        debug_steps = getattr(cfg, "debug_steps", 0)
        if debug_steps and (i + 1) >= debug_steps:
            break

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_acc / max(n_batches, 1),
        "mean_exit": total_exit / max(n_images, 1),
    }


def main():
    parser = base_argparser("FoveaTer ASP ImageNet Training")
    args = parser.parse_args()
    overrides = parse_overrides(args)
    cfg = load_config(args.config or "configs/imagenet_foveater.yaml", overrides)
    set_seed(cfg.seed)
    device = cfg.device

    print(f"\n{'='*60}")
    print("  FoveaTer ASP ImageNet")
    print(f"  Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")
    print(f"  Classes: {cfg.num_classes}  Max fixations: {cfg.max_fixations}")
    print(f"  Device: {device}")
    if getattr(cfg, "smoke", False):
        print("  Mode: synthetic smoke data")
    print(f"{'='*60}\n")

    train_loader, val_loader, _ = _build_loaders(cfg, device)

    model = _build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    if hasattr(model, "param_count"):
        print(f"Param split: {model.param_count()}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=bool(cfg.use_amp and device.type == "cuda"))

    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc={best_acc*100:.2f}%")

    run_name = f"imagenet_foveater_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(cfg.log_dir, f"{run_name}.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_acc,mean_exit,lr,time\n")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_m = _train_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        val_m = _evaluate(model, val_loader, device, cfg)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
            f"loss={train_m['loss']:.4f} train={train_m['acc']*100:.2f}% "
            f"val={val_m['acc']*100:.2f}% "
            f"exit={val_m['mean_exit']:.2f}/{cfg.max_fixations} "
            f"lr={lr_now:.2e} | {elapsed:.0f}s"
        )

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_acc,
            "config": cfg.to_dict(),
        }
        torch.save(ckpt, os.path.join(cfg.ckpt_dir, "imagenet_foveater_last.pt"))

        if val_m["acc"] > best_acc:
            best_acc = val_m["acc"]
            ckpt["best_metric"] = best_acc
            torch.save(ckpt, os.path.join(cfg.ckpt_dir, "imagenet_foveater_best.pt"))
            print(f"    >> New best val: {best_acc*100:.2f}%")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch+1},{train_m['loss']:.4f},{train_m['acc']*100:.2f},"
                f"{val_m['acc']*100:.2f},{val_m['mean_exit']:.3f},"
                f"{lr_now:.6g},{elapsed:.1f}\n"
            )

    print(f"\nBest validation accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
