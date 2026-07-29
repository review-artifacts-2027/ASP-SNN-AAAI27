"""
train_scanobj.py — Train ASP-SNN on ScanObjectNN PB-T50-RS classification.

Usage:
    python train_scanobj.py [--config configs/scanobj_cls.yaml] [--resume ckpt.pt]
"""

import math
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR

from config import load_config, set_seed, base_argparser, parse_overrides
from datasets.scanobjectnn import ScanObjectNNDataset
from models.asp_classifier import ASPClassifier


# Minimum epochs of base training to bother with SWA averaging.
# Below this, SWA averages too few snapshots to be meaningful and can
# actually HURT performance. Reference: Izmailov et al. UAI 2018.
_SWA_MIN_AVERAGING_EPOCHS = 10


def main():
    parser = base_argparser("ASP-SNN ScanObjectNN Training")
    args = parser.parse_args()
    overrides = parse_overrides(args)

    config_path = args.config or "configs/scanobj_cls.yaml"
    cfg = load_config(config_path, overrides)
    set_seed(cfg.seed)
    device = cfg.device

    print(f"\n{'='*60}")
    print(f"  ASP-SNN ScanObjectNN PB-T50-RS Classification")
    print(f"  Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = ScanObjectNNDataset(cfg.data_dir, 'train', cfg)
    # P0 FIX: Validation must use UN-AUGMENTED data so val acc tracks test acc
    val_ds_clean = ScanObjectNNDataset(cfg.data_dir, 'train', cfg, force_no_aug=True)
    test_ds = ScanObjectNNDataset(cfg.data_dir, 'test', cfg)

    val_frac = getattr(cfg, 'val_fraction', 0.1)
    n_val = int(len(train_ds) * val_frac)
    n_train = len(train_ds) - n_val
    # Generate split indices deterministically and apply to both datasets
    indices = torch.randperm(
        len(train_ds),
        generator=torch.Generator().manual_seed(cfg.seed),
    ).tolist()
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    train_sub = Subset(train_ds, train_indices)
    val_sub = Subset(val_ds_clean, val_indices)  # uses clean data
    print(f"Train: {n_train} | Val: {n_val} (no aug) | Test: {len(test_ds)}")

    # drop_last safety: only drop if we have enough samples to spare
    drop_last = n_train >= cfg.batch_size * 2

    pw = cfg.num_workers > 0
    train_loader = DataLoader(
        train_sub, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=drop_last, persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_sub, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=pw,
    )

    # ── Model ─────────────────────────────────────────────────────────
    cfg.in_channels = 6
    model = ASPClassifier(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── SWA model — only enable if averaging window is meaningful ─────
    use_swa_cfg = getattr(cfg, 'use_swa', False)
    swa_start = int(cfg.epochs * getattr(cfg, 'swa_start_frac', 0.75))
    averaging_window = cfg.epochs - swa_start
    use_swa = use_swa_cfg and (averaging_window >= _SWA_MIN_AVERAGING_EPOCHS)

    if use_swa_cfg and not use_swa:
        print(f"[SWA] Disabled — averaging window ({averaging_window} epochs) "
              f"< {_SWA_MIN_AVERAGING_EPOCHS} required. "
              f"Run for more epochs to enable SWA.")

    swa_model = AveragedModel(model).to(device) if use_swa else None

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_lambda(epoch):
        warmup = getattr(cfg, 'warmup_epochs', 20)
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    swa_scheduler = SWALR(optimizer, swa_lr=getattr(cfg, 'swa_lr', 5e-5)) if use_swa else None
    scaler = GradScaler(enabled=cfg.use_amp)

    label_smooth = getattr(cfg, 'label_smooth', 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # Aux weights and their sum (for normalised loss display)
    aux_w = ASPClassifier.aux_weights(cfg.T)
    aux_w_sum = sum(aux_w)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        best_val_acc = ckpt.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_val_acc*100:.2f}%")

    # ── Logging ───────────────────────────────────────────────────────
    run_name = f"scanobj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(cfg.log_dir, f"{run_name}.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_acc,lr,time\n")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        tau = max(cfg.tau_end, cfg.tau_start * (cfg.tau_decay ** epoch))
        model.gumbel_tau.fill_(tau)

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = total_correct = total_samples = 0
        n_total_batches = len(train_loader)
        log_every = max(1, n_total_batches // 10)

        for batch_idx, (slices, geo, labels) in enumerate(train_loader):
            slices = slices.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = labels.size(0)

            with autocast(device_type=device.type, enabled=cfg.use_amp):
                logits_all = model(slices, geo, training=True)
                T_actual = len(logits_all)
                # Normalize by sum of aux weights so effective LR matches config
                # Without this, loss is ~2.58x higher than single-timestep CE,
                # inflating the effective learning rate proportionally.
                loss = sum(
                    aux_w[t] * criterion(logits_all[t], labels)
                    for t in range(min(T_actual, len(aux_w)))
                ) / aux_w_sum

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * B
            preds = logits_all[-1].argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += B

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_total_batches:
                elapsed = time.time() - t0
                per_batch = elapsed / (batch_idx + 1)
                remaining = per_batch * (n_total_batches - batch_idx - 1)
                print(f"  ep{epoch+1} [{batch_idx+1:4d}/{n_total_batches}] "
                      f"loss={loss.item():.4f} eta={remaining:.0f}s", flush=True)

        # LR scheduling
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Loss is already normalized by aux_w_sum in the backward pass
        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)
        lr_now = optimizer.param_groups[0]['lr']

        # ── Validation ────────────────────────────────────────────────
        eval_interval = getattr(cfg, 'eval_interval', 1)
        if (epoch + 1) % eval_interval == 0 or epoch == cfg.epochs - 1:
            val_acc = _evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} train={train_acc*100:.1f}% "
                f"val={val_acc*100:.2f}% | {elapsed:.0f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_metric': best_val_acc,
                }, os.path.join(cfg.ckpt_dir, 'scanobj_best.pt'))
                print(f"    >> New best val: {val_acc*100:.2f}%")

            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.4f},{train_acc*100:.2f},"
                        f"{val_acc*100:.2f},{lr_now:.2e},{elapsed:.0f}\n")
        else:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} train={train_acc*100:.1f}% | {elapsed:.0f}s"
            )

        # Save last for resume
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_metric': best_val_acc,
        }, os.path.join(cfg.ckpt_dir, 'scanobj_last.pt'))

    # ── SWA BN update ─────────────────────────────────────────────────
    if use_swa and swa_model is not None:
        print(f"\nSWA averaged {cfg.epochs - swa_start} epochs of training.")
        print("Updating SWA batch norm statistics ...")
        _update_bn(swa_model, train_loader, device)
        swa_acc = _evaluate(swa_model, test_loader, device)
        print(f"SWA test accuracy: {swa_acc*100:.2f}%")
        torch.save({
            'epoch': cfg.epochs,
            'model': swa_model.module.state_dict(),
            'best_metric': swa_acc,
        }, os.path.join(cfg.ckpt_dir, 'scanobj_swa.pt'))

    # ── Final test ────────────────────────────────────────────────────
    best_ckpt_path = os.path.join(cfg.ckpt_dir, 'scanobj_best.pt')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model'])
        test_acc = _evaluate(model, test_loader, device)
        print(f"\nFinal test accuracy (best ckpt): {test_acc*100:.2f}%")
    print(f"Checkpoint: {best_ckpt_path}")


def _evaluate(model, loader, device):
    """Evaluate classification accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for slices, geo, labels in loader:
            slices = slices.to(device)
            geo = geo.to(device)
            labels = labels.to(device)
            logits_all = model(slices, geo, training=False)
            preds = logits_all[-1].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def _update_bn(swa_model, loader, device):
    """Manual BN update for SWA (model.forward needs multiple args)."""
    for module in swa_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.reset_running_stats()
            module.train()
            module.momentum = None

    swa_model.train()
    with torch.no_grad():
        for slices, geo, _ in loader:
            slices = slices.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            swa_model(slices, geo, training=False)
    swa_model.eval()


if __name__ == "__main__":
    main()
