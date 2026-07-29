"""
train_shapenet.py — Train ASP-SNN on ShapeNetPart part segmentation.

Usage:
    python train_shapenet.py [--config configs/shapenet_seg.yaml] [--resume ckpt.pt]
"""

import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import load_config, set_seed, base_argparser, parse_overrides
from datasets.shapenetpart import (
    ShapeNetPartDataset, CATEGORY_TO_PARTS, CATEGORY_NAMES,
    NUM_PARTS, NUM_CATEGORIES,
)
from models.asp_segmentor import ASPSegmentor


# ── Loss ──────────────────────────────────────────────────────────────────

def seg_loss_fn(part_logits, part_labels, cat_ids):
    """
    CE loss with category-aware part masking.
    Invalid part logits masked to -1e9 in fp32 (NOT fp16 — avoids overflow).
    No label_smoothing (incompatible with -1e9 masking).
    """
    B, N, P = part_logits.shape
    device = part_logits.device

    # P1 FIX: pre-fetch cat_ids to CPU once instead of .item() inside the
    # inner loop (which forces B CPU syncs per forward pass).
    cat_ids_cpu = cat_ids.detach().cpu().tolist()
    valid_mask = torch.zeros(B, P, device=device, dtype=torch.bool)
    for b in range(B):
        for pid in CATEGORY_TO_PARTS[cat_ids_cpu[b]]:
            valid_mask[b, pid] = True

    mask_expanded = valid_mask.unsqueeze(1).expand(B, N, P)
    logits_masked = part_logits.float().clone()
    logits_masked[~mask_expanded] = -1e9

    logits_flat = logits_masked.reshape(B * N, P)
    labels_flat = part_labels.reshape(B * N)
    return F.cross_entropy(logits_flat, labels_flat)


# ── mIoU ──────────────────────────────────────────────────────────────────

def compute_instance_miou(pred_parts, true_parts, cat_ids, n_points):
    n_shapes = len(cat_ids)
    iou_per_shape = []
    cat_ious = {i: [] for i in range(NUM_CATEGORIES)}

    for i in range(n_shapes):
        start = i * n_points
        end = start + n_points
        p = pred_parts[start:end]
        g = true_parts[start:end]
        cat = int(cat_ids[i])
        parts = CATEGORY_TO_PARTS[cat]

        ious = []
        for part in parts:
            pred_mask = (p == part)
            true_mask = (g == part)
            union = np.logical_or(pred_mask, true_mask).sum()
            inter = np.logical_and(pred_mask, true_mask).sum()
            if union == 0:
                continue
            ious.append(inter / union)

        if ious:
            shape_iou = float(np.mean(ious))
            iou_per_shape.append(shape_iou)
            cat_ious[cat].append(shape_iou)

    inst_miou = float(np.mean(iou_per_shape)) if iou_per_shape else 0.0
    per_cat = {}
    populated_ious = []
    for cat, ious in cat_ious.items():
        if ious:
            per_cat[CATEGORY_NAMES[cat]] = float(np.mean(ious))
            populated_ious.append(per_cat[CATEGORY_NAMES[cat]])
        else:
            # Category absent from test split → mark explicitly but don't drag
            # cls_miou down with a 0.
            per_cat[CATEGORY_NAMES[cat]] = float('nan')
    cls_miou = float(np.mean(populated_ious)) if populated_ious else 0.0
    return inst_miou, cls_miou, per_cat


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = base_argparser("ASP-SNN ShapeNetPart Training")
    args = parser.parse_args()
    overrides = parse_overrides(args)

    config_path = args.config or "configs/shapenet_seg.yaml"
    cfg = load_config(config_path, overrides)
    set_seed(cfg.seed)
    device = cfg.device

    print(f"\n{'='*60}")
    print(f"  ASP-SNN ShapeNetPart Part Segmentation")
    print(f"  Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = ShapeNetPartDataset(cfg.data_dir, 'train', cfg)
    test_ds = ShapeNetPartDataset(cfg.data_dir, 'test', cfg)

    pw = cfg.num_workers > 0
    # drop_last safety: only drop if we have enough samples to spare
    drop_last = len(train_ds) >= cfg.batch_size * 2
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=drop_last, persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=pw,
    )

    # ── Model ─────────────────────────────────────────────────────────
    # Override config for segmentor
    cfg.num_classes = NUM_PARTS
    cfg.num_categories = NUM_CATEGORIES
    cfg.use_category = True
    cfg.in_channels = 6

    model = ASPSegmentor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Optimizer with differential LR ────────────────────────────────
    enc_scale = getattr(cfg, 'encoder_lr_scale', 0.1)
    encoder_params = (
        list(model.feature_extractor.parameters()) +
        list(model.slice_transformer.parameters()) +
        list(model.pos_proj.parameters())
    )
    enc_ids = set(id(p) for p in encoder_params)
    new_params = [p for p in model.parameters() if id(p) not in enc_ids]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg.lr * enc_scale},
        {"params": new_params, "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)

    # Warmup + cosine LR
    def lr_lambda(epoch):
        warmup = getattr(cfg, 'warmup_epochs', 10)
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_amp)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_inst_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        best_inst_iou = ckpt.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_inst_iou*100:.2f}%")

    # ── Logging ───────────────────────────────────────────────────────
    run_name = f"shapenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(cfg.log_dir, f"{run_name}.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,inst_miou,cls_miou,lr,time\n")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        # Gumbel temperature annealing
        tau = max(cfg.tau_end, cfg.tau_start * (cfg.tau_decay ** epoch))
        model.gumbel_tau.fill_(tau)

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = n_batches = 0
        n_total_batches = len(train_loader)
        log_every = max(1, n_total_batches // 10)

        for batch_idx, (slices, geo, pts_xyz, sid_arr, part_labels, cat_ids) in enumerate(train_loader):
            slices = slices.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            pts_xyz = pts_xyz.to(device, non_blocking=True)
            sid_arr = sid_arr.to(device, non_blocking=True)
            part_labels = part_labels.to(device, non_blocking=True)
            cat_ids = cat_ids.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=cfg.use_amp):
                part_logits, _ = model(
                    slices, geo, sid_arr, cat_ids, pts_xyz, training=True
                )
                loss = seg_loss_fn(part_logits, part_labels, cat_ids)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_total_batches:
                elapsed = time.time() - t0
                per_batch = elapsed / (batch_idx + 1)
                remaining = per_batch * (n_total_batches - batch_idx - 1)
                print(f"  ep{epoch+1} [{batch_idx+1:4d}/{n_total_batches}] "
                      f"loss={loss.item():.4f} eta={remaining:.0f}s", flush=True)

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[-1]['lr']

        # ── Eval ──────────────────────────────────────────────────────
        eval_interval = getattr(cfg, 'eval_interval', 5)
        if (epoch + 1) % eval_interval == 0 or epoch == cfg.epochs - 1:
            model.eval()
            all_preds, all_true, all_cats = [], [], []

            with torch.no_grad():
                for slices, geo, pts_xyz, sid_arr, part_labels, cat_ids in test_loader:
                    slices = slices.to(device)
                    geo = geo.to(device)
                    pts_xyz = pts_xyz.to(device)
                    sid_arr = sid_arr.to(device)
                    cat_ids = cat_ids.to(device)
                    B, N = part_labels.shape

                    part_logits, _ = model(
                        slices, geo, sid_arr, cat_ids, pts_xyz, training=False
                    )

                    for b in range(B):
                        cat = int(cat_ids[b].item())
                        lgt = part_logits[b]
                        valid = torch.tensor(CATEGORY_TO_PARTS[cat], device=device)
                        lgt_valid = lgt[:, valid]
                        pred_local = lgt_valid.argmax(dim=-1)
                        pred_global = valid[pred_local]
                        all_preds.append(pred_global.cpu().numpy())
                        all_true.append(part_labels[b].numpy())
                        all_cats.append(cat)

            all_preds = np.concatenate(all_preds)
            all_true = np.concatenate(all_true)
            inst_iou, cls_iou, per_cat = compute_instance_miou(
                all_preds, all_true, np.array(all_cats), test_ds.n_points
            )

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} | "
                f"Inst mIoU={inst_iou*100:.2f}% "
                f"Cls mIoU={cls_iou*100:.2f}% | {elapsed:.0f}s"
            )

            # Per-category printout every 25 epochs
            if (epoch + 1) % 25 == 0:
                for cn, iou in sorted(per_cat.items(), key=lambda x: x[1]):
                    print(f"    {cn:<14} {iou*100:5.1f}%")

            # Save best
            if inst_iou > best_inst_iou:
                best_inst_iou = inst_iou
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_metric': best_inst_iou,
                    'inst_iou': inst_iou,
                    'cls_iou': cls_iou,
                }, os.path.join(cfg.ckpt_dir, 'shapenet_best.pt'))
                print(f"    >> New best: {inst_iou*100:.2f}%")

            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.4f},"
                        f"{inst_iou*100:.2f},{cls_iou*100:.2f},"
                        f"{lr_now:.2e},{elapsed:.0f}\n")
        else:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} | {elapsed:.0f}s"
            )

        # Save last checkpoint for resume
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_metric': best_inst_iou,
        }, os.path.join(cfg.ckpt_dir, 'shapenet_last.pt'))

    print(f"\nDone. Best Instance mIoU: {best_inst_iou*100:.2f}%")
    print(f"Checkpoint: {cfg.ckpt_dir}/shapenet_best.pt")


if __name__ == "__main__":
    main()
