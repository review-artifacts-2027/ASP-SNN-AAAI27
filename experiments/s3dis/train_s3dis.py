"""
train_s3dis.py — Train ASP-SNN on S3DIS Area 5 scene segmentation.

Tier A upgrades:
  A4. TET-style loss (equal-weight per-timestep CE + MSE regularization),
      replacing the previous weighted-sum + confidence-penalty loss which
      was actively rewarding overconfidence.
  Mid-epoch eval uses evaluate_s3dis_aggregated (A1) so training-time
  mIoU reflects the true test-time metric.

Tier B upgrade:
  B1. Lovász-Softmax auxiliary loss — directly optimizes mIoU, the eval
      metric. Especially valuable on tail classes (beam, column, board)
      that dominate the macro-averaged mIoU gap.
      Reference: Berman et al. CVPR 2018.

Tier E upgrade:
  E1. Room-level ASP prior — precomputed room summaries are projected into
      the initial LIF belief state u_0 via a zero-init learnable projection,
      seeding the ASP loop with room-level context instead of zeros.
"""

import argparse
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
from datasets.s3dis import (
    S3DISDataset, compute_class_weights, CLASS_NAMES, NUM_CLASSES,
)
from models.asp_segmentor import ASPSegmentor
from models.lovasz_losses import LovaszSoftmaxLoss
from eval_s3dis import evaluate_s3dis_aggregated, compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helper (best-effort)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(log_path: str, out_dir: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs, losses, mious, maccs, oas, lrs = [], [], [], [], [], []
        with open(log_path, 'r') as f:
            f.readline()  # header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                epochs.append(int(parts[0]))
                losses.append(float(parts[1]))
                mious.append(float(parts[2]))
                maccs.append(float(parts[3]))
                oas.append(float(parts[4]))
                lrs.append(float(parts[5]))

        if not epochs:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('S3DIS ASP-SNN Training Curves', fontsize=14,
                     fontweight='bold')

        axes[0].plot(epochs, mious, 'b-o', markersize=3, label='mIoU')
        axes[0].plot(epochs, maccs, 'g--s', markersize=3, label='mAcc')
        axes[0].set_title('Segmentation Metrics')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('%')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        if mious:
            best_ep = epochs[mious.index(max(mious))]
            axes[0].axvline(best_ep, color='r', linestyle=':', alpha=0.7,
                            label=f'Best ep{best_ep}')
            axes[0].legend()

        axes[1].plot(epochs, losses, 'r-o', markersize=3)
        axes[1].set_title('Training Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(epochs, lrs, 'm-o', markersize=3)
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR (log scale)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(out_dir, 's3dis_training_curves.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Plot] Training curves saved → {out_path}")
    except Exception as e:
        print(f"[Plot] Warning: could not generate plot ({e})")


# ─────────────────────────────────────────────────────────────────────────────
#  KD teacher (PointNet-style per-point segmentation)
# ─────────────────────────────────────────────────────────────────────────────
class PointNetSegTeacher(nn.Module):
    """Lightweight PointNet segmentation teacher for knowledge distillation."""

    def __init__(self, num_classes: int, in_channels: int = 7):
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.seg_head = nn.Sequential(
            nn.Conv1d(1024 + 128, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, pts_feat):  # [B, N, C]
        x = pts_feat.permute(0, 2, 1)                # [B, C, N]
        local_feat = self.local_mlp(x)               # [B, 128, N]
        global_feat = self.global_mlp(local_feat) \
            .max(dim=-1, keepdim=True).values         # [B, 1024, 1]
        global_feat = global_feat.expand(
            -1, -1, local_feat.size(-1))              # [B, 1024, N]
        combined = torch.cat(
            [local_feat, global_feat], dim=1)         # [B, 1152, N]
        return self.seg_head(combined).permute(0, 2, 1)  # [B, N, C]


def seg_kd_loss(student_logits, teacher_logits,
                T: float = 4.0) -> torch.Tensor:
    """Per-point KL divergence loss for segmentation KD."""
    B, N, C = student_logits.shape
    s = F.log_softmax(student_logits.reshape(B * N, C) / T, dim=-1)
    t = F.softmax(teacher_logits.detach().reshape(B * N, C) / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


# ─────────────────────────────────────────────────────────────────────────────
#  Metric helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Compute per-class IoU, mIoU, OA, and mAcc."""
    ious, accs = [], []
    for c in range(num_classes):
        pred_c = (pred == c)
        true_c = (target == c)
        inter = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        ious.append(inter / union if union > 0 else float('nan'))
        true_count = true_c.sum()
        accs.append(inter / true_count if true_count > 0 else float('nan'))

    miou = float(np.nanmean(ious))
    macc = float(np.nanmean(accs))
    oa = float((pred == target).sum() / max(len(target), 1))
    return miou, macc, oa, {CLASS_NAMES[i]: ious[i]
                            for i in range(num_classes)}


# ─────────────────────────────────────────────────────────────────────────────
#  A4: TET loss replacing active_loss_seg
# ─────────────────────────────────────────────────────────────────────────────
def active_loss_seg_tet(logits_final, logits_all, labels, criterion,
                        tet_lambda: float = 0.05):
    """
    TET loss for segmentation: equal-weight mean of per-timestep CE, with
    optional MSE regularization pulling each timestep toward the final.

    Reference: Deng et al. "Temporal Efficient Training of Spiking Neural
    Networks via Gradient Re-weighting", ICLR 2022.

    Replaces active_loss_seg which used quadratic-ramp + confidence penalty.
    The penalty rewarded overconfidence and hurt mIoU on rare classes.

    Args:
        logits_final: [B*N, C]  final-timestep logits, flattened
        logits_all:   list of [B, N, C]  per-timestep logits (or belief list)
        labels:       [B*N]     flattened ground-truth labels
        criterion:    nn.CrossEntropyLoss (with class weights + ignore_index)
        tet_lambda:   MSE reg weight (0 disables)

    Returns:
        loss: scalar
    """
    if not logits_all:
        return criterion(logits_final, labels)

    C = logits_final.shape[-1]

    # Check if logits_all contains actual per-timestep logits or belief states.
    # Belief states from ASPSegmentor have shape [B, hidden_dim] (no N dim),
    # while per-timestep logits have shape [B, N, C].
    # Only apply TET CE if they are actually per-timestep logits.
    first = logits_all[0]
    if first.dim() == 3 and first.shape[-1] == C:
        # Per-timestep logits available (dense_tet mode)
        ce_terms = torch.stack([
            criterion(l.reshape(-1, C), labels) for l in logits_all
        ])
        loss = ce_terms.mean()

        if tet_lambda > 0 and len(logits_all) > 1:
            final = logits_all[-1].detach()
            mse_terms = torch.stack([
                F.mse_loss(logits_all[t], final)
                for t in range(len(logits_all) - 1)
            ])
            loss = loss + tet_lambda * mse_terms.mean()

        return loss
    else:
        # logits_all contains belief states, not logits — fall back to
        # single-timestep CE on the final output.
        return criterion(logits_final, labels)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = base_argparser("ASP-SNN S3DIS Training")
    args = parser.parse_args()
    overrides = parse_overrides(args)

    config_path = args.config or "configs/s3dis_seg.yaml"
    cfg = load_config(config_path, overrides)
    set_seed(cfg.seed)
    device = cfg.device

    test_area = getattr(cfg, 'test_area', 5)
    train_areas_str = ", ".join(
        str(a) for a in [1, 2, 3, 4, 5, 6] if a != test_area
    )

    print(f"\n{'='*60}")
    print(f"  ASP-SNN S3DIS Scene Segmentation")
    print(f"  Protocol: train on Areas {{{train_areas_str}}}, "
          f"test on Area {test_area}")
    print(f"  Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")
    print(f"  Block: {cfg.block_size}m  Points/block: {cfg.num_points}  "
          f"Points/slice: {cfg.points_per_slice}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = S3DISDataset(cfg.data_dir, 'train', cfg)
    test_ds = S3DISDataset(cfg.data_dir, 'test', cfg)

    pw = cfg.num_workers > 0
    drop_last = len(train_ds) >= cfg.batch_size * 2
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=drop_last, persistent_workers=pw,
    )

    # ── Class weights ─────────────────────────────────────────────────
    class_weights = None
    if getattr(cfg, 'use_class_weights', True):
        weights_np = compute_class_weights(cfg.data_dir, test_area)
        class_weights = torch.from_numpy(weights_np).to(device)
        print(f"[Weights] class weights: "
              f"{[f'{w:.2f}' for w in weights_np.tolist()]}")

    # ── Model ─────────────────────────────────────────────────────────
    in_ch = 3
    if getattr(cfg, 'use_rgb', True):
        in_ch += 3
    if getattr(cfg, 'use_height', True):
        in_ch += 1
    cfg.in_channels = in_ch
    cfg.num_classes = NUM_CLASSES
    cfg.use_category = False
    cfg.num_categories = 0

    # ── KD teacher setup ─────────────────────────────────────────────
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    kd_teacher = None
    kd_teacher_epochs = int(getattr(cfg, 'kd_teacher_epochs', 0))
    kd_temp = float(getattr(cfg, 'kd_temp', 4.0))
    kd_lam = float(getattr(cfg, 'kd_lam', 0.5))
    teacher_ckpt = os.path.join(cfg.ckpt_dir, 's3dis_kd_teacher.pt')

    if kd_teacher_epochs > 0:
        kd_teacher = PointNetSegTeacher(
            num_classes=NUM_CLASSES, in_channels=in_ch,
        ).to(device)

        if os.path.exists(teacher_ckpt):
            print(f"\n[KD] Found saved teacher → {teacher_ckpt}")
            kd_teacher.load_state_dict(
                torch.load(teacher_ckpt, map_location=device,
                           weights_only=False)
            )
            print("[KD] Teacher loaded. Skipping pre-training.")
        else:
            print(f"\n[KD] Pre-training PointNet seg teacher "
                  f"({kd_teacher_epochs} ep, T={kd_temp}, λ={kd_lam})")
            kd_teacher.train()
            t_opt = torch.optim.AdamW(kd_teacher.parameters(), lr=1e-3,
                                      weight_decay=1e-4)
            t_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                t_opt, T_max=kd_teacher_epochs, eta_min=1e-5,
            )
            t_criterion = nn.CrossEntropyLoss(
                weight=class_weights, ignore_index=-1,
            )

            for t_ep in range(kd_teacher_epochs):
                t0_ep = time.time()
                t_loss_sum = t_n = 0
                n_t_batches = len(train_loader)
                log_every_t = max(1, n_t_batches // 10)

                for batch_idx_t, batch_data_t in enumerate(train_loader):
                    # Teacher only needs pts_feat and labels
                    pts_feat_b = batch_data_t[2].to(device, non_blocking=True)
                    sem_labels_b = batch_data_t[4].to(device, non_blocking=True)

                    t_logits = kd_teacher(pts_feat_b)
                    B_t, N_t, C_t = t_logits.shape
                    t_loss = t_criterion(
                        t_logits.reshape(B_t * N_t, C_t),
                        sem_labels_b.reshape(B_t * N_t),
                    )
                    t_opt.zero_grad()
                    t_loss.backward()
                    nn.utils.clip_grad_norm_(kd_teacher.parameters(), 1.0)
                    t_opt.step()
                    t_loss_sum += float(t_loss.detach()) * B_t
                    t_n += B_t

                    if (batch_idx_t + 1) % log_every_t == 0 or \
                       (batch_idx_t + 1) == n_t_batches:
                        elapsed_t = time.time() - t0_ep
                        print(
                            f"  [Teacher] ep{t_ep+1} "
                            f"[{batch_idx_t+1:4d}/{n_t_batches}] "
                            f"loss={t_loss.item():.4f} "
                            f"elapsed={elapsed_t:.0f}s",
                            flush=True,
                        )
                t_sch.step()
                print(f"  [Teacher] Ep {t_ep+1:2d}/{kd_teacher_epochs}  "
                      f"total_loss={t_loss_sum / max(t_n, 1):.4f} "
                      f"time={time.time()-t0_ep:.0f}s", flush=True)
            torch.save(kd_teacher.state_dict(), teacher_ckpt)
            print(f"[KD] Teacher saved → {teacher_ckpt}")

        kd_teacher.eval()

    # ── Student model ─────────────────────────────────────────────────
    model = ASPSegmentor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Input channels: {in_ch} (xyz"
          f"{'+rgb' if getattr(cfg, 'use_rgb', True) else ''}"
          f"{'+height' if getattr(cfg, 'use_height', True) else ''})")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    def lr_lambda(epoch):
        warmup = getattr(cfg, 'warmup_epochs', 5)
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_amp)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-1,
    )

    # ── Loss mode (A4: TET) ──────────────────────────────────────────
    loss_mode = getattr(cfg, 'loss_mode', 'tet')
    tet_lambda = float(getattr(cfg, 'tet_lambda', 0.05))
    print(f"Loss mode: {loss_mode}"
          + (f" (TET lambda={tet_lambda})" if loss_mode == 'tet' else ""))

    # ── B1: Lovász-Softmax auxiliary loss ─────────────────────────────
    use_lovasz = getattr(cfg, 'use_lovasz', True)
    lovasz_lambda = float(getattr(cfg, 'lovasz_lambda', 1.0))
    lovasz_criterion = LovaszSoftmaxLoss(
        ignore_index=-1, classes="present",
    ) if use_lovasz else None
    if use_lovasz:
        print(f"Lovász-Softmax loss: enabled (λ={lovasz_lambda})")

    # ── E1: room-level prior config ──────────────────────────────────
    use_room_prior = getattr(cfg, 'use_room_prior', False)
    if use_room_prior:
        print(f"Room-level prior (E1): enabled "
              f"(K_room={getattr(cfg, 'room_prior_anchors', 64)}, "
              f"k={getattr(cfg, 'room_prior_k', 32)})")

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device,
                          weights_only=False)
        # Tolerate missing keys (new E1 projection layers won't be in old ckpt)
        missing, unexpected = model.load_state_dict(
            ckpt['model'], strict=False,
        )
        if missing:
            print(f"  [resume] missing keys ({len(missing)}): "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  [resume] unexpected keys ({len(unexpected)})")
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        best_miou = ckpt.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch}, "
              f"best mIoU: {best_miou*100:.2f}%")

    # ── Logging ───────────────────────────────────────────────────────
    os.makedirs(cfg.log_dir, exist_ok=True)
    shared_log_path = os.path.join(cfg.log_dir, 's3dis_train_log.csv')
    log_path = shared_log_path
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,miou,macc,oa,lr,time\n")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        tau = max(cfg.tau_end, cfg.tau_start * (cfg.tau_decay ** epoch))
        model.gumbel_tau.fill_(tau)

        if hasattr(model, 'lif_head') and \
           hasattr(model.lif_head, 'reset_spike_stats'):
            model.lif_head.reset_spike_stats()

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = n_batches = 0
        n_total_batches = len(train_loader)
        log_every = max(1, n_total_batches // 20)

        for batch_idx, batch_data in enumerate(train_loader):
            # ── Unpack (handles with/without room prior) ──────────
            #
            # Dataset returns:
            #   Without room prior: (slices, geo, pts_feat, sid_arr,
            #                        sem_labels, cat_id)              → 6 items
            #   With room prior:    (slices, geo, pts_feat, sid_arr,
            #                        sem_labels, cat_id, room_summary) → 7 items
            #
            slices       = batch_data[0].to(device, non_blocking=True)
            geo          = batch_data[1].to(device, non_blocking=True)
            pts_feat     = batch_data[2].to(device, non_blocking=True)
            sid_arr      = batch_data[3].to(device, non_blocking=True)
            sem_labels   = batch_data[4].to(device, non_blocking=True)
            cat_ids      = batch_data[5].to(device, non_blocking=True)

            if use_room_prior and len(batch_data) > 6:
                room_summary = batch_data[6].to(device, non_blocking=True)
            else:
                room_summary = None

            with autocast(device_type=device.type, enabled=cfg.use_amp):
                logits_final, logits_all = model(
                    slices, geo, sid_arr, cat_ids, pts_feat,
                    room_summary=room_summary,
                    training=True,
                )
                B, N, C = logits_final.shape

                # A4: TET loss (equal-weight per-timestep CE + MSE reg)
                loss = active_loss_seg_tet(
                    logits_final.reshape(B * N, C),
                    logits_all,
                    sem_labels.reshape(B * N),
                    criterion,
                    tet_lambda=tet_lambda,
                )

                # B1: Lovász-Softmax on final-timestep logits.
                # Applied only to final logits (not per-timestep) because
                # Lovász is expensive (sort per class per point).
                if lovasz_criterion is not None:
                    lov = lovasz_criterion(logits_final, sem_labels)
                    loss = loss + lovasz_lambda * lov

                # KD loss (if teacher exists)
                if kd_teacher is not None:
                    with torch.no_grad():
                        t_logits = kd_teacher(pts_feat)
                    loss = loss + kd_lam * seg_kd_loss(
                        logits_final, t_logits, kd_temp,
                    )

                # Spike firing-rate penalty (if applicable)
                if hasattr(model, 'lif_head') and \
                   hasattr(model.lif_head, 'mean_firing_rate'):
                    loss = loss + 0.01 * model.lif_head.mean_firing_rate()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_every == 0 or \
               (batch_idx + 1) == n_total_batches:
                elapsed = time.time() - t0
                per_batch = elapsed / (batch_idx + 1)
                remaining = per_batch * (n_total_batches - batch_idx - 1)
                gpu_mem = (torch.cuda.max_memory_allocated() / 1e9
                           if torch.cuda.is_available() else 0)
                print(
                    f"  ep{epoch+1} [{batch_idx+1:4d}/{n_total_batches}] "
                    f"loss={loss.item():.4f} eta={remaining:.0f}s "
                    f"gpu_mem={gpu_mem:.1f}GB",
                    flush=True,
                )

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]['lr']

        # ── Eval (A1: room-aggregated, single vote during training) ──
        eval_interval = getattr(cfg, 'eval_interval', 5)
        if (epoch + 1) % eval_interval == 0 or epoch == cfg.epochs - 1:
            model.eval()

            all_preds, all_true = evaluate_s3dis_aggregated(
                model, test_ds, cfg, device, n_votes=1,
            )
            miou, macc, oa, per_class = compute_iou(
                all_preds, all_true, NUM_CLASSES,
            )

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} | "
                f"mIoU={miou*100:.2f}% mAcc={macc*100:.2f}% "
                f"OA={oa*100:.2f}% | {elapsed:.0f}s"
            )

            # Per-class print every eval so we track tail class progress
            for name, iou in sorted(
                per_class.items(),
                key=lambda x: x[1] if not np.isnan(x[1]) else 0,
            ):
                v = iou * 100 if not np.isnan(iou) else 0.0
                print(f"    {name:<12} {v:5.1f}%")

            if miou > best_miou:
                best_miou = miou
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_metric': best_miou,
                    'miou': miou,
                    'macc': macc,
                    'oa': oa,
                }, os.path.join(cfg.ckpt_dir, 's3dis_best.pt'))
                print(f"    >> New best mIoU: {miou*100:.2f}%")

            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.4f},"
                        f"{miou*100:.2f},{macc*100:.2f},{oa*100:.2f},"
                        f"{lr_now:.2e},{elapsed:.0f}\n")
        else:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} | {elapsed:.0f}s"
            )

        # Save last for resume
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_metric': best_miou,
        }, os.path.join(cfg.ckpt_dir, 's3dis_last.pt'))

    print(f"\nDone. Best mIoU: {best_miou*100:.2f}%")
    print(f"Checkpoint: {cfg.ckpt_dir}/s3dis_best.pt")
    print(f"\nRun full-TTA eval with:")
    print(f"  python eval_s3dis.py --ckpt {cfg.ckpt_dir}/s3dis_best.pt "
          f"--n_votes 3 --per_class")

    plot_training_curves(log_path, cfg.log_dir)


if __name__ == "__main__":
    main()