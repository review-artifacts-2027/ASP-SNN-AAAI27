"""
train_s3dis.py — Train ASP-SNN on S3DIS Area 5 scene segmentation.

Usage:
    python train_s3dis.py [--config configs/s3dis_seg.yaml] [--resume ckpt.pt]
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
from datasets.s3dis import S3DISDataset, CLASS_NAMES, NUM_CLASSES, compute_class_weights
from models.asp_segmentor import ASPSegmentor


def plot_training_curves(log_path: str, out_dir: str):
    """
    Read the CSV training log and plot mIoU, loss, and LR curves.
    Saved as s3dis_training_curves.png in out_dir.
    Works across resume runs since all epochs append to the same CSV.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend — works on Kaggle/Colab
        import matplotlib.pyplot as plt
        import csv

        epochs, losses, mious, maccs, lrs = [], [], [], [], []
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('miou', '') == '':
                    continue  # skip non-eval epochs
                epochs.append(int(row['epoch']))
                losses.append(float(row['train_loss']))
                mious.append(float(row['miou']))
                maccs.append(float(row['macc']))
                lrs.append(float(row['lr']))

        if not epochs:
            print("[Plot] No eval epochs found in log yet.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('S3DIS ASP-SNN Training Curves', fontsize=14, fontweight='bold')

        axes[0].plot(epochs, mious, 'b-o', markersize=3, label='mIoU')
        axes[0].plot(epochs, maccs, 'g--s', markersize=3, label='mAcc')
        axes[0].set_title('Segmentation Metrics')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('%')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        if mious:
            best_ep = epochs[mious.index(max(mious))]
            axes[0].axvline(best_ep, color='r', linestyle=':', alpha=0.7, label=f'Best ep{best_ep}')
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
        x = pts_feat.permute(0, 2, 1)   # [B, C, N]
        local_feat = self.local_mlp(x)  # [B, 128, N]
        global_feat = self.global_mlp(local_feat).max(dim=-1, keepdim=True).values  # [B, 1024, 1]
        global_feat = global_feat.expand(-1, -1, local_feat.size(-1))  # [B, 1024, N]
        combined = torch.cat([local_feat, global_feat], dim=1)  # [B, 1152, N]
        return self.seg_head(combined).permute(0, 2, 1)  # [B, N, num_classes]


def seg_kd_loss(student_logits, teacher_logits, T: float = 4.0) -> torch.Tensor:
    """Per-point KL divergence loss for segmentation KD."""
    B, N, C = student_logits.shape
    s = F.log_softmax(student_logits.reshape(B * N, C) / T, dim=-1)
    t = F.softmax(teacher_logits.detach().reshape(B * N, C) / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)



def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Compute per-class IoU, mIoU, OA, and mAcc."""
    ious = []
    accs = []
    for c in range(num_classes):
        pred_c = (pred == c)
        true_c = (target == c)
        inter = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        if union > 0:
            ious.append(inter / union)
        else:
            ious.append(float('nan'))
        true_count = true_c.sum()
        if true_count > 0:
            accs.append(inter / true_count)
        else:
            accs.append(float('nan'))

    iou_arr = np.array(ious)
    acc_arr = np.array(accs)
    miou = float(np.nanmean(iou_arr))
    macc = float(np.nanmean(acc_arr))
    oa = float((pred == target).sum() / max(len(target), 1))
    return miou, macc, oa, {CLASS_NAMES[i]: ious[i] for i in range(num_classes)}


def active_loss_seg(logits_final, logits_all, labels, criterion):
    """
    Computes final loss + auxiliary intermediate loss + confidence penalty.
    labels: [B*N]
    logits_final: [B*N, C]
    logits_all: list of [B, N, C]
    """
    import torch.nn.functional as F
    
    loss = criterion(logits_final, labels)
    
    if len(logits_all) > 1:
        # Auxiliary KD / hard-label CE loss
        aux = sum(criterion(l.reshape(-1, l.shape[-1]), labels) for l in logits_all[:-1])
        loss = loss + 0.1 * aux / (len(logits_all) - 1)
        
        # Confidence regularisation
        conf_penalty = 0
        S = len(logits_all)
        for i, l in enumerate(logits_all):
            w = (S - i) / S
            probs = F.softmax(l.reshape(-1, l.shape[-1]), dim=-1)
            max_p = probs.max(dim=-1).values
            # Filter out ignore_index (-1)
            valid = labels != -1
            if valid.sum() > 0:
                conf_penalty += w * (1.0 - max_p[valid]).mean()
                
        loss = loss + 0.05 * conf_penalty / S
        
    return loss


def main():
    parser = base_argparser("ASP-SNN S3DIS Training")
    args = parser.parse_args()
    overrides = parse_overrides(args)

    config_path = args.config or "configs/s3dis_seg.yaml"
    cfg = load_config(config_path, overrides)
    set_seed(cfg.seed)
    device = cfg.device

    test_area = getattr(cfg, 'test_area', 5)
    train_areas_str = ", ".join(str(a) for a in [1, 2, 3, 4, 5, 6] if a != test_area)

    print(f"\n{'='*60}")
    print(f"  ASP-SNN S3DIS Scene Segmentation")
    print(f"  Protocol: train on Areas {{{train_areas_str}}}, test on Area {test_area}")
    print(f"  Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = S3DISDataset(cfg.data_dir, 'train', cfg)
    test_ds = S3DISDataset(cfg.data_dir, 'test', cfg)

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

    # ── Class weights ─────────────────────────────────────────────────
    class_weights = None
    if getattr(cfg, 'use_class_weights', True):
        print("Computing class weights from training areas ...")
        weights_np = compute_class_weights(cfg.data_dir,
                                           getattr(cfg, 'test_area', 5))
        class_weights = torch.from_numpy(weights_np).to(device)
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:<12} weight={weights_np[i]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────
    # Determine input channels from config
    in_ch = 3
    if getattr(cfg, 'use_rgb', True):
        in_ch += 3
    if getattr(cfg, 'use_height', True):
        in_ch += 1
    cfg.in_channels = in_ch
    cfg.num_classes = NUM_CLASSES
    cfg.use_category = False
    cfg.num_categories = 0

    # ── KD teacher (optional) ─────────────────────────────────────────
    kd_teacher_epochs = int(getattr(cfg, 'kd_teacher_epochs', 0))
    kd_temp = float(getattr(cfg, 'kd_temp', 4.0))
    kd_lam  = float(getattr(cfg, 'kd_lam', 0.5))
    kd_teacher = None
    teacher_ckpt = os.path.join(cfg.ckpt_dir, "s3dis_teacher.pth")

    if kd_teacher_epochs > 0:
        kd_teacher = PointNetSegTeacher(NUM_CLASSES, in_channels=in_ch).to(device)

        # If a saved teacher already exists (e.g. resuming after Kaggle timeout),
        # skip re-training and just load it — saves ~2 hours per resume run.
        if os.path.exists(teacher_ckpt):
            print(f"\n[KD] Found saved teacher checkpoint → loading from {teacher_ckpt}")
            kd_teacher.load_state_dict(
                torch.load(teacher_ckpt, map_location=device, weights_only=False)
            )
            print("[KD] Teacher loaded successfully. Skipping pre-training.")
        else:
            print(f"\n[KD] Pre-training PointNet seg teacher ({kd_teacher_epochs} ep, T={kd_temp}, λ={kd_lam})")
            kd_teacher.train()
            t_opt = torch.optim.AdamW(kd_teacher.parameters(), lr=1e-3, weight_decay=1e-4)
            t_sch = torch.optim.lr_scheduler.CosineAnnealingLR(t_opt, T_max=kd_teacher_epochs, eta_min=1e-5)
            t_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
            for t_ep in range(kd_teacher_epochs):
                t0_ep = time.time()
                t_loss_sum = t_n = 0
                n_t_batches = len(train_loader)
                log_every_t = max(1, n_t_batches // 10)
                for batch_idx_t, (slices_b, geo_b, pts_feat_b, sid_b, sem_labels_b, cat_b) in enumerate(train_loader):
                    pts_feat_b   = pts_feat_b.to(device, non_blocking=True)
                    sem_labels_b = sem_labels_b.to(device, non_blocking=True)
                    t_logits = kd_teacher(pts_feat_b)
                    B, N, C = t_logits.shape
                    t_loss = t_criterion(t_logits.reshape(B*N, C), sem_labels_b.reshape(B*N))
                    t_opt.zero_grad(); t_loss.backward()
                    nn.utils.clip_grad_norm_(kd_teacher.parameters(), 1.0)
                    t_opt.step()
                    t_loss_sum += float(t_loss.detach()) * B
                    t_n        += B

                    if (batch_idx_t + 1) % log_every_t == 0 or (batch_idx_t + 1) == n_t_batches:
                        elapsed_t = time.time() - t0_ep
                        print(
                            f"  [Teacher] ep{t_ep+1} [{batch_idx_t+1:4d}/{n_t_batches}] "
                            f"loss={t_loss.item():.4f} elapsed={elapsed_t:.0f}s",
                            flush=True,
                        )
                t_sch.step()
                print(f"  [Teacher] Ep {t_ep+1:2d}/{kd_teacher_epochs}  total_loss={t_loss_sum/t_n:.4f} time={time.time()-t0_ep:.0f}s", flush=True)
            torch.save(kd_teacher.state_dict(), teacher_ckpt)
            print(f"[KD] Teacher saved → {teacher_ckpt}")

        kd_teacher.eval()


    model = ASPSegmentor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Input channels: {in_ch} (xyz"
          f"{'+rgb' if getattr(cfg, 'use_rgb', True) else ''}"
          f"{'+height' if getattr(cfg, 'use_height', True) else ''})")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_lambda(epoch):
        warmup = getattr(cfg, 'warmup_epochs', 5)
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_amp)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-1,  # safety: if any invalid labels exist
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        best_miou = ckpt.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou*100:.2f}%")

    # ── Logging — use ONE shared log across all resume runs ───────────
    # s3dis_train_log.csv accumulates across runs; timestamped runs get their own file too
    shared_log_path = os.path.join(cfg.log_dir, 's3dis_train_log.csv')
    run_name = f"s3dis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = shared_log_path  # always append to the shared log
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,miou,macc,oa,lr,time\n")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        tau = max(cfg.tau_end, cfg.tau_start * (cfg.tau_decay ** epoch))
        model.gumbel_tau.fill_(tau)

        # Reset LIF spike statistics for sparsity rate calculation
        if hasattr(model, 'lif_head') and hasattr(model.lif_head, 'reset_spike_stats'):
            model.lif_head.reset_spike_stats()

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = n_batches = 0
        n_total_batches = len(train_loader)
        log_every = max(1, n_total_batches // 20)  # ~20 progress prints per epoch

        for batch_idx, (slices, geo, pts_feat, sid_arr, sem_labels, cat_ids) in enumerate(train_loader):
            slices = slices.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            pts_feat = pts_feat.to(device, non_blocking=True)
            sid_arr = sid_arr.to(device, non_blocking=True)
            sem_labels = sem_labels.to(device, non_blocking=True)
            cat_ids = cat_ids.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=cfg.use_amp):
                logits_final, logits_all = model(
                    slices, geo, sid_arr, cat_ids, pts_feat, training=True
                )
                # logits_final: [B, N, 13]
                B, N, C = logits_final.shape
                
                loss = active_loss_seg(
                    logits_final.reshape(B * N, C),
                    logits_all,
                    sem_labels.reshape(B * N),
                    criterion
                )

                if kd_teacher is not None:
                    with torch.no_grad():
                        t_logits = kd_teacher(pts_feat)
                    loss = loss + kd_lam * seg_kd_loss(logits_final, t_logits, kd_temp)
                
                # Add spike firing-rate penalty if applicable
                if hasattr(model, 'lif_head') and hasattr(model.lif_head, 'mean_firing_rate'):
                    loss = loss + 0.01 * model.lif_head.mean_firing_rate()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            # Per-batch progress with ETA — fixes "no console output for 8 minutes"
            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_total_batches:
                elapsed = time.time() - t0
                per_batch = elapsed / (batch_idx + 1)
                remaining = per_batch * (n_total_batches - batch_idx - 1)
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(
                    f"  ep{epoch+1} [{batch_idx+1:4d}/{n_total_batches}] "
                    f"loss={loss.item():.4f} eta={remaining:.0f}s "
                    f"gpu_mem={gpu_mem:.1f}GB",
                    flush=True,
                )

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]['lr']

        # ── Eval ──────────────────────────────────────────────────────
        eval_interval = getattr(cfg, 'eval_interval', 5)
        if (epoch + 1) % eval_interval == 0 or epoch == cfg.epochs - 1:
            model.eval()
            all_preds, all_true = [], []

            with torch.no_grad():
                for slices, geo, pts_feat, sid_arr, sem_labels, cat_ids in test_loader:
                    slices = slices.to(device)
                    geo = geo.to(device)
                    pts_feat = pts_feat.to(device)
                    sid_arr = sid_arr.to(device)
                    cat_ids = cat_ids.to(device)

                    logits_final, _ = model(
                        slices, geo, sid_arr, cat_ids, pts_feat, training=False
                    )
                    preds = logits_final.argmax(dim=-1)  # [B, N]
                    all_preds.append(preds.cpu().numpy().reshape(-1))
                    all_true.append(sem_labels.numpy().reshape(-1))

            all_preds = np.concatenate(all_preds)
            all_true = np.concatenate(all_true)
            miou, macc, oa, per_class = compute_iou(
                all_preds, all_true, NUM_CLASSES
            )

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} | "
                f"mIoU={miou*100:.2f}% mAcc={macc*100:.2f}% "
                f"OA={oa*100:.2f}% | {elapsed:.0f}s"
            )

            if (epoch + 1) % 25 == 0 or epoch == cfg.epochs - 1:
                for name, iou in sorted(per_class.items(),
                                        key=lambda x: x[1] if not np.isnan(x[1]) else 0):
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

    # Plot training curves from shared log
    plot_training_curves(log_path, cfg.log_dir)


if __name__ == "__main__":
    main()
