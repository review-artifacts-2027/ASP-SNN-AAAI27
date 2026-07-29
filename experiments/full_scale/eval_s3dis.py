"""
eval_s3dis.py — Evaluate ASP-SNN on S3DIS Area 5 test set.

Reports mIoU, mAcc, OA, and optional per-class IoU breakdown.

Usage:
    python eval_s3dis.py --ckpt checkpoints/s3dis_best.pt [--per_class]
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_config, set_seed
from datasets.s3dis import S3DISDataset, CLASS_NAMES, NUM_CLASSES
from models.asp_segmentor import ASPSegmentor


def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Compute mIoU, mAcc, OA, per-class IoU, per-class Acc."""
    ious = []
    accs = []
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
    return miou, macc, oa, ious, accs


def main():
    p = argparse.ArgumentParser(description="Evaluate S3DIS")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/s3dis_seg.yaml")
    p.add_argument("--per_class", action="store_true",
                   help="Print per-class IoU and accuracy")
    p.add_argument("--batch", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.batch:
        cfg.batch_size = args.batch
    set_seed(cfg.seed)
    device = cfg.device

    # Dataset
    test_ds = S3DISDataset(cfg.data_dir, 'test', cfg)
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Model
    in_ch = 3
    if getattr(cfg, 'use_rgb', True):
        in_ch += 3
    if getattr(cfg, 'use_height', True):
        in_ch += 1
    cfg.in_channels = in_ch
    cfg.num_classes = NUM_CLASSES
    cfg.use_category = False
    cfg.num_categories = 0

    model = ASPSegmentor(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    # Handle DataParallel-saved checkpoints
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    print(f"\nCheckpoint : {args.ckpt}")
    print(f"Epoch      : {ckpt.get('epoch', '?')}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Test blocks: {len(test_ds)}")

    # Evaluate
    all_preds, all_true = [], []

    with torch.no_grad():
        for slices, geo, pts_feat, sid_arr, sem_labels, cat_ids in loader:
            slices = slices.to(device)
            geo = geo.to(device)
            pts_feat = pts_feat.to(device)
            sid_arr = sid_arr.to(device)
            cat_ids = cat_ids.to(device)

            logits, _ = model(
                slices, geo, sid_arr, cat_ids, pts_feat, training=False
            )
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_true.append(sem_labels.numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    miou, macc, oa, ious, accs = compute_metrics(
        all_preds, all_true, NUM_CLASSES
    )

    print(f"\n{'='*50}")
    print(f"  mIoU              : {miou*100:.2f}%")
    print(f"  mAcc              : {macc*100:.2f}%")
    print(f"  Overall Accuracy  : {oa*100:.2f}%")
    print(f"  Points evaluated  : {len(all_preds):,}")
    print(f"{'='*50}")

    if args.per_class:
        print(f"\n  {'Class':<12} {'IoU':>6}  {'Acc':>6}")
        print(f"  {'-'*28}")
        for i in range(NUM_CLASSES):
            iou_v = ious[i] * 100 if not np.isnan(ious[i]) else 0.0
            acc_v = accs[i] * 100 if not np.isnan(accs[i]) else 0.0
            bar = "#" * int(iou_v / 100 * 20)
            print(f"  {CLASS_NAMES[i]:<12} {iou_v:5.1f}% {acc_v:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()