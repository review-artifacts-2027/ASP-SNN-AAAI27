"""
eval_shapenet.py — Evaluate ASP-SNN on ShapeNetPart test set.

Usage:
    python eval_shapenet.py --ckpt checkpoints/shapenet_best.pt [--per_cat]
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_config, set_seed
from datasets.shapenetpart import (
    ShapeNetPartDataset, CATEGORY_TO_PARTS, CATEGORY_NAMES,
    NUM_PARTS, NUM_CATEGORIES,
)
from models.asp_segmentor import ASPSegmentor


def compute_instance_miou(pred_parts, true_parts, cat_ids, n_points):
    """Compute instance mIoU, class mIoU, and per-category IoU."""
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
            per_cat[CATEGORY_NAMES[cat]] = float('nan')
    cls_miou = float(np.mean(populated_ious)) if populated_ious else 0.0
    return inst_miou, cls_miou, per_cat


def main():
    p = argparse.ArgumentParser(description="Evaluate ShapeNetPart")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/shapenet_seg.yaml")
    p.add_argument("--per_cat", action="store_true",
                   help="Print per-category IoU breakdown")
    p.add_argument("--batch", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.batch:
        cfg.batch_size = args.batch
    set_seed(cfg.seed)
    device = cfg.device

    # Dataset
    test_ds = ShapeNetPartDataset(cfg.data_dir, 'test', cfg)
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Model
    cfg.num_classes = NUM_PARTS
    cfg.num_categories = NUM_CATEGORIES
    cfg.use_category = True
    cfg.in_channels = 6
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

    # Evaluate
    all_preds, all_true, all_cats = [], [], []

    with torch.no_grad():
        for slices, geo, pts_xyz, sid_arr, part_labels, cat_ids in loader:
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

    print(f"\n{'='*50}")
    print(f"  Instance mIoU : {inst_iou*100:.2f}%")
    print(f"  Class mIoU    : {cls_iou*100:.2f}%")
    print(f"{'='*50}")

    if args.per_cat:
        print(f"\n  Per-category IoU:")
        for name, iou in sorted(per_cat.items(), key=lambda x: x[1]):
            bar = "#" * int(iou * 30)
            print(f"    {name:<14} {iou*100:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
