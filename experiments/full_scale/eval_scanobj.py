"""
eval_scanobj.py — Evaluate ASP-SNN on ScanObjectNN PB-T50-RS test set.

Supports test-time augmentation (TTA) with N-vote averaging.

Usage:
    python eval_scanobj.py --ckpt checkpoints/scanobj_best.pt
    python eval_scanobj.py --ckpt checkpoints/scanobj_best.pt --n_votes 10
"""

import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_config, set_seed
from datasets.scanobjectnn import ScanObjectNNDataset
from datasets.slicing import compute_geo_torch
from models.asp_classifier import ASPClassifier


def augment_vote_gpu(slices, geo):
    """
    One random z-rotation augmentation on GPU tensors for TTA.

    P1 FIX: previously only the centroid (geo[:,:,:3]) was rotated and
    variance/max_dist were left stale. Now we recompute the FULL 8-dim geo
    descriptor from the rotated slices so SSP sees consistent geometry.
    """
    device, dtype = slices.device, slices.dtype
    theta = float(np.random.uniform(0, 2 * math.pi))
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor(
        [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]],
        device=device, dtype=dtype,
    )

    B, M, K, C = slices.shape
    slices_aug = slices.clone()
    # Rotate xyz channels
    slices_aug[:, :, :, :3] = slices[:, :, :, :3] @ rot
    # If normals are present (ShapeNet uses zero-pad, ScanObjectNN uses zero)
    # rotate them too — harmless for zeros, correct for real normals
    if C >= 6:
        slices_aug[:, :, :, 3:6] = slices[:, :, :, 3:6] @ rot

    # Recompute the full 8-dim geo descriptor (centroid, variance, max_dist,
    # dist_to_origin) from the rotated slices on GPU
    geo_aug = compute_geo_torch(slices_aug)

    return slices_aug, geo_aug


def evaluate(model, loader, device, n_votes=1):
    """Evaluate with optional TTA."""
    model.eval()
    all_probs = []
    all_labels = []
    total_slices = 0
    total_samples = 0

    with torch.no_grad():
        for slices, geo, labels in loader:
            slices = slices.to(device)
            geo = geo.to(device)
            B = slices.shape[0]

            summed = torch.zeros(B, model.cfg.num_classes, device=device)

            for v in range(n_votes):
                if v == 0:
                    s_v, g_v = slices, geo
                else:
                    s_v, g_v = augment_vote_gpu(slices, geo)

                logits_all = model(s_v, g_v, training=False)
                summed += logits_all[-1].softmax(dim=-1)

                if v == 0:
                    total_slices += len(logits_all) * B
                    total_samples += B

            all_probs.append((summed / n_votes).cpu())
            all_labels.append(labels)

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = probs.argmax(dim=-1)

    # Overall accuracy
    oa = (preds == labels).float().mean().item()

    # Per-class accuracy
    num_classes = model.cfg.num_classes
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        per_class_total[c] = mask.sum().item()
        per_class_correct[c] = (preds[mask] == c).sum().item()

    per_class_acc = per_class_correct / per_class_total.clamp(min=1)
    macc = per_class_acc.mean().item()
    avg_slices = total_slices / max(total_samples, 1)

    return oa, macc, per_class_acc, avg_slices


def main():
    p = argparse.ArgumentParser(description="Evaluate ScanObjectNN")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/scanobj_cls.yaml")
    p.add_argument("--n_votes", type=int, default=1,
                   help="Number of TTA votes (1=no TTA)")
    p.add_argument("--batch", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.batch:
        cfg.batch_size = args.batch
    set_seed(cfg.seed)
    device = cfg.device

    # Dataset
    test_ds = ScanObjectNNDataset(cfg.data_dir, 'test', cfg)
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Model
    cfg.in_channels = 6
    model = ASPClassifier(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    # Handle DataParallel-saved checkpoints
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)

    print(f"\nCheckpoint : {args.ckpt}")
    print(f"Epoch      : {ckpt.get('epoch', '?')}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"TTA votes  : {args.n_votes}")

    # Evaluate
    oa, macc, per_class_acc, avg_slices = evaluate(
        model, loader, device, args.n_votes
    )

    print(f"\n{'='*50}")
    print(f"  Overall Accuracy  : {oa*100:.2f}%")
    print(f"  Mean Class Acc    : {macc*100:.2f}%")
    print(f"  Avg slices used   : {avg_slices:.2f} / {cfg.T}")
    print(f"{'='*50}")

    # Per-class
    print(f"\n  Per-class accuracy ({cfg.num_classes} classes):")
    for c in range(cfg.num_classes):
        acc = per_class_acc[c].item()
        bar = "#" * int(acc * 30)
        print(f"    Class {c:2d}  {acc*100:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
