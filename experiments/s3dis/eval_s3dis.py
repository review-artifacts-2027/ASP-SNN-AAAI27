"""
eval_s3dis.py — Evaluate ASP-SNN on S3DIS Area 5 test set.

Tier A upgrades:
  A1. Per-room softmax probability aggregation across overlapping blocks.
      The sliding-window test blocks overlap at 50% stride. Previously,
      predictions from overlapping blocks were simply concatenated and
      every point entered the mIoU calculation once per block — leading
      to inconsistent predictions being double-counted. Now, per-point
      softmax probabilities are accumulated into a room-level buffer
      keyed by the point's room-relative index, and the final prediction
      is argmax over the averaged probability. Each point appears EXACTLY
      ONCE in the metric.

  A2. Test-time voting (TTA) via random z-rotation. The same block is run
      n_votes times with different random z-rotations, and the softmax
      probabilities are averaged before argmax. The first vote is always
      the identity transform (theta=0) for determinism. Geometry descriptors
      are recomputed on-GPU from the rotated slices.

Tier E upgrade:
  E1. Room-level prior passthrough — when cfg.use_room_prior=True, the
      dataset returns precomputed room summaries alongside each block.
      The evaluator forwards them to the model (which injects them into
      the initial LIF belief state). When use_room_prior=False, behavior
      is identical to the pre-E1 pipeline.

Usage:
    python eval_s3dis.py --ckpt checkpoints/s3dis_best.pt --n_votes 3 --per_class
"""

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_config, set_seed
from datasets.s3dis import S3DISDataset, CLASS_NAMES, NUM_CLASSES
from datasets.slicing import compute_geo_torch
from models.asp_segmentor import ASPSegmentor


def _rotate_z_batch(tensor: torch.Tensor, theta: float) -> torch.Tensor:
    """
    Rotate the first 3 channels of the last dimension by angle theta
    around the z-axis. Works on any tensor shape [..., C] where C >= 3.

    Non-xyz channels (RGB, height, etc.) are left untouched.
    """
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor(
        [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]],
        dtype=tensor.dtype, device=tensor.device,
    )
    out = tensor.clone()
    # [..., :3] @ R^T  =  [..., :3] @ R  (R is orthogonal and column-format)
    out[..., :3] = out[..., :3] @ R
    return out


@torch.no_grad()
def evaluate_s3dis_aggregated(model, test_ds, cfg, device,
                              n_votes: int = 1):
    """
    Per-room probability aggregation with optional TTA voting.

    For each overlapping sliding-window block, the model's per-point softmax
    probabilities are accumulated into a room-level [P_room, C] buffer using
    the point's room-relative index. After all blocks (and votes) are
    processed, each point's final prediction is argmax over its averaged
    probability, so a point appears exactly once in the metric.

    Args:
        model:    ASPSegmentor in eval mode
        test_ds:  S3DISDataset with split='test'
        cfg:      config namespace
        device:   torch device
        n_votes:  TTA count. First vote = identity; subsequent votes = random
                  z-rotation with geo descriptors recomputed on-GPU.

    Returns:
        preds_all: [P_total] int64  concatenated predictions (unique room pts)
        true_all:  [P_total] int64  matching ground-truth labels
    """
    # Enable metadata output (room_idx + orig_indices) from the dataset
    test_ds._return_meta = True

    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Pre-allocate room-level probability accumulators
    room_probs = [
        np.zeros((len(r), NUM_CLASSES), dtype=np.float64)
        for r in test_ds.rooms
    ]
    room_counts = [
        np.zeros(len(r), dtype=np.int32) for r in test_ds.rooms
    ]

    use_room_prior = getattr(cfg, 'use_room_prior', False)

    for vote in range(max(1, n_votes)):
        # First vote: identity. Subsequent: random z-rotation.
        theta = 0.0 if vote == 0 else float(
            np.random.uniform(0, 2 * math.pi)
        )

        for batch in loader:
            # ── Unpack ────────────────────────────────────────────
            # Dataset with _return_meta=True returns:
            #   Without room prior: 8 items (6 base + room_idx + orig_indices)
            #   With room prior:    9 items (7 base + room_idx + orig_indices)
            #
            # The room_idx and orig_indices are always the LAST two items.
            #
            idx_of_room_id = -2
            idx_of_orig    = -1

            slices     = batch[0].to(device, non_blocking=True)
            geo        = batch[1].to(device, non_blocking=True)
            pts_feat   = batch[2].to(device, non_blocking=True)
            sid_arr    = batch[3].to(device, non_blocking=True)
            # sem_labels = batch[4]  — not needed for inference
            cat_ids    = batch[5].to(device, non_blocking=True)

            if use_room_prior and len(batch) == 9:
                room_summary = batch[6].to(device, non_blocking=True)
            else:
                room_summary = None

            room_ids     = batch[idx_of_room_id]    # keep on CPU
            orig_indices = batch[idx_of_orig]        # keep on CPU

            # ── TTA rotation ──────────────────────────────────────
            if theta != 0.0:
                slices   = _rotate_z_batch(slices, theta)
                pts_feat = _rotate_z_batch(pts_feat, theta)
                # Recompute geo from rotated slices so positional encoding
                # and SSP stay consistent (P0 fix pattern).
                geo = compute_geo_torch(slices)

            # ── Forward ───────────────────────────────────────────
            logits, _ = model(
                slices, geo, sid_arr, cat_ids, pts_feat,
                room_summary=room_summary,
                training=False,
            )
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()  # [B,N,C]

            room_ids_np = room_ids.numpy().astype(np.int64)
            orig_np     = orig_indices.numpy().astype(np.int64)  # [B, N]

            # ── Accumulate per-room ───────────────────────────────
            for b in range(probs.shape[0]):
                r_id = int(room_ids_np[b])
                idx  = orig_np[b]
                # np.add.at handles duplicate indices correctly
                np.add.at(room_probs[r_id], idx, probs[b])
                np.add.at(room_counts[r_id], idx, 1)

    # ── Finalize: one prediction per unique room point ────────────────
    all_preds, all_true = [], []
    for r_id in range(len(test_ds.rooms)):
        counts = room_counts[r_id]
        covered = counts > 0
        if not covered.any():
            continue
        avg = room_probs[r_id][covered] / counts[covered, None]
        preds = avg.argmax(axis=1).astype(np.int64)
        true = test_ds.rooms[r_id][covered, 6].astype(np.int64)
        all_preds.append(preds)
        all_true.append(true)

    # Reset the flag so the dataset returns to its normal shape
    test_ds._return_meta = False

    return np.concatenate(all_preds), np.concatenate(all_true)


def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    num_classes: int):
    """Compute mIoU, mAcc, OA, per-class IoU, per-class Acc."""
    ious, accs = [], []
    for c in range(num_classes):
        pred_c = (pred == c)
        true_c = (target == c)
        inter  = np.logical_and(pred_c, true_c).sum()
        union  = np.logical_or(pred_c, true_c).sum()
        ious.append(inter / union if union > 0 else float('nan'))
        true_count = true_c.sum()
        accs.append(inter / true_count if true_count > 0 else float('nan'))

    miou = float(np.nanmean(ious))
    macc = float(np.nanmean(accs))
    oa   = float((pred == target).sum() / max(len(target), 1))
    return miou, macc, oa, ious, accs


def main():
    p = argparse.ArgumentParser(description="Evaluate ASP-SNN on S3DIS")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to model checkpoint")
    p.add_argument("--config", type=str, default="configs/s3dis_seg.yaml",
                   help="Path to config YAML")
    p.add_argument("--per_class", action="store_true",
                   help="Print per-class IoU and accuracy")
    p.add_argument("--n_votes", type=int, default=3,
                   help="Number of TTA votes (1 = no augmentation)")
    p.add_argument("--batch", type=int, default=None,
                   help="Override batch size")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.batch:
        cfg.batch_size = args.batch
    set_seed(cfg.seed)
    device = cfg.device

    # ── Dataset ───────────────────────────────────────────────────────
    test_ds = S3DISDataset(cfg.data_dir, 'test', cfg)

    # ── Model config ──────────────────────────────────────────────────
    in_ch = 3
    if getattr(cfg, 'use_rgb', True):
        in_ch += 3
    if getattr(cfg, 'use_height', True):
        in_ch += 1
    cfg.in_channels    = in_ch
    cfg.num_classes    = NUM_CLASSES
    cfg.use_category   = False
    cfg.num_categories = 0

    model = ASPSegmentor(cfg).to(device)

    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    # strict=False to tolerate missing E1 keys in pre-E1 checkpoints
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ckpt] missing keys ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    model.eval()

    print(f"\nCheckpoint : {args.ckpt}")
    print(f"Epoch      : {ckpt.get('epoch', '?')}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Test rooms : {len(test_ds.rooms)}")
    print(f"Test blocks: {len(test_ds)}")
    print(f"TTA votes  : {args.n_votes}")
    print(f"Room prior : {getattr(cfg, 'use_room_prior', False)}")

    # ── Run evaluation ────────────────────────────────────────────────
    preds, true = evaluate_s3dis_aggregated(
        model, test_ds, cfg, device, n_votes=args.n_votes,
    )

    miou, macc, oa, ious, accs = compute_metrics(preds, true, NUM_CLASSES)

    print(f"\n{'='*50}")
    print(f"  mIoU              : {miou*100:.2f}%")
    print(f"  mAcc              : {macc*100:.2f}%")
    print(f"  Overall Accuracy  : {oa*100:.2f}%")
    print(f"  Points evaluated  : {len(preds):,}  (unique room points)")
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