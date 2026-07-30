"""
eval_shapenet.py — Evaluate ASP-SNN on ShapeNetPart test set.

Usage:
    python eval_shapenet.py --ckpt checkpoints/shapenet_best.pt
    python eval_shapenet.py --ckpt ... --per_cat
    python eval_shapenet.py --ckpt ... --energy       # Batch 5: full energy report

Batch 5 update — full energy report:
    When --energy is passed, attaches spike-rate monitors to every spiking
    component (LIF head, spiking encoder if used, spiking seg head if used)
    and prints a full breakdown including system-level α ratio. This is the
    number that goes in the paper's efficiency headline.
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


# ── Encoder LIF hook-based monitor (mirrors eval_scanobj.py) ──────────────

class _EncoderSpikeMonitor:
    """
    Hooks the spiking encoder's LIF layers and records their spike outputs
    to compute mean firing rate for the energy report. Attach with
    .attach(spiking_encoder_impl), detach with .detach().
    """

    def __init__(self):
        self.total_spikes = 0.0
        self.total_neurons = 0
        self._handles = []

    def _make_hook(self):
        def hook(module, inp, out):
            # `out` is a spike tensor {0, 1}. Sum gives spike count; numel
            # gives total neurons across the batch.
            self.total_spikes += float(out.sum().item())
            self.total_neurons += out.numel()
        return hook

    def attach(self, spiking_encoder_impl):
        """Attach hooks to every _EncoderLIF found on the module tree."""
        # Try known attribute names first for a clean hook.
        for name in ['lif1', 'lif2_edge', 'lif3', 'lif4']:
            layer = getattr(spiking_encoder_impl, name, None)
            if layer is not None:
                h = layer.register_forward_hook(self._make_hook())
                self._handles.append(h)

        # Fallback: any submodule whose class name looks like a LIF layer.
        if len(self._handles) == 0:
            for m in spiking_encoder_impl.modules():
                cls_name = m.__class__.__name__
                if 'LIF' in cls_name or cls_name.endswith('LIF'):
                    h = m.register_forward_hook(self._make_hook())
                    self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def mean_rate(self):
        if self.total_neurons == 0:
            return 0.0
        return self.total_spikes / self.total_neurons

    def reset(self):
        self.total_spikes = 0.0
        self.total_neurons = 0


# ── mIoU (unchanged) ──────────────────────────────────────────────────────

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
    p.add_argument("--energy", action="store_true",
                   help="Attach spike monitors and print energy report (Batch 5)")
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
    cfg.num_classes    = NUM_PARTS
    cfg.num_categories = NUM_CATEGORIES
    cfg.use_category   = True
    cfg.in_channels    = 6
    model = ASPSegmentor(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] missing keys (using fresh init): {len(missing)} entries")
    if unexpected:
        print(f"  [load] unexpected keys (ignored):        {len(unexpected)} entries")
    model.eval()

    enc_type = getattr(cfg, 'encoder_type', 'analog')
    seg_type = getattr(cfg, 'seg_head_type', 'analog')
    print(f"\nCheckpoint : {args.ckpt}")
    print(f"Epoch      : {ckpt.get('epoch', '?')}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder    : {enc_type}"
          f"{f'  (T_enc={cfg.encoder_T})' if enc_type == 'spiking' else ''}")
    print(f"Seg head   : {seg_type}"
          f"{f'  (T_seg={cfg.seg_head_T})' if seg_type == 'spiking' else ''}")

    # ── Batch 5: attach spike-rate monitors when --energy ──────────────
    head_logger    = None
    enc_monitor    = None
    seg_head_logger = None

    if args.energy:
        from models.lif import SpikeRateLogger

        # LIF head (always spiking)
        head_logger = SpikeRateLogger()
        model.lif_head.spike_monitor = head_logger

        # Spiking encoder (only if encoder_type == 'spiking')
        if enc_type == 'spiking':
            impl = getattr(model.feature_extractor, 'impl', None)
            if impl is not None and impl.__class__.__name__ == 'SpikingEdgeConvEncoder':
                enc_monitor = _EncoderSpikeMonitor()
                enc_monitor.attach(impl)
            else:
                print("  WARN: encoder_type=spiking but SpikingEdgeConvEncoder "
                      "not found on model.feature_extractor.impl — encoder "
                      "firing rate will not be measured.")

        # Spiking seg head (only if seg_head_type == 'spiking')
        if seg_type == 'spiking':
            seg_head_logger = SpikeRateLogger()
            if hasattr(model.seg_head, 'spike_monitor'):
                model.seg_head.spike_monitor = seg_head_logger
            else:
                print("  WARN: seg_head_type=spiking but seg_head has no "
                      "spike_monitor attribute — seg-head firing rate will "
                      "not be measured.")

    # ── Evaluate ──────────────────────────────────────────────────────
    all_preds, all_true, all_cats = [], [], []

    with torch.no_grad():
        for batch in loader:
            (coarse_slices, coarse_geo, pts_xyz, coarse_sid_arr,
             fine_slices,   fine_geo,   fine_sid_arr,
             part_labels, _bnd_labels, cat_ids) = batch

            coarse_slices  = coarse_slices.to(device)
            coarse_geo     = coarse_geo.to(device)
            pts_xyz        = pts_xyz.to(device)
            coarse_sid_arr = coarse_sid_arr.to(device)
            fine_slices    = fine_slices.to(device)
            fine_geo       = fine_geo.to(device)
            fine_sid_arr   = fine_sid_arr.to(device)
            cat_ids        = cat_ids.to(device)
            B, N = part_labels.shape

            part_logits, _ = model(
                coarse_slices, coarse_geo, coarse_sid_arr,
                cat_ids, pts_xyz,
                fine_slices=fine_slices,
                fine_geo=fine_geo,
                fine_sid_arr=fine_sid_arr,
                training=False,
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

    # ── Batch 5: Energy report ────────────────────────────────────────
    if args.energy:
        from models.energy import compute_energy, print_energy_report

        head_fr = head_logger.mean_rate() if head_logger else 0.0
        enc_fr  = enc_monitor.mean_rate() if enc_monitor else None
        seg_fr  = seg_head_logger.mean_rate() if seg_head_logger else None

        energy = compute_energy(
            cfg,
            mean_firing_rate_head=head_fr,
            mean_firing_rate_encoder=enc_fr,
            mean_firing_rate_seg_head=seg_fr,
        )
        print_energy_report(energy)

        # Detach monitors for cleanliness
        model.lif_head.spike_monitor = None
        if enc_monitor is not None:
            enc_monitor.detach()
        if seg_head_logger is not None and hasattr(model.seg_head, 'spike_monitor'):
            model.seg_head.spike_monitor = None

    print()


if __name__ == "__main__":
    main()