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


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _make_valid_mask(cat_ids, num_parts, device):
    B = cat_ids.shape[0]
    cat_ids_cpu = cat_ids.detach().cpu().tolist()
    valid_mask = torch.zeros(B, num_parts, device=device, dtype=torch.bool)
    for b in range(B):
        for pid in CATEGORY_TO_PARTS[cat_ids_cpu[b]]:
            valid_mask[b, pid] = True
    return valid_mask


def _seg_ce_from_mask(part_logits, part_labels, valid_mask):
    B, N, P = part_logits.shape
    mask_expanded = valid_mask.unsqueeze(1).expand(B, N, P)
    logits_masked = part_logits.float().clone()
    logits_masked[~mask_expanded] = -1e9

    logits_flat = logits_masked.reshape(B * N, P)
    labels_flat = part_labels.reshape(B * N)
    return F.cross_entropy(logits_flat, labels_flat)


def seg_loss_fn(part_logits, part_labels, cat_ids):
    valid_mask = _make_valid_mask(cat_ids, part_logits.shape[-1],
                                  part_logits.device)
    return _seg_ce_from_mask(part_logits, part_labels, valid_mask)


def compute_seg_loss(part_logits, per_timestep_logits, part_labels, cat_ids,
                     loss_mode='ce', tet_lambda=0.0):
    device     = part_logits.device
    num_parts  = part_logits.shape[-1]
    valid_mask = _make_valid_mask(cat_ids, num_parts, device)

    use_dense = (loss_mode == 'dense_tet'
                 and per_timestep_logits is not None
                 and len(per_timestep_logits) > 0)

    if use_dense:
        ce_terms = torch.stack([
            _seg_ce_from_mask(lt, part_labels, valid_mask)
            for lt in per_timestep_logits
        ])
        loss = ce_terms.mean()
        if tet_lambda > 0 and len(per_timestep_logits) > 1:
            final = per_timestep_logits[-1].detach()
            mse_terms = torch.stack([
                F.mse_loss(lt, final)
                for lt in per_timestep_logits[:-1]
            ])
            loss = loss + tet_lambda * mse_terms.mean()
        return loss

    return _seg_ce_from_mask(part_logits, part_labels, valid_mask)


def compute_boundary_loss(bnd_logits, bnd_labels, pos_weight=None):
    if bnd_logits is None:
        return torch.zeros((), device=bnd_labels.device)

    if pos_weight is None:
        num_pos = bnd_labels.sum().clamp(min=1.0)
        num_neg = (bnd_labels.numel() - num_pos).clamp(min=1.0)
        pw = (num_neg / num_pos).clamp(max=20.0)
    else:
        pw = torch.tensor(float(pos_weight), device=bnd_logits.device)

    return F.binary_cross_entropy_with_logits(
        bnd_logits.reshape(-1),
        bnd_labels.reshape(-1),
        pos_weight=pw,
    )


def warm_start_encoder_from_analog(model, ckpt_path: str, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"warm_start_from checkpoint not found: {ckpt_path}")

    print(f"\n[warm-start] loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    src_state = ckpt.get('model', ckpt)
    src_state = {k.replace('module.', ''): v for k, v in src_state.items()}

    missing, unexpected = model.load_state_dict(src_state, strict=False)
    loaded = [k for k in src_state.keys() if k not in unexpected]

    print(f"  loaded  {len(loaded):>4d} tensors (compatible with new model)")
    print(f"  skipped {len(unexpected):>4d} tensors (structure changed / removed)")
    print(f"  fresh   {len(missing):>4d} tensors (LIF-specific / new components)")

    return loaded, unexpected


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
    torch.backends.cudnn.benchmark = True

    parser = base_argparser("ASP-SNN ShapeNetPart Training")
    args = parser.parse_args()
    overrides = parse_overrides(args)

    config_path = args.config or "configs/shapenet_seg.yaml"
    cfg = load_config(config_path, overrides)
    set_seed(cfg.seed)
    device = cfg.device

    n_gpus = torch.cuda.device_count()
    use_dp = (n_gpus > 1) and (device.type == 'cuda')

    enc_type   = getattr(cfg, 'encoder_type', 'analog')
    seg_type   = getattr(cfg, 'seg_head_type', 'analog')
    warm_start = getattr(cfg, 'warm_start_from', None)

    print(f"\n{'='*60}")
    print(f"  ASP-SNN ShapeNetPart Part Segmentation")
    print(f"  Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")
    print(f"  Encoder    : {enc_type}"
          f"{f'  (T_enc={cfg.encoder_T})' if enc_type == 'spiking' else ''}")
    print(f"  Seg head   : {seg_type}"
          f"{f'  (T_seg={cfg.seg_head_T})' if seg_type == 'spiking' else ''}")
    print(f"  Warm start : {warm_start or 'no (from scratch)'}")
    print(f"  Device     : {device}")
    print(f"  GPUs       : {n_gpus}"
          f"{'  (DataParallel)' if use_dp else ''}")
    print(f"{'='*60}\n")

    train_ds = ShapeNetPartDataset(cfg.data_dir, 'train', cfg)
    test_ds  = ShapeNetPartDataset(cfg.data_dir, 'test',  cfg)

    pw_ = cfg.num_workers > 0
    drop_last = len(train_ds) >= cfg.batch_size * 2
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=drop_last, persistent_workers=pw_,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=pw_,
    )

    cfg.num_classes    = NUM_PARTS
    cfg.num_categories = NUM_CATEGORIES
    cfg.use_category   = True
    cfg.in_channels    = 6

    model = ASPSegmentor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    if warm_start:
        warm_start_encoder_from_analog(model, warm_start, device)

    loss_mode        = getattr(cfg, 'loss_mode', 'ce')
    tet_lambda       = getattr(cfg, 'tet_lambda', 0.0)
    bnd_loss_weight  = getattr(cfg, 'bnd_loss_weight', 0.1)
    use_bnd          = getattr(cfg, 'use_boundary_aware', True)
    print(f"Loss mode: {loss_mode}" +
          (f" (TET lambda={tet_lambda})" if loss_mode == 'dense_tet' else ""))
    if use_bnd:
        print(f"Boundary BCE weight: {bnd_loss_weight}")

    enc_scale = getattr(cfg, 'encoder_lr_scale', 0.1)
    encoder_params = (
        list(model.feature_extractor.parameters()) +
        list(model.slice_transformer.parameters()) +
        list(model.pos_proj.parameters())
    )
    if model.fine_encoder is not None:
        encoder_params += list(model.fine_encoder.parameters())
    if model.fine_pos_proj is not None:
        encoder_params += list(model.fine_pos_proj.parameters())

    enc_ids    = set(id(p) for p in encoder_params)
    new_params = [p for p in model.parameters() if id(p) not in enc_ids]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg.lr * enc_scale},
        {"params": new_params,     "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)

    def lr_lambda(epoch):
        warmup = getattr(cfg, 'warmup_epochs', 10)
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_amp)

    start_epoch = 0
    best_inst_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f"  [resume] missing keys: {len(missing)}")
        if unexpected:
            print(f"  [resume] unexpected keys: {len(unexpected)}")
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        best_inst_iou = ckpt.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_inst_iou*100:.2f}%")

    if use_dp:
        gpu_ids = list(range(n_gpus))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[DP] Wrapping model with DataParallel across GPUs {gpu_ids}\n")

    run_name = f"shapenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(cfg.log_dir, f"{run_name}.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,inst_miou,cls_miou,lr,time\n")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        tau = max(cfg.tau_end, cfg.tau_start * (cfg.tau_decay ** epoch))

        unwrap(model).gumbel_tau.fill_(tau)

        model.train()
        total_loss = n_batches = 0
        n_total_batches = len(train_loader)
        log_every = max(1, n_total_batches // 10)

        for batch_idx, batch in enumerate(train_loader):
            (coarse_slices, coarse_geo, pts_xyz, coarse_sid_arr,
             fine_slices,   fine_geo,   fine_sid_arr,
             part_labels, bnd_labels, cat_ids) = batch

            coarse_slices   = coarse_slices.to(device,   non_blocking=True)
            coarse_geo      = coarse_geo.to(device,      non_blocking=True)
            pts_xyz         = pts_xyz.to(device,         non_blocking=True)
            coarse_sid_arr  = coarse_sid_arr.to(device,  non_blocking=True)
            fine_slices     = fine_slices.to(device,     non_blocking=True)
            fine_geo        = fine_geo.to(device,        non_blocking=True)
            fine_sid_arr    = fine_sid_arr.to(device,    non_blocking=True)
            part_labels     = part_labels.to(device,     non_blocking=True)
            bnd_labels      = bnd_labels.to(device,      non_blocking=True)
            cat_ids         = cat_ids.to(device,         non_blocking=True)

            with autocast(device_type=device.type, enabled=cfg.use_amp):
                part_logits, aux = model(
                    coarse_slices, coarse_geo, coarse_sid_arr,
                    cat_ids, pts_xyz,
                    fine_slices=fine_slices,
                    fine_geo=fine_geo,
                    fine_sid_arr=fine_sid_arr,
                    training=True,
                )
                per_t     = aux.get('per_timestep_logits', None) if isinstance(aux, dict) else None
                bnd_logts = aux.get('bnd_logits', None) if isinstance(aux, dict) else None

                seg_loss = compute_seg_loss(
                    part_logits, per_t, part_labels, cat_ids,
                    loss_mode=loss_mode, tet_lambda=tet_lambda,
                )

                if use_bnd and bnd_logts is not None:
                    bnd_loss = compute_boundary_loss(bnd_logts, bnd_labels)
                    loss = seg_loss + bnd_loss_weight * bnd_loss
                else:
                    loss = seg_loss

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

        eval_interval = getattr(cfg, 'eval_interval', 5)
        if (epoch + 1) % eval_interval == 0 or epoch == cfg.epochs - 1:
            model.eval()
            all_preds, all_true, all_cats = [], [], []

            with torch.no_grad():
                for batch in test_loader:
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

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} | "
                f"Inst mIoU={inst_iou*100:.2f}% "
                f"Cls mIoU={cls_iou*100:.2f}% | {elapsed:.0f}s"
            )

            if (epoch + 1) % 25 == 0:
                for cn, iou in sorted(per_cat.items(), key=lambda x: x[1]):
                    print(f"    {cn:<14} {iou*100:5.1f}%")

            if inst_iou > best_inst_iou:
                best_inst_iou = inst_iou

                torch.save({
                    'epoch': epoch + 1,
                    'model': unwrap(model).state_dict(),
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

        torch.save({
            'epoch': epoch + 1,
            'model': unwrap(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_metric': best_inst_iou,
        }, os.path.join(cfg.ckpt_dir, 'shapenet_last.pt'))

    print(f"\nDone. Best Instance mIoU: {best_inst_iou*100:.2f}%")
    print(f"Checkpoint: {cfg.ckpt_dir}/shapenet_best.pt")

if __name__ == "__main__":
    main()
