"""
smoke_test.py — Quick sanity check that all models build and forward-pass
without errors. Runs on CPU with tiny synthetic data in ~30 seconds.

Usage:
    python smoke_test.py

This does NOT require any downloaded datasets. It creates random tensors
matching the exact shapes each model expects, runs a forward + backward
pass, and verifies output shapes. If this passes, the code has no import
errors, shape mismatches, or device issues.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from config import Config


def make_cfg(**overrides):
    """Create a minimal config object for testing."""
    defaults = dict(
        feat_dim=512, k_edge=10, transformer_heads=4, transformer_ffn_dim=512,
        hidden_dim=512, num_lif_layers=3, lif_leak=0.9, lif_threshold=1.0,
        d_ssp=128, T=3, exit_threshold=0.4,
        tau_start=1.0, tau_end=0.1, tau_decay=0.95,
        num_slices=4, points_per_slice=16, geo_dim=8,
        num_points=64, num_classes=15, in_channels=6,
        num_categories=0, use_category=False, point_feat_dim=64,
        use_rgb=False, use_height=False,
        cls_head_dims=None, cls_head_dropout=None,
    )
    defaults.update(overrides)
    return Config(defaults)


def test_classifier():
    """Test ASPClassifier with synthetic data."""
    print("[1/8] Testing ASPClassifier ... ", end="", flush=True)
    from models.asp_classifier import ASPClassifier

    cfg = make_cfg(num_classes=15, cls_head_dims=[128, 64],
                   cls_head_dropout=[0.3, 0.2])
    model = ASPClassifier(cfg)
    model.train()

    B, M, K, C = 2, cfg.num_slices, cfg.points_per_slice, 6
    slices = torch.randn(B, M, K, C)
    geo = torch.randn(B, M, 8)
    geo[:, :, 6] = torch.rand(B, M)  # max_dist used for sorting

    logits_all = model(slices, geo, training=True)
    assert len(logits_all) == cfg.T, f"Expected {cfg.T} timesteps, got {len(logits_all)}"
    assert logits_all[-1].shape == (B, 15), f"Wrong shape: {logits_all[-1].shape}"

    # Backward pass
    loss = sum(F.cross_entropy(lg, torch.zeros(B, dtype=torch.long))
               for lg in logits_all)
    loss.backward()

    # Inference mode (with early exit)
    model.eval()
    with torch.no_grad():
        logits_eval = model(slices, geo, training=False)
    assert logits_eval[-1].shape == (B, 15)

    print(f"OK  ({len(logits_all)} train steps, {len(logits_eval)} eval steps)")


def test_segmentor_shapenet():
    """Test ASPSegmentor in ShapeNetPart mode."""
    print("[2/8] Testing ASPSegmentor (ShapeNet) ... ", end="", flush=True)
    from models.asp_segmentor import ASPSegmentor

    cfg = make_cfg(num_classes=50, num_categories=16,
                   use_category=True, in_channels=6)
    model = ASPSegmentor(cfg)
    model.train()

    B, M, K = 2, cfg.num_slices, cfg.points_per_slice
    N = cfg.num_points
    slices = torch.randn(B, M, K, 6)
    geo = torch.randn(B, M, 8)
    geo[:, :, 6] = torch.rand(B, M)
    sid_arr = torch.randint(0, M, (B, N))
    cat_ids = torch.randint(0, 16, (B,))
    pts_xyz = torch.randn(B, N, 3)

    logits, beliefs = model(slices, geo, sid_arr, cat_ids, pts_xyz, training=True)
    assert logits.shape == (B, N, 50), f"Wrong shape: {logits.shape}"
    assert len(beliefs) == cfg.T

    loss = F.cross_entropy(logits.reshape(B * N, 50),
                           torch.randint(0, 50, (B * N,)))
    loss.backward()
    print(f"OK  (output {logits.shape})")


def test_segmentor_s3dis():
    """Test ASPSegmentor in S3DIS mode (no category, 7-ch input)."""
    print("[3/8] Testing ASPSegmentor (S3DIS) ... ", end="", flush=True)
    from models.asp_segmentor import ASPSegmentor

    cfg = make_cfg(num_classes=13, num_categories=0,
                   use_category=False, in_channels=7,
                   use_rgb=True, use_height=True)
    model = ASPSegmentor(cfg)
    model.train()

    B, M, K = 2, cfg.num_slices, cfg.points_per_slice
    N = cfg.num_points
    slices = torch.randn(B, M, K, 7)
    geo = torch.randn(B, M, 8)
    geo[:, :, 6] = torch.rand(B, M)
    sid_arr = torch.randint(0, M, (B, N))
    cat_ids = torch.zeros(B, dtype=torch.long)
    pts_feat = torch.randn(B, N, 7)  # xyz + rgb + height

    logits, beliefs = model(slices, geo, sid_arr, cat_ids, pts_feat, training=True)
    assert logits.shape == (B, N, 13), f"Wrong shape: {logits.shape}"

    loss = F.cross_entropy(logits.reshape(B * N, 13),
                           torch.randint(0, 13, (B * N,)))
    loss.backward()
    print(f"OK  (output {logits.shape})")


def test_slicing():
    """Test FPS + KNN slicing pipeline."""
    print("[4/8] Testing slicing pipeline ... ", end="", flush=True)
    from datasets.slicing import slice_point_cloud, assign_points_to_slices

    N, C, M, K = 256, 6, 8, 32
    pts = np.random.randn(N, C).astype(np.float32)
    slices, geo, anchors = slice_point_cloud(pts, M, K)
    assert slices.shape == (M, K, C), f"Slices: {slices.shape}"
    assert geo.shape == (M, 8), f"Geo: {geo.shape}"
    assert anchors.shape == (M, 3), f"Anchors: {anchors.shape}"

    sid = assign_points_to_slices(pts[:, :3], anchors)
    assert sid.shape == (N,), f"SID: {sid.shape}"
    assert sid.min() >= 0 and sid.max() < M

    print(f"OK  (slices {slices.shape}, geo {geo.shape})")


def test_early_exit():
    """Verify early exit does not crash and returns valid logits."""
    print("[5/8] Testing early exit ... ", end="", flush=True)
    from models.asp_classifier import ASPClassifier

    cfg = make_cfg(num_classes=15, exit_threshold=0.0)  # always exit at t=0
    model = ASPClassifier(cfg)
    model.eval()
    B, M, K = 2, cfg.num_slices, cfg.points_per_slice
    slices = torch.randn(B, M, K, 6)
    geo = torch.randn(B, M, 8)
    geo[:, :, 6] = torch.rand(B, M)
    with torch.no_grad():
        logits = model(slices, geo, training=False)
    assert len(logits) >= 1, "Early exit returned empty logits"
    assert logits[-1].shape == (B, 15)
    print(f"OK  (exited at t={len(logits)})")


def test_batch_size_1():
    """BatchNorm can break with batch_size=1 in train mode. Verify eval works."""
    print("[6/8] Testing batch_size=1 eval ... ", end="", flush=True)
    from models.asp_classifier import ASPClassifier

    cfg = make_cfg(num_classes=15)
    model = ASPClassifier(cfg)
    model.eval()
    slices = torch.randn(1, cfg.num_slices, cfg.points_per_slice, 6)
    geo = torch.randn(1, cfg.num_slices, 8)
    geo[:, :, 6] = torch.rand(1, cfg.num_slices)
    with torch.no_grad():
        out = model(slices, geo, training=False)
    assert out[-1].shape == (1, 15), "Batch size 1 eval failed"
    print("OK")


def test_foveater_asp():
    """Test FoveaTer ASP image model with synthetic ImageNet-like data."""
    print("[7/8] Testing FoveaTerASP (ImageNet) ... ", end="", flush=True)
    from models.foveater_asp import FoveaTerASP

    model = FoveaTerASP(
        num_classes=10,
        image_size=64,
        feature_grid=4,
        embed_dim=48,
        depth=1,
        num_heads=3,
        max_fixations=2,
        max_tokens=8,
    )
    model.train()

    images = torch.randn(2, 3, 64, 64)
    labels = torch.randint(0, 10, (2,))

    logits_final, logits_all = model.forward_active_train(
        images, max_fixations=2, random_initial=True
    )
    assert len(logits_all) == 2, f"Expected 2 fixations, got {len(logits_all)}"
    assert logits_final.shape == (2, 10), f"Wrong shape: {logits_final.shape}"
    loss = sum(F.cross_entropy(logits, labels) for logits in logits_all)
    loss.backward()

    model.eval()
    with torch.no_grad():
        logits, exit_step, history = model.forward_active_infer(
            images, threshold=2.0, max_fixations=2, initial_fixation="center"
        )
    assert logits.shape == (2, 10), f"Wrong eval shape: {logits.shape}"
    assert exit_step == 2, f"Expected full 2-fixation eval, got {exit_step}"
    assert history.shape == (2, 2, 2), f"Wrong fixation history: {history.shape}"
    print(f"OK  ({len(logits_all)} fixations)")


def test_checkpoint_roundtrip():
    """Verify save/load cycle preserves model predictions."""
    print("[8/8] Testing checkpoint save/load ... ", end="", flush=True)
    import tempfile
    from models.asp_classifier import ASPClassifier

    cfg = make_cfg(num_classes=15)
    model = ASPClassifier(cfg)
    model.eval()
    slices = torch.randn(1, cfg.num_slices, cfg.points_per_slice, 6)
    geo = torch.randn(1, cfg.num_slices, 8)
    geo[:, :, 6] = torch.rand(1, cfg.num_slices)
    with torch.no_grad():
        out1 = model(slices, geo, training=False)[-1]
    path = "test_ckpt.pt"
    try:
        with open(path, 'wb') as f:
            torch.save({'model': model.state_dict()}, f)
        with open(path, 'rb') as f:
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
        model2 = ASPClassifier(cfg)
        model2.load_state_dict(ckpt['model'])
        model2.eval()
        with torch.no_grad():
            out2 = model2(slices, geo, training=False)[-1]
    finally:
        import gc
        gc.collect()  # force garbage collection to release torch.load file handles on Windows
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass
    assert torch.allclose(out1, out2, atol=1e-5), "Checkpoint round-trip changed predictions"
    print("OK")


def main():
    print("=" * 55)
    print("  ASP-SNN Smoke Test")
    print("  Device: CPU (no GPU required)")
    print("=" * 55)
    print()

    passed = 0
    failed = 0

    for test_fn in [test_classifier, test_segmentor_shapenet,
                    test_segmentor_s3dis, test_slicing,
                    test_early_exit, test_batch_size_1,
                    test_foveater_asp, test_checkpoint_roundtrip]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL  ({e})")

    print()
    print("=" * 55)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
        print("  The code is ready to run on real data.")
    else:
        print(f"  {passed} passed, {failed} FAILED")
        print("  Fix the failures before running on cluster.")
    print("=" * 55)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
