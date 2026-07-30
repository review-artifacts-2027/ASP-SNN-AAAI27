"""
datasets/transforms.py — Data augmentation for point cloud tasks.

Batch 3 adds `augment_points_only`: applies rigid transform + jitter to a
RAW point cloud, before slicing. This is the entry point used by the
Batch 3 shapenet loader, which slices AFTER augmentation. Applying the
transform once at the point level automatically keeps coarse and fine
slicings geometrically consistent, and also fixes the classic aug/geo
bug where geometry descriptors were computed pre-augmentation on slices
that were then augmented (leading to catastrophic train/val gap).

Batch 1 A4 additions retained in augment_seg (used by S3DIS):
    - Wider anisotropic scale range
    - Optional random axis mirror
    - Tighter jitter defaults
    - Optional random point resampling
"""

import numpy as np


# ── Rotation primitives (unchanged from original) ─────────────────────

def random_rotation_z():
    """Rotation about the z axis. Returns [3, 3]."""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]], dtype=np.float32)


def random_so3_tilt(max_rad=0.26):
    """Small perturbation from upright: yaw + small pitch/roll.
    Returns [3, 3]."""
    Rz = random_rotation_z()
    ax = np.random.uniform(-max_rad, max_rad)
    ay = np.random.uniform(-max_rad, max_rad)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    Rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]], np.float32)
    Ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]], np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _random_mirror(R: np.ndarray, cfg) -> np.ndarray:
    """
    Apply random axis-mirror to a rotation matrix.
    Left-multiplies R by a diagonal flip. Handled in matrix form so it
    composes cleanly with rotation/scale.
    """
    flip_x = getattr(cfg, 'aug_flip_x', False) and (np.random.random() < 0.5)
    flip_y = getattr(cfg, 'aug_flip_y', False) and (np.random.random() < 0.5)
    flip_z = getattr(cfg, 'aug_flip_z', False) and (np.random.random() < 0.5)
    if not (flip_x or flip_y or flip_z):
        return R
    D = np.eye(3, dtype=np.float32)
    if flip_x: D[0, 0] = -1.
    if flip_y: D[1, 1] = -1.
    if flip_z: D[2, 2] = -1.
    return (D @ R).astype(np.float32)


# ── Classification-side augmentation (unchanged) ──────────────────────

def augment_slices(slices: np.ndarray, cfg) -> np.ndarray:
    """
    Augment slices for classification tasks (ScanObjectNN).

    UNCHANGED — do not touch the ScanObjectNN tuning that already produces
    good results.
    """
    M, K, C = slices.shape
    result = slices.copy()

    if getattr(cfg, 'aug_rotate_so3', False):
        R = random_so3_tilt(getattr(cfg, 'aug_so3_tilt', 0.26))
    elif getattr(cfg, 'aug_rotate_z', False):
        R = random_rotation_z()
    else:
        R = np.eye(3, dtype=np.float32)

    result[:, :, :3] = result[:, :, :3] @ R
    if C >= 6:
        result[:, :, 3:6] = result[:, :, 3:6] @ R

    lo = getattr(cfg, 'aug_scale_lo', 0.8)
    hi = getattr(cfg, 'aug_scale_hi', 1.25)
    scale = np.random.uniform(lo, hi, (1, 1, 3)).astype(np.float32)
    result[:, :, :3] *= scale

    t_range = getattr(cfg, 'aug_translate', 0.1)
    trans = np.random.uniform(-t_range, t_range, (1, 1, 3)).astype(np.float32)
    result[:, :, :3] += trans

    sigma = getattr(cfg, 'aug_jitter_sigma', 0.01)
    clip  = getattr(cfg, 'aug_jitter_clip', 0.05)
    jitter = np.clip(np.random.normal(0, sigma, (M, K, 3)), -clip, clip)
    result[:, :, :3] += jitter.astype(np.float32)

    drop_rate = getattr(cfg, 'aug_slice_dropout', 0.1)
    for m in range(M):
        ratio = np.random.random() * drop_rate
        drop_idx = np.where(np.random.random(K) < ratio)[0]
        if len(drop_idx) > 0:
            result[m, drop_idx] = result[m, 0]

    pdrop = getattr(cfg, 'aug_point_dropout', 0.0)
    if pdrop > 0:
        ratio = np.random.random() * pdrop
        for m in range(M):
            drop_idx = np.where(np.random.random(K) < ratio)[0]
            if len(drop_idx) > 0:
                result[m, drop_idx] = result[m, 0]

    return result.astype(np.float32)


# ── Batch 3: point-level augmentation (used by ShapeNet) ──────────────

def augment_points_only(pts_xyz: np.ndarray, cfg) -> np.ndarray:
    """
    Apply rigid transform + jitter to a raw point cloud BEFORE slicing.

    This is the Batch 3 entry point for ShapeNetPart. Slicing happens
    downstream on the augmented points, which:
      1) Keeps coarse and fine slicings geometrically consistent (both
         see the SAME augmented data).
      2) Eliminates the classic aug/geo bug where geometry descriptors
         are computed pre-augmentation on slices that get augmented after.

    Args:
        pts_xyz: [N, 3] normalised xyz
        cfg:     config object with aug_* fields

    Returns:
        pts_aug: [N, 3] augmented xyz (float32)
    """
    N = pts_xyz.shape[0]

    # Rotation
    if getattr(cfg, 'aug_rotate_so3', False):
        R = random_so3_tilt(getattr(cfg, 'aug_so3_tilt', 0.26))
    elif getattr(cfg, 'aug_rotate_z', True):
        R = random_rotation_z()
    else:
        R = np.eye(3, dtype=np.float32)

    # Optional random axis mirror (A4)
    R = _random_mirror(R, cfg)

    # Anisotropic scale (A4 defaults are 0.66 / 1.5)
    lo = getattr(cfg, 'aug_scale_lo', 0.66)
    hi = getattr(cfg, 'aug_scale_hi', 1.5)
    scale = np.random.uniform(lo, hi, (1, 3)).astype(np.float32)

    # Translation
    t_range = getattr(cfg, 'aug_translate', 0.1)
    trans = np.random.uniform(-t_range, t_range, (1, 3)).astype(np.float32)

    # Apply rigid transform
    pts_aug = pts_xyz.copy()
    pts_aug = (pts_aug @ R) * scale + trans

    # Jitter (A4 defaults are σ=0.005, clip=0.02)
    sigma = getattr(cfg, 'aug_jitter_sigma', 0.005)
    clip  = getattr(cfg, 'aug_jitter_clip',  0.02)
    if sigma > 0:
        jitter = np.clip(
            np.random.normal(0, sigma, pts_aug.shape), -clip, clip
        ).astype(np.float32)
        pts_aug = pts_aug + jitter

    return pts_aug.astype(np.float32)


# ── S3DIS-side augmentation (unchanged, kept for backward compat) ─────

def augment_seg(slices: np.ndarray, pts_features: np.ndarray,
                cfg, part_labels: np.ndarray = None) -> tuple:
    """
    S3DIS-style: augment already-sliced tensors + per-point features with a
    shared transform. Preserved unchanged for S3DIS backwards compat.

    ShapeNetPart Batch 3 no longer uses this — it augments points BEFORE
    slicing via augment_points_only.
    """
    M, K, C = slices.shape
    N, F = pts_features.shape

    if getattr(cfg, 'aug_rotate_so3', False):
        R = random_so3_tilt(getattr(cfg, 'aug_so3_tilt', 0.26))
    elif getattr(cfg, 'aug_rotate_z', True):
        R = random_rotation_z()
    else:
        R = np.eye(3, dtype=np.float32)

    R = _random_mirror(R, cfg)

    lo = getattr(cfg, 'aug_scale_lo', 0.66)
    hi = getattr(cfg, 'aug_scale_hi', 1.5)
    scale = np.random.uniform(lo, hi, (1, 3)).astype(np.float32)

    t_range = getattr(cfg, 'aug_translate', 0.1)
    trans = np.random.uniform(-t_range, t_range, (1, 3)).astype(np.float32)

    pts_aug = pts_features.copy()
    pts_aug[:, :3] = (pts_aug[:, :3] @ R) * scale + trans

    result = slices.copy()
    result[:, :, :3] = (result[:, :, :3] @ R) * scale + trans
    if C >= 6:
        if not getattr(cfg, 'use_rgb', False):
            result[:, :, 3:6] = result[:, :, 3:6] @ R

    sigma = getattr(cfg, 'aug_jitter_sigma', 0.005)
    clip  = getattr(cfg, 'aug_jitter_clip',  0.02)
    jitter = np.clip(np.random.normal(0, sigma, (M, K, 3)), -clip, clip)
    result[:, :, :3] += jitter.astype(np.float32)

    drop_rate = getattr(cfg, 'aug_slice_dropout', 0.1)
    for m in range(M):
        ratio = np.random.random() * drop_rate
        drop_idx = np.where(np.random.random(K) < ratio)[0]
        if len(drop_idx) > 0:
            result[m, drop_idx] = result[m, 0]

    if getattr(cfg, 'use_rgb', False) and F > 3:
        cdrop = getattr(cfg, 'aug_color_drop', 0.0)
        if cdrop > 0 and np.random.random() < cdrop:
            pts_aug[:, 3:6] = 0.0
            result[:, :, 3:6] = 0.0
        cjit = getattr(cfg, 'aug_color_jitter', 0.0)
        if cjit > 0:
            noise = np.random.uniform(-cjit, cjit, (1, 3)).astype(np.float32)
            pts_aug[:, 3:6] = np.clip(pts_aug[:, 3:6] + noise, 0.0, 1.0)
            result[:, :, 3:6] = np.clip(
                result[:, :, 3:6] + noise.reshape(1, 1, 3), 0.0, 1.0
            )

    do_resample = getattr(cfg, 'aug_resample', False)
    if do_resample:
        idx = np.random.choice(N, size=N, replace=True)
        pts_aug = pts_aug[idx]
        if part_labels is not None:
            part_labels_out = part_labels[idx]
    else:
        if part_labels is not None:
            part_labels_out = part_labels

    if part_labels is not None:
        return (result.astype(np.float32),
                pts_aug.astype(np.float32),
                part_labels_out)
    return result.astype(np.float32), pts_aug.astype(np.float32)