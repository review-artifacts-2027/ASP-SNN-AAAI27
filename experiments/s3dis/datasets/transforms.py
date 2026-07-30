"""
datasets/transforms.py — Point cloud augmentation functions.

Two main entry points:
    augment_slices()  — for classification (modifies slices only)
    augment_seg()     — for segmentation (shared transform on slices + pts)

Augmentations added (novel contributions on top of baseline):
    pointwolf_seg()   — simplified PointWOLF local non-linear distortion.
                        Selects N_anchors random points, applies local
                        anisotropic scaling to nearby points. Simulates
                        real-world LiDAR scan non-linearities.

All augmentations are numpy-based and applied in __getitem__.
"""

import numpy as np


def random_rotation_z() -> np.ndarray:
    """Random rotation matrix around z-axis. Returns [3, 3]."""
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], np.float32)


def random_so3_tilt(max_rad: float = 0.26) -> np.ndarray:
    """SO(3) tilt: random z-rotation + small x/y tilts. Returns [3, 3]."""
    Rz = random_rotation_z()
    ax = np.random.uniform(-max_rad, max_rad)
    ay = np.random.uniform(-max_rad, max_rad)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    Rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]], np.float32)
    Ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]], np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def augment_slices(slices: np.ndarray, cfg) -> np.ndarray:
    """
    Augment slices for classification tasks (ScanObjectNN).

    Args:
        slices: [M, K, C]
        cfg:    config with aug_* parameters

    Returns:
        augmented slices: [M, K, C]
    """
    M, K, C = slices.shape
    result = slices.copy()

    # Rotation
    if getattr(cfg, 'aug_rotate_so3', False):
        R = random_so3_tilt(getattr(cfg, 'aug_so3_tilt', 0.26))
    elif getattr(cfg, 'aug_rotate_z', False):
        R = random_rotation_z()
    else:
        R = np.eye(3, dtype=np.float32)

    result[:, :, :3] = result[:, :, :3] @ R
    if C >= 6:
        result[:, :, 3:6] = result[:, :, 3:6] @ R

    # Anisotropic scale
    lo = getattr(cfg, 'aug_scale_lo', 0.8)
    hi = getattr(cfg, 'aug_scale_hi', 1.25)
    scale = np.random.uniform(lo, hi, (1, 1, 3)).astype(np.float32)
    result[:, :, :3] *= scale

    # Translation
    t_range = getattr(cfg, 'aug_translate', 0.1)
    trans = np.random.uniform(-t_range, t_range, (1, 1, 3)).astype(np.float32)
    result[:, :, :3] += trans

    # Jitter
    sigma = getattr(cfg, 'aug_jitter_sigma', 0.01)
    clip = getattr(cfg, 'aug_jitter_clip', 0.05)
    jitter = np.clip(np.random.normal(0, sigma, (M, K, 3)), -clip, clip)
    result[:, :, :3] += jitter.astype(np.float32)

    # Per-slice point dropout
    drop_rate = getattr(cfg, 'aug_slice_dropout', 0.1)
    for m in range(M):
        ratio = np.random.random() * drop_rate
        drop_idx = np.where(np.random.random(K) < ratio)[0]
        if len(drop_idx) > 0:
            result[m, drop_idx] = result[m, 0]

    # Global point dropout (ScanObjectNN)
    pdrop = getattr(cfg, 'aug_point_dropout', 0.0)
    if pdrop > 0:
        ratio = np.random.random() * pdrop
        for m in range(M):
            drop_idx = np.where(np.random.random(K) < ratio)[0]
            if len(drop_idx) > 0:
                result[m, drop_idx] = result[m, 0]

    return result.astype(np.float32)


def pointwolf_seg(slices: np.ndarray, pts_features: np.ndarray,
                  cfg) -> tuple:
    """
    Simplified PointWOLF: Local non-linear distortion.

    Selects num_anchors random anchor points and applies local anisotropic
    scaling to all points within a radius of each anchor.
    Applied consistently to both slices and per-point features.

    Reference: Kim et al., PointWOLF (ECCV 2022).

    Args:
        slices:       [M, K, C]
        pts_features: [N, F]  (first 3 dims are xyz)
        cfg:          config with aug_pointwolf_* parameters

    Returns:
        (augmented_slices, augmented_pts_features)
    """
    if not getattr(cfg, 'aug_pointwolf', False):
        return slices, pts_features

    n_anchors  = getattr(cfg, 'aug_pointwolf_num_anchors', 3)
    radius     = getattr(cfg, 'aug_pointwolf_radius', 0.3)
    scale_lo   = getattr(cfg, 'aug_pointwolf_scale_lo', 0.7)
    scale_hi   = getattr(cfg, 'aug_pointwolf_scale_hi', 1.3)

    slices_out = slices.copy()
    pts_out    = pts_features.copy()

    # Sample anchors from per-point xyz (always first 3 dims of pts_features)
    N = pts_features.shape[0]
    anchor_idxs = np.random.choice(N, n_anchors, replace=False)
    anchors = pts_features[anchor_idxs, :3]  # [n_anchors, 3]

    for anchor in anchors:
        scale = np.random.uniform(scale_lo, scale_hi, 3).astype(np.float32)

        # Apply to pts_features xyz
        delta_pts = pts_out[:, :3] - anchor  # [N, 3]
        dist_pts  = np.linalg.norm(delta_pts, axis=1)  # [N]
        mask_pts  = dist_pts < radius
        if mask_pts.sum() > 0:
            pts_out[mask_pts, :3] = anchor + delta_pts[mask_pts] * scale

        # Apply same distortion to slices xyz (each slice is [K, C])
        M, K, _ = slices_out.shape
        for m in range(M):
            delta_sl = slices_out[m, :, :3] - anchor  # [K, 3]
            dist_sl  = np.linalg.norm(delta_sl, axis=1)  # [K]
            mask_sl  = dist_sl < radius
            if mask_sl.sum() > 0:
                slices_out[m, mask_sl, :3] = (
                    anchor + delta_sl[mask_sl] * scale
                )

    return slices_out.astype(np.float32), pts_out.astype(np.float32)


def augment_seg(slices: np.ndarray, pts_features: np.ndarray,
                cfg) -> tuple:
    """
    Augment slices AND per-point features with SHARED transform.

    Critical: slices and pts must receive the same rotation/scale/translate
    otherwise the segmentation head learns to ignore xyz features.

    Args:
        slices:       [M, K, C]
        pts_features: [N, F]  (first 3 dims are xyz)
        cfg:          config

    Returns:
        (augmented_slices, augmented_pts_features)
    """
    M, K, C = slices.shape
    N, F = pts_features.shape

    # Shared rotation
    if getattr(cfg, 'aug_rotate_so3', False):
        R = random_so3_tilt(getattr(cfg, 'aug_so3_tilt', 0.26))
    elif getattr(cfg, 'aug_rotate_z', True):
        R = random_rotation_z()
    else:
        R = np.eye(3, dtype=np.float32)

    # Shared scale
    lo = getattr(cfg, 'aug_scale_lo', 0.8)
    hi = getattr(cfg, 'aug_scale_hi', 1.25)
    scale = np.random.uniform(lo, hi, (1, 3)).astype(np.float32)

    # Shared translation
    t_range = getattr(cfg, 'aug_translate', 0.1)
    trans = np.random.uniform(-t_range, t_range, (1, 3)).astype(np.float32)

    # Apply to pts_features FIRST (xyz in first 3 dims)
    pts_aug = pts_features.copy()
    pts_aug[:, :3] = (pts_aug[:, :3] @ R) * scale + trans

    # Apply to slices
    result = slices.copy()
    result[:, :, :3] = (result[:, :, :3] @ R) * scale + trans
    if C >= 6:
        # Rotate normals / rgb stays untouched (normals are in 3:6 for ShapeNet)
        # For S3DIS, channels 3:6 are RGB — do NOT rotate them
        if not getattr(cfg, 'use_rgb', False):
            result[:, :, 3:6] = result[:, :, 3:6] @ R

    # Jitter (slices only)
    sigma = getattr(cfg, 'aug_jitter_sigma', 0.01)
    clip = getattr(cfg, 'aug_jitter_clip', 0.05)
    jitter = np.clip(np.random.normal(0, sigma, (M, K, 3)), -clip, clip)
    result[:, :, :3] += jitter.astype(np.float32)

    # Per-slice dropout
    drop_rate = getattr(cfg, 'aug_slice_dropout', 0.1)
    for m in range(M):
        ratio = np.random.random() * drop_rate
        drop_idx = np.where(np.random.random(K) < ratio)[0]
        if len(drop_idx) > 0:
            result[m, drop_idx] = result[m, 0]

    # Color augmentation (S3DIS)
    if getattr(cfg, 'use_rgb', False) and F > 3:
        # Color dropout: zero out RGB with probability
        cdrop = getattr(cfg, 'aug_color_drop', 0.0)
        if cdrop > 0 and np.random.random() < cdrop:
            pts_aug[:, 3:6] = 0.0
            result[:, :, 3:6] = 0.0

        # Color jitter (additive)
        cjit = getattr(cfg, 'aug_color_jitter', 0.0)
        if cjit > 0:
            noise = np.random.uniform(-cjit, cjit, (1, 3)).astype(np.float32)
            pts_aug[:, 3:6] = np.clip(pts_aug[:, 3:6] + noise, 0.0, 1.0)
            result[:, :, 3:6] = np.clip(result[:, :, 3:6] + noise.reshape(1,1,3), 0.0, 1.0)

        # Color brightness multiplicative jitter
        cbright = getattr(cfg, 'aug_color_bright', 0.0)
        if cbright > 0:
            factor = np.random.uniform(1.0 - cbright, 1.0 + cbright, (1, 3)).astype(np.float32)
            pts_aug[:, 3:6] = np.clip(pts_aug[:, 3:6] * factor, 0.0, 1.0)
            result[:, :, 3:6] = np.clip(result[:, :, 3:6] * factor.reshape(1,1,3), 0.0, 1.0)

    # Global point dropout (simulate sparse LiDAR scans)
    # Randomly zero-out a fraction of points in the block (slices + pts)
    gpdrop = getattr(cfg, 'aug_global_point_dropout', 0.0)
    if gpdrop > 0 and np.random.random() < 0.5:  # apply with 50% probability
        ratio = np.random.uniform(0, gpdrop)
        # Drop from pts_aug
        drop_mask = np.random.random(N) < ratio
        if drop_mask.sum() > 0:
            pts_aug[drop_mask] = pts_aug[0]  # replace with first point
        # Drop from slices
        for m in range(M):
            sl_drop = np.random.random(K) < ratio
            if sl_drop.sum() > 0:
                result[m, sl_drop] = result[m, 0]

    # PointWOLF local non-linear distortion (applied after rigid transforms)
    if getattr(cfg, 'aug_pointwolf', False):
        result, pts_aug = pointwolf_seg(result, pts_aug, cfg)

    return result.astype(np.float32), pts_aug.astype(np.float32)
