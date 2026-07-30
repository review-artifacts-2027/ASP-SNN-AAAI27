"""
datasets/transforms.py — Data augmentation for point cloud tasks.

Batch 3 adds `augment_points_only`: applies rigid transform + jitter to a
RAW point cloud, before slicing. This is the entry point used by the
Batch 3 shapenet loader, which slices AFTER augmentation. Applying the
transform once at the point level automatically keeps coarse and fine
slicings geometrically consistent, and also fixes the classic aug/geo
bug where geometry descriptors were computed pre-augmentation on slices
that were then augmented (leading to catastrophic train/val gap).

Only the point-level ShapeNetPart augmentation path is retained here.
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
