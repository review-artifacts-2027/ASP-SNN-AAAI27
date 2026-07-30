"""
datasets/slicing.py — FPS, KNN slicing, and geometry descriptor computation.

All point cloud datasets are sliced into M local patches via:
    1. Farthest Point Sampling (FPS) to find M anchor points
    2. KNN around each anchor to form slices of K points each
    3. 8-dim geometry descriptor per slice

Geometry descriptor (8-dim):
    [0:3]  centroid xyz
    [3:6]  per-axis variance
    [6]    max distance from centroid (used for sorting)
    [7]    distance from slice centroid to cloud centroid
"""

import numpy as np


def fps(points: np.ndarray, npoint: int, seed: int = None) -> np.ndarray:
    """
    Iterative Farthest Point Sampling on xyz (first 3 columns).

    Args:
        points: [N, C] with C >= 3
        npoint: number of anchor points to select
        seed:   if provided, fixes the random starting point for
                reproducible slicing at test time.

    Returns:
        centroids: [npoint] int32 indices into points
    """
    N = points.shape[0]
    xyz = points[:, :3]
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N, dtype=np.float64) * 1e10
    if seed is not None:
        rng = np.random.default_rng(seed)
        farthest = int(rng.integers(0, N))
    else:
        farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        farthest = int(np.argmax(distance))
    return centroids


def build_slices(points: np.ndarray, anchors: np.ndarray,
                 k: int = 64) -> np.ndarray:
    """
    KNN slicing around anchor points.

    Args:
        points:  [N, C]
        anchors: [M] indices
        k:       points per slice

    Returns:
        slices: [M, K, C]
    """
    xyz = points[:, :3]
    N = points.shape[0]
    slices = []
    for idx in anchors:
        dist = np.linalg.norm(xyz - xyz[idx], axis=1)
        nn_idx = np.argsort(dist)[:k]
        # If fewer than k points available, pad by repeating last
        if len(nn_idx) < k:
            pad = np.repeat(nn_idx[-1:], k - len(nn_idx))
            nn_idx = np.concatenate([nn_idx, pad])
        slices.append(points[nn_idx])
    return np.stack(slices)


def compute_geo(slice_pts: np.ndarray) -> np.ndarray:
    """
    8-dim geometry descriptor for one slice (numpy version).

    Args:
        slice_pts: [K, C] with C >= 3

    Returns:
        geo: [8] float32
    """
    xyz = slice_pts[:, :3]
    centroid = xyz.mean(axis=0)
    variance = xyz.var(axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=1)
    max_dist = float(np.max(dists))
    dist_to_origin = float(np.linalg.norm(centroid))
    return np.concatenate([
        centroid, variance, [max_dist], [dist_to_origin]
    ]).astype(np.float32)


def compute_geo_torch(slices: "torch.Tensor") -> "torch.Tensor":
    """
    Torch version of compute_geo that operates batched on GPU.

    Args:
        slices: [B, M, K, C] with C >= 3

    Returns:
        geo: [B, M, 8] float32

    Used during TTA to recompute geo descriptors after rotation augmentation
    on GPU, avoiding stale variance/max_dist fields.
    """
    import torch
    xyz = slices[..., :3]                                            # [B,M,K,3]
    centroid = xyz.mean(dim=2)                                       # [B,M,3]
    variance = xyz.var(dim=2, unbiased=False)                        # [B,M,3]
    dists = torch.linalg.norm(
        xyz - centroid.unsqueeze(2), dim=-1
    )                                                                 # [B,M,K]
    max_dist = dists.max(dim=-1).values.unsqueeze(-1)                # [B,M,1]
    dist_to_origin = torch.linalg.norm(
        centroid, dim=-1, keepdim=True
    )                                                                 # [B,M,1]
    return torch.cat([centroid, variance, max_dist, dist_to_origin], dim=-1)


def slice_point_cloud(points: np.ndarray, num_slices: int = 16,
                      points_per_slice: int = 64, seed: int = None):
    """
    Full slicing pipeline: FPS -> KNN -> geo descriptors.

    Args:
        points:          [N, C] point cloud (C >= 3)
        num_slices:      M
        points_per_slice: K
        seed:            optional FPS seed for deterministic test-time slicing

    Returns:
        slices:     [M, K, C]
        geo:        [M, 8]
        anchor_xyz: [M, 3]
    """
    anchors = fps(points, num_slices, seed=seed)
    slices = build_slices(points, anchors, points_per_slice)
    geo = np.stack([compute_geo(s) for s in slices])
    anchor_xyz = points[anchors, :3]
    return slices, geo, anchor_xyz


def assign_points_to_slices(pts_xyz: np.ndarray,
                            anchor_xyz: np.ndarray) -> np.ndarray:
    """
    Assign each point to its nearest anchor (for segmentation).

    Args:
        pts_xyz:    [N, 3]
        anchor_xyz: [M, 3]

    Returns:
        sid_arr: [N] int32, slice index per point
    """
    # [N, M] pairwise distances
    dists = np.linalg.norm(
        pts_xyz[:, None, :] - anchor_xyz[None, :, :], axis=2
    )
    return dists.argmin(axis=1).astype(np.int32)