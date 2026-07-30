"""FPS slicing and 6-D geometry descriptors (computed offline; zero MACs online).

Descriptor g_m in R^6 per FPS anchor m (paper Sec. 2.1):
    [0:3] centroid XYZ of the slice (normalized cloud frame)
    [3]   distance of slice centroid to the point-cloud centroid
    [4]   intra-cluster spread  (mean point distance to slice centroid)
    [5]   normalized point count (Voronoi occupancy of the anchor / N)

Component names (for the A3 ablation): cx, cy, cz, dist, spread, count.
"""
from __future__ import annotations

import torch

DESC_COMPONENTS = ["cx", "cy", "cz", "dist", "spread", "count"]
DESC_DIM = len(DESC_COMPONENTS)


def farthest_point_sampling(points: torch.Tensor, k: int) -> torch.Tensor:
    """points: (B, N, 3) -> anchor indices (B, K). Deterministic start (point 0)."""
    B, N, _ = points.shape
    device = points.device
    idx = torch.zeros(B, k, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_arange = torch.arange(B, device=device)
    for i in range(k):
        idx[:, i] = farthest
        centroid = points[batch_arange, farthest].unsqueeze(1)      # (B,1,3)
        d = ((points - centroid) ** 2).sum(-1)                      # (B,N)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return idx


def knn_group(points: torch.Tensor, anchors_xyz: torch.Tensor, p: int) -> torch.Tensor:
    """Group P nearest neighbours of each anchor. -> (B, K, P, 3)."""
    d = torch.cdist(anchors_xyz, points)                            # (B,K,N)
    nn_idx = d.topk(p, largest=False).indices                       # (B,K,P)
    B, K, P = nn_idx.shape
    flat = nn_idx.reshape(B, K * P)
    grouped = torch.gather(points, 1, flat.unsqueeze(-1).expand(-1, -1, 3))
    return grouped.reshape(B, K, P, 3)


def slice_point_cloud(points: torch.Tensor, k: int = 16, p: int = 64):
    """Full offline slicing pipeline.

    Returns:
        slices (B, K, P, 3)  local regions (KNN of each FPS anchor)
        desc   (B, K, 6)     geometry descriptors
        anchors_xyz (B, K, 3)
    """
    B, N, _ = points.shape
    a_idx = farthest_point_sampling(points, k)                      # (B,K)
    anchors_xyz = torch.gather(points, 1, a_idx.unsqueeze(-1).expand(-1, -1, 3))
    slices = knn_group(points, anchors_xyz, p)                      # (B,K,P,3)

    centroid_slice = slices.mean(dim=2)                             # (B,K,3)
    cloud_centroid = points.mean(dim=1, keepdim=True)               # (B,1,3)
    dist = (centroid_slice - cloud_centroid).norm(dim=-1, keepdim=True)         # (B,K,1)
    spread = (slices - centroid_slice.unsqueeze(2)).norm(dim=-1).mean(-1, keepdim=True)

    # Voronoi occupancy: fraction of cloud points nearest to anchor m.
    d_all = torch.cdist(points, anchors_xyz)                        # (B,N,K)
    owner = d_all.argmin(-1)                                        # (B,N)
    count = torch.zeros(B, k, device=points.device, dtype=points.dtype)
    count.scatter_add_(1, owner, torch.ones_like(owner, dtype=points.dtype))
    count = (count / N).unsqueeze(-1)                               # (B,K,1)

    desc = torch.cat([centroid_slice, dist, spread, count], dim=-1) # (B,K,6)
    return slices, desc, anchors_xyz


def mask_descriptor(desc: torch.Tensor, drop: list[str]) -> torch.Tensor:
    """A3 ablation: zero-out named components (keeps dimensionality fixed)."""
    if not drop:
        return desc
    desc = desc.clone()
    for name in drop:
        desc[..., DESC_COMPONENTS.index(name)] = 0.0
    return desc


def normalize_cloud(points: torch.Tensor) -> torch.Tensor:
    """Center at origin and scale to unit sphere (PointNet convention)."""
    points = points - points.mean(dim=-2, keepdim=True)
    scale = points.norm(dim=-1).amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return points / scale.unsqueeze(-1)
