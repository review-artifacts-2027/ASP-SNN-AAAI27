from __future__ import annotations

import torch

DESC_COMPONENTS = ["cx", "cy", "cz", "dist", "spread", "count"]
DESC_DIM = len(DESC_COMPONENTS)


def farthest_point_sampling(points: torch.Tensor, k: int) -> torch.Tensor:
    B, N, _ = points.shape
    device = points.device
    idx = torch.zeros(B, k, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_arange = torch.arange(B, device=device)
    for i in range(k):
        idx[:, i] = farthest
        centroid = points[batch_arange, farthest].unsqueeze(1)
        d = ((points - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return idx


def knn_group(points: torch.Tensor, anchors_xyz: torch.Tensor, p: int) -> torch.Tensor:
    d = torch.cdist(anchors_xyz, points)
    nn_idx = d.topk(p, largest=False).indices
    B, K, P = nn_idx.shape
    flat = nn_idx.reshape(B, K * P)
    grouped = torch.gather(points, 1, flat.unsqueeze(-1).expand(-1, -1, 3))
    return grouped.reshape(B, K, P, 3)


def slice_point_cloud(points: torch.Tensor, k: int = 16, p: int = 64):
    B, N, _ = points.shape
    a_idx = farthest_point_sampling(points, k)
    anchors_xyz = torch.gather(points, 1, a_idx.unsqueeze(-1).expand(-1, -1, 3))
    slices = knn_group(points, anchors_xyz, p)

    centroid_slice = slices.mean(dim=2)
    cloud_centroid = points.mean(dim=1, keepdim=True)
    dist = (centroid_slice - cloud_centroid).norm(dim=-1, keepdim=True)
    spread = (slices - centroid_slice.unsqueeze(2)).norm(dim=-1).mean(-1, keepdim=True)

    d_all = torch.cdist(points, anchors_xyz)
    owner = d_all.argmin(-1)
    count = torch.zeros(B, k, device=points.device, dtype=points.dtype)
    count.scatter_add_(1, owner, torch.ones_like(owner, dtype=points.dtype))
    count = (count / N).unsqueeze(-1)

    desc = torch.cat([centroid_slice, dist, spread, count], dim=-1)
    return slices, desc, anchors_xyz


def mask_descriptor(desc: torch.Tensor, drop: list[str]) -> torch.Tensor:
    if not drop:
        return desc
    desc = desc.clone()
    for name in drop:
        desc[..., DESC_COMPONENTS.index(name)] = 0.0
    return desc


def normalize_cloud(points: torch.Tensor) -> torch.Tensor:
    points = points - points.mean(dim=-2, keepdim=True)
    scale = points.norm(dim=-1).amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return points / scale.unsqueeze(-1)
