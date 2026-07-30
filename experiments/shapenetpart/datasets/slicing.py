import numpy as np


def fps(points: np.ndarray, npoint: int, seed: int = None) -> np.ndarray:
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

    xyz = points[:, :3]
    N = points.shape[0]
    slices = []
    for idx in anchors:
        dist = np.linalg.norm(xyz - xyz[idx], axis=1)
        nn_idx = np.argsort(dist)[:k]

        if len(nn_idx) < k:
            pad = np.repeat(nn_idx[-1:], k - len(nn_idx))
            nn_idx = np.concatenate([nn_idx, pad])
        slices.append(points[nn_idx])
    return np.stack(slices)


def compute_geo(slice_pts: np.ndarray) -> np.ndarray:
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
    import torch
    xyz = slices[..., :3]
    centroid = xyz.mean(dim=2)
    variance = xyz.var(dim=2, unbiased=False)
    dists = torch.linalg.norm(
        xyz - centroid.unsqueeze(2), dim=-1
    )
    max_dist = dists.max(dim=-1).values.unsqueeze(-1)
    dist_to_origin = torch.linalg.norm(
        centroid, dim=-1, keepdim=True
    )
    return torch.cat([centroid, variance, max_dist, dist_to_origin], dim=-1)


def slice_point_cloud(points: np.ndarray, num_slices: int = 16,
                      points_per_slice: int = 64, seed: int = None):

    anchors = fps(points, num_slices, seed=seed)
    slices = build_slices(points, anchors, points_per_slice)
    geo = np.stack([compute_geo(s) for s in slices])
    anchor_xyz = points[anchors, :3]
    return slices, geo, anchor_xyz


def assign_points_to_slices(pts_xyz: np.ndarray,
                            anchor_xyz: np.ndarray) -> np.ndarray:

    dists = np.linalg.norm(
        pts_xyz[:, None, :] - anchor_xyz[None, :, :], axis=2
    )
    return dists.argmin(axis=1).astype(np.int32)
