import numpy as np
import torch
import torch.nn as nn


def compute_room_summary_np(room_points: np.ndarray,
                            n_anchors: int = 64,
                            k_neighbors: int = 32,
                            use_rgb: bool = True,
                            use_height: bool = True,
                            room_z_bounds: tuple = None,
                            seed: int = 0) -> np.ndarray:

    xyz = room_points[:, :3]
    N = len(xyz)

    rng = np.random.default_rng(seed)
    anchors = np.zeros(n_anchors, dtype=np.int64)
    distance = np.full(N, 1e10, dtype=np.float64)
    farthest = int(rng.integers(0, N))
    for i in range(n_anchors):
        anchors[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        distance = np.minimum(distance, dist)
        farthest = int(np.argmax(distance))

    per_anchor_feats = []
    for a in anchors:
        d = np.linalg.norm(xyz - xyz[a], axis=1)
        idx = np.argpartition(d, min(k_neighbors, N - 1))[:k_neighbors]
        patch = room_points[idx]

        pxyz = patch[:, :3]
        centroid = pxyz.mean(axis=0)
        spread = pxyz.std(axis=0)

        feats = [centroid, spread]

        if use_rgb:
            rgb = patch[:, 3:6] / 255.0
            feats.append(rgb.mean(axis=0))
            feats.append(rgb.std(axis=0))

        if use_height and room_z_bounds is not None:
            z_min, z_max = room_z_bounds
            z_norm = np.clip(
                (patch[:, 2] - z_min) / max(z_max - z_min, 1e-6),
                0.0, 1.0,
            )
            feats.append(np.array([z_norm.mean(), z_norm.std()],
                                  dtype=np.float32))

        per_anchor_feats.append(np.concatenate(feats).astype(np.float32))

    stacked = np.stack(per_anchor_feats)
    pooled = np.concatenate([
        stacked.mean(axis=0),
        stacked.max(axis=0),
    ])
    return pooled.astype(np.float32)


def summary_dim(use_rgb: bool = True, use_height: bool = True) -> int:
    per_anchor = 3 + 3
    if use_rgb:
        per_anchor += 6
    if use_height:
        per_anchor += 2
    return per_anchor * 2


class RoomPriorProjection(nn.Module):
    def __init__(self, room_summary_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(room_summary_dim, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim, bias=False),
        )

        nn.init.zeros_(self.proj[-1].weight)

    def forward(self, room_summary: torch.Tensor) -> torch.Tensor:
        return self.proj(room_summary)
