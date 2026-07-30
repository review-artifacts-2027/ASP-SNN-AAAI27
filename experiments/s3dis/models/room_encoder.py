"""
models/room_encoder.py — Room-level ASP prior (E1).

Purpose:
    The ASP loop currently resets its belief state to zeros at the start
    of every block, wasting the "belief" work between adjacent blocks
    from the same room. This is especially wasteful because room-level
    identity (office vs. hallway vs. conference room vs. storage) is
    highly informative for the per-point class distribution — an office
    has different clutter statistics than a hallway.

    E1 addresses this by computing a single 512-dim summary vector per
    ROOM once (precomputed at dataset-init time using the FROZEN encoder
    from the main model, or a randomly-initialized encoder followed by
    a learnable projection). This summary is then projected into the
    initial belief state u_0 of the LIF head, so each block's ASP loop
    starts with room-level prior knowledge rather than zeros.

Design decisions:
    - PRECOMPUTED: room summaries are computed once per room at dataset
      init time, not per-batch. Frozen features are fast, deterministic,
      and require no gradients through the room encoder.
    - INIT-ONLY (not per-timestep fusion): the room prior is injected
      only into u_0 (the initial belief state). This preserves the ASP
      thesis — the LIF membrane state "acts as a compressed summary of
      everything it has observed up to this point" — by seeding that
      summary with a room-level prior. Per-timestep fusion would blur
      the "membrane state as belief" story and cost 6x more compute.
    - LEARNABLE PROJECTION: the projection from room summary → belief
      init is learnable (a Linear layer inside ASPSegmentor), so training
      can decide how much room-level bias to apply. Zero-initialized so
      early training starts identical to the no-prior baseline and only
      grows the effect when it helps.

    Architecture rationale:
        FROZEN(EdgeConv encoder) → [K_room, 512] per-anchor tokens
                                → mean+max pool → [1024] room descriptor
                                → LEARNABLE Linear(1024→512) belief init
"""

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
    """
    Precompute a fixed-length descriptor for a whole room.

    Uses FPS over the whole room to pick K_room anchors, KNN to form
    patches, then computes handcrafted geometric + color statistics per
    anchor and pools across anchors. This is a FROZEN featurization — no
    neural network needed for the summary itself; the learnable part is
    the projection into the belief state, which lives in ASPSegmentor.

    We use handcrafted features (rather than the neural encoder) because:
      1. They can be computed once at dataset init in numpy without a GPU.
      2. They're deterministic and don't require the model to exist yet.
      3. Room-level statistics (mean color, height distribution, spatial
         extent) are already strong signals for room type; a learnable
         projection on top gives the model room to disambiguate further.

    Args:
        room_points:   [N, 7]  x,y,z,r,g,b,label
        n_anchors:     K_room, number of FPS anchors over the whole room
        k_neighbors:   K per anchor
        use_rgb:       include color statistics
        use_height:    include normalized height statistics
        room_z_bounds: (z_min, z_max) for per-room height normalization
        seed:          FPS starting-point seed

    Returns:
        summary: [D_room] float32
                 D_room = n_anchors * feats_per_anchor
                 feats_per_anchor depends on flags but is deterministic.
    """
    xyz = room_points[:, :3]
    N = len(xyz)

    # FPS to n_anchors anchors
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

    # For each anchor: KNN patch → per-patch statistics
    per_anchor_feats = []
    for a in anchors:
        d = np.linalg.norm(xyz - xyz[a], axis=1)
        idx = np.argpartition(d, min(k_neighbors, N - 1))[:k_neighbors]
        patch = room_points[idx]

        pxyz = patch[:, :3]
        centroid = pxyz.mean(axis=0)
        spread = pxyz.std(axis=0)

        feats = [centroid, spread]  # 3 + 3

        if use_rgb:
            rgb = patch[:, 3:6] / 255.0
            feats.append(rgb.mean(axis=0))         # 3
            feats.append(rgb.std(axis=0))          # 3

        if use_height and room_z_bounds is not None:
            z_min, z_max = room_z_bounds
            z_norm = np.clip(
                (patch[:, 2] - z_min) / max(z_max - z_min, 1e-6),
                0.0, 1.0,
            )
            feats.append(np.array([z_norm.mean(), z_norm.std()],
                                  dtype=np.float32))  # 2

        per_anchor_feats.append(np.concatenate(feats).astype(np.float32))

    # Pool across anchors: mean + max, then flatten
    stacked = np.stack(per_anchor_feats)             # [n_anchors, D_anchor]
    pooled = np.concatenate([
        stacked.mean(axis=0),
        stacked.max(axis=0),
    ])
    return pooled.astype(np.float32)


def summary_dim(use_rgb: bool = True, use_height: bool = True) -> int:
    """Return the dimensionality of the pooled room summary."""
    per_anchor = 3 + 3
    if use_rgb:
        per_anchor += 6
    if use_height:
        per_anchor += 2
    return per_anchor * 2   # mean + max pooling


class RoomPriorProjection(nn.Module):
    """
    Learnable projection from precomputed room summary into the LIF
    initial belief state u_0.

    Zero-initialized so early training starts identical to no-prior
    baseline. Growth of the projection weights is driven by the CE +
    Lovász loss — if the room prior helps mIoU, weights grow; if not,
    they stay near zero.

    Applied to layer 0 only (the deepest layer receiving direct external
    input in the LIF stack). Subsequent layers pick up the room signal
    through the residual chain.

    Reference for zero-init "warm start" of an auxiliary pathway:
        Bachlechner et al. "ReZero is All You Need." UAI 2021.
    """

    def __init__(self, room_summary_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(room_summary_dim, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, hidden_dim, bias=False),
        )
        # Zero-init the final linear so u_0 starts at 0 as in baseline
        nn.init.zeros_(self.proj[-1].weight)

    def forward(self, room_summary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            room_summary: [B, D_room]  precomputed frozen summaries

        Returns:
            u_init: [B, hidden_dim]  additive contribution to layer-0 u_0
        """
        return self.proj(room_summary)