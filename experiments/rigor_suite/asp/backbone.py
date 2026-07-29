"""Spiking region encoders ("Local KNN Backbone").

Both encoders map a region to a feature e_m in R^D with spiking activations,
and are applied to ALL K regions in parallel during training (paper Sec. 4).

PointSliceEncoder : (B, K, P, 3) local KNN slices  -> (B, K, D)
PatchSliceEncoder : (B, K, F)    flattened patches -> (B, K, D)   (CIFAR adapter)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .lif import spike_fn


class PointSliceEncoder(nn.Module):
    """Per-point spiking MLP (3->h->D) + max-pool over the slice (PointNet-lite)."""

    def __init__(self, d_model: int = 128, hidden: int = 64, sg_slope: float = 4.0):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.sg_slope = sg_slope
        self.hidden, self.d_model = hidden, d_model
        self.last_firing_rate = torch.tensor(0.0)

    def forward(self, slices: torch.Tensor, anchors_xyz: torch.Tensor) -> torch.Tensor:
        B, K, P, _ = slices.shape
        rel = slices - anchors_xyz.unsqueeze(2)          # local frame
        x = rel.reshape(B * K * P, 3)
        s1 = spike_fn(self.bn1(self.fc1(x)) - 0.0, self.sg_slope)
        h = self.bn2(self.fc2(s1))
        s2 = spike_fn(h, self.sg_slope)
        self.last_firing_rate = 0.5 * (s1.mean() + s2.mean())
        feat = (s2 * h).reshape(B, K, P, self.d_model)   # spike-gated membrane
        return feat.amax(dim=2)                          # (B, K, D)


class PatchSliceEncoder(nn.Module):
    """Flattened image patch -> spiking MLP -> (B, K, D). Same interface."""

    def __init__(self, patch_dim: int, d_model: int = 128, hidden: int = 128,
                 sg_slope: float = 4.0):
        super().__init__()
        self.fc1 = nn.Linear(patch_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.sg_slope = sg_slope
        self.d_model = d_model
        self.last_firing_rate = torch.tensor(0.0)

    def forward(self, patches: torch.Tensor, anchors_xyz=None) -> torch.Tensor:
        B, K, F_ = patches.shape
        x = patches.reshape(B * K, F_)
        s1 = spike_fn(self.bn1(self.fc1(x)), self.sg_slope)
        h = self.bn2(self.fc2(s1))
        s2 = spike_fn(h, self.sg_slope)
        self.last_firing_rate = 0.5 * (s1.mean() + s2.mean())
        return (s2 * h).reshape(B, K, self.d_model)
