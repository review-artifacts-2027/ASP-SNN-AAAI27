"""Synthetic geometric-primitive point clouds (8 classes).

Purpose: CPU-runnable verification of every theoretical claim before GPU-scale
runs, and controlled stress tests (occlusion/density) where ground-truth region
informativeness is known by construction. NOT a benchmark substitute.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset

from ..geometry import normalize_cloud, slice_point_cloud

CLASSES = ["sphere", "cube", "cylinder", "cone", "torus", "pyramid",
           "cross", "helix"]


def _unit(rng, n):
    return torch.rand(n, generator=rng)


def make_cloud(cls: int, n: int, rng: torch.Generator) -> torch.Tensor:
    u, v, w = _unit(rng, n), _unit(rng, n), _unit(rng, n)
    if cls == 0:      # sphere (surface)
        th, ph = 2 * math.pi * u, torch.acos(2 * v - 1)
        p = torch.stack([th.cos() * ph.sin(), th.sin() * ph.sin(), ph.cos()], -1)
    elif cls == 1:    # cube surface
        face = (u * 6).long().clamp(max=5)
        a, b = 2 * v - 1, 2 * w - 1
        p = torch.zeros(n, 3)
        for f in range(6):
            m = face == f
            axis, sign = f // 2, 1.0 if f % 2 == 0 else -1.0
            other = [i for i in range(3) if i != axis]
            p[m, axis] = sign
            p[m, other[0]] = a[m]
            p[m, other[1]] = b[m]
    elif cls == 2:    # cylinder
        th, z = 2 * math.pi * u, 2 * v - 1
        p = torch.stack([th.cos(), th.sin(), z], -1)
    elif cls == 3:    # cone
        th, h = 2 * math.pi * u, v
        r = 1 - h
        p = torch.stack([r * th.cos(), r * th.sin(), 2 * h - 1], -1)
    elif cls == 4:    # torus
        th, ph = 2 * math.pi * u, 2 * math.pi * v
        R, r = 1.0, 0.35
        p = torch.stack([(R + r * ph.cos()) * th.cos(),
                         (R + r * ph.cos()) * th.sin(), r * ph.sin()], -1)
    elif cls == 5:    # pyramid (4 triangular faces + base)
        base = torch.stack([2 * v - 1, 2 * w - 1, -torch.ones(n)], -1)
        apexward = u.unsqueeze(-1)
        p = base * (1 - apexward) + torch.tensor([0.0, 0.0, 1.0]) * apexward
    elif cls == 6:    # 3D cross of 3 orthogonal bars
        axis = (u * 3).long().clamp(max=2)
        p = (torch.stack([2 * v - 1, 2 * w - 1, _unit(rng, n) * 0.3 - 0.15], -1))
        out = torch.zeros(n, 3)
        for a in range(3):
            m = axis == a
            long_axis = p[m, 0]
            thick = p[m, 1:] * 0.15
            cols = [a] + [i for i in range(3) if i != a]
            out[m.nonzero().squeeze(-1), cols[0]] = long_axis
            out[m.nonzero().squeeze(-1), cols[1]] = thick[:, 0]
            out[m.nonzero().squeeze(-1), cols[2]] = thick[:, 1]
        p = out
    else:             # helix
        t = 4 * math.pi * u
        p = torch.stack([0.8 * t.cos(), 0.8 * t.sin(), (t / (2 * math.pi)) - 1], -1)
        p = p + (torch.stack([v, w, _unit(rng, n)], -1) - 0.5) * 0.12
    return p


def augment(p: torch.Tensor, rng: torch.Generator, jitter: float = 0.01) -> torch.Tensor:
    th = torch.rand(1, generator=rng).item() * 2 * math.pi   # z-rotation
    c, s = math.cos(th), math.sin(th)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    scale = 0.8 + 0.4 * torch.rand(1, generator=rng).item()
    p = (p @ R.t()) * scale
    p = p + torch.randn(p.shape, generator=rng) * jitter
    return p


class SyntheticPointDataset(Dataset):
    def __init__(self, n_per_class: int = 100, n_points: int = 256,
                 k_slices: int = 16, points_per_slice: int = 32,
                 seed: int = 0, corruption=None, severity: int = 0):
        self.rng = torch.Generator().manual_seed(seed)
        self.k, self.p = k_slices, points_per_slice
        self.corruption, self.severity = corruption, severity
        self.items = []
        for c in range(len(CLASSES)):
            for _ in range(n_per_class):
                self.items.append((make_cloud(c, n_points, self.rng), c))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cloud, label = self.items[i]
        g = torch.Generator().manual_seed(i * 7919 + 13)
        pts = normalize_cloud(augment(cloud.clone(), g).unsqueeze(0)).squeeze(0)
        if self.corruption is not None and self.severity > 0:
            pts = self.corruption(pts, self.severity, g)
            pts = normalize_cloud(pts.unsqueeze(0)).squeeze(0)
        slices, desc, anchors = slice_point_cloud(pts.unsqueeze(0), self.k, self.p)
        return slices.squeeze(0), desc.squeeze(0), anchors.squeeze(0), label
