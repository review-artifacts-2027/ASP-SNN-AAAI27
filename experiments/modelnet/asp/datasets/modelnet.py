"""ModelNet10/40 loader (modelnet40_normal_resampled format, PointNet++ standard).

Download (once):
    https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
Extract to <root>/modelnet40_normal_resampled/.  ModelNet10 uses the
modelnet10_shape_names.txt / *_train.txt / *_test.txt splits in the same folder.

Each shape file: N rows "x,y,z,nx,ny,nz"; we read xyz, sample n_points, and
apply the standard normalization + augmentation (random z-rotation, scale,
jitter) at train time.
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from ..geometry import normalize_cloud, slice_point_cloud


class ModelNetDataset(Dataset):
    def __init__(self, root: str, variant: int = 40, split: str = "train",
                 n_points: int = 1024, k_slices: int = 16,
                 points_per_slice: int = 128, augment: bool = None,
                 corruption=None, severity: int = 0, cache: bool = True):
        base = os.path.join(root, "modelnet40_normal_resampled")
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"{base} not found. Download modelnet40_normal_resampled.zip "
                "(see module docstring) and extract into the data root.")
        names_file = os.path.join(base, f"modelnet{variant}_shape_names.txt")
        split_file = os.path.join(base, f"modelnet{variant}_{split}.txt")
        self.classes = [l.strip() for l in open(names_file)]
        ids = [l.strip() for l in open(split_file)]
        self.files, self.labels = [], []
        for sid in ids:
            cls = "_".join(sid.split("_")[:-1])
            self.files.append(os.path.join(base, cls, sid + ".txt"))
            self.labels.append(self.classes.index(cls))
        self.n_points, self.k, self.p = n_points, k_slices, points_per_slice
        self.augment = (split == "train") if augment is None else augment
        self.corruption, self.severity = corruption, severity
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.files)

    def _load(self, i):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        import numpy as np
        arr = np.loadtxt(self.files[i], delimiter=",", dtype="float32")[:, :3]
        pts = torch.from_numpy(arr)
        if self._cache is not None:
            self._cache[i] = pts
        return pts

    def __getitem__(self, i):
        pts = self._load(i)
        g = torch.Generator().manual_seed(i * 6151 + 29)
        idx = torch.randperm(pts.shape[0], generator=g)[: self.n_points]
        pts = pts[idx]
        pts = normalize_cloud(pts.unsqueeze(0)).squeeze(0)
        if self.augment:
            import math
            th = torch.rand(1, generator=g).item() * 2 * math.pi
            c, s = math.cos(th), math.sin(th)
            R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            pts = (pts @ R.t()) * (0.8 + 0.4 * torch.rand(1, generator=g).item())
            pts = pts + torch.randn(pts.shape, generator=g) * 0.01
        if self.corruption is not None and self.severity > 0:
            pts = self.corruption(pts, self.severity, g)
            pts = normalize_cloud(pts.unsqueeze(0)).squeeze(0)
        slices, desc, anchors = slice_point_cloud(pts.unsqueeze(0), self.k, self.p)
        return slices.squeeze(0), desc.squeeze(0), anchors.squeeze(0), self.labels[i]
