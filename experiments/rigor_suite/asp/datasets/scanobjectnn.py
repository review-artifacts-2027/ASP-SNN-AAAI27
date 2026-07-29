"""ScanObjectNN loader — PB-T50-RS (hardest variant; proposal Sec. 7.1).

Download (registration required): https://hkust-vgd.github.io/scanobjectnn/
Place h5 files at:
    <root>/scanobjectnn/main_split/training_objectdataset_augmentedrot_scale75.h5
    <root>/scanobjectnn/main_split/test_objectdataset_augmentedrot_scale75.h5
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from ..geometry import normalize_cloud, slice_point_cloud


class ScanObjectNNDataset(Dataset):
    NUM_CLASSES = 15

    def __init__(self, root: str, split: str = "train", n_points: int = 1024,
                 k_slices: int = 16, points_per_slice: int = 128,
                 corruption=None, severity: int = 0):
        import h5py
        fname = ("training" if split == "train" else "test") \
            + "_objectdataset_augmentedrot_scale75.h5"
        path = os.path.join(root, "scanobjectnn", "main_split", fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} missing — see module docstring.")
        with h5py.File(path, "r") as f:
            self.data = torch.from_numpy(f["data"][:]).float()
            self.labels = torch.from_numpy(f["label"][:]).long()
        self.n_points, self.k, self.p = n_points, k_slices, points_per_slice
        self.corruption, self.severity = corruption, severity

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i * 4409 + 17)
        pts = self.data[i]
        idx = torch.randperm(pts.shape[0], generator=g)[: self.n_points]
        pts = normalize_cloud(pts[idx].unsqueeze(0)).squeeze(0)
        if self.corruption is not None and self.severity > 0:
            pts = self.corruption(pts, self.severity, g)
            pts = normalize_cloud(pts.unsqueeze(0)).squeeze(0)
        slices, desc, anchors = slice_point_cloud(pts.unsqueeze(0), self.k, self.p)
        return slices.squeeze(0), desc.squeeze(0), anchors.squeeze(0), \
            int(self.labels[i])
