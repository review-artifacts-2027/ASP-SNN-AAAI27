"""
datasets/shapenetpart.py — ShapeNetPart HDF5 dataset for part segmentation.

16 categories, 50 part labels, 14007 train / 2874 test shapes.
Each shape has 2048 points with global part labels in [0, 49].

Returns per sample:
    slices       [M, K, 6]   sliced point cloud (xyz + zero normals)
    geo          [M, 8]      geometry descriptors
    pts_xyz      [N, 3]      normalised xyz for PerPointBranch
    sid_arr      [N]          slice assignment per point
    part_labels  [N]          ground truth part ids (0-49)
    cat_id       int          category index (0-15)
"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset

from .slicing import slice_point_cloud, assign_points_to_slices, compute_geo
from .transforms import augment_seg


# ── Category and part definitions ──────────────────────────────────────

CATEGORY_NAMES = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
    'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',
    'Rocket', 'Skateboard', 'Table',
]

# Part label ranges per category (global labels, 0-indexed)
CATEGORY_TO_PARTS = {
    0:  [0, 1, 2, 3],          # Airplane
    1:  [4, 5],                 # Bag
    2:  [6, 7],                 # Cap
    3:  [8, 9, 10, 11],        # Car
    4:  [12, 13, 14, 15],      # Chair
    5:  [16, 17, 18],          # Earphone
    6:  [19, 20, 21],          # Guitar
    7:  [22, 23],              # Knife
    8:  [24, 25, 26, 27],      # Lamp
    9:  [28, 29],              # Laptop
    10: [30, 31, 32, 33, 34, 35],  # Motorbike
    11: [36, 37],              # Mug
    12: [38, 39, 40],          # Pistol
    13: [41, 42, 43],          # Rocket
    14: [44, 45, 46],          # Skateboard
    15: [47, 48, 49],          # Table
}

NUM_PARTS = 50
NUM_CATEGORIES = 16


class ShapeNetPartDataset(Dataset):

    def __init__(self, data_dir: str, split: str, cfg=None):
        """
        Args:
            data_dir: path to shapenet_part_seg_hdf5_data/
            split:    'train' or 'test'
            cfg:      config object with slicing/aug parameters
        """
        assert split in ('train', 'test')
        self.split = split
        self.cfg = cfg
        self.n_points = getattr(cfg, 'num_points', 2048)

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")

        # Discover h5 files dynamically (train0.h5 ... trainN.h5)
        pattern = os.path.join(data_dir, f"{split}*.h5")
        h5_files = sorted(glob.glob(pattern))
        if not h5_files:
            raise FileNotFoundError(
                f"No {split}*.h5 files found in {data_dir}. "
                f"Run: python datasets/download.py --shapenet"
            )

        # Optional RAM cap applied during loading to avoid OOM on low-memory hosts
        max_shapes = getattr(cfg, 'max_shapes', 0)

        all_pts, all_cat, all_pid = [], [], []
        n_loaded = 0
        for path in h5_files:
            with h5py.File(path, 'r') as f:
                n = f['data'].shape[0]
                if max_shapes and n_loaded + n > max_shapes:
                    n = max_shapes - n_loaded
                all_pts.append(f['data'][:n].astype(np.float32))
                all_cat.append(f['label'][:n].astype(np.int64))
                all_pid.append(f['pid'][:n].astype(np.int32))
                n_loaded += n
            if max_shapes and n_loaded >= max_shapes:
                break

        self.pts = np.concatenate(all_pts, axis=0)
        cats = np.concatenate(all_cat, axis=0)
        self.cats = cats.squeeze(-1) if cats.ndim == 2 else cats
        self.pids = np.concatenate(all_pid, axis=0)

        print(f"[ShapeNetPart] '{split}': {len(self.pts)} shapes, "
              f"{NUM_CATEGORIES} categories, {NUM_PARTS} parts")

    def __len__(self):
        return len(self.pts)

    def _normalise(self, pts):
        """Centre and scale to unit sphere."""
        pts = pts - pts.mean(axis=0)
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 0:
            pts = pts / scale
        return pts.astype(np.float32)

    def __getitem__(self, idx):
        cat_id = int(self.cats[idx])
        part_labels = self.pids[idx][:self.n_points].astype(np.int64)

        # Normalise xyz
        raw_xyz = self.pts[idx][:self.n_points, :3]
        pts_n = self._normalise(raw_xyz)

        # Pad to 6 channels (xyz + zero normals)
        pts6 = np.concatenate(
            [pts_n, np.zeros((len(pts_n), 3), dtype=np.float32)], axis=1
        )

        # Slice
        M = getattr(self.cfg, 'num_slices', 16)
        K = getattr(self.cfg, 'points_per_slice', 128)
        # Deterministic FPS at test time for reproducible metrics
        fps_seed = idx if self.split == 'test' else None
        slices, geo, anchor_xyz = slice_point_cloud(pts6, M, K, seed=fps_seed)

        # Assign each point to nearest slice
        sid_arr = assign_points_to_slices(pts_n, anchor_xyz)

        # Augment (training only) — shared transform on slices + pts
        if self.split == 'train' and self.cfg is not None:
            slices, pts_n = augment_seg(slices, pts_n, self.cfg)
            # P0 FIX: Recompute geo from augmented slices so positional encoding
            # and SSP see the same geometry as the encoder.
            # Without this, train and test see different (geo, points) distributions.
            geo = np.stack([compute_geo(s) for s in slices])

        return (
            slices.astype(np.float32),        # [M, K, 6]
            geo.astype(np.float32),           # [M, 8]
            pts_n.astype(np.float32),         # [N, 3]
            sid_arr.astype(np.int64),         # [N]
            part_labels,                      # [N]
            cat_id,                           # int
        )
