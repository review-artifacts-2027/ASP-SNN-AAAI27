"""
datasets/shapenetpart.py — ShapeNetPart HDF5 dataset for part segmentation.

16 categories, 50 part labels, 14007 train / 2874 test shapes.
Each shape has 2048 points with global part labels in [0, 49].

Batch 3 upgrades:
    B1 — per-point boundary labels bnd_labels[N] computed once from the k-NN
         graph over raw normalized xyz. A point is a boundary if any of its
         k nearest neighbours has a different part label. Rigid transforms
         preserve boundary structure, so this is aug-invariant.
    B4 — fine-scale slicing (M_fine, K_fine) alongside coarse. Uses the same
         FPS slicer as coarse. Both scales come from the SAME augmented point
         cloud so their geometries are consistent.
    (Fix) augmentation is now applied to the raw point cloud BEFORE slicing,
         which naturally fixes the aug/geo consistency bug we hit previously
         (geo was computed pre-augmentation, slices post — root cause of
         the ~97/67/37 train/val/test collapse).

Returns per sample (10 items — see __getitem__ docstring for shapes):
    coarse_slices     [M, K, 6]
    coarse_geo        [M, 8]
    pts_xyz           [N, 3]
    coarse_sid_arr    [N]
    fine_slices       [M_f, K_f, 6]
    fine_geo          [M_f, 8]
    fine_sid_arr      [N]
    part_labels       [N]
    bnd_labels        [N]
    cat_id            int
"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset

from .slicing import slice_point_cloud, assign_points_to_slices
from .transforms import augment_points_only


# ── Category and part definitions ──────────────────────────────────────

CATEGORY_NAMES = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
    'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',
    'Rocket', 'Skateboard', 'Table',
]

CATEGORY_TO_PARTS = {
    0:  [0, 1, 2, 3],           # Airplane
    1:  [4, 5],                 # Bag
    2:  [6, 7],                 # Cap
    3:  [8, 9, 10, 11],         # Car
    4:  [12, 13, 14, 15],       # Chair
    5:  [16, 17, 18],           # Earphone
    6:  [19, 20, 21],           # Guitar
    7:  [22, 23],               # Knife
    8:  [24, 25, 26, 27],       # Lamp
    9:  [28, 29],               # Laptop
    10: [30, 31, 32, 33, 34, 35],  # Motorbike
    11: [36, 37],               # Mug
    12: [38, 39, 40],           # Pistol
    13: [41, 42, 43],           # Rocket
    14: [44, 45],               # Skateboard
    15: [46, 47, 48, 49],       # Table
}

NUM_PARTS = 50
NUM_CATEGORIES = 16


# ── Boundary label computation (B1) ────────────────────────────────────

def compute_boundary_labels(pts_xyz: np.ndarray,
                            part_labels: np.ndarray,
                            k: int = 10) -> np.ndarray:
    """
    A point is a boundary point if any of its k nearest neighbours has a
    different part label. Computed on raw normalised xyz. Rigid transforms
    preserve k-NN so this is aug-invariant.

    Args:
        pts_xyz:     [N, 3] float
        part_labels: [N]    int
        k:           number of neighbours to check

    Returns:
        bnd_labels: [N] float32 (0.0 or 1.0)
    """
    N = pts_xyz.shape[0]
    k = min(k, N - 1)

    # Squared pairwise distances via inner products (16MB peak at N=2048).
    sq = (pts_xyz ** 2).sum(axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (pts_xyz @ pts_xyz.T)
    np.fill_diagonal(dist2, np.inf)

    # k-NN indices (unsorted top-k via argpartition — faster than argsort)
    knn_idx = np.argpartition(dist2, k - 1, axis=1)[:, :k]  # [N, k]

    neighbor_labels = part_labels[knn_idx]                  # [N, k]
    same = (neighbor_labels == part_labels[:, None])        # [N, k]
    is_boundary = ~same.all(axis=1)                         # [N]
    return is_boundary.astype(np.float32)


class ShapeNetPartDataset(Dataset):

    def __init__(self, data_dir: str, split: str, cfg=None):
        assert split in ('train', 'test')
        self.split = split
        self.cfg = cfg
        self.n_points = getattr(cfg, 'num_points', 2048)

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")

        pattern = os.path.join(data_dir, f"{split}*.h5")
        h5_files = sorted(glob.glob(pattern))
        if not h5_files:
            raise FileNotFoundError(
                f"No {split}*.h5 files found in {data_dir}. "
                f"Run: python datasets/download.py --shapenet"
            )

        all_pts, all_cat, all_pid = [], [], []
        for path in h5_files:
            with h5py.File(path, 'r') as f:
                all_pts.append(f['data'][:].astype(np.float32))
                all_cat.append(f['label'][:].astype(np.int64))
                all_pid.append(f['pid'][:].astype(np.int64))

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
        """
        Returns 10 items:
            coarse_slices  [M, K, 6]     coarse per-slice point clouds
            coarse_geo     [M, 8]        coarse geometry descriptors
            pts_xyz        [N, 3]        (augmented) normalised xyz
            coarse_sid_arr [N]           coarse slice id per point
            fine_slices    [M_f, K_f, 6] fine per-slice point clouds
            fine_geo       [M_f, 8]      fine geometry descriptors
            fine_sid_arr   [N]           fine slice id per point
            part_labels    [N]           ground-truth part ids
            bnd_labels     [N]           boundary indicator (float 0/1)
            cat_id         int           category index
        """
        cat_id = int(self.cats[idx])
        part_labels = self.pids[idx][:self.n_points].astype(np.int64)

        raw_xyz = self.pts[idx][:self.n_points, :3]
        pts_n   = self._normalise(raw_xyz)                    # [N, 3]

        # ── B1: compute boundary labels once, before augmentation ─────
        k_bnd = getattr(self.cfg, 'bnd_k', 10)
        bnd_labels = compute_boundary_labels(pts_n, part_labels, k=k_bnd)

        # ── Augmentation happens on raw points (not on slices) ────────
        # This is a CLEAN pipeline: single transform applied to all points,
        # then both coarse and fine slicing operate on the same augmented
        # data. Naturally avoids the aug/geo consistency bug where geo was
        # computed from pre-augmentation slices.
        if self.split == 'train' and self.cfg is not None:
            pts_n = augment_points_only(pts_n, self.cfg)

        # Pad to 6 channels for the encoder (xyz + zero normals)
        pts6 = np.concatenate(
            [pts_n, np.zeros((len(pts_n), 3), dtype=np.float32)], axis=1
        )

        # ── Coarse slicing (M=16, K=128 by default) ───────────────────
        M      = getattr(self.cfg, 'num_slices', 16)
        K      = getattr(self.cfg, 'points_per_slice', 128)
        # Deterministic FPS at test time for reproducible metrics
        fps_seed = idx if self.split == 'test' else None
        coarse_slices, coarse_geo, coarse_anchor_xyz = slice_point_cloud(
            pts6, M, K, seed=fps_seed,
        )
        coarse_sid_arr = assign_points_to_slices(pts_n, coarse_anchor_xyz)

        # ── B4: fine slicing (M_fine=64, K_fine=32 by default) ────────
        M_f = getattr(self.cfg, 'num_slices_fine', 64)
        K_f = getattr(self.cfg, 'points_per_slice_fine', 32)
        # Use a different seed offset so coarse and fine FPS don't collide
        fps_seed_fine = (idx + 10_000) if self.split == 'test' else None
        fine_slices, fine_geo, fine_anchor_xyz = slice_point_cloud(
            pts6, M_f, K_f, seed=fps_seed_fine,
        )
        fine_sid_arr = assign_points_to_slices(pts_n, fine_anchor_xyz)

        return (
            coarse_slices.astype(np.float32),   # [M, K, 6]
            coarse_geo.astype(np.float32),      # [M, 8]
            pts_n.astype(np.float32),           # [N, 3]
            coarse_sid_arr.astype(np.int64),    # [N]
            fine_slices.astype(np.float32),     # [M_f, K_f, 6]
            fine_geo.astype(np.float32),        # [M_f, 8]
            fine_sid_arr.astype(np.int64),      # [N]
            part_labels,                        # [N]
            bnd_labels,                         # [N] float32
            cat_id,                             # int
        )