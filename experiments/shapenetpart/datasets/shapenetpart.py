import os
import glob
import numpy as np
from torch.utils.data import Dataset

from .slicing import slice_point_cloud, assign_points_to_slices
from .transforms import augment_points_only

CATEGORY_NAMES = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
    'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',
    'Rocket', 'Skateboard', 'Table',
]

CATEGORY_TO_PARTS = {
    0:  [0, 1, 2, 3],
    1:  [4, 5],
    2:  [6, 7],
    3:  [8, 9, 10, 11],
    4:  [12, 13, 14, 15],
    5:  [16, 17, 18],
    6:  [19, 20, 21],
    7:  [22, 23],
    8:  [24, 25, 26, 27],
    9:  [28, 29],
    10: [30, 31, 32, 33, 34, 35],
    11: [36, 37],
    12: [38, 39, 40],
    13: [41, 42, 43],
    14: [44, 45],
    15: [46, 47, 48, 49],
}

NUM_PARTS = 50
NUM_CATEGORIES = 16


def compute_boundary_labels(pts_xyz: np.ndarray,
                            part_labels: np.ndarray,
                            k: int = 10) -> np.ndarray:

    N = pts_xyz.shape[0]
    k = min(k, N - 1)

    sq = (pts_xyz ** 2).sum(axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (pts_xyz @ pts_xyz.T)
    np.fill_diagonal(dist2, np.inf)

    knn_idx = np.argpartition(dist2, k - 1, axis=1)[:, :k]

    neighbor_labels = part_labels[knn_idx]
    same = (neighbor_labels == part_labels[:, None])
    is_boundary = ~same.all(axis=1)
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
        pts = pts - pts.mean(axis=0)
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 0:
            pts = pts / scale
        return pts.astype(np.float32)

    def __getitem__(self, idx):
        cat_id = int(self.cats[idx])
        part_labels = self.pids[idx][:self.n_points].astype(np.int64)

        raw_xyz = self.pts[idx][:self.n_points, :3]
        pts_n   = self._normalise(raw_xyz)

        k_bnd = getattr(self.cfg, 'bnd_k', 10)
        bnd_labels = compute_boundary_labels(pts_n, part_labels, k=k_bnd)

        if self.split == 'train' and self.cfg is not None:
            pts_n = augment_points_only(pts_n, self.cfg)

        pts6 = np.concatenate(
            [pts_n, np.zeros((len(pts_n), 3), dtype=np.float32)], axis=1
        )

        M      = getattr(self.cfg, 'num_slices', 16)
        K      = getattr(self.cfg, 'points_per_slice', 128)

        fps_seed = idx if self.split == 'test' else None
        coarse_slices, coarse_geo, coarse_anchor_xyz = slice_point_cloud(
            pts6, M, K, seed=fps_seed,
        )
        coarse_sid_arr = assign_points_to_slices(pts_n, coarse_anchor_xyz)

        M_f = getattr(self.cfg, 'num_slices_fine', 64)
        K_f = getattr(self.cfg, 'points_per_slice_fine', 32)

        fps_seed_fine = (idx + 10_000) if self.split == 'test' else None
        fine_slices, fine_geo, fine_anchor_xyz = slice_point_cloud(
            pts6, M_f, K_f, seed=fps_seed_fine,
        )
        fine_sid_arr = assign_points_to_slices(pts_n, fine_anchor_xyz)

        return (
            coarse_slices.astype(np.float32),
            coarse_geo.astype(np.float32),
            pts_n.astype(np.float32),
            coarse_sid_arr.astype(np.int64),
            fine_slices.astype(np.float32),
            fine_geo.astype(np.float32),
            fine_sid_arr.astype(np.int64),
            part_labels,
            bnd_labels,
            cat_id,
        )
