"""
datasets/scanobjectnn.py — ScanObjectNN dataset for classification.

PB_T50_RS variant (hardest): 15 classes, real-world scanned objects with
background clutter, occlusion, and sensor noise.

The H5 files contain:
    data:  [N_shapes, 2048, 3]  xyz coordinates
    label: [N_shapes]           class labels (0-14)

Returns per sample:
    slices  [M, K, 6]   (xyz padded with zero normals)
    geo     [M, 8]      geometry descriptors
    label   int          class index (0-14)
"""

import os
import numpy as np
from torch.utils.data import Dataset

from .slicing import slice_point_cloud, compute_geo
from .transforms import augment_slices


# File names for PB_T50_RS variant
_TRAIN_FILE = "training_objectdataset_augmentedrot_scale75.h5"
_TEST_FILE = "test_objectdataset_augmentedrot_scale75.h5"


class ScanObjectNNDataset(Dataset):

    def __init__(self, data_dir: str, split: str, cfg=None,
                 force_no_aug: bool = False):
        """
        Args:
            data_dir: path to ScanObjectNN/main_split/
            split:    'train' or 'test'
            cfg:      config object
            force_no_aug: if True, disable augmentation even when split='train'.
                          Used for the validation subset which should match
                          the test distribution, not the augmented train one.
        """
        assert split in ('train', 'test')
        self.split = split
        self.cfg = cfg
        self.force_no_aug = force_no_aug
        self.n_points = getattr(cfg, 'num_points', 2048)

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")

        fname = _TRAIN_FILE if split == 'train' else _TEST_FILE
        h5_path = os.path.join(data_dir, fname)

        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"ScanObjectNN file not found: {h5_path}\n"
                f"Download from https://hkust-vgd.github.io/scanobjectnn/ "
                f"and place H5 files in {data_dir}/"
            )

        with h5py.File(h5_path, 'r') as f:
            self.pts = f['data'][:].astype(np.float32)    # [N, 2048, 3]
            self.labels = f['label'][:].astype(np.int64)  # [N]

        # Flatten label if needed
        if self.labels.ndim == 2:
            self.labels = self.labels.squeeze(-1)

        print(f"[ScanObjectNN] '{split}': {len(self.pts)} shapes, "
              f"15 classes (PB_T50_RS)")

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
        label = int(self.labels[idx])

        # Sample points if we have more than needed.
        # P2 FIX: at test time use a per-idx deterministic RNG so the same
        # sample always produces the same point subset across eval passes.
        raw = self.pts[idx]  # [2048, 3]
        if self.split == 'test':
            rng = np.random.default_rng(idx)
        else:
            rng = np.random

        if len(raw) > self.n_points:
            choice = rng.choice(len(raw), self.n_points, replace=False)
            raw = raw[choice]
        elif len(raw) < self.n_points:
            choice = rng.choice(len(raw), self.n_points, replace=True)
            raw = raw[choice]

        pts_n = self._normalise(raw)

        # Pad to 6 channels (xyz + zero normals)
        pts6 = np.concatenate(
            [pts_n, np.zeros((len(pts_n), 3), dtype=np.float32)], axis=1
        )

        # Slice
        M = getattr(self.cfg, 'num_slices', 16)
        K = getattr(self.cfg, 'points_per_slice', 128)
        fps_seed = idx if self.split == 'test' else None
        slices, geo, _ = slice_point_cloud(pts6, M, K, seed=fps_seed)

        # Augment (training only, unless force_no_aug)
        if self.split == 'train' and self.cfg is not None and not self.force_no_aug:
            slices = augment_slices(slices, self.cfg)
            # P0 FIX: Recompute geo from augmented slices so positional encoding
            # and SSP see the same geometry as the encoder.
            geo = np.stack([compute_geo(s) for s in slices])

        return (
            slices.astype(np.float32),  # [M, K, 6]
            geo.astype(np.float32),     # [M, 8]
            label,                      # int
        )
