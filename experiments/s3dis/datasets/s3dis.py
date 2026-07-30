"""
datasets/s3dis.py — S3DIS dataset for indoor scene segmentation.

Stanford Large-Scale 3D Indoor Spaces: 6 areas, 271 rooms, 13 classes.
Protocol: train on Areas 1,2,3,4,6 — test on Area 5.

Each room is stored as an .npy file with columns:
    [x, y, z, r, g, b, semantic_label]  (RGB in 0-255)

Returns per sample (train mode, 7 items with E1 enabled):
    slices        [M, K, C]
    geo           [M, 8]
    pts_features  [N, F]
    sid_arr       [N]
    sem_labels    [N]
    cat_id        0
    room_summary  [D_room]     precomputed room-level prior (E1)

Returns per sample (eval mode when _return_meta=True, 9 items):
    ... same 7 items ...
    room_idx      int
    orig_indices  [N]

If cfg.use_room_prior is False, room_summary is omitted (6 / 8 items).
"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset

from .slicing import slice_point_cloud, assign_points_to_slices, compute_geo
from .transforms import augment_seg
from models.room_encoder import compute_room_summary_np, summary_dim


CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window',
    'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter',
]
NUM_CLASSES = 13

TRAIN_AREAS = [1, 2, 3, 4, 6]
TEST_AREA = 5


class S3DISDataset(Dataset):
    """S3DIS dataset with block-based sampling + optional room-level prior."""

    def __init__(self, data_dir: str, split: str, cfg=None):
        assert split in ('train', 'test')
        self.split = split
        self.cfg = cfg
        self.n_points = getattr(cfg, 'num_points', 4096)
        self.block_size = getattr(cfg, 'block_size', 1.0)
        self.use_rgb = getattr(cfg, 'use_rgb', True)
        self.use_height = getattr(cfg, 'use_height', True)

        # E1: room-level prior config
        self.use_room_prior = getattr(cfg, 'use_room_prior', False)
        self.room_prior_anchors = getattr(cfg, 'room_prior_anchors', 64)
        self.room_prior_k = getattr(cfg, 'room_prior_k', 32)

        self._return_meta = False

        test_area = getattr(cfg, 'test_area', 5)
        if split == 'train':
            areas = [a for a in [1, 2, 3, 4, 5, 6] if a != test_area]
        else:
            areas = [test_area]

        self.rooms = []
        npy_paths = self._discover_rooms(data_dir, areas)

        if len(npy_paths) == 0:
            raise FileNotFoundError(
                f"No S3DIS .npy room files found for areas {areas} in {data_dir}\n"
                f"Expected either:\n"
                f"  - {data_dir}/Area_N/*.npy   (folder layout)\n"
                f"  - {data_dir}/raw/Area_N_*.npy   (flat layout, OpenPoints)\n"
                f"Run: python datasets/download.py --s3dis"
            )

        for npy_path in npy_paths:
            room_data = np.load(npy_path)
            self.rooms.append(room_data.astype(np.float32))

        # Per-room z bounds
        self.room_z_bounds = []
        for room in self.rooms:
            z = room[:, 2]
            z_min = float(z.min())
            z_max = float(z.max())
            if z_max - z_min < 1e-6:
                z_max = z_min + 1.0
            self.room_z_bounds.append((z_min, z_max))

        # E1: precompute room summaries (once, cached to disk)
        self.room_summaries = None
        if self.use_room_prior:
            self.room_summaries = self._compute_or_load_summaries(
                data_dir, npy_paths, split, test_area,
            )
            print(f"[S3DIS] Room summaries: {self.room_summaries.shape} "
                  f"(K_room={self.room_prior_anchors}, k={self.room_prior_k})")

        self.room_sizes = [len(r) for r in self.rooms]
        self.total_points = sum(self.room_sizes)

        if split == 'train':
            epoch_len = getattr(cfg, 'epoch_len', -1)
            if epoch_len > 0:
                self._len = epoch_len
            else:
                self._len = self.total_points // self.n_points
        else:
            self.test_blocks = self._precompute_test_blocks()
            self._len = len(self.test_blocks)

        print(f"[S3DIS] '{split}': {len(self.rooms)} rooms, "
              f"{self.total_points:,} points, {self._len} samples/epoch")

    def _compute_or_load_summaries(self, data_dir: str, npy_paths: list,
                                   split: str, test_area: int) -> np.ndarray:
        """Compute [num_rooms, D_room] summaries once, cache to disk."""
        cache_name = (
            f"s3dis_room_summaries_area{test_area}_{split}_"
            f"K{self.room_prior_anchors}_k{self.room_prior_k}_"
            f"rgb{int(self.use_rgb)}_h{int(self.use_height)}.npy"
        )
        cache_path = os.path.join(data_dir, cache_name)

        if os.path.exists(cache_path):
            summaries = np.load(cache_path).astype(np.float32)
            expected_D = summary_dim(self.use_rgb, self.use_height) * \
                         self.room_prior_anchors * 2 // (2 * self.room_prior_anchors) * self.room_prior_anchors
            # Sanity check on cached dim
            if summaries.shape == (len(self.rooms),
                                   summary_dim(self.use_rgb, self.use_height)):
                print(f"[S3DIS] Loaded cached room summaries → {cache_path}")
                return summaries
            print(f"[S3DIS] Cache dim mismatch, recomputing summaries.")

        print(f"[S3DIS] Precomputing room summaries "
              f"({len(self.rooms)} rooms, one-time cost)...")
        summaries = []
        for ri, room in enumerate(self.rooms):
            s = compute_room_summary_np(
                room,
                n_anchors=self.room_prior_anchors,
                k_neighbors=self.room_prior_k,
                use_rgb=self.use_rgb,
                use_height=self.use_height,
                room_z_bounds=self.room_z_bounds[ri],
                seed=ri,   # per-room seed for determinism
            )
            summaries.append(s)
        summaries = np.stack(summaries).astype(np.float32)

        try:
            np.save(cache_path, summaries)
            print(f"[S3DIS] Cached room summaries → {cache_path}")
        except Exception as e:
            print(f"[S3DIS] Warning: could not cache summaries ({e})")

        return summaries

    def _precompute_test_blocks(self):
        stride = self.block_size * 0.5
        blocks = []
        for ri, room in enumerate(self.rooms):
            xyz = room[:, :3]
            x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
            x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
            cx = x_min + self.block_size / 2
            while cx < x_max:
                cy = y_min + self.block_size / 2
                while cy < y_max:
                    blocks.append((ri, cx, cy))
                    cy += stride
                cx += stride
        return blocks

    def __len__(self):
        return self._len

    @staticmethod
    def _discover_rooms(data_dir: str, areas: list) -> list:
        found = []
        for area in areas:
            area_dir = os.path.join(data_dir, f"Area_{area}")
            if os.path.isdir(area_dir):
                found.extend(sorted(glob.glob(os.path.join(area_dir, "*.npy"))))
                continue

            patterns = [
                os.path.join(data_dir, "raw", f"Area_{area}_*.npy"),
                os.path.join(data_dir, f"Area_{area}_*.npy"),
            ]
            for pat in patterns:
                files = sorted(glob.glob(pat))
                if files:
                    found.extend(files)
                    break

        return found

    def _sample_block(self, room: np.ndarray,
                      cx: float = None, cy: float = None,
                      return_indices: bool = False):
        xyz = room[:, :3]
        half = self.block_size / 2

        if cx is None:
            x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
            x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
            cx = np.random.uniform(x_min + half, max(x_min + half, x_max - half))
            cy = np.random.uniform(y_min + half, max(y_min + half, y_max - half))

        mask = (
            (xyz[:, 0] >= cx - half) & (xyz[:, 0] < cx + half) &
            (xyz[:, 1] >= cy - half) & (xyz[:, 1] < cy + half)
        )
        orig_idx = np.where(mask)[0]
        block_pts = room[orig_idx]

        if len(block_pts) == 0:
            dists = np.linalg.norm(xyz[:, :2] - np.array([cx, cy]), axis=1)
            orig_idx = np.argsort(dists)[:self.n_points]
            block_pts = room[orig_idx]

        if len(block_pts) >= self.n_points:
            choice = np.random.choice(len(block_pts), self.n_points,
                                      replace=False)
        else:
            choice = np.random.choice(len(block_pts), self.n_points,
                                      replace=True)

        block_out = block_pts[choice]
        if return_indices:
            return block_out, orig_idx[choice]
        return block_out

    def _prepare_features(self, block: np.ndarray, room_idx: int):
        xyz = block[:, :3].copy()
        rgb = block[:, 3:6].copy() / 255.0
        labels = block[:, 6].astype(np.int64)

        z_min, z_max = self.room_z_bounds[room_idx]
        z_vals = block[:, 2]
        height = ((z_vals - z_min) / (z_max - z_min)).astype(np.float32)
        height = np.clip(height, 0.0, 1.0)

        xyz = xyz - xyz.mean(axis=0)

        parts = [xyz]
        if self.use_rgb:
            parts.append(rgb)
        if self.use_height:
            parts.append(height.reshape(-1, 1))
        pts_for_slicing = np.concatenate(parts, axis=1).astype(np.float32)

        pts_features = pts_for_slicing.copy()

        return pts_for_slicing, pts_features, labels

    def __getitem__(self, idx):
        if self.split == 'train':
            room_idx = np.random.randint(0, len(self.rooms))
            block = self._sample_block(self.rooms[room_idx])
            orig_indices = None

            if getattr(self.cfg, 'aug_cutmix', False) and \
               np.random.random() < getattr(self.cfg, 'aug_cutmix_prob', 0.5):
                room_idx2 = np.random.randint(0, len(self.rooms))
                block2 = self._sample_block(self.rooms[room_idx2])

                xyz = block[:, :3]
                x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
                x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

                cx = np.random.uniform(x_min, x_max)
                cy = np.random.uniform(y_min, y_max)
                w = np.random.uniform(0.2, 0.5) * (x_max - x_min)
                h = np.random.uniform(0.2, 0.5) * (y_max - y_min)

                mask = (
                    (xyz[:, 0] >= cx - w/2) & (xyz[:, 0] < cx + w/2) &
                    (xyz[:, 1] >= cy - h/2) & (xyz[:, 1] < cy + h/2)
                )

                num_replace = mask.sum()
                if num_replace > 0:
                    replace_idx = np.random.choice(len(block2), num_replace,
                                                   replace=True)
                    block[mask] = block2[replace_idx]
        else:
            room_idx, cx, cy = self.test_blocks[idx]
            block, orig_indices = self._sample_block(
                self.rooms[room_idx], cx, cy, return_indices=True,
            )

        pts_for_slicing, pts_features, sem_labels = self._prepare_features(
            block, room_idx
        )

        M = getattr(self.cfg, 'num_slices', 16)
        K = getattr(self.cfg, 'points_per_slice', 256)
        fps_seed = idx if self.split == 'test' else None
        slices, geo, anchor_xyz = slice_point_cloud(pts_for_slicing, M, K,
                                                   seed=fps_seed)

        sid_arr = assign_points_to_slices(
            pts_for_slicing[:, :3], anchor_xyz
        )

        if self.split == 'train' and self.cfg is not None:
            slices, pts_features = augment_seg(slices, pts_features, self.cfg)
            geo = np.stack([compute_geo(s) for s in slices])

        base = (
            slices.astype(np.float32),
            geo.astype(np.float32),
            pts_features.astype(np.float32),
            sid_arr.astype(np.int64),
            sem_labels,
            0,
        )

        # E1: append precomputed room summary
        if self.use_room_prior:
            base = base + (self.room_summaries[room_idx].copy(),)

        if self._return_meta:
            if orig_indices is None:
                orig_indices = np.zeros(self.n_points, dtype=np.int64)
            base = base + (
                np.int64(room_idx),
                orig_indices.astype(np.int64),
            )

        return base


def compute_class_weights(data_dir: str, test_area: int = 5) -> np.ndarray:
    """Compute inverse-frequency class weights from training areas."""
    cache_path = os.path.join(
        data_dir, f"s3dis_class_weights_area{test_area}.npy"
    )
    if os.path.exists(cache_path):
        return np.load(cache_path).astype(np.float32)

    train_areas = [a for a in [1, 2, 3, 4, 5, 6] if a != test_area]
    npy_paths = S3DISDataset._discover_rooms(data_dir, train_areas)

    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for npy_path in npy_paths:
        room = np.load(npy_path)
        labels = room[:, 6].astype(int)
        for c in range(NUM_CLASSES):
            counts[c] += (labels == c).sum()

    total = counts.sum()
    freq = counts / total
    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.max()
    weights = weights.astype(np.float32)
    try:
        np.save(cache_path, weights)
    except Exception:
        pass
    return weights