"""
datasets/s3dis.py — S3DIS dataset for indoor scene segmentation.

Stanford Large-Scale 3D Indoor Spaces: 6 areas, 271 rooms, 13 classes.
Protocol: train on Areas 1,2,3,4,6 — test on Area 5.

Each room is stored as an .npy file with columns:
    [x, y, z, r, g, b, semantic_label]  (RGB in 0-255)

Training: random 1m x 1m blocks, N=4096 points per block.
Testing:  sliding-window blocks over full rooms, aggregate predictions.

Returns per sample:
    slices        [M, K, C]    C = in_channels from config
    geo           [M, 8]       geometry descriptors
    pts_features  [N, F]       per-point features for PerPointBranch
    sid_arr       [N]          slice assignment
    sem_labels    [N]          semantic labels (0-12)
    cat_id        0            dummy (no category conditioning)
"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset

from .slicing import slice_point_cloud, assign_points_to_slices, compute_geo
from .transforms import augment_seg


# 13 semantic classes
CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window',
    'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter',
]
NUM_CLASSES = 13

# Areas for train/test split
TRAIN_AREAS = [1, 2, 3, 4, 6]
TEST_AREA = 5


class S3DISDataset(Dataset):
    """
    S3DIS dataset with block-based sampling.

    During training: randomly sample blocks from rooms.
    During testing:  iterate over all blocks in test area rooms.
    """

    def __init__(self, data_dir: str, split: str, cfg=None):
        assert split in ('train', 'test')
        self.split = split
        self.cfg = cfg
        self.n_points = getattr(cfg, 'num_points', 4096)
        self.block_size = getattr(cfg, 'block_size', 1.0)
        self.use_rgb = getattr(cfg, 'use_rgb', True)
        self.use_height = getattr(cfg, 'use_height', True)

        test_area = getattr(cfg, 'test_area', 5)
        if split == 'train':
            areas = [a for a in [1, 2, 3, 4, 5, 6] if a != test_area]
        else:
            areas = [test_area]

        # Load all room files — support BOTH layouts:
        #   Folder layout:  data_dir/Area_N/room_name.npy
        #   Flat layout:    data_dir/raw/Area_N_room_name.npy  (OpenPoints)
        #   Flat layout:    data_dir/Area_N_room_name.npy      (alternate)
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
            room_data = np.load(npy_path)  # [N, 7]: x,y,z,r,g,b,label
            self.rooms.append(room_data.astype(np.float32))

        # P0 FIX: Pre-compute per-ROOM z bounds for room-relative height
        # normalization. Previously height was per-BLOCK which made floor=0
        # and ceiling=1 meaningless within a small block.
        # Now: height = (z - room_z_min) / (room_z_max - room_z_min) for the
        # WHOLE room → floor of room = 0, ceiling of room = 1 (semantic).
        self.room_z_bounds = []
        for room in self.rooms:
            z = room[:, 2]
            z_min = float(z.min())
            z_max = float(z.max())
            # Guard against degenerate rooms (single floor scan, etc.)
            if z_max - z_min < 1e-6:
                z_max = z_min + 1.0
            self.room_z_bounds.append((z_min, z_max))

        # For training: create a flat index of (room_idx, point_count)
        # so we can sample uniformly across rooms proportional to size
        self.room_sizes = [len(r) for r in self.rooms]
        self.total_points = sum(self.room_sizes)

        if split == 'train':
            # Each "sample" is one random block — we define epoch length
            # as total_points // n_points to see each point ~once per epoch
            epoch_len = getattr(cfg, 'epoch_len', -1)
            if epoch_len > 0:
                self._len = epoch_len
            else:
                self._len = self.total_points // self.n_points
        else:
            # For testing: pre-compute all block centres for sliding window
            self.test_blocks = self._precompute_test_blocks()
            self._len = len(self.test_blocks)

        print(f"[S3DIS] '{split}': {len(self.rooms)} rooms, "
              f"{self.total_points:,} points, {self._len} samples/epoch")

    def _precompute_test_blocks(self):
        """Pre-compute (room_idx, cx, cy) for sliding-window test blocks."""
        stride = self.block_size * 0.5  # 50% overlap
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
        """
        Find all room .npy files for the requested areas.
        Supports two common layouts:

            (1) Folder layout (our default):
                data_dir/Area_N/room_name.npy

            (2) Flat layout (OpenPoints preprocessed s3disfull.tar):
                data_dir/raw/Area_N_room_name.npy
                data_dir/Area_N_room_name.npy

        Returns sorted list of full file paths.
        """
        found = []
        for area in areas:
            # (1) Folder layout
            area_dir = os.path.join(data_dir, f"Area_{area}")
            if os.path.isdir(area_dir):
                found.extend(sorted(glob.glob(os.path.join(area_dir, "*.npy"))))
                continue

            # (2) Flat layout — look in raw/ subfolder first, then data_dir
            patterns = [
                os.path.join(data_dir, "raw", f"Area_{area}_*.npy"),
                os.path.join(data_dir, f"Area_{area}_*.npy"),
            ]
            for pat in patterns:
                files = sorted(glob.glob(pat))
                if files:
                    found.extend(files)
                    break  # only one layout per area

        return found

    def _sample_block(self, room: np.ndarray,
                      cx: float = None, cy: float = None):
        """
        Extract a block of points from a room.

        Args:
            room: [N, 7] full room data
            cx, cy: block centre (None = random for training)

        Returns:
            block: [n_points, 7]
        """
        xyz = room[:, :3]
        half = self.block_size / 2

        if cx is None:
            # Random block centre (training)
            x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
            x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
            cx = np.random.uniform(x_min + half, max(x_min + half, x_max - half))
            cy = np.random.uniform(y_min + half, max(y_min + half, y_max - half))

        # Select points within block
        mask = (
            (xyz[:, 0] >= cx - half) & (xyz[:, 0] < cx + half) &
            (xyz[:, 1] >= cy - half) & (xyz[:, 1] < cy + half)
        )
        block_pts = room[mask]

        if len(block_pts) == 0:
            # Fallback: take nearest n_points to centre
            dists = np.linalg.norm(xyz[:, :2] - np.array([cx, cy]), axis=1)
            idx = np.argsort(dists)[:self.n_points]
            block_pts = room[idx]

        # Sample to exact n_points
        if len(block_pts) >= self.n_points:
            choice = np.random.choice(len(block_pts), self.n_points,
                                      replace=False)
        else:
            choice = np.random.choice(len(block_pts), self.n_points,
                                      replace=True)
        return block_pts[choice]

    def _prepare_features(self, block: np.ndarray, room_idx: int):
        """
        Prepare block data into sliceable point cloud and per-point features.

        Args:
            block:    [N, 7]  x,y,z,r,g,b,label
            room_idx: index into self.room_z_bounds for per-room height normalization

        Returns:
            pts_for_slicing: [N, C]  for encoder (C matches in_channels)
            pts_features:    [N, F]  for PerPointBranch
            sem_labels:      [N]     int labels (0-12)
        """
        xyz = block[:, :3].copy()
        rgb = block[:, 3:6].copy() / 255.0  # normalise to [0,1]
        labels = block[:, 6].astype(np.int64)

        # P0 FIX: Height feature is normalized PER-ROOM, not per-block.
        # Computed BEFORE centering xyz since we need the absolute z values.
        # Now floor of room = 0, ceiling of room = 1 (semantic meaning).
        z_min, z_max = self.room_z_bounds[room_idx]
        z_vals = block[:, 2]
        height = ((z_vals - z_min) / (z_max - z_min)).astype(np.float32)
        height = np.clip(height, 0.0, 1.0)  # guard against outliers

        # Centre xyz within the block (height feature stays room-relative)
        xyz = xyz - xyz.mean(axis=0)

        # Build slicing input: xyz + rgb + (optional height)
        # The encoder expects in_channels dimensions
        parts = [xyz]
        if self.use_rgb:
            parts.append(rgb)
        if self.use_height:
            parts.append(height.reshape(-1, 1))
        pts_for_slicing = np.concatenate(parts, axis=1).astype(np.float32)

        # Per-point features for PerPointBranch (same channels)
        pts_features = pts_for_slicing.copy()

        return pts_for_slicing, pts_features, labels

    def __getitem__(self, idx):
        if self.split == 'train':
            # Random room, random block
            room_idx = np.random.randint(0, len(self.rooms))
            block = self._sample_block(self.rooms[room_idx])

            # PointCutMix (Batch/Sample-level Mixup)
            if getattr(self.cfg, 'aug_cutmix', False) and np.random.random() < getattr(self.cfg, 'aug_cutmix_prob', 0.5):
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
                    replace_idx = np.random.choice(len(block2), num_replace, replace=True)
                    block[mask] = block2[replace_idx]
        else:
            # Deterministic test block
            room_idx, cx, cy = self.test_blocks[idx]
            block = self._sample_block(self.rooms[room_idx], cx, cy)

        # Pass room_idx so height is normalized per-ROOM (not per-block)
        pts_for_slicing, pts_features, sem_labels = self._prepare_features(
            block, room_idx
        )

        # Slice
        M = getattr(self.cfg, 'num_slices', 16)
        K = getattr(self.cfg, 'points_per_slice', 256)
        fps_seed = idx if self.split == 'test' else None
        slices, geo, anchor_xyz = slice_point_cloud(pts_for_slicing, M, K, seed=fps_seed)

        # Assign points to slices (using xyz only)
        sid_arr = assign_points_to_slices(
            pts_for_slicing[:, :3], anchor_xyz
        )

        # Augment (training only)
        if self.split == 'train' and self.cfg is not None:
            slices, pts_features = augment_seg(slices, pts_features, self.cfg)
            # P0 FIX: Recompute geo from augmented slices so positional encoding
            # and SSP see the same geometry as the encoder.
            geo = np.stack([compute_geo(s) for s in slices])

        return (
            slices.astype(np.float32),          # [M, K, C]
            geo.astype(np.float32),             # [M, 8]
            pts_features.astype(np.float32),    # [N, F]
            sid_arr.astype(np.int64),           # [N]
            sem_labels,                         # [N]
            0,                                  # dummy cat_id
        )


def compute_class_weights(data_dir: str, test_area: int = 5) -> np.ndarray:
    """
    Compute inverse-frequency class weights from training areas.
    Supports both folder and flat S3DIS layouts.
    Results are cached to data_dir/s3dis_class_weights.npy to avoid
    recomputing on every training start.

    Returns:
        weights: [13] float32 normalised so max = 1.0
    """
    cache_path = os.path.join(data_dir, f"s3dis_class_weights_area{test_area}.npy")
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

    # Inverse frequency, normalised
    total = counts.sum()
    freq = counts / total
    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.max()  # normalise so max weight = 1.0
    weights = weights.astype(np.float32)
    # Cache for subsequent runs
    try:
        np.save(cache_path, weights)
    except Exception:
        pass
    return weights
