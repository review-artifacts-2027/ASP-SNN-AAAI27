"""
datasets/convert_shapenet_raw.py — Convert raw ShapeNetPart (PartAnnotation)
to the HDF5 format our loader expects.

Input format (Kaggle "shapenetcore_partanno_segmentation_benchmark_v0_normal"):
    raw_dir/
        synsetoffset2category.txt   # category_name <tab> synset_id
        02691156/                   # synset id (Airplane)
            points/*.pts            # text files: x y z nx ny nz part_label
        02773838/                   # synset id (Bag)
            points/*.pts
        ...
        train_test_split/
            shuffled_train_file_list.json
            shuffled_val_file_list.json
            shuffled_test_file_list.json

Output format (matches Stanford HDF5 download):
    out_dir/
        train0.h5 .. trainN.h5
        test0.h5 .. testM.h5
    Each h5 contains:
        data:  [num_shapes, 2048, 3]   float32
        label: [num_shapes, 1]         int64  (0-15 category)
        pid:   [num_shapes, 2048]      int64  (0-49 global part)
        all_object_categories.txt

Usage:
    python datasets/convert_shapenet_raw.py \
        --raw_dir data/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        --out_dir data/shapenet_part_seg_hdf5_data
"""

import argparse
import json
import os

import numpy as np


# Standard 16 ShapeNetPart categories (synset id → category index 0-15)
SYNSET_TO_CAT = {
    "02691156": 0,   # Airplane
    "02773838": 1,   # Bag
    "02954340": 2,   # Cap
    "02958343": 3,   # Car
    "03001627": 4,   # Chair
    "03261776": 5,   # Earphone
    "03467517": 6,   # Guitar
    "03624134": 7,   # Knife
    "03636649": 8,   # Lamp
    "03642806": 9,   # Laptop
    "03790512": 10,  # Motorbike
    "03797390": 11,  # Mug
    "03948459": 12,  # Pistol
    "04099429": 13,  # Rocket
    "04225987": 14,  # Skateboard
    "04379243": 15,  # Table
}

CATEGORY_NAMES = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
    'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',
    'Rocket', 'Skateboard', 'Table',
]

# Per-category part label offsets — global part label is offset + local part
PART_OFFSET = {
    0: 0,   1: 4,   2: 6,   3: 8,   4: 12,  5: 16,  6: 19,  7: 22,
    8: 24,  9: 28,  10: 30, 11: 36, 12: 38, 13: 41, 14: 44, 15: 47,
}
PART_COUNT = {
    0: 4,   1: 2,   2: 2,   3: 4,   4: 4,   5: 3,   6: 3,   7: 2,
    8: 4,   9: 2,   10: 6,  11: 2,  12: 3,  13: 3,  14: 3,  15: 3,
}
CAT_TO_GLOBAL_PARTS = {
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
    14: [44, 45, 46],
    15: [47, 48, 49],
}

NUM_POINTS_PER_SHAPE = 2048
SHAPES_PER_H5 = 2048  # how many shapes per output h5 file


def to_global_part_labels(part_n: np.ndarray, cat_idx: int):
    """Return category-global ShapeNetPart labels, or None if invalid."""
    observed_parts = set(map(int, np.unique(part_n)))
    valid_parts = set(CAT_TO_GLOBAL_PARTS[cat_idx])
    if observed_parts.issubset(valid_parts):
        return part_n

    local_count = PART_COUNT[cat_idx]
    if part_n.min() >= 1 and part_n.max() <= local_count:
        return part_n - 1 + PART_OFFSET[cat_idx]
    if part_n.min() >= 0 and part_n.max() < local_count:
        return part_n + PART_OFFSET[cat_idx]
    return None


def find_label_file(pts_path: str):
    """Locate separate PartAnnotation labels for a points file."""
    stem = os.path.splitext(os.path.basename(pts_path))[0]
    points_dir = os.path.dirname(pts_path)
    synset_dir = (
        os.path.dirname(points_dir)
        if os.path.basename(points_dir) == "points"
        else points_dir
    )
    candidates = []
    for label_dir_name in ("points_label", "labels", "seg", "segmentation"):
        label_dir = os.path.join(synset_dir, label_dir_name)
        for ext in (".seg", ".txt", ".pts"):
            candidates.append(os.path.join(label_dir, f"{stem}{ext}"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_shape(pts_path: str) -> tuple:
    """
    Load a .pts file from PartAnnotation.

    Each row has either:
        x y z part_label                 (4 cols, basic variant)
        x y z nx ny nz part_label        (7 cols, _normal variant)
    """
    data = np.loadtxt(pts_path).astype(np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 3:
        label_path = find_label_file(pts_path)
        if label_path is None:
            raise ValueError(f"No separate label file found for {pts_path}")
        part = np.loadtxt(label_path).astype(np.int64)
        if part.ndim > 1:
            part = part.reshape(-1)
        xyz = data[:, :3]
        if len(part) != len(xyz):
            raise ValueError(
                f"Point/label count mismatch for {pts_path}: "
                f"{len(xyz)} points vs {len(part)} labels"
            )
    elif data.shape[1] == 4:
        xyz, part = data[:, :3], data[:, 3]
    elif data.shape[1] >= 7:
        xyz, part = data[:, :3], data[:, -1]
    else:
        raise ValueError(f"Unexpected column count in {pts_path}: {data.shape}")
    return xyz, part.astype(np.int64)


def resample(xyz: np.ndarray, part: np.ndarray, n: int) -> tuple:
    """Sample exactly n points (with replacement if needed)."""
    if len(xyz) >= n:
        idx = np.random.choice(len(xyz), n, replace=False)
    else:
        idx = np.random.choice(len(xyz), n, replace=True)
    return xyz[idx], part[idx]


def load_split_list(raw_dir: str, split: str) -> list:
    """Parse the train_test_split JSON file."""
    fname = f"shuffled_{split}_file_list.json"
    path = os.path.join(raw_dir, "train_test_split", fname)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        items = json.load(f)
    # Items can look like "shape_data/02691156/abcd1234" or include
    # "points/abcd1234.pts" depending on the mirror.
    parsed = []
    for item in items:
        parsed_item = normalise_split_item(item)
        if parsed_item is None:
            continue
        parsed.append(parsed_item)
    return parsed


def normalise_split_item(item: str):
    parts = item.replace("\\", "/").strip().split("/")
    for i, part in enumerate(parts):
        if part not in SYNSET_TO_CAT:
            continue
        tail = parts[i + 1:]
        if not tail:
            return None
        if "points" in tail:
            pidx = tail.index("points")
            if pidx + 1 >= len(tail):
                return None
            name = tail[pidx + 1]
        else:
            name = tail[-1]
        return part, os.path.splitext(name)[0]
    return None


def find_pts_file(raw_dir: str, synset: str, name: str) -> str:
    """Locate the .pts/.txt file for a given shape."""
    candidates = [
        os.path.join(raw_dir, synset, "expert_verified", "points", f"{name}.pts"),
        os.path.join(raw_dir, synset, "expert_verified", "points", f"{name}.txt"),
        os.path.join(raw_dir, synset, "expert_verified", f"{name}.pts"),
        os.path.join(raw_dir, synset, "expert_verified", f"{name}.txt"),
        os.path.join(raw_dir, synset, "points", f"{name}.pts"),
        os.path.join(raw_dir, synset, "points", f"{name}.txt"),
        os.path.join(raw_dir, synset, f"{name}.pts"),
        os.path.join(raw_dir, synset, f"{name}.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def next_h5_index(out_dir: str, prefix: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    next_idx = 0
    for fname in os.listdir(out_dir):
        if not (fname.startswith(prefix) and fname.endswith(".h5")):
            continue
        idx_text = fname[len(prefix):-3]
        if idx_text.isdigit():
            next_idx = max(next_idx, int(idx_text) + 1)
    return next_idx


def write_h5_chunk(out_dir: str, prefix: str, idx: int,
                   data: np.ndarray, label: np.ndarray, pid: np.ndarray):
    """Write one h5 chunk."""
    import h5py
    path = os.path.join(out_dir, f"{prefix}{idx}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("label", data=label, compression="gzip")
        f.create_dataset("pid", data=pid, compression="gzip")
    print(f"  wrote {path}  ({len(data)} shapes)")


def convert_split(raw_dir: str, out_dir: str, split: str, prefix: str):
    """Convert one split (train/val/test) to HDF5 chunks."""
    pairs = load_split_list(raw_dir, split)
    if not pairs:
        print(f"  [skip] no entries for split '{split}'")
        return 0

    np.random.seed(0)  # deterministic resampling

    data_buf, label_buf, pid_buf = [], [], []
    chunk_idx = next_h5_index(out_dir, prefix)
    n_written = 0

    for i, (synset, name) in enumerate(pairs):
        pts_path = find_pts_file(raw_dir, synset, name)
        if pts_path is None:
            continue

        try:
            xyz, part = load_shape(pts_path)
        except Exception as e:
            print(f"  [warn] failed to read {pts_path}: {e}")
            continue

        xyz_n, part_n = resample(xyz, part, NUM_POINTS_PER_SHAPE)

        cat_idx = SYNSET_TO_CAT[synset]

        part_global = to_global_part_labels(part_n, cat_idx)
        if part_global is None:
            print(f"  [warn] unexpected label range in {name}: "
                  f"[{part_n.min()},{part_n.max()}], skipping")
            continue

        data_buf.append(xyz_n)
        label_buf.append(cat_idx)
        pid_buf.append(part_global)

        if len(data_buf) >= SHAPES_PER_H5:
            arr_data  = np.stack(data_buf).astype(np.float32)
            arr_label = np.array(label_buf, dtype=np.int64).reshape(-1, 1)
            arr_pid   = np.stack(pid_buf).astype(np.int64)
            write_h5_chunk(out_dir, prefix, chunk_idx,
                           arr_data, arr_label, arr_pid)
            data_buf, label_buf, pid_buf = [], [], []
            chunk_idx += 1
            n_written += len(arr_data)

        if (i + 1) % 500 == 0:
            print(f"  ... processed {i+1}/{len(pairs)}")

    # Final partial chunk
    if data_buf:
        arr_data  = np.stack(data_buf).astype(np.float32)
        arr_label = np.array(label_buf, dtype=np.int64).reshape(-1, 1)
        arr_pid   = np.stack(pid_buf).astype(np.int64)
        write_h5_chunk(out_dir, prefix, chunk_idx,
                       arr_data, arr_label, arr_pid)
        n_written += len(arr_data)

    return n_written


def write_metadata(out_dir: str):
    """Write all_object_categories.txt so loader treats this as valid."""
    path = os.path.join(out_dir, "all_object_categories.txt")
    with open(path, "w") as f:
        for name, sid in zip(CATEGORY_NAMES, SYNSET_TO_CAT.keys()):
            f.write(f"{name}\t{sid}\n")


def convert_shapenet_archive_streaming(archive_path: str, out_dir: str) -> tuple:
    """
    Convert a ShapeNetPart ZIP archive directly to HDF5 without extracting to disk.

    Uses a SINGLE sequential pass through the archive to read all pts/seg data
    into memory (~200 MB), then writes HDF5 chunks. This avoids the random-seek
    penalty of reading by name from a large ZIP file.

    Requires only ~550 MB total (200 MB RAM + 350 MB HDF5 output).

    Returns (n_train, n_test) tuple of shapes written.
    """
    import zipfile

    try:
        import h5py  # noqa: F401 — validated before we start
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

    os.makedirs(out_dir, exist_ok=True)
    size_mb = os.path.getsize(archive_path) // 1_000_000
    print(f"[ShapeNet] Streaming archive -> HDF5  ({size_mb} MB)")

    # ── Single sequential pass: read all pts + seg data into memory ───────
    # Sequential reads from a ZIP are fast (100+ MB/s) vs random seeks (50ms each).
    pts_raw = {}   # (synset, stem) → bytes
    seg_raw = {}   # (synset, stem) → bytes

    print("[ShapeNet] Reading archive (single sequential pass)...", flush=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        total = len(zf.infolist())
        for i, info in enumerate(zf.infolist()):
            entry = info.filename
            if not (entry.endswith(".pts") or entry.endswith(".seg")):
                continue
            norm = entry.replace("\\", "/").split("/")
            for j, part in enumerate(norm):
                if part not in SYNSET_TO_CAT:
                    continue
                if j + 1 >= len(norm):
                    break
                stem = os.path.splitext(norm[-1])[0]
                key = (part, stem)
                data = zf.read(entry)
                if entry.endswith(".pts"):
                    pts_raw[key] = data
                else:
                    seg_raw[key] = data
                break
            if (i + 1) % 40000 == 0:
                print(f"  scanned {i+1}/{total} entries, "
                      f"pts={len(pts_raw)} seg={len(seg_raw)}", flush=True)

    # ── Build valid pairs (have both pts + seg) ───────────────────────────
    pairs = [
        (synset, stem, pts_raw[(synset, stem)], seg_raw[(synset, stem)])
        for (synset, stem) in sorted(pts_raw)
        if (synset, stem) in seg_raw
    ]
    print(f"[ShapeNet] {len(pairs)} shapes with both pts + seg")

    # ── Stratified 80/20 split by synset ─────────────────────────────────
    np.random.seed(42)
    train_pairs, test_pairs = [], []
    by_synset = {}
    for p in pairs:
        by_synset.setdefault(p[0], []).append(p)

    for synset in sorted(by_synset):
        bucket = by_synset[synset]
        order = np.random.permutation(len(bucket))
        n_tr = max(1, int(len(bucket) * 0.8))
        for k in order[:n_tr]:
            train_pairs.append(bucket[k])
        for k in order[n_tr:]:
            test_pairs.append(bucket[k])

    print(f"[ShapeNet] Split: {len(train_pairs)} train / {len(test_pairs)} test")

    # ── Convert helper (data already in RAM — no I/O bottleneck) ─────────
    def _process(split_pairs, prefix):
        buf_data, buf_label, buf_pid = [], [], []
        chunk = n_written = n_skip = 0

        for i, (synset, stem, pts_bytes, seg_bytes) in enumerate(split_pairs):
            try:
                # Parse XYZ from .pts (first 3 columns, space-separated)
                pts_lines = pts_bytes.decode("latin-1").splitlines()
                rows = []
                for ln in pts_lines:
                    v = ln.split()
                    if len(v) >= 3:
                        rows.append((float(v[0]), float(v[1]), float(v[2])))
                if not rows:
                    n_skip += 1
                    continue
                xyz = np.array(rows, dtype=np.float32)

                # Parse labels from .seg (one integer per line)
                seg_lines = seg_bytes.decode("latin-1").splitlines()
                part = np.array(
                    [int(ln) for ln in seg_lines if ln.strip()],
                    dtype=np.int64,
                )

                if len(part) != len(xyz):
                    n_skip += 1
                    continue

                xyz_n, part_n = resample(xyz, part, NUM_POINTS_PER_SHAPE)
                cat_idx = SYNSET_TO_CAT[synset]

                part_g = to_global_part_labels(part_n, cat_idx)
                if part_g is None:
                    n_skip += 1
                    continue

                buf_data.append(xyz_n)
                buf_label.append(cat_idx)
                buf_pid.append(part_g)

                if len(buf_data) >= SHAPES_PER_H5:
                    write_h5_chunk(
                        out_dir, prefix, chunk,
                        np.stack(buf_data).astype(np.float32),
                        np.array(buf_label, np.int64).reshape(-1, 1),
                        np.stack(buf_pid).astype(np.int64),
                    )
                    n_written += len(buf_data)
                    buf_data, buf_label, buf_pid = [], [], []
                    chunk += 1

                if (i + 1) % 2000 == 0:
                    print(f"  ... {i+1}/{len(split_pairs)}", flush=True)

            except Exception:
                n_skip += 1

        if buf_data:
            write_h5_chunk(
                out_dir, prefix, chunk,
                np.stack(buf_data).astype(np.float32),
                np.array(buf_label, np.int64).reshape(-1, 1),
                np.stack(buf_pid).astype(np.int64),
            )
            n_written += len(buf_data)

        if n_skip:
            print(f"  [warn] {n_skip} shapes skipped")
        return n_written

    # ── Run ───────────────────────────────────────────────────────────
    print("[ShapeNet] Converting train split...")
    n_tr = _process(train_pairs, "train")
    print("[ShapeNet] Converting test split...")
    n_te = _process(test_pairs, "test")

    write_metadata(out_dir)
    print(f"[ShapeNet] Streaming done: {n_tr} train + {n_te} test shapes")
    return n_tr, n_te


def main():
    p = argparse.ArgumentParser(
        description="Convert raw PartAnnotation ShapeNetPart to HDF5"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--raw_dir", type=str, default=None,
                   help="Root of extracted raw PartAnnotation (contains synset folders)")
    g.add_argument("--archive", type=str, default=None,
                   help="Path to a ShapeNetPart ZIP archive — convert without extraction")
    p.add_argument("--out_dir", type=str,
                   default="data/shapenet_part_seg_hdf5_data",
                   help="Output directory for HDF5 files")
    args = p.parse_args()

    if args.archive:
        os.makedirs(args.out_dir, exist_ok=True)
        n_tr, n_te = convert_shapenet_archive_streaming(args.archive, args.out_dir)
        print(f"\nDone. {n_tr} train + {n_te} test shapes in {args.out_dir}/")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Converting raw -> HDF5")
    print(f"  raw:  {args.raw_dir}")
    print(f"  out:  {args.out_dir}")

    # Train split (combine train + val into train h5)
    print("\n[train + val]")
    n_tr = convert_split(args.raw_dir, args.out_dir, "train", "train")
    n_va = convert_split(args.raw_dir, args.out_dir, "val", "train")
    print(f"  total train+val: {n_tr + n_va} shapes")

    # Test split
    print("\n[test]")
    n_te = convert_split(args.raw_dir, args.out_dir, "test", "test")
    print(f"  total test: {n_te} shapes")

    write_metadata(args.out_dir)
    print(f"\nDone. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()
