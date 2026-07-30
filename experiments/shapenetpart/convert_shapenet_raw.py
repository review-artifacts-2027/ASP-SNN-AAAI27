#!/usr/bin/env python3
"""
Convert raw ShapeNetPart (v0_normal benchmark layout) into the standard
PointNet-style HDF5 format used by shapenet_part_seg_hdf5_data folders.

Expected --raw_dir layout (this is what the Kaggle mitkir/shapenet mirror
extracts to):
    <raw_dir>/synsetoffset2category.txt
    <raw_dir>/train_test_split/shuffled_{train,val,test}_file_list.json
    <raw_dir>/<synset_id>/<shape_id>.txt   (7 cols: x y z nx ny nz part_label)

Output (--out_dir):
    all_object_categories.txt
    {train,val,test}_hdf5_file_list.txt
    ply_data_{train,val,test}<N>.h5   with datasets: data, label, pid

Usage:
    python convert_shapenet_raw.py \
        --raw_dir /kaggle/working/data/shapenetcore_partanno_segmentation_benchmark_v0_normal \
        --out_dir /kaggle/working/data/shapenet_part_seg_hdf5_data
"""
import argparse
import json
import os

import h5py
import numpy as np

# Standard 16-category ShapeNetPart mapping (synset IDs match the official
# benchmark and torch_geometric's ShapeNet loader).
CATEGORY_IDS = {
    'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343',
    'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134',
    'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390',
    'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243',
}
CATEGORY_ORDER = list(CATEGORY_IDS.keys())
SYNSET_TO_LABEL = {synset: i for i, synset in enumerate(CATEGORY_IDS.values())}

# Global part-id offsets — MUST match datasets/shapenetpart.py's
# CATEGORY_TO_PARTS exactly (note: Skateboard=2 parts, Table=4 parts in
# that file, which differs from the common PointNet convention of 3/3 —
# using their exact mapping here so generated pid values stay consistent
# with what their loader/eval code assumes per category).
SEG_CLASSES = {
    'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15], 'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21],
    'Knife': [22, 23], 'Lamp': [24, 25, 26, 27], 'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40],
    'Rocket': [41, 42, 43], 'Skateboard': [44, 45], 'Table': [46, 47, 48, 49],
}
SYNSET_TO_VALID_PARTS = {
    CATEGORY_IDS[cat]: set(SEG_CLASSES[cat]) for cat in CATEGORY_ORDER
}


def load_split(raw_dir, split_name):
    path = os.path.join(raw_dir, 'train_test_split', f'shuffled_{split_name}_file_list.json')
    with open(path) as f:
        entries = json.load(f)
    out = []
    for e in entries:
        parts = e.replace('\\', '/').split('/')
        synset, shape_id = parts[-2], parts[-1]
        out.append((synset, shape_id))
    return out


def sample_points(xyz, seg, num_points, rng):
    n = xyz.shape[0]
    replace = n < num_points
    idx = rng.choice(n, num_points, replace=replace)
    return xyz[idx], seg[idx]


def convert_entries(raw_dir, out_dir, out_split_name, entries, num_points, chunk_size, seed):
    """Write entries (list of (synset, shape_id)) to train*.h5 / test*.h5
    files, matching the glob pattern datasets/shapenetpart.py expects."""
    rng = np.random.default_rng(seed)
    chunk_idx = 0
    skipped = 0
    total_written = 0

    for start in range(0, len(entries), chunk_size):
        chunk = entries[start:start + chunk_size]
        data = np.zeros((len(chunk), num_points, 3), dtype=np.float32)
        label = np.zeros((len(chunk), 1), dtype=np.uint8)
        pid = np.zeros((len(chunk), num_points), dtype=np.uint8)
        valid = 0

        for synset, shape_id in chunk:
            txt_path = os.path.join(raw_dir, synset, f'{shape_id}.txt')
            if not os.path.exists(txt_path):
                skipped += 1
                continue
            raw = np.loadtxt(txt_path, dtype=np.float32)
            if raw.ndim == 1:
                raw = raw[None, :]
            if raw.shape[1] < 7:
                skipped += 1
                continue
            xyz = raw[:, 0:3]
            # NOTE: the raw label column in this dataset is already the
            # GLOBAL 0-49 part id, not a per-category local index -- do
            # NOT add a category offset here. (Confirmed empirically:
            # adding the offset produced ids exactly one category-offset
            # too high, e.g. Knife shapes landing in Skateboard's range.)
            global_part = raw[:, 6].astype(int)

            valid_ids = SYNSET_TO_VALID_PARTS[synset]
            seen_ids = set(np.unique(global_part).tolist())
            if not seen_ids.issubset(valid_ids):
                skipped += 1
                continue

            xyz_s, part_s = sample_points(xyz, global_part, num_points, rng)

            data[valid] = xyz_s
            label[valid, 0] = SYNSET_TO_LABEL[synset]
            pid[valid] = part_s
            valid += 1

        if valid == 0:
            continue

        # Filename MUST start with the split name ("train"/"test") to match
        # the loader's glob(f"{split}*.h5") — no "ply_data_" prefix.
        fname = f'{out_split_name}{chunk_idx}.h5'
        with h5py.File(os.path.join(out_dir, fname), 'w') as hf:
            hf.create_dataset('data', data=data[:valid])
            hf.create_dataset('label', data=label[:valid])
            hf.create_dataset('pid', data=pid[:valid])
        chunk_idx += 1
        total_written += valid
        print(f'[{out_split_name}] wrote {fname} ({valid} shapes)')

    if skipped:
        print(f'[{out_split_name}] WARNING: skipped {skipped} shapes (missing file or bad column count)')
    print(f'[{out_split_name}] total: {total_written} shapes written')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--num_points', type=int, default=2048)
    ap.add_argument('--chunk_size', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # datasets/shapenetpart.py only reads "train" and "test" — merge the
    # official train+val splits into "train" (12137+1870=14007, matching
    # the loader's own docstring).
    train_entries = load_split(args.raw_dir, 'train') + load_split(args.raw_dir, 'val')
    test_entries = load_split(args.raw_dir, 'test')

    convert_entries(args.raw_dir, args.out_dir, 'train', train_entries,
                     args.num_points, args.chunk_size, args.seed)
    convert_entries(args.raw_dir, args.out_dir, 'test', test_entries,
                     args.num_points, args.chunk_size, args.seed)

    print('Done ->', args.out_dir)


if __name__ == '__main__':
    main()