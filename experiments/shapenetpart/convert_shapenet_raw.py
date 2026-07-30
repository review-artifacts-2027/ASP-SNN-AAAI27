#!/usr/bin/env python3

import argparse
import json
import os

import h5py
import numpy as np

CATEGORY_IDS = {
    'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343',
    'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134',
    'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390',
    'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243',
}
CATEGORY_ORDER = list(CATEGORY_IDS.keys())
SYNSET_TO_LABEL = {synset: i for i, synset in enumerate(CATEGORY_IDS.values())}

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

    train_entries = load_split(args.raw_dir, 'train') + load_split(args.raw_dir, 'val')
    test_entries = load_split(args.raw_dir, 'test')

    convert_entries(args.raw_dir, args.out_dir, 'train', train_entries,
                     args.num_points, args.chunk_size, args.seed)
    convert_entries(args.raw_dir, args.out_dir, 'test', test_entries,
                     args.num_points, args.chunk_size, args.seed)

    print('Done ->', args.out_dir)

if __name__ == '__main__':
    main()
