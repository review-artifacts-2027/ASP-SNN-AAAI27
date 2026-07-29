"""
datasets/download.py — Download and prepare all three datasets.

Download methods per dataset:
    ShapeNetPart  : Stanford -> Google Drive -> Kaggle PartAnnotation
    ScanObjectNN  : HuggingFace mirror (no form needed) → gdown fallback
    S3DIS         : gdown from Google Drive (OpenPoints preprocessed) → manual fallback

All methods tested from US university networks.

Usage:
    python datasets/download.py --all
    python datasets/download.py --shapenet
    python datasets/download.py --scanobj
    python datasets/download.py --s3dis
    python datasets/download.py --s3dis_preprocess /path/to/Stanford3dDataset_v1.2
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

DATA_ROOT = "data"


# ═══════════════════════════════════════════════════════════════════════
#  ShapeNetPart — Stanford direct download (no auth)
# ═══════════════════════════════════════════════════════════════════════

SHAPENET_URL = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
# Public Google Drive mirror (used by PointNeXt and others)
SHAPENET_GDRIVE_ID = "1tEnSGAdgfp-NPVS5y_ALD8eF18bzwhM_"
SHAPENET_KAGGLE_SLUG = "majdouline20/shapenetpart-dataset"
SHAPENET_KAGGLE_URL = (
    "https://www.kaggle.com/datasets/majdouline20/shapenetpart-dataset"
)
SHAPENET_DIR = "shapenet_part_seg_hdf5_data"


def download_shapenet():
    """
    ShapeNetPart HDF5 — multi-source download.
    Tries in order:
      0. Cached kagglehub archive (stream to HDF5, no extraction needed)
      1. Stanford direct URL
      2. Google Drive mirror via gdown
      3. Kaggle raw PartAnnotation download + conversion
      4. Manual instructions
    """
    out_dir = os.path.join(DATA_ROOT, SHAPENET_DIR)
    sentinel = os.path.join(out_dir, "all_object_categories.txt")

    if os.path.exists(sentinel):
        print(f"[ShapeNet] Already present at {out_dir}")
        return True

    os.makedirs(DATA_ROOT, exist_ok=True)

    # ── Method 0: cached kagglehub archive (fastest, no extraction) ────
    archive_path = _find_kagglehub_cached_archive()
    if archive_path:
        mb = os.path.getsize(archive_path) // 1_000_000
        print(f"[ShapeNet] Cached archive found ({mb} MB) — streaming to HDF5 "
              "(no extraction needed)...")
        if _stream_convert_shapenet(archive_path, out_dir):
            return True

    # ── Method 1: Stanford direct ──────────────────────────────────────
    print(f"[ShapeNet] Trying Stanford direct ({SHAPENET_URL}) ...")
    zip_path = os.path.join(DATA_ROOT, "shapenet_part_seg_hdf5_data.zip")
    try:
        _download_with_progress(SHAPENET_URL, zip_path, timeout=30)
        print(f"[ShapeNet] Extracting ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_ROOT)
        os.remove(zip_path)
        if os.path.exists(sentinel):
            print(f"[ShapeNet] Done (from Stanford)")
            return True
    except Exception as e:
        print(f"[ShapeNet] Stanford failed: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # ── Method 2: gdown from Google Drive ──────────────────────────────
    print("[ShapeNet] Trying Google Drive mirror via gdown ...")
    try:
        import gdown
        gdown.download(id=SHAPENET_GDRIVE_ID, output=zip_path, quiet=False)
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1_000_000:
            print(f"[ShapeNet] Extracting ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_ROOT)
            os.remove(zip_path)
            if os.path.exists(sentinel):
                print(f"[ShapeNet] Done (from gdown mirror)")
                return True
    except ImportError:
        print("[ShapeNet] gdown not installed (pip install gdown)")
    except Exception as e:
        print(f"[ShapeNet] gdown failed: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # ── Method 3: Manual instructions ──────────────────────────────────
    print(f"[ShapeNet] Trying Kaggle dataset ({SHAPENET_KAGGLE_SLUG}) ...")
    try:
        if download_shapenet_kaggle():
            return True
    except Exception as e:
        print(f"[ShapeNet] Kaggle failed: {e}")

    print()
    print("=" * 60)
    print("  ShapeNetPart auto-download failed from all sources")
    print("=" * 60)
    print()
    print("  Option A: Download from Kaggle and convert to HDF5")
    print("    1. Install dependencies:  pip install kagglehub kaggle")
    print("    2. If needed, set up Kaggle API credentials:")
    print("       https://www.kaggle.com/docs/api")
    print(f"    3. Dataset: {SHAPENET_KAGGLE_URL}")
    print("    4. Re-run:")
    print("       python datasets/download.py --shapenet_kaggle")
    print("    5. Or convert an already extracted raw folder:")
    print("       python datasets/convert_shapenet_raw.py \\")
    print("           --raw_dir data/PartAnnotation \\")
    print("           --out_dir data/shapenet_part_seg_hdf5_data")
    print()
    print("  Option B: Manual Stanford download")
    print(f"    1. Visit:  {SHAPENET_URL}")
    print(f"    2. Place the .zip in {DATA_ROOT}/")
    print(f"    3. Re-run: python datasets/download.py --shapenet")
    print()
    return False


def _find_kagglehub_cached_archive():
    """Return the path to a cached kagglehub archive for ShapeNetPart, or None."""
    owner, dataset = SHAPENET_KAGGLE_SLUG.split("/")
    dataset_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets", owner, dataset
    )
    if not os.path.isdir(dataset_dir):
        return None
    for fname in sorted(os.listdir(dataset_dir), reverse=True):
        if fname.endswith(".archive"):
            path = os.path.join(dataset_dir, fname)
            if os.path.getsize(path) > 100_000_000:  # at least 100 MB = real archive
                return path
    return None


def _stream_convert_shapenet(archive_path: str, out_dir: str) -> bool:
    """Convert a cached kagglehub archive to HDF5 using the streaming path."""
    sentinel = os.path.join(out_dir, "all_object_categories.txt")
    try:
        try:
            from datasets.convert_shapenet_raw import (
                convert_shapenet_archive_streaming, write_metadata,
            )
        except ImportError:
            from convert_shapenet_raw import (
                convert_shapenet_archive_streaming, write_metadata,
            )
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        n_tr, n_te = convert_shapenet_archive_streaming(archive_path, out_dir)
        if os.path.exists(sentinel) and n_tr > 0 and n_te > 0:
            print(f"[ShapeNet] Done via streaming: {n_tr} train + {n_te} test")
            return True
    except Exception as e:
        print(f"[ShapeNet] Streaming conversion failed: {e}")
    return False


def download_shapenet_kaggle():
    """
    Download the Kaggle ShapeNetPart PartAnnotation dataset and convert it.

    Tries in order:
      1. Streaming conversion from a pre-cached archive (no extraction needed)
      2. kagglehub download + full extraction (first-time, enough disk space)
      3. Streaming conversion from newly-downloaded archive (extraction failed)
      4. Kaggle CLI extraction
    """
    out_dir = os.path.join(DATA_ROOT, SHAPENET_DIR)
    sentinel = os.path.join(out_dir, "all_object_categories.txt")
    if os.path.exists(sentinel):
        print(f"[ShapeNet] Already present at {out_dir}")
        return True

    os.makedirs(DATA_ROOT, exist_ok=True)

    # ── Fast path: archive already cached → skip 2.3 GB extraction entirely ──
    archive_path = _find_kagglehub_cached_archive()
    if archive_path:
        mb = os.path.getsize(archive_path) // 1_000_000
        print(f"[ShapeNet] Cached archive found ({mb} MB) — streaming to HDF5 "
              "(no extraction needed)...")
        if _stream_convert_shapenet(archive_path, out_dir):
            return True

    # ── Normal path: download + extract via kagglehub ─────────────────────────
    download_root = os.path.join(DATA_ROOT, "_kaggle_shapenetpart")
    os.makedirs(download_root, exist_ok=True)

    raw_dir = None
    try:
        import kagglehub
        print("[ShapeNet] Downloading Kaggle dataset with kagglehub ...")
        kaggle_path = kagglehub.dataset_download(SHAPENET_KAGGLE_SLUG)
        raw_dir = _find_shapenet_raw_dir(kaggle_path)
        if raw_dir is None:
            print(f"[ShapeNet] Could not find PartAnnotation under {kaggle_path}")
    except ImportError:
        print("[ShapeNet] kagglehub not installed (pip install kagglehub)")
    except Exception as e:
        print(f"[ShapeNet] kagglehub failed: {e}")

    # Extraction succeeded → use disk-based converter
    if raw_dir is not None:
        return prepare_shapenet_raw(raw_dir)

    # Extraction failed (disk full?) → try streaming from newly-cached archive
    archive_path = _find_kagglehub_cached_archive()
    if archive_path:
        if _stream_convert_shapenet(archive_path, out_dir):
            return True

    raw_dir = _download_shapenet_with_kaggle_cli(download_root)

    if raw_dir is None:
        print("[ShapeNet] Kaggle download did not produce a raw PartAnnotation folder.")
        return False

    return prepare_shapenet_raw(raw_dir)


def _download_shapenet_with_kaggle_cli(download_root: str):
    """Download the Kaggle dataset with the official Kaggle CLI."""
    print("[ShapeNet] Trying Kaggle CLI ...")
    kaggle_cmd = _kaggle_cli_command()
    if kaggle_cmd is None:
        print("[ShapeNet] kaggle CLI not found (pip install kaggle)")
        return None

    cmd = kaggle_cmd + [
        "datasets",
        "download",
        "-d",
        SHAPENET_KAGGLE_SLUG,
        "-p",
        download_root,
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ShapeNet] Kaggle CLI failed with exit code {e.returncode}")
        return None

    return _find_shapenet_raw_dir(download_root)


def _kaggle_cli_command():
    """Return a command prefix for the Kaggle CLI, including Windows user scripts."""
    for name in ("kaggle", "kaggle.exe"):
        exe = shutil.which(name)
        if exe:
            return [exe]

    candidates = []
    if os.name == "nt":
        candidates.append(
            os.path.join(
                os.path.expanduser("~"),
                "AppData",
                "Roaming",
                "Python",
                f"Python{sys.version_info.major}{sys.version_info.minor}",
                "Scripts",
                "kaggle.exe",
            )
        )
        candidates.append(
            os.path.join(os.path.dirname(sys.executable), "Scripts", "kaggle.exe")
        )

    for exe in candidates:
        if os.path.exists(exe):
            return [exe]
    return None


def prepare_shapenet_raw(raw_dir: str):
    """Convert an extracted ShapeNetPart PartAnnotation folder to HDF5."""
    raw_dir = _find_shapenet_raw_dir(raw_dir)
    if raw_dir is None:
        print("[ShapeNet] Could not find raw ShapeNetPart PartAnnotation files.")
        return False

    out_dir = os.path.join(DATA_ROOT, SHAPENET_DIR)
    sentinel = os.path.join(out_dir, "all_object_categories.txt")
    print(f"[ShapeNet] Converting raw PartAnnotation from {raw_dir}")

    try:
        from datasets.convert_shapenet_raw import convert_split, write_metadata
    except Exception:
        from convert_shapenet_raw import convert_split, write_metadata

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n_train = convert_split(raw_dir, out_dir, "train", "train")
    n_val = convert_split(raw_dir, out_dir, "val", "train")
    n_test = convert_split(raw_dir, out_dir, "test", "test")
    write_metadata(out_dir)

    if os.path.exists(sentinel) and n_train + n_val > 0 and n_test > 0:
        print(
            f"[ShapeNet] Ready at {out_dir}: "
            f"{n_train + n_val} train/val + {n_test} test shapes"
        )
        return True

    print("[ShapeNet] Conversion finished but no usable train/test H5 files were written.")
    return False


def _find_shapenet_raw_dir(root: str):
    """Find a raw ShapeNetPart folder containing synset dirs and split JSONs."""
    if not root or not os.path.exists(root):
        return None

    expected_synsets = {
        "02691156", "02773838", "02954340", "02958343",
        "03001627", "03261776", "03467517", "03624134",
        "03636649", "03642806", "03790512", "03797390",
        "03948459", "04099429", "04225987", "04379243",
    }
    for cur, dirs, files in os.walk(root):
        has_splits = "train_test_split" in dirs
        synset_hits = len(expected_synsets.intersection(dirs))
        if has_splits and synset_hits >= 8:
            return cur
        if "synsetoffset2category.txt" in files and synset_hits >= 8:
            return cur
    return None


# ═══════════════════════════════════════════════════════════════════════
#  ScanObjectNN — HuggingFace mirror (no form needed!)
# ═══════════════════════════════════════════════════════════════════════

# This HuggingFace mirror contains the PB_T50_RS variant (hardest)
# and does NOT require filling out any license form.
SCANOBJ_HF_URL = (
    "https://huggingface.co/datasets/cminst/ScanObjectNN/resolve/main/"
    "scanobjectnn_PB_T50_RS_h5.zip"
)
# Google Drive fallback (OpenPoints preprocessed tar)
SCANOBJ_GDRIVE_ID = "1iM3mhMJ_N0x5pytcP831l3ZFwbLmbwzi"


def download_scanobjectnn():
    """
    ScanObjectNN PB_T50_RS — auto-download from HuggingFace mirror.
    No license form required for the HF mirror.
    Falls back to Google Drive if HF fails.

    Expected result:
        data/ScanObjectNN/main_split/
            training_objectdataset_augmentedrot_scale75.h5  (11,416 shapes)
            test_objectdataset_augmentedrot_scale75.h5      (2,882 shapes)
    """
    out_dir = os.path.join(DATA_ROOT, "ScanObjectNN", "main_split")
    train_file = os.path.join(out_dir,
                              "training_objectdataset_augmentedrot_scale75.h5")
    test_file = os.path.join(out_dir,
                             "test_objectdataset_augmentedrot_scale75.h5")

    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"[ScanObjectNN] Already present at {out_dir}")
        return True

    os.makedirs(out_dir, exist_ok=True)

    # ── Method 1: HuggingFace direct download ─────────────────────────
    print("[ScanObjectNN] Downloading from HuggingFace mirror ...")
    hf_zip = os.path.join(DATA_ROOT, "scanobjectnn_hf.zip")
    try:
        _download_with_progress(SCANOBJ_HF_URL, hf_zip)
        print("[ScanObjectNN] Extracting ...")
        with zipfile.ZipFile(hf_zip, 'r') as zf:
            zf.extractall(os.path.join(DATA_ROOT, "ScanObjectNN"))
        os.remove(hf_zip)

        # The HF zip may extract with a slightly different structure
        # Verify the files ended up in the right place
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("[ScanObjectNN] Done (from HuggingFace)")
            return True

        # Check if files extracted to a subfolder
        for root, dirs, files in os.walk(os.path.join(DATA_ROOT, "ScanObjectNN")):
            for f in files:
                if f == "training_objectdataset_augmentedrot_scale75.h5":
                    src = os.path.join(root, f)
                    if src != train_file:
                        os.makedirs(out_dir, exist_ok=True)
                        os.rename(src, train_file)
                if f == "test_objectdataset_augmentedrot_scale75.h5":
                    src = os.path.join(root, f)
                    if src != test_file:
                        os.makedirs(out_dir, exist_ok=True)
                        os.rename(src, test_file)

        if os.path.exists(train_file) and os.path.exists(test_file):
            print("[ScanObjectNN] Done (from HuggingFace, relocated)")
            return True
    except Exception as e:
        print(f"[ScanObjectNN] HuggingFace failed: {e}")
        if os.path.exists(hf_zip):
            os.remove(hf_zip)

    # ── Method 2: Google Drive via gdown ──────────────────────────────
    print("[ScanObjectNN] Trying Google Drive via gdown ...")
    try:
        import gdown
        tar_path = os.path.join(DATA_ROOT, "ScanObjectNN.tar")
        gdown.download(id=SCANOBJ_GDRIVE_ID, output=tar_path, quiet=False)

        print("[ScanObjectNN] Extracting ...")
        with tarfile.open(tar_path, 'r') as tf:
            tf.extractall(DATA_ROOT)
        os.remove(tar_path)

        if os.path.exists(train_file) and os.path.exists(test_file):
            print("[ScanObjectNN] Done (from Google Drive)")
            return True
    except Exception as e:
        print(f"[ScanObjectNN] gdown failed: {e}")

    # ── Method 3: Manual instructions ─────────────────────────────────
    print()
    print("=" * 60)
    print("  ScanObjectNN auto-download failed")
    print("=" * 60)
    print()
    print("  Option 1: Download manually from the official site")
    print("    1. Visit: https://hkust-vgd.github.io/scanobjectnn/")
    print("    2. Fill the license form to get the download link")
    print("    3. Download h5_files.zip")
    print("    4. Extract main_split/ to:")
    print(f"       {out_dir}/")
    print()
    print("  Option 2: Try the direct link")
    print(f"    wget {SCANOBJ_HF_URL}")
    print(f"    unzip scanobjectnn_PB_T50_RS_h5.zip -d {os.path.join(DATA_ROOT, 'ScanObjectNN')}")
    print()
    return False


# ═══════════════════════════════════════════════════════════════════════
#  S3DIS — Google Drive (OpenPoints preprocessed) or manual
# ═══════════════════════════════════════════════════════════════════════

S3DIS_GDRIVE_ID = "1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y"


def download_s3dis():
    """
    S3DIS preprocessed per-room .npy files.
    Each file: [N_points, 7] = x, y, z, r, g, b, semantic_label

    Primary: gdown from Google Drive (OpenPoints format)
    Fallback: Manual download from Stanford + preprocessing
    """
    out_dir = os.path.join(DATA_ROOT, "s3dis")
    sentinel = os.path.join(out_dir, "Area_5")

    if os.path.isdir(sentinel):
        import glob
        npys = glob.glob(os.path.join(sentinel, "*.npy"))
        if len(npys) > 0:
            print(f"[S3DIS] Already present at {out_dir} ({len(npys)} rooms in Area_5)")
            return True

    os.makedirs(out_dir, exist_ok=True)

    # ── Method 1: gdown from Google Drive ─────────────────────────────
    print("[S3DIS] Downloading preprocessed data via gdown ...")
    try:
        import gdown
        zip_path = os.path.join(DATA_ROOT, "s3dis_processed.zip")
        gdown.download(id=S3DIS_GDRIVE_ID, output=zip_path, quiet=False)

        print("[S3DIS] Extracting (this may take a few minutes) ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
        os.remove(zip_path)

        # Verify structure — look for Area_* directories
        found_areas = [d for d in os.listdir(out_dir)
                       if d.startswith("Area_") and os.path.isdir(os.path.join(out_dir, d))]
        if len(found_areas) >= 6:
            print(f"[S3DIS] Done: {len(found_areas)} areas")
            return True

        # If extracted into a subdirectory, relocate
        for item in os.listdir(out_dir):
            sub = os.path.join(out_dir, item)
            if os.path.isdir(sub) and item not in found_areas:
                for child in os.listdir(sub):
                    if child.startswith("Area_"):
                        os.rename(os.path.join(sub, child),
                                  os.path.join(out_dir, child))
                        found_areas.append(child)

        if len(found_areas) >= 6:
            print(f"[S3DIS] Done: {len(found_areas)} areas (relocated)")
            return True

    except ImportError:
        print("[S3DIS] gdown not installed. Install with: pip install gdown")
    except Exception as e:
        print(f"[S3DIS] gdown failed: {e}")

    # ── Method 2: Manual instructions ─────────────────────────────────
    print()
    print("=" * 60)
    print("  S3DIS auto-download failed")
    print("=" * 60)
    print()
    print("  Option 1: Install gdown and retry")
    print("    pip install gdown")
    print("    python datasets/download.py --s3dis")
    print()
    print("  Option 2: Download raw S3DIS + preprocess")
    print("    1. Get Stanford3dDataset_v1.2_Aligned_Version.zip from:")
    print("       http://buildingparser.stanford.edu/dataset.html")
    print("    2. Extract it")
    print("    3. Run: python datasets/download.py --s3dis_preprocess /path/to/Stanford3dDataset_v1.2_Aligned_Version")
    print()
    print("  Option 3: Download OpenPoints preprocessed S3DIS")
    print(f"    gdown --id {S3DIS_GDRIVE_ID} -O data/s3dis_processed.zip")
    print(f"    unzip data/s3dis_processed.zip -d data/s3dis/")
    print()
    return False


def preprocess_s3dis_raw(raw_dir: str):
    """
    Preprocess raw Stanford S3DIS into per-room .npy files.

    Raw structure:
        Stanford3dDataset_v1.2_Aligned_Version/Area_N/room_name/Annotations/*.txt

    Output:
        data/s3dis/Area_N/room_name.npy  — [N_points, 7]: x,y,z,r,g,b,label
    """
    import numpy as np

    out_dir = os.path.join(DATA_ROOT, "s3dis")
    os.makedirs(out_dir, exist_ok=True)

    CLASS_MAP = {name: i for i, name in enumerate([
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window',
        'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter',
    ])}

    total_rooms = 0
    for area_idx in range(1, 7):
        area_name = f"Area_{area_idx}"
        area_raw = os.path.join(raw_dir, area_name)
        area_out = os.path.join(out_dir, area_name)
        os.makedirs(area_out, exist_ok=True)

        if not os.path.isdir(area_raw):
            print(f"[Preprocess] Skipping {area_name} (not found in {raw_dir})")
            continue

        rooms = sorted([d for d in os.listdir(area_raw)
                        if os.path.isdir(os.path.join(area_raw, d))])

        area_count = 0
        for room_name in rooms:
            anno_dir = os.path.join(area_raw, room_name, "Annotations")
            if not os.path.isdir(anno_dir):
                continue

            room_pts = []
            for anno_file in sorted(os.listdir(anno_dir)):
                if not anno_file.endswith('.txt'):
                    continue
                class_name = '_'.join(anno_file.split('_')[:-1])
                label = CLASS_MAP.get(class_name, CLASS_MAP['clutter'])

                fpath = os.path.join(anno_dir, anno_file)
                try:
                    pts = np.loadtxt(fpath)
                except Exception:
                    continue
                if pts.ndim == 1:
                    pts = pts.reshape(1, -1)
                if pts.shape[1] < 6:
                    continue

                labels_col = np.full((len(pts), 1), label, dtype=np.float32)
                room_pts.append(
                    np.concatenate([pts[:, :6].astype(np.float32), labels_col], axis=1)
                )

            if room_pts:
                room_data = np.concatenate(room_pts, axis=0)
                np.save(os.path.join(area_out, f"{room_name}.npy"), room_data)
                area_count += 1

        total_rooms += area_count
        print(f"[Preprocess] {area_name}: {area_count} rooms")

    print(f"[Preprocess] Done: {total_rooms} rooms total in {out_dir}")


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _download_with_progress(url: str, dest: str, timeout: int = 60):
    """Download with progress bar and connection timeout."""
    import socket
    socket.setdefaulttimeout(timeout)
    try:
        from tqdm import tqdm

        class _Hook(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with _Hook(unit='B', unit_scale=True, miniters=1,
                   desc=os.path.basename(dest)) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
    except ImportError:
        print(f"  Downloading {os.path.basename(dest)} (no progress bar) ...")
        urllib.request.urlretrieve(url, dest)
    finally:
        socket.setdefaulttimeout(None)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Download ASP-SNN datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python datasets/download.py --all
    python datasets/download.py --scanobj
    python datasets/download.py --shapenet_kaggle
    python datasets/download.py --shapenet_raw /data/PartAnnotation
    python datasets/download.py --s3dis_preprocess /data/Stanford3dDataset_v1.2_Aligned_Version
        """,
    )
    p.add_argument("--all", action="store_true",
                   help="Download all three datasets")
    p.add_argument("--shapenet", action="store_true",
                   help="Download ShapeNetPart HDF5")
    p.add_argument("--shapenet_kaggle", action="store_true",
                   help="Download ShapeNetPart from Kaggle and convert to HDF5")
    p.add_argument("--shapenet_raw", type=str, default=None,
                   help="Convert local raw ShapeNetPart PartAnnotation to HDF5")
    p.add_argument("--scanobj", action="store_true",
                   help="Download ScanObjectNN PB_T50_RS")
    p.add_argument("--s3dis", action="store_true",
                   help="Download S3DIS preprocessed")
    p.add_argument("--s3dis_preprocess", type=str, default=None,
                   help="Preprocess raw S3DIS from Stanford directory")
    args = p.parse_args()

    if args.s3dis_preprocess:
        import numpy as np
        preprocess_s3dis_raw(args.s3dis_preprocess)
        return

    if args.shapenet_raw:
        ok = prepare_shapenet_raw(args.shapenet_raw)
        print()
        print("=" * 60)
        print("  Download Summary")
        print("=" * 60)
        print(f"  {'ShapeNetPart':<15} {'READY' if ok else 'NEEDS ATTENTION'}")
        print("=" * 60)
        return

    results = {}

    if args.all or args.shapenet:
        results['ShapeNetPart'] = download_shapenet()

    if args.shapenet_kaggle:
        results['ShapeNetPart'] = download_shapenet_kaggle()

    if args.all or args.scanobj:
        results['ScanObjectNN'] = download_scanobjectnn()

    if args.all or args.s3dis:
        results['S3DIS'] = download_s3dis()

    if not any([
        args.all,
        args.shapenet,
        args.shapenet_kaggle,
        args.scanobj,
        args.s3dis,
    ]):
        p.print_help()
        return

    # Summary
    print()
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "READY" if ok else "NEEDS ATTENTION"
        print(f"  {name:<15} {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
