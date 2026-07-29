"""
imagenet.py
===========
ImageNet/ImageNet-100 dataloaders for the FoveaTer ASP runner.

Expected layout:

    DATA_ROOT/
      train/
        class_a/*.JPEG
        class_b/*.JPEG
      val/
        class_a/*.JPEG
        class_b/*.JPEG

ImageNet-100 works with the same layout, just with 100 class folders.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_torchvision():
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - import guard for environments
        raise ImportError(
            "torchvision is required for ImageNet datasets and transforms"
        ) from exc
    return datasets, transforms


def imagenet_transform(
    train: bool,
    image_size: int = 224,
    resize_size: int = 256,
):
    _, transforms = _require_torchvision()

    if train:
        ops = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
        if hasattr(transforms, "RandAugment"):
            ops.append(transforms.RandAugment())
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _split_path(root: str | Path, split: str) -> Path:
    root = Path(root)
    candidates = [root / split]
    if split == "val":
        candidates.append(root / "validation")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find ImageNet split '{split}' under {root}. "
        "Expected train/ and val/ class-folder directories."
    )


def build_imagenet_dataset(
    root: str | Path,
    split: str,
    image_size: int = 224,
    resize_size: int = 256,
):
    datasets, _ = _require_torchvision()

    split_dir = _split_path(root, split)
    return datasets.ImageFolder(
        split_dir,
        transform=imagenet_transform(
            train=(split == "train"),
            image_size=image_size,
            resize_size=resize_size,
        ),
    )


def build_imagenet_loaders(
    root: str | Path,
    batch_size: int = 128,
    workers: int = 8,
    image_size: int = 224,
    resize_size: int = 256,
    pin_memory: bool = True,
):
    train_ds = build_imagenet_dataset(
        root, "train", image_size=image_size, resize_size=resize_size
    )
    val_ds = build_imagenet_dataset(
        root, "val", image_size=image_size, resize_size=resize_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, len(train_ds.classes), train_ds.classes
