"""CIFAR-10/100 adapted to the SNN slice pipeline (patch-slice mode).

Adaptation (proposal Sec. 7.2, K matched to the SNN's 16 slices):
    32x32 image -> 4x4 grid of 8x8 patches = K=16 "slices".
    Each patch is a flattened 8*8*3 = 192-D region processed by the spiking
    PatchSliceEncoder; the SSP sees a 6-D descriptor analog per patch:

        [0] row (normalized, centered)          ~ centroid x
        [1] col (normalized, centered)          ~ centroid y
        [2] distance to image center            ~ dist-to-centroid
        [3] intensity std within patch          ~ intra-cluster spread
        [4] edge density (mean |grad|)          ~ occupancy / point count
        [5] mean intensity                      ~ centroid z analog

    This preserves the SSP interface exactly: same 6-D descriptor slot, same
    masking, same Gumbel/argmax selection, same LIF head and early exit.

Requires torchvision (auto-downloads CIFAR to <root>).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def image_to_patch_slices(img: torch.Tensor, grid: int = 4):
    """img: (3,32,32) normalized -> patches (K,192), desc (K,6), anchors (K,3)."""
    C, H, W = img.shape
    ph, pw = H // grid, W // grid
    patches, desc, anchors = [], [], []
    gray = img.mean(0)                                     # (H,W)
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    for r in range(grid):
        for c in range(grid):
            pat = img[:, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            patches.append(pat.reshape(-1))
            row = (r + 0.5) / grid * 2 - 1
            col = (c + 0.5) / grid * 2 - 1
            dist = (row ** 2 + col ** 2) ** 0.5
            spread = pat.std().item()
            eg = gx[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw - 1].abs().mean() \
                + gy[r * ph:(r + 1) * ph - 1, c * pw:(c + 1) * pw].abs().mean()
            desc.append([row, col, dist, spread, eg.item() / 2, pat.mean().item()])
            anchors.append([row, col, 0.0])
    return (torch.stack(patches), torch.tensor(desc, dtype=torch.float32),
            torch.tensor(anchors, dtype=torch.float32))


class CIFARPatchDataset(Dataset):
    def __init__(self, root: str, variant: int = 10, split: str = "train",
                 grid: int = 4, corruption=None, severity: int = 0,
                 limit: int | None = None):
        import torchvision
        cls = torchvision.datasets.CIFAR10 if variant == 10 \
            else torchvision.datasets.CIFAR100
        self.ds = cls(root, train=(split == "train"), download=True)
        self.grid = grid
        self.num_classes = variant
        self.corruption, self.severity = corruption, severity
        self.limit = limit
        self.augment = split == "train"

    def __len__(self):
        return min(len(self.ds), self.limit) if self.limit else len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]
        x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
        g = torch.Generator().manual_seed(i * 2741 + 7)
        if self.augment:
            if torch.rand(1, generator=g).item() < 0.5:
                x = x.flip(-1)                              # horizontal flip
            pad = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
            dx = torch.randint(0, 9, (2,), generator=g)
            x = pad[:, dx[0]:dx[0] + 32, dx[1]:dx[1] + 32]  # random crop
        x = (x - MEAN) / STD
        patches, desc, anchors = image_to_patch_slices(x, self.grid)
        if self.corruption is not None and self.severity > 0:
            patches = self.corruption(patches, self.severity, g)
        return patches, desc, anchors, label
