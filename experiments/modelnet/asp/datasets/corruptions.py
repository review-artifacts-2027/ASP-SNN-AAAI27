"""ModelNet40-C-style corruptions (proposal Sec. 8.2) for points and patches.

Every function: f(x, severity in 1..5, torch.Generator) -> corrupted x.
Point clouds are resampled back to N points so downstream slicing is unchanged.
"""
from __future__ import annotations

import torch


def _resample(pts: torch.Tensor, n: int, g: torch.Generator) -> torch.Tensor:
    if pts.shape[0] == n:
        return pts
    if pts.shape[0] > n:
        idx = torch.randperm(pts.shape[0], generator=g)[:n]
    else:
        idx = torch.randint(0, pts.shape[0], (n,), generator=g)
    return pts[idx]


# ------------------------------------------------------------- point clouds
def gaussian_noise(pts, severity, g):
    sig = [0.01, 0.02, 0.03, 0.04, 0.05][severity - 1]
    return pts + torch.randn(pts.shape, generator=g) * sig


def uniform_outliers(pts, severity, g):
    frac = [0.02, 0.05, 0.08, 0.12, 0.16][severity - 1]
    n_out = max(1, int(frac * pts.shape[0]))
    out = torch.rand(n_out, 3, generator=g) * 2 - 1
    keep = pts[torch.randperm(pts.shape[0], generator=g)[: pts.shape[0] - n_out]]
    return torch.cat([keep, out])


def dropout(pts, severity, g):
    p = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    n_keep = max(8, int((1 - p) * pts.shape[0]))
    keep = pts[torch.randperm(pts.shape[0], generator=g)[:n_keep]]
    return _resample(keep, pts.shape[0], g)


def occlusion_halfspace(pts, severity, g):
    """Remove the deepest fraction along a random direction (self-occlusion)."""
    frac = [0.15, 0.25, 0.35, 0.45, 0.55][severity - 1]
    d = torch.randn(3, generator=g)
    d = d / d.norm()
    proj = pts @ d
    thresh = torch.quantile(proj, frac)
    keep = pts[proj >= thresh]
    return _resample(keep, pts.shape[0], g)


def clutter(pts, severity, g):
    n_cl = [8, 16, 24, 32, 48][severity - 1]
    centers = torch.rand(3, 3, generator=g) * 2 - 1
    blob = (centers.repeat_interleave(n_cl // 3 + 1, 0)[:n_cl]
            + torch.randn(n_cl, 3, generator=g) * 0.05)
    keep = pts[torch.randperm(pts.shape[0], generator=g)[: pts.shape[0] - n_cl]]
    return torch.cat([keep, blob])


def rotation_jitter(pts, severity, g):
    ang = [2, 5, 10, 15, 25][severity - 1] * 3.14159 / 180
    a = (torch.rand(3, generator=g) * 2 - 1) * ang
    cx, sx = torch.cos(a[0]), torch.sin(a[0])
    cy, sy = torch.cos(a[1]), torch.sin(a[1])
    cz, sz = torch.cos(a[2]), torch.sin(a[2])
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return pts @ (Rz @ Ry @ Rx).t()


def density_gradient(pts, severity, g):
    """S2: non-uniform sensor-like density along a random axis."""
    k = [1.0, 2.0, 3.0, 4.0, 6.0][severity - 1]
    d = torch.randn(3, generator=g); d = d / d.norm()
    proj = pts @ d
    z = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    keep_prob = torch.sigmoid(k * (0.5 - z) * 4)
    keep = pts[torch.rand(pts.shape[0], generator=g) < keep_prob]
    if keep.shape[0] < 8:
        keep = pts[:8]
    return _resample(keep, pts.shape[0], g)


POINT_CORRUPTIONS = {"gaussian_noise": gaussian_noise,
                     "uniform_outliers": uniform_outliers,
                     "dropout": dropout,
                     "occlusion": occlusion_halfspace,
                     "clutter": clutter,
                     "rotation_jitter": rotation_jitter,
                     "density_gradient": density_gradient}


# ----------------------------------------------------------------- patches
def patch_gaussian_noise(patches, severity, g):
    sig = [0.04, 0.08, 0.12, 0.16, 0.2][severity - 1]
    return patches + torch.randn(patches.shape, generator=g) * sig


def patch_occlusion(patches, severity, g):
    """Zero out entire random patches (spatially structured occlusion)."""
    n = [2, 4, 6, 8, 10][severity - 1]
    out = patches.clone()
    idx = torch.randperm(patches.shape[0], generator=g)[:n]
    out[idx] = 0.0
    return out


def patch_contrast(patches, severity, g):
    c = [0.75, 0.6, 0.45, 0.3, 0.2][severity - 1]
    return patches * c


PATCH_CORRUPTIONS = {"gaussian_noise": patch_gaussian_noise,
                     "occlusion": patch_occlusion,
                     "contrast": patch_contrast}
