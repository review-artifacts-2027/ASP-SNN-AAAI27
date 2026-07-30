from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lif import spike_fn


class PointSliceEncoder(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 64, sg_slope: float = 4.0):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.sg_slope = sg_slope
        self.hidden, self.d_model = hidden, d_model
        self.last_firing_rate = torch.tensor(0.0)

    def forward(self, slices: torch.Tensor, anchors_xyz: torch.Tensor) -> torch.Tensor:
        B, K, P, _ = slices.shape
        rel = slices - anchors_xyz.unsqueeze(2)
        x = rel.reshape(B * K * P, 3)
        s1 = spike_fn(self.bn1(self.fc1(x)) - 0.0, self.sg_slope)
        h = self.bn2(self.fc2(s1))
        s2 = spike_fn(h, self.sg_slope)
        self.last_firing_rate = 0.5 * (s1.mean() + s2.mean())
        feat = (s2 * h).reshape(B, K, P, self.d_model)
        return feat.amax(dim=2)


class PatchSliceEncoder(nn.Module):
    def __init__(self, patch_dim: int, d_model: int = 128, hidden: int = 128,
                 sg_slope: float = 4.0):
        super().__init__()
        self.fc1 = nn.Linear(patch_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.sg_slope = sg_slope
        self.d_model = d_model
        self.last_firing_rate = torch.tensor(0.0)

    def forward(self, patches: torch.Tensor, anchors_xyz=None) -> torch.Tensor:
        B, K, F_ = patches.shape
        x = patches.reshape(B * K, F_)
        s1 = spike_fn(self.bn1(self.fc1(x)), self.sg_slope)
        h = self.bn2(self.fc2(s1))
        s2 = spike_fn(h, self.sg_slope)
        self.last_firing_rate = 0.5 * (s1.mean() + s2.mean())
        return (s2 * h).reshape(B, K, self.d_model)


class SpikingConvPatchStem(nn.Module):
    def __init__(self, patch_dim: int = 192, k_slices: int = 16,
                 d_model: int = 128,
                 stage_channels=(48, 96, 128, 128),
                 in_channels: int = 3, sg_slope: float = 4.0):
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError(
                f"stage_channels must have length 4, got {len(stage_channels)}"
            )
        if stage_channels[-1] != d_model:
            raise ValueError(
                f"final stage channels {stage_channels[-1]} must equal "
                f"d_model={d_model}."
            )
        g = int(math.isqrt(k_slices))
        if g * g != k_slices:
            raise ValueError(
                f"cannot infer square grid from k_slices={k_slices}"
            )
        area = patch_dim // in_channels
        p = int(math.isqrt(area))
        if p * p * in_channels != patch_dim:
            raise ValueError(
                f"cannot infer square patch shape from patch_dim={patch_dim}, "
                f"in_channels={in_channels}"
            )
        self.patch_dim   = patch_dim
        self.k_slices    = k_slices
        self.grid_hw     = (g, g)
        self.patch_hw    = (p, p)
        self.in_channels = in_channels
        self.d_model     = d_model
        self.sg_slope    = sg_slope
        self.stage_channels = tuple(int(c) for c in stage_channels)

        c0 = in_channels
        c1, c2, c3, c4 = self.stage_channels
        self.conv1 = nn.Conv2d(c0, c1, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(c1); self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(c2); self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(c3); self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1, bias=False); self.bn4 = nn.BatchNorm2d(c4)
        self.last_firing_rate = torch.tensor(0.0)

    def _patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        B, K, D = patches.shape
        C, ph, pw = self.in_channels, self.patch_hw[0], self.patch_hw[1]
        gh, gw = self.grid_hw
        p = patches.view(B, gh, gw, C, ph, pw)
        p = p.permute(0, 3, 1, 4, 2, 5).contiguous()
        return p.view(B, C, gh * ph, gw * pw)

    def forward(self, patches: torch.Tensor, anchors_xyz=None) -> torch.Tensor:
        img = self._patches_to_image(patches)
        h = self.bn1(self.conv1(img)); s1 = spike_fn(h, self.sg_slope); h = self.pool1(s1)
        h = self.bn2(self.conv2(h));   s2 = spike_fn(h, self.sg_slope); h = self.pool2(s2)
        h = self.bn3(self.conv3(h));   s3 = spike_fn(h, self.sg_slope); h = self.pool3(s3)
        h_pre = self.bn4(self.conv4(h)); s4 = spike_fn(h_pre, self.sg_slope); gated = s4 * h_pre
        self.last_firing_rate = 0.25 * (s1.mean() + s2.mean() + s3.mean() + s4.mean())
        B_, C_, H_, W_ = gated.shape
        return gated.permute(0, 2, 3, 1).contiguous().view(B_, H_ * W_, C_)


class SpikingPerPatchEncoder(nn.Module):
    def __init__(self, patch_dim: int = 192, k_slices: int = 16,
                 d_model: int = 128,
                 stage_channels=(32, 64, 128),
                 in_channels: int = 3, sg_slope: float = 4.0):
        super().__init__()
        if len(stage_channels) != 3:
            raise ValueError(
                f"SpikingPerPatchEncoder expects 3 stages, got {len(stage_channels)}"
            )
        if stage_channels[-1] != d_model:
            raise ValueError(
                f"final stage channels {stage_channels[-1]} must equal "
                f"d_model={d_model}. Adjust `perpatch_stage_channels` or `d_model`."
            )
        area = patch_dim // in_channels
        p = int(math.isqrt(area))
        if p * p * in_channels != patch_dim:
            raise ValueError(
                f"cannot infer square patch shape from patch_dim={patch_dim}, "
                f"in_channels={in_channels}"
            )

        if p % 4 != 0 and p >= 4:
            raise ValueError(
                f"SpikingPerPatchEncoder needs patch_hw divisible by 4 for the "
                f"two MaxPool2 stages; got p={p}."
            )

        self.patch_dim   = patch_dim
        self.k_slices    = k_slices
        self.patch_hw    = (p, p)
        self.in_channels = in_channels
        self.d_model     = d_model
        self.sg_slope    = sg_slope
        self.stage_channels = tuple(int(c) for c in stage_channels)

        c0 = in_channels
        c1, c2, c3 = self.stage_channels

        self.conv1 = nn.Conv2d(c0, c1, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(c3)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.last_firing_rate = torch.tensor(0.0)

    def _encode_batched(self, x_flat: torch.Tensor) -> torch.Tensor:
        N, D = x_flat.shape
        x = x_flat.view(N, self.in_channels,
                        self.patch_hw[0], self.patch_hw[1])

        h  = self.bn1(self.conv1(x))
        s1 = spike_fn(h, self.sg_slope)
        h  = self.pool1(s1)

        h  = self.bn2(self.conv2(h))
        s2 = spike_fn(h, self.sg_slope)
        h  = self.pool2(s2)

        h_pre = self.bn3(self.conv3(h))
        s3    = spike_fn(h_pre, self.sg_slope)
        gated = s3 * h_pre

        self.last_firing_rate = (s1.mean() + s2.mean() + s3.mean()) / 3.0

        out = self.gap(gated).flatten(1)
        return out

    def forward(self, patches: torch.Tensor, anchors_xyz=None) -> torch.Tensor:
        B, K, D = patches.shape
        e = self._encode_batched(patches.reshape(B * K, D))
        return e.reshape(B, K, self.d_model)

    def encode_selected(self, patches: torch.Tensor,
                        indices: torch.Tensor) -> torch.Tensor:

        B = patches.shape[0]

        gather_idx = torch.arange(B, device=patches.device)
        selected = patches[gather_idx, indices]
        return self._encode_batched(selected)
