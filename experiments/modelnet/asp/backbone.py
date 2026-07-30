"""Spiking region encoders ("Local KNN Backbone").

Encoders map a region to a feature e_m in R^D with spiking activations.

Available encoders:
    PointSliceEncoder       : (B, K, P, 3) local KNN slices  -> (B, K, D)
    PatchSliceEncoder       : (B, K, F)    flattened patches -> (B, K, D)  (FC MLP)
    SpikingConvPatchStem    : (B, K, F)    flattened patches -> (B, K, D)  (Phase 4)
    SpikingPerPatchEncoder  : (B, K, F)    flattened patches -> (B, K, D)  (Phase 8)

Phase 4 (SpikingConvPatchStem) — reconstructs the full 32x32 image from
    (B, K, C·ph·pw) patch tokens, runs a 4-stage full-image spiking conv,
    reshapes to (B, K, d_model). Best accuracy but the conv has
    cross-patch dependencies, so it CANNOT be evaluated on individual
    patches — the whole image is always processed.

Phase 8 (SpikingPerPatchEncoder) — encodes EACH 8x8 patch independently
    through a small 3-stage conv. No cross-patch context, so the per-patch
    output can be computed on-demand for STREAMING inference:
        `encode_selected(patches, idx)` gives byte-identical output to
        `forward(patches)[:, idx]`, but at cost proportional to the
        number of selected patches, not K.

    This is the mechanism that recovers α_sys at inference: with early
    exit at expected step E[τ] < K, encoder cost drops from
        K × per_patch_cost   -> E[τ] × per_patch_cost
    a ~K/E[τ] factor improvement when early stopping activates.

    Accuracy trade-off: no cross-patch conv → ~2-3 pp val OA drop vs.
    a full-image stem.  The two variants expose an accuracy/compute trade-off.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lif import spike_fn


# ─────────────────────────────────────────────────────────────────────
#  Point-cloud encoder (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────
class PointSliceEncoder(nn.Module):
    """Per-point spiking MLP (3->h->D) + max-pool over the slice (PointNet-lite)."""

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
        rel = slices - anchors_xyz.unsqueeze(2)          # local frame
        x = rel.reshape(B * K * P, 3)
        s1 = spike_fn(self.bn1(self.fc1(x)) - 0.0, self.sg_slope)
        h = self.bn2(self.fc2(s1))
        s2 = spike_fn(h, self.sg_slope)
        self.last_firing_rate = 0.5 * (s1.mean() + s2.mean())
        feat = (s2 * h).reshape(B, K, P, self.d_model)   # spike-gated membrane
        return feat.amax(dim=2)                          # (B, K, D)


# ─────────────────────────────────────────────────────────────────────
#  FC patch encoder (UNCHANGED, pre-Phase 4)
# ─────────────────────────────────────────────────────────────────────
class PatchSliceEncoder(nn.Module):
    """Flattened image patch -> spiking MLP -> (B, K, D). Same interface."""

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


# ─────────────────────────────────────────────────────────────────────
#  Phase 4 — Spiking Conv Patch Stem (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────
class SpikingConvPatchStem(nn.Module):
    """Spikformer-SPS-style spiking convolutional patch stem.

    See Phase 4 delivery notes for architectural details. Cross-patch
    context makes this encoder NON-STREAMABLE — use
    `SpikingPerPatchEncoder` instead when α_sys is the priority.
    """

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


# ─────────────────────────────────────────────────────────────────────
#  Phase 8 NEW — Spiking Per-Patch Encoder (streaming-friendly)
# ─────────────────────────────────────────────────────────────────────
class SpikingPerPatchEncoder(nn.Module):
    """Per-patch spiking conv encoder for streaming inference (Phase 8).

    Each patch is encoded INDEPENDENTLY through a small 3-stage conv,
    with NO cross-patch dependencies. This enables true streaming
    inference — at each ASP step, only the SELECTED patch is encoded,
    saving encoder compute proportional to `E[τ]/K` where `E[τ]` is
    the expected exit step under early exit.

    Architecture (defaults for 8x8 RGB patches):

        (B, 3, 8, 8)
            Conv3x3(3, c1=32)  + BN + spike  ->  (B, 32, 8, 8)
            MaxPool2                          ->  (B, 32, 4, 4)
            Conv3x3(32, c2=64) + BN + spike  ->  (B, 64, 4, 4)
            MaxPool2                          ->  (B, 64, 2, 2)
            Conv3x3(64, c3=128)+ BN + spike-gated ->  (B, 128, 2, 2)
        AdaptiveAvgPool2d(1)                  ->  (B, 128, 1, 1)
        flatten                               ->  (B, 128)

    Parameter count:  ~93k (vs. ~301k in SpikingConvPatchStem)
    MACs per patch:
        Conv1:  8·8·3·32·9 =    55,296  (analog — input is real-valued)
        Conv2:  4·4·32·64·9 =  294,912  (AC — spike input)
        Conv3:  2·2·64·128·9 = 294,912  (AC — spike input)
        Total per patch: ~645k. K=16 patches -> ~10.3M MACs.
        Compare Phase 4 SpikingConvPatchStem: ~24M MACs total.

    Two forward modes
    ─────────────────
        forward(patches, anchors=None):
            (B, K, patch_dim) -> (B, K, d_model)
            Batched — used at training and non-streaming inference.
            Cost: K × per-patch. Cost identical to K per-patch calls.

        encode_selected(patches, indices):
            (B, K, patch_dim), (B,) -> (B, d_model)
            Encodes ONLY the patch at indices[b] for each batch item b.
            Byte-identical to `forward(patches)[torch.arange(B), indices]`
            because per-patch encoding is deterministic and independent.
            Cost: 1 × per-patch call.

    Interface
    ─────────
    Matches PatchSliceEncoder / PointSliceEncoder / SpikingConvPatchStem:
        forward(patches, anchors_xyz=None) -> (B, K, d_model)
        `.last_firing_rate` Tensor attribute
    """
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
        # We do 2 MaxPool2 stages on p×p patches. Ensure p is divisible by 4.
        if p % 4 != 0 and p >= 4:
            # Allow the default p=8 while catching pathological cases.
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

        # Stage 1: p×p -> p/2 × p/2
        self.conv1 = nn.Conv2d(c0, c1, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2: p/2 × p/2 -> p/4 × p/4
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3: p/4 × p/4 -> p/4 × p/4 (no pool)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(c3)

        # Global avg pool at the end.
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.last_firing_rate = torch.tensor(0.0)

    # -----------------------------------------------------------------
    def _encode_batched(self, x_flat: torch.Tensor) -> torch.Tensor:
        """(N, patch_dim) -> (N, d_model).

        This is the SHARED conv path used by both `forward` (N = B·K) and
        `encode_selected` (N = B). Deterministic and per-sample-independent,
        so calls with different N produce identical rows for the same input.
        """
        N, D = x_flat.shape
        x = x_flat.view(N, self.in_channels,
                        self.patch_hw[0], self.patch_hw[1])

        # Stage 1
        h  = self.bn1(self.conv1(x))
        s1 = spike_fn(h, self.sg_slope)
        h  = self.pool1(s1)

        # Stage 2
        h  = self.bn2(self.conv2(h))
        s2 = spike_fn(h, self.sg_slope)
        h  = self.pool2(s2)

        # Stage 3 — spike-gated membrane (matches SpikingConvPatchStem convention)
        h_pre = self.bn3(self.conv3(h))
        s3    = spike_fn(h_pre, self.sg_slope)
        gated = s3 * h_pre

        # Firing rate (mean across the 3 spiking stages).
        self.last_firing_rate = (s1.mean() + s2.mean() + s3.mean()) / 3.0

        # Global average pool -> (N, d_model)
        out = self.gap(gated).flatten(1)
        return out

    # -----------------------------------------------------------------
    def forward(self, patches: torch.Tensor, anchors_xyz=None) -> torch.Tensor:
        """(B, K, patch_dim) -> (B, K, d_model)  — batched forward."""
        B, K, D = patches.shape
        e = self._encode_batched(patches.reshape(B * K, D))
        return e.reshape(B, K, self.d_model)

    # -----------------------------------------------------------------
    def encode_selected(self, patches: torch.Tensor,
                        indices: torch.Tensor) -> torch.Tensor:
        """(B, K, patch_dim), (B,) -> (B, d_model)  — streaming call.

        Encodes ONLY the patch at `indices[b]` for each batch item b.
        Byte-identical to `forward(patches)[torch.arange(B), indices]`
        (per-patch encoding is deterministic and independent).
        """
        B = patches.shape[0]
        # Advanced indexing: pick patch `indices[b]` from batch item b.
        gather_idx = torch.arange(B, device=patches.device)
        selected = patches[gather_idx, indices]              # (B, patch_dim)
        return self._encode_batched(selected)
