"""
models/encoder.py — EdgeConv feature extractor (analog + spiking variants).

Two variants controlled by `cfg.encoder_type`:

    'analog'   (default, backward compatible):
        Standard EdgeConv with ReLU. This is the original ASP-SNN encoder.
        Runs in ~1 forward pass. Pays E_MAC per operation.

    'spiking':
        Replaces every ReLU with a LIF neuron. Whole encoder runs for
        T_enc timesteps per sample and spike outputs are summed over
        time then normalized by T_enc to keep the output range comparable
        to the analog version. Payload downstream (transformer, SSP,
        ASP loop, classification head) is IDENTICAL — same interface,
        same output shape.

        Design follows the canonical direct-training approach used by:
          - Spiking PointNet (Ren et al., NeurIPS 2023)
          - Spiking PointCNN (Wu et al., MDPI 2024)
          - SpikingRTNH (Paek & Kong, 2025)
          - Spike-driven Transformer family (Yao et al.)

        The purpose is to shift the encoder from MAC-dominated compute to
        AC-dominated compute (SOPs = firing_rate * T_enc * FLOPs).
        Reference: Horowitz ISSCC 2014 for E_MAC=4.6pJ, E_AC=0.9pJ (45nm).

Takes [B, M, K, C] slices and produces [B, M, feat_dim] tokens either way.
Uses static kNN on xyz coordinates — never recomputed in feature space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lif import _spike  # ATan surrogate, AMP-safe, shared with LIF head


def knn_xyz(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Static k-NN on xyz.  xyz: [BM, N, 3] -> idx: [BM, N, k]."""
    BM, N, _ = xyz.shape
    dist = torch.cdist(xyz, xyz)
    diag = torch.eye(N, device=xyz.device, dtype=xyz.dtype).unsqueeze(0) * 1e9
    _, idx = (dist + diag).topk(k, dim=-1, largest=False)
    return idx


def build_edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x: [BM,N,C], idx: [BM,N,k] -> [BM, 2C, N, k] for Conv2d."""
    BM, N, C = x.shape
    k = idx.shape[-1]
    bm_idx = torch.arange(BM, device=x.device).view(BM, 1, 1).expand(BM, N, k)
    nbrs = x[bm_idx, idx]                              # [BM, N, k, C]
    x_ctr = x.unsqueeze(2).expand(BM, N, k, C)         # [BM, N, k, C]
    edge = torch.cat([x_ctr, nbrs - x_ctr], dim=-1)    # [BM, N, k, 2C]
    return edge.permute(0, 3, 1, 2).contiguous()        # [BM, 2C, N, k]


# ══════════════════════════════════════════════════════════════════════════
#  Analog EdgeConv encoder (original — unchanged behavior)
# ══════════════════════════════════════════════════════════════════════════

class AnalogEdgeConvEncoder(nn.Module):
    """
    Single static EdgeConv + Conv1d widening + global max-pool.
    ORIGINAL ARCHITECTURE. Not modified.

    Input  : slices [B, M, K, C]  (C >= 3, first 3 = xyz)
    Output : feats  [B, M, feat_dim]
    """

    def __init__(self, feat_dim: int = 512, k_edge: int = 20,
                 in_channels: int = 6):
        super().__init__()
        self.k = k_edge
        self.in_channels = in_channels

        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv1d(128, 256, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, feat_dim, 1, bias=False)
        self.bn3   = nn.BatchNorm1d(feat_dim)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        B, M, K, C = slices.shape
        BM = B * M
        x   = slices.reshape(BM, K, C)
        xyz = x[:, :, :3]

        centroid = xyz.mean(dim=1, keepdim=True)
        rel_xyz  = xyz - centroid                        # [BM, K, 3]

        if C >= self.in_channels:
            extra   = x[:, :, 3:self.in_channels]
            feat_in = torch.cat([rel_xyz, extra], dim=-1)
        elif C > 3:
            extra   = x[:, :, 3:]
            pad_dim = self.in_channels - C
            pad     = torch.zeros(BM, K, pad_dim, device=x.device, dtype=x.dtype)
            feat_in = torch.cat([rel_xyz, extra, pad], dim=-1)
        else:
            pad_dim = self.in_channels - 3
            if pad_dim > 0:
                pad     = torch.zeros(BM, K, pad_dim, device=x.device, dtype=x.dtype)
                feat_in = torch.cat([rel_xyz, pad], dim=-1)
            else:
                feat_in = rel_xyz

        idx  = knn_xyz(xyz, self.k)
        edge = build_edge_features(feat_in, idx)         # [BM, 2*in_ch, K, k]
        feat = self.edge_conv(edge)                      # [BM, 128, K, k]
        feat = feat.max(dim=-1).values                   # [BM, 128, K]
        feat = self.relu(self.bn2(self.conv2(feat)))     # [BM, 256, K]
        feat = self.relu(self.bn3(self.conv3(feat)))     # [BM, feat_dim, K]
        feat = feat.max(dim=-1).values                   # [BM, feat_dim]
        return feat.view(B, M, -1)


# ══════════════════════════════════════════════════════════════════════════
#  Spiking EdgeConv encoder — Tier 2
# ══════════════════════════════════════════════════════════════════════════

class _EncoderLIF(nn.Module):
    """
    Stateless-in-init LIF wrapper for encoder use.

    Unlike the head's LIFCell (which is scalar-timesteps for the ASP loop),
    encoder LIF operates on Conv2d/Conv1d feature maps [B, C, ...] and
    holds its (u, s) state internally between the T_enc timesteps of ONE
    forward pass. Reset via .reset_state() before each new forward.

    Same dynamics as the head LIF:
        u_t = leak * u_{t-1} + inp - threshold * s_{t-1}
        s_t = spike(u_t - threshold)  (ATan surrogate, AMP-safe)
    """

    def __init__(self, leak: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.leak = leak
        self.threshold = threshold
        self.u = None
        self.s = None

    def reset_state(self):
        self.u = None
        self.s = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """inp: any shape [B, ...]. Returns binary spike tensor same shape."""
        if self.u is None:
            self.u = torch.zeros_like(inp)
            self.s = torch.zeros_like(inp)
        u = self.leak * self.u + inp - self.threshold * self.s
        s = _spike(u - self.threshold)
        self.u = u
        self.s = s
        return s

    @property
    def spike_out(self):
        return self.s


class SpikingEdgeConvEncoder(nn.Module):
    """
    Spiking version of the EdgeConv encoder. Every ReLU is replaced by an
    _EncoderLIF neuron. The whole encoder runs for T_enc timesteps per
    sample, and the outputs are averaged over time to yield a real-valued
    [B, M, feat_dim] tensor compatible with the downstream transformer.

    Why average, not sum:
      Averaging keeps the output range comparable to the analog encoder's
      max-pool (both are O(1) magnitudes), so the pretrained transformer
      / SSP / classifier heads don't need retuning. Summing would grow
      linearly with T_enc and destabilize training.

    Why static input encoding:
      The input geometry is a static point cloud (not a temporal signal),
      so following the direct-training convention we feed the same edge
      features at every timestep. The LIF neurons produce a temporal
      spike code from this static input. Reference: Ren et al. Spiking
      PointNet (NeurIPS 2023), Section 3.2.

    The pipeline preserves EVERY structural choice of the analog encoder
    (2C->128->128 EdgeConv, 128->256->feat_dim widening, kNN, edge
    features, both max-pools). Only the ReLUs are swapped.
    """

    def __init__(self, feat_dim: int = 512, k_edge: int = 20,
                 in_channels: int = 6, T_enc: int = 4,
                 lif_leak: float = 0.9, lif_threshold: float = 1.0):
        super().__init__()
        self.k = k_edge
        self.in_channels = in_channels
        self.T_enc = T_enc

        # Two Conv2d + BN blocks (edge conv)
        self.conv1 = nn.Conv2d(in_channels * 2, 128, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(128)
        self.lif1  = _EncoderLIF(lif_leak, lif_threshold)

        self.conv2_edge = nn.Conv2d(128, 128, 1, bias=False)
        self.bn2_edge   = nn.BatchNorm2d(128)
        self.lif2_edge  = _EncoderLIF(lif_leak, lif_threshold)

        # Two Conv1d + BN blocks (widening)
        self.conv2 = nn.Conv1d(128, 256, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(256)
        self.lif3  = _EncoderLIF(lif_leak, lif_threshold)

        self.conv3 = nn.Conv1d(256, feat_dim, 1, bias=False)
        self.bn3   = nn.BatchNorm1d(feat_dim)
        self.lif4  = _EncoderLIF(lif_leak, lif_threshold)

    def _reset_all(self):
        for m in [self.lif1, self.lif2_edge, self.lif3, self.lif4]:
            m.reset_state()

    def _build_input(self, slices: torch.Tensor):
        """
        Prepare edge features & feat_in once (static across T_enc steps).
        Returns (edge, xyz) reused every timestep.
        """
        B, M, K, C = slices.shape
        BM = B * M
        x   = slices.reshape(BM, K, C)
        xyz = x[:, :, :3]

        centroid = xyz.mean(dim=1, keepdim=True)
        rel_xyz  = xyz - centroid

        if C >= self.in_channels:
            extra   = x[:, :, 3:self.in_channels]
            feat_in = torch.cat([rel_xyz, extra], dim=-1)
        elif C > 3:
            extra   = x[:, :, 3:]
            pad_dim = self.in_channels - C
            pad     = torch.zeros(BM, K, pad_dim, device=x.device, dtype=x.dtype)
            feat_in = torch.cat([rel_xyz, extra, pad], dim=-1)
        else:
            pad_dim = self.in_channels - 3
            if pad_dim > 0:
                pad     = torch.zeros(BM, K, pad_dim, device=x.device, dtype=x.dtype)
                feat_in = torch.cat([rel_xyz, pad], dim=-1)
            else:
                feat_in = rel_xyz

        idx  = knn_xyz(xyz, self.k)
        edge = build_edge_features(feat_in, idx)   # [BM, 2*in_ch, K, k]
        return edge, B, M, K, BM

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        """
        [B, M, K, C] -> [B, M, feat_dim].
        Runs for T_enc timesteps internally, averages spike outputs across time.
        """
        edge, B, M, K, BM = self._build_input(slices)
        self._reset_all()

        # Accumulator for time-averaged final features
        feat_accum = None

        for t in range(self.T_enc):
            # Edge conv block 1
            h = self.bn1(self.conv1(edge))                 # [BM, 128, K, k]
            h = self.lif1(h)                               # spikes
            h = self.bn2_edge(self.conv2_edge(h))          # [BM, 128, K, k]
            h = self.lif2_edge(h)                          # spikes

            # Max-pool across edges (k) — preserves original structure
            h = h.max(dim=-1).values                       # [BM, 128, K]

            # Widening blocks
            h = self.bn2(self.conv2(h))                    # [BM, 256, K]
            h = self.lif3(h)
            h = self.bn3(self.conv3(h))                    # [BM, feat_dim, K]
            h = self.lif4(h)

            # Global max-pool across K (matches analog encoder)
            h = h.max(dim=-1).values                       # [BM, feat_dim]

            if feat_accum is None:
                feat_accum = h
            else:
                feat_accum = feat_accum + h

        # Time-average keeps output range comparable to analog encoder
        feat = feat_accum / float(self.T_enc)              # [BM, feat_dim]
        return feat.view(B, M, -1)


# ══════════════════════════════════════════════════════════════════════════
#  Factory — selects encoder from cfg
# ══════════════════════════════════════════════════════════════════════════

class EdgeConvFeatureExtractor(nn.Module):
    """
    Backward-compatible factory. If cfg.encoder_type == 'spiking' it builds
    the spiking encoder, otherwise the original analog one.

    Kept as an nn.Module (not a function) so downstream code that does
    `list(model.feature_extractor.parameters())` continues to work.
    """

    def __init__(self, feat_dim: int = 512, k_edge: int = 20,
                 in_channels: int = 6, encoder_type: str = 'analog',
                 T_enc: int = 4, lif_leak: float = 0.9,
                 lif_threshold: float = 1.0):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'spiking':
            self.impl = SpikingEdgeConvEncoder(
                feat_dim=feat_dim, k_edge=k_edge,
                in_channels=in_channels, T_enc=T_enc,
                lif_leak=lif_leak, lif_threshold=lif_threshold,
            )
        else:
            self.impl = AnalogEdgeConvEncoder(
                feat_dim=feat_dim, k_edge=k_edge, in_channels=in_channels,
            )

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        return self.impl(slices)