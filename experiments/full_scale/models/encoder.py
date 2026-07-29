"""
models/encoder.py — EdgeConv feature extractor for per-slice encoding.

Takes [B, M, K, C] slices and produces [B, M, feat_dim] tokens.
Uses static kNN on xyz coordinates (first 3 channels) — never recomputed
in feature space.

Supports variable input channels:
    C=6  xyz + normals      (ShapeNetPart, ScanObjectNN)
    C=7  xyz + rgb + height (S3DIS)
    C=3  xyz only           (fallback — pads with zeros)
"""

import torch
import torch.nn as nn


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


class EdgeConvFeatureExtractor(nn.Module):
    """
    Single static EdgeConv + Conv1d widening + global max-pool.

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
