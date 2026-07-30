import torch
import torch.nn as nn
import torch.nn.functional as F

from .lif import _spike


def knn_xyz(xyz: torch.Tensor, k: int) -> torch.Tensor:
    BM, N, _ = xyz.shape
    dist = torch.cdist(xyz, xyz)
    diag = torch.eye(N, device=xyz.device, dtype=xyz.dtype).unsqueeze(0) * 1e9
    _, idx = (dist + diag).topk(k, dim=-1, largest=False)
    return idx


def build_edge_features(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    BM, N, C = x.shape
    k = idx.shape[-1]
    bm_idx = torch.arange(BM, device=x.device).view(BM, 1, 1).expand(BM, N, k)
    nbrs = x[bm_idx, idx]
    x_ctr = x.unsqueeze(2).expand(BM, N, k, C)
    edge = torch.cat([x_ctr, nbrs - x_ctr], dim=-1)
    return edge.permute(0, 3, 1, 2).contiguous()


class AnalogEdgeConvEncoder(nn.Module):
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
        edge = build_edge_features(feat_in, idx)
        feat = self.edge_conv(edge)
        feat = feat.max(dim=-1).values
        feat = self.relu(self.bn2(self.conv2(feat)))
        feat = self.relu(self.bn3(self.conv3(feat)))
        feat = feat.max(dim=-1).values
        return feat.view(B, M, -1)


class _EncoderLIF(nn.Module):
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
    def __init__(self, feat_dim: int = 512, k_edge: int = 20,
                 in_channels: int = 6, T_enc: int = 4,
                 lif_leak: float = 0.9, lif_threshold: float = 1.0):
        super().__init__()
        self.k = k_edge
        self.in_channels = in_channels
        self.T_enc = T_enc

        self.conv1 = nn.Conv2d(in_channels * 2, 128, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(128)
        self.lif1  = _EncoderLIF(lif_leak, lif_threshold)

        self.conv2_edge = nn.Conv2d(128, 128, 1, bias=False)
        self.bn2_edge   = nn.BatchNorm2d(128)
        self.lif2_edge  = _EncoderLIF(lif_leak, lif_threshold)

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
        edge = build_edge_features(feat_in, idx)
        return edge, B, M, K, BM

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        edge, B, M, K, BM = self._build_input(slices)
        self._reset_all()

        feat_accum = None

        for t in range(self.T_enc):
            h = self.bn1(self.conv1(edge))
            h = self.lif1(h)
            h = self.bn2_edge(self.conv2_edge(h))
            h = self.lif2_edge(h)

            h = h.max(dim=-1).values

            h = self.bn2(self.conv2(h))
            h = self.lif3(h)
            h = self.bn3(self.conv3(h))
            h = self.lif4(h)

            h = h.max(dim=-1).values

            if feat_accum is None:
                feat_accum = h
            else:
                feat_accum = feat_accum + h

        feat = feat_accum / float(self.T_enc)
        return feat.view(B, M, -1)


class EdgeConvFeatureExtractor(nn.Module):
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
