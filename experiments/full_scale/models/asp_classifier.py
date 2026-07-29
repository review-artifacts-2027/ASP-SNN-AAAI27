"""
models/asp_classifier.py — ASP-SNN for point cloud classification.

Used for ScanObjectNN (15 classes) and can also serve ModelNet40 (40 classes).

Forward pass:
    1. Slicing is done in the dataset loader (not here).
    2. Encoder: EdgeConv on each slice -> [B, M, 512] tokens
    3. Positional encoding: Linear(3->512) on centroid xyz
    4. Cross-slice transformer: single layer, 4 heads
    5. ASP loop (T steps):
       - SSP scores unvisited slices
       - Gumbel-softmax (train) / argmax (eval) selects one
       - Fuse selected feature + belief context
       - LIF step -> logits + updated belief
       - Early exit if confidence margin > threshold (eval only)
    6. Inference: logit averaging across all timesteps

The classification head in the LIF module can be either a single Linear
or a 3-layer MLP (configured via cls_head_dims in the YAML config).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EdgeConvFeatureExtractor
from .ssp import SSP
from .lif import MultiLayerLIF


class ASPClassifier(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_ch = getattr(cfg, 'in_channels', 6)

        self.feature_extractor = EdgeConvFeatureExtractor(
            feat_dim=cfg.feat_dim,
            k_edge=cfg.k_edge,
            in_channels=in_ch,
        )

        self.pos_proj = nn.Linear(3, cfg.feat_dim, bias=False)

        self.slice_transformer = nn.TransformerEncoderLayer(
            d_model=cfg.feat_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        self.ssp = SSP(
            belief_dim=cfg.hidden_dim,
            geo_dim=cfg.geo_dim,
            d_ssp=cfg.d_ssp,
        )

        self.belief_to_feat = nn.Linear(cfg.hidden_dim, cfg.feat_dim, bias=False)
        self.belief_norm = nn.LayerNorm(cfg.hidden_dim)

        # Deep MLP head if configured, otherwise single Linear
        cls_dims = getattr(cfg, 'cls_head_dims', None)
        cls_drop = getattr(cfg, 'cls_head_dropout', None)

        self.lif_head = MultiLayerLIF(
            feat_dim=cfg.feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=cfg.num_classes,
            num_layers=cfg.num_lif_layers,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            spike_dropout=getattr(cfg, 'spike_dropout', 0.0),
            cls_head_dims=cls_dims,
            cls_head_dropout=cls_drop,
        )

        self.register_buffer('gumbel_tau',
                             torch.tensor(float(cfg.tau_start)))

    @staticmethod
    def aux_weights(T: int) -> list:
        """Quadratic ramp: earlier timesteps get lower weight."""
        if T == 1:
            return [1.0]
        return [0.1 + 0.9 * (t / (T - 1)) ** 2 for t in range(T)]

    def forward(self, slices, geo, training=True):
        """
        Args:
            slices:   [B, M, K, C]  per-slice point clouds
            geo:      [B, M, 8]     geometry descriptors
            training: bool

        Returns:
            logits_all: list of [B, num_classes] per ASP timestep
        """
        B, M, K, _ = slices.shape
        device = slices.device

        # Sort slices by max_dist descending (geo[:,:,6])
        order     = geo[:, :, 6].argsort(dim=1, descending=True)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        slices_ord = slices[batch_idx, order]
        geo_ord    = geo[batch_idx, order]

        # Encode all slices once
        all_feats = self.feature_extractor(slices_ord)       # [B, M, feat_dim]
        pos       = self.pos_proj(geo_ord[:, :, :3])
        all_feats = all_feats + pos
        all_feats = self.slice_transformer(all_feats)

        # ASP loop
        states     = self.lif_head.init_state(B, device)
        belief     = torch.zeros(B, self.cfg.hidden_dim, device=device)
        vis_mask   = torch.zeros(B, M, dtype=torch.bool, device=device)
        logits_all = []

        exit_thr = getattr(self.cfg, 'exit_threshold', 0.4)

        for t in range(self.cfg.T):
            scores = self.ssp(belief, geo_ord, vis_mask)

            if training:
                w = F.gumbel_softmax(
                    scores, tau=self.gumbel_tau.item(), hard=True, dim=-1,
                )
            else:
                w = F.one_hot(scores.argmax(dim=-1), M).float()

            sel_idx  = w.argmax(dim=-1)
            vis_mask = vis_mask.clone()
            vis_mask[torch.arange(B, device=device), sel_idx] = True

            e_t = (w.unsqueeze(-1) * all_feats).sum(dim=1)
            e_t = e_t + self.belief_to_feat(states[-1][0].detach())

            logits, states, u_last = self.lif_head.step(e_t, states)
            logits_all.append(logits)

            belief = self.belief_norm(u_last.detach())

            # Early exit at inference
            if not training:
                probs = logits.softmax(dim=-1)
                top2  = probs.topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1]).min().item()
                if margin > exit_thr:
                    break

        # Inference logit averaging
        if not training and len(logits_all) > 1:
            logits_all[-1] = torch.stack(logits_all, dim=0).mean(dim=0)

        return logits_all
