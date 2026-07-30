"""
models/asp_segmentor.py — ASP-SNN for point cloud segmentation.

Handles both:
    ShapeNetPart : use_category=True,  num_classes=50 parts, num_categories=16
    S3DIS        : use_category=False, num_classes=13 scene classes

Architecture:
    Shared with classifier:
        EdgeConv encoder -> [B, M, feat_dim]
        Transformer -> [B, M, feat_dim]
        SSP + ASP loop + LIF -> belief [B, hidden_dim]

    Segmentation-specific:
        PerPointBranch: pts_xyz [B,N,3] -> MLP -> [B,N, point_feat_dim]
        SegHead MLP: [local | global | point_feat | (cat_onehot) | xyz] -> [B,N, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .room_encoder import RoomPriorProjection, summary_dim

from .encoder import EdgeConvFeatureExtractor
from .ssp import SSP
from .lif import MultiLayerLIF


class PerPointBranch(nn.Module):
    """
    Per-point xyz -> unique feature.  Breaks the problem where all ~128
    points in the same slice share an identical 512-dim feature.

    pts_xyz [B, N, 3] -> [B, N, out_dim]
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_dim, 1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: [B, N, in_dim] -> [B, N, out_dim]"""
        x = pts.transpose(1, 2)    # [B, in_dim, N]
        x = self.mlp(x)            # [B, out_dim, N]
        return x.transpose(1, 2)   # [B, N, out_dim]


class SegmentationHead(nn.Module):
    """
    Per-point MLP for segmentation.

    Input per point:
        ShapeNet: [local(512) | global(512) | point(64) | cat(16) | xyz(3)] = 1107
        S3DIS:    [local(512) | global(512) | point(64) | xyz(3)]           = 1091
    """

    def __init__(self, feat_dim: int = 512, point_feat_dim: int = 64,
                 num_classes: int = 50, num_categories: int = 0,
                 xyz_dim: int = 3):
        super().__init__()
        in_dim = feat_dim * 2 + point_feat_dim + num_categories + xyz_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, local_feats, global_feat, point_feats,
                cat_onehot, pts_xyz):
        """
        local_feats  : [B, N, feat_dim]
        global_feat  : [B, feat_dim]
        point_feats  : [B, N, point_feat_dim]
        cat_onehot   : [B, num_cats] or None
        pts_xyz      : [B, N, 3]

        Returns: [B, N, num_classes]
        """
        B, N, _ = local_feats.shape
        g = global_feat.unsqueeze(1).expand(B, N, -1)

        parts = [local_feats, g, point_feats]
        if cat_onehot is not None:
            c = cat_onehot.unsqueeze(1).expand(B, N, -1).float()
            parts.append(c)
        parts.append(pts_xyz)

        x = torch.cat(parts, dim=-1)    # [B, N, in_dim]
        x = x.reshape(B * N, -1)
        return self.mlp(x).reshape(B, N, -1)


class ASPSegmentor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_category = getattr(cfg, 'use_category', False)
        self.num_classes  = cfg.num_classes
        self.num_cats     = getattr(cfg, 'num_categories', 0) if self.use_category else 0

        in_ch = getattr(cfg, 'in_channels', 6)
        point_feat_dim = getattr(cfg, 'point_feat_dim', 64)

        # Shared encoder
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
        self.belief_norm    = nn.LayerNorm(cfg.hidden_dim)

        # LIF head — num_classes=1 stub (not used for segmentation output)
        self.lif_head = MultiLayerLIF(
            feat_dim=cfg.feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=1,
            num_layers=cfg.num_lif_layers,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            spike_dropout=getattr(cfg, 'spike_dropout', 0.0),
            lif_learnable=getattr(cfg, 'lif_learnable', True),  # default True = learnable LIF
        )
        # ── E1: Room-level ASP prior ─────────────────────────────────
        self.use_room_prior = getattr(cfg, 'use_room_prior', False)
        if self.use_room_prior:
            D_room = summary_dim(
                use_rgb=getattr(cfg, 'use_rgb', True),
                use_height=getattr(cfg, 'use_height', True),
            ) * getattr(cfg, 'room_prior_anchors', 64)
            # Note: pooled summary is mean+max over anchors, so effective
            # dim is 2 * per_anchor. summary_dim() already returns 2 * per_anchor.
            D_room = summary_dim(
                use_rgb=getattr(cfg, 'use_rgb', True),
                use_height=getattr(cfg, 'use_height', True),
            )
            self.room_prior_proj = RoomPriorProjection(
                room_summary_dim=D_room,
                hidden_dim=cfg.hidden_dim,
            )
        else:
            self.room_prior_proj = None

        self.register_buffer('gumbel_tau',
                             torch.tensor(float(cfg.tau_start)))

        # Per-point branch
        # For S3DIS we feed xyz+rgb+height (7 dims) to PerPointBranch
        pp_in = 3
        if getattr(cfg, 'use_height', False) and getattr(cfg, 'use_rgb', False):
            pp_in = 7  # xyz + rgb + height
        elif getattr(cfg, 'use_rgb', False):
            pp_in = 6  # xyz + rgb
        self.point_branch = PerPointBranch(in_dim=pp_in, out_dim=point_feat_dim)

        # Segmentation head
        self.seg_head = SegmentationHead(
            feat_dim=cfg.feat_dim,
            point_feat_dim=point_feat_dim,
            num_classes=self.num_classes,
            num_categories=self.num_cats,
            xyz_dim=3,
        )

    @staticmethod
    def aux_weights(T: int) -> list:
        if T == 1:
            return [1.0]
        return [0.1 + 0.9 * (t / (T - 1)) ** 2 for t in range(T)]

    def forward(self, slices, geo, sid_arr, cat_ids, pts_features,
                room_summary=None, fine_slices=None, fine_geo=None,
                fine_sid_arr=None, training=True):
        """
        Args:
            slices:       [B, M, K, C]
            geo:          [B, M, 8]
            sid_arr:      [B, N]
            cat_ids:      [B]
            pts_features: [B, N, F]
            room_summary: [B, D_room] or None
                          E1: precomputed room-level prior. When provided
                          AND self.use_room_prior=True, seeds the initial
                          LIF belief state via a zero-init learnable
                          projection. When None or use_room_prior=False,
                          behavior is identical to the pre-E1 baseline.
            training:     bool

        Returns:
            part_logits:  [B, N, num_classes]
            aux:          dict with belief_list, per_timestep_logits, bnd_logits
        """
        B, M, K, _ = slices.shape
        N      = sid_arr.shape[1]
        device = slices.device
        

        # Defensive: verify pts_features channel dim matches PerPointBranch
        pp_in_dim = self.point_branch.mlp[0].in_channels
        assert pts_features.shape[-1] == pp_in_dim, (
            f"pts_features has {pts_features.shape[-1]} channels but "
            f"PerPointBranch expects {pp_in_dim}. Check use_rgb/use_height config."
        )

        # Category one-hot (ShapeNet) or None (S3DIS)
        if self.use_category and cat_ids is not None:
            cat_onehot = F.one_hot(cat_ids.long(), self.num_cats)
        else:
            cat_onehot = None

        # Per-point branch
        point_feats = self.point_branch(pts_features)  # [B, N, point_feat_dim]

        # Encode all slices (original order — NOT sorted)
        all_feats = self.feature_extractor(slices)       # [B, M, feat_dim]
        pos       = self.pos_proj(geo[:, :, :3])
        all_feats = all_feats + pos
        all_feats = self.slice_transformer(all_feats)

        # Sort for SSP only
        order     = geo[:, :, 6].argsort(dim=1, descending=True)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        geo_ord          = geo[batch_idx, order]
        all_feats_sorted = all_feats[batch_idx, order]

        # Extract xyz from pts_features (always first 3 dims)
        pts_xyz = pts_features[:, :, :3]
        
        # Per-point local features via direct lookup (original slice order)
        b_idx       = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        local_feats = all_feats[b_idx, sid_arr.long()]  # [B, N, feat_dim]

        # ASP loop — compute intermediate logits for active loss
        states      = self.lif_head.init_state(B, device)
        if self.room_prior_proj is not None and room_summary is not None:
            u_init = self.room_prior_proj(room_summary)        # [B, hidden_dim]
            u0, s0 = states[0]
            states[0] = (u0 + u_init, s0)
        belief      = torch.zeros(B, self.cfg.hidden_dim, device=device)
        vis_mask    = torch.zeros(B, M, dtype=torch.bool, device=device)
        belief_list = []
        logits_all  = []

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

            e_t = (w.unsqueeze(-1) * all_feats_sorted).sum(dim=1)
            e_t = e_t + self.belief_to_feat(states[-1][0].detach())

            _, states, u_last = self.lif_head.step(e_t, states)
            belief = self.belief_norm(u_last.detach())
            belief_list.append(belief)
            
            # Compute intermediate segmentation logits for this step
            global_feat_t = torch.stack(belief_list, dim=0).mean(dim=0)
            logits_t = self.seg_head(
                local_feats, global_feat_t, point_feats, cat_onehot, pts_xyz,
            )
            logits_all.append(logits_t)

        return logits_all[-1], logits_all