import torch
import torch.nn as nn
import torch.nn.functional as F
from .room_encoder import RoomPriorProjection, summary_dim

from .encoder import EdgeConvFeatureExtractor
from .ssp import SSP
from .lif import MultiLayerLIF


class PerPointBranch(nn.Module):
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
        x = pts.transpose(1, 2)
        x = self.mlp(x)
        return x.transpose(1, 2)


class SegmentationHead(nn.Module):
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

        B, N, _ = local_feats.shape
        g = global_feat.unsqueeze(1).expand(B, N, -1)

        parts = [local_feats, g, point_feats]
        if cat_onehot is not None:
            c = cat_onehot.unsqueeze(1).expand(B, N, -1).float()
            parts.append(c)
        parts.append(pts_xyz)

        x = torch.cat(parts, dim=-1)
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

        self.lif_head = MultiLayerLIF(
            feat_dim=cfg.feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=1,
            num_layers=cfg.num_lif_layers,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            spike_dropout=getattr(cfg, 'spike_dropout', 0.0),
            lif_learnable=getattr(cfg, 'lif_learnable', True),
        )

        self.use_room_prior = getattr(cfg, 'use_room_prior', False)
        if self.use_room_prior:
            D_room = summary_dim(
                use_rgb=getattr(cfg, 'use_rgb', True),
                use_height=getattr(cfg, 'use_height', True),
            ) * getattr(cfg, 'room_prior_anchors', 64)

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

        pp_in = 3
        if getattr(cfg, 'use_height', False) and getattr(cfg, 'use_rgb', False):
            pp_in = 7
        elif getattr(cfg, 'use_rgb', False):
            pp_in = 6
        self.point_branch = PerPointBranch(in_dim=pp_in, out_dim=point_feat_dim)

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

        B, M, K, _ = slices.shape
        N      = sid_arr.shape[1]
        device = slices.device

        pp_in_dim = self.point_branch.mlp[0].in_channels
        assert pts_features.shape[-1] == pp_in_dim, (
            f"pts_features has {pts_features.shape[-1]} channels but "
            f"PerPointBranch expects {pp_in_dim}. Check use_rgb/use_height config."
        )

        if self.use_category and cat_ids is not None:
            cat_onehot = F.one_hot(cat_ids.long(), self.num_cats)
        else:
            cat_onehot = None

        point_feats = self.point_branch(pts_features)

        all_feats = self.feature_extractor(slices)
        pos       = self.pos_proj(geo[:, :, :3])
        all_feats = all_feats + pos
        all_feats = self.slice_transformer(all_feats)

        order     = geo[:, :, 6].argsort(dim=1, descending=True)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        geo_ord          = geo[batch_idx, order]
        all_feats_sorted = all_feats[batch_idx, order]

        pts_xyz = pts_features[:, :, :3]

        b_idx       = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        local_feats = all_feats[b_idx, sid_arr.long()]

        states      = self.lif_head.init_state(B, device)
        if self.room_prior_proj is not None and room_summary is not None:
            u_init = self.room_prior_proj(room_summary)
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

            global_feat_t = torch.stack(belief_list, dim=0).mean(dim=0)
            logits_t = self.seg_head(
                local_feats, global_feat_t, point_feats, cat_onehot, pts_xyz,
            )
            logits_all.append(logits_t)

        return logits_all[-1], logits_all
