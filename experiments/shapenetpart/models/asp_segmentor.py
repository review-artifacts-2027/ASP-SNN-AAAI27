import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EdgeConvFeatureExtractor
from .ssp import SSP
from .lif import MultiLayerLIF, _spike


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


class BoundaryHead(nn.Module):
    def __init__(self, feat_dim: int = 512, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, local_feats: torch.Tensor) -> torch.Tensor:
        B, N, D = local_feats.shape
        x = local_feats.reshape(B * N, D)
        return self.mlp(x).reshape(B, N)


class _SegLIFCell(nn.Module):
    def __init__(self, dim: int, leak: float = 0.9, threshold: float = 1.0,
                 use_mpbn: bool = True, learnable_params: bool = True):
        super().__init__()
        self.use_mpbn = use_mpbn
        self.learnable_params = learnable_params
        self.mpbn = nn.BatchNorm1d(dim) if use_mpbn else nn.Identity()

        if learnable_params:
            import math as _math
            leak = min(max(leak, 1e-4), 1.0 - 1e-4)
            raw_leak_init = _math.log(leak / (1.0 - leak))
            raw_thr_init  = _math.log(_math.expm1(threshold))
            self.raw_leak      = nn.Parameter(torch.tensor(raw_leak_init))
            self.raw_threshold = nn.Parameter(torch.tensor(raw_thr_init))
        else:
            self.register_buffer('_leak', torch.tensor(float(leak)))
            self.register_buffer('_threshold', torch.tensor(float(threshold)))

    @property
    def leak(self):
        if self.learnable_params:
            return torch.sigmoid(self.raw_leak)
        return self._leak

    @property
    def threshold(self):
        if self.learnable_params:
            return F.softplus(self.raw_threshold)
        return self._threshold

    def step(self, inp: torch.Tensor,
             u_prev: torch.Tensor, s_prev: torch.Tensor):
        leak = self.leak
        thr  = self.threshold
        u_t = leak * u_prev + inp - thr * s_prev
        u_for_spike = self.mpbn(u_t)
        s_t = _spike(u_for_spike - thr)
        return u_t, s_t


class SegmentationHead(nn.Module):
    def __init__(self, feat_dim: int = 512, point_feat_dim: int = 64,
                 num_classes: int = 50, num_categories: int = 0,
                 xyz_dim: int = 3, fine_feat_dim: int = 0):
        super().__init__()
        in_dim = feat_dim * 2 + point_feat_dim + num_categories + xyz_dim + fine_feat_dim

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
                cat_onehot, pts_xyz, fine_feats=None):
        B, N, _ = local_feats.shape
        g = global_feat.unsqueeze(1).expand(B, N, -1)

        parts = [local_feats, g, point_feats]
        if cat_onehot is not None:
            c = cat_onehot.unsqueeze(1).expand(B, N, -1).float()
            parts.append(c)
        parts.append(pts_xyz)
        if fine_feats is not None:
            parts.append(fine_feats)

        x = torch.cat(parts, dim=-1)
        x = x.reshape(B * N, -1)
        return self.mlp(x).reshape(B, N, -1)


class SpikingSegmentationHead(nn.Module):
    def __init__(self, feat_dim: int = 512, point_feat_dim: int = 64,
                 num_classes: int = 50, num_categories: int = 0,
                 xyz_dim: int = 3, fine_feat_dim: int = 0,
                 T_seg: int = 2, leak: float = 0.9, threshold: float = 1.0,
                 use_mpbn: bool = True, learnable_params: bool = True):
        super().__init__()
        self.T_seg = T_seg

        in_dim = feat_dim * 2 + point_feat_dim + num_categories + xyz_dim + fine_feat_dim

        self.fc1 = nn.Linear(in_dim, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.lif1 = _SegLIFCell(256, leak, threshold, use_mpbn, learnable_params)

        self.fc2 = nn.Linear(256, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.lif2 = _SegLIFCell(256, leak, threshold, use_mpbn, learnable_params)

        self.fc3 = nn.Linear(256, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.lif3 = _SegLIFCell(128, leak, threshold, use_mpbn, learnable_params)

        self.classifier = nn.Linear(128, num_classes)

        self.spike_monitor = None

    def forward(self, local_feats, global_feat, point_feats,
                cat_onehot, pts_xyz, fine_feats=None):
        B, N, _ = local_feats.shape
        g = global_feat.unsqueeze(1).expand(B, N, -1)

        parts = [local_feats, g, point_feats]
        if cat_onehot is not None:
            c = cat_onehot.unsqueeze(1).expand(B, N, -1).float()
            parts.append(c)
        parts.append(pts_xyz)
        if fine_feats is not None:
            parts.append(fine_feats)

        x = torch.cat(parts, dim=-1)
        x_flat = x.reshape(B * N, -1)
        device = x_flat.device
        dtype  = x_flat.dtype

        z1 = self.bn1(self.fc1(x_flat))

        u1 = torch.zeros(B * N, 256, device=device, dtype=dtype)
        s1 = torch.zeros(B * N, 256, device=device, dtype=dtype)
        u2 = torch.zeros(B * N, 256, device=device, dtype=dtype)
        s2 = torch.zeros(B * N, 256, device=device, dtype=dtype)
        u3 = torch.zeros(B * N, 128, device=device, dtype=dtype)
        s3 = torch.zeros(B * N, 128, device=device, dtype=dtype)

        mem_accum = torch.zeros(B * N, 128, device=device, dtype=dtype)

        for t in range(self.T_seg):
            u1, s1 = self.lif1.step(z1, u1, s1)
            if self.spike_monitor is not None:
                self.spike_monitor.record(0, s1)

            z2 = self.bn2(self.fc2(s1))
            u2, s2 = self.lif2.step(z2, u2, s2)
            if self.spike_monitor is not None:
                self.spike_monitor.record(1, s2)

            z3 = self.bn3(self.fc3(s2))
            u3, s3 = self.lif3.step(z3, u3, s3)
            if self.spike_monitor is not None:
                self.spike_monitor.record(2, s3)

            mem_accum = mem_accum + u3

        mem_avg = mem_accum / self.T_seg
        logits  = self.classifier(mem_avg)
        return logits.reshape(B, N, -1)


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
            encoder_type=getattr(cfg, 'encoder_type', 'analog'),
            T_enc=getattr(cfg, 'encoder_T', 4),
            lif_leak=getattr(cfg, 'encoder_lif_leak', getattr(cfg, 'lif_leak', 0.9)),
            lif_threshold=getattr(cfg, 'encoder_lif_threshold', getattr(cfg, 'lif_threshold', 1.0)),
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

        self.use_multiscale = getattr(cfg, 'use_multiscale', True)
        self.fine_feat_dim  = getattr(cfg, 'fine_feat_dim', 128) if self.use_multiscale else 0
        if self.use_multiscale:
            self.fine_encoder = EdgeConvFeatureExtractor(
                feat_dim=self.fine_feat_dim,
                k_edge=getattr(cfg, 'fine_k_edge', 8),
                in_channels=in_ch,
                encoder_type='analog',
                T_enc=1,
                lif_leak=cfg.lif_leak,
                lif_threshold=cfg.lif_threshold,
            )
            self.fine_pos_proj = nn.Linear(3, self.fine_feat_dim, bias=False)
        else:
            self.fine_encoder  = None
            self.fine_pos_proj = None

        self.use_boundary_aware = getattr(cfg, 'use_boundary_aware', True)
        if self.use_boundary_aware:
            self.boundary_head = BoundaryHead(
                feat_dim=cfg.feat_dim,
                hidden=getattr(cfg, 'boundary_hidden', 128),
            )
        else:
            self.boundary_head = None

        self.ssp = SSP(
            belief_dim=cfg.hidden_dim,
            geo_dim=cfg.geo_dim,
            d_ssp=cfg.d_ssp,
            mode=getattr(cfg, 'ssp_mode', 'learned'),
            num_categories=self.num_cats,
            cat_emb_dim=getattr(cfg, 'cat_emb_dim', 8),
            use_boundary_bias=self.use_boundary_aware,
        )
        self.belief_to_feat = nn.Linear(cfg.hidden_dim, cfg.feat_dim, bias=False)
        self.belief_norm    = nn.LayerNorm(cfg.hidden_dim)

        self.global_context = nn.Sequential(
            nn.Linear(cfg.feat_dim * 2, cfg.feat_dim, bias=False),
            nn.LayerNorm(cfg.feat_dim),
            nn.GELU(),
        )
        self.context_gate = nn.Parameter(torch.tensor(0.5))

        self.belief_to_seg_global = nn.Linear(cfg.hidden_dim, cfg.feat_dim, bias=False)
        nn.init.zeros_(self.belief_to_seg_global.weight)

        self.lif_head = MultiLayerLIF(
            feat_dim=cfg.feat_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=1,
            num_layers=cfg.num_lif_layers,
            leak=cfg.lif_leak,
            threshold=cfg.lif_threshold,
            learnable_params=getattr(cfg, 'lif_learnable', False),
            use_mpbn=getattr(cfg, 'lif_use_mpbn', False),
        )

        self.register_buffer('gumbel_tau',
                             torch.tensor(float(cfg.tau_start)))

        pp_in = 3
        if getattr(cfg, 'use_height', False) and getattr(cfg, 'use_rgb', False):
            pp_in = 7
        elif getattr(cfg, 'use_rgb', False):
            pp_in = 6
        self.point_branch = PerPointBranch(in_dim=pp_in, out_dim=point_feat_dim)

        self.seg_head_type = getattr(cfg, 'seg_head_type', 'spiking')
        if self.seg_head_type == 'spiking':
            self.seg_head = SpikingSegmentationHead(
                feat_dim=cfg.feat_dim,
                point_feat_dim=point_feat_dim,
                num_classes=self.num_classes,
                num_categories=self.num_cats,
                xyz_dim=3,
                fine_feat_dim=self.fine_feat_dim,
                T_seg=getattr(cfg, 'seg_head_T', 2),
                leak=getattr(cfg, 'seg_head_lif_leak', cfg.lif_leak),
                threshold=getattr(cfg, 'seg_head_lif_threshold', cfg.lif_threshold),
                use_mpbn=getattr(cfg, 'lif_use_mpbn', True),
                learnable_params=getattr(cfg, 'lif_learnable', True),
            )
        elif self.seg_head_type == 'analog':
            self.seg_head = SegmentationHead(
                feat_dim=cfg.feat_dim,
                point_feat_dim=point_feat_dim,
                num_classes=self.num_classes,
                num_categories=self.num_cats,
                xyz_dim=3,
                fine_feat_dim=self.fine_feat_dim,
            )
        else:
            raise ValueError(f"Unknown seg_head_type: {self.seg_head_type}")

    @staticmethod
    def aux_weights(T: int) -> list:
        if T == 1:
            return [1.0]
        return [0.1 + 0.9 * (t / (T - 1)) ** 2 for t in range(T)]

    def _soft_feature_propagation(self, all_feats, geo, pts_features, sid_arr):
        B, M, feat_dim = all_feats.shape
        N = pts_features.shape[1]
        device = all_feats.device

        use_soft_fp = getattr(self.cfg, 'use_soft_fp', True)
        if not use_soft_fp:
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
            return all_feats[b_idx, sid_arr.long()]

        k_fp   = getattr(self.cfg, 'fp_k', 3)
        fp_eps = getattr(self.cfg, 'fp_epsilon', 1e-8)

        anchor_xyz = geo[:, :, :3]
        points_xyz = pts_features[:, :, :3]

        dist = torch.cdist(points_xyz.float(), anchor_xyz.float())
        k_use = min(k_fp, M)
        knn_dist, knn_idx = dist.topk(k_use, dim=-1, largest=False)

        inv_dist = 1.0 / (knn_dist + fp_eps)
        weights  = inv_dist / inv_dist.sum(dim=-1, keepdim=True)

        b_idx_k = (torch.arange(B, device=device)
                   .view(B, 1, 1).expand(B, N, k_use))
        knn_feats = all_feats[b_idx_k, knn_idx]

        weights_typed = weights.to(knn_feats.dtype)
        return (weights_typed.unsqueeze(-1) * knn_feats).sum(dim=-2)

    def _slice_boundary_scores(self, bnd_probs, sid_arr, M):
        B, N = bnd_probs.shape
        device = bnd_probs.device
        sums   = torch.zeros(B, M, device=device, dtype=bnd_probs.dtype)
        counts = torch.zeros(B, M, device=device, dtype=bnd_probs.dtype)
        sums.scatter_add_(1, sid_arr, bnd_probs)
        counts.scatter_add_(1, sid_arr, torch.ones_like(bnd_probs))
        return sums / counts.clamp(min=1.0)

    def forward(self, slices, geo, sid_arr, cat_ids, pts_features,
                fine_slices=None, fine_geo=None, fine_sid_arr=None,
                training=True):
        B, M, K, _ = slices.shape
        N      = sid_arr.shape[1]
        device = slices.device

        pp_in_dim = self.point_branch.mlp[0].in_channels
        assert pts_features.shape[-1] == pp_in_dim, (
            f"pts_features has {pts_features.shape[-1]} channels but "
            f"PerPointBranch expects {pp_in_dim}."
        )

        compute_per_t = training and (
            getattr(self.cfg, 'loss_mode', 'aux') == 'dense_tet'
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

        ctx_mean   = all_feats.mean(dim=1)
        ctx_max    = all_feats.max(dim=1).values
        global_ctx = self.global_context(torch.cat([ctx_mean, ctx_max], dim=-1))

        local_feats = self._soft_feature_propagation(
            all_feats, geo, pts_features, sid_arr,
        )

        fine_local_feats = None
        if self.use_multiscale and fine_slices is not None:
            fine_all_feats = self.fine_encoder(fine_slices)
            fine_pos       = self.fine_pos_proj(fine_geo[:, :, :3])
            fine_all_feats = fine_all_feats + fine_pos
            fine_local_feats = self._soft_feature_propagation(
                fine_all_feats, fine_geo, pts_features, fine_sid_arr,
            )

        bnd_logits       = None
        slice_bnd_scores = None
        if self.boundary_head is not None:
            bnd_logits = self.boundary_head(local_feats)
            bnd_probs  = torch.sigmoid(bnd_logits)
            slice_bnd_scores = self._slice_boundary_scores(
                bnd_probs, sid_arr.long(), M,
            )

        pts_xyz = pts_features[:, :, :3]

        order     = geo[:, :, 6].argsort(dim=1, descending=True)
        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        geo_ord          = geo[batch_idx, order]
        all_feats_sorted = all_feats[batch_idx, order]
        bnd_scores_ord = None
        if slice_bnd_scores is not None:
            bnd_scores_ord = slice_bnd_scores[batch_idx, order]

        states              = self.lif_head.init_state(B, device)
        belief              = torch.zeros(B, self.cfg.hidden_dim, device=device)
        vis_mask            = torch.zeros(B, M, dtype=torch.bool, device=device)
        belief_list         = []
        per_timestep_logits = [] if compute_per_t else None
        gate                = torch.sigmoid(self.context_gate)
        u_last              = None

        for t in range(self.cfg.T):
            scores = self.ssp(
                belief, geo_ord, vis_mask,
                cat_ids=cat_ids,
                boundary_scores=bnd_scores_ord,
            )

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
            e_t = e_t + gate * global_ctx

            _, states, u_last = self.lif_head.step(e_t, states)

            if compute_per_t:
                belief_live = self.belief_norm(u_last)
                global_t = global_ctx + self.belief_to_seg_global(belief_live)
                logits_t = self.seg_head(
                    local_feats, global_t, point_feats, cat_onehot, pts_xyz,
                    fine_feats=fine_local_feats,
                )
                per_timestep_logits.append(logits_t)

            belief = self.belief_norm(u_last.detach())
            belief_list.append(belief)

        if compute_per_t:
            part_logits = per_timestep_logits[-1]
        else:
            final_belief = self.belief_norm(u_last) if u_last is not None else belief
            final_global = global_ctx + self.belief_to_seg_global(final_belief)
            part_logits = self.seg_head(
                local_feats, final_global, point_feats, cat_onehot, pts_xyz,
                fine_feats=fine_local_feats,
            )

        aux = {
            'belief_list':         belief_list,
            'per_timestep_logits': per_timestep_logits,
            'bnd_logits':          bnd_logits,
        }
        return part_logits, aux
