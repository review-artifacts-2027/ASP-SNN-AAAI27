"""
models/ssp.py — Slice Selection Policy with ablation modes.

Modes (set via cfg.ssp_mode):
    'learned'   — attention-based scoring between LIF belief state and
                  per-slice geometry descriptors (the proposed method).
    'random'    — assign random scores to unvisited slices. Go/no-go
                  ablation baseline.
    'fps_order' — score slices by their FPS order (descending). Tests
                  whether the learned policy beats a fixed ordering.

Category conditioning (A3, Batch 1):
    num_categories > 0 → learned category embedding added to the key.

Boundary bias (B1, Batch 3):
    boundary_scores > 0 → learned scalar `boundary_weight` used to bias
    SSP scoring toward slices with high predicted-boundary probability.
    Weight is zero-initialised so training smoothly grows the bias
    only if boundary predictions become useful. When boundary head is
    off or its scores are None, SSP behaves identically to Batch 2.

Reference: Mnih et al., Recurrent Models of Visual Attention, NeurIPS 2014
(conditioning attention on task metadata & downstream signals).
"""

import torch
import torch.nn as nn


class SSP(nn.Module):
    """Slice Selection Policy — supports learned / random / fps_order modes."""

    def __init__(self, belief_dim: int, geo_dim: int = 8,
                 d_ssp: int = 128, mode: str = 'learned',
                 num_categories: int = 0, cat_emb_dim: int = 8,
                 use_boundary_bias: bool = False):
        super().__init__()
        assert mode in ('learned', 'random', 'fps_order'), \
            f"Unknown ssp_mode: {mode}"
        self.mode = mode

        self.key_proj   = nn.Linear(belief_dim, d_ssp, bias=False)
        self.query_proj = nn.Linear(geo_dim, d_ssp, bias=False)
        self.scale      = d_ssp ** -0.5

        # ── A3 (Batch 1): Category conditioning ────────────────────────
        self.num_categories = num_categories
        if num_categories > 0:
            self.cat_emb      = nn.Embedding(num_categories, cat_emb_dim)
            self.cat_key_proj = nn.Linear(cat_emb_dim, d_ssp, bias=False)
            nn.init.normal_(self.cat_emb.weight, std=0.02)
            nn.init.zeros_(self.cat_key_proj.weight)
        else:
            self.cat_emb      = None
            self.cat_key_proj = None

        # ── B1 (Batch 3): Boundary bias ───────────────────────────────
        # boundary_weight is zero-initialised so early training (when the
        # boundary head is still noise) doesn't corrupt SSP decisions.
        # As the boundary head learns to predict boundaries, training
        # naturally grows this weight if the bias improves seg loss.
        self.use_boundary_bias = use_boundary_bias
        if use_boundary_bias:
            self.boundary_weight = nn.Parameter(torch.tensor(0.0))
        else:
            self.boundary_weight = None

    def forward(
        self,
        belief:          torch.Tensor,           # [B, hidden_dim]
        geo:             torch.Tensor,           # [B, M, geo_dim]
        vis_mask:        torch.Tensor,           # [B, M]  bool
        cat_ids:         torch.Tensor = None,    # [B]  optional
        boundary_scores: torch.Tensor = None,    # [B, M]  optional (B1)
    ) -> torch.Tensor:
        """
        Args:
            belief          : LIF belief state
            geo             : per-slice geometry descriptors
            vis_mask        : True at visited slices
            cat_ids         : category ids for A3 (ShapeNetPart)
            boundary_scores : per-slice mean boundary probability [0,1] for B1.
                              Ignored when use_boundary_bias was False.

        Returns:
            scores: [B, M] with visited entries masked to -1e9.
        """
        B, M, _ = geo.shape
        device = geo.device

        if self.mode == 'learned':
            key = self.key_proj(belief)                              # [B, d_ssp]

            # A3: category augmentation of the key
            if self.cat_emb is not None and cat_ids is not None:
                cat_key = self.cat_key_proj(self.cat_emb(cat_ids.long()))
                key = key + cat_key                                  # [B, d_ssp]

            query  = self.query_proj(geo)                             # [B, M, d_ssp]
            scores = (query * key.unsqueeze(1)).sum(-1) * self.scale  # [B, M]

            # B1: additive boundary bias. Bias grows/shrinks via training
            # depending on whether the boundary signal helps seg loss.
            if self.boundary_weight is not None and boundary_scores is not None:
                scores = scores + self.boundary_weight * boundary_scores

        elif self.mode == 'random':
            scores = torch.rand(B, M, device=device)

        else:  # 'fps_order'
            ramp = torch.linspace(1.0, 0.0, M, device=device)
            scores = ramp.unsqueeze(0).expand(B, M).contiguous()

        return scores.masked_fill(vis_mask, -1e9)