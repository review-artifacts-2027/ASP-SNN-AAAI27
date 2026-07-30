import torch
import torch.nn as nn


class SSP(nn.Module):
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

        self.num_categories = num_categories
        if num_categories > 0:
            self.cat_emb      = nn.Embedding(num_categories, cat_emb_dim)
            self.cat_key_proj = nn.Linear(cat_emb_dim, d_ssp, bias=False)
            nn.init.normal_(self.cat_emb.weight, std=0.02)
            nn.init.zeros_(self.cat_key_proj.weight)
        else:
            self.cat_emb      = None
            self.cat_key_proj = None

        self.use_boundary_bias = use_boundary_bias
        if use_boundary_bias:
            self.boundary_weight = nn.Parameter(torch.tensor(0.0))
        else:
            self.boundary_weight = None

    def forward(
        self,
        belief:          torch.Tensor,
        geo:             torch.Tensor,
        vis_mask:        torch.Tensor,
        cat_ids:         torch.Tensor = None,
        boundary_scores: torch.Tensor = None,
    ) -> torch.Tensor:
        B, M, _ = geo.shape
        device = geo.device

        if self.mode == 'learned':
            key = self.key_proj(belief)

            if self.cat_emb is not None and cat_ids is not None:
                cat_key = self.cat_key_proj(self.cat_emb(cat_ids.long()))
                key = key + cat_key

            query  = self.query_proj(geo)
            scores = (query * key.unsqueeze(1)).sum(-1) * self.scale

            if self.boundary_weight is not None and boundary_scores is not None:
                scores = scores + self.boundary_weight * boundary_scores

        elif self.mode == 'random':
            scores = torch.rand(B, M, device=device)

        else:
            ramp = torch.linspace(1.0, 0.0, M, device=device)
            scores = ramp.unsqueeze(0).expand(B, M).contiguous()

        return scores.masked_fill(vis_mask, -1e9)
