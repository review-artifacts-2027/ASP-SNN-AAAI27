import torch
import torch.nn as nn


class SSP(nn.Module):
    def __init__(self, belief_dim: int, geo_dim: int = 8,
                 d_ssp: int = 128):
        super().__init__()
        self.key_proj   = nn.Linear(belief_dim, d_ssp, bias=False)
        self.query_proj = nn.Linear(geo_dim, d_ssp, bias=False)
        self.scale      = d_ssp ** -0.5

    def forward(
        self,
        belief:   torch.Tensor,
        geo:      torch.Tensor,
        vis_mask: torch.Tensor,
    ) -> torch.Tensor:
        key   = self.key_proj(belief)
        query = self.query_proj(geo)
        scores = (query * key.unsqueeze(1)).sum(-1) * self.scale
        return scores.masked_fill(vis_mask, -1e9)
