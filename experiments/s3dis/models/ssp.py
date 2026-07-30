"""
models/ssp.py — Slice Selection Policy.

Scores unvisited slices via scaled dot-product attention between the
current LIF belief state (hidden_dim) and per-slice geometry descriptors
(geo_dim=8).

At each ASP timestep the SSP produces a score vector [B, M] with
visited entries masked to -inf so that Gumbel-softmax (training) or
argmax (inference) never re-selects a slice.
"""

import torch
import torch.nn as nn


class SSP(nn.Module):
    """Slice Selection Policy — attention-based scoring."""

    def __init__(self, belief_dim: int, geo_dim: int = 8,
                 d_ssp: int = 128):
        super().__init__()
        self.key_proj   = nn.Linear(belief_dim, d_ssp, bias=False)
        self.query_proj = nn.Linear(geo_dim, d_ssp, bias=False)
        self.scale      = d_ssp ** -0.5

    def forward(
        self,
        belief:   torch.Tensor,   # [B, hidden_dim]
        geo:      torch.Tensor,   # [B, M, geo_dim]
        vis_mask: torch.Tensor,   # [B, M]  bool — True = already visited
    ) -> torch.Tensor:
        """
        Returns:
            scores: [B, M] with visited entries masked to a very negative value

        Note: we use a large finite negative value (-1e9) instead of -inf so
        that Gumbel-softmax and argmax remain numerically stable even in the
        edge case where ALL slices have been visited (would happen if T >= M).
        With true -inf the softmax over all -inf values would produce NaN.
        """
        key   = self.key_proj(belief)                           # [B, d_ssp]
        query = self.query_proj(geo)                            # [B, M, d_ssp]
        scores = (query * key.unsqueeze(1)).sum(-1) * self.scale  # [B, M]
        return scores.masked_fill(vis_mask, -1e9)
