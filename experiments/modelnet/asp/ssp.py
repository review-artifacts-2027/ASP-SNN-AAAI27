from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -1e9


class SSP(nn.Module):
    def __init__(self, d_model: int, d_desc: int = 6, d_ssp: int = 64,
                 rank: int = 0, policy: str = "ssp", use_mask: bool = True):
        super().__init__()
        assert policy in {"ssp", "random", "fixed", "geometry_only"}
        self.d_model, self.d_desc, self.d_ssp = d_model, d_desc, d_ssp
        self.policy, self.use_mask, self.rank = policy, use_mask, rank
        if rank and rank > 0:
            self.Wk = nn.Sequential(nn.Linear(d_model, rank, bias=False),
                                    nn.Linear(rank, d_ssp, bias=False))
        else:
            self.Wk = nn.Linear(d_model, d_ssp, bias=False)
        self.Wq = nn.Linear(d_desc, d_ssp, bias=False)
        if policy == "geometry_only":
            self.static_key = nn.Parameter(torch.randn(d_ssp) / d_ssp ** 0.5)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def bilinear_form(self) -> torch.Tensor:
        wq = self.Wq.weight
        if isinstance(self.Wk, nn.Sequential):
            wk = self.Wk[1].weight @ self.Wk[0].weight
        else:
            wk = self.Wk.weight
        return wk.t() @ wq

    def scores(self, u: torch.Tensor, desc: torch.Tensor,
               visited: torch.Tensor) -> torch.Tensor:

        if self.policy == "geometry_only":
            key = self.static_key.unsqueeze(0).expand(u.shape[0], -1)
        else:
            key = self.Wk(u)
        query = self.Wq(desc)
        s = torch.einsum("bd,bkd->bk", key, query) / self.d_ssp ** 0.5
        if self.policy == "random":
            s = torch.rand_like(s)
        elif self.policy == "fixed":
            K = s.shape[1]
            s = -torch.arange(K, device=s.device, dtype=s.dtype).expand_as(s)
        if self.use_mask:
            s = s.masked_fill(visited, MASK_VALUE)
        return s

    def select(self, scores: torch.Tensor, hard_inference: bool,
               tau: float = 1.0) -> torch.Tensor:

        if hard_inference or self.policy in {"random", "fixed"}:
            idx = scores.argmax(-1)
            return F.one_hot(idx, scores.shape[-1]).to(scores.dtype)
        return F.gumbel_softmax(scores, tau=tau, hard=True)
