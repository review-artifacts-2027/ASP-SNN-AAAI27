"""Slice Selection Policy (SSP): lightweight cross-attention over unvisited anchors.

    score(m) = (W_k u_{t-1})^T (W_q g_m) / sqrt(d_ssp)          (paper Eq. 1)

Design notes backed by Theorem "rank cap" (paper/theory_section.tex, Thm. 5):
the bilinear form B = W_k^T W_q has rank <= min(d_ssp, 6, D) = 6, so any SSP
with d_ssp >= 6 is *exactly* expressible with d_ssp = 6.  The `rank` option
factorizes W_k = A B (A: d_ssp x r, B: r x D) to hit the ~2K-parameter budget
without any loss of expressiveness for r >= 6.

Policy modes:
    ssp            membrane-driven scoring (the method)
    random         uniform over unvisited anchors (A5 baseline)
    fixed          FPS ordinal order (fixed-order SNN baseline)
    geometry_only  static learned saliency, no membrane feedback (S3/T3 probe)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -1e9  # used instead of -inf: safe under Gumbel noise / softmax


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

    # ----------------------------------------------------------------- params
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def bilinear_form(self) -> torch.Tensor:
        """Effective B = W_k^T W_q in R^{D x d_desc} (used by verification V4)."""
        wq = self.Wq.weight                                   # (d_ssp, d_desc)
        if isinstance(self.Wk, nn.Sequential):
            wk = self.Wk[1].weight @ self.Wk[0].weight        # (d_ssp, D)
        else:
            wk = self.Wk.weight
        return wk.t() @ wq                                    # (D, d_desc)

    # ----------------------------------------------------------------- scores
    def scores(self, u: torch.Tensor, desc: torch.Tensor,
               visited: torch.Tensor) -> torch.Tensor:
        """u: (B,D) membrane; desc: (B,K,6); visited: (B,K) bool -> (B,K)."""
        if self.policy == "geometry_only":
            key = self.static_key.unsqueeze(0).expand(u.shape[0], -1)
        else:
            key = self.Wk(u)                                  # (B, d_ssp)
        query = self.Wq(desc)                                 # (B, K, d_ssp)
        s = torch.einsum("bd,bkd->bk", key, query) / self.d_ssp ** 0.5
        if self.policy == "random":
            s = torch.rand_like(s)          # uniform priority, no learning signal
        elif self.policy == "fixed":
            K = s.shape[1]
            s = -torch.arange(K, device=s.device, dtype=s.dtype).expand_as(s)
        if self.use_mask:
            s = s.masked_fill(visited, MASK_VALUE)
        return s

    # -------------------------------------------------------------- selection
    def select(self, scores: torch.Tensor, hard_inference: bool,
               tau: float = 1.0) -> torch.Tensor:
        """Return a (B,K) one-hot selection.

        Training: Gumbel-softmax(hard=True) straight-through (paper Sec. 4).
        Inference: hard argmax, O(1) overhead.
        Non-learned policies (random/fixed) never need gradients -> argmax.
        """
        if hard_inference or self.policy in {"random", "fixed"}:
            idx = scores.argmax(-1)
            return F.one_hot(idx, scores.shape[-1]).to(scores.dtype)
        return F.gumbel_softmax(scores, tau=tau, hard=True)
