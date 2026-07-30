from __future__ import annotations

import torch
import torch.nn as nn

from .lif import LIFCell


class MultiLayerLIFHead(nn.Module):
    def __init__(self, dim: int, n_layers: int = 3, tau: float = 2.0,
                 v_th: float = 1.0, learnable: bool = True,
                 sg_slope: float = 4.0,
                 use_mpbn: bool = True, use_residual: bool = True):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.dim          = dim
        self.n_layers     = n_layers
        self.use_mpbn     = bool(use_mpbn)
        self.use_residual = bool(use_residual)
        self.v_th         = v_th
        self.sg_slope     = sg_slope

        self.linears = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(n_layers)
        ])

        if self.use_mpbn:
            self.mpbns = nn.ModuleList([
                nn.BatchNorm1d(dim) for _ in range(n_layers)
            ])
        else:
            self.mpbns = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

        self.cells = nn.ModuleList([
            LIFCell(dim, tau=tau, v_th=v_th, learnable=learnable,
                    sg_slope=sg_slope)
            for _ in range(n_layers)
        ])

        self._layer_frs: list[torch.Tensor] = []

    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        for cell in self.cells:
            cell.reset_state(batch, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        self._layer_frs = []
        for L in range(self.n_layers):
            z = self.linears[L](h)
            if self.use_mpbn:
                z = self.mpbns[L](z)

            if self.use_residual and L > 0:
                z = z + h.to(z.dtype)
            spikes = self.cells[L](z)

            self._layer_frs.append(spikes.mean())
            h = spikes
        return h

    @property
    def membrane(self) -> torch.Tensor | None:
        return self.cells[-1].membrane

    @property
    def last_firing_rate(self) -> torch.Tensor:
        if not self._layer_frs:
            return torch.zeros((), device=next(self.parameters()).device)
        return self._layer_frs[-1]

    @property
    def mean_firing_rate(self) -> torch.Tensor:
        if not self._layer_frs:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(self._layer_frs).mean()
