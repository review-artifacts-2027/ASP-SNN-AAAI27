"""Multi-layer LIF head with MPBN and pre-spike residuals (Phase 5).

Drop-in replacement for the single-layer LIFCell in ASPModel. Preserves
the interface exactly, so ASPModel can swap in this head with just a
change of `head_type` in the config — no other change needed:

    self.head(x)                # (B, D) -> (B, D) last-layer spikes
    self.head.reset_state(B, ...)
    self.head.membrane          # (B, D) LAST cell's pre-reset membrane
                                # (the belief state read by the SSP)

Motivation
──────────
After Phase 4 (spiking conv patch stem), each per-slice feature carries
significantly more information. The single-layer LIF head from the
original design becomes the accuracy bottleneck because 1 layer is not
enough capacity to consume the richer features. Phase 5 stacks 3 LIF
layers, adds Membrane Potential Batch Normalisation (MPBN) around each
cell's integrator input, and adds pre-spike residual shortcuts to
unblock gradients through the surrogate.

Per-layer design
────────────────
For each layer L in {0, …, n_layers-1}:

    z_L    = Linear_L(h_{L-1})                     # h_{-1} = x (input),
                                                   # else spikes from L-1
    if use_mpbn:
        z_L = BN_L(z_L)                            # MPBN — normalises the
                                                   # data flow into the
                                                   # firing function
    if use_residual and L > 0:
        z_L = z_L + h_{L-1}                        # pre-spike residual
                                                   # (spike shortcut, keeps
                                                   #  the surrogate-gradient
                                                   #  path from vanishing)
    spikes_L = LIFCell_L(z_L)                       # cell membrane u_L
                                                   # updated internally
    h_L      = spikes_L                             # feeds next layer

Output:  h_{N-1}                    — spikes from the last cell
Belief:  cells[-1].membrane        — deepest layer's pre-reset u_t

References
──────────
    * MPBN: Guo et al., "Membrane Potential Batch Normalization for
      Spiking Neural Networks", ICCV 2023. Applied to the input driving
      each LIF integrator (not to the membrane post-integrate) — this
      matches the ASP paper's own S3DIS pipeline where MPBN is used the
      same way in `S3DIS/models/lif.py`.
    * Pre-spike residual (Spikingformer): Zhou et al., "Spikingformer:
      Spike-driven Residual Learning for Transformer-based SNN", 2023.
    * Multi-layer LIF stack: Fang et al., "Incorporating Learnable
      Membrane Time Constants…", ICCV 2021 (PLIF) — this file uses
      per-layer, per-channel learnable α via asp.lif.LIFCell.

Numerical notes
───────────────
    * Under bf16 AMP, PyTorch usually keeps BN in fp32 and returns fp32.
      The residual add is written as `z + h.to(z.dtype)` to prevent a
      dtype-mismatch RuntimeError.
    * MPBN folds into the LIF threshold at inference (running-mean and
      running-var become a rescale+offset of the threshold), so there is
      ZERO inference-time cost from MPBN.

Firing-rate reporting
─────────────────────
    * `last_firing_rate` is the mean firing rate of the LAST layer's
      spike output (matches the encoder's convention and what the
      existing composite loss consumes via `out["firing_rate"]`).
    * `mean_firing_rate` is the mean across ALL layers. Not currently
      wired into the loss but exposed for α accounting later.

Compatibility
─────────────
    * Uses `LIFCell` from `asp.lif`. If `n_layers=1`, `use_mpbn=False`,
      and `use_residual=False`, this module behaves IDENTICALLY to a
      standalone LIFCell with an extra Linear pre-projection.
    * `reset_state(B, device, dtype)` iterates over all cells.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .lif import LIFCell


class MultiLayerLIFHead(nn.Module):
    """N-layer stacked LIF head. Drop-in replacement for LIFCell in ASPModel."""

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

        # Per-layer linear projection (no bias — BN or LIF handles bias).
        self.linears = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(n_layers)
        ])

        # Per-layer MPBN (or Identity to skip).
        if self.use_mpbn:
            self.mpbns = nn.ModuleList([
                nn.BatchNorm1d(dim) for _ in range(n_layers)
            ])
        else:
            self.mpbns = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

        # Per-layer LIF cell.
        # Uses asp.lif.LIFCell so the exact learnable-alpha and soft-reset
        # semantics of the original single-layer head are preserved.
        self.cells = nn.ModuleList([
            LIFCell(dim, tau=tau, v_th=v_th, learnable=learnable,
                    sg_slope=sg_slope)
            for _ in range(n_layers)
        ])

        # Firing-rate tracking (populated in forward; helpful for α reporting).
        self._layer_frs: list[torch.Tensor] = []

    # -----------------------------------------------------------------
    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        """Reset all cells' membrane and post-reset carry to zeros."""
        for cell in self.cells:
            cell.reset_state(batch, device, dtype)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) -> spikes (B, D) from the FINAL layer."""
        h = x
        self._layer_frs = []
        for L in range(self.n_layers):
            z = self.linears[L](h)
            if self.use_mpbn:
                z = self.mpbns[L](z)
            # Pre-spike residual. Layer 0 has no residual because the input
            # `x` is a real-valued fused feature from self.proj(e_t) — adding
            # it would mix scales badly and is not what the reference
            # architectures do.
            if self.use_residual and L > 0:
                # Cast the residual to the current dtype to survive AMP
                # BN's fp32 promotion. If dtypes already match this is a no-op.
                z = z + h.to(z.dtype)
            spikes = self.cells[L](z)
            # Track firing rate per layer.
            self._layer_frs.append(spikes.mean())
            h = spikes
        return h

    # -----------------------------------------------------------------
    @property
    def membrane(self) -> torch.Tensor | None:
        """Belief state read by the SSP: LAST layer's pre-reset u_t."""
        return self.cells[-1].membrane

    # -----------------------------------------------------------------
    @property
    def last_firing_rate(self) -> torch.Tensor:
        """Firing rate of the FINAL LIF layer for this most recent forward."""
        if not self._layer_frs:
            return torch.zeros((), device=next(self.parameters()).device)
        return self._layer_frs[-1]

    # -----------------------------------------------------------------
    @property
    def mean_firing_rate(self) -> torch.Tensor:
        """Mean firing rate across ALL LIF layers (for α reporting)."""
        if not self._layer_frs:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(self._layer_frs).mean()