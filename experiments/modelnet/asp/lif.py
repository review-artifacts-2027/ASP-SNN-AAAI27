"""Leaky Integrate-and-Fire neurons with surrogate gradients (pure PyTorch).

Membrane dynamics (proposal Sec. 4.2):
    u_t = alpha * u_{t-1} + (1 - alpha) * x_t,   alpha = sigmoid(raw_alpha)  (learnable, per-channel)
    o_t = H(u_t - V_th)                          (Heaviside; surrogate gradient in backward)
    u_t <- u_t - o_t * V_th                      (soft reset)

The *pre-reset* membrane is exposed as `.membrane` so the SSP can read the
belief state u_{t-1} exactly as defined in the paper.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _SurrogateSpike(torch.autograd.Function):
    """Heaviside forward / scaled-sigmoid derivative backward (SG slope k)."""

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x)
        ctx.k = k
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        k = ctx.k
        s = torch.sigmoid(k * x)
        return grad_out * k * s * (1.0 - s), None


def spike_fn(x: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    return _SurrogateSpike.apply(x, k)


class LIFCell(nn.Module):
    """Stateful learnable-LIF cell driven one timestep at a time."""

    def __init__(self, dim: int, tau: float = 2.0, v_th: float = 1.0,
                 learnable: bool = True, sg_slope: float = 4.0):
        super().__init__()
        alpha0 = math.exp(-1.0 / tau)
        raw = math.log(alpha0 / (1.0 - alpha0))  # sigmoid^-1(alpha0)
        if learnable:
            self.raw_alpha = nn.Parameter(torch.full((dim,), raw))
        else:
            self.register_buffer("raw_alpha", torch.full((dim,), raw))
        self.v_th = v_th
        self.sg_slope = sg_slope
        self.dim = dim
        self.membrane = None  # pre-reset u_t (belief state read by the SSP)
        self._u = None        # post-reset carry

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        self._u = torch.zeros(batch, self.dim, device=device, dtype=dtype)
        self.membrane = self._u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._u is None or self._u.shape[0] != x.shape[0]:
            self.reset_state(x.shape[0], x.device, x.dtype)
        a = self.alpha
        u = a * self._u + (1.0 - a) * x
        self.membrane = u                      # u_t, pre-reset
        o = spike_fn(u - self.v_th, self.sg_slope)
        self._u = u - o * self.v_th            # soft reset
        return o


class LeakyReadout(nn.Module):
    """Non-spiking leaky integrator readout producing logits y_t."""

    def __init__(self, in_dim: int, num_classes: int, tau: float = 2.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        alpha0 = math.exp(-1.0 / tau)
        self.raw_alpha = nn.Parameter(torch.full((num_classes,),
                                                 math.log(alpha0 / (1 - alpha0))))
        self._v = None

    def reset_state(self, batch: int, device, dtype=torch.float32) -> None:
        self._v = torch.zeros(batch, self.fc.out_features, device=device, dtype=dtype)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        if self._v is None or self._v.shape[0] != spikes.shape[0]:
            self.reset_state(spikes.shape[0], spikes.device, spikes.dtype)
        a = torch.sigmoid(self.raw_alpha)
        self._v = a * self._v + (1.0 - a) * self.fc(spikes)
        return self._v
