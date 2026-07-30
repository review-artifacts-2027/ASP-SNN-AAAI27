"""
models/lif.py — Multi-layer Leaky Integrate-and-Fire temporal head.

Three stacked LIF cells with residual connections and surrogate gradients.
Produces per-timestep outputs through the residual chain (NOT raw membrane).

Dynamics (soft reset):
    u_t = leak * u_prev + inp - threshold * s_prev
    s_t = Heaviside(u_t - threshold)  [surrogate: ATan]

Surrogate gradient — ATan:
    Reference: Fang et al. ICCV 2021 "Incorporating Learnable Membrane
    Time Constants to Enhance Learning of Spiking Neural Networks"
    Used by SPM (our reference backbone) and all modern SNN papers.
    g(x) = alpha / (2 * (1 + (alpha * pi * x / 2)^2)), alpha=2.0

Learnable LIF (Novel contribution):
    leak and threshold are per-neuron nn.Parameters, constrained via
    sigmoid (leak ∈ (0,1)) and softplus (threshold > 0). This allows
    the network to learn optimal membrane dynamics per-neuron.

Neuromorphic Augmentation — Temporal Spike Dropout:
    During training, spikes are randomly zeroed with probability
    spike_dropout. This simulates neuromorphic sensor event dropout
    and acts as a regulariser specific to spiking networks.

AMP safety: Custom autograd forward/backward are decorated with
    @custom_fwd(cast_inputs=torch.float32) / @custom_bwd to prevent
    fp16 overflow in the pow(2) term when AMP is enabled.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Backward-compatible AMP decorators (PyTorch 2.4+ vs 2.1-2.3)
try:
    from torch.amp import custom_fwd, custom_bwd
    _amp_fwd = custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    _amp_bwd = custom_bwd(device_type="cuda")
except (ImportError, TypeError):
    try:
        from torch.cuda.amp import custom_fwd, custom_bwd
        _amp_fwd = custom_fwd(cast_inputs=torch.float32)
        _amp_bwd = custom_bwd()
    except ImportError:
        _amp_fwd = lambda fn: fn
        _amp_bwd = lambda fn: fn

_ATAN_ALPHA = 2.0


class _ATanSurrogateSpike(torch.autograd.Function):
    """Heaviside with ATan surrogate gradient (AMP-safe)."""

    @staticmethod
    @_amp_fwd
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    @_amp_bwd
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        alpha = _ATAN_ALPHA
        denom = 1.0 + (alpha * math.pi * x / 2.0).pow(2)
        return grad * (alpha / (2.0 * denom))


def _spike(x: torch.Tensor) -> torch.Tensor:
    return _ATanSurrogateSpike.apply(x)


class LIFCell(nn.Module):
    """
    Single LIF neuron layer with soft reset and learnable dynamics.

    leak      : per-neuron, learned via sigmoid so it stays in (0, 1)
    threshold : per-neuron, learned via softplus so it stays > 0
    spike_dropout: neuromorphic augmentation — randomly zeros spikes
                   during training to simulate sensor event loss
    """

    def __init__(self, in_dim: int, out_dim: int,
                 leak: float = 0.9, threshold: float = 1.0,
                 spike_dropout: float = 0.0,
                 lif_learnable: bool = True):
        super().__init__()
        self.out_dim        = out_dim
        self.spike_dropout  = spike_dropout
        self.lif_learnable  = lif_learnable
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

        if lif_learnable:
            # Learnable per-neuron leak: sigmoid(log_leak) ∈ (0, 1)
            init_log_leak = math.log(leak / (1.0 - leak + 1e-6))
            self.log_leak = nn.Parameter(
                torch.full((out_dim,), init_log_leak)
            )
            # Learnable per-neuron threshold: softplus(raw_thr) > 0
            init_raw_thr = math.log(math.exp(threshold) - 1.0 + 1e-6)
            self.raw_threshold = nn.Parameter(
                torch.full((out_dim,), init_raw_thr)
            )
        else:
            # Fixed LIF — constants, not trained (ablation baseline)
            self.register_buffer('_fixed_leak',      torch.tensor(leak))
            self.register_buffer('_fixed_threshold', torch.tensor(threshold))

    @property
    def eff_leak(self) -> torch.Tensor:
        """Effective leak ∈ (0, 1) for current batch device."""
        if self.lif_learnable:
            return torch.sigmoid(self.log_leak)
        return self._fixed_leak

    @property
    def eff_threshold(self) -> torch.Tensor:
        """Effective threshold > 0 for current batch device."""
        if self.lif_learnable:
            return F.softplus(self.raw_threshold)
        return self._fixed_threshold

    def step(self, x: torch.Tensor, u_prev: torch.Tensor,
             s_prev: torch.Tensor):
        """One timestep: x [B, in_dim] -> (u_new, s_new) each [B, out_dim]."""
        inp = F.relu(self.bn(self.fc(x)))
        leak = self.eff_leak          # [out_dim]
        thr  = self.eff_threshold     # [out_dim]
        u_t  = leak * u_prev + inp - thr * s_prev
        s_t  = _spike(u_t - thr)

        # Neuromorphic Augmentation: Temporal Spike Dropout
        if self.training and self.spike_dropout > 0.0:
            keep = (torch.rand_like(s_t) >= self.spike_dropout).float()
            s_t  = s_t * keep

        return u_t, s_t


class SpikeRateLogger:
    """
    Records per-layer spike counts for energy efficiency reporting.

    Usage:
        logger = SpikeRateLogger()
        model.lif_head.spike_monitor = logger
        evaluate(model, ...)
        print(logger.mean_rate(), logger.per_layer_rates())
        model.lif_head.spike_monitor = None
    """

    def __init__(self):
        self.counts = {}
        self.totals = {}

    def record(self, layer_idx: int, spikes: torch.Tensor):
        n = spikes.numel()
        s = spikes.sum().item()
        self.counts[layer_idx] = self.counts.get(layer_idx, 0) + s
        self.totals[layer_idx] = self.totals.get(layer_idx, 0) + n

    def mean_rate(self) -> float:
        if not self.totals:
            return 0.0
        return sum(self.counts.values()) / sum(self.totals.values())

    def per_layer_rates(self) -> dict:
        return {k: self.counts[k] / self.totals[k]
                for k in sorted(self.totals.keys())}

    def reset(self):
        self.counts.clear()
        self.totals.clear()


class MultiLayerLIF(nn.Module):
    """
    3-layer LIF head with residual connections and learnable dynamics.

    Input:  e_t [B, feat_dim]  (fused slice feature per ASP step)
    Output: logits [B, num_classes], new_states, u_last [B, hidden_dim]

    spike_dropout: neuromorphic augmentation probability passed to each LIFCell.
    """

    def __init__(self, feat_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 3, leak: float = 0.9,
                 threshold: float = 1.0,
                 spike_dropout: float = 0.0,
                 lif_learnable: bool = True,
                 cls_head_dims: list = None,
                 cls_head_dropout: list = None):
        self._spike_sum   = 0.0
        self._spike_total = 0
        super().__init__()
        self.num_layers   = num_layers
        self.hidden_dim   = hidden_dim
        self.spike_monitor = None
        self._spike_sum   = 0.0
        self._spike_total = 0

        dims_in = [feat_dim] + [hidden_dim] * (num_layers - 1)
        self.cells = nn.ModuleList([
            LIFCell(d_in, hidden_dim, leak, threshold,
                    spike_dropout=spike_dropout,
                    lif_learnable=lif_learnable)
            for d_in in dims_in
        ])

        self.shortcut = (
            nn.Linear(feat_dim, hidden_dim, bias=False)
            if feat_dim != hidden_dim else nn.Identity()
        )

        if cls_head_dims and len(cls_head_dims) > 0:
            if cls_head_dropout is None:
                cls_head_dropout = [0.5] * len(cls_head_dims)
            layers = []
            prev_dim = hidden_dim
            for dim, drop in zip(cls_head_dims, cls_head_dropout):
                layers.extend([
                    nn.Linear(prev_dim, dim, bias=False),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(drop),
                ])
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, num_classes))
            self.fc_out = nn.Sequential(*layers)
        else:
            self.fc_out = nn.Linear(hidden_dim, num_classes)

    def init_state(self, B: int, device: torch.device) -> list:
        return [
            (torch.zeros(B, self.hidden_dim, device=device),
             torch.zeros(B, self.hidden_dim, device=device))
            for _ in range(self.num_layers)
        ]

    def step(self, e_t: torch.Tensor, states: list):
        """
        One ASP timestep through all LIF cells.

        Returns:
            logits     : [B, num_classes]
            new_states : updated (u, s) per layer
            u_last     : [B, hidden_dim] — last cell MEMBRANE for belief
                         (NOT the residual chain output x which goes to fc_out)
        """
        new_states = []
        x = e_t

        for i, (cell, (u, s)) in enumerate(zip(self.cells, states)):
            u_new, s_new = cell.step(x, u, s)
            new_states.append((u_new, s_new))

            # Track firing rate for energy efficiency reporting
            self._spike_sum   += s_new.sum().item()
            self._spike_total += s_new.numel()

            if self.spike_monitor is not None:
                self.spike_monitor.record(i, s_new)

            if i == 0:
                x = u_new + self.shortcut(e_t)
            else:
                x = u_new + x

        logits = self.fc_out(x)
        u_last = new_states[-1][0]
        return logits, new_states, u_last

    def mean_firing_rate(self) -> float:
        """
        Returns mean spike firing rate accumulated since last reset.
        Called by train_s3dis.py to compute the sparsity regularisation loss.
        """
        if self._spike_total == 0:
            return 0.0
        return self._spike_sum / self._spike_total

    def reset_spike_stats(self):
        """Reset counters at the start of each epoch."""
        self._spike_sum   = 0.0
        self._spike_total = 0