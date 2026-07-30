import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_dim: int, out_dim: int,
                 leak: float = 0.9, threshold: float = 1.0,
                 learnable_params: bool = False,
                 use_mpbn: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.learnable_params = learnable_params
        self.use_mpbn = use_mpbn

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

        if learnable_params:
            leak = min(max(leak, 1e-4), 1 - 1e-4)
            raw_leak_init = math.log(leak / (1.0 - leak))

            raw_thr_init = math.log(math.expm1(threshold))
            self.raw_leak = nn.Parameter(torch.tensor(raw_leak_init))
            self.raw_threshold = nn.Parameter(torch.tensor(raw_thr_init))
        else:
            self.register_buffer('_leak', torch.tensor(float(leak)))
            self.register_buffer('_threshold', torch.tensor(float(threshold)))

        if use_mpbn:
            self.mpbn = nn.BatchNorm1d(out_dim)

    @property
    def leak(self):
        if self.learnable_params:
            return torch.sigmoid(self.raw_leak)
        return self._leak

    @property
    def threshold(self):
        if self.learnable_params:
            return F.softplus(self.raw_threshold)
        return self._threshold

    def step(self, x, u_prev, s_prev):
        leak = self.leak
        threshold = self.threshold

        inp = F.relu(self.bn(self.fc(x)))
        u_t = leak * u_prev + inp - threshold * s_prev

        u_for_spike = self.mpbn(u_t) if self.use_mpbn else u_t

        s_t = _spike(u_for_spike - threshold)
        return u_t, s_t


class SpikeRateLogger:
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
    def __init__(self, feat_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 3, leak: float = 0.9,
                 threshold: float = 1.0,
                 cls_head_dims: list = None,
                 cls_head_dropout: list = None,
                 learnable_params: bool = False,
                 use_mpbn: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.spike_monitor = None

        dims_in = [feat_dim] + [hidden_dim] * (num_layers - 1)
        self.cells = nn.ModuleList([
            LIFCell(d_in, hidden_dim, leak, threshold,
                    learnable_params=learnable_params, use_mpbn=use_mpbn)
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
        new_states = []
        x = e_t

        for i, (cell, (u, s)) in enumerate(zip(self.cells, states)):
            u_new, s_new = cell.step(x, u, s)
            new_states.append((u_new, s_new))

            if self.spike_monitor is not None:
                self.spike_monitor.record(i, s_new)

            if i == 0:
                x = u_new + self.shortcut(e_t)
            else:
                x = u_new + x

        logits = self.fc_out(x)
        u_last = new_states[-1][0]
        return logits, new_states, u_last
