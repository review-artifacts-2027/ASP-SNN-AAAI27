"""Full-observation, non-spiking teacher for logit distillation into ASP.

The student is constrained twice: it is spiking, and at exit step t it has seen
only t of the K slices. The natural teacher therefore removes exactly those two
constraints -- full-precision ReLU activations and all K slices visible at once
-- while consuming the *identical* (slices, desc, anchors) tensors, so no extra
data pipeline is needed and the logits align class-for-class.

Used by train_kd.py; `asp.train.composite_loss` consumes its logits via the
`teacher_logits` argument.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .geometry import DESC_DIM


class SliceTeacher(nn.Module):
    """PointNet-lite per slice -> pool over all K slices -> classifier."""

    def __init__(self, num_classes: int, d_model: int = 256, hidden: int = 128,
                 dropout: float = 0.4):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model), nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
        )
        # Per-slice feature = maxpool over its points, concatenated with the
        # 6-D geometry descriptor the SSP also sees.
        self.slice_mlp = nn.Sequential(
            nn.Linear(d_model + DESC_DIM, d_model), nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.d_model = d_model

    def forward(self, slices: torch.Tensor, desc: torch.Tensor,
                anchors_xyz: torch.Tensor) -> torch.Tensor:
        B, K, P, _ = slices.shape
        rel = slices - anchors_xyz.unsqueeze(2)              # local frame, as student
        x = self.point_mlp(rel.reshape(B * K * P, 3))
        x = x.reshape(B, K, P, self.d_model).amax(dim=2)     # (B,K,D) pool points
        x = torch.cat([x, desc], dim=-1)                     # (B,K,D+6)
        x = self.slice_mlp(x.reshape(B * K, -1)).reshape(B, K, self.d_model)
        pooled = torch.cat([x.amax(dim=1), x.mean(dim=1)], dim=-1)   # (B,2D)
        return self.head(pooled)
