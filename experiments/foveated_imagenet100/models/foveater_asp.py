"""
foveater_asp.py
===============
Image-domain ASP model based on the FoveaTer paper.

The existing ASP stack in this repository works on point-cloud slices. This
module keeps the same active perception interface while adapting it to
ImageNet-style images:

  image -> CNN feature map -> fixation-dependent foveated tokens
        -> transformer -> class-token attention -> next fixation

Main paper details retained here:
  - 14 x 14 feature map for 224 x 224 images.
  - Square foveated pooling regions with 1, 3, 5, and 7 receptive fields.
  - Up to 29 pooled tokens per fixation, padded inside a batch.
  - Class-token attention from the last block drives the next fixation.
  - An inhibition-of-return map discourages revisiting the same region.
  - Logits are produced from the average class-token state over fixations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_trunc_normal(tensor: torch.Tensor, std: float = 0.02) -> None:
    if hasattr(nn.init, "trunc_normal_"):
        nn.init.trunc_normal_(tensor, std=std)
    else:
        nn.init.normal_(tensor, std=std)


@dataclass(frozen=True)
class PoolSpec:
    dy: int
    dx: int
    kernel: int
    scale_id: int
    ring: int


def build_foveater_pool_specs() -> list[PoolSpec]:
    """
    Build a 49-center square foveation layout.

    The paper describes 49 pooling centers in a 27 x 27 visual field with
    receptive fields 1, 3, 5, and 7. This implementation uses a 7 x 7 lattice
    around the current fixation. The central 3 x 3 lattice cells are treated as
    foveal, no-pooling samples; outer cells use larger kernels.
    """
    offsets = [-9, -6, -3, 0, 3, 6, 9]
    specs: list[PoolSpec] = []

    for gy, dy in enumerate(offsets):
        for gx, dx in enumerate(offsets):
            ring = max(abs(gy - 3), abs(gx - 3))
            if ring <= 1:
                kernel, scale_id = 1, 0
            elif ring == 2:
                kernel, scale_id = 3, 1
            else:
                # Keep both large peripheral scales represented.
                is_corner = abs(gy - 3) == 3 and abs(gx - 3) == 3
                kernel, scale_id = (7, 3) if is_corner else (5, 2)
            specs.append(PoolSpec(dy=dy, dx=dx, kernel=kernel,
                                  scale_id=scale_id, ring=ring))

    specs.sort(key=lambda s: (s.ring, abs(s.dy) + abs(s.dx), s.dy, s.dx))
    return specs


class FoveaTerBackbone(nn.Module):
    """
    Small hybrid CNN stem that emits a 14 x 14 feature map for 224 x 224 input.

    This matches the FoveaTer transformer-front-end assumption without adding a
    heavyweight dependency on timm. It can be replaced by a stronger backbone
    later as long as it returns [B, embed_dim, H, W].
    """

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FoveationModule(nn.Module):
    """
    Fixation-dependent square pooling over a CNN feature map.

    Args:
        max_tokens: paper default is 29 active tokens after dropping pooling
            centers outside the original 14 x 14 image.
        feature_grid: target feature-map grid size. Inputs are pooled to this
            size by FoveaTerASP before this module runs.
    """

    def __init__(self, max_tokens: int = 29, feature_grid: int = 14):
        super().__init__()
        self.max_tokens = max_tokens
        self.feature_grid = feature_grid
        self.specs = build_foveater_pool_specs()

    def forward(
        self,
        features: torch.Tensor,
        fixations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, H, W]
            fixations: [B, 2] integer feature-map coordinates in (y, x)

        Returns:
            tokens: [B, max_tokens, C]
            valid_mask: [B, max_tokens], True for real tokens.
            centers: [B, max_tokens, 2], feature-map coordinates for tokens.
            scale_ids: [B, max_tokens], 0..3 for 1/3/5/7 pooling scales.
        """
        bsz, channels, height, width = features.shape
        device = features.device
        dtype = features.dtype

        all_tokens = torch.zeros(bsz, self.max_tokens, channels,
                                 device=device, dtype=dtype)
        valid_mask = torch.zeros(bsz, self.max_tokens,
                                 device=device, dtype=torch.bool)
        centers = torch.full((bsz, self.max_tokens, 2), -1,
                             device=device, dtype=torch.long)
        scale_ids = torch.zeros(bsz, self.max_tokens,
                                device=device, dtype=torch.long)

        fixations = fixations.to(device=device, dtype=torch.long)

        for b in range(bsz):
            fy = int(fixations[b, 0].clamp(0, height - 1).item())
            fx = int(fixations[b, 1].clamp(0, width - 1).item())

            out_i = 0
            for spec in self.specs:
                cy = fy + spec.dy
                cx = fx + spec.dx
                if cy < 0 or cy >= height or cx < 0 or cx >= width:
                    continue

                radius = spec.kernel // 2
                y0 = max(0, cy - radius)
                y1 = min(height, cy + radius + 1)
                x0 = max(0, cx - radius)
                x1 = min(width, cx + radius + 1)
                if y0 >= y1 or x0 >= x1:
                    continue

                pooled = features[b, :, y0:y1, x0:x1].mean(dim=(1, 2))
                all_tokens[b, out_i] = pooled
                valid_mask[b, out_i] = True
                centers[b, out_i, 0] = cy
                centers[b, out_i, 1] = cx
                scale_ids[b, out_i] = spec.scale_id

                out_i += 1
                if out_i >= self.max_tokens:
                    break

        return all_tokens, valid_mask, centers, scale_ids


class FoveaTerBlock(nn.Module):
    """Transformer encoder block that returns attention weights."""

    def __init__(
        self,
        dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.norm1(x)
        attn_out, attn = self.attn(
            q, q, q,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x, attn


class FoveaTerASP(nn.Module):
    """
    Foveated Transformer with ASP-style sequential inference.

    Public methods mirror the point-cloud ASP wrappers where possible:
        forward_active_train(images) -> logits_final, logits_all
        forward_active_infer(images, threshold) -> logits, exit_step, fixations
    """

    def __init__(
        self,
        num_classes: int = 1000,
        image_size: int = 224,
        feature_grid: int = 14,
        embed_dim: int = 192,
        depth: int = 9,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        max_fixations: int = 5,
        max_tokens: int = 29,
        dropout: float = 0.0,
        accumulator_decay: float = 0.5,
        ior_strength: float = 1.0,
        initial_fixation: str = "random",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.feature_grid = feature_grid
        self.embed_dim = embed_dim
        self.max_fixations = max_fixations
        self.initial_fixation = initial_fixation
        self.accumulator_decay = accumulator_decay
        self.ior_strength = ior_strength
        self.temporal_dim = embed_dim

        self.backbone = FoveaTerBackbone(embed_dim=embed_dim)
        self.foveation = FoveationModule(
            max_tokens=max_tokens,
            feature_grid=feature_grid,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, feature_grid * feature_grid, embed_dim)
        )
        self.scale_embed = nn.Embedding(4, embed_dim)

        self.blocks = nn.ModuleList([
            FoveaTerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.register_buffer("gumbel_tau", torch.tensor(1.0))
        self.register_buffer("last_fixation_history", torch.empty(0), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        _init_trunc_normal(self.cls_token)
        _init_trunc_normal(self.cls_pos)
        _init_trunc_normal(self.spatial_pos_embed)
        _init_trunc_normal(self.scale_embed.weight)
        nn.init.zeros_(self.head.bias)
        _init_trunc_normal(self.head.weight)

    def _feature_map(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.shape[-2:] != (self.feature_grid, self.feature_grid):
            features = F.adaptive_avg_pool2d(
                features, (self.feature_grid, self.feature_grid)
            )
        return features

    def _initial_fixations(
        self,
        batch_size: int,
        device: torch.device,
        mode: str | None = None,
        fixations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fixations is not None:
            return fixations.to(device=device, dtype=torch.long)

        mode = mode or self.initial_fixation
        if mode == "random":
            return torch.randint(
                low=0,
                high=self.feature_grid,
                size=(batch_size, 2),
                device=device,
            )
        if mode == "center":
            center = self.feature_grid // 2
            return torch.full((batch_size, 2), center,
                              device=device, dtype=torch.long)
        raise ValueError("initial fixation must be 'random' or 'center'")

    def _encode_fixation(
        self,
        features: torch.Tensor,
        fixations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, mask, centers, scale_ids = self.foveation(features, fixations)
        bsz = features.size(0)
        width = self.feature_grid

        safe_y = centers[..., 0].clamp(min=0, max=self.feature_grid - 1)
        safe_x = centers[..., 1].clamp(min=0, max=self.feature_grid - 1)
        flat = (safe_y * width + safe_x).long()
        spatial = self.spatial_pos_embed[0, flat]
        spatial = spatial * mask.unsqueeze(-1)
        scale = self.scale_embed(scale_ids) * mask.unsqueeze(-1)

        token_seq = tokens + spatial + scale
        cls = self.cls_token.expand(bsz, -1, -1) + self.cls_pos
        x = torch.cat([cls, token_seq], dim=1)

        cls_mask = torch.zeros(bsz, 1, device=features.device, dtype=torch.bool)
        key_padding_mask = torch.cat([cls_mask, ~mask], dim=1)

        last_attn = None
        for block in self.blocks:
            x, last_attn = block(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        cls_state = x[:, 0]
        if last_attn is None:
            token_attn = torch.zeros_like(mask, dtype=features.dtype)
        else:
            token_attn = last_attn.mean(dim=1)[:, 0, 1:]
            token_attn = token_attn.masked_fill(~mask, 0.0)

        return cls_state, token_attn, centers, mask

    def _update_maps(
        self,
        confidence_map: torch.Tensor,
        ior_map: torch.Tensor,
        fixations: torch.Tensor,
        centers: torch.Tensor,
        token_attn: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, height, width = confidence_map.shape
        confidence_map = confidence_map * self.accumulator_decay
        ior_map = ior_map * self.accumulator_decay

        for b in range(bsz):
            for j in range(centers.size(1)):
                if not bool(valid_mask[b, j].item()):
                    continue
                cy = int(centers[b, j, 0].item())
                cx = int(centers[b, j, 1].item())
                confidence_map[b, cy, cx] = (
                    confidence_map[b, cy, cx] + token_attn[b, j]
                )

            fy = int(fixations[b, 0].item())
            fx = int(fixations[b, 1].item())
            y0 = max(0, fy - 1)
            y1 = min(height, fy + 2)
            x0 = max(0, fx - 1)
            x1 = min(width, fx + 2)
            ior_map[b, y0:y1, x0:x1] = ior_map[b, y0:y1, x0:x1] + 1.0

        return confidence_map, ior_map

    def _next_fixation(
        self,
        confidence_map: torch.Tensor,
        ior_map: torch.Tensor,
    ) -> torch.Tensor:
        score = confidence_map - self.ior_strength * ior_map
        flat_idx = score.flatten(1).argmax(dim=-1)
        y = flat_idx // self.feature_grid
        x = flat_idx % self.feature_grid
        return torch.stack([y, x], dim=-1).long()

    def _logits_from_states(self, cls_states: list[torch.Tensor]) -> torch.Tensor:
        state = torch.stack(cls_states, dim=1).mean(dim=1)
        return self.head(state)

    def forward_active_train(
        self,
        images: torch.Tensor,
        max_fixations: int | None = None,
        initial_fixations: torch.Tensor | None = None,
        random_initial: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        max_fixations = max_fixations or self.max_fixations
        features = self._feature_map(images)
        bsz = images.size(0)
        device = images.device

        mode = "random" if random_initial else "center"
        fixations = self._initial_fixations(
            bsz, device, mode=mode, fixations=initial_fixations
        )
        confidence_map = torch.zeros(
            bsz, self.feature_grid, self.feature_grid, device=device
        )
        ior_map = torch.zeros_like(confidence_map)

        cls_states: list[torch.Tensor] = []
        logits_all: list[torch.Tensor] = []
        history: list[torch.Tensor] = []

        for step in range(max_fixations):
            history.append(fixations.detach())
            cls_state, token_attn, centers, valid_mask = self._encode_fixation(
                features, fixations
            )
            cls_states.append(cls_state)
            logits_all.append(self._logits_from_states(cls_states))

            if step + 1 < max_fixations:
                confidence_map, ior_map = self._update_maps(
                    confidence_map, ior_map, fixations,
                    centers, token_attn, valid_mask
                )
                fixations = self._next_fixation(confidence_map, ior_map)

        self.last_fixation_history = torch.stack(history, dim=1)
        return logits_all[-1], logits_all

    def forward_active_infer(
        self,
        images: torch.Tensor,
        threshold: float = 0.7,
        max_fixations: int | None = None,
        class_thresholds: torch.Tensor | None = None,
        initial_fixation: str = "center",
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        max_fixations = max_fixations or self.max_fixations
        features = self._feature_map(images)
        bsz = images.size(0)
        device = images.device

        fixations = self._initial_fixations(
            bsz, device, mode=initial_fixation, fixations=None
        )
        confidence_map = torch.zeros(
            bsz, self.feature_grid, self.feature_grid, device=device
        )
        ior_map = torch.zeros_like(confidence_map)

        cls_states: list[torch.Tensor] = []
        history: list[torch.Tensor] = []
        last_logits: torch.Tensor | None = None

        for step in range(max_fixations):
            history.append(fixations.detach())
            cls_state, token_attn, centers, valid_mask = self._encode_fixation(
                features, fixations
            )
            cls_states.append(cls_state)
            last_logits = self._logits_from_states(cls_states)

            probs = F.softmax(last_logits, dim=-1)
            pred = probs.argmax(dim=-1)
            conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
            if class_thresholds is not None:
                thresholds = class_thresholds.to(device=device)[pred]
            else:
                thresholds = torch.full_like(conf, threshold)
            if bool((conf >= thresholds).all().item()):
                self.last_fixation_history = torch.stack(history, dim=1)
                return last_logits, step + 1, self.last_fixation_history

            if step + 1 < max_fixations:
                confidence_map, ior_map = self._update_maps(
                    confidence_map, ior_map, fixations,
                    centers, token_attn, valid_mask
                )
                fixations = self._next_fixation(confidence_map, ior_map)

        self.last_fixation_history = torch.stack(history, dim=1)
        assert last_logits is not None
        return last_logits, max_fixations, self.last_fixation_history

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            logits, _ = self.forward_active_train(
                images, random_initial=(self.initial_fixation == "random")
            )
            return logits
        logits, _, _ = self.forward_active_infer(
            images,
            max_fixations=self.max_fixations,
            initial_fixation="center",
        )
        return logits

    def set_gumbel_tau(self, tau: float) -> None:
        # Kept for compatibility with existing ASP training utilities.
        self.gumbel_tau.fill_(float(tau))

    def reset_state(self, batch_size: int, device=None) -> None:
        # The image FoveaTer path is transformer-based and stateless.
        return None

    def get_firing_rates(self) -> dict:
        return {}

    def mean_firing_rate(self) -> float:
        return 0.0

    def param_count(self) -> dict:
        backbone = sum(p.numel() for p in self.backbone.parameters())
        foveation = 0
        transformer = sum(p.numel() for p in self.blocks.parameters())
        embeddings = (
            self.cls_token.numel()
            + self.cls_pos.numel()
            + self.spatial_pos_embed.numel()
            + sum(p.numel() for p in self.scale_embed.parameters())
        )
        norm = sum(p.numel() for p in self.norm.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "backbone": backbone,
            "foveation": foveation,
            "transformer": transformer,
            "embeddings": embeddings,
            "norm": norm,
            "head": head,
            "total": total,
        }


def build_foveater_asp_tiny(num_classes: int = 1000, **kwargs) -> FoveaTerASP:
    """Convenience builder matching the paper's DeiT-Tiny scale."""
    defaults = {
        "embed_dim": 192,
        "depth": 9,
        "num_heads": 3,
        "max_fixations": 5,
        "max_tokens": 29,
    }
    defaults.update(kwargs)
    return FoveaTerASP(num_classes=num_classes, **defaults)
