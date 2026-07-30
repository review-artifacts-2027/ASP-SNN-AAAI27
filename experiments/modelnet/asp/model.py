from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import PatchSliceEncoder, PointSliceEncoder
from .geometry import DESC_DIM, mask_descriptor
from .lif import LIFCell, LeakyReadout
from .ssp import SSP


@dataclass
class ASPConfig:
    modality: str = "points"
    num_classes: int = 8
    d_model: int = 128
    k_slices: int = 16
    points_per_slice: int = 64
    patch_dim: int = 192
    enc_hidden: int = 64
    d_ssp: int = 64
    ssp_rank: int = 0
    policy: str = "ssp"
    use_mask: bool = True
    drop_desc: list = field(default_factory=list)
    tau_mem: float = 2.0
    v_th: float = 1.0
    sg_slope: float = 4.0
    theta: float = 0.7
    theta_entropy: float | None = None

    conv_stage_channels: list = field(default_factory=lambda: [48, 96, 128, 128])

    head_type: str = "single"
    head_n_layers: int = 3
    head_use_mpbn: bool = True
    head_use_residual: bool = True

    use_global_ctx: bool = False
    global_ctx_gate_init: float = 0.0

    perpatch_stage_channels: list = field(default_factory=lambda: [32, 64, 128])

    use_streaming_inference: bool = False
    stream_stop_when_all_exited: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ASPConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ASPModel(nn.Module):
    def __init__(self, cfg: ASPConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.modality == "points":
            self.encoder = PointSliceEncoder(cfg.d_model, cfg.enc_hidden, cfg.sg_slope)
        elif cfg.modality == "patches":
            self.encoder = PatchSliceEncoder(cfg.patch_dim, cfg.d_model,
                                             cfg.enc_hidden, cfg.sg_slope)
        elif cfg.modality == "patches_conv":
            from .backbone import SpikingConvPatchStem
            self.encoder = SpikingConvPatchStem(
                patch_dim=cfg.patch_dim,
                k_slices=cfg.k_slices,
                d_model=cfg.d_model,
                stage_channels=cfg.conv_stage_channels,
                sg_slope=cfg.sg_slope,
            )
        elif cfg.modality == "patches_conv_streaming":
            from .backbone import SpikingPerPatchEncoder
            self.encoder = SpikingPerPatchEncoder(
                patch_dim=cfg.patch_dim,
                k_slices=cfg.k_slices,
                d_model=cfg.d_model,
                stage_channels=cfg.perpatch_stage_channels,
                sg_slope=cfg.sg_slope,
            )
        else:
            raise ValueError(
                f"unknown modality: {cfg.modality!r}. Expected one of "
                f"'points', 'patches', 'patches_conv', 'patches_conv_streaming'."
            )
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

        if cfg.head_type == "single":
            self.head = LIFCell(cfg.d_model, cfg.tau_mem, cfg.v_th,
                                True, cfg.sg_slope)
        elif cfg.head_type == "multi":
            from .multi_head import MultiLayerLIFHead
            self.head = MultiLayerLIFHead(
                dim=cfg.d_model, n_layers=cfg.head_n_layers,
                tau=cfg.tau_mem, v_th=cfg.v_th, learnable=True,
                sg_slope=cfg.sg_slope,
                use_mpbn=cfg.head_use_mpbn,
                use_residual=cfg.head_use_residual,
            )
        else:
            raise ValueError(
                f"unknown head_type: {cfg.head_type!r}. Expected 'single' or 'multi'."
            )

        self.use_global_ctx = bool(cfg.use_global_ctx)
        if self.use_global_ctx:
            self.ctx_proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
            self.ctx_gate = nn.Parameter(
                torch.tensor(float(cfg.global_ctx_gate_init))
            )

        if cfg.use_streaming_inference and not hasattr(self.encoder, "encode_selected"):
            raise ValueError(
                f"use_streaming_inference=True but encoder "
                f"{type(self.encoder).__name__} does not implement "
                f"encode_selected(). Streaming requires "
                f"modality='patches_conv_streaming' (or another streamable "
                f"encoder). Fix: change modality, or set "
                f"use_streaming_inference=false."
            )

        self.readout = LeakyReadout(cfg.d_model, cfg.num_classes, cfg.tau_mem)
        self.ssp = SSP(cfg.d_model, DESC_DIM, cfg.d_ssp, cfg.ssp_rank,
                       cfg.policy, cfg.use_mask)

    def ssp_param_count(self) -> int:
        return self.ssp.param_count()

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _prep(self, regions, desc):
        desc = mask_descriptor(desc, self.cfg.drop_desc)
        B = regions.shape[0]
        self.head.reset_state(B, regions.device)
        self.readout.reset_state(B, regions.device)
        return desc, B

    @staticmethod
    def _margin_entropy(logits: torch.Tensor):
        p = F.softmax(logits, dim=-1)
        top2 = p.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(p.clamp_min(1e-12).log() * p).sum(-1)
        return margin, entropy

    def _global_ctx(self, feats: torch.Tensor) -> torch.Tensor | None:
        if not self.use_global_ctx:
            return None
        mean_ctx = feats.mean(dim=1)
        max_ctx  = feats.amax(dim=1)
        combined = torch.cat([mean_ctx, max_ctx], dim=-1)
        return self.ctx_proj(combined)

    def forward_train(self, regions, desc, anchors_xyz=None, tau_gumbel: float = 1.0):
        desc, B = self._prep(regions, desc)
        K = self.cfg.k_slices
        feats = self.encoder(regions, anchors_xyz)
        ctx = self._global_ctx(feats)
        visited = torch.zeros(B, K, dtype=torch.bool, device=regions.device)
        logits_all, sel_all, fr = [], [], []
        u_prev = self.head.membrane
        for _t in range(K):
            scores = self.ssp.scores(u_prev, desc, visited)
            w = self.ssp.select(scores, hard_inference=False, tau=tau_gumbel)
            e_t = torch.einsum("bk,bkd->bd", w, feats)
            h_t = self.proj(e_t)
            if ctx is not None:
                h_t = h_t + self.ctx_gate * ctx
            spikes = self.head(h_t)
            logits_all.append(self.readout(spikes))
            sel_all.append(w)
            fr.append(spikes.mean())
            visited = visited | (w.detach() > 0.5)
            u_prev = self.head.membrane
        return {"logits": torch.stack(logits_all, 1),
                "sel_onehot": torch.stack(sel_all, 1),
                "firing_rate": torch.stack(fr).mean() + self.encoder.last_firing_rate}

    @torch.no_grad()
    def forward_infer(self, regions, desc, anchors_xyz=None,
                      theta: float | None = None, max_steps: int | None = None,
                      keep_membrane: bool = False):

        theta = self.cfg.theta if theta is None else theta
        desc, B = self._prep(regions, desc)
        K = self.cfg.k_slices
        T = max_steps or K
        device = regions.device
        visited = torch.zeros(B, K, dtype=torch.bool, device=device)
        exit_step = torch.full((B,), T, dtype=torch.long, device=device)
        exit_logits = torch.zeros(B, self.cfg.num_classes, device=device)
        logits_all, margins_all, sels, membranes = [], [], [], []

        streaming = bool(self.cfg.use_streaming_inference)
        if not streaming:
            feats = self.encoder(regions, anchors_xyz)
            ctx = self._global_ctx(feats)
        else:
            feats = None

            ctx = None
            if self.use_global_ctx:
                ctx = None

        u_prev = self.head.membrane

        for t in range(T):
            scores = self.ssp.scores(u_prev, desc, visited)
            w = self.ssp.select(scores, hard_inference=True)
            sel_idx = w.argmax(-1)

            if not streaming:
                e_t = torch.einsum("bk,bkd->bd", w, feats)
            else:
                e_t = self.encoder.encode_selected(regions, sel_idx)

            h_t = self.proj(e_t)
            if ctx is not None:
                h_t = h_t + self.ctx_gate * ctx

            spikes = self.head(h_t)
            logits = self.readout(spikes)
            margin, entropy = self._margin_entropy(logits)
            ok = margin > theta
            if self.cfg.theta_entropy is not None:
                import math
                ok = ok & (entropy < math.log(self.cfg.num_classes) * self.cfg.theta_entropy)
            newly = ok & (exit_step == T)
            exit_step[newly] = t + 1
            exit_logits[newly] = logits[newly]
            logits_all.append(logits)
            margins_all.append(margin)
            sels.append(sel_idx)
            if keep_membrane:
                membranes.append(self.head.membrane.clone())
            visited = visited | (w > 0.5)
            u_prev = self.head.membrane

            if streaming and self.cfg.stream_stop_when_all_exited:
                if bool((exit_step != T).all().item()):
                    break

        never = exit_step == T
        if len(logits_all) > 0:
            exit_logits[never] = logits_all[-1][never]

        out = {"logits": torch.stack(logits_all, 1),
               "margins": torch.stack(margins_all, 1),
               "selections": torch.stack(sels, 1),
               "exit_step": exit_step,
               "exit_logits": exit_logits}
        if keep_membrane:
            out["membranes"] = torch.stack(membranes, 1)
        return out
