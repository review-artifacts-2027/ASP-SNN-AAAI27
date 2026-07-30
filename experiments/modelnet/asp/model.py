"""Active Spiking Perception model: SSP-driven slice loop + LIF head + early exit.

Inference loop (paper Fig. 1):
    u_{t-1} --SSP--> select slice m* --encoder--> e_{m*} --LIF head--> logits y_t
    exit when margin P(top1) - P(top2) > theta  (optional entropy AND-criterion)

Phase-by-phase change log
─────────────────────────
    Phase 4:  `modality: "patches_conv"`         -> SpikingConvPatchStem
    Phase 5:  `head_type: "multi"`               -> MultiLayerLIFHead
    Phase 6:  `use_global_ctx: true`             -> global ctx pathway (default off)
    Phase 8:  `modality: "patches_conv_streaming"` -> SpikingPerPatchEncoder
              `use_streaming_inference: true`      -> lazy per-patch encoding

Phase 8 mechanics
─────────────────
The Phase 4 SpikingConvPatchStem runs a full 32x32 conv encoder, which
computes tokens for ALL K=16 patches in one pass and cannot be
partially executed. Its encoder cost (~24M MACs) is paid on every
inference regardless of early exit, so α_sys is ~0.7× (worse than the
ANN baseline).

Phase 8 introduces SpikingPerPatchEncoder — a small 3-stage conv that
processes each 8x8 patch independently. Two forward modes:

    encoder.forward(regions)                        # batched over all K
        used at TRAINING (need all K feats for the Gumbel-ST loop)
        and at NON-streaming inference (theta very high, τ = K).

    encoder.encode_selected(regions, sel_idx)       # one patch per batch item
        used at STREAMING inference: at each ASP step, only the argmax
        slice is encoded, saving encoder compute proportional to E[τ]/K.

Byte-identical semantics under both modes: the per-patch encoder has
no cross-patch dependencies, so `encode_selected(regions, k)` gives the
same result as `forward(regions)[:, k, :]`.

Trade-off recorded in the paper's Pareto table:
    * Phase 4 (SpikingConvPatchStem): higher accuracy, α_sys ~0.7×
    * Phase 8 (SpikingPerPatchEncoder + streaming): -2..3 pp accuracy,
      α_sys ~3-4× at θ=0.7.
"""
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
    modality: str = "points"          # points | patches | patches_conv | patches_conv_streaming
    num_classes: int = 8
    d_model: int = 128
    k_slices: int = 16
    points_per_slice: int = 64
    patch_dim: int = 192              # optional 8x8x3 image-patch interface
    enc_hidden: int = 64
    d_ssp: int = 64
    ssp_rank: int = 0                 # 0 = full W_k; >=6 keeps exact expressiveness (Thm 5)
    policy: str = "ssp"               # ssp | random | fixed | geometry_only
    use_mask: bool = True             # A2 ablation switch
    drop_desc: list = field(default_factory=list)   # A3 ablation, e.g. ["spread"]
    tau_mem: float = 2.0
    v_th: float = 1.0
    sg_slope: float = 4.0
    theta: float = 0.7                # margin exit threshold (A1 sweep)
    theta_entropy: float | None = None  # if set: AND H(p) < log(C)*theta_entropy
    # Phase 4 — Spiking Conv Patch Stem channel widths.
    conv_stage_channels: list = field(default_factory=lambda: [48, 96, 128, 128])
    # Phase 5 — Multi-layer LIF head config.
    head_type: str = "single"
    head_n_layers: int = 3
    head_use_mpbn: bool = True
    head_use_residual: bool = True
    # Phase 6 — Global-context pathway (default off).
    use_global_ctx: bool = False
    global_ctx_gate_init: float = 0.0
    # Phase 8 — Per-patch streaming encoder config + inference-time streaming.
    perpatch_stage_channels: list = field(default_factory=lambda: [32, 64, 128])
    #   `use_streaming_inference`: at inference, only encode the selected
    #   slice per step (requires a streamable encoder; currently only the
    #   'patches_conv_streaming' modality supports this).
    #   `stream_stop_when_all_exited`: break the inference loop as soon
    #   as every batch item has hit the margin exit criterion. Combined
    #   with the per-patch encoder, this is what actually reduces α_sys.
    use_streaming_inference: bool = False
    stream_stop_when_all_exited: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ASPConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ASPModel(nn.Module):
    def __init__(self, cfg: ASPConfig):
        super().__init__()
        self.cfg = cfg
        # ── Encoder dispatch ──────────────────────────────────────────
        if cfg.modality == "points":
            self.encoder = PointSliceEncoder(cfg.d_model, cfg.enc_hidden, cfg.sg_slope)
        elif cfg.modality == "patches":
            self.encoder = PatchSliceEncoder(cfg.patch_dim, cfg.d_model,
                                             cfg.enc_hidden, cfg.sg_slope)
        elif cfg.modality == "patches_conv":
            # Phase 4: full-image spiking conv patch stem.
            from .backbone import SpikingConvPatchStem
            self.encoder = SpikingConvPatchStem(
                patch_dim=cfg.patch_dim,
                k_slices=cfg.k_slices,
                d_model=cfg.d_model,
                stage_channels=cfg.conv_stage_channels,
                sg_slope=cfg.sg_slope,
            )
        elif cfg.modality == "patches_conv_streaming":
            # Phase 8: per-patch spiking conv encoder (streamable).
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

        # ── Head dispatch (Phase 5) ──────────────────────────────────
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

        # ── Global-context pathway (Phase 6, default off) ─────────────
        self.use_global_ctx = bool(cfg.use_global_ctx)
        if self.use_global_ctx:
            self.ctx_proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
            self.ctx_gate = nn.Parameter(
                torch.tensor(float(cfg.global_ctx_gate_init))
            )

        # ── Phase 8 streaming inference guard ────────────────────────
        # If streaming is requested but the encoder can't stream, error early.
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

    # ------------------------------------------------------------------ utils
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

    # --------------------------------------------------------------- training
    def forward_train(self, regions, desc, anchors_xyz=None, tau_gumbel: float = 1.0):
        """Run all K steps with differentiable Gumbel-ST selection.

        At training we always encode ALL K slices upfront (the Gumbel-
        softmax mix requires it). Streaming is an inference-only
        optimization.
        """
        desc, B = self._prep(regions, desc)
        K = self.cfg.k_slices
        feats = self.encoder(regions, anchors_xyz)             # (B,K,D) in parallel
        ctx = self._global_ctx(feats)                          # Phase 6 (None when off)
        visited = torch.zeros(B, K, dtype=torch.bool, device=regions.device)
        logits_all, sel_all, fr = [], [], []
        u_prev = self.head.membrane                            # zeros at t=0
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

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def forward_infer(self, regions, desc, anchors_xyz=None,
                      theta: float | None = None, max_steps: int | None = None,
                      keep_membrane: bool = False):
        """Hard-argmax loop; records the step at which each sample would exit.

        Two paths:
          A) Non-streaming (default): encoder runs on all K slices upfront,
             loop runs T=K steps recording margins per step.
          B) Streaming (Phase 8, `cfg.use_streaming_inference=True`):
             encoder is called once per SSP step on the SELECTED slice
             only. If all batch items have exited by step t, break early
             (`cfg.stream_stop_when_all_exited=True`). This gives:
                encoder_cost ≈ max_over_batch(τ_b) × per_patch_cost
             which drops α_sys inference cost substantially when
             E[τ] << K.
        """
        theta = self.cfg.theta if theta is None else theta
        desc, B = self._prep(regions, desc)
        K = self.cfg.k_slices
        T = max_steps or K
        device = regions.device
        visited = torch.zeros(B, K, dtype=torch.bool, device=device)
        exit_step = torch.full((B,), T, dtype=torch.long, device=device)
        exit_logits = torch.zeros(B, self.cfg.num_classes, device=device)
        logits_all, margins_all, sels, membranes = [], [], [], []

        # Encoder handling — differs by path.
        streaming = bool(self.cfg.use_streaming_inference)
        if not streaming:
            feats = self.encoder(regions, anchors_xyz)         # (B, K, D)
            ctx = self._global_ctx(feats)
        else:
            feats = None                                       # computed lazily
            # Global ctx requires ALL K tokens; incompatible with streaming.
            # We simply disable it in this path — training with streaming
            # inference should also set use_global_ctx=false to match.
            ctx = None
            if self.use_global_ctx:
                # Silently disable; the training config would have caught
                # this mismatch already, but be robust here too.
                ctx = None

        u_prev = self.head.membrane
        # ── Main loop ────────────────────────────────────────────────
        for t in range(T):
            scores = self.ssp.scores(u_prev, desc, visited)
            w = self.ssp.select(scores, hard_inference=True)
            sel_idx = w.argmax(-1)                             # (B,)

            # Get e_t via one of the two paths.
            if not streaming:
                e_t = torch.einsum("bk,bkd->bd", w, feats)
            else:
                # Streaming: encode ONLY the selected patch.
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

            # Phase 8: early-break the loop when all batch items have exited.
            # This is what actually converts encoder compute savings into
            # observable wall-clock / MAC savings.
            if streaming and self.cfg.stream_stop_when_all_exited:
                if bool((exit_step != T).all().item()):
                    break

        # For samples that never exited during the recorded steps, use the
        # last computed logits as the fallback exit.
        never = exit_step == T
        if len(logits_all) > 0:
            exit_logits[never] = logits_all[-1][never]

        out = {"logits": torch.stack(logits_all, 1),      # (B, T_actual, C)
               "margins": torch.stack(margins_all, 1),
               "selections": torch.stack(sels, 1),
               "exit_step": exit_step,
               "exit_logits": exit_logits}
        if keep_membrane:
            out["membranes"] = torch.stack(membranes, 1)
        return out
