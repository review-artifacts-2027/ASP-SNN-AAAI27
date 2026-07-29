"""Active Spiking Perception model: SSP-driven slice loop + LIF head + early exit.

Inference loop (paper Fig. 1):
    u_{t-1} --SSP--> select slice m* --encoder--> e_{m*} --LIF head--> logits y_t
    exit when margin P(top1) - P(top2) > theta  (optional entropy AND-criterion)

Training uses Gumbel-softmax (hard=True) straight-through selection and
computes all K per-step logits for the composite loss.
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
    modality: str = "points"          # points | patches
    num_classes: int = 8
    d_model: int = 128
    k_slices: int = 16
    points_per_slice: int = 64
    patch_dim: int = 192              # 8*8*3 for CIFAR
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

    @classmethod
    def from_dict(cls, d: dict) -> "ASPConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ASPModel(nn.Module):
    def __init__(self, cfg: ASPConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.modality == "points":
            self.encoder = PointSliceEncoder(cfg.d_model, cfg.enc_hidden, cfg.sg_slope)
        else:
            self.encoder = PatchSliceEncoder(cfg.patch_dim, cfg.d_model,
                                             cfg.enc_hidden, cfg.sg_slope)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.head = LIFCell(cfg.d_model, cfg.tau_mem, cfg.v_th, True, cfg.sg_slope)
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

    # --------------------------------------------------------------- training
    def forward_train(self, regions, desc, anchors_xyz=None, tau_gumbel: float = 1.0):
        """Run all K steps with differentiable Gumbel-ST selection.

        Returns dict: logits (B,T,C), sel_onehot (B,T,K), firing_rate (scalar).
        """
        desc, B = self._prep(regions, desc)
        K = self.cfg.k_slices
        feats = self.encoder(regions, anchors_xyz)             # (B,K,D) in parallel
        visited = torch.zeros(B, K, dtype=torch.bool, device=regions.device)
        logits_all, sel_all, fr = [], [], []
        u_prev = self.head.membrane                            # zeros at t=0
        for _t in range(K):
            scores = self.ssp.scores(u_prev, desc, visited)
            w = self.ssp.select(scores, hard_inference=False, tau=tau_gumbel)
            e_t = torch.einsum("bk,bkd->bd", w, feats)
            spikes = self.head(self.proj(e_t))
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

        The full trajectory is computed so one pass yields metrics for ANY
        theta (exit steps are re-derivable from the stored margins).
        """
        theta = self.cfg.theta if theta is None else theta
        desc, B = self._prep(regions, desc)
        K = self.cfg.k_slices
        T = max_steps or K
        feats = self.encoder(regions, anchors_xyz)
        visited = torch.zeros(B, K, dtype=torch.bool, device=regions.device)
        device = regions.device
        exit_step = torch.full((B,), T, dtype=torch.long, device=device)
        exit_logits = torch.zeros(B, self.cfg.num_classes, device=device)
        logits_all, margins_all, sels, membranes = [], [], [], []
        u_prev = self.head.membrane
        for t in range(T):
            scores = self.ssp.scores(u_prev, desc, visited)
            w = self.ssp.select(scores, hard_inference=True)
            sel_idx = w.argmax(-1)                             # (B,)
            e_t = torch.einsum("bk,bkd->bd", w, feats)
            spikes = self.head(self.proj(e_t))
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
        never = exit_step == T
        exit_logits[never] = logits_all[-1][never]
        out = {"logits": torch.stack(logits_all, 1),      # (B,T,C)
               "margins": torch.stack(margins_all, 1),    # (B,T)
               "selections": torch.stack(sels, 1),        # (B,T)
               "exit_step": exit_step,                    # (B,)
               "exit_logits": exit_logits}                # (B,C)
        if keep_membrane:
            out["membranes"] = torch.stack(membranes, 1)  # (B,T,D)
        return out
