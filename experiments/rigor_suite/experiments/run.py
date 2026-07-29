"""Unified experiment runner for all ablations (A1-A5) and strengthening
experiments (S1-S6) on every dataset.

Usage:
    python -m experiments.run --base configs/base/modelnet40.yaml \
        --exp configs/ablations/A1_theta.yaml [--seeds 0 1 2 3 4] [--epochs N]
    python -m experiments.run --base configs/base/cifar100.yaml \
        --exp configs/strengthening/S1_occlusion.yaml

Outputs: results/<experiment>/<dataset>/seed<k>/<variant>/summary.json (+model.pt)
and a flat rows.csv per experiment for aggregation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asp import metrics as M                              # noqa: E402
from asp.datasets import build_dataset, corruptions       # noqa: E402
from asp.eval import evaluate                             # noqa: E402
from asp.model import ASPConfig, ASPModel                 # noqa: E402
from asp.train import train_model                         # noqa: E402


def make_loaders(cfg, corruption_fn=None, severity=0):
    from torch.utils.data import DataLoader
    c = dict(cfg)
    c["_corruption_fn"], c["_severity"] = corruption_fn, severity
    tr = build_dataset(cfg["dataset"], "train", dict(cfg))
    te = build_dataset(cfg["dataset"], "test", c)
    bs = cfg.get("batch_size", 64)
    nw = cfg.get("num_workers", 0)
    return (DataLoader(tr, bs, shuffle=True, num_workers=nw, drop_last=True),
            DataLoader(te, bs, shuffle=False, num_workers=nw))


def summarize(model, ev, cfg, thetas):
    rows = M.theta_sweep(ev["raw"]["logits"], ev["raw"]["margins"],
                         ev["raw"]["labels"], thetas)
    for r in rows:
        r.update(M.energy_proxy_pj(r["avg_slices"], cfg.get("points_per_slice", 64),
                                   cfg.get("enc_hidden", 64), cfg["d_model"],
                                   cfg["k_slices"], cfg["d_ssp"]))
        C = cfg["num_classes"]
        r["risk_bound_thm1"] = (C - 1) * (1 - r["theta"]) / C
        r["bound_satisfied"] = r["risk_exited"] <= r["risk_bound_thm1"] + 1e-9
    return {"acc_full_T": ev["acc_full_T"], "theta_rows": rows,
            "revisit": ev["revisit"], "drift": ev["drift"], "ece": ev["ece"],
            "aurc": ev["risk_coverage"]["aurc"],
            "ssp_param_count": model.ssp_param_count(),
            "total_param_count": model.total_param_count(),
            "ssp_param_frac": model.ssp_param_count() / model.total_param_count()}


@torch.no_grad()
def forced_start_eval(model, loader, device, force_steps: int, theta: float):
    """S3: first `force_steps` selections forced to the policy's WORST-ranked
    unvisited slice, then the policy resumes. Returns accuracy + avg slices."""
    model.eval()
    accs, exits = [], []
    for regions, desc, anchors, labels in loader:
        regions, desc, labels = regions.to(device), desc.to(device), labels.to(device)
        anchors = anchors.to(device) if anchors is not None else None
        from asp.geometry import mask_descriptor
        d = mask_descriptor(desc, model.cfg.drop_desc)
        B, K = regions.shape[0], model.cfg.k_slices
        model.head.reset_state(B, device); model.readout.reset_state(B, device)
        feats = model.encoder(regions, anchors)
        visited = torch.zeros(B, K, dtype=torch.bool, device=device)
        u = model.head.membrane
        exit_step = torch.full((B,), K, dtype=torch.long, device=device)
        exit_logits = torch.zeros(B, model.cfg.num_classes, device=device)
        for t in range(K):
            s = model.ssp.scores(u, d, visited)
            if t < force_steps:                       # adversarial: pick the worst
                masked_min = s.masked_fill(visited, 1e9)
                idx = masked_min.argmin(-1)
            else:
                idx = s.argmax(-1)
            w = torch.nn.functional.one_hot(idx, K).float()
            e = torch.einsum("bk,bkd->bd", w, feats)
            logits = model.readout(model.head(model.proj(e)))
            margin, _ = model._margin_entropy(logits)
            newly = (margin > theta) & (exit_step == K)
            exit_step[newly] = t + 1; exit_logits[newly] = logits[newly]
            visited = visited | (w > 0.5); u = model.head.membrane
        never = exit_step == K
        exit_logits[never] = logits[never]
        accs.append((exit_logits.argmax(-1) == labels).float())
        exits.append(exit_step.float())
    return {"accuracy": torch.cat(accs).mean().item(),
            "avg_slices": torch.cat(exits).mean().item()}


def run_variant(base_cfg, variant, exp, seed, device, out_dir, epochs_override):
    cfg = copy.deepcopy(base_cfg)
    cfg.update({k: v for k, v in variant.items() if k != "name"})
    if epochs_override:
        cfg["epochs"] = epochs_override
    torch.manual_seed(seed)
    vdir = os.path.join(out_dir, f"seed{seed}", variant["name"])
    os.makedirs(vdir, exist_ok=True)
    thetas = exp.get("eval_thetas", [cfg.get("theta", 0.7)])
    rows_out = []

    if exp["type"] == "transfer" and variant["name"] == "transfer_frozen":
        tr, te = make_loaders(cfg)
        model = ASPModel(ASPConfig.from_dict(cfg)).to(device)
        ck = torch.load(exp["source_ckpt"], map_location=device)
        src = {k: v for k, v in ck.items() if k.startswith("ssp.")
               and k in model.state_dict()
               and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(src, strict=False)
        for n, p in model.ssp.named_parameters():
            p.requires_grad = False
        model, hist = _train_prebuilt(model, cfg, tr, te, device)
    else:
        tr, te = make_loaders(cfg)
        model, hist = train_model(cfg, tr, te, device)

    ev = evaluate(model, te, device, thetas)
    summary = summarize(model, ev, cfg, thetas)
    summary.update({"variant": variant["name"], "seed": seed,
                    "experiment": exp["experiment"], "dataset": cfg["dataset"]})

    if exp["type"] == "eval_corruption":
        fn = corruptions.POINT_CORRUPTIONS.get(exp["corruption"]) \
            if cfg["modality"] == "points" else \
            corruptions.PATCH_CORRUPTIONS.get(exp["corruption"])
        corr_rows = []
        for sev in exp["severities"]:
            _, te_c = make_loaders(cfg, fn if sev > 0 else None, sev)
            ev_c = evaluate(model, te_c, device, thetas)
            r = M.theta_sweep(ev_c["raw"]["logits"], ev_c["raw"]["margins"],
                              ev_c["raw"]["labels"], thetas)[0]
            r.update({"severity": sev, "acc_full_T": ev_c["acc_full_T"]})
            corr_rows.append(r)
        summary["corruption_rows"] = corr_rows

    if exp["type"] == "forced_start":
        summary["forced_rows"] = [
            dict(force_steps=f, **forced_start_eval(model, te, device, f, thetas[0]))
            for f in exp["force_steps"]]

    if exp.get("temperature_scaling"):
        summary["temp_scaled"] = _temp_scale(ev, thetas, cfg["num_classes"])

    per_cls = M.per_class_accuracy(ev["raw"]["logits"][:, -1].argmax(-1),
                                   ev["raw"]["labels"], cfg["num_classes"])
    es = M.exits_from_margins(ev["raw"]["margins"], thetas[0])
    cls_exit = [es[ev["raw"]["labels"] == c].float().mean().item()
                if (ev["raw"]["labels"] == c).any() else float("nan")
                for c in range(cfg["num_classes"])]
    summary["per_class_acc"] = per_cls
    summary["per_class_exit"] = cls_exit
    summary["exit_hist"] = torch.bincount(es, minlength=cfg["k_slices"] + 1).tolist()

    torch.save(model.state_dict(), os.path.join(vdir, "model.pt"))
    with open(os.path.join(vdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(vdir, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    for r in summary["theta_rows"]:
        rows_out.append({"experiment": exp["experiment"], "dataset": cfg["dataset"],
                         "variant": variant["name"], "seed": seed, **{
                             k: v for k, v in r.items() if not isinstance(v, list)},
                         "acc_full_T": summary["acc_full_T"],
                         "ssp_params": summary["ssp_param_count"],
                         "revisit_rate": summary["revisit"]["revisit_rate"],
                         "final_distinct": summary["revisit"]["final_distinct"],
                         "delta_hat": summary["drift"]["delta_hat"],
                         "ece": summary["ece"], "aurc": summary["aurc"]})
    return rows_out


def _train_prebuilt(model, cfg, tr, te, device):
    """train_model but with an externally built (partially frozen) model."""
    import asp.train as T
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg.get("lr", 1e-3))
    hist = []
    for ep in range(cfg.get("epochs", 30)):
        model.train()
        tau = T.anneal_tau(ep, cfg.get("epochs", 30))
        for regions, desc, anchors, labels in tr:
            regions, desc, labels = (regions.to(device), desc.to(device),
                                     labels.to(device))
            anchors = anchors.to(device)
            out = model.forward_train(regions, desc, anchors, tau)
            loss = T.composite_loss(out, labels, cfg.get("lambda_exit", 0.1),
                                    cfg.get("lambda_sparse", 0.01))
            opt.zero_grad(); loss.backward(); opt.step()
        hist.append({"epoch": ep, "tau": tau})
    return model, hist


def _temp_scale(ev, thetas, C):
    """Fit softmax temperature on half the test margins (proper: use val split)."""
    import torch.nn.functional as F
    logits, labels = ev["raw"]["logits"][:, -1], ev["raw"]["labels"]
    n = len(labels) // 2
    T_ = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T_], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits[:n] / T_.clamp_min(0.05), labels[:n])
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        p = F.softmax(ev["raw"]["logits"][n:] / T_.clamp_min(0.05), -1)
        top2 = p.topk(2, -1).values
        marg = top2[..., 0] - top2[..., 1]
        rows = M.theta_sweep(ev["raw"]["logits"][n:], marg, labels[n:], thetas)
    return {"temperature": T_.item(), "theta_rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    base_cfg = yaml.safe_load(open(a.base))
    exp = yaml.safe_load(open(a.exp))
    seeds = a.seeds if a.seeds is not None else base_cfg.get("seeds", [0])
    out_dir = os.path.join(a.out, exp["experiment"], base_cfg["dataset"])
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for variant in exp["sweep"]:
        for seed in seeds:
            print(f"=== {exp['experiment']} | {base_cfg['dataset']} | "
                  f"{variant['name']} | seed {seed} ===")
            all_rows += run_variant(base_cfg, variant, exp, seed, a.device,
                                    out_dir, a.epochs)
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(os.path.join(out_dir, "rows.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_rows)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
