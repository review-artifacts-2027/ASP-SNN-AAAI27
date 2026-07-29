"""Train ASP with a full-observation teacher (logit KD) + real data augmentation.

Stage 1: train `asp.teacher.SliceTeacher` (non-spiking, sees all K slices).
Stage 2: train the ASP student, adding KL(student_t || teacher) at every step t.

Emits the same artifact set as experiments/run.py so results are directly
comparable against the no-KD baseline:
    teacher.pt  model.pt  summary.json  history.json  teacher_history.json

Usage:
    python train_kd.py --config configs/run/modelnet40.yaml --out results_kd/... \
        --teacher-epochs 60 --lambda-kd 1.0 --kd-temp 4.0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asp import metrics as M                      # noqa: E402
from asp.datasets import build_dataset            # noqa: E402
from asp.eval import evaluate                     # noqa: E402
from asp.model import ASPConfig, ASPModel         # noqa: E402
from asp.teacher import SliceTeacher              # noqa: E402
from asp.train import anneal_tau, composite_loss  # noqa: E402
from experiments.run import summarize             # noqa: E402


def make_loaders(cfg):
    tr = build_dataset(cfg["dataset"], "train", dict(cfg))
    te = build_dataset(cfg["dataset"], "test", dict(cfg))
    bs, nw = cfg.get("batch_size", 128), cfg.get("num_workers", 0)
    return (DataLoader(tr, bs, shuffle=True, num_workers=nw, drop_last=True,
                       persistent_workers=nw > 0),
            DataLoader(te, bs, shuffle=False, num_workers=nw,
                       persistent_workers=nw > 0))


@torch.no_grad()
def teacher_accuracy(teacher, loader, device):
    teacher.eval()
    correct = total = 0
    for regions, desc, anchors, labels in loader:
        regions, desc, anchors, labels = (regions.to(device), desc.to(device),
                                          anchors.to(device), labels.to(device))
        pred = teacher(regions, desc, anchors).argmax(-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def train_teacher(cfg, tr, te, device, epochs, lr, log):
    t = SliceTeacher(cfg["num_classes"], d_model=cfg.get("teacher_d_model", 256),
                     hidden=cfg.get("teacher_hidden", 128)).to(device)
    opt = torch.optim.AdamW(t.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist, best = [], 0.0
    for ep in range(epochs):
        t.train()
        tot = n = 0
        t0 = time.time()
        for regions, desc, anchors, labels in tr:
            regions, desc, anchors, labels = (regions.to(device), desc.to(device),
                                              anchors.to(device), labels.to(device))
            loss = F.cross_entropy(t(regions, desc, anchors), labels,
                                   label_smoothing=0.2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * labels.numel(); n += labels.numel()
        sched.step()
        row = {"epoch": ep, "train_loss": tot / n, "sec": time.time() - t0}
        if ep % 5 == 0 or ep == epochs - 1:
            row["test_acc"] = teacher_accuracy(t, te, device)
            best = max(best, row["test_acc"])
        hist.append(row)
        log("[teacher %03d] " % ep + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()))
    return t, hist, best


def train_student_kd(cfg, teacher, tr, te, device, lambda_kd, kd_temp, log,
                     kd_schedule="ramp"):
    model = ASPModel(ASPConfig.from_dict(cfg)).to(device)
    epochs = cfg["epochs"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                           weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if teacher is not None:
        teacher.eval()
    hist = []
    for ep in range(epochs):
        model.train()
        tau = anneal_tau(ep, epochs, cfg.get("tau_start", 1.0), cfg.get("tau_end", 0.5))
        tot = n = 0
        t0 = time.time()
        for regions, desc, anchors, labels in tr:
            regions, desc, anchors, labels = (regions.to(device), desc.to(device),
                                              anchors.to(device), labels.to(device))
            with torch.no_grad():
                tl = teacher(regions, desc, anchors) if teacher is not None else None
            out = model.forward_train(regions, desc, anchors, tau_gumbel=tau)
            loss = composite_loss(out, labels, cfg.get("lambda_exit", 0.1),
                                  cfg.get("lambda_sparse", 0.01),
                                  teacher_logits=tl, lambda_kd=lambda_kd,
                                  kd_temp=kd_temp, kd_schedule=kd_schedule)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * labels.numel(); n += labels.numel()
        sched.step()
        row = {"epoch": ep, "tau_gumbel": tau, "train_loss": tot / n,
               "sec": time.time() - t0}
        if ep % cfg.get("eval_every", 5) == 0 or ep == epochs - 1:
            ev = evaluate(model, te, device, thetas=[cfg.get("theta", 0.7)])
            row.update({"test_acc_full": ev["acc_full_T"],
                        "test_acc_theta": ev["theta_rows"][0]["accuracy"],
                        "avg_slices": ev["theta_rows"][0]["avg_slices"]})
        hist.append(row)
        log("[student %03d] " % ep + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()))
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-epochs", type=int, default=60)
    ap.add_argument("--teacher-lr", type=float, default=1e-3)
    ap.add_argument("--teacher-ckpt", default=None,
                    help="reuse an already-trained teacher instead of retraining")
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--kd-temp", type=float, default=4.0)
    ap.add_argument("--kd-schedule", choices=["all", "ramp", "final"],
                    default="ramp",
                    help="which steps the teacher supervises (see composite_loss)")
    ap.add_argument("--no-kd", action="store_true",
                    help="ablation: augmentation fix only, teacher disabled")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    tr, te = make_loaders(cfg)
    dev = a.device

    teacher, thist, tbest = None, [], None
    if not a.no_kd:
        if a.teacher_ckpt:
            print(f"### teacher: reusing {a.teacher_ckpt}")
            teacher = SliceTeacher(cfg["num_classes"],
                                   d_model=cfg.get("teacher_d_model", 256),
                                   hidden=cfg.get("teacher_hidden", 128)).to(dev)
            teacher.load_state_dict(torch.load(a.teacher_ckpt, map_location=dev))
            tbest = teacher_accuracy(teacher, te, dev)
        else:
            print(f"### teacher: {a.teacher_epochs} epochs on {cfg['dataset']}")
            teacher, thist, tbest = train_teacher(cfg, tr, te, dev, a.teacher_epochs,
                                                  a.teacher_lr, print)
            torch.save(teacher.state_dict(), os.path.join(a.out, "teacher.pt"))
        print(f"### teacher test acc = {tbest:.4f}")

    print(f"### student: {cfg['epochs']} epochs, lambda_kd={a.lambda_kd} "
          f"T={a.kd_temp} kd={'off' if a.no_kd else 'on'}")
    model, hist = train_student_kd(cfg, teacher, tr, te, dev,
                                   a.lambda_kd, a.kd_temp, print,
                                   kd_schedule=a.kd_schedule)

    thetas = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ev = evaluate(model, te, dev, thetas)
    summary = summarize(model, ev, cfg, thetas)
    summary.update({"variant": "kd" if not a.no_kd else "aug_only",
                    "seed": a.seed, "experiment": "KD", "dataset": cfg["dataset"],
                    "teacher_best_acc": tbest, "lambda_kd": a.lambda_kd,
                    "kd_temp": a.kd_temp, "kd_schedule": a.kd_schedule})
    es = M.exits_from_margins(ev["raw"]["margins"], cfg.get("theta", 0.7))
    summary["exit_hist"] = torch.bincount(es, minlength=cfg["k_slices"] + 1).tolist()
    summary["per_class_acc"] = M.per_class_accuracy(
        ev["raw"]["logits"][:, -1].argmax(-1), ev["raw"]["labels"], cfg["num_classes"])

    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    for name, obj in (("summary.json", summary), ("history.json", hist),
                      ("teacher_history.json", thist)):
        with open(os.path.join(a.out, name), "w") as f:
            json.dump(obj, f, indent=1)
    print(f"### DONE {cfg['dataset']}: student acc_full_T={summary['acc_full_T']:.4f} "
          f"teacher={tbest}")


if __name__ == "__main__":
    main()
