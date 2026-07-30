"""
experiments/run_policy_order.py — the M=16 selection ablation on ModelNet40.

Answers Supplementary D.3 / reviewer W1-W3 with real data. For each seed it:
  1. trains three backbones END-TO-END: learned (ssp), random, fps_order (fixed);
  2. on the LEARNED backbone, additionally evaluates the ORACLE-GREEDY ceiling
     and the held-representation SWAPS (random/fps on the same frozen weights);
  3. reports the accuracy@k anytime curve (confound-free) and tau_bar/acc@theta.

Two comparisons come out of one run:
  * END-TO-END   : learned vs random vs fps_order   (each on its own backbone)
                   -> "does training a membrane-driven policy help?"
  * HELD-REP     : learned vs swap_random vs swap_fps vs oracle (one backbone)
                   -> "given ONE representation, does the READING ORDER matter,
                       and how far is the learned order from the oracle?"

Usage
-----
    python -m experiments.run_policy_order \
        --base configs/base/modelnet40_M16_ablation.yaml \
        --exp  configs/ablations/A5b_policy_M16.yaml \
        --seeds 0 1 2 --epochs 60 --device cuda --out results

Drop asp/oracle.py and asp/anytime.py into the package first. No other file
needs editing.
"""
from __future__ import annotations

import argparse, copy, csv, json, os, sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asp.datasets import build_dataset                    # noqa: E402
from asp.eval import collect                              # noqa: E402
from asp.model import ASPConfig, ASPModel                 # noqa: E402
from asp.train import train_model                         # noqa: E402
from asp import anytime as AT                             # noqa: E402
from asp.oracle import collect_oracle, collect_with_policy  # noqa: E402


def make_loaders(cfg):
    tr = build_dataset(cfg["dataset"], "train", dict(cfg))
    te = build_dataset(cfg["dataset"], "test", dict(cfg))
    bs, nw = cfg.get("batch_size", 64), cfg.get("num_workers", 0)
    return (DataLoader(tr, bs, shuffle=True, num_workers=nw, drop_last=True),
            DataLoader(te, bs, shuffle=False, num_workers=nw))


def train_one(cfg, policy, tr, te, device):
    c = copy.deepcopy(cfg); c["policy"] = policy
    model = ASPModel(ASPConfig.from_dict(c)).to(device)
    model, _ = train_model(c, tr, te, device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    base = yaml.safe_load(open(a.base))
    exp = yaml.safe_load(open(a.exp))
    if a.epochs:
        base["epochs"] = a.epochs
    seeds = a.seeds if a.seeds is not None else base.get("seeds", [0, 1, 2])
    ks = exp.get("anytime_ks", [1, 2, 3, 4, 6, 8, 12, 16])
    thetas = exp.get("eval_thetas", [0.2, 0.3, 0.4, 0.5])
    dev = a.device
    out_dir = os.path.join(a.out, exp["experiment"], base["dataset"])
    os.makedirs(out_dir, exist_ok=True)

    # rule -> list of per-seed summaries
    RULES = ["learned", "random", "fps_order", "oracle", "swap_random", "swap_fps"]
    bucket = {r: [] for r in RULES}
    flat_rows = []

    for seed in seeds:
        print(f"\n########## seed {seed} ##########", flush=True)
        torch.manual_seed(seed)
        tr, te = make_loaders(base)

        # --- END-TO-END backbones ---
        m_learned = train_one(base, "ssp",    tr, te, dev)
        m_random  = train_one(base, "random", tr, te, dev)
        m_fps     = train_one(base, "fixed",  tr, te, dev)

        raws = {
            "learned":   collect(m_learned, te, dev),                 # own rule
            "random":    collect(m_random,  te, dev),
            "fps_order": collect(m_fps,     te, dev),
            # --- HELD-REP on the learned backbone ---
            "oracle":      collect_oracle(m_learned, te, dev),
            "swap_random": collect_with_policy(m_learned, te, dev, "random"),
            "swap_fps":    collect_with_policy(m_learned, te, dev, "fixed"),
        }

        for rule, raw in raws.items():
            s = AT.summarize_rule(raw, thetas, ks)
            s.update({"rule": rule, "seed": seed})
            bucket[rule].append(s)
            row = {"experiment": exp["experiment"], "dataset": base["dataset"],
                   "rule": rule, "seed": seed, "acc_full_K": s["acc_full_K"]}
            row.update({f"acc@{k}": s["anytime"][k] for k in ks})
            for tr_ in s["theta_rows"]:
                row[f"tau@{tr_['theta']}"] = tr_["tau_bar"]
                row[f"acc@theta{tr_['theta']}"] = tr_["acc_theta"]
            flat_rows.append(row)
        json.dump(raws and {r: {"seed": seed} for r in raws},
                  open(os.path.join(out_dir, f"seed{seed}_done.json"), "w"))

    # ---------------- aggregate + write ----------------
    if flat_rows:
        keys = sorted({k for r in flat_rows for k in r})
        with open(os.path.join(out_dir, "rows.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(flat_rows)

    def agg(rule, field_getter):
        vals = [field_getter(s) for s in bucket[rule]]
        return np.mean(vals), np.std(vals)

    # ---- Table S4 (real data) ----
    lines = []
    lines.append("=" * 110)
    lines.append(f"TABLE S4 (real data) — {base['dataset']}  M={base['k_slices']}  "
                 f"seeds={seeds}  epochs={base['epochs']}")
    lines.append("Accuracy@k anytime curve (mean±std %).  tau-bar forced < M via low theta.")
    lines.append("=" * 110)
    hdr = "rule".ljust(13) + "".join(f"  acc@{k:<2}".ljust(11) for k in ks) + "  acc@K"
    lines.append(hdr); lines.append("-" * len(hdr))
    order = ["fps_order", "random", "learned", "oracle", "swap_random", "swap_fps"]
    for rule in order:
        if not bucket[rule]:
            continue
        cells = []
        for k in ks:
            mu, sd = agg(rule, lambda s, k=k: s["anytime"][k] * 100)
            cells.append(f"{mu:5.1f}±{sd:3.1f}".ljust(11))
        mu, sd = agg(rule, lambda s: s["acc_full_K"] * 100)
        lines.append(rule.ljust(13) + "  " + "  ".join(cells) + f"  {mu:4.1f}±{sd:3.1f}")
    lines.append("-" * len(hdr))
    # early-exit view
    lines.append("\nEarly-exit view (tau-bar | acc@theta), mean±std:")
    th_hdr = "rule".ljust(13) + "".join(f"  θ={t}: taū / acc".ljust(20) for t in thetas)
    lines.append(th_hdr)
    for rule in order:
        if not bucket[rule]:
            continue
        cells = []
        for i, t in enumerate(thetas):
            tmu, tsd = agg(rule, lambda s, i=i: s["theta_rows"][i]["tau_bar"])
            amu, asd = agg(rule, lambda s, i=i: s["theta_rows"][i]["acc_theta"] * 100)
            cells.append(f"{tmu:4.1f} / {amu:4.1f}".ljust(20))
        lines.append(rule.ljust(13) + "".join(cells))
    # headline gaps at a small budget
    kk = ks[2] if len(ks) > 2 else ks[0]
    def m(rule): return agg(rule, lambda s: s["anytime"][kk] * 100)[0]
    lines.append("\n" + "=" * 110)
    lines.append(f"HEADLINE @k={kk}:  fps={m('fps_order'):.1f}  random={m('random'):.1f}  "
                 f"learned={m('learned'):.1f}  oracle={m('oracle'):.1f}")
    lines.append(f"  learned−random = {m('learned')-m('random'):+.1f} pp    "
                 f"oracle−learned = {m('oracle')-m('learned'):+.1f} pp    "
                 f"oracle−random = {m('oracle')-m('random'):+.1f} pp")
    lines.append("  Interpretation:")
    lines.append("    learned>random>fps beyond seed noise  -> THESIS VALIDATED")
    lines.append("    oracle>>random≈learned                -> structure exists, POLICY UNDERPOWERED")
    lines.append("    oracle≈random                         -> selection doesn't help at this M -> pivot")
    lines.append("=" * 110)
    report = "\n".join(lines)
    print(report)
    open(os.path.join(out_dir, "table_s4.txt"), "w").write(report)
    print("\nwrote:", os.path.join(out_dir, "rows.csv"), "and table_s4.txt")


if __name__ == "__main__":
    main()