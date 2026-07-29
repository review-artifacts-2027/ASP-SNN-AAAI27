"""Aggregate results across seeds: mean +/- 95% CI, Welch t-tests vs baseline.

Usage: python -m experiments.aggregate --dir results/A5_policy/modelnet40 \
           [--baseline ssp] [--metric accuracy]
Writes summary.csv and summary.md next to rows.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asp.metrics import ci95, welch_ttest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--metrics", nargs="*",
                    default=["accuracy", "avg_slices", "risk_exited",
                             "acc_full_T", "revisit_rate", "delta_hat",
                             "ece", "energy_ratio"])
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.dir, "rows.csv"))))
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["variant"], r.get("theta", ""))
        for m in a.metrics:
            if r.get(m) not in (None, ""):
                groups[key][m].append(float(r[m]))
    out_rows, md = [], ["| variant | theta | " + " | ".join(a.metrics) + " |",
                        "|---" * (2 + len(a.metrics)) + "|"]
    for (variant, theta), md_ in sorted(groups.items()):
        row = {"variant": variant, "theta": theta}
        cells = []
        for m in a.metrics:
            if md_.get(m):
                s = ci95(md_[m])
                row[f"{m}_mean"], row[f"{m}_ci95"] = s["mean"], s["ci95"]
                cells.append(f"{s['mean']:.4f}±{s['ci95']:.4f}")
            else:
                cells.append("-")
        out_rows.append(row)
        md.append(f"| {variant} | {theta} | " + " | ".join(cells) + " |")
    if a.baseline:
        md += ["", f"Welch t-tests vs `{a.baseline}` (accuracy):", ""]
        base = defaultdict(list)
        for (v, th), md_ in groups.items():
            if v == a.baseline:
                base[th] = md_.get("accuracy", [])
        for (v, th), md_ in sorted(groups.items()):
            if v != a.baseline and base.get(th) and md_.get("accuracy"):
                t = welch_ttest(md_["accuracy"], base[th])
                md.append(f"- {v} (theta={th}): diff={t['mean_diff']:+.4f}, "
                          f"t={t['t']:.2f}, p~{t['p_normal_approx']:.4f}")
    keys = sorted({k for r in out_rows for k in r})
    with open(os.path.join(a.dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out_rows)
    open(os.path.join(a.dir, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
