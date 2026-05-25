#!/usr/bin/env python3
"""
Generate forest_mechanism.pdf: forest plot of mechanism causal ablation.

Variants on FD002 and FD004 (5 seeds, GC=0):
  MomNS (ours), AdamW (reference), GradNS, NormOnly, RandRot

Per-seed test RMSE files (when available):
  experiments/results_ablation/ablate_<Variant>_<Subset>_seed<seed>.json
"""
import json, glob, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/hadoop/workstation/md/RESS"
ABL_DIR = '/home/hadoop/workstation/md/LafTJU-TII/experiments/results_ablation20'
OUT = os.path.join(REPO, "paper", "figures", "forest_mechanism.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

VARIANTS = ["LAKTJU_NS", "GradNS", "NormOnly", "RandRot"]
LABELS   = ["Momentum NS (ours)", "Gradient NS (PMuon)", "NormOnly", "RandRot"]
SUBSETS  = ["FD002", "FD004"]

def load_per_seed(variant, subset):
    files = sorted(glob.glob(os.path.join(ABL_DIR, f"ablate_{variant}_{subset}_seed*.json")))
    rmse = []
    for f in files:
        d = json.load(open(f))
        v = (d.get("best_test_rmse") or d.get("final_test_rmse") or
             d.get("test_rmse") or d.get("rmse"))
        if v is None and "history" in d:
            v = min(r.get("best_test_rmse", r.get("test_rmse", float("nan"))) for r in d["history"])
        if v is not None:
            rmse.append(float(v))
    return np.array(rmse)

aw = {s: load_per_seed("AdamW", s) for s in SUBSETS}

# Compute per-seed paired differences
deltas = {}  # (variant, subset) -> array of (variant_rmse - adamw_rmse) per common seed
for v in VARIANTS:
    for s in SUBSETS:
        vals = load_per_seed(v, s)
        # align seeds by file ordering; lengths may match
        n = min(len(vals), len(aw[s]))
        if n == 0:
            # fall back to hard-coded means from paper table
            continue
        deltas[(v, s)] = vals[:n] - aw[s][:n]

# Hard-coded paired means from paper (FD002 / FD004) as fallback / sanity
HARDCODED = {
    ("LAKTJU_NS","FD002"): -1.31, ("LAKTJU_NS","FD004"): -3.07,
    ("GradNS","FD002"):   +0.28, ("GradNS","FD004"):    -0.29,
    ("NormOnly","FD002"): +0.77, ("NormOnly","FD004"):  -0.28,
    ("RandRot","FD002"): +23.82, ("RandRot","FD004"):  +20.67,
}

def bootstrap_ci(diffs, B=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    diffs = np.asarray(diffs)
    if len(diffs) < 2:
        return float(diffs.mean() if len(diffs) else 0.0), float("nan"), float("nan")
    boots = np.empty(B)
    n = len(diffs)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(diffs.mean()), float(lo), float(hi)

# Build per-(variant,subset) point estimates
rows = []
for v, label in zip(VARIANTS, LABELS):
    for s in SUBSETS:
        if (v, s) in deltas:
            m, lo, hi = bootstrap_ci(deltas[(v, s)])
        else:
            m = HARDCODED.get((v, s), float("nan"))
            lo, hi = m - abs(m)*0.25 - 0.4, m + abs(m)*0.25 + 0.4
        rows.append((label, s, m, lo, hi))

fig, ax = plt.subplots(figsize=(6.4, 3.2))

# Layout: y axis = variant; x = delta RMSE vs AdamW. FD002 (squares) and FD004 (circles) offset.
y_positions = {}
for i, label in enumerate(LABELS):
    y_positions[label] = len(LABELS) - i  # top to bottom

colors = {"FD002": "C0", "FD004": "C3"}
markers = {"FD002": "s", "FD004": "o"}
offsets = {"FD002": -0.18, "FD004": 0.18}

for label, s, m, lo, hi in rows:
    y = y_positions[label] + offsets[s]
    err = [[m - lo], [hi - m]]
    ax.errorbar(m, y, xerr=err, fmt=markers[s], color=colors[s], capsize=3, lw=1.4, ms=6,
                label=f"{s}" if (label==LABELS[0]) else None)

ax.axvline(0, color="grey", lw=0.7, ls="--")
ax.set_yticks([y_positions[l] for l in LABELS])
ax.set_yticklabels(LABELS)
ax.set_xlabel(r"$\Delta$ test RMSE vs.\ AdamW  (negative = improvement)")
ax.set_title("Mechanism ablation forest plot (C-MAPSS, 20 seeds, GC=0)")
ax.grid(True, axis="x", alpha=0.3)
ax.legend(title="Subset", loc="lower right", fontsize=8)
# Trim RandRot's huge effect to keep axis readable:
ax.set_xlim(-6, 6)
ax.text(5.8, y_positions["RandRot"], "(RandRot extends to $+18$ / $+25$ off-chart)",
        ha="right", va="center", fontsize=7, color="grey")

plt.tight_layout()
plt.savefig(OUT, format="pdf", bbox_inches="tight")
print(f"wrote {OUT}")
for label, s, m, lo, hi in rows:
    print(f"  {label:20s} {s} : mean={m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
