"""
FEMTO variance decomposition + sp88 val/test gap (Plan B).

For each (condition, opt), the audit's Δ has two sources of uncertainty:
  (a) seed variance  : 20 runs per (cond, opt, partition) cell
  (b) partition variance : 3+ partitions per (cond, opt), each holding out
      a different test bearing

We decompose: total Var(Δ) ≈ Var_partition(mean_Δ) + mean_partition(Var_seed).
If partition variance >> seed variance, the optimizer's apparent advantage
is dominated by which bearing is held out, not by random training noise.

We also pull per-partition val/test RMSE gap (a proxy for the
train/held-out distribution shift) to quantify why sp=88 is "hard".
"""
import os, json, glob
import numpy as np

RES = os.path.expanduser('~/pmo_femto/results')
OUT_MD = os.path.expanduser('~/pmo_femto/results/VARIANCE_DECOMP.md')

OPTS = ['PMO', 'Adan', 'RAdam', 'Lion']
CONDS = ['cond1', 'cond2']
PARTITIONS = [2024, 7, 88, 1, 2]


def load_phase_b(opt, sp, cond):
    """Return list of (seed, val, test) at this partition's matched best config."""
    runs = []
    for f in glob.glob(os.path.join(RES, f'femto_{cond}_{opt}_seed*_*.json')):
        try:
            d = json.load(open(f))
            if d['split_info']['split_seed'] != sp: continue
            c = d['config']
            if c['optimizer'] != opt or c['condition'] != cond: continue
            runs.append((c['seed'], c['beta1'], c['grad_clip'], c['lr'],
                         d['best_val_rmse'], d['best_test_rmse']))
        except Exception: pass
    return runs


def matched_test(runs):
    """Pick best-by-val at seed=42, return per-seed test RMSE of that config."""
    s42 = [r for r in runs if r[0] == 42]
    if not s42: return None, None
    best = min(s42, key=lambda r: r[4])
    matched = sorted([r for r in runs
        if abs(r[1]-best[1])<1e-9 and abs(r[2]-best[2])<1e-9 and abs(r[3]-best[3])<1e-12])
    return [r[5] for r in matched], [r[4] for r in matched]


md = ["# FEMTO Plan-B variance decomposition + sp=88 val/test gap\n",
      "## Per-partition val/test RMSE gap (distribution-shift proxy)\n",
      "| Cond | Opt | sp | val mean | test mean | val/test ratio |",
      "|---|---|---|---|---|---|"]
for cond in CONDS:
    for opt in OPTS + ['AdamW']:
        for sp in PARTITIONS:
            runs = load_phase_b(opt, sp, cond)
            te, va = matched_test(runs)
            if te and va and len(te) >= 5:
                vm, tm = np.mean(va), np.mean(te)
                md.append(f"| {cond} | {opt} | {sp} | {vm:.2f} | {tm:.2f} | "
                          f"{(tm/max(vm,1e-6)):.2f}× |")

md.append("")
md.append("## Variance decomposition (per cond, opt): how much of $\\Delta$ uncertainty is partition vs seed\n")
md.append("| Cond | Opt | sd_seed (mean over partitions) | sd_partition (across mean-Δ) | partition/seed ratio |")
md.append("|---|---|---|---|---|")
for cond in CONDS:
    for opt in OPTS:
        # for each partition, compute paired Δ test-RMSE = opt − AdamW per seed
        partition_means = []   # mean Δ at each partition
        seed_sds = []          # sd of Δ across 20 seeds within each partition
        for sp in PARTITIONS:
            aw_te, _ = matched_test(load_phase_b('AdamW', sp, cond))
            op_te, _ = matched_test(load_phase_b(opt, sp, cond))
            if aw_te is None or op_te is None: continue
            n = min(len(aw_te), len(op_te))
            if n < 5: continue
            d = np.array(op_te[:n]) - np.array(aw_te[:n])
            partition_means.append(d.mean())
            seed_sds.append(d.std(ddof=1))
        if len(partition_means) >= 2:
            sd_partition = float(np.std(partition_means, ddof=1))
            sd_seed = float(np.mean(seed_sds))
            md.append(f"| {cond} | {opt} | {sd_seed:.2f} | {sd_partition:.2f} | "
                      f"{(sd_partition/max(sd_seed,1e-6)):.2f}× |")

md.append("\n\n*Partition variance > seed variance means optimizer apparent advantage is dominated by which bearing is held out.*\n")
text = "\n".join(md) + "\n"
os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
open(OUT_MD, 'w').write(text)
print(text)
print(f"wrote {OUT_MD}")
