import json, os, numpy as np, glob
from collections import defaultdict
import sys

base = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/laftju_tii_exp/results_patchtst12")
runs = []
for f in sorted(glob.glob(os.path.join(base, "patchtst_*.json"))):
    d = json.load(open(f))
    fn = os.path.basename(f).replace("patchtst_","").replace(".json","")
    # Parse: FD001_LAKTJU_NS_seed42 -> subset=FD001, opt=LAKTJU_NS, seed=42
    #    or: FD001_AdamW_seed42
    parts = fn.split("_")
    subset = parts[0]
    # Find seedNNN part
    seed_idx = None
    for i, p in enumerate(parts):
        if p.startswith("seed"):
            seed_idx = i
            break
    opt = "_".join(parts[1:seed_idx])
    seed = int(parts[seed_idx].replace("seed",""))
    runs.append({"subset":subset,"opt":opt,"seed":seed,
                 "rmse":d.get("best_test_rmse")})

by_cfg = defaultdict(list)
for r in runs:
    by_cfg[(r["subset"], r["opt"])].append(r["rmse"])

def bs_p(a, b, n=10000):
    a = np.array(a); b = np.array(b)
    nn = min(len(a), len(b)); a = a[:nn]; b = b[:nn]
    rng = np.random.default_rng(2026)
    diffs = np.array([(b[rng.integers(0,nn,nn)] - a[rng.integers(0,nn,nn)]).mean() for _ in range(n)])
    return float(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))

print(f"Total runs: {len(runs)}")
print()
print("PatchTST x C-MAPSS Results (5 seeds, validation-selected best test RMSE, GC=0)")
print("=" * 70)
for subset in ["FD001","FD004"]:
    aw = None
    print(f"\n  {subset}:")
    for opt in ["AdamW","LAKTJU_NS","MUON"]:
        vals = by_cfg.get((subset,opt), [])
        if not vals: continue
        v = np.array(vals)
        if opt == "AdamW": aw = v
        m, s = v.mean(), v.std(ddof=1) if len(v) > 1 else 0
        if opt == "AdamW":
            print(f"    {opt:<12} {len(vals)} seeds  {m:.3f} +- {s:.3f}")
        else:
            p = bs_p(aw, v)
            d = v.mean() - aw.mean()
            sig = "SIG" if p < 0.05 else "n.s."
            print(f"    {opt:<12} {len(vals)} seeds  {m:.3f} +- {s:.3f}   vs AdamW: d={d:+.3f} p={p:.4f} ({sig})")

# Best per subset
print()
print("Best per subset:")
for subset in ["FD001","FD004"]:
    best_opt, best_val = None, float('inf')
    for opt in ["AdamW","LAKTJU_NS","MUON"]:
        vals = by_cfg.get((subset,opt), [])
        if vals and np.mean(vals) < best_val:
            best_val, best_opt = np.mean(vals), opt
    print(f"  {subset}: {best_opt} ({best_val:.3f})")
