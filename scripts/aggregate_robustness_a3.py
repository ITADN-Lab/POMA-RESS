"""
A3 (down-scoped, paper-consistent): robustness stats for the C-MAPSS GC=0
validation-selected primary protocol. Mirrors aggregate_fair_baseline.py
selection (per subset/opt pick the GC=0 LR with the most seeds), then reports
paired Cohen's d, win-rate (PMO<AdamW), and per-seed dispersion (IQR).
NO new correlation claims; pure summary of the SAME runs behind the paper table.
"""
import json, glob, os, re, math
from collections import defaultdict

EXP  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIR = os.path.join(EXP, "results_aggregated", "fair12")

rows = defaultdict(list)  # (sub,opt,lr) -> [(seed,best_test_rmse)]
for fp in glob.glob(os.path.join(FAIR, "cmapss_*_seed*_*_*.json")):
    try: d = json.load(open(fp))
    except Exception: continue
    c = d.get("config", {})
    if float(c.get("grad_clip", 0.0)) != 0.0: continue          # GC=0 fair
    sub, opt = c.get("subset"), c.get("optimizer")
    if opt not in ("AdamW", "LAKTJU_NS"): continue
    rows[(sub, opt, f"{c.get('lr'):.0e}")].append((int(c.get("seed",0)),
                                                   d.get("best_test_rmse")))

# canonical LR pick: per (sub,opt) the LR with most distinct seeds
pick = {}
for (sub,opt,lr), lst in rows.items():
    nseed = len({s for s,_ in lst})
    if (sub,opt) not in pick or nseed > pick[(sub,opt)][1]:
        pick[(sub,opt)] = (lr, nseed)

def stats(v):
    n=len(v); m=sum(v)/n
    sd=math.sqrt(sum((x-m)**2 for x in v)/(n-1)) if n>1 else 0.0
    s=sorted(v); q=lambda p:s[min(n-1,int(p*(n-1)+0.5))]
    return n,m,sd,q(.25),q(.75)

print(f"{'Sub':5} {'n':>2} {'AdamW(m±sd)':>14} {'PMO(m±sd)':>14} {'Δ':>6} {'d':>6} {'win':>6} {'AdamW IQR':>13} {'PMO IQR':>13}")
for sub in ["FD001","FD002","FD003","FD004"]:
    aw_lr = pick.get((sub,"AdamW"),(None,))[0]; ns_lr = pick.get((sub,"LAKTJU_NS"),(None,))[0]
    aw = {s:r for s,r in rows.get((sub,"AdamW",aw_lr),[]) if r is not None}
    ns = {s:r for s,r in rows.get((sub,"LAKTJU_NS",ns_lr),[]) if r is not None}
    common = sorted(set(aw)&set(ns))
    if not common: print(f"{sub:5}  no paired seeds"); continue
    a=[aw[s] for s in common]; b=[ns[s] for s in common]
    na,ma,sda,a25,a75 = stats(a); nb,mb,sdb,b25,b75 = stats(b)
    delta = mb-ma
    # paired Cohen's d (mean diff / sd of diffs)
    diffs=[x-y for x,y in zip(a,b)]; md=sum(diffs)/len(diffs)
    sdd=math.sqrt(sum((x-md)**2 for x in diffs)/(len(diffs)-1)) if len(diffs)>1 else 0.0
    d = md/sdd if sdd else float('nan')
    wins=sum(1 for x,y in zip(a,b) if y<x)
    print(f"{sub:5} {len(common):2d} {ma:6.2f}±{sda:4.2f}   {mb:6.2f}±{sdb:4.2f}   "
          f"{delta:+5.2f} {d:+5.2f} {wins:2d}/{len(common):<3d} "
          f"[{a25:5.2f},{a75:5.2f}] [{b25:5.2f},{b75:5.2f}]")
