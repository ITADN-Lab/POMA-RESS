"""
A3 deep-dive on EXISTING data (no new experiments).
Quantifies the core mechanistic claim: measured early-training gradient
correlation (rho_g) and momentum conditioning kappa(M) predict the
PMO (=LAKTJU_NS) minus AdamW RMSE gain.

Sources:
  - results_aggregated/fair12/*fair*gc0.0*   (C-MAPSS, GC=0 fair protocol)
  - results_xjtu/xjtu_*_seed*.json           (cross-condition splits)
Outputs per-subset: paired Delta, win-rate, Cohen's d, mean rho_g, mean kappa,
and the across-task Spearman/Pearson correlation (rho_g -> gain).
"""
import json, glob, os, re, math
from collections import defaultdict

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIR = os.path.join(EXP, "results_aggregated", "fair12")

def load(fp):
    try: return json.load(open(fp))
    except Exception: return None

def early_rho_kappa(d, max_epoch=30):
    sh = d.get("spectral_history") or []
    rg = [e["grad_corr"] for e in sh if e.get("epoch",1e9) <= max_epoch and e.get("grad_corr") is not None]
    ka = [e["kappa_M"]  for e in sh if e.get("epoch",1e9) <= max_epoch and e.get("kappa_M")  is not None]
    return (sum(rg)/len(rg) if rg else None, sum(ka)/len(ka) if ka else None)

# ---- C-MAPSS GC=0 fair: AdamW vs LAKTJU_NS, paired by seed ----
pat = re.compile(r"cmapss_(FD00[0-9])_([A-Za-z_]+)_seed(\d+)_fair_.*gc0\.0_")
buf = defaultdict(dict)   # (subset, seed) -> {opt: (rmse, rho, kappa)}
for fp in glob.glob(os.path.join(FAIR, "*fair*gc0.0*")):
    m = pat.search(os.path.basename(fp))
    if not m: continue
    sub, opt, seed = m.group(1), m.group(2), m.group(3)
    if opt not in ("AdamW", "LAKTJU_NS"): continue
    d = load(fp)
    if not d: continue
    rho, ka = early_rho_kappa(d)
    buf[(sub, seed)][opt] = (d["best_test_rmse"], rho, ka)

def cohend(xs, ys):
    n=len(xs)
    if n<2: return float("nan")
    mx=sum(xs)/n; my=sum(ys)/n
    vx=sum((a-mx)**2 for a in xs)/(n-1); vy=sum((a-my)**2 for a in ys)/(n-1)
    sp=math.sqrt(((n-1)*vx+(n-1)*vy)/(2*n-2)) or float("nan")
    return (mx-my)/sp if sp==sp and sp!=0 else float("nan")

print("=== C-MAPSS (GC=0 fair, paired by seed): AdamW vs PMO ===")
print(f"{'sub':6} {'n':>2} {'AdamW':>8} {'PMO':>8} {'Delta':>7} {'win':>6} {'d':>6} {'rho_g':>7} {'kappa':>9}")
sub_gain={}; sub_rho={}; sub_kappa={}
for sub in ["FD001","FD002","FD003","FD004"]:
    aw=[]; ns=[]; rhos=[]; kaps=[]
    for (s,seed),v in buf.items():
        if s!=sub or "AdamW" not in v or "LAKTJU_NS" not in v: continue
        aw.append(v["AdamW"][0]); ns.append(v["LAKTJU_NS"][0])
        for opt in ("AdamW","LAKTJU_NS"):
            if v[opt][1] is not None: rhos.append(v[opt][1])
            if v[opt][2] is not None: kaps.append(v[opt][2])
    if not aw:
        print(f"{sub:6}  no paired data"); continue
    n=len(aw); mAW=sum(aw)/n; mNS=sum(ns)/n; delta=mNS-mAW
    wins=sum(1 for a,b in zip(aw,ns) if b<a)
    d=cohend(aw,ns)
    mrho=sum(rhos)/len(rhos) if rhos else float("nan")
    mka =sum(kaps)/len(kaps) if kaps else float("nan")
    sub_gain[sub]=delta; sub_rho[sub]=mrho; sub_kappa[sub]=mka
    print(f"{sub:6} {n:2d} {mAW:8.2f} {mNS:8.2f} {delta:+7.2f} {wins:2d}/{n:<3d} {d:+6.2f} {mrho:7.3f} {mka:9.1f}")

# ---- across-subset correlation rho_g -> gain (negative gain = improvement) ----
def spearman(xs, ys):
    def rank(v):
        s=sorted(range(len(v)), key=lambda i:v[i]); r=[0]*len(v)
        for i,idx in enumerate(s): r[idx]=i+1
        return r
    rx,ry=rank(xs),rank(ys); n=len(xs)
    dd=sum((a-b)**2 for a,b in zip(rx,ry))
    return 1-6*dd/(n*(n*n-1)) if n>1 else float("nan")
def pearson(xs,ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    num=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    den=math.sqrt(sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))
    return num/den if den else float("nan")

subs=[s for s in ["FD001","FD002","FD003","FD004"] if s in sub_gain]
g=[sub_gain[s] for s in subs]; r=[sub_rho[s] for s in subs]; k=[sub_kappa[s] for s in subs]
print("\n=== Mechanism correlation across C-MAPSS subsets ===")
print(f"subsets={subs}")
print(f"gain(PMO-AdamW)={[round(x,2) for x in g]}  rho_g={[round(x,3) for x in r]}  kappa={[round(x,1) for x in k]}")
print(f"Spearman(rho_g, gain) = {spearman(r,g):+.3f}  (more negative gain at higher rho_g => improvement tracks correlation)")
print(f"Pearson (rho_g, gain) = {pearson(r,g):+.3f}")
print(f"Spearman(kappa, gain) = {spearman(k,g):+.3f}")

# ---- XJTU cross-condition: per split best LR per optimizer, NS vs AdamW ----
XJ = os.path.join(EXP, "results_xjtu")
xpat=re.compile(r"xjtu_(OC\d_OC\d_to_OC\d)_(?:lrpilot_)?([A-Za-z_]+)_lr([0-9e\-\.]+)_seed(\d+)\.json")
xbuf=defaultdict(lambda: defaultdict(list))
for fp in glob.glob(os.path.join(XJ,"xjtu_*_seed*.json")):
    m=xpat.search(os.path.basename(fp))
    if not m: continue
    split,opt,lr,seed=m.groups()
    if opt not in ("AdamW","LAKTJU_NS"): continue
    d=load(fp)
    if not d: continue
    rm=d.get("best_test_rmse") or d.get("final_test_rmse")
    if rm is not None: xbuf[split][(opt,lr)].append(rm)
print("\n=== XJTU cross-condition (mean best_test_rmse, best LR per optimizer) ===")
for split in sorted(xbuf):
    best={}
    for (opt,lr),v in xbuf[split].items():
        mu=sum(v)/len(v)
        if opt not in best or mu<best[opt][0]: best[opt]=(mu,lr,len(v))
    if "AdamW" in best and "LAKTJU_NS" in best:
        aw=best["AdamW"]; ns=best["LAKTJU_NS"]
        print(f"{split:18}  AdamW {aw[0]:.2f}(lr{aw[1]},n{aw[2]})  PMO {ns[0]:.2f}(lr{ns[1]},n{ns[2]})  Delta {ns[0]-aw[0]:+.2f}")
