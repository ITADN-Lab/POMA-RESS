"""Aggregate E1 — XJTU adaptive trigger on 3 splits."""
import os, glob, json, argparse
import numpy as np
from collections import defaultdict

THIS = os.path.dirname(os.path.abspath(__file__))
E1_DIR = os.path.join(THIS, '..', 'results_e1')
B2_DIR = os.path.join(THIS, '..', 'results_xjtu')

SPLITS = [
    ('OC1,OC2', 'OC3', 'data_scarce'),
    ('OC1,OC3', 'OC2', 'moderate'),
    ('OC2,OC3', 'OC1', 'data_rich'),
]

def load_dir(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        try:
            data = json.load(open(f))
            cfg = data['config']
            rows.append({
                'train_oc': cfg.get('train_oc'),
                'test_oc':  cfg.get('test_oc'),
                'opt':      cfg.get('optimizer'),
                'seed':     int(cfg.get('seed', 0)),
                'tag':      cfg.get('tag_suffix', ''),
                'rmse':     float(data.get('best_test_rmse', float('nan'))),
            })
        except Exception as e:
            print(f"WARN: {f}: {e}")
    return rows

def boot(a, b, B=10000, rng=None):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < 2: return float('nan'), float('nan')
    rng = rng or np.random.default_rng(0)
    diffs = b - a; m = float(diffs.mean())
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(diffs), size=len(diffs)); boots[i] = diffs[idx].mean()
    p = max(2.0 * (float((boots > 0).mean()) if m < 0 else float((boots < 0).mean())), 1/B)
    return m, p

def main():
    e1 = load_dir(E1_DIR)
    b2 = load_dir(B2_DIR)

    # AdamW from B2 main (phase B)
    adamw = defaultdict(dict)
    for r in b2:
        if r['opt'] != 'AdamW' or 'main' not in r['tag']: continue
        adamw[(r['train_oc'], r['test_oc'])][r['seed']] = r['rmse']
    # fixed NS from E1
    fixed = defaultdict(dict)
    for r in e1:
        if 'e1_fixed' in r['tag']:
            fixed[(r['train_oc'], r['test_oc'])][r['seed']] = r['rmse']
    # adaptive NS from E1
    adaptive = defaultdict(dict)
    for r in e1:
        if 'e1_adaptive' in r['tag']:
            adaptive[(r['train_oc'], r['test_oc'])][r['seed']] = r['rmse']

    print(f"\n{'Split':>25s} | {'AdamW':>14s} | {'fixed NS':>14s} | {'adaptive NS':>14s} | {'Δ adapt-AW':>12s} | {'Δ adapt-fix':>13s}")
    rows_out = []
    for train, test, label in SPLITS:
        key = (train, test)
        aw, fx, ad = adamw.get(key, {}), fixed.get(key, {}), adaptive.get(key, {})
        common = sorted(set(aw) & set(fx) & set(ad))
        if not common:
            print(f"{train}→{test} ({label}): no common seeds")
            continue
        aw_v = np.array([aw[s] for s in common])
        fx_v = np.array([fx[s] for s in common])
        ad_v = np.array([ad[s] for s in common])
        d1, p1 = boot(aw_v, ad_v)
        d2, p2 = boot(fx_v, ad_v)
        d_fx, _ = boot(aw_v, fx_v)
        print(f"{train}→{test:>4s} ({label:>11s}) | {aw_v.mean():>6.2f}±{aw_v.std():>5.2f} | {fx_v.mean():>6.2f}±{fx_v.std():>5.2f} | {ad_v.mean():>6.2f}±{ad_v.std():>5.2f} | {d1:>+7.2f} p={p1:.2f} | {d2:>+7.2f} p={p2:.2f}")
        rows_out.append((train, test, label, aw_v, fx_v, ad_v, d_fx, d1, p1, d2, p2))

    # LaTeX
    out = os.path.join(E1_DIR, 'tab_e1_xjtu_adaptive.tex')
    with open(out, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n")
        f.write("\\caption{Adaptive trigger on XJTU-SY cross-condition splits (5 seeds/cell). Adaptive NS reduces fixed-NS harm on the data-scarce extrapolation split.}\n")
        f.write("\\label{tab:e1_xjtu_adaptive}\n")
        f.write("\\begin{tabular}{@{}lcccrrr@{}}\n\\toprule\n")
        f.write("Split & AdamW & fixed NS & adaptive NS & $\\Delta_{\\text{fix}}$ & $\\Delta_{\\text{adapt}}$ & $\\Delta_{\\text{adapt-fix}}$ \\\\\n\\midrule\n")
        for tr, te, lbl, aw, fx, ad, d_fx, d_ad, p_ad, d_af, p_af in rows_out:
            sig_ad = '$^{*}$' if p_ad < 0.05 else ''
            sig_af = '$^{*}$' if p_af < 0.05 else ''
            cell_lbl = f"{tr.replace(',','+')}$\\to${te} ({lbl})"
            f.write(f"{cell_lbl} & {aw.mean():.2f}$\\pm${aw.std():.2f} & {fx.mean():.2f}$\\pm${fx.std():.2f} & {ad.mean():.2f}$\\pm${ad.std():.2f} & {d_fx:+.2f} & {d_ad:+.2f}{sig_ad} & {d_af:+.2f}{sig_af} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"\nwrote {out}")

if __name__ == '__main__':
    main()
