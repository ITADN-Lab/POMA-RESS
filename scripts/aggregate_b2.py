"""
Aggregate B2 — XJTU-SY cross-condition sweep (Phase A LR pilot + Phase B full seeds).

For each (train_oc, test_oc) split and each optimizer, pick best LR from Phase A (seed=42),
then aggregate Phase B per-seed runs at that best LR.

Output:
  - Console table of per-split mean ± std
  - Paired bootstrap Δ(NS vs AdamW) per split
  - LaTeX tab_b2_xjtu.tex
"""
import os, sys, json, glob, argparse
import numpy as np
from collections import defaultdict

THIS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(THIS, '..', 'results_xjtu')

PRIMARY_OPTS   = ['AdamW', 'LAKTJU_NS', 'MUON']
SECONDARY_OPTS = ['Adan', 'SOAP']
ALL_OPTS = PRIMARY_OPTS + SECONDARY_OPTS

SPLITS = [
    ('OC1,OC2', 'OC3'),
    ('OC1,OC3', 'OC2'),
    ('OC2,OC3', 'OC1'),
]

def load(save_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(save_dir, 'xjtu_*.json'))):
        try:
            d = json.load(open(f))
            cfg = d['config']
            rows.append({
                'train_oc': cfg.get('train_oc'),
                'test_oc':  cfg.get('test_oc'),
                'opt':      cfg.get('optimizer'),
                'lr':       float(cfg.get('lr', 0.0)),
                'seed':     int(cfg.get('seed', 0)),
                'phase':    'pilot' if 'lrpilot' in cfg.get('tag_suffix', '') else 'main',
                'best_test_rmse': float(d.get('best_test_rmse', float('nan'))),
                'best_val_rmse':  float(d.get('best_val_rmse',  float('nan'))),
                'file': os.path.basename(f),
            })
        except Exception as e:
            print(f"WARN: skip {f}: {e}")
    return rows

def boot(a, b, B=10000, rng=None):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < 2:
        return float('nan'), float('nan'), (float('nan'), float('nan'))
    rng = rng or np.random.default_rng(0)
    diffs = b - a
    n = len(diffs)
    mean_diff = float(diffs.mean())
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n); boots[i] = diffs[idx].mean()
    p = 2.0 * (float((boots > 0).mean()) if mean_diff < 0 else float((boots < 0).mean()))
    p = max(p, 1.0 / B)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean_diff, p, (float(lo), float(hi))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', default=DEFAULT_DIR)
    p.add_argument('--out_dir',  default=DEFAULT_DIR)
    args = p.parse_args()
    args.save_dir = os.path.abspath(args.save_dir)
    args.out_dir  = os.path.abspath(args.out_dir)

    rows = load(args.save_dir)
    print(f"[aggregate_b2] {len(rows)} runs from {args.save_dir}")

    # Phase A: pick best LR per (split, opt) using seed=42 pilot
    best_lr = {}  # (train,test,opt) -> (lr, val_rmse)
    for r in rows:
        if r['phase'] != 'pilot' or r['seed'] != 42:
            continue
        k = (r['train_oc'], r['test_oc'], r['opt'])
        if k not in best_lr or r['best_val_rmse'] < best_lr[k][1]:
            best_lr[k] = (r['lr'], r['best_val_rmse'])

    print("\n=== Phase A best LR ===")
    for k, v in sorted(best_lr.items()):
        print(f"  {k[0]:>9s}→{k[1]:>3s}  {k[2]:12s} lr={v[0]:.0e}  val_rmse={v[1]:.2f}")

    # Phase B: aggregate per (split, opt) at best LR
    table = defaultdict(lambda: defaultdict(dict))  # (train,test) -> opt -> seed -> rmse
    for r in rows:
        if r['phase'] != 'main':
            continue
        k = (r['train_oc'], r['test_oc'], r['opt'])
        if k not in best_lr:
            continue
        if abs(r['lr'] - best_lr[k][0]) > 1e-12:
            continue
        table[(r['train_oc'], r['test_oc'])][r['opt']][r['seed']] = r['best_test_rmse']

    print("\n=== Phase B main results (best LR per cell, paired bootstrap NS vs AdamW) ===")
    print(f"{'split':>16s} {'opt':>12s} {'mean±std (n)':>18s} {'Δ vs AdamW':>14s} {'p':>8s}")
    paired_results = {}
    for split in SPLITS:
        train_oc, test_oc = split
        aw = table.get(split, {}).get('AdamW', {})
        ns = table.get(split, {}).get('LAKTJU_NS', {})
        common = sorted(set(aw) & set(ns))
        paired_results[split] = {'common_seeds': common}
        for opt in ALL_OPTS:
            cell = table.get(split, {}).get(opt, {})
            vals = np.array([cell[s] for s in sorted(cell)])
            if len(vals) == 0:
                print(f"{train_oc:>9s}→{test_oc:>3s} {opt:>12s} (no data)")
                continue
            mean, std = float(vals.mean()), float(vals.std())
            if opt == 'AdamW':
                print(f"{train_oc:>9s}→{test_oc:>3s} {opt:>12s} {mean:>8.2f}±{std:>5.2f}({len(vals):>2d}) {'(ref)':>14s}")
            elif opt == 'LAKTJU_NS' and common:
                aw_v = np.array([aw[s] for s in common])
                ns_v = np.array([ns[s] for s in common])
                diff, p, ci = boot(aw_v, ns_v)
                paired_results[split]['NS_vs_AW'] = (diff, p, ci)
                print(f"{train_oc:>9s}→{test_oc:>3s} {opt:>12s} {mean:>8.2f}±{std:>5.2f}({len(vals):>2d}) "
                      f"{diff:>+8.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] {p:>8.3f}")
            else:
                print(f"{train_oc:>9s}→{test_oc:>3s} {opt:>12s} {mean:>8.2f}±{std:>5.2f}({len(vals):>2d})")

    # LaTeX table
    out = os.path.join(args.out_dir, 'tab_b2_xjtu.tex')
    with open(out, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{1.5pt}\n")
        f.write("\\caption{XJTU-SY Bearing RUL (LSTM, cross-condition split). $20$-feature minute-level descriptors (10 time+frequency per channel $\\times$ 2 channels), validation-selected test RMSE. AdamW/LAKTJU-NS/MUON: 10 seeds; Adan/SOAP: 5 seeds. Best LR per (split $\\times$ optimizer) chosen on seed-42 pilot. \\textbf{Bold} = best per row.}\n")
        f.write("\\label{tab:b2_xjtu}\n")
        f.write("\\resizebox{\\columnwidth}{!}{%\n")
        f.write("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n")
        f.write("Split & AdamW & Adan & MUON & SOAP & LAKTJU-NS \\\\\n\\midrule\n")
        for split in SPLITS:
            train_oc, test_oc = split
            cells = {}
            for opt in ALL_OPTS:
                cell = table.get(split, {}).get(opt, {})
                vals = np.array([cell[s] for s in sorted(cell)])
                cells[opt] = (float(vals.mean()), float(vals.std()), len(vals)) if len(vals) else (float('nan'), float('nan'), 0)
            means = [(opt, cells[opt][0]) for opt in ALL_OPTS if cells[opt][2] > 0]
            best_opt = min(means, key=lambda x: x[1])[0] if means else None
            row = []
            for opt in ['AdamW', 'Adan', 'MUON', 'SOAP', 'LAKTJU_NS']:
                m, s, n = cells[opt]
                if n == 0:
                    row.append('---')
                elif opt == best_opt:
                    row.append(f"\\textbf{{{m:.2f}}}$\\pm${s:.2f}")
                else:
                    row.append(f"{m:.2f}$\\pm${s:.2f}")
            label = f"{train_oc.replace(',','+')}$\\to${test_oc}"
            f.write(f"{label} & {' & '.join(row)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n")
        # significance footnote
        f.write("\\\\[1pt]\n{\\scriptsize Paired bootstrap LAKTJU-NS vs.\\ AdamW (10 seeds): ")
        notes = []
        for split in SPLITS:
            sp = paired_results.get(split, {}).get('NS_vs_AW')
            if sp is None: continue
            d, p, ci = sp
            sig = '$^{*}$' if p < 0.05 else ''
            notes.append(f"{split[0].replace(',','+')}$\\to${split[1]} $\\Delta{{=}}{d:+.2f}$ [{ci[0]:+.2f},{ci[1]:+.2f}] $p{{=}}{p:.3f}${sig}")
        f.write('; '.join(notes) + ".}\n")
        f.write("\\end{table}\n")
    print(f"\nwrote {out}")

if __name__ == '__main__':
    main()
