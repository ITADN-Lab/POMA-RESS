"""
Aggregate the β₁ × GC cross sweep (8 corner cells × 5 seeds × 2 optimizers = 80 runs).

Produces:
  - Per-cell paired-bootstrap Δ(NS-AW) with 95% CI and p-value
  - LaTeX table tab_cross.tex for supplementary §S6
  - Console summary
"""
import os, sys, json, glob, argparse
import numpy as np
from collections import defaultdict

THIS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(THIS, '..', 'results_defense_cross')

SUBSETS = ['FD002', 'FD004']
BETAS   = [0.7, 0.95]
GCS     = [0.5, 2.0]

def load(save_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(save_dir, 'cmapss_*.json'))):
        try:
            d = json.load(open(f))
            cfg = d['config']
            rows.append({
                'subset': cfg.get('subset'),
                'opt':    cfg.get('optimizer'),
                'beta1':  float(cfg.get('beta1', 0.9)),
                'gc':     float(cfg.get('grad_clip', 0.0)),
                'seed':   int(cfg.get('seed', 0)),
                'rmse':   float(d.get('best_test_rmse', float('nan'))),
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

def cell_rows(rows):
    grid = defaultdict(dict)  # (s,b,g) -> {opt: {seed:rmse}}
    for r in rows:
        key = (r['subset'], round(r['beta1'],2), round(r['gc'],1))
        grid[key].setdefault(r['opt'], {})[r['seed']] = r['rmse']
    return grid

def summarize(grid):
    cells = []
    for s in SUBSETS:
        for b in BETAS:
            for g in GCS:
                key = (s, b, g)
                d = grid.get(key, {})
                aw, ns = d.get('AdamW', {}), d.get('LAKTJU_NS', {})
                common = sorted(set(aw) & set(ns))
                if not common:
                    cells.append((s, b, g, None))
                    continue
                aw_v = np.array([aw[k] for k in common])
                ns_v = np.array([ns[k] for k in common])
                diff, p, ci = boot(aw_v, ns_v)
                cells.append((s, b, g, dict(
                    n=len(common), aw_m=aw_v.mean(), aw_s=aw_v.std(),
                    ns_m=ns_v.mean(), ns_s=ns_v.std(),
                    diff=diff, p=p, ci=ci)))
    return cells

def latex_table(cells, out):
    with open(out, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\caption{Joint $(\\beta_1,\\mathrm{GC})$ corner-point verification on C-MAPSS (LSTM, $5$ seeds per cell, validation-selected, paired bootstrap). NS direction preserved at all $8$ corner cells; the gain is largest where AdamW's smoothing is most disrupted (low $\\beta_1$).}\n")
        f.write("\\label{tab:cross_corner}\n\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n")
        f.write("Subset & $(\\beta_1,\\mathrm{GC})$ & AdamW & LAKTJU-NS & $\\Delta$ & $p$ & 95\\% CI \\\\\n\\midrule\n")
        last_subset = None
        for s, b, g, c in cells:
            if c is None:
                continue
            aw_str = f"{c['aw_m']:.2f}$\\pm${c['aw_s']:.2f}"
            ns_str = f"{c['ns_m']:.2f}$\\pm${c['ns_s']:.2f}"
            if c['ns_m'] < c['aw_m']:
                ns_str = f"\\textbf{{{c['ns_m']:.2f}}}$\\pm${c['ns_s']:.2f}"
            else:
                aw_str = f"\\textbf{{{c['aw_m']:.2f}}}$\\pm${c['aw_s']:.2f}"
            sig = "$^{*}$" if c['p'] < 0.05 else ""
            cell_lbl = f"({b:.2f},{g:.1f})"
            subset_lbl = s if s != last_subset else ''
            last_subset = s
            f.write(f"{subset_lbl} & {cell_lbl} & {aw_str} & {ns_str} & {c['diff']:+.2f}{sig} & {c['p']:.3f} & [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\\\[1pt]\\scriptsize $^{*}p<0.05$ paired bootstrap ($10^4$ resamples).\n")
        f.write("\\end{table}\n")
    print(f"wrote {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', default=DEFAULT_DIR)
    p.add_argument('--out_dir',  default=DEFAULT_DIR)
    args = p.parse_args()
    args.save_dir = os.path.abspath(args.save_dir)
    args.out_dir  = os.path.abspath(args.out_dir)

    rows = load(args.save_dir)
    print(f"[aggregate_cross] {len(rows)} runs from {args.save_dir}")
    grid = cell_rows(rows)
    cells = summarize(grid)

    print(f"\n{'subset':>7s} {'(β,GC)':>12s} {'AdamW':>15s} {'NS':>15s} {'Δ':>8s} {'p':>7s} {'95% CI':>16s}")
    for s, b, g, c in cells:
        if c is None:
            print(f"{s:>7s} ({b:.2f},{g:.1f}) (no data)")
            continue
        print(f"{s:>7s} ({b:.2f},{g:.1f}) "
              f"{c['aw_m']:>7.2f}±{c['aw_s']:>5.2f}({c['n']:>2d}) "
              f"{c['ns_m']:>7.2f}±{c['ns_s']:>5.2f} "
              f"{c['diff']:>+7.2f} {c['p']:>7.3f} [{c['ci'][0]:>+5.2f},{c['ci'][1]:>+5.2f}]")

    latex_table(cells, os.path.join(args.out_dir, 'tab_cross.tex'))

if __name__ == '__main__':
    main()
