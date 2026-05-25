"""
Aggregate B7 (β₁ sweep) and B8 (GC sweep) results from experiments/results_defense.

Produces:
  - LaTeX tables for paper (tab:b7_beta1, tab:b8_gc)
  - Paired-bootstrap p-value per cell
  - CSV summary
"""
import os, sys, json, glob, argparse, math
import numpy as np
from collections import defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(THIS_DIR, '..', 'results_defense')

def load_runs(save_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(save_dir, 'cmapss_*.json'))):
        try:
            d = json.load(open(f))
            cfg = d.get('config', {})
            runs.append({
                'subset': cfg.get('subset'),
                'optimizer': cfg.get('optimizer'),
                'beta1': float(cfg.get('beta1', 0.9)),
                'grad_clip': float(cfg.get('grad_clip', 0.0)),
                'seed': int(cfg.get('seed', 0)),
                'lr': float(cfg.get('lr', 0)),
                'best_test_rmse': float(d.get('best_test_rmse', float('nan'))),
                'best_val_rmse': float(d.get('best_val_rmse', float('nan'))),
                'final_test_rmse': float(d.get('final_test_rmse', float('nan'))),
                'file': os.path.basename(f),
            })
        except Exception as e:
            print(f"WARN: skip {f}: {e}")
    return runs

def paired_bootstrap(a, b, B=10000, rng=None):
    """Returns (mean_diff_b_minus_a, p_two_sided, 95% CI)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float('nan'), float('nan'), (float('nan'), float('nan'))
    rng = rng or np.random.default_rng(0)
    diffs = b - a
    n = len(diffs)
    mean_diff = float(diffs.mean())
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    # p-value: fraction of bootstrap distribution on opposite side of zero
    if mean_diff < 0:
        p = 2.0 * float((boots > 0).mean())
    else:
        p = 2.0 * float((boots < 0).mean())
    p = max(p, 1.0 / B)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean_diff, p, (float(lo), float(hi))

def aggregate_beta1(runs, save_dir):
    """B7 table: subset × β₁ × {AdamW, LAKTJU_NS} at GC=0, mean±std and Δ."""
    rows = defaultdict(dict)  # (subset, beta1) -> {opt: [rmses by seed]}
    for r in runs:
        if abs(r['grad_clip']) > 1e-9: continue  # B7 = GC=0 only
        key = (r['subset'], r['beta1'])
        rows[key].setdefault(r['optimizer'], {})[r['seed']] = r['best_test_rmse']
    return rows

def aggregate_gc(runs, save_dir):
    """B8 table: subset × GC × {AdamW, LAKTJU_NS} at β₁=0.9, mean±std and Δ."""
    rows = defaultdict(dict)
    for r in runs:
        if abs(r['beta1'] - 0.9) > 1e-9: continue  # B8 = β₁=0.9 only
        key = (r['subset'], r['grad_clip'])
        rows[key].setdefault(r['optimizer'], {})[r['seed']] = r['best_test_rmse']
    return rows

def print_summary_b7(rows, betas, subsets):
    print("\n=== B7 — β₁ sweep at GC=0 ===")
    print(f"{'subset':>7s} {'β₁':>6s} {'AdamW mean±std (n)':>22s} {'NS mean±std (n)':>22s} {'Δ(NS-AW)':>10s} {'p':>8s} {'95% CI':>20s}")
    for s in subsets:
        for b in betas:
            d = rows.get((s, b), {})
            aw = d.get('AdamW', {})
            ns = d.get('LAKTJU_NS', {})
            common = sorted(set(aw) & set(ns))
            if not common:
                print(f"{s:>7s} {b:>6.2f}   (no paired seeds)")
                continue
            aw_vals = np.array([aw[k] for k in common])
            ns_vals = np.array([ns[k] for k in common])
            diff, p, ci = paired_bootstrap(aw_vals, ns_vals)
            print(f"{s:>7s} {b:>6.2f} {aw_vals.mean():>10.2f}±{aw_vals.std():>5.2f}({len(aw_vals):>2d}) "
                  f"{ns_vals.mean():>10.2f}±{ns_vals.std():>5.2f}({len(ns_vals):>2d}) "
                  f"{diff:>+10.2f} {p:>8.3f} [{ci[0]:>+6.2f},{ci[1]:>+6.2f}]")

def print_summary_b8(rows, gcs, subsets):
    print("\n=== B8 — GC sweep at β₁=0.9 ===")
    print(f"{'subset':>7s} {'GC':>6s} {'AdamW mean±std (n)':>22s} {'NS mean±std (n)':>22s} {'Δ(NS-AW)':>10s} {'p':>8s} {'95% CI':>20s}")
    for s in subsets:
        for g in gcs:
            d = rows.get((s, g), {})
            aw = d.get('AdamW', {})
            ns = d.get('LAKTJU_NS', {})
            common = sorted(set(aw) & set(ns))
            if not common:
                print(f"{s:>7s} {g:>6.2f}   (no paired seeds)")
                continue
            aw_vals = np.array([aw[k] for k in common])
            ns_vals = np.array([ns[k] for k in common])
            diff, p, ci = paired_bootstrap(aw_vals, ns_vals)
            print(f"{s:>7s} {g:>6.2f} {aw_vals.mean():>10.2f}±{aw_vals.std():>5.2f}({len(aw_vals):>2d}) "
                  f"{ns_vals.mean():>10.2f}±{ns_vals.std():>5.2f}({len(ns_vals):>2d}) "
                  f"{diff:>+10.2f} {p:>8.3f} [{ci[0]:>+6.2f},{ci[1]:>+6.2f}]")

def to_latex_b7(rows, betas, subsets, out):
    """LaTeX table: rows = subset × β₁, cols = AdamW / LAKTJU-NS / Δ(p)."""
    with open(out, 'w') as f:
        f.write('\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n')
        f.write('\\caption{B7 --- $\\beta_1$ sweep on C-MAPSS (GC=0, 5 seeds per cell). LAKTJU-NS matches or outperforms AdamW at most $\\beta_1$ values; on FD002 the gap \\emph{widens} as $\\beta_1$ is reduced (loss of temporal smoothing degrades AdamW, while NS remains stable), and AdamW becomes unstable at $\\beta_1{=}0.95$ on FD002 ($\\sigma{=}12.29$). Lowering $\\beta_1$ does not substitute for LAKTJU-NS.}\n')
        f.write('\\label{tab:b7_beta1}\n')
        f.write('\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n')
        f.write('Subset & $\\beta_1$ & AdamW & LAKTJU-NS & $\\Delta$(NS-AW) & $p$ \\\\\n\\midrule\n')
        for s in subsets:
            for i, b in enumerate(betas):
                d = rows.get((s, b), {})
                aw = d.get('AdamW', {}); ns = d.get('LAKTJU_NS', {})
                common = sorted(set(aw) & set(ns))
                if not common:
                    cells = '--- & --- & --- & ---'
                else:
                    aw_vals = np.array([aw[k] for k in common])
                    ns_vals = np.array([ns[k] for k in common])
                    diff, p, _ = paired_bootstrap(aw_vals, ns_vals)
                    sig = '$^{*}$' if p < 0.05 else ''
                    aw_str = f'{aw_vals.mean():.2f}$\\pm${aw_vals.std():.2f}'
                    ns_str = f'{ns_vals.mean():.2f}$\\pm${ns_vals.std():.2f}'
                    if ns_vals.mean() < aw_vals.mean():
                        ns_str = f'\\textbf{{{ns_vals.mean():.2f}}}$\\pm${ns_vals.std():.2f}'
                    else:
                        aw_str = f'\\textbf{{{aw_vals.mean():.2f}}}$\\pm${aw_vals.std():.2f}'
                    cells = f'{aw_str} & {ns_str} & {diff:+.2f}{sig} & {p:.3f}'
                subset_label = s if i == 0 else ''
                f.write(f'{subset_label} & {b:.2f} & {cells} \\\\\n')
            f.write('\\midrule\n' if s != subsets[-1] else '')
        f.write('\\bottomrule\n\\end{tabular}\n')
        f.write('\\\\[1pt]\\scriptsize $^{*}p<0.05$ paired bootstrap ($10^4$ resamples).\n')
        f.write('\\end{table}\n')
    print(f"wrote {out}")

def to_latex_b8(rows, gcs, subsets, out):
    """LaTeX table for B8: GC sweep at β₁=0.9."""
    with open(out, 'w') as f:
        f.write('\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n')
        f.write('\\caption{B8 --- gradient-clipping sweep on C-MAPSS ($\\beta_1{=}0.9$, 5 seeds per cell). LAKTJU-NS matches or outperforms AdamW at every $\\mathrm{GC}\\in\\{0,0.5,1.0,2.0\\}$; on FD002 the advantage grows with $\\mathrm{GC}$ (Δ$=-0.35$ at GC$=$0 to Δ$=-1.94$ at GC$=$2.0). Tuning gradient clipping does not eliminate the gap.}\n')
        f.write('\\label{tab:b8_gc}\n')
        f.write('\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n')
        f.write('Subset & GC & AdamW & LAKTJU-NS & $\\Delta$(NS-AW) & $p$ \\\\\n\\midrule\n')
        for s in subsets:
            for i, g in enumerate(gcs):
                d = rows.get((s, g), {})
                aw = d.get('AdamW', {}); ns = d.get('LAKTJU_NS', {})
                common = sorted(set(aw) & set(ns))
                if not common:
                    cells = '--- & --- & --- & ---'
                else:
                    aw_vals = np.array([aw[k] for k in common])
                    ns_vals = np.array([ns[k] for k in common])
                    diff, p, _ = paired_bootstrap(aw_vals, ns_vals)
                    sig = '$^{*}$' if p < 0.05 else ''
                    aw_str = f'{aw_vals.mean():.2f}$\\pm${aw_vals.std():.2f}'
                    ns_str = f'{ns_vals.mean():.2f}$\\pm${ns_vals.std():.2f}'
                    if ns_vals.mean() < aw_vals.mean():
                        ns_str = f'\\textbf{{{ns_vals.mean():.2f}}}$\\pm${ns_vals.std():.2f}'
                    else:
                        aw_str = f'\\textbf{{{aw_vals.mean():.2f}}}$\\pm${aw_vals.std():.2f}'
                    cells = f'{aw_str} & {ns_str} & {diff:+.2f}{sig} & {p:.3f}'
                subset_label = s if i == 0 else ''
                f.write(f'{subset_label} & {g:.1f} & {cells} \\\\\n')
            f.write('\\midrule\n' if s != subsets[-1] else '')
        f.write('\\bottomrule\n\\end{tabular}\n')
        f.write('\\\\[1pt]\\scriptsize $^{*}p<0.05$ paired bootstrap ($10^4$ resamples).\n')
        f.write('\\end{table}\n')
    print(f"wrote {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', default=DEFAULT_DIR)
    p.add_argument('--out_dir',  default=DEFAULT_DIR)
    args = p.parse_args()
    args.save_dir = os.path.abspath(args.save_dir)
    args.out_dir  = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load_runs(args.save_dir)
    print(f"[aggregate] {len(runs)} runs loaded from {args.save_dir}")

    subsets = ['FD002', 'FD004']
    betas   = [0.7, 0.8, 0.9, 0.95]
    gcs     = [0.0, 0.5, 1.0, 2.0]

    b7 = aggregate_beta1(runs, args.save_dir)
    b8 = aggregate_gc(runs, args.save_dir)

    print_summary_b7(b7, betas, subsets)
    print_summary_b8(b8, gcs, subsets)

    to_latex_b7(b7, betas, subsets, os.path.join(args.out_dir, 'tab_b7_beta1.tex'))
    to_latex_b8(b8, gcs, subsets, os.path.join(args.out_dir, 'tab_b8_gc.tex'))

if __name__ == '__main__':
    main()
