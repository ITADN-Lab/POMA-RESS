"""Aggregate C-MAPSS fair-baseline results from 12号:
  - For each (subset × optimizer × GC), find best LR by validation; report best-LR seed-level stats.
  - Use validation-selected best_test_rmse (already computed in train_cmapss_rul.py).
  - Compute paired bootstrap p-values (LAKTJU-NS vs each baseline at best LR per GC).
  - Output: (a) summary table, (b) LaTeX snippets ready to paste.
"""
import os, json, glob, argparse, sys
from collections import defaultdict
import numpy as np


def bootstrap_diff_p(a, b, n_boot=10000, rng_seed=2026):
    """Two-sided paired bootstrap p-value for mean(b) - mean(a)."""
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return float('nan')
    a = a[:n]; b = b[:n]
    obs = b.mean() - a.mean()
    rng = np.random.default_rng(rng_seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = (b[idx] - a[idx]).mean()
        diffs.append(d)
    diffs = np.array(diffs)
    if obs >= 0:
        p = (diffs >= 2 * obs - obs).mean()  # would equal mean of resampled around 0
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())  # standard two-sided
    return float(p)


def load_all(save_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(save_dir, 'cmapss_*_seed*_*_*.json'))):
        try:
            d = json.load(open(f))
            cfg = d.get('config', {})
            runs.append({
                'file': f,
                'subset': cfg.get('subset'),
                'optimizer': cfg.get('optimizer'),
                'gc': float(cfg.get('grad_clip', 0.0)),
                'lr': float(cfg.get('lr', 0.0)),
                'seed': int(cfg.get('seed', 0)),
                'best_val_rmse': d.get('best_val_rmse'),
                'best_test_rmse': d.get('best_test_rmse'),
                'final_test_rmse': (d.get('history', [{}])[-1].get('test_rmse')
                                    if d.get('history') else None),
            })
        except Exception as e:
            print(f"WARN: skip {f}: {e}", file=sys.stderr)
    return runs


def best_lr_by_val(runs):
    """For each (subset, opt, gc), pick LR with the most seeds (Phase B re-ran best LR
    from Phase A across multiple seeds, so the LR with most seeds is the chosen one).
    Ties broken by lowest mean val_rmse."""
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[(r['subset'], r['optimizer'], r['gc'], r['lr'])].append(r)
    # Stats per (subset, opt, gc, lr): n_seeds (deduped), mean val
    stats = {}
    for k, lst in by_cfg.items():
        seeds = sorted({r['seed'] for r in lst})
        vals = [r['best_val_rmse'] for r in lst if r['best_val_rmse']]
        if vals:
            stats[k] = (len(seeds), np.mean(vals))
    # Pick LR with most seeds, tie-break by val_rmse
    best = {}
    for (subset, opt, gc, lr), (nseeds, val) in stats.items():
        key = (subset, opt, gc)
        if key not in best:
            best[key] = (lr, val, nseeds)
        else:
            cur_lr, cur_val, cur_n = best[key]
            if (nseeds > cur_n) or (nseeds == cur_n and val < cur_val):
                best[key] = (lr, val, nseeds)
    return {k: (lr, val) for k, (lr, val, _) in best.items()}


def aggregate(runs, best_lr_map):
    """For each (subset × opt × gc) at best LR, compute seed-level stats."""
    by_cfg = defaultdict(list)
    for r in runs:
        bk = (r['subset'], r['optimizer'], r['gc'])
        if bk in best_lr_map and abs(r['lr'] - best_lr_map[bk][0]) < 1e-9:
            by_cfg[bk].append(r)
    stats = {}
    for k, lst in by_cfg.items():
        # dedup by seed (keep best val_rmse seed if duplicates)
        by_seed = {}
        for r in lst:
            if r['seed'] not in by_seed or r['best_val_rmse'] < by_seed[r['seed']]['best_val_rmse']:
                by_seed[r['seed']] = r
        seed_runs = list(by_seed.values())
        if not seed_runs:
            continue
        test_rmses = [r['best_test_rmse'] for r in seed_runs]
        final_rmses = [r['final_test_rmse'] for r in seed_runs if r['final_test_rmse']]
        stats[k] = {
            'lr': best_lr_map[k][0],
            'n_seeds': len(seed_runs),
            'best_test_mean': float(np.mean(test_rmses)),
            'best_test_std': float(np.std(test_rmses, ddof=1)) if len(test_rmses) > 1 else 0.0,
            'final_test_mean': float(np.mean(final_rmses)) if final_rmses else None,
            'final_test_std': float(np.std(final_rmses, ddof=1)) if len(final_rmses) > 1 else 0.0,
            'seeds': sorted(by_seed.keys()),
            'all_test_rmses': test_rmses,
        }
    return stats


def print_table(stats):
    rows = sorted(stats.items())
    print(f"\n{'Subset':<7} {'Optimizer':<12} {'GC':<5} {'LR':<10} {'N':<3} {'best_test (mean±std)':<25} {'final_test':<20}")
    print('-' * 90)
    for (subset, opt, gc), s in rows:
        bs = f"{s['best_test_mean']:.2f}±{s['best_test_std']:.2f}"
        fs = (f"{s['final_test_mean']:.2f}±{s['final_test_std']:.2f}" if s['final_test_mean'] else "-")
        print(f"{subset:<7} {opt:<12} {gc:<5.1f} {s['lr']:<10.4f} {s['n_seeds']:<3} {bs:<25} {fs:<20}")


def significance_table(stats):
    """For each subset×GC, compute paired bootstrap p of LAKTJU_NS vs each other optimizer."""
    print("\n=== Significance: LAKTJU_NS vs others (paired bootstrap, two-sided) ===")
    print(f"{'Subset':<7} {'GC':<5} {'Baseline':<12} {'NS<base?':<12} {'p-value':<10}")
    print('-' * 55)
    by_subset_gc = defaultdict(dict)
    for (subset, opt, gc), s in stats.items():
        by_subset_gc[(subset, gc)][opt] = s
    for (subset, gc), opt_stats in sorted(by_subset_gc.items()):
        ns = opt_stats.get('LAKTJU_NS')
        if not ns: continue
        for opt, s in sorted(opt_stats.items()):
            if opt == 'LAKTJU_NS': continue
            p = bootstrap_diff_p(ns['all_test_rmses'], s['all_test_rmses'])
            delta = ns['best_test_mean'] - s['best_test_mean']
            ns_better = "yes" if delta < 0 else "no"
            print(f"{subset:<7} {gc:<5.1f} {opt:<12} NS-{opt}={delta:+.2f} ({ns_better}) p={p:.4f}")


def latex_main_table(stats):
    """Generate LaTeX table comparable to tab:cmapss in paper."""
    subsets = ['FD001', 'FD002', 'FD003', 'FD004']
    optimizers = ['AdamW', 'Adan', 'MUON', 'SOAP', 'LAKTJU_NS']
    print("\n=== LaTeX: tab:cmapss_fair (best LR per (subset×opt) at GC=0) ===")
    print(r"\begin{tabular}{@{}lccccc@{}}")
    print(r"\toprule")
    print(r"Subset & " + " & ".join(optimizers) + r" \\")
    print(r"\midrule")
    for subset in subsets:
        cells = [subset]
        for opt in optimizers:
            s = stats.get((subset, opt, 0.0))
            if s:
                cells.append(f"{s['best_test_mean']:.2f}$\\pm${s['best_test_std']:.2f}")
            else:
                cells.append('-')
        print(' & '.join(cells) + r' \\')
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True)
    args = ap.parse_args()
    runs = load_all(args.save_dir)
    print(f"Loaded {len(runs)} runs from {args.save_dir}")
    if not runs:
        return
    best = best_lr_by_val(runs)
    stats = aggregate(runs, best)
    print_table(stats)
    significance_table(stats)
    latex_main_table(stats)


if __name__ == '__main__':
    main()
