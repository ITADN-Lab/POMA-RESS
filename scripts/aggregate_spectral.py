"""Aggregate step-wise spectral tracking from local 13号:
  - For each (subset × optimizer), compute mean κ_pre/κ_post (LAKTJU_NS) or sample κ (others) over training
  - Plot/print the κ trajectory averaged across seeds
  - Output: summary table + LaTeX snippet for the mechanism table
"""
import os, json, glob, argparse, sys
from collections import defaultdict
import numpy as np


def load_all(save_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(save_dir, 'spectral_*.json'))):
        try:
            d = json.load(open(f))
            cfg = d.get('config', {})
            out.append({
                'file': f,
                'subset': cfg.get('subset'),
                'optimizer': cfg.get('optimizer'),
                'seed': int(cfg.get('seed', 0)),
                'best_test_rmse': d.get('best_test_rmse'),
                'spectral_log': d.get('spectral_log', []),
            })
        except Exception as e:
            print(f"WARN: skip {f}: {e}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True)
    args = ap.parse_args()
    runs = load_all(args.save_dir)
    print(f"Loaded {len(runs)} spectral runs")
    if not runs:
        return

    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[(r['subset'], r['optimizer'])].append(r)

    print(f"\n{'Subset':<7} {'Optimizer':<12} {'N':<3} {'Mean κ_pre':<12} {'Mean κ_post':<14} {'NS reduction':<14} {'Mean erank_pre':<15} {'Mean erank_post':<15} {'best_test':<10}")
    print('-' * 110)
    summary_rows = []
    for (subset, opt), lst in sorted(by_cfg.items()):
        seeds = sorted({r['seed'] for r in lst})
        # collect κ values
        if 'NS' in opt:
            kappa_pre, kappa_post, erank_pre, erank_post = [], [], [], []
            for r in lst:
                for ev in r['spectral_log']:
                    if ev.get('event') == 'NS':
                        if ev.get('kappa_pre'): kappa_pre.append(ev['kappa_pre'])
                        if ev.get('kappa_post'): kappa_post.append(ev['kappa_post'])
                        if ev.get('erank_pre'): erank_pre.append(ev['erank_pre'])
                        if ev.get('erank_post'): erank_post.append(ev['erank_post'])
            if not kappa_pre:
                continue
            kp = np.array(kappa_pre); kpo = np.array(kappa_post)
            ep = np.array(erank_pre); epo = np.array(erank_post)
            test_rmses = [r['best_test_rmse'] for r in lst if r['best_test_rmse']]
            row = {
                'subset': subset, 'optimizer': opt, 'n_seeds': len(seeds),
                'kappa_pre_mean': float(kp.mean()),
                'kappa_pre_median': float(np.median(kp)),
                'kappa_post_mean': float(kpo.mean()),
                'kappa_post_median': float(np.median(kpo)),
                'erank_pre_mean': float(ep.mean()),
                'erank_post_mean': float(epo.mean()),
                'reduction_factor': float(np.median(kp) / np.median(kpo)) if np.median(kpo) > 0 else None,
                'best_test_mean': float(np.mean(test_rmses)) if test_rmses else None,
            }
            summary_rows.append(row)
            print(f"{subset:<7} {opt:<12} {len(seeds):<3} {row['kappa_pre_mean']:<12.2e} {row['kappa_post_mean']:<14.2e} "
                  f"{row['reduction_factor']:<14.1e} {row['erank_pre_mean']:<15.2f} {row['erank_post_mean']:<15.2f} "
                  f"{row['best_test_mean']:<10.2f}")
        else:
            kappas, eranks = [], []
            for r in lst:
                for ev in r['spectral_log']:
                    if ev.get('event') == 'sample':
                        if ev.get('kappa'): kappas.append(ev['kappa'])
                        if ev.get('erank'): eranks.append(ev['erank'])
            if not kappas:
                continue
            ks = np.array(kappas); es = np.array(eranks)
            test_rmses = [r['best_test_rmse'] for r in lst if r['best_test_rmse']]
            row = {
                'subset': subset, 'optimizer': opt, 'n_seeds': len(seeds),
                'kappa_mean': float(ks.mean()),
                'kappa_median': float(np.median(ks)),
                'erank_mean': float(es.mean()),
                'best_test_mean': float(np.mean(test_rmses)) if test_rmses else None,
            }
            summary_rows.append(row)
            print(f"{subset:<7} {opt:<12} {len(seeds):<3} {row['kappa_mean']:<12.2e} {'-':<14} "
                  f"{'-':<14} {row['erank_mean']:<15.2f} {'-':<15} {row['best_test_mean']:<10.2f}")

    # Save summary JSON
    out_json = os.path.join(args.save_dir, '_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary_rows, f, indent=2, default=float)
    print(f"\nSaved summary to {out_json}")

    # LaTeX table
    print("\n=== LaTeX: tab:spectral_stepwise ===")
    print(r"\begin{tabular}{@{}llrrrr@{}}")
    print(r"\toprule")
    print(r"Subset & Optimizer & median $\kappa$ pre & median $\kappa$ post & reduction & best test RMSE \\")
    print(r"\midrule")
    for row in summary_rows:
        if 'kappa_pre_median' in row:
            print(f"{row['subset']} & {row['optimizer']} & {row['kappa_pre_median']:.2e} & "
                  f"{row['kappa_post_median']:.2e} & {row['reduction_factor']:.1e}$\\times$ & "
                  f"{row['best_test_mean']:.2f} \\\\")
        else:
            print(f"{row['subset']} & {row['optimizer']} & {row['kappa_median']:.2e} & "
                  f"-- & -- & {row['best_test_mean']:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
