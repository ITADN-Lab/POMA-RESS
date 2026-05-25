"""Aggregate SECOM full from 14号: per-optimizer best LR + significance vs AdamW."""
import os, json, glob, argparse, sys
from collections import defaultdict
import numpy as np


def bootstrap_p(a, b, n_boot=10000, seed=2026):
    a = np.array(a); b = np.array(b)
    n = min(len(a), len(b))
    if n < 2: return float('nan')
    a = a[:n]; b = b[:n]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append((b[idx] - a[idx]).mean())
    diffs = np.array(diffs)
    return float(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True)
    args = ap.parse_args()
    runs = []
    for f in sorted(glob.glob(os.path.join(args.save_dir, 'secom_*.json'))):
        try:
            d = json.load(open(f))
            cfg = d.get('config', {})
            runs.append({
                'opt': cfg.get('optimizer'),
                'lr': float(cfg.get('lr', 0)),
                'seed': int(cfg.get('seed', 0)),
                'best_val_acc': d.get('best_val_acc'),
                'best_test_acc': d.get('best_test_acc'),
            })
        except Exception as e:
            print(f"WARN: {f}: {e}", file=sys.stderr)
    print(f"Loaded {len(runs)} SECOM runs")
    if not runs: return

    # Group by (opt, lr) → seeds list
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[(r['opt'], r['lr'])].append(r)
    # Best LR per opt by mean val_acc
    by_opt = defaultdict(list)
    for (opt, lr), lst in by_cfg.items():
        if not lst: continue
        mean_val = np.mean([x['best_val_acc'] for x in lst if x['best_val_acc']])
        by_opt[opt].append((lr, mean_val, lst))
    best = {}
    for opt, cands in by_opt.items():
        best[opt] = max(cands, key=lambda x: x[1])  # (lr, val, runs)

    print(f"\n{'Optimizer':<12} {'LR':<10} {'N':<3} {'best_test_acc (mean±std)':<28}")
    print('-' * 60)
    test_arrays = {}
    for opt in sorted(best):
        lr, val, lst = best[opt]
        accs = [r['best_test_acc'] for r in lst if r['best_test_acc']]
        m, s = np.mean(accs), np.std(accs, ddof=1) if len(accs) > 1 else 0
        test_arrays[opt] = accs
        print(f"{opt:<12} {lr:<10.4f} {len(accs):<3} {m:.4f}±{s:.4f}")

    print("\n=== Significance vs AdamW (paired bootstrap, two-sided) ===")
    if 'AdamW' in test_arrays:
        for opt in sorted(test_arrays):
            if opt == 'AdamW': continue
            p = bootstrap_p(test_arrays['AdamW'], test_arrays[opt])
            d = np.mean(test_arrays[opt]) - np.mean(test_arrays['AdamW'])
            print(f"  AdamW vs {opt}: Δ={d:+.4f}, p={p:.4f}")

    print("\n=== LaTeX: tab:secom ===")
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(r"Optimizer & LR & N & Test Accuracy & vs AdamW \\")
    print(r"\midrule")
    for opt in sorted(best, key=lambda o: -np.mean(test_arrays.get(o, [0]))):
        lr, _, lst = best[opt]
        accs = test_arrays[opt]
        m, s = np.mean(accs), np.std(accs, ddof=1) if len(accs) > 1 else 0
        vs = '---' if opt == 'AdamW' else f"$p{{=}}{bootstrap_p(test_arrays.get('AdamW', []), accs):.3f}$"
        print(f"{opt} & ${lr:.0e}$ & {len(accs)} & {m*100:.2f}$\\pm${s*100:.2f}\\% & {vs} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
