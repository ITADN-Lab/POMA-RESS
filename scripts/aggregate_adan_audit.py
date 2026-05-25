"""
Aggregate the Adan-vs-AdamW audit case study (Plan B second case study).
Same structure as aggregate_leakfree.py but for AdamW vs Adan; reuses the
existing AdamW results in results_leakfree/.
"""
import os, sys, json, glob
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, '..', 'results_leakfree')
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
OPTS = ['AdamW', 'Adan']
BUDGETS = [1, 2, 3, 6, 9, 18, 36]
PARTITIONS = [2024, 7, 99]
N_ORDER = 4000


def load_all():
    runs = []
    for f in glob.glob(os.path.join(RES, 'lf_*.json')):
        try:
            d = json.load(open(f))
            c = d['config']
            if c['optimizer'] not in OPTS:
                continue
            runs.append({
                'subset': c['subset'], 'opt': c['optimizer'],
                'beta1': c['beta1'], 'gc': c['grad_clip'], 'lr': c['lr'],
                'seed': c['seed'],
                'split_seed': d.get('split_info', {}).get('split_seed', 2024),
                'val': d['best_val_rmse'], 'test': d['best_test_rmse'],
            })
        except Exception as e:
            print(f"WARN skip {f}: {e}")
    return runs


def budget_curve(grid):
    val = np.array([g['val'] for g in grid])
    test = np.array([g['test'] for g in grid])
    n = len(grid)
    rng = np.random.RandomState(12345)
    out = {}
    for B in BUDGETS:
        if B > n: continue
        picks = []
        for _ in range(N_ORDER):
            idx = rng.choice(n, size=B, replace=False)
            picks.append(test[idx[np.argmin(val[idx])]])
        picks = np.array(picks)
        out[B] = (float(picks.mean()), float(picks.std()))
    return out


def paired_stats(a, b):
    a, b = np.array(a, float), np.array(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = b - a
    rng = np.random.RandomState(2024)
    boot = np.sort([rng.choice(diff, n, replace=True).mean() for _ in range(10000)])
    return {
        'n': n, 'mean_a': float(a.mean()), 'sd_a': float(a.std(ddof=1)),
        'mean_b': float(b.mean()), 'sd_b': float(b.std(ddof=1)),
        'delta': float(diff.mean()),
        'ci': [float(boot[250]), float(boot[9750])],
        'p_boot': float(2 * min((boot > 0).mean(), (boot < 0).mean())),
        'dz': float(diff.mean() / (diff.std(ddof=1) + 1e-12)),
    }


def best_of_grid(runs, subset, opt, split_seed, seed=42):
    grid = [r for r in runs if r['subset'] == subset and r['opt'] == opt
            and r['split_seed'] == split_seed and r['seed'] == seed]
    if not grid:
        return None, []
    return min(grid, key=lambda g: g['val']), grid


def main():
    runs = load_all()
    print(f"loaded {len(runs)} AdamW/Adan runs")
    if not runs:
        sys.exit("no runs")

    summary = {'equal_budget': {}, 'budget_curve': {}, 'phase_b': {},
               'partition_sensitivity': {}}
    md = ["# Adan vs AdamW: second-case-study audit on C-MAPSS\n",
          f"{len(runs)} AdamW+Adan runs in results_leakfree/. Same leak-free "
          "engine-level equal-budget audit protocol as the PMO case study.\n"]

    # 20-seed Phase B
    md.append("## Equal-budget comparison, 20 seeds (Phase B)\n")
    md.append("| Subset | AdamW | Adan | Δ(Adan−AW) | 95% CI | p | dz |")
    md.append("|---|---|---|---|---|---|---|")
    for s in SUBSETS:
        sv = {}
        for o in OPTS:
            best, _ = best_of_grid(runs, s, o, 2024, seed=42)
            if best is None: continue
            matched = [r for r in runs if r['subset']==s and r['opt']==o
                       and r['split_seed']==2024
                       and abs(r['beta1']-best['beta1'])<1e-9
                       and abs(r['gc']-best['gc'])<1e-9
                       and abs(r['lr']-best['lr'])<1e-12]
            sv[o] = sorted(matched, key=lambda r: r['seed'])
        if 'AdamW' in sv and 'Adan' in sv and len(sv['AdamW'])>1:
            st = paired_stats([r['test'] for r in sv['AdamW']],
                              [r['test'] for r in sv['Adan']])
            summary['phase_b'][s] = {**st,
                'adamw_tests': [r['test'] for r in sv['AdamW']],
                'adan_tests': [r['test'] for r in sv['Adan']]}
            md.append(f"| {s} | {st['mean_a']:.2f}±{st['sd_a']:.2f} | "
                      f"{st['mean_b']:.2f}±{st['sd_b']:.2f} | {st['delta']:+.2f} "
                      f"| [{st['ci'][0]:+.2f},{st['ci'][1]:+.2f}] | "
                      f"{st['p_boot']:.3f} | {st['dz']:+.2f} | (n={st['n']}) |")

    # Partition sensitivity
    md.append("\n## Validation-partition sensitivity (best-of-36, seed 42)\n")
    md.append("| Subset | Partition | AdamW | Adan | Δ(Adan−AW) |")
    md.append("|---|---|---|---|---|")
    for s in SUBSETS:
        for sp in PARTITIONS:
            row = {}
            for o in OPTS:
                best, _ = best_of_grid(runs, s, o, sp, seed=42)
                if best: row[o] = best['test']
            if 'AdamW' in row and 'Adan' in row:
                d = row['Adan'] - row['AdamW']
                summary['partition_sensitivity'].setdefault(s, {})[sp] = {
                    'AdamW': row['AdamW'], 'Adan': row['Adan'], 'delta': d}
                md.append(f"| {s} | sp{sp} | {row['AdamW']:.2f} | "
                          f"{row['Adan']:.2f} | {d:+.2f} |")

    # Budget curve
    md.append("\n## Tuning-budget curve (partition 2024, seed 42)\n")
    for s in SUBSETS:
        for o in OPTS:
            _, grid = best_of_grid(runs, s, o, 2024, seed=42)
            if len(grid) >= 36:
                summary['budget_curve'].setdefault(s, {})[o] = budget_curve(grid)
    for s in SUBSETS:
        bc = summary['budget_curve'].get(s, {})
        if 'AdamW' not in bc or 'Adan' not in bc: continue
        md.append(f"\n**{s}** — Δ(Adan−AW) by budget: " +
                  ", ".join(f"B{B}:{bc['Adan'][B][0]-bc['AdamW'][B][0]:+.2f}"
                            for B in BUDGETS if B in bc['AdamW']))

    with open(os.path.join(RES, 'adan_audit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RES, 'ADAN_AUDIT_ANALYSIS.md'), 'w') as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {RES}/ADAN_AUDIT_ANALYSIS.md + adan_audit_summary.json")


if __name__ == '__main__':
    main()
