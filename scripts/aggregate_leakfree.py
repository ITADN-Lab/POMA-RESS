"""
Aggregate the leak-free engine-level C-MAPSS re-evaluation (Plan B).

Produces:
  1. Equal-budget result : best-of-36 config (by engine-level validation RMSE)
     for AdamW vs POMA, per subset, with 20-seed error bars (Phase B).
  2. Tuning-budget curve : expected test RMSE of "try B random configs, keep
     the best by engine-val", averaged over random orderings (seed 42).
  3. Partition sensitivity : the equal-budget AdamW-vs-POMA gap recomputed under
     3 independent leave-engines-out validation partitions.

Outputs LEAKFREE_ANALYSIS.md, leakfree_summary.json, and the budget-curve PDF.
"""
import os, sys, json, glob
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
RES = '/home/hadoop/workstation/md/LafTJU-TII/experiments/results_leakfree'
FIG = '/home/hadoop/workstation/md/RESS/paper/figures/tuning_budget_curve.pdf'
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
OPTS = ['AdamW', 'PMO']
BUDGETS = [1, 2, 3, 6, 9, 18, 36]
PARTITIONS = [2024, 7, 99]
N_ORDER = 4000


def load_all():
    runs = []
    for f in glob.glob(os.path.join(RES, 'lf_*.json')):
        try:
            d = json.load(open(f))
            c = d['config']
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
        if B > n:
            continue
        picks = []
        for _ in range(N_ORDER):
            idx = rng.choice(n, size=B, replace=False)
            picks.append(test[idx[np.argmin(val[idx])]])
        picks = np.array(picks)
        out[B] = (float(picks.mean()), float(picks.std()))
    return out


def paired_stats(a, b):
    """Paired diff b-a (per-seed)."""
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
    print(f"loaded {len(runs)} runs")
    if not runs:
        sys.exit("no runs")

    summary = {'equal_budget': {}, 'budget_curve': {}, 'phase_b': {},
               'partition_sensitivity': {}}
    md = ["# Leak-free engine-level C-MAPSS re-evaluation (Plan B)\n",
          f"{len(runs)} runs. Validation = leave-engines-out (disjoint engine "
          "partition). HP selected by engine-level validation RMSE; test RMSE "
          "on the standard C-MAPSS test engines.\n"]

    # ---- Phase B: 20-seed equal-budget comparison (partition 2024) ----
    md.append("## Equal-budget comparison, 20 seeds (Phase B, partition 2024)\n")
    md.append("Best engine-val config per (subset,opt) re-run at 20 seeds.\n")
    md.append("| Subset | AdamW | POMA | Δ(POMA−AW) | 95% CI | p | dz |")
    md.append("|---|---|---|---|---|---|---|")
    for s in SUBSETS:
        sv = {}
        for o in OPTS:
            best, _ = best_of_grid(runs, s, o, 2024, seed=42)
            if best is None:
                continue
            matched = [r for r in runs if r['subset'] == s and r['opt'] == o
                       and r['split_seed'] == 2024
                       and abs(r['beta1'] - best['beta1']) < 1e-9
                       and abs(r['gc'] - best['gc']) < 1e-9
                       and abs(r['lr'] - best['lr']) < 1e-12]
            sv[o] = sorted(matched, key=lambda r: r['seed'])
        if 'AdamW' in sv and 'PMO' in sv and len(sv['AdamW']) > 1:
            st = paired_stats([r['test'] for r in sv['AdamW']],
                              [r['test'] for r in sv['PMO']])
            summary['phase_b'][s] = {**st,
                'adamw_tests': [r['test'] for r in sv['AdamW']],
                'pmo_tests': [r['test'] for r in sv['PMO']]}
            md.append(f"| {s} | {st['mean_a']:.2f}±{st['sd_a']:.2f} | "
                      f"{st['mean_b']:.2f}±{st['sd_b']:.2f} | {st['delta']:+.2f} "
                      f"| [{st['ci'][0]:+.2f},{st['ci'][1]:+.2f}] | "
                      f"{st['p_boot']:.3f} | {st['dz']:+.2f} | "
                      f"(n={st['n']}) |")

    # ---- Partition sensitivity (seed 42, best-of-36 per partition) ----
    md.append("\n## Validation-partition sensitivity (best-of-36, seed 42)\n")
    md.append("Equal-budget best-config test RMSE recomputed under 3 independent "
              "leave-engines-out partitions.\n")
    md.append("| Subset | Partition | AdamW | POMA | Δ(POMA−AW) |")
    md.append("|---|---|---|---|---|")
    for s in SUBSETS:
        summary['partition_sensitivity'][s] = {}
        for sp in PARTITIONS:
            row = {}
            for o in OPTS:
                best, _ = best_of_grid(runs, s, o, sp, seed=42)
                if best:
                    row[o] = best['test']
            if 'AdamW' in row and 'PMO' in row:
                d = row['PMO'] - row['AdamW']
                summary['partition_sensitivity'][s][sp] = {
                    'AdamW': row['AdamW'], 'PMO': row['PMO'], 'delta': d}
                md.append(f"| {s} | sp{sp} | {row['AdamW']:.2f} | "
                          f"{row['PMO']:.2f} | {d:+.2f} |")

    # ---- Budget curve (partition 2024, seed 42) ----
    md.append("\n## Tuning-budget curve (partition 2024, seed 42)\n")
    for s in SUBSETS:
        for o in OPTS:
            _, grid = best_of_grid(runs, s, o, 2024, seed=42)
            if len(grid) >= 36:
                summary['budget_curve'].setdefault(s, {})[o] = budget_curve(grid)
            elif grid:
                print(f"WARN {s}/{o}: {len(grid)}/36 grid configs")
    for s in SUBSETS:
        bc = summary['budget_curve'].get(s, {})
        if 'AdamW' not in bc or 'PMO' not in bc:
            continue
        md.append(f"\n**{s}** — Δ(POMA−AW) by budget: " +
                  ", ".join(f"B{B}:{bc['PMO'][B][0]-bc['AdamW'][B][0]:+.2f}"
                            for B in BUDGETS if B in bc['AdamW']))

    with open(os.path.join(RES, 'leakfree_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RES, 'LEAKFREE_ANALYSIS.md'), 'w') as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {RES}/LEAKFREE_ANALYSIS.md + leakfree_summary.json")
    plot_budget_curve(summary['budget_curve'])


def plot_budget_curve(bc_all):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}")
        return
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    colors = {'AdamW': '#c0392b', 'PMO': '#2c3e50'}
    for ax, s in zip(axes, SUBSETS):
        bc = bc_all.get(s, {})
        if not bc:
            ax.set_visible(False)
            continue
        for o in OPTS:
            if o not in bc:
                continue
            Bs = [B for B in BUDGETS if B in bc[o]]
            ax.plot(Bs, [bc[o][B][0] for B in Bs], 'o-', color=colors[o],
                    label=('POMA' if o=='PMO' else o), ms=4, lw=1.6)
        ax.set_xscale('log'); ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(B) for B in BUDGETS], fontsize=7)
        ax.set_title(s, fontsize=10)
        ax.set_xlabel('tuning budget (configs tried)', fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('expected test RMSE', fontsize=9)
    axes[0].legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches='tight')
    print(f"wrote {FIG}")


if __name__ == '__main__':
    main()
