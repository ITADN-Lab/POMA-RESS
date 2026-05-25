"""
Winner-flip and selection-regret analysis (Codex gpt-5.5 Plan A #3).
For each subset and each candidate optimizer, simulate "draw B configs, keep
best by engine-val, report (test, winner-vs-AdamW)" over many random draws,
and report:
  - selection regret: expected test gap vs the full-budget best.
  - winner-flip rate: fraction of draws whose PMO-vs-AdamW conclusion differs
    from the full-budget conclusion.
"""
import os, json, glob
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, '..', 'results_leakfree')
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
OPTS = ['AdamW', 'PMO', 'Adan']
BUDGETS = [1, 2, 3, 6, 9, 18, 36]
N_DRAWS = 4000


def load_grid(subset, opt):
    grid = []
    for f in glob.glob(os.path.join(RES, f'lf_{subset}_{opt}_seed42_*.json')):
        try:
            d = json.load(open(f))
            if d.get('split_info', {}).get('split_seed') != 2024:
                continue
            c = d['config']
            grid.append({'val': d['best_val_rmse'],
                         'test': d['best_test_rmse']})
        except Exception:
            pass
    return grid


def best_of_B(grid, B, rng):
    val = np.array([g['val'] for g in grid])
    test = np.array([g['test'] for g in grid])
    idx = rng.choice(len(grid), size=B, replace=False)
    return float(test[idx[np.argmin(val[idx])]])


def main():
    rng = np.random.RandomState(12345)
    md = ["# Winner-flip and selection-regret analysis\n",
          "How does the AdamW-vs-method comparison change with tuning budget? "
          "For each (subset, method) we draw B configurations from the 36-config "
          "leak-free grid, keep the best by engine-level val, and ask: (1) does "
          "the winner (lower test) match the full-budget winner? (2) what is "
          "the test-RMSE regret vs the full-budget best?\n"]
    md.append("| Subset | Comparison | B | Winner agreement (%) | Selection regret |")
    md.append("|---|---|---|---|---|")
    summary = {}
    for s in SUBSETS:
        for method in ('PMO', 'Adan'):
            g_aw = load_grid(s, 'AdamW')
            g_m = load_grid(s, method)
            if len(g_aw) < 36 or len(g_m) < 36:
                continue
            # full-budget winner (lowest val → its test):
            val_aw = np.array([g['val'] for g in g_aw])
            test_aw = np.array([g['test'] for g in g_aw])
            val_m = np.array([g['val'] for g in g_m])
            test_m = np.array([g['test'] for g in g_m])
            full_aw_test = float(test_aw[np.argmin(val_aw)])
            full_m_test = float(test_m[np.argmin(val_m)])
            full_winner = method if full_m_test < full_aw_test else 'AdamW'
            summary.setdefault(s, {})[method] = {
                'full_aw': full_aw_test, 'full_m': full_m_test,
                'full_winner': full_winner, 'by_budget': {}}
            for B in BUDGETS:
                agree, regret_aw, regret_m = 0, [], []
                for _ in range(N_DRAWS):
                    ia = rng.choice(36, B, replace=False)
                    im = rng.choice(36, B, replace=False)
                    t_a = float(test_aw[ia[np.argmin(val_aw[ia])]])
                    t_m = float(test_m[im[np.argmin(val_m[im])]])
                    w = method if t_m < t_a else 'AdamW'
                    if w == full_winner:
                        agree += 1
                    regret_aw.append(t_a - full_aw_test)
                    regret_m.append(t_m - full_m_test)
                ag = agree / N_DRAWS * 100
                reg_aw, reg_m = float(np.mean(regret_aw)), float(np.mean(regret_m))
                summary[s][method]['by_budget'][B] = {
                    'winner_agree_pct': ag,
                    'regret_AdamW': reg_aw,
                    'regret_method': reg_m,
                }
                md.append(f"| {s} | {method} vs AdamW | {B} | {ag:.0f}% "
                          f"| AW {reg_aw:+.2f}, {method} {reg_m:+.2f} |")
            md.append("")

    with open(os.path.join(RES, 'WINNER_FLIP_ANALYSIS.md'), 'w') as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RES, 'winner_flip_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md))
    print(f"\nwrote {RES}/WINNER_FLIP_ANALYSIS.md")


if __name__ == '__main__':
    main()
