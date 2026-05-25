"""
Multi-optimizer partition sensitivity + extended winner-flip for the 4-optimizer
audit panel (PMO/Adan/RAdam/Lion vs AdamW). Generates supplementary tables.
"""
import os, json, glob
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, '..', 'results_leakfree')
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
OPTS = ['PMO', 'Adan', 'RAdam', 'Lion']
PARTITIONS = [2024, 7, 99]
BUDGETS = [1, 2, 3, 6, 9, 18, 36]
N_DRAWS = 4000


def load_grid(opt, subset, split_seed, seed=42):
    runs = []
    for f in glob.glob(os.path.join(RES, f'lf_{subset}_{opt}_seed{seed}_*.json')):
        try:
            d = json.load(open(f))
            if d.get('split_info', {}).get('split_seed') != split_seed:
                continue
            c = d['config']
            if c['optimizer'] != opt or c['subset'] != subset:
                continue
            runs.append({'beta1': c['beta1'], 'gc': c['grad_clip'],
                         'lr': c['lr'], 'val': d['best_val_rmse'],
                         'test': d['best_test_rmse']})
        except Exception:
            pass
    return runs


def best_of(grid):
    return min(grid, key=lambda g: g['val']) if grid else None


def main():
    md = ["# Multi-optimizer partition sensitivity + winner-flip (full panel)\n",
          "Extends partition sensitivity and winner-flip rate to all four "
          "case-study optimizers (PMO, Adan, RAdam, Lion) vs AdamW.\n"]

    # === partition sensitivity ===
    md.append("## Partition sensitivity (best-of-36 seed-42 Δ per partition)\n")
    md.append("| Subset | Optimizer | sp2024 Δ | sp7 Δ | sp99 Δ |")
    md.append("|---|---|---|---|---|")
    psens = {}
    for s in SUBSETS:
        for o in OPTS:
            row = {}
            adamw_row = {}
            for sp in PARTITIONS:
                opt_grid = load_grid(o, s, sp)
                aw_grid = load_grid('AdamW', s, sp)
                best_o = best_of(opt_grid)
                best_aw = best_of(aw_grid)
                if best_o and best_aw:
                    row[sp] = best_o['test'] - best_aw['test']
                    adamw_row[sp] = best_aw['test']
            if len(row) == 3:
                psens.setdefault(s, {})[o] = row
                md.append(f"| {s} | {o} | {row[2024]:+.2f} | {row[7]:+.2f} | {row[99]:+.2f} |")

    # === winner-flip (full 16 pairs) ===
    md.append("\n## Winner-flip rate (full 4-optimizer panel, 16 subset-optimizer pairs)\n")
    md.append("| Subset | Opt | B=1 | B=2 | B=3 | B=6 | B=9 | B=18 | B=36 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    rng = np.random.RandomState(12345)
    wflip = {}
    all_agree = {B: [] for B in BUDGETS}
    for s in SUBSETS:
        for o in OPTS:
            g_aw = load_grid('AdamW', s, 2024)
            g_o = load_grid(o, s, 2024)
            if len(g_aw) < 36 or len(g_o) < 36:
                md.append(f"| {s} | {o} | (incomplete: AdamW {len(g_aw)}, {o} {len(g_o)}) | | | | | | |")
                continue
            val_aw = np.array([r['val'] for r in g_aw])
            test_aw = np.array([r['test'] for r in g_aw])
            val_o = np.array([r['val'] for r in g_o])
            test_o = np.array([r['test'] for r in g_o])
            full_aw_test = float(test_aw[np.argmin(val_aw)])
            full_o_test = float(test_o[np.argmin(val_o)])
            full_winner = o if full_o_test < full_aw_test else 'AdamW'
            cells = []
            wflip.setdefault(s, {})[o] = {}
            for B in BUDGETS:
                agree = 0
                for _ in range(N_DRAWS):
                    ia = rng.choice(36, B, replace=False)
                    im = rng.choice(36, B, replace=False)
                    t_a = float(test_aw[ia[np.argmin(val_aw[ia])]])
                    t_o = float(test_o[im[np.argmin(val_o[im])]])
                    w = o if t_o < t_a else 'AdamW'
                    if w == full_winner: agree += 1
                pct = agree / N_DRAWS * 100
                wflip[s][o][B] = pct
                all_agree[B].append(pct)
                cells.append(f"{pct:.0f}")
            md.append(f"| {s} | {o} | " + " | ".join(cells) + " |")
    # Mean / worst across 16 pairs
    md.append("")
    md.append("**Aggregate across 16 (subset, optimizer) pairs:**\n")
    md.append("| Budget B | Mean agreement | Worst pair | Mean flip rate |")
    md.append("|---|---|---|---|")
    for B in BUDGETS:
        if not all_agree[B]: continue
        mean_ag = float(np.mean(all_agree[B]))
        worst = float(min(all_agree[B]))
        flip = 100 - mean_ag
        md.append(f"| {B} | {mean_ag:.0f}% | {worst:.0f}% | {flip:.0f}% |")

    out_md = os.path.join(RES, 'MULTI_PARTITION_WINNERFLIP.md')
    with open(out_md, 'w') as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_md}")


if __name__ == '__main__':
    main()
