"""
FEMTO multi-partition robustness aggregator. For each (condition, optimizer),
report 20-seed paired Δ vs AdamW under each of 3 leave-bearings-out partitions
(sp 2024 / 7 / 99). Cond3 omitted as boundary (only 3 bearings).
"""
import os, json, glob, argparse
import numpy as np

CONDITIONS = ['cond1', 'cond2']
ALL_OPTS = ['PMO', 'Adan', 'RAdam', 'Lion']
PARTITIONS = [2024, 7, 88, 1, 2]   # 5 distinct (val_idx,test_idx) partitions


def load_runs(res_dir, opt, split_seed):
    runs = []
    for f in glob.glob(os.path.join(res_dir, f'femto_*_{opt}_seed*_*.json')):
        try:
            d = json.load(open(f))
            c = d['config']
            if c['optimizer'] != opt: continue
            if d.get('split_info', {}).get('split_seed') != split_seed: continue
            runs.append({
                'condition': c['condition'], 'beta1': c['beta1'],
                'gc': c['grad_clip'], 'lr': c['lr'], 'seed': c['seed'],
                'val': d['best_val_rmse'], 'test': d['best_test_rmse']})
        except Exception: pass
    return runs


def paired_stats(a, b):
    a, b = np.array(a, float), np.array(b, float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
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


def best_of_grid(runs, seed=42):
    grid = [r for r in runs if r['seed'] == seed]
    if not grid: return None
    return min(grid, key=lambda g: g['val'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res_dir', default=os.path.expanduser('~/pmo_femto/results'))
    args = ap.parse_args()

    md = ["# FEMTO multi-partition robustness check (Plan B1)\n",
          "Each (condition, optimizer) re-run at each partition's own best 36-config × 20 seeds. "
          "cond3 omitted (only 3 bearings → degenerate splits regardless of seed). "
          f"Each partition draws a different held-out test bearing ({len(PARTITIONS)} partitions total).\n",
          "| Cond | Opt | " + " | ".join(f"sp={p}" for p in PARTITIONS) + " | replication |",
          "|---|---|" + "|".join(["---"] * len(PARTITIONS)) + "|---|"]
    summary = {}
    for cond in CONDITIONS:
        for opt in ALL_OPTS:
            row = [cond, opt]
            sigwins = ties = sigloss = 0
            for sp in PARTITIONS:
                adamw_runs = [r for r in load_runs(args.res_dir, 'AdamW', sp)
                              if r['condition'] == cond]
                opt_runs = [r for r in load_runs(args.res_dir, opt, sp)
                            if r['condition'] == cond]
                best_aw = best_of_grid(adamw_runs, 42)
                best_opt = best_of_grid(opt_runs, 42)
                if best_aw is None or best_opt is None:
                    row.append("—"); continue
                m_aw = sorted([r for r in adamw_runs
                    if abs(r['beta1']-best_aw['beta1'])<1e-9
                    and abs(r['gc']-best_aw['gc'])<1e-9
                    and abs(r['lr']-best_aw['lr'])<1e-12], key=lambda r: r['seed'])
                m_o = sorted([r for r in opt_runs
                    if abs(r['beta1']-best_opt['beta1'])<1e-9
                    and abs(r['gc']-best_opt['gc'])<1e-9
                    and abs(r['lr']-best_opt['lr'])<1e-12], key=lambda r: r['seed'])
                if len(m_aw) < 5 or len(m_o) < 5:
                    row.append(f"n<5 ({len(m_o)})"); continue
                st = paired_stats([r['test'] for r in m_aw],
                                  [r['test'] for r in m_o])
                summary.setdefault(cond, {}).setdefault(opt, {})[sp] = st
                marker = "**" if st['p_boot'] < 0.05 else ""
                row.append(f"{marker}{st['delta']:+.2f}{marker} (p={st['p_boot']:.3f}, n={st['n']})")
                if st['p_boot'] < 0.05:
                    if st['delta'] < 0: sigwins += 1
                    else: sigloss += 1
                else:
                    ties += 1
            row.append(f"{sigwins} sig win / {ties} tie / {sigloss} sig loss")
            md.append("| " + " | ".join(row) + " |")

    md.append("")
    md.append("## Headline (signed Δ in test RMSE; bold = significant p<0.05)\n")
    md.append("| Condition | Optimizer | wins replicate at | verdict |")
    md.append("|---|---|---|---|")
    for cond in CONDITIONS:
        for opt in ALL_OPTS:
            d = summary.get(cond, {}).get(opt, {})
            wins = [sp for sp, st in d.items() if st['p_boot'] < 0.05 and st['delta'] < 0]
            losses = [sp for sp, st in d.items() if st['p_boot'] < 0.05 and st['delta'] > 0]
            n_part = len(PARTITIONS)
            n_wins = len(wins)
            n_loss = len(losses)
            if n_wins == n_part:
                verdict = f"ROBUST WIN ({n_wins}/{n_part})"
            elif n_wins >= 2 and n_loss == 0:
                verdict = f"PARTITION-CONDITIONAL WIN ({n_wins}/{n_part})"
            elif n_wins >= 2 and n_loss >= 1:
                verdict = f"MIXED ({n_wins} win / {n_loss} loss / {n_part-n_wins-n_loss} tie)"
            elif n_wins >= 1:
                verdict = f"PARTIAL ({n_wins}/{n_part})"
            elif n_loss:
                verdict = f"NET LOSS ({n_loss}/{n_part})"
            else:
                verdict = "TIE"
            md.append(f"| {cond} | {opt} | sp{wins or '—'} | {verdict} |")

    out_md = os.path.join(args.res_dir, 'FEMTO_MULTIPART_ANALYSIS.md')
    with open(out_md, 'w') as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.res_dir, 'femto_multipart_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md))
    print(f"\nwrote {out_md}")


if __name__ == '__main__':
    main()
