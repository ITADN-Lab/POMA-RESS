"""
Aggregate FEMTO audit results (B1 Plan): compare each candidate optimizer
(PMO, Adan, RAdam, Lion) against tuned AdamW under leak-free bearing-level
equal-budget protocol, Phase B 20-seed verdict per (condition, optimizer).
"""
import os, json, glob
import numpy as np
import argparse

CONDITIONS = ['cond1', 'cond2', 'cond3']
ALL_OPTS = ['PMO', 'Adan', 'RAdam', 'Lion']


def load_runs(res_dir, opt, split_seed=2024):
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


def best_of_grid(runs, seed=42):
    grid = [r for r in runs if r['seed'] == seed]
    if not grid: return None
    return min(grid, key=lambda g: g['val'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res_dir', default=os.path.expanduser('~/pmo_femto/results'))
    args = ap.parse_args()

    md = ["# FEMTO real-industrial audit panel (Plan B1)\n",
          "Leak-free bearing-level equal-budget protocol applied to PMO, Adan, RAdam, Lion vs AdamW on PHM 2012 PRONOSTIA bearing data. Phase B 20-seed paired bootstrap.\n",
          "| Condition | Optimizer | AdamW (mean±sd) | Opt (mean±sd) | Δ(Opt−AW) | p | dz | n |",
          "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for c in CONDITIONS:
        adamw_runs = load_runs(args.res_dir, 'AdamW')
        adamw_c = [r for r in adamw_runs if r['condition'] == c]
        best_aw = best_of_grid(adamw_c, seed=42)
        if best_aw is None:
            md.append(f"| {c} | — | no AdamW Phase A | | | | | 0 |")
            continue
        matched_aw = sorted([r for r in adamw_c
            if abs(r['beta1']-best_aw['beta1'])<1e-9
            and abs(r['gc']-best_aw['gc'])<1e-9
            and abs(r['lr']-best_aw['lr'])<1e-12], key=lambda r: r['seed'])
        for opt in ALL_OPTS:
            opt_runs = load_runs(args.res_dir, opt)
            opt_c = [r for r in opt_runs if r['condition'] == c]
            best_opt = best_of_grid(opt_c, seed=42)
            if best_opt is None: continue
            matched_opt = sorted([r for r in opt_c
                if abs(r['beta1']-best_opt['beta1'])<1e-9
                and abs(r['gc']-best_opt['gc'])<1e-9
                and abs(r['lr']-best_opt['lr'])<1e-12], key=lambda r: r['seed'])
            if len(matched_aw) < 2 or len(matched_opt) < 2:
                md.append(f"| {c} | {opt} | n<2 (incomplete) | | | | | {len(matched_opt)} |")
                continue
            st = paired_stats([r['test'] for r in matched_aw],
                              [r['test'] for r in matched_opt])
            summary.setdefault(c, {})[opt] = {**st,
                'cfg': {'b1': best_opt['beta1'], 'gc': best_opt['gc'], 'lr': best_opt['lr']}}
            md.append(f"| {c} | {opt} | {st['mean_a']:.2f}±{st['sd_a']:.2f} | "
                      f"{st['mean_b']:.2f}±{st['sd_b']:.2f} | "
                      f"{st['delta']:+.2f} | {st['p_boot']:.3f} | "
                      f"{st['dz']:+.2f} | {st['n']} |")
    md.append("")
    md.append("## Headline (Δ in test RMSE; bold = significant p<0.05)\n")
    md.append("| Condition | PMO Δ | Adan Δ | RAdam Δ | Lion Δ |")
    md.append("|---|---|---|---|---|")
    for c in CONDITIONS:
        row = [f"{c}"]
        for opt in ALL_OPTS:
            x = summary.get(c, {}).get(opt)
            if x is None:
                row.append("—")
            else:
                d = x['delta']; p = x['p_boot']
                row.append(f"**{d:+.2f}** (p={p:.3f})" if p < 0.05 else f"{d:+.2f} (p={p:.3f})")
        md.append("| " + " | ".join(row) + " |")
    out_md = os.path.join(args.res_dir, 'FEMTO_AUDIT_ANALYSIS.md')
    with open(out_md, 'w') as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.res_dir, 'femto_audit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md))
    print(f"\nwrote {out_md}")


if __name__ == '__main__':
    main()
