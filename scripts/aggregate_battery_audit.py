"""
Aggregate NASA Battery audit (Plan B v2 minimal replication).
4 candidates × 2 partitions × 3 seeds (paired vs AdamW).
"""
import os, json, glob, argparse
import numpy as np

CANDIDATES = ['PMO', 'Adan', 'RAdam', 'Lion']
PARTITIONS = [2024, 7]


def load_runs(res_dir, opt, sp):
    runs = []
    for f in glob.glob(os.path.join(res_dir, f'battery_{opt}_seed*_*.json')):
        try:
            d = json.load(open(f))
            if d['split_info']['split_seed'] != sp: continue
            c = d['config']
            if c['optimizer'] != opt: continue
            runs.append({'beta1': c['beta1'], 'gc': c['grad_clip'], 'lr': c['lr'],
                         'seed': c['seed'], 'val': d['best_val_rmse'],
                         'test': d['best_test_rmse']})
        except Exception: pass
    return runs


def best_of_grid(runs, seed=42):
    g = [r for r in runs if r['seed'] == seed]
    return min(g, key=lambda r: r['val']) if g else None


def paired_stats(a, b):
    a, b = np.array(a, float), np.array(b, float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    diff = b - a
    rng = np.random.RandomState(2024)
    boot = np.sort([rng.choice(diff, n, replace=True).mean() for _ in range(10000)])
    return {'n': n, 'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'delta': float(diff.mean()),
            'ci': [float(boot[250]), float(boot[9750])],
            'p_boot': float(2 * min((boot > 0).mean(), (boot < 0).mean())),
            'dz': float(diff.mean() / (diff.std(ddof=1) + 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res_dir', default=os.path.expanduser('~/pmo_battery/results'))
    args = ap.parse_args()

    md = ["# NASA Battery audit panel (Plan B v2 minimal replication)\n",
          "Cross-domain (electrochemical, vs C-MAPSS turbofan + FEMTO bearings) — "
          "4 candidates × 2 partitions × 3 seeds. Per-partition Phase A (12-config "
          "grid) picks each opt's best, then 3 seeds × best-config paired vs AdamW.\n",
          "| Opt | sp | AdamW (mean) | Opt (mean) | Δ | 95% CI | p | dz | n |",
          "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for sp in PARTITIONS:
        adamw_runs = load_runs(args.res_dir, 'AdamW', sp)
        best_aw = best_of_grid(adamw_runs, 42)
        if best_aw is None:
            print(f"WARN no AdamW for sp{sp}"); continue
        matched_aw = sorted([r for r in adamw_runs
            if abs(r['beta1']-best_aw['beta1'])<1e-9
            and abs(r['gc']-best_aw['gc'])<1e-9
            and abs(r['lr']-best_aw['lr'])<1e-12], key=lambda r: r['seed'])
        for opt in CANDIDATES:
            opt_runs = load_runs(args.res_dir, opt, sp)
            best_o = best_of_grid(opt_runs, 42)
            if best_o is None: continue
            matched_o = sorted([r for r in opt_runs
                if abs(r['beta1']-best_o['beta1'])<1e-9
                and abs(r['gc']-best_o['gc'])<1e-9
                and abs(r['lr']-best_o['lr'])<1e-12], key=lambda r: r['seed'])
            if len(matched_aw) < 3 or len(matched_o) < 3: continue
            st = paired_stats([r['test'] for r in matched_aw],
                              [r['test'] for r in matched_o])
            summary.setdefault(opt, {})[sp] = st
            mk = "**" if st['p_boot'] < 0.05 else ""
            md.append(f"| {opt} | {sp} | {st['mean_a']:.2f} | {st['mean_b']:.2f} | "
                      f"{mk}{st['delta']:+.2f}{mk} | "
                      f"[{st['ci'][0]:+.2f}, {st['ci'][1]:+.2f}] | "
                      f"{st['p_boot']:.3f} | {st['dz']:+.2f} | {st['n']} |")
    md.append("")
    md.append("## Headline (Δ vs AdamW, bold = sig p<0.05)\n")
    md.append("| Opt | sp=2024 | sp=7 |")
    md.append("|---|---|---|")
    for opt in CANDIDATES:
        d = summary.get(opt, {})
        row = [opt]
        for sp in PARTITIONS:
            x = d.get(sp)
            if x is None: row.append("—")
            else:
                mk = "**" if x['p_boot'] < 0.05 else ""
                row.append(f"{mk}{x['delta']:+.2f}{mk} (p={x['p_boot']:.3f})")
        md.append("| " + " | ".join(row) + " |")
    out_md = os.path.join(args.res_dir, 'BATTERY_AUDIT_ANALYSIS.md')
    open(out_md, 'w').write("\n".join(md) + "\n")
    json.dump(summary, open(os.path.join(args.res_dir, 'battery_summary.json'), 'w'),
              indent=2)
    print("\n".join(md)); print(f"\nwrote {out_md}")


if __name__ == '__main__':
    main()
