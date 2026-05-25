"""
Aggregate RAdam-FD004 partition × 20-seed: compute paired Δ(RAdam - AdamW) at
each partition using the partition-specific best configurations.
"""
import os, json, glob
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(THIS, '..', 'results_leakfree')
SUBSET = 'FD004'
PARTITIONS = [2024, 7, 99]


def best_cfg(opt, sp, seed=42):
    best = None
    for f in glob.glob(os.path.join(RES, f'lf_{SUBSET}_{opt}_seed{seed}_*.json')):
        try:
            d = json.load(open(f))
            if d.get('split_info', {}).get('split_seed') != sp: continue
            c = d['config']
            if c['optimizer'] != opt or c['subset'] != SUBSET: continue
            v = d['best_val_rmse']
            if best is None or v < best[0]:
                best = (v, c['beta1'], c['grad_clip'], c['lr'])
        except: pass
    return best


def matched_seeds(opt, sp, b1, gc, lr):
    out = []
    for f in glob.glob(os.path.join(RES, f'lf_{SUBSET}_{opt}_seed*_*.json')):
        try:
            d = json.load(open(f))
            if d.get('split_info', {}).get('split_seed') != sp: continue
            c = d['config']
            if c['optimizer'] != opt: continue
            if abs(c['beta1']-b1)>1e-9 or abs(c['grad_clip']-gc)>1e-9 or abs(c['lr']-lr)>1e-12: continue
            out.append({'seed': c['seed'], 'test': d['best_test_rmse']})
        except: pass
    return sorted(out, key=lambda r: r['seed'])


def paired_stats(a, b):
    a, b = np.array(a, float), np.array(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = b - a
    rng = np.random.RandomState(2024)
    boot = np.sort([rng.choice(diff, n, replace=True).mean() for _ in range(10000)])
    return {
        'n': n,
        'mean_aw': float(a.mean()), 'sd_aw': float(a.std(ddof=1)),
        'mean_rd': float(b.mean()), 'sd_rd': float(b.std(ddof=1)),
        'delta': float(diff.mean()),
        'ci': [float(boot[250]), float(boot[9750])],
        'p_boot': float(2 * min((boot > 0).mean(), (boot < 0).mean())),
        'dz': float(diff.mean() / (diff.std(ddof=1) + 1e-12)),
    }


def main():
    md = ["# RAdam-FD004 partition × 20-seed verification\n"]
    md.append("Per-partition best configurations re-run at 20 seeds; paired Δ(RAdam $-$ AdamW) reported.\n")
    md.append("| Partition | AdamW best cfg | RAdam best cfg | AdamW (mean±sd) | RAdam (mean±sd) | Δ | 95% CI | p | dz | n |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for sp in PARTITIONS:
        aw_b = best_cfg('AdamW', sp)
        rd_b = best_cfg('RAdam', sp)
        if aw_b is None or rd_b is None:
            md.append(f"| sp{sp} | missing | | | | | | | | |")
            continue
        aw_runs = matched_seeds('AdamW', sp, aw_b[1], aw_b[2], aw_b[3])
        rd_runs = matched_seeds('RAdam', sp, rd_b[1], rd_b[2], rd_b[3])
        if len(aw_runs) < 2 or len(rd_runs) < 2:
            md.append(f"| sp{sp} | n_aw={len(aw_runs)} n_rd={len(rd_runs)} | | | | | | | | |")
            continue
        st = paired_stats([r['test'] for r in aw_runs], [r['test'] for r in rd_runs])
        aw_cfg = f"b{aw_b[1]} gc{aw_b[2]} lr{aw_b[3]:.0e}"
        rd_cfg = f"b{rd_b[1]} gc{rd_b[2]} lr{rd_b[3]:.0e}"
        md.append(f"| sp{sp} | {aw_cfg} | {rd_cfg} | {st['mean_aw']:.2f}±{st['sd_aw']:.2f} | "
                  f"{st['mean_rd']:.2f}±{st['sd_rd']:.2f} | {st['delta']:+.2f} | "
                  f"[{st['ci'][0]:+.2f},{st['ci'][1]:+.2f}] | "
                  f"{st['p_boot']:.3f} | {st['dz']:+.2f} | {st['n']} |")
    out_md = os.path.join(RES, 'RADAM_PARTITION_20SEED.md')
    with open(out_md, 'w') as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_md}")


if __name__ == '__main__':
    main()
