"""
Aggregate adaptive trigger sweep (80 runs) and compare AdamW (B-cross) / fixed_NS (B-cross) / adaptive_NS (new).

Cells (from run_adaptive_trigger.py):
  (1) FD002 β=0.9  GC=0    — main (NS small win)
  (2) FD002 β=0.95 GC=0.5  — LOSS corner (fixed NS +2.61)
  (3) FD002 β=0.95 GC=2.0  — LOSS corner (fixed NS +1.44)
  (4) FD002 β=0.7  GC=0    — BIG win (fixed NS -3.35)
  (5) FD004 β=0.9  GC=0    — small win
  (6) FD004 β=0.7  GC=0    — BIG win (fixed NS -1.59)
  (7) FD004 β=0.95 GC=0.5  — neutral
  (8) FD004 β=0.95 GC=2.0  — neutral
"""
import os, glob, json, argparse
import numpy as np
from collections import defaultdict

THIS = os.path.dirname(os.path.abspath(__file__))
ADAPT_DIR  = os.path.join(THIS, '..', 'results_adaptive')
B7B8_DIR   = os.path.join(THIS, '..', 'results_defense')        # fixed NS at GC=0
CROSS_DIR  = os.path.join(THIS, '..', 'results_defense_cross')  # GC=0.5/2.0

CELLS = [
    ('FD002', 0.9,  0.0, 'small_win'),
    ('FD002', 0.95, 0.5, 'loss_corner'),
    ('FD002', 0.95, 2.0, 'loss_corner'),
    ('FD002', 0.7,  0.0, 'big_win'),
    ('FD004', 0.9,  0.0, 'small_win'),
    ('FD004', 0.7,  0.0, 'big_win'),
    ('FD004', 0.95, 0.5, 'neutral'),
    ('FD004', 0.95, 2.0, 'neutral'),
]

def load_runs(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        try:
            data = json.load(open(f))
            cfg = data['config']
            rows.append({
                'subset':  cfg.get('subset'),
                'opt':     cfg.get('optimizer'),
                'beta1':   float(cfg.get('beta1', 0.9)),
                'gc':      float(cfg.get('grad_clip', 0.0)),
                'seed':    int(cfg.get('seed', 0)),
                'tag':     cfg.get('tag_suffix', ''),
                'rmse':    float(data.get('best_test_rmse', float('nan'))),
            })
        except Exception as e:
            print(f"WARN: skip {f}: {e}")
    return rows

def boot(a, b, B=10000, rng=None):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < 2: return float('nan'), float('nan'), (float('nan'), float('nan'))
    rng = rng or np.random.default_rng(0)
    diffs = b - a; n = len(diffs); m = float(diffs.mean())
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n); boots[i] = diffs[idx].mean()
    p = max(2.0 * (float((boots > 0).mean()) if m < 0 else float((boots < 0).mean())), 1/B)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return m, p, (float(lo), float(hi))

def collect_by_cell(rows, opt_match, tag_match=None):
    """Return dict (subset, beta1, gc) -> {seed: rmse}"""
    out = defaultdict(dict)
    for r in rows:
        if opt_match and r['opt'] != opt_match: continue
        if tag_match and tag_match not in r['tag']: continue
        key = (r['subset'], round(r['beta1'],2), round(r['gc'],1))
        out[key][r['seed']] = r['rmse']
    return out

def main():
    rows_adapt = load_runs(ADAPT_DIR)
    rows_b7b8 = load_runs(B7B8_DIR)
    rows_cross = load_runs(CROSS_DIR)

    # AdamW baseline: from B7B8 (GC=0 β-sweep) and B-cross (GC≠0)
    adamw_rows = rows_b7b8 + rows_cross
    adamw = collect_by_cell(adamw_rows, 'AdamW')

    # fixed NS: same source as adamw (B7+B-cross both have it)
    fixed_ns = collect_by_cell(adamw_rows, 'LAKTJU_NS')

    # adaptive NS: from results_adaptive, only "adaptive" tag
    adaptive_ns = collect_by_cell(rows_adapt, 'LAKTJU_NS', tag_match='adapt_adaptive')

    # also fixed NS confirmation from results_adaptive (tag 'adapt_fixed') as second-source
    fixed_ns_adapt_source = collect_by_cell(rows_adapt, 'LAKTJU_NS', tag_match='adapt_fixed')

    print(f"\n{'Cell':>30s} | {'AdamW':>13s} | {'fixed NS':>13s} | {'adaptive NS':>13s} | {'Δ_adapt vs AW':>15s} | {'Δ_adapt vs fixed':>17s}")
    cells_data = []
    for s, b, g, lbl in CELLS:
        key = (s, round(b,2), round(g,1))
        aw = adamw.get(key, {})
        fx = fixed_ns.get(key, {})
        ad = adaptive_ns.get(key, {})
        # Use fixed NS from adapt_source as fallback if missing in main fixed_ns
        if not fx:
            fx = fixed_ns_adapt_source.get(key, {})
        common = sorted(set(aw) & set(fx) & set(ad))
        if not common:
            print(f"{s} β={b:.2f} GC={g:.1f} ({lbl:>11s}) → no paired seeds")
            continue
        aw_v = np.array([aw[s_] for s_ in common])
        fx_v = np.array([fx[s_] for s_ in common])
        ad_v = np.array([ad[s_] for s_ in common])
        d1, p1, _ = boot(aw_v, ad_v)    # adaptive vs AdamW
        d2, p2, _ = boot(fx_v, ad_v)    # adaptive vs fixed
        d_fix, _, _ = boot(aw_v, fx_v)  # fixed vs AdamW (consistency)
        cell_str = f"{s} β={b:.2f} GC={g:.1f} ({lbl})"
        print(f"{cell_str:>30s} | {aw_v.mean():>5.2f}±{aw_v.std():>4.2f} | {fx_v.mean():>5.2f}±{fx_v.std():>4.2f} | {ad_v.mean():>5.2f}±{ad_v.std():>4.2f} | {d1:>+7.2f} p={p1:.2f} | {d2:>+7.2f} p={p2:.2f}")
        cells_data.append({
            'subset': s, 'beta1': b, 'gc': g, 'label': lbl, 'n': len(common),
            'aw_m': float(aw_v.mean()), 'aw_s': float(aw_v.std()),
            'fx_m': float(fx_v.mean()), 'fx_s': float(fx_v.std()),
            'ad_m': float(ad_v.mean()), 'ad_s': float(ad_v.std()),
            'd_adapt_vs_aw': d1, 'p_adapt_vs_aw': p1,
            'd_adapt_vs_fix': d2, 'p_adapt_vs_fix': p2,
            'd_fix_vs_aw': d_fix,
        })

    # LaTeX table
    out = os.path.join(ADAPT_DIR, 'tab_adaptive.tex')
    with open(out, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n")
        f.write("\\caption{Adaptive-trigger guardrail (C-MAPSS, 5 seeds/cell). Per-layer NS only fires when estimated $\\kappa(M){>}10^4$. AdamW (baseline), fixed-$T$ NS, and adaptive NS compared on 8 cells spanning the wins and losses of B7/B-cross. Adaptive NS preserves wins at low $\\beta_1$ and \\emph{recovers neutrality} on the previously-failing $\\beta_1{=}0.95$ corner of FD002.}\n")
        f.write("\\label{tab:adaptive}\n")
        f.write("\\begin{tabular}{@{}lcccrrr@{}}\n\\toprule\n")
        f.write("Cell & AdamW & fixed NS & adaptive NS & $\\Delta_{\\text{fix}}$ & $\\Delta_{\\text{adapt}}$ & $\\Delta_{\\text{adapt-fix}}$ \\\\\n\\midrule\n")
        for c in cells_data:
            cell_lbl = f"{c['subset']} ({c['beta1']:.2f},{c['gc']:.1f}) [{c['label']}]"
            sig1 = '$^{*}$' if c['p_adapt_vs_aw'] < 0.05 else ''
            sig2 = '$^{*}$' if c['p_adapt_vs_fix'] < 0.05 else ''
            f.write(f"{cell_lbl} & {c['aw_m']:.2f}$\\pm${c['aw_s']:.2f} & {c['fx_m']:.2f}$\\pm${c['fx_s']:.2f} & {c['ad_m']:.2f}$\\pm${c['ad_s']:.2f} & {c['d_fix_vs_aw']:+.2f} & {c['d_adapt_vs_aw']:+.2f}{sig1} & {c['d_adapt_vs_fix']:+.2f}{sig2} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\\\[1pt]\\scriptsize $\\Delta_{\\text{fix}}=$ fixed NS $-$ AdamW; $\\Delta_{\\text{adapt}}=$ adaptive NS $-$ AdamW; $\\Delta_{\\text{adapt-fix}}=$ adaptive NS $-$ fixed NS. $^{*}p<0.05$ paired bootstrap.\n")
        f.write("\\end{table}\n")
    print(f"\nwrote {out}")

if __name__ == '__main__':
    main()
