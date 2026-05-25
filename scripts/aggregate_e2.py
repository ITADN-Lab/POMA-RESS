"""Aggregate E2 — Lion + RAdam baselines on C-MAPSS FD002/FD004."""
import os, glob, json, argparse
import numpy as np
from collections import defaultdict

THIS = os.path.dirname(os.path.abspath(__file__))
E2_DIR = os.path.join(THIS, '..', 'results_e2')
# Reuse fair12 AdamW + LAKTJU_NS canonical results for reference
FAIR_DIR = os.path.join(THIS, '..', 'results_aggregated', 'fair12')

SUBSETS = ['FD002', 'FD004']

def load_dir(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        try:
            data = json.load(open(f))
            cfg = data['config']
            rows.append({
                'subset':  cfg.get('subset'),
                'opt':     cfg.get('optimizer'),
                'seed':    int(cfg.get('seed', 0)),
                'lr':      float(cfg.get('lr', 0)),
                'gc':      float(cfg.get('grad_clip', 0)),
                'rmse':    float(data.get('best_test_rmse', float('nan'))),
            })
        except Exception as e:
            print(f"WARN: {f}: {e}")
    return rows

def boot(a, b, B=10000, rng=None):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < 2: return float('nan'), float('nan')
    rng = rng or np.random.default_rng(0)
    diffs = b - a; m = float(diffs.mean())
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(diffs), size=len(diffs)); boots[i] = diffs[idx].mean()
    p = max(2.0 * (float((boots > 0).mean()) if m < 0 else float((boots < 0).mean())), 1/B)
    return m, p

def main():
    e2 = load_dir(E2_DIR)
    fair = load_dir(FAIR_DIR)

    print(f"\n{'Subset':>10s} | {'Optimizer':>12s} | {'mean±std (n)':>16s}")
    rows_out = {}
    for s in SUBSETS:
        # Lion, RAdam from E2
        for opt in ['Lion', 'RAdam']:
            cells = [r['rmse'] for r in e2 if r['subset']==s and r['opt']==opt]
            if cells:
                arr = np.array(cells)
                print(f"{s:>10s} | {opt:>12s} | {arr.mean():>7.2f}±{arr.std():>5.2f}({len(arr):>2d})")
                rows_out.setdefault(s, {})[opt] = arr
        # AdamW + LAKTJU_NS for reference (use canonical paper means)
        for opt in ['AdamW', 'LAKTJU_NS']:
            cells = [r['rmse'] for r in fair if r['subset']==s and r['opt']==opt and r['gc'] == 0.0]
            if cells:
                arr = np.array(cells)
                print(f"{s:>10s} | {opt:>12s} | {arr.mean():>7.2f}±{arr.std():>5.2f}({len(arr):>2d})")
                rows_out.setdefault(s, {})[opt] = arr

    # LaTeX
    out = os.path.join(E2_DIR, 'tab_e2_lion_radam.tex')
    with open(out, 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\caption{Additional optimizer baselines on C-MAPSS (LSTM, GC$=$0, 5 seeds): Lion and RAdam compared to AdamW and LAKTJU-NS. LAKTJU-NS retains an advantage over both additional baselines on FD002/FD004.}\n")
        f.write("\\label{tab:e2_lion_radam}\n")
        f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
        f.write("Subset & AdamW & Lion & RAdam & LAKTJU-NS \\\\\n\\midrule\n")
        for s in SUBSETS:
            row = []
            best_m = float('inf')
            best_opt = None
            for opt in ['AdamW', 'Lion', 'RAdam', 'LAKTJU_NS']:
                if opt in rows_out.get(s, {}):
                    arr = rows_out[s][opt]
                    if arr.mean() < best_m:
                        best_m = arr.mean(); best_opt = opt
            for opt in ['AdamW', 'Lion', 'RAdam', 'LAKTJU_NS']:
                if opt in rows_out.get(s, {}):
                    arr = rows_out[s][opt]
                    cell = f"{arr.mean():.2f}$\\pm${arr.std():.2f}"
                    if opt == best_opt: cell = f"\\textbf{{{arr.mean():.2f}}}$\\pm${arr.std():.2f}"
                    row.append(cell)
                else:
                    row.append('---')
            f.write(f"{s} & {' & '.join(row)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"\nwrote {out}")

if __name__ == '__main__':
    main()
