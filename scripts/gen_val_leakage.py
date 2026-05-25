"""
Validation-leakage diagnostic (Codex gpt-5.5 Plan A #4).
Compares window-val vs engine-val on the SAME 36-config grid at seed 42, for
AdamW on each C-MAPSS subset.

Outputs:
  - Per-subset: best-by-window-val test RMSE vs best-by-engine-val test RMSE
    (the SELECTION REGRET of using the leaky split for hyperparameter selection).
  - Per-subset: median val RMSE under each protocol (showing window-val
    routinely reports impossibly-low validation, the leakage symptom).
  - VAL_LEAKAGE_DIAGNOSTIC.md + summary JSON.
"""
import os, json, glob
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
RES_LF = os.path.join(THIS, '..', 'results_leakfree')
RES_WV = os.path.join(THIS, '..', 'results_window_val')
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']


def load_grid(res_dir, subset, opt='AdamW', seed=42, prefix='lf'):
    out = []
    for f in glob.glob(os.path.join(res_dir, f'{prefix}_{subset}_{opt}_seed{seed}_*.json')):
        try:
            d = json.load(open(f))
            c = d['config']
            if c['optimizer'] != opt: continue
            if prefix == 'lf':
                if d.get('split_info', {}).get('split_seed') != 2024: continue
                if d.get('split_info', {}).get('mode') != 'engine': continue
            else:
                if d.get('split_info', {}).get('mode') != 'window': continue
            out.append({
                'beta1': c['beta1'], 'gc': c['grad_clip'], 'lr': c['lr'],
                'val': d['best_val_rmse'],
                'test': d['best_test_rmse'],
            })
        except Exception:
            pass
    return out


def main():
    md = ["# Validation-leakage diagnostic\n",
          "For each C-MAPSS subset, the same AdamW 36-config grid is run under "
          "two validation protocols: \\textbf{engine-val} (leak-free leave-engines-out) "
          "and \\textbf{window-val} (the legacy 85/15 random-window split). "
          "We compare what each protocol selects as the best configuration and "
          "what test RMSE results.\n"]
    md.append("| Subset | Engine-val best test | Window-val best test | Selection regret | "
              "Median val (engine) | Median val (window) | Leakage ratio |")
    md.append("|---|---|---|---|---|---|---|")
    summary = {}
    for s in SUBSETS:
        e = load_grid(RES_LF, s, 'AdamW', 42, prefix='lf')
        w = load_grid(RES_WV, s, 'AdamW', 42, prefix='wv')
        if not e or not w:
            md.append(f"| {s} | (engine n={len(e)}) | (window n={len(w)}) | --- | --- | --- | --- |")
            continue
        e_best = min(e, key=lambda r: r['val'])
        w_best = min(w, key=lambda r: r['val'])
        regret = w_best['test'] - e_best['test']
        e_med = float(np.median([r['val'] for r in e]))
        w_med = float(np.median([r['val'] for r in w]))
        ratio = e_med / w_med if w_med > 0 else float('nan')
        summary[s] = {
            'engine_best_test': e_best['test'],
            'window_best_test': w_best['test'],
            'selection_regret': regret,
            'median_val_engine': e_med,
            'median_val_window': w_med,
            'leakage_ratio': ratio,
            'engine_best_cfg': {'b1': e_best['beta1'], 'gc': e_best['gc'], 'lr': e_best['lr']},
            'window_best_cfg': {'b1': w_best['beta1'], 'gc': w_best['gc'], 'lr': w_best['lr']},
        }
        md.append(f"| {s} | {e_best['test']:.2f} | {w_best['test']:.2f} | "
                  f"{regret:+.2f} | {e_med:.2f} | {w_med:.2f} | "
                  f"{ratio:.1f}× |")

    if summary:
        regrets = [v['selection_regret'] for v in summary.values()]
        ratios = [v['leakage_ratio'] for v in summary.values()]
        md.append("")
        md.append(f"**Headline:** window-val selection costs on average "
                  f"{np.mean(regrets):+.2f} test RMSE vs engine-val selection "
                  f"({sum(r>0 for r in regrets)}/{len(regrets)} subsets are worse "
                  f"under window-val). Window-val reports validation RMSEs that are "
                  f"{np.mean(ratios):.1f}\\times larger than engine-val's---that is, "
                  f"the leakage makes the validation metric appear ${{\\sim}}{np.mean(ratios):.0f}{{\\times}}$ "
                  f"better than the engine-level reality (the symptom of validation leakage).")

    out_md = os.path.join(RES_LF, '../VAL_LEAKAGE_DIAGNOSTIC.md')
    with open(out_md, 'w') as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RES_LF, '../val_leakage_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md))
    print(f"\nwrote {out_md}")


if __name__ == '__main__':
    main()
