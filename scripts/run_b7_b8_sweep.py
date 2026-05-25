"""
B7 (β₁ sweep) + B8 (GC sweep) — defense experiments for TII rebuttal.

Sweep over the union grid, deduping identical configs:
  • subsets: FD002, FD004
  • optimizers: AdamW, LAKTJU_NS
  • β₁: {0.7, 0.8, 0.9, 0.95}
  • GC: {0.0, 0.5, 1.0, 2.0}
  • seeds: {42, 123, 456, 789, 1024}

B7 matrix = full β₁ × seeds at GC=0.0 (fair baseline) → 2×2×4×5 = 80 runs
B8 matrix = full GC × seeds at β₁=0.9 (default)     → 2×2×4×5 = 80 runs
Overlap (β₁=0.9 & GC=0.0): 2×2×5 = 20 runs
Net unique: 80 + 80 − 20 = 140 runs

LR per (subset × optimizer) taken from prior best-LR table:
  AdamW   FD002 = 3e-4, FD004 = 3e-4
  LAKTJU_NS FD002 = 1e-3, FD004 = 1e-3

Outputs: experiments/results_defense/cmapss_<subset>_<opt>_seed<seed>_b<beta1>_g<gc>.json
"""
import os, sys, json, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_cmapss_rul.py')

SUBSETS    = ['FD002', 'FD004']
OPTIMIZERS = ['AdamW', 'LAKTJU_NS']
BETA1_GRID = [0.7, 0.8, 0.9, 0.95]
GC_GRID    = [0.0, 0.5, 1.0, 2.0]
SEEDS      = [42, 123, 456, 789, 1024]

BEST_LR = {
    # Calibrated from paper's primary fair-baseline runs (GC=0, 5 seeds):
    #   AdamW FD002 lr=1e-3 (mean 18.18), FD004 lr=1e-3 (mean 22.55)
    #   LAKTJU_NS FD002 lr=3e-4 (mean 17.16), FD004 lr=1e-3 (mean 21.91)
    ('AdamW',     'FD002'): 1e-3,
    ('AdamW',     'FD004'): 1e-3,
    ('LAKTJU_NS', 'FD002'): 3e-4,
    ('LAKTJU_NS', 'FD004'): 1e-3,
}

DEFAULT_BETA1 = 0.9
DEFAULT_GC    = 0.0

def build_matrix():
    """Returns list of (subset, opt, beta1, gc, seed) configs, deduplicated."""
    seen = set()
    configs = []
    # B7: full β₁ sweep at GC=0
    for subset, opt, beta1, seed in product(SUBSETS, OPTIMIZERS, BETA1_GRID, SEEDS):
        key = (subset, opt, beta1, DEFAULT_GC, seed)
        if key not in seen:
            seen.add(key)
            configs.append(key)
    # B8: full GC sweep at β₁=0.9
    for subset, opt, gc, seed in product(SUBSETS, OPTIMIZERS, GC_GRID, SEEDS):
        key = (subset, opt, DEFAULT_BETA1, gc, seed)
        if key not in seen:
            seen.add(key)
            configs.append(key)
    return configs

def output_path(save_dir, subset, opt, beta1, gc, seed):
    tag = f"b{beta1:.2f}_g{gc:.1f}_seed{seed}"
    return os.path.join(save_dir, f"cmapss_{subset}_{opt}_{tag}.json")

def run_one(cfg, args):
    subset, opt, beta1, gc, seed = cfg
    out = output_path(args.save_dir, subset, opt, beta1, gc, seed)
    if os.path.exists(out) and not args.force:
        return cfg, 'SKIP', 0.0, ''
    lr = BEST_LR[(opt, subset)]
    cmd = [
        args.python_bin, TRAIN,
        '--data_dir',    args.data_dir,
        '--subset',      subset,
        '--optimizer',   opt,
        '--epochs',      str(args.epochs),
        '--batch_size',  '256',
        '--lr',          str(lr),
        '--beta1',       str(beta1),
        '--grad_clip',   str(gc),
        '--seed',        str(seed),
        '--save_dir',    args.save_dir,
        '--tag_suffix',  f"b{beta1:.2f}_g{gc:.1f}_seed{seed}",
        '--ns_interval', '100', '--ns_steps', '1', '--ns_max_dim', '256',
        '--weight_decay', '1e-4',
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        dt = time.time() - t0
        if p.returncode != 0:
            return cfg, f'FAIL rc={p.returncode}', dt, (p.stderr or '')[-400:]
        return cfg, 'OK', dt, ''
    except subprocess.TimeoutExpired:
        return cfg, 'TIMEOUT', time.time() - t0, ''
    except Exception as e:
        return cfg, f'EXC {type(e).__name__}', time.time() - t0, str(e)[-200:]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=os.path.join(os.path.dirname(THIS_DIR), '..', 'data', 'cmapss'))
    p.add_argument('--save_dir', default=os.path.join(os.path.dirname(THIS_DIR), 'results_defense'))
    p.add_argument('--python_bin', default=sys.executable)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs',  type=int, default=100)
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='If >0, only run first N configs (for sanity)')
    args = p.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    matrix = build_matrix()
    if args.limit > 0:
        matrix = matrix[:args.limit]
    print(f"[B7+B8 sweep] {len(matrix)} configs (workers={args.workers}, epochs={args.epochs}, save_dir={args.save_dir})")
    if args.dry_run:
        for c in matrix:
            print('  ', c)
        return

    t_start = time.time()
    n_ok, n_skip, n_fail = 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, cfg, args): cfg for cfg in matrix}
        for i, fut in enumerate(as_completed(futs), 1):
            cfg, status, dt, err = fut.result()
            tag = f"{cfg[0]}/{cfg[1]} β={cfg[2]:.2f} GC={cfg[3]:.1f} s{cfg[4]}"
            print(f"[{i:3d}/{len(matrix)}] {tag} → {status} ({dt:.0f}s)")
            if status == 'OK':   n_ok += 1
            elif status == 'SKIP': n_skip += 1
            else:
                n_fail += 1
                if err: print(f"      stderr: {err[-200:]}")
    total = time.time() - t_start
    print(f"\n=== DONE in {total/60:.1f} min: ok={n_ok} skip={n_skip} fail={n_fail} ===")

if __name__ == '__main__':
    main()
