"""
B-cross: β₁ × GC joint sweep — extends B7/B8 to defend against the
"reviewer might claim joint hyperparameter tuning closes the gap" argument.

Matrix:
  subsets: FD002, FD004
  optimizers: AdamW, LAKTJU_NS
  β₁: {0.7, 0.95}     (extremes only; B7 already covered {0.7,0.8,0.9,0.95} at GC=0)
  GC: {0.5, 2.0}      (extremes only; B8 already covered {0,0.5,1,2} at β₁=0.9)
  seeds: {42, 123, 456, 789, 1024}

Default split:
  --subset_filter FD002  → 40 runs on one machine
  --subset_filter FD004  → 40 runs on the other
"""
import os, sys, json, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_cmapss_rul.py')

SUBSETS    = ['FD002', 'FD004']
OPTIMIZERS = ['AdamW', 'LAKTJU_NS']
BETA1_GRID = [0.7, 0.95]
GC_GRID    = [0.5, 2.0]
SEEDS      = [42, 123, 456, 789, 1024]

# Calibrated from paper's primary fair-baseline runs (same as run_b7_b8_sweep.py)
BEST_LR = {
    ('AdamW',     'FD002'): 1e-3,
    ('AdamW',     'FD004'): 1e-3,
    ('LAKTJU_NS', 'FD002'): 3e-4,
    ('LAKTJU_NS', 'FD004'): 1e-3,
}

def build_matrix(subset_filter=None):
    cfgs = []
    for subset, opt, b, g, s in product(SUBSETS, OPTIMIZERS, BETA1_GRID, GC_GRID, SEEDS):
        if subset_filter and subset != subset_filter:
            continue
        cfgs.append((subset, opt, b, g, s))
    return cfgs

def output_path(save_dir, subset, opt, beta1, gc, seed):
    return os.path.join(save_dir, f"cmapss_{subset}_{opt}_cross_b{beta1:.2f}_g{gc:.1f}_seed{seed}.json")

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
        '--tag_suffix',  f"cross_b{beta1:.2f}_g{gc:.1f}_seed{seed}",
        '--ns_interval', '100', '--ns_steps', '1', '--ns_max_dim', '256',
        '--weight_decay', '1e-4',
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        dt = time.time() - t0
        if p.returncode != 0:
            return cfg, f'FAIL rc={p.returncode}', dt, (p.stderr or '')[-300:]
        return cfg, 'OK', dt, ''
    except subprocess.TimeoutExpired:
        return cfg, 'TIMEOUT', time.time() - t0, ''
    except Exception as e:
        return cfg, f'EXC {type(e).__name__}', time.time() - t0, str(e)[-200:]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--python_bin', default=sys.executable)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs',  type=int, default=100)
    p.add_argument('--subset_filter', default=None, choices=[None, 'FD002', 'FD004'])
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    matrix = build_matrix(args.subset_filter)
    print(f"[B-cross] {len(matrix)} configs (filter={args.subset_filter}, workers={args.workers}, epochs={args.epochs})")
    print(f"save_dir = {args.save_dir}")
    t0 = time.time()
    ok=skip=fail=0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, c, args): c for c in matrix}
        for i, f in enumerate(as_completed(futs), 1):
            cfg, st, dt, err = f.result()
            tag = f"{cfg[0]}/{cfg[1]} β={cfg[2]:.2f} G={cfg[3]:.1f} s{cfg[4]}"
            print(f"[{i:3d}/{len(matrix)}] {tag} → {st} ({dt:.0f}s)")
            if st == 'OK': ok += 1
            elif st == 'SKIP': skip += 1
            else:
                fail += 1
                if err: print(f"     err: {err[-200:]}")
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min: ok={ok} skip={skip} fail={fail}")

if __name__ == '__main__':
    main()
