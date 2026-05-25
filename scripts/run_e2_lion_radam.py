"""
E2 — Lion + RAdam baselines on C-MAPSS FD002/FD004 (5 seeds).

Defends against "stronger baseline" Reviewer concern.
Output format same as B7+B8 so it can be aggregated together.
"""
import os, sys, json, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_cmapss_rul.py')

SUBSETS = ['FD002', 'FD004']
OPTIMIZERS = ['Lion', 'RAdam']
SEEDS = [42, 123, 456, 789, 1024]

# LR per (opt, subset) — initial reasonable defaults; can be tuned if needed
BEST_LR = {
    ('Lion', 'FD002'):  3e-4,
    ('Lion', 'FD004'):  3e-4,
    ('RAdam','FD002'):  1e-3,
    ('RAdam','FD004'):  1e-3,
}

def run_one(cfg, args):
    subset, opt, seed = cfg
    lr = BEST_LR[(opt, subset)]
    out = os.path.join(args.save_dir, f"cmapss_{subset}_{opt}_e2_seed{seed}.json")
    if os.path.exists(out) and not args.force:
        return cfg, 'SKIP', 0.0, ''
    cmd = [
        args.python_bin, TRAIN,
        '--data_dir',    args.data_dir,
        '--subset',      subset,
        '--optimizer',   opt,
        '--epochs',      str(args.epochs),
        '--batch_size',  '256',
        '--lr',          str(lr),
        '--beta1',       '0.9',
        '--grad_clip',   '0.0',
        '--seed',        str(seed),
        '--save_dir',    args.save_dir,
        '--tag_suffix',  f"e2_seed{seed}",
        '--weight_decay','1e-4',
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        dt = time.time() - t0
        if p.returncode != 0:
            return cfg, f'FAIL rc={p.returncode}', dt, (p.stderr or '')[-300:]
        return cfg, 'OK', dt, ''
    except subprocess.TimeoutExpired:
        return cfg, 'TIMEOUT', time.time() - t0, ''

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--python_bin', default=sys.executable)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs',  type=int, default=100)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    matrix = [(s, o, sd) for s, o, sd in product(SUBSETS, OPTIMIZERS, SEEDS)]
    print(f"[E2 Lion+RAdam] {len(matrix)} configs, workers={args.workers}, epochs={args.epochs}")

    t0 = time.time()
    ok=skip=fail=0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, c, args): c for c in matrix}
        for i, f in enumerate(as_completed(futs), 1):
            cfg, st, dt, err = f.result()
            print(f"[{i:3d}/{len(matrix)}] {cfg[0]}/{cfg[1]} s{cfg[2]} → {st} ({dt:.0f}s)")
            if st == 'OK': ok += 1
            elif st == 'SKIP': skip += 1
            else:
                fail += 1
                if err: print(f"     err: {err[-200:]}")
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min: ok={ok} skip={skip} fail={fail}")

if __name__ == '__main__':
    main()
