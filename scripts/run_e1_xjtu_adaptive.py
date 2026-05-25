"""
E1 — adaptive trigger on XJTU-SY 3 splits (5 seeds × 3 splits × 2 variants = 30 runs).

Test if adaptive κ-trigger (threshold 1e4) avoids the data-scarce split (OC1+OC2→OC3)
NS regression (+6.14 with fixed NS) while preserving the data-rich split (OC2+OC3→OC1) win.

LR per split chosen from B2 Phase A pilot (we hard-code based on prior runs):
  OC1,OC2→OC3 NS: lr=3e-3
  OC1,OC3→OC2 NS: lr=3e-4
  OC2,OC3→OC1 NS: lr=3e-3
"""
import os, sys, json, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_xjtu_rul.py')

SPLITS = [
    ('OC1,OC2', 'OC3', 3e-3, 'data_scarce'),    # NS loses here with fixed
    ('OC1,OC3', 'OC2', 3e-4, 'moderate'),       # borderline
    ('OC2,OC3', 'OC1', 3e-3, 'data_rich'),      # NS wins here with fixed
]
SEEDS = [42, 123, 456, 789, 1024]
VARIANTS = ['fixed', 'adaptive']

KAPPA_THRESHOLD = 1e4

def run_one(cfg, args):
    train_oc, test_oc, lr, _label, variant, seed = cfg
    suffix = f"e1_{variant}_seed{seed}"
    out_train = train_oc.replace(',', '+')
    out_test = test_oc.replace(',', '+')
    expected = os.path.join(args.save_dir, f"xjtu_{out_train}_to_{out_test}_{suffix}.json")
    if os.path.exists(expected) and not args.force:
        return cfg, 'SKIP', 0.0, ''
    cmd = [
        args.python_bin, TRAIN,
        '--cache_dir',   args.cache_dir,
        '--train_oc',    train_oc,
        '--test_oc',     test_oc,
        '--optimizer',   'LAKTJU_NS',
        '--epochs',      str(args.epochs),
        '--batch_size',  '128',
        '--lr',          str(lr),
        '--beta1',       '0.9',
        '--grad_clip',   '0.0',
        '--seed',        str(seed),
        '--save_dir',    args.save_dir,
        '--tag_suffix',  suffix,
        '--ns_interval', '100',
        '--ns_steps',    '1',
        '--ns_max_dim',  '256',
        '--weight_decay','1e-4',
    ]
    if variant == 'adaptive':
        cmd += ['--adaptive_trigger', '--kappa_threshold', str(KAPPA_THRESHOLD)]
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
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--save_dir',  required=True)
    p.add_argument('--python_bin', default=sys.executable)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs',  type=int, default=100)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    args.cache_dir = os.path.abspath(args.cache_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    matrix = []
    for split in SPLITS:
        for variant in VARIANTS:
            for seed in SEEDS:
                matrix.append((split[0], split[1], split[2], split[3], variant, seed))
    print(f"[E1 XJTU adaptive] {len(matrix)} configs, workers={args.workers}, epochs={args.epochs}")

    t0 = time.time()
    ok=skip=fail=0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, c, args): c for c in matrix}
        for i, f in enumerate(as_completed(futs), 1):
            cfg, st, dt, err = f.result()
            print(f"[{i:2d}/{len(matrix)}] {cfg[0]}→{cfg[1]} {cfg[4]:>8s} s{cfg[5]} → {st} ({dt:.0f}s)")
            if st == 'OK': ok += 1
            elif st == 'SKIP': skip += 1
            else:
                fail += 1
                if err: print(f"     err: {err[-200:]}")
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min: ok={ok} skip={skip} fail={fail}")

if __name__ == '__main__':
    main()
