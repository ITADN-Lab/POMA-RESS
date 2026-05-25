"""
B-adaptive: compare fixed NS (T=100, every step), adaptive NS (κ-trigger), and AdamW
on cells where fixed NS WINS and cells where fixed NS LOSES (from B-cross).

Hypothesis: adaptive trigger preserves wins on regular cells, avoids losses on corner cells.

Cells (5 seeds × 2 subsets × 3 variants × 4 cells = 120 runs):
  (1) FD002 β=0.9  GC=0    — fixed NS canonical win (-0.35 main paper)
  (2) FD002 β=0.95 GC=0.5  — fixed NS LOSS corner (+2.61 cross)
  (3) FD002 β=0.95 GC=2.0  — fixed NS LOSS corner (+1.44 cross)
  (4) FD002 β=0.7  GC=0    — fixed NS big win (-3.35 B7)
  (5) FD004 β=0.9  GC=0    — fixed NS small win
  (6) FD004 β=0.7  GC=0    — fixed NS big win (-1.59 B7)
  (7) FD004 β=0.95 GC=0.5  — neutral
  (8) FD004 β=0.95 GC=2.0  — neutral

Variants:
  AdamW                 — baseline (no NS)
  LAKTJU_NS fixed_T=100 — current method
  LAKTJU_NS adaptive κ=1e4 — proposed guardrail

Note: AdamW runs reuse the B-cross result files (already computed) to save compute.
Only LAKTJU_NS fixed + adaptive are new (160 runs total).
"""
import os, sys, json, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_cmapss_rul.py')

CELLS = [
    # (subset, beta1, gc, comment)
    ('FD002', 0.9,  0.0, 'main'),
    ('FD002', 0.95, 0.5, 'loss_corner'),
    ('FD002', 0.95, 2.0, 'loss_corner'),
    ('FD002', 0.7,  0.0, 'big_win'),
    ('FD004', 0.9,  0.0, 'small_win'),
    ('FD004', 0.7,  0.0, 'big_win'),
    ('FD004', 0.95, 0.5, 'neutral'),
    ('FD004', 0.95, 2.0, 'neutral'),
]

SEEDS = [42, 123, 456, 789, 1024]
VARIANTS = ['fixed', 'adaptive']  # AdamW result reused from B7/B-cross

BEST_LR = {
    ('FD002',): 3e-4,   # NS best LR on FD002 (paper protocol)
    ('FD004',): 1e-3,   # NS best LR on FD004
}

KAPPA_THRESHOLD = 1e4

def cfg_to_path(save_dir, subset, beta1, gc, variant, seed):
    return os.path.join(save_dir, f"adapt_{subset}_{variant}_b{beta1:.2f}_g{gc:.1f}_seed{seed}.json")

def run_one(cfg, args):
    subset, beta1, gc, _comment, variant, seed = cfg
    out = cfg_to_path(args.save_dir, subset, beta1, gc, variant, seed)
    if os.path.exists(out) and not args.force:
        return cfg, 'SKIP', 0.0, ''
    lr = BEST_LR[(subset,)]
    cmd = [
        args.python_bin, TRAIN,
        '--data_dir',    args.data_dir,
        '--subset',      subset,
        '--optimizer',   'LAKTJU_NS',
        '--epochs',      str(args.epochs),
        '--batch_size',  '256',
        '--lr',          str(lr),
        '--beta1',       str(beta1),
        '--grad_clip',   str(gc),
        '--seed',        str(seed),
        '--save_dir',    args.save_dir,
        '--tag_suffix',  f"adapt_{variant}_b{beta1:.2f}_g{gc:.1f}_seed{seed}",
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
    p.add_argument('--data_dir', default=os.path.join(os.path.dirname(THIS_DIR), 'data', 'cmapss'))
    p.add_argument('--save_dir', default=os.path.join(os.path.dirname(THIS_DIR), 'results_adaptive'))
    p.add_argument('--python_bin', default=sys.executable)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs',  type=int, default=100)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    matrix = []
    for cell in CELLS:
        for variant in VARIANTS:
            for seed in SEEDS:
                matrix.append((cell[0], cell[1], cell[2], cell[3], variant, seed))
    print(f"[adaptive] {len(matrix)} configs, workers={args.workers}, epochs={args.epochs}")

    t0 = time.time()
    ok=skip=fail=0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, c, args): c for c in matrix}
        for i, f in enumerate(as_completed(futs), 1):
            cfg, st, dt, err = f.result()
            tag = f"{cfg[0]} β={cfg[1]:.2f} GC={cfg[2]:.1f} {cfg[4]:>8s} s{cfg[5]}"
            print(f"[{i:3d}/{len(matrix)}] {tag} → {st} ({dt:.0f}s)")
            if st == 'OK': ok += 1
            elif st == 'SKIP': skip += 1
            else:
                fail += 1
                if err: print(f"     err: {err[-200:]}")
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min: ok={ok} skip={skip} fail={fail}")

if __name__ == '__main__':
    main()
