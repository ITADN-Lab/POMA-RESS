"""
B2 — XJTU-SY Bearing RUL sweep, distributed across local GPUs.

Cross-condition evaluation: 3 splits {train OC1+OC2 / test OC3, train OC1+OC3 / test OC2, train OC2+OC3 / test OC1}
Optimizers (primary, 10 seeds): AdamW, LAKTJU_NS, MUON
Optimizers (secondary, 5 seeds): Adan, SOAP
Total runs: (3 splits × 3 opts × 10 seeds) + (3 × 2 × 5) = 90 + 30 = 120 runs.

Best LR per (optimizer × split) chosen from a 3-LR pilot sweep at seed=42 only.
The pilot LR sweep is built in too; we run Phase A (pilot, 1 seed), then Phase B (best LR, full seeds).
"""
import os, sys, json, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_xjtu_rul.py')

SPLITS = [
    ('OC1,OC2', 'OC3'),
    ('OC1,OC3', 'OC2'),
    ('OC2,OC3', 'OC1'),
]
PRIMARY_OPTS = ['AdamW', 'LAKTJU_NS', 'MUON']
SECONDARY_OPTS = ['Adan', 'SOAP']
PRIMARY_SEEDS = [42, 123, 456, 789, 1024, 2024, 2048, 3, 7, 11]
SECONDARY_SEEDS = [42, 123, 456, 789, 1024]

LR_GRID = {
    'AdamW':     [3e-4, 1e-3, 3e-3],
    'LAKTJU_NS': [3e-4, 1e-3, 3e-3],
    'MUON':      [3e-3, 1e-2, 3e-2],
    'SOAP':      [3e-4, 1e-3, 3e-3],
    'Adan':      [1e-3, 3e-3, 1e-2],
}

def build_phase_a(args):
    """Phase A: LR pilot with seed=42 only."""
    out = []
    all_opts = PRIMARY_OPTS + SECONDARY_OPTS
    for train_oc, test_oc in SPLITS:
        for opt in all_opts:
            for lr in LR_GRID[opt]:
                out.append((train_oc, test_oc, opt, lr, 42, args.epochs))
    return out

def collect_best_lr(save_dir):
    """Pick best (lowest val_rmse) LR per (split, opt) from Phase A results."""
    import glob
    best = {}
    for f in glob.glob(os.path.join(save_dir, 'xjtu_*_lrpilot*.json')):
        try:
            d = json.load(open(f))
            cfg = d.get('config', {})
            key = (cfg.get('train_oc'), cfg.get('test_oc'), cfg.get('optimizer'))
            val_rmse = d.get('best_val_rmse', float('inf'))
            lr = cfg.get('lr')
            if key not in best or val_rmse < best[key][1]:
                best[key] = (lr, val_rmse, f)
        except Exception as e:
            print(f"WARN: skip {f}: {e}")
    return best

def build_phase_b(args, best_lr):
    """Phase B: best-LR full seeds (10 for primary, 5 for secondary)."""
    out = []
    for train_oc, test_oc in SPLITS:
        for opt in PRIMARY_OPTS:
            key = (train_oc, test_oc, opt)
            if key not in best_lr:
                print(f"WARN: no Phase A best LR for {key}; using default")
                lr = LR_GRID[opt][1]
            else:
                lr = best_lr[key][0]
            for seed in PRIMARY_SEEDS:
                out.append((train_oc, test_oc, opt, lr, seed, args.epochs))
        for opt in SECONDARY_OPTS:
            key = (train_oc, test_oc, opt)
            if key not in best_lr:
                lr = LR_GRID[opt][1]
            else:
                lr = best_lr[key][0]
            for seed in SECONDARY_SEEDS:
                out.append((train_oc, test_oc, opt, lr, seed, args.epochs))
    return out

def output_path(save_dir, train_oc, test_oc, opt, lr, seed, tag):
    test = test_oc.replace(',', '+')
    train = train_oc.replace(',', '+')
    return os.path.join(save_dir, f"xjtu_{train}_to_{test}_{opt}_lr{lr:.0e}_seed{seed}_{tag}.json")

def run_one(cfg, args, tag):
    train_oc, test_oc, opt, lr, seed, epochs = cfg
    out = output_path(args.save_dir, train_oc, test_oc, opt, lr, seed, tag)
    if os.path.exists(out) and not args.force:
        return cfg, 'SKIP', 0.0, ''
    suffix = f"{tag}_{opt}_lr{lr:.0e}_seed{seed}"
    cmd = [
        args.python_bin, TRAIN,
        '--cache_dir',  args.cache_dir,
        '--train_oc',   train_oc,
        '--test_oc',    test_oc,
        '--optimizer',  opt,
        '--epochs',     str(epochs),
        '--batch_size', str(args.batch_size),
        '--lr',         str(lr),
        '--seed',       str(seed),
        '--grad_clip',  '0.0',
        '--save_dir',   args.save_dir,
        '--tag_suffix', suffix,
        '--ns_interval', '100', '--ns_steps', '1', '--ns_max_dim', '256',
        '--weight_decay', '1e-4',
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--save_dir', default=os.path.join(THIS_DIR, '..', 'results_xjtu'))
    p.add_argument('--python_bin', default=sys.executable)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--phase', choices=['A', 'B', 'AB'], default='AB',
                   help='A = LR pilot only; B = full seeds; AB = both')
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    args.cache_dir = os.path.abspath(args.cache_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    def run_phase(configs, tag):
        print(f"\n[Phase {tag}] {len(configs)} configs, workers={args.workers}")
        if args.dry_run:
            for c in configs: print('  ', c)
            return
        t_start = time.time()
        ok = skip = fail = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, c, args, tag): c for c in configs}
            for i, f in enumerate(as_completed(futs), 1):
                cfg, st, dt, err = f.result()
                tag_str = f"{cfg[0]}→{cfg[1]} {cfg[2]} lr={cfg[3]:.0e} s{cfg[4]}"
                print(f"[{i:3d}/{len(configs)}] {tag_str} → {st} ({dt:.0f}s)")
                if st == 'OK': ok += 1
                elif st == 'SKIP': skip += 1
                else:
                    fail += 1
                    if err: print(f"     err: {err[-200:]}")
        print(f"=== Phase {tag} done in {(time.time()-t_start)/60:.1f} min: ok={ok} skip={skip} fail={fail}")

    if args.phase in ('A', 'AB'):
        run_phase(build_phase_a(args), 'lrpilot')
    if args.phase in ('B', 'AB'):
        best_lr = collect_best_lr(args.save_dir)
        print(f"\n[Phase B] Best-LR table from Phase A:")
        for k, (lr, val, _) in sorted(best_lr.items()):
            print(f"  {k}: lr={lr:.0e}  val_rmse={val:.2f}")
        run_phase(build_phase_b(args, best_lr), 'main')

if __name__ == '__main__':
    main()
