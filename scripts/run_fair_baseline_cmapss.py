"""C-MAPSS Fair-Baseline Runner: 5 optimizers × 4 subsets × 2 GC × multi-LR × multi-seed.

Designed for parallel execution on a single 5090 (32GB) with 4-6 worker processes.

Phase A: LR sweep (1 seed=42) — find best LR per (subset × optimizer × GC)
Phase B: Best-LR re-run with 5 seeds — generate the seed-level statistics for the paper

Outputs JSON to --save_dir; aggregator at end prints summary.
"""
import os, sys, json, time, argparse, subprocess, signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_cmapss_rul.py')

OPTIMIZERS = ['AdamW', 'LAKTJU_NS', 'MUON', 'SOAP', 'Adan']
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
GC_CONFIGS = [0.0, 1.0]
LR_GRID = {
    'AdamW':     [3e-4, 1e-3, 3e-3],
    'LAKTJU_NS': [3e-4, 1e-3, 3e-3],
    'MUON':      [3e-3, 1e-2, 3e-2],
    'SOAP':      [3e-4, 1e-3, 3e-3],
    'Adan':      [1e-3, 3e-3, 1e-2],
}
PHASE_B_SEEDS = [42, 123, 456, 789, 1024]

def make_tag(opt, subset, gc, lr, seed):
    return f"fair_{subset}_{opt}_gc{gc}_lr{lr:.0e}_seed{seed}"

def run_one(args, opt, subset, gc, lr, seed, save_dir, python_bin, epochs):
    tag = make_tag(opt, subset, gc, lr, seed)
    out = os.path.join(save_dir, f"cmapss_{subset}_{opt}_seed{seed}_{tag}.json")
    if os.path.exists(out):
        return tag, 'SKIP', None
    cmd = [
        python_bin, TRAIN,
        '--data_dir', args.data_dir,
        '--subset', subset,
        '--optimizer', opt,
        '--epochs', str(epochs),
        '--batch_size', '256',
        '--lr', str(lr),
        '--seed', str(seed),
        '--grad_clip', str(gc),
        '--save_dir', save_dir,
        '--tag_suffix', tag,
        '--ns_interval', '100', '--ns_steps', '1', '--ns_max_dim', '256',
        '--weight_decay', '1e-4',
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = (p.returncode == 0)
        return tag, ('OK' if ok else f'FAIL rc={p.returncode}'), {
            'time': time.time() - t0,
            'stderr_tail': (p.stderr[-400:] if not ok else ''),
        }
    except subprocess.TimeoutExpired:
        return tag, 'TIMEOUT', {'time': time.time() - t0}

def collect_best_lr(save_dir):
    """Phase A → pick best (lowest val_rmse) LR per (subset, opt, gc, seed=42)."""
    import glob
    best = {}  # (subset, opt, gc) -> (lr, val_rmse)
    for f in glob.glob(os.path.join(save_dir, 'cmapss_*_seed42_*.json')):
        try:
            d = json.load(open(f))
            cfg = d.get('config', {})
            subset, opt, gc, lr = cfg.get('subset'), cfg.get('optimizer'), cfg.get('grad_clip'), cfg.get('lr')
            val_rmse = d.get('best_val_rmse', float('inf'))
            key = (subset, opt, gc)
            if key not in best or val_rmse < best[key][1]:
                best[key] = (lr, val_rmse, f)
        except Exception as e:
            print(f"WARN: skip {f}: {e}")
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--phase', choices=['A', 'B', 'AB'], default='AB')
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    ap.add_argument('--optimizers', default='AdamW,LAKTJU_NS,MUON,SOAP,Adan')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    subsets = args.subsets.split(',')
    optimizers = args.optimizers.split(',')

    # Phase A: LR sweep with seed 42
    if 'A' in args.phase:
        jobs = []
        for opt in optimizers:
            for subset in subsets:
                for gc in GC_CONFIGS:
                    for lr in LR_GRID[opt]:
                        jobs.append((opt, subset, gc, lr, 42))
        print(f"[Phase A] {len(jobs)} jobs, {args.workers} workers")
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, args, *j, args.save_dir, args.python_bin, args.epochs)
                    for j in jobs]
            for f in as_completed(futs):
                tag, status, meta = f.result()
                done += 1
                eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
                print(f"  [{done:3d}/{len(jobs)}] {status:12s} {tag} (ETA {eta:.1f}min)", flush=True)
        print(f"[Phase A] Done in {(time.time()-t0)/60:.1f} min")

    if 'B' in args.phase:
        best = collect_best_lr(args.save_dir)
        print(f"[Phase B] Best LRs found for {len(best)} (subset,opt,gc) combos")
        for k, v in sorted(best.items()):
            print(f"  {k}: lr={v[0]:.0e} val_rmse={v[1]:.3f}")
        jobs = []
        for (subset, opt, gc), (lr, _, _) in best.items():
            if subset not in subsets or opt not in optimizers:
                continue
            for seed in PHASE_B_SEEDS:
                if seed == 42:
                    continue  # already done in Phase A
                jobs.append((opt, subset, gc, lr, seed))
        print(f"[Phase B] {len(jobs)} jobs")
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, args, *j, args.save_dir, args.python_bin, args.epochs)
                    for j in jobs]
            for f in as_completed(futs):
                tag, status, meta = f.result()
                done += 1
                eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
                print(f"  [{done:3d}/{len(jobs)}] {status:12s} {tag} (ETA {eta:.1f}min)", flush=True)
        print(f"[Phase B] Done in {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
