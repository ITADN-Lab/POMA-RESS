"""Runner: PatchTST × C-MAPSS FD001/FD004 × {AdamW, NS, MUON} × 5 seeds × 2 LRs.

For 12号 RTX 5090.
Phase A: 2-LR sweep per (subset × optimizer), 1 seed
Phase B: best LR × 5 seeds

Approx total: (2 subsets × 3 optimizers × 2 LRs × 1 seed) + (2 × 3 × 1 LR × 5 seeds) = 12 + 30 = 42 jobs
"""
import os, sys, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_cmapss_patchtst.py')

SUBSETS = ['FD001', 'FD004']
OPTIMIZERS = ['AdamW', 'LAKTJU_NS', 'MUON']
LR_GRID = {'AdamW': [3e-4, 1e-3], 'LAKTJU_NS': [3e-4, 1e-3], 'MUON': [1e-2, 3e-2]}
PHASE_B_SEEDS = [42, 123, 456, 789, 1024]


def run_one(data_dir, subset, opt, lr, seed, save_dir, python_bin, epochs=100):
    tag = f"pt_{subset}_{opt}_lr{lr:.0e}_seed{seed}"
    out = os.path.join(save_dir, f"patchtst_{subset}_{opt}_seed{seed}_{tag}.json")
    if os.path.exists(out):
        return tag, 'SKIP'
    cmd = [python_bin, TRAIN, '--data_dir', data_dir, '--subset', subset,
           '--optimizer', opt, '--lr', str(lr), '--seed', str(seed),
           '--epochs', str(epochs), '--batch_size', '128',
           '--grad_clip', '0', '--weight_decay', '1e-4',
           '--ns_interval', '100', '--ns_steps', '1', '--ns_max_dim', '256',
           '--save_dir', save_dir]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return tag, ('OK' if p.returncode == 0 else f'FAIL {p.stderr[-300:]}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--subsets', default='FD001,FD004')
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    subsets = args.subsets.split(',')

    # Phase A: LR sweep
    jobs_a = [(s, o, l, 42) for s in subsets for o in OPTIMIZERS for l in LR_GRID[o]]
    print(f"[Phase A] {len(jobs_a)} jobs")
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.data_dir, *j, args.save_dir, args.python_bin) for j in jobs_a]
        for f in as_completed(futs):
            tag, status = f.result(); done += 1
            eta = (time.time()-t0)/done*(len(jobs_a)-done)/60
            print(f"  [{done:3d}/{len(jobs_a)}] {status:15s} {tag} (ETA {eta:.1f}min)", flush=True)
    print(f"[Phase A] done in {(time.time()-t0)/60:.1f} min")

    # Pick best LR per (subset, opt) — lowest best_val_rmse
    import json, glob
    best_lr = {}
    for s in subsets:
        for o in OPTIMIZERS:
            best_lr[(s, o)] = None
            best_val = float('inf')
            for l in LR_GRID[o]:
                pat = os.path.join(args.save_dir, f"patchtst_{s}_{o}_seed42_pt_{s}_{o}_lr{l:.0e}_seed42.json")
                fs = glob.glob(pat)
                if fs:
                    try:
                        d = json.load(open(fs[0]))
                        if d.get('best_val_rmse', float('inf')) < best_val:
                            best_val = d['best_val_rmse']
                            best_lr[(s, o)] = l
                    except: pass
    print(f"[Phase B] best LRs: {best_lr}")

    # Phase B: multi-seed
    jobs_b = [(s, o, best_lr[(s, o)], seed) for s in subsets for o in OPTIMIZERS
              for seed in PHASE_B_SEEDS if best_lr.get((s, o)) is not None]
    print(f"[Phase B] {len(jobs_b)} jobs")
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.data_dir, *j, args.save_dir, args.python_bin) for j in jobs_b]
        for f in as_completed(futs):
            tag, status = f.result(); done += 1
            eta = (time.time()-t0)/done*(len(jobs_b)-done)/60
            print(f"  [{done:3d}/{len(jobs_b)}] {status:15s} {tag} (ETA {eta:.1f}min)", flush=True)
    print(f"[Phase B] done in {(time.time()-t0)/60:.1f} min. All results in {args.save_dir}")

if __name__ == '__main__':
    main()
