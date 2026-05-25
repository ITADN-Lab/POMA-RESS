"""Runner: UCI Appliances Energy LSTM × {AdamW, NS, MUON, SOAP} × 5 seeds × 2 LRs.

For 本机 RTX 5090.
Phase A: LR sweep (2 LRs × 1 seed)
Phase B: best LR × 5 seeds
"""
import os, sys, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

TRAIN = os.path.join(os.path.dirname(__file__), 'train_uci_energy.py')
OPTIMIZERS = ['AdamW', 'LAKTJU_NS', 'MUON', 'SOAP']
LR_GRID = {'AdamW': [1e-3, 3e-3], 'LAKTJU_NS': [1e-3, 3e-3], 'MUON': [1e-2, 3e-2], 'SOAP': [1e-3, 3e-3]}
PHASE_B_SEEDS = [42, 123, 456, 789, 1024]


def run_one(opt, lr, seed, save_dir, python_bin, epochs=100):
    tag = f"energy_{opt}_lr{lr:.0e}_seed{seed}"
    out = os.path.join(save_dir, f"energy_lstm_{opt}_seed{seed}_{tag}.json")
    if os.path.exists(out):
        return tag, 'SKIP'
    cmd = [python_bin, TRAIN, '--optimizer', opt, '--lr', str(lr), '--seed', str(seed),
           '--epochs', str(epochs), '--batch_size', '64', '--model', 'lstm', '--window', '10',
           '--save_dir', save_dir]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return tag, ('OK' if p.returncode == 0 else f'FAIL {p.stderr[-300:]}'), time.time()-t0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--phase', choices=['A','B','AB'], default='AB')
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.phase in ('A', 'AB'):
        jobs_a = [(o, l, 42) for o in OPTIMIZERS for l in LR_GRID[o]]
        print(f"[Phase A] {len(jobs_a)} jobs"); t0 = time.time(); done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, *j, args.save_dir, args.python_bin) for j in jobs_a]
            for f in as_completed(futs):
                tag, status, _ = f.result(); done += 1
                print(f"  [{done:3d}/{len(jobs_a)}] {status:15s} {tag}", flush=True)
        print(f"[Phase A] done in {(time.time()-t0)/60:.1f} min")

    # Best LR per opt
    import json, glob
    best_lr = {}
    for o in OPTIMIZERS:
        best_lr[o] = LR_GRID[o][0]
        best_val = float('inf')
        for l in LR_GRID[o]:
            for f in glob.glob(os.path.join(args.save_dir, f"energy_lstm_{o}_seed42_energy_{o}_lr{l:.0e}_seed42.json")):
                try:
                    d = json.load(open(f))
                    if d.get('best_val_rmse', float('inf')) < best_val:
                        best_val = d['best_val_rmse']; best_lr[o] = l
                except: pass
    print(f"Best LRs: {best_lr}")

    if args.phase in ('B', 'AB'):
        jobs_b = [(o, best_lr[o], s) for o in OPTIMIZERS for s in PHASE_B_SEEDS if s != 42]
        print(f"[Phase B] {len(jobs_b)} jobs"); t0 = time.time(); done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, *j, args.save_dir, args.python_bin) for j in jobs_b]
            for f in as_completed(futs):
                tag, status, _ = f.result(); done += 1
                print(f"  [{done:3d}/{len(jobs_b)}] {status:15s} {tag}", flush=True)
        print(f"[Phase B] done in {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
