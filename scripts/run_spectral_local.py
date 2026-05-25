"""Step-wise spectral tracking on C-MAPSS (FD001/FD003/FD004) × {AdamW, LAKTJU_NS, MUON} × 5 seeds.

Records pre-NS / post-NS κ(M) and effective rank for LAKTJU_NS, and periodic samples for AdamW/MUON.

Designed for local 13号 (RTX 5090, 32GB), parallel 4 workers.
"""
import os, sys, json, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_cmapss_spectral.py')

SUBSETS = ['FD001', 'FD003', 'FD004']
OPTIMIZERS = ['AdamW', 'LAKTJU_NS', 'MUON']
SEEDS = [42, 123, 456, 789, 1024]
LR = {'AdamW': 1e-3, 'LAKTJU_NS': 1e-3, 'MUON': 1e-2}
GC = 0.0  # fair baseline (GC=1 hurts AdamW)

def run_one(args, opt, subset, seed, save_dir, python_bin):
    out = os.path.join(save_dir, f"spectral_{subset}_{opt}_seed{seed}.json")
    if os.path.exists(out):
        return f"{subset}/{opt}/{seed}", 'SKIP'
    cmd = [
        python_bin, TRAIN,
        '--data_dir', args.data_dir,
        '--subset', subset,
        '--optimizer', opt,
        '--epochs', '100',
        '--batch_size', '256',
        '--lr', str(LR[opt]),
        '--seed', str(seed),
        '--grad_clip', str(GC),
        '--save_dir', save_dir,
        '--ns_interval', '100', '--ns_steps', '1', '--ns_max_dim', '256',
        '--weight_decay', '1e-4',
    ]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return f"{subset}/{opt}/{seed}", ('OK' if p.returncode == 0 else f"FAIL {p.stderr[-200:]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    jobs = [(opt, sub, seed) for opt in OPTIMIZERS for sub in SUBSETS for seed in SEEDS]
    print(f"[Spectral local] {len(jobs)} jobs, {args.workers} workers")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args, *j, args.save_dir, args.python_bin) for j in jobs]
        for f in as_completed(futs):
            tag, status = f.result()
            done += 1
            eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
            print(f"  [{done:3d}/{len(jobs)}] {status:20s} {tag} (ETA {eta:.1f}min)", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
