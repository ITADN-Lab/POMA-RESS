"""SECOM full industrial scenario: 6 optimizers × 7 seeds × 2 LRs.

For 14号 (RTX 5090, 32GB). SECOM is small; can run 4-6 workers.
"""
import os, sys, json, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_secom.py')

OPTIMIZERS = ['AdamW', 'LAKTJU_NS', 'LAKTJU_Lite', 'MUON', 'SOAP', 'Adan']
SEEDS = [42, 123, 456, 789, 1024, 2024, 2048]
LR_GRID = {
    'AdamW': [1e-3, 3e-3],
    'LAKTJU_NS': [1e-3, 3e-3],
    'LAKTJU_Lite': [1e-3, 3e-3],
    'MUON': [1e-3, 1e-2],
    'SOAP': [1e-3, 3e-3],
    'Adan': [3e-3, 1e-2],
}

def run_one(args, opt, lr, seed, save_dir, python_bin):
    tag = f"secom_{opt}_lr{lr:.0e}_seed{seed}"
    out = os.path.join(save_dir, f"{tag}.json")
    if os.path.exists(out):
        return tag, 'SKIP'
    cmd = [
        python_bin, TRAIN,
        '--optimizer', opt,
        '--epochs', '100',
        '--batch_size', '64',
        '--lr', str(lr),
        '--seed', str(seed),
        '--ns_interval', '50', '--ns_steps', '1', '--ns_max_dim', '256',
        '--save_dir', save_dir,
        '--weight_decay', '1e-4',
    ]
    # train_secom.py expects to find data in ./data — set cwd to data parent
    env = os.environ.copy()
    p = subprocess.run(cmd, cwd=args.work_dir, capture_output=True, text=True, timeout=600, env=env)
    # The script saves to f"secom_{opt}_seed{seed}.json" by default; need to either rename or modify.
    src = os.path.join(save_dir, f"secom_{opt}_seed{seed}.json")
    if os.path.exists(src) and src != out:
        os.rename(src, out)
    return tag, ('OK' if p.returncode == 0 else f'FAIL {p.stderr[-200:]}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--work_dir', required=True, help='Directory containing data/ subdir for SECOM')
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    jobs = []
    for opt in OPTIMIZERS:
        for lr in LR_GRID[opt]:
            for seed in SEEDS:
                jobs.append((opt, lr, seed))
    print(f"[SECOM full] {len(jobs)} jobs ({len(OPTIMIZERS)} opts × {len(SEEDS)} seeds × ~2 LRs), {args.workers} workers")
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
