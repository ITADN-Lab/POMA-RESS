"""C-MAPSS extra seeds: add 5 more seeds to main fair-baseline configs.

For (FD001-004, AdamW/LAKTJU_NS/MUON, GC=0, best-LR) add seeds 2024/2048/4096/8192/16384.
Output: same format as Phase B in run_fair_baseline_cmapss.py.

Usage: python run_cmapss_extra_seeds.py --data_dir ... --save_dir ... --subset FD001 --python_bin ... --workers 5

Run separately per machine with --subset to avoid conflicts.
"""
import os, sys, json, time, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

TRAIN = os.path.join(os.path.dirname(__file__), 'train_cmapss_rul.py')
# Best LR per (subset, optimizer) from Phase A (GC=0)
BEST_LR = {
    ('FD001', 'AdamW'): 3e-4, ('FD001', 'LAKTJU_NS'): 3e-4, ('FD001', 'MUON'): 3e-2,
    ('FD002', 'AdamW'): 1e-3, ('FD002', 'LAKTJU_NS'): 3e-4, ('FD002', 'MUON'): 3e-3,
    ('FD003', 'AdamW'): 1e-3, ('FD003', 'LAKTJU_NS'): 3e-3, ('FD003', 'MUON'): 1e-2,
    ('FD004', 'AdamW'): 1e-3, ('FD004', 'LAKTJU_NS'): 1e-3, ('FD004', 'MUON'): 1e-2,
}
EXTRA_SEEDS = [20480, 40960, 61440, 81920, 102400, 122880, 143360, 163840, 184320, 204800]

def run_one(args, subset, opt, seed, save_dir, python_bin):
    lr = BEST_LR[(subset, opt)]
    tag = f"extra_{subset}_{opt}_lr{lr:.0e}_seed{seed}"
    out = os.path.join(save_dir, f"cmapss_{subset}_{opt}_seed{seed}_{tag}.json")
    if os.path.exists(out):
        return tag, 'SKIP'
    cmd = [python_bin, TRAIN, '--data_dir', args.data_dir, '--subset', subset,
           '--optimizer', opt, '--lr', str(lr), '--seed', str(seed),
           '--epochs', '100', '--batch_size', '256', '--grad_clip', '0',
           '--weight_decay', '1e-4', '--ns_interval', '100', '--ns_steps', '1',
           '--ns_max_dim', '256', '--save_dir', save_dir, '--tag_suffix', tag]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return tag, ('OK' if p.returncode == 0 else f'FAIL'), time.time()-t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    subsets = args.subsets.split(',')
    optimizers = ['AdamW', 'LAKTJU_NS', 'MUON']
    jobs = [(sub, opt, seed) for sub in subsets for opt in optimizers for seed in EXTRA_SEEDS]
    print(f"[Extra seeds] {len(jobs)} jobs x {args.workers} workers, subsets={subsets}")
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args, *j, args.save_dir, args.python_bin) for j in jobs]
        for f in as_completed(futs):
            tag, status, dt = f.result(); done += 1
            eta = (time.time()-t0)/done*(len(jobs)-done)/60
            print(f"[{done:3d}/{len(jobs)}] {status:8s} {tag} ({dt:.0f}s ETA {eta:.1f}min)", flush=True)
    print(f"Extra seeds done in {(time.time()-t0)/60:.1f} min. Results in {args.save_dir}")

if __name__ == '__main__':
    main()
