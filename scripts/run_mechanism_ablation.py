"""Mechanism causal ablation on C-MAPSS FD002/FD004.

5 variants × 2 subsets × 5 seeds = 50 jobs.
For 本机 RTX 5090, 5 workers.
"""
import os, sys, json, time, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

VARIANTS = ['AdamW', 'LAKTJU_NS', 'GradNS', 'NormOnly', 'RandRot']
SUBSETS = ['FD002', 'FD004']
SEEDS = [42, 123, 456, 789, 1024]
# Best LR from fair baseline: all use 1e-3 for AdamW/NS on FD002/FD004 (GC=0)
LR = {'AdamW': 1e-3, 'LAKTJU_NS': 3e-4, 'GradNS': 1e-3, 'NormOnly': 1e-3, 'RandRot': 1e-3}

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, 'train_cmapss_ablation_variants.py')

def run_one(args, variant, subset, seed, save_dir, python_bin):
    out = os.path.join(save_dir, f"ablate_{variant}_{subset}_seed{seed}.json")
    if os.path.exists(out):
        return f"{variant}/{subset}/seed{seed}", 'SKIP', 0
    cmd = [python_bin, TRAIN, '--data_dir', args.data_dir, '--subset', subset,
           '--variant', variant, '--lr', str(LR[variant]), '--seed', str(seed),
           '--epochs', '100', '--batch_size', '256', '--grad_clip', '0',
           '--weight_decay', '1e-4', '--interval', '100', '--steps', '1',
           '--max_dim', '256', '--save_dir', save_dir]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return f"{variant}/{subset}/{seed}", ('OK' if p.returncode == 0 else 'FAIL'), time.time()-t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    jobs = [(v, s, seed) for v in VARIANTS for s in SUBSETS for seed in SEEDS]
    print(f"[Mechanism ablation] {len(jobs)} jobs, {args.workers} workers")
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args, *j, args.save_dir, args.python_bin) for j in jobs]
        for f in as_completed(futs):
            tag, status, dt = f.result(); done += 1
            eta = (time.time()-t0)/done*(len(jobs)-done)/60
            print(f"[{done:3d}/{len(jobs)}] {status:8s} {tag} ({dt:.0f}s ETA {eta:.1f}min)", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
