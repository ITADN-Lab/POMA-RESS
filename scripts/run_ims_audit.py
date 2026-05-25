"""
NASA IMS Bearing audit runner — RESS-native external replication
(Plan B v3).  Mirror of run_battery_audit.py: 4 candidates × 2
partitions × 3 seeds + 12-config Phase A grid per (opt, partition).
"""
import os, sys, time, subprocess, json, glob, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS, 'train_ims_leakfree.py')

OPTS = ['AdamW', 'PMO', 'Adan', 'RAdam', 'Lion']
BETA1_PER_OPT = {
    'AdamW': [0.85, 0.9, 0.95], 'PMO': [0.85, 0.9, 0.95],
    'Adan': [0.85, 0.9, 0.95], 'RAdam': [0.8, 0.9, 0.95],
    'Lion': [0.9, 0.95, 0.99],
}
GC_GRID = [0.0, 1.0]
LR_PER_OPT = {
    'AdamW': [1e-3, 3e-3], 'PMO': [1e-3, 3e-3],
    'Adan': [1e-3, 3e-3], 'RAdam': [1e-3, 3e-3],
    'Lion': [3e-4, 1e-3],
}
PARTITIONS = [2024, 7]
SEEDS_B = [42, 123, 456]
MAX_RETRY = 3


def cfg_tag(b1, gc, lr): return f"b{b1}_gc{gc}_lr{lr:.0e}"
def out_name(opt, b1, gc, lr, seed, sp):
    tag = cfg_tag(b1, gc, lr); sp_s = '' if sp == 2024 else f"_sp{sp}"
    return f"ims_{opt}_seed{seed}_{tag}{sp_s}.json"


def run_one(data_dir, save_dir, py, opt, b1, gc, lr, seed, sp):
    tag = cfg_tag(b1, gc, lr); nm = f"{opt}/{tag}/s{seed}/sp{sp}"
    out = os.path.join(save_dir, out_name(opt, b1, gc, lr, seed, sp))
    if os.path.exists(out): return nm, 'SKIP'
    suffix = tag if sp == 2024 else f"{tag}_sp{sp}"
    cmd = [py, TRAIN, '--data_dir', data_dir, '--optimizer', opt,
           '--val_split', 'engine', '--split_seed', str(sp),
           '--epochs', '50', '--batch_size', '256',
           '--lr', str(lr), '--beta1', str(b1), '--grad_clip', str(gc),
           '--seed', str(seed), '--save_dir', save_dir, '--tag_suffix', suffix,
           '--weight_decay', '1e-4']
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for attempt in range(1, MAX_RETRY + 1):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=1200, env=env)
            if p.returncode == 0 and os.path.exists(out):
                return nm, 'OK' if attempt == 1 else f'OK(r{attempt})'
            return nm, f"FAIL rc={p.returncode} {(p.stderr or '')[-160:]}"
        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRY: time.sleep(20); continue
            return nm, 'TIMEOUT'
    return nm, 'FAIL exhausted'


def collect_best(save_dir, opt, sp, seed=42):
    best = None
    for f in glob.glob(os.path.join(save_dir, f'ims_{opt}_seed{seed}_*.json')):
        try:
            d = json.load(open(f))
            if d['split_info']['split_seed'] != sp: continue
            c = d['config']
            if c['optimizer'] != opt: continue
            v = d.get('best_val_rmse', float('inf'))
            if best is None or v < best[0]:
                best = (v, c['beta1'], c['grad_clip'], c['lr'])
        except Exception: pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--phase', default='AB')
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    def grid_for(opt):
        return list(product(BETA1_PER_OPT[opt], GC_GRID, LR_PER_OPT[opt]))

    if 'A' in args.phase:
        jobs = []
        for sp in PARTITIONS:
            for o in OPTS:
                for b1, gc, lr in grid_for(o):
                    jobs.append((o, b1, gc, lr, 42, sp))
        print(f"[Phase A] {len(jobs)} jobs, {args.workers} workers", flush=True)
        t0, done = time.time(), 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, args.data_dir, args.save_dir,
                              args.python_bin, *j) for j in jobs]
            for f in as_completed(futs):
                nm, st = f.result(); done += 1
                eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
                print(f"  [{done:4d}/{len(jobs)}] {st:14s} {nm} (ETA {eta:.1f}min)", flush=True)
        print(f"[Phase A] done {(time.time()-t0)/60:.1f}min", flush=True)

    if 'B' in args.phase:
        jobs = []
        for sp in PARTITIONS:
            for o in OPTS:
                best = collect_best(args.save_dir, o, sp, 42)
                if not best: continue
                _, b1, gc, lr = best
                print(f"[Phase B] best {o}/sp{sp}: b1={b1} gc={gc} lr={lr:.0e}")
                for seed in SEEDS_B:
                    jobs.append((o, b1, gc, lr, seed, sp))
        print(f"[Phase B] {len(jobs)} jobs", flush=True)
        t0, done = time.time(), 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, args.data_dir, args.save_dir,
                              args.python_bin, *j) for j in jobs]
            for f in as_completed(futs):
                nm, st = f.result(); done += 1
                eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
                print(f"  [{done:4d}/{len(jobs)}] {st:14s} {nm} (ETA {eta:.1f}min)", flush=True)
        print(f"[Phase B] done {(time.time()-t0)/60:.1f}min", flush=True)
    print("ALL DONE", flush=True)


if __name__ == '__main__':
    main()
