"""
Plan B runner: leak-free engine-level equal-budget re-evaluation of C-MAPSS.

Phase A : 36-config beta1 x GC x LR grid, seed 42, partition 2024, for
          {AdamW, PMO} x {FD001-FD004} -> 288 runs. Equal-budget + budget curve.
Phase B : re-run the best engine-val config per (subset, optimizer) at 20 seeds
          on partition 2024 -> paired error bars for AdamW vs PMO.
Phase P : 36-config grid, seed 42, for 2 ALTERNATIVE engine partitions ->
          validation-partition sensitivity (is the conclusion split-robust?).

Hyper-parameter selection is ALWAYS by engine-level validation RMSE.
"""
import os, sys, json, glob, time, argparse, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS, 'train_cmapss_leakfree.py')

OPTIMIZERS = ['AdamW', 'PMO']
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
BETA1_GRID = [0.8, 0.9, 0.95]
GC_GRID = [0.0, 0.5, 1.0, 2.0]
LR_GRID = [3e-4, 1e-3, 3e-3]            # 3 x 4 x 3 = 36 configs
MAIN_PARTITION = 2024
ALT_PARTITIONS = [7, 99]
# 20-seed set for Phase B (seed 42 already produced in Phase A)
PHASE_B_SEEDS = [123, 456, 789, 1024, 2, 7, 13, 21, 34, 55,
                 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
MAX_RETRY = 6


def cfg_tag(b1, gc, lr):
    return f"b{b1}_gc{gc}_lr{lr:.0e}"


def out_name(subset, opt, b1, gc, lr, seed, split_seed):
    tag = cfg_tag(b1, gc, lr)
    sp = '' if split_seed == MAIN_PARTITION else f"_sp{split_seed}"
    return f"lf_{subset}_{opt}_seed{seed}_{tag}{sp}.json"


def run_one(data_dir, save_dir, python_bin, epochs,
            subset, opt, b1, gc, lr, seed, split_seed):
    tag = cfg_tag(b1, gc, lr)
    name = f"{subset}/{opt}/{tag}/s{seed}/sp{split_seed}"
    out = os.path.join(save_dir, out_name(subset, opt, b1, gc, lr, seed, split_seed))
    if os.path.exists(out):
        return name, 'SKIP'
    suffix = tag if split_seed == MAIN_PARTITION else f"{tag}_sp{split_seed}"
    cmd = [
        python_bin, TRAIN,
        '--data_dir', data_dir, '--subset', subset, '--optimizer', opt,
        '--val_split', 'engine', '--split_seed', str(split_seed),
        '--epochs', str(epochs), '--batch_size', '256',
        '--lr', str(lr), '--beta1', str(b1), '--grad_clip', str(gc),
        '--seed', str(seed), '--save_dir', save_dir, '--tag_suffix', suffix,
        '--weight_decay', '1e-4',
        '--ns_interval', '100', '--ns_steps', '1', '--ns_max_dim', '256',
    ]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for attempt in range(1, MAX_RETRY + 1):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=1200, env=env)
            if p.returncode == 0 and os.path.exists(out):
                return name, ('OK' if attempt == 1 else f'OK(retry{attempt})')
            if 'out of memory' in (p.stderr or '') and attempt < MAX_RETRY:
                time.sleep(45 * attempt)
                continue
            return name, f"FAIL rc={p.returncode} {(p.stderr or '')[-160:]}"
        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRY:
                time.sleep(30)
                continue
            return name, 'TIMEOUT'
    return name, 'FAIL exhausted-retries'


def collect_best(save_dir, seed=42, split_seed=MAIN_PARTITION):
    """Best (lowest engine-val RMSE) config per (subset, opt)."""
    best = {}
    for f in glob.glob(os.path.join(save_dir, f'lf_*_seed{seed}_*.json')):
        try:
            d = json.load(open(f))
            if d.get('split_info', {}).get('split_seed') != split_seed:
                continue
            c = d['config']
            key = (c['subset'], c['optimizer'])
            v = d.get('best_val_rmse', float('inf'))
            if key not in best or v < best[key][0]:
                best[key] = (v, c['beta1'], c['grad_clip'], c['lr'])
        except Exception as e:
            print(f"WARN skip {f}: {e}")
    return best


def launch(jobs, save_dir, args, label):
    print(f"[{label}] {len(jobs)} jobs, {args.workers} workers", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.data_dir, save_dir, args.python_bin,
                          args.epochs, *j) for j in jobs]
        for f in as_completed(futs):
            name, status = f.result()
            done += 1
            eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
            print(f"  [{done:4d}/{len(jobs)}] {status:14s} {name} "
                  f"(ETA {eta:.1f}min)", flush=True)
    print(f"[{label}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--phase', default='ABP')   # any subset of A,B,P
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    grid = list(product(BETA1_GRID, GC_GRID, LR_GRID))

    if 'A' in args.phase:
        jobs = [(s, o, b1, gc, lr, 42, MAIN_PARTITION)
                for s in SUBSETS for o in OPTIMIZERS for b1, gc, lr in grid]
        launch(jobs, args.save_dir, args, 'Phase A')

    if 'B' in args.phase:
        best = collect_best(args.save_dir, seed=42, split_seed=MAIN_PARTITION)
        print("[Phase B] best engine-val configs (partition 2024):")
        for k, v in sorted(best.items()):
            print(f"  {k}: val={v[0]:.3f} b1={v[1]} gc={v[2]} lr={v[3]:.0e}")
        jobs = []
        for (subset, opt), (_, b1, gc, lr) in best.items():
            for seed in PHASE_B_SEEDS:
                jobs.append((subset, opt, b1, gc, lr, seed, MAIN_PARTITION))
        launch(jobs, args.save_dir, args, 'Phase B (20-seed)')

    if 'P' in args.phase:
        jobs = [(s, o, b1, gc, lr, 42, sp)
                for sp in ALT_PARTITIONS
                for s in SUBSETS for o in OPTIMIZERS for b1, gc, lr in grid]
        launch(jobs, args.save_dir, args, 'Phase P (partition sensitivity)')

    print("ALL DONE", flush=True)


if __name__ == '__main__':
    main()
