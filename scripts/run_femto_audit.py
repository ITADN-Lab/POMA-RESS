"""
Plan B1 — real-industrial replication of the leak-free equal-budget audit
on FEMTO PHM 2012 bearing dataset (3 operating conditions).

Mirrors run_multi_audit.py but with:
  * subsets -> conditions (cond1/cond2/cond3)
  * train script -> train_femto_leakfree.py
  * 5-optimizer panel (AdamW + PMO + Adan + RAdam + Lion), all on the same
    equal-budget 36-config grid as the C-MAPSS audit (one grid per optimizer,
    optimizer-specific β1/LR ranges following their original recommended
    settings — same 36 cells per optimizer).

Phase A: Phase-A 36 configs × 5 optimizers × 3 conditions × seed42 = 540 runs
Phase B: best-config × 20 seeds × 5 optimizers × 3 conditions = 300 runs
Phase P: partition sensitivity (2 alt seeds × 36 cfg × 5 opt × 3 cond) = 1080
Total: ~1920 runs.  At ~3 min/run on RTX PRO 6000 with 10 workers ≈ 10-12h.
"""
import os, sys, json, glob, time, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS, 'train_femto_leakfree.py')

OPTIMIZERS = ['AdamW', 'PMO', 'Adan', 'RAdam', 'Lion']
CONDITIONS = ['cond1', 'cond2', 'cond3']
BETA1_PER_OPT = {
    'AdamW': [0.85, 0.9, 0.95],
    'PMO':   [0.85, 0.9, 0.95],
    'Adan':  [0.85, 0.9, 0.95],
    'RAdam': [0.8, 0.9, 0.95],
    'Lion':  [0.9, 0.95, 0.99],
}
GC_GRID = [0.0, 0.5, 1.0, 2.0]
LR_PER_OPT = {
    'AdamW': [3e-4, 1e-3, 3e-3],
    'PMO':   [3e-4, 1e-3, 3e-3],
    'Adan':  [3e-4, 1e-3, 3e-3],
    'RAdam': [3e-4, 1e-3, 3e-3],
    'Lion':  [1e-4, 3e-4, 1e-3],
}
MAIN_PARTITION = 2024
ALT_PARTITIONS = [7, 99]
PHASE_B_SEEDS = [123, 456, 789, 1024, 2, 7, 13, 21, 34, 55,
                 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
MAX_RETRY = 5


def cfg_tag(b1, gc, lr):
    return f"b{b1}_gc{gc}_lr{lr:.0e}"


def out_name(condition, opt, b1, gc, lr, seed, split_seed):
    tag = cfg_tag(b1, gc, lr)
    sp = '' if split_seed == MAIN_PARTITION else f"_sp{split_seed}"
    return f"femto_{condition}_{opt}_seed{seed}_{tag}{sp}.json"


def run_one(data_dir, save_dir, python_bin, epochs,
            condition, opt, b1, gc, lr, seed, split_seed):
    tag = cfg_tag(b1, gc, lr)
    name = f"{condition}/{opt}/{tag}/s{seed}/sp{split_seed}"
    out = os.path.join(save_dir, out_name(condition, opt, b1, gc, lr, seed, split_seed))
    if os.path.exists(out):
        return name, 'SKIP'
    suffix = tag if split_seed == MAIN_PARTITION else f"{tag}_sp{split_seed}"
    cmd = [
        python_bin, TRAIN,
        '--data_dir', data_dir, '--condition', condition, '--optimizer', opt,
        '--val_split', 'engine', '--split_seed', str(split_seed),
        '--epochs', str(epochs), '--batch_size', '256',
        '--lr', str(lr), '--beta1', str(b1), '--grad_clip', str(gc),
        '--seed', str(seed), '--save_dir', save_dir, '--tag_suffix', suffix,
        '--weight_decay', '1e-4',
    ]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for attempt in range(1, MAX_RETRY + 1):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=1200, env=env)
            if p.returncode == 0 and os.path.exists(out):
                return name, ('OK' if attempt == 1 else f'OK(retry{attempt})')
            if 'out of memory' in (p.stderr or '') and attempt < MAX_RETRY:
                time.sleep(45 * attempt); continue
            return name, f"FAIL rc={p.returncode} {(p.stderr or '')[-160:]}"
        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRY:
                time.sleep(30); continue
            return name, 'TIMEOUT'
    return name, 'FAIL exhausted-retries'


def collect_best(save_dir, opt, seed=42, split_seed=MAIN_PARTITION):
    best = {}
    for f in glob.glob(os.path.join(save_dir, f'femto_*_{opt}_seed{seed}_*.json')):
        try:
            d = json.load(open(f))
            if d.get('split_info', {}).get('split_seed') != split_seed: continue
            c = d['config']
            if c['optimizer'] != opt: continue
            key = c['condition']
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
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--phase', default='ABP')
    ap.add_argument('--only_opts', default='',
                    help='comma-separated subset of OPTIMIZERS to run')
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    opts = OPTIMIZERS if not args.only_opts else args.only_opts.split(',')

    def grid_for(opt):
        return [(b1, gc, lr) for b1, gc, lr in
                product(BETA1_PER_OPT[opt], GC_GRID, LR_PER_OPT[opt])]

    if 'A' in args.phase:
        jobs = [(c, o, b1, gc, lr, 42, MAIN_PARTITION)
                for c in CONDITIONS for o in opts
                for b1, gc, lr in grid_for(o)]
        launch(jobs, args.save_dir, args, f'Phase A ({len(opts)} opt grid)')

    if 'B' in args.phase:
        jobs = []
        for o in opts:
            best = collect_best(args.save_dir, o, 42, MAIN_PARTITION)
            print(f"[Phase B] best configs for {o}:")
            for k, v in sorted(best.items()):
                print(f"  {k}: val={v[0]:.3f} b1={v[1]} gc={v[2]} lr={v[3]:.0e}")
            for condition, (_, b1, gc, lr) in best.items():
                for seed in PHASE_B_SEEDS:
                    jobs.append((condition, o, b1, gc, lr, seed, MAIN_PARTITION))
        launch(jobs, args.save_dir, args, 'Phase B (20-seed)')

    if 'P' in args.phase:
        jobs = [(c, o, b1, gc, lr, 42, sp)
                for sp in ALT_PARTITIONS for c in CONDITIONS for o in opts
                for b1, gc, lr in grid_for(o)]
        launch(jobs, args.save_dir, args, 'Phase P (partition sensitivity)')

    print("ALL DONE", flush=True)


if __name__ == '__main__':
    main()
