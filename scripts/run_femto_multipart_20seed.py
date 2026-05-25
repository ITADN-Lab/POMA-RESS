"""
FEMTO multi-partition robustness check.

After Phase A/B/P finish, for each (condition, optimizer) we re-run the
best-of-36 config at each of 3 partitions (sp 2024 / 7 / 99) × 20 seeds.
This converts the single-test-bearing-per-partition design into effectively
3 different test bearings per (cond, opt), strengthening the cross-bearing
generalization claim on cond1+cond2.

Only cond1+cond2 are run here (cond3 has only 3 bearings → degenerate
splits regardless of partition seed; cond3 stays as boundary indicator).
"""
import os, sys, json, glob, time, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS, 'train_femto_leakfree.py')

OPTIMIZERS = ['AdamW', 'PMO', 'Adan', 'RAdam', 'Lion']
CONDITIONS = ['cond1', 'cond2']
PARTITIONS = [2024, 7, 88, 1, 2]  # sp99 collided with sp7; sp1/sp2 add (6,2)/(4,1) splits
SEEDS = [42, 123, 456, 789, 1024, 2, 7, 13, 21, 34,
         55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
MAX_RETRY = 5


def cfg_tag(b1, gc, lr):
    return f"b{b1}_gc{gc}_lr{lr:.0e}"


def best_config(res_dir, condition, opt, split_seed):
    """Pick best-by-val seed-42 config under a given partition."""
    files = glob.glob(os.path.join(
        res_dir, f'femto_{condition}_{opt}_seed42_*.json'))
    best = None
    for f in files:
        try:
            d = json.load(open(f))
            if d['split_info']['split_seed'] != split_seed: continue
            c = d['config']
            if c['optimizer'] != opt or c['condition'] != condition: continue
            v = d['best_val_rmse']
            if best is None or v < best[0]:
                best = (v, c['beta1'], c['grad_clip'], c['lr'])
        except Exception: pass
    return best


def out_name(condition, opt, b1, gc, lr, seed, split_seed):
    tag = cfg_tag(b1, gc, lr)
    sp = '' if split_seed == 2024 else f"_sp{split_seed}"
    return f"femto_{condition}_{opt}_seed{seed}_{tag}{sp}.json"


def run_one(data_dir, save_dir, python_bin, epochs,
            condition, opt, b1, gc, lr, seed, split_seed):
    tag = cfg_tag(b1, gc, lr)
    name = f"{condition}/{opt}/{tag}/s{seed}/sp{split_seed}"
    out = os.path.join(save_dir, out_name(condition, opt, b1, gc, lr, seed, split_seed))
    if os.path.exists(out):
        return name, 'SKIP'
    suffix = tag if split_seed == 2024 else f"{tag}_sp{split_seed}"
    cmd = [python_bin, TRAIN,
           '--data_dir', data_dir, '--condition', condition, '--optimizer', opt,
           '--val_split', 'engine', '--split_seed', str(split_seed),
           '--epochs', str(epochs), '--batch_size', '256',
           '--lr', str(lr), '--beta1', str(b1), '--grad_clip', str(gc),
           '--seed', str(seed), '--save_dir', save_dir, '--tag_suffix', suffix,
           '--weight_decay', '1e-4']
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=80)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    jobs = []
    for opt in OPTIMIZERS:
        for cond in CONDITIONS:
            for sp in PARTITIONS:
                bc = best_config(args.save_dir, cond, opt, sp)
                if bc is None:
                    print(f"[warn] no Phase A best for {cond}/{opt}/sp{sp}")
                    continue
                _, b1, gc, lr = bc
                print(f"[plan] {cond}/{opt}/sp{sp}: b1={b1} gc={gc} lr={lr:.0e}")
                for seed in SEEDS:
                    jobs.append((cond, opt, b1, gc, lr, seed, sp))
    print(f"\nTotal jobs: {len(jobs)}", flush=True)

    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.data_dir, args.save_dir, args.python_bin,
                          args.epochs, *j) for j in jobs]
        for f in as_completed(futs):
            name, status = f.result()
            done += 1
            eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
            print(f"  [{done:4d}/{len(jobs)}] {status:14s} {name} "
                  f"(ETA {eta:.1f}min)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == '__main__':
    main()
