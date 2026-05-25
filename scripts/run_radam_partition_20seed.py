"""
Plan B (Codex v4 optional): firm up RAdam-FD004 partition-fragility statement
by running 20 seeds at each of the 3 leave-engines-out partitions.

For each partition, the partition-specific best RAdam config (selected by
seed-42 grid val) is re-run at 19 extra seeds. AdamW's partition-specific
best is also re-run. Compares paired 20-seed Δ per partition.
"""
import os, sys, json, glob, time, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS, 'train_cmapss_leakfree.py')

SUBSET = 'FD004'
OPTS = ['AdamW', 'RAdam']
PARTITIONS = [2024, 7, 99]
EXTRA_SEEDS = [123, 456, 789, 1024, 2, 7, 13, 21, 34, 55,
               89, 144, 233, 377, 610, 987, 1597, 2584, 4181]
MAX_RETRY = 6


def best_at_partition(res_dir, opt, partition, seed=42):
    """Pick best (lowest engine-val) config for opt at this partition from seed-42 grid."""
    best = None
    for f in glob.glob(os.path.join(res_dir, f'lf_{SUBSET}_{opt}_seed{seed}_*.json')):
        try:
            d = json.load(open(f))
            if d.get('split_info', {}).get('split_seed') != partition:
                continue
            c = d['config']
            if c['optimizer'] != opt or c['subset'] != SUBSET:
                continue
            v = d['best_val_rmse']
            if best is None or v < best[0]:
                best = (v, c['beta1'], c['grad_clip'], c['lr'])
        except Exception:
            pass
    return best


def cfg_tag(b1, gc, lr):
    return f"b{b1}_gc{gc}_lr{lr:.0e}"


def out_name(opt, b1, gc, lr, seed, sp):
    tag = cfg_tag(b1, gc, lr)
    suffix = '' if sp == 2024 else f'_sp{sp}'
    return f'lf_{SUBSET}_{opt}_seed{seed}_{tag}{suffix}.json'


def run_one(data_dir, save_dir, python_bin, epochs, opt, b1, gc, lr, seed, sp):
    tag = cfg_tag(b1, gc, lr)
    out = os.path.join(save_dir, out_name(opt, b1, gc, lr, seed, sp))
    name = f"{SUBSET}/{opt}/{tag}/s{seed}/sp{sp}"
    if os.path.exists(out):
        return name, 'SKIP'
    suffix = tag if sp == 2024 else f"{tag}_sp{sp}"
    cmd = [
        python_bin, TRAIN,
        '--data_dir', data_dir, '--subset', SUBSET, '--optimizer', opt,
        '--val_split', 'engine', '--split_seed', str(sp),
        '--epochs', str(epochs), '--batch_size', '256',
        '--lr', str(lr), '--beta1', str(b1), '--grad_clip', str(gc),
        '--seed', str(seed), '--save_dir', save_dir, '--tag_suffix', suffix,
        '--weight_decay', '1e-4',
    ]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for attempt in range(1, MAX_RETRY + 1):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
            if p.returncode == 0 and os.path.exists(out):
                return name, ('OK' if attempt == 1 else f'OK(retry{attempt})')
            if 'out of memory' in (p.stderr or '') and attempt < MAX_RETRY:
                time.sleep(45 * attempt); continue
            return name, f"FAIL rc={p.returncode}"
        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRY:
                time.sleep(30); continue
            return name, 'TIMEOUT'
    return name, 'FAIL'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    jobs = []
    for opt in OPTS:
        for sp in PARTITIONS:
            best = best_at_partition(args.save_dir, opt, sp, seed=42)
            if best is None:
                print(f"WARN no best config for {opt} sp{sp}")
                continue
            _, b1, gc, lr = best
            print(f"{opt} sp{sp}: best b1={b1} gc={gc} lr={lr:.0e}")
            for seed in EXTRA_SEEDS:
                jobs.append((opt, b1, gc, lr, seed, sp))

    print(f"\ntotal jobs: {len(jobs)} (will SKIP already-done)")
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.data_dir, args.save_dir, args.python_bin,
                          args.epochs, *j) for j in jobs]
        for f in as_completed(futs):
            name, status = f.result()
            done += 1
            eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
            print(f"  [{done:3d}/{len(jobs)}] {status:14s} {name} (ETA {eta:.1f}min)", flush=True)
    print(f"DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
