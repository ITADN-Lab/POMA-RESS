"""Phase A grid at sp=88 — needed to define each optimizer's best config
under the new partition (sp99 was redundant with sp7 due to 7-bearing
split degeneracy)."""
import os, sys, time, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS, 'train_femto_leakfree.py')

OPTIMIZERS = ['AdamW', 'PMO', 'Adan', 'RAdam', 'Lion']
CONDITIONS = ['cond1', 'cond2']
SP = int(os.environ.get('FEMTO_SP', 88))
BETA1_PER_OPT = {
    'AdamW': [0.85, 0.9, 0.95], 'PMO': [0.85, 0.9, 0.95],
    'Adan': [0.85, 0.9, 0.95], 'RAdam': [0.8, 0.9, 0.95],
    'Lion': [0.9, 0.95, 0.99],
}
GC_GRID = [0.0, 0.5, 1.0, 2.0]
LR_PER_OPT = {
    'AdamW': [3e-4, 1e-3, 3e-3], 'PMO': [3e-4, 1e-3, 3e-3],
    'Adan': [3e-4, 1e-3, 3e-3], 'RAdam': [3e-4, 1e-3, 3e-3],
    'Lion': [1e-4, 3e-4, 1e-3],
}


def cfg_tag(b1, gc, lr):
    return f"b{b1}_gc{gc}_lr{lr:.0e}"


def run_one(data_dir, save_dir, python_bin, condition, opt, b1, gc, lr):
    tag = cfg_tag(b1, gc, lr)
    suffix = f"{tag}_sp{SP}"
    out = os.path.join(save_dir, f"femto_{condition}_{opt}_seed42_{suffix}.json")
    if os.path.exists(out):
        return f"{condition}/{opt}/{tag}", 'SKIP'
    cmd = [python_bin, TRAIN,
           '--data_dir', data_dir, '--condition', condition, '--optimizer', opt,
           '--val_split', 'engine', '--split_seed', str(SP),
           '--epochs', '80', '--batch_size', '256',
           '--lr', str(lr), '--beta1', str(b1), '--grad_clip', str(gc),
           '--seed', '42', '--save_dir', save_dir, '--tag_suffix', suffix,
           '--weight_decay', '1e-4']
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=900, env=env)
        if p.returncode == 0:
            return f"{condition}/{opt}/{tag}", 'OK'
        return f"{condition}/{opt}/{tag}", f"FAIL {p.returncode}"
    except subprocess.TimeoutExpired:
        return f"{condition}/{opt}/{tag}", 'TIMEOUT'


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()

    jobs = []
    for c in CONDITIONS:
        for o in OPTIMIZERS:
            for b1, gc, lr in product(BETA1_PER_OPT[o], GC_GRID, LR_PER_OPT[o]):
                jobs.append((c, o, b1, gc, lr))
    print(f"sp={SP} Phase A: {len(jobs)} jobs, {args.workers} workers", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, args.data_dir, args.save_dir, args.python_bin, *j) for j in jobs]
        for f in as_completed(futs):
            name, status = f.result()
            done += 1
            eta = (time.time() - t0) / max(done, 1) * (len(jobs) - done) / 60
            print(f"  [{done:4d}/{len(jobs)}] {status:8s} {name} (ETA {eta:.1f}min)", flush=True)
    print(f"sp={SP} Phase A done in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
