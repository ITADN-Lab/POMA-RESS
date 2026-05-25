"""
Plan A #4 (Codex gpt-5.5): validation-leakage diagnostic.
Re-run the same 36-config AdamW grid with --val_split window (the leaky 85/15
random-window split) so we can compare, per subset, the selection regret of
window-val vs engine-val (already in results_leakfree/).

Small budget: AdamW only, seed 42 only, 144 runs total. ~30 min on 11号 with
4 workers (sharing the GPU with the multi-audit Phase P).
"""
import os, sys, json, glob, time, subprocess, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

THIS = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS, 'train_cmapss_leakfree.py')

SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
BETA1 = [0.8, 0.9, 0.95]
GC = [0.0, 0.5, 1.0, 2.0]
LR = [3e-4, 1e-3, 3e-3]


def cfg_tag(b1, gc, lr):
    return f"b{b1}_gc{gc}_lr{lr:.0e}"


def out_name(subset, b1, gc, lr, seed):
    return f"wv_{subset}_AdamW_seed{seed}_{cfg_tag(b1,gc,lr)}.json"


def run_one(data_dir, save_dir, python_bin, epochs, subset, b1, gc, lr, seed):
    tag = cfg_tag(b1, gc, lr)
    name = f"{subset}/AdamW/{tag}/s{seed}/window"
    out = os.path.join(save_dir, out_name(subset, b1, gc, lr, seed))
    if os.path.exists(out):
        return name, 'SKIP'
    suffix = tag
    cmd = [
        python_bin, TRAIN,
        '--data_dir', data_dir, '--subset', subset, '--optimizer', 'AdamW',
        '--val_split', 'window',
        '--epochs', str(epochs), '--batch_size', '256',
        '--lr', str(lr), '--beta1', str(b1), '--grad_clip', str(gc),
        '--seed', str(seed), '--save_dir', save_dir,
        '--tag_suffix', suffix,
        '--weight_decay', '1e-4',
    ]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for attempt in range(1, 5):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=1200, env=env)
            # rename the output to wv_* (the trainer writes lf_*_<tag>.json by default)
            default_out = os.path.join(save_dir, f"lf_{subset}_AdamW_seed{seed}_{tag}.json")
            if os.path.exists(default_out) and not os.path.exists(out):
                os.rename(default_out, out)
            if p.returncode == 0 and os.path.exists(out):
                return name, 'OK' if attempt == 1 else f'OK(retry{attempt})'
            if 'out of memory' in (p.stderr or '') and attempt < 4:
                time.sleep(60 * attempt); continue
            return name, f"FAIL rc={p.returncode} {(p.stderr or '')[-160:]}"
        except subprocess.TimeoutExpired:
            if attempt < 4:
                time.sleep(30); continue
            return name, 'TIMEOUT'
    return name, 'FAIL'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    jobs = [(s, b1, gc, lr, 42)
            for s in SUBSETS for b1, gc, lr in product(BETA1, GC, LR)]
    print(f"window-val grid: {len(jobs)} jobs, {args.workers} workers", flush=True)
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
