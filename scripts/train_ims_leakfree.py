"""
NASA IMS Bearing RUL — LEAK-FREE bearing-level audit (Plan B v3,
RESS-native dataset).

NASA Ames Prognostics Data Repository, Set No.4 "Bearings" — three
independent run-to-failure test sets, each with 4 bearings instrumented
by an accelerometer.

Set 1: 2156 acquisitions, 8 channels (2 accels/bearing), test ended on
       bearing 3 inner-race defect.
Set 2:  984 acquisitions, 4 channels (1 accel/bearing), test ended on
       bearing 1 outer-race defect.
Set 3: 6324 acquisitions, 4 channels (1 accel/bearing), test ended on
       bearing 3 rolling-element defect.

We treat each (set, bearing) pair as a unit; each file becomes one
acquisition; per-file features are RMS, kurtosis, peak-to-peak.
RUL target at acquisition t is min(N_total - t, RUL_CAP).
"""
import os, sys, json, time, argparse, datetime, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS)

from train_cmapss_leakfree import (
    LSTM_RUL, build_optimizer, train_one_epoch, evaluate, ArrayDataset,
)

RUL_CAP = 100
DEFAULT_SPLIT_SEED = 2024
# Bearings across the 3 test sets (12 total)
IMS_BEARINGS = [
    ('1st_test', 'b1'), ('1st_test', 'b2'),
    ('1st_test', 'b3'), ('1st_test', 'b4'),
    ('2nd_test', 'b1'), ('2nd_test', 'b2'),
    ('2nd_test', 'b3'), ('2nd_test', 'b4'),
    ('3rd_test', 'b1'), ('3rd_test', 'b2'),
    ('3rd_test', 'b3'), ('3rd_test', 'b4'),
]


def _read_set_dir(set_dir, bearing_idx, n_channels_per_bearing):
    """Return Nx3 feature matrix for one bearing in one test set."""
    files = sorted(glob.glob(os.path.join(set_dir, '*')))
    files = [f for f in files if os.path.isfile(f) and not f.endswith('.txt')]
    rows = []
    for f in files:
        try:
            # whitespace-separated, no header
            a = np.loadtxt(f)
            # column layout: bearing j has channels [j*npc, ..., j*npc+npc-1]
            cols = list(range(bearing_idx * n_channels_per_bearing,
                              (bearing_idx + 1) * n_channels_per_bearing))
            sig = a[:, cols].mean(axis=1) if len(cols) > 1 else a[:, cols[0]]
        except Exception:
            continue
        if len(sig) < 100:
            continue
        rms = float(np.sqrt(np.mean(sig * sig)))
        kurt = float(((sig - sig.mean()) ** 4).mean() / (sig.std() ** 4 + 1e-9))
        p2p = float(sig.max() - sig.min())
        rows.append([rms, kurt, p2p])
    return np.array(rows, dtype=np.float32) if rows else None


def _load_ims_bearing(data_dir, set_name, bearing):
    cache = os.path.join(data_dir, f'_cache_{set_name}_{bearing}.npy')
    if os.path.isfile(cache):
        return np.load(cache)
    set_dir = os.path.join(data_dir, set_name)
    if not os.path.isdir(set_dir):
        raise FileNotFoundError(f'{set_dir} missing')
    bidx = int(bearing.lstrip('b')) - 1
    npc = 2 if set_name == '1st_test' else 1
    mat = _read_set_dir(set_dir, bidx, npc)
    if mat is None or len(mat) < 50:
        raise RuntimeError(f'{set_name}/{bearing} parsed empty / too short')
    np.save(cache, mat)
    return mat


def build_ims_datasets(data_dir, window_size=30, val_split='engine',
                       val_frac=0.20, split_seed=DEFAULT_SPLIT_SEED):
    """Cross-set bearing-level leave-out audit.  Pool the 12 IMS bearings
    across all 3 test sets and partition them by split_seed.
    """
    mats = []
    for set_name, bearing in IMS_BEARINGS:
        try:
            mats.append((f'{set_name}/{bearing}',
                         _load_ims_bearing(data_dir, set_name, bearing)))
        except Exception as e:
            print(f'[warn] skipping {set_name}/{bearing}: {e}', file=sys.stderr)
    if len(mats) < 6:
        raise RuntimeError(f'only {len(mats)} bearings usable')

    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(mats))
    n = len(mats)
    n_val = max(1, int(round(val_frac * n)))
    n_test = max(1, int(round(val_frac * n)))
    test_idx = sorted(perm[:n_test].tolist())
    val_idx = sorted(perm[n_test:n_test + n_val].tolist())
    train_idx = sorted(perm[n_test + n_val:].tolist())
    train = [mats[i] for i in train_idx]
    val = [mats[i] for i in val_idx]
    test = [mats[i] for i in test_idx]

    pool = np.concatenate([m for _, m in train], axis=0)
    mean = pool.mean(axis=0); std = pool.std(axis=0) + 1e-8

    def build(mats):
        S, T = [], []
        for bid, mat in mats:
            feats = (mat - mean) / std
            n = len(feats)
            for i in range(window_size, n + 1):
                S.append(feats[i - window_size:i])
                T.append(min(n - i, RUL_CAP))
        return np.array(S, dtype=np.float32), np.array(T, dtype=np.float32)

    tr_s, tr_t = build(train)
    va_s, va_t = build(val)
    te_s, te_t = build(test)
    split_info = {'mode': 'bearing',
                  'train_bearings': [b for b, _ in train],
                  'val_bearings':   [b for b, _ in val],
                  'test_bearings':  [b for b, _ in test],
                  'split_seed': split_seed}
    return (ArrayDataset(tr_s, tr_t), ArrayDataset(va_s, va_t),
            ArrayDataset(te_s, te_t), pool.shape[1], split_info)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=os.path.expanduser('~/pmo_ims/data/src'))
    p.add_argument('--optimizer', default='AdamW',
                   choices=['AdamW', 'PMO', 'LAKTJU_NS', 'Adan', 'RAdam', 'Lion'])
    p.add_argument('--val_split', default='engine')
    p.add_argument('--split_seed', type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window_size', type=int, default=30)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--save_dir', default=os.path.expanduser('~/pmo_ims/results'))
    p.add_argument('--tag_suffix', default='')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds, input_dim, split_info = build_ims_datasets(
        args.data_dir, args.window_size, args.val_split, split_seed=args.split_seed)
    print(f"NASA IMS Bearing: train={len(train_ds)} val={len(val_ds)} "
          f"test={len(test_ds)} input_dim={input_dim} | {split_info}", flush=True)

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    sl = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = LSTM_RUL(input_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
    opt, sched = build_optimizer(args, model)
    crit = nn.MSELoss()

    best_val = float('inf'); best_test = float('inf'); best_mae = float('inf')
    best_ep = -1; hist = []; t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, tl, opt, crit, device, args.grad_clip)
        vr, vm = evaluate(model, vl, crit, device)
        tr_te, tr_ma = evaluate(model, sl, crit, device)
        sched.step(vr)
        if vr < best_val:
            best_val, best_test, best_mae, best_ep = vr, tr_te, tr_ma, ep
        hist.append({'epoch': ep, 'train_loss': tr, 'val_rmse': vr, 'test_rmse': tr_te})
        if ep % 15 == 0 or ep == args.epochs:
            print(f"[{ep:3d}/{args.epochs}] tr={tr:.2f} val={vr:.2f} test={tr_te:.2f} "
                  f"best={best_test:.2f}@{best_ep} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"ims_{args.optimizer}_seed{args.seed}"
    if args.tag_suffix: tag += f"_{args.tag_suffix}"
    out = os.path.join(args.save_dir, f"{tag}.json")
    json.dump({'config': vars(args), 'val_split': args.val_split,
               'split_info': split_info, 'best_val_rmse': best_val,
               'best_test_rmse': best_test, 'best_test_mae': best_mae,
               'best_epoch': best_ep, 'final_test_rmse': hist[-1]['test_rmse'],
               'history': hist, 'total_time': time.time() - t0,
               'input_dim': input_dim, 'timestamp': datetime.datetime.now().isoformat()},
              open(out, 'w'), indent=2, default=float)
    print(f"Saved {out} | best_val={best_val:.3f} best_test={best_test:.3f}")


if __name__ == '__main__':
    main()
