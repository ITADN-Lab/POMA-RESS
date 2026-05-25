"""
FEMTO PHM 2012 Bearing RUL — LEAK-FREE bearing-level validation (Plan B1).

Real-industrial replication of the leak-free equal-budget audit protocol on
PRONOSTIA accelerated-bearing-degradation data (FEMTO-ST Institute, France).

Differences from train_cmapss_leakfree.py:
  * "engine" -> "bearing" (the indivisible unit of leave-out)
  * 3 operating conditions: cond1 (1800rpm/4000N), cond2 (1650rpm/4200N),
    cond3 (1500rpm/5000N) — analogous to FD001..FD003/4
  * Raw measurement: bi-axial vibration sampled at 25.6 kHz, 0.1 s every 10 s
    -> we compress each 0.1 s window to (RMS_H, RMS_V, Kurt_H, Kurt_V,
        Peak2Peak_H, Peak2Peak_V) — 6 statistical features per acquisition
  * Per-bearing time-series is the sequence of these 6-D feature vectors
    over the bearing's life; the RUL target at index t is (T_total - t)
    clamped at RUL_CAP, where T_total is the bearing's total acquisition count
  * Leave-bearings-out: within each condition, the bearings are partitioned
    into (train | bearing-val | test) by SPLIT_SEED — fixed across configs/seeds
"""
import os
import sys
import json
import time
import argparse
import datetime
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..', '..'))   # repo root
sys.path.insert(0, os.path.join(_THIS, '..'))

# Re-use the C-MAPSS leak-free script's optimizer builder + LSTM + train/eval.
from train_cmapss_leakfree import (
    LSTM_RUL, build_optimizer, train_one_epoch, evaluate, ArrayDataset,
)

RUL_CAP = 125          # mirrors C-MAPSS convention; total bearing lives are
                       # typically 800-2700 acquisitions; we cap to keep target
                       # comparable across bearings.
DEFAULT_SPLIT_SEED = 2024

# FEMTO bearings by operating condition.  In the official challenge the
# Learning set has bearing*_1 and *_2 (full run-to-failure), the Test set
# has *_3 .. *_7 (truncated, ground-truth RUL).  We use ALL bearings as
# run-to-failure-equivalent series; the loader handles missing tail data.
FEMTO_BEARINGS = {
    'cond1': ['Bearing1_1', 'Bearing1_2', 'Bearing1_3', 'Bearing1_4',
              'Bearing1_5', 'Bearing1_6', 'Bearing1_7'],
    'cond2': ['Bearing2_1', 'Bearing2_2', 'Bearing2_3', 'Bearing2_4',
              'Bearing2_5', 'Bearing2_6', 'Bearing2_7'],
    'cond3': ['Bearing3_1', 'Bearing3_2', 'Bearing3_3'],
}

# Ground-truth final-RUL for the truncated test bearings (PHM 2012 challenge
# revealed after challenge close; values are RUL at end of provided segment,
# in seconds).  Source: FEMTO-ST original challenge metadata.
FEMTO_TEST_FINAL_RUL_SEC = {
    'Bearing1_3': 5730, 'Bearing1_4': 339,  'Bearing1_5': 1610,
    'Bearing1_6': 1460, 'Bearing1_7': 7570,
    'Bearing2_3': 7530, 'Bearing2_4': 1390, 'Bearing2_5': 3090,
    'Bearing2_6': 1290, 'Bearing2_7': 580,
    'Bearing3_3': 820,
}


def _read_bearing_dir(bearing_dir):
    """Return Nx6 feature matrix for one bearing.

    Reads acc_*.csv files in time order.  Each csv has 2560 rows
    (0.1s at 25.6kHz) with columns [h, m, s, us, vib_h, vib_v].
    We collapse each file to 6 features: (rms_h, rms_v, kurt_h, kurt_v,
    p2p_h, p2p_v).
    """
    files = sorted(glob.glob(os.path.join(bearing_dir, 'acc_*.csv')))
    rows = []
    for f in files:
        try:
            # FEMTO CSVs are semicolon-separated; older mirrors use comma.
            try:
                a = np.loadtxt(f, delimiter=';', usecols=(4, 5))
            except (ValueError, IndexError):
                a = np.loadtxt(f, delimiter=',', usecols=(4, 5))
        except Exception:
            continue
        if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 100:
            continue
        h, v = a[:, 0], a[:, 1]
        rms_h, rms_v = np.sqrt(np.mean(h * h)), np.sqrt(np.mean(v * v))
        kurt_h = float(((h - h.mean()) ** 4).mean() / (h.std() ** 4 + 1e-9))
        kurt_v = float(((v - v.mean()) ** 4).mean() / (v.std() ** 4 + 1e-9))
        p2p_h = float(h.max() - h.min())
        p2p_v = float(v.max() - v.min())
        rows.append([rms_h, rms_v, kurt_h, kurt_v, p2p_h, p2p_v])
    return np.array(rows, dtype=np.float32) if rows else None


def _load_bearing(data_dir, bearing):
    """Cached per-bearing loader: writes a .npy beside the bearing folder.

    Searches Learning_set, Test_set, and Full_Test_Set subdirectories.
    """
    cache = os.path.join(data_dir, f'_cache_{bearing}.npy')
    if os.path.isfile(cache):
        return np.load(cache)
    # Prefer Full_Test_Set (extended to run-to-failure post-challenge)
    # over Test_set (truncated mid-life).  Learning_set bearings already run
    # to failure.
    found = None
    for sub in ('Learning_set', 'Full_Test_Set', 'Test_set', '.'):
        cand = os.path.join(data_dir, sub, bearing)
        if os.path.isdir(cand):
            found = cand
            break
    if found is None:
        raise FileNotFoundError(f'bearing {bearing} not found under {data_dir}')
    mat = _read_bearing_dir(found)
    if mat is None or len(mat) < 50:
        raise RuntimeError(f'bearing {bearing} parsed empty / too short')
    np.save(cache, mat)
    return mat


def build_femto_windows(bearing_mats, sensor_cols, mean, std, window_size,
                        truncate_final_rul=None):
    """Sliding windows + capped RUL targets for a list of (id, matrix)."""
    samples, targets = [], []
    for bid, mat in bearing_mats:
        feats = (mat[:, sensor_cols] - mean) / std
        n = len(feats)
        # If a truncated-test final-RUL is known (in seconds), use it to
        # set the actual RUL at the LAST window; otherwise assume the
        # bearing ran to failure and RUL_at_end = 0.
        final_rul_idx = 0
        if truncate_final_rul is not None and bid in truncate_final_rul:
            # acquisitions are taken every 10 s; convert.
            final_rul_idx = max(0, int(round(truncate_final_rul[bid] / 10.0)))
        for i in range(window_size, n + 1):
            samples.append(feats[i - window_size:i])
            # RUL counts down from (n - i + final_rul_idx) at the window-end.
            rul = min(n - i + final_rul_idx, RUL_CAP)
            targets.append(float(rul))
    if not samples:
        raise RuntimeError('build_femto_windows: empty')
    return (np.array(samples, dtype=np.float32),
            np.array(targets, dtype=np.float32))


def build_femto_datasets(data_dir, condition, window_size,
                         val_split='engine', val_frac=0.20,
                         split_seed=DEFAULT_SPLIT_SEED):
    """Return (train_ds, val_ds, test_ds, input_dim, split_info).

    Within each operating condition we partition bearings into:
      * train  (~60%, run-to-failure series)
      * bearing-val (~20%, leave-bearings-out)
      * test   (~20%, held-out bearings)
    The partition is fixed by split_seed.
    """
    bearings = FEMTO_BEARINGS[condition]
    mats = []
    for b in bearings:
        try:
            mats.append((b, _load_bearing(data_dir, b)))
        except Exception as e:
            print(f'[warn] skipping {b}: {e}', file=sys.stderr)
    if len(mats) < 3:
        raise RuntimeError(f'{condition}: only {len(mats)} usable bearings')

    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(len(mats))
    n = len(mats)
    n_val = max(1, int(round(val_frac * n)))
    n_test = max(1, int(round(val_frac * n)))
    test_idx = sorted(perm[:n_test].tolist())
    val_idx = sorted(perm[n_test:n_test + n_val].tolist())
    train_idx = sorted(perm[n_test + n_val:].tolist())

    train_mats = [mats[i] for i in train_idx]
    val_mats = [mats[i] for i in val_idx]
    test_mats = [mats[i] for i in test_idx]

    if val_split != 'engine':
        # Window-level leaky split for the T1 diagnostic (matches the
        # C-MAPSS audit's contrast).
        all_mats = mats
        # use all bearings, no leave-out; random window split below.
        feat_pool = np.concatenate([m for _, m in all_mats], axis=0)
        sensor_std = feat_pool.std(axis=0)
        sensor_cols = sorted(np.where(sensor_std > 1e-4)[0].tolist())
        train_pool = feat_pool[:, sensor_cols]
        mean = train_pool.mean(axis=0)
        std = train_pool.std(axis=0) + 1e-8
        full_s, full_t = build_femto_windows(all_mats, sensor_cols, mean, std,
                                             window_size,
                                             truncate_final_rul=FEMTO_TEST_FINAL_RUL_SEC)
        full = ArrayDataset(full_s, full_t)
        nv = int(val_frac * len(full))
        nt = len(full) - nv
        g = torch.Generator().manual_seed(split_seed)
        train_ds, val_ds = torch.utils.data.random_split(full, [nt, nv], generator=g)
        # For window-mode test we re-use full as proxy (leaky-protocol diagnostic).
        test_ds = ArrayDataset(full_s[:1], full_t[:1])
        split_info = {'mode': 'window', 'n_train': nt, 'n_val': nv,
                      'split_seed': split_seed}
        return train_ds, val_ds, test_ds, len(sensor_cols), split_info

    # engine (= bearing) leak-free mode
    train_pool = np.concatenate([m for _, m in train_mats], axis=0)
    sensor_std = train_pool.std(axis=0)
    sensor_cols = sorted(np.where(sensor_std > 1e-4)[0].tolist())
    if len(sensor_cols) < 3:
        sensor_cols = list(range(train_pool.shape[1]))
    tr_pool = train_pool[:, sensor_cols]
    mean = tr_pool.mean(axis=0)
    std = tr_pool.std(axis=0) + 1e-8

    # All bearings (Learning_set + Full_Test_Set) ran to failure → no
    # residual-RUL truncation needed.
    tr_s, tr_t = build_femto_windows(train_mats, sensor_cols, mean, std,
                                     window_size, truncate_final_rul=None)
    va_s, va_t = build_femto_windows(val_mats, sensor_cols, mean, std,
                                     window_size, truncate_final_rul=None)
    te_s, te_t = build_femto_windows(test_mats, sensor_cols, mean, std,
                                     window_size, truncate_final_rul=None)
    split_info = {'mode': 'engine',
                  'train_bearings': [b for b, _ in train_mats],
                  'val_bearings': [b for b, _ in val_mats],
                  'test_bearings': [b for b, _ in test_mats],
                  'split_seed': split_seed}
    return (ArrayDataset(tr_s, tr_t), ArrayDataset(va_s, va_t),
            ArrayDataset(te_s, te_t), len(sensor_cols), split_info)


def parse_args():
    p = argparse.ArgumentParser(description='FEMTO bearing RUL — leak-free audit')
    p.add_argument('--data_dir', type=str,
                   default=os.path.expanduser('~/pmo_femto/data/processed'))
    p.add_argument('--condition', type=str, default='cond1',
                   choices=['cond1', 'cond2', 'cond3'])
    p.add_argument('--optimizer', type=str, default='AdamW',
                   choices=['AdamW', 'PMO', 'LAKTJU_NS', 'Adan', 'RAdam', 'Lion'])
    p.add_argument('--val_split', type=str, default='engine',
                   choices=['engine', 'window'])
    p.add_argument('--split_seed', type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window_size', type=int, default=30)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--save_dir', type=str,
                   default=os.path.expanduser('~/pmo_femto/results'))
    p.add_argument('--tag_suffix', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds, input_dim, split_info = build_femto_datasets(
        args.data_dir, args.condition, args.window_size, args.val_split,
        split_seed=args.split_seed)
    print(f"FEMTO {args.condition} val_split={args.val_split}: "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"input_dim={input_dim} | {split_info}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2)

    model = LSTM_RUL(input_dim, args.hidden_dim, args.num_layers,
                     args.dropout).to(device)
    optimizer, scheduler = build_optimizer(args, model)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_test_rmse = float('inf')
    best_test_mae = float('inf')
    best_epoch = -1
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, args.grad_clip)
        val_rmse, val_mae = evaluate(model, val_loader, criterion, device)
        test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_rmse)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_test_rmse = test_rmse
            best_test_mae = test_mae
            best_epoch = epoch
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_rmse': val_rmse, 'test_rmse': test_rmse})
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] train={train_loss:.2f} "
                  f"val={val_rmse:.2f} test={test_rmse:.2f} "
                  f"best_test={best_test_rmse:.2f}@{best_epoch} "
                  f"({time.time()-start:.0f}s)", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"femto_{args.condition}_{args.optimizer}_seed{args.seed}"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"
    result = {
        'config': vars(args), 'val_split': args.val_split,
        'split_info': split_info,
        'best_val_rmse': best_val_rmse,
        'best_test_rmse': best_test_rmse,
        'best_test_mae': best_test_mae,
        'best_epoch': best_epoch,
        'final_test_rmse': history[-1]['test_rmse'],
        'history': history, 'total_time': time.time() - start,
        'input_dim': input_dim,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    out_path = os.path.join(args.save_dir, f"{tag}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved {out_path} | best_val={best_val_rmse:.3f} "
          f"best_test={best_test_rmse:.3f}")


if __name__ == '__main__':
    main()
