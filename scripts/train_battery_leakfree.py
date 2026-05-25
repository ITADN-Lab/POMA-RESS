"""
NASA Li-ion Battery RUL — LEAK-FREE battery-level audit (Plan B v2).

Cross-domain replication of the audit: electrochemical degradation (NASA
PCoE Battery Data Set #5: B0005, B0006, B0007, B0018) instead of
mechanical bearings/turbofans.  Each battery is run through charge /
discharge / impedance cycles until end-of-life (EOL = capacity dropped
to 1.4 Ah from ~2 Ah nominal).  The RUL target at cycle t is
(EOL_cycle - t) clamped at RUL_CAP.

Protocol: leave-batteries-out (analogue of leave-engines-out / -bearings),
fixed split per split_seed, same equal $\\beta_1\\times$GC$\\times$LR
36-config tuning budget per optimizer, same 4-optimizer panel
(AdamW, PMO, Adan, RAdam, Lion), same 20-seed paired bootstrap.

Input: per-cycle features (discharge capacity, voltage curve summary,
current curve summary, temperature, internal-resistance proxy) = a small
multi-D time-series of length = cycles-to-failure per battery.
"""
import os, sys, json, time, argparse, datetime
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..', '..'))
sys.path.insert(0, os.path.join(_THIS, '..'))

from train_cmapss_leakfree import (
    LSTM_RUL, build_optimizer, train_one_epoch, evaluate, ArrayDataset,
)

RUL_CAP = 100
DEFAULT_SPLIT_SEED = 2024

# 4 batteries in PCoE Battery Data Set #5
BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018']


def _load_battery_mat(data_dir, b):
    """Return Nx6 feature matrix for one battery, one row per discharge cycle.

    Features per cycle (5): [discharge_capacity, mean_voltage, mean_current,
    mean_temp, cycle_index_norm].
    The NASA .mat files contain a top-level struct named after the battery
    with a 'cycle' array; each 'cycle' has type ('charge'/'discharge'/
    'impedance') + 'data' substruct (Voltage_measured, Current_measured,
    Temperature_measured, Time, Capacity etc).  We extract one feature
    vector per *discharge* cycle.
    """
    cache = os.path.join(data_dir, f'_cache_{b}.npy')
    if os.path.isfile(cache):
        return np.load(cache)
    from scipy.io import loadmat
    # files live as <b>.mat or <data_dir>/<b>.mat
    candidates = [
        os.path.join(data_dir, f'{b}.mat'),
        os.path.join(data_dir, '1. BatteryAgingARC-FY08Q4', f'{b}.mat'),
        os.path.join(data_dir, 'BatteryAgingARC-FY08Q4', f'{b}.mat'),
    ]
    fp = next((c for c in candidates if os.path.isfile(c)), None)
    if fp is None:
        # search recursively
        for root, _, files in os.walk(data_dir):
            if f'{b}.mat' in files:
                fp = os.path.join(root, f'{b}.mat')
                break
    if fp is None:
        raise FileNotFoundError(f'battery {b} .mat not found under {data_dir}')
    m = loadmat(fp, squeeze_me=True, struct_as_record=False)
    bs = m[b]
    cycles = bs.cycle
    rows = []
    for idx, c in enumerate(cycles):
        t = str(c.type)
        if t != 'discharge': continue
        d = c.data
        try:
            cap = float(np.atleast_1d(d.Capacity)[0])
        except Exception:
            cap = float('nan')
        try:
            v = np.atleast_1d(d.Voltage_measured).astype(float)
            i = np.atleast_1d(d.Current_measured).astype(float)
            tm = np.atleast_1d(d.Temperature_measured).astype(float)
            mv = float(v.mean()); mi = float(i.mean()); mt = float(tm.mean())
        except Exception:
            mv = mi = mt = float('nan')
        rows.append([cap, mv, mi, mt, float(idx)])
    arr = np.array(rows, dtype=np.float32)
    if len(arr) < 20:
        raise RuntimeError(f'battery {b} parsed only {len(arr)} cycles')
    # normalise the cycle-index column by total cycles
    arr[:, -1] /= max(1, arr[-1, -1])
    np.save(cache, arr)
    return arr


def build_battery_datasets(data_dir, window_size=20, val_split='engine',
                           val_frac=0.25, split_seed=DEFAULT_SPLIT_SEED):
    """Return (train_ds, val_ds, test_ds, input_dim, split_info)."""
    mats = []
    for b in BATTERIES:
        try:
            mats.append((b, _load_battery_mat(data_dir, b)))
        except Exception as e:
            print(f'[warn] skipping {b}: {e}', file=sys.stderr)
    if len(mats) < 3:
        raise RuntimeError(f'only {len(mats)} batteries usable')

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

    # normalise on train pool
    pool = np.concatenate([m for _, m in train], axis=0)
    sensor_cols = list(range(pool.shape[1]))
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
    split_info = {'mode': 'battery',
                  'train_batteries': [b for b, _ in train],
                  'val_batteries':   [b for b, _ in val],
                  'test_batteries':  [b for b, _ in test],
                  'split_seed': split_seed}
    return (ArrayDataset(tr_s, tr_t), ArrayDataset(va_s, va_t),
            ArrayDataset(te_s, te_t), pool.shape[1], split_info)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=os.path.expanduser('~/pmo_battery/data/src'))
    p.add_argument('--optimizer', default='AdamW',
                   choices=['AdamW', 'PMO', 'LAKTJU_NS', 'Adan', 'RAdam', 'Lion'])
    p.add_argument('--val_split', default='engine')
    p.add_argument('--split_seed', type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window_size', type=int, default=20)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--save_dir', default=os.path.expanduser('~/pmo_battery/results'))
    p.add_argument('--tag_suffix', default='')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds, input_dim, split_info = build_battery_datasets(
        args.data_dir, args.window_size, args.val_split, split_seed=args.split_seed)
    print(f"NASA Battery: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"input_dim={input_dim} | {split_info}", flush=True)

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
    tag = f"battery_{args.optimizer}_seed{args.seed}"
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
