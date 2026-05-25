"""
C-MAPSS RUL — LEAK-FREE engine-level validation re-evaluation (Plan B).

Difference from train_cmapss_rul.py:
  --val_split engine : validation windows are drawn from a DISJOINT set of
                       engines (leave-engines-out), so validation RMSE cannot
                       be inflated by windows overlapping the training engines.
                       The engine partition is fixed by SPLIT_SEED (constant
                       across every config and every training seed) so the
                       hyper-parameter comparison is fair.
                       Normalisation statistics are computed from the TRAINING
                       engines only.
  --val_split window : the legacy leaky 85/15 random-window split (kept so the
                       two protocols can be compared head-to-head).

The C-MAPSS test set is, by construction, a disjoint set of engines, so the
reported test RMSE is always leak-free; only the train/val split changes.

Optimizers: AdamW, LAKTJU_NS (the optimizer the paper calls PMO).
"""
import os
import sys
import json
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, '..', '..'))   # repo root
sys.path.insert(0, os.path.join(_THIS, '..'))

STD_THRESHOLD = 0.01
RUL_CAP = 125
DEFAULT_SPLIT_SEED = 2024   # FIXED engine partition — never tied to training seed


def load_cmapss_txt(filepath):
    return np.loadtxt(filepath)


def get_sensor_cols(rows):
    """Active (non-constant) sensor indices, computed from the given rows only."""
    all_sensors = rows[:, 5:]
    sensor_std = all_sensors.std(axis=0)
    active = np.where(sensor_std > STD_THRESHOLD)[0]
    if len(active) < 5:
        active = np.arange(21)
    return sorted(active.tolist())


def build_windows(data, engine_ids_keep, sensor_cols, mean, std, window_size):
    """Sliding windows + capped RUL targets for the engines in engine_ids_keep."""
    samples, targets = [], []
    eid_col = data[:, 0].astype(int)
    for eid in engine_ids_keep:
        engine_data = data[eid_col == eid]
        sensors = engine_data[:, 5:][:, sensor_cols]
        sensors = (sensors - mean) / std
        max_cycle = len(engine_data)
        for i in range(window_size, max_cycle + 1):
            samples.append(sensors[i - window_size:i])
            targets.append(min(max_cycle - i, RUL_CAP))
    return (np.array(samples, dtype=np.float32),
            np.array(targets, dtype=np.float32))


class ArrayDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples, self.targets = samples, targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32))


def make_test_set(data_dir, subset, sensor_cols, mean, std, window_size):
    """Last window per test engine; true RUL from the RUL file."""
    test_data = load_cmapss_txt(os.path.join(data_dir, f'test_{subset}.txt'))
    rul_data = load_cmapss_txt(os.path.join(data_dir, f'RUL_{subset}.txt'))
    eid_col = test_data[:, 0].astype(int)
    samples, targets = [], []
    for idx, eid in enumerate(np.unique(eid_col)):
        engine_data = test_data[eid_col == eid]
        sensors = engine_data[:, 5:][:, sensor_cols]
        sensors = (sensors - mean) / std
        if len(sensors) >= window_size:
            window = sensors[-window_size:]
        else:
            pad = np.zeros((window_size - len(sensors), len(sensor_cols)),
                           dtype=np.float32)
            window = np.concatenate([pad, sensors], axis=0)
        samples.append(window.astype(np.float32))
        targets.append(min(float(rul_data[idx]), RUL_CAP))
    return ArrayDataset(np.array(samples, dtype=np.float32),
                        np.array(targets, dtype=np.float32))


def build_datasets(data_dir, subset, window_size, val_split, val_frac=0.15,
                   split_seed=DEFAULT_SPLIT_SEED):
    """Return (train_ds, val_ds, test_ds, input_dim, split_info)."""
    train_data = load_cmapss_txt(os.path.join(data_dir, f'train_{subset}.txt'))
    eid_col = train_data[:, 0].astype(int)
    all_engines = np.unique(eid_col)

    if val_split == 'engine':
        # leave-engines-out: a fixed disjoint engine partition
        rng = np.random.RandomState(split_seed)
        perm = rng.permutation(all_engines)
        n_val = max(1, int(round(val_frac * len(perm))))
        val_engines = sorted(perm[:n_val].tolist())
        train_engines = sorted(perm[n_val:].tolist())
        # sensor selection + normalisation from TRAIN engines only
        train_rows = train_data[np.isin(eid_col, train_engines)]
        sensor_cols = get_sensor_cols(train_rows)
        tr_sens = train_rows[:, 5:][:, sensor_cols]
        mean = tr_sens.mean(axis=0)
        std = tr_sens.std(axis=0) + 1e-8
        tr_s, tr_t = build_windows(train_data, train_engines, sensor_cols,
                                   mean, std, window_size)
        va_s, va_t = build_windows(train_data, val_engines, sensor_cols,
                                   mean, std, window_size)
        train_ds = ArrayDataset(tr_s, tr_t)
        val_ds = ArrayDataset(va_s, va_t)
        split_info = {'mode': 'engine', 'n_train_engines': len(train_engines),
                      'n_val_engines': len(val_engines),
                      'val_engines': val_engines, 'split_seed': split_seed}
    elif val_split == 'window':
        # legacy leaky 85/15 random-window split
        sensor_cols = get_sensor_cols(train_data)
        all_sens = train_data[:, 5:][:, sensor_cols]
        mean = all_sens.mean(axis=0)
        std = all_sens.std(axis=0) + 1e-8
        s, t = build_windows(train_data, all_engines.tolist(), sensor_cols,
                             mean, std, window_size)
        full = ArrayDataset(s, t)
        n_val = int(val_frac * len(full))
        n_tr = len(full) - n_val
        g = torch.Generator().manual_seed(split_seed)
        train_ds, val_ds = torch.utils.data.random_split(full, [n_tr, n_val],
                                                         generator=g)
        split_info = {'mode': 'window', 'n_train_windows': n_tr,
                      'n_val_windows': n_val, 'split_seed': split_seed}
    else:
        raise ValueError(val_split)

    test_ds = make_test_set(data_dir, subset, sensor_cols, mean, std, window_size)
    return train_ds, val_ds, test_ds, len(sensor_cols), split_info


class LSTM_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)


def build_optimizer(args, model):
    params = model.parameters()
    if args.optimizer == 'AdamW':
        lr = args.lr or 1e-3
        opt = optim.AdamW(params, lr=lr, betas=(args.beta1, 0.999),
                          weight_decay=args.weight_decay)
    elif args.optimizer in ('PMO', 'LAKTJU_NS'):
        from optimizer.LAKTJU_NS import LAKTJU_NS
        lr = args.lr or 1e-3
        opt = LAKTJU_NS(params, lr=lr, betas=(args.beta1, 0.999),
                        weight_decay=args.weight_decay,
                        ns_interval=args.ns_interval, ns_steps=args.ns_steps,
                        ns_max_dim=args.ns_max_dim)
    elif args.optimizer == 'Adan':
        from adan_pytorch import Adan
        lr = args.lr or 1e-3
        opt = Adan(list(params), lr=lr,
                   betas=(args.beta1, 0.92, 0.99),
                   weight_decay=args.weight_decay)
    elif args.optimizer == 'RAdam':
        lr = args.lr or 1e-3
        opt = optim.RAdam(params, lr=lr, betas=(args.beta1, 0.999),
                          weight_decay=args.weight_decay)
    elif args.optimizer == 'Lion':
        # Lion: sign-based update with EMA, custom implementation matching
        # train_cmapss_rul.py's Lion mode (Chen et al., 2024).
        from types import MethodType
        lr = args.lr or 3e-4
        wd = args.weight_decay
        opt = optim.AdamW(params, lr=lr, betas=(args.beta1, 0.99),
                          weight_decay=0.0)
        def lion_step(self, closure=None):
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                b1, _ = group['betas']
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    st = self.state[p]
                    if 'exp_avg' not in st:
                        st['exp_avg'] = torch.zeros_like(p)
                    ea = st['exp_avg']
                    ea.mul_(b1).add_(grad, alpha=1 - b1)
                    if wd != 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(ea.sign(), alpha=-lr)
            return loss
        opt.step = MethodType(lion_step, opt)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    sched = lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5)
    return opt, sched


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds.append(model(x).cpu().numpy())
        targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    return rmse, mae


def parse_args():
    p = argparse.ArgumentParser(description='C-MAPSS RUL — leak-free re-evaluation')
    p.add_argument('--data_dir', type=str, default='../data/cmapss')
    p.add_argument('--subset', type=str, default='FD001',
                   choices=['FD001', 'FD002', 'FD003', 'FD004'])
    p.add_argument('--optimizer', type=str, default='AdamW',
                   choices=['AdamW', 'PMO', 'LAKTJU_NS', 'Adan', 'RAdam', 'Lion'])
    p.add_argument('--val_split', type=str, default='engine',
                   choices=['engine', 'window'])
    p.add_argument('--split_seed', type=int, default=DEFAULT_SPLIT_SEED,
                   help='engine-partition seed (fixed across configs/seeds)')
    p.add_argument('--epochs', type=int, default=100)
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
    p.add_argument('--save_dir', type=str, default='../results_leakfree')
    p.add_argument('--tag_suffix', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds, input_dim, split_info = build_datasets(
        args.data_dir, args.subset, args.window_size, args.val_split,
        split_seed=args.split_seed)
    print(f"C-MAPSS {args.subset} val_split={args.val_split}: "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"input_dim={input_dim} | {split_info}")

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
                  f"val_rmse={val_rmse:.2f} test_rmse={test_rmse:.2f} "
                  f"best_test={best_test_rmse:.2f}@{best_epoch} "
                  f"({time.time()-start:.0f}s)", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"lf_{args.subset}_{args.optimizer}_seed{args.seed}"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"
    result = {
        'config': vars(args),
        'val_split': args.val_split,
        'split_info': split_info,
        'best_val_rmse': best_val_rmse,
        'best_test_rmse': best_test_rmse,
        'best_test_mae': best_test_mae,
        'best_epoch': best_epoch,
        'final_test_rmse': history[-1]['test_rmse'],
        'history': history,
        'total_time': time.time() - start,
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
