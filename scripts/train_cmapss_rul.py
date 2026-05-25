"""
NASA C-MAPSS Turbofan Engine RUL Prediction for LafTJU-TII Paper.

Dataset: NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
  - 4 sub-datasets (FD001-FD004) with different operating conditions and fault modes
  - 21 sensor readings per time step, 100 engines per subset
  - Task: Predict Remaining Useful Life (RUL) — regression

Model: LSTM (2-layer) for time series RUL prediction
Optimizers: AdamW, LAKTJU_NS_Adam, LAKTJU_NS, LAKTJU_Lite
"""
import os
import sys
import math
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
# Support both deployment layouts:
#   (a) LafTJU-TII repo: <repo>/experiments/scripts/  → add <repo>/ for "optimizer.*"
#   (b) flat remote:     ~/laftju_tii_run/scripts/    → add ~/laftju_tii_run/ for "optimizer.*"
sys.path.insert(0, os.path.join(_THIS, '..', '..'))   # repo root (layout a)
sys.path.insert(0, os.path.join(_THIS, '..'))         # one-level up (layout b)


# ────────────────────────────────────────────────────────────────
# Data: NASA C-MAPSS
# ────────────────────────────────────────────────────────────────

# Sensor selection: auto-detect near-constant sensors per subset
# Default for FD001; for FD002-004, computed dynamically from training data
DEFAULT_DROP_SENSORS = [0, 4, 5, 9, 15, 17, 18]
STD_THRESHOLD = 0.01  # sensors with std < threshold considered near-constant

def get_sensor_cols(train_data, subset='FD001'):
    """Auto-detect non-constant sensors from training data.
    Returns list of sensor column indices (0-20 within the 21 sensor columns).
    """
    all_sensors = train_data[:, 5:]  # 21 sensor columns
    sensor_std = all_sensors.std(axis=0)
    # Drop sensors with near-zero standard deviation
    active = np.where(sensor_std > STD_THRESHOLD)[0]
    if len(active) < 5:  # safety: keep at least 5 sensors
        active = np.arange(21)
    return sorted(active.tolist())

CMAPSS_CONFIGS = {
    'FD001': {'num_conditions': 1, 'num_faults': 1},
    'FD002': {'num_conditions': 6, 'num_faults': 1},
    'FD003': {'num_conditions': 1, 'num_faults': 2},
    'FD004': {'num_conditions': 6, 'num_faults': 2},
}

RUL_CAP = 125  # Piecewise linear cap


def load_cmapss_txt(filepath):
    """Load a C-MAPSS train/test/RUL text file into numpy arrays.

    Train file columns: engine_id, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1..21
    """
    data = np.loadtxt(filepath)
    return data


class CMAPSSDataset(Dataset):
    """NASA C-MAPSS RUL Prediction Dataset.

    Processes raw time series into fixed-length sliding windows with RUL targets.

    Expected data directory structure:
        data_dir/
        ├── train_FD001.txt
        ├── test_FD001.txt
        ├── RUL_FD001.txt
        ├── train_FD002.txt
        ...
    """
    def __init__(self, data_dir, subset='FD001', window_size=30,
                 mode='train', normalize=True, rul_cap=125):
        self.window_size = window_size
        self.rul_cap = rul_cap
        self.mode = mode
        self.subset = subset

        # Load train data for normalization stats
        train_path = os.path.join(data_dir, f'train_{subset}.txt')
        train_data = load_cmapss_txt(train_path)

        # Auto-detect active sensors for this subset
        self.sensor_cols = get_sensor_cols(train_data, subset)
        n_sensors = len(self.sensor_cols)

        # Compute normalization stats from training data
        all_sensors = train_data[:, 5:]  # 21 sensor columns
        self.sensor_mean = all_sensors[:, self.sensor_cols].mean(axis=0)
        self.sensor_std = all_sensors[:, self.sensor_cols].std(axis=0) + 1e-8

        if mode == 'train':
            self.samples, self.targets = self._process_train(train_data)
        else:
            test_path = os.path.join(data_dir, f'test_{subset}.txt')
            rul_path = os.path.join(data_dir, f'RUL_{subset}.txt')
            test_data = load_cmapss_txt(test_path)
            rul_data = load_cmapss_txt(rul_path)
            self.samples, self.targets = self._process_test(test_data, rul_data)

        print(f"C-MAPSS {subset} ({mode}): {len(self.samples)} windows, "
              f"sensor_dim={len(self.sensor_cols)}, window={window_size}")

    def _normalize_sensors(self, data):
        """Select relevant sensors and normalize."""
        sensors = data[:, 5:][:, self.sensor_cols]  # first extract 21 sensor cols, then select
        return (sensors - self.sensor_mean) / self.sensor_std

    def _process_train(self, data):
        """Process training data: sliding windows with RUL targets."""
        samples, targets = [], []
        engine_ids = data[:, 0].astype(int)
        unique_engines = np.unique(engine_ids)

        for eid in unique_engines:
            engine_data = data[engine_ids == eid]
            sensor_normed = self._normalize_sensors(engine_data)
            max_cycle = len(engine_data)

            # RUL = max_cycle - current_cycle, capped
            for i in range(self.window_size, max_cycle + 1):
                window = sensor_normed[i - self.window_size:i]
                rul = min(max_cycle - i, self.rul_cap)
                samples.append(window)
                targets.append(rul)

        return np.array(samples, dtype=np.float32), np.array(targets, dtype=np.float32)

    def _process_test(self, test_data, rul_data):
        """Process test data: last window per engine, true RUL from file."""
        samples, targets = [], []
        engine_ids = test_data[:, 0].astype(int)
        unique_engines = np.unique(engine_ids)

        for idx, eid in enumerate(unique_engines):
            engine_data = test_data[engine_ids == eid]
            sensor_normed = self._normalize_sensors(engine_data)

            # Take the last window
            if len(engine_data) >= self.window_size:
                window = sensor_normed[-self.window_size:]
            else:
                # Pad with zeros if shorter than window
                pad = np.zeros((self.window_size - len(engine_data), len(self.sensor_cols)),
                               dtype=np.float32)
                window = np.concatenate([pad, sensor_normed], axis=0)

            true_rul = min(float(rul_data[idx]), self.rul_cap)
            samples.append(window)
            targets.append(true_rul)

        return np.array(samples, dtype=np.float32), np.array(targets, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # (W, S)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)   # scalar
        return x, y


# ────────────────────────────────────────────────────────────────
# Model: LSTM for RUL Prediction
# ────────────────────────────────────────────────────────────────

class LSTM_RUL(nn.Module):
    """2-layer LSTM for Remaining Useful Life prediction.

    Architecture:
      LSTM(input_dim → hidden_dim × 2 layers) → dropout → fc(hidden → hidden//2) → relu → fc(hidden//2 → 1)
    """
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
        # x: (B, W, S)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # Last layer hidden state
        return out.squeeze(-1)


# ────────────────────────────────────────────────────────────────
# Optimizer factory
# ────────────────────────────────────────────────────────────────

def build_optimizer(args, model):
    params = model.parameters()
    opt_name = args.optimizer

    if opt_name == 'SGD':
        lr = args.lr or 0.01
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif opt_name == 'Adam':
        lr = args.lr or 1e-3
        optimizer = optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'AdamW':
        lr = args.lr or 1e-3
        optimizer = optim.AdamW(params, lr=lr, betas=(args.beta1, 0.999),
                                weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'Lion':
        # Sign-based optimizer following the Lion paper (Chen et al., 2023)
        # Uses EMA of gradient with sign-based update
        from types import MethodType
        lr = args.lr or 3e-4
        wd = args.weight_decay
        betas_lion = (0.9, 0.99)
        optimizer = optim.AdamW(params, lr=lr, betas=betas_lion, weight_decay=0.0)
        def lion_step(self, closure=None):
            loss = None
            if closure is not None: loss = closure()
            for group in self.param_groups:
                beta1, beta2 = group['betas']
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    state = self.state[p]
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # Sign-based update: direction from EMA, magnitude from current grad
                    update = exp_avg.sign()
                    if wd != 0:
                        p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-lr)
            return loss
        optimizer.step = MethodType(lion_step, optimizer)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'RAdam':
        lr = args.lr or 1e-3
        optimizer = optim.RAdam(params, lr=lr, betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'Adan':
        from adan_pytorch import Adan
        lr = args.lr or 5e-3
        optimizer = Adan(list(params), lr=lr, betas=(0.98, 0.92, 0.99),
                         weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'LAKTJU_NS_Adam':
        from optimizer.LAKTJU_NS_Adam import LAKTJU_NS_Adam
        lr = args.lr or 1e-3
        optimizer = LAKTJU_NS_Adam(params, lr=lr, betas=(args.beta1, 0.999),
                                   weight_decay=args.weight_decay,
                                   ns_interval=args.ns_interval, ns_steps=args.ns_steps,
                                   ns_max_dim=args.ns_max_dim)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        lr = args.lr or 1e-3
        optimizer = LAKTJU_NS(params, lr=lr, betas=(args.beta1, 0.999),
                              weight_decay=args.weight_decay,
                              ns_interval=args.ns_interval, ns_steps=args.ns_steps,
                              ns_max_dim=args.ns_max_dim,
                              adaptive_trigger=getattr(args, 'adaptive_trigger', False),
                              kappa_threshold=getattr(args, 'kappa_threshold', 1e4))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'LAKTJU_Lite':
        from optimizer.LAKTJU_Lite import LAKTJU_Lite
        lr = args.lr or 1e-3
        optimizer = LAKTJU_Lite(params, lr=lr, beta1=0.9, beta2=0.999,
                                weight_decay=args.weight_decay,
                                total_steps=args.epochs * 200)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name in ('MUON', 'Muon'):
        from heavyball import Muon
        lr = args.lr or 1e-3
        optimizer = Muon(list(params), lr=lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'SOAP':
        from heavyball import SOAP
        lr = args.lr or 1e-3
        optimizer = SOAP(list(params), lr=lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    elif opt_name == 'Adan':
        from adan_pytorch import Adan
        lr = args.lr or 1e-3
        optimizer = Adan(list(params), lr=lr, weight_decay=args.weight_decay,
                         betas=(0.98, 0.92, 0.99))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer, scheduler


# ────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────

def compute_kappa(optimizer):
    """Compute momentum buffer condition number κ(M) from optimizer state.
    Uses the first 2D parameter's exp_avg (momentum buffer), with robust SVD."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.dim() < 2:
                continue
            state = optimizer.state.get(p, {})
            if 'exp_avg' not in state:
                continue
            m = state['exp_avg'].detach()
            m2d = m.view(m.size(0), -1).double()  # float64 for numerical stability
            try:
                _, s, _ = torch.linalg.svd(m2d, full_matrices=False)
                # Filter very small singular values (numerical noise)
                s = s[s > 1e-10]
                if len(s) >= 2:
                    kappa = (s[0] / s[-1]).item()
                    return min(kappa, 1e6)  # clamp to avoid numerical extremes
            except:
                pass
            break
    return 1.0


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=0.0, prev_grad_snapshot=None):
    """Train one epoch. Optionally computes inter-epoch gradient correlation.
    Returns (avg_loss, grad_corr_float, grad_snapshot_tensor).
    """
    model.train()
    total_loss, total = 0, 0
    grad_corr_sum, grad_corr_count = 0.0, 0
    first_grad_snapshot = None

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        # Capture gradient snapshot on first batch (for next epoch's correlation)
        if batch_idx == 0:
            grads = []
            for p in model.parameters():
                if p.grad is not None and p.dim() >= 2:
                    grads.append(p.grad.detach().flatten()[:2000])
            if grads:
                first_grad_snapshot = torch.cat(grads).clone()

        # Compute correlation with previous epoch's gradient (every 3rd batch)
        if prev_grad_snapshot is not None and batch_idx % 3 == 0:
            grads = []
            for p in model.parameters():
                if p.grad is not None and p.dim() >= 2:
                    grads.append(p.grad.detach().flatten()[:2000])
            if grads:
                g = torch.cat(grads)
                # Align lengths
                min_len = min(len(g), len(prev_grad_snapshot))
                g_aligned = g[:min_len]
                prev_aligned = prev_grad_snapshot[:min_len]
                g_norm = g_aligned.norm() + 1e-8
                prev_norm = prev_aligned.norm() + 1e-8
                corr = (g_aligned @ prev_aligned) / (g_norm * prev_norm)
                grad_corr_sum += corr.item()
                grad_corr_count += 1

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    avg_grad_corr = grad_corr_sum / max(grad_corr_count, 1) if grad_corr_count > 0 else 0.0
    return total_loss / total, avg_grad_corr, first_grad_snapshot


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))

    return total_loss / total, rmse, mae


def parse_args():
    p = argparse.ArgumentParser(description='C-MAPSS RUL Prediction — LafTJU-TII')
    p.add_argument('--data_dir', type=str, default='../data/cmapss')
    p.add_argument('--subset', type=str, default='FD001',
                   choices=['FD001', 'FD002', 'FD003', 'FD004'])
    p.add_argument('--optimizer', type=str, default='AdamW',
                   choices=['SGD', 'Adam', 'AdamW', 'LAKTJU_NS_Adam', 'Lion', 'RAdam', 'Adan',
                            'LAKTJU_NS', 'LAKTJU_Lite', 'MUON', 'Muon', 'SOAP'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window_size', type=int, default=30)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--save_dir', type=str, default='../results')
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--tag_suffix', type=str, default='')
    p.add_argument('--beta1', type=float, default=0.9,
                   help='First moment EMA coefficient (β₁). Used by AdamW, LAKTJU_NS, LAKTJU_NS_Adam.')
    p.add_argument('--adaptive_trigger', action='store_true',
                   help='LAKTJU_NS only: trigger NS per-layer when estimated κ(M) > threshold.')
    p.add_argument('--kappa_threshold', type=float, default=1e4,
                   help='LAKTJU_NS adaptive trigger threshold (default 1e4).')
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_ds = CMAPSSDataset(args.data_dir, subset=args.subset,
                             window_size=args.window_size, mode='train')
    test_ds = CMAPSSDataset(args.data_dir, subset=args.subset,
                            window_size=args.window_size, mode='test')

    # Split train into train/val (85/15)
    n_train = len(train_ds)
    n_val = int(0.15 * n_train)
    n_tr = n_train - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    input_dim = len(train_ds.dataset.sensor_cols) if hasattr(train_ds, 'dataset') else len(train_ds.sensor_cols)
    model = LSTM_RUL(input_dim=input_dim, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM_RUL: {n_params:,} parameters, input_dim={input_dim}")

    # Optimizer
    optimizer, scheduler = build_optimizer(args, model)
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_test_rmse = float('inf')
    history = []

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"cmapss_{args.subset}_{args.optimizer}_seed{args.seed}"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"
    start = time.time()

    # Spectral diagnostic tracking
    spectral_history = []
    prev_grad_snapshot = None

    for epoch in range(1, args.epochs + 1):
        train_loss, grad_corr, grad_snapshot = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, args.grad_clip, prev_grad_snapshot)
        prev_grad_snapshot = grad_snapshot  # store for next epoch
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, device)
        test_loss, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)

        # Track spectral diagnostic every 10 epochs
        if epoch % 10 == 0:
            kappa_m = compute_kappa(optimizer)
            spectral_history.append({
                'epoch': epoch,
                'kappa_M': float(kappa_m),
                'grad_corr': float(grad_corr),
            })

        # Scheduler step (ReduceLROnPlateau)
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_rmse)
        else:
            scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_test_rmse = test_rmse
            best_test_mae = test_mae

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss, 'val_rmse': val_rmse, 'val_mae': val_mae,
            'test_loss': test_loss, 'test_rmse': test_rmse, 'test_mae': test_mae,
            'best_test_rmse': best_test_rmse,
        })

        if epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - start
            print(f"[{epoch:3d}/{args.epochs}] train_loss={train_loss:.2f} "
                  f"val_rmse={val_rmse:.2f} test_rmse={test_rmse:.2f} "
                  f"best={best_test_rmse:.2f} ({elapsed:.0f}s)")

    total_time = time.time() - start

    # Save results
    result = {
        'config': vars(args),
        'final_test_rmse': history[-1]['test_rmse'],
        'best_test_rmse': best_test_rmse,
        'best_test_mae': best_test_mae,
        'best_val_rmse': best_val_rmse,
        'history': history,
        'spectral_history': spectral_history,
        'total_time': total_time,
        'n_params': n_params,
        'input_dim': input_dim,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    out_path = os.path.join(args.save_dir, f"{tag}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Results saved to {out_path}")
    print(f"Best test RMSE: {best_test_rmse:.2f}, MAE: {best_test_mae:.2f}")


if __name__ == '__main__':
    main()
