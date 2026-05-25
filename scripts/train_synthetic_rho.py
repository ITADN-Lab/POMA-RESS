"""
Synthetic Industrial Process Dataset with Controllable Gradient Correlation.
Tests the dual diagnostic (κ>50 or ρ_g>0.8) across varying ρ_g levels.
Generates temporally correlated multivariate time series with tunable ρ_g.
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_correlated_series(n_samples=5000, n_features=20, rho_g=0.85, seed=42):
    """Generate time series with controllable inter-step gradient correlation."""
    rng = np.random.default_rng(seed)
    # Base signal: slow-varying latent factors
    t = np.linspace(0, 20*np.pi, n_samples)
    latent = np.column_stack([
        np.sin(t * (i+1) * 0.3) + 0.5 * np.cos(t * (i+1) * 0.7)
        for i in range(5)
    ])  # (n_samples, 5)
    # Mix to n_features with correlation
    W = rng.normal(0, 1, (5, n_features))
    X_signal = latent @ W  # deterministic signal component
    # Add noise scaled by (1-rho_g): lower rho_g = more noise = less correlation
    noise_scale = np.sqrt(1 - rho_g**2) / max(rho_g, 1e-4)
    X = X_signal + noise_scale * rng.normal(0, 1, X_signal.shape)
    # Regression target: non-linear function of first 3 latent factors
    y = (latent[:, 0]**2 + 0.5 * latent[:, 1] * latent[:, 2] +
         rng.normal(0, 0.1, n_samples))
    return X.astype(np.float32), y.astype(np.float32)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def build_optimizer(args, model, steps_per_epoch):
    params = model.parameters()
    if args.optimizer == 'AdamW':
        opt = optim.AdamW(params, lr=args.lr or 1e-3, weight_decay=args.weight_decay)
    elif args.optimizer == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        opt = LAKTJU_NS(params, lr=args.lr or 1e-3, betas=(0.9,0.999),
                        weight_decay=args.weight_decay, ns_interval=args.ns_interval,
                        ns_steps=args.ns_steps, ns_max_dim=args.ns_max_dim, min_ndim=2)
    else: raise ValueError(args.optimizer)
    return opt, lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


def train_epoch(model, loader, opt, criterion):
    model.train(); total_loss, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
        loss = criterion(model(x), y); loss.backward(); opt.step()
        total_loss += loss.item() * x.size(0); total += x.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); total_loss, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = criterion(model(x), y)
        total_loss += loss.item() * x.size(0); total += x.size(0)
    return total_loss / total


def parse_args():
    p = argparse.ArgumentParser(description='Synthetic ρ_g Diagnostic Validation')
    p.add_argument('--optimizer', default='LAKTJU_NS', choices=['AdamW','LAKTJU_NS'])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rho_g', type=float, default=0.85)
    p.add_argument('--n_samples', type=int, default=5000)
    p.add_argument('--n_features', type=int, default=20)
    p.add_argument('--ns_interval', type=int, default=50)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--save_dir', default='./results')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    X, y = generate_correlated_series(args.n_samples, args.n_features, args.rho_g, args.seed)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=args.seed)

    tr_ld = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)), batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(y_va)), batch_size=args.batch_size)
    ts_ld = DataLoader(TensorDataset(torch.FloatTensor(X_ts), torch.FloatTensor(y_ts)), batch_size=args.batch_size)

    model = MLPRegressor(args.n_features).to(DEVICE)
    opt, sched = build_optimizer(args, model, len(tr_ld))
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"synth_rho{args.rho_g}_{args.optimizer}_seed{args.seed}"
    best_va, best_ts = float('inf'), float('inf')
    start = time.time()

    for epoch in range(1, args.epochs+1):
        tr_l = train_epoch(model, tr_ld, opt, criterion)
        va_l = evaluate(model, va_ld, criterion)
        ts_l = evaluate(model, ts_ld, criterion)
        sched.step()
        if va_l < best_va:
            best_va = va_l; best_ts = ts_l

    elapsed = time.time() - start
    result = {'config': vars(args), 'best_val_loss': float(best_va),
              'best_test_loss': float(best_ts), 'total_time': elapsed}
    fp = os.path.join(args.save_dir, f"{tag}.json")
    with open(fp, 'w') as f: json.dump(result, f, indent=2, default=float)
    print(f"ρ_g={args.rho_g} {args.optimizer}: test_loss={best_ts:.4f} ({elapsed:.0f}s)")

if __name__ == '__main__': main()
