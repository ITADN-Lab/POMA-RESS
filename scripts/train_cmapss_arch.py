"""
Ablation 2.2: Non-LSTM architectures for C-MAPSS RUL prediction.
Tests 1D-CNN and Transformer on FD001/FD004 with AdamW vs LAKTJU-NS.
"""
import os, sys, json, time, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from train_cmapss_rul import CMAPSSDataset, evaluate, compute_kappa


class CNN1D_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, W, S) -> (B, S, W) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class Transformer_RUL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 30, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*2,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, W, S)
        x = self.proj(x) + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling over time
        return self.fc(x).squeeze(-1)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, total = 0, 0
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


def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = CMAPSSDataset(args.data_dir, subset=args.subset, window_size=30, mode='train')
    test_ds = CMAPSSDataset(args.data_dir, subset=args.subset, window_size=30, mode='test')
    n_val = int(0.15 * len(train_ds))
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [len(train_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    input_dim = test_ds.sensor_cols.__len__() if hasattr(test_ds, 'sensor_cols') else train_ds.dataset.sensor_cols.__len__()

    if args.model == 'CNN1D':
        model = CNN1D_RUL(input_dim).to(device)
    else:
        model = Transformer_RUL(input_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    if args.optimizer == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        optimizer = LAKTJU_NS(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                              weight_decay=1e-4, ns_interval=100, ns_steps=1, ns_max_dim=256)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_test_rmse = float('inf')
    start = time.time()

    for epoch in range(1, 101):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, val_rmse, _ = evaluate(model, val_loader, criterion, device)
        _, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_test_rmse = test_rmse
            best_test_mae = test_mae

        if epoch % 20 == 0:
            print(f"  [{epoch}/100] val_rmse={val_rmse:.2f} test_rmse={test_rmse:.2f} best={best_test_rmse:.2f}")

    total_time = time.time() - start
    tag = f"cmapss_{args.subset}_{args.model}_{args.optimizer}_seed{args.seed}"
    result = {
        'config': {'subset': args.subset, 'model': args.model, 'optimizer': args.optimizer, 'seed': args.seed},
        'best_test_rmse': best_test_rmse,
        'best_test_mae': best_test_mae,
        'total_time': total_time,
        'n_params': n_params,
    }
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"{tag}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"  => {args.model}/{args.optimizer}/seed{args.seed}: RMSE={best_test_rmse:.2f} ({total_time:.1f}s)")
    return best_test_rmse


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='../data/cmapss')
    p.add_argument('--subset', default='FD001')
    p.add_argument('--model', choices=['CNN1D', 'Transformer'], default='CNN1D')
    p.add_argument('--optimizer', choices=['AdamW', 'LAKTJU_NS'], default='AdamW')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_dir', default='../results/ablation_2_2')
    args = p.parse_args()
    run_experiment(args)
