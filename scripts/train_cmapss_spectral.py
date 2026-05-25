"""C-MAPSS training with detailed spectral logging.

Wraps the optimizer with a SpectralTracker that records pre-NS / post-NS κ(M) and erank
for LAKTJU_NS, and periodic samples for AdamW/MUON. Outputs JSON with full spectral_log.
"""
import os, sys, json, time, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..'))

from train_cmapss_rul import CMAPSSDataset, LSTM_RUL, train_one_epoch, evaluate
from spectral_tracker import attach_tracker


def build_optimizer_local(name, params, lr, wd):
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        return LAKTJU_NS(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd,
                         ns_interval=100, ns_steps=1, ns_max_dim=256, min_ndim=2)
    elif name == 'MUON':
        from heavyball import Muon
        return Muon(list(params), lr=lr, weight_decay=wd)
    else:
        raise ValueError(name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--subset', default='FD001')
    p.add_argument('--optimizer', default='LAKTJU_NS')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window_size', type=int, default=30)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--tag_suffix', type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = CMAPSSDataset(args.data_dir, subset=args.subset, window_size=args.window_size, mode='train')
    test_ds = CMAPSSDataset(args.data_dir, subset=args.subset, window_size=args.window_size, mode='test')

    n_train = len(train_ds)
    n_val = int(0.15 * n_train)
    train_ds_split, val_ds = torch.utils.data.random_split(
        train_ds, [n_train - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds_split, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    input_dim = len(train_ds.sensor_cols)
    model = LSTM_RUL(input_dim=input_dim, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, dropout=args.dropout).to(device)

    base_opt = build_optimizer_local(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    spectral_log = []
    optimizer = attach_tracker(base_opt, args.optimizer, spectral_log, sample_interval=100)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"spectral_{args.subset}_{args.optimizer}_seed{args.seed}"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"
    out_path = os.path.join(args.save_dir, f"{tag}.json")

    history = []
    best_val_rmse, best_test_rmse, best_test_mae = float('inf'), float('inf'), float('inf')
    final_test_rmse = float('inf')
    prev_grad_snapshot = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, grad_corr, prev_grad_snapshot = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip, prev_grad_snapshot)
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, device)
        test_loss, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
        if val_rmse < best_val_rmse:
            best_val_rmse, best_test_rmse, best_test_mae = val_rmse, test_rmse, test_mae
        final_test_rmse = test_rmse
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_rmse': val_rmse,
                        'test_rmse': test_rmse, 'best_val_rmse': best_val_rmse,
                        'best_test_rmse': best_test_rmse, 'grad_corr': grad_corr})
        if epoch % 20 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(f"[{epoch:3d}/{args.epochs}] train_loss={train_loss:.2f} val_rmse={val_rmse:.2f} "
                  f"test_rmse={test_rmse:.2f} best={best_test_rmse:.2f} ({elapsed:.0f}s)")

    result = {
        'config': vars(args),
        'final_test_rmse': float(final_test_rmse),
        'best_test_rmse': float(best_test_rmse),
        'best_test_mae': float(best_test_mae),
        'best_val_rmse': float(best_val_rmse),
        'history': history,
        'spectral_log': spectral_log,
        'spectral_log_size': len(spectral_log),
        'total_time': time.time() - t0,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved {out_path} | spectral events={len(spectral_log)}")


if __name__ == '__main__':
    main()
