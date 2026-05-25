"""C-MAPSS training with PER-LAYER spectral logging (B2)."""
import os, sys, json, time, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..'))
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..'))

from train_cmapss_rul import CMAPSSDataset, LSTM_RUL, train_one_epoch, evaluate
from spectral_tracker_perlayer import attach_tracker_perlayer


def build_optimizer_local(name, params, lr, wd, ns_interval, ns_steps, ns_max_dim):
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        return LAKTJU_NS(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd,
                         ns_interval=ns_interval, ns_steps=ns_steps,
                         ns_max_dim=ns_max_dim, min_ndim=2)
    raise ValueError(name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--subset', default='FD002')
    p.add_argument('--optimizer', default='LAKTJU_NS')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
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
    p.add_argument('--tag_suffix', type=str, default='perlayer')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = CMAPSSDataset(args.data_dir, subset=args.subset,
                             window_size=args.window_size, mode='train')
    test_ds = CMAPSSDataset(args.data_dir, subset=args.subset,
                            window_size=args.window_size, mode='test')
    n_train = len(train_ds); n_val = int(0.15 * n_train)
    train_ds_split, val_ds = torch.utils.data.random_split(
        train_ds, [n_train - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds_split, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    input_dim = len(train_ds.sensor_cols)
    model = LSTM_RUL(input_dim=input_dim, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, dropout=args.dropout).to(device)

    base_opt = build_optimizer_local(args.optimizer, model.parameters(), args.lr,
                                     args.weight_decay, args.ns_interval,
                                     args.ns_steps, args.ns_max_dim)
    spectral_log = []
    optimizer = attach_tracker_perlayer(base_opt, args.optimizer, spectral_log)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"spectral_perlayer_{args.subset}_{args.optimizer}_seed{args.seed}"
    out_path = os.path.join(args.save_dir, f"{tag}.json")

    history = []
    best_val_rmse = best_test_rmse = best_test_mae = float('inf')
    final_test_rmse = float('inf'); prev = None; t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, grad_corr, prev = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip, prev)
        _, val_rmse, _ = evaluate(model, val_loader, criterion, device)
        _, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
        if val_rmse < best_val_rmse:
            best_val_rmse, best_test_rmse, best_test_mae = val_rmse, test_rmse, test_mae
        final_test_rmse = test_rmse
        history.append({'epoch': epoch, 'val_rmse': val_rmse, 'test_rmse': test_rmse,
                        'best_test_rmse': best_test_rmse, 'grad_corr': grad_corr})
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] val={val_rmse:.2f} test={test_rmse:.2f} "
                  f"best={best_test_rmse:.2f} ({time.time()-t0:.0f}s) "
                  f"NSevents={len(spectral_log)}")

    n_layers = spectral_log[0]['n_layers'] if spectral_log else 0
    result = {
        'config': vars(args),
        'final_test_rmse': float(final_test_rmse),
        'best_test_rmse': float(best_test_rmse),
        'best_test_mae': float(best_test_mae),
        'best_val_rmse': float(best_val_rmse),
        'history': history,
        'n_layers_tracked': n_layers,
        'spectral_log_perlayer': spectral_log,
        'spectral_log_size': len(spectral_log),
        'total_time': time.time() - t0,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved {out_path} | NS events={len(spectral_log)} | layers/event={n_layers}")


if __name__ == '__main__':
    main()
