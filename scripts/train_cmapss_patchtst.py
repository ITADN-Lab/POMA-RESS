"""PatchTST on C-MAPSS: architecture generalization experiment.

Tests whether LAKTJU-NS benefits transfer from LSTM to modern
Transformer-based time-series models on industrial RUL prediction.

Optimizers: AdamW, LAKTJU_NS, MUON
GC=0 (fair baseline), 5 seeds, best LR per optimizer

Usage:
    python train_cmapss_patchtst.py --subset FD001 --optimizer AdamW --lr 1e-3 --seed 42
"""
import os, sys, json, time, argparse, datetime, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..'))
from train_cmapss_rul import CMAPSSDataset, evaluate
from models_patchtst import PatchTST_RUL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_optimizer(name, params, lr, wd, ns_interval=100):
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == 'LAKTJU_NS':
        from optimizer.LAKTJU_NS import LAKTJU_NS
        return LAKTJU_NS(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd,
                         ns_interval=ns_interval, ns_steps=1, ns_max_dim=256, min_ndim=2)
    elif name in ('MUON', 'Muon'):
        from heavyball import Muon
        return Muon(list(params), lr=lr, weight_decay=wd)
    elif name == 'SOAP':
        from heavyball import SOAP
        return SOAP(list(params), lr=lr, weight_decay=wd)
    elif name == 'Adan':
        from adan_pytorch import Adan
        return Adan(list(params), lr=lr, betas=(0.98, 0.92, 0.99), weight_decay=wd)
    else:
        raise ValueError(name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--subset', default='FD001', choices=['FD001','FD002','FD003','FD004'])
    p.add_argument('--optimizer', default='LAKTJU_NS', choices=['AdamW','LAKTJU_NS','MUON','SOAP','Adan'])
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--ns_interval', type=int, default=100)
    p.add_argument('--ns_steps', type=int, default=1)
    p.add_argument('--ns_max_dim', type=int, default=256)
    p.add_argument('--save_dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    train_ds = CMAPSSDataset(args.data_dir, subset=args.subset, window_size=30, mode='train')
    test_ds = CMAPSSDataset(args.data_dir, subset=args.subset, window_size=30, mode='test')

    n_train, n_val = len(train_ds), int(0.15 * len(train_ds))
    train_split, val_split = torch.utils.data.random_split(
        train_ds, [n_train - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_split, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_split, args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=2)

    input_dim = len(train_ds.sensor_cols)
    model = PatchTST_RUL(n_vars=input_dim, seq_len=30, patch_len=6, stride=3,
                         d_model=128, n_heads=4, n_layers=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PatchTST: {n_params:,} params, input_dim={input_dim}, subset={args.subset}")

    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.ns_interval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"patchtst_{args.subset}_{args.optimizer}_seed{args.seed}"
    out_path = os.path.join(args.save_dir, f"{tag}.json")

    best_val_rmse, best_test_rmse = float('inf'), float('inf')
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
        train_loss = total_loss / total

        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, DEVICE)
        test_loss, test_rmse, test_mae = evaluate(model, test_loader, criterion, DEVICE)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_rmse)
        else:
            scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse, best_test_rmse, best_test_mae = val_rmse, test_rmse, test_mae

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_rmse': val_rmse,
                        'test_rmse': test_rmse, 'best_val_rmse': best_val_rmse,
                        'best_test_rmse': best_test_rmse})
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] train_loss={train_loss:.2f} val_rmse={val_rmse:.2f} "
                  f"test_rmse={test_rmse:.2f} best={best_test_rmse:.2f} ({(time.time()-t0)/60:.1f}min)")

    result = {
        'config': vars(args), 'n_params': n_params, 'input_dim': input_dim,
        'best_test_rmse': float(best_test_rmse), 'best_val_rmse': float(best_val_rmse),
        'best_test_mae': float(best_test_mae),
        'final_test_rmse': float(history[-1]['test_rmse']),
        'history': history[-50:], 'total_time': time.time() - t0,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Saved {out_path} | best_test_rmse={best_test_rmse:.2f}")


if __name__ == '__main__':
    main()
