"""C-MAPSS training with mechanism ablation variants.

Usage: python train_cmapss_ablation_variants.py --variant LAKTJU_NS --subset FD002 --seed 42
"""
import os, sys, json, time, argparse, datetime, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '..'))

from train_cmapss_rul import CMAPSSDataset, LSTM_RUL, evaluate
from ablation_variants import ABLATION_VARIANTS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--subset', default='FD002')
    p.add_argument('--variant', default='LAKTJU_NS', choices=list(ABLATION_VARIANTS.keys()))
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--grad_clip', type=float, default=0.0)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--interval', type=int, default=100)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--max_dim', type=int, default=256)
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
        train_ds, [n_train-n_val, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_split, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_split, args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=2)

    input_dim = len(train_ds.sensor_cols)
    model = LSTM_RUL(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.3).to(DEVICE)

    factory = ABLATION_VARIANTS[args.variant]
    if args.variant == 'AdamW':
        optimizer = factory(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.variant in ('NormOnly', 'RandRot'):
        optimizer = factory(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay, interval=args.interval,
                           max_dim=args.max_dim)
    else:
        optimizer = factory(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay, interval=args.interval,
                           steps=args.steps, max_dim=args.max_dim)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"ablate_{args.variant}_{args.subset}_seed{args.seed}.json")

    best_val_rmse, best_test_rmse = float('inf'), float('inf')
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if args.grad_clip > 0: nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()*x.size(0); total += x.size(0)

        val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, DEVICE)
        test_loss, test_rmse, test_mae = evaluate(model, test_loader, criterion, DEVICE)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse, best_test_rmse = val_rmse, test_rmse

        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"[{epoch:3d}/{args.epochs}] val={val_rmse:.2f} test={test_rmse:.2f} best={best_test_rmse:.2f}")

    result = {'config': vars(args), 'best_test_rmse': float(best_test_rmse),
              'best_val_rmse': float(best_val_rmse), 'total_time': time.time()-t0,
              'timestamp': datetime.datetime.now().isoformat()}
    with open(out_path,'w') as f: json.dump(result, f, indent=2, default=float)
    print(f"Saved {out_path} best_test_rmse={best_test_rmse:.2f}")

if __name__ == '__main__':
    main()
